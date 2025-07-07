"""
This script uses the Sam2 model to generate masks from video frames based on input points.
It initializes the model, adds new points, propagates the prompts throughout the video,
and saves the resultant masked images.

This work uses the Sam2 model as described in:
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
"""
import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../sam2"))
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = os.path.join(os.path.dirname(__file__), "../sam2/checkpoints/sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # This should be relative to the sam2 package

prompts_source_path = "data/frames_and_prompts0/prompts/prompts_per_frame.npy"
frames_source_dir = "data/frames_and_prompts0/frames"

masks_output_dir = "data/dataset/targets"
frames_output_dir = "data/dataset/images"

def build_predictor():
    print("Building predictor...")
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    print("Predictor built successfully.")
    return predictor

def initialize_state(predictor, frames_output_dir):
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = predictor.init_state(frames_output_dir)
    print("State initialized.")
    return state

def add_inital_prompts(predictor, state, points):
    print("Adding points from prompts...")
    labels = np.array([1] * len(points), dtype=np.int8)
    predictor.add_new_points_or_box(state, frame_idx=0, points=points, labels=labels, obj_id=1)


def image_predict(output_dir, frames_output_dir, visual_prompts, debug=True):
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=torch.device("cuda")))
    print("Generating masks for images...")
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted([
        f for f in os.listdir(frames_output_dir)
        if os.path.isfile(os.path.join(frames_output_dir, f)) and f.endswith('.jpg')
    ])

    for i, frame_file in enumerate(frame_files):
        image_path = os.path.join(frames_output_dir, frame_file)
        image = Image.open(image_path).convert("RGB")

        points = visual_prompts[i]
        labels = np.array([1] * len(points), dtype=np.int8)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )

            # Select the mask with highest score
            best_idx = np.argmax(scores)
            best_mask = (masks[best_idx] > 0).astype(np.uint8)

        h, w = best_mask.shape
        mask_rgb = np.repeat(best_mask.reshape(h, w, 1), 3, axis=2)

        # Write mask to file
        output_path = os.path.join(output_dir, frame_file.replace('.jpg', '_mask.jpg'))
        cv2.imwrite(output_path, mask_rgb)
        

        if debug:
            orig = cv2.imread(image_path)
            overlay = orig.copy()
            overlay[best_mask == 1] = (255, 105, 250)  # Magenta overlay
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, orig, 1 - alpha, 0, orig)
            cv2.imshow('Masked Image', orig)
            cv2.waitKey(1)
            del orig, overlay

        del masks, best_mask, mask_rgb

    print('Masks created and saved')
    cv2.destroyAllWindows()



def propagate_and_save_masks(predictor, state, batch_prompts, frame_idx_offset, debug):
    print("Propagating in video...")
    curr_idx = frame_idx_offset
    batch_frame_idx = 0
    for batch_frame_idx, _, masks in predictor.propagate_in_video(state):
        curr_idx += batch_frame_idx
        # Convert confident mask to binary mask
        mask = (masks[0].cpu().numpy() > 0).astype(np.uint8)
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask = np.repeat(mask.reshape(h, w, 1), 3, axis=2) # Convert to RGB for visualization

        # Check if all points in visual prompts are covered by the mask
        if len(batch_prompts[batch_frame_idx]) > 0:
            for point in batch_prompts[batch_frame_idx]:
                if point is None or mask[point[1], point[0], 0] != 1:
                    print(f"Warning: Point {point} in frame {curr_idx} is not covered by the mask.")
                    return batch_frame_idx + 1
        else:
            continue  # Skip if no prompts for this frame
        
        # Check if any mask exists and contains at least one '1'
        if not np.any(mask):
            print(f"No valid masks generated for frame {curr_idx}. Skipping this frame.")
            continue
        # Image path
        if append:
            image_name = f"{append_frame_idx + curr_idx:05d}.jpg"
        else:
            image_name = f"{curr_idx:05d}.jpg"

        # Write mask to file
        output_path = os.path.join(masks_output_dir, image_name) 
        cv2.imwrite(output_path, mask)
        # Write corresponding frame to file
        src_path = os.path.join(frames_source_dir, batch_files[batch_frame_idx])
        dst_path = os.path.join(frames_output_dir, image_name)
        shutil.copy(src_path, dst_path)

        if debug:            
            # Read original image and apply mask overlay
            image_path = os.path.join(src_path)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read image at {image_path}. Skipping this frame.")
                continue
            overlay = image.copy()
            overlay[mask[:, :, 0] == 1] = (255, 105, 250)
            # Draw a green circle at point [0]
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            cv2.imshow('Masked Image', image)
            cv2.waitKey(1)

    print('Masks created and saved')
    cv2.destroyAllWindows()
    return batch_frame_idx

def process_video_frames(predictor, state, batch_prompts, frame_idx, debug):
    i = 0
    while i < len(batch_prompts) and (batch_prompts[i] is None or len(batch_prompts[i]) == 0):
        i += 1
    if i < len(batch_prompts):
        add_inital_prompts(predictor, state, batch_prompts[i])
        stopped_frame = propagate_and_save_masks(predictor, state, batch_prompts, frame_idx, debug)
        return stopped_frame
    else:
        return i

if __name__ == "__main__":
    debug = False
    append = False
    batch_size = 32
    os.makedirs(masks_output_dir, exist_ok=True)
    os.makedirs(frames_output_dir, exist_ok=True)

    # Load points and labels from prompts_source_path
    visual_prompts = np.load(prompts_source_path, allow_pickle=True)
    frame_idx = 0

    # List files sorted in frames_source_dir
    frame_files = sorted([
        f for f in os.listdir(frames_source_dir)
        if os.path.isfile(os.path.join(frames_source_dir, f)) and f.endswith('.jpg')
    ])

    append_frame_idx = 0
    if append:
        # Get the latest file name from frames_output_dir
        existing_files = sorted([
            f for f in os.listdir(frames_output_dir)
            if os.path.isfile(os.path.join(frames_output_dir, f)) and f.endswith('.jpg')
        ])
        if existing_files:
            latest_file = existing_files[-1]
            append_frame_idx = int(latest_file.split('.')[0]) + 1

    # Build the predictor
    predictor = build_predictor()

    # Batch frames in the expected sam2 format for processing large media
    while frame_idx < len(frame_files):
        temp_dir = os.path.join(masks_output_dir, "temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        # Copy batch of frames to temp_dir
        batch_files = frame_files[frame_idx:frame_idx+batch_size]
        batch_prompts = visual_prompts[frame_idx:frame_idx+batch_size]
        for i, file in enumerate(batch_files):
            src_path = os.path.join(frames_source_dir, file)
            dst_path = os.path.join(temp_dir, f"{i:05d}.jpg") # Sam2 name format
            shutil.copy(src_path, dst_path)

        state = initialize_state(predictor, temp_dir)
        print(f"Processing from {frame_idx} to {frame_idx + len(batch_files) - 1}...")
        frame_idx += process_video_frames(predictor, state, batch_prompts, frame_idx, debug)
        del state

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
