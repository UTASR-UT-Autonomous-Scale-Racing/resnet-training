# Instructions

1. `cd Sam2Masks`

2. **Sam2 Installation:**
    git clone https://github.com/facebookresearch/sam2.git && cd sam2

3. **Install Sam2:**
    pip install -e .
    *Note: If you encounter errors, check your network speed or try downloading the package independently.*

4. **Download Checkpoint:**
    `cd checkpoints/`
    `wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`

5. **Set config file**
    `configs/sam2.1/sam2.1_hiera_l.yaml`

6. **Create JPG Folder:**
    `python generate_frames_and_prompts.py`

7. **Run Video Masking:**
    `python generate_sam2_masks.py`

8. **Train ResNet:**
    `cd resnet/`
    `python train.py`

9. **Run Inference:**
    `cd resnet/`
    `python inference.py`

**Summary of Outputs:**

- `data/frames_and_prompts`: Contains extracted JPG images from the video and prompts for each frame.
- `data/sam2_masked_frames`: Contains masked images with pixel values of 0 and 1.
- `checkpoints/mask_rcnn_model.onnx`: Contains the trained Mask R-CNN model.

