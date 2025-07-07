import cv2
import numpy as np
from typing import Tuple, List, Optional
import os
import sqlite3
import shutil

def process_recording(
    source_path: str,
    colour_min: np.ndarray,
    colour_max: np.ndarray,
    colorspace: str,
    width: int = 640,
    height: int = 400,
    debug: bool = False,
    num_frames: Optional[int] = None,
    output_dir: str = "output_dir",
) -> None:
    """
    Process a video or database recording to extract frames and generate prompts.

    Args:
        source_path (str): Path to the video or database file.
        colour_min (np.ndarray): Minimum color threshold for masking.
        colour_max (np.ndarray): Maximum color threshold for masking.
        colorspace (str): Colorspace conversion string for cv2.
        width (int): Width of the frames.
        height (int): Height of the frames.
        debug (bool): Whether to enable debug mode.
        num_frames (Optional[int]): Number of frames to process. None for all frames.
        output_dir (str): Directory to save output frames and prompts.

    Returns:
        None
    """
    frame_count = 0
    prompts_list = []

    # Create directories for saving frames and prompts
    frames_dir = f"{output_dir}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    prompts_dir = f"{output_dir}/prompts"
    os.makedirs(prompts_dir, exist_ok=True)

    # Process video file
    if source_path.endswith(".mp4"):
        source = cv2.VideoCapture(source_path)

        while (not num_frames or frame_count < num_frames):
            ret, frame = source.read()
            if not ret:
                break

            # Process each frame and generate prompts
            processed_frame, mask, prompts = process_frame(
                frame, colour_min, colour_max, colorspace
            )
            prompts_list.append(prompts)

            # Display debug information if enabled
            if debug:
                combined_frame = np.vstack((processed_frame, mask))
                cv2.imshow('DEBUG', combined_frame)
                cv2.waitKey(10)

            frame_count += 1

        source.release()

    # Process database file
    elif source_path.endswith(".db"):
        conn = sqlite3.connect(source_path)
        cursor = conn.cursor()

        # Query all rows from the state_vector_data table, ordered by id
        cursor.execute("SELECT * FROM state_vector_data ORDER BY id ASC")

        # Iterate over each row in the query result
        while (not num_frames or frame_count < num_frames):
            row = cursor.fetchone()
            if not row:
                break

            # Convert binary data to numpy arrays for depth and RGB frames
            rgb_frame = np.frombuffer(row[8], dtype=np.uint8)

            # Reshape frames to expected dimensions
            rgb_frame = rgb_frame.reshape((height, width, 3))

            # Save the frame as an image
            frame_filename = os.path.join(frames_dir, f"{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, rgb_frame)

            # Process the frame to generate prompts
            prompts = process_frame(rgb_frame, colour_min, colour_max, colorspace)
            if prompts == 1:
                print("Exiting due to user input.")
                break
            
            prompts_list.append(np.asarray(prompts, dtype=np.int16))

            frame_count += 1

        conn.close()

    else:
        raise ValueError("Unsupported video source format. Please provide a .mp4 or .db file.")
    print(f"Processed {frame_count} frames.")

    # Clean up and save prompts
    cv2.destroyAllWindows()
    if prompts_list:
        np.save(f"{prompts_dir}/prompts_per_frame.npy", np.array(prompts_list, dtype=object))
    else:
        print("An error occurred, no prompts generated.")


def _enclose_polygons(mask: np.ndarray) -> np.ndarray:
    """
    Enclose polygons in the mask by filling gaps between contours.

    Args:
        mask (np.ndarray): Input binary mask.

    Returns:
        np.ndarray: Modified mask with enclosed polygons.
    """
    def _fill(row: np.ndarray) -> None:
        indices = np.where(row == 255)[0]
        for i in range(0, len(indices) - 1, 2):
            row[indices[i] + 1 : indices[i + 1]] = 255

    _fill(mask[0])
    _fill(mask[-1])
    return mask

def process_frame(
    frame: np.ndarray,
    colour_min: np.ndarray,
    colour_max: np.ndarray,
    colorspace: str
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Process a single frame to generate a mask and prompts.

    Args:
        frame (np.ndarray): Input frame.
        colour_min (np.ndarray): Minimum color threshold for masking.
        colour_max (np.ndarray): Maximum color threshold for masking.
        colorspace (str): Colorspace conversion string for cv2.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]: Processed frame, mask, and prompts.
    """
    original_frame = frame.copy()
    frame = cv2.cvtColor(frame, eval(colorspace))
    height, width = frame.shape[:2]
    crop = int(height / 2.5)

    # Crop the frame and apply a negative region of interest
    mask = frame[crop:]
    mask = negative_region_of_interest(mask)

    # Create a binary mask based on color thresholds
    colour_mask = cv2.inRange(
        mask, colour_min, colour_max
    )
    colour_mask = _enclose_polygons(colour_mask)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(colour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame = original_frame.copy()
    prompts = []
    valid_contours = []
    threshold_area =  300  # Minimum area for valid contours

    # Filter contours based on area and draw them on the frame
    for cnt in contours:
        if cv2.contourArea(cnt) > threshold_area:
            valid_contours.append(cnt)
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2, offset=(0, crop))

    # Generate prompts in concentric rings
    center_x, center_y = width // 2, height // 2
    max_radius = max(center_x, center_y)
    num_rings = 15
    points_per_ring = 20
    j = 2

    prompts_per_frame = []
    for ring in range(0, num_rings+2):
        if ring % j == 0:
            radius = (ring / num_rings) * max_radius
            for i in range(points_per_ring):
                angle = (2 * np.pi / points_per_ring)* i + ring + 35
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                # Check if the point lies within any valid contour
                if any(cv2.pointPolygonTest(cnt, (x, y - crop), False) >= 0 for cnt in valid_contours):
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                    prompts_per_frame.append([x, y])
                else:
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                if ring == 0:
                    break

    # Callback function for mouse interaction
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for point in prompts_per_frame:
                if np.linalg.norm(np.array([x, y]) - np.array(point)) <= 5:
                    prompts_per_frame.remove(point)
                    cv2.circle(frame, (point[0], point[1]), 3, (255, 0, 0), -1)
                    break

    # Display the frame and mask for debugging
    while True:
        mask = cv2.cvtColor(colour_mask, cv2.COLOR_GRAY2BGR)
        combined_frame = np.vstack((frame, mask))
        cv2.imshow('DEBUG', combined_frame)
        cv2.setMouseCallback('DEBUG', mouse_callback)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):  # Proceed to the next frame
            break
        elif key == ord('l'):  # Clear prompts for the current frame
            prompts_per_frame = []
            break
        elif key == ord('q'):  # Quit processing
            return 1
    
    prompts.extend(prompts_per_frame)

    return prompts

def negative_region_of_interest(frame: np.ndarray) -> np.ndarray:
    """
    Apply a negative region of interest mask to the frame.

    Args:
        frame (np.ndarray): Input frame.

    Returns:
        np.ndarray: Masked frame.
    """
    # Optionally, you can define a region of interest
    #height, width = frame.shape[:2]
    #trapezoid = np.array([
    #    [(0, height), (width, height), (int(0.75 * width), height // 2), (int(0.25 * width), height // 2)]
    #], dtype=np.int32)

    mask = np.ones_like(frame[:, :, 0]) * 255
    #cv2.fillPoly(mask, [trapezoid], 1)
    return cv2.bitwise_and(frame, frame, mask=mask)

def display_lines(frame: np.ndarray, lines: Optional[np.ndarray]) -> None:
    """
    Display lines on the frame.

    Args:
        frame (np.ndarray): Input frame.
        lines (Optional[np.ndarray]): Detected lines.

    Returns:
        None
    """
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4) if line.ndim == 2 else line
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=10)

if __name__ == "__main__":
    # Define input source path and output directory
    source_path = "data/recording.db"  # Path to the video or database file
    output_dir = "data/frames_and_prompts"
    
    debug = True  # Enable debug mode
    colorspace_min = np.array([0, 0, 0], np.uint8)  # Minimum color threshold
    colorspace_max = np.array([33, 112, 123], np.uint8)  # Maximum color threshold
    num_frames = None  # Set to None to process all frames

    # Remove existing output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Process the recording
    process_recording(source_path, colorspace_min, colorspace_max,
                        colorspace="cv2.COLOR_BGR2XYZ", num_frames=num_frames,
                        debug=debug, output_dir=output_dir)
