import cv2
import numpy as np

def create_batch_collage(frames_list):
    """
    Takes a list of 5 frames (1280x720) and stitches them into a 2560x2160 collage.
    Returns the giant collage image.
    """
    # Assuming all frames are your standard 720(H) x 1280(W)
    frame_h, frame_w = frames_list[0].shape[:2]
    
    # Create a giant black canvas for a 3-row, 2-column grid
    # Height = 720 * 3 = 2160. Width = 1280 * 2 = 2560
    collage = np.zeros((frame_h * 3, frame_w * 2, 3), dtype=np.uint8)
    
    # Define the (Y, X) starting pixel coordinates for each of the 6 grid slots
    grid_positions = [
        (0, 0),                 (0, frame_w),             # Row 1 (Frames 0, 1)
        (frame_h, 0),           (frame_h, frame_w),       # Row 2 (Frames 2, 3)
        (frame_h * 2, 0),       (frame_h * 2, frame_w)    # Row 3 (Frame 4, Empty)
    ]
    
    # Paste each frame into its assigned slot on the giant canvas
    for i, frame in enumerate(frames_list):
        y_offset, x_offset = grid_positions[i]
        collage[y_offset : y_offset + frame_h, x_offset : x_offset + frame_w] = frame
        
    return collage


def decode_collage_labels(custom_labels, original_w, original_h):
    """Translates AWS coordinates from the giant collage back into 5 separate original frames."""
    collage_w = original_w * 2
    collage_h = original_h * 3
    separated_results = [[] for _ in range(5)]
    
    for label in custom_labels:
        box = label.get("Geometry", {}).get("BoundingBox")
        if not box: continue
            
        abs_left = int(box["Left"] * collage_w)
        abs_top = int(box["Top"] * collage_h)
        abs_width = int(box["Width"] * collage_w)
        abs_height = int(box["Height"] * collage_h)
        
        # Determine grid column and row, safely clamped to max grid size
        col = min(abs_left // original_w, 1)
        row = min(abs_top // original_h, 2)
        frame_index = (row * 2) + col
        
        if frame_index >= 5: continue
            
        local_left = abs_left - (col * original_w)
        local_top = abs_top - (row * original_h)
        
        new_label = {
            "Name": label["Name"],
            "Confidence": label["Confidence"],
            "Geometry": {
                "BoundingBox": {
                    "Left": max(0, local_left / original_w),
                    "Top": max(0, local_top / original_h),
                    "Width": abs_width / original_w,
                    "Height": abs_height / original_h
                }
            }
        }
        separated_results[frame_index].append(new_label)
        
    return separated_results