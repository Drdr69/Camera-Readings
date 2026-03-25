import cv2
import numpy as np
import argparse
import json
import os

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img

def calibrate_camera(image_path, reference_width_mm, reference_height_mm):
    """
    To ensure 100% accuracy, you need a known reference object in the same plane
    as the box you are measuring. A checkerboard, a coin, or a printed 
    calibration target of exact known dimensions.
    
    This script helps extract the pixel-to-millimeter ratio by drawing a box 
    around the reference object.
    """
    img = load_image(image_path)
    clone = img.copy()
    ref_points = []
    
    # Simple mouse callback to select the reference object
    def click_and_crop(event, x, y, flags, param):
        nonlocal ref_points, clone
        
        # If the left mouse button was clicked, record the starting (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_points = [(x, y)]

        # If the left mouse button was released, record the ending (x, y) coordinates
        elif event == cv2.EVENT_LBUTTONUP:
            ref_points.append((x, y))

            # Draw a rectangle around the region of interest
            cv2.rectangle(clone, ref_points[0], ref_points[1], (0, 255, 0), 2)
            cv2.imshow("Calibrate", clone)
            
    print("Instructions:")
    print("1. A window will open showing your image.")
    print("2. Click and drag to draw a box perfectly around your reference object.")
    print("3. Press 'c' to confirm the box.")
    print("4. Press 'r' to reset if you make a mistake.")
            
    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", click_and_crop)

    while True:
        cv2.imshow("Calibrate", clone)
        key = cv2.waitKey(1) & 0xFF

        # If 'r' is pressed, reset the cropping region
        if key == ord("r"):
            clone = img.copy()
            ref_points = []

        # If 'c' is pressed, break from the loop
        elif key == ord("c"):
            if len(ref_points) == 2:
                break
            else:
                print("Please draw a box first.")

    cv2.destroyAllWindows()
    
    if len(ref_points) == 2:
        # Calculate pixel dimensions of the selected reference object
        pixel_width = abs(ref_points[1][0] - ref_points[0][0])
        pixel_height = abs(ref_points[1][1] - ref_points[0][1])
        
        # Calculate ratio: pixels per millimeter
        pixels_per_mm_w = pixel_width / reference_width_mm
        pixels_per_mm_h = pixel_height / reference_height_mm
        
        # We take the average or handle width/height separately if the camera aspect ratio is skewed
        avg_pixels_per_mm = (pixels_per_mm_w + pixels_per_mm_h) / 2.0
        
        print(f"\n--- Calibration Results ---")
        print(f"Reference Object Size: {reference_width_mm}mm x {reference_height_mm}mm")
        print(f"Object Pixel Size: {pixel_width}px x {pixel_height}px")
        print(f"Pixels per mm (Width): {pixels_per_mm_w:.4f}")
        print(f"Pixels per mm (Height): {pixels_per_mm_h:.4f}")
        print(f"Average Pixels per mm: {avg_pixels_per_mm:.4f}")
        
        # Save to a config file
        config = {
            "pixels_per_mm": avg_pixels_per_mm,
            "pixels_per_mm_w": pixels_per_mm_w,
            "pixels_per_mm_h": pixels_per_mm_h
        }
        
        with open("calibration_config.json", "w") as f:
            json.dump(config, f, indent=4)
            
        print("\nCalibration saved to 'calibration_config.json'")
        return avg_pixels_per_mm
    else:
        print("Calibration cancelled or failed.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate pixel-to-millimeter ratio")
    parser.add_argument("--image", type=str, required=True, help="Path to the image containing the reference object")
    parser.add_argument("--ref_width", type=float, required=True, help="Width of the reference object in millimeters")
    parser.add_argument("--ref_height", type=float, required=True, help="Height of the reference object in millimeters")
    
    args = parser.parse_args()
    
    calibrate_camera(args.image, args.ref_width, args.ref_height)
