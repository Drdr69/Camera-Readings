import cv2
import numpy as np
import argparse
import json
import os
from ultralytics import YOLO

def load_config(config_path="calibration_config.json"):
    """
    Load pixel-to-millimeter ratio from calibration config.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError("Calibration file 'calibration_config.json' not found. Run calibrate.py first.")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return config.get("pixels_per_mm", None), config.get("pixels_per_mm_w", None), config.get("pixels_per_mm_h", None)

def measure_box(image_path, model_path, calibration_path="calibration_config.json", output_dir="results"):
    """
    Uses instance segmentation model to perfectly trace irregular edges and measure bounding box.
    """
    # 1. Load the pre-trained or fine-tuned YOLOv8 segmentation model
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    # 2. Load calibration config
    pixels_per_mm, pixels_per_mm_w, pixels_per_mm_h = load_config(calibration_path)
    if pixels_per_mm is None:
        raise ValueError("Invalid calibration config.")
    
    # 3. Read image
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # 4. Run inference (predict the mask of the box)
    results = model(img)
    
    # Check if a box was detected
    if len(results[0].boxes) == 0:
        print("No box detected in the image.")
        return
    
    # Check if masks exist
    if results[0].masks is None:
        print("Objects detected, but no segmentation masks were generated.")
        return

    # We assume the first detected object is our target box
    # Extract the mask (the pixel-perfect blob covering the box)
    mask = results[0].masks.data[0].cpu().numpy()
    
    # Resize mask to original image dimensions (since YOLO might resize it during inference)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert mask to binary (0 and 255)
    binary_mask = (mask * 255).astype(np.uint8)
    
    # Find contours from the pixel-perfect mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No valid contour found for the box mask.")
        return
        
    # Get the largest contour (assuming it's the main box)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # --- DIMENSION EXTRACTION EXPLANATION ---
    # `cv2.minAreaRect` finds the smallest possible rotated rectangle that completely encloses
    # the contour (which represents our irregular shape). This is crucial because an irregular
    # box might not be perfectly axis-aligned with the camera. A standard bounding box
    # (`cv2.boundingRect`) would be too large if the box is tilted.
    # The output `rect` contains: (center(x, y), (width, height), angle of rotation)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # The dimensions are extracted from rect[1] which holds (width, height) in pixels
    pixel_width, pixel_height = rect[1]
    
    # --- PIXEL TO MILLIMETER CONVERSION ---
    # `pixels_per_mm` is a ratio calculated during the calibration step (`calibrate.py`).
    # By dividing the pixel measurement by this ratio, we convert the units to millimeters.
    # Example: If width is 600 pixels, and 1 mm = 10 pixels, 600 / 10 = 60 mm.
    width_mm = pixel_width / pixels_per_mm
    height_mm = pixel_height / pixels_per_mm
    
    print("\n--- Measurement Results ---")
    print(f"Pixel Width: {pixel_width:.2f}px")
    print(f"Pixel Height: {pixel_height:.2f}px")
    print(f"Calculated Width: {width_mm:.2f} mm")
    print(f"Calculated Height: {height_mm:.2f} mm")
    
    # --- Visualization ---
    result_img = img.copy()
    
    # Draw the contour (the irregular shape)
    cv2.drawContours(result_img, [largest_contour], -1, (0, 255, 0), 2)
    
    # Draw the bounding box
    cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)
    
    # Put text for dimensions
    text_w = f"Width: {width_mm:.2f}mm"
    text_h = f"Height: {height_mm:.2f}mm"
    
    # Add labels near the box
    cv2.putText(result_img, text_w, (int(box[0][0]), int(box[0][1] - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(result_img, text_h, (int(box[0][0]), int(box[0][1] - 35)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save the result
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"measured_{base_name}")
    cv2.imwrite(output_path, result_img)
    print(f"\nSaved measurement result to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure irregular boxes in millimeters using YOLO segmentation.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to measure")
    parser.add_argument("--model", type=str, default="yolov8x-seg.pt", help="Path to the trained YOLO segmentation model (.pt file) - Defaults to the most accurate model")
    parser.add_argument("--config", type=str, default="calibration_config.json", help="Path to calibration config JSON")
    
    args = parser.parse_args()
    
    measure_box(args.image, args.model, args.config)
