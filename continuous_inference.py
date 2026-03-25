import cv2
import numpy as np
import argparse
import os
import json
from ultralytics import YOLO
import time

def load_config(config_path="calibration_config.json"):
    """
    Load pixel-to-millimeter ratio from calibration config.
    """
    if not os.path.exists(config_path):
        print("Warning: Calibration file 'calibration_config.json' not found. "
              "Measurements will be in pixels.")
        return None, None, None
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return config.get("pixels_per_mm", None), config.get("pixels_per_mm_w", None), config.get("pixels_per_mm_h", None)

def upload_data(image, measurement_data):
    """
    Placeholder function for uploading data and image.
    Update this function when the upload destination is known 
    (e.g., AWS S3, REST API, FTP, database, etc.)
    """
    # To encode the image for sending over HTTP:
    # _, img_encoded = cv2.imencode('.jpg', image)
    # response = requests.post("https://api.example.com/upload", files={"image": img_encoded.tobytes()}, data=measurement_data)
    
    # Placeholder: Do nothing yet. Kept empty until the destination is known.
    pass

def continuous_inference(model_path="yolov8n-seg.pt", calibration_path="calibration_config.json"):
    """
    Runs a never-ending loop capturing from the camera, running deep learning model, 
    and uploading the results continuously.
    """
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    pixels_per_mm, _, _ = load_config(calibration_path)
    
    print("Opening video capture (camera 0)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open video device (camera 0). Is a webcam connected?")
        
    print("--- Starting continuous inference loop ---")
    print("Press 'q' in the camera window to gracefully stop the program.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera. Retrying...")
                time.sleep(1)
                continue
            
            # Run deep-learning inference
            results = model(frame, verbose=False) # verbose=False to keep terminal logs clean
            
            measurement_data = {
                "boxes_detected": 0,
                "measurements": []
            }
            
            # Make a copy of the camera frame to draw bounding boxes on
            result_img = frame.copy()
            
            if len(results[0].boxes) > 0:
                # Iterate through all detected objects
                measurement_data["boxes_detected"] = len(results[0].boxes)
                
                # Extract pixel-perfect masks
                masks_data = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
                
                for idx, mask in enumerate(masks_data):
                    # Resize mask to original image dimensions to draw over the frame
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (mask_resized * 255).astype(np.uint8)
                    
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                        
                    largest_contour = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(largest_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    
                    pixel_width, pixel_height = rect[1]
                    
                    # Convert to millimeters if calibration was found
                    width_val = pixel_width / pixels_per_mm if pixels_per_mm else pixel_width
                    height_val = pixel_height / pixels_per_mm if pixels_per_mm else pixel_height
                    unit = "mm" if pixels_per_mm else "px"
                    
                    measurement_data["measurements"].append({
                        "id": idx,
                        "width": width_val,
                        "height": height_val,
                        "unit": unit
                    })
                    
                    # Draw visual feedback
                    cv2.drawContours(result_img, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)
                    
                    text_w = f"W: {width_val:.2f}{unit}"
                    text_h = f"H: {height_val:.2f}{unit}"
                    
                    cv2.putText(result_img, text_w, (int(box[0][0]), int(box[0][1] - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(result_img, text_h, (int(box[0][0]), int(box[0][1] - 30)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # --- UPLOAD COMPONENT ---
            # Call our placeholder function with the annotated image and measurement payload.
            upload_data(image=result_img, measurement_data=measurement_data)
            
            # Show the live feed feedback continuously
            cv2.imshow("Continuous Measurement Feed", result_img)
            
            # Exit loop gracefully if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit signal received. Shutting down...")
                break
                
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down gracefully...")
        
    finally:
        print("Releasing camera and closing windows.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live camera instance segmentation and measurement.")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Path to the trained YOLO segmentation model (.pt file)")
    parser.add_argument("--config", type=str, default="calibration_config.json", help="Path to calibration config JSON")
    
    args = parser.parse_args()
    
    continuous_inference(args.model, args.config)
