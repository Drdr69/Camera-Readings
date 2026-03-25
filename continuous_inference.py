import cv2
import numpy as np
import argparse
import os
import json
from ultralytics import YOLO
import time
import glob

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_config(config_path="calibration_config.json"):
    """
    Reads the calibration file to figure out how many pixels equal one millimeter.
    If you haven't run `calibrate.py` yet, this file won't exist, and the program 
    will just measure things in raw 'pixels' instead of 'millimeters'.
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
    Simulates uploading the processed image and measurement data.
    Saves the image with drawn bounding boxes and the measured dimensions
    (height and width in millimeters) to an 'uploaded_results' directory.
    
    The 'image' is the picture with the boxes drawn on it.
    The 'measurement_data' is a dictionary containing the width/height numbers.
    """
    output_dir = "uploaded_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a unique timestamp for filenames
    timestamp = int(time.time() * 1000)
    image_filename = os.path.join(output_dir, f"result_{timestamp}.jpg")
    json_filename = os.path.join(output_dir, f"data_{timestamp}.json")

    # 1. Save the processed image to disk
    cv2.imwrite(image_filename, image)

    # 2. Save the measurement data as JSON
    with open(json_filename, "w") as f:
        json.dump(measurement_data, f, indent=4)

    print(f"Uploaded! Saved image to {image_filename} and data to {json_filename}")

    # --- EXAMPLE OF REAL UPLOAD VIA HTTP POST (COMMENTED OUT) ---
    # import requests
    #
    # url = "https://your-api-endpoint.com/upload"
    # payload = {'data': json.dumps(measurement_data)}
    # files = [
    #   ('image', ('image.jpg', open(image_filename, 'rb'), 'image/jpeg'))
    # ]
    # headers = {'Authorization': 'Bearer YOUR_TOKEN'}
    #
    # try:
    #     response = requests.post(url, headers=headers, data=payload, files=files)
    #     if response.status_code == 200:
    #         print("Successfully uploaded to remote server!")
    #     else:
    #         print(f"Failed to upload. Status code: {response.status_code}")
    # except Exception as e:
    #     print(f"Error uploading to server: {e}")
    # -------------------------------------------------------------


# ==============================================================================
# MAIN PROCESSING LOGIC
# ==============================================================================

def process_and_upload_frame(frame, model, pixels_per_mm):
    """
    This function takes a single picture (frame), asks the AI (YOLO model) to 
    find objects in it, calculates their sizes, and then pretends to upload them.
    
    Returns:
        result_img: The picture with the colored boxes and text drawn on it.
    """
    # 1. Ask the AI model to look at the picture and find shapes.
    # verbose=False tells the AI to stay quiet and not spam the terminal with text.
    results = model(frame, verbose=False) 
    
    # We will store all the size data we find in this dictionary
    measurement_data = {
        "boxes_detected": 0,
        "measurements": []
    }
    
    # 2. Make a copy of the picture so we can draw colorful boxes on it later
    result_img = frame.copy()
    
    # 3. Check if the AI actually found anything
    if len(results[0].boxes) > 0:
        measurement_data["boxes_detected"] = len(results[0].boxes)
        
        # Extract the "masks" (the pixel-perfect colorful blobs the AI drew over the objects)
        masks_data = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []
        
        if len(masks_data) == 0:
            print("Notice: Objects detected, but no segmentation masks were generated.")

        # Loop through every single object the AI found in the picture
        for idx, mask in enumerate(masks_data):
            # The AI might have squished the mask, so we stretch it back to the picture's original size
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Convert the mask into a simple black-and-white silhouette
            binary_mask = (mask_resized * 255).astype(np.uint8)
            
            # Find the outline (contour) of the silhouette
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            # Get the biggest outline (to ignore tiny specks of noise)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # --- DIMENSION EXTRACTION EXPLANATION ---
            # cv2.minAreaRect generates the smallest enclosing rotated rectangle.
            # This handles irregular boxes at an angle, preventing a standard
            # axis-aligned bounding box from being excessively large.
            # It provides: ((center_x, center_y), (width, height), angle)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Extract the raw width and height in pixels from the rectangle
            pixel_width, pixel_height = rect[1]
            
            # --- PIXEL TO MILLIMETER CONVERSION ---
            # Uses the average pixel-to-millimeter ratio from calibration_config.json.
            # We divide the pixel width/height by the ratio (pixels/mm) to get millimeters.
            # If `calibrate.py` hasn't been run, `pixels_per_mm` is None, and we fall back to pixels.
            width_val = pixel_width / pixels_per_mm if pixels_per_mm else pixel_width
            height_val = pixel_height / pixels_per_mm if pixels_per_mm else pixel_height
            unit = "mm" if pixels_per_mm else "px"
            
            # Save these numbers into our data dictionary so we can upload them later
            measurement_data["measurements"].append({
                "id": idx,
                "width": width_val,
                "height": height_val,
                "unit": unit
            })
            
            # === DRAWING VISUAL FEEDBACK ===
            # Draw a green line perfectly tracing the odd shape of the object
            cv2.drawContours(result_img, [largest_contour], -1, (0, 255, 0), 2)
            # Draw a red box around the object
            cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)
            
            # Write the width and height text above the box in blue
            text_w = f"W: {width_val:.2f}{unit}"
            text_h = f"H: {height_val:.2f}{unit}"
            cv2.putText(result_img, text_w, (int(box[0][0]), int(box[0][1] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(result_img, text_h, (int(box[0][0]), int(box[0][1] - 30)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 4. Give the final picture and the numbers to our upload function
    upload_data(image=result_img, measurement_data=measurement_data)
    
    return result_img


# ==============================================================================
# THE INFINITE LOOP
# ==============================================================================

def continuous_inference(model_path="yolov8x-seg.pt", calibration_path="calibration_config.json", source="0"):
    """
    This is the main function that never stops running.
    It can read from either a webcam, OR a folder on your computer (like a Google Drive folder).
    """
    print(f"Loading AI Model from {model_path}...")
    model = YOLO(model_path)
    
    # Load our pixel-to-millimeter ratio
    pixels_per_mm, _, _ = load_config(calibration_path)
    
    # Check if the user typed "0" (for a camera) or a folder path (like "C:/Google Drive/Images")
    if source.isdigit():
        is_camera = True
        print(f"Opening webcam #{source}...")
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            print("ERROR: Could not open the webcam. Make sure it is plugged in.")
            return
    else:
        is_camera = False
        print(f"Monitoring folder: '{source}' for new images...")
        if not os.path.exists(source):
            print(f"ERROR: The folder '{source}' does not exist. Please create it first.")
            return
        # We need to remember which images we've already uploaded so we don't upload them twice!
        processed_images = set()

    print("\n--- Starting Continuous Program ---")
    print("Press the 'q' key on your keyboard to stop the program.\n")
    
    try:
        # 'while True' means this code will loop forever and never shut down naturally
        while True:
            
            # ==========================================
            # MODE A: WEBCAM
            # ==========================================
            if is_camera:
                # Take one picture from the camera
                success, frame = cap.read()
                if not success:
                    print("Failed to grab a picture from the camera. Trying again...")
                    time.sleep(1)
                    continue
                
                # Process the picture (Find the boxes, draw them, and upload them)
                result_img = process_and_upload_frame(frame, model, pixels_per_mm)
                
                # Show the picture on the screen live
                cv2.imshow("Live Camera Feed", result_img)
                
                # Wait 1 millisecond. If the user presses 'q', break out of the infinite loop to stop.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nUser pressed 'q'. Stopping the program...")
                    break
            
            # ==========================================
            # MODE B: GOOGLE DRIVE FOLDER (OR LOCAL DIRECTORY)
            # ==========================================
            else:
                # Look inside the folder for any file ending in .jpg or .png
                search_pattern = os.path.join(source, "*.*")
                all_files = glob.glob(search_pattern)
                
                # Filter out anything that isn't an image, and anything we already processed
                new_images = []
                for file in all_files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file not in processed_images:
                        new_images.append(file)
                
                # If we found brand-new images in the folder...
                if len(new_images) > 0:
                    for img_path in new_images:
                        print(f"New image found! Processing: {os.path.basename(img_path)}")
                        
                        # Read the image file from the computer
                        frame = cv2.imread(img_path)
                        if frame is not None:
                            # Process it (Find boxes, draw them, and upload them)
                            result_img = process_and_upload_frame(frame, model, pixels_per_mm)
                            
                            # Show the picture on the screen
                            cv2.imshow("Folder Monitoring Feed", result_img)
                            cv2.waitKey(1) # Refresh the window
                            
                            # Remember that we processed this image so we don't do it again
                            processed_images.add(img_path)
                        else:
                            print(f"Could not read {img_path}. It might be corrupted.")
                
                # Wait for 1 second before checking the Google Drive folder again for new files
                time.sleep(1)
                
                # If the user presses 'q', stop the program.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nUser pressed 'q'. Stopping the program...")
                    break

    except KeyboardInterrupt:
        # If the user presses CTRL+C in the terminal, shut down gracefully
        print("\nProgram interrupted by user. Shutting down...")
        
    finally:
        # --- CLEAN UP ---
        # When breaking out of the loop, close the camera and the video windows
        print("Cleaning up resources...")
        if is_camera:
            cap.release()
        cv2.destroyAllWindows()


# ==============================================================================
# SCRIPT STARTING POINT
# ==============================================================================
if __name__ == "__main__":
    # This section sets up the commands you can type when running the script.
    parser = argparse.ArgumentParser(description="Continuous measurement pipeline. Reads from Webcam or a Folder.")
    
    # --model: Lets you change which AI model file you are using. Default is the most accurate version.
    parser.add_argument("--model", type=str, default="yolov8x-seg.pt",
                        help="Path to the trained YOLO segmentation model (.pt file). Default uses max accuracy model.")
    
    # --config: Lets you specify the calibration file
    parser.add_argument("--config", type=str, default="calibration_config.json", 
                        help="Path to calibration config JSON")
    
    # --source: The MOST IMPORTANT ONE. "0" means webcam. A folder path means "monitor this folder".
    parser.add_argument("--source", type=str, default="0", 
                        help="Source of images. Use '0' for webcam, or provide a folder path (like 'C:/Google Drive/Box Images') to monitor that folder for new images.")
    
    args = parser.parse_args()
    
    # Start the infinite loop!
    continuous_inference(args.model, args.config, args.source)
