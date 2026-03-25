import cv2
import time
import argparse
import threading
import os
import json
from queue import Queue
from ultralytics import YOLO
from measurement_engine import MeasurementEngine

# =====================================================================================
# TENSORRT EXPORT UTILITY
# =====================================================================================

def export_to_tensorrt(model_path: str):
    """
    Converts a PyTorch (.pt) weights file into a highly optimized TensorRT engine (.engine).
    This is critical for edge deployment on factory floor NVIDIA GPUs to meet the <50ms latency.
    """
    print(f"Exporting {model_path} to TensorRT...")
    model = YOLO(model_path)

    # Export parameters:
    # `format="engine"` targets NVIDIA TensorRT.
    # `half=True` uses FP16 precision, drastically reducing memory bandwidth and increasing speed.
    # `simplify=True` removes redundant nodes in the computational graph before TRT conversion.
    # `imgsz=640` is our production inference resolution.
    model.export(format="engine", half=True, simplify=True, imgsz=640)
    print(f"✅ Export successful. The .engine file will be placed next to {model_path}")
    print(f"Please restart this script using the new .engine file for maximum speed.")


# =====================================================================================
# HIGH-SPEED PRODUCTION INFERENCE PIPELINE
# =====================================================================================

class ProductionPipeline:
    """
    Phase 4: Edge Optimization & Production Inference Loop.

    This implements a multi-threaded Producer-Consumer model to ensure that
    frame grabbing (I/O) does not block the Deep Learning Inference (GPU),
    and that inference does not block the Measurement/Math operations (CPU).
    """
    def __init__(self, model_path: str, source: int = 0, z_distance_mm: float = 500.0):
        self.source = source
        self.upload_dir = "uploaded_results"

        # Ensure upload directory exists
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)

        # Load the optimized model (preferably a .engine file, but handles .pt automatically)
        print(f"Loading YOLO-Pose model from {model_path}...")
        self.model = YOLO(model_path)

        # Initialize the deterministic measurement and color extraction module
        # with the intrinsic camera calibration logic
        self.measurement_engine = MeasurementEngine(z_distance_mm=z_distance_mm)

        # We use Queues to safely pass data between threads without locking
        # Maxsize=2 ensures we don't build up a huge backlog of stale frames if the conveyor stops.
        # We always want the "freshest" frame.
        self.frame_queue = Queue(maxsize=2)
        self.inference_queue = Queue(maxsize=2)
        self.display_queue = Queue(maxsize=2)

        # Control flags
        self.running = True

    def _thread_1_frame_grabber(self):
        """
        Thread 1: The Producer.
        Continuously reads frames from the industrial camera buffer.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("ERROR: Camera failed to initialize.")
            self.running = False
            return

        print("Thread 1: Camera Grabber Started.")
        while self.running:
            success, frame = cap.read()
            if success:
                # If the queue is full, drop the oldest frame to avoid latency pileup
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass

                self.frame_queue.put(frame)
            else:
                print("Warning: Dropped camera frame.")
                time.sleep(0.01)

        cap.release()

    def _thread_2_inference_engine(self):
        """
        Thread 2: The GPU Worker.
        Takes frames from the queue and runs the YOLO-Pose TensorRT engine.
        """
        print("Thread 2: Inference Engine Started.")
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

                # Run inference: verbose=False prevents console spam, stream=False processes immediately
                t1 = time.time()
                results = self.model(frame, verbose=False, stream=False)
                inference_time = (time.time() - t1) * 1000 # milliseconds

                if self.inference_queue.full():
                    try:
                        self.inference_queue.get_nowait()
                    except:
                        pass

                # Pass the raw frame, the YOLO results, and the latency metric to the next thread
                self.inference_queue.put((frame, results, inference_time))
            else:
                # Prevent CPU spin-locking if the camera is slow
                time.sleep(0.001)

    def _thread_3_measurement_and_logic(self):
        """
        Thread 3: The Consumer (Math & PLC Output).
        Extracts keypoints, calculates millimeters, checks Pass/Fail rules,
        draws diagnostics, and outputs the final frame.
        """
        print("Thread 3: Measurement & Logic Started.")
        while self.running:
            if not self.inference_queue.empty():
                frame, results, inference_time = self.inference_queue.get()

                pass_fail_status = "PASS"
                status_color = (0, 255, 0) # Green

                # Check if any objects were detected
                if len(results[0].boxes) > 0 and results[0].keypoints is not None:

                    # Assume one package per frame for this industrial station
                    box = results[0].boxes[0]
                    # Convert tensor coordinates to standard Python list of [x, y] tuples
                    kpts_tensor = results[0].keypoints.xy[0].cpu().numpy()

                    # Only proceed if we detected the required 5 keypoints
                    if len(kpts_tensor) >= 5:
                        # Extract the keypoints: Top-Left, Top-Right, Bottom-Right, Bottom-Left, Center
                        kpts = [(float(pt[0]), float(pt[1])) for pt in kpts_tensor]

                        # --- 1. Compute Dimensions (Math) ---
                        dims = self.measurement_engine.compute_dimensions(kpts)
                        width_mm = dims["width"]
                        height_mm = dims["height"]

                        # --- 2. Compute Asymmetry (Alignment) ---
                        asym_mm = self.measurement_engine.compute_asymmetry(kpts)

                        # --- 3. Extract Color (Lighting/Print Quality) ---
                        # Extract the bounding box coordinates (x1, y1, x2, y2)
                        bbox = box.xyxy[0].cpu().numpy()
                        color_data = self.measurement_engine.extract_color(
                            frame, (bbox[0], bbox[1]), (bbox[2], bbox[3])
                        )
                        avg_hsv = color_data["hsv"]

                        # --- HARDCODED PASS/FAIL LOGIC BLOCK ---
                        # In a real factory, these thresholds would come from a PLC or a database.
                        reasons = []
                        if width_mm > 220.0 or width_mm < 200.0:
                            reasons.append("Width Error")
                        if asym_mm > 15.0:
                            reasons.append("Misaligned Artwork")
                        if avg_hsv[2] < 100: # Value (Brightness) is too low
                            reasons.append("Too Dark (Lighting/Print)")

                        if reasons:
                            pass_fail_status = f"FAIL: {', '.join(reasons)}"
                            status_color = (0, 0, 255) # Red
                            # Trigger PLC communication here (e.g., Modbus/TCP or digital GPIO output)
                            # plc.send_reject_signal()

                        # --- UPLOAD RESULTS (Data Persistence) ---
                        # Save the processed image and physical dimensions for later review
                        # In production, this can easily be swapped with an HTTP POST request to a database
                        timestamp = int(time.time() * 1000)

                        # Pack measurement payload
                        payload = {
                            "timestamp": timestamp,
                            "status": pass_fail_status,
                            "width_mm": round(width_mm, 2),
                            "height_mm": round(height_mm, 2),
                            "asymmetry_mm": round(asym_mm, 2),
                            "color_hsv": list(avg_hsv)
                        }

                        json_path = os.path.join(self.upload_dir, f"result_{timestamp}.json")
                        with open(json_path, "w") as f:
                            json.dump(payload, f, indent=4)

                        if self.display_queue.full():
                            try:
                                self.display_queue.get_nowait()
                            except:
                                pass

                        # Pass the result details down the pipeline to the main thread for drawing and upload
                        self.display_queue.put((frame, bbox, kpts, status_color, pass_fail_status, width_mm, height_mm, asym_mm, avg_hsv, inference_time, timestamp))
                        continue

                # If no object was found, pass an empty result
                if self.display_queue.full():
                    try:
                        self.display_queue.get_nowait()
                    except:
                        pass
                self.display_queue.put((frame, None, None, None, None, None, None, None, None, inference_time, None))
            else:
                time.sleep(0.001)

    def start(self):
        """
        Spawns the three background threads (I/O Grabber, GPU Inference, CPU Math),
        while running the GUI and upload saving in the Main Thread to guarantee thread-safety.
        """
        print("\n--- Starting Industrial AOI Pipeline ---")
        print("Press 'q' on the video window to stop.")

        t1 = threading.Thread(target=self._thread_1_frame_grabber, daemon=True)
        t2 = threading.Thread(target=self._thread_2_inference_engine, daemon=True)
        t3 = threading.Thread(target=self._thread_3_measurement_and_logic, daemon=True)

        t1.start()
        t2.start()
        t3.start()

        try:
            # Main Thread: UI and File Upload saving
            # This ensures cv2.imshow runs securely on the main loop.
            while self.running:
                if not self.display_queue.empty():
                    # Extract the processed data from Thread 3
                    frame, bbox, kpts, status_color, pass_fail_status, width_mm, height_mm, asym_mm, avg_hsv, inference_time, timestamp = self.display_queue.get()

                    display_frame = frame.copy()

                    # If an object was detected, draw the diagnostics
                    if bbox is not None:
                        # Draw bounding box
                        cv2.rectangle(display_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), status_color, 2)

                        # Draw keypoints (blue dots)
                        for pt in kpts:
                            if pt[0] != 0 and pt[1] != 0: # Ensure valid point
                                cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)

                        # Draw Text Overlays
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(display_frame, pass_fail_status, (20, 40), font, 1.0, status_color, 3)
                        cv2.putText(display_frame, f"W: {width_mm:.1f}mm, H: {height_mm:.1f}mm", (20, 80), font, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"Asym: {asym_mm:.1f}mm", (20, 110), font, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"HSV: {avg_hsv}", (20, 140), font, 0.7, (255, 255, 255), 2)

                        # --- UPLOAD: Save the Annotated Image ---
                        img_path = os.path.join(self.upload_dir, f"result_{timestamp}.jpg")
                        cv2.imwrite(img_path, display_frame)

                    # Display the total latency (including YOLO)
                    cv2.putText(display_frame, f"Inference Latency: {inference_time:.1f}ms", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Safe to call in Main Thread
                    cv2.imshow("Industrial AOI Output", display_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                else:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt caught. Shutting down...")
            self.running = False

        print("Waiting for threads to close...")
        cv2.destroyAllWindows()
        print("Pipeline successfully terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Industrial AOI Production Inference")
    parser.add_argument("--export", type=str, default="", help="If provided, exports the given .pt model to TensorRT (.engine) and exits.")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="Path to the trained YOLO-Pose model (use .engine for production)")
    parser.add_argument("--source", type=int, default=0, help="Camera index (e.g., 0 for USB, or industrial GigE interface)")
    parser.add_argument("--z_distance", type=float, default=500.0, help="Z-axis distance from camera to conveyor belt in mm")

    args = parser.parse_args()

    if args.export:
        export_to_tensorrt(args.export)
    else:
        pipeline = ProductionPipeline(model_path=args.model, source=args.source, z_distance_mm=args.z_distance)
        pipeline.start()
