# Industrial Automated Optical Inspection (AOI) Pipeline

This project implements a highly optimized, end-to-end Deep Learning pipeline using **YOLO-Pose**, **TensorRT**, and **OpenCV**. It is designed for high-speed, continuous-motion conveyor belts where latency must be strictly kept under 50ms per frame.

This documentation serves as a comprehensive guide on how to add images, train the model, tune parameters for different results, and deploy the system.

---

## The Workflow: 4 Phases

The architecture is strictly divided into four distinct phases. You execute these python scripts in order.

### Phase 1: Data Preparation (`data_prep.py`)
This script takes your messy raw images and labels and perfectly structures them for YOLO training.

**How to add images:**
1. Collect raw images from your factory camera (e.g., `.jpg` or `.png`).
2. Label them using a tool like [CVAT](https://www.cvat.ai/) or [Roboflow](https://roboflow.com/). You must label **1 bounding box** (the package) and **5 Keypoints** in this exact order:
   - Keypoint 0: Top-Left Corner
   - Keypoint 1: Top-Right Corner
   - Keypoint 2: Bottom-Right Corner
   - Keypoint 3: Bottom-Left Corner
   - Keypoint 4: Center Artwork/Label (used for asymmetry measurement)
3. Export the annotations in **YOLO-Pose format** (.txt files).
4. Place all raw images in a folder (e.g., `raw_data/images`) and all label texts in another folder (e.g., `raw_data/labels`).

**Running Phase 1:**
```bash
python data_prep.py --src_imgs path/to/raw_data/images --src_lbls path/to/raw_data/labels --dest ./aoi_dataset --sample 500
```
* **What it does:** It randomly samples 500 images (to avoid overfitting on a specific time period), splits them 80/20 into train/validation sets, builds the `images/train`, `labels/train` folders inside `./aoi_dataset`, and auto-generates the `dataset.yaml` file.
* **Important Paths:** The output folder (default `./aoi_dataset`) is what you will use for the next phase.

---

### Phase 2: Model Training (`train_pose.py`)
This script trains the lightweight `yolov8n-pose.pt` model using custom augmentations designed for conveyor belts.

**Running Phase 2:**
```bash
python train_pose.py --data ./aoi_dataset/dataset.yaml --epochs 100 --imgsz 640 --batch 16
```

**Parameters to change for different results (Inside `train_pose.py`):**
Open `train_pose.py` and locate the `model.train()` block to tune these industrial augmentations:
* `degrees=5.0`: Change this if your packages rotate significantly on the belt (e.g., `15.0`).
* `hsv_v=0.4`: Modifies brightness variation. Increase if your factory lighting fluctuates drastically.
* `imgsz`: Default is `640`. If you need to detect extremely tiny print features, you can increase this to `1280` (but inference will be slower).
* **Model Selection:** The script hardcodes `yolov8n-pose.pt` (Nano). If you have a powerful GPU and need higher accuracy, change it to `yolov8s-pose.pt` (Small) or `yolov8m-pose.pt` (Medium).

**Important Paths:** Once training finishes, the best weights are saved in:
`aoi_pose_project/industrial_model/weights/best.pt`.

---

### Phase 3: The Math Engine (`measurement_engine.py`)
You don't run this file directly. It is imported by Phase 4. It contains the core OpenCV deterministic mathematics.

**Critical Calibration Parameters:**
Currently, `measurement_engine.py` uses a placeholder 1080p intrinsic camera matrix. **To get accurate millimeters**, you must calibrate your specific physical camera using a checkerboard pattern (`cv2.calibrateCamera`).

Once calibrated, open `measurement_engine.py` and update the `__init__` function with your true hardware values:
```python
self.camera_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float32)

self.dist_coeffs = np.array([k1, k2, p1, p2, k3])
```

---

### Phase 4: Production Inference (`production_inference.py`)
This is the main script that runs endlessly on the factory floor. It uses a high-speed, multi-threaded Producer-Consumer loop to ensure the camera never drops frames while the GPU works.

**Step 4A: Export to TensorRT (Required for speed)**
```bash
python production_inference.py --export aoi_pose_project/industrial_model/weights/best.pt
```
This converts your PyTorch `.pt` file into a `.engine` file optimized for your specific NVIDIA GPU in FP16 precision.

**Step 4B: Run the Factory Loop**
```bash
python production_inference.py --model aoi_pose_project/industrial_model/weights/best.engine --source 0 --z_distance 500.0
```

**Parameters to change:**
* `--source`: `0` is usually your first USB webcam. Change to a GigE IP address (e.g., `"rtsp://192.168.1.10"`) for industrial cameras.
* `--z_distance`: The physical distance from your camera lens to the conveyor belt in millimeters. **This is strictly required for the math to convert 2D pixels to 3D millimeters.** If you raise the camera higher, you must update this number.

**Where do the results go?**
As requested, the pipeline continuously saves the processed frames (with drawn diagnostics) and a JSON file containing the physical dimensions (Width, Height, Asymmetry, HSV Color) to the `uploaded_results/` folder. In a production environment, you can modify the JSON dump block to execute an HTTP POST request to your central database.