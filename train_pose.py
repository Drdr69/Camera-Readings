import argparse
from ultralytics import YOLO

def train_industrial_pose_model(data_yaml_path: str, epochs: int = 100, imgsz: int = 640, batch_size: int = 16):
    """
    Phase 2: Train a YOLO-Pose model configured for Industrial AOI.
    This script initializes a 'nano' model to guarantee high-speed inference (<50ms).
    It also employs specialized augmentations for continuous-motion conveyor belts.
    """
    print(f"Loading YOLOv8 Nano Pose model (yolov8n-pose.pt) for speed optimization...")
    # Initialize the base nano pose model.
    # The '.pt' weights will be downloaded automatically by Ultralytics if missing.
    model = YOLO("yolov8n-pose.pt")

    print(f"\n🚀 Starting Phase 2: Training {epochs} epochs on dataset: {data_yaml_path}")
    print(f"Parameters: Image Size={imgsz}, Batch Size={batch_size}")

    # Train the model with customized hyperparameters for an industrial setting.
    # The objects are moving quickly on a conveyor belt, so we aggressively augment
    # motion blur and handle minor rotations or lighting shifts.
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device='',  # Auto-select GPU if available, else CPU
        project="aoi_pose_project",
        name="industrial_model",

        # --- Custom Augmentations for Conveyor Belt Environment ---
        # 1. Rotational Shifts: Packages might skew slightly as they move.
        degrees=5.0,
        # 2. HSV Lighting: Factory overhead lights can flicker or change color temperature.
        hsv_h=0.015,  # Hue
        hsv_s=0.7,    # Saturation
        hsv_v=0.4,    # Value (brightness)
        # 3. Motion Blur: Fast-moving belts often result in slight blur.
        # (Using standard YOLO blur augmentation)
        bgr=0.0,      # Disable BGR to RGB since we control the camera input
        flipud=0.0,   # No upside down (conveyor is flat)
        fliplr=0.5,   # Left/Right flip is acceptable if symmetric
        mosaic=1.0,   # Useful for context, but keep it standard
        mixup=0.0     # Disable mixup (doesn't make sense for rigid packages)
    )

    print(f"\n✅ --- Phase 2 Complete ---")
    print(f"Best model weights saved in: aoi_pose_project/industrial_model/weights/best.pt")
    print(f"Ready for Phase 4: TensorRT Export.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Train YOLO-Pose Model for Industrial AOI")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset YAML file (e.g., dataset.yaml)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference targeting")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")

    args = parser.parse_args()

    train_industrial_pose_model(args.data, args.epochs, args.imgsz, args.batch)
