import argparse
from ultralytics import YOLO

def train_model(data_yaml, epochs=100, imgsz=640, batch_size=16):
    """
    Train a YOLOv8 instance segmentation model on custom data.
    Instance segmentation is ideal for irregular shapes as it provides a pixel-perfect mask.
    """
    print(f"Loading pre-trained YOLOv8 segmentation model...")
    # Load a pre-trained segmentation model. 
    # 'yolov8n-seg.pt' is fast, but you can change to 'yolov8x-seg.pt' for maximum accuracy if you have a good GPU.
    model = YOLO('yolov8n-seg.pt')

    print(f"Starting training for {epochs} epochs on dataset: {data_yaml}")
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device='', # Automatically choose GPU if available, else CPU
        project='box_measurement',
        name='segmentation_model'
    )
    
    # After training, the best weights will be saved in the run directory.
    print(f"\n--- Training Complete ---")
    print(f"Check the 'box_measurement/segmentation_model' directory for training logs and the best model weights ('weights/best.pt').")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation Model for Irregular Boxes")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset YAML file (e.g., dataset.yaml)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    train_model(args.data, args.epochs, args.imgsz, args.batch)
