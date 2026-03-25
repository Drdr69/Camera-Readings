import os
import random
import shutil
import argparse
from pathlib import Path

def setup_yolo_directories(base_dir: Path):
    """
    Creates the standard YOLO directory structure required for training:
    - images/train
    - images/val
    - labels/train
    - labels/val
    """
    dirs = [
        base_dir / "images" / "train",
        base_dir / "images" / "val",
        base_dir / "labels" / "train",
        base_dir / "labels" / "val"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def create_dataset_yaml(base_dir: Path):
    """
    Generates the dataset.yaml file specifically configured for
    1 class ('packaged_product') and 5 keypoints.
    """
    yaml_content = f"""# YOLO-Pose Dataset Configuration for Industrial AOI
path: {base_dir.absolute()} # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')

# Classes
names:
  0: packaged_product

# Keypoints Configuration
# 5 keypoints: 4 corners + 1 center point for artwork alignment
kpt_shape: [5, 3] # [number of keypoints, number of dim (x, y, visible)]
"""
    yaml_path = base_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"✅ Generated dataset.yaml at {yaml_path}")

def prepare_yolo_pose_dataset(source_image_dir: str, source_label_dir: str, dest_dir: str, sample_size: int = 500, split_ratio: float = 0.8):
    """
    Extracts a random subset of data from a source directory, splits it 80/20,
    and organizes it into a standard YOLO-Pose dataset structure.
    """
    src_img_path = Path(source_image_dir)
    src_lbl_path = Path(source_label_dir)
    dst_path = Path(dest_dir)

    if not src_img_path.exists():
        raise FileNotFoundError(f"Source image directory not found: {src_img_path}")

    # Setup YOLO directories
    print(f"Creating YOLO directory structure in {dst_path}...")
    setup_yolo_directories(dst_path)

    # Get all valid images
    # Assuming .jpg or .png for industrial cameras
    all_images = list(src_img_path.glob("*.jpg")) + list(src_img_path.glob("*.png"))

    if not all_images:
        print("⚠️ No images found in source directory.")
        return

    # Randomly sample to avoid over-representation of a specific time window
    sample_size = min(sample_size, len(all_images))
    print(f"Randomly sampling {sample_size} images out of {len(all_images)} available...")
    sampled_images = random.sample(all_images, sample_size)

    # Calculate split index (80/20 Train/Val split)
    split_idx = int(sample_size * split_ratio)
    train_images = sampled_images[:split_idx]
    val_images = sampled_images[split_idx:]

    print(f"Splitting data: {len(train_images)} Train | {len(val_images)} Validation")

    def copy_data(image_list, split_type):
        """Helper to copy images and their corresponding label files."""
        for img in image_list:
            # Determine destination paths
            dst_img_file = dst_path / "images" / split_type / img.name

            # The label should have the exact same name but .txt extension
            lbl_file = src_lbl_path / f"{img.stem}.txt"
            dst_lbl_file = dst_path / "labels" / split_type / f"{img.stem}.txt"

            # Copy image
            shutil.copy2(img, dst_img_file)

            # Copy label if it exists (highly required for Pose, but handle gracefully)
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_lbl_file)
            else:
                print(f"⚠️ Missing label file for {img.name}")

    print("Copying Training files...")
    copy_data(train_images, "train")

    print("Copying Validation files...")
    copy_data(val_images, "val")

    # Generate the yaml configuration
    create_dataset_yaml(dst_path)
    print("\n🚀 Dataset preparation complete! Ready for Phase 2 (Training).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Prepare Dataset for YOLO-Pose Industrial AOI")
    parser.add_argument("--src_imgs", type=str, required=True, help="Path to raw source images")
    parser.add_argument("--src_lbls", type=str, required=True, help="Path to raw source labels (.txt files)")
    parser.add_argument("--dest", type=str, default="./aoi_dataset", help="Destination directory for YOLO dataset")
    parser.add_argument("--sample", type=int, default=500, help="Number of random samples to extract")

    args = parser.parse_args()
    prepare_yolo_pose_dataset(args.src_imgs, args.src_lbls, args.dest, args.sample)
