"""Download COCO val2017 and convert to UrbanEye 5-class YOLO format.

Downloads COCO validation set (5000 images, ~1GB) and filters for
the 5 UrbanEye driving classes: vehicle, pedestrian, cyclist,
traffic_light, traffic_sign.

This serves as the baseline training dataset before CARLA data
generation is available.
"""

import json
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# UrbanEye class definitions
URBANEYE_CLASSES = ["vehicle", "pedestrian", "cyclist", "traffic_light", "traffic_sign"]

# COCO category ID -> UrbanEye class ID mapping
COCO_TO_URBANEYE = {
    1: 1,  # person -> pedestrian
    2: 2,  # bicycle -> cyclist
    3: 0,  # car -> vehicle
    4: 0,  # motorcycle -> vehicle
    6: 0,  # bus -> vehicle
    8: 0,  # truck -> vehicle
    10: 3,  # traffic light -> traffic_light
    13: 4,  # stop sign -> traffic_sign
}

DATA_DIR = Path(os.path.expanduser("~/urbaneye/data"))
COCO_IMAGES_URL = "https://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name}...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            sys.stdout.write(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=reporthook)
    print()


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a zip file."""
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


def convert_coco_to_yolo(
    coco_json_path: Path,
    images_dir: Path,
    output_dir: Path,
    split: str = "val",
) -> dict:
    """Convert COCO annotations to YOLO format for our 5 classes."""
    print(f"\nConverting COCO -> YOLO ({split})...")

    with open(coco_json_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Build image ID -> filename mapping
    id_to_file = {}
    id_to_size = {}
    for img in coco["images"]:
        id_to_file[img["id"]] = img["file_name"]
        id_to_size[img["id"]] = (img["width"], img["height"])

    # Group annotations by image
    img_annotations: dict[int, list] = {}
    for ann in coco["annotations"]:
        coco_cat = ann["category_id"]
        if coco_cat not in COCO_TO_URBANEYE:
            continue
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # Create output directories
    img_out = output_dir / "images" / split
    lbl_out = output_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = {name: 0 for name in URBANEYE_CLASSES}
    images_copied = 0
    skipped = 0

    for img_id, anns in img_annotations.items():
        filename = id_to_file.get(img_id)
        if filename is None:
            continue

        src_img = images_dir / filename
        if not src_img.exists():
            skipped += 1
            continue

        w, h = id_to_size[img_id]
        yolo_lines = []

        for ann in anns:
            coco_cat = ann["category_id"]
            urbaneye_cls = COCO_TO_URBANEYE.get(coco_cat)
            if urbaneye_cls is None:
                continue

            # COCO bbox: [x, y, width, height] (absolute pixels)
            bx, by, bw, bh = ann["bbox"]

            # Skip tiny boxes
            if bw < 2 or bh < 2:
                continue

            # Convert to YOLO: center_x, center_y, width, height (normalized)
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h

            # Clamp to [0, 1]
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))

            yolo_lines.append(f"{urbaneye_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[URBANEYE_CLASSES[urbaneye_cls]] += 1

        if yolo_lines:
            # Copy image
            dst_img = img_out / filename
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # Write label
            stem = Path(filename).stem
            lbl_path = lbl_out / f"{stem}.txt"
            with open(lbl_path, "w") as f:
                f.write("\n".join(yolo_lines))

            images_copied += 1

    print(f"  Images with driving objects: {images_copied} (skipped {skipped})")
    print("  Class distribution:")
    for name, count in stats.items():
        print(f"    {name}: {count}")

    return stats


def create_dataset_yaml(output_dir: Path) -> Path:
    """Create YOLO dataset.yaml config file."""
    yaml_content = f"""# UrbanEye Baseline Dataset (COCO-derived, 5 driving classes)
path: {output_dir}
train: images/train
val: images/val

nc: 5
names:
  0: vehicle
  1: pedestrian
  2: cyclist
  3: traffic_light
  4: traffic_sign
"""
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nDataset config written to: {yaml_path}")
    return yaml_path


def split_train_val(output_dir: Path, val_ratio: float = 0.2) -> None:
    """Split the val set into train/val for our purposes.

    COCO val2017 has 5000 images. We use 80% for training, 20% for validation.
    """
    random.seed(42)

    val_images = sorted((output_dir / "images" / "val").glob("*.jpg"))
    val_labels = output_dir / "labels" / "val"

    if not val_images:
        print("No images to split!")
        return

    # Create train directories
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.shuffle(val_images)
    split_idx = int(len(val_images) * (1 - val_ratio))

    train_imgs = val_images[:split_idx]

    moved = 0
    for img_path in train_imgs:
        stem = img_path.stem
        lbl_path = val_labels / f"{stem}.txt"

        # Move image
        dst_img = train_img_dir / img_path.name
        shutil.move(str(img_path), str(dst_img))

        # Move label
        if lbl_path.exists():
            dst_lbl = train_lbl_dir / lbl_path.name
            shutil.move(str(lbl_path), str(dst_lbl))
            moved += 1

    remaining_val = len(val_images) - len(train_imgs)
    print(f"\nDataset split: {moved} train / {remaining_val} val")


def main():
    print("=" * 60)
    print("UrbanEye Dataset Preparation")
    print("COCO val2017 -> 5-class driving dataset")
    print("=" * 60)

    downloads = DATA_DIR / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    # Step 1: Download COCO val2017 images
    print("\n[1/5] Downloading COCO val2017 images (~1GB)...")
    img_zip = downloads / "val2017.zip"
    download_file(COCO_IMAGES_URL, img_zip)

    # Step 2: Download COCO annotations
    print("\n[2/5] Downloading COCO annotations (~252MB)...")
    ann_zip = downloads / "annotations_trainval2017.zip"
    download_file(COCO_ANNOTS_URL, ann_zip)

    # Step 3: Extract
    print("\n[3/5] Extracting...")
    coco_dir = DATA_DIR / "coco"
    if not (coco_dir / "val2017").exists():
        extract_zip(img_zip, coco_dir)
    if not (coco_dir / "annotations").exists():
        extract_zip(ann_zip, coco_dir)

    # Step 4: Convert to YOLO format
    print("\n[4/5] Converting to YOLO format...")
    output_dir = DATA_DIR / "urbaneye_baseline"
    stats = convert_coco_to_yolo(
        coco_json_path=coco_dir / "annotations" / "instances_val2017.json",
        images_dir=coco_dir / "val2017",
        output_dir=output_dir,
        split="val",
    )

    # Step 5: Split into train/val
    print("\n[5/5] Splitting into train/val (80/20)...")
    split_train_val(output_dir, val_ratio=0.2)

    # Create dataset.yaml
    create_dataset_yaml(output_dir)

    # Summary
    total_objects = sum(stats.values())
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"Total driving objects: {total_objects}")
    print(f"Dataset location: {output_dir}")
    print(f"Config: {output_dir / 'dataset.yaml'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
