"""Domain adaptation pipeline for UrbanEye.

Implements Layer 3 of the 3-layer domain adaptation strategy:
mixed CARLA (80%) + BDD100K (20%) fine-tuning with backbone freezing.

Layer 1: Randomized CARLA weather/lighting (Phase 3)
Layer 2: Albumentations augmentation pipeline (Phase 4)
Layer 3: Mixed training data with fine-tuning (this module)

BDD100K has 10 detection classes that map to our 5 UrbanEye classes.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from urbaneye.utils.constants import CLASS_NAMES
from urbaneye.utils.io_helpers import ensure_dir

# BDD100K category name -> UrbanEye class ID mapping
BDD100K_CLASS_MAPPING: dict[str, int] = {
    "car": 0,  # vehicle
    "truck": 0,  # vehicle
    "bus": 0,  # vehicle
    "train": 0,  # vehicle (rare)
    "pedestrian": 1,  # pedestrian
    "rider": 2,  # cyclist
    "bicycle": 2,  # cyclist
    "motorcycle": 2,  # cyclist
    "traffic light": 3,  # traffic_light
    "traffic sign": 4,  # traffic_sign
}

# All 10 BDD100K detection classes
BDD100K_ALL_CLASSES: list[str] = [
    "car",
    "truck",
    "bus",
    "train",
    "pedestrian",
    "rider",
    "bicycle",
    "motorcycle",
    "traffic light",
    "traffic sign",
]


@dataclass
class DomainAdaptConfig:
    """Fine-tuning configuration for domain adaptation.

    Uses 10x lower learning rate and frozen backbone to prevent
    catastrophic forgetting of features learned from CARLA data.
    """

    base_weights: str = "best.pt"
    epochs: int = 20
    lr0: float = 0.0001  # 10x lower than initial training
    lrf: float = 0.01
    freeze_layers: int = 10  # Freeze first 10 backbone layers
    carla_ratio: float = 0.8
    bdd100k_ratio: float = 0.2
    img_size: int = 640
    batch_size: int = 16
    momentum: float = 0.937
    weight_decay: float = 0.0005
    amp: bool = True
    device: str = "0"
    workers: int = 4
    patience: int = 10
    project: str = "runs/detect"
    name: str = "urbaneye_adapted"

    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors: list[str] = []
        if self.epochs <= 0:
            errors.append(f"epochs must be positive, got {self.epochs}")
        if not (0 < self.lr0 <= 0.01):
            errors.append(f"lr0 should be low for fine-tuning (0, 0.01], got {self.lr0}")
        if self.freeze_layers < 0:
            errors.append(f"freeze_layers must be non-negative, got {self.freeze_layers}")
        if not (0 < self.carla_ratio < 1):
            errors.append(f"carla_ratio must be in (0, 1), got {self.carla_ratio}")
        if not (0 < self.bdd100k_ratio < 1):
            errors.append(f"bdd100k_ratio must be in (0, 1), got {self.bdd100k_ratio}")
        if abs(self.carla_ratio + self.bdd100k_ratio - 1.0) > 0.001:
            errors.append(
                f"carla_ratio + bdd100k_ratio must equal 1.0, "
                f"got {self.carla_ratio} + {self.bdd100k_ratio} = {self.carla_ratio + self.bdd100k_ratio}"
            )
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        return errors

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        """Convert to Ultralytics model.train() kwargs."""
        return {
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.img_size,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "amp": self.amp,
            "device": self.device,
            "workers": self.workers,
            "patience": self.patience,
            "project": self.project,
            "name": self.name,
            "freeze": self.freeze_layers,
            "plots": True,
        }


class BDD100KAdapter:
    """Convert BDD100K detection annotations to YOLO format.

    BDD100K uses JSON annotations with 10 object categories.
    This adapter maps them to UrbanEye's 5 driving classes.
    """

    def __init__(self) -> None:
        self.class_mapping = BDD100K_CLASS_MAPPING
        self.stats: dict[str, int] = {name: 0 for name in CLASS_NAMES}

    def convert_annotations(self, bdd_json_path: Path, output_dir: Path) -> int:
        """Convert BDD100K JSON annotations to YOLO .txt format.

        Args:
            bdd_json_path: Path to BDD100K detection JSON file.
            output_dir: Directory to write YOLO label files.

        Returns:
            Number of images with valid annotations.
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        with open(bdd_json_path, encoding="utf-8") as f:
            bdd_data = json.load(f)

        converted = 0
        for frame in bdd_data:
            frame_name = frame.get("name", "")
            labels = frame.get("labels", [])

            yolo_lines = []
            for label in labels:
                category = label.get("category", "")
                urbaneye_cls = self.class_mapping.get(category)
                if urbaneye_cls is None:
                    continue

                box2d = label.get("box2d")
                if box2d is None:
                    continue

                x1 = float(box2d.get("x1", 0))
                y1 = float(box2d.get("y1", 0))
                x2 = float(box2d.get("x2", 0))
                y2 = float(box2d.get("y2", 0))

                # BDD100K images are 1280x720
                img_w, img_h = 1280, 720
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                # Validate
                if w <= 0 or h <= 0:
                    continue
                cx = max(0.001, min(0.999, cx))
                cy = max(0.001, min(0.999, cy))
                w = max(0.001, min(1.0, w))
                h = max(0.001, min(1.0, h))

                yolo_lines.append(f"{urbaneye_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                self.stats[CLASS_NAMES[urbaneye_cls]] += 1

            if yolo_lines:
                stem = Path(frame_name).stem
                label_path = output_dir / f"{stem}.txt"
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))
                converted += 1

        return converted

    def filter_by_classes(self, annotations: list[dict]) -> list[dict]:
        """Keep only annotations matching our 5 target classes.

        Args:
            annotations: List of BDD100K annotation dicts.

        Returns:
            Filtered list containing only mapped classes.
        """
        return [ann for ann in annotations if ann.get("category", "") in self.class_mapping]

    def get_stats(self) -> dict[str, int]:
        """Return conversion statistics per class."""
        return dict(self.stats)


def create_mixed_dataset(
    primary_dir: Path,
    secondary_dir: Path,
    output_dir: Path,
    primary_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    """Create a mixed training dataset from two data sources.

    Samples images from primary (e.g., CARLA) and secondary (e.g., BDD100K)
    at the specified ratio, creating a unified YOLO dataset.

    Args:
        primary_dir: Primary dataset root (with images/ and labels/).
        secondary_dir: Secondary dataset root.
        output_dir: Output directory for the mixed dataset.
        primary_ratio: Fraction of images from primary source (0-1).
        seed: Random seed for reproducible sampling.

    Returns:
        Path to the generated dataset.yaml.
    """
    random.seed(seed)
    output_dir = Path(output_dir)

    for split in ("train", "val"):
        img_out = ensure_dir(output_dir / "images" / split)
        lbl_out = ensure_dir(output_dir / "labels" / split)

        # Collect images from both sources
        primary_imgs = sorted((primary_dir / "images" / split).glob("*.*"))
        secondary_imgs = sorted((secondary_dir / "images" / split).glob("*.*"))

        if not primary_imgs and not secondary_imgs:
            continue

        # Sample according to ratio
        total = len(primary_imgs) + len(secondary_imgs)
        n_primary = int(total * primary_ratio)
        n_secondary = total - n_primary

        sampled_primary = random.sample(primary_imgs, min(n_primary, len(primary_imgs)))
        sampled_secondary = random.sample(secondary_imgs, min(n_secondary, len(secondary_imgs)))

        # Copy files
        for img_path in sampled_primary + sampled_secondary:
            src_label = img_path.parent.parent.parent / "labels" / split / f"{img_path.stem}.txt"

            shutil.copy2(img_path, img_out / img_path.name)
            if src_label.exists():
                shutil.copy2(src_label, lbl_out / f"{img_path.stem}.txt")

    # Generate dataset.yaml
    yaml_content = f"""# UrbanEye Mixed Dataset (Primary {primary_ratio:.0%} + Secondary {1 - primary_ratio:.0%})
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
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def fine_tune(config: DomainAdaptConfig, data_yaml: str) -> Path:
    """Run domain adaptation fine-tuning.

    Loads the base weights, freezes backbone layers, and fine-tunes
    on the mixed dataset with a lower learning rate.

    Args:
        config: Fine-tuning configuration.
        data_yaml: Path to the mixed dataset YAML.

    Returns:
        Path to the best fine-tuned weights.

    Raises:
        ValueError: If config validation fails.
    """
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid adaptation config: {'; '.join(errors)}")

    from ultralytics import YOLO

    model = YOLO(config.base_weights)
    kwargs = config.to_ultralytics_kwargs()
    kwargs["data"] = data_yaml

    results = model.train(**kwargs)
    return Path(results.save_dir) / "weights" / "best.pt"
