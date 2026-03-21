"""Visualization utilities for UrbanEye.

Provides bounding box drawing, class distribution plotting, augmentation
visualization grids, and dataset summary statistics.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from urbaneye.utils.constants import CLASS_COLORS, CLASS_ID_TO_NAME, CLASS_NAMES


def draw_bboxes(
    image: np.ndarray,
    bboxes: list[list[float]],
    class_ids: list[int],
    confidences: list[float] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes with class labels on an image.

    Args:
        image: Input image (H x W x 3, BGR uint8).
        bboxes: YOLO-format bboxes [[cx, cy, w, h], ...] (normalized 0-1).
        class_ids: Class ID for each bbox.
        confidences: Optional confidence scores per bbox.
        thickness: Box line thickness.
        font_scale: Label font scale.

    Returns:
        Annotated image copy (original is not modified).
    """
    if not bboxes:
        return image.copy()

    annotated = image.copy()
    h, w = annotated.shape[:2]

    for i, (bbox, cls_id) in enumerate(zip(bboxes, class_ids)):
        cx, cy, bw, bh = bbox

        # Convert YOLO normalized to pixel coordinates
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Get class color
        class_name = CLASS_ID_TO_NAME.get(cls_id, f"cls_{cls_id}")
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        label = class_name
        if confidences and i < len(confidences):
            label = f"{class_name} {confidences[i]:.2f}"

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(annotated, (x1, y1 - label_h - baseline - 4), (x1 + label_w, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
        )

    return annotated


def plot_class_distribution(label_dir: Path) -> dict[str, int]:
    """Count class occurrences across all label files in a directory.

    Args:
        label_dir: Directory containing YOLO .txt label files.

    Returns:
        Dictionary mapping class name to count.
    """
    counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}

    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        return counts

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        cls_name = CLASS_ID_TO_NAME.get(cls_id)
                        if cls_name:
                            counts[cls_name] += 1
                    except ValueError:
                        continue

    return counts


def create_dataset_summary(dataset_dir: Path) -> dict[str, int | dict[str, int]]:
    """Compute summary statistics for a YOLO dataset.

    Args:
        dataset_dir: Root dataset directory containing images/ and labels/.

    Returns:
        Dictionary with total_images, total_labels, and per-split counts.
    """
    dataset_dir = Path(dataset_dir)
    summary: dict[str, int | dict[str, int]] = {
        "total_images": 0,
        "total_labels": 0,
    }

    splits: dict[str, int] = {}
    for split in ("train", "val", "test"):
        img_dir = dataset_dir / "images" / split
        if img_dir.is_dir():
            count = len(list(img_dir.glob("*.*")))
            splits[split] = count
            summary["total_images"] = int(summary["total_images"]) + count

        label_dir = dataset_dir / "labels" / split
        if label_dir.is_dir():
            count = len(list(label_dir.glob("*.txt")))
            summary["total_labels"] = int(summary["total_labels"]) + count

    summary["splits"] = splits
    return summary
