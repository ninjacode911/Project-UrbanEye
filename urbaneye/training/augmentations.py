"""Albumentations-based augmentation pipeline for domain adaptation.

Applies real-world camera degradations (noise, blur, JPEG artifacts, weather
effects) to CARLA synthetic images during training, bridging the sim-to-real
domain gap. Three augmentation levels support different training phases.

All transforms are YOLO-bbox-compatible via A.BboxParams.
"""

from __future__ import annotations

import albumentations as A
import numpy as np


def get_train_augmentation(level: str = "medium") -> A.Compose:
    """Build a training augmentation pipeline at the specified level.

    Args:
        level: Augmentation intensity. One of:
            - "light": Minimal augmentation (validation/sanity checks)
            - "medium": Standard training augmentation
            - "heavy": Aggressive augmentation for domain adaptation

    Returns:
        Albumentations Compose pipeline with YOLO bbox support.

    Raises:
        ValueError: If level is not one of the valid options.
    """
    valid_levels = {"light", "medium", "heavy"}
    if level not in valid_levels:
        raise ValueError(f"level must be one of {valid_levels}, got '{level}'")

    if level == "light":
        transforms = [
            A.HorizontalFlip(p=0.5),
        ]
    elif level == "medium":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.ImageCompression(quality_range=(70, 95), p=0.3),
            A.Perspective(scale=(0.02, 0.05), p=0.1),
            A.ColorJitter(p=0.3),
        ]
    else:  # heavy
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.GaussNoise(p=0.4),
            A.GaussianBlur(blur_limit=(3, 9), p=0.3),
            A.ImageCompression(quality_range=(50, 90), p=0.4),
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=0.15),
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.1),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),
            A.Perspective(scale=(0.02, 0.08), p=0.15),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.4),
        ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def apply_augmentation(
    image: np.ndarray,
    bboxes: list[list[float]],
    class_labels: list[int],
    level: str = "medium",
    seed: int | None = None,
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    """Apply augmentation to an image with YOLO-format bounding boxes.

    Args:
        image: Input image (H x W x 3, uint8).
        bboxes: YOLO-format bounding boxes [[cx, cy, w, h], ...].
        class_labels: Class IDs corresponding to each bbox.
        level: Augmentation level ("light", "medium", "heavy").
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of (augmented_image, augmented_bboxes, augmented_labels).
    """
    pipeline = get_train_augmentation(level)

    if seed is not None:
        import random

        random.seed(seed)
        np.random.seed(seed)

    result = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

    aug_image = result["image"]
    aug_bboxes = [list(b) for b in result["bboxes"]]
    aug_labels = result["class_labels"]

    # Validate output bboxes
    aug_bboxes = validate_augmented_bboxes(aug_bboxes)

    return aug_image, aug_bboxes, aug_labels


def validate_augmented_bboxes(bboxes: list[list[float]]) -> list[list[float]]:
    """Clip bounding boxes to [0, 1] range and remove degenerate boxes.

    Args:
        bboxes: List of YOLO-format bboxes [[cx, cy, w, h], ...].

    Returns:
        Filtered and clipped bounding boxes.
    """
    valid: list[list[float]] = []
    for bbox in bboxes:
        if len(bbox) != 4:
            continue
        cx, cy, w, h = bbox

        # Clip to valid range
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        # Remove zero-area boxes
        if w <= 0 or h <= 0:
            continue

        valid.append([cx, cy, w, h])

    return valid
