"""YOLOv11 training configuration, launcher, and model export.

Provides a dataclass-based training configuration with validation,
Ultralytics kwargs conversion, and model export utilities. The training
script (scripts/train_baseline.py) calls these functions.

Training is designed for two GPU tiers:
  - RTX 5070 Laptop (8.5GB VRAM): YOLOv11n batch=16, YOLOv11m batch=8
  - Kaggle T4 (16GB VRAM): YOLOv11m batch=16
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """YOLOv11 training hyperparameters.

    All values match the UrbanEye project spec (PDF Section 7.2).
    """

    model_variant: str = "yolo11n.pt"
    data_yaml: str = "dataset.yaml"
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    mosaic: float = 1.0
    mixup: float = 0.1
    cos_lr: bool = True
    amp: bool = True
    device: str = "0"
    workers: int = 4
    patience: int = 20
    save_period: int = 10
    project: str = "runs/detect"
    name: str = "urbaneye_v1"
    degrees: float = 5.0
    perspective: float = 0.001
    flipud: float = 0.0
    fliplr: float = 0.5
    cls: float = 0.5
    plots: bool = True
    export_formats: list[str] = field(default_factory=lambda: ["onnx"])

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty list means valid."""
        errors: list[str] = []

        if self.epochs <= 0:
            errors.append(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        if self.img_size <= 0 or self.img_size % 32 != 0:
            errors.append(f"img_size must be positive and divisible by 32, got {self.img_size}")
        if not (0 < self.lr0 <= 1.0):
            errors.append(f"lr0 must be in (0, 1], got {self.lr0}")
        if not (0 < self.lrf <= 1.0):
            errors.append(f"lrf must be in (0, 1], got {self.lrf}")
        if not (0 <= self.momentum <= 1.0):
            errors.append(f"momentum must be in [0, 1], got {self.momentum}")
        if self.weight_decay < 0:
            errors.append(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.warmup_epochs < 0:
            errors.append(f"warmup_epochs must be non-negative, got {self.warmup_epochs}")
        if not (0 <= self.mosaic <= 1.0):
            errors.append(f"mosaic must be in [0, 1], got {self.mosaic}")
        if not (0 <= self.mixup <= 1.0):
            errors.append(f"mixup must be in [0, 1], got {self.mixup}")
        if self.workers < 0:
            errors.append(f"workers must be non-negative, got {self.workers}")
        if self.patience <= 0:
            errors.append(f"patience must be positive, got {self.patience}")

        valid_variants = {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"}
        if self.model_variant not in valid_variants:
            errors.append(
                f"model_variant must be one of {valid_variants}, got '{self.model_variant}'"
            )

        valid_formats = {"onnx", "torchscript", "openvino", "engine", "coreml"}
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                errors.append(f"invalid export format '{fmt}', must be one of {valid_formats}")

        return errors

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs dict for ultralytics YOLO model.train().

        Returns:
            Dictionary matching ultralytics train() parameter names.
        """
        return {
            "data": self.data_yaml,
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.img_size,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "degrees": self.degrees,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "cls": self.cls,
            "amp": self.amp,
            "device": self.device,
            "workers": self.workers,
            "patience": self.patience,
            "save_period": self.save_period,
            "project": self.project,
            "name": self.name,
            "plots": self.plots,
        }

    @classmethod
    def for_rtx5070_laptop(cls, model_size: str = "n") -> TrainConfig:
        """Create config optimized for RTX 5070 Laptop (8.5GB VRAM).

        Args:
            model_size: 'n' for nano (batch=16) or 'm' for medium (batch=8).
        """
        if model_size == "n":
            return cls(
                model_variant="yolo11n.pt",
                batch_size=16,
                name="yolov11n_rtx5070",
            )
        elif model_size == "m":
            return cls(
                model_variant="yolo11m.pt",
                batch_size=8,
                name="yolov11m_rtx5070",
            )
        else:
            raise ValueError(f"model_size must be 'n' or 'm', got '{model_size}'")

    @classmethod
    def for_kaggle_t4(cls, model_size: str = "m") -> TrainConfig:
        """Create config optimized for Kaggle T4 (16GB VRAM).

        Args:
            model_size: 'n' for nano or 'm' for medium (default).
        """
        return cls(
            model_variant=f"yolo11{model_size}.pt",
            batch_size=16,
            epochs=100,
            name=f"yolov11{model_size}_kaggle",
        )


def train(config: TrainConfig) -> Path:
    """Launch YOLOv11 training with the given configuration.

    Args:
        config: Training configuration.

    Returns:
        Path to the best model weights (best.pt).

    Raises:
        ValueError: If config validation fails.
    """
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid training config: {'; '.join(errors)}")

    from ultralytics import YOLO

    model = YOLO(config.model_variant)
    results = model.train(**config.to_ultralytics_kwargs())
    return Path(results.save_dir) / "weights" / "best.pt"


def export_model(
    weights_path: Path,
    formats: list[str] | None = None,
    img_size: int = 640,
) -> list[Path]:
    """Export trained model to deployment formats.

    Args:
        weights_path: Path to trained .pt weights file.
        formats: List of export formats (default: ["onnx"]).
        img_size: Image size for exported model.

    Returns:
        List of paths to exported model files.
    """
    if formats is None:
        formats = ["onnx"]

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    exported: list[Path] = []
    for fmt in formats:
        path = model.export(format=fmt, imgsz=img_size, half=False)
        exported.append(Path(path))

    return exported
