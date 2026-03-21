"""Tests for urbaneye.training.train_yolov11 module."""

from __future__ import annotations

from urbaneye.training.train_yolov11 import TrainConfig


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_default_values_match_spec(self) -> None:
        """Default config matches UrbanEye PDF spec."""
        cfg = TrainConfig()
        assert cfg.model_variant == "yolo11n.pt"
        assert cfg.epochs == 50
        assert cfg.batch_size == 16
        assert cfg.img_size == 640
        assert cfg.lr0 == 0.001
        assert cfg.lrf == 0.01
        assert cfg.momentum == 0.937
        assert cfg.weight_decay == 0.0005
        assert cfg.warmup_epochs == 3
        assert cfg.mosaic == 1.0
        assert cfg.mixup == 0.1
        assert cfg.amp is True

    def test_validate_valid_config(self) -> None:
        """Valid config produces no errors."""
        cfg = TrainConfig()
        assert cfg.validate() == []

    def test_validate_negative_epochs(self) -> None:
        """Negative epochs caught."""
        cfg = TrainConfig(epochs=-1)
        errors = cfg.validate()
        assert any("epochs" in e for e in errors)

    def test_validate_zero_batch_size(self) -> None:
        """Zero batch size caught."""
        cfg = TrainConfig(batch_size=0)
        errors = cfg.validate()
        assert any("batch_size" in e for e in errors)

    def test_validate_img_size_not_divisible_by_32(self) -> None:
        """Image size not divisible by 32 caught."""
        cfg = TrainConfig(img_size=641)
        errors = cfg.validate()
        assert any("img_size" in e for e in errors)

    def test_validate_lr0_too_high(self) -> None:
        """LR > 1.0 caught."""
        cfg = TrainConfig(lr0=1.5)
        errors = cfg.validate()
        assert any("lr0" in e for e in errors)

    def test_validate_lr0_zero(self) -> None:
        """LR = 0 caught."""
        cfg = TrainConfig(lr0=0.0)
        errors = cfg.validate()
        assert any("lr0" in e for e in errors)

    def test_validate_invalid_model_variant(self) -> None:
        """Unknown model variant caught."""
        cfg = TrainConfig(model_variant="yolo99.pt")
        errors = cfg.validate()
        assert any("model_variant" in e for e in errors)

    def test_validate_invalid_export_format(self) -> None:
        """Invalid export format caught."""
        cfg = TrainConfig(export_formats=["invalid_format"])
        errors = cfg.validate()
        assert any("export format" in e for e in errors)

    def test_validate_negative_workers(self) -> None:
        """Negative workers caught."""
        cfg = TrainConfig(workers=-1)
        errors = cfg.validate()
        assert any("workers" in e for e in errors)

    def test_validate_mosaic_out_of_range(self) -> None:
        """Mosaic > 1.0 caught."""
        cfg = TrainConfig(mosaic=1.5)
        errors = cfg.validate()
        assert any("mosaic" in e for e in errors)


class TestTrainConfigKwargs:
    """Tests for to_ultralytics_kwargs conversion."""

    def test_contains_required_keys(self) -> None:
        """Kwargs dict contains all required Ultralytics parameters."""
        cfg = TrainConfig()
        kwargs = cfg.to_ultralytics_kwargs()
        required = [
            "data",
            "epochs",
            "batch",
            "imgsz",
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "mosaic",
            "mixup",
            "amp",
            "device",
            "project",
            "name",
        ]
        for key in required:
            assert key in kwargs, f"Missing key: {key}"

    def test_batch_key_name(self) -> None:
        """Ultralytics uses 'batch' not 'batch_size'."""
        cfg = TrainConfig(batch_size=8)
        kwargs = cfg.to_ultralytics_kwargs()
        assert kwargs["batch"] == 8
        assert "batch_size" not in kwargs

    def test_imgsz_key_name(self) -> None:
        """Ultralytics uses 'imgsz' not 'img_size'."""
        cfg = TrainConfig(img_size=640)
        kwargs = cfg.to_ultralytics_kwargs()
        assert kwargs["imgsz"] == 640
        assert "img_size" not in kwargs

    def test_values_match_config(self) -> None:
        """Kwargs values match the config object."""
        cfg = TrainConfig(epochs=100, lr0=0.01, batch_size=32)
        kwargs = cfg.to_ultralytics_kwargs()
        assert kwargs["epochs"] == 100
        assert kwargs["lr0"] == 0.01
        assert kwargs["batch"] == 32


class TestTrainConfigFactories:
    """Tests for factory class methods."""

    def test_rtx5070_nano(self) -> None:
        """RTX 5070 nano config uses batch=16."""
        cfg = TrainConfig.for_rtx5070_laptop("n")
        assert cfg.model_variant == "yolo11n.pt"
        assert cfg.batch_size == 16
        assert cfg.validate() == []

    def test_rtx5070_medium(self) -> None:
        """RTX 5070 medium config uses batch=8 (VRAM constraint)."""
        cfg = TrainConfig.for_rtx5070_laptop("m")
        assert cfg.model_variant == "yolo11m.pt"
        assert cfg.batch_size == 8
        assert cfg.validate() == []

    def test_kaggle_t4(self) -> None:
        """Kaggle T4 config uses batch=16 and 100 epochs."""
        cfg = TrainConfig.for_kaggle_t4("m")
        assert cfg.model_variant == "yolo11m.pt"
        assert cfg.batch_size == 16
        assert cfg.epochs == 100
        assert cfg.validate() == []

    def test_rtx5070_invalid_size(self) -> None:
        """Invalid model size raises ValueError."""
        try:
            TrainConfig.for_rtx5070_laptop("x")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
