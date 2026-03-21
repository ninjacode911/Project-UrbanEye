"""Tests for urbaneye.training.domain_adapt module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urbaneye.training.domain_adapt import (
    BDD100K_ALL_CLASSES,
    BDD100K_CLASS_MAPPING,
    BDD100KAdapter,
    DomainAdaptConfig,
    create_mixed_dataset,
)


class TestBDD100KClassMapping:
    """Tests for BDD100K to UrbanEye class mapping."""

    def test_all_10_bdd_classes_mapped(self) -> None:
        """All 10 BDD100K classes are present in the mapping."""
        for cls in BDD100K_ALL_CLASSES:
            assert cls in BDD100K_CLASS_MAPPING, f"Missing mapping for: {cls}"

    def test_car_maps_to_vehicle(self) -> None:
        assert BDD100K_CLASS_MAPPING["car"] == 0

    def test_truck_maps_to_vehicle(self) -> None:
        assert BDD100K_CLASS_MAPPING["truck"] == 0

    def test_bus_maps_to_vehicle(self) -> None:
        assert BDD100K_CLASS_MAPPING["bus"] == 0

    def test_pedestrian_maps_to_pedestrian(self) -> None:
        assert BDD100K_CLASS_MAPPING["pedestrian"] == 1

    def test_rider_maps_to_cyclist(self) -> None:
        assert BDD100K_CLASS_MAPPING["rider"] == 2

    def test_bicycle_maps_to_cyclist(self) -> None:
        assert BDD100K_CLASS_MAPPING["bicycle"] == 2

    def test_traffic_light_maps_correctly(self) -> None:
        assert BDD100K_CLASS_MAPPING["traffic light"] == 3

    def test_traffic_sign_maps_correctly(self) -> None:
        assert BDD100K_CLASS_MAPPING["traffic sign"] == 4


class TestBDD100KAdapter:
    """Tests for BDD100KAdapter."""

    @pytest.fixture
    def mock_bdd_json(self, tmp_path: Path) -> Path:
        """Create a mock BDD100K detection JSON file."""
        data = [
            {
                "name": "frame_001.jpg",
                "labels": [
                    {"category": "car", "box2d": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}},
                    {"category": "pedestrian", "box2d": {"x1": 50, "y1": 100, "x2": 80, "y2": 250}},
                    {"category": "dog", "box2d": {"x1": 500, "y1": 500, "x2": 600, "y2": 600}},
                ],
            },
            {
                "name": "frame_002.jpg",
                "labels": [
                    {
                        "category": "traffic light",
                        "box2d": {"x1": 400, "y1": 50, "x2": 420, "y2": 100},
                    },
                ],
            },
            {
                "name": "frame_003.jpg",
                "labels": [
                    {"category": "dog", "box2d": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}},
                ],
            },
        ]
        json_path = tmp_path / "bdd_labels.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")
        return json_path

    def test_convert_annotations(self, mock_bdd_json: Path, tmp_path: Path) -> None:
        """Converts valid BDD100K annotations to YOLO format."""
        adapter = BDD100KAdapter()
        output = tmp_path / "labels"
        count = adapter.convert_annotations(mock_bdd_json, output)
        # frame_001 has car + pedestrian (dog skipped), frame_002 has traffic light
        # frame_003 only has dog (skipped) so no output
        assert count == 2

    def test_skips_unmapped_classes(self, mock_bdd_json: Path, tmp_path: Path) -> None:
        """Classes not in mapping (e.g., dog) are skipped."""
        adapter = BDD100KAdapter()
        output = tmp_path / "labels"
        adapter.convert_annotations(mock_bdd_json, output)
        # frame_003 only had 'dog' which is unmapped -> no file
        assert not (output / "frame_003.txt").exists()

    def test_yolo_format_correct(self, mock_bdd_json: Path, tmp_path: Path) -> None:
        """Output files are in correct YOLO format."""
        adapter = BDD100KAdapter()
        output = tmp_path / "labels"
        adapter.convert_annotations(mock_bdd_json, output)

        label = (output / "frame_001.txt").read_text(encoding="utf-8")
        lines = label.strip().split("\n")
        assert len(lines) == 2  # car + pedestrian

        parts = lines[0].split()
        assert len(parts) == 5
        assert parts[0] == "0"  # car -> vehicle (class 0)
        for val in parts[1:]:
            assert 0 <= float(val) <= 1

    def test_stats_tracking(self, mock_bdd_json: Path, tmp_path: Path) -> None:
        """Conversion stats are tracked per class."""
        adapter = BDD100KAdapter()
        adapter.convert_annotations(mock_bdd_json, tmp_path / "labels")
        stats = adapter.get_stats()
        assert stats["vehicle"] == 1
        assert stats["pedestrian"] == 1
        assert stats["traffic_light"] == 1
        assert stats["cyclist"] == 0
        assert stats["traffic_sign"] == 0

    def test_filter_by_classes(self) -> None:
        """filter_by_classes keeps only mapped categories."""
        adapter = BDD100KAdapter()
        annotations = [
            {"category": "car"},
            {"category": "dog"},
            {"category": "pedestrian"},
            {"category": "cat"},
        ]
        filtered = adapter.filter_by_classes(annotations)
        assert len(filtered) == 2
        assert filtered[0]["category"] == "car"
        assert filtered[1]["category"] == "pedestrian"


class TestDomainAdaptConfig:
    """Tests for DomainAdaptConfig."""

    def test_default_values(self) -> None:
        """Default config matches spec."""
        cfg = DomainAdaptConfig()
        assert cfg.epochs == 20
        assert cfg.lr0 == 0.0001
        assert cfg.freeze_layers == 10
        assert cfg.carla_ratio == 0.8
        assert cfg.bdd100k_ratio == 0.2

    def test_validate_valid(self) -> None:
        """Valid config produces no errors."""
        cfg = DomainAdaptConfig()
        assert cfg.validate() == []

    def test_validate_ratios_must_sum_to_one(self) -> None:
        """carla_ratio + bdd100k_ratio must equal 1.0."""
        cfg = DomainAdaptConfig(carla_ratio=0.5, bdd100k_ratio=0.3)
        errors = cfg.validate()
        assert any("must equal 1.0" in e for e in errors)

    def test_validate_lr_too_high_for_finetuning(self) -> None:
        """LR > 0.01 is suspicious for fine-tuning."""
        cfg = DomainAdaptConfig(lr0=0.1)
        errors = cfg.validate()
        assert any("lr0" in e for e in errors)

    def test_validate_negative_freeze(self) -> None:
        """Negative freeze layers caught."""
        cfg = DomainAdaptConfig(freeze_layers=-1)
        errors = cfg.validate()
        assert any("freeze_layers" in e for e in errors)

    def test_to_ultralytics_kwargs(self) -> None:
        """Kwargs contain freeze parameter."""
        cfg = DomainAdaptConfig()
        kwargs = cfg.to_ultralytics_kwargs()
        assert kwargs["freeze"] == 10
        assert kwargs["lr0"] == 0.0001
        assert kwargs["epochs"] == 20


class TestCreateMixedDataset:
    """Tests for create_mixed_dataset."""

    @pytest.fixture
    def two_datasets(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create two mock YOLO datasets."""
        for name, count in [("primary", 10), ("secondary", 5)]:
            for split in ("train", "val"):
                img_dir = tmp_path / name / "images" / split
                lbl_dir = tmp_path / name / "labels" / split
                img_dir.mkdir(parents=True)
                lbl_dir.mkdir(parents=True)
                for i in range(count):
                    (img_dir / f"{name}_{i:04d}.jpg").write_bytes(b"fake")
                    (lbl_dir / f"{name}_{i:04d}.txt").write_text(
                        "0 0.5 0.5 0.2 0.3", encoding="utf-8"
                    )
        return tmp_path / "primary", tmp_path / "secondary"

    def test_creates_dataset_yaml(self, two_datasets: tuple[Path, Path], tmp_path: Path) -> None:
        """Creates a valid dataset.yaml."""
        primary, secondary = two_datasets
        yaml_path = create_mixed_dataset(primary, secondary, tmp_path / "mixed")
        assert yaml_path.exists()
        content = yaml_path.read_text(encoding="utf-8")
        assert "nc: 5" in content
        assert "vehicle" in content

    def test_creates_image_directories(
        self, two_datasets: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Creates train and val image directories."""
        primary, secondary = two_datasets
        create_mixed_dataset(primary, secondary, tmp_path / "mixed")
        assert (tmp_path / "mixed" / "images" / "train").is_dir()
        assert (tmp_path / "mixed" / "images" / "val").is_dir()

    def test_copies_labels(self, two_datasets: tuple[Path, Path], tmp_path: Path) -> None:
        """Labels are copied alongside images."""
        primary, secondary = two_datasets
        create_mixed_dataset(primary, secondary, tmp_path / "mixed")
        labels = list((tmp_path / "mixed" / "labels" / "train").glob("*.txt"))
        assert len(labels) > 0

    def test_respects_ratio(self, two_datasets: tuple[Path, Path], tmp_path: Path) -> None:
        """Mixed dataset approximately respects the specified ratio."""
        primary, secondary = two_datasets
        create_mixed_dataset(primary, secondary, tmp_path / "mixed", primary_ratio=0.8)
        train_imgs = list((tmp_path / "mixed" / "images" / "train").glob("*.*"))
        primary_count = sum(1 for p in train_imgs if "primary" in p.name)
        total = len(train_imgs)
        if total > 0:
            ratio = primary_count / total
            assert 0.6 <= ratio <= 1.0  # Allow some flexibility due to small sample
