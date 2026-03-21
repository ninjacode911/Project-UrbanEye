"""Shared pytest fixtures for UrbanEye test suite.

Provides reusable fixtures for temporary directories, sample images,
sample bounding boxes, and configuration loading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def sample_image() -> np.ndarray:
    """Provide a 640x640x3 sample image (random pixels, uint8)."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_small() -> np.ndarray:
    """Provide a 320x320x3 sample image for fast tests."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(320, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_bboxes_yolo() -> list[list[float]]:
    """Provide sample YOLO-format bounding boxes [class_id, cx, cy, w, h]."""
    return [
        [0, 0.5, 0.5, 0.2, 0.3],  # vehicle, center
        [1, 0.3, 0.7, 0.1, 0.25],  # pedestrian
        [2, 0.8, 0.4, 0.05, 0.15],  # cyclist
        [3, 0.15, 0.1, 0.03, 0.06],  # traffic_light
        [4, 0.9, 0.2, 0.04, 0.08],  # traffic_sign
    ]


@pytest.fixture
def sample_detections() -> np.ndarray:
    """Provide sample detection array [x1, y1, x2, y2, confidence, class_id]."""
    return np.array(
        [
            [100, 150, 200, 300, 0.92, 0],  # vehicle
            [300, 400, 350, 550, 0.85, 1],  # pedestrian
            [500, 200, 530, 280, 0.73, 2],  # cyclist
            [50, 30, 70, 60, 0.68, 3],  # traffic_light
            [580, 80, 600, 120, 0.61, 4],  # traffic_sign
        ],
        dtype=np.float32,
    )


@pytest.fixture
def project_config(tmp_path: Path) -> dict:
    """Load and return a test project configuration."""
    config = {
        "project": {"name": "UrbanEye", "version": "0.1.0"},
        "detection": {
            "num_classes": 5,
            "class_names": [
                "vehicle",
                "pedestrian",
                "cyclist",
                "traffic_light",
                "traffic_sign",
            ],
            "confidence_threshold": 0.25,
            "img_size": 640,
        },
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return config


@pytest.fixture
def sample_yaml_file(tmp_path: Path) -> Path:
    """Create a temporary valid YAML file and return its path."""
    data = {"key": "value", "nested": {"inner": 42}}
    yaml_path = tmp_path / "test.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return yaml_path


@pytest.fixture
def sample_label_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample YOLO label files."""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    # 3 sample label files with different class distributions
    labels = [
        "0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.25\n",
        "0 0.4 0.4 0.15 0.2\n2 0.8 0.3 0.05 0.1\n",
        "1 0.6 0.6 0.1 0.2\n3 0.15 0.1 0.03 0.06\n4 0.9 0.2 0.04 0.08\n",
    ]
    for i, content in enumerate(labels):
        (label_dir / f"frame_{i:06d}.txt").write_text(content, encoding="utf-8")
    return label_dir
