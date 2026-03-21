"""Tests for urbaneye.utils.visualization module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from urbaneye.utils.visualization import (
    create_dataset_summary,
    draw_bboxes,
    plot_class_distribution,
)


class TestDrawBboxes:
    """Tests for draw_bboxes function."""

    def test_returns_same_shape(self, sample_image: np.ndarray) -> None:
        """Annotated image has same shape as input."""
        bboxes = [[0.5, 0.5, 0.2, 0.3]]
        class_ids = [0]
        result = draw_bboxes(sample_image, bboxes, class_ids)
        assert result.shape == sample_image.shape

    def test_does_not_modify_original(self, sample_image: np.ndarray) -> None:
        """Original image is not modified."""
        original = sample_image.copy()
        bboxes = [[0.5, 0.5, 0.2, 0.3]]
        draw_bboxes(sample_image, bboxes, [0])
        np.testing.assert_array_equal(sample_image, original)

    def test_empty_bboxes_returns_copy(self, sample_image: np.ndarray) -> None:
        """Empty bbox list returns an unmodified copy."""
        result = draw_bboxes(sample_image, [], [])
        np.testing.assert_array_equal(result, sample_image)

    def test_with_confidences(self, sample_image: np.ndarray) -> None:
        """Drawing with confidence scores doesn't crash."""
        bboxes = [[0.5, 0.5, 0.2, 0.3]]
        result = draw_bboxes(sample_image, bboxes, [0], confidences=[0.95])
        assert result.shape == sample_image.shape

    def test_multiple_classes(self, sample_image: np.ndarray) -> None:
        """Drawing multiple classes with different colors."""
        bboxes = [[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]]
        result = draw_bboxes(sample_image, bboxes, [0, 1])
        assert result.shape == sample_image.shape


class TestPlotClassDistribution:
    """Tests for plot_class_distribution function."""

    def test_counts_classes(self, sample_label_dir: Path) -> None:
        """Correctly counts class occurrences across label files."""
        counts = plot_class_distribution(sample_label_dir)
        assert counts["vehicle"] == 2  # class 0 appears in 2 files
        assert counts["pedestrian"] == 2  # class 1 appears in 2 files
        assert counts["cyclist"] == 1  # class 2 in 1 file
        assert counts["traffic_light"] == 1
        assert counts["traffic_sign"] == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns zero counts."""
        empty_dir = tmp_path / "empty_labels"
        empty_dir.mkdir()
        counts = plot_class_distribution(empty_dir)
        assert all(v == 0 for v in counts.values())

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Non-existent directory returns zero counts."""
        counts = plot_class_distribution(tmp_path / "missing")
        assert all(v == 0 for v in counts.values())


class TestCreateDatasetSummary:
    """Tests for create_dataset_summary function."""

    def test_counts_images_and_labels(self, tmp_path: Path) -> None:
        """Correctly counts images and labels across splits."""
        # Create train split
        train_img = tmp_path / "images" / "train"
        train_img.mkdir(parents=True)
        (train_img / "img1.jpg").write_bytes(b"fake")
        (train_img / "img2.jpg").write_bytes(b"fake")

        train_lbl = tmp_path / "labels" / "train"
        train_lbl.mkdir(parents=True)
        (train_lbl / "img1.txt").write_text("0 0.5 0.5 0.2 0.3", encoding="utf-8")
        (train_lbl / "img2.txt").write_text("1 0.3 0.7 0.1 0.2", encoding="utf-8")

        summary = create_dataset_summary(tmp_path)
        assert summary["total_images"] == 2
        assert summary["total_labels"] == 2

    def test_empty_dataset(self, tmp_path: Path) -> None:
        """Empty dataset returns zero counts."""
        summary = create_dataset_summary(tmp_path)
        assert summary["total_images"] == 0
        assert summary["total_labels"] == 0
