"""Tests for urbaneye.training.augmentations module."""

from __future__ import annotations

import numpy as np
import pytest

A = pytest.importorskip("albumentations")

from urbaneye.training.augmentations import (  # noqa: E402
    apply_augmentation,
    get_train_augmentation,
    validate_augmented_bboxes,
)


@pytest.fixture
def sample_aug_image() -> np.ndarray:
    """Provide a 640x640x3 image for augmentation tests."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_aug_bboxes() -> list[list[float]]:
    """Sample YOLO bboxes (cx, cy, w, h) for augmentation tests."""
    return [
        [0.5, 0.5, 0.2, 0.3],
        [0.3, 0.7, 0.1, 0.15],
    ]


@pytest.fixture
def sample_aug_labels() -> list[int]:
    """Sample class labels matching bboxes."""
    return [0, 1]


class TestGetTrainAugmentation:
    """Tests for get_train_augmentation function."""

    def test_returns_compose(self) -> None:
        """Returns an Albumentations Compose object."""
        pipeline = get_train_augmentation("medium")
        assert isinstance(pipeline, A.Compose)

    def test_all_levels_valid(self) -> None:
        """All three levels instantiate without error."""
        for level in ("light", "medium", "heavy"):
            pipeline = get_train_augmentation(level)
            assert isinstance(pipeline, A.Compose)

    def test_invalid_level_raises(self) -> None:
        """Invalid level raises ValueError."""
        with pytest.raises(ValueError, match="level must be one of"):
            get_train_augmentation("extreme")

    def test_light_has_fewer_transforms(self) -> None:
        """Light level has fewer transforms than medium."""
        light = get_train_augmentation("light")
        medium = get_train_augmentation("medium")
        # Light has 1 transform, medium has more
        assert len(light.transforms) < len(medium.transforms)

    def test_has_bbox_params(self) -> None:
        """Pipeline includes YOLO bbox parameters."""
        pipeline = get_train_augmentation("medium")
        params = pipeline.processors.get("bboxes")
        assert params is not None


class TestApplyAugmentation:
    """Tests for apply_augmentation function."""

    def test_output_shape_matches_input(
        self,
        sample_aug_image: np.ndarray,
        sample_aug_bboxes: list[list[float]],
        sample_aug_labels: list[int],
    ) -> None:
        """Augmented image has same shape as input."""
        aug_img, _, _ = apply_augmentation(
            sample_aug_image, sample_aug_bboxes, sample_aug_labels, "light", seed=42
        )
        assert aug_img.shape == sample_aug_image.shape

    def test_bboxes_in_valid_range(
        self,
        sample_aug_image: np.ndarray,
        sample_aug_bboxes: list[list[float]],
        sample_aug_labels: list[int],
    ) -> None:
        """All output bboxes are in [0, 1] range."""
        _, aug_bboxes, _ = apply_augmentation(
            sample_aug_image, sample_aug_bboxes, sample_aug_labels, "medium", seed=42
        )
        for bbox in aug_bboxes:
            for val in bbox:
                assert 0.0 <= val <= 1.0, f"bbox value {val} out of [0,1] range"

    def test_class_labels_preserved(
        self,
        sample_aug_image: np.ndarray,
        sample_aug_bboxes: list[list[float]],
        sample_aug_labels: list[int],
    ) -> None:
        """Class labels are preserved through augmentation (same set)."""
        _, _, aug_labels = apply_augmentation(
            sample_aug_image, sample_aug_bboxes, sample_aug_labels, "light", seed=42
        )
        # Labels should be a subset (some may be removed by min_visibility)
        for label in aug_labels:
            assert label in sample_aug_labels

    def test_seed_parameter_accepted(
        self,
        sample_aug_image: np.ndarray,
        sample_aug_bboxes: list[list[float]],
        sample_aug_labels: list[int],
    ) -> None:
        """Seed parameter is accepted and produces valid output."""
        aug_img, aug_bboxes, aug_labels = apply_augmentation(
            sample_aug_image, sample_aug_bboxes, sample_aug_labels, "light", seed=123
        )
        assert aug_img.shape == sample_aug_image.shape
        assert isinstance(aug_bboxes, list)
        assert isinstance(aug_labels, list)

    def test_empty_bboxes(self, sample_aug_image: np.ndarray) -> None:
        """Augmentation works with no bboxes."""
        aug_img, aug_bboxes, aug_labels = apply_augmentation(
            sample_aug_image, [], [], "medium", seed=42
        )
        assert aug_img.shape == sample_aug_image.shape
        assert aug_bboxes == []
        assert aug_labels == []


class TestValidateAugmentedBboxes:
    """Tests for validate_augmented_bboxes function."""

    def test_valid_bboxes_pass_through(self) -> None:
        """Valid bboxes are returned unchanged."""
        bboxes = [[0.5, 0.5, 0.2, 0.3]]
        result = validate_augmented_bboxes(bboxes)
        assert result == bboxes

    def test_clips_out_of_range(self) -> None:
        """Values > 1.0 are clipped to 1.0."""
        bboxes = [[1.1, 0.5, 0.2, 0.3]]
        result = validate_augmented_bboxes(bboxes)
        assert result[0][0] == 1.0

    def test_clips_negative(self) -> None:
        """Negative values are clipped to 0.0."""
        bboxes = [[-0.1, 0.5, 0.2, 0.3]]
        result = validate_augmented_bboxes(bboxes)
        assert result[0][0] == 0.0

    def test_removes_zero_area(self) -> None:
        """Zero-area bboxes are removed."""
        bboxes = [[0.5, 0.5, 0.0, 0.3]]
        result = validate_augmented_bboxes(bboxes)
        assert result == []

    def test_removes_wrong_length(self) -> None:
        """Bboxes with wrong number of values are removed."""
        bboxes = [[0.5, 0.5, 0.2]]  # Only 3 values
        result = validate_augmented_bboxes(bboxes)
        assert result == []

    def test_empty_input(self) -> None:
        """Empty input returns empty list."""
        assert validate_augmented_bboxes([]) == []
