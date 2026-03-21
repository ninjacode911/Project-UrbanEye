"""Integration tests for urbaneye.demo.app module."""

from __future__ import annotations

import numpy as np

from urbaneye.demo.app import ProcessingResult, annotate_frame
from urbaneye.tracking.deepsort_pipeline import TrackedObject


class TestAnnotateFrame:
    """Tests for frame annotation function."""

    def test_returns_same_shape(self) -> None:
        """Annotated frame has same shape as input."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = [
            TrackedObject(
                track_id=1,
                bbox=np.array([100, 100, 200, 200]),
                confidence=0.9,
                class_id=0,
                class_name="vehicle",
                is_confirmed=True,
            )
        ]
        result = annotate_frame(frame, tracks, fps=30.0)
        assert result.shape == frame.shape

    def test_empty_tracks(self) -> None:
        """Empty tracks returns frame with FPS overlay only."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = annotate_frame(frame, [], fps=25.0)
        assert result.shape == frame.shape

    def test_does_not_modify_original(self) -> None:
        """Original frame is not modified."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        annotate_frame(frame, [], fps=30.0)
        np.testing.assert_array_equal(frame, original)

    def test_multiple_tracks(self) -> None:
        """Multiple tracks are all annotated."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = [
            TrackedObject(1, np.array([10, 10, 50, 50]), 0.9, 0, "vehicle", True),
            TrackedObject(2, np.array([100, 100, 200, 200]), 0.8, 1, "pedestrian", True),
            TrackedObject(3, np.array([300, 50, 350, 100]), 0.7, 2, "cyclist", True),
        ]
        result = annotate_frame(frame, tracks, fps=30.0)
        assert result.shape == frame.shape


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_default_values(self) -> None:
        """Default result has zero values."""
        result = ProcessingResult()
        assert result.total_frames == 0
        assert result.total_detections == 0
        assert result.avg_fps == 0.0

    def test_class_counts_default_empty(self) -> None:
        """Class counts default to empty dict."""
        result = ProcessingResult()
        assert result.class_counts == {}
