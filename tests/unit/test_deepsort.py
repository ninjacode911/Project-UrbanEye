"""Tests for urbaneye.tracking.deepsort_pipeline module.

Uses mocks to avoid deep-sort-realtime model downloads in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from urbaneye.tracking.deepsort_pipeline import DeepSORTPipeline, TrackedObject


class TestTrackedObject:
    """Tests for TrackedObject dataclass."""

    def test_to_dict(self) -> None:
        """to_dict produces JSON-serializable output."""
        obj = TrackedObject(
            track_id=1,
            bbox=np.array([10, 20, 50, 80]),
            confidence=0.95,
            class_id=0,
            class_name="vehicle",
            is_confirmed=True,
            age=10,
            hits=8,
        )
        d = obj.to_dict()
        assert d["track_id"] == 1
        assert d["bbox_xyxy"] == [10, 20, 50, 80]
        assert d["confidence"] == 0.95
        assert d["class_name"] == "vehicle"
        assert d["is_confirmed"] is True

    def test_to_dict_with_float_bbox(self) -> None:
        """to_dict handles float bboxes."""
        obj = TrackedObject(
            track_id=1,
            bbox=np.array([10.5, 20.3, 50.7, 80.1]),
            confidence=0.8,
            class_id=1,
            class_name="pedestrian",
            is_confirmed=True,
        )
        d = obj.to_dict()
        assert len(d["bbox_xyxy"]) == 4


class TestDeepSORTPipeline:
    """Tests for DeepSORTPipeline."""

    def test_lazy_initialization(self) -> None:
        """Tracker is not initialized until first update call."""
        pipeline = DeepSORTPipeline()
        assert pipeline._initialized is False
        assert pipeline._tracker is None

    def test_default_config(self) -> None:
        """Default config matches spec values."""
        pipeline = DeepSORTPipeline()
        assert pipeline.max_age == 70
        assert pipeline.n_init == 3
        assert pipeline.max_cosine_distance == 0.3
        assert pipeline.nn_budget == 100
        assert pipeline.embedder == "mobilenet"

    def test_frame_required(self) -> None:
        """update() raises ValueError when frame is None."""
        pipeline = DeepSORTPipeline()
        det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
        with pytest.raises(ValueError, match="frame"):
            pipeline.update(det, frame=None)

    def test_config_loading(self, tmp_path) -> None:
        """Config is loaded from YAML file."""
        config = tmp_path / "ds_config.yaml"
        config.write_text(
            "max_age: 100\nn_init: 5\nmax_cosine_distance: 0.5\n",
            encoding="utf-8",
        )
        pipeline = DeepSORTPipeline(config_path=config)
        assert pipeline.max_age == 100
        assert pipeline.n_init == 5
        assert pipeline.max_cosine_distance == 0.5

    def test_reset_clears_state(self) -> None:
        """reset() clears tracker and initialization flag."""
        pipeline = DeepSORTPipeline()
        pipeline._initialized = True
        pipeline._tracker = MagicMock()
        pipeline.reset()
        assert pipeline._initialized is False
        assert pipeline._tracker is None

    @patch("urbaneye.tracking.deepsort_pipeline.DeepSORTPipeline._init_tracker")
    def test_update_with_mock_tracker(self, mock_init) -> None:
        """update() processes detections when tracker is mocked."""
        pipeline = DeepSORTPipeline()
        pipeline._initialized = True
        pipeline._tracker = MagicMock()
        pipeline._tracker.tracks = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)

        result = pipeline.update(det, frame=frame)
        pipeline._tracker.update_tracks.assert_called_once()
        assert isinstance(result, list)


class TestDualTracker:
    """Tests for DualTracker unified interface."""

    def test_bytetrack_creates_bytetrack(self) -> None:
        """BYTETRACK type creates ByteTrack backend."""
        from urbaneye.tracking.dual_tracker import DualTracker, TrackerType

        tracker = DualTracker(TrackerType.BYTETRACK)
        assert tracker._bytetrack is not None
        assert tracker._deepsort is None

    def test_deepsort_creates_deepsort(self) -> None:
        """DEEPSORT type creates DeepSORT backend."""
        from urbaneye.tracking.dual_tracker import DualTracker, TrackerType

        tracker = DualTracker(TrackerType.DEEPSORT)
        assert tracker._deepsort is not None
        assert tracker._bytetrack is None

    def test_bytetrack_returns_tracked_objects(self) -> None:
        """ByteTrack backend produces TrackedObject output."""
        from urbaneye.tracking.dual_tracker import DualTracker, TrackerType

        tracker = DualTracker(TrackerType.BYTETRACK)
        det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)

        # Run enough frames to get active tracks
        for _ in range(5):
            result = tracker.update(det)

        assert all(isinstance(obj, TrackedObject) for obj in result)

    def test_deepsort_requires_frame(self) -> None:
        """DeepSORT backend raises ValueError without frame."""
        from urbaneye.tracking.dual_tracker import DualTracker, TrackerType

        tracker = DualTracker(TrackerType.DEEPSORT)
        det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
        with pytest.raises(ValueError, match="frame"):
            tracker.update(det, frame=None)

    def test_reset(self) -> None:
        """reset() works for both backends."""
        from urbaneye.tracking.dual_tracker import DualTracker, TrackerType

        for tt in (TrackerType.BYTETRACK, TrackerType.DEEPSORT):
            tracker = DualTracker(tt)
            tracker.reset()  # Should not raise

    def test_from_config(self, tmp_path) -> None:
        """from_config creates correct tracker type."""
        from urbaneye.tracking.dual_tracker import DualTracker, TrackerType

        config = tmp_path / "tracker.yaml"
        config.write_text("tracker_type: bytetrack\nhigh_thresh: 0.7\n", encoding="utf-8")
        tracker = DualTracker.from_config(config)
        assert tracker.tracker_type == TrackerType.BYTETRACK

    def test_tracked_object_has_class_name(self) -> None:
        """TrackedObject from ByteTrack includes class name."""
        from urbaneye.tracking.dual_tracker import DualTracker, TrackerType

        tracker = DualTracker(TrackerType.BYTETRACK)
        det = np.array([[100, 100, 200, 200, 0.9, 1]], dtype=np.float32)
        for _ in range(5):
            result = tracker.update(det)
        if result:
            assert result[0].class_name == "pedestrian"
