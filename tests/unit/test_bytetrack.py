"""Tests for urbaneye.tracking.bytetrack_pipeline and tracking.utils."""

from __future__ import annotations

import numpy as np

from urbaneye.tracking.bytetrack_pipeline import ByteTrackPipeline, TrackState
from urbaneye.tracking.utils import (
    compute_iou,
    compute_iou_matrix,
    linear_assignment_solve,
    tlbr_to_tlwh,
    tlwh_to_xyah,
    xyah_to_tlbr,
)


class TestComputeIoU:
    """Tests for IoU computation."""

    def test_identical_boxes(self) -> None:
        """Identical boxes have IoU = 1.0."""
        box = np.array([10, 10, 50, 50])
        assert abs(compute_iou(box, box) - 1.0) < 1e-6

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes have IoU = 0.0."""
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        """Partially overlapping boxes have 0 < IoU < 1."""
        a = np.array([0, 0, 20, 20])
        b = np.array([10, 10, 30, 30])
        iou = compute_iou(a, b)
        assert 0 < iou < 1

    def test_known_value(self) -> None:
        """IoU of known boxes matches hand-calculated value."""
        a = np.array([0, 0, 20, 20])  # area = 400
        b = np.array([10, 10, 30, 30])  # area = 400
        # intersection: [10,10,20,20] = 100
        # union: 400 + 400 - 100 = 700
        expected = 100 / 700
        assert abs(compute_iou(a, b) - expected) < 1e-6


class TestComputeIoUMatrix:
    """Tests for IoU matrix computation."""

    def test_shape(self) -> None:
        """IoU matrix has shape (N, M)."""
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        b = np.array([[5, 5, 15, 15], [25, 25, 35, 35], [0, 0, 5, 5]])
        result = compute_iou_matrix(a, b)
        assert result.shape == (2, 3)

    def test_diagonal_for_same_boxes(self) -> None:
        """Same boxes produce diagonal of 1.0."""
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        result = compute_iou_matrix(boxes, boxes)
        np.testing.assert_array_almost_equal(np.diag(result), [1.0, 1.0])

    def test_empty_input(self) -> None:
        """Empty inputs produce empty matrix."""
        result = compute_iou_matrix(np.empty((0, 4)), np.array([[0, 0, 10, 10]]))
        assert result.shape == (0, 1)


class TestLinearAssignment:
    """Tests for Hungarian Algorithm wrapper."""

    def test_perfect_matching(self) -> None:
        """Low-cost diagonal produces perfect matching."""
        cost = np.array([[0.1, 0.9], [0.9, 0.1]])
        matches, unm_r, unm_c = linear_assignment_solve(cost, threshold=0.5)
        assert len(matches) == 2
        assert len(unm_r) == 0
        assert len(unm_c) == 0

    def test_threshold_filtering(self) -> None:
        """High-cost matches are filtered by threshold."""
        cost = np.array([[0.1, 0.9], [0.9, 0.1]])
        matches, _, _ = linear_assignment_solve(cost, threshold=0.05)
        assert len(matches) == 0  # Both optimal matches exceed 0.05

    def test_empty_matrix(self) -> None:
        """Empty cost matrix returns all unmatched."""
        matches, unm_r, unm_c = linear_assignment_solve(np.empty((3, 0)), threshold=1.0)
        assert len(matches) == 0
        assert len(unm_r) == 3


class TestBboxConversions:
    """Tests for bbox format conversions."""

    def test_tlbr_to_tlwh(self) -> None:
        """[x1,y1,x2,y2] → [x1,y1,w,h]."""
        result = tlbr_to_tlwh(np.array([10, 20, 50, 80]))
        np.testing.assert_array_equal(result, [10, 20, 40, 60])

    def test_tlwh_to_xyah(self) -> None:
        """[x1,y1,w,h] → [cx,cy,a,h]."""
        result = tlwh_to_xyah(np.array([10, 20, 40, 60]))
        assert result[0] == 30  # cx = 10 + 40/2
        assert result[1] == 50  # cy = 20 + 60/2
        assert abs(result[2] - 40 / 60) < 1e-6  # aspect ratio
        assert result[3] == 60  # h

    def test_roundtrip_tlbr_xyah(self) -> None:
        """Converting tlbr→tlwh→xyah→tlbr recovers the original."""
        original = np.array([10.0, 20.0, 50.0, 80.0])
        tlwh = tlbr_to_tlwh(original)
        xyah = tlwh_to_xyah(tlwh)
        recovered = xyah_to_tlbr(xyah)
        np.testing.assert_array_almost_equal(recovered, original)


class TestByteTrackPipeline:
    """Tests for the full ByteTrack tracker."""

    def _make_detection(
        self, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9, cls: int = 0
    ) -> np.ndarray:
        """Helper: create a single detection array."""
        return np.array([[x1, y1, x2, y2, conf, cls]], dtype=np.float32)

    def test_empty_detections(self) -> None:
        """No detections returns empty track list."""
        tracker = ByteTrackPipeline()
        tracks = tracker.update(np.empty((0, 6)))
        assert tracks == []

    def test_single_detection_creates_track(self) -> None:
        """One detection creates a new track (initially NEW, not ACTIVE)."""
        tracker = ByteTrackPipeline()
        det = self._make_detection(100, 100, 200, 200)
        tracker.update(det)
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].state == TrackState.NEW

    def test_track_becomes_active_after_min_hits(self) -> None:
        """Track transitions NEW → ACTIVE after min_hits consecutive matches."""
        tracker = ByteTrackPipeline()
        det = self._make_detection(100, 100, 200, 200)

        for _ in range(5):
            tracker.update(det)

        active = tracker.active_tracks
        assert len(active) >= 1
        assert active[0].state == TrackState.ACTIVE

    def test_track_becomes_lost_when_unmatched(self) -> None:
        """Active track becomes LOST when detection disappears."""
        tracker = ByteTrackPipeline()
        det = self._make_detection(100, 100, 200, 200)

        # Build up active track
        for _ in range(5):
            tracker.update(det)
        assert len(tracker.active_tracks) >= 1

        # Remove detection → track should become lost
        for _ in range(3):
            tracker.update(np.empty((0, 6)))

        lost = [t for t in tracker.tracks if t.state == TrackState.LOST]
        assert len(lost) >= 1

    def test_lost_track_deleted_after_max_lost(self) -> None:
        """Lost track is deleted after max_lost frames."""
        tracker = ByteTrackPipeline()
        tracker.max_lost = 5
        det = self._make_detection(100, 100, 200, 200)

        # Build track
        for _ in range(5):
            tracker.update(det)

        # Lose it for > max_lost frames
        for _ in range(10):
            tracker.update(np.empty((0, 6)))

        # Track should be removed
        assert len(tracker.tracks) == 0

    def test_track_ids_monotonically_increasing(self) -> None:
        """Track IDs are assigned in increasing order."""
        tracker = ByteTrackPipeline()
        det1 = self._make_detection(100, 100, 200, 200)
        det2 = self._make_detection(400, 400, 500, 500)

        tracker.update(det1)
        tracker.update(np.vstack([det1[0], det2[0]]))

        ids = sorted([t.track_id for t in tracker.tracks])
        assert ids == sorted(set(ids))  # All unique
        assert ids[-1] > ids[0]  # Increasing

    def test_two_stage_matching(self) -> None:
        """Low-confidence detections match in stage 2."""
        tracker = ByteTrackPipeline()
        # Build track with high-conf detection
        high_det = self._make_detection(100, 100, 200, 200, conf=0.9)
        for _ in range(5):
            tracker.update(high_det)

        # Now send low-conf detection at same position
        low_det = self._make_detection(105, 105, 205, 205, conf=0.3)
        tracker.update(low_det)

        # Track should still be active (matched via stage 2)
        active = [t for t in tracker.tracks if t.state in (TrackState.ACTIVE, TrackState.NEW)]
        assert len(active) >= 1

    def test_reset_clears_state(self) -> None:
        """reset() clears all tracks and counters."""
        tracker = ByteTrackPipeline()
        det = self._make_detection(100, 100, 200, 200)
        tracker.update(det)
        assert len(tracker.tracks) > 0

        tracker.reset()
        assert len(tracker.tracks) == 0
        assert tracker.frame_id == 0
        assert tracker._track_id_count == 0

    def test_multiple_objects_tracked(self) -> None:
        """Multiple spatially separated objects get separate tracks."""
        tracker = ByteTrackPipeline()
        dets = np.array(
            [
                [100, 100, 150, 150, 0.9, 0],
                [400, 400, 450, 450, 0.9, 1],
                [700, 100, 750, 150, 0.85, 0],
            ],
            dtype=np.float32,
        )

        for _ in range(5):
            tracker.update(dets)

        assert len(tracker.active_tracks) == 3

    def test_class_preserved_in_track(self) -> None:
        """Track preserves the detection class ID."""
        tracker = ByteTrackPipeline()
        det = self._make_detection(100, 100, 200, 200, conf=0.9, cls=2)
        tracker.update(det)
        assert tracker.tracks[0].class_id == 2
