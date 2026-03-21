"""End-to-end pipeline smoke tests.

Verifies that data flows correctly between modules:
detection output → tracker input → evaluator input → report output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from urbaneye.evaluation.detection_evaluator import DetectionEvaluator
from urbaneye.evaluation.generate_report import ReportGenerator
from urbaneye.evaluation.mot_evaluator import MOTEvaluator, MOTMetrics
from urbaneye.tracking.bytetrack_pipeline import ByteTrackPipeline
from urbaneye.tracking.deepsort_pipeline import TrackedObject
from urbaneye.tracking.dual_tracker import DualTracker, TrackerType


class TestDetectionToTracking:
    """Verify detection output format feeds into trackers."""

    def test_bytetrack_accepts_detection_format(self) -> None:
        """ByteTrack accepts standard Nx6 detection array."""
        tracker = ByteTrackPipeline()
        detections = np.array(
            [[100, 100, 200, 200, 0.9, 0], [300, 300, 400, 400, 0.8, 1]],
            dtype=np.float32,
        )
        for _ in range(5):
            tracks = tracker.update(detections)
        assert all(hasattr(t, "track_id") for t in tracks)

    def test_dual_tracker_bytetrack_output(self) -> None:
        """DualTracker (ByteTrack) returns TrackedObject instances."""
        tracker = DualTracker(TrackerType.BYTETRACK)
        detections = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
        for _ in range(5):
            result = tracker.update(detections)
        assert all(isinstance(obj, TrackedObject) for obj in result)

    def test_empty_detections_no_crash(self) -> None:
        """Empty detections don't crash any tracker."""
        for tt in (TrackerType.BYTETRACK,):
            tracker = DualTracker(tt)
            result = tracker.update(np.empty((0, 6), dtype=np.float32))
            assert isinstance(result, list)


class TestTrackingToEvaluation:
    """Verify tracker output can be evaluated."""

    def test_mot_evaluator_with_tracked_objects(self) -> None:
        """MOTEvaluator accepts tracked object format."""
        gt_frames = [
            [{"id": 1, "bbox": [100, 100, 200, 200], "class_id": 0}],
            [{"id": 1, "bbox": [110, 110, 210, 210], "class_id": 0}],
        ]
        pred_frames = [
            [{"id": 1, "bbox": [100, 100, 200, 200], "class_id": 0}],
            [{"id": 1, "bbox": [110, 110, 210, 210], "class_id": 0}],
        ]
        evaluator = MOTEvaluator()
        metrics = evaluator.evaluate(gt_frames, pred_frames)
        assert isinstance(metrics, MOTMetrics)
        assert metrics.mota > 0

    def test_detection_evaluator_runs(self) -> None:
        """DetectionEvaluator produces metrics from detection format."""
        gt = [{"bbox": [10, 10, 50, 50], "class_id": 0}]
        pred = [{"bbox": [10, 10, 50, 50], "class_id": 0, "confidence": 0.9}]
        evaluator = DetectionEvaluator()
        metrics = evaluator.evaluate(gt, pred)
        assert metrics.map50 > 0


class TestEvaluationToReport:
    """Verify evaluation results produce valid reports."""

    def test_full_report_pipeline(self, tmp_path: Path) -> None:
        """Evaluation metrics → Markdown report file."""
        bt_metrics = MOTMetrics(mota=0.75, motp=0.85, idf1=0.70, id_switches=100)
        ds_metrics = MOTMetrics(mota=0.70, motp=0.82, idf1=0.75, id_switches=60)

        gen = ReportGenerator(tracker_metrics={"ByteTrack": bt_metrics, "DeepSORT": ds_metrics})
        out = tmp_path / "report.md"
        report = gen.generate_markdown(output_path=out)

        assert out.exists()
        assert "ByteTrack" in report
        assert "DeepSORT" in report
        assert "Comparison" in report


class TestFullPipelineSmoke:
    """End-to-end: detections → tracker → evaluator → report."""

    def test_complete_pipeline(self, tmp_path: Path) -> None:
        """Run complete pipeline with synthetic multi-frame data."""
        # Simulate 10 frames with 2 objects
        tracker = DualTracker(TrackerType.BYTETRACK)
        gt_frames = []
        pred_frames = []

        for frame_id in range(10):
            offset = frame_id * 5
            detections = np.array(
                [
                    [100 + offset, 100, 200 + offset, 200, 0.9, 0],
                    [400, 300, 500, 400, 0.85, 1],
                ],
                dtype=np.float32,
            )
            tracks = tracker.update(detections)

            gt_frames.append(
                [
                    {"id": 1, "bbox": [100 + offset, 100, 200 + offset, 200], "class_id": 0},
                    {"id": 2, "bbox": [400, 300, 500, 400], "class_id": 1},
                ]
            )
            pred_frames.append(
                [
                    {"id": t.track_id, "bbox": t.bbox.tolist(), "class_id": t.class_id}
                    for t in tracks
                ]
            )

        # Evaluate
        evaluator = MOTEvaluator()
        metrics = evaluator.evaluate(gt_frames, pred_frames)

        # Generate report
        gen = ReportGenerator(tracker_metrics={"ByteTrack": metrics})
        report = gen.generate_markdown(output_path=tmp_path / "smoke_report.md")

        assert metrics.num_matches > 0
        assert (tmp_path / "smoke_report.md").exists()
        assert "MOTA" in report
