"""Tests for urbaneye.evaluation.detection_evaluator module."""

from __future__ import annotations

from urbaneye.evaluation.detection_evaluator import DetectionEvaluator, DetectionMetrics, compute_ap


class TestComputeAP:
    """Tests for AP computation."""

    def test_perfect_detection(self) -> None:
        """Perfect precision-recall curve gives AP = 1.0."""
        recalls = [0.5, 1.0]
        precisions = [1.0, 1.0]
        assert abs(compute_ap(recalls, precisions) - 1.0) < 0.01

    def test_no_detections(self) -> None:
        """Empty input gives AP = 0."""
        assert compute_ap([], []) == 0.0

    def test_decreasing_precision(self) -> None:
        """Decreasing precision gives lower AP."""
        recalls = [0.2, 0.4, 0.6, 0.8, 1.0]
        precisions = [1.0, 0.8, 0.6, 0.4, 0.2]
        ap = compute_ap(recalls, precisions)
        assert 0 < ap < 1


class TestDetectionMetrics:
    """Tests for DetectionMetrics dataclass."""

    def test_meets_targets_carla(self) -> None:
        """CARLA target: mAP@50 >= 0.70."""
        m = DetectionMetrics(map50=0.75)
        assert m.meets_targets("carla")["map50"] is True

    def test_meets_targets_bdd100k(self) -> None:
        """BDD100K target: mAP@50 >= 0.50."""
        m = DetectionMetrics(map50=0.55)
        assert m.meets_targets("bdd100k")["map50"] is True

    def test_to_dict(self) -> None:
        """to_dict is serializable."""
        m = DetectionMetrics(map50=0.7, precision=0.8, recall=0.6, per_class_ap={"vehicle": 0.9})
        d = m.to_dict()
        assert d["map50"] == 0.7
        assert d["per_class_ap"]["vehicle"] == 0.9


class TestDetectionEvaluator:
    """Tests for DetectionEvaluator."""

    def test_perfect_detection(self) -> None:
        """Perfect overlap gives high mAP."""
        gt = [{"bbox": [10, 10, 50, 50], "class_id": 0}]
        pred = [{"bbox": [10, 10, 50, 50], "class_id": 0, "confidence": 0.9}]
        evaluator = DetectionEvaluator()
        metrics = evaluator.evaluate(gt, pred)
        assert metrics.map50 > 0.9

    def test_no_detections(self) -> None:
        """No predictions gives zero metrics."""
        gt = [{"bbox": [10, 10, 50, 50], "class_id": 0}]
        evaluator = DetectionEvaluator()
        metrics = evaluator.evaluate(gt, [])
        assert metrics.map50 == 0.0

    def test_no_gt(self) -> None:
        """No ground truth handles gracefully."""
        evaluator = DetectionEvaluator()
        metrics = evaluator.evaluate([], [])
        assert metrics.map50 == 0.0

    def test_per_class_ap(self) -> None:
        """Per-class AP is computed for each class."""
        gt = [
            {"bbox": [10, 10, 50, 50], "class_id": 0},
            {"bbox": [100, 100, 150, 150], "class_id": 1},
        ]
        pred = [
            {"bbox": [10, 10, 50, 50], "class_id": 0, "confidence": 0.9},
            {"bbox": [100, 100, 150, 150], "class_id": 1, "confidence": 0.8},
        ]
        evaluator = DetectionEvaluator()
        metrics = evaluator.evaluate(gt, pred)
        assert "vehicle" in metrics.per_class_ap
        assert "pedestrian" in metrics.per_class_ap
