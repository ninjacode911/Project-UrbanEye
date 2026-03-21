"""Tests for urbaneye.evaluation.mot_evaluator module."""

from __future__ import annotations

from urbaneye.evaluation.mot_evaluator import MOTEvaluator, MOTMetrics


class TestMOTMetrics:
    """Tests for MOTMetrics dataclass."""

    def test_meets_targets_all_pass(self) -> None:
        """Perfect metrics meet all targets."""
        m = MOTMetrics(mota=0.9, motp=0.9, idf1=0.8, id_switches=50)
        targets = m.meets_targets()
        assert all(targets.values())

    def test_meets_targets_mota_fails(self) -> None:
        """Low MOTA fails target."""
        m = MOTMetrics(mota=0.5, motp=0.9, idf1=0.8, id_switches=50)
        assert m.meets_targets()["mota"] is False

    def test_to_dict(self) -> None:
        """to_dict produces serializable output."""
        m = MOTMetrics(mota=0.75, motp=0.85, idf1=0.70, id_switches=100)
        d = m.to_dict()
        assert d["mota"] == 0.75
        assert d["id_switches"] == 100


class TestMOTEvaluator:
    """Tests for MOTEvaluator."""

    def test_perfect_tracking(self) -> None:
        """Ground truth == predictions → MOTA = 1.0, ID Switches = 0."""
        gt = [
            [{"id": 1, "bbox": [10, 10, 50, 50], "class_id": 0}],
            [{"id": 1, "bbox": [15, 15, 55, 55], "class_id": 0}],
        ]
        pred = [
            [{"id": 1, "bbox": [10, 10, 50, 50], "class_id": 0}],
            [{"id": 1, "bbox": [15, 15, 55, 55], "class_id": 0}],
        ]
        evaluator = MOTEvaluator()
        metrics = evaluator.evaluate(gt, pred)
        assert metrics.mota == 1.0
        assert metrics.id_switches == 0
        assert metrics.num_matches == 2

    def test_all_false_positives(self) -> None:
        """Predictions with no GT → all FP, MOTA undefined."""
        gt = [[], []]
        pred = [
            [{"id": 1, "bbox": [10, 10, 50, 50], "class_id": 0}],
            [{"id": 2, "bbox": [100, 100, 150, 150], "class_id": 0}],
        ]
        evaluator = MOTEvaluator()
        metrics = evaluator.evaluate(gt, pred)
        assert metrics.num_false_positives == 2
        assert metrics.num_matches == 0

    def test_all_misses(self) -> None:
        """GT with no predictions → all FN, MOTA low."""
        gt = [
            [{"id": 1, "bbox": [10, 10, 50, 50], "class_id": 0}],
            [{"id": 1, "bbox": [15, 15, 55, 55], "class_id": 0}],
        ]
        pred = [[], []]
        evaluator = MOTEvaluator()
        metrics = evaluator.evaluate(gt, pred)
        assert metrics.num_misses == 2
        assert metrics.mota <= 0.0  # MOTA = 1 - FN/GT = 0 when all missed

    def test_id_switch_detected(self) -> None:
        """ID switch is counted when prediction ID changes for same GT object."""
        gt = [
            [{"id": 1, "bbox": [10, 10, 50, 50], "class_id": 0}],
            [{"id": 1, "bbox": [12, 12, 52, 52], "class_id": 0}],
        ]
        pred = [
            [{"id": 100, "bbox": [10, 10, 50, 50], "class_id": 0}],
            [{"id": 200, "bbox": [12, 12, 52, 52], "class_id": 0}],  # Different pred ID!
        ]
        evaluator = MOTEvaluator()
        metrics = evaluator.evaluate(gt, pred)
        assert metrics.id_switches == 1

    def test_motp_is_average_iou(self) -> None:
        """MOTP equals average IoU of matched pairs."""
        gt = [[{"id": 1, "bbox": [0, 0, 100, 100], "class_id": 0}]]
        pred = [[{"id": 1, "bbox": [0, 0, 100, 100], "class_id": 0}]]
        evaluator = MOTEvaluator()
        metrics = evaluator.evaluate(gt, pred)
        assert abs(metrics.motp - 1.0) < 0.01  # Perfect overlap

    def test_empty_sequence(self) -> None:
        """Empty GT and pred produce zero metrics."""
        evaluator = MOTEvaluator()
        metrics = evaluator.evaluate([], [])
        assert metrics.mota == 0.0
        assert metrics.num_matches == 0

    def test_per_class_evaluation(self) -> None:
        """Per-class evaluation separates by class_id."""
        gt = [
            [
                {"id": 1, "bbox": [10, 10, 50, 50], "class_id": 0},
                {"id": 2, "bbox": [100, 100, 150, 150], "class_id": 1},
            ],
        ]
        pred = [
            [
                {"id": 1, "bbox": [10, 10, 50, 50], "class_id": 0},
                {"id": 2, "bbox": [100, 100, 150, 150], "class_id": 1},
            ],
        ]
        evaluator = MOTEvaluator()
        per_class = evaluator.evaluate_per_class(gt, pred)
        assert "vehicle" in per_class
        assert "pedestrian" in per_class
        assert per_class["vehicle"].num_matches == 1
        assert per_class["pedestrian"].num_matches == 1
