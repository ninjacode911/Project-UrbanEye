"""Tests for urbaneye.evaluation.generate_report module."""

from __future__ import annotations

from pathlib import Path

from urbaneye.evaluation.detection_evaluator import DetectionMetrics
from urbaneye.evaluation.generate_report import ReportGenerator
from urbaneye.evaluation.mot_evaluator import MOTMetrics


class TestReportGenerator:
    """Tests for ReportGenerator."""

    def test_generates_markdown(self) -> None:
        """Report is valid Markdown with headers."""
        det = DetectionMetrics(map50=0.75, precision=0.8, recall=0.6)
        gen = ReportGenerator(detection_metrics=det)
        report = gen.generate_markdown()
        assert "# UrbanEye Evaluation Report" in report
        assert "mAP@50" in report

    def test_tracking_section(self) -> None:
        """Report includes tracker metrics."""
        metrics = {"ByteTrack": MOTMetrics(mota=0.75, motp=0.85, idf1=0.7, id_switches=100)}
        gen = ReportGenerator(tracker_metrics=metrics)
        report = gen.generate_markdown()
        assert "ByteTrack" in report
        assert "MOTA" in report

    def test_comparison_section(self) -> None:
        """Report includes comparison when two trackers present."""
        metrics = {
            "ByteTrack": MOTMetrics(mota=0.75, motp=0.85, idf1=0.65, id_switches=150),
            "DeepSORT": MOTMetrics(mota=0.70, motp=0.83, idf1=0.72, id_switches=80),
        }
        gen = ReportGenerator(tracker_metrics=metrics)
        report = gen.generate_markdown()
        assert "Comparison" in report
        assert "wins" in report

    def test_writes_to_file(self, tmp_path: Path) -> None:
        """Report is written to file when path provided."""
        det = DetectionMetrics(map50=0.5)
        gen = ReportGenerator(detection_metrics=det)
        out = tmp_path / "report.md"
        gen.generate_markdown(output_path=out)
        assert out.exists()
        assert "mAP@50" in out.read_text(encoding="utf-8")

    def test_per_class_ap_in_report(self) -> None:
        """Per-class AP table appears in report."""
        det = DetectionMetrics(
            map50=0.7,
            per_class_ap={"vehicle": 0.8, "pedestrian": 0.7, "cyclist": 0.5},
        )
        gen = ReportGenerator(detection_metrics=det)
        report = gen.generate_markdown()
        assert "vehicle" in report
        assert "pedestrian" in report

    def test_empty_report(self) -> None:
        """Report with no metrics still has header."""
        gen = ReportGenerator()
        report = gen.generate_markdown()
        assert "UrbanEye Evaluation Report" in report
