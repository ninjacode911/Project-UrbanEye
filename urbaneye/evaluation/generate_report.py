"""Evaluation report generator — Markdown output.

Generates comparison reports for ByteTrack vs DeepSORT tracking
results and per-class detection/tracking breakdowns.
"""

from __future__ import annotations

from pathlib import Path

from urbaneye.evaluation.detection_evaluator import DetectionMetrics
from urbaneye.evaluation.mot_evaluator import MOTMetrics
from urbaneye.utils.constants import CLASS_NAMES


class ReportGenerator:
    """Generate evaluation reports in Markdown format.

    Args:
        detection_metrics: Optional detection evaluation results.
        tracker_metrics: Optional dict of tracker_name -> MOTMetrics.
    """

    def __init__(
        self,
        detection_metrics: DetectionMetrics | None = None,
        tracker_metrics: dict[str, MOTMetrics] | None = None,
    ) -> None:
        self.detection_metrics = detection_metrics
        self.tracker_metrics = tracker_metrics or {}

    def generate_markdown(self, output_path: Path | None = None) -> str:
        """Generate full Markdown evaluation report.

        Args:
            output_path: Optional path to write the report file.

        Returns:
            Report content as a string.
        """
        sections: list[str] = []
        sections.append("# UrbanEye Evaluation Report\n")

        if self.detection_metrics:
            sections.append(self._detection_section())

        if self.tracker_metrics:
            sections.append(self._tracking_section())
            if len(self.tracker_metrics) >= 2:
                sections.append(self._comparison_section())

        report = "\n".join(sections)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")

        return report

    def _detection_section(self) -> str:
        """Generate detection metrics section."""
        m = self.detection_metrics
        lines = [
            "## Detection Metrics\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| mAP@50 | {m.map50:.4f} |",
            f"| mAP@50-95 | {m.map50_95:.4f} |",
            f"| Precision | {m.precision:.4f} |",
            f"| Recall | {m.recall:.4f} |",
            "",
        ]

        if m.per_class_ap:
            lines.append("### Per-Class AP@50\n")
            lines.append("| Class | AP@50 |")
            lines.append("|-------|-------|")
            for cls_name in CLASS_NAMES:
                ap = m.per_class_ap.get(cls_name, 0.0)
                lines.append(f"| {cls_name} | {ap:.4f} |")
            lines.append("")

        return "\n".join(lines)

    def _tracking_section(self) -> str:
        """Generate tracking metrics section."""
        lines = [
            "## Tracking Metrics\n",
            "| Metric | " + " | ".join(self.tracker_metrics.keys()) + " |",
            "|--------|" + "|".join(["-------"] * len(self.tracker_metrics)) + "|",
        ]

        metric_names = [
            ("MOTA", "mota"),
            ("MOTP", "motp"),
            ("IDF1", "idf1"),
            ("ID Switches", "id_switches"),
            ("False Positives", "num_false_positives"),
            ("Misses (FN)", "num_misses"),
            ("Matches", "num_matches"),
        ]

        for display_name, attr_name in metric_names:
            values = []
            for metrics in self.tracker_metrics.values():
                val = getattr(metrics, attr_name)
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            lines.append(f"| {display_name} | " + " | ".join(values) + " |")

        lines.append("")
        return "\n".join(lines)

    def _comparison_section(self) -> str:
        """Generate ByteTrack vs DeepSORT comparison."""
        names = list(self.tracker_metrics.keys())
        lines = [
            "## Tracker Comparison\n",
        ]

        # Determine winner per metric
        comparisons = [
            ("MOTA", "mota", True),  # higher is better
            ("MOTP", "motp", True),
            ("IDF1", "idf1", True),
            ("ID Switches", "id_switches", False),  # lower is better
        ]

        for display, attr, higher_better in comparisons:
            values = {name: getattr(self.tracker_metrics[name], attr) for name in names}
            if higher_better:
                winner = max(values, key=values.get)
            else:
                winner = min(values, key=values.get)
            lines.append(f"- **{display}**: {winner} wins ({values[winner]})")

        lines.append("")
        return "\n".join(lines)
