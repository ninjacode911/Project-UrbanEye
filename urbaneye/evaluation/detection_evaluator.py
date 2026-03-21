"""Detection evaluation metrics — mAP, precision, recall.

Computes COCO-style detection metrics for UrbanEye's 5 classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from urbaneye.tracking.utils import compute_iou
from urbaneye.utils.constants import CLASS_NAMES


@dataclass
class DetectionMetrics:
    """Detection evaluation results."""

    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    per_class_ap: dict[str, float] = field(default_factory=dict)

    def meets_targets(self, domain: str = "carla") -> dict[str, bool]:
        """Check against project target metrics."""
        if domain == "carla":
            return {"map50": self.map50 >= 0.70}
        return {"map50": self.map50 >= 0.50}

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "map50": round(self.map50, 4),
            "map50_95": round(self.map50_95, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "per_class_ap": {k: round(v, 4) for k, v in self.per_class_ap.items()},
        }


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    """Compute Average Precision using the all-point interpolation method.

    Args:
        recalls: List of recall values (sorted ascending).
        precisions: List of precision values.

    Returns:
        Average Precision value.
    """
    if not recalls or not precisions:
        return 0.0

    # Add sentinel values
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]

    # Monotonically decreasing precision
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Compute area under curve
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


class DetectionEvaluator:
    """Evaluate detection performance with mAP metrics.

    Args:
        iou_threshold: IoU threshold for matching (default 0.5 for mAP@50).
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        gt_boxes: list[dict],
        pred_boxes: list[dict],
    ) -> DetectionMetrics:
        """Compute detection metrics.

        Args:
            gt_boxes: List of ground truth dicts with 'bbox', 'class_id'.
            pred_boxes: List of prediction dicts with 'bbox', 'class_id', 'confidence'.

        Returns:
            DetectionMetrics with mAP and per-class AP.
        """
        if not gt_boxes and not pred_boxes:
            return DetectionMetrics()

        # Sort predictions by confidence (descending)
        pred_sorted = sorted(pred_boxes, key=lambda x: x.get("confidence", 0), reverse=True)

        per_class_ap: dict[str, float] = {}
        total_tp = 0
        total_fp = 0
        total_gt = len(gt_boxes)

        for cls_id, cls_name in enumerate(CLASS_NAMES):
            cls_gt = [g for g in gt_boxes if g["class_id"] == cls_id]
            cls_pred = [p for p in pred_sorted if p["class_id"] == cls_id]

            if not cls_gt:
                per_class_ap[cls_name] = 0.0
                total_fp += len(cls_pred)
                continue

            if not cls_pred:
                per_class_ap[cls_name] = 0.0
                continue

            # Match predictions to ground truth
            matched_gt = set()
            tp_list: list[int] = []
            fp_list: list[int] = []

            for pred in cls_pred:
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(cls_gt):
                    if gt_idx in matched_gt:
                        continue
                    iou = compute_iou(
                        np.array(pred["bbox"]),
                        np.array(gt["bbox"]),
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched_gt.add(best_gt_idx)
                    total_tp += 1
                else:
                    tp_list.append(0)
                    fp_list.append(1)
                    total_fp += 1

            # Compute precision-recall curve
            tp_cumsum = np.cumsum(tp_list)
            fp_cumsum = np.cumsum(fp_list)
            recalls = (tp_cumsum / len(cls_gt)).tolist()
            precisions = (tp_cumsum / (tp_cumsum + fp_cumsum)).tolist()

            per_class_ap[cls_name] = compute_ap(recalls, precisions)

        # Overall mAP is mean of per-class APs (only classes with GT)
        valid_aps = [
            ap
            for cls_name, ap in per_class_ap.items()
            if any(g["class_id"] == CLASS_NAMES.index(cls_name) for g in gt_boxes)
        ]
        map50 = float(np.mean(valid_aps)) if valid_aps else 0.0

        # Precision and recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0

        return DetectionMetrics(
            map50=map50,
            precision=precision,
            recall=recall,
            per_class_ap=per_class_ap,
        )
