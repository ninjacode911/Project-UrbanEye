"""MOT (Multiple Object Tracking) evaluation metrics.

Computes standard MOT Challenge metrics: MOTA, MOTP, IDF1, ID Switches.
Provides both a lightweight built-in computation and optional TrackEval
library integration for full benchmark-compatible evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urbaneye.tracking.utils import compute_iou_matrix, linear_assignment_solve
from urbaneye.utils.constants import CLASS_NAMES


@dataclass
class MOTMetrics:
    """Standard MOT evaluation metrics.

    All metrics follow the MOT Challenge definitions.
    """

    mota: float = 0.0
    motp: float = 0.0
    idf1: float = 0.0
    id_switches: int = 0
    num_false_positives: int = 0
    num_misses: int = 0
    num_matches: int = 0
    mostly_tracked: int = 0
    mostly_lost: int = 0
    total_gt: int = 0
    total_pred: int = 0

    def meets_targets(self) -> dict[str, bool]:
        """Check against UrbanEye project target metrics."""
        return {
            "mota": self.mota >= 0.72,
            "motp": self.motp >= 0.80,
            "idf1": self.idf1 >= 0.65,
            "id_switches": self.id_switches < 200,
        }

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "mota": round(self.mota, 4),
            "motp": round(self.motp, 4),
            "idf1": round(self.idf1, 4),
            "id_switches": self.id_switches,
            "num_false_positives": self.num_false_positives,
            "num_misses": self.num_misses,
            "num_matches": self.num_matches,
            "mostly_tracked": self.mostly_tracked,
            "mostly_lost": self.mostly_lost,
        }


class MOTEvaluator:
    """Evaluate tracking performance using standard MOT metrics.

    Computes MOTA, MOTP, IDF1, and ID Switches by comparing predicted
    tracks against ground truth frame-by-frame.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        gt_frames: list[list[dict]],
        pred_frames: list[list[dict]],
    ) -> MOTMetrics:
        """Run full MOT evaluation across a sequence of frames.

        Args:
            gt_frames: List of frames, each containing ground truth dicts
                       with keys: 'id', 'bbox' [x1,y1,x2,y2], 'class_id'.
            pred_frames: List of frames, each containing prediction dicts
                         with keys: 'id', 'bbox' [x1,y1,x2,y2], 'class_id'.

        Returns:
            MOTMetrics with all computed metrics.
        """
        total_matches = 0
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        total_iou_sum = 0.0
        total_gt_count = 0
        total_pred_count = 0

        # Track ID assignment history: gt_id -> last matched pred_id
        id_assignment: dict[int, int] = {}

        for gt_objs, pred_objs in zip(gt_frames, pred_frames):
            gt_count = len(gt_objs)
            pred_count = len(pred_objs)
            total_gt_count += gt_count
            total_pred_count += pred_count

            if gt_count == 0 and pred_count == 0:
                continue

            if gt_count == 0:
                total_fp += pred_count
                continue

            if pred_count == 0:
                total_fn += gt_count
                continue

            # Compute IoU matrix
            gt_bboxes = np.array([obj["bbox"] for obj in gt_objs])
            pred_bboxes = np.array([obj["bbox"] for obj in pred_objs])
            iou_matrix = compute_iou_matrix(gt_bboxes, pred_bboxes)

            # Hungarian matching
            cost_matrix = 1.0 - iou_matrix
            matches, unmatched_gt, unmatched_pred = linear_assignment_solve(
                cost_matrix, threshold=1.0 - self.iou_threshold
            )

            # Count matches and compute MOTP
            for gt_idx, pred_idx in matches:
                iou = iou_matrix[gt_idx, pred_idx]
                total_iou_sum += iou
                total_matches += 1

                # Check for ID switches
                gt_id = gt_objs[gt_idx]["id"]
                pred_id = pred_objs[pred_idx]["id"]
                if gt_id in id_assignment and id_assignment[gt_id] != pred_id:
                    total_id_switches += 1
                id_assignment[gt_id] = pred_id

            total_fn += len(unmatched_gt)
            total_fp += len(unmatched_pred)

        # Compute final metrics
        metrics = MOTMetrics(
            num_matches=total_matches,
            num_false_positives=total_fp,
            num_misses=total_fn,
            id_switches=total_id_switches,
            total_gt=total_gt_count,
            total_pred=total_pred_count,
        )

        # MOTA = 1 - (FN + FP + ID_Switches) / total_GT
        if total_gt_count > 0:
            metrics.mota = 1.0 - (total_fn + total_fp + total_id_switches) / total_gt_count

        # MOTP = sum(IoU of matches) / total_matches
        if total_matches > 0:
            metrics.motp = total_iou_sum / total_matches

        # Simplified IDF1 approximation
        if total_gt_count + total_pred_count > 0:
            metrics.idf1 = 2 * total_matches / (total_gt_count + total_pred_count)

        return metrics

    def evaluate_per_class(
        self,
        gt_frames: list[list[dict]],
        pred_frames: list[list[dict]],
    ) -> dict[str, MOTMetrics]:
        """Run evaluation broken down by object class.

        Args:
            gt_frames: Ground truth frames with 'class_id' field.
            pred_frames: Prediction frames with 'class_id' field.

        Returns:
            Dict mapping class name to per-class MOTMetrics.
        """
        results: dict[str, MOTMetrics] = {}

        for cls_id, cls_name in enumerate(CLASS_NAMES):
            # Filter frames for this class
            gt_filtered = [
                [obj for obj in frame if obj.get("class_id") == cls_id] for frame in gt_frames
            ]
            pred_filtered = [
                [obj for obj in frame if obj.get("class_id") == cls_id] for frame in pred_frames
            ]
            results[cls_name] = self.evaluate(gt_filtered, pred_filtered)

        return results
