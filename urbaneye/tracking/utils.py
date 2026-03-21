"""Tracking utility functions — IoU computation, bbox conversion, assignment.

All functions are pure numpy — no external tracking library dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Compute IoU between two bounding boxes in [x1, y1, x2, y2] format.

    Args:
        bbox_a: First bounding box (4,).
        bbox_b: Second bounding box (4,).

    Returns:
        Intersection over Union value in [0, 1].
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def compute_iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute NxM IoU matrix between two sets of bounding boxes.

    Args:
        bboxes_a: First set of boxes (N, 4) in [x1, y1, x2, y2] format.
        bboxes_b: Second set of boxes (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        IoU matrix of shape (N, M).
    """
    n = len(bboxes_a)
    m = len(bboxes_b)
    iou = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            iou[i, j] = compute_iou(bboxes_a[i], bboxes_b[j])

    return iou


def linear_assignment_solve(
    cost_matrix: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve linear assignment problem with threshold filtering.

    Uses the Hungarian Algorithm via scipy to find optimal matching,
    then filters out matches above the cost threshold.

    Args:
        cost_matrix: NxM cost matrix (lower is better).
        threshold: Maximum cost for a valid match.

    Returns:
        Tuple of (matches, unmatched_rows, unmatched_cols).
        matches: List of (row, col) index pairs.
        unmatched_rows: Row indices with no valid match.
        unmatched_cols: Column indices with no valid match.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= threshold:
            matches.append((r, c))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)


def tlbr_to_tlwh(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x1, y1, w, h]."""
    result = bbox.copy()
    result[2] = bbox[2] - bbox[0]
    result[3] = bbox[3] - bbox[1]
    return result


def tlwh_to_xyah(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, w, h] to [cx, cy, aspect_ratio, h]."""
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    a = bbox[2] / bbox[3] if bbox[3] > 0 else 0
    return np.array([cx, cy, a, bbox[3]])


def xyah_to_tlbr(bbox: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, aspect_ratio, h] to [x1, y1, x2, y2]."""
    w = bbox[2] * bbox[3]
    h = bbox[3]
    x1 = bbox[0] - w / 2
    y1 = bbox[1] - h / 2
    x2 = bbox[0] + w / 2
    y2 = bbox[1] + h / 2
    return np.array([x1, y1, x2, y2])
