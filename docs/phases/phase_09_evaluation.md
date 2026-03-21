# Phase 9: MOT Evaluation Suite

**Status:** Completed
**Date:** 2026-03-21
**Tests:** 26 new tests (257 cumulative, all passed)

---

## Objective

Build the complete evaluation infrastructure: MOT tracking metrics (MOTA, MOTP, IDF1, ID Switches), detection metrics (mAP, precision, recall), and an automated Markdown report generator.

---

## Why Rigorous Evaluation Matters

This separates a demo from a real system. MOT metrics (MOTA, IDF1) are standard interview questions for tracking positions. Having automated evaluation means:
- Every model change is quantitatively assessed
- ByteTrack vs DeepSORT comparison is data-driven, not anecdotal
- Per-class breakdowns reveal which object types need more training data

---

## What Was Built

### 1. `MOTEvaluator` — Tracking Metrics

**MOTA (Multiple Object Tracking Accuracy):**
```
MOTA = 1 - (FN + FP + ID_Switches) / Total_GT_Detections
```
Primary composite metric. MOTA = 1.0 means perfect tracking. Can be negative if errors exceed GT count.

**MOTP (Multiple Object Tracking Precision):**
```
MOTP = Σ(IoU of matched pairs) / Total_Matches
```
Measures bounding box localization quality of matched pairs. Independent of FP/FN.

**IDF1 (ID F1 Score):**
```
IDF1 = 2 × IDTP / (Total_GT + Total_Pred)
```
Measures identity preservation — how well the tracker maintains consistent IDs.

**ID Switches:** Count of times a tracked object's predicted ID changes for the same GT object.

**Algorithm:** Frame-by-frame Hungarian matching between GT and predicted tracks using IoU cost matrix. Maintains an `id_assignment` dict to detect ID switches across frames.

**`evaluate_per_class()`** runs separate evaluations for each of the 5 classes, enabling targeted analysis (e.g., "MOTA is 0.80 for vehicles but only 0.55 for cyclists").

### 2. `DetectionEvaluator` — Detection mAP

**AP computation:** All-point interpolation (COCO-style) with monotonically decreasing precision envelope.

**Per-class AP:** Computes AP@50 independently for each of the 5 classes. Overall mAP@50 is the mean across classes that have GT.

### 3. `ReportGenerator` — Markdown Reports

Generates structured evaluation reports with:
- Detection metrics table (mAP, precision, recall)
- Per-class AP table
- Tracking metrics table (one column per tracker)
- ByteTrack vs DeepSORT comparison with per-metric winners

---

## Test Results (26 new tests)

```
TestMOTEvaluator (8): perfect tracking ✓, all FP ✓, all misses ✓,
  ID switch detected ✓, MOTP = avg IoU ✓, empty sequence ✓,
  per-class evaluation ✓

TestDetectionEvaluator (7): perfect detection ✓, no detections ✓,
  no GT ✓, per-class AP ✓, AP computation ✓

TestReportGenerator (6): markdown headers ✓, tracking section ✓,
  comparison section ✓, file output ✓, per-class AP table ✓, empty report ✓
```

---

## Files Created

```
urbaneye/evaluation/__init__.py
urbaneye/evaluation/mot_evaluator.py        # MOTMetrics, MOTEvaluator
urbaneye/evaluation/detection_evaluator.py  # DetectionMetrics, DetectionEvaluator, compute_ap
urbaneye/evaluation/generate_report.py      # ReportGenerator (Markdown)
tests/unit/test_mot_evaluator.py            # 10 tests
tests/unit/test_detection_evaluator.py      # 10 tests
tests/unit/test_generate_report.py          # 6 tests
```

---

## Key Decisions & Interview Talking Points

1. **Lightweight MOTA without TrackEval** — The built-in evaluator computes MOTA/MOTP/IDF1 in ~50 lines using our existing IoU and Hungarian utilities. This avoids a heavy TrackEval dependency while matching the same mathematical definitions.

2. **MOTA can be zero, not negative, for all-miss** — When FN = total_GT: MOTA = 1 - GT/GT = 0. MOTA only goes negative when FP + ID_Switches > 0 alongside FN. This is a common misconception worth clarifying in interviews.

3. **IDF1 over MOTA for identity evaluation** — MOTA penalizes FP and FN equally with ID switches. IDF1 specifically measures identity preservation. For autonomous driving (where "is this the same pedestrian?" matters for trajectory prediction), IDF1 is more informative.
