# Phase 8: DeepSORT + Dual Tracker Interface

**Status:** Completed
**Date:** 2026-03-21
**Tests:** 15 new tests (231 cumulative, all passed)

---

## Objective

Integrate DeepSORT via `deep-sort-realtime` with MobileNetV2 Re-ID, and build the unified `DualTracker` interface that both the evaluation suite and demo app consume.

---

## Why Both Trackers

| | ByteTrack (Phase 7) | DeepSORT (This Phase) |
|---|---|---|
| **Matching** | IoU + motion (Kalman) | IoU + motion + **appearance** (Re-ID) |
| **Re-ID** | None | MobileNetV2, 128-dim embeddings |
| **Speed** | 60+ FPS (CPU) | 25-35 FPS (CPU) |
| **Identity through occlusion** | Lost after ~30 frames | Maintained via appearance memory |
| **Best for** | Real-time demo, speed-critical | Safety evaluation, identity persistence |

The comparison is a strong interview talking point: "We implemented both and can quantitatively show when appearance-based tracking outperforms motion-only tracking."

---

## What Was Built

### 1. `TrackedObject` — Unified Output Dataclass

Both trackers produce `TrackedObject` instances with identical fields:

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | int | Unique persistent identifier |
| `bbox` | np.ndarray | [x1, y1, x2, y2] pixel coordinates |
| `confidence` | float | Detection confidence (0-1) |
| `class_id` | int | UrbanEye class ID (0-4) |
| `class_name` | str | Human-readable class name |
| `is_confirmed` | bool | Whether track has enough hits |
| `age` | int | Frames since track creation |
| `hits` | int | Number of detection matches |

`to_dict()` method produces JSON-serializable output for logging and API responses.

### 2. `DeepSORTPipeline` — Appearance-Based Tracker

**Lazy initialization:** The `deep-sort-realtime` library and MobileNetV2 Re-ID model (~15MB) are only loaded on the first `update()` call. This prevents:
- Heavy imports slowing down module loading
- Model downloads during unit testing
- Import failures when deep-sort-realtime isn't installed

**Configuration:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_age` | 70 | Longer than ByteTrack's 30 — allows Re-ID through longer occlusions |
| `n_init` | 3 | Same as ByteTrack — 3 hits to confirm |
| `max_cosine_distance` | 0.3 | Re-ID matching threshold — lower = stricter |
| `nn_budget` | 100 | Max gallery size per track — limits memory per tracked object |
| `embedder` | mobilenet | MobileNetV2 for fast 128-dim appearance features |

**Key difference from ByteTrack:** `update(detections, frame)` requires the `frame` argument because DeepSORT needs the raw image to extract appearance features. ByteTrack ignores the frame entirely.

### 3. `DualTracker` — Unified Interface

```python
# Same API regardless of backend
tracker = DualTracker(TrackerType.BYTETRACK)  # or DEEPSORT
tracks = tracker.update(detections, frame)    # Returns list[TrackedObject]
```

**`_strack_to_tracked_object()`** converts ByteTrack's internal `STrack` to the unified `TrackedObject` format. This is the adapter pattern — the evaluation suite and demo never see `STrack`.

**`from_config(yaml_path)`** reads `tracker_type` from YAML and creates the appropriate backend.

---

## Test Results

```
tests/unit/test_deepsort.py — 15 tests

TestTrackedObject (2):
  - to_dict produces serializable output ✓
  - handles float bboxes ✓

TestDeepSORTPipeline (6):
  - lazy initialization (not loaded until update) ✓
  - default config matches spec ✓
  - frame=None raises ValueError ✓
  - config loading from YAML ✓
  - reset clears state ✓
  - update with mocked tracker ✓

TestDualTracker (7):
  - BYTETRACK creates ByteTrack backend ✓
  - DEEPSORT creates DeepSORT backend ✓
  - ByteTrack returns TrackedObject ✓
  - DeepSORT requires frame ✓
  - reset works for both ✓
  - from_config loads correct type ✓
  - TrackedObject includes class name ✓
```

**All 15 new tests passed.**

---

## Files Created

```
urbaneye/tracking/deepsort_pipeline.py         # DeepSORT wrapper + TrackedObject
urbaneye/tracking/dual_tracker.py              # DualTracker unified interface
urbaneye/tracking/tracker_config/deepsort.yaml # DeepSORT hyperparameters
tests/unit/test_deepsort.py                    # 15 tests
```

---

## Key Decisions & Interview Talking Points

1. **Lazy initialization** — `DeepSORTPipeline._init_tracker()` only runs on first `update()`. This means importing the tracking module doesn't trigger a 15MB model download. Critical for CI (where deep-sort-realtime may not be installed) and for fast module loading.

2. **Adapter pattern for unified output** — `DualTracker._strack_to_tracked_object()` adapts ByteTrack's `STrack` to `TrackedObject`. The evaluation suite and demo app only know about `TrackedObject` — they're decoupled from the tracking backend.

3. **Frame required for DeepSORT, optional for ByteTrack** — This is enforced with a `ValueError`. ByteTrack is pure motion-based (IoU + Kalman), so it only needs detection coordinates. DeepSORT extracts appearance features from the image, so it physically needs the pixel data.

4. **max_age: 70 vs 30** — DeepSORT uses 70 frames (vs ByteTrack's 30) before deleting a lost track because Re-ID can re-identify an object after much longer occlusions. A pedestrian who walks behind a parked car for 3 seconds (60 frames at 20fps) can be re-identified by appearance when they reappear.

5. **TrackedObject.to_dict() for JSON logging** — Every tracked object can be serialized for the per-frame JSON tracking log (used by the evaluation suite and the demo's metrics panel).
