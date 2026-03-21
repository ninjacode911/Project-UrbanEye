# Phase 7: ByteTrack Real-Time Tracker

**Status:** Completed
**Date:** 2026-03-21
**Tests:** 36 new tests (216 cumulative, all passed)

---

## Objective

Implement ByteTrack multi-object tracker from scratch — Kalman Filter for motion prediction, Hungarian Algorithm for optimal assignment, and the two-stage matching strategy that is ByteTrack's key innovation. No external tracking library dependency.

---

## Why From Scratch

1. **Interview depth** — "I implemented the Kalman Filter and Hungarian Algorithm from scratch" demonstrates deeper understanding than "I called a library"
2. **No unmaintained dependency** — ByteTrack Python packages are often outdated or have complex CUDA dependencies
3. **Customizable** — We can add per-class tracking pools, custom gating, and velocity estimation
4. **Only ~400 lines of core logic** — The algorithm is elegant and compact
5. **Dependencies: only numpy + scipy** — No compiled extensions or GPU requirements

---

## What Was Built

### 1. `urbaneye/tracking/kalman_filter.py` — Kalman Filter (8D State)

**State vector (8D):** `[cx, cy, aspect_ratio, height, vx, vy, va, vh]`
- (cx, cy): bounding box center
- a: aspect ratio (width/height) — tracks shape changes during occlusion
- h: height — tracks scale changes as objects approach/recede
- (vx, vy, va, vh): respective velocities

**Measurement vector (4D):** `[cx, cy, aspect_ratio, height]`

**Constant-velocity motion model:** `F` (8x8 state transition) adds velocity to position each timestep. This simple model works well for frame-to-frame tracking at 20+ FPS where objects don't accelerate much between frames.

**Key methods:**
| Method | What It Does | Math |
|--------|-------------|------|
| `initiate(measurement)` | Create new track state | `mean = [m, 0_vel]`, `cov = diag(σ²)` |
| `predict(mean, cov)` | Project forward 1 frame | `x' = F·x`, `P' = F·P·Fᵀ + Q` |
| `update(mean, cov, measurement)` | Correct with detection | `K = P·Hᵀ·S⁻¹`, `x = x + K·(z - H·x)` |
| `gating_distance(mean, cov, measurements)` | Mahalanobis distance | `d² = (z-μ)ᵀ·S⁻¹·(z-μ)` |

**Process noise `Q`** scales with object height — larger objects have proportionally more uncertainty. This prevents the filter from being overconfident about small objects (which are harder to localize precisely).

### 2. `urbaneye/tracking/bytetrack_pipeline.py` — Two-Stage Tracker

#### ByteTrack's Key Innovation (ECCV 2022)

Traditional trackers only match **high-confidence** detections (score ≥ 0.6) with tracks. ByteTrack's insight: **low-confidence detections** (score 0.1-0.6) are often partially occluded objects. Discarding them loses valuable tracking information.

```
Frame N detections
    │
    ├── High confidence (≥ 0.6) ──> Stage 1: Match with active tracks
    │                                   │
    │                                   ├── Matched → Update track
    │                                   └── Unmatched tracks → go to Stage 2
    │
    └── Low confidence (0.1-0.6) ──> Stage 2: Match with remaining tracks
                                        │
                                        ├── Matched → Rescue occluded track!
                                        └── Unmatched → mark track as LOST
```

#### Track Lifecycle State Machine

```
Detection appears → NEW (unconfirmed, not rendered)
    │
    ├── Matched 3 consecutive frames → ACTIVE (confirmed, rendered in output)
    │       │
    │       ├── Continues matching → stays ACTIVE
    │       └── No match found → LOST (Kalman predicts position)
    │               │
    │               ├── Re-matched within 30 frames → back to ACTIVE
    │               └── 30 frames without match → DELETED (removed)
    │
    └── Never reaches 3 matches → DELETED
```

#### `ByteTrackPipeline.update(detections)` Algorithm

```python
def update(self, detections: np.ndarray) -> list[STrack]:
    # 1. Split detections by confidence
    high_dets = detections[scores >= 0.6]
    low_dets = detections[0.1 <= scores < 0.6]

    # 2. Predict all existing tracks forward (Kalman)
    for track in self.tracks:
        track.predict(self.kf)

    # 3. Stage 1: High-conf ↔ active tracks (IoU + Hungarian)
    matches_1, unmatched_tracks, unmatched_dets = associate(active, high_dets)

    # 4. Stage 2: Low-conf ↔ remaining unmatched tracks
    matches_2, still_unmatched, _ = associate(remaining, low_dets)

    # 5. Mark unmatched active → LOST, old LOST → DELETED
    # 6. Try matching lost tracks with unmatched high-conf dets
    # 7. Create new tracks from unmatched high-conf detections
    # 8. Return confirmed active tracks
```

#### `STrack` Dataclass

Each tracked object stores: `track_id`, `bbox`, `score`, `class_id`, `state`, Kalman `mean`/`covariance`, `hits`, `time_since_update`.

### 3. `urbaneye/tracking/utils.py` — Tracking Utilities

| Function | Purpose |
|----------|---------|
| `compute_iou(a, b)` | Single-pair IoU in [x1,y1,x2,y2] format |
| `compute_iou_matrix(A, B)` | NxM IoU matrix for batch association |
| `linear_assignment_solve(cost, threshold)` | Hungarian Algorithm with threshold (scipy) |
| `tlbr_to_tlwh`, `tlwh_to_xyah`, `xyah_to_tlbr` | Bbox format converters (roundtrip-safe) |

### 4. `tracker_config/bytetrack.yaml` — Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `high_thresh` | 0.6 | Stage 1 confidence cutoff |
| `low_thresh` | 0.1 | Stage 2 confidence cutoff |
| `match_thresh` | 0.8 | IoU threshold for matching |
| `max_lost` | 30 | Frames before deleting lost track |
| `min_hits` | 3 | Frames before confirming new track |

---

## Test Results

```
tests/unit/test_kalman_filter.py — 13 tests

TestKalmanFilterInit (5):
  - 8D state vector ✓
  - 8x8 covariance ✓
  - position matches measurement ✓
  - initial velocity is zero ✓
  - covariance is positive definite ✓

TestKalmanFilterPredict (3):
  - moves state by velocity ✓
  - increases uncertainty ✓
  - preserves symmetry ✓

TestKalmanFilterUpdate (3):
  - pulls toward measurement ✓
  - reduces uncertainty ✓
  - predict-update cycle converges to true velocity ✓

TestKalmanFilterGating (2):
  - close measurement → low distance ✓
  - far measurement → high distance ✓

tests/unit/test_bytetrack.py — 23 tests

TestComputeIoU (4):
  - identical boxes = 1.0 ✓, no overlap = 0.0 ✓
  - partial overlap in (0,1) ✓, known value matches ✓

TestComputeIoUMatrix (3):
  - correct shape ✓, diagonal for same = 1.0 ✓, empty input ✓

TestLinearAssignment (3):
  - perfect matching ✓, threshold filtering ✓, empty matrix ✓

TestBboxConversions (3):
  - tlbr→tlwh ✓, tlwh→xyah ✓, roundtrip recovers original ✓

TestByteTrackPipeline (10):
  - empty detections → empty tracks ✓
  - single detection → NEW track ✓
  - NEW → ACTIVE after min_hits ✓
  - ACTIVE → LOST when unmatched ✓
  - LOST → DELETED after max_lost ✓
  - track IDs monotonically increasing ✓
  - two-stage matching works ✓
  - reset clears state ✓
  - multiple objects tracked separately ✓
  - class ID preserved ✓
```

**All 36 new tests passed.**

---

## Files Created

```
urbaneye/tracking/__init__.py                 # Tracking subpackage
urbaneye/tracking/kalman_filter.py            # 8D Kalman Filter (~150 lines)
urbaneye/tracking/bytetrack_pipeline.py       # ByteTrack tracker (~280 lines)
urbaneye/tracking/utils.py                    # IoU, Hungarian, bbox converters (~100 lines)
urbaneye/tracking/tracker_config/bytetrack.yaml  # Hyperparameters
tests/unit/test_kalman_filter.py              # 13 tests
tests/unit/test_bytetrack.py                  # 23 tests
```

---

## Key Decisions & Interview Talking Points

1. **Aspect ratio in state vector, not width** — The Kalman state uses `[cx, cy, aspect_ratio, height]` rather than `[cx, cy, width, height]`. Aspect ratio changes less than width/height during occlusion (a car's aspect ratio stays ~2:1 even as it gets partially hidden). This makes the motion model more stable.

2. **Process noise scales with object height** — `σ_position = height / 20`. Larger objects (trucks at 200px) have more positional uncertainty than small objects (traffic lights at 30px). This prevents the filter from being overconfident about large objects.

3. **Two-stage matching rescues occluded objects** — In a dense traffic scene, a partially occluded pedestrian might have confidence 0.3 (below typical 0.5 threshold). ByteTrack's Stage 2 matches this detection with the existing track, maintaining identity through the occlusion. This is the paper's primary contribution.

4. **Cholesky decomposition for Kalman gain** — We use `scipy.linalg.cho_factor` + `cho_solve` instead of explicit matrix inversion. Cholesky is numerically stable and ~2x faster than `np.linalg.inv` for symmetric positive-definite matrices.

5. **Empty-detection ACTIVE→LOST transition bug** — Initially, the empty-detection branch only incremented `time_since_update` but didn't transition ACTIVE→LOST. Caught by tests. This is why testing the tracker lifecycle is critical — state machine bugs are invisible until edge cases hit.

6. **IoU cost matrix, not distance** — We compute `cost = 1 - IoU` so that the Hungarian Algorithm minimizes cost (maximizing IoU). The threshold `1 - match_thresh + 1.0` converts the IoU threshold to cost space.
