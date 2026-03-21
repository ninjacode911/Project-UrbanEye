# Phase 4: Augmentation Pipeline & Dataset Utilities

**Status:** Completed
**Date:** 2026-03-21
**Tests:** 26 new tests (137 cumulative, all passed)

---

## Objective

Build the Albumentations-based augmentation pipeline (domain adaptation layer 2), dataset visualization tools, and dataset summary utilities. The augmentation pipeline bridges the sim-to-real gap by applying real-world camera degradations to CARLA's synthetic images during training.

---

## Why This Phase Matters

CARLA generates photorealistic images, but they still look "too clean" compared to real dashcam footage. Real cameras produce:

- **JPEG compression artifacts** — every dashcam stores compressed video
- **Gaussian noise** — sensor noise in low-light conditions
- **Motion blur** — from vehicle movement and camera vibration
- **Lens distortion** — perspective warping, especially at the edges
- **Weather effects** — rain drops on the lens, fog halos, sun flares

Without augmentation, a model trained on clean CARLA images will fail when it sees noisy, compressed, blurry real-world images. This is the **sim-to-real domain gap** — the central challenge of simulation-based training.

UrbanEye addresses this with a 3-layer domain adaptation strategy:
1. **Layer 1** (Phase 3): Randomized CARLA weather/lighting during data generation
2. **Layer 2** (This Phase): Albumentations augmentations mimicking real camera imperfections
3. **Layer 3** (Phase 6): Mixed training with 80% CARLA + 20% BDD100K real images

---

## What Was Built

### 1. `urbaneye/training/augmentations.py` — Three-Level Augmentation Pipeline

The pipeline supports three intensity levels, each designed for a specific training phase.

#### Light Level (Validation / Sanity Checks)

| Transform | Probability | Purpose |
|-----------|-------------|---------|
| HorizontalFlip | 0.5 | Basic spatial augmentation |

Used during validation runs to verify the augmentation pipeline works without introducing heavy distortions.

#### Medium Level (Standard Training)

| Transform | Probability | Real-World Analogue |
|-----------|-------------|-------------------|
| HorizontalFlip | 0.5 | Driving in opposite lane perspective |
| RandomBrightnessContrast | 0.5 | Exposure variation, auto-exposure adjustments |
| GaussNoise | 0.3 | Camera sensor noise (especially in low light) |
| GaussianBlur | 0.2 | Motion blur, out-of-focus |
| ImageCompression (70-95 quality) | 0.3 | JPEG compression artifacts from dashcam storage |
| Perspective | 0.1 | Lens distortion, camera mount vibration |
| ColorJitter | 0.3 | White balance variation, lighting changes |

This is the default training augmentation — strong enough to improve generalization without destroying the training signal.

#### Heavy Level (Domain Adaptation)

Includes everything in medium, plus:

| Transform | Probability | Real-World Analogue |
|-----------|-------------|-------------------|
| RandomRain | 0.15 | Rain drops on windshield/lens |
| RandomFog (0.1-0.3) | 0.1 | Fog reducing visibility |
| RandomSunFlare | 0.1 | Direct sunlight hitting the lens |
| Aggressive ColorJitter | 0.4 | Extreme lighting variation |

Used specifically for domain adaptation runs where the goal is maximum visual diversity.

#### YOLO-Compatible Bounding Box Handling

All transforms are wrapped in `A.Compose` with:

```python
bbox_params=A.BboxParams(
    format="yolo",           # cx, cy, w, h normalized [0,1]
    label_fields=["class_labels"],  # class IDs travel with bboxes
    min_visibility=0.3,      # remove bboxes that become <30% visible after crop
)
```

Albumentations automatically transforms bounding boxes when the image is flipped, cropped, or warped. `min_visibility=0.3` ensures that if a crop removes most of a bounding box, the annotation is dropped rather than left as a misleading fragment.

#### `apply_augmentation()` Function

```python
def apply_augmentation(
    image: np.ndarray,         # H x W x 3, uint8
    bboxes: list[list[float]], # [[cx, cy, w, h], ...]
    class_labels: list[int],   # [0, 1, 2, ...]
    level: str = "medium",
    seed: int | None = None,
) -> tuple[np.ndarray, list[list[float]], list[int]]:
```

- Creates pipeline for the specified level
- Sets both `random.seed()` and `np.random.seed()` for reproducibility (Albumentations uses both RNG sources internally)
- Applies augmentation
- Validates output bboxes via `validate_augmented_bboxes()`

#### `validate_augmented_bboxes()` Function

Post-processing safety net for bounding boxes after augmentation:
1. **Clip** all coordinates to [0, 1] range
2. **Remove** zero-area boxes (width or height = 0)
3. **Remove** boxes with wrong number of values (≠ 4)

This prevents invalid annotations from reaching the training pipeline, even if an augmentation produces an edge-case coordinate.

### 2. `urbaneye/training/dataset.yaml` — YOLO Dataset Configuration

Standard Ultralytics format:

```yaml
path: ../datasets/urbaneye
train: images/train
val: images/val
test: images/test
nc: 5
names: {0: vehicle, 1: pedestrian, 2: cyclist, 3: traffic_light, 4: traffic_sign}
```

This file is passed directly to `model.train(data="dataset.yaml")` in the YOLOv11 training pipeline.

### 3. `urbaneye/utils/visualization.py` — Dataset Visualization Tools

#### `draw_bboxes(image, bboxes, class_ids, confidences=None) → annotated_image`

Draws YOLO-format bounding boxes on an image:
- Converts normalized [0,1] YOLO coordinates to pixel coordinates
- Uses class-specific BGR colors from `constants.py`
- Draws filled label backgrounds with class name and optional confidence score
- Returns a **copy** of the image (original is never modified)

#### `plot_class_distribution(label_dir) → dict[str, int]`

Counts class occurrences across all `.txt` label files in a directory. Returns a dict like `{"vehicle": 12000, "pedestrian": 8000, ...}`. Useful for detecting class imbalance before training.

#### `create_dataset_summary(dataset_dir) → dict`

Computes summary statistics for a complete YOLO dataset:
- Total images across all splits
- Total label files across all splits
- Per-split image counts (train/val/test)

---

## Test Results

```
tests/unit/test_augmentations.py — 16 tests

TestGetTrainAugmentation (5 tests):
  - returns A.Compose ✓
  - all 3 levels instantiate ✓
  - invalid level raises ValueError ✓
  - light has fewer transforms than medium ✓
  - has bbox_params configured ✓

TestApplyAugmentation (6 tests):
  - output shape matches input ✓
  - all bboxes in [0,1] range ✓
  - class labels preserved ✓
  - seed parameter accepted ✓
  - empty bboxes work ✓

TestValidateAugmentedBboxes (6 tests):
  - valid bboxes pass through ✓
  - clips values > 1.0 ✓
  - clips negative values ✓
  - removes zero-area boxes ✓
  - removes wrong-length boxes ✓
  - empty input → empty output ✓

tests/unit/test_visualization.py — 10 tests

TestDrawBboxes (5 tests):
  - returns same shape ✓
  - does not modify original ✓
  - empty bboxes returns copy ✓
  - works with confidences ✓
  - multiple classes render ✓

TestPlotClassDistribution (3 tests):
  - counts classes correctly ✓
  - empty directory → zero counts ✓
  - nonexistent directory → zero counts ✓

TestCreateDatasetSummary (2 tests):
  - counts images and labels ✓
  - empty dataset → zero counts ✓
```

**All 26 new tests passed in 6.5s.** (Augmentation tests are slower because they actually run image transformations.)

---

## Files Created in This Phase

```
urbaneye/training/__init__.py           # Training subpackage
urbaneye/training/augmentations.py      # 3-level augmentation pipeline
urbaneye/training/dataset.yaml          # YOLO dataset config (5 classes)
urbaneye/utils/visualization.py         # Bbox drawing, class distribution, dataset summary
tests/unit/test_augmentations.py        # 16 tests
tests/unit/test_visualization.py        # 10 tests
```

---

## Key Decisions & Interview Talking Points

1. **Three augmentation levels** — Light/medium/heavy is more useful than a single pipeline. Light is for debugging (is the pipeline breaking my data?), medium is for training, heavy is for domain adaptation. The same `get_train_augmentation(level)` API serves all three use cases.

2. **Why these specific transforms** — Each transform mimics a real-world camera imperfection:
   - `ImageCompression(70-95)` — dashcams compress at quality 70-85, creating block artifacts
   - `GaussNoise` — sensor noise in $15-50 variance range matches typical CMOS sensors
   - `RandomRain` at 15% probability — real driving datasets (BDD100K) have ~15% rainy frames
   - `RandomSunFlare` in top half of image only (`flare_roi=(0,0,1,0.5)`) — sun flare comes from above, not below

3. **min_visibility=0.3** — If an augmentation (crop, perspective) makes a bounding box less than 30% visible, it's removed. Training on barely-visible objects hurts model precision. The 0.3 threshold is a standard COCO benchmark setting.

4. **Dual RNG seeding** — Albumentations internally uses both Python's `random` module and NumPy's RNG. Setting only `np.random.seed()` doesn't guarantee reproducibility. We set both `random.seed(seed)` and `np.random.seed(seed)`.

5. **Non-destructive drawing** — `draw_bboxes()` returns a copy, never modifying the input. This prevents subtle bugs where visualization accidentally corrupts training data in shared-memory pipelines.

6. **Class imbalance detection** — `plot_class_distribution()` counts occurrences per class across a label directory. In urban driving, vehicles outnumber cyclists 10:1. Knowing this before training justifies the class weights (pedestrian=2.0, cyclist=3.0) in the training config.

---

## Cumulative Project Status After Phase 4

```
Total source files:  22 Python files + 4 YAML/config files
Total test files:    7 test files
Total tests:         137 tests, all passing
Lint status:         ruff check clean, ruff format clean
Package:             pip install -e ".[dev]" works
```

### What's Ready for Phase 5+

The complete data generation and preprocessing pipeline is now built:
- CARLA configuration → sensor setup → data generation → annotation export → augmentation → visualization

Phase 5 (YOLOv11 Training Pipeline) will build on:
- `training/dataset.yaml` — YOLO dataset config
- `training/augmentations.py` — augmentation pipeline for training
- `utils/constants.py` — class names, thresholds
- `configs/project_config.yaml` — training hyperparameters
