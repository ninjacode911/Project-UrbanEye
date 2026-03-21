# Phase 6: Domain Adaptation Pipeline

**Status:** Completed
**Date:** 2026-03-21
**Tests:** 24 new tests (180 cumulative, all passed)

---

## Objective

Implement Layer 3 of the 3-layer domain adaptation strategy: BDD100K annotation conversion, mixed dataset creation, and fine-tuning configuration with backbone freezing.

---

## Why This Phase Matters

The **sim-to-real domain gap** is the central challenge of simulation-based training. A model trained purely on synthetic data (CARLA) will fail on real-world images because:
- CARLA textures are too clean (no sensor noise, no compression artifacts)
- CARLA lighting is physically simulated but lacks real-world variation
- CARLA object diversity is limited (fewer vehicle makes/models)

UrbanEye addresses this with three layers:
1. **Layer 1 (Phase 3):** Randomized CARLA weather/lighting during data generation
2. **Layer 2 (Phase 4):** Albumentations augmentations (noise, blur, JPEG artifacts)
3. **Layer 3 (This Phase):** Mixed training with 80% CARLA + 20% BDD100K real-world images

Layer 3 is the most impactful: by fine-tuning on real images with a frozen backbone and 10x lower learning rate, the model adapts its detection head to real-world appearance patterns without losing the features learned from synthetic data.

---

## What Was Built

### 1. `BDD100KAdapter` — Annotation Format Converter

Converts BDD100K's JSON annotation format (10 categories) to UrbanEye's YOLO format (5 classes).

**BDD100K → UrbanEye class mapping (all 10 classes covered):**

| BDD100K Class | UrbanEye Class | ID | Rationale |
|---------------|---------------|-----|-----------|
| car | vehicle | 0 | Primary collision risk |
| truck | vehicle | 0 | Same tracking behavior as cars |
| bus | vehicle | 0 | Large vehicle variant |
| train | vehicle | 0 | Rare but maps to vehicle |
| pedestrian | pedestrian | 1 | Direct match |
| rider | cyclist | 2 | Person on bicycle/motorcycle |
| bicycle | cyclist | 2 | Unmanned bicycle still a cyclist hazard |
| motorcycle | cyclist | 2 | Similar size/behavior to bicycles |
| traffic light | traffic_light | 3 | Direct match |
| traffic sign | traffic_sign | 4 | Direct match |

**Key method:** `convert_annotations(bdd_json_path, output_dir) → count`
- Parses BDD100K JSON (frame-level structure with `name` and `labels` fields)
- Converts `box2d` pixel coordinates to YOLO normalized format (BDD100K images are 1280x720)
- Skips unmapped categories (e.g., "dog" in COCO-overlap)
- Tracks per-class conversion statistics
- Returns count of images with valid annotations

### 2. `DomainAdaptConfig` — Fine-Tuning Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lr0` | 0.0001 | 10x lower than initial training — prevents catastrophic forgetting |
| `freeze_layers` | 10 | Freeze first 10 backbone layers — preserve feature extraction |
| `epochs` | 20 | Short fine-tune — enough to adapt, not enough to overfit |
| `carla_ratio` | 0.8 | 80% synthetic data maintains volume |
| `bdd100k_ratio` | 0.2 | 20% real data closes the domain gap |
| `patience` | 10 | Aggressive early stopping for fine-tuning |

**Validation enforces:** ratios sum to 1.0, LR ≤ 0.01 (higher is suspicious for fine-tuning), non-negative freeze layers. The `to_ultralytics_kwargs()` method passes `freeze=10` which Ultralytics uses to freeze the first N backbone layers.

### 3. `create_mixed_dataset()` — Dataset Mixer

Combines two YOLO datasets at a specified ratio:
1. Collects images from both `primary/images/train/` and `secondary/images/train/`
2. Samples according to `primary_ratio` (0.8 by default)
3. Copies images and labels to a unified output directory
4. Generates `dataset.yaml` pointing to the mixed data
5. Deterministic via `seed=42`

---

## Test Results

```
tests/unit/test_domain_adapt.py — 24 tests

TestBDD100KClassMapping (8 tests):
  - all 10 BDD100K classes mapped ✓
  - car → vehicle ✓
  - truck → vehicle ✓
  - bus → vehicle ✓
  - pedestrian → pedestrian ✓
  - rider → cyclist ✓
  - bicycle → cyclist ✓
  - traffic light → traffic_light ✓
  - traffic sign → traffic_sign ✓

TestBDD100KAdapter (5 tests):
  - converts valid annotations (2 of 3 frames have driving objects) ✓
  - skips unmapped classes (dog filtered out) ✓
  - output in correct YOLO format (5 fields, normalized [0,1]) ✓
  - stats tracking per class ✓
  - filter_by_classes keeps only mapped categories ✓

TestDomainAdaptConfig (6 tests):
  - default values match spec ✓
  - valid config → no errors ✓
  - ratios must sum to 1.0 ✓
  - lr > 0.01 caught (too high for fine-tuning) ✓
  - negative freeze layers caught ✓
  - to_ultralytics_kwargs includes freeze parameter ✓

TestCreateMixedDataset (4 tests):
  - creates valid dataset.yaml ✓
  - creates image directories ✓
  - copies labels alongside images ✓
  - respects primary/secondary ratio ✓
```

**All 24 new tests passed.**

---

## Files Created

```
urbaneye/training/domain_adapt.py        # BDD100KAdapter, DomainAdaptConfig, create_mixed_dataset, fine_tune
tests/unit/test_domain_adapt.py          # 24 tests
```

---

## Key Decisions & Interview Talking Points

1. **10x lower learning rate (0.0001 vs 0.001)** — Fine-tuning with the original LR would erase the features learned from CARLA data ("catastrophic forgetting"). The 10x reduction lets the detection head adapt while preserving backbone representations.

2. **Freeze 10 backbone layers** — The early layers extract low-level features (edges, textures) that transfer well between domains. The later layers and detection head are domain-specific and need to adapt.

3. **80/20 ratio, not 50/50** — The synthetic data provides volume and diversity (50K frames, 144 weather/map configs). Real data provides appearance grounding. Too much real data risks overfitting to BDD100K's specific camera and geography.

4. **BDD100K's 10→5 class mapping** — Multiple BDD100K categories collapse into one UrbanEye class. `rider`, `bicycle`, and `motorcycle` all become `cyclist` because from a perception standpoint they have similar size, speed, and tracking behavior. The model doesn't need to distinguish a ridden bicycle from a motorcycle — both are "cyclist-scale road users."

5. **Validation enforces LR ceiling** — `lr0 > 0.01` is flagged as an error because no reasonable fine-tuning should use a learning rate that high. This catches copy-paste errors from the initial training config.
