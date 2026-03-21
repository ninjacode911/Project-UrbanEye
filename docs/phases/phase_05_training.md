# Phase 5: YOLOv11 Training Pipeline

**Status:** Completed
**Date:** 2026-03-21
**Tests:** 19 new tests (156 cumulative, all passed)
**Training:** YOLOv11n, 50 epochs, 41.9 min on RTX 5070 Laptop

---

## Objective

Build the complete YOLOv11 training pipeline — configuration dataclass with validation, Ultralytics kwargs conversion, GPU-specific factory methods, dataset preparation from COCO, and a working baseline model trained on real hardware.

---

## Why This Phase Matters

This is where UrbanEye transitions from infrastructure to actual machine learning. The training pipeline must:

1. **Work on multiple GPU tiers** — RTX 5070 Laptop (8.5GB VRAM) and Kaggle T4 (16GB VRAM) need different batch sizes
2. **Validate configurations before training** — A typo in `lr0` shouldn't waste 4 hours of GPU time
3. **Produce deployment-ready artifacts** — best.pt for further training, best.onnx for HuggingFace demo
4. **Establish baseline metrics** — Without a baseline, we can't measure the improvement from CARLA data and domain adaptation

---

## What Was Built

### 1. `urbaneye/training/train_yolov11.py` — Training Configuration Module

#### `TrainConfig` Dataclass

All YOLOv11 hyperparameters in a single, validated dataclass:

| Parameter | Default | Spec Source | Notes |
|-----------|---------|-------------|-------|
| `model_variant` | `yolo11n.pt` | PDF Section 7.2 | n (nano), s, m, l, x sizes |
| `epochs` | 50 | Baseline; 100 for full run | Early stopping via `patience=20` |
| `batch_size` | 16 | Fits YOLOv11n in 8.5GB | 8 for YOLOv11m |
| `img_size` | 640 | Standard YOLO training | Must be divisible by 32 |
| `lr0` | 0.001 | PDF Section 7.2 | Initial learning rate |
| `lrf` | 0.01 | PDF Section 7.2 | Final LR = lr0 * lrf (cosine) |
| `momentum` | 0.937 | PDF Section 7.2 | AdamW momentum |
| `weight_decay` | 0.0005 | PDF Section 7.2 | L2 regularization |
| `warmup_epochs` | 3 | PDF Section 7.2 | Linear warmup |
| `mosaic` | 1.0 | PDF Section 7.2 | 4-image mosaic augmentation |
| `mixup` | 0.1 | PDF Section 7.2 | Image blending augmentation |
| `amp` | True | — | Automatic Mixed Precision (FP16) |

#### `validate()` Method

Returns a list of error strings. Catches 12 classes of invalid configurations:
- Negative/zero epochs, batch size, image size
- Image size not divisible by 32 (YOLO architecture requirement)
- Learning rate out of (0, 1] range
- Momentum, mosaic, mixup out of [0, 1] range
- Negative weight decay, workers
- Unknown model variant (only yolo11n/s/m/l/x.pt accepted)
- Invalid export format (only onnx/torchscript/openvino/engine/coreml)

**Why validate before training:** A training run takes hours. Catching a bad config at startup instead of mid-training saves significant time and GPU cost.

#### `to_ultralytics_kwargs()` Method

Converts the dataclass to the exact dict format Ultralytics `model.train()` expects. Key translations:
- `batch_size` → `batch` (Ultralytics uses `batch`)
- `img_size` → `imgsz` (Ultralytics uses `imgsz`)

This separation of our internal naming from Ultralytics' API means if Ultralytics changes parameter names in a future version, we only update one method.

#### Factory Class Methods

Two GPU-specific presets:

```python
TrainConfig.for_rtx5070_laptop("n")  # batch=16, fits 8.5GB
TrainConfig.for_rtx5070_laptop("m")  # batch=8, fits 8.5GB
TrainConfig.for_kaggle_t4("m")       # batch=16, fits 16GB, 100 epochs
```

These encode hardware-specific knowledge so users don't need to manually calculate VRAM budgets.

#### `train()` and `export_model()` Functions

- `train(config)` — Validates config, loads YOLO model, calls `model.train()`, returns path to `best.pt`
- `export_model(weights_path, formats, img_size)` — Exports to ONNX, TorchScript, etc. Returns list of exported file paths.

### 2. `scripts/prepare_dataset.py` — COCO→UrbanEye Dataset Preparation

A standalone script that downloads COCO val2017 and converts it to our 5-class YOLO format.

#### COCO→UrbanEye Class Mapping

| COCO Category ID | COCO Name | UrbanEye Class | UrbanEye ID |
|-------------------|-----------|---------------|-------------|
| 1 | person | pedestrian | 1 |
| 2 | bicycle | cyclist | 2 |
| 3 | car | vehicle | 0 |
| 4 | motorcycle | vehicle | 0 |
| 6 | bus | vehicle | 0 |
| 8 | truck | vehicle | 0 |
| 10 | traffic light | traffic_light | 3 |
| 13 | stop sign | traffic_sign | 4 |

All 5 UrbanEye classes are covered. Multiple COCO categories map to `vehicle` (car, bus, truck, motorcycle).

#### Pipeline Steps

1. **Download COCO val2017 images** (778MB, 5000 images) via `wget`
2. **Download COCO annotations** (252MB, JSON format) via `wget`
3. **Extract** both zip files
4. **Convert** COCO JSON → YOLO .txt format:
   - Parse `instances_val2017.json`
   - Filter for our 8 COCO categories
   - Convert bbox from COCO format `[x, y, w, h]` (absolute pixels) to YOLO format `[cx, cy, w, h]` (normalized 0-1)
   - Clamp coordinates to [0.001, 0.999]
   - Skip tiny boxes (< 2px)
5. **Split** into 80% train / 20% val (2,422 train / 606 val)
6. **Generate** `dataset.yaml` for Ultralytics

#### Dataset Statistics

| Class | Count | Percentage |
|-------|-------|-----------|
| pedestrian | 10,993 | 73.2% |
| vehicle | 3,003 | 20.0% |
| traffic_light | 633 | 4.2% |
| cyclist | 316 | 2.1% |
| traffic_sign | 75 | 0.5% |
| **Total** | **15,020** | **100%** |

**Key observation:** Severe class imbalance — pedestrians dominate (73%) while traffic_sign (0.5%) and cyclist (2.1%) are rare. This matches real-world driving distributions but means the model will be biased toward pedestrian detection. Class weights (Phase 6) will help address this.

### 3. `scripts/train_baseline.py` — Training Launch Script

CLI-based training launcher optimized for the RTX 5070 Laptop:

```bash
python3 train_baseline.py                    # YOLOv11n, 50 epochs
python3 train_baseline.py --model m          # YOLOv11m (more accurate)
python3 train_baseline.py --epochs 100       # longer training
python3 train_baseline.py --resume           # resume from checkpoint
```

Features:
- Detects dataset.yaml automatically from `~/urbaneye/data/urbaneye_baseline/`
- Prints config summary before training starts
- Saves checkpoints every 10 epochs
- Auto-exports ONNX after training completes
- Resume support from last checkpoint

---

## Training Results

### Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| VRAM | 8,151 MiB (8.5GB) |
| VRAM Used | 2,380 MiB (29%) |
| PyTorch | 2.10.0+cu128 |
| CUDA | 13.2 |
| Ultralytics | 8.4.21 |
| Python | 3.10.12 |

### Model Architecture

```
YOLO11n summary: 182 layers, 2,590,815 parameters, 6.4 GFLOPs
Transferred 448/499 items from pretrained weights (COCO pre-trained)
```

The model is fine-tuned from COCO-pretrained weights (transfer learning), not trained from scratch. This gives a significant head start since the backbone already knows how to extract visual features.

### Metrics by Epoch

| Metric | Epoch 1 | Epoch 25 | Best (Epoch 48) | Final (50) |
|--------|---------|----------|-----------------|------------|
| mAP@50 | ~0.15 | ~0.43 | **0.4735** | 0.4728 |
| mAP@50-95 | ~0.06 | ~0.26 | **0.2921** | 0.2905 |
| Precision | ~0.30 | ~0.55 | 0.6140 | 0.6740 |
| Recall | ~0.20 | ~0.40 | 0.4350 | 0.4083 |
| box_loss | 1.69 | 1.33 | 1.26 | 1.25 |
| cls_loss | 3.63 | 1.15 | 0.96 | 0.96 |
| dfl_loss | 1.62 | 1.30 | 1.26 | 1.25 |

### Training Performance

| Metric | Value |
|--------|-------|
| Total training time | 41.9 minutes |
| Time per epoch | ~50 seconds |
| Batches per epoch | 152 (2,422 images / 16 batch) |
| Speed | ~3.4 batches/second |
| AMP | Enabled (FP16 mixed precision) |

### Saved Artifacts

| File | Size | Purpose |
|------|------|---------|
| `best.pt` | 5.3 MB | Best model weights (epoch 48) |
| `best.onnx` | 11 MB | ONNX export for CPU inference |
| `last.pt` | 5.3 MB | Final epoch weights |
| `epoch{0,10,20,30,40}.pt` | 16 MB each | Periodic checkpoints |
| `results.png` | — | Training curves plot |
| `confusion_matrix.png` | — | Per-class confusion matrix |
| `BoxPR_curve.png` | — | Precision-Recall curves |
| `labels.jpg` | — | Dataset class distribution |
| `val_batch{0,1,2}_pred.jpg` | — | Validation predictions |

### Gap Analysis

| Metric | Achieved | Target | Gap | Expected Fix |
|--------|----------|--------|-----|-------------|
| mAP@50 | 0.47 | 0.70 | -0.23 | CARLA data (50K frames) + YOLOv11m |
| mAP@50-95 | 0.29 | 0.45 | -0.16 | More data + larger model + more epochs |
| Recall | 0.41 | — | Low | Class weights for rare classes |

**Why the gap exists:**
1. **Dataset size** — 2,422 training images vs planned 50,000 CARLA frames (20x more data)
2. **Domain mismatch** — COCO contains indoor/nature scenes, not just driving
3. **Class imbalance** — traffic_sign has only 75 examples (0.5%)
4. **Model size** — YOLOv11n (2.6M params) is the smallest variant
5. **No driving-specific augmentation** — No rain, fog, night augmentation applied

---

## Test Results

```
tests/unit/test_train_config.py — 19 tests

TestTrainConfig (11 tests):
  - default values match PDF spec ✓
  - valid config → no errors ✓
  - negative epochs caught ✓
  - zero batch size caught ✓
  - img_size not divisible by 32 caught ✓
  - lr0 too high caught ✓
  - lr0 zero caught ✓
  - invalid model variant caught ✓
  - invalid export format caught ✓
  - negative workers caught ✓
  - mosaic out of range caught ✓

TestTrainConfigKwargs (4 tests):
  - contains all required Ultralytics keys ✓
  - 'batch' key (not 'batch_size') ✓
  - 'imgsz' key (not 'img_size') ✓
  - values match config ✓

TestTrainConfigFactories (4 tests):
  - RTX 5070 nano: batch=16 ✓
  - RTX 5070 medium: batch=8 ✓
  - Kaggle T4: batch=16, 100 epochs ✓
  - invalid size raises ValueError ✓
```

**All 19 new tests passed. 156 cumulative tests, all passing.**

---

## Files Created in This Phase

```
# Main package
urbaneye/training/train_yolov11.py      # TrainConfig dataclass + train/export functions

# Scripts (WSL2 execution)
scripts/prepare_dataset.py               # COCO→UrbanEye 5-class dataset preparation
scripts/train_baseline.py                # CLI training launcher for RTX 5070

# Tests
tests/unit/test_train_config.py          # 19 tests for TrainConfig

# Training outputs (on WSL2: ~/urbaneye/runs/yolov11n_baseline/)
weights/best.pt                          # Best model (5.3MB)
weights/best.onnx                        # ONNX export (11MB)
weights/last.pt                          # Final epoch weights
results.csv                              # Per-epoch metrics
results.png                              # Training curves
confusion_matrix.png                     # Per-class confusion
```

---

## Key Decisions & Interview Talking Points

1. **RTX 5070 Laptop has 8.5GB VRAM, not 12GB** — The laptop variant of the 5070 has less VRAM than the desktop. We discovered this during environment setup and adjusted batch sizes: YOLOv11n at batch=16 uses only 2.4GB (29% VRAM), leaving headroom for larger batches or YOLOv11m.

2. **COCO as baseline instead of waiting for CARLA** — Rather than blocking on CARLA installation (multi-day process), we used COCO val2017 filtered for driving classes. This validated the entire pipeline in 42 minutes while providing a real baseline to compare against future CARLA-trained models.

3. **Transfer learning from COCO pre-trained weights** — Starting from `yolo11n.pt` (pre-trained on 80 COCO classes) and fine-tuning for our 5 classes. The model transferred 448/499 weight layers. This is far more effective than training from scratch with only 2,422 images.

4. **Config validation prevents wasted GPU time** — `TrainConfig.validate()` catches 12 categories of errors before `model.train()` is called. A training run takes 42 minutes; catching a bad LR at startup saves hours of debugging.

5. **Factory methods encode hardware knowledge** — `TrainConfig.for_rtx5070_laptop("n")` means the user doesn't need to know that 8.5GB VRAM fits batch=16 for nano but only batch=8 for medium. The factory captures this GPU-specific constraint.

6. **ONNX auto-export** — The training script automatically exports to ONNX after training completes. This means the HuggingFace demo (Phase 10) always has a fresh ONNX model without a manual export step.

7. **mAP@50 = 0.47 baseline is expected** — With only 2,422 images from a non-driving dataset using the smallest model, 0.47 is actually reasonable. The COCO pre-trained backbone provides strong feature extraction; the fine-tuning adapts the detection head to our 5 classes. The jump to 0.70+ will come from 20x more data, driving-specific augmentation, and the larger YOLOv11m model.
