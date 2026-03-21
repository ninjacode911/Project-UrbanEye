# UrbanEye — System Architecture

## Overview

UrbanEye is structured as a 4-stage pipeline where each stage is independently executable, documented, and reproducible.

## Pipeline Stages

### Stage 1: Synthetic Data Generation (CARLA)

CARLA 0.9.15 simulator generates photorealistic driving scenes with automatic ground truth annotations.

**Components:**
- **Ego Vehicle Autopilot** — Drives through 8 CARLA town maps autonomously
- **Multi-Sensor Rig** — Synchronized RGB (1920x1080), Depth (640x480), and Semantic Segmentation (640x480) cameras
- **Annotation Exporter** — Projects 3D bounding boxes to YOLO format with zero manual labeling
- **Scenario Runner** — Scripts edge cases: jaywalking, emergency vehicles, adverse weather, night driving

**Configuration Matrix:** 8 maps x 6 weather presets x 3 time-of-day = 144 unique scenarios

**Output:** 50,000+ YOLO-annotated image-label pairs uploaded to Kaggle as a public dataset.

### Stage 2: Model Training with Domain Adaptation

YOLOv11 fine-tuned on CARLA synthetic data with a 3-layer domain adaptation strategy.

**3-Layer Domain Adaptation:**
1. **Randomized CARLA conditions** — Weather, lighting, and time-of-day variation during data generation
2. **Albumentations augmentation** — Gaussian noise, motion blur, JPEG artifacts, rain/fog effects
3. **Mixed training data** — 80% CARLA + 20% BDD100K real-world images with backbone freezing

**Models Trained:**
- YOLOv11n (nano) — fastest, for real-time demo (25+ FPS on CPU)
- YOLOv11m (medium) — most accurate, for evaluation

**Export:** ONNX (CPU deployment) + TorchScript (GPU inference)

### Stage 3: Inference & Tracking

Dual tracker system processes YOLOv11 detections frame-by-frame.

**ByteTrack (Real-Time):**
- Two-stage matching: associates both high and low-confidence detections
- Kalman Filter for motion prediction
- Hungarian Algorithm for optimal assignment
- 60+ FPS on CPU, 120+ FPS on GPU

**DeepSORT (Identity-Persistent):**
- MobileNetV2 Re-ID feature extractor (128-dim embeddings)
- Cosine distance matching for appearance-based re-identification
- Cascade matching prioritizes recent tracks
- Maintains identity through long occlusions

**DualTracker Interface:**
Unified API that switches between ByteTrack and DeepSORT, producing consistent `TrackedObject` output regardless of backend.

### Stage 4: Evaluation & Demo

**MOT Evaluation:**
- MOTA (Multiple Object Tracking Accuracy)
- MOTP (Multiple Object Tracking Precision)
- IDF1 (Identity F1 Score)
- ID Switches, Mostly Tracked/Lost percentages
- Per-class breakdown for all 5 detection classes
- ByteTrack vs DeepSORT comparison report

**HuggingFace Spaces Demo:**
- Gradio interface: upload video or select CARLA test clips
- Tracker selection, confidence slider, class filter
- Annotated output video with track IDs and metrics panel

## Module Dependencies

```
urbaneye/utils/ ──────────────────────────────────────────────┐
    │                                                          │
    v                                                          │
urbaneye/carla/ ──> urbaneye/training/ ──> urbaneye/tracking/  │
                         │                      │              │
                         v                      v              │
                    urbaneye/evaluation/ <───────┘              │
                         │                                     │
                         v                                     │
                    urbaneye/demo/ <────────────────────────────┘
```

## Data Flow

```
CARLA World State
    │
    ├── RGB Camera ──> images/*.jpg
    ├── Depth Camera ──> depth_maps/ (validation only)
    ├── Semantic Seg ──> seg_maps/ (annotation quality check)
    └── Bounding Box API ──> labels/*.txt (YOLO format)
                                │
                                v
                        YOLOv11 Training
                                │
                                v
                        Model Weights (.pt)
                                │
                                v
                   ┌────────────┴────────────┐
                   │                         │
              ONNX Export              TorchScript
              (CPU demo)              (GPU inference)
                   │                         │
                   v                         v
              ByteTrack              DeepSORT
                   │                         │
                   └────────┬────────────────┘
                            │
                            v
                     TrackedObject[]
                            │
                   ┌────────┴────────┐
                   │                 │
              MOT Evaluation    Annotated Video
              (MOTA, IDF1)     (bboxes + IDs)
```
