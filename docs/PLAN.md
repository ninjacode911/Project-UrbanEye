# UrbanEye — Implementation Plan & Progress Tracker

> **Autonomous Driving Perception Pipeline**
> CARLA Simulation | YOLOv11 Detection | ByteTrack + DeepSORT Tracking | Zero Hardware Cost

**Author:** Navnit Amrutharaj
**Created:** 2026-03-21
**Target Completion:** 12 Phases (~12 working days)
**Status:** In Progress

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Summary](#architecture-summary)
3. [Phase Dependency Graph](#phase-dependency-graph)
4. [Training Environment Strategy](#training-environment-strategy)
5. [Phase 1: Project Scaffolding & CI/CD](#phase-1-project-scaffolding--cicd)
6. [Phase 2: CARLA Sensor Config & Annotation Exporter](#phase-2-carla-sensor-config--annotation-exporter)
7. [Phase 3: CARLA Data Generator & Scenario Runner](#phase-3-carla-data-generator--scenario-runner)
8. [Phase 4: Augmentation Pipeline & Dataset Utilities](#phase-4-augmentation-pipeline--dataset-utilities)
9. [Phase 5: YOLOv11 Training Pipeline](#phase-5-yolov11-training-pipeline)
10. [Phase 6: Domain Adaptation Pipeline](#phase-6-domain-adaptation-pipeline)
11. [Phase 7: ByteTrack Real-Time Tracker](#phase-7-bytetrack-real-time-tracker)
12. [Phase 8: DeepSORT + Dual Tracker Interface](#phase-8-deepsort--dual-tracker-interface)
13. [Phase 9: MOT Evaluation Suite](#phase-9-mot-evaluation-suite)
14. [Phase 10: HuggingFace Spaces Demo](#phase-10-huggingface-spaces-demo)
15. [Phase 11: Integration Tests & Performance Benchmarks](#phase-11-integration-tests--performance-benchmarks)
16. [Phase 12: Final Documentation, Security Audit & Release](#phase-12-final-documentation-security-audit--release)
17. [Target Metrics](#target-metrics)
18. [Tech Stack Reference](#tech-stack-reference)

---

## Project Overview



**The 4-Stage Pipeline:**
1. **Synthetic Data Generation** — CARLA simulator generates 50,000+ auto-annotated frames across 144 scenario configurations (8 maps x 6 weather x 3 time-of-day)
2. **Model Training with Domain Adaptation** — YOLOv11 trained on Kaggle T4 GPUs with 3-layer domain adaptation strategy to bridge the sim-to-real gap
3. **Inference & Tracking** — Dual tracker system: ByteTrack (60+ FPS, motion-based) and DeepSORT (25-35 FPS, appearance-based Re-ID)
4. **Evaluation & Demo** — Standard MOT metrics (MOTA, IDF1) + interactive HuggingFace Spaces Gradio demo

**5 Detection Classes:** Vehicle, Pedestrian, Cyclist, Traffic Light, Traffic Sign

**Core Thesis:** Real AV companies spend $100M+ on perception systems. UrbanEye proves the entire methodology — synthetic data generation, domain adaptation, multi-object tracking, MOT evaluation — can be executed with CARLA, YOLOv11, and free GPUs. The research techniques are identical; only the hardware budget is different.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        URBANEYE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 1: DATA GENERATION (CARLA 0.9.15 on WSL2)                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Ego      │───>│ Multi-   │───>│ YOLO         │───>│ 50K+         │  │
│  │ Vehicle  │    │ Sensor   │    │ Annotation   │    │ Annotated    │  │
│  │ Autopilot│    │ Capture  │    │ Export       │    │ Frames       │  │
│  └──────────┘    └──────────┘    └──────────────┘    └──────────────┘  │
│       │                                                                 │
│       ├── 8 CARLA maps (Town01-Town10HD)                               │
│       ├── 6 weather presets (Clear/Cloudy/Wet/Rain/Sunset)             │
│       ├── 3 time-of-day (noon/sunset/night)                            │
│       └── ScenarioRunner: jaywalking, emergency vehicle, dense traffic │
│                                                                         │
│  STAGE 2: TRAINING (Kaggle T4 / RTX 5070 Local)                       │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ YOLOv11  │───>│ Albumentations│───>│ Domain       │                 │
│  │ n + m    │    │ Augmentation  │    │ Adaptation   │                 │
│  │ Training │    │ Pipeline      │    │ (BDD100K)    │                 │
│  └──────────┘    └──────────────┘    └──────────────┘                  │
│       │                                                                 │
│       ├── 100 epochs, batch 16, cosine LR, AMP                        │
│       ├── 3-layer domain adaptation (weather/augment/mixed-data)       │
│       └── Export: ONNX + TorchScript                                   │
│                                                                         │
│  STAGE 3: INFERENCE + TRACKING                                         │
│  ┌──────────────┐         ┌──────────────┐                             │
│  │ ByteTrack    │         │ DeepSORT     │                             │
│  │ (60+ FPS)    │         │ (25-35 FPS)  │                             │
│  │ Motion-based │         │ Re-ID based  │                             │
│  │ 2-stage match│         │ MobileNetV2  │                             │
│  └──────┬───────┘         └──────┬───────┘                             │
│         │                        │                                      │
│         └────────┬───────────────┘                                      │
│                  │                                                       │
│           ┌──────▼───────┐                                              │
│           │ DualTracker  │  ← Unified interface                         │
│           │ (switchable) │                                              │
│           └──────────────┘                                              │
│                                                                         │
│  STAGE 4: EVALUATION + DEMO                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ MOT Metrics  │    │ Report       │    │ HuggingFace  │              │
│  │ MOTA/IDF1    │    │ Generator    │    │ Spaces Demo  │              │
│  │ Per-class    │    │ MD + HTML    │    │ Gradio 4.x   │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Dependency Graph

```
Phase 1 (Scaffold + CI)
    │
    v
Phase 2 (Sensor Config + Annotation Exporter)
    │
    v
Phase 3 (Data Generator + Scenario Runner)
    │
    v
Phase 4 (Augmentations + Dataset Utils)
    │
    ├──> Phase 5 (YOLOv11 Training) ──> Phase 6 (Domain Adaptation)
    │                                         │
    v                                         v
Phase 7 (ByteTrack) ───┐                     │
    │                   │                     │
    v                   v                     │
Phase 8 (DeepSORT) ──> Phase 9 (Evaluation) <─┘
    │                   │
    v                   v
Phase 10 (HF Demo) <───┘
    │
    v
Phase 11 (Integration Tests)
    │
    v
Phase 12 (Docs + Security + Release)
```

---

## Training Environment Strategy

| Task | Recommended Environment | Reason |
|------|------------------------|--------|
| CARLA data generation | **WSL2 + RTX 5070** | Requires GPU for rendering, local gives fastest iteration |
| YOLOv11 training (100 epochs) | **Kaggle T4** or **Local RTX 5070** | Either works; Kaggle avoids tying up local GPU |
| Domain adaptation (20 epochs) | **Kaggle T4** or **Local RTX 5070** | Shorter run, either is fine |
| DeepSORT Re-ID model | **No training needed** | Pre-trained on Market-1501 |
| Quick experiments / debugging | **Local RTX 5070** | Fastest feedback loop |

**Decision point:** Before each training run, we'll discuss whether to use local GPU or Kaggle/Colab based on the specific situation.

---

## Phase 1: Project Scaffolding & CI/CD

**Status:** `[ ] Not Started`
**Objective:** Establish complete project skeleton with packaging, linting, testing, and CI pipeline.

### Why This Phase Comes First
Without proper scaffolding, every subsequent phase accumulates tech debt. CI from day one means every change is automatically verified — ruff linting catches style issues, pytest catches regressions.

### What Gets Built

| File | Purpose | Interview Relevance |
|------|---------|-------------------|
| `pyproject.toml` | Modern Python packaging with optional dependency groups | Shows knowledge of Python packaging best practices (PEP 621) |
| `.github/workflows/ci.yml` | GitHub Actions CI: lint + test on Python 3.11/3.12 | Demonstrates CI/CD competency |
| `urbaneye/utils/constants.py` | 5 class names, BGR colors, CARLA class mapping | Centralized configuration prevents magic numbers |
| `urbaneye/utils/io_helpers.py` | YAML loading, path validation, directory helpers | Reusable utilities across all modules |
| `tests/conftest.py` | Shared pytest fixtures | Proper test architecture |
| `.gitignore` | Excludes model weights, data, secrets | Security-conscious development |

### Technical Decisions

**Why pyproject.toml over setup.py:** PEP 621 standard, declarative configuration, single source of truth for project metadata, tool configuration (ruff, pytest), and dependencies.

**Why optional dependency groups:**
- `[dev]` — pytest, ruff, pytest-mock (development only)
- `[training]` — ultralytics, albumentations (only needed for training)
- `[tracking]` — deep-sort-realtime, scipy, lap (only for tracking module)
- `[demo]` — gradio (only for demo deployment)

This prevents installing heavy ML packages when only running the demo or tests.

**Why ruff over black+isort+flake8:** Single tool replaces three. Faster, more configurable, actively maintained. Line length 100 (standard for ML projects where variable names tend to be longer).

### Test Strategy
- `test_io_helpers.py` validates all utility functions with edge cases
- Verify `pip install -e ".[dev]"` works cleanly
- Verify `ruff check .` produces zero warnings
- Verify `pytest tests/unit/ -v` passes

### Exit Criteria
- [ ] Package installs cleanly
- [ ] Linting passes with zero errors
- [ ] All unit tests pass
- [ ] CI workflow is valid YAML

---

## Phase 2: CARLA Sensor Config & Annotation Exporter

**Status:** `[ ] Not Started`
**Objective:** Build sensor configuration dataclasses and the 3D→YOLO annotation exporter. Pure math — fully testable without CARLA installed.

### Why This Design (Dependency Injection)
CARLA only runs on Linux (WSL2) with GPU access. By designing all CARLA interaction behind well-defined Python dataclasses and accepting CARLA objects as parameters (rather than creating them internally), every function can be tested with mock objects on any platform. This is a critical architectural decision.

### What Gets Built

| File | Purpose | Interview Relevance |
|------|---------|-------------------|
| `sensor_config.py` | `CameraSensorConfig`, `SensorSuite`, `SimulationConfig` dataclasses | Clean separation of configuration from runtime |
| `annotation_exporter.py` | 3D bounding box → 2D YOLO format projection + validation | Core computer vision math: perspective projection |
| `carla_config.yaml` | 8 maps, 6 weather presets, 3 time-of-day, 5 classes | 144 unique scenario configurations |

### The Coordinate Transformation (Key Algorithm)

The annotation exporter performs a critical transformation:
1. **World coordinates** (3D) → CARLA actor bounding box vertices in world space
2. **Camera projection** → Project 3D vertices onto 2D image plane using camera intrinsics
3. **YOLO format** → Convert pixel coordinates to normalized (0-1) center-x, center-y, width, height
4. **Validation** → Filter out-of-frame objects, behind-camera objects, zero-area boxes

This is the same transformation every LiDAR/camera fusion system uses in real AV stacks.

### Test Strategy
- Known 3D box + known camera → verify exact YOLO output
- Behind-camera objects → returns None
- Edge-of-frame objects → filtered correctly
- Zero-area degenerate boxes → filtered
- **No CARLA imports in any test file**

### Exit Criteria
- [ ] All tests pass without CARLA installed
- [ ] Annotation exporter handles all edge cases
- [ ] ruff clean

---

## Phase 3: CARLA Data Generator & Scenario Runner

**Status:** `[ ] Not Started`
**Objective:** Build the ego-vehicle driving loop, multi-sensor capture, and scenario runner scripts.

### What Gets Built

| File | Purpose | Interview Relevance |
|------|---------|-------------------|
| `data_generator.py` | `CarlaDataGenerator` class — full driving + capture loop | Shows understanding of simulation-based data pipelines |
| `base_scenario.py` | Abstract base class for all scenarios | Design pattern: Strategy pattern for scenario composition |
| `pedestrian_crossing.py` | Jaywalking edge case scenario | Safety-critical test scenario |
| `adverse_weather.py` | Rain/fog/night cycling | Domain robustness testing |
| `emergency_vehicle.py` | Rare vehicle type detection | Long-tail distribution handling |

### Key Design: Scenario Composability
Scenarios implement a common interface (`BaseScenario`) and can be composed — multiple scenarios run simultaneously during a single data generation session. This mirrors how real AV testing frameworks like Waymo's ScenarioRunner work.

### Data Flow
```
CarlaDataGenerator
    │
    ├── setup_world(map, weather) → Load map, configure weather/time
    ├── spawn_ego_vehicle() → Place ego car at random spawn point
    ├── attach_sensors() → RGB + Depth + Semantic cameras
    │
    └── generate_dataset(num_frames) ──loop──┐
        │                                     │
        ├── capture_frame() → FrameData       │
        │   ├── rgb_image (1920x1080)         │
        │   ├── depth_map (640x480)           │
        │   └── semantic_map (640x480)        │
        │                                     │
        ├── scenario.tick() → advance scenarios│
        │                                     │
        ├── annotation_exporter.export() → .txt│
        │                                     │
        └── save image + label ───────────────┘
```

### Output Structure (Standard YOLO)
```
output_dir/
├── images/
│   ├── train/   (~40,000 frames)
│   ├── val/     (~5,000 frames)
│   └── test/    (~5,000 frames)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata.csv  (town, weather, time, npc_count per frame)
```

### Test Strategy (All Mock-Based)
- Mock `carla.Client`, `carla.World`, `carla.Actor`
- Verify correct CARLA API call sequence
- Verify output directory structure matches YOLO spec
- Verify scenario lifecycle (setup → tick → cleanup)

### Exit Criteria
- [ ] All mock-based tests pass without CARLA
- [ ] Output structure matches YOLO format
- [ ] Scenario runner is extensible (new scenario = 1 new file)

---

## Phase 4: Augmentation Pipeline & Dataset Utilities

**Status:** `[ ] Not Started`
**Objective:** Build domain adaptation layer 2 (Albumentations) and dataset visualization tools.

### Why Augmentations Are Critical
CARLA images look "too clean" — perfect lighting, no sensor noise, no motion blur. Real dashcams have JPEG compression artifacts, lens distortion, varying exposure, rain drops on the lens. The augmentation pipeline applies these real-world degradations during training so the model generalizes to real data.

### The 3-Level Augmentation Strategy

| Level | When Used | Augmentations |
|-------|-----------|---------------|
| **Light** | Validation sanity checks | HorizontalFlip only |
| **Medium** | Standard training | Flip, brightness/contrast, Gaussian noise, blur, JPEG artifacts, perspective |
| **Heavy** | Domain adaptation | All medium + RandomRain, RandomFog, RandomSunFlare, aggressive ColorJitter |

### YOLO-Compatible Bbox Handling
Albumentations natively supports YOLO-format bounding box transformations. When an image is flipped or cropped, the bounding boxes are automatically transformed. `min_visibility=0.3` ensures boxes that become mostly cropped out are removed.

### Test Strategy
- Augmented image shape matches input (no accidental resizing)
- All output bounding boxes in [0, 1] range (no invalid coordinates)
- Class labels preserved through augmentation
- Reproducible with fixed random seed
- Zero-area boxes removed, out-of-range coordinates clipped

### Exit Criteria
- [ ] Augmented bboxes never exceed [0,1]
- [ ] All 3 levels work correctly
- [ ] Visualization tools produce valid outputs
- [ ] All tests pass

---

## Phase 5: YOLOv11 Training Pipeline

**Status:** `[ ] Not Started`
**Objective:** Build training configuration, launcher script, and Kaggle notebooks.

### Training Configuration (From PDF Spec)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | YOLOv11m (medium) | Best accuracy/speed tradeoff for T4 |
| Epochs | 100 | Standard for convergence on 50K images |
| Batch size | 16 | Fits T4 16GB VRAM |
| Image size | 640 | Standard YOLO training resolution |
| Learning rate | 0.001 → 0.00001 | Cosine schedule with warmup |
| Mosaic | 1.0 | Increases small object density |
| Mixup | 0.1 | Regularization |
| AMP | True | 2x faster training on T4 |
| Class weights | ped=2.0, cyclist=3.0 | Compensate safety-critical class imbalance |

### Kaggle Notebook Structure
1. Install ultralytics + albumentations
2. Mount CARLA dataset from Kaggle Datasets
3. Import and configure `TrainConfig`
4. Run training with progress bars
5. Plot training curves (loss, mAP per epoch)
6. Export to ONNX + TorchScript
7. Quick inference on sample images
8. Download weights

**NOTE:** When this phase reaches actual training, Navnit will be notified to choose local RTX 5070 or Kaggle T4.

### Exit Criteria
- [ ] Config validation catches all invalid states
- [ ] Kwargs conversion produces correct Ultralytics parameters
- [ ] Notebooks are well-structured and documented

---

## Phase 6: Domain Adaptation Pipeline

**Status:** `[ ] Not Started`
**Objective:** Bridge the sim-to-real gap with mixed CARLA + BDD100K training.

### The 3-Layer Strategy
1. **Layer 1 (Phase 3):** Randomized CARLA weather/lighting during data generation
2. **Layer 2 (Phase 4):** Albumentations augmentations mimicking real camera imperfections
3. **Layer 3 (This Phase):** Mixed training with 80% CARLA + 20% BDD100K real images

### BDD100K Class Mapping

| BDD100K Class | UrbanEye Class |
|---------------|---------------|
| car, truck, bus, train | vehicle |
| pedestrian | pedestrian |
| rider, bicycle, motorcycle | cyclist |
| traffic light | traffic_light |
| traffic sign | traffic_sign |

### Fine-Tuning Strategy
- **Base weights:** best.pt from Phase 5 (CARLA-only training)
- **Freeze layers:** First 10 backbone layers (preserve learned features)
- **Learning rate:** 0.0001 (10x lower — prevents catastrophic forgetting)
- **Epochs:** 20 (short fine-tune)
- **Expected improvement:** BDD100K mAP@50 from ~0.35 → ~0.55

### Exit Criteria
- [ ] BDD100K→YOLO conversion works correctly
- [ ] Mixed dataset maintains 80/20 ratio
- [ ] All tests pass

---

## Phase 7: ByteTrack Real-Time Tracker

**Status:** `[ ] Not Started`
**Objective:** Implement ByteTrack from scratch — the most algorithmically complex module.

### Why From Scratch (Not a Library)
1. **Interview depth:** "I implemented the Kalman Filter and Hungarian Algorithm from scratch" carries more weight than "I called a library"
2. **No unmaintained dependency:** ByteTrack Python packages are often outdated or have complex dependencies
3. **Custom per-class tracking:** We can separate tracking pools per object class (vehicles vs pedestrians)
4. **Only ~300 lines of core logic:** The algorithm is elegant and compact

### ByteTrack's Key Innovation (ECCV 2022)

Traditional trackers only associate **high-confidence** detections with tracks. ByteTrack's insight: **low-confidence detections** (partially occluded objects) are valuable too.

**Two-Stage Matching:**
```
Frame N detections
    │
    ├── High confidence (≥ 0.6) ──> Stage 1: Match with active tracks (IoU + Hungarian)
    │                                   │
    │                                   ├── Matched → Update track
    │                                   └── Unmatched tracks remain
    │
    └── Low confidence (≥ 0.1) ──> Stage 2: Match with remaining unmatched tracks
                                        │
                                        ├── Matched → Update track (rescued from occlusion!)
                                        └── Still unmatched → increment lost counter
```

### Kalman Filter State Model

| State Variable | Meaning |
|---------------|---------|
| x | Bounding box center X |
| y | Bounding box center Y |
| a | Aspect ratio (width/height) |
| h | Height |
| vx | Velocity in X |
| vy | Velocity in Y |
| va | Rate of change of aspect ratio |
| vh | Rate of change of height |

**Measurement:** [x, y, a, h] (observed from detection)
**Prediction:** Uses constant-velocity motion model to predict next frame position

### Track Lifecycle
```
Detection appears → NEW (unconfirmed)
    │
    ├── Matched for min_hits (3) consecutive frames → ACTIVE (confirmed, rendered)
    │       │
    │       ├── Continues matching → stays ACTIVE
    │       └── No match found → LOST (Kalman predicts position)
    │               │
    │               ├── Re-matched within max_lost (30) frames → back to ACTIVE
    │               └── max_lost exceeded → DELETED (removed permanently)
    │
    └── Not matched for min_hits frames → DELETED
```

### Exit Criteria
- [ ] All Kalman Filter tests pass
- [ ] Track lifecycle state machine works correctly
- [ ] Two-stage matching correctly rescues occluded detections
- [ ] IoU computation matches manual calculations
- [ ] No external ByteTrack library dependency

---

## Phase 8: DeepSORT + Dual Tracker Interface

**Status:** `[ ] Not Started`
**Objective:** Integrate DeepSORT and build the unified `DualTracker` interface.

### ByteTrack vs DeepSORT Comparison

| Feature | ByteTrack | DeepSORT |
|---------|-----------|----------|
| Matching | IoU + motion (Kalman) | IoU + motion + **appearance** (Re-ID) |
| Re-ID | None | MobileNetV2, 128-dim embeddings |
| Speed | 60+ FPS (CPU) | 25-35 FPS (CPU) |
| Occlusion handling | Low-confidence detection rescue | Appearance re-identification |
| Best for | Real-time demo, speed-critical | Identity persistence, safety evaluation |
| MOTA (typical) | ~77% | ~70% |
| ID Switches | Higher | **Lower** (main advantage) |

### DualTracker Design
```python
# User-facing API is identical regardless of backend
tracker = DualTracker(TrackerType.BYTETRACK)  # or DEEPSORT
tracks = tracker.update(detections, frame)
# Returns: list[TrackedObject] — same format for both
```

### Exit Criteria
- [ ] Both trackers produce TrackedObject through same interface
- [ ] DeepSORT lazy-initializes (no model download at import time)
- [ ] Config files parameterize both trackers

---

## Phase 9: MOT Evaluation Suite

**Status:** `[ ] Not Started`
**Objective:** Build standard MOT metrics computation and automated report generation.

### Key Metrics Explained

| Metric | Formula | What It Measures | Target |
|--------|---------|-----------------|--------|
| **MOTA** | 1 - (FN + FP + IDsw) / GT | Overall tracking accuracy | ≥ 0.72 |
| **MOTP** | Σ(IoU of matches) / matches | Bbox localization precision | ≥ 0.80 |
| **IDF1** | 2×IDTP / (2×IDTP + IDFP + IDFN) | Identity preservation | ≥ 0.65 |
| **ID Switches** | Count of ID changes | Identity stability | < 200 |

### Report Output
The report generator produces a Markdown document with:
- Overall metrics table
- ByteTrack vs DeepSORT comparison
- Per-class breakdown (vehicle, pedestrian, cyclist, traffic light, traffic sign)
- Pass/fail against target thresholds

### Exit Criteria
- [ ] Metrics match expected values on synthetic test data
- [ ] Report generator produces valid Markdown
- [ ] Per-class breakdown works for all 5 classes

---

## Phase 10: HuggingFace Spaces Demo

**Status:** `[ ] Not Started`
**Objective:** Build the public-facing Gradio demo.

### Demo UI Layout
```
┌─────────────────────────────────────────────────────┐
│  UrbanEye — Autonomous Driving Perception Demo      │
├──────────────────────┬──────────────────────────────┤
│  📹 Upload Video     │  🎬 Annotated Output         │
│  [drag & drop]       │  [bboxes + track IDs + FPS]  │
├──────────────────────┴──────────────────────────────┤
│  Tracker: ○ ByteTrack  ○ DeepSORT                   │
│  Confidence: ═══════●═══ 0.45                       │
│  Classes: ☑ Vehicle ☑ Pedestrian ☑ Cyclist          │
│           ☑ Traffic Light ☑ Traffic Sign             │
├─────────────────────────────────────────────────────┤
│  📊 Metrics:                                        │
│  Detections: 127 | Active Tracks: 14 | FPS: 28.3   │
├─────────────────────────────────────────────────────┤
│  🎞 Sample Clips: [Sunny] [Rain] [Night] [Dense]   │
└─────────────────────────────────────────────────────┘
```

### Self-Contained Deployment
`hf_space/app.py` is completely self-contained — no imports from the `urbaneye` package. This ensures HuggingFace Spaces can run it with just `pip install -r requirements.txt` without installing the full project.

### Exit Criteria
- [ ] Gradio app launches locally
- [ ] Video processing works end-to-end
- [ ] HF Spaces README has correct metadata

---

## Phase 11: Integration Tests & Performance Benchmarks

**Status:** `[ ] Not Started`
**Objective:** End-to-end pipeline testing and FPS benchmarks.

### What Gets Tested
- Module boundary compatibility (detection output → tracker input → evaluator input)
- Full pipeline: synthetic video → detect → track → evaluate → report
- Both trackers through DualTracker interface
- Edge cases: empty frames, single detection, 1000 detections per frame
- Performance: ByteTrack processes 100 detections x 1000 frames under 1 second

### Exit Criteria
- [ ] All integration tests pass
- [ ] Full pipeline completes in < 30 seconds
- [ ] No circular dependencies

---

## Phase 12: Final Documentation, Security Audit & Release

**Status:** `[ ] Not Started`
**Objective:** Complete all docs, security audit, and prepare for GitHub release.

### Security Audit Checklist
- [ ] No `.env` files in repository
- [ ] No API keys or tokens in any file
- [ ] `.gitignore` covers all sensitive patterns
- [ ] No hardcoded local machine paths
- [ ] Model weights in `.gitignore` (downloaded separately)
- [ ] CI workflow doesn't expose secrets
- [ ] CARLA imports guarded with try/except
- [ ] Only collaborator: ninjacode911

### Documentation Deliverables
- `README.md` — Full project README with badges, architecture, results, demo link
- `docs/ARCHITECTURE.md` — System architecture with diagrams
- `docs/PHASES.md` — Phase completion log
- `docs/TRAINING_GUIDE.md` — Step-by-step Kaggle/local training guide
- `docs/DEPLOYMENT_GUIDE.md` — HuggingFace Spaces deployment guide

### Exit Criteria
- [ ] CI pipeline green on GitHub
- [ ] All documentation complete
- [ ] Security audit passes
- [ ] HF demo is live and functional

---

## Target Metrics

### Detection Targets

| Metric | CARLA Test Set | BDD100K (After DA) |
|--------|---------------|-------------------|
| mAP@50 (all classes) | ≥ 0.70 | ≥ 0.50 |
| mAP@50:95 | ≥ 0.45 | — |
| mAP@50 — Vehicles | ≥ 0.80 | — |
| mAP@50 — Pedestrians | ≥ 0.65 | — |
| mAP@50 — Cyclists | ≥ 0.60 | — |
| YOLOv11n ONNX FPS (CPU) | ≥ 25 | — |

### Tracking Targets

| Metric | ByteTrack Target | DeepSORT Target |
|--------|-----------------|----------------|
| MOTA | ≥ 0.72 | ≥ 0.68 |
| MOTP | ≥ 0.80 | ≥ 0.80 |
| IDF1 | ≥ 0.65 | ≥ 0.70 |
| ID Switches (per seq) | < 200 | < 80 |
| Processing FPS (CPU) | ≥ 60 | ≥ 25 |

---

## Tech Stack Reference

| Tool | Version/License | Role |
|------|----------------|------|
| CARLA Simulator | 0.9.15, MIT | Photorealistic AV simulator |
| YOLOv11 (Ultralytics) | AGPL-3.0 | Object detection (5 classes) |
| Kaggle T4 GPU | Free (30hr/week) | Cloud training |
| RTX 5070 (Local) | — | Local training + CARLA |
| ByteTrack | Apache 2.0 (our implementation) | Real-time MOT tracker |
| DeepSORT | deep-sort-realtime, MIT | Identity-persistent tracker |
| TrackEval | Google, open source | MOT metrics evaluation |
| Albumentations | MIT | Domain adaptation augmentations |
| BDD100K | UC Berkeley, free | Real-world driving images |
| Gradio 4.x | Apache 2.0 | Demo UI |
| HuggingFace Spaces | Free CPU tier | Demo hosting |
| OpenCV | Apache 2.0 | Video I/O, frame annotation |
| Supervision (Roboflow) | MIT | Tracking visualization |
