# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Install with specific extras
pip install -e ".[training]"    # ultralytics, albumentations
pip install -e ".[tracking]"    # deep-sort-realtime, scipy, lap, filterpy
pip install -e ".[demo]"        # gradio, onnxruntime
pip install -e ".[all]"         # everything

# Run all tests
pytest tests/ -v --timeout=60

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run a single test file
pytest tests/unit/test_bytetrack.py -v

# Run a single test class or method
pytest tests/unit/test_bytetrack.py::TestByteTrackPipeline::test_two_stage_matching -v

# Lint
ruff check .

# Format check
ruff format --check .

# Auto-fix lint + format
ruff check --fix . && ruff format .
```

## Architecture

UrbanEye is a 4-stage autonomous driving perception pipeline: **Data Generation → Training → Tracking → Evaluation/Demo**. All CARLA interaction is dependency-injected so the entire codebase is testable without CARLA installed.

### Core Design Pattern: No CARLA Required for Testing

Every module that touches CARLA accepts the CARLA client/world as a constructor parameter (`Any` type). Tests inject `MagicMock` objects. This is the fundamental architectural decision — pure math functions (projection, IoU, Kalman filter) have zero external dependencies, and orchestration classes (generators, trackers) accept injected dependencies.

### Module Dependency Graph

```
utils/constants.py          ← Shared by everything (5 classes, colors, thresholds)
utils/io_helpers.py          ← YAML loading, path utils
     │
     ├── carla/sensor_config.py      ← Dataclasses (CameraSensorConfig, SensorSuite, SimulationConfig)
     ├── carla/annotation_exporter.py ← 3D→2D projection, YOLO format conversion
     ├── carla/data_generator.py     ← CarlaDataGenerator orchestration (depends on sensor_config + annotation_exporter)
     ├── carla/scenario_runner/      ← BaseScenario ABC + 3 concrete scenarios
     │
     ├── training/augmentations.py   ← Albumentations 3-level pipeline (light/medium/heavy)
     ├── training/train_yolov11.py   ← TrainConfig dataclass + Ultralytics launcher
     ├── training/domain_adapt.py    ← BDD100KAdapter + DomainAdaptConfig + mixed dataset creation
     │
     ├── tracking/kalman_filter.py   ← From-scratch Kalman Filter (8D state, constant-velocity)
     ├── tracking/utils.py           ← IoU, Hungarian Algorithm, bbox conversions
     ├── tracking/bytetrack_pipeline.py ← ByteTrack from scratch (STrack, two-stage matching)
     ├── tracking/deepsort_pipeline.py  ← DeepSORT wrapper (lazy-init, deep-sort-realtime)
     ├── tracking/dual_tracker.py    ← DualTracker unified interface → TrackedObject output
     │
     ├── evaluation/mot_evaluator.py      ← MOTA/MOTP/IDF1/ID Switches
     ├── evaluation/detection_evaluator.py ← mAP@50, per-class AP
     ├── evaluation/generate_report.py    ← Markdown report generator
     │
     └── demo/app.py               ← Gradio video processing + frame annotation
```

### Key Types That Cross Module Boundaries

- **Detection format**: `np.ndarray` shape `(N, 6)` — `[x1, y1, x2, y2, confidence, class_id]`. This is the universal detection format consumed by both trackers.
- **`TrackedObject`** (in `deepsort_pipeline.py`): Unified output from both ByteTrack and DeepSORT via `DualTracker`. Contains `track_id`, `bbox`, `confidence`, `class_id`, `class_name`.
- **`MOTMetrics`** / **`DetectionMetrics`**: Dataclasses consumed by `ReportGenerator`.

### ByteTrack is From Scratch

`tracking/bytetrack_pipeline.py` and `tracking/kalman_filter.py` are a custom implementation — not a library wrapper. The Kalman Filter uses `scipy.linalg` for Cholesky decomposition. The Hungarian Algorithm uses `scipy.optimize.linear_sum_assignment`. Track lifecycle: NEW → ACTIVE (after `min_hits`) → LOST → DELETED (after `max_lost`).

### DeepSORT is Lazy-Initialized

`DeepSORTPipeline._init_tracker()` only runs on first `update()` call. This prevents model downloads at import time. Tests mock `_init_tracker` and `_tracker` to avoid the dependency.

### 5 Detection Classes (0-indexed)

`0: vehicle, 1: pedestrian, 2: cyclist, 3: traffic_light, 4: traffic_sign` — defined in `utils/constants.py`. CARLA semantic tags, BDD100K categories, and COCO categories all map to these 5 IDs.

## Configuration

- `configs/project_config.yaml` — Project-wide settings (detection thresholds, training hyperparameters, tracking params, evaluation targets)
- `urbaneye/carla/carla_config.yaml` — CARLA sensor rig and scenario configuration
- `urbaneye/tracking/tracker_config/bytetrack.yaml` — ByteTrack hyperparameters
- `urbaneye/tracking/tracker_config/deepsort.yaml` — DeepSORT hyperparameters
- `urbaneye/training/dataset.yaml` — YOLO dataset config (5 classes, paths)

## CI

GitHub Actions on push/PR to `main`: ruff lint → ruff format check → pytest unit tests. Matrix: Python 3.11 + 3.12 on ubuntu-latest.

## Important Constraints

- CARLA only runs on Linux (WSL2 with GPU). All CARLA-dependent code must remain testable via mocks on any platform.
- Training requires either local RTX 5070 or Kaggle T4. `ultralytics` is an optional dependency — never import it at module top level in non-training code.
- `deep-sort-realtime` is an optional dependency — only imported inside `DeepSORTPipeline._init_tracker()`.
- The `annotations_trainval2017.zip` (252MB) in the repo root is gitignored and must never be committed.
