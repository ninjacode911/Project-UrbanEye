# Phase 1: Project Scaffolding, Tooling & CI/CD

**Status:** Completed
**Date:** 2026-03-21
**Tests:** 18/18 passed

---

## Objective

Establish the complete project skeleton — directory structure, Python packaging, linting (ruff), testing (pytest), GitHub Actions CI, shared constants, utility functions, and test fixtures — so that every subsequent phase builds on a verified, reproducible foundation.

---

## Why This Phase Exists

Without proper scaffolding from day one, every subsequent phase accumulates technical debt: inconsistent imports, missing dependencies, untested code paths, manual formatting. By investing in infrastructure first:

- **CI from day one** means every push is automatically linted and tested
- **pyproject.toml** gives us a single source of truth for dependencies, tool configs, and metadata
- **Shared constants** prevent magic numbers scattered across modules
- **Utility functions** (YAML loading, path helpers) are tested once and reused everywhere

This mirrors how production engineering teams work — infrastructure before features.

---

## What Was Built

### 1. `pyproject.toml` — Modern Python Packaging (PEP 621)

The entire project is packaged as an installable Python package using the modern `pyproject.toml` standard (replacing the legacy `setup.py`).

**Key design decisions:**

- **`requires-python = ">=3.11"`** — Python 3.11+ for native `tuple[int, int]` type hints, `match` statements, and performance improvements.
- **Optional dependency groups** to avoid installing heavy ML packages when not needed:
  - `[dev]` — pytest, ruff, pytest-mock (development and testing)
  - `[training]` — ultralytics, albumentations (only for model training)
  - `[tracking]` — deep-sort-realtime, scipy, lap (only for tracking module)
  - `[demo]` — gradio, onnxruntime (only for HuggingFace demo)
  - `[all]` — everything combined

This means `pip install -e ".[dev]"` gives you a working development environment without downloading PyTorch, CUDA, or Gradio.

- **Ruff configuration** — Line length 100 (standard for ML projects where variable names are longer), target Python 3.11, select rules: E (pycodestyle), F (pyflakes), I (isort), W (warnings), UP (pyupgrade).
- **Pytest configuration** — Test timeout of 60 seconds (prevents hung tests), custom markers for `slow` and `integration` tests.

### 2. `.github/workflows/ci.yml` — GitHub Actions CI Pipeline

Runs on every push to `main` and every pull request:

```yaml
strategy:
  matrix:
    python-version: ["3.11", "3.12"]
```

**Steps:**
1. Checkout code
2. Set up Python (matrix: 3.11 + 3.12)
3. Install package with dev dependencies
4. Run `ruff check .` + `ruff format --check .` (lint + format verification)
5. Run `pytest tests/unit/ -v -x --timeout=60` (unit tests, fail-fast)

**Why matrix testing:** Ensures compatibility across Python versions. If a feature works on 3.12 but breaks 3.11, CI catches it before merge.

### 3. `urbaneye/utils/constants.py` — Centralized Configuration

All magic numbers, class definitions, and default thresholds live in one file:

- **`CLASS_NAMES`** — The 5 detection classes: `["vehicle", "pedestrian", "cyclist", "traffic_light", "traffic_sign"]`. Order matches YOLO class IDs (0-indexed).
- **`CLASS_COLORS`** — BGR color per class for OpenCV visualization. Each class has a distinct, high-contrast color.
- **`CLASS_NAME_TO_ID` / `CLASS_ID_TO_NAME`** — Bidirectional mapping dictionaries, generated from `CLASS_NAMES` using `enumerate()`.
- **`CARLA_SEMANTIC_TAGS`** — Maps CARLA's semantic segmentation tags (integers) to our class names. Multiple CARLA tags map to one UrbanEye class (e.g., CARLA tags 10, 11, 14, 15 all map to "vehicle").
- **Detection thresholds** — `DEFAULT_CONFIDENCE_THRESHOLD = 0.25`, `DEFAULT_NMS_IOU_THRESHOLD = 0.45`, `DEFAULT_IMG_SIZE = 640`.
- **Tracker defaults** — ByteTrack (high_thresh=0.6, low_thresh=0.1, max_lost=30) and DeepSORT (max_age=70, max_cosine_distance=0.3) hyperparameters.
- **CARLA maps, weather presets, time-of-day** — 8 maps, 6 weather presets, 3 time options = 144 total scenario configurations.

**Why centralize constants:** When a threshold changes (e.g., confidence from 0.25 to 0.3), it changes in one place. No grep-and-replace across 20 files.

### 4. `urbaneye/utils/io_helpers.py` — Reusable I/O Utilities

Four utility functions used across all modules:

| Function | Purpose | Error Handling |
|----------|---------|---------------|
| `load_yaml(path)` | Safe YAML loading with `yaml.safe_load` | Raises `FileNotFoundError` for missing file, `YAMLError` for invalid YAML or non-dict content |
| `ensure_dir(path)` | Create directory + parents, idempotent | Returns path for chaining |
| `validate_image_path(path)` | Check file exists + has valid image extension | Returns bool, supports .jpg/.jpeg/.png/.bmp/.tiff/.webp |
| `get_project_root()` | Walk up from current file to find `pyproject.toml` | Raises `FileNotFoundError` if not found |

**Key implementation detail for `load_yaml`:** Uses `yaml.safe_load` (not `yaml.load`) to prevent arbitrary code execution from malicious YAML files. Also validates the return type is a dict — rejects YAML files that contain a list at the top level.

### 5. `.gitignore` — Security-Conscious Exclusions

Explicitly excludes:
- **Model weights** (`.pt`, `.onnx`, `.torchscript`) — too large for git, downloaded separately
- **Data directories** (`data/`, `datasets/`) — 50K+ images don't belong in git
- **Training outputs** (`runs/`, `wandb/`) — ephemeral, regeneratable
- **Video files** (`.mp4`, `.avi`) — large binary files
- **Secrets** (`.env`, `*.pem`, `credentials.json`) — never committed
- **IDE/OS** files (`.vscode/`, `.DS_Store`, `Thumbs.db`)

### 6. `tests/conftest.py` — Shared Test Fixtures

Provides reusable fixtures for all test files:

- `tmp_output_dir` — Temporary directory for test outputs (auto-cleaned by pytest)
- `sample_image` / `sample_image_small` — Random pixel arrays (640x640 and 320x320) with fixed seed for reproducibility
- `sample_bboxes_yolo` — 5 sample YOLO bounding boxes, one per class
- `sample_detections` — 5 detection arrays in [x1, y1, x2, y2, confidence, class_id] format
- `project_config` — Test configuration loaded from a temporary YAML file
- `sample_yaml_file` — Temporary valid YAML file for io_helpers tests
- `sample_label_dir` — Temporary directory with 3 YOLO label files for visualization tests

### 7. Other Files

- `LICENSE` — Apache 2.0 license
- `requirements.txt` / `requirements_carla.txt` — Pinned runtime dependencies
- `configs/project_config.yaml` — Central project configuration with all hyperparameters
- `README.md` — Project overview with badges, architecture diagram, quick start
- `docs/ARCHITECTURE.md` — System architecture with pipeline diagrams

---

## Test Results

```
tests/unit/test_io_helpers.py — 18 tests

TestLoadYaml (6 tests):
  - load valid YAML ✓
  - missing file raises FileNotFoundError ✓
  - malformed YAML raises YAMLError ✓
  - empty YAML returns empty dict ✓
  - list YAML raises YAMLError ✓
  - accepts string path ✓

TestEnsureDir (3 tests):
  - creates new nested directory ✓
  - idempotent on existing directory ✓
  - returns path for chaining ✓

TestValidateImagePath (7 tests):
  - valid .jpg ✓, .png ✓, .bmp ✓
  - invalid .txt extension ✓
  - nonexistent file ✓
  - case-insensitive (.JPG) ✓
  - .jpeg extension ✓

TestGetProjectRoot (2 tests):
  - finds directory containing pyproject.toml ✓
  - root contains urbaneye/ package ✓
```

**All 18 tests passed in 0.27s.**

---

## Files Created in This Phase

```
Project-UrbanEye/
├── .github/workflows/ci.yml          # GitHub Actions CI pipeline
├── .gitignore                         # Security-conscious exclusions
├── LICENSE                            # Apache 2.0
├── README.md                          # Project overview + quick start
├── pyproject.toml                     # Python packaging + tool config
├── requirements.txt                   # Runtime dependencies
├── requirements_carla.txt             # CARLA-specific dependencies
├── configs/project_config.yaml        # Central project configuration
├── docs/ARCHITECTURE.md               # System architecture
├── docs/PHASES.md                     # Phase tracking
├── docs/PLAN.md                       # Full implementation plan
├── urbaneye/__init__.py               # Package root (__version__ = "0.1.0")
├── urbaneye/utils/__init__.py         # Utils subpackage
├── urbaneye/utils/constants.py        # Class names, colors, thresholds, maps
├── urbaneye/utils/io_helpers.py       # YAML loading, path helpers
├── tests/__init__.py                  # Test package
├── tests/conftest.py                  # Shared fixtures
├── tests/unit/__init__.py             # Unit test package
├── tests/unit/test_io_helpers.py      # 18 tests for io_helpers
└── tests/integration/__init__.py      # Integration test package (empty)
```

---

## Key Decisions & Interview Talking Points

1. **pyproject.toml over setup.py** — PEP 621 is the modern standard. Declarative configuration eliminates the need for imperative Python code in packaging.

2. **Optional dependency groups** — A `pip install -e ".[dev]"` takes 30 seconds. A `pip install -e ".[all]"` takes 5 minutes (PyTorch, CUDA). Separating them respects developer time.

3. **Ruff over Black+isort+flake8** — Single tool, 10-100x faster, actively maintained by Astral. One config block in pyproject.toml replaces three separate config files.

4. **Fixed random seeds in fixtures** — `np.random.default_rng(seed=42)` ensures reproducible test data. Tests that depend on random data are flaky tests.

5. **`yaml.safe_load` over `yaml.load`** — Prevents arbitrary code execution from malicious YAML. `yaml.load` can instantiate any Python object, `safe_load` only produces basic Python types.
