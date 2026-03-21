# Phase 2: CARLA Sensor Configuration & Annotation Exporter

**Status:** Completed
**Date:** 2026-03-21
**Tests:** 58 new tests (76 cumulative, all passed)

---

## Objective

Build the data generation infrastructure: sensor configuration dataclasses, the 3D-to-YOLO annotation conversion math, and a CARLA configuration file — all fully testable without a running CARLA simulator.

---

## Why This Phase Matters

CARLA only runs on Linux (WSL2) with GPU access. If the data generation code can only be tested with CARLA running, development is bottlenecked by hardware availability. By designing all CARLA interaction behind **pure Python dataclasses** and **pure math functions**, we can:

- Develop and test on Windows without CARLA installed
- Run CI on GitHub Actions (Ubuntu, no GPU)
- Iterate rapidly on annotation logic without waiting for simulator startup (~30s)

This is the same **dependency injection** pattern used in production AV stacks — the perception code doesn't create its own camera; it receives one.

---

## What Was Built

### 1. `urbaneye/carla/sensor_config.py` — Sensor Configuration Dataclasses

Three nested dataclasses define the complete CARLA sensor setup:

#### `CameraSensorConfig`

Represents a single camera sensor with all parameters needed for CARLA blueprint configuration:

| Field | Type | Default | Spec Source |
|-------|------|---------|-------------|
| `width` | int | 1920 | PDF Section 6.1: RGB at 1920x1080 |
| `height` | int | 1080 | PDF Section 6.1 |
| `fov` | float | 90.0 | PDF Section 6.1: 90° FOV |
| `fps` | float | 20.0 | PDF Section 6.1: 20 FPS |
| `position` | tuple | (2.0, 0.0, 1.4) | PDF Section 6.1: dashcam position |
| `sensor_type` | str | "rgb" | One of: rgb, depth, semantic_seg |

**`validate()` method** returns a list of error strings (empty = valid). Checks: positive width/height, FOV in (0, 180], positive FPS, valid sensor type, 3-component position.

**`to_carla_blueprint_attrs()`** converts to the string dict format CARLA expects for `sensor.set_attribute()`.

#### `SensorSuite`

Groups three cameras matching the UrbanEye spec:
- **RGB** — 1920x1080, 90° FOV (primary training data)
- **Depth** — 640x480, 90° FOV (for occlusion filtering and distance validation)
- **Semantic** — 640x480, 90° FOV (for annotation quality verification)

All sensors share the same dashcam mount position `(2.0, 0.0, 1.4)` — forward of vehicle center, centered, 1.4m height.

**`all_sensors()`** returns `list[tuple[str, CameraSensorConfig]]` for iteration. **`validate()`** propagates errors from all three sensors with prefixed names (e.g., `"rgb: width must be positive"`).

#### `SimulationConfig`

Defines the simulation environment parameters:
- **Synchronous mode** (enabled by default) — ensures all sensors capture the same simulation tick, preventing frame misalignment
- **Fixed delta** = 1/20 = 0.05s — matches the 20 FPS capture rate
- **8 maps** (Town01-Town10HD), **6 weather presets**, **3 time-of-day** = **144 total scenario configurations**
- **NPC ranges** — 50-150 vehicles, 30-80 pedestrians per scenario

**`total_scenario_configs`** property computes `len(maps) * len(weather_presets) * len(time_of_day)` = 144.

**`from_yaml(path)`** class method loads config from YAML. Uses an instance default (`defaults = cls()`) for fallback values rather than class attributes (dataclass `field(default_factory=...)` creates instance attributes, not class attributes — a subtle Python gotcha that was caught and fixed during testing).

#### `FullCarlaConfig`

Combines `SensorSuite` + `SimulationConfig` with a unified `validate()` and `from_yaml()`.

### 2. `urbaneye/carla/annotation_exporter.py` — 3D-to-YOLO Coordinate Transformation

This module contains the core computer vision math that converts CARLA's 3D bounding boxes into YOLO-format 2D annotations. This is the same projection math used in every camera-based AV perception system.

#### The Transformation Pipeline

```
3D World Coordinates (CARLA)
         │
         ▼
    world_to_camera()           ← 4x4 extrinsic matrix (world → camera frame)
         │
         ▼
    Camera Coordinates
         │
         ▼
    Perspective Division         ← Divide by Z (depth)
         │
         ▼
    2D Pixel Coordinates         ← 3x3 intrinsic matrix K
         │
         ▼
    pixel_bbox_to_yolo()        ← Normalize to [0, 1], convert to center format
         │
         ▼
    YOLO Format: "class_id cx cy w h"
```

#### `build_camera_intrinsics(width, height, fov_degrees) → K (3x3)`

Constructs the pinhole camera intrinsic matrix:

```
K = | fx   0   cx |      fx = width / (2 * tan(fov/2))
    |  0   fy  cy |      fy = fx  (square pixels)
    |  0    0   1 |      cx = width / 2,  cy = height / 2
```

The focal length `fx` is derived from the horizontal field of view — wider FOV means shorter focal length (objects appear smaller).

#### `world_to_camera(world_point, camera_transform, camera_intrinsics) → pixel | None`

Projects a single 3D point to 2D:
1. Transform to camera coordinates: `cam_point = camera_transform @ world_point`
2. Check depth: if `cam_point[2] <= 0`, point is behind camera → return `None`
3. Project: `pixel = K @ cam_point[:3]`, then divide by Z
4. Return `(u, v)` pixel coordinates

#### `bbox_3d_to_2d(corners_3d, camera_transform, camera_intrinsics) → (x_min, y_min, x_max, y_max) | None`

Projects all 8 corners of a 3D bounding box, computes the 2D axis-aligned bounding rectangle. Returns `None` if all corners are behind the camera.

#### `pixel_bbox_to_yolo(x_min, y_min, x_max, y_max, class_id, img_width, img_height) → str | None`

Converts pixel bounding box to YOLO format with validation:
1. **Clamp** coordinates to image boundaries
2. **Normalize** to [0, 1]: `cx = (x_min + x_max) / 2 / img_width`
3. **Filter invalid boxes**: zero area, center outside frame, very small (< 0.001), invalid class ID
4. **Return** format: `"class_id cx cy w h"` with 6 decimal places

#### `validate_yolo_annotation(line) → bool`

Validates a single YOLO annotation line: 5 fields, integer class ID in [0, NUM_CLASSES), all coordinates in [0, 1], positive width and height.

#### `export_frame_annotations(annotations, output_path) → Path`

Writes a list of YOLO annotation strings to a `.txt` file. Creates parent directories automatically.

#### `load_annotations(label_path) → list[tuple]`

Reads a YOLO label file back into structured tuples. Skips invalid lines gracefully.

### 3. `urbaneye/carla/carla_config.yaml` — Complete CARLA Configuration

Defines all parameters for data generation in a single YAML file:
- Simulation settings (sync mode, 20 FPS, rendering)
- Sensor parameters (RGB 1920x1080, Depth 640x480, Semantic 640x480)
- 8 CARLA maps with inline comments (e.g., "Town04 — Highway with on/off ramps")
- 6 weather presets
- 3 time-of-day options
- 5 detection class definitions
- NPC spawn ranges (50-150 vehicles, 30-80 pedestrians)
- Dataset split ratios (80% train / 10% val / 10% test, by scenario not by frame to prevent data leakage)

---

## Test Results

```
tests/unit/test_sensor_config.py — 28 tests

TestCameraSensorConfig (10 tests):
  - default RGB matches spec (1920x1080, 90 FOV, dashcam position) ✓
  - validate: valid config → no errors ✓
  - validate: negative width caught ✓
  - validate: zero height caught ✓
  - validate: FOV > 180 caught ✓
  - validate: FOV = 0 caught ✓
  - validate: negative FPS caught ✓
  - validate: invalid sensor type caught ✓
  - aspect ratio computed correctly ✓
  - to_carla_blueprint_attrs format correct ✓

TestSensorSuite (7 tests):
  - 3 sensors (rgb, depth, semantic) ✓
  - RGB is 1920x1080 ✓
  - Depth is 640x480 ✓
  - Semantic is 640x480 ✓
  - valid suite → no errors ✓
  - propagates sensor errors with prefix ✓
  - all sensors at same dashcam position ✓

TestSimulationConfig (9 tests):
  - sync mode enabled by default ✓
  - fixed delta = 0.05 (20 FPS) ✓
  - 8 default maps ✓
  - 6 default weather presets ✓
  - 144 total scenario configs ✓
  - valid config → no errors ✓
  - negative delta caught ✓
  - empty maps caught ✓
  - from_yaml loads correctly ✓

TestFullCarlaConfig (2 tests):
  - default is valid ✓
  - from_yaml loads sensors correctly ✓

tests/unit/test_annotation_exporter.py — 30 tests

TestBuildCameraIntrinsics (5 tests):
  - 3x3 shape ✓
  - principal point at image center ✓
  - positive focal length ✓
  - square pixels (fx == fy) ✓
  - wider FOV → shorter focal length ✓

TestWorldToCamera (3 tests):
  - point in front → valid pixel ✓
  - point behind camera → None ✓
  - accepts homogeneous coordinates ✓

TestBbox3dTo2d (2 tests):
  - visible box → valid 2D bbox ✓
  - behind camera → None ✓

TestPixelBboxToYolo (6 tests):
  - center box valid ✓
  - zero area → None ✓
  - negative area → None ✓
  - invalid class ID → None ✓
  - tiny box → None ✓
  - edge box clamped ✓

TestCaralBboxToYolo (1 test):
  - full pipeline: 3D → YOLO annotation ✓

TestValidateYoloAnnotation (7 tests):
  - valid annotation passes ✓
  - all 5 class IDs valid ✓
  - invalid class ID fails ✓
  - out-of-range coords fail ✓
  - zero dimensions fail ✓
  - wrong field count fails ✓
  - non-numeric fails ✓

TestExportFrameAnnotations (3 tests):
  - writes file correctly ✓
  - creates parent directories ✓
  - empty annotations → empty file ✓

TestLoadAnnotations (3 tests):
  - loads valid file ✓
  - missing file → empty list ✓
  - skips invalid lines ✓
```

**All 58 new tests passed.**

---

## Files Created in This Phase

```
urbaneye/carla/__init__.py                 # CARLA subpackage
urbaneye/carla/sensor_config.py            # 4 dataclasses, validation, YAML loading
urbaneye/carla/annotation_exporter.py      # 3D→YOLO projection, export, validation
urbaneye/carla/carla_config.yaml           # Complete CARLA configuration
tests/unit/test_sensor_config.py           # 28 tests
tests/unit/test_annotation_exporter.py     # 30 tests
```

---

## Key Decisions & Interview Talking Points

1. **Dependency injection over direct CARLA imports** — `sensor_config.py` has zero CARLA imports. All CARLA objects are passed as parameters, enabling mock-based testing. This is the Strategy pattern applied to sensor configuration.

2. **Pinhole camera model** — The intrinsic matrix `K` is derived from field-of-view using `fx = width / (2 * tan(fov/2))`. This is the standard model used in OpenCV, KITTI, and every camera calibration paper.

3. **Multi-layer validation in annotation export** — A YOLO annotation goes through 5 filters: behind-camera check, frame-boundary clamping, zero-area removal, minimum-size threshold, class ID validation. Each filter is independently testable.

4. **`from_yaml` with instance defaults** — Dataclass `field(default_factory=...)` creates instance attributes, not class attributes. Accessing `cls.maps` raises `AttributeError`. The fix: `defaults = cls()` creates a default instance. This is a common Python dataclass pitfall worth mentioning in interviews.

5. **Scenario-based dataset split** — Splitting by scenario (not by frame) prevents data leakage. If frame 100 and frame 101 from the same scenario end up in train and test respectively, the model isn't really being tested on unseen data.
