"""Centralized constants for the UrbanEye pipeline.

All class definitions, color mappings, CARLA semantic tags, and default
thresholds are defined here to prevent magic numbers across modules.
"""

from __future__ import annotations

# Detection class names — order matches YOLO class IDs (0-indexed)
CLASS_NAMES: list[str] = [
    "vehicle",
    "pedestrian",
    "cyclist",
    "traffic_light",
    "traffic_sign",
]

NUM_CLASSES: int = len(CLASS_NAMES)

# BGR color per class for OpenCV visualization
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "vehicle": (0, 255, 0),  # Green
    "pedestrian": (0, 0, 255),  # Red
    "cyclist": (255, 165, 0),  # Orange
    "traffic_light": (0, 255, 255),  # Yellow
    "traffic_sign": (255, 0, 255),  # Magenta
}

# Mapping from class name to YOLO class ID
CLASS_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Mapping from YOLO class ID to class name
CLASS_ID_TO_NAME: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# CARLA semantic segmentation tag to UrbanEye class name
# Reference: https://carla.readthedocs.io/en/0.9.15/ref_sensors/#semantic-segmentation-camera
CARLA_SEMANTIC_TAGS: dict[int, str] = {
    10: "vehicle",  # Car
    11: "vehicle",  # Truck
    12: "pedestrian",  # Pedestrian
    13: "cyclist",  # Rider (cyclist/motorcyclist)
    14: "vehicle",  # Motorcycle
    15: "vehicle",  # Bicycle (unmanned)
    18: "traffic_light",  # TrafficLight
    19: "traffic_sign",  # TrafficSign
}

# Default detection thresholds
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
DEFAULT_NMS_IOU_THRESHOLD: float = 0.45
DEFAULT_IMG_SIZE: int = 640

# CARLA sensor defaults
DEFAULT_RGB_WIDTH: int = 1920
DEFAULT_RGB_HEIGHT: int = 1080
DEFAULT_RGB_FOV: float = 90.0
DEFAULT_SENSOR_FPS: float = 20.0
DEFAULT_DASHCAM_POSITION: tuple[float, float, float] = (2.0, 0.0, 1.4)

# Tracking defaults
BYTETRACK_HIGH_THRESH: float = 0.6
BYTETRACK_LOW_THRESH: float = 0.1
BYTETRACK_MATCH_THRESH: float = 0.8
BYTETRACK_MAX_LOST: int = 30
BYTETRACK_MIN_HITS: int = 3

DEEPSORT_MAX_AGE: int = 70
DEEPSORT_N_INIT: int = 3
DEEPSORT_MAX_COSINE_DISTANCE: float = 0.3
DEEPSORT_NN_BUDGET: int = 100

# CARLA maps used for data generation
CARLA_MAPS: list[str] = [
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
]

# Weather presets
WEATHER_PRESETS: list[str] = [
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "HardRainNoon",
    "ClearSunset",
    "CloudySunset",
]

# Time-of-day options
TIME_OF_DAY: list[str] = [
    "noon",
    "sunset",
    "night",
]
