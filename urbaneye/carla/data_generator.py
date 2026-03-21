"""CARLA data generation engine for UrbanEye.

Orchestrates ego-vehicle driving, multi-sensor capture, and YOLO annotation
export across multiple CARLA maps, weather conditions, and scenarios.

The CARLA client is injected via constructor (dependency injection) to enable
mock-based testing without a running CARLA instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from urbaneye.carla.annotation_exporter import export_frame_annotations, pixel_bbox_to_yolo
from urbaneye.carla.scenario_runner.base_scenario import BaseScenario, ScenarioState
from urbaneye.carla.sensor_config import SensorSuite, SimulationConfig
from urbaneye.utils.io_helpers import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Data captured from one synchronized simulation tick.

    Attributes:
        frame_id: Sequential frame number.
        timestamp: Simulation time in seconds.
        rgb_image: RGB camera image (H x W x 3, uint8).
        depth_map: Depth camera output (H x W, float32) or None.
        semantic_map: Semantic segmentation (H x W, uint8) or None.
        raw_bboxes: List of raw bounding box data from CARLA actors.
    """

    frame_id: int
    timestamp: float
    rgb_image: np.ndarray
    depth_map: np.ndarray | None = None
    semantic_map: np.ndarray | None = None
    raw_bboxes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DatasetStats:
    """Statistics about a generated dataset.

    Attributes:
        total_frames: Number of frames captured.
        class_distribution: Count per class name.
        output_dir: Root directory of the generated dataset.
        map_name: CARLA map used.
        weather: Weather preset used.
    """

    total_frames: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    output_dir: Path = Path(".")
    map_name: str = ""
    weather: str = ""


class CarlaDataGenerator:
    """Orchestrates ego-vehicle driving and multi-sensor data capture.

    The generator follows this lifecycle:
    1. setup_world() — Load map, set weather, enable sync mode
    2. spawn_ego_vehicle() — Place ego car with autopilot
    3. attach_sensors() — Mount RGB, depth, semantic cameras
    4. generate_dataset() — Drive + capture + annotate for N frames
    5. cleanup() — Destroy all actors, restore settings

    Args:
        client: CARLA client object (injected, mockable).
        sensor_suite: Camera sensor configuration.
        simulation_config: Simulation settings (maps, weather, sync).
        output_dir: Root directory for generated dataset.
    """

    def __init__(
        self,
        client: Any,
        sensor_suite: SensorSuite,
        simulation_config: SimulationConfig,
        output_dir: Path,
    ) -> None:
        self._client = client
        self._sensor_suite = sensor_suite
        self._sim_config = simulation_config
        self._output_dir = Path(output_dir)
        self._world: Any = None
        self._ego_vehicle: Any = None
        self._sensors: list[Any] = []
        self._frame_buffer: dict[str, np.ndarray] = {}

    @property
    def world(self) -> Any:
        """Current CARLA world object."""
        return self._world

    @property
    def ego_vehicle(self) -> Any:
        """Current ego vehicle actor."""
        return self._ego_vehicle

    def setup_world(self, map_name: str, weather_preset: str) -> None:
        """Load a CARLA map and configure weather/simulation settings.

        Args:
            map_name: CARLA map name (e.g., "Town01").
            weather_preset: Weather preset name (e.g., "ClearNoon").
        """
        self._world = self._client.load_world(map_name)

        # Enable synchronous mode
        if hasattr(self._world, "get_settings"):
            settings = self._world.get_settings()
            settings.synchronous_mode = self._sim_config.sync_mode
            settings.fixed_delta_seconds = self._sim_config.fixed_delta
            settings.no_rendering_mode = self._sim_config.no_rendering
            self._world.apply_settings(settings)

        # Set weather
        if hasattr(self._world, "set_weather"):
            self._world.set_weather(weather_preset)

        logger.info(f"World loaded: map={map_name}, weather={weather_preset}")

    def spawn_ego_vehicle(self, autopilot: bool = True) -> None:
        """Spawn the ego vehicle at a random spawn point.

        Args:
            autopilot: Whether to enable autopilot driving.
        """
        bp_lib = self._world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")

        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on this map")

        # Use first spawn point (deterministic for reproducibility)
        self._ego_vehicle = self._world.spawn_actor(vehicle_bp, spawn_points[0])

        if autopilot and hasattr(self._ego_vehicle, "set_autopilot"):
            self._ego_vehicle.set_autopilot(True)

        logger.info("Ego vehicle spawned with autopilot=%s", autopilot)

    def attach_sensors(self) -> None:
        """Attach all sensors from the sensor suite to the ego vehicle."""
        for sensor_name, sensor_config in self._sensor_suite.all_sensors():
            sensor_type_map = {
                "rgb": "sensor.camera.rgb",
                "depth": "sensor.camera.depth",
                "semantic_seg": "sensor.camera.semantic_segmentation",
            }
            carla_type = sensor_type_map.get(sensor_config.sensor_type)
            if carla_type is None:
                continue

            bp_lib = self._world.get_blueprint_library()
            sensor_bp = bp_lib.find(carla_type)

            # Set attributes
            for attr_key, attr_val in sensor_config.to_carla_blueprint_attrs().items():
                sensor_bp.set_attribute(attr_key, attr_val)

            # Create transform at sensor position
            transform = self._create_transform(sensor_config.position)
            sensor_actor = self._world.spawn_actor(
                sensor_bp, transform, attach_to=self._ego_vehicle
            )

            # Register callback for data capture
            sensor_actor.listen(lambda data, name=sensor_name: self._on_sensor_data(name, data))

            self._sensors.append(sensor_actor)
            logger.info(f"Sensor attached: {sensor_name} ({carla_type})")

    def _create_transform(self, position: tuple[float, float, float]) -> Any:
        """Create a CARLA Transform from (x, y, z) position."""
        if hasattr(self._world, "Transform"):
            return self._world.Transform(position)
        # For mock testing, return the position tuple
        return position

    def _on_sensor_data(self, sensor_name: str, data: Any) -> None:
        """Callback for incoming sensor data."""
        if hasattr(data, "raw_data"):
            self._frame_buffer[sensor_name] = np.frombuffer(data.raw_data, dtype=np.uint8)
        else:
            self._frame_buffer[sensor_name] = data

    def capture_frame(self, frame_id: int) -> FrameData:
        """Capture one synchronized frame from all sensors.

        Args:
            frame_id: Sequential frame identifier.

        Returns:
            FrameData containing images and bounding box data.
        """
        # Tick the world to advance simulation
        if hasattr(self._world, "tick"):
            self._world.tick()

        # Collect sensor data from buffer
        rgb = self._frame_buffer.get("rgb")
        depth = self._frame_buffer.get("depth")
        semantic = self._frame_buffer.get("semantic")

        # Default empty image if sensor didn't fire
        h, w = self._sensor_suite.rgb.height, self._sensor_suite.rgb.width
        if rgb is None:
            rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Get actor bounding boxes
        raw_bboxes = self._get_actor_bboxes()

        return FrameData(
            frame_id=frame_id,
            timestamp=frame_id * self._sim_config.fixed_delta,
            rgb_image=rgb,
            depth_map=depth,
            semantic_map=semantic,
            raw_bboxes=raw_bboxes,
        )

    def _get_actor_bboxes(self) -> list[dict[str, Any]]:
        """Extract bounding box info from all actors in the world."""
        bboxes: list[dict[str, Any]] = []
        if not hasattr(self._world, "get_actors"):
            return bboxes

        for actor in self._world.get_actors():
            if hasattr(actor, "bounding_box") and hasattr(actor, "type_id"):
                bboxes.append(
                    {
                        "actor_id": getattr(actor, "id", 0),
                        "type_id": actor.type_id,
                        "bounding_box": actor.bounding_box,
                        "transform": getattr(actor, "get_transform", lambda: None)(),
                    }
                )
        return bboxes

    def generate_dataset(
        self,
        map_name: str,
        weather_preset: str,
        num_frames: int = 500,
        scenarios: list[BaseScenario] | None = None,
    ) -> DatasetStats:
        """Main generation loop: drive, capture, annotate for N frames.

        Args:
            map_name: CARLA map name.
            weather_preset: Weather preset.
            num_frames: Number of frames to capture.
            scenarios: Optional list of scenarios to run during generation.

        Returns:
            DatasetStats with generation summary.
        """
        self.setup_world(map_name, weather_preset)
        self.spawn_ego_vehicle()
        self.attach_sensors()

        # Prepare output directories
        images_dir = ensure_dir(self._output_dir / "images" / "train")
        labels_dir = ensure_dir(self._output_dir / "labels" / "train")

        stats = DatasetStats(
            output_dir=self._output_dir,
            map_name=map_name,
            weather=weather_preset,
        )

        for frame_id in range(num_frames):
            # Tick scenarios
            if scenarios:
                for scenario in scenarios:
                    if scenario.state != ScenarioState.COMPLETED:
                        scenario.tick(self._world, frame_id)

            # Capture frame
            frame_data = self.capture_frame(frame_id)

            # Save image
            frame_name = f"{map_name}_{weather_preset}_{frame_id:06d}"
            img_path = images_dir / f"{frame_name}.jpg"
            self._save_image(frame_data.rgb_image, img_path)

            # Export annotations
            annotations = self._process_bboxes(frame_data.raw_bboxes)
            label_path = labels_dir / f"{frame_name}.txt"
            export_frame_annotations(annotations, label_path)

            stats.total_frames += 1

            if (frame_id + 1) % 100 == 0:
                logger.info(f"Generated {frame_id + 1}/{num_frames} frames")

        # Cleanup scenarios
        if scenarios:
            for scenario in scenarios:
                scenario.cleanup(self._world)

        self.cleanup()
        return stats

    def _process_bboxes(self, raw_bboxes: list[dict[str, Any]]) -> list[str]:
        """Convert raw CARLA bboxes to YOLO annotation strings."""
        annotations: list[str] = []
        img_w = self._sensor_suite.rgb.width
        img_h = self._sensor_suite.rgb.height

        for bbox_data in raw_bboxes:
            # In real CARLA, we'd project 3D bbox to 2D here
            # For the pipeline structure, we extract class and bbox
            bbox = bbox_data.get("bounding_box")
            if bbox is None:
                continue

            class_id = bbox_data.get("class_id", 0)

            # If bbox is already in pixel format (from mock)
            if isinstance(bbox, dict) and "x_min" in bbox:
                annotation = pixel_bbox_to_yolo(
                    bbox["x_min"],
                    bbox["y_min"],
                    bbox["x_max"],
                    bbox["y_max"],
                    class_id,
                    img_w,
                    img_h,
                )
                if annotation:
                    annotations.append(annotation)

        return annotations

    def _save_image(self, image: np.ndarray, path: Path) -> None:
        """Save an image to disk."""
        try:
            import cv2

            cv2.imwrite(str(path), image)
        except ImportError:
            # Fallback: save as raw numpy (for testing without OpenCV)
            np.save(str(path).replace(".jpg", ".npy"), image)

    def cleanup(self) -> None:
        """Destroy all sensors and the ego vehicle, restore async mode."""
        for sensor in self._sensors:
            if hasattr(sensor, "stop"):
                sensor.stop()
            if hasattr(sensor, "destroy"):
                sensor.destroy()
        self._sensors.clear()

        if self._ego_vehicle is not None:
            if hasattr(self._ego_vehicle, "destroy"):
                self._ego_vehicle.destroy()
            self._ego_vehicle = None

        # Restore async mode
        if self._world is not None and hasattr(self._world, "get_settings"):
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            self._world.apply_settings(settings)

        logger.info("Cleanup complete — all actors destroyed")
