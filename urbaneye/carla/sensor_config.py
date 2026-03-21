"""Sensor configuration dataclasses for CARLA data generation.

Defines camera parameters, sensor suites, and simulation settings as
pure Python dataclasses — no CARLA dependency. Configuration can be
loaded from YAML or constructed programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from urbaneye.utils.io_helpers import load_yaml


@dataclass
class CameraSensorConfig:
    """Configuration for a single CARLA camera sensor.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fov: Field of view in degrees.
        fps: Capture frames per second.
        position: Mount position (x, y, z) relative to vehicle center.
        sensor_type: One of 'rgb', 'depth', 'semantic_seg'.
    """

    width: int = 1920
    height: int = 1080
    fov: float = 90.0
    fps: float = 20.0
    position: tuple[float, float, float] = (2.0, 0.0, 1.4)
    sensor_type: str = "rgb"

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty list means valid."""
        errors: list[str] = []
        if self.width <= 0:
            errors.append(f"width must be positive, got {self.width}")
        if self.height <= 0:
            errors.append(f"height must be positive, got {self.height}")
        if not (0.0 < self.fov <= 180.0):
            errors.append(f"fov must be in (0, 180], got {self.fov}")
        if self.fps <= 0:
            errors.append(f"fps must be positive, got {self.fps}")
        valid_types = {"rgb", "depth", "semantic_seg"}
        if self.sensor_type not in valid_types:
            errors.append(f"sensor_type must be one of {valid_types}, got '{self.sensor_type}'")
        if len(self.position) != 3:
            errors.append(f"position must have 3 components, got {len(self.position)}")
        return errors

    @property
    def aspect_ratio(self) -> float:
        """Width / height ratio."""
        return self.width / self.height

    def to_carla_blueprint_attrs(self) -> dict[str, str]:
        """Convert to CARLA blueprint attribute dict for sensor.set_attribute()."""
        return {
            "image_size_x": str(self.width),
            "image_size_y": str(self.height),
            "fov": str(self.fov),
        }


@dataclass
class SensorSuite:
    """Complete multi-sensor rig for the ego vehicle.

    Matches the UrbanEye spec: RGB (1920x1080) + Depth (640x480) +
    Semantic Segmentation (640x480), all at the dashcam position.
    """

    rgb: CameraSensorConfig = field(
        default_factory=lambda: CameraSensorConfig(
            width=1920,
            height=1080,
            fov=90.0,
            fps=20.0,
            position=(2.0, 0.0, 1.4),
            sensor_type="rgb",
        )
    )
    depth: CameraSensorConfig = field(
        default_factory=lambda: CameraSensorConfig(
            width=640,
            height=480,
            fov=90.0,
            fps=20.0,
            position=(2.0, 0.0, 1.4),
            sensor_type="depth",
        )
    )
    semantic: CameraSensorConfig = field(
        default_factory=lambda: CameraSensorConfig(
            width=640,
            height=480,
            fov=90.0,
            fps=20.0,
            position=(2.0, 0.0, 1.4),
            sensor_type="semantic_seg",
        )
    )

    def validate(self) -> list[str]:
        """Validate all sensors in the suite."""
        errors: list[str] = []
        for name in ("rgb", "depth", "semantic"):
            sensor = getattr(self, name)
            for err in sensor.validate():
                errors.append(f"{name}: {err}")
        return errors

    def all_sensors(self) -> list[tuple[str, CameraSensorConfig]]:
        """Return list of (name, config) tuples for iteration."""
        return [("rgb", self.rgb), ("depth", self.depth), ("semantic", self.semantic)]


@dataclass
class SimulationConfig:
    """CARLA simulation settings.

    Attributes:
        sync_mode: Whether to use synchronous mode (required for aligned sensors).
        fixed_delta: Time step in seconds (1/fps).
        no_rendering: Disable rendering for faster data generation.
        maps: List of CARLA maps to use.
        weather_presets: List of weather presets to cycle through.
        time_of_day: List of time-of-day options.
        npc_vehicles_range: (min, max) NPC vehicles to spawn per scenario.
        npc_pedestrians_range: (min, max) NPC pedestrians to spawn per scenario.
    """

    sync_mode: bool = True
    fixed_delta: float = 1.0 / 20.0
    no_rendering: bool = False
    maps: list[str] = field(
        default_factory=lambda: [
            "Town01",
            "Town02",
            "Town03",
            "Town04",
            "Town05",
            "Town06",
            "Town07",
            "Town10HD",
        ]
    )
    weather_presets: list[str] = field(
        default_factory=lambda: [
            "ClearNoon",
            "CloudyNoon",
            "WetNoon",
            "HardRainNoon",
            "ClearSunset",
            "CloudySunset",
        ]
    )
    time_of_day: list[str] = field(default_factory=lambda: ["noon", "sunset", "night"])
    npc_vehicles_range: tuple[int, int] = (50, 150)
    npc_pedestrians_range: tuple[int, int] = (30, 80)

    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors: list[str] = []
        if self.fixed_delta <= 0:
            errors.append(f"fixed_delta must be positive, got {self.fixed_delta}")
        if not self.maps:
            errors.append("maps list cannot be empty")
        if not self.weather_presets:
            errors.append("weather_presets list cannot be empty")
        if (
            self.npc_vehicles_range[0] < 0
            or self.npc_vehicles_range[1] < self.npc_vehicles_range[0]
        ):
            errors.append(f"invalid npc_vehicles_range: {self.npc_vehicles_range}")
        if (
            self.npc_pedestrians_range[0] < 0
            or self.npc_pedestrians_range[1] < self.npc_pedestrians_range[0]
        ):
            errors.append(f"invalid npc_pedestrians_range: {self.npc_pedestrians_range}")
        return errors

    @property
    def total_scenario_configs(self) -> int:
        """Total unique scenario combinations (maps x weather x time)."""
        return len(self.maps) * len(self.weather_presets) * len(self.time_of_day)

    @classmethod
    def from_yaml(cls, path: Path) -> SimulationConfig:
        """Load simulation config from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            Populated SimulationConfig instance.
        """
        data = load_yaml(path)
        sim = data.get("simulation", {})
        defaults = cls()
        return cls(
            sync_mode=sim.get("sync_mode", True),
            fixed_delta=sim.get("fixed_delta_seconds", 1.0 / 20.0),
            no_rendering=sim.get("no_rendering", False),
            maps=data.get("maps", defaults.maps),
            weather_presets=data.get("weather_presets", defaults.weather_presets),
            time_of_day=data.get("time_of_day", defaults.time_of_day),
        )


@dataclass
class FullCarlaConfig:
    """Complete CARLA configuration combining sensors and simulation settings."""

    sensors: SensorSuite = field(default_factory=SensorSuite)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def validate(self) -> list[str]:
        """Validate the complete configuration."""
        errors = self.sensors.validate()
        errors.extend(self.simulation.validate())
        return errors

    @classmethod
    def from_yaml(cls, path: Path) -> FullCarlaConfig:
        """Load complete config from YAML."""
        data = load_yaml(path)

        # Parse sensor configs
        sensors_data = data.get("sensors", {})
        suite = SensorSuite()
        if "rgb" in sensors_data:
            rgb = sensors_data["rgb"]
            suite.rgb = CameraSensorConfig(
                width=rgb.get("width", 1920),
                height=rgb.get("height", 1080),
                fov=rgb.get("fov", 90.0),
                fps=rgb.get("fps", 20.0),
                position=tuple(rgb.get("position", [2.0, 0.0, 1.4])),
                sensor_type="rgb",
            )
        if "depth" in sensors_data:
            depth = sensors_data["depth"]
            suite.depth = CameraSensorConfig(
                width=depth.get("width", 640),
                height=depth.get("height", 480),
                fov=depth.get("fov", 90.0),
                position=tuple(depth.get("position", [2.0, 0.0, 1.4])),
                sensor_type="depth",
            )
        if "semantic" in sensors_data:
            sem = sensors_data["semantic"]
            suite.semantic = CameraSensorConfig(
                width=sem.get("width", 640),
                height=sem.get("height", 480),
                fov=sem.get("fov", 90.0),
                position=tuple(sem.get("position", [2.0, 0.0, 1.4])),
                sensor_type="semantic_seg",
            )

        sim = SimulationConfig.from_yaml(path)
        return cls(sensors=suite, simulation=sim)
