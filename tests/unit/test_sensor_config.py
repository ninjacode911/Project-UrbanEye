"""Tests for urbaneye.carla.sensor_config module."""

from __future__ import annotations

from pathlib import Path

import yaml

from urbaneye.carla.sensor_config import (
    CameraSensorConfig,
    FullCarlaConfig,
    SensorSuite,
    SimulationConfig,
)


class TestCameraSensorConfig:
    """Tests for CameraSensorConfig dataclass."""

    def test_default_rgb_matches_spec(self) -> None:
        """Default RGB config matches PDF spec: 1920x1080, 90 FOV, dashcam position."""
        cfg = CameraSensorConfig()
        assert cfg.width == 1920
        assert cfg.height == 1080
        assert cfg.fov == 90.0
        assert cfg.fps == 20.0
        assert cfg.position == (2.0, 0.0, 1.4)
        assert cfg.sensor_type == "rgb"

    def test_validate_valid_config(self) -> None:
        """Valid config produces no errors."""
        cfg = CameraSensorConfig()
        assert cfg.validate() == []

    def test_validate_negative_width(self) -> None:
        """Negative width is detected."""
        cfg = CameraSensorConfig(width=-100)
        errors = cfg.validate()
        assert any("width" in e for e in errors)

    def test_validate_zero_height(self) -> None:
        """Zero height is detected."""
        cfg = CameraSensorConfig(height=0)
        errors = cfg.validate()
        assert any("height" in e for e in errors)

    def test_validate_fov_over_180(self) -> None:
        """FOV > 180 is detected."""
        cfg = CameraSensorConfig(fov=200.0)
        errors = cfg.validate()
        assert any("fov" in e for e in errors)

    def test_validate_fov_zero(self) -> None:
        """FOV = 0 is detected."""
        cfg = CameraSensorConfig(fov=0.0)
        errors = cfg.validate()
        assert any("fov" in e for e in errors)

    def test_validate_negative_fps(self) -> None:
        """Negative fps is detected."""
        cfg = CameraSensorConfig(fps=-5.0)
        errors = cfg.validate()
        assert any("fps" in e for e in errors)

    def test_validate_invalid_sensor_type(self) -> None:
        """Unknown sensor type is detected."""
        cfg = CameraSensorConfig(sensor_type="lidar")
        errors = cfg.validate()
        assert any("sensor_type" in e for e in errors)

    def test_aspect_ratio(self) -> None:
        """Aspect ratio is correctly computed."""
        cfg = CameraSensorConfig(width=1920, height=1080)
        assert abs(cfg.aspect_ratio - 16.0 / 9.0) < 0.01

    def test_to_carla_blueprint_attrs(self) -> None:
        """Blueprint attributes are correctly formatted."""
        cfg = CameraSensorConfig(width=1920, height=1080, fov=90.0)
        attrs = cfg.to_carla_blueprint_attrs()
        assert attrs == {"image_size_x": "1920", "image_size_y": "1080", "fov": "90.0"}


class TestSensorSuite:
    """Tests for SensorSuite dataclass."""

    def test_default_suite_has_three_sensors(self) -> None:
        """Default suite contains rgb, depth, and semantic sensors."""
        suite = SensorSuite()
        sensors = suite.all_sensors()
        assert len(sensors) == 3
        names = [name for name, _ in sensors]
        assert names == ["rgb", "depth", "semantic"]

    def test_default_rgb_is_1920x1080(self) -> None:
        """RGB sensor defaults to 1920x1080."""
        suite = SensorSuite()
        assert suite.rgb.width == 1920
        assert suite.rgb.height == 1080

    def test_default_depth_is_640x480(self) -> None:
        """Depth sensor defaults to 640x480."""
        suite = SensorSuite()
        assert suite.depth.width == 640
        assert suite.depth.height == 480

    def test_default_semantic_is_640x480(self) -> None:
        """Semantic seg sensor defaults to 640x480."""
        suite = SensorSuite()
        assert suite.semantic.width == 640
        assert suite.semantic.height == 480

    def test_validate_valid_suite(self) -> None:
        """Valid suite produces no errors."""
        suite = SensorSuite()
        assert suite.validate() == []

    def test_validate_propagates_sensor_errors(self) -> None:
        """Suite validation includes errors from individual sensors."""
        suite = SensorSuite()
        suite.rgb = CameraSensorConfig(width=-1)
        errors = suite.validate()
        assert any("rgb" in e for e in errors)

    def test_all_sensors_at_same_position(self) -> None:
        """All sensors are mounted at the same dashcam position."""
        suite = SensorSuite()
        positions = [cfg.position for _, cfg in suite.all_sensors()]
        assert all(pos == (2.0, 0.0, 1.4) for pos in positions)


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_default_sync_mode_enabled(self) -> None:
        """Synchronous mode is enabled by default."""
        cfg = SimulationConfig()
        assert cfg.sync_mode is True

    def test_default_fixed_delta_is_20fps(self) -> None:
        """Fixed delta corresponds to 20 FPS."""
        cfg = SimulationConfig()
        assert abs(cfg.fixed_delta - 0.05) < 0.001

    def test_default_8_maps(self) -> None:
        """Default config has 8 CARLA maps."""
        cfg = SimulationConfig()
        assert len(cfg.maps) == 8
        assert "Town01" in cfg.maps
        assert "Town10HD" in cfg.maps

    def test_default_6_weather_presets(self) -> None:
        """Default config has 6 weather presets."""
        cfg = SimulationConfig()
        assert len(cfg.weather_presets) == 6

    def test_total_scenario_configs(self) -> None:
        """Total combinations = 8 maps x 6 weather x 3 time = 144."""
        cfg = SimulationConfig()
        assert cfg.total_scenario_configs == 144

    def test_validate_valid_config(self) -> None:
        """Valid config produces no errors."""
        cfg = SimulationConfig()
        assert cfg.validate() == []

    def test_validate_negative_delta(self) -> None:
        """Negative fixed_delta is detected."""
        cfg = SimulationConfig(fixed_delta=-0.01)
        errors = cfg.validate()
        assert any("fixed_delta" in e for e in errors)

    def test_validate_empty_maps(self) -> None:
        """Empty maps list is detected."""
        cfg = SimulationConfig(maps=[])
        errors = cfg.validate()
        assert any("maps" in e for e in errors)

    def test_from_yaml(self, tmp_path: Path) -> None:
        """from_yaml correctly loads configuration."""
        config_data = {
            "simulation": {"sync_mode": True, "fixed_delta_seconds": 0.05},
            "maps": ["Town01", "Town03"],
            "weather_presets": ["ClearNoon"],
            "time_of_day": ["noon"],
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        cfg = SimulationConfig.from_yaml(yaml_path)
        assert cfg.sync_mode is True
        assert cfg.maps == ["Town01", "Town03"]
        assert cfg.weather_presets == ["ClearNoon"]


class TestFullCarlaConfig:
    """Tests for FullCarlaConfig dataclass."""

    def test_default_is_valid(self) -> None:
        """Default FullCarlaConfig passes validation."""
        cfg = FullCarlaConfig()
        assert cfg.validate() == []

    def test_from_yaml_loads_sensors(self, tmp_path: Path) -> None:
        """from_yaml correctly loads sensor configuration."""
        config_data = {
            "simulation": {"sync_mode": True, "fixed_delta_seconds": 0.05},
            "sensors": {
                "rgb": {"width": 1280, "height": 720, "fov": 90.0, "position": [2.0, 0.0, 1.4]},
                "depth": {"width": 320, "height": 240, "fov": 90.0},
                "semantic": {"width": 320, "height": 240, "fov": 90.0},
            },
            "maps": ["Town01"],
            "weather_presets": ["ClearNoon"],
            "time_of_day": ["noon"],
        }
        yaml_path = tmp_path / "carla_config.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        cfg = FullCarlaConfig.from_yaml(yaml_path)
        assert cfg.sensors.rgb.width == 1280
        assert cfg.sensors.rgb.height == 720
        assert cfg.sensors.depth.width == 320
