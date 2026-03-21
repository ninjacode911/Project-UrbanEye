"""Adverse weather scenario for CARLA.

Cycles through rain, fog, and night conditions to generate training
data that makes the model robust to visual degradation — the primary
sim-to-real gap challenge for autonomous driving perception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from urbaneye.carla.scenario_runner.base_scenario import (
    BaseScenario,
    ScenarioConfig,
)


@dataclass
class WeatherConfig:
    """Single weather configuration.

    Attributes:
        name: Human-readable name.
        cloudiness: Cloud coverage percentage (0-100).
        precipitation: Rain intensity (0-100).
        fog_density: Fog density (0-100).
        sun_altitude_angle: Sun angle in degrees (-90 to 90).
        wetness: Road wetness (0-100).
    """

    name: str = "ClearNoon"
    cloudiness: float = 0.0
    precipitation: float = 0.0
    fog_density: float = 0.0
    sun_altitude_angle: float = 70.0
    wetness: float = 0.0


# Pre-defined weather configurations matching the PDF spec
WEATHER_PRESETS: list[WeatherConfig] = [
    WeatherConfig("ClearNoon", 10, 0, 0, 70, 0),
    WeatherConfig("CloudyNoon", 80, 0, 0, 70, 0),
    WeatherConfig("WetNoon", 50, 30, 10, 70, 50),
    WeatherConfig("HardRainNoon", 90, 80, 20, 70, 100),
    WeatherConfig("ClearSunset", 10, 0, 0, 5, 0),
    WeatherConfig("CloudySunset", 80, 0, 15, 5, 0),
    WeatherConfig("NightClear", 10, 0, 0, -80, 0),
    WeatherConfig("NightRain", 90, 60, 30, -80, 80),
]


@dataclass
class AdverseWeatherConfig(ScenarioConfig):
    """Configuration for adverse weather scenario.

    Attributes:
        presets: Weather presets to cycle through.
        change_interval: Frames between weather changes.
    """

    presets: list[WeatherConfig] = field(default_factory=lambda: list(WEATHER_PRESETS))
    change_interval: int = 100


class AdverseWeatherScenario(BaseScenario):
    """Cycles through weather conditions during data generation.

    This scenario tests:
    - Detection robustness in rain, fog, and night
    - Reduced visibility handling
    - Wet road reflections
    - Headlight glare in nighttime
    """

    def __init__(self, config: AdverseWeatherConfig | None = None) -> None:
        super().__init__(config or AdverseWeatherConfig())
        self._weather_config = config or AdverseWeatherConfig()
        self._current_preset_idx: int = 0

    @property
    def name(self) -> str:
        return "adverse_weather"

    @property
    def description(self) -> str:
        return (
            f"Cycles through {len(self._weather_config.presets)} weather presets "
            f"every {self._weather_config.change_interval} frames"
        )

    @property
    def current_weather(self) -> WeatherConfig:
        """Currently active weather preset."""
        return self._weather_config.presets[self._current_preset_idx]

    def setup(self, world: Any) -> None:
        """Apply the first weather preset.

        Args:
            world: CARLA world object with set_weather() method.
        """
        self._apply_weather(world)

    def _apply_weather(self, world: Any) -> None:
        """Apply the current weather preset to the world."""
        if not hasattr(world, "set_weather"):
            return
        preset = self.current_weather
        world.set_weather(
            cloudiness=preset.cloudiness,
            precipitation=preset.precipitation,
            fog_density=preset.fog_density,
            sun_altitude_angle=preset.sun_altitude_angle,
            wetness=preset.wetness,
        )

    def _tick_impl(self, world: Any, frame_id: int) -> None:
        """Advance to next weather preset at interval.

        Args:
            world: CARLA world object.
            frame_id: Current frame number.
        """
        if self._frame_count > 0 and self._frame_count % self._weather_config.change_interval == 0:
            self._current_preset_idx = (self._current_preset_idx + 1) % len(
                self._weather_config.presets
            )
            self._apply_weather(world)
