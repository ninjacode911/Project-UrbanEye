"""Tests for urbaneye.carla.scenario_runner module.

All tests use mock objects — no CARLA installation required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from urbaneye.carla.scenario_runner.adverse_weather import (
    AdverseWeatherConfig,
    AdverseWeatherScenario,
    WeatherConfig,
)
from urbaneye.carla.scenario_runner.base_scenario import (
    BaseScenario,
    ScenarioState,
)
from urbaneye.carla.scenario_runner.emergency_vehicle import (
    EmergencyVehicleConfig,
    EmergencyVehicleScenario,
)
from urbaneye.carla.scenario_runner.pedestrian_crossing import (
    PedestrianCrossingConfig,
    PedestrianCrossingScenario,
)


class TestBaseScenario:
    """Tests for BaseScenario abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """BaseScenario is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseScenario()  # type: ignore[abstract]

    def test_initial_state_is_pending(self) -> None:
        """New scenarios start in PENDING state."""
        scenario = PedestrianCrossingScenario()
        assert scenario.state == ScenarioState.PENDING

    def test_tick_transitions_to_active(self) -> None:
        """First tick transitions from PENDING to ACTIVE."""
        scenario = PedestrianCrossingScenario()
        world = MagicMock()
        state = scenario.tick(world, 0)
        assert state == ScenarioState.ACTIVE

    def test_tick_completes_after_duration(self) -> None:
        """Scenario completes after duration_frames ticks."""
        scenario = PedestrianCrossingScenario(PedestrianCrossingConfig(duration_frames=5))
        world = MagicMock()

        for i in range(4):
            state = scenario.tick(world, i)
            assert state == ScenarioState.ACTIVE

        state = scenario.tick(world, 5)
        assert state == ScenarioState.COMPLETED

    def test_tick_repeats_when_configured(self) -> None:
        """Scenario with repeat=True loops instead of completing."""
        config = PedestrianCrossingConfig(duration_frames=3, repeat=True)
        scenario = PedestrianCrossingScenario(config)
        world = MagicMock()

        for i in range(3):
            scenario.tick(world, i)

        # Should still be active (restarted)
        state = scenario.tick(world, 3)
        assert state == ScenarioState.ACTIVE

    def test_cleanup_destroys_actors(self) -> None:
        """cleanup destroys all spawned actors."""
        scenario = PedestrianCrossingScenario()
        mock_actor = MagicMock()
        scenario._spawned_actors = [mock_actor]

        world = MagicMock()
        scenario.cleanup(world)
        mock_actor.destroy.assert_called_once()
        assert len(scenario._spawned_actors) == 0

    def test_frame_count_increments(self) -> None:
        """Frame count increments with each tick."""
        scenario = PedestrianCrossingScenario()
        world = MagicMock()

        for i in range(5):
            scenario.tick(world, i)

        assert scenario.frame_count == 5


class TestPedestrianCrossingScenario:
    """Tests for PedestrianCrossingScenario."""

    def test_name(self) -> None:
        """Scenario has correct name."""
        scenario = PedestrianCrossingScenario()
        assert scenario.name == "pedestrian_crossing"

    def test_description_includes_count(self) -> None:
        """Description includes pedestrian count."""
        config = PedestrianCrossingConfig(num_pedestrians=8)
        scenario = PedestrianCrossingScenario(config)
        assert "8" in scenario.description

    def test_setup_spawns_pedestrians(self) -> None:
        """setup spawns the configured number of pedestrians."""
        config = PedestrianCrossingConfig(num_pedestrians=3)
        scenario = PedestrianCrossingScenario(config)

        world = MagicMock()
        bp_lib = MagicMock()
        bp_lib.filter.return_value = [MagicMock()]
        world.get_blueprint_library.return_value = bp_lib
        world.get_random_location_from_navigation.return_value = MagicMock()
        actor = MagicMock()
        world.spawn_actor.return_value = actor

        scenario.setup(world)
        assert world.spawn_actor.call_count == 3

    def test_default_config(self) -> None:
        """Default config has reasonable values."""
        config = PedestrianCrossingConfig()
        assert config.num_pedestrians == 5
        assert config.crossing_speed == 1.4


class TestAdverseWeatherScenario:
    """Tests for AdverseWeatherScenario."""

    def test_name(self) -> None:
        """Scenario has correct name."""
        scenario = AdverseWeatherScenario()
        assert scenario.name == "adverse_weather"

    def test_initial_weather(self) -> None:
        """Initial weather is the first preset."""
        presets = [WeatherConfig("Rain", precipitation=80)]
        config = AdverseWeatherConfig(presets=presets)
        scenario = AdverseWeatherScenario(config)
        assert scenario.current_weather.name == "Rain"

    def test_weather_changes_at_interval(self) -> None:
        """Weather changes every change_interval frames."""
        presets = [
            WeatherConfig("Clear"),
            WeatherConfig("Rain", precipitation=80),
        ]
        config = AdverseWeatherConfig(presets=presets, change_interval=3, duration_frames=100)
        scenario = AdverseWeatherScenario(config)

        world = MagicMock()

        # First 3 ticks: Clear
        for i in range(3):
            scenario.tick(world, i)
        assert scenario.current_weather.name == "Clear"

        # After interval, should change to Rain
        scenario.tick(world, 3)
        assert scenario.current_weather.name == "Rain"

    def test_setup_applies_weather(self) -> None:
        """setup calls world.set_weather."""
        scenario = AdverseWeatherScenario()
        world = MagicMock()
        scenario.setup(world)
        world.set_weather.assert_called_once()


class TestEmergencyVehicleScenario:
    """Tests for EmergencyVehicleScenario."""

    def test_name(self) -> None:
        """Scenario has correct name."""
        scenario = EmergencyVehicleScenario()
        assert scenario.name == "emergency_vehicle"

    def test_description_includes_count(self) -> None:
        """Description includes vehicle count."""
        config = EmergencyVehicleConfig(num_vehicles=3)
        scenario = EmergencyVehicleScenario(config)
        assert "3" in scenario.description

    def test_setup_spawns_vehicles(self) -> None:
        """setup spawns the configured number of emergency vehicles."""
        config = EmergencyVehicleConfig(num_vehicles=2)
        scenario = EmergencyVehicleScenario(config)

        world = MagicMock()
        bp_lib = MagicMock()
        bp_lib.find.return_value = MagicMock()
        world.get_blueprint_library.return_value = bp_lib
        carla_map = MagicMock()
        carla_map.get_spawn_points.return_value = [MagicMock(), MagicMock(), MagicMock()]
        world.get_map.return_value = carla_map
        actor = MagicMock()
        world.spawn_actor.return_value = actor

        scenario.setup(world)
        assert world.spawn_actor.call_count == 2

    def test_default_config(self) -> None:
        """Default config has reasonable values."""
        config = EmergencyVehicleConfig()
        assert config.num_vehicles == 2
        assert config.approach_from_behind is True
