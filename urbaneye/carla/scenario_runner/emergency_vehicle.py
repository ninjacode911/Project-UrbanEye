"""Emergency vehicle scenario for CARLA.

Spawns ambulances and fire trucks with active lights, testing detection
of rare vehicle types that appear infrequently in real-world datasets
but are critical for autonomous driving safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from urbaneye.carla.scenario_runner.base_scenario import (
    BaseScenario,
    ScenarioConfig,
)

# CARLA vehicle blueprints for emergency vehicles
EMERGENCY_BLUEPRINTS: list[str] = [
    "vehicle.carlamotors.firetruck",
    "vehicle.ford.ambulance",
    "vehicle.mercedes.sprinter",
]


@dataclass
class EmergencyVehicleConfig(ScenarioConfig):
    """Configuration for emergency vehicle scenario.

    Attributes:
        num_vehicles: Number of emergency vehicles to spawn.
        approach_from_behind: Whether vehicles approach from behind ego.
    """

    num_vehicles: int = 2
    approach_from_behind: bool = True


class EmergencyVehicleScenario(BaseScenario):
    """Spawns emergency vehicles with active lights/sirens.

    This scenario tests:
    - Detection of rare vehicle types (ambulance, fire truck)
    - Handling of flashing lights (visual noise)
    - Tracking of fast-approaching vehicles from behind
    """

    def __init__(self, config: EmergencyVehicleConfig | None = None) -> None:
        super().__init__(config or EmergencyVehicleConfig())
        self._ev_config = config or EmergencyVehicleConfig()

    @property
    def name(self) -> str:
        return "emergency_vehicle"

    @property
    def description(self) -> str:
        return (
            f"Spawns {self._ev_config.num_vehicles} emergency vehicles "
            f"{'approaching from behind' if self._ev_config.approach_from_behind else 'in scene'}"
        )

    def setup(self, world: Any) -> None:
        """Spawn emergency vehicles at spawn points.

        Args:
            world: CARLA world object with spawn_actor() method.
        """
        if not hasattr(world, "get_blueprint_library"):
            return

        bp_lib = world.get_blueprint_library()

        for i in range(self._ev_config.num_vehicles):
            bp_name = EMERGENCY_BLUEPRINTS[i % len(EMERGENCY_BLUEPRINTS)]
            bp = bp_lib.find(bp_name)
            if bp is None:
                continue

            spawn_points = world.get_map().get_spawn_points()
            if spawn_points and i < len(spawn_points):
                actor = world.spawn_actor(bp, spawn_points[i])
                if actor is not None:
                    if hasattr(actor, "set_autopilot"):
                        actor.set_autopilot(True)
                    self._spawned_actors.append(actor)

    def _tick_impl(self, world: Any, frame_id: int) -> None:
        """Emergency vehicles drive on autopilot; no per-tick action needed."""
        pass
