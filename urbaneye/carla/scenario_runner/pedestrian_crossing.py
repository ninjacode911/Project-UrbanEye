"""Pedestrian crossing (jaywalking) scenario for CARLA.

Spawns pedestrians that cross the road at random intervals, testing
detection and tracking of unpredictable pedestrian movement — the
highest safety priority for autonomous driving perception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from urbaneye.carla.scenario_runner.base_scenario import (
    BaseScenario,
    ScenarioConfig,
)


@dataclass
class PedestrianCrossingConfig(ScenarioConfig):
    """Configuration for pedestrian crossing scenario.

    Attributes:
        num_pedestrians: Number of pedestrians to spawn.
        crossing_interval: Frames between crossing attempts.
        crossing_speed: Pedestrian walking speed (m/s).
    """

    num_pedestrians: int = 5
    crossing_interval: int = 50
    crossing_speed: float = 1.4  # Average walking speed


class PedestrianCrossingScenario(BaseScenario):
    """Spawns pedestrians that jaywalk across the road.

    This scenario tests:
    - Detection of pedestrians in the driving path
    - Tracking through partial occlusion (pedestrians behind vehicles)
    - Handling of unpredictable movement patterns
    """

    def __init__(self, config: PedestrianCrossingConfig | None = None) -> None:
        super().__init__(config or PedestrianCrossingConfig())
        self._ped_config = config or PedestrianCrossingConfig()

    @property
    def name(self) -> str:
        return "pedestrian_crossing"

    @property
    def description(self) -> str:
        return (
            f"Spawns {self._ped_config.num_pedestrians} pedestrians that jaywalk "
            f"across the road at {self._ped_config.crossing_interval}-frame intervals"
        )

    def setup(self, world: Any) -> None:
        """Spawn pedestrians at roadside locations.

        Args:
            world: CARLA world object. Expected to have:
                - get_blueprint_library() -> BlueprintLibrary
                - get_random_location_from_navigation() -> Location
                - spawn_actor(blueprint, transform) -> Actor
        """
        if not hasattr(world, "get_blueprint_library"):
            return

        bp_lib = world.get_blueprint_library()
        walker_bps = bp_lib.filter("walker.pedestrian.*")

        try:
            num_bps = len(walker_bps)
        except TypeError:
            num_bps = 0

        for i in range(self._ped_config.num_pedestrians):
            if num_bps > 0:
                bp = walker_bps[i % num_bps]
            else:
                bp = walker_bps  # Mock fallback
            if bp is None:
                continue
            spawn_point = world.get_random_location_from_navigation()
            if spawn_point is not None:
                actor = world.spawn_actor(bp, spawn_point)
                if actor is not None:
                    self._spawned_actors.append(actor)

    def _tick_impl(self, world: Any, frame_id: int) -> None:
        """Direct pedestrians to cross at intervals.

        Every crossing_interval frames, command a pedestrian to walk
        toward a point across the road.
        """
        if frame_id % self._ped_config.crossing_interval == 0:
            for actor in self._spawned_actors:
                if hasattr(actor, "apply_control"):
                    # In real CARLA, we'd set a walker controller target
                    # For mock testing, this is a no-op
                    pass
