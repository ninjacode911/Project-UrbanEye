"""Abstract base class for CARLA scenario runner scripts.

All scenarios implement this interface, enabling composable execution
during data generation. Multiple scenarios can run simultaneously.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ScenarioState(Enum):
    """Current state of a running scenario."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ScenarioConfig:
    """Base configuration for scenarios.

    Attributes:
        duration_frames: How many frames the scenario should run.
        repeat: Whether the scenario should loop after completion.
    """

    duration_frames: int = 200
    repeat: bool = False


class BaseScenario(ABC):
    """Abstract base class for all CARLA scenarios.

    Scenarios follow a lifecycle: setup → tick (repeated) → cleanup.
    The CARLA world object is injected (dependency injection) to enable
    mock-based testing without a running CARLA instance.
    """

    def __init__(self, config: ScenarioConfig | None = None) -> None:
        self._config = config or ScenarioConfig()
        self._state = ScenarioState.PENDING
        self._frame_count: int = 0
        self._spawned_actors: list[Any] = []

    @property
    def state(self) -> ScenarioState:
        """Current scenario state."""
        return self._state

    @property
    def frame_count(self) -> int:
        """Number of frames since scenario started."""
        return self._frame_count

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scenario name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this scenario tests."""
        ...

    @abstractmethod
    def setup(self, world: Any) -> None:
        """Initialize the scenario: spawn actors, configure environment.

        Args:
            world: CARLA world object (or mock for testing).
        """
        ...

    @abstractmethod
    def _tick_impl(self, world: Any, frame_id: int) -> None:
        """Scenario-specific tick logic. Override in subclasses.

        Args:
            world: CARLA world object.
            frame_id: Current frame number.
        """
        ...

    def tick(self, world: Any, frame_id: int) -> ScenarioState:
        """Advance the scenario by one frame.

        Handles state transitions and calls _tick_impl for scenario-specific logic.

        Args:
            world: CARLA world object.
            frame_id: Current frame number.

        Returns:
            Current scenario state after this tick.
        """
        if self._state == ScenarioState.PENDING:
            self.setup(world)
            self._state = ScenarioState.ACTIVE

        if self._state == ScenarioState.ACTIVE:
            self._tick_impl(world, frame_id)
            self._frame_count += 1

            if self._frame_count >= self._config.duration_frames:
                if self._config.repeat:
                    self._frame_count = 0
                else:
                    self._state = ScenarioState.COMPLETED

        return self._state

    def cleanup(self, world: Any) -> None:
        """Remove all scenario-spawned actors and reset state.

        Args:
            world: CARLA world object.
        """
        for actor in self._spawned_actors:
            if hasattr(actor, "destroy"):
                actor.destroy()
        self._spawned_actors.clear()
        self._state = ScenarioState.COMPLETED
