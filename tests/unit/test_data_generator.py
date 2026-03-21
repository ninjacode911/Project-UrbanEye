"""Tests for urbaneye.carla.data_generator module.

All tests use mock objects — no CARLA installation required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from urbaneye.carla.data_generator import CarlaDataGenerator, DatasetStats, FrameData
from urbaneye.carla.sensor_config import SensorSuite, SimulationConfig


@pytest.fixture
def mock_carla_world() -> MagicMock:
    """Create a mock CARLA world with all required methods."""
    world = MagicMock()

    # Settings
    settings = MagicMock()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    world.get_settings.return_value = settings

    # Blueprint library
    bp_lib = MagicMock()
    bp_lib.find.return_value = MagicMock()
    bp_lib.filter.return_value = [MagicMock()]
    world.get_blueprint_library.return_value = bp_lib

    # Map with spawn points
    carla_map = MagicMock()
    carla_map.get_spawn_points.return_value = [MagicMock(), MagicMock()]
    world.get_map.return_value = carla_map

    # Actors
    world.get_actors.return_value = []

    # Spawn actor returns a mock actor
    actor = MagicMock()
    actor.id = 1
    actor.type_id = "vehicle.tesla.model3"
    world.spawn_actor.return_value = actor

    return world


@pytest.fixture
def mock_client(mock_carla_world: MagicMock) -> MagicMock:
    """Create a mock CARLA client."""
    client = MagicMock()
    client.load_world.return_value = mock_carla_world
    return client


@pytest.fixture
def generator(mock_client: MagicMock, tmp_path: Path) -> CarlaDataGenerator:
    """Create a data generator with mock client."""
    return CarlaDataGenerator(
        client=mock_client,
        sensor_suite=SensorSuite(),
        simulation_config=SimulationConfig(),
        output_dir=tmp_path / "dataset",
    )


class TestCarlaDataGenerator:
    """Tests for CarlaDataGenerator class."""

    def test_setup_world_loads_map(
        self, generator: CarlaDataGenerator, mock_client: MagicMock
    ) -> None:
        """setup_world calls client.load_world with correct map name."""
        generator.setup_world("Town01", "ClearNoon")
        mock_client.load_world.assert_called_once_with("Town01")

    def test_setup_world_sets_weather(
        self, generator: CarlaDataGenerator, mock_client: MagicMock
    ) -> None:
        """setup_world sets the weather preset."""
        generator.setup_world("Town01", "HardRainNoon")
        generator.world.set_weather.assert_called_once_with("HardRainNoon")

    def test_setup_world_enables_sync_mode(
        self, generator: CarlaDataGenerator, mock_client: MagicMock
    ) -> None:
        """setup_world enables synchronous mode."""
        generator.setup_world("Town01", "ClearNoon")
        generator.world.get_settings()
        generator.world.apply_settings.assert_called()

    def test_spawn_ego_vehicle(self, generator: CarlaDataGenerator) -> None:
        """spawn_ego_vehicle creates ego vehicle with autopilot."""
        generator.setup_world("Town01", "ClearNoon")
        generator.spawn_ego_vehicle()
        assert generator.ego_vehicle is not None
        generator.ego_vehicle.set_autopilot.assert_called_once_with(True)

    def test_spawn_ego_vehicle_no_autopilot(self, generator: CarlaDataGenerator) -> None:
        """spawn_ego_vehicle can disable autopilot."""
        generator.setup_world("Town01", "ClearNoon")
        generator.spawn_ego_vehicle(autopilot=False)
        generator.ego_vehicle.set_autopilot.assert_not_called()

    def test_attach_sensors_creates_three(self, generator: CarlaDataGenerator) -> None:
        """attach_sensors creates RGB, depth, and semantic sensors."""
        generator.setup_world("Town01", "ClearNoon")
        generator.spawn_ego_vehicle()
        generator.attach_sensors()
        # 3 sensors in the default suite
        assert generator.world.spawn_actor.call_count >= 3

    def test_capture_frame_returns_frame_data(self, generator: CarlaDataGenerator) -> None:
        """capture_frame returns a valid FrameData object."""
        generator.setup_world("Town01", "ClearNoon")
        frame = generator.capture_frame(0)
        assert isinstance(frame, FrameData)
        assert frame.frame_id == 0
        assert isinstance(frame.rgb_image, np.ndarray)

    def test_capture_frame_sequential_ids(self, generator: CarlaDataGenerator) -> None:
        """Frame IDs are sequential."""
        generator.setup_world("Town01", "ClearNoon")
        for i in range(5):
            frame = generator.capture_frame(i)
            assert frame.frame_id == i

    def test_capture_frame_timestamp(self, generator: CarlaDataGenerator) -> None:
        """Timestamp is computed from frame_id * fixed_delta."""
        generator.setup_world("Town01", "ClearNoon")
        frame = generator.capture_frame(20)
        expected_time = 20 * (1.0 / 20.0)
        assert abs(frame.timestamp - expected_time) < 0.001

    def test_generate_dataset_creates_directories(
        self, generator: CarlaDataGenerator, tmp_path: Path
    ) -> None:
        """generate_dataset creates the YOLO directory structure."""
        generator.generate_dataset("Town01", "ClearNoon", num_frames=3)
        output = tmp_path / "dataset"
        assert (output / "images" / "train").is_dir()
        assert (output / "labels" / "train").is_dir()

    def test_generate_dataset_returns_stats(self, generator: CarlaDataGenerator) -> None:
        """generate_dataset returns DatasetStats with correct frame count."""
        stats = generator.generate_dataset("Town01", "ClearNoon", num_frames=5)
        assert isinstance(stats, DatasetStats)
        assert stats.total_frames == 5
        assert stats.map_name == "Town01"
        assert stats.weather == "ClearNoon"

    def test_generate_dataset_creates_label_files(
        self, generator: CarlaDataGenerator, tmp_path: Path
    ) -> None:
        """generate_dataset creates label files for each frame."""
        generator.generate_dataset("Town01", "ClearNoon", num_frames=3)
        labels_dir = tmp_path / "dataset" / "labels" / "train"
        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) == 3

    def test_cleanup_destroys_ego(self, generator: CarlaDataGenerator) -> None:
        """cleanup destroys the ego vehicle."""
        generator.setup_world("Town01", "ClearNoon")
        generator.spawn_ego_vehicle()
        ego = generator.ego_vehicle
        generator.cleanup()
        ego.destroy.assert_called_once()
        assert generator.ego_vehicle is None


class TestFrameData:
    """Tests for FrameData dataclass."""

    def test_default_optional_fields(self) -> None:
        """Optional fields default to None/empty."""
        frame = FrameData(frame_id=0, timestamp=0.0, rgb_image=np.zeros((1, 1, 3)))
        assert frame.depth_map is None
        assert frame.semantic_map is None
        assert frame.raw_bboxes == []

    def test_all_fields(self) -> None:
        """All fields can be populated."""
        frame = FrameData(
            frame_id=42,
            timestamp=2.1,
            rgb_image=np.zeros((1080, 1920, 3), dtype=np.uint8),
            depth_map=np.zeros((480, 640), dtype=np.float32),
            semantic_map=np.zeros((480, 640), dtype=np.uint8),
            raw_bboxes=[{"actor_id": 1, "type_id": "vehicle"}],
        )
        assert frame.frame_id == 42
        assert frame.rgb_image.shape == (1080, 1920, 3)
        assert len(frame.raw_bboxes) == 1


class TestDatasetStats:
    """Tests for DatasetStats dataclass."""

    def test_default_values(self) -> None:
        """Default stats are zero/empty."""
        stats = DatasetStats()
        assert stats.total_frames == 0
        assert stats.class_distribution == {}
