"""Unified tracker interface for ByteTrack and DeepSORT.

DualTracker provides a consistent API regardless of which tracking
backend is used. The evaluation suite and demo app consume TrackedObject
instances without knowing which tracker produced them.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import yaml

from urbaneye.tracking.bytetrack_pipeline import ByteTrackPipeline, STrack
from urbaneye.tracking.deepsort_pipeline import DeepSORTPipeline, TrackedObject
from urbaneye.utils.constants import CLASS_ID_TO_NAME


class TrackerType(Enum):
    """Available tracker backends."""

    BYTETRACK = "bytetrack"
    DEEPSORT = "deepsort"


class DualTracker:
    """Unified interface for both ByteTrack and DeepSORT trackers.

    Provides a consistent update() API that returns TrackedObject instances
    regardless of which backend is active.

    Args:
        tracker_type: Which tracker to use.
        config_path: Optional YAML config for the selected tracker.
    """

    def __init__(
        self,
        tracker_type: TrackerType = TrackerType.BYTETRACK,
        config_path: Path | None = None,
    ) -> None:
        self.tracker_type = tracker_type

        if tracker_type == TrackerType.BYTETRACK:
            self._bytetrack = ByteTrackPipeline(config_path)
            self._deepsort = None
        else:
            self._deepsort = DeepSORTPipeline(config_path)
            self._bytetrack = None

    def update(
        self, detections: np.ndarray, frame: np.ndarray | None = None
    ) -> list[TrackedObject]:
        """Process one frame of detections through the selected tracker.

        Args:
            detections: Nx6 array [x1, y1, x2, y2, confidence, class_id].
            frame: BGR image. Required for DeepSORT, optional for ByteTrack.

        Returns:
            List of TrackedObject instances.

        Raises:
            ValueError: If DeepSORT is active but frame is None.
        """
        if self.tracker_type == TrackerType.BYTETRACK:
            return self._update_bytetrack(detections)
        else:
            if frame is None:
                raise ValueError("DeepSORT requires 'frame' for Re-ID feature extraction")
            return self._deepsort.update(detections, frame)

    def _update_bytetrack(self, detections: np.ndarray) -> list[TrackedObject]:
        """Run ByteTrack and convert STrack → TrackedObject."""
        stracks = self._bytetrack.update(detections)
        return [self._strack_to_tracked_object(s) for s in stracks]

    @staticmethod
    def _strack_to_tracked_object(strack: STrack) -> TrackedObject:
        """Convert ByteTrack's STrack to the unified TrackedObject."""
        return TrackedObject(
            track_id=strack.track_id,
            bbox=strack.bbox,
            confidence=strack.score,
            class_id=strack.class_id,
            class_name=CLASS_ID_TO_NAME.get(strack.class_id, f"cls_{strack.class_id}"),
            is_confirmed=strack.is_confirmed,
            age=strack.hits,
            hits=strack.hits,
        )

    def reset(self) -> None:
        """Reset the active tracker's state."""
        if self._bytetrack is not None:
            self._bytetrack.reset()
        if self._deepsort is not None:
            self._deepsort.reset()

    @classmethod
    def from_config(cls, config_path: Path) -> DualTracker:
        """Create a DualTracker from a YAML config file.

        The config must contain a 'tracker_type' key ("bytetrack" or "deepsort").

        Args:
            config_path: Path to tracker config YAML.

        Returns:
            Configured DualTracker instance.
        """
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        tracker_type_str = cfg.get("tracker_type", "bytetrack")
        tracker_type = TrackerType(tracker_type_str)
        return cls(tracker_type=tracker_type, config_path=config_path)
