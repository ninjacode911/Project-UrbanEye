"""DeepSORT tracker with appearance-based re-identification.

Uses the deep-sort-realtime library with MobileNetV2 Re-ID features.
Maintains identity through long occlusions via 128-dim appearance embeddings.

Lazy-initialized to avoid model download at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from urbaneye.utils.constants import CLASS_ID_TO_NAME


@dataclass
class TrackedObject:
    """Unified tracked object representation for both trackers.

    This is the common output format produced by both ByteTrack and DeepSORT
    through the DualTracker interface.
    """

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    is_confirmed: bool
    age: int = 0
    hits: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "track_id": int(self.track_id),
            "bbox_xyxy": self.bbox.tolist()
            if isinstance(self.bbox, np.ndarray)
            else list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "is_confirmed": bool(self.is_confirmed),
            "age": int(self.age),
            "hits": int(self.hits),
        }


class DeepSORTPipeline:
    """DeepSORT tracker with appearance-based re-identification.

    Uses deep-sort-realtime library with custom configuration.
    Re-ID model: MobileNetV2 producing 128-dim embeddings.
    Matching: Cascade matching with cosine distance + Mahalanobis gating.

    The tracker is lazy-initialized: the deep-sort-realtime library and
    Re-ID model are only loaded when update() is first called. This prevents
    heavy imports and model downloads at module import time.

    Args:
        config_path: Optional YAML config for hyperparameters.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self.max_age: int = 70
        self.n_init: int = 3
        self.max_cosine_distance: float = 0.3
        self.nn_budget: int = 100
        self.embedder: str = "mobilenet"
        self.half: bool = True

        if config_path is not None:
            self._load_config(config_path)

        self._tracker: Any = None  # Lazy init
        self._initialized: bool = False

    def _load_config(self, path: Path) -> None:
        """Load tracker hyperparameters from YAML."""
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.max_age = cfg.get("max_age", self.max_age)
        self.n_init = cfg.get("n_init", self.n_init)
        self.max_cosine_distance = cfg.get("max_cosine_distance", self.max_cosine_distance)
        self.nn_budget = cfg.get("nn_budget", self.nn_budget)
        self.embedder = cfg.get("embedder", self.embedder)

    def _init_tracker(self) -> None:
        """Initialize deep-sort-realtime tracker (lazy, on first update)."""
        if self._initialized:
            return
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            self._tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget,
                embedder=self.embedder,
                half=self.half,
            )
            self._initialized = True
        except ImportError:
            raise ImportError(
                "deep-sort-realtime is required for DeepSORT. "
                "Install with: pip install deep-sort-realtime"
            )

    def update(
        self, detections: np.ndarray, frame: np.ndarray | None = None
    ) -> list[TrackedObject]:
        """Process one frame of detections.

        Args:
            detections: Nx6 array [x1, y1, x2, y2, confidence, class_id].
            frame: BGR image for Re-ID feature extraction. Required for DeepSORT.

        Returns:
            List of TrackedObject with persistent IDs.

        Raises:
            ValueError: If frame is None (required for appearance extraction).
        """
        if frame is None:
            raise ValueError("DeepSORT requires 'frame' argument for Re-ID feature extraction")

        self._init_tracker()

        if len(detections) == 0:
            self._tracker.update_tracks([], frame=frame)
            return self._get_tracked_objects()

        # Convert to deep-sort-realtime format: list of ([x1,y1,w,h], confidence, class_name)
        ds_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            w = x2 - x1
            h = y2 - y1
            cls_name = CLASS_ID_TO_NAME.get(int(cls_id), f"cls_{int(cls_id)}")
            ds_detections.append(([x1, y1, w, h], float(conf), cls_name))

        # Update tracker
        self._tracker.update_tracks(ds_detections, frame=frame)
        return self._get_tracked_objects()

    def _get_tracked_objects(self) -> list[TrackedObject]:
        """Extract TrackedObject list from internal tracker state."""
        objects: list[TrackedObject] = []
        if self._tracker is None:
            return objects

        for track in self._tracker.tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            cls_name = track.get_det_class() or "unknown"

            # Reverse lookup class ID from name
            from urbaneye.utils.constants import CLASS_NAME_TO_ID

            cls_id = CLASS_NAME_TO_ID.get(cls_name, -1)

            objects.append(
                TrackedObject(
                    track_id=track.track_id,
                    bbox=np.array(ltrb),
                    confidence=track.get_det_conf() or 0.0,
                    class_id=cls_id,
                    class_name=cls_name,
                    is_confirmed=True,
                    age=track.age,
                    hits=track.hits,
                )
            )
        return objects

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker = None
        self._initialized = False
