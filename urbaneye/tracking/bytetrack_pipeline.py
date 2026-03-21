"""ByteTrack multi-object tracker — implemented from scratch.

ByteTrack's key innovation (ECCV 2022): associate BOTH high-confidence
and low-confidence detections. Low-confidence detections are often partially
occluded objects that naive trackers would miss.

Two-stage matching:
  Stage 1: High-confidence detections ↔ active tracks (IoU + Hungarian)
  Stage 2: Low-confidence detections ↔ remaining unmatched tracks

Only depends on numpy, scipy (Hungarian Algorithm). No external tracking library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

from urbaneye.tracking.kalman_filter import KalmanFilter
from urbaneye.tracking.utils import (
    compute_iou_matrix,
    linear_assignment_solve,
    tlbr_to_tlwh,
    tlwh_to_xyah,
    xyah_to_tlbr,
)


class TrackState(Enum):
    """Lifecycle state of a tracked object."""

    NEW = 0
    ACTIVE = 1
    LOST = 2
    DELETED = 3


@dataclass
class STrack:
    """Single tracked object (Short-lived Track).

    Attributes:
        track_id: Unique track identifier.
        bbox: Current bounding box [x1, y1, x2, y2].
        score: Detection confidence score.
        class_id: Object class ID.
        state: Current lifecycle state.
        mean: Kalman filter state vector (8D).
        covariance: Kalman filter covariance matrix (8x8).
        start_frame: Frame when track was first created.
        hits: Number of times matched with a detection.
        time_since_update: Frames since last detection match.
    """

    track_id: int
    bbox: np.ndarray
    score: float
    class_id: int
    state: TrackState = TrackState.NEW
    mean: np.ndarray = field(default_factory=lambda: np.zeros(8))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8))
    start_frame: int = 0
    hits: int = 1
    time_since_update: int = 0

    def predict(self, kf: KalmanFilter) -> None:
        """Predict next state using Kalman Filter."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, detection: np.ndarray, score: float, class_id: int, kf: KalmanFilter) -> None:
        """Update track with a matched detection.

        Args:
            detection: Detected bbox [x1, y1, x2, y2].
            score: Detection confidence.
            class_id: Detection class.
            kf: Kalman filter instance.
        """
        self.bbox = detection
        self.score = score
        self.class_id = class_id

        # Convert bbox to measurement format [cx, cy, a, h]
        tlwh = tlbr_to_tlwh(detection)
        measurement = tlwh_to_xyah(tlwh)
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)

        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.NEW and self.hits >= 3:
            self.state = TrackState.ACTIVE
        elif self.state == TrackState.LOST:
            self.state = TrackState.ACTIVE

    def to_tlbr(self) -> np.ndarray:
        """Get current bbox in [x1, y1, x2, y2] format from Kalman state."""
        return xyah_to_tlbr(self.mean[:4])

    @property
    def is_confirmed(self) -> bool:
        """Whether this track has been confirmed (enough hits)."""
        return self.state == TrackState.ACTIVE


class ByteTrackPipeline:
    """ByteTrack multi-object tracker with two-stage association.

    Stage 1: Match high-confidence detections (>= high_thresh) to
             existing tracks using IoU distance + Hungarian Algorithm.
    Stage 2: Match remaining low-confidence detections (>= low_thresh)
             to unmatched tracks from Stage 1.

    Args:
        config_path: Optional YAML config file for hyperparameters.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self.high_thresh: float = 0.6
        self.low_thresh: float = 0.1
        self.match_thresh: float = 0.8
        self.max_lost: int = 30
        self.min_hits: int = 3

        if config_path is not None:
            self._load_config(config_path)

        self.kf = KalmanFilter()
        self.tracks: list[STrack] = []
        self._track_id_count: int = 0
        self.frame_id: int = 0

    def _load_config(self, path: Path) -> None:
        """Load tracker hyperparameters from YAML."""
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.high_thresh = cfg.get("high_thresh", self.high_thresh)
        self.low_thresh = cfg.get("low_thresh", self.low_thresh)
        self.match_thresh = cfg.get("match_thresh", self.match_thresh)
        self.max_lost = cfg.get("max_lost", self.max_lost)
        self.min_hits = cfg.get("min_hits", self.min_hits)

    def _next_id(self) -> int:
        """Get next unique track ID."""
        self._track_id_count += 1
        return self._track_id_count

    def update(self, detections: np.ndarray) -> list[STrack]:
        """Process one frame of detections.

        This is the core ByteTrack algorithm:
        1. Split detections into high and low confidence
        2. Predict all existing tracks forward
        3. First association: high-confidence ↔ active tracks
        4. Second association: low-confidence ↔ unmatched tracks
        5. Handle unmatched tracks (mark lost or delete)
        6. Create new tracks from unmatched high-confidence detections
        7. Return confirmed active tracks

        Args:
            detections: Nx6 array [x1, y1, x2, y2, confidence, class_id].
                        Can be empty (Nx6 with N=0).

        Returns:
            List of active, confirmed tracks.
        """
        self.frame_id += 1

        # Handle empty detections
        if len(detections) == 0:
            for track in self.tracks:
                track.predict(self.kf)  # predict() already increments time_since_update
                if track.state == TrackState.ACTIVE:
                    track.state = TrackState.LOST
                if track.state == TrackState.LOST and track.time_since_update > self.max_lost:
                    track.state = TrackState.DELETED
            self._remove_deleted_tracks()
            return self.active_tracks

        # Split detections by confidence
        scores = detections[:, 4]
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (scores < self.high_thresh)

        high_dets = detections[high_mask]
        low_dets = detections[low_mask]

        # Predict all tracks forward
        for track in self.tracks:
            track.predict(self.kf)

        # Separate active and lost tracks
        active_tracks = [t for t in self.tracks if t.state in (TrackState.NEW, TrackState.ACTIVE)]
        lost_tracks = [t for t in self.tracks if t.state == TrackState.LOST]

        # ---- STAGE 1: High-confidence ↔ Active tracks ----
        matches_1, unmatched_tracks_1, unmatched_dets_1 = self._associate(active_tracks, high_dets)

        # Update matched tracks
        for track_idx, det_idx in matches_1:
            track = active_tracks[track_idx]
            det = high_dets[det_idx]
            track.update(det[:4], det[4], int(det[5]), self.kf)

        # ---- STAGE 2: Low-confidence ↔ Remaining unmatched tracks ----
        remaining_tracks = [active_tracks[i] for i in unmatched_tracks_1]
        matches_2, unmatched_tracks_2, _ = self._associate(remaining_tracks, low_dets)

        # Update matched tracks from stage 2
        for track_idx, det_idx in matches_2:
            track = remaining_tracks[track_idx]
            det = low_dets[det_idx]
            track.update(det[:4], det[4], int(det[5]), self.kf)

        # ---- Handle unmatched tracks ----
        for idx in unmatched_tracks_2:
            track = remaining_tracks[idx]
            if track.state == TrackState.ACTIVE:
                track.state = TrackState.LOST

        # Also try to match lost tracks with unmatched high-conf detections
        unmatched_high_dets = (
            high_dets[list(unmatched_dets_1)] if unmatched_dets_1 else np.empty((0, 6))
        )
        if len(lost_tracks) > 0 and len(unmatched_high_dets) > 0:
            matches_lost, _, remaining_det_indices = self._associate(
                lost_tracks, unmatched_high_dets
            )
            for track_idx, det_idx in matches_lost:
                track = lost_tracks[track_idx]
                det = unmatched_high_dets[det_idx]
                track.update(det[:4], det[4], int(det[5]), self.kf)
            # Update unmatched_dets_1 to only contain truly unmatched
            unmatched_dets_final = [unmatched_dets_1[i] for i in remaining_det_indices]
        else:
            unmatched_dets_final = list(unmatched_dets_1)

        # Mark old lost tracks as deleted
        for track in self.tracks:
            if track.state == TrackState.LOST and track.time_since_update > self.max_lost:
                track.state = TrackState.DELETED

        # ---- Create new tracks from unmatched high-confidence detections ----
        for det_idx in unmatched_dets_final:
            det = high_dets[det_idx]
            new_track = self._init_track(det)
            self.tracks.append(new_track)

        # Clean up deleted tracks
        self._remove_deleted_tracks()

        return self.active_tracks

    def _associate(
        self, tracks: list[STrack], detections: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate tracks with detections using IoU + Hungarian Algorithm.

        Args:
            tracks: List of existing tracks.
            detections: Nx6 detection array.

        Returns:
            Tuple of (matches, unmatched_track_indices, unmatched_det_indices).
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Compute IoU cost matrix
        track_bboxes = np.array([t.to_tlbr() for t in tracks])
        det_bboxes = detections[:, :4]
        iou_matrix = compute_iou_matrix(track_bboxes, det_bboxes)

        # Convert IoU to cost (1 - IoU)
        cost_matrix = 1.0 - iou_matrix

        return linear_assignment_solve(cost_matrix, threshold=1.0 - self.match_thresh + 1.0)

    def _init_track(self, detection: np.ndarray) -> STrack:
        """Initialize a new track from a detection.

        Args:
            detection: [x1, y1, x2, y2, confidence, class_id].

        Returns:
            New STrack instance.
        """
        bbox = detection[:4]
        tlwh = tlbr_to_tlwh(bbox)
        measurement = tlwh_to_xyah(tlwh)
        mean, covariance = self.kf.initiate(measurement)

        return STrack(
            track_id=self._next_id(),
            bbox=bbox,
            score=float(detection[4]),
            class_id=int(detection[5]),
            state=TrackState.NEW,
            mean=mean,
            covariance=covariance,
            start_frame=self.frame_id,
        )

    def _remove_deleted_tracks(self) -> None:
        """Remove tracks marked as DELETED."""
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

    @property
    def active_tracks(self) -> list[STrack]:
        """Return only confirmed, active tracks."""
        return [t for t in self.tracks if t.state == TrackState.ACTIVE]

    def reset(self) -> None:
        """Clear all tracks and reset state."""
        self.tracks.clear()
        self._track_id_count = 0
        self.frame_id = 0
