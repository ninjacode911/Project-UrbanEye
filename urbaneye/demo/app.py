"""UrbanEye Gradio Demo — Development version.

Interactive demo for the autonomous driving perception pipeline.
Upload a video or select a sample clip, choose a tracker, adjust
confidence, and view annotated results with metrics.

This is the development version that imports from the urbaneye package.
The hf_space/app.py is the self-contained HuggingFace Spaces version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from urbaneye.tracking.deepsort_pipeline import TrackedObject
from urbaneye.tracking.dual_tracker import DualTracker, TrackerType
from urbaneye.utils.constants import (
    CLASS_COLORS,
    CLASS_NAMES,
)


@dataclass
class ProcessingResult:
    """Result of processing a video through the pipeline."""

    output_path: str = ""
    total_frames: int = 0
    total_detections: int = 0
    total_tracks: int = 0
    avg_fps: float = 0.0
    class_counts: dict[str, int] = field(default_factory=dict)
    id_switch_estimate: int = 0


def annotate_frame(
    frame: np.ndarray,
    tracks: list[TrackedObject],
    fps: float = 0.0,
) -> np.ndarray:
    """Draw tracking annotations on a single frame.

    Args:
        frame: BGR image.
        tracks: List of tracked objects to draw.
        fps: Current processing FPS for overlay.

    Returns:
        Annotated frame copy.
    """
    annotated = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.bbox[:4]]
        color = CLASS_COLORS.get(track.class_name, (255, 255, 255))

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label: class + track ID + confidence
        label = f"{track.class_name} #{track.track_id} {track.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_h - 6), (x1 + label_w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # FPS overlay
    if fps > 0:
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

    # Track count overlay
    cv2.putText(
        annotated,
        f"Tracks: {len(tracks)}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    return annotated


def process_video(
    video_path: str,
    tracker_type: str = "ByteTrack",
    confidence: float = 0.25,
    selected_classes: list[str] | None = None,
    model_path: str | None = None,
) -> ProcessingResult:
    """Process a video through the detection + tracking pipeline.

    Args:
        video_path: Path to input video file.
        tracker_type: "ByteTrack" or "DeepSORT".
        confidence: Detection confidence threshold.
        selected_classes: Classes to detect (None = all).
        model_path: Path to ONNX or .pt model weights.

    Returns:
        ProcessingResult with output path and metrics.
    """
    if selected_classes is None:
        selected_classes = list(CLASS_NAMES)

    result = ProcessingResult()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return result

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Setup output
    output_path = str(Path(video_path).with_suffix(".out.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    # Setup tracker
    tt = TrackerType.DEEPSORT if tracker_type == "DeepSORT" else TrackerType.BYTETRACK
    tracker = DualTracker(tt)

    class_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # In production: run YOLO inference here
        # For now, tracker processes empty detections (demo skeleton)
        detections = np.empty((0, 6), dtype=np.float32)

        # Run tracker
        tracks = tracker.update(detections, frame=frame if tt == TrackerType.DEEPSORT else None)

        # Filter by selected classes
        tracks = [t for t in tracks if t.class_name in selected_classes]

        # Count per-class detections
        for t in tracks:
            class_counts[t.class_name] = class_counts.get(t.class_name, 0) + 1

        # Annotate and write
        annotated = annotate_frame(frame, tracks, fps=fps_in)
        writer.write(annotated)

        frame_count += 1
        result.total_detections += len(tracks)

    cap.release()
    writer.release()

    result.output_path = output_path
    result.total_frames = frame_count
    result.class_counts = class_counts
    result.avg_fps = fps_in

    return result
