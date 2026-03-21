"""YOLO annotation export from CARLA bounding box data.

Converts 3D bounding boxes from CARLA world space to 2D YOLO-format
annotations (class_id center_x center_y width height, normalized 0-1).

All functions are pure math — no CARLA dependency required. This enables
complete unit testing without a running CARLA instance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from urbaneye.utils.constants import NUM_CLASSES


def world_to_camera(
    world_point: np.ndarray,
    camera_transform: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> np.ndarray | None:
    """Project a 3D world point onto the 2D camera image plane.

    Uses the standard pinhole camera model:
        pixel = K @ [R|t] @ world_point

    Args:
        world_point: 3D point in world coordinates, shape (3,) or (4,) homogeneous.
        camera_transform: 4x4 camera extrinsic matrix (world-to-camera transform).
        camera_intrinsics: 3x3 camera intrinsic matrix.

    Returns:
        2D pixel coordinates as (u, v) ndarray, or None if the point is behind the camera.
    """
    if world_point.shape[0] == 3:
        world_point = np.append(world_point, 1.0)

    # Transform to camera coordinates
    cam_point = camera_transform @ world_point
    # cam_point: [x_cam, y_cam, z_cam, 1]

    # z_cam is the depth; if negative, point is behind camera
    if cam_point[2] <= 0:
        return None

    # Project to image plane: pixel = K @ [x/z, y/z, 1]
    projected = camera_intrinsics @ cam_point[:3]
    u = projected[0] / projected[2]
    v = projected[1] / projected[2]

    return np.array([u, v])


def build_camera_intrinsics(width: int, height: int, fov_degrees: float) -> np.ndarray:
    """Build the 3x3 camera intrinsic matrix from sensor parameters.

    Uses the standard pinhole camera model where focal length is derived
    from the field of view.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fov_degrees: Horizontal field of view in degrees.

    Returns:
        3x3 intrinsic matrix K.
    """
    fov_rad = np.deg2rad(fov_degrees)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # Square pixels
    cx = width / 2.0
    cy = height / 2.0

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def bbox_3d_to_2d(
    corners_3d: np.ndarray,
    camera_transform: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Project 3D bounding box corners to 2D pixel bounding box.

    Args:
        corners_3d: 8x3 array of 3D bounding box corner coordinates.
        camera_transform: 4x4 world-to-camera transformation matrix.
        camera_intrinsics: 3x3 camera intrinsic matrix.

    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates, or None if
        all corners are behind the camera.
    """
    projected_points: list[np.ndarray] = []

    for corner in corners_3d:
        pixel = world_to_camera(corner, camera_transform, camera_intrinsics)
        if pixel is not None:
            projected_points.append(pixel)

    if len(projected_points) == 0:
        return None

    points = np.array(projected_points)
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))

    return (x_min, y_min, x_max, y_max)


def pixel_bbox_to_yolo(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    class_id: int,
    img_width: int,
    img_height: int,
) -> str | None:
    """Convert pixel bounding box to YOLO format annotation string.

    YOLO format: class_id center_x center_y width height (all normalized 0-1).

    Args:
        x_min, y_min, x_max, y_max: Pixel bounding box coordinates.
        class_id: Integer class ID (0-indexed).
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        YOLO annotation string, or None if the bbox is invalid.
    """
    # Clamp to image boundaries
    x_min = max(0.0, min(x_min, img_width))
    y_min = max(0.0, min(y_min, img_height))
    x_max = max(0.0, min(x_max, img_width))
    y_max = max(0.0, min(y_max, img_height))

    # Compute normalized YOLO coordinates
    cx = (x_min + x_max) / 2.0 / img_width
    cy = (y_min + y_max) / 2.0 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height

    # Filter invalid boxes
    if w <= 0 or h <= 0:
        return None

    # Filter boxes that are mostly outside the frame
    if cx <= 0 or cx >= 1 or cy <= 0 or cy >= 1:
        return None

    # Filter very small boxes (noise)
    if w < 0.001 or h < 0.001:
        return None

    # Validate class ID
    if class_id < 0 or class_id >= NUM_CLASSES:
        return None

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def carla_bbox_to_yolo(
    corners_3d: np.ndarray,
    camera_transform: np.ndarray,
    camera_intrinsics: np.ndarray,
    class_id: int,
    img_width: int,
    img_height: int,
) -> str | None:
    """Full pipeline: 3D CARLA bbox → 2D projection → YOLO annotation.

    Args:
        corners_3d: 8x3 array of 3D bounding box corners.
        camera_transform: 4x4 world-to-camera transformation matrix.
        camera_intrinsics: 3x3 camera intrinsic matrix.
        class_id: Integer class ID.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        YOLO annotation string, or None if the object is not visible.
    """
    bbox_2d = bbox_3d_to_2d(corners_3d, camera_transform, camera_intrinsics)
    if bbox_2d is None:
        return None

    x_min, y_min, x_max, y_max = bbox_2d
    return pixel_bbox_to_yolo(x_min, y_min, x_max, y_max, class_id, img_width, img_height)


def export_frame_annotations(
    annotations: list[str],
    output_path: Path,
) -> Path:
    """Write YOLO annotations to a label file.

    Args:
        annotations: List of YOLO annotation strings (one per line).
        output_path: Path to write the .txt label file.

    Returns:
        The output path (for chaining).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(annotations))
    return output_path


def validate_yolo_annotation(line: str) -> bool:
    """Validate a single YOLO annotation line.

    Expected format: class_id center_x center_y width height
    All values normalized to [0, 1] except class_id (integer >= 0).

    Args:
        line: A single line from a YOLO label file.

    Returns:
        True if the annotation is valid.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return False

    try:
        class_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return False

    if class_id < 0 or class_id >= NUM_CLASSES:
        return False

    for val in (cx, cy, w, h):
        if val < 0 or val > 1:
            return False

    if w <= 0 or h <= 0:
        return False

    return True


def load_annotations(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO annotations from a label file.

    Args:
        label_path: Path to the .txt label file.

    Returns:
        List of (class_id, cx, cy, w, h) tuples.
    """
    annotations: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return annotations

    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if validate_yolo_annotation(line):
                parts = line.split()
                annotations.append(
                    (
                        int(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                )
    return annotations
