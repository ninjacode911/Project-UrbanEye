"""Tests for urbaneye.carla.annotation_exporter module.

All tests use synthetic data — no CARLA import required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from urbaneye.carla.annotation_exporter import (
    bbox_3d_to_2d,
    build_camera_intrinsics,
    carla_bbox_to_yolo,
    export_frame_annotations,
    load_annotations,
    pixel_bbox_to_yolo,
    validate_yolo_annotation,
    world_to_camera,
)


class TestBuildCameraIntrinsics:
    """Tests for build_camera_intrinsics function."""

    def test_shape_is_3x3(self) -> None:
        """Intrinsic matrix is 3x3."""
        K = build_camera_intrinsics(1920, 1080, 90.0)
        assert K.shape == (3, 3)

    def test_principal_point_at_center(self) -> None:
        """Principal point (cx, cy) is at image center."""
        K = build_camera_intrinsics(1920, 1080, 90.0)
        assert K[0, 2] == 1920 / 2.0
        assert K[1, 2] == 1080 / 2.0

    def test_focal_length_positive(self) -> None:
        """Focal length fx and fy are positive."""
        K = build_camera_intrinsics(640, 480, 90.0)
        assert K[0, 0] > 0
        assert K[1, 1] > 0

    def test_square_pixels(self) -> None:
        """fx equals fy (square pixels)."""
        K = build_camera_intrinsics(1920, 1080, 90.0)
        assert abs(K[0, 0] - K[1, 1]) < 1e-10

    def test_wider_fov_means_shorter_focal(self) -> None:
        """Wider FOV produces smaller focal length."""
        K_narrow = build_camera_intrinsics(640, 480, 60.0)
        K_wide = build_camera_intrinsics(640, 480, 120.0)
        assert K_narrow[0, 0] > K_wide[0, 0]


class TestWorldToCamera:
    """Tests for world_to_camera function."""

    def test_point_in_front_of_camera(self) -> None:
        """Point in front of camera projects to valid pixel coordinates."""
        # Identity camera transform (camera at origin, looking down +Z)
        cam_transform = np.eye(4)
        K = build_camera_intrinsics(640, 480, 90.0)
        # Point at (0, 0, 5) — directly in front
        point = np.array([0.0, 0.0, 5.0])
        pixel = world_to_camera(point, cam_transform, K)
        assert pixel is not None
        # Should project near image center
        assert abs(pixel[0] - 320) < 1.0
        assert abs(pixel[1] - 240) < 1.0

    def test_point_behind_camera_returns_none(self) -> None:
        """Point behind camera returns None."""
        cam_transform = np.eye(4)
        K = build_camera_intrinsics(640, 480, 90.0)
        # Point at (0, 0, -5) — behind camera
        point = np.array([0.0, 0.0, -5.0])
        pixel = world_to_camera(point, cam_transform, K)
        assert pixel is None

    def test_accepts_homogeneous_coordinates(self) -> None:
        """Function accepts 4D homogeneous coordinates."""
        cam_transform = np.eye(4)
        K = build_camera_intrinsics(640, 480, 90.0)
        point = np.array([0.0, 0.0, 5.0, 1.0])
        pixel = world_to_camera(point, cam_transform, K)
        assert pixel is not None


class TestBbox3dTo2d:
    """Tests for bbox_3d_to_2d function."""

    def _make_cube_corners(self, center: tuple[float, float, float], size: float) -> np.ndarray:
        """Helper: create 8 corners of a cube."""
        cx, cy, cz = center
        half = size / 2
        corners = []
        for dx in (-half, half):
            for dy in (-half, half):
                for dz in (-half, half):
                    corners.append([cx + dx, cy + dy, cz + dz])
        return np.array(corners)

    def test_visible_box_returns_bbox(self) -> None:
        """Visible box returns valid 2D bounding box."""
        corners = self._make_cube_corners((0, 0, 10), 2.0)
        cam_transform = np.eye(4)
        K = build_camera_intrinsics(640, 480, 90.0)
        result = bbox_3d_to_2d(corners, cam_transform, K)
        assert result is not None
        x_min, y_min, x_max, y_max = result
        assert x_min < x_max
        assert y_min < y_max

    def test_behind_camera_returns_none(self) -> None:
        """Box entirely behind camera returns None."""
        corners = self._make_cube_corners((0, 0, -10), 2.0)
        cam_transform = np.eye(4)
        K = build_camera_intrinsics(640, 480, 90.0)
        result = bbox_3d_to_2d(corners, cam_transform, K)
        assert result is None


class TestPixelBboxToYolo:
    """Tests for pixel_bbox_to_yolo function."""

    def test_center_box_valid(self) -> None:
        """Box in the center of the image produces valid YOLO annotation."""
        result = pixel_bbox_to_yolo(100, 100, 200, 300, 0, 640, 480)
        assert result is not None
        parts = result.split()
        assert len(parts) == 5
        assert parts[0] == "0"
        cx, cy, w, h = [float(p) for p in parts[1:]]
        assert 0 < cx < 1
        assert 0 < cy < 1
        assert 0 < w < 1
        assert 0 < h < 1

    def test_zero_area_returns_none(self) -> None:
        """Zero-area box (same min/max) returns None."""
        result = pixel_bbox_to_yolo(100, 100, 100, 100, 0, 640, 480)
        assert result is None

    def test_negative_area_returns_none(self) -> None:
        """Inverted box (max < min) returns None after clamping."""
        result = pixel_bbox_to_yolo(200, 200, 100, 100, 0, 640, 480)
        assert result is None

    def test_invalid_class_id_returns_none(self) -> None:
        """Class ID outside valid range returns None."""
        assert pixel_bbox_to_yolo(100, 100, 200, 200, -1, 640, 480) is None
        assert pixel_bbox_to_yolo(100, 100, 200, 200, 99, 640, 480) is None

    def test_tiny_box_returns_none(self) -> None:
        """Very small box (< 0.001 normalized) returns None."""
        # Box width = 0.1 pixel on a 640-wide image = 0.1/640 < 0.001
        result = pixel_bbox_to_yolo(100, 100, 100.05, 100.05, 0, 640, 480)
        assert result is None

    def test_box_at_edge_clamped(self) -> None:
        """Box extending beyond image edge is clamped."""
        result = pixel_bbox_to_yolo(-50, -50, 100, 100, 0, 640, 480)
        assert result is not None
        parts = result.split()
        cx = float(parts[1])
        assert 0 < cx < 1


class TestCaralBboxToYolo:
    """Tests for the full carla_bbox_to_yolo pipeline."""

    def test_visible_box_produces_annotation(self) -> None:
        """Visible 3D box produces valid YOLO annotation."""
        # Create a simple cube in front of the camera
        half = 1.0
        corners = []
        for dx in (-half, half):
            for dy in (-half, half):
                for dz in (-half, half):
                    corners.append([dx, dy, 10.0 + dz])
        corners = np.array(corners)

        cam_transform = np.eye(4)
        K = build_camera_intrinsics(640, 480, 90.0)

        result = carla_bbox_to_yolo(corners, cam_transform, K, 0, 640, 480)
        assert result is not None
        assert validate_yolo_annotation(result)


class TestValidateYoloAnnotation:
    """Tests for validate_yolo_annotation function."""

    def test_valid_annotation(self) -> None:
        """Valid annotation passes validation."""
        assert validate_yolo_annotation("0 0.500000 0.500000 0.200000 0.300000") is True

    def test_all_classes_valid(self) -> None:
        """All 5 class IDs are valid."""
        for cls in range(5):
            assert validate_yolo_annotation(f"{cls} 0.5 0.5 0.2 0.3") is True

    def test_invalid_class_id(self) -> None:
        """Class ID out of range fails."""
        assert validate_yolo_annotation("5 0.5 0.5 0.2 0.3") is False
        assert validate_yolo_annotation("-1 0.5 0.5 0.2 0.3") is False

    def test_coords_out_of_range(self) -> None:
        """Coordinates > 1.0 fail."""
        assert validate_yolo_annotation("0 1.5 0.5 0.2 0.3") is False
        assert validate_yolo_annotation("0 0.5 -0.1 0.2 0.3") is False

    def test_zero_dimensions(self) -> None:
        """Zero width or height fails."""
        assert validate_yolo_annotation("0 0.5 0.5 0.0 0.3") is False
        assert validate_yolo_annotation("0 0.5 0.5 0.2 0.0") is False

    def test_wrong_field_count(self) -> None:
        """Wrong number of fields fails."""
        assert validate_yolo_annotation("0 0.5 0.5 0.2") is False
        assert validate_yolo_annotation("0 0.5 0.5 0.2 0.3 extra") is False

    def test_non_numeric_fails(self) -> None:
        """Non-numeric values fail."""
        assert validate_yolo_annotation("abc 0.5 0.5 0.2 0.3") is False


class TestExportFrameAnnotations:
    """Tests for export_frame_annotations function."""

    def test_writes_file(self, tmp_path: Path) -> None:
        """Annotations are written to the correct file."""
        annotations = [
            "0 0.500000 0.500000 0.200000 0.300000",
            "1 0.300000 0.700000 0.100000 0.250000",
        ]
        out = tmp_path / "labels" / "frame_000001.txt"
        result = export_frame_annotations(annotations, out)
        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "0 0.500000" in content
        assert "1 0.300000" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created automatically."""
        out = tmp_path / "deep" / "nested" / "label.txt"
        export_frame_annotations(["0 0.5 0.5 0.1 0.1"], out)
        assert out.exists()

    def test_empty_annotations(self, tmp_path: Path) -> None:
        """Empty annotation list creates empty file."""
        out = tmp_path / "empty.txt"
        export_frame_annotations([], out)
        assert out.exists()
        assert out.read_text(encoding="utf-8") == ""


class TestLoadAnnotations:
    """Tests for load_annotations function."""

    def test_load_valid_file(self, tmp_path: Path) -> None:
        """Valid label file is parsed correctly."""
        label = tmp_path / "test.txt"
        label.write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.25\n", encoding="utf-8")
        result = load_annotations(label)
        assert len(result) == 2
        assert result[0] == (0, 0.5, 0.5, 0.2, 0.3)
        assert result[1] == (1, 0.3, 0.7, 0.1, 0.25)

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Missing file returns empty list."""
        result = load_annotations(tmp_path / "missing.txt")
        assert result == []

    def test_skips_invalid_lines(self, tmp_path: Path) -> None:
        """Invalid lines are skipped."""
        label = tmp_path / "mixed.txt"
        label.write_text("0 0.5 0.5 0.2 0.3\ninvalid line\n1 0.3 0.7 0.1 0.25\n", encoding="utf-8")
        result = load_annotations(label)
        assert len(result) == 2
