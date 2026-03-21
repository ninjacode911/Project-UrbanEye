"""Tests for urbaneye.utils.io_helpers module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from urbaneye.utils.io_helpers import (
    ensure_dir,
    get_project_root,
    load_yaml,
    validate_image_path,
)


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_load_valid_yaml(self, sample_yaml_file: Path) -> None:
        """load_yaml returns correct dict for valid YAML."""
        result = load_yaml(sample_yaml_file)
        assert result == {"key": "value", "nested": {"inner": 42}}

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """load_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="YAML file not found"):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_load_malformed_yaml_raises(self, tmp_path: Path) -> None:
        """load_yaml raises YAMLError for invalid YAML content."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{{invalid: yaml: [}", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_yaml(bad_yaml)

    def test_load_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """load_yaml returns empty dict for empty YAML file."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        result = load_yaml(empty)
        assert result == {}

    def test_load_yaml_with_list_raises(self, tmp_path: Path) -> None:
        """load_yaml raises YAMLError when YAML contains a list, not a dict."""
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError, match="Expected a YAML mapping"):
            load_yaml(list_yaml)

    def test_load_yaml_accepts_path_as_string(self, sample_yaml_file: Path) -> None:
        """load_yaml accepts string path in addition to Path object."""
        result = load_yaml(sample_yaml_file)
        assert isinstance(result, dict)


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_creates_new_directory(self, tmp_path: Path) -> None:
        """ensure_dir creates a directory that does not exist."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        assert not new_dir.exists()
        result = ensure_dir(new_dir)
        assert new_dir.is_dir()
        assert result == new_dir

    def test_idempotent_on_existing_directory(self, tmp_path: Path) -> None:
        """ensure_dir does not raise on existing directory."""
        existing = tmp_path / "existing"
        existing.mkdir()
        result = ensure_dir(existing)
        assert existing.is_dir()
        assert result == existing

    def test_returns_path_for_chaining(self, tmp_path: Path) -> None:
        """ensure_dir returns the path for method chaining."""
        result = ensure_dir(tmp_path / "chain")
        assert isinstance(result, Path)


class TestValidateImagePath:
    """Tests for validate_image_path function."""

    def test_valid_jpg(self, tmp_path: Path) -> None:
        """validate_image_path returns True for existing .jpg file."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")  # Minimal JPEG header
        assert validate_image_path(img) is True

    def test_valid_png(self, tmp_path: Path) -> None:
        """validate_image_path returns True for existing .png file."""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG")
        assert validate_image_path(img) is True

    def test_valid_bmp(self, tmp_path: Path) -> None:
        """validate_image_path returns True for existing .bmp file."""
        img = tmp_path / "test.bmp"
        img.write_bytes(b"BM")
        assert validate_image_path(img) is True

    def test_invalid_extension(self, tmp_path: Path) -> None:
        """validate_image_path returns False for non-image extension."""
        txt = tmp_path / "test.txt"
        txt.write_text("not an image", encoding="utf-8")
        assert validate_image_path(txt) is False

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """validate_image_path returns False for non-existent file."""
        assert validate_image_path(tmp_path / "missing.jpg") is False

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """validate_image_path handles uppercase extensions."""
        img = tmp_path / "test.JPG"
        img.write_bytes(b"\xff\xd8\xff\xe0")
        assert validate_image_path(img) is True

    def test_jpeg_extension(self, tmp_path: Path) -> None:
        """validate_image_path accepts .jpeg extension."""
        img = tmp_path / "test.jpeg"
        img.write_bytes(b"\xff\xd8\xff\xe0")
        assert validate_image_path(img) is True


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_finds_project_root(self) -> None:
        """get_project_root returns a directory containing pyproject.toml."""
        root = get_project_root()
        assert root.is_dir()
        assert (root / "pyproject.toml").is_file()

    def test_root_contains_urbaneye_package(self) -> None:
        """get_project_root returns directory containing urbaneye/ package."""
        root = get_project_root()
        assert (root / "urbaneye").is_dir()
