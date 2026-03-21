"""File I/O and path utilities for UrbanEye.

Provides safe YAML loading, directory creation, image path validation,
and project root detection used across all modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Valid image extensions for the pipeline
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise yaml.YAMLError(f"Expected a YAML mapping (dict), got {type(data).__name__}")
    return data


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path (for chaining).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_image_path(path: Path) -> bool:
    """Check if a path points to a valid image file.

    Validates both file existence and extension.

    Args:
        path: Path to validate.

    Returns:
        True if the file exists and has a valid image extension.
    """
    path = Path(path)
    return path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS


def get_project_root() -> Path:
    """Find the UrbanEye project root by looking for pyproject.toml.

    Walks up from this file's directory until pyproject.toml is found.

    Returns:
        Path to the project root directory.

    Raises:
        FileNotFoundError: If pyproject.toml cannot be found in any parent.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml in parent directories)")
