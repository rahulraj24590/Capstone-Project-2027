"""
Application settings loader for LandslideGuard.

Reads config/settings.yaml and provides a typed settings object.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings() -> dict:
    """Load and cache application settings from config/settings.yaml.

    Returns:
        Nested dictionary of settings.

    Raises:
        FileNotFoundError: If settings.yaml does not exist.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Settings file not found: {_CONFIG_PATH}. "
            "Ensure config/settings.yaml exists."
        )
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_setting(*keys: str, default=None):
    """Retrieve a nested setting value by key path.

    Example:
        get_setting("risk", "thresholds", "low")  # → 0.35

    Args:
        *keys: Sequence of nested dictionary keys.
        default: Value returned if the key path does not exist.

    Returns:
        Setting value or default.
    """
    settings = load_settings()
    value = settings
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value
