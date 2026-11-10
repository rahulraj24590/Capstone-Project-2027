"""
Model registry for LandslideGuard.

Tracks model versions, manages serialisation/deserialisation, and
prevents silent overwriting of previously evaluated models.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

MODELS_DIR = Path(get_setting("paths", "models") or "models")
REGISTRY_FILE = MODELS_DIR / "registry.json"


def _load_registry() -> dict:
    """Load model registry JSON."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": []}


def _save_registry(registry: dict) -> None:
    """Persist model registry JSON."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def save_model(
    model,
    model_name: str,
    metadata: dict | None = None,
    overwrite: bool = False,
) -> Path:
    """Serialise and register a model.

    Args:
        model: Fitted sklearn-compatible model or pipeline.
        model_name: Base name without extension.
        metadata: Arbitrary metadata to store in the registry
                  (e.g., metrics, feature list, dataset version).
        overwrite: If False, raises if a model with the same name exists.

    Returns:
        Path to the saved model file.

    Raises:
        FileExistsError: If model file exists and overwrite=False.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name}.pkl"

    if model_path.exists() and not overwrite:
        raise FileExistsError(
            f"Model '{model_path}' already exists. "
            "Set overwrite=True to replace it."
        )

    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    # Update registry
    registry = _load_registry()
    entry = {
        "name": model_name,
        "path": str(model_path),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    # Remove existing entry with same name
    registry["models"] = [m for m in registry["models"] if m["name"] != model_name]
    registry["models"].append(entry)
    _save_registry(registry)
    logger.info("Registered model '%s' in registry.", model_name)

    return model_path


def load_model(model_name: str):
    """Load a model by name from the models directory.

    Args:
        model_name: Model name (without .pkl extension).

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model '{model_path}' not found. "
            "Run the training pipeline first: python -m src.models.train"
        )
    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)
    return model


def list_models() -> list[dict]:
    """Return all registered models.

    Returns:
        List of model registry entries.
    """
    return _load_registry().get("models", [])


def get_model_info(model_name: str) -> dict | None:
    """Retrieve registry metadata for a named model.

    Args:
        model_name: Model name.

    Returns:
        Registry entry dict or None if not found.
    """
    registry = _load_registry()
    for entry in registry.get("models", []):
        if entry["name"] == model_name:
            return entry
    return None


def active_model_exists() -> bool:
    """Check if the configured active model file exists.

    Returns:
        True if the model file is present on disk.
    """
    active = get_setting("model", "active_model") or "best_model.pkl"
    return (MODELS_DIR / active).exists()
