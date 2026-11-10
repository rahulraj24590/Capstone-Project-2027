"""
Model inference module for LandslideGuard.

Provides a unified interface for loading the active model and making
predictions. Validates inputs and raises explicit errors if the model
is not available or inputs are out of expected range.

Design:
- Predictions always use the saved model artefact.
- No fallback to random or hard-coded predictions.
- Missing required features raise ValidationError.
- Out-of-range features produce warnings, not errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_registry import load_model, get_model_info
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting
from src.utils.validation import validate_feature_dict, ValidationResult

logger = get_logger(__name__)

MODELS_DIR = Path(get_setting("paths", "models") or "models")
ACTIVE_MODEL_NAME = get_setting("model", "active_model") or "best_model"
# Remove .pkl extension if present
ACTIVE_MODEL_NAME = ACTIVE_MODEL_NAME.replace(".pkl", "")


def load_feature_columns() -> list[str]:
    """Load the list of feature columns used during training.

    Returns:
        List of feature column names.

    Raises:
        FileNotFoundError: If the feature columns file does not exist.
    """
    feature_path = MODELS_DIR / "feature_columns.json"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {feature_path}. "
            "Ensure the model has been trained."
        )
    with open(feature_path, "r") as f:
        return json.load(f)["feature_columns"]


def load_training_stats() -> dict:
    """Load training statistics for NaN imputation.

    Returns:
        Dict with 'feature_medians' key.
    """
    stats_path = MODELS_DIR / "training_stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            return json.load(f)
    return {"feature_medians": {}}


class Predictor:
    """Wraps the active model for inference.

    Usage:
        predictor = Predictor()
        result = predictor.predict({"slope_degrees": 28, "rainfall_24h_mm": 85, ...})
    """

    def __init__(self) -> None:
        self._model = None
        self._feature_cols: list[str] | None = None
        self._training_stats: dict = {}
        self._model_info: dict | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model on first prediction."""
        if self._model is None:
            self._model = load_model(ACTIVE_MODEL_NAME)
            self._feature_cols = load_feature_columns()
            self._training_stats = load_training_stats()
            self._model_info = get_model_info(ACTIVE_MODEL_NAME)
            logger.info(
                "Predictor loaded model '%s' with %d features.",
                ACTIVE_MODEL_NAME, len(self._feature_cols),
            )

    def predict(
        self,
        features: dict,
        return_proba: bool = True,
    ) -> dict:
        """Run inference for a single observation.

        Args:
            features: Dict mapping feature name → value.
                      Missing optional features are imputed with training medians.
                      Missing required features raise ValueError.
            return_proba: If True, return probability; otherwise return hard class.

        Returns:
            Dict with keys:
            - probability: Float in [0, 1].
            - risk_category: str ('Low', 'Moderate', 'High', 'Very High').
            - predicted_class: int (0 or 1).
            - warnings: List of warning strings.
            - errors: List of error strings.
            - feature_values: Dict of values actually used.
            - model_name: str.
        """
        self._ensure_loaded()
        assert self._feature_cols is not None

        warnings: list[str] = []
        errors: list[str] = []

        # Validate features
        required = ["slope_degrees", "elevation_m", "rainfall_24h_mm", "rainfall_72h_mm"]
        optional = [c for c in self._feature_cols if c not in required]
        validation = validate_feature_dict(features, required, optional)
        warnings.extend(validation.warnings)

        if not validation.is_valid:
            errors.extend(validation.errors)
            return {
                "probability": None,
                "risk_category": "Unknown",
                "predicted_class": None,
                "warnings": warnings,
                "errors": errors,
                "feature_values": features,
                "model_name": ACTIVE_MODEL_NAME,
            }

        # Build feature vector in correct order
        medians = self._training_stats.get("feature_medians", {})
        feature_row = []
        used_features: dict = {}

        for col in self._feature_cols:
            val = features.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                # Impute with training median
                median_val = medians.get(col, 0.0)
                feature_row.append(float(median_val))
                used_features[col] = float(median_val)
                warnings.append(
                    f"Feature '{col}' was missing and imputed with training median "
                    f"({median_val:.4f}). Prediction may be less reliable."
                )
            else:
                feature_row.append(float(val))
                used_features[col] = float(val)

        X = np.array([feature_row])

        # Predict
        if hasattr(self._model, "predict_proba"):
            prob = float(self._model.predict_proba(X)[0, 1])
        else:
            prob = float(self._model.predict(X)[0])
            warnings.append("Model does not support probability output; using hard prediction.")

        pred_class = int(prob >= 0.5)
        risk_category = probability_to_risk_category(prob)

        return {
            "probability": prob,
            "risk_category": risk_category,
            "predicted_class": pred_class,
            "warnings": warnings,
            "errors": errors,
            "feature_values": used_features,
            "model_name": ACTIVE_MODEL_NAME,
            "model_info": self._model_info,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a DataFrame.

        Args:
            df: DataFrame where each row is one observation.

        Returns:
            DataFrame with 'probability', 'risk_category', 'predicted_class' columns added.
        """
        self._ensure_loaded()
        assert self._feature_cols is not None

        medians = self._training_stats.get("feature_medians", {})
        feature_df = df.reindex(columns=self._feature_cols)

        # Impute missing values with training medians
        for col in self._feature_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(medians.get(col, 0.0))

        X = feature_df.values.astype(float)

        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(X)[:, 1]
        else:
            probs = self._model.predict(X).astype(float)

        result = df.copy()
        result["probability"] = probs
        result["predicted_class"] = (probs >= 0.5).astype(int)
        result["risk_category"] = pd.Series(probs).map(probability_to_risk_category).values

        return result


def probability_to_risk_category(probability: float) -> str:
    """Convert a probability to a risk category label.

    Thresholds are loaded from config/settings.yaml.
    These thresholds are PROVISIONAL and have not been validated
    against confirmed landslide observations.

    Args:
        probability: Float in [0, 1].

    Returns:
        Risk category string.
    """
    from src.utils.settings import get_setting
    thresholds = get_setting("risk", "thresholds") or {}
    low = float(thresholds.get("low", 0.35))
    moderate = float(thresholds.get("moderate", 0.55))
    high = float(thresholds.get("high", 0.75))

    if probability < low:
        return "Low"
    elif probability < moderate:
        return "Moderate"
    elif probability < high:
        return "High"
    else:
        return "Very High"


# Module-level singleton
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Return the module-level Predictor singleton.

    Returns:
        Loaded Predictor instance.
    """
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor
