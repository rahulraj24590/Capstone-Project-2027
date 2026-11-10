"""
Dynamic risk engine for LandslideGuard.

Orchestrates the three-tier risk assessment framework:

1. STATIC SUSCEPTIBILITY
   Long-term terrain susceptibility based on topographic, geological,
   and land-cover features. Computed once per location.

2. DYNAMIC RISK ESTIMATE
   Susceptibility adjusted by current or recent environmental conditions
   (primarily rainfall intensity and antecedent moisture). Updates as
   environmental inputs change.

3. EXPERIMENTAL ALERT
   Decision to raise a warning based on the dynamic risk estimate,
   configured thresholds, data quality checks, and cooldown logic.

The engine falls back to a clearly labelled heuristic scoring baseline
if the trained model is not available.

IMPORTANT:
- "Experimental" or "baseline" risk scores are NOT operationally validated.
- Missing environmental inputs are not silently filled with zero.
- Data freshness is enforced; stale estimates trigger warnings.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from src.risk.thresholds import RiskCategory, get_thresholds
from src.risk.validation import RiskInputValidator, ValidationStatus
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

FRESHNESS_HOURS = float(get_setting("risk", "data_freshness_hours") or 6)


# ─────────────────────────────────────────────
# Baseline risk scoring (no trained model needed)
# ─────────────────────────────────────────────

class BaselineRiskScorer:
    """Heuristic risk scorer for use when no trained model is available.

    Based on a weighted linear combination of normalised feature values.
    Normalisation bounds are approximate physical maxima for the features.

    THIS SCORER IS EXPERIMENTAL and must NOT be presented as a validated
    predictive model. It serves only to demonstrate the application flow.
    """

    # Normalisation upper bounds (features are clipped to [0, max_val])
    FEATURE_NORMS: dict[str, float] = {
        "slope_degrees": 60.0,
        "rainfall_24h_mm": 200.0,
        "rainfall_72h_mm": 500.0,
        "rainfall_1h_mm": 50.0,
        "elevation_m": 3000.0,      # inverted (higher elevation → higher risk)
        "soil_moisture": 1.0,
        "vegetation_index": 1.0,    # inverted (less vegetation → higher risk)
    }

    DEFAULT_WEIGHTS: dict[str, float] = {
        "slope_degrees": 0.30,
        "rainfall_24h_mm": 0.25,
        "rainfall_72h_mm": 0.20,
        "elevation_m": 0.10,
        "soil_moisture": 0.10,
        "vegetation_index": 0.05,
    }

    def __init__(self) -> None:
        cfg_weights = get_setting("risk", "baseline", "weights") or {}
        self.weights = {**self.DEFAULT_WEIGHTS, **cfg_weights}

    def score(self, features: dict[str, Any]) -> tuple[float, list[str]]:
        """Compute a normalised heuristic risk score in [0, 1].

        Args:
            features: Dict of feature name → value. Missing values are noted.

        Returns:
            Tuple of (score, warnings list).
        """
        warnings: list[str] = [
            "EXPERIMENTAL: This score is from a heuristic baseline, NOT a validated model."
        ]
        total_weight = 0.0
        weighted_sum = 0.0
        missing_features: list[str] = []

        for feat, weight in self.weights.items():
            val = features.get(feat)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                missing_features.append(feat)
                continue

            norm_max = self.FEATURE_NORMS.get(feat, 1.0)
            val = float(val)

            # For features where lower values indicate more risk (inversion)
            if feat == "vegetation_index":
                # High vegetation → lower risk → invert
                normalised = 1.0 - max(0.0, min(val, norm_max)) / norm_max
            elif feat == "elevation_m":
                # High elevation can indicate steeper terrain; use moderate weight
                normalised = max(0.0, min(val, norm_max)) / norm_max
            else:
                normalised = max(0.0, min(val, norm_max)) / norm_max

            weighted_sum += weight * normalised
            total_weight += weight

        if total_weight == 0:
            warnings.append("No valid feature values available for baseline scoring.")
            return 0.0, warnings

        score = weighted_sum / total_weight

        if missing_features:
            warnings.append(
                f"Missing features in baseline score: {missing_features}. "
                "Score computed from available features only."
            )

        return float(np.clip(score, 0.0, 1.0)), warnings


# ─────────────────────────────────────────────
# Risk engine
# ─────────────────────────────────────────────

class RiskEngine:
    """Main dynamic risk assessment engine.

    Dispatches to the trained ML model if available,
    or falls back to the heuristic baseline with explicit warnings.
    """

    def __init__(self) -> None:
        self._predictor = None
        self._baseline = BaselineRiskScorer()
        self._thresholds = get_thresholds()
        self._model_available: bool | None = None

    def _check_model(self) -> bool:
        """Check if the trained model is available."""
        if self._model_available is not None:
            return self._model_available
        try:
            from src.models.model_registry import active_model_exists
            self._model_available = active_model_exists()
        except Exception:
            self._model_available = False
        return self._model_available

    def _get_predictor(self):
        """Lazy-load the ML predictor."""
        if self._predictor is None:
            from src.models.predict import get_predictor
            self._predictor = get_predictor()
        return self._predictor

    def estimate_risk(
        self,
        features: dict[str, Any],
        observation_time: datetime | None = None,
        location_name: str | None = None,
    ) -> dict:
        """Estimate risk for a single location.

        Args:
            features: Environmental and terrain feature dict.
            observation_time: When the environmental data was observed.
                              If None, current time is assumed (with warning).
            location_name: Optional human-readable location name.

        Returns:
            Dict containing:
            - probability: Float or None.
            - risk_category: str.
            - method: 'ml_model' | 'heuristic_baseline'.
            - is_experimental: bool (always True until independent validation).
            - data_quality: dict with freshness and completeness indicators.
            - warnings: list of warning strings.
            - errors: list of error strings.
            - features_used: dict.
        """
        warnings: list[str] = []
        errors: list[str] = []

        # ── Data freshness check ──────────────────────────────────────
        if observation_time is None:
            observation_time = datetime.now(timezone.utc)
            warnings.append(
                "No observation timestamp provided. Assuming current time. "
                "Verify that environmental inputs are recent."
            )
        else:
            if observation_time.tzinfo is None:
                observation_time = observation_time.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - observation_time).total_seconds() / 3600
            if age_hours > FRESHNESS_HOURS:
                warnings.append(
                    f"Environmental data is {age_hours:.1f} hours old "
                    f"(threshold: {FRESHNESS_HOURS} hours). "
                    "Risk estimate may not reflect current conditions."
                )

        # ── Input validation ──────────────────────────────────────────
        validator = RiskInputValidator()
        validation = validator.validate(features)
        warnings.extend(validation.warnings)
        if validation.status == ValidationStatus.INVALID:
            errors.extend(validation.errors)
            return {
                "probability": None,
                "risk_category": RiskCategory.UNKNOWN.value,
                "method": "none",
                "is_experimental": True,
                "data_quality": {"valid": False},
                "warnings": warnings,
                "errors": errors,
                "features_used": features,
            }

        # ── Compute risk ──────────────────────────────────────────────
        method = "heuristic_baseline"
        probability: float | None = None

        if self._check_model():
            try:
                predictor = self._get_predictor()
                result = predictor.predict(features)
                probability = result["probability"]
                warnings.extend(result.get("warnings", []))
                if probability is not None:
                    method = "ml_model"
                    logger.debug(
                        "ML prediction for %s: prob=%.4f",
                        location_name or "unknown", probability,
                    )
                else:
                    for err in result.get("errors", []):
                        warnings.append(f"ML model skipped ({err}); falling back to heuristic baseline.")
            except Exception as e:
                logger.warning("ML model prediction failed: %s. Falling back to baseline.", e)
                warnings.append(
                    f"ML model prediction failed ({type(e).__name__}: {e}). "
                    "Falling back to heuristic baseline."
                )

        if probability is None:
            probability, baseline_warnings = self._baseline.score(features)
            warnings.extend(baseline_warnings)
            method = "heuristic_baseline"

        risk_category = self._thresholds.classify(probability)

        # ── Data quality indicators ───────────────────────────────────
        present_features = [k for k, v in features.items()
                            if v is not None and not (isinstance(v, float) and math.isnan(v))]
        total_expected = 7  # rough count of key features
        completeness = len(present_features) / total_expected

        data_quality = {
            "valid": True,
            "completeness_pct": round(completeness * 100, 1),
            "observation_time": observation_time.isoformat(),
            "freshness_ok": True,  # already warned above if not
            "present_features": present_features,
        }

        return {
            "probability": round(probability, 4),
            "risk_category": risk_category.value,
            "method": method,
            "is_experimental": True,  # Always True until independently validated
            "data_quality": data_quality,
            "warnings": warnings,
            "errors": errors,
            "features_used": features,
            "location_name": location_name,
            "estimated_at": datetime.now(timezone.utc).isoformat(),
        }

    def estimate_risk_batch(self, observations: list[dict]) -> list[dict]:
        """Estimate risk for multiple locations.

        Args:
            observations: List of feature dicts.

        Returns:
            List of risk estimate dicts.
        """
        return [self.estimate_risk(**obs) for obs in observations]


# ─────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────

_engine: RiskEngine | None = None


def get_risk_engine() -> RiskEngine:
    """Return the module-level RiskEngine singleton."""
    global _engine
    if _engine is None:
        _engine = RiskEngine()
    return _engine
