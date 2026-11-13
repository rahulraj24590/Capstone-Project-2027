"""
Application-wide input validation utilities for LandslideGuard.

Provides functions to validate coordinates, feature values, and
environmental inputs before they are passed to the risk engine or model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Validation result dataclass
# ─────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Container for validation outcomes."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def __bool__(self) -> bool:
        return self.is_valid


# ─────────────────────────────────────────────
# Coordinate validation
# ─────────────────────────────────────────────

def validate_coordinates(latitude: Any, longitude: Any) -> ValidationResult:
    """Validate geographic coordinates.

    Args:
        latitude: Latitude value to check.
        longitude: Longitude value to check.

    Returns:
        ValidationResult indicating whether coordinates are valid.
    """
    result = ValidationResult(is_valid=True)

    # Check types
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        result.add_error(
            f"Latitude and longitude must be numeric. "
            f"Got lat={latitude!r}, lon={longitude!r}."
        )
        return result

    # Check for NaN / Inf
    if math.isnan(lat) or math.isinf(lat):
        result.add_error(f"Latitude is not a finite number: {lat}")
    if math.isnan(lon) or math.isinf(lon):
        result.add_error(f"Longitude is not a finite number: {lon}")

    if not result.is_valid:
        return result

    # Range check
    if not (-90.0 <= lat <= 90.0):
        result.add_error(
            f"Latitude {lat} is out of range [-90, 90]. "
            "Note: latitude and longitude may have been swapped."
        )
    if not (-180.0 <= lon <= 180.0):
        result.add_error(
            f"Longitude {lon} is out of range [-180, 180]. "
            "Note: latitude and longitude may have been swapped."
        )

    # Check for (0, 0) — likely a default / missing value
    if lat == 0.0 and lon == 0.0:
        result.add_warning(
            "Coordinates (0°, 0°) are in the Gulf of Guinea. "
            "This may indicate a missing or default coordinate value."
        )

    return result


# ─────────────────────────────────────────────
# Feature range validation
# ─────────────────────────────────────────────

# Expected valid ranges for each feature.
# These are physical / instrumentation limits, not model training bounds.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "slope_degrees": (0.0, 90.0),
    "elevation_m": (-500.0, 9000.0),
    "rainfall_1h_mm": (0.0, 500.0),
    "rainfall_24h_mm": (0.0, 2000.0),
    "rainfall_48h_mm": (0.0, 3500.0),
    "rainfall_72h_mm": (0.0, 5000.0),
    "vegetation_index": (-1.0, 1.0),
    "soil_moisture": (0.0, 1.0),
}


def validate_feature_value(
    feature_name: str,
    value: Any,
    allow_none: bool = False,
) -> ValidationResult:
    """Validate a single feature value.

    Args:
        feature_name: Name of the feature (must match FEATURE_RANGES key).
        value: Value to validate.
        allow_none: If True, None values produce a warning, not an error.

    Returns:
        ValidationResult.
    """
    result = ValidationResult(is_valid=True)

    if value is None:
        if allow_none:
            result.add_warning(f"Feature '{feature_name}' is missing (None).")
        else:
            result.add_error(f"Feature '{feature_name}' is required but is None.")
        return result

    try:
        v = float(value)
    except (TypeError, ValueError):
        result.add_error(
            f"Feature '{feature_name}' must be numeric. Got {value!r}."
        )
        return result

    if math.isnan(v):
        result.add_error(f"Feature '{feature_name}' is NaN.")
        return result

    if math.isinf(v):
        result.add_error(f"Feature '{feature_name}' is infinite: {v}.")
        return result

    if feature_name in FEATURE_RANGES:
        lo, hi = FEATURE_RANGES[feature_name]
        if not (lo <= v <= hi):
            result.add_warning(
                f"Feature '{feature_name}' value {v} is outside the expected "
                f"range [{lo}, {hi}]. The model may produce unreliable estimates."
            )

    return result


def validate_feature_dict(
    features: dict[str, Any],
    required_features: list[str],
    optional_features: list[str] | None = None,
) -> ValidationResult:
    """Validate a dictionary of input features.

    Args:
        features: Dictionary of feature name → value.
        required_features: Features that must be present and valid.
        optional_features: Features that may be absent (warnings only).

    Returns:
        ValidationResult aggregating all individual checks.
    """
    result = ValidationResult(is_valid=True)
    optional_features = optional_features or []

    for feat in required_features:
        sub = validate_feature_value(feat, features.get(feat), allow_none=False)
        result.errors.extend(sub.errors)
        result.warnings.extend(sub.warnings)
        if not sub.is_valid:
            result.is_valid = False

    for feat in optional_features:
        if feat in features:
            sub = validate_feature_value(feat, features[feat], allow_none=True)
            result.warnings.extend(sub.warnings)

    return result


# ─────────────────────────────────────────────
# Rainfall consistency check
# ─────────────────────────────────────────────

def validate_rainfall_consistency(
    rainfall_1h: float | None,
    rainfall_24h: float | None,
    rainfall_72h: float | None,
) -> ValidationResult:
    """Check that rainfall accumulations are physically consistent.

    1-hour rainfall should not exceed 24-hour rainfall, and 24-hour
    should not exceed 72-hour, unless data comes from different sources.

    Args:
        rainfall_1h: 1-hour rainfall (mm).
        rainfall_24h: 24-hour rainfall (mm).
        rainfall_72h: 72-hour rainfall (mm).

    Returns:
        ValidationResult with warnings for inconsistencies.
    """
    result = ValidationResult(is_valid=True)

    if rainfall_1h is not None and rainfall_24h is not None:
        if rainfall_1h > rainfall_24h:
            result.add_warning(
                f"1-hour rainfall ({rainfall_1h} mm) exceeds 24-hour rainfall "
                f"({rainfall_24h} mm). This may indicate a data error or "
                "that the values come from different time periods."
            )

    if rainfall_24h is not None and rainfall_72h is not None:
        if rainfall_24h > rainfall_72h:
            result.add_warning(
                f"24-hour rainfall ({rainfall_24h} mm) exceeds 72-hour rainfall "
                f"({rainfall_72h} mm). This is physically possible only if "
                "the 72-hour window ended before peak rainfall."
            )

    return result
