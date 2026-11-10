"""
Input validation for the risk engine.

Validates environmental and terrain feature dictionaries before
they are passed to the risk estimation pipeline.

Separated from utils/validation.py to allow risk-specific rules
(e.g., checking that all rainfall values are from the same time window).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.utils.validation import (
    validate_coordinates,
    validate_feature_value,
    validate_rainfall_consistency,
    FEATURE_RANGES,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationStatus(str, Enum):
    """Validation outcome status."""
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID = "invalid"


@dataclass
class RiskValidationResult:
    """Container for risk input validation outcomes."""

    status: ValidationStatus = ValidationStatus.VALID
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.status = ValidationStatus.INVALID

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        if self.status == ValidationStatus.VALID:
            self.status = ValidationStatus.VALID_WITH_WARNINGS


class RiskInputValidator:
    """Validates feature dictionaries for risk estimation."""

    # Features that should ideally be present; absence triggers a WARNING, not an error.
    # Hard errors are only raised for clearly invalid values (e.g., negative rainfall,
    # out-of-range coordinates). Missing features reduce estimate quality but do not
    # prevent the heuristic baseline from running.
    PREFERRED_FEATURES = ["slope_degrees", "elevation_m"]
    # At least one rainfall variable should be present for dynamic risk
    RAINFALL_FEATURES = ["rainfall_1h_mm", "rainfall_24h_mm", "rainfall_72h_mm"]

    def validate(self, features: dict[str, Any]) -> RiskValidationResult:
        """Validate a feature dictionary.

        Args:
            features: Dict of feature name → value.

        Returns:
            RiskValidationResult with status, errors, and warnings.
            Status is INVALID only if coordinates are wrong or values are out of
            physical range (e.g., negative rainfall). Missing features produce
            VALID_WITH_WARNINGS so the heuristic baseline can still run.
        """
        result = RiskValidationResult()

        if not features:
            result.add_warning(
                "No feature values provided. Risk estimate will use heuristic defaults only."
            )
            return result

        # 1. Preferred features — warn if absent, check range if present
        for feat in self.PREFERRED_FEATURES:
            val = features.get(feat)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                result.add_warning(
                    f"Preferred feature '{feat}' is missing. "
                    "Risk estimate quality is reduced."
                )
            else:
                sub = validate_feature_value(feat, val, allow_none=False)
                # Range errors for present features become errors
                result.errors.extend(sub.errors)
                result.warnings.extend(sub.warnings)
                if not sub.is_valid:
                    result.status = ValidationStatus.INVALID

        # 2. Optional features — range check only
        optional = [
            f for f in FEATURE_RANGES
            if f not in self.PREFERRED_FEATURES and f in features
        ]
        for feat in optional:
            sub = validate_feature_value(feat, features[feat], allow_none=True)
            result.warnings.extend(sub.warnings)
            # Propagate hard errors (e.g., negative rainfall) as INVALID
            if sub.errors:
                result.errors.extend(sub.errors)
                result.status = ValidationStatus.INVALID

        # 3. Rainfall presence check
        has_rainfall = any(
            features.get(f) is not None
            and not (isinstance(features.get(f), float) and math.isnan(features.get(f)))
            for f in self.RAINFALL_FEATURES
        )
        if not has_rainfall:
            result.add_warning(
                "No rainfall data provided. Dynamic risk estimation will rely only on "
                "terrain features. The estimate will NOT reflect current weather conditions."
            )

        # 4. Rainfall consistency
        r1h = features.get("rainfall_1h_mm")
        r24h = features.get("rainfall_24h_mm")
        r72h = features.get("rainfall_72h_mm")
        consistency = validate_rainfall_consistency(r1h, r24h, r72h)
        result.warnings.extend(consistency.warnings)

        # 5. Coordinate validation (if coordinates are provided)
        lat = features.get("latitude")
        lon = features.get("longitude")
        if lat is not None and lon is not None:
            coord_result = validate_coordinates(lat, lon)
            result.errors.extend(coord_result.errors)
            result.warnings.extend(coord_result.warnings)
            if not coord_result.is_valid:
                result.status = ValidationStatus.INVALID

        # Upgrade status if there are warnings but no errors
        if result.status == ValidationStatus.VALID and result.warnings:
            result.status = ValidationStatus.VALID_WITH_WARNINGS

        return result
