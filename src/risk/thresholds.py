"""
Risk threshold definitions for LandslideGuard.

Defines risk category boundaries and provides utility functions
for threshold-based classification.

IMPORTANT: These thresholds are PROVISIONAL. They have not been
calibrated against verified historical landslide event data or
validated by a geological authority. They must not be used as the
basis for operational early-warning decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from src.utils.settings import get_setting


class RiskCategory(str, Enum):
    """Enumeration of risk category labels."""

    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"
    UNKNOWN = "Unknown"

    @classmethod
    def ordered(cls) -> list["RiskCategory"]:
        """Return categories in ascending severity order."""
        return [cls.LOW, cls.MODERATE, cls.HIGH, cls.VERY_HIGH]


@dataclass(frozen=True)
class Thresholds:
    """Probability thresholds for risk category boundaries.

    Each threshold is the UPPER bound (exclusive) for that category.
    Probabilities at or above the 'high' threshold fall into VERY_HIGH.
    """

    low: float
    moderate: float
    high: float

    @classmethod
    def from_config(cls) -> "Thresholds":
        """Load thresholds from application configuration.

        Returns:
            Thresholds instance populated from config/settings.yaml.
        """
        t = get_setting("risk", "thresholds") or {}
        return cls(
            low=float(t.get("low", 0.35)),
            moderate=float(t.get("moderate", 0.55)),
            high=float(t.get("high", 0.75)),
        )

    def classify(self, probability: float) -> RiskCategory:
        """Classify a probability value into a risk category.

        Args:
            probability: Float in [0, 1].

        Returns:
            RiskCategory enum value.
        """
        if probability is None or probability != probability:  # NaN check
            return RiskCategory.UNKNOWN
        p = float(probability)
        if p < self.low:
            return RiskCategory.LOW
        elif p < self.moderate:
            return RiskCategory.MODERATE
        elif p < self.high:
            return RiskCategory.HIGH
        else:
            return RiskCategory.VERY_HIGH

    def exceeds_alert_threshold(
        self,
        probability: float,
        alert_category: str = "High",
    ) -> bool:
        """Check if a probability meets or exceeds the alert trigger threshold.

        Args:
            probability: Estimated probability.
            alert_category: Minimum category for alert ('Moderate', 'High', 'Very High').

        Returns:
            True if risk meets or exceeds the alert category.
        """
        category = self.classify(probability)
        ordered = [c.value for c in RiskCategory.ordered()]
        try:
            return ordered.index(category.value) >= ordered.index(alert_category)
        except ValueError:
            return False


# Module-level singleton
_thresholds: Thresholds | None = None


def get_thresholds() -> Thresholds:
    """Return the module-level Thresholds singleton.

    Returns:
        Thresholds loaded from configuration.
    """
    global _thresholds
    if _thresholds is None:
        _thresholds = Thresholds.from_config()
    return _thresholds


# ─────────────────────────────────────────────
# Colour mapping (for visualisations)
# ─────────────────────────────────────────────

RISK_COLORS: dict[str, str] = {
    RiskCategory.LOW.value: "#2ecc71",       # Green
    RiskCategory.MODERATE.value: "#f39c12",  # Orange
    RiskCategory.HIGH.value: "#e74c3c",      # Red
    RiskCategory.VERY_HIGH.value: "#8e44ad", # Purple
    RiskCategory.UNKNOWN.value: "#95a5a6",   # Grey
}

RISK_EMOJI: dict[str, str] = {
    RiskCategory.LOW.value: "🟢",
    RiskCategory.MODERATE.value: "🟡",
    RiskCategory.HIGH.value: "🔴",
    RiskCategory.VERY_HIGH.value: "🟣",
    RiskCategory.UNKNOWN.value: "⚪",
}
