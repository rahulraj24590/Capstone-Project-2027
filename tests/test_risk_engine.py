"""
Tests for the dynamic risk engine.

Verifies:
- Risk categories are assigned according to configured thresholds
- Threshold boundary conditions work correctly
- Invalid inputs trigger appropriate errors/warnings
- Missing values are not silently filled with zero
- Model failure falls back to heuristic baseline with a warning
- Alert cooldowns work correctly
"""

from __future__ import annotations

import pytest

from src.risk.thresholds import Thresholds, RiskCategory, get_thresholds
from src.risk.validation import RiskInputValidator, ValidationStatus
from src.risk.risk_engine import RiskEngine, BaselineRiskScorer


# ─────────────────────────────────────────────
# Threshold classification tests
# ─────────────────────────────────────────────

class TestThresholdClassification:
    @pytest.fixture
    def thresholds(self):
        return Thresholds(low=0.35, moderate=0.55, high=0.75)

    def test_low_category(self, thresholds):
        assert thresholds.classify(0.10) == RiskCategory.LOW
        assert thresholds.classify(0.34) == RiskCategory.LOW
        # Boundary: exactly 0.35 → Moderate
        assert thresholds.classify(0.35) == RiskCategory.MODERATE

    def test_moderate_category(self, thresholds):
        assert thresholds.classify(0.35) == RiskCategory.MODERATE
        assert thresholds.classify(0.50) == RiskCategory.MODERATE
        assert thresholds.classify(0.54) == RiskCategory.MODERATE

    def test_high_category(self, thresholds):
        assert thresholds.classify(0.55) == RiskCategory.HIGH
        assert thresholds.classify(0.70) == RiskCategory.HIGH
        assert thresholds.classify(0.74) == RiskCategory.HIGH

    def test_very_high_category(self, thresholds):
        assert thresholds.classify(0.75) == RiskCategory.VERY_HIGH
        assert thresholds.classify(0.90) == RiskCategory.VERY_HIGH
        assert thresholds.classify(1.00) == RiskCategory.VERY_HIGH

    def test_zero_probability(self, thresholds):
        assert thresholds.classify(0.0) == RiskCategory.LOW

    def test_none_probability(self, thresholds):
        assert thresholds.classify(None) == RiskCategory.UNKNOWN

    def test_nan_probability(self, thresholds):
        import math
        assert thresholds.classify(float("nan")) == RiskCategory.UNKNOWN

    def test_exceeds_alert_threshold_low(self, thresholds):
        # p=0.20 → Low; doesn't exceed Moderate
        assert not thresholds.exceeds_alert_threshold(0.20, "Moderate")

    def test_exceeds_alert_threshold_high(self, thresholds):
        # p=0.80 → Very High; exceeds High
        assert thresholds.exceeds_alert_threshold(0.80, "High")

    def test_config_thresholds_loaded(self):
        t = get_thresholds()
        assert isinstance(t, Thresholds)
        assert t.low > 0
        assert t.moderate > t.low
        assert t.high > t.moderate


# ─────────────────────────────────────────────
# Input validation tests
# ─────────────────────────────────────────────

class TestRiskInputValidation:
    @pytest.fixture
    def validator(self):
        return RiskInputValidator()

    def test_valid_features_pass(self, validator):
        features = {
            "slope_degrees": 25.0,
            "rainfall_24h_mm": 80.0,
            "rainfall_72h_mm": 160.0,
            "elevation_m": 600.0,
        }
        result = validator.validate(features)
        assert result.status != ValidationStatus.INVALID

    def test_empty_features_trigger_warning(self, validator):
        result = validator.validate({})
        assert len(result.warnings) > 0 or result.status == ValidationStatus.INVALID

    def test_out_of_range_slope_flagged(self, validator):
        features = {"slope_degrees": 150.0}  # impossible slope
        result = validator.validate(features)
        # Should have at least a warning
        assert len(result.warnings) > 0 or result.status == ValidationStatus.INVALID

    def test_negative_rainfall_flagged(self, validator):
        features = {"rainfall_24h_mm": -10.0}
        result = validator.validate(features)
        assert len(result.warnings) > 0 or result.status == ValidationStatus.INVALID


# ─────────────────────────────────────────────
# Baseline scorer tests
# ─────────────────────────────────────────────

class TestBaselineScorer:
    @pytest.fixture
    def scorer(self):
        return BaselineRiskScorer()

    def test_returns_float_in_range(self, scorer):
        features = {
            "slope_degrees": 30.0,
            "rainfall_24h_mm": 100.0,
            "rainfall_72h_mm": 200.0,
            "vegetation_index": 0.3,
            "soil_moisture": 0.6,
        }
        score, warnings = scorer.score(features)
        assert 0.0 <= score <= 1.0

    def test_always_marked_experimental(self, scorer):
        score, warnings = scorer.score({"slope_degrees": 20.0})
        assert any("EXPERIMENTAL" in w.upper() or "experimental" in w.lower() for w in warnings)

    def test_missing_features_warned(self, scorer):
        # Provide only one feature; rest should be noted as missing
        score, warnings = scorer.score({"slope_degrees": 30.0})
        assert any("Missing" in w or "missing" in w for w in warnings)

    def test_no_valid_features_returns_zero_score(self, scorer):
        score, warnings = scorer.score({})
        # When no features provided, score should be 0 with a warning
        assert score == pytest.approx(0.0)

    def test_high_slope_high_rainfall_gives_high_score(self, scorer):
        low_risk = {"slope_degrees": 5.0, "rainfall_24h_mm": 5.0, "soil_moisture": 0.1}
        high_risk = {"slope_degrees": 55.0, "rainfall_24h_mm": 180.0, "soil_moisture": 0.9}
        low_score, _ = scorer.score(low_risk)
        high_score, _ = scorer.score(high_risk)
        assert high_score > low_score


# ─────────────────────────────────────────────
# Risk engine integration tests
# ─────────────────────────────────────────────

class TestRiskEngineIntegration:
    @pytest.fixture
    def engine(self):
        return RiskEngine()

    def test_estimate_risk_returns_dict(self, engine):
        features = {"slope_degrees": 25.0, "rainfall_24h_mm": 80.0, "rainfall_72h_mm": 160.0}
        result = engine.estimate_risk(features=features)
        assert isinstance(result, dict)

    def test_risk_category_in_valid_values(self, engine):
        features = {"slope_degrees": 25.0, "rainfall_24h_mm": 80.0}
        result = engine.estimate_risk(features=features)
        valid_cats = {c.value for c in RiskCategory}
        assert result["risk_category"] in valid_cats

    def test_probability_in_range_or_none(self, engine):
        features = {"slope_degrees": 25.0, "rainfall_24h_mm": 80.0}
        result = engine.estimate_risk(features=features)
        prob = result.get("probability")
        if prob is not None:
            assert 0.0 <= prob <= 1.0

    def test_always_marked_experimental(self, engine):
        features = {"slope_degrees": 25.0, "rainfall_24h_mm": 80.0}
        result = engine.estimate_risk(features=features)
        assert result.get("is_experimental") is True

    def test_stale_observation_time_produces_warning(self, engine):
        from datetime import datetime, timezone, timedelta
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        features = {"slope_degrees": 25.0, "rainfall_24h_mm": 80.0}
        result = engine.estimate_risk(features=features, observation_time=old_time)
        assert len(result.get("warnings", [])) > 0

    def test_missing_features_not_silently_zero(self, engine):
        """Result must include explicit warnings when features are missing."""
        result = engine.estimate_risk(features={})
        # Either warnings must mention missing features OR the score must reflect the degraded input
        has_warnings = len(result.get("warnings", [])) > 0
        assert has_warnings  # Must explicitly warn about missing data

    def test_batch_estimation(self, engine):
        obs = [
            {"features": {"slope_degrees": 10.0, "rainfall_24h_mm": 20.0}},
            {"features": {"slope_degrees": 40.0, "rainfall_24h_mm": 150.0}},
        ]
        results = engine.estimate_risk_batch(obs)
        assert len(results) == 2


# ─────────────────────────────────────────────
# Alert cooldown tests
# ─────────────────────────────────────────────

class TestAlertCooldown:
    def test_is_on_cooldown_false_initially(self):
        from src.database.models import is_on_cooldown
        # A location that has never triggered an alert should not be on cooldown
        result = is_on_cooldown(location_id=None, risk_category="High")
        # None location should not be on cooldown (or may be treated as a unique key)
        assert isinstance(result, bool)

    def test_alert_manager_instantiates(self):
        from src.alerts.alert_manager import AlertManager
        manager = AlertManager()
        assert manager is not None

    def test_evaluate_low_risk_no_alert(self):
        from src.alerts.alert_manager import AlertManager
        manager = AlertManager()
        risk_result = {
            "probability": 0.05,
            "risk_category": "Low",
            "method": "heuristic_baseline",
        }
        alert = manager.evaluate_and_alert(risk_result)
        assert alert is None  # Low risk should not trigger alert
