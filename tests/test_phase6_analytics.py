"""Comprehensive tests for Phase 6 Intelligent Environmental Analysis, Forecasting & Drift Monitoring."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.database.db as db_mod
import src.database.models as models_mod
from api import (
    PredictionOutcomeRequest,
    get_forecast_risk_trend,
    get_model_drift_monitoring,
    get_rainfall_intelligence,
    record_model_prediction_outcome,
)
from src.monitoring.drift_monitor import compute_model_drift_report
from src.monitoring.environmental_analytics import (
    analyze_rainfall_trends,
    detect_rainfall_anomalies,
    forecast_risk_trend,
)


def _run_coro_sync(coro):
    """Execute a coroutine synchronously without creating an asyncio event loop."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch):
    """Point database operations at a temporary SQLite DB for Phase 6 tests."""
    db_file = tmp_path / "test_phase6.db"

    @contextmanager
    def _tmp_ctx(_db_path=None):
        conn = db_mod.get_connection(db_file)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(db_mod, "DB_PATH", db_file)
    monkeypatch.setattr(db_mod, "db_context", _tmp_ctx)
    monkeypatch.setattr(models_mod, "db_context", _tmp_ctx)
    db_mod.initialize_database(db_file)
    return db_file


def _build_baseline_daily(n_days: int = 120, start: date = date(2026, 5, 1)) -> dict:
    rng = np.random.default_rng(42)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(n_days)]
    # Realistic variable rainfall (~8 mm mean, ~6 mm std)
    vals = np.clip(rng.normal(loc=8.0, scale=5.0, size=n_days), 0.0, 45.0).round(2).tolist()
    return {"time": dates, "precipitation_sum": vals}


def test_anomaly_detection_guards_and_four_classifications():
    # 1. Insufficient historical baseline (< 30 days) NEVER fabricates a baseline
    short_baseline = _build_baseline_daily(n_days=12)
    res_short = detect_rainfall_anomalies(recent_24h_mm=50.0, baseline_daily=short_baseline)
    assert res_short["baseline_available"] is False
    assert res_short["z_score"] is None
    assert res_short["isolation_forest_available"] is False
    assert "Insufficient Historical Baseline" in res_short["classification"]

    # 2. Missing recent 24h rainfall never replaced with zero
    rich_baseline = _build_baseline_daily(n_days=90)
    res_missing = detect_rainfall_anomalies(recent_24h_mm=None, baseline_daily=rich_baseline)
    assert res_missing["baseline_available"] is True
    assert res_missing["z_score"] is None
    assert "Insufficient Recent 24h" in res_missing["classification"]
    assert res_missing["isolation_forest_available"] is True

    # 3. Four documented anomaly classifications across empirical archive
    stats = detect_rainfall_anomalies(recent_24h_mm=8.0, baseline_daily=rich_baseline)
    assert stats["classification"] == "Normal rainfall"
    assert stats["baseline_stats"] is not None
    assert len(stats["history_series"]) == 90

    mean_v = stats["baseline_stats"]["mean_mm"]
    std_v = stats["baseline_stats"]["std_mm"]
    res_above = detect_rainfall_anomalies(
        recent_24h_mm=round(mean_v + 1.35 * std_v, 2), baseline_daily=rich_baseline
    )
    assert res_above["classification"] == "Above-normal rainfall"

    res_unusual = detect_rainfall_anomalies(
        recent_24h_mm=round(mean_v + 2.35 * std_v, 2), baseline_daily=rich_baseline
    )
    assert res_unusual["classification"] == "Unusual rainfall"

    res_extreme = detect_rainfall_anomalies(
        recent_24h_mm=max(140.0, round(mean_v + 4.0 * std_v, 2)), baseline_daily=rich_baseline
    )
    assert res_extreme["classification"] == "Extreme rainfall"
    assert res_extreme["above_rolling_threshold"] is True

    # 4. Zero-variance constant baseline handled safely without ZeroDivisionError
    dry_baseline = {
        "time": [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(40)],
        "precipitation_sum": [0.0] * 40,
    }
    res_dry = detect_rainfall_anomalies(recent_24h_mm=0.0, baseline_daily=dry_baseline)
    assert res_dry["classification"] == "Normal rainfall"
    assert res_dry["baseline_stats"]["zero_variance"] is True


def test_multiscale_trends_and_seasonal_coverage_guards():
    baseline = _build_baseline_daily(n_days=150, start=date(2026, 5, 1))
    trends = analyze_rainfall_trends(
        baseline_daily=baseline,
        daily_historical=None,
        recent_24h_mm=16.0,
    )
    assert trends["has_data"] is True
    assert trends["valid_days_count"] == 150
    assert trends["weekly_trend"]["complete_weeks"] >= 18
    assert trends["monthly_trend"]["complete_months"] >= 4
    assert trends["historical_comparison"]["baseline_available"] is True

    seasons = {row["season"]: row for row in trends["seasonal_comparison"]}
    assert seasons["Monsoon (JJAS)"]["sufficient_coverage"] is True
    assert seasons["Monsoon (JJAS)"]["mean_daily_mm"] is not None
    # May-Sep dataset has zero Jan-Feb days -> Winter (JF) must report insufficient coverage
    assert seasons["Winter (JF)"]["sufficient_coverage"] is False
    assert "Insufficient Seasonal Coverage" in seasons["Winter (JF)"]["status"]


def test_forecast_risk_trend_validation_and_uncertainty():
    base_t = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    hist_times = [(base_t - timedelta(hours=72 - i)).isoformat() for i in range(72)]
    fcst_times = [(base_t + timedelta(hours=i + 1)).isoformat() for i in range(72)]

    complete_weather = {
        "elevation_m": 1250.0,
        "historical_hourly": {
            "time": hist_times,
            "precipitation": [0.8] * 72,
            "soil_moisture_0_to_7cm": [0.42] * 72,
        },
        "forecast_hourly": {
            "time": fcst_times,
            "precipitation": [2.5] * 24 + [3.5] * 24 + [4.5] * 24,
            "soil_moisture_0_to_7cm": [0.55] * 24 + [0.65] * 24 + [0.75] * 24,
        },
    }
    res_full = forecast_risk_trend(
        weather_payload=complete_weather,
        terrain_features={"slope_degrees": 32.0, "elevation_m": 1250.0},
    )
    assert res_full["is_experimental"] is True
    assert res_full["cannot_predict_exact_timing"] is True
    assert res_full["model_is_synthetic_dev"] is True
    for h_key in ("24h", "48h", "72h"):
        hz = res_full["horizons"][h_key]
        assert hz["probability"] is not None
        assert 0.0 <= hz["lower_probability"] <= hz["probability"] <= hz["upper_probability"] <= 1.0
        assert hz["uncertainty_width"] > 0.0
        assert any("forecast" in item.lower() for item in hz["forecast_derived_inputs"])

    # Incomplete forecast at +30h -> +24h succeeds cleanly, +48h and +72h explicitly Unavailable
    incomplete_weather = {
        "elevation_m": 1250.0,
        "historical_hourly": complete_weather["historical_hourly"],
        "forecast_hourly": {
            "time": fcst_times[:50],
            "precipitation": [2.0] * 30 + [None] + [2.0] * 19,
            "soil_moisture_0_to_7cm": [0.5] * 50,
        },
    }
    res_inc = forecast_risk_trend(
        weather_payload=incomplete_weather,
        terrain_features={"slope_degrees": 32.0, "elevation_m": 1250.0},
    )
    assert res_inc["horizons"]["24h"]["probability"] is not None
    assert res_inc["horizons"]["48h"]["probability"] is None
    assert "Unavailable" in res_inc["horizons"]["48h"]["status"]
    assert res_inc["horizons"]["72h"]["probability"] is None

    # Empty / None payload -> honest Unavailable state
    res_none = forecast_risk_trend(weather_payload=None)
    assert "Unavailable" in res_none["status"]
    assert res_none["horizons"]["24h"]["probability"] is None


def test_model_drift_monitoring_evidence_gating_and_outcomes(isolated_db):
    # 1. Insufficient evidence initially -> never claims confirmed drift
    report_empty = compute_model_drift_report()
    assert report_empty["drift_confirmed"] is False
    assert "Insufficient Evidence" in report_empty["overall_status"]

    # 2. Stationary production predictions matching synthetic reference split
    from src.models.model_registry import load_model

    ref_df = pd.read_parquet("data/processed/dataset.parquet").iloc[:300].copy()
    feat_cols = [
        "slope_degrees",
        "elevation_m",
        "rainfall_24h_mm",
        "rainfall_72h_mm",
        "rainfall_1h_mm",
        "vegetation_index",
        "soil_moisture",
    ]
    model = load_model("best_model")
    ref_slice = ref_df.iloc[:40]
    ref_slice_probs = model.predict_proba(ref_slice[feat_cols].to_numpy(dtype=float))[:, 1]
    stationary_preds = pd.DataFrame([
        {
            "id": idx + 1,
            "probability": float(ref_slice_probs[idx]),
            "input_features": {c: float(row[c]) for c in feat_cols},
            "warnings": [],
        }
        for idx, (_, row) in enumerate(ref_slice.iterrows())
    ])
    report_nom = compute_model_drift_report(
        predictions_df=stationary_preds,
        env_df=pd.DataFrame(),
        outcomes_df=pd.DataFrame(),
        reference_df=ref_df,
    )
    assert report_nom["drift_confirmed"] is False
    assert report_nom["data_quality_summary"]["feature_vectors_analyzed"] == 40

    # 3. Severe feature shift + high missingness triggers drift_confirmed=True
    shifted_preds = pd.DataFrame([
        {
            "id": idx + 1,
            "probability": 0.92,
            "input_features": {
                "slope_degrees": 48.0 + (idx % 5),
                "elevation_m": 2400.0 + idx * 5.0,
                "rainfall_24h_mm": 350.0 + idx * 3.0,
                "rainfall_72h_mm": 800.0 + idx * 5.0,
                "rainfall_1h_mm": None,  # 100% missing rainfall_1h_mm
                "vegetation_index": 0.12,
                "soil_moisture": 0.92,
            },
            "warnings": ["Feature 'rainfall_1h_mm' was missing and imputed with training median"],
        }
        for idx in range(35)
    ])
    report_shift = compute_model_drift_report(
        predictions_df=shifted_preds,
        env_df=pd.DataFrame(),
        outcomes_df=pd.DataFrame(),
        reference_df=ref_df,
    )
    assert report_shift["drift_confirmed"] is True
    assert report_shift["overall_status"] == "Drift Indicators Confirmed"
    miss_map = {r["feature"]: r for r in report_shift["missingness_drift"]}
    assert miss_map["rainfall_1h_mm"]["status"] == "Elevated Missingness"

    # 4. Labelled outcome feedback via SQLite + API endpoint triggers live performance drift check
    loc_id = models_mod.upsert_location(name="Drift Ridge", latitude=27.1, longitude=88.3)
    for i in range(12):
        # Create overconfident wrong predictions (prob=0.92, actual=0 and prob=0.08, actual=1)
        pred_p = 0.92 if i < 6 else 0.08
        act_y = 0 if i < 6 else 1
        pid = models_mod.save_prediction(
            probability=pred_p,
            risk_category="Very High" if pred_p > 0.5 else "Low",
            model_name="best_model",
            method="ml_model",
            input_features={
                "slope_degrees": 20.0,
                "elevation_m": 500.0,
                "rainfall_1h_mm": 5.0,
                "rainfall_24h_mm": 25.0,
                "rainfall_72h_mm": 60.0,
                "vegetation_index": 0.5,
                "soil_moisture": 0.4,
            },
            location_id=loc_id,
        )
        out_resp = _run_coro_sync(
            record_model_prediction_outcome(
                PredictionOutcomeRequest(
                    prediction_id=pid,
                    actual_label=act_y,
                    outcome_source="pytest_field_audit",
                )
            )
        )
        assert out_resp["status"] == "recorded"

    drift_api_resp = _run_coro_sync(get_model_drift_monitoring(limit=100))
    assert drift_api_resp["performance_drift"]["labelled_samples"] == 12
    assert drift_api_resp["performance_drift"]["degradation_detected"] is True
    assert drift_api_resp["drift_confirmed"] is True


def test_phase6_weather_and_forecast_api_endpoints(monkeypatch, isolated_db):
    from src.data_providers import weather_provider as wp_mod

    base_t = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    mock_payload = {
        "latitude": 10.1,
        "longitude": 77.1,
        "timezone": "Asia/Kolkata",
        "utc_offset_seconds": 19800,
        "elevation_m": 1100.0,
        "historical_hourly": {
            "time": [(base_t - timedelta(hours=72 - i)).isoformat() for i in range(72)],
            "precipitation": [1.0] * 72,
            "soil_moisture_0_to_7cm": [0.45] * 72,
        },
        "forecast_hourly": {
            "time": [(base_t + timedelta(hours=i + 1)).isoformat() for i in range(72)],
            "precipitation": [2.0] * 72,
            "soil_moisture_0_to_7cm": [0.60] * 72,
        },
        "daily_historical": None,
        "archive_baseline_daily": _build_baseline_daily(n_days=90),
        "provenance": {"provider": "Open-Meteo", "is_cached": False},
    }
    monkeypatch.setattr(wp_mod.WeatherProvider, "fetch_separated_weather", lambda self, **kw: mock_payload)

    intel = _run_coro_sync(get_rainfall_intelligence(latitude=10.1, longitude=77.1))
    assert intel["anomaly_detection"]["baseline_available"] is True
    assert intel["anomaly_detection"]["classification"] in {
        "Normal rainfall",
        "Above-normal rainfall",
        "Unusual rainfall",
        "Extreme rainfall",
    }
    assert intel["trend_analysis"]["has_data"] is True

    fcst = _run_coro_sync(
        get_forecast_risk_trend(latitude=10.1, longitude=77.1, slope_degrees=30.0, elevation_m=1100.0)
    )
    assert fcst["cannot_predict_exact_timing"] is True
    assert fcst["horizons"]["24h"]["probability"] is not None
    assert fcst["horizons"]["72h"]["probability"] is not None
