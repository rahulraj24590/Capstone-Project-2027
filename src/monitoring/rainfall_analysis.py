"""
Dynamic rainfall analytics and risk-delta integration for LandslideGuard.

Scientific integrity guarantees:
- Separates past reanalysis, current snapshot, future forecast, and manual inputs.
- Never replaces missing rainfall observations with zero.
- Never fabricates baseline anomalies when historical archive records are absent.
- Never retrains ML models during environmental updates.
- Statistical risk associations are experimental and never assert causation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.database.models import get_latest_prediction, save_prediction
from src.risk.risk_engine import FRESHNESS_HOURS, get_risk_engine
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

SCIENTIFIC_DISCLAIMER = (
    "Experimental model-based association only. Rainfall correlations do not establish "
    "landslide causation, and outputs are NOT official emergency warnings."
)


def _is_valid_nonneg(val: Any) -> bool:
    if val is None:
        return False
    try:
        num = float(val)
    except (TypeError, ValueError):
        return False
    return math.isfinite(num) and num >= 0.0


def _strict_window_sum(series: list[Any], window: int) -> tuple[float | None, int]:
    """Compute window accumulation ONLY when all `window` trailing hours are valid.

    Never replaces missing values (None/NaN/negative) with zero.
    """
    if len(series) < window:
        missing_in_series = sum(1 for v in series if not _is_valid_nonneg(v))
        return None, missing_in_series + (window - len(series))

    tail = series[-window:]
    missing = sum(1 for v in tail if not _is_valid_nonneg(v))
    if missing > 0:
        return None, missing

    total = sum(float(v) for v in tail)
    return round(total, 2), 0


def _compute_ols_slope(values: list[float]) -> float | None:
    """Compute ordinary least-squares slope in mm per hour for evenly spaced hourly steps."""
    n = len(values)
    if n < 6:
        return None
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den <= 0.0:
        return 0.0
    return num / den


def compute_rainfall_metrics(
    historical_hourly: dict[str, Any],
    forecast_hourly: dict[str, Any] | None = None,
    baseline_daily: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute strict rainfall accumulations, 24h trend, forecast totals, and anomaly."""
    hist_precip = list(historical_hourly.get("precipitation", []) or [])
    valid_hist_vals = [float(v) for v in hist_precip if _is_valid_nonneg(v)]
    missing_total = len(hist_precip) - len(valid_hist_vals)

    r_1h = (
        round(float(hist_precip[-1]), 2)
        if hist_precip and _is_valid_nonneg(hist_precip[-1])
        else None
    )
    r_24h, miss_24 = _strict_window_sum(hist_precip, 24)
    r_48h, miss_48 = _strict_window_sum(hist_precip, 48)
    r_72h, miss_72 = _strict_window_sum(hist_precip, 72)

    # 24-hour trend using valid observations from the trailing 24-hour slice
    tail_24 = [float(v) for v in hist_precip[-24:] if _is_valid_nonneg(v)]
    slope = _compute_ols_slope(tail_24)
    if slope is None:
        trend_label = "Insufficient Data"
        trend_slope = None
    else:
        trend_slope = round(slope, 4)
        if slope > 0.05:
            trend_label = "Increasing"
        elif slope < -0.05:
            trend_label = "Decreasing"
        else:
            trend_label = "Stable"

    # Historical daily baseline anomaly (requires >= 30 valid ERA5 daily entries & valid 24h sum)
    baseline_daily_mean_mm: float | None = None
    anomaly_24h_mm: float | None = None
    anomaly_ratio: float | None = None
    anomaly_status = "Insufficient Baseline"
    baseline_days_count = 0

    if baseline_daily:
        daily_sums = [
            float(v)
            for v in (baseline_daily.get("precipitation_sum", []) or [])
            if _is_valid_nonneg(v)
        ]
        baseline_days_count = len(daily_sums)
        if baseline_days_count >= 30:
            baseline_daily_mean_mm = round(sum(daily_sums) / baseline_days_count, 2)
            if r_24h is not None:
                anomaly_24h_mm = round(r_24h - baseline_daily_mean_mm, 2)
                if baseline_daily_mean_mm > 0:
                    anomaly_ratio = round(r_24h / baseline_daily_mean_mm, 2)
                if anomaly_24h_mm > 5.0:
                    anomaly_status = "Above Historical Daily Mean"
                elif anomaly_24h_mm < -2.0:
                    anomaly_status = "Below Historical Daily Mean"
                else:
                    anomaly_status = "Near Historical Daily Mean"
            else:
                anomaly_status = "Insufficient Recent 24h Data"

    # Separate forecast totals (strictly labelled as forecast, never observation)
    fcst_precip = list((forecast_hourly or {}).get("precipitation", []) or [])
    fcst_valid = [float(v) for v in fcst_precip if _is_valid_nonneg(v)]
    fcst_missing = len(fcst_precip) - len(fcst_valid)
    fcst_is_complete = bool(fcst_precip) and fcst_missing == 0
    if not fcst_precip:
        fcst_status = "Unavailable"
    elif fcst_missing == 0:
        fcst_status = "Complete"
    else:
        fcst_status = f"Incomplete ({fcst_missing} missing hours)"
    fcst_24h, _ = _strict_window_sum(fcst_precip[:24], 24) if len(fcst_precip) >= 24 else (None, 24)
    # Only compute aggregate forecast total when the forecast horizon has no missing slots
    fcst_total = round(sum(fcst_valid), 2) if fcst_is_complete else None

    return {
        "rainfall_1h_mm": r_1h,
        "rainfall_24h_mm": r_24h,
        "rainfall_48h_mm": r_48h,
        "rainfall_72h_mm": r_72h,
        "valid_hours_total": len(valid_hist_vals),
        "missing_hours_total": missing_total,
        "missing_hours_24h": miss_24,
        "missing_hours_48h": miss_48,
        "missing_hours_72h": miss_72,
        "trend_slope_mm_per_hr": trend_slope,
        "trend_label": trend_label,
        "baseline_daily_mean_mm": baseline_daily_mean_mm,
        "baseline_days_count": baseline_days_count,
        "anomaly_24h_mm": anomaly_24h_mm,
        "anomaly_ratio": anomaly_ratio,
        "anomaly_status": anomaly_status,
        "forecast_24h_mm": fcst_24h,
        "forecast_total_mm": fcst_total,
        "forecast_hours_count": len(fcst_precip),
        "forecast_valid_hours_count": len(fcst_valid),
        "forecast_missing_hours": fcst_missing,
        "forecast_is_complete": fcst_is_complete,
        "forecast_data_status": fcst_status,
    }


def build_hourly_series_dataframe(weather: dict[str, Any]) -> pd.DataFrame:
    """Build combined hourly DataFrame cleanly tagging Past Reanalysis vs Forecast."""
    rows: list[dict[str, Any]] = []
    for block_key, label, is_fcst in [
        ("historical_hourly", "Past Reanalysis (Model-Derived)", False),
        ("forecast_hourly", "Forecast (Predicted)", True),
    ]:
        block = weather.get(block_key) or {}
        times = block.get("time", []) or []
        precip = block.get("precipitation", []) or []
        temp = block.get("temperature_2m", []) or []
        rh = block.get("relative_humidity_2m", []) or []
        wind = block.get("wind_speed_10m", []) or []
        soil = block.get("soil_moisture_0_to_7cm", []) or []

        for i, t in enumerate(times):
            rows.append({
                "time": t,
                "precipitation_mm": precip[i] if i < len(precip) else None,
                "temperature_2m_c": temp[i] if i < len(temp) else None,
                "relative_humidity_2m_pct": rh[i] if i < len(rh) else None,
                "wind_speed_10m_kmh": wind[i] if i < len(wind) else None,
                "soil_moisture_0_to_7cm": soil[i] if i < len(soil) else None,
                "series_type": label,
                "is_forecast": is_fcst,
            })

    if not rows:
        # Fallback if caller provided flat 'hourly' only
        hourly = weather.get("hourly") or {}
        times = hourly.get("time", []) or []
        precip = hourly.get("precipitation", []) or []
        for i, t in enumerate(times):
            rows.append({
                "time": t,
                "precipitation_mm": precip[i] if i < len(precip) else None,
                "series_type": "Hourly (Unspecified Split)",
                "is_forecast": False,
            })

    return pd.DataFrame(rows)


def build_daily_series_dataframe(weather: dict[str, Any]) -> pd.DataFrame:
    """Build daily DataFrame separating past/current dates from forecast dates."""
    rows: list[dict[str, Any]] = []
    for block_key, label, is_fcst in [
        ("daily_historical", "Past / Current-Date (Reanalysis)", False),
        ("daily_forecast", "Forecast (Predicted)", True),
    ]:
        block = weather.get(block_key) or {}
        times = block.get("time", []) or []
        precip = block.get("precipitation_sum", []) or []
        tmax = block.get("temperature_2m_max", []) or []
        tmin = block.get("temperature_2m_min", []) or []

        for i, d in enumerate(times):
            rows.append({
                "date": d,
                "precipitation_sum_mm": precip[i] if i < len(precip) else None,
                "temperature_max_c": tmax[i] if i < len(tmax) else None,
                "temperature_min_c": tmin[i] if i < len(tmin) else None,
                "series_type": label,
                "is_forecast": is_fcst,
            })

    return pd.DataFrame(rows)


def _describe_influencing_factors(
    current_features: dict[str, Any],
    prev_features: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Explain which inputs changed or sit near elevated thresholds (non-causal)."""
    factors: list[dict[str, Any]] = []
    monitored = [
        ("rainfall_1h_mm", "1-Hour Rainfall", "mm", 10.0),
        ("rainfall_24h_mm", "24-Hour Accumulated Rainfall", "mm", 50.0),
        ("rainfall_48h_mm", "48-Hour Accumulated Rainfall", "mm", 80.0),
        ("rainfall_72h_mm", "72-Hour Accumulated Rainfall", "mm", 110.0),
        ("soil_moisture", "Model Soil Moisture (0-7cm)", "m³/m³", 0.40),
        ("slope_degrees", "Terrain Slope", "deg", 25.0),
    ]

    for key, label, unit, ref_threshold in monitored:
        cur_val = current_features.get(key)
        if cur_val is None or not isinstance(cur_val, (int, float)) or not math.isfinite(float(cur_val)):
            continue
        cur_f = float(cur_val)
        prev_val = prev_features.get(key) if prev_features else None
        delta = None
        if prev_val is not None and isinstance(prev_val, (int, float)) and math.isfinite(float(prev_val)):
            delta = round(cur_f - float(prev_val), 3)

        status = "Above Reference Threshold" if cur_f >= ref_threshold else "Within Reference Range"
        factors.append({
            "feature": key,
            "label": label,
            "current_value": round(cur_f, 3),
            "previous_value": round(float(prev_val), 3) if delta is not None else None,
            "delta": delta,
            "unit": unit,
            "reference_note": status,
        })

    # Prioritize features that changed most cleanly, then elevated values
    factors.sort(
        key=lambda item: (
            0 if item["delta"] is not None and abs(item["delta"]) > 0 else 1,
            -abs(item["delta"] or 0.0),
            -abs(item["current_value"]),
        )
    )
    return factors


def evaluate_rainfall_risk_change(
    features: dict[str, Any],
    observation_time: datetime | None = None,
    location_id: int | None = None,
    location_name: str | None = None,
    source_label: str = "Open-Meteo Model-Derived Reanalysis",
    is_manual: bool = False,
    previous_prediction: dict[str, Any] | None = None,
    risk_engine=None,
    persist: bool = True,
) -> dict[str, Any]:
    """Validate rainfall inputs, compute experimental risk, persist, and explain change."""
    if previous_prediction is None and location_id is not None:
        previous_prediction = get_latest_prediction(location_id)

    prev_features: dict[str, Any] | None = None
    if previous_prediction and previous_prediction.get("input_features"):
        raw_prev = previous_prediction["input_features"]
        if isinstance(raw_prev, str):
            try:
                prev_features = json.loads(raw_prev)
            except json.JSONDecodeError:
                prev_features = None
        elif isinstance(raw_prev, dict):
            prev_features = raw_prev

    now_utc = datetime.now(timezone.utc)
    if observation_time is None:
        freshness_status = "Manual User Input (Time Assumed Current)" if is_manual else "Timestamp Missing (Assumed Current)"
        age_hours = 0.0
    else:
        obs_utc = (
            observation_time.replace(tzinfo=timezone.utc)
            if observation_time.tzinfo is None
            else observation_time.astimezone(timezone.utc)
        )
        age_hours = round(max(0.0, (now_utc - obs_utc).total_seconds() / 3600.0), 2)
        if is_manual:
            freshness_status = "User-Provided Data (Manual Fallback)"
        elif age_hours <= FRESHNESS_HOURS:
            freshness_status = f"Fresh ({age_hours:.1f}h old)"
        else:
            freshness_status = f"Stale ({age_hours:.1f}h old; threshold {FRESHNESS_HOURS}h)"

    engine = risk_engine or get_risk_engine()
    result = engine.estimate_risk(
        features=features,
        observation_time=observation_time,
        location_name=location_name,
    )

    warnings = list(result.get("warnings", []))
    errors = list(result.get("errors", []))
    if is_manual:
        warnings.append("Manual Fallback Input: Values are user-provided, NOT live telemetry.")

    current_prob = result.get("probability")
    current_cat = result.get("risk_category", "Unknown")

    prediction_id = None
    if persist and not errors:
        model_name = get_setting("model", "active_model") or "best_model"
        prediction_id = save_prediction(
            probability=current_prob,
            risk_category=current_cat,
            model_name=model_name,
            method=result.get("method", "unknown"),
            input_features=features,
            warnings=warnings,
            location_id=location_id,
            is_experimental=True,
        )

    prev_prob = previous_prediction.get("probability") if previous_prediction else None
    prev_cat = previous_prediction.get("risk_category") if previous_prediction else None

    prob_delta = (
        round(float(current_prob) - float(prev_prob), 4)
        if (current_prob is not None and prev_prob is not None)
        else None
    )

    if prev_cat is None:
        category_change = f"Initial Assessment ({current_cat})"
    elif prev_cat == current_cat:
        category_change = f"Unchanged ({current_cat})"
    else:
        category_change = f"{prev_cat} -> {current_cat}"

    available_rainfall = {
        k: features.get(k)
        for k in ("rainfall_1h_mm", "rainfall_24h_mm", "rainfall_48h_mm", "rainfall_72h_mm")
    }

    return {
        "prediction_id": prediction_id,
        "location_id": location_id,
        "location_name": location_name,
        "source_label": source_label,
        "is_manual_input": bool(is_manual),
        "is_experimental": True,
        "previous_estimate": {
            "probability": round(float(prev_prob), 4) if prev_prob is not None else None,
            "risk_category": prev_cat,
            "predicted_at": previous_prediction.get("predicted_at") if previous_prediction else None,
        },
        "current_estimate": {
            "probability": round(float(current_prob), 4) if current_prob is not None else None,
            "risk_category": current_cat,
            "method": result.get("method"),
            "predicted_at": now_utc.isoformat(),
        },
        "probability_delta": prob_delta,
        "category_change": category_change,
        "data_freshness": {
            "status": freshness_status,
            "age_hours": age_hours,
            "observation_time": observation_time.isoformat() if observation_time else None,
        },
        "available_rainfall_mm": available_rainfall,
        "influencing_factors": _describe_influencing_factors(features, prev_features),
        "warnings": warnings,
        "errors": errors,
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
    }

