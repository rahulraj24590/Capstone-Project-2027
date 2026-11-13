"""
Phase 6 Environmental Analytics & Experimental Risk Trend Forecasting for LandslideGuard.

Implements:
1. Rainfall anomaly detection against real historical archive baselines:
   - Empirical Z-score and quantile ranking
   - Rolling 7-day statistical mean / standard-deviation threshold envelope
   - Unsupervised Isolation Forest outlier detection (when >= 60 baseline days exist)
   - Transparent 4-tier classification (Normal, Above-normal, Unusual, Extreme)
   - Strict refusal to classify anomalies using fabricated baselines
2. Multi-scale rainfall trend analysis (daily, weekly, monthly, seasonal, historical)
   with explicit temporal coverage thresholds and zero-imputation prohibited
3. 24h / 48h / 72h experimental risk trend forecasting driven by actual weather
   forecast inputs, explicit forecast-input attribution, lead-time uncertainty bands,
   and explicit disclosure that exact landslide timing cannot be predicted
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.model_registry import active_model_exists
from src.monitoring.rainfall_analysis import (
    SCIENTIFIC_DISCLAIMER,
    _compute_ols_slope,
    _is_valid_nonneg,
    _strict_window_sum,
)
from src.risk.risk_engine import get_risk_engine
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

FORECAST_TIMING_DISCLAIMER = (
    "Experimental forecast trajectory only: indicates whether projected meteorological inputs "
    "align with higher or lower model risk scores across 24h/48h/72h windows. "
    "This system CANNOT predict exact landslide timing or failure onset, and trained ML weights "
    "currently originate from synthetic development data rather than field-validated events."
)

_IMD_SEASONS: list[tuple[str, set[int]]] = [
    ("Winter (JF)", {1, 2}),
    ("Pre-Monsoon (MAM)", {3, 4, 5}),
    ("Monsoon (JJAS)", {6, 7, 8, 9}),
    ("Post-Monsoon (OND)", {10, 11, 12}),
]


def _extract_valid_daily_series(
    baseline_daily: dict[str, Any] | None,
) -> tuple[list[str], list[float], int]:
    """Extract aligned valid non-negative (date, precipitation_mm) pairs without zero-imputation."""
    if not isinstance(baseline_daily, dict) or not baseline_daily:
        return [], [], 0

    raw_precip = list(baseline_daily.get("precipitation_sum") or [])
    raw_times = list(baseline_daily.get("time") or [])
    dates: list[str] = []
    values: list[float] = []
    invalid_count = 0

    for idx, raw_val in enumerate(raw_precip):
        if not _is_valid_nonneg(raw_val):
            invalid_count += 1
            continue
        dt_str = (
            str(raw_times[idx])
            if idx < len(raw_times) and raw_times[idx] is not None
            else f"day-{idx + 1}"
        )
        dates.append(dt_str)
        values.append(float(raw_val))

    if dates and len(dates) == len(values) and not str(dates[0]).startswith("day-"):
        paired = sorted(zip(dates, values), key=lambda pair: pair[0])
        dates = [p[0] for p in paired]
        values = [p[1] for p in paired]

    return dates, values, invalid_count


def _classify_anomaly_tier(
    recent_mm: float,
    z_score: float | None,
    p90: float,
    p95: float,
    p99: float,
    iso_anomaly: bool | None,
    imd_very_heavy_mm: float,
    z_above: float,
    z_unusual: float,
    z_extreme: float,
    baseline_days_count: int = 0,
) -> tuple[str, list[str]]:
    """Map empirical baseline comparisons deterministically into the 4 documented classes."""
    reasons: list[str] = []
    # Require either z >= 3.0, documented IMD Very Heavy (>=115.6 mm/24h & >=P95),
    # or empirical P99 backed by >= 180 historical archive days for Extreme classification.
    extreme_by_p99 = (
        baseline_days_count >= 180 and p99 > 0.0 and recent_mm >= p99 and recent_mm > 0.0
    )
    if (
        (z_score is not None and z_score >= z_extreme)
        or extreme_by_p99
        or (recent_mm >= imd_very_heavy_mm and recent_mm >= max(p95, 1.0))
    ):
        if z_score is not None and z_score >= z_extreme:
            reasons.append(f"Z-score {z_score:.2f} >= {z_extreme:.1f} sigma")
        if extreme_by_p99:
            reasons.append(f"Exceeds empirical archive 99th percentile ({p99:.1f} mm/day)")
        if recent_mm >= imd_very_heavy_mm:
            reasons.append(f"Exceeds documented IMD Very Heavy threshold ({imd_very_heavy_mm:.1f} mm/24h)")
        return "Extreme rainfall", reasons

    if (
        (z_score is not None and z_score >= z_unusual)
        or (p95 > 0.0 and recent_mm >= p95)
        or (bool(iso_anomaly) and (z_score is None or z_score >= 1.5))
    ):
        if z_score is not None and z_score >= z_unusual:
            reasons.append(f"Z-score {z_score:.2f} in [{z_unusual:.1f}, {z_extreme:.1f}) sigma")
        if p95 > 0.0 and recent_mm >= p95:
            reasons.append(f"Exceeds empirical archive 95th percentile ({p95:.1f} mm/day)")
        if iso_anomaly:
            reasons.append("Flagged as multivariate outlier by Isolation Forest on historical baseline")
        return "Unusual rainfall", reasons

    if (
        (z_score is not None and z_score >= z_above)
        or (p90 > 0.0 and recent_mm >= p90)
        or (z_score is None and recent_mm > 0.0)
    ):
        if z_score is not None and z_score >= z_above:
            reasons.append(f"Z-score {z_score:.2f} in [{z_above:.1f}, {z_unusual:.1f}) sigma")
        elif z_score is None and recent_mm > 0.0:
            reasons.append("Non-zero rainfall relative to zero-variance dry historical baseline")
        if p90 > 0.0 and recent_mm >= p90:
            reasons.append(f"Exceeds empirical archive 90th percentile ({p90:.1f} mm/day)")
        return "Above-normal rainfall", reasons

    reasons.append("Within normal empirical historical variation")
    return "Normal rainfall", reasons


def detect_rainfall_anomalies(
    recent_24h_mm: float | None,
    baseline_daily: dict[str, Any] | None,
    rolling_window_days: int | None = None,
) -> dict[str, Any]:
    """Detect rainfall anomalies strictly from authentic historical daily baselines.

    Uses Z-score, empirical quantiles, rolling statistical envelopes, and
    Isolation Forest (when sufficient variance & >= 60 baseline days exist).
    Never invents or simulates baselines when archive records are insufficient.
    """
    cfg = get_setting("analytics", "anomaly") or {}
    min_days = int(cfg.get("min_baseline_days", 30))
    min_if_days = int(cfg.get("min_isolation_forest_days", 60))
    z_above = float(cfg.get("z_above_normal", 1.0))
    z_unusual = float(cfg.get("z_unusual", 2.0))
    z_extreme = float(cfg.get("z_extreme", 3.0))
    imd_very_heavy = float(cfg.get("imd_very_heavy_mm", 115.6))
    contamination = float(cfg.get("isolation_forest_contamination", 0.05))
    roll_win = int(rolling_window_days or cfg.get("rolling_window_days", 7))

    dates, values, invalid_days = _extract_valid_daily_series(baseline_daily)
    n_valid = len(values)

    if n_valid < min_days:
        return {
            "baseline_available": False,
            "baseline_days_count": n_valid,
            "min_baseline_days_required": min_days,
            "invalid_baseline_days": invalid_days,
            "recent_24h_mm": float(recent_24h_mm) if _is_valid_nonneg(recent_24h_mm) else None,
            "classification": "Unavailable (Insufficient Historical Baseline)",
            "status": "Insufficient Historical Baseline",
            "z_score": None,
            "percentile_rank": None,
            "rolling_mean_mm": None,
            "rolling_upper_threshold_mm": None,
            "above_rolling_threshold": None,
            "isolation_forest_available": False,
            "isolation_forest_anomaly": None,
            "isolation_forest_score": None,
            "baseline_stats": None,
            "reasons": [
                f"Historical baseline has {n_valid} valid days; at least {min_days} days required before anomaly classification."
            ],
            "history_series": [],
        }

    arr = np.asarray(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n_valid > 1 else 0.0
    median_val = float(np.median(arr))
    p75, p90, p95, p99 = [float(x) for x in np.percentile(arr, [75, 90, 95, 99])]
    iqr_val = float(p75 - np.percentile(arr, 25))
    max_val = float(np.max(arr))

    # Rolling statistical thresholds across available historical baseline
    s = pd.Series(arr)
    min_p = max(3, min(roll_win, n_valid))
    roll_mean_s = s.rolling(window=roll_win, min_periods=min_p).mean()
    roll_std_s = s.rolling(window=roll_win, min_periods=min_p).std(ddof=1).fillna(0.0)
    roll_upper_s = roll_mean_s + z_unusual * roll_std_s

    recent_rolling_mean = float(np.mean(arr[-roll_win:]))
    recent_rolling_std = float(np.std(arr[-roll_win:], ddof=1)) if len(arr[-roll_win:]) > 1 else 0.0
    rolling_upper_thresh = recent_rolling_mean + z_unusual * max(recent_rolling_std, std_val * 0.5)

    baseline_stats = {
        "mean_mm": round(mean_val, 3),
        "std_mm": round(std_val, 3),
        "median_mm": round(median_val, 3),
        "iqr_mm": round(iqr_val, 3),
        "p75_mm": round(p75, 3),
        "p90_mm": round(p90, 3),
        "p95_mm": round(p95, 3),
        "p99_mm": round(p99, 3),
        "max_mm": round(max_val, 3),
        "zero_variance": bool(std_val == 0.0),
    }

    # Build unsupervised Isolation Forest on empirical historical series if large enough
    iso_available = False
    iso_model: IsolationForest | None = None
    hist_iso_flags = [False] * n_valid
    if n_valid >= min_if_days and std_val > 1e-6:
        try:
            roll_feats = roll_mean_s.fillna(mean_val).to_numpy(dtype=float)
            x_hist = np.column_stack([arr, roll_feats])
            iso_model = IsolationForest(
                contamination=min(0.25, max(0.01, contamination)),
                random_state=42,
            )
            preds = iso_model.fit_predict(x_hist)
            hist_iso_flags = [bool(p == -1) for p in preds]
            iso_available = True
        except Exception as exc:
            logger.warning("Isolation Forest skipped cleanly: %s", exc)
            iso_available = False

    history_series: list[dict[str, Any]] = []
    for idx in range(n_valid):
        val = float(arr[idx])
        z_pt = round((val - mean_val) / std_val, 3) if std_val > 1e-9 else None
        rm_val = float(roll_mean_s.iloc[idx]) if pd.notna(roll_mean_s.iloc[idx]) else mean_val
        ru_val = float(roll_upper_s.iloc[idx]) if pd.notna(roll_upper_s.iloc[idx]) else rolling_upper_thresh
        history_series.append({
            "date": dates[idx],
            "precipitation_mm": round(val, 2),
            "rolling_mean_mm": round(rm_val, 2),
            "rolling_upper_threshold_mm": round(ru_val, 2),
            "z_score": z_pt,
            "is_statistical_outlier": bool((z_pt is not None and z_pt >= z_unusual) or hist_iso_flags[idx]),
        })

    if not _is_valid_nonneg(recent_24h_mm):
        return {
            "baseline_available": True,
            "baseline_days_count": n_valid,
            "min_baseline_days_required": min_days,
            "invalid_baseline_days": invalid_days,
            "recent_24h_mm": None,
            "classification": "Unavailable (Insufficient Recent 24h Rainfall)",
            "status": "Insufficient Recent 24h Data",
            "z_score": None,
            "percentile_rank": None,
            "rolling_mean_mm": round(recent_rolling_mean, 2),
            "rolling_upper_threshold_mm": round(rolling_upper_thresh, 2),
            "above_rolling_threshold": None,
            "isolation_forest_available": iso_available,
            "isolation_forest_anomaly": None,
            "isolation_forest_score": None,
            "baseline_stats": baseline_stats,
            "reasons": [
                "Historical baseline exists, but recent 24h rainfall total is unavailable (missing values are never zero-filled)."
            ],
            "history_series": history_series,
        }

    recent_f = float(recent_24h_mm)
    z_score = round((recent_f - mean_val) / std_val, 3) if std_val > 1e-9 else (0.0 if recent_f == mean_val else None)
    percentile_rank = round(float(np.mean(arr <= recent_f) * 100.0), 1)
    above_rolling = bool(recent_f > rolling_upper_thresh)

    iso_anomaly: bool | None = None
    iso_score: float | None = None
    if iso_available and iso_model is not None:
        # Recent 7-day average incorporating latest observation
        tail_vals = list(arr[-(roll_win - 1) :]) + [recent_f] if roll_win > 1 else [recent_f]
        x_now = np.array([[recent_f, float(np.mean(tail_vals))]], dtype=float)
        iso_anomaly = bool(iso_model.predict(x_now)[0] == -1)
        iso_score = round(float(iso_model.decision_function(x_now)[0]), 4)

    classification, reasons = _classify_anomaly_tier(
        recent_mm=recent_f,
        z_score=z_score,
        p90=p90,
        p95=p95,
        p99=p99,
        iso_anomaly=iso_anomaly,
        imd_very_heavy_mm=imd_very_heavy,
        z_above=z_above,
        z_unusual=z_unusual,
        z_extreme=z_extreme,
        baseline_days_count=n_valid,
    )
    if above_rolling:
        reasons.append(
            f"Exceeds {roll_win}-day rolling upper threshold ({rolling_upper_thresh:.1f} mm/day)"
        )

    return {
        "baseline_available": True,
        "baseline_days_count": n_valid,
        "min_baseline_days_required": min_days,
        "invalid_baseline_days": invalid_days,
        "recent_24h_mm": round(recent_f, 2),
        "classification": classification,
        "status": "Computed from Historical Archive",
        "z_score": z_score,
        "percentile_rank": percentile_rank,
        "rolling_mean_mm": round(recent_rolling_mean, 2),
        "rolling_upper_threshold_mm": round(rolling_upper_thresh, 2),
        "above_rolling_threshold": above_rolling,
        "isolation_forest_available": iso_available,
        "isolation_forest_anomaly": iso_anomaly,
        "isolation_forest_score": iso_score,
        "baseline_stats": baseline_stats,
        "reasons": reasons,
        "history_series": history_series,
    }


def _classify_slope_direction(slope: float | None, threshold: float = 0.05) -> str:
    if slope is None:
        return "Insufficient Data"
    if slope > threshold:
        return "Increasing"
    if slope < -threshold:
        return "Decreasing"
    return "Stable"


def analyze_rainfall_trends(
    baseline_daily: dict[str, Any] | None,
    daily_historical: dict[str, Any] | None = None,
    recent_24h_mm: float | None = None,
) -> dict[str, Any]:
    """Compute daily, weekly, monthly, historical, and seasonal rainfall trends."""
    cfg = get_setting("analytics", "trends") or {}
    min_week_days = int(cfg.get("min_weekly_valid_days", 5))
    min_month_days = int(cfg.get("min_monthly_valid_days", 20))
    min_season_days = int(cfg.get("min_seasonal_valid_days", 15))
    roll_win = int(cfg.get("rolling_window_days", 7))

    # Merge baseline_daily and daily_historical without duplicates (latest source wins per date)
    date_map: dict[str, float] = {}
    invalid_days = 0
    for block in (baseline_daily, daily_historical):
        d_list, v_list, inv = _extract_valid_daily_series(block)
        invalid_days += inv
        for dt_str, val in zip(d_list, v_list):
            if not dt_str.startswith("day-"):
                date_map[dt_str] = val

    if not date_map:
        return {
            "has_data": False,
            "valid_days_count": 0,
            "invalid_days_skipped": invalid_days,
            "daily_trend": {"status": "Insufficient Data", "slope_mm_per_day": None, "records": []},
            "weekly_trend": {"status": "Insufficient Data", "slope_mm_per_week": None, "records": []},
            "monthly_trend": {"status": "Insufficient Data", "slope_mm_per_month": None, "records": []},
            "seasonal_comparison": [],
            "historical_comparison": {"baseline_available": False, "recent_24h_mm": recent_24h_mm},
        }

    df = pd.DataFrame(
        [{"date_str": k, "precipitation_mm": v} for k, v in sorted(date_map.items())]
    )
    df["date"] = pd.to_datetime(df["date_str"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        return {
            "has_data": False,
            "valid_days_count": 0,
            "invalid_days_skipped": invalid_days,
            "daily_trend": {"status": "Insufficient Data", "slope_mm_per_day": None, "records": []},
            "weekly_trend": {"status": "Insufficient Data", "slope_mm_per_week": None, "records": []},
            "monthly_trend": {"status": "Insufficient Data", "slope_mm_per_month": None, "records": []},
            "seasonal_comparison": [],
            "historical_comparison": {"baseline_available": False, "recent_24h_mm": recent_24h_mm},
        }

    # 1. Daily trend + rolling 7-day average
    df["rolling_7d_mean_mm"] = (
        df["precipitation_mm"].rolling(window=roll_win, min_periods=max(2, min(roll_win, len(df)))).mean().round(2)
    )
    daily_slope = _compute_ols_slope(df["precipitation_mm"].tolist())
    daily_records = [
        {
            "date": row["date_str"],
            "precipitation_mm": round(float(row["precipitation_mm"]), 2),
            "rolling_7d_mean_mm": (
                round(float(row["rolling_7d_mean_mm"]), 2)
                if pd.notna(row["rolling_7d_mean_mm"])
                else None
            ),
        }
        for _, row in df.iterrows()
    ]

    # 2. Weekly trend (require >= min_week_days valid days per ISO week)
    iso = df["date"].dt.isocalendar()
    df["iso_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    wk_grouped = (
        df.groupby("iso_week", sort=True)
        .agg(
            valid_days=("precipitation_mm", "count"),
            total_mm=("precipitation_mm", "sum"),
            mean_daily_mm=("precipitation_mm", "mean"),
        )
        .reset_index()
    )
    complete_weeks = wk_grouped[wk_grouped["valid_days"] >= min_week_days].copy()
    weekly_slope = (
        _compute_ols_slope(complete_weeks["total_mm"].tolist())
        if len(complete_weeks) >= 6
        else None
    )
    weekly_records = [
        {
            "week": str(r["iso_week"]),
            "valid_days": int(r["valid_days"]),
            "total_mm": round(float(r["total_mm"]), 2),
            "mean_daily_mm": round(float(r["mean_daily_mm"]), 2),
        }
        for _, r in complete_weeks.iterrows()
    ]

    # 3. Monthly trend (require >= min_month_days valid days per month)
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    mo_grouped = (
        df.groupby("year_month", sort=True)
        .agg(
            valid_days=("precipitation_mm", "count"),
            total_mm=("precipitation_mm", "sum"),
            mean_daily_mm=("precipitation_mm", "mean"),
            max_daily_mm=("precipitation_mm", "max"),
        )
        .reset_index()
    )
    complete_months = mo_grouped[mo_grouped["valid_days"] >= min_month_days].copy()
    monthly_slope = (
        _compute_ols_slope(complete_months["total_mm"].tolist())
        if len(complete_months) >= 6
        else None
    )
    monthly_records = [
        {
            "month": str(r["year_month"]),
            "valid_days": int(r["valid_days"]),
            "total_mm": round(float(r["total_mm"]), 2),
            "mean_daily_mm": round(float(r["mean_daily_mm"]), 2),
            "max_daily_mm": round(float(r["max_daily_mm"]), 2),
        }
        for _, r in complete_months.iterrows()
    ]

    # 4. Seasonal comparison (IMD seasons with explicit sufficiency check)
    df["month_num"] = df["date"].dt.month
    seasonal_rows: list[dict[str, Any]] = []
    for season_label, month_set in _IMD_SEASONS:
        sub = df[df["month_num"].isin(month_set)]["precipitation_mm"]
        n_s = int(len(sub))
        if n_s >= min_season_days:
            seasonal_rows.append({
                "season": season_label,
                "valid_days": n_s,
                "sufficient_coverage": True,
                "status": "Available",
                "mean_daily_mm": round(float(sub.mean()), 2),
                "total_mm": round(float(sub.sum()), 2),
                "max_daily_mm": round(float(sub.max()), 2),
                "p90_daily_mm": round(float(np.percentile(sub.to_numpy(dtype=float), 90)), 2),
            })
        else:
            seasonal_rows.append({
                "season": season_label,
                "valid_days": n_s,
                "sufficient_coverage": False,
                "status": f"Insufficient Seasonal Coverage ({n_s}/{min_season_days} valid days)",
                "mean_daily_mm": None,
                "total_mm": None,
                "max_daily_mm": None,
                "p90_daily_mm": None,
            })

    # 5. Historical comparison summary
    arr = df["precipitation_mm"].to_numpy(dtype=float)
    latest_7d_mm = (
        round(float(df["precipitation_mm"].tail(7).sum()), 2)
        if len(df) >= 7
        else None
    )
    hist_comp = {
        "baseline_available": len(df) >= 30,
        "valid_days": int(len(df)),
        "archive_mean_daily_mm": round(float(np.mean(arr)), 2),
        "archive_p50_mm": round(float(np.percentile(arr, 50)), 2),
        "archive_p90_mm": round(float(np.percentile(arr, 90)), 2),
        "archive_p95_mm": round(float(np.percentile(arr, 95)), 2),
        "recent_7d_mm": latest_7d_mm,
        "recent_24h_mm": round(float(recent_24h_mm), 2) if _is_valid_nonneg(recent_24h_mm) else None,
        "recent_vs_archive_ratio": (
            round(float(recent_24h_mm) / float(np.mean(arr)), 2)
            if _is_valid_nonneg(recent_24h_mm) and float(np.mean(arr)) > 0
            else None
        ),
    }

    return {
        "has_data": True,
        "valid_days_count": int(len(df)),
        "invalid_days_skipped": invalid_days,
        "date_range": {"start": str(df["date_str"].iloc[0]), "end": str(df["date_str"].iloc[-1])},
        "daily_trend": {
            "status": _classify_slope_direction(daily_slope, threshold=0.05),
            "slope_mm_per_day": round(daily_slope, 4) if daily_slope is not None else None,
            "records": daily_records,
        },
        "weekly_trend": {
            "status": _classify_slope_direction(weekly_slope, threshold=0.5),
            "slope_mm_per_week": round(weekly_slope, 3) if weekly_slope is not None else None,
            "complete_weeks": len(weekly_records),
            "skipped_incomplete_weeks": int(len(wk_grouped) - len(complete_weeks)),
            "records": weekly_records,
        },
        "monthly_trend": {
            "status": _classify_slope_direction(monthly_slope, threshold=1.0),
            "slope_mm_per_month": round(monthly_slope, 3) if monthly_slope is not None else None,
            "complete_months": len(monthly_records),
            "skipped_incomplete_months": int(len(mo_grouped) - len(complete_months)),
            "records": monthly_records,
        },
        "seasonal_comparison": seasonal_rows,
        "historical_comparison": hist_comp,
    }


def forecast_risk_trend(
    weather_payload: dict[str, Any] | None,
    terrain_features: dict[str, Any] | None = None,
    risk_engine=None,
) -> dict[str, Any]:
    """Estimate experimental 24h / 48h / 72h risk trends from actual forecast weather inputs.

    Strict guarantees:
    - Uses ONLY actual non-missing weather forecast precipitation & soil moisture.
    - Combines trailing historical reanalysis only for the overlapping 72h antecedent tail
      (clearly attributed per-horizon as historical vs forecast-derived).
    - Computes symmetric forecast uncertainty bounds that widen with lead time.
    - Returns explicit Unavailable status if forecast horizon hours are missing or invalid.
    - Explicitly discloses synthetic training basis and inability to predict failure timing.
    """
    cfg = get_setting("analytics", "forecast_risk") or {}
    horizons = [int(h) for h in (cfg.get("horizons_hours") or [24, 48, 72])]
    step_hours = max(1, int(cfg.get("step_hours", 6)))
    sens_pct = float(cfg.get("rainfall_sensitivity_pct", 0.25))

    terrain = dict(terrain_features or {})
    if weather_payload and terrain.get("elevation_m") is None:
        terrain["elevation_m"] = weather_payload.get("elevation_m")

    model_exists = active_model_exists()
    model_is_synthetic = True
    model_name = "heuristic_baseline"
    if model_exists:
        model_name = "best_model"
        # Dataset inventory / README confirms current artifacts are trained on synthetic data
        ds_path = get_setting("paths", "data_processed") or "data/processed"
        try:
            ds_file = f"{ds_path}/dataset.parquet"
            sample_syn = pd.read_parquet(ds_file, columns=["is_synthetic"])
            if not sample_syn.empty and (sample_syn["is_synthetic"] == 0).all():
                model_is_synthetic = False
        except Exception:
            model_is_synthetic = True

    limitations = [
        FORECAST_TIMING_DISCLAIMER,
        "Meteorological forecast uncertainty grows with lead time (+24h -> +48h -> +72h).",
        "Soil moisture from Open-Meteo is model-derived surface layer (0-7cm), not an in-situ piezometer reading.",
    ]
    if model_is_synthetic:
        limitations.insert(
            0,
            "Active ML model ('best_model') was trained on synthetic development data and is NOT field-calibrated.",
        )

    if not isinstance(weather_payload, dict):
        return {
            "status": "Unavailable (No Weather Payload)",
            "model_name": model_name,
            "model_available": model_exists,
            "model_is_synthetic_dev": model_is_synthetic,
            "horizons": {
                f"{h}h": {"horizon_hours": h, "status": "Unavailable", "probability": None}
                for h in horizons
            },
            "trajectory": [],
            "forecast_derived_inputs": [],
            "limitations": limitations,
            "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
        }

    hist_blk = weather_payload.get("historical_hourly") or {}
    fcst_blk = weather_payload.get("forecast_hourly") or {}
    hist_precip = list(hist_blk.get("precipitation") or [])
    fcst_precip = list(fcst_blk.get("precipitation") or [])
    fcst_times = list(fcst_blk.get("time") or [])
    fcst_soil = list(fcst_blk.get("soil_moisture_0_to_7cm") or [])
    hist_soil = list(hist_blk.get("soil_moisture_0_to_7cm") or [])

    if not fcst_precip:
        return {
            "status": "Unavailable (No Forecast Data)",
            "model_name": model_name,
            "model_available": model_exists,
            "model_is_synthetic_dev": model_is_synthetic,
            "horizons": {
                f"{h}h": {
                    "horizon_hours": h,
                    "status": "Unavailable (Empty Forecast Stream)",
                    "probability": None,
                }
                for h in horizons
            },
            "trajectory": [],
            "forecast_derived_inputs": [],
            "limitations": limitations,
            "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
        }

    engine = risk_engine or get_risk_engine()
    hist_tail = hist_precip[-72:]
    combined_precip = hist_tail + fcst_precip
    hist_offset = len(hist_tail)

    # Identify which steps to evaluate (every `step_hours` up to max horizon, plus target horizons)
    max_h = max(horizons)
    eval_steps = sorted(
        {h for h in range(step_hours, max_h + 1, step_hours)} | set(horizons)
    )

    trajectory: list[dict[str, Any]] = []
    horizon_summary: dict[str, dict[str, Any]] = {}

    for h in eval_steps:
        horizon_key = f"{h}h"
        if len(fcst_precip) < h:
            rec = {
                "horizon_hours": h,
                "forecast_time": None,
                "status": f"Unavailable (Only {len(fcst_precip)}/{h} forecast hours provided)",
                "probability": None,
                "lower_probability": None,
                "upper_probability": None,
                "uncertainty_width": None,
                "risk_category": "Unknown",
                "method": "none",
                "forecast_rainfall_sum_mm": None,
                "rolling_24h_mm": None,
                "rolling_72h_mm": None,
                "forecast_derived_inputs": [],
                "warnings": [f"Forecast stream shorter than +{h}h horizon."],
            }
            trajectory.append(rec)
            if h in horizons:
                horizon_summary[horizon_key] = rec
            continue

        # Strictly require zero missing forecast slots from 0..h
        fcst_slice = fcst_precip[:h]
        fcst_sum, fcst_missing = _strict_window_sum(fcst_slice, h)
        if fcst_sum is None:
            rec = {
                "horizon_hours": h,
                "forecast_time": fcst_times[h - 1] if h - 1 < len(fcst_times) else None,
                "status": f"Unavailable (Incomplete Forecast: {fcst_missing} missing hours)",
                "probability": None,
                "lower_probability": None,
                "upper_probability": None,
                "uncertainty_width": None,
                "risk_category": "Unknown",
                "method": "none",
                "forecast_rainfall_sum_mm": None,
                "rolling_24h_mm": None,
                "rolling_72h_mm": None,
                "forecast_derived_inputs": [],
                "warnings": [f"Cannot evaluate +{h}h risk with {fcst_missing} missing forecast hour(s)."],
            }
            trajectory.append(rec)
            if h in horizons:
                horizon_summary[horizon_key] = rec
            continue

        end_idx = hist_offset + h
        combined_up_to_h = combined_precip[:end_idx]
        r_1h = round(float(fcst_slice[-1]), 2)
        r_24h, _ = _strict_window_sum(combined_up_to_h, 24)
        r_72h, _ = _strict_window_sum(combined_up_to_h, 72)

        # Latest valid soil moisture up to +h (forecast preferred, then recent historical)
        soil_val: float | None = None
        soil_is_fcst = False
        valid_fcst_soil = [float(v) for v in fcst_soil[:h] if _is_valid_nonneg(v)]
        if valid_fcst_soil:
            soil_val = valid_fcst_soil[-1]
            soil_is_fcst = True
        else:
            valid_hist_soil = [float(v) for v in hist_soil if _is_valid_nonneg(v)]
            if valid_hist_soil:
                soil_val = valid_hist_soil[-1]

        features = {
            "slope_degrees": terrain.get("slope_degrees"),
            "elevation_m": terrain.get("elevation_m"),
            "vegetation_index": terrain.get("vegetation_index"),
            "rainfall_1h_mm": r_1h,
            "rainfall_24h_mm": r_24h,
            "rainfall_72h_mm": r_72h,
            "soil_moisture": soil_val,
        }

        nominal_res = engine.estimate_risk(features=features)
        nom_prob = nominal_res.get("probability")
        warnings = list(nominal_res.get("warnings") or [])

        fcst_inputs = ["rainfall_1h_mm (forecast)"]
        if h >= 24 and r_24h is not None:
            fcst_inputs.append("rainfall_24h_mm (100% forecast)")
        elif r_24h is not None:
            fcst_inputs.append("rainfall_24h_mm (historical antecedent + forecast)")
        if h >= 72 and r_72h is not None:
            fcst_inputs.append("rainfall_72h_mm (100% forecast)")
        elif r_72h is not None:
            fcst_inputs.append("rainfall_72h_mm (historical antecedent + forecast)")
        if soil_is_fcst:
            fcst_inputs.append("soil_moisture (model-derived forecast)")

        # Lead-time widening rainfall sensitivity band for honest uncertainty bounds
        lead_scale = 1.0 + 0.25 * max(0.0, (h / 24.0) - 1.0)
        eff_band = min(0.60, sens_pct * lead_scale)
        lower_prob: float | None = None
        upper_prob: float | None = None
        uncertainty_width: float | None = None

        if nom_prob is not None:
            low_feats = dict(features)
            high_feats = dict(features)
            for rf_key in ("rainfall_1h_mm", "rainfall_24h_mm", "rainfall_72h_mm"):
                if features[rf_key] is not None:
                    low_feats[rf_key] = round(max(0.0, float(features[rf_key]) * (1.0 - eff_band)), 4)
                    high_feats[rf_key] = round(float(features[rf_key]) * (1.0 + eff_band), 4)
            low_res = engine.estimate_risk(features=low_feats)
            high_res = engine.estimate_risk(features=high_feats)
            p_lo = low_res.get("probability")
            p_hi = high_res.get("probability")
            candidates = [p for p in (p_lo, nom_prob, p_hi) if p is not None]
            if candidates:
                # Guarantee non-zero minimum uncertainty spread reflecting lead time even at saturation
                min_spread = 0.02 * lead_scale
                raw_lo = min(candidates)
                raw_hi = max(candidates)
                if raw_hi - raw_lo < min_spread:
                    lower_prob = round(max(0.0, float(nom_prob) - min_spread / 2.0), 4)
                    upper_prob = round(min(1.0, float(nom_prob) + min_spread / 2.0), 4)
                else:
                    lower_prob = round(float(raw_lo), 4)
                    upper_prob = round(float(raw_hi), 4)
                uncertainty_width = round(upper_prob - lower_prob, 4)

        status_str = (
            "Available (Experimental — Synthetic ML Model)"
            if (nom_prob is not None and nominal_res.get("method") == "ml_model" and model_is_synthetic)
            else ("Available (Experimental)" if nom_prob is not None else "Unavailable (Validation Error)")
        )

        rec = {
            "horizon_hours": h,
            "forecast_time": fcst_times[h - 1] if h - 1 < len(fcst_times) else None,
            "status": status_str,
            "probability": nom_prob,
            "lower_probability": lower_prob,
            "upper_probability": upper_prob,
            "uncertainty_width": uncertainty_width,
            "sensitivity_pct": round(eff_band * 100.0, 1),
            "risk_category": nominal_res.get("risk_category", "Unknown"),
            "method": nominal_res.get("method", "none"),
            "forecast_rainfall_sum_mm": fcst_sum,
            "rolling_24h_mm": r_24h,
            "rolling_72h_mm": r_72h,
            "soil_moisture": round(soil_val, 3) if soil_val is not None else None,
            "forecast_derived_inputs": fcst_inputs,
            "warnings": warnings,
            "errors": list(nominal_res.get("errors") or []),
        }
        trajectory.append(rec)
        if h in horizons:
            horizon_summary[horizon_key] = rec

    avail_count = sum(1 for v in horizon_summary.values() if v.get("probability") is not None)
    overall_status = (
        "Forecast Risk Available (Experimental)"
        if avail_count == len(horizons)
        else ("Partially Available" if avail_count > 0 else "Unavailable (Insufficient Valid Forecast Inputs)")
    )

    return {
        "status": overall_status,
        "is_experimental": True,
        "cannot_predict_exact_timing": True,
        "model_name": model_name,
        "model_available": model_exists,
        "model_is_synthetic_dev": model_is_synthetic,
        "horizons": horizon_summary,
        "trajectory": trajectory,
        "limitations": limitations,
        "scientific_disclaimer": FORECAST_TIMING_DISCLAIMER,
    }
