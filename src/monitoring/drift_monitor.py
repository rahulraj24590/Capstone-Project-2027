"""
Phase 6 Model Drift & Data Quality Monitoring Engine for LandslideGuard.

Monitors four evidence-gated dimensions:
1. Feature distribution changes (PSI, standardized mean shift, KS two-sample test)
2. Missing / imputed feature rate changes vs. training baseline
3. Prediction score distribution changes (PSI, KS test, elevated-risk share delta)
4. Live model performance changes once verified ground-truth outcomes arrive

Strict scientific integrity rules:
- Never claims drift is confirmed unless minimum sample sizes and configured
  statistical/performance thresholds are satisfied.
- Clearly states when reference statistics originate from synthetic dev data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.database.models import (
    get_prediction_outcomes,
    get_recent_environmental_observations,
    get_recent_predictions,
)
from src.models.model_registry import active_model_exists, load_model
from src.models.predict import load_feature_columns, load_training_stats
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

METRICS_PATH = Path(get_setting("paths", "reports") or "reports") / "metrics.json"
DATASET_PATH = Path(get_setting("paths", "data_processed") or "data/processed") / "dataset.parquet"


def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float | None:
    """Compute Population Stability Index using quantile bins anchored on expected data."""
    if len(expected) < 10 or len(actual) < 10:
        return None
    exp_clean = expected[np.isfinite(expected)]
    act_clean = actual[np.isfinite(actual)]
    if len(exp_clean) < 10 or len(act_clean) < 10:
        return None

    # Adapt quantile bin count so smaller sample slices still maintain >=10 obs/bin on average
    effective_bins = min(bins, max(4, len(act_clean) // 10))
    edges = np.quantile(exp_clean, np.linspace(0.0, 1.0, effective_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        # Constant/near-constant baseline cannot support multi-bin PSI cleanly
        return 0.0 if np.allclose(act_clean.mean(), exp_clean.mean()) else None

    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts, _ = np.histogram(exp_clean, bins=edges)
    act_counts, _ = np.histogram(act_clean, bins=edges)

    eps = 1e-4
    exp_pct = np.clip(exp_counts / max(len(exp_clean), 1), eps, None)
    act_pct = np.clip(act_counts / max(len(act_clean), 1), eps, None)
    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()

    psi_val = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(max(0.0, psi_val), 4)


def _ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple[float | None, float | None]:
    """Run two-sample Kolmogorov-Smirnov test safely."""
    exp_clean = expected[np.isfinite(expected)]
    act_clean = actual[np.isfinite(actual)]
    if len(exp_clean) < 5 or len(act_clean) < 5:
        return None, None
    res = ks_2samp(exp_clean, act_clean)
    return round(float(res.statistic), 4), round(float(res.pvalue), 6)


def _parse_json_field(val: Any, default: Any) -> Any:
    if isinstance(val, type(default)):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, type(default)) else default
        except json.JSONDecodeError:
            return default
    return default


def load_reference_baseline(reference_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Load reference feature, prediction, missingness, and test-performance baselines."""
    try:
        feature_cols = load_feature_columns()
    except FileNotFoundError:
        feature_cols = [
            "slope_degrees",
            "elevation_m",
            "rainfall_24h_mm",
            "rainfall_72h_mm",
            "rainfall_1h_mm",
            "vegetation_index",
            "soil_moisture",
        ]

    stats_json = load_training_stats()
    feature_medians = dict(stats_json.get("feature_medians") or {})

    ref_df = reference_df
    ref_source = "caller_reference_dataframe"
    if ref_df is None and DATASET_PATH.exists():
        try:
            full_df = pd.read_parquet(DATASET_PATH)
            # Match training chronological split (~68% / 1360 train rows)
            n_train = max(1, int(len(full_df) * 0.68))
            ref_df = full_df.iloc[:n_train].copy()
            ref_source = "dataset.parquet (chronological training slice; synthetic)"
        except Exception as exc:
            logger.warning("Failed reading reference dataset parquet: %s", exc)
            ref_df = None

    ref_arrays: dict[str, np.ndarray] = {}
    ref_summary: dict[str, dict[str, float]] = {}
    ref_missing_rates: dict[str, float] = {col: 0.0 for col in feature_cols}
    ref_probs = np.array([], dtype=float)

    if ref_df is not None and not ref_df.empty:
        for col in feature_cols:
            if col not in ref_df.columns:
                continue
            series = pd.to_numeric(ref_df[col], errors="coerce")
            ref_missing_rates[col] = round(float(series.isna().mean()), 4)
            valid = series.dropna().to_numpy(dtype=float)
            if len(valid) > 0:
                ref_arrays[col] = valid
                ref_summary[col] = {
                    "mean": round(float(np.mean(valid)), 4),
                    "std": round(float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0, 4),
                    "median": round(float(np.median(valid)), 4),
                }

        # Baseline prediction distribution using saved model on available reference rows
        if active_model_exists() and all(c in ref_df.columns for c in feature_cols):
            try:
                model = load_model("best_model")
                x_ref = ref_df[feature_cols].fillna(pd.Series(feature_medians)).to_numpy(dtype=float)
                if hasattr(model, "predict_proba"):
                    ref_probs = np.asarray(model.predict_proba(x_ref)[:, 1], dtype=float)
            except Exception as exc:
                logger.warning("Could not score reference baseline for prediction drift: %s", exc)

    ref_test_metrics: dict[str, Any] = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                all_m = json.load(f)
            test_key = next((k for k in all_m if str(k).endswith("_test")), None)
            if test_key and isinstance(all_m[test_key], dict):
                ref_test_metrics = {
                    k: all_m[test_key].get(k)
                    for k in ("precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "n_samples")
                }
        except Exception:
            ref_test_metrics = {}

    return {
        "feature_columns": feature_cols,
        "reference_source": ref_source if ref_df is not None else "training_stats.json (medians only)",
        "reference_arrays": ref_arrays,
        "reference_summary": ref_summary,
        "reference_missing_rates": ref_missing_rates,
        "reference_probabilities": ref_probs,
        "reference_test_metrics": ref_test_metrics,
        "reference_is_synthetic": True,
    }


def _extract_feature_frames(
    predictions_df: pd.DataFrame | None,
    env_df: pd.DataFrame | None,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Extract current feature matrix and explicit/imputed missing counts cleanly."""
    rows: list[dict[str, float | None]] = []
    imputed_counts: dict[str, int] = {col: 0 for col in feature_cols}

    if predictions_df is not None and not predictions_df.empty and "input_features" in predictions_df.columns:
        for _, rec in predictions_df.iterrows():
            feats = _parse_json_field(rec.get("input_features"), {})
            warns = _parse_json_field(rec.get("warnings"), [])
            warn_text = " ".join(str(w) for w in warns).lower()
            row_out: dict[str, float | None] = {}
            for col in feature_cols:
                raw_v = feats.get(col)
                was_imputed = f"feature '{col.lower()}' was missing" in warn_text
                if was_imputed:
                    imputed_counts[col] += 1
                    row_out[col] = None
                elif raw_v is None:
                    row_out[col] = None
                else:
                    try:
                        fv = float(raw_v)
                        row_out[col] = fv if math.isfinite(fv) else None
                    except (TypeError, ValueError):
                        row_out[col] = None
            rows.append(row_out)

    # Supplement from environmental_observations if prediction records carry no feature payloads
    if not rows and env_df is not None and not env_df.empty:
        for _, rec in env_df.iterrows():
            row_out = {}
            for col in feature_cols:
                raw_v = rec.get(col)
                if raw_v is None or pd.isna(raw_v):
                    row_out[col] = None
                else:
                    try:
                        fv = float(raw_v)
                        row_out[col] = fv if math.isfinite(fv) else None
                    except (TypeError, ValueError):
                        row_out[col] = None
            rows.append(row_out)

    if not rows:
        return pd.DataFrame(columns=feature_cols), {col: 0.0 for col in feature_cols}

    feat_df = pd.DataFrame(rows, columns=feature_cols)
    miss_rates = {col: round(float(feat_df[col].isna().mean()), 4) for col in feature_cols}
    return feat_df, miss_rates


def compute_model_drift_report(
    predictions_df: pd.DataFrame | None = None,
    env_df: pd.DataFrame | None = None,
    outcomes_df: pd.DataFrame | None = None,
    reference_df: pd.DataFrame | None = None,
    limit: int = 300,
) -> dict[str, Any]:
    """Evaluate feature, missingness, prediction, and outcome performance drift safely."""
    cfg = get_setting("analytics", "drift") or {}
    min_feat_n = int(cfg.get("min_feature_samples", 20))
    min_out_n = int(cfg.get("min_outcome_samples", 10))
    psi_mod = float(cfg.get("psi_moderate_threshold", 0.10))
    psi_high = float(cfg.get("psi_high_threshold", 0.25))
    ks_p_thresh = float(cfg.get("ks_p_threshold", 0.01))
    miss_delta_thresh = float(cfg.get("missing_rate_delta_threshold", 0.10))
    perf_drop_thresh = float(cfg.get("perf_drop_threshold", 0.10))

    if predictions_df is None:
        predictions_df = get_recent_predictions(limit=limit)
    if env_df is None:
        env_df = get_recent_environmental_observations(limit=limit)
    if outcomes_df is None:
        outcomes_df = get_prediction_outcomes(limit=limit)

    ref = load_reference_baseline(reference_df=reference_df)
    feature_cols = ref["feature_columns"]
    ref_arrays = ref["reference_arrays"]
    ref_summary = ref["reference_summary"]
    ref_miss = ref["reference_missing_rates"]
    ref_probs = ref["reference_probabilities"]
    ref_test = ref["reference_test_metrics"]

    feat_df, cur_miss = _extract_feature_frames(predictions_df, env_df, feature_cols)
    n_feature_rows = int(len(feat_df))

    # ── 1 & 2. Feature Distribution & Missing Data Changes ───────────
    feature_drift_rows: list[dict[str, Any]] = []
    missingness_rows: list[dict[str, Any]] = []
    confirmed_feature_shifts: list[str] = []
    watchlist_signals: list[str] = []

    for col in feature_cols:
        miss_base = float(ref_miss.get(col, 0.0))
        miss_cur = float(cur_miss.get(col, 0.0))
        miss_delta = round(miss_cur - miss_base, 4)

        if n_feature_rows < min_feat_n:
            miss_status = f"Insufficient Samples ({n_feature_rows}/{min_feat_n})"
        elif miss_delta >= miss_delta_thresh:
            miss_status = "Elevated Missingness"
            watchlist_signals.append(f"{col} missing/imputed rate +{miss_delta * 100:.1f}%")
        else:
            miss_status = "Nominal"

        missingness_rows.append({
            "feature": col,
            "sample_count": n_feature_rows,
            "baseline_missing_rate": miss_base,
            "current_missing_rate": miss_cur,
            "missing_rate_delta": miss_delta,
            "status": miss_status,
        })

        cur_arr = (
            pd.to_numeric(feat_df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if col in feat_df.columns and not feat_df.empty
            else np.array([], dtype=float)
        )
        ref_arr = ref_arrays.get(col, np.array([], dtype=float))
        ref_stat = ref_summary.get(col, {})

        if len(cur_arr) < min_feat_n or len(ref_arr) < min_feat_n:
            feature_drift_rows.append({
                "feature": col,
                "current_samples": int(len(cur_arr)),
                "reference_mean": ref_stat.get("mean"),
                "current_mean": round(float(np.mean(cur_arr)), 4) if len(cur_arr) > 0 else None,
                "mean_shift_z": None,
                "psi": None,
                "ks_stat": None,
                "ks_p_value": None,
                "status": f"Insufficient Samples ({len(cur_arr)}/{min_feat_n})",
                "shift_detected": False,
            })
            continue

        cur_mean = float(np.mean(cur_arr))
        ref_mean = float(ref_stat.get("mean", np.mean(ref_arr)))
        ref_std = float(ref_stat.get("std", np.std(ref_arr, ddof=1)))
        mean_shift_z = round((cur_mean - ref_mean) / ref_std, 3) if ref_std > 1e-9 else None
        psi = _compute_psi(ref_arr, cur_arr)
        ks_stat, ks_p = _ks_test(ref_arr, cur_arr)

        # Require both material magnitude (PSI) and statistical corroboration (KS p < thresh)
        # before confirming feature distribution drift to avoid small-sample noise triggers.
        if (psi is not None and psi >= psi_high and ks_p is None) or (
            psi is not None and psi >= 0.15 and ks_p is not None and ks_p < ks_p_thresh
        ):
            status = "Significant Distribution Shift"
            shift_flag = True
            confirmed_feature_shifts.append(f"{col} (PSI={psi}, KS p={ks_p})")
        elif psi is not None and psi >= psi_mod:
            status = "Moderate Distribution Shift (Watch)"
            shift_flag = False
            watchlist_signals.append(f"{col} moderate shift (PSI={psi})")
        else:
            status = "Stable"
            shift_flag = False

        feature_drift_rows.append({
            "feature": col,
            "current_samples": int(len(cur_arr)),
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(cur_mean, 4),
            "mean_shift_z": mean_shift_z,
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_p_value": ks_p,
            "status": status,
            "shift_detected": shift_flag,
        })

    # ── 3. Prediction Distribution Changes ───────────────────────────
    cur_probs = (
        pd.to_numeric(predictions_df["probability"], errors="coerce").dropna().to_numpy(dtype=float)
        if predictions_df is not None and not predictions_df.empty and "probability" in predictions_df.columns
        else np.array([], dtype=float)
    )
    pred_drift: dict[str, Any] = {
        "current_samples": int(len(cur_probs)),
        "reference_samples": int(len(ref_probs)),
        "current_mean_probability": round(float(np.mean(cur_probs)), 4) if len(cur_probs) > 0 else None,
        "reference_mean_probability": round(float(np.mean(ref_probs)), 4) if len(ref_probs) > 0 else None,
        "current_high_risk_share": round(float(np.mean(cur_probs >= 0.55)), 4) if len(cur_probs) > 0 else None,
        "reference_high_risk_share": round(float(np.mean(ref_probs >= 0.55)), 4) if len(ref_probs) > 0 else None,
        "psi": None,
        "ks_stat": None,
        "ks_p_value": None,
        "shift_detected": False,
        "status": f"Insufficient Samples ({len(cur_probs)}/{min_feat_n})",
        "current_probabilities": [round(float(p), 4) for p in cur_probs.tolist()],
        "reference_probabilities": [round(float(p), 4) for p in ref_probs[:500].tolist()],
    }

    if len(cur_probs) >= min_feat_n and len(ref_probs) >= min_feat_n:
        p_psi = _compute_psi(ref_probs, cur_probs)
        p_ks, p_pval = _ks_test(ref_probs, cur_probs)
        pred_drift["psi"] = p_psi
        pred_drift["ks_stat"] = p_ks
        pred_drift["ks_p_value"] = p_pval
        if (p_psi is not None and p_psi >= psi_high) and (p_pval is None or p_pval < ks_p_thresh):
            pred_drift["shift_detected"] = True
            pred_drift["status"] = "Prediction Distribution Shift Observed"
            confirmed_feature_shifts.append(f"prediction_probability (PSI={p_psi}, KS p={p_pval})")
        elif p_psi is not None and p_psi >= psi_mod:
            pred_drift["status"] = "Moderate Prediction Shift (Watch)"
            watchlist_signals.append(f"Prediction score moderate shift (PSI={p_psi})")
        else:
            pred_drift["status"] = "Stable"

    # ── 4. Performance Changes When Labelled Outcomes Exist ──────────
    perf_drift: dict[str, Any] = {
        "labelled_samples": 0,
        "min_required_samples": min_out_n,
        "status": f"Awaiting Labelled Outcomes (0/{min_out_n})",
        "degradation_detected": False,
        "live_metrics": None,
        "reference_test_metrics": ref_test,
        "f1_delta": None,
        "brier_delta": None,
    }

    if outcomes_df is not None and not outcomes_df.empty:
        valid_out = outcomes_df.dropna(subset=["actual_label", "predicted_probability"]).copy()
        valid_out["actual_label"] = pd.to_numeric(valid_out["actual_label"], errors="coerce")
        valid_out["predicted_probability"] = pd.to_numeric(valid_out["predicted_probability"], errors="coerce")
        valid_out = valid_out[valid_out["actual_label"].isin([0, 1]) & valid_out["predicted_probability"].between(0.0, 1.0)]

        n_out = int(len(valid_out))
        perf_drift["labelled_samples"] = n_out
        if n_out < min_out_n:
            perf_drift["status"] = f"Awaiting Sufficient Labelled Outcomes ({n_out}/{min_out_n})"
        else:
            y_true = valid_out["actual_label"].to_numpy(dtype=int)
            y_prob = valid_out["predicted_probability"].to_numpy(dtype=float)
            y_pred = (y_prob >= 0.5).astype(int)

            live_brier = round(float(brier_score_loss(y_true, y_prob)), 4)
            live_acc = round(float(accuracy_score(y_true, y_pred)), 4)
            live_prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
            live_rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
            live_f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
            has_both_classes = len(np.unique(y_true)) == 2
            live_roc = round(float(roc_auc_score(y_true, y_prob)), 4) if has_both_classes else None
            live_pr = round(float(average_precision_score(y_true, y_prob)), 4) if has_both_classes else None

            perf_drift["live_metrics"] = {
                "accuracy": live_acc,
                "precision": live_prec,
                "recall": live_rec,
                "f1": live_f1,
                "roc_auc": live_roc,
                "pr_auc": live_pr,
                "brier_score": live_brier,
                "positive_outcomes": int(y_true.sum()),
            }

            ref_f1 = ref_test.get("f1")
            ref_brier = ref_test.get("brier_score")
            f1_delta = round(live_f1 - float(ref_f1), 4) if ref_f1 is not None else None
            brier_delta = round(live_brier - float(ref_brier), 4) if ref_brier is not None else None
            perf_drift["f1_delta"] = f1_delta
            perf_drift["brier_delta"] = brier_delta

            degraded = bool(
                (has_both_classes and f1_delta is not None and f1_delta <= -perf_drop_thresh)
                or (brier_delta is not None and brier_delta >= perf_drop_thresh)
            )
            perf_drift["degradation_detected"] = degraded
            if degraded:
                perf_drift["status"] = "Performance Degradation Detected on Labelled Outcomes"
                confirmed_feature_shifts.append(
                    f"labelled_performance (F1 delta={f1_delta}, Brier delta={brier_delta})"
                )
            else:
                perf_drift["status"] = "Nominal on Labelled Outcomes"

    # ── 5. Evidence-Gated Overall Conclusion ─────────────────────────
    any_sufficient = n_feature_rows >= min_feat_n or perf_drift["labelled_samples"] >= min_out_n
    drift_confirmed = bool(confirmed_feature_shifts)

    if not any_sufficient:
        overall_status = "Insufficient Evidence to Assess Drift"
    elif drift_confirmed:
        overall_status = "Drift Indicators Confirmed"
    elif watchlist_signals:
        overall_status = "Watchlist — Moderate Shift or Elevated Missingness (Drift Unconfirmed)"
    else:
        overall_status = "No Confirmed Drift Detected"

    avg_completeness = (
        round((1.0 - float(np.mean(list(cur_miss.values())))) * 100.0, 1)
        if n_feature_rows > 0 and cur_miss
        else None
    )

    return {
        "overall_status": overall_status,
        "drift_confirmed": drift_confirmed,
        "confirmed_indicators": confirmed_feature_shifts,
        "watchlist_signals": watchlist_signals,
        "reference_source": ref["reference_source"],
        "reference_is_synthetic": ref["reference_is_synthetic"],
        "data_quality_summary": {
            "prediction_records_analyzed": int(len(predictions_df)) if predictions_df is not None else 0,
            "feature_vectors_analyzed": n_feature_rows,
            "labelled_outcomes_count": perf_drift["labelled_samples"],
            "avg_feature_completeness_pct": avg_completeness,
            "min_feature_samples_required": min_feat_n,
            "min_outcome_samples_required": min_out_n,
        },
        "feature_drift": feature_drift_rows,
        "missingness_drift": missingness_rows,
        "prediction_drift": pred_drift,
        "performance_drift": perf_drift,
        "methodology_note": (
            "Feature and prediction drift are evaluated via Population Stability Index (PSI) and two-sample KS tests "
            "anchored against the training reference distribution (minimum 20 production records required). "
            "Performance drift is evaluated only once >=10 verified ground-truth outcomes are recorded."
        ),
    }
