"""
Explainable AI (XAI) module for LandslideGuard.

Provides global and individual prediction explanations built strictly from actual
trained models and verified datasets:
- SHAP attribution (LinearExplainer / TreeExplainer where model type is supported)
- Permutation feature importance (sklearn.inspection.permutation_importance)
- Tree-based built-in feature importance where supported
- Human-readable contribution statements with causal humility guarantees
- Bounded in-memory caching to avoid redundant SHAP computations

Scientific integrity guarantees:
- Never fabricates SHAP values or feature importances when a model is missing,
  untrained, unsupported, or encounters an error.
- Reorders inputs strictly to the model's trained feature_columns order.
- Clearly separates feature importance, SHAP contribution, correlation, and causation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.models.model_registry import get_model_info, load_model
from src.models.predict import (
    load_feature_columns,
    load_training_stats,
    probability_to_risk_category,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# In-memory bounded caches to prevent repeated SHAP calculation on identical inputs
_MAX_CACHE_ENTRIES = 128
_LOCAL_EXPLANATION_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_GLOBAL_EXPLANATION_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()

EXPLAINABILITY_CONCEPTS: list[dict[str, str]] = [
    {
        "concept": "Feature Importance (Global)",
        "definition": (
            "Measures how much a feature influences predictions or validation performance "
            "(e.g., PR-AUC drop under permutation or tree split impurity reduction) across an "
            "evaluation dataset. Shows overall sensitivity, not direction for one point."
        ),
    },
    {
        "concept": "SHAP Contribution (Local & Additive)",
        "definition": (
            "Shapley Additive exPlanations decompose an individual model prediction into additive "
            "feature shifts from the expected baseline output (base_value + sum(SHAP) = output). "
            "Positive values push risk higher; negative values push risk lower."
        ),
    },
    {
        "concept": "Correlation",
        "definition": (
            "Measures statistical co-movement between two variables in observational or synthetic "
            "data. Strongly correlated predictors can share attribution weight without physical meaning."
        ),
    },
    {
        "concept": "Causation",
        "definition": (
            "Physical cause-and-effect slope failure mechanism governed by geotechnical stability, "
            "pore-water pressure, and geology. ML models and SHAP NEVER prove physical causation."
        ),
    },
]

NON_CAUSAL_DISCLAIMER = (
    "Explanations describe how the statistical model weighed input features for this "
    "experimental estimate. High SHAP contribution or feature importance reflects model "
    "association, NOT physical landslide causation."
)

# Domain metadata restricted strictly to features a trained model may actually use
FEATURE_METADATA_MAP: dict[str, dict[str, str]] = {
    "rainfall_1h_mm": {"label": "1-Hour Rainfall (mm)", "category": "Rainfall"},
    "rainfall_24h_mm": {"label": "24-Hour Rainfall (mm)", "category": "Rainfall"},
    "rainfall_48h_mm": {"label": "48-Hour Rainfall (mm)", "category": "Rainfall"},
    "rainfall_72h_mm": {"label": "72-Hour Rainfall (mm)", "category": "Rainfall"},
    "slope_degrees": {"label": "Terrain Slope (°)", "category": "Slope"},
    "elevation_m": {"label": "Elevation (m)", "category": "Elevation"},
    "soil_moisture": {"label": "Soil Moisture (m³/m³)", "category": "Soil"},
    "vegetation_index": {"label": "Vegetation / Land Cover Index", "category": "Land Cover"},
    "land_cover": {"label": "Land Cover Class", "category": "Land Cover"},
}


def describe_model_feature(col: str) -> dict[str, str]:
    """Return human-readable label and domain category for an active model feature."""
    if col in FEATURE_METADATA_MAP:
        return {"feature": col, **FEATURE_METADATA_MAP[col]}
    clean = col.replace("_", " ").title()
    return {"feature": col, "label": clean, "category": "Validated Model Input"}


def _infer_dataset_provenance(parquet_path: Path = Path("data/processed/dataset.parquet")) -> dict[str, Any]:
    """Inspect active processed dataset metadata without inventing external provenance."""
    if not parquet_path.exists():
        return {"source": "Unavailable (dataset.parquet not found)", "is_synthetic": None}
    try:
        df_head = pd.read_parquet(parquet_path).head(25)
        is_syn: bool | None = None
        if "is_synthetic" in df_head.columns:
            syn_vals = df_head["is_synthetic"].dropna()
            if not syn_vals.empty:
                is_syn = bool(int(syn_vals.iloc[0]) == 1)
        src_col = None
        for col in ("source", "data_source", "dataset_source"):
            if col in df_head.columns and df_head[col].notna().any():
                src_col = str(df_head[col].dropna().iloc[0])
                break
        if is_syn is True:
            source_desc = f"Synthetic Development Dataset ({parquet_path.as_posix()}, is_synthetic=1)"
        elif src_col:
            source_desc = f"{src_col} ({parquet_path.as_posix()})"
        else:
            source_desc = str(parquet_path.as_posix())
        return {"source": source_desc, "is_synthetic": is_syn}
    except Exception:
        return {"source": str(parquet_path.as_posix()), "is_synthetic": None}


def build_model_metadata_summary(
    model_name: str,
    model_info: dict | None,
    feature_cols: list[str] | None,
) -> dict[str, Any]:
    """Assemble transparent model provenance, evaluation metrics, and validation info."""
    meta = (model_info or {}).get("metadata", {}) if model_info else {}
    ds_info = _infer_dataset_provenance()
    dataset_source = meta.get("dataset_source") or ds_info["source"]
    saved_at = (model_info or {}).get("saved_at") if model_info else None
    alg = meta.get("algorithm") or (
        model_name.replace("candidate_", "") if model_name.startswith("candidate_") else "Unavailable"
    )
    active_cols = list(feature_cols or meta.get("feature_columns") or [])
    feature_descriptors = [describe_model_feature(c) for c in active_cols]

    scalar_keys = ("split", "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score")
    raw_test = meta.get("test_metrics") or {}
    raw_val = meta.get("validation_metrics") or {}
    test_summary = {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in raw_test.items() if k in scalar_keys}
    val_summary = {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in raw_val.items() if k in scalar_keys}

    validation_approach = meta.get("validation_approach")
    if not validation_approach:
        if test_summary or val_summary or meta.get("train_samples") is not None:
            validation_approach = (
                "Chronological temporal split (or stratified random fallback when dates absent: "
                "68% train / 12% validation / 20% held-out test); selected by validation PR-AUC"
            )
        else:
            validation_approach = "Unavailable"

    return {
        "model_name": model_name,
        "algorithm": alg,
        "model_version": _get_model_version_tag(model_name, model_info),
        "training_date": str(saved_at)[:19].replace("T", " ") + " UTC" if saved_at else "Unavailable",
        "training_dataset_source": dataset_source,
        "is_synthetic_training": ds_info["is_synthetic"],
        "train_samples": meta.get("train_samples"),
        "test_samples": meta.get("test_samples"),
        "feature_list": active_cols,
        "feature_descriptors": feature_descriptors,
        "evaluation_metrics": {
            "test": test_summary,
            "validation": val_summary,
        },
        "validation_approach": validation_approach,
    }


def _build_explanation_limitations(
    missing_features: list[str],
    shap_available: bool,
    shap_unit: str | None,
    model_metadata: dict[str, Any],
) -> list[str]:
    """Generate transparent limitations for an individual prediction explanation."""
    limits = [NON_CAUSAL_DISCLAIMER]
    if model_metadata.get("is_synthetic_training") is True:
        limits.append(
            "Model artifacts were trained on synthetic development data (is_synthetic=1); "
            "attributions illustrate pipeline mechanics, not validated real-world geotechnical susceptibility."
        )
    if missing_features:
        limits.append(
            f"Missing/unavailable inputs ({', '.join(missing_features)}) were filled with training medians "
            "and do not reflect site measurements."
        )
    if shap_available:
        limits.append(
            f"SHAP contributions measure additive shift relative to the background sample baseline in '{shap_unit or 'score'}' units, "
            "which varies with the chosen background reference set."
        )
    else:
        limits.append("SHAP attribution is unavailable for this estimator type; no values were fabricated.")
    return limits


def clear_explanation_cache() -> None:
    """Clear local and global SHAP/explainability caches."""
    _LOCAL_EXPLANATION_CACHE.clear()
    _GLOBAL_EXPLANATION_CACHE.clear()


def _cache_put(cache: OrderedDict[str, dict[str, Any]], key: str, value: dict[str, Any]) -> None:
    if key in cache:
        cache.move_to_end(key)
    cache[key] = value
    while len(cache) > _MAX_CACHE_ENTRIES:
        cache.popitem(last=False)


def _unwrap_pipeline(model: Any) -> tuple[Any, Any]:
    """Return (preprocessor_or_None, final_estimator) for sklearn Pipeline or bare estimator."""
    if hasattr(model, "named_steps") and isinstance(model.named_steps, dict):
        steps = list(model.named_steps.items())
        clf = steps[-1][1]
        if len(steps) == 1:
            return None, clf
        if len(steps) == 2 and hasattr(steps[0][1], "transform"):
            return steps[0][1], clf
        # Multi-step pipeline: apply all transformers prior to final estimator
        from sklearn.pipeline import Pipeline
        return Pipeline(steps[:-1]), clf
    return None, model


def _validate_fitted_model(model: Any) -> tuple[bool, str | None]:
    """Verify model object exists and is trained/fitted."""
    if model is None:
        return False, "No trained model provided or loaded."
    try:
        check_is_fitted(model)
    except NotFittedError as exc:
        return False, f"Model is not trained/fitted: {exc}"
    except Exception as exc:
        # Some wrapper estimators raise TypeError if not sklearn-standard; check classifier attributes
        _, clf = _unwrap_pipeline(model)
        if not any(
            hasattr(clf, attr)
            for attr in ("coef_", "feature_importances_", "classes_", "n_features_in_", "get_booster")
        ):
            return False, f"Unable to verify trained model state: {exc}"
    return True, None


def _get_model_version_tag(model_name: str, model_info: dict | None) -> str:
    """Build a concise deterministic version tag from registry metadata."""
    if not model_info:
        return f"{model_name} (unregistered)"
    meta = model_info.get("metadata", {})
    alg = meta.get("algorithm") or model_name.replace("candidate_", "")
    saved_at = str(model_info.get("saved_at", "unknown"))[:19]
    return f"{model_name} [{alg} @ {saved_at}]"


def _resolve_model_and_metadata(
    model_name: str = "best_model",
    model: Any = None,
    feature_cols: list[str] | None = None,
    medians: dict[str, float] | None = None,
) -> tuple[Any, list[str] | None, dict[str, float], dict | None, str | None]:
    """Load model, ordered feature columns, training medians, and registry info safely."""
    loaded_model = model
    if loaded_model is None:
        try:
            loaded_model = load_model(model_name)
        except FileNotFoundError as exc:
            return None, None, {}, None, str(exc)
        except Exception as exc:
            return None, None, {}, None, f"Failed to load model '{model_name}': {exc}"

    is_ready, fit_err = _validate_fitted_model(loaded_model)
    if not is_ready:
        return loaded_model, None, {}, None, fit_err

    model_info = get_model_info(model_name)
    cols = feature_cols
    if cols is None:
        if model_info and model_info.get("metadata", {}).get("feature_columns"):
            cols = list(model_info["metadata"]["feature_columns"])
        else:
            try:
                cols = load_feature_columns()
            except FileNotFoundError as exc:
                return loaded_model, None, {}, model_info, str(exc)

    resolved_medians = medians
    if resolved_medians is None:
        try:
            resolved_medians = load_training_stats().get("feature_medians", {})
        except Exception:
            resolved_medians = {}

    return loaded_model, cols, resolved_medians, model_info, None


def _build_ordered_feature_vector(
    features: dict[str, Any],
    feature_cols: list[str],
    medians: dict[str, float],
) -> tuple[np.ndarray | None, dict[str, float], list[str], list[str], list[str], list[str]]:
    """Reorder an input feature dict strictly into `feature_cols` order.

    Requires at least one valid finite numeric model feature so empty or wholly
    unavailable inputs never yield fabricated median-only explanations.
    """
    warnings: list[str] = []
    errors: list[str] = []
    missing_features: list[str] = []
    available_features: list[str] = []
    used: dict[str, float] = {}
    row_vals: list[float] = []

    if not isinstance(features, dict) or not features:
        return None, {}, [], list(feature_cols), [], [
            "Insufficient input data: no model features were provided; refusing to generate explanation."
        ]

    # Report extra keys ignored vs expected feature order for transparency
    for col in feature_cols:
        raw_val = features.get(col)
        is_missing = raw_val is None
        if isinstance(raw_val, float) and (math.isnan(raw_val) or math.isinf(raw_val)):
            if math.isinf(raw_val):
                errors.append(f"Feature '{col}' must be finite; got {raw_val}.")
                continue
            is_missing = True

        if is_missing:
            missing_features.append(col)
            fill_val = medians.get(col)
            if fill_val is None:
                errors.append(f"Feature '{col}' is missing and no training median fallback exists.")
                continue
            row_vals.append(float(fill_val))
            used[col] = float(fill_val)
            warnings.append(
                f"Feature '{col}' was missing and filled with training median ({float(fill_val):.4f}) "
                "for explanation continuity."
            )
        else:
            try:
                num = float(raw_val)
            except (TypeError, ValueError):
                errors.append(f"Feature '{col}' must be numeric; got {raw_val!r}.")
                continue
            if not math.isfinite(num):
                errors.append(f"Feature '{col}' is not finite ({num}).")
                continue
            row_vals.append(num)
            used[col] = num
            available_features.append(col)

    if not available_features:
        errors.append(
            "Insufficient input data: zero valid model features were available; "
            "refusing to generate median-only explanation."
        )

    if errors:
        return None, used, available_features, missing_features, warnings, errors

    return np.array([row_vals], dtype=float), used, available_features, missing_features, warnings, []


def _align_dataframe_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    medians: dict[str, float] | None = None,
) -> tuple[np.ndarray | None, pd.DataFrame | None, list[str], list[str]]:
    """Validate and reorder a DataFrame into exact `feature_cols` column order."""
    warnings: list[str] = []
    errors: list[str] = []
    if df is None or df.empty:
        return None, None, [], ["Evaluation dataset is empty."]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        return None, None, [], [
            f"Evaluation dataset is missing required model features: {missing_cols}."
        ]

    ordered_df = df[feature_cols].copy()
    if list(df.columns)[:len(feature_cols)] != feature_cols:
        warnings.append(
            "Input DataFrame columns were reordered to match model training feature order."
        )

    for col in feature_cols:
        ordered_df[col] = pd.to_numeric(ordered_df[col], errors="coerce")
        if ordered_df[col].isna().any():
            fill = (medians or {}).get(col)
            if fill is None:
                fill = float(ordered_df[col].median()) if ordered_df[col].notna().any() else None
            if fill is None:
                errors.append(f"Feature '{col}' contains only NaN values with no fallback median.")
                return None, None, warnings, errors
            n_na = int(ordered_df[col].isna().sum())
            ordered_df[col] = ordered_df[col].fillna(float(fill))
            warnings.append(f"Imputed {n_na} missing value(s) in '{col}' using median ({float(fill):.4f}).")

    return ordered_df.values.astype(float), ordered_df, warnings, []


def _compute_shap_matrix(
    clf: Any,
    X_transformed: np.ndarray,
    X_bg_transformed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Run compatible SHAP explainer for linear or tree estimators without fabrication.

    Returns:
        (shap_values_2d, base_values_1d, explainer_name, shap_unit)
    """
    import shap

    is_linear = hasattr(clf, "coef_")
    is_tree = (
        hasattr(clf, "feature_importances_")
        and (
            hasattr(clf, "estimators_")
            or "xgb" in type(clf).__module__.lower()
            or "tree" in type(clf).__name__.lower()
            or "forest" in type(clf).__name__.lower()
            or "boosting" in type(clf).__name__.lower()
        )
    )

    if is_tree:
        try:
            explainer = shap.TreeExplainer(clf)
            explanation = explainer(X_transformed, check_additivity=False)
            explainer_name = f"TreeExplainer ({type(clf).__name__})"
        except TypeError:
            explainer = shap.TreeExplainer(clf)
            explanation = explainer(X_transformed)
            explainer_name = f"TreeExplainer ({type(clf).__name__})"
    elif is_linear:
        if X_bg_transformed is None or len(X_bg_transformed) == 0:
            X_bg_transformed = X_transformed
        explainer = shap.LinearExplainer(clf, X_bg_transformed)
        explanation = explainer(X_transformed)
        explainer_name = f"LinearExplainer ({type(clf).__name__})"
    else:
        raise TypeError(
            f"Unsupported estimator type '{type(clf).__name__}' for native SHAP "
            "(expected linear coef_ or supported tree ensemble)."
        )

    vals = np.asarray(explanation.values, dtype=float)
    base_vals = np.asarray(explanation.base_values, dtype=float)

    # Normalise 3D binary classifier outputs (n_samples, n_features, 2) -> positive class [:, :, 1]
    if vals.ndim == 3:
        class_idx = 1 if vals.shape[2] > 1 else 0
        vals = vals[:, :, class_idx]
        if base_vals.ndim == 2:
            base_vals = base_vals[:, class_idx]
        elif base_vals.ndim == 1 and base_vals.shape[0] == 2:
            base_vals = np.full(vals.shape[0], float(base_vals[class_idx]))
        shap_unit = "probability_margin"
    else:
        if base_vals.ndim == 0:
            base_vals = np.full(vals.shape[0], float(base_vals))
        elif base_vals.ndim == 2:
            base_vals = base_vals[:, -1]
        shap_unit = "log_odds"

    if vals.shape != X_transformed.shape:
        raise RuntimeError(
            f"Unexpected SHAP shape {vals.shape} for input shape {X_transformed.shape}."
        )

    return vals, base_vals, explainer_name, shap_unit


def _load_default_background(feature_cols: list[str], medians: dict[str, float], max_rows: int = 120) -> np.ndarray:
    """Load real dataset rows as background reference, or single median row if parquet absent."""
    parquet_path = Path("data/processed/dataset.parquet")
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            X_bg, _, _, errs = _align_dataframe_features(df.head(max_rows), feature_cols, medians)
            if X_bg is not None and not errs:
                return X_bg
        except Exception as exc:
            logger.debug("Could not read default background parquet: %s", exc)
    return np.array([[float(medians.get(c, 0.0)) for c in feature_cols]], dtype=float)


def _format_human_statements(
    contributions: list[dict[str, Any]],
    probability: float | None,
    risk_category: str,
) -> tuple[str, list[str]]:
    """Build non-causal human-readable explanation statements from actual SHAP values."""
    if not contributions:
        return "Model attribution values could not be computed for this prediction.", []

    top_pos = [c for c in contributions if c["shap_value"] > 0][:3]
    top_neg = [c for c in contributions if c["shap_value"] < 0][:2]
    statements: list[str] = []

    for item in contributions[:4]:
        direction_phrase = (
            "higher (increasing predicted risk)"
            if item["shap_value"] > 0
            else "lower (decreasing predicted risk)"
            if item["shap_value"] < 0
            else "essentially neutral relative to baseline"
        )
        statements.append(
            f"The model assigned importance to {item['feature']} "
            f"(input={item['feature_value']:.3f}, SHAP={item['shap_value']:+.4f}) "
            f"when producing this estimate, pushing output {direction_phrase}."
        )

    prob_str = f"{probability * 100:.1f}%" if probability is not None else "N/A"
    pos_str = ", ".join(c["feature"] for c in top_pos) if top_pos else "none"
    neg_str = ", ".join(c["feature"] for c in top_neg) if top_neg else "none"
    summary = (
        f"Estimated category is {risk_category} ({prob_str}). "
        f"Top upward model drivers: {pos_str}. Top downward drivers: {neg_str}. "
        "Note: SHAP reflects model sensitivity to feature values, not physical landslide causation."
    )
    return summary, statements


def explain_prediction(
    features: dict[str, Any],
    model_name: str = "best_model",
    model: Any = None,
    feature_cols: list[str] | None = None,
    medians: dict[str, float] | None = None,
    background_df: pd.DataFrame | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Compute an individual prediction and compatible SHAP attribution explanation."""
    resolved_model, cols, resolved_medians, model_info, load_err = _resolve_model_and_metadata(
        model_name=model_name,
        model=model,
        feature_cols=feature_cols,
        medians=medians,
    )
    version_tag = _get_model_version_tag(model_name, model_info)
    model_meta = build_model_metadata_summary(model_name, model_info, cols)

    if load_err or cols is None:
        return {
            "available": False,
            "shap_available": False,
            "model_name": model_name,
            "model_version": version_tag,
            "model_metadata": model_meta,
            "probability": None,
            "predicted_class": None,
            "risk_category": "Unknown",
            "feature_values": {},
            "available_features": [],
            "missing_features": [],
            "contributions": [],
            "positive_contributors": [],
            "negative_contributors": [],
            "base_value": None,
            "shap_unit": None,
            "explainer_type": None,
            "human_readable_summary": "Model explanation unavailable because no trained model could be loaded.",
            "feature_statements": [],
            "explanation_limitations": _build_explanation_limitations([], False, None, model_meta),
            "warnings": [],
            "errors": [load_err or "Model unavailable."],
            "from_cache": False,
            "scientific_disclaimer": NON_CAUSAL_DISCLAIMER,
        }

    X_raw, used_features, avail_feats, missing_feats, warnings, vec_errors = _build_ordered_feature_vector(
        features, cols, resolved_medians
    )
    if vec_errors or X_raw is None:
        return {
            "available": False,
            "shap_available": False,
            "model_name": model_name,
            "model_version": version_tag,
            "model_metadata": model_meta,
            "probability": None,
            "predicted_class": None,
            "risk_category": "Unknown",
            "feature_values": used_features,
            "available_features": avail_feats,
            "missing_features": missing_feats,
            "contributions": [],
            "positive_contributors": [],
            "negative_contributors": [],
            "base_value": None,
            "shap_unit": None,
            "explainer_type": None,
            "human_readable_summary": "Invalid or unresolvable input features prevented prediction explanation.",
            "feature_statements": [],
            "explanation_limitations": _build_explanation_limitations(missing_feats, False, None, model_meta),
            "warnings": warnings,
            "errors": vec_errors,
            "from_cache": False,
            "scientific_disclaimer": NON_CAUSAL_DISCLAIMER,
        }

    # Check deterministic cache for custom or default model calls
    cache_payload = json.dumps(
        {
            "model": model_name if model is None else f"custom_{id(model)}",
            "version": version_tag,
            "cols": cols,
            "vals": [round(float(used_features[c]), 8) for c in cols],
        },
        sort_keys=True,
    )
    cache_key = hashlib.md5(cache_payload.encode()).hexdigest()
    if use_cache and cache_key in _LOCAL_EXPLANATION_CACHE:
        cached_copy = dict(_LOCAL_EXPLANATION_CACHE[cache_key])
        cached_copy["from_cache"] = True
        return cached_copy

    # Compute actual prediction
    try:
        if hasattr(resolved_model, "predict_proba"):
            prob = float(resolved_model.predict_proba(X_raw)[0, 1])
        else:
            prob = float(resolved_model.predict(X_raw)[0])
            warnings.append("Estimator has no predict_proba; using binary prediction score.")
    except Exception as exc:
        return {
            "available": False,
            "shap_available": False,
            "model_name": model_name,
            "model_version": version_tag,
            "probability": None,
            "predicted_class": None,
            "risk_category": "Unknown",
            "feature_values": used_features,
            "missing_features": missing_feats,
            "contributions": [],
            "positive_contributors": [],
            "negative_contributors": [],
            "base_value": None,
            "shap_unit": None,
            "explainer_type": None,
            "human_readable_summary": f"Model inference failed: {exc}",
            "feature_statements": [],
            "warnings": warnings,
            "errors": [f"Model inference failed: {exc}"],
            "from_cache": False,
            "scientific_disclaimer": NON_CAUSAL_DISCLAIMER,
        }

    pred_class = int(prob >= 0.5)
    risk_category = probability_to_risk_category(prob)

    preproc, clf = _unwrap_pipeline(resolved_model)
    try:
        X_in = preproc.transform(X_raw) if preproc is not None else X_raw
    except Exception as exc:
        X_in = X_raw
        warnings.append(f"Preprocessor transform skipped during explanation: {exc}")

    if background_df is not None and not background_df.empty:
        X_bg_raw, _, bg_warns, bg_errs = _align_dataframe_features(background_df, cols, resolved_medians)
        warnings.extend(bg_warns)
        if X_bg_raw is None or bg_errs:
            X_bg_raw = _load_default_background(cols, resolved_medians)
    else:
        X_bg_raw = _load_default_background(cols, resolved_medians)

    X_bg = preproc.transform(X_bg_raw) if preproc is not None else X_bg_raw

    shap_available = False
    shap_error = None
    contributions: list[dict[str, Any]] = []
    pos_contribs: list[dict[str, Any]] = []
    neg_contribs: list[dict[str, Any]] = []
    base_value: float | None = None
    shap_unit: str | None = None
    explainer_type: str | None = None

    try:
        shap_vals, base_vals, explainer_type, shap_unit = _compute_shap_matrix(clf, X_in, X_bg)
        row_shap = shap_vals[0]
        base_value = round(float(base_vals[0]), 5)
        for idx, col in enumerate(cols):
            sv = float(row_shap[idx])
            direction = "positive" if sv > 1e-6 else "negative" if sv < -1e-6 else "neutral"
            f_desc = describe_model_feature(col)
            contributions.append({
                "feature": col,
                "feature_label": f_desc["label"],
                "feature_category": f_desc["category"],
                "feature_value": round(float(used_features[col]), 4),
                "is_imputed": col in missing_feats,
                "shap_value": round(sv, 5),
                "abs_shap": round(abs(sv), 5),
                "direction": direction,
            })
        contributions.sort(key=lambda r: r["abs_shap"], reverse=True)
        pos_contribs = [c for c in contributions if c["shap_value"] > 0]
        neg_contribs = sorted(
            [c for c in contributions if c["shap_value"] < 0],
            key=lambda r: r["shap_value"],
        )
        shap_available = True
    except Exception as exc:
        shap_error = f"SHAP attribution unavailable for {type(clf).__name__}: {exc}"
        warnings.append(shap_error)
        logger.warning("SHAP local explanation skipped: %s", shap_error)

    summary, statements = _format_human_statements(contributions, prob, risk_category)
    if not shap_available and shap_error:
        summary = (
            f"Estimated category is {risk_category} ({prob * 100:.1f}%), "
            f"but SHAP attribution is unavailable ({shap_error})."
        )

    result = {
        "available": True,
        "shap_available": shap_available,
        "shap_error": shap_error,
        "model_name": model_name,
        "model_version": version_tag,
        "model_metadata": model_meta,
        "probability": round(prob, 5),
        "predicted_class": pred_class,
        "risk_category": risk_category,
        "feature_columns_ordered": cols,
        "feature_values": used_features,
        "available_features": avail_feats,
        "missing_features": missing_feats,
        "contributions": contributions,
        "positive_contributors": pos_contribs,
        "negative_contributors": neg_contribs,
        "base_value": base_value,
        "shap_unit": shap_unit,
        "explainer_type": explainer_type,
        "human_readable_summary": summary,
        "feature_statements": statements,
        "explanation_limitations": _build_explanation_limitations(
            missing_feats, shap_available, shap_unit, model_meta
        ),
        "warnings": warnings,
        "errors": [],
        "from_cache": False,
        "scientific_disclaimer": NON_CAUSAL_DISCLAIMER,
    }
    if use_cache:
        _cache_put(_LOCAL_EXPLANATION_CACHE, cache_key, dict(result))
    return result


def explain_global(
    model_name: str = "best_model",
    model: Any = None,
    feature_cols: list[str] | None = None,
    df_eval: pd.DataFrame | None = None,
    target_col: str = "landslide_occurred",
    max_samples: int = 150,
    n_permutations: int = 8,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Compute global feature importance, permutation importance, and SHAP summary/beeswarm data."""
    resolved_model, cols, resolved_medians, model_info, load_err = _resolve_model_and_metadata(
        model_name=model_name,
        model=model,
        feature_cols=feature_cols,
    )
    version_tag = _get_model_version_tag(model_name, model_info)
    model_meta = build_model_metadata_summary(model_name, model_info, cols)

    if load_err or cols is None:
        return {
            "available": False,
            "shap_available": False,
            "model_name": model_name,
            "model_version": version_tag,
            "model_metadata": model_meta,
            "sample_count": 0,
            "feature_importance": [],
            "permutation_importance": [],
            "shap_summary": [],
            "method_comparison": [],
            "beeswarm_records": [],
            "warnings": [],
            "errors": [load_err or "Model unavailable."],
            "from_cache": False,
            "scientific_disclaimer": NON_CAUSAL_DISCLAIMER,
        }

    # Resolve evaluation dataset
    warnings: list[str] = []
    if df_eval is None:
        parquet_path = Path("data/processed/dataset.parquet")
        if not parquet_path.exists():
            return {
                "available": False,
                "shap_available": False,
                "model_name": model_name,
                "model_version": version_tag,
                "sample_count": 0,
                "feature_importance": [],
                "permutation_importance": [],
                "shap_summary": [],
                "beeswarm_records": [],
                "warnings": [],
                "errors": [f"Evaluation dataset not found at {parquet_path}."],
                "from_cache": False,
                "scientific_disclaimer": NON_CAUSAL_DISCLAIMER,
            }
        df_full = pd.read_parquet(parquet_path)
        # Prefer held-out chronological tail split matching training evaluation
        test_frac = 0.20
        if "event_date" in df_full.columns and df_full["event_date"].notna().any():
            df_full = df_full.sort_values("event_date")
        cutoff = int(len(df_full) * (1.0 - test_frac))
        df_eval = df_full.iloc[cutoff:].copy() if cutoff < len(df_full) else df_full.copy()

    X_all, ordered_df, align_warns, align_errs = _align_dataframe_features(
        df_eval, cols, resolved_medians
    )
    warnings.extend(align_warns)
    if X_all is None or ordered_df is None or align_errs:
        return {
            "available": False,
            "shap_available": False,
            "model_name": model_name,
            "model_version": version_tag,
            "sample_count": 0,
            "feature_importance": [],
            "permutation_importance": [],
            "shap_summary": [],
            "beeswarm_records": [],
            "warnings": warnings,
            "errors": align_errs,
            "from_cache": False,
            "scientific_disclaimer": NON_CAUSAL_DISCLAIMER,
        }

    cache_key_payload = json.dumps(
        {
            "model": model_name if model is None else f"custom_{id(model)}",
            "version": version_tag,
            "cols": cols,
            "n_rows": len(X_all),
            "max_samples": int(max_samples),
            "n_permutations": int(n_permutations),
            "first_sum": round(float(np.sum(X_all[:5])), 4),
        },
        sort_keys=True,
    )
    cache_key = hashlib.md5(cache_key_payload.encode()).hexdigest()
    if use_cache and cache_key in _GLOBAL_EXPLANATION_CACHE:
        cached = dict(_GLOBAL_EXPLANATION_CACHE[cache_key])
        cached["from_cache"] = True
        return cached

    # Deterministic representative subsample for responsive SHAP & permutation compute
    rng = np.random.default_rng(42)
    n_total = len(X_all)
    if n_total > max_samples:
        sample_idx = np.sort(rng.choice(n_total, size=max_samples, replace=False))
        X_sample = X_all[sample_idx]
        sampled_df = ordered_df.iloc[sample_idx].reset_index(drop=True)
    else:
        sample_idx = np.arange(n_total)
        X_sample = X_all
        sampled_df = ordered_df.reset_index(drop=True)

    preproc, clf = _unwrap_pipeline(resolved_model)

    # 1. Built-in importance (Tree feature_importances_ or Linear |coef_|)
    intrinsic_rows: list[dict[str, Any]] = []
    intrinsic_method = None
    if hasattr(clf, "feature_importances_"):
        intrinsic_method = "tree_feature_importance"
        vals = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        intrinsic_method = "standardized_linear_coefficient_abs"
        vals = np.abs(np.asarray(clf.coef_, dtype=float).ravel())
    else:
        vals = None

    if vals is not None and len(vals) == len(cols):
        for col, imp in zip(cols, vals):
            f_desc = describe_model_feature(col)
            intrinsic_rows.append({
                "feature": col,
                "feature_label": f_desc["label"],
                "feature_category": f_desc["category"],
                "importance": round(float(imp), 6),
                "method": intrinsic_method,
            })
        intrinsic_rows.sort(key=lambda r: r["importance"], reverse=True)
        for rank, row in enumerate(intrinsic_rows, start=1):
            row["rank"] = rank

    # 2. Permutation Feature Importance (requires target column with >= 2 classes)
    perm_rows: list[dict[str, Any]] = []
    perm_error = None
    if target_col in df_eval.columns:
        y_all = pd.to_numeric(df_eval[target_col], errors="coerce").dropna()
        if len(y_all) == len(df_eval) and y_all.nunique() >= 2:
            y_sub = y_all.iloc[sample_idx].values.astype(int)
            if len(np.unique(y_sub)) >= 2:
                try:
                    from sklearn.inspection import permutation_importance

                    scoring = "average_precision" if hasattr(resolved_model, "predict_proba") else "f1"
                    perm = permutation_importance(
                        resolved_model,
                        X_sample,
                        y_sub,
                        n_repeats=max(2, int(n_permutations)),
                        random_state=42,
                        scoring=scoring,
                    )
                    for col, m, s in zip(cols, perm.importances_mean, perm.importances_std):
                        f_desc = describe_model_feature(col)
                        perm_rows.append({
                            "feature": col,
                            "feature_label": f_desc["label"],
                            "feature_category": f_desc["category"],
                            "importance_mean": round(float(m), 6),
                            "importance_std": round(float(s), 6),
                            "scoring": scoring,
                        })
                    perm_rows.sort(key=lambda r: r["importance_mean"], reverse=True)
                    for rank, row in enumerate(perm_rows, start=1):
                        row["rank"] = rank
                except Exception as exc:
                    perm_error = f"Permutation importance failed: {exc}"
                    warnings.append(perm_error)
            else:
                perm_error = "Subsampled target contains fewer than 2 classes for permutation scoring."
                warnings.append(perm_error)
        else:
            perm_error = f"Target '{target_col}' lacks two classes for permutation scoring."
    else:
        perm_error = f"Target column '{target_col}' not provided; permutation importance skipped."

    # 3. Global SHAP Summary & Beeswarm Values
    shap_summary: list[dict[str, Any]] = []
    beeswarm_records: list[dict[str, Any]] = []
    shap_available = False
    shap_error = None
    shap_unit = None
    explainer_type = None

    try:
        X_in = preproc.transform(X_sample) if preproc is not None else X_sample
        shap_matrix, _, explainer_type, shap_unit = _compute_shap_matrix(clf, X_in, X_in)

        # Min-max normalise raw feature values per feature strictly for beeswarm colour mapping
        col_mins = sampled_df.min(axis=0)
        col_maxs = sampled_df.max(axis=0)

        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        mean_signed = np.mean(shap_matrix, axis=0)
        std_shap = np.std(shap_matrix, axis=0)

        for idx, col in enumerate(cols):
            f_desc = describe_model_feature(col)
            shap_summary.append({
                "feature": col,
                "feature_label": f_desc["label"],
                "feature_category": f_desc["category"],
                "mean_abs_shap": round(float(mean_abs[idx]), 5),
                "mean_shap": round(float(mean_signed[idx]), 5),
                "std_shap": round(float(std_shap[idx]), 5),
                "min_shap": round(float(np.min(shap_matrix[:, idx])), 5),
                "max_shap": round(float(np.max(shap_matrix[:, idx])), 5),
            })

        shap_summary.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
        for rank, row in enumerate(shap_summary, start=1):
            row["rank"] = rank

        span_map = {}
        for col in cols:
            lo = float(col_mins[col])
            hi = float(col_maxs[col])
            span_map[col] = (lo, hi - lo if (hi - lo) > 1e-12 else 1.0)

        for row_i in range(len(sampled_df)):
            for col_j, col in enumerate(cols):
                raw_fval = float(sampled_df.iloc[row_i][col])
                lo, span = span_map[col]
                norm_val = (raw_fval - lo) / span
                beeswarm_records.append({
                    "sample_idx": int(sample_idx[row_i]),
                    "feature": col,
                    "shap_value": round(float(shap_matrix[row_i, col_j]), 5),
                    "feature_value": round(raw_fval, 4),
                    "feature_value_norm": round(float(np.clip(norm_val, 0.0, 1.0)), 4),
                })

        shap_available = True
    except Exception as exc:
        shap_error = f"Global SHAP computation unavailable for {type(clf).__name__}: {exc}"
        warnings.append(shap_error)
        logger.warning("Global SHAP failed: %s", shap_error)

    # Normalized share (%) per explainer strictly across the trained features
    shap_map = {r["feature"]: r["mean_abs_shap"] for r in shap_summary}
    perm_map = {r["feature"]: max(0.0, float(r["importance_mean"])) for r in perm_rows}
    intr_map = {r["feature"]: max(0.0, float(r["importance"])) for r in intrinsic_rows}
    shap_total = sum(shap_map.values()) or 1.0
    perm_total = sum(perm_map.values()) or 1.0
    intr_total = sum(intr_map.values()) or 1.0
    method_comparison: list[dict[str, Any]] = []
    for col in cols:
        f_desc = describe_model_feature(col)
        method_comparison.append({
            "feature": col,
            "feature_label": f_desc["label"],
            "feature_category": f_desc["category"],
            "shap_share_pct": round((shap_map[col] / shap_total) * 100.0, 2) if col in shap_map and shap_summary else None,
            "permutation_share_pct": round((perm_map[col] / perm_total) * 100.0, 2) if col in perm_map and perm_rows else None,
            "intrinsic_share_pct": round((intr_map[col] / intr_total) * 100.0, 2) if col in intr_map and intrinsic_rows else None,
        })

    result = {
        "available": bool(shap_available or intrinsic_rows or perm_rows),
        "shap_available": shap_available,
        "shap_error": shap_error,
        "perm_error": perm_error,
        "model_name": model_name,
        "model_version": version_tag,
        "model_metadata": model_meta,
        "explainer_type": explainer_type,
        "shap_unit": shap_unit,
        "sample_count": int(len(sampled_df)),
        "total_eval_rows": int(n_total),
        "feature_columns": cols,
        "feature_importance": intrinsic_rows,
        "intrinsic_method": intrinsic_method,
        "permutation_importance": perm_rows,
        "shap_summary": shap_summary,
        "method_comparison": method_comparison,
        "beeswarm_records": beeswarm_records,
        "warnings": warnings,
        "errors": [],
        "from_cache": False,
        "scientific_disclaimer": NON_CAUSAL_DISCLAIMER,
    }
    if use_cache:
        _cache_put(_GLOBAL_EXPLANATION_CACHE, cache_key, dict(result))
    return result

