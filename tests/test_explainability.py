"""Unit and API tests for Phase 4 Explainable AI (SHAP & Feature Importance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from api import (
    ExplainPredictionRequest,
    explain_single_prediction,
    get_global_explainability,
)
from src.models.explainability import (
    EXPLAINABILITY_CONCEPTS,
    clear_explanation_cache,
    explain_global,
    explain_prediction,
)
from src.models.predict import Predictor, load_feature_columns


def _run_coro_sync(coro):
    """Execute a coroutine synchronously without creating an asyncio socketpair."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


@pytest.fixture(autouse=True)
def _clean_xai_cache():
    clear_explanation_cache()
    yield
    clear_explanation_cache()


def test_model_loading_and_shap_compatibility_across_saved_models():
    sample = {
        "slope_degrees": 32.0,
        "elevation_m": 1250.0,
        "rainfall_24h_mm": 110.0,
        "rainfall_72h_mm": 195.0,
        "rainfall_1h_mm": 22.0,
        "vegetation_index": 0.28,
        "soil_moisture": 0.68,
    }
    for model_name in ("best_model", "candidate_random_forest", "candidate_gradient_boosting", "candidate_xgboost"):
        res = explain_prediction(sample, model_name=model_name, use_cache=False)
        assert res["available"] is True, f"{model_name} failed: {res.get('errors')}"
        assert res["shap_available"] is True, f"{model_name} SHAP error: {res.get('shap_error')}"
        assert res["model_version"].startswith(model_name)
        assert len(res["contributions"]) == len(load_feature_columns())
        assert res["base_value"] is not None
        assert any(
            "assigned importance to" in stmt and "when producing this estimate" in stmt
            for stmt in res["feature_statements"]
        )


def test_feature_ordering_invariance_dict_and_dataframe():
    ordered = {
        "slope_degrees": 28.0,
        "elevation_m": 900.0,
        "rainfall_24h_mm": 85.0,
        "rainfall_72h_mm": 140.0,
        "rainfall_1h_mm": 15.0,
        "vegetation_index": 0.35,
        "soil_moisture": 0.55,
    }
    reversed_dict = {k: ordered[k] for k in reversed(list(ordered.keys()))}

    r1 = explain_prediction(ordered, model_name="best_model", use_cache=False)
    r2 = explain_prediction(reversed_dict, model_name="best_model", use_cache=False)
    assert r1["probability"] == pytest.approx(r2["probability"], abs=1e-9)

    s1 = {c["feature"]: c["shap_value"] for c in r1["contributions"]}
    s2 = {c["feature"]: c["shap_value"] for c in r2["contributions"]}
    for feat in ordered:
        assert s1[feat] == pytest.approx(s2[feat], abs=1e-9)

    # Reordered global DataFrame columns must still yield identical SHAP summary & permutation values
    df = pd.read_parquet("data/processed/dataset.parquet").tail(60).copy()
    feat_cols = load_feature_columns()
    shuffled_df = df[list(reversed(feat_cols)) + ["landslide_occurred"]]

    g1 = explain_global(model_name="best_model", df_eval=df, max_samples=50, n_permutations=3, use_cache=False)
    g2 = explain_global(model_name="best_model", df_eval=shuffled_df, max_samples=50, n_permutations=3, use_cache=False)
    assert g1["available"] is True and g2["available"] is True
    assert any("reordered" in w.lower() for w in g2["warnings"])
    m1 = {r["feature"]: r["mean_abs_shap"] for r in g1["shap_summary"]}
    m2 = {r["feature"]: r["mean_abs_shap"] for r in g2["shap_summary"]}
    for feat in feat_cols:
        assert m1[feat] == pytest.approx(m2[feat], abs=1e-9)


def test_individual_explanation_split_missing_features_and_cache():
    partial = {
        "slope_degrees": 36.0,
        "rainfall_24h_mm": 140.0,
        # Intentionally omit elevation_m, vegetation_index, soil_moisture, etc.
    }
    first = explain_prediction(partial, model_name="best_model", use_cache=True)
    assert first["available"] is True
    assert first["shap_available"] is True
    assert first["from_cache"] is False
    assert "elevation_m" in first["missing_features"]
    assert len(first["warnings"]) > 0
    assert len(first["positive_contributors"]) + len(first["negative_contributors"]) > 0
    assert "causation" in first["scientific_disclaimer"].lower()

    # Identical call hits bounded cache immediately
    second = explain_prediction(partial, model_name="best_model", use_cache=True)
    assert second["from_cache"] is True
    assert second["probability"] == first["probability"]
    assert second["contributions"] == first["contributions"]


def test_global_feature_importance_permutation_beeswarm_and_concepts():
    res = explain_global(
        model_name="candidate_random_forest",
        max_samples=60,
        n_permutations=4,
        use_cache=True,
    )
    assert res["available"] is True
    assert res["shap_available"] is True
    assert res["sample_count"] == 60
    assert len(res["shap_summary"]) == 7
    assert len(res["permutation_importance"]) == 7
    assert len(res["feature_importance"]) == 7
    assert len(res["beeswarm_records"]) == 60 * 7
    assert res["from_cache"] is False

    # Second call returns cached result
    cached = explain_global(
        model_name="candidate_random_forest",
        max_samples=60,
        n_permutations=4,
        use_cache=True,
    )
    assert cached["from_cache"] is True

    concept_names = {c["concept"] for c in EXPLAINABILITY_CONCEPTS}
    assert any("Feature Importance" in c for c in concept_names)
    assert any("SHAP" in c for c in concept_names)
    assert "Correlation" in concept_names
    assert "Causation" in concept_names


def test_missing_and_untrained_model_handling():
    missing_local = explain_prediction({"slope_degrees": 20.0}, model_name="nonexistent_model_999")
    assert missing_local["available"] is False
    assert missing_local["shap_available"] is False
    assert missing_local["contributions"] == []
    assert len(missing_local["errors"]) >= 1

    missing_global = explain_global(model_name="nonexistent_model_999")
    assert missing_global["available"] is False
    assert missing_global["shap_summary"] == []
    assert len(missing_global["errors"]) >= 1

    untrained = LogisticRegression()
    untrained_res = explain_prediction(
        {"slope_degrees": 25.0},
        model_name="untrained_lr",
        model=untrained,
        feature_cols=["slope_degrees"],
        medians={"slope_degrees": 15.0},
        use_cache=False,
    )
    assert untrained_res["available"] is False
    assert "not trained" in untrained_res["errors"][0].lower()


def test_unsupported_model_type_and_shap_error_never_fabricates(monkeypatch):
    # Fitted KNN classifier lacks coef_ and tree structure -> SHAP must report unavailable cleanly
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(int)
    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)

    knn_local = explain_prediction(
        {"slope_degrees": 1.0, "rainfall_24h_mm": 0.5},
        model_name="knn_unsupported",
        model=knn,
        feature_cols=["slope_degrees", "rainfall_24h_mm"],
        medians={"slope_degrees": 0.0, "rainfall_24h_mm": 0.0},
        use_cache=False,
    )
    assert knn_local["available"] is True
    assert knn_local["shap_available"] is False
    assert knn_local["contributions"] == []
    assert "unsupported" in (knn_local["shap_error"] or "").lower()

    # Global KNN still runs permutation importance cleanly while keeping shap_available=False
    df_knn = pd.DataFrame({
        "slope_degrees": X[:, 0],
        "rainfall_24h_mm": X[:, 1],
        "landslide_occurred": y,
    })
    knn_global = explain_global(
        model_name="knn_unsupported",
        model=knn,
        feature_cols=["slope_degrees", "rainfall_24h_mm"],
        df_eval=df_knn,
        n_permutations=3,
        use_cache=False,
    )
    assert knn_global["shap_available"] is False
    assert knn_global["shap_summary"] == []
    assert len(knn_global["permutation_importance"]) == 2

    # Missing required column in DataFrame returns clear error
    bad_df_global = explain_global(
        model_name="best_model",
        df_eval=pd.DataFrame({"wrong_col": [1.0, 2.0]}),
        use_cache=False,
    )
    assert bad_df_global["available"] is False
    assert "missing required model features" in bad_df_global["errors"][0].lower()


def test_explainability_api_endpoints():
    global_payload = _run_coro_sync(
        get_global_explainability(model_name="best_model", max_samples=40, include_beeswarm=True)
    )
    assert global_payload["shap_available"] is True
    assert len(global_payload["shap_summary"]) == 7
    assert len(global_payload["beeswarm_records"]) == 40 * 7

    req = ExplainPredictionRequest(
        model_name="best_model",
        slope_degrees=30.0,
        elevation_m=1000.0,
        rainfall_1h_mm=15.0,
        rainfall_24h_mm=90.0,
        rainfall_72h_mm=150.0,
        vegetation_index=0.35,
        soil_moisture=0.60,
    )
    pred_payload = _run_coro_sync(explain_single_prediction(req))
    assert pred_payload["available"] is True
    assert pred_payload["shap_available"] is True
    assert len(pred_payload["contributions"]) == 7

    with pytest.raises(HTTPException) as exc_info:
        _run_coro_sync(get_global_explainability(model_name="missing_model_xyz", max_samples=20))
    assert exc_info.value.status_code == 503


def test_zero_input_refusal_never_fabricates_shap():
    empty_res = explain_prediction({}, model_name="best_model", use_cache=False)
    assert empty_res["available"] is False
    assert empty_res["shap_available"] is False
    assert empty_res["contributions"] == []
    assert any("insufficient input" in e.lower() or "no model features" in e.lower() for e in empty_res["errors"])

    all_none = explain_prediction(
        {col: None for col in load_feature_columns()},
        model_name="best_model",
        use_cache=False,
    )
    assert all_none["available"] is False
    assert all_none["shap_available"] is False
    assert all_none["contributions"] == []


def test_prediction_explanation_consistency_and_model_metadata():
    sample = {
        "slope_degrees": 34.0,
        "elevation_m": 1350.0,
        "rainfall_24h_mm": 135.0,
        "rainfall_72h_mm": 220.0,
        "rainfall_1h_mm": 28.0,
        "vegetation_index": 0.25,
        "soil_moisture": 0.72,
        "unused_extra_input": 999.0,
    }
    pred = Predictor().predict(sample)
    exp = explain_prediction(sample, model_name="best_model", use_cache=False)

    assert exp["available"] is True and exp["shap_available"] is True
    assert exp["probability"] == pytest.approx(pred["probability"], abs=1e-4)
    assert exp["risk_category"] == pred["risk_category"]
    assert exp["predicted_class"] == pred["predicted_class"]

    # Unused keys must never appear in feature list or contributions
    trained_cols = load_feature_columns()
    assert exp["feature_columns_ordered"] == trained_cols
    assert {c["feature"] for c in exp["contributions"]} == set(trained_cols)
    assert "unused_extra_input" not in {c["feature"] for c in exp["contributions"]}

    # Full model metadata & limitations present
    meta = exp["model_metadata"]
    assert meta["model_name"] == "best_model"
    assert meta["training_dataset_source"] != ""
    assert meta["training_date"] != "Unavailable"
    assert meta["feature_list"] == trained_cols
    assert meta["model_version"].startswith("best_model")
    assert "test" in meta["evaluation_metrics"] and "f1" in meta["evaluation_metrics"]["test"]
    assert meta["validation_approach"] != "Unavailable"
    assert any("causation" in lim.lower() for lim in exp["explanation_limitations"])

    # Global method comparison covers only trained features
    g_res = explain_global(model_name="candidate_random_forest", max_samples=50, n_permutations=3, use_cache=False)
    assert {r["feature"] for r in g_res["method_comparison"]} == set(trained_cols)
    assert all(r["intrinsic_share_pct"] is not None for r in g_res["method_comparison"])

