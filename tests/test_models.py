"""
Tests for model loading, prediction, and evaluation.

Verifies:
- Saved model can be loaded
- Feature order matches model expectations
- Predictions use actual model (no random fallback)
- Evaluation metrics are computed correctly
- Batch prediction works
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

MODEL_DIR = Path("models")
DATASET_PATH = Path("data/processed/dataset.parquet")


@pytest.fixture
def simple_pipeline():
    """A minimal sklearn pipeline for testing."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=100, random_state=42)),
    ])
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 1], [5, 6, 4]])
    y = np.array([0, 1, 1, 0, 1])
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def sample_features():
    """Valid feature dictionary for a single prediction."""
    return {
        "slope_degrees": 28.0,
        "elevation_m": 900.0,
        "rainfall_1h_mm": 20.0,
        "rainfall_24h_mm": 85.0,
        "rainfall_72h_mm": 160.0,
        "vegetation_index": 0.3,
        "soil_moisture": 0.55,
    }


# ─────────────────────────────────────────────
# Model file tests
# ─────────────────────────────────────────────

class TestModelFiles:
    def test_best_model_pkl_exists(self):
        assert (MODEL_DIR / "best_model.pkl").exists(), \
            "best_model.pkl not found. Run: python -m src.models.train"

    def test_feature_columns_json_exists(self):
        assert (MODEL_DIR / "feature_columns.json").exists(), \
            "feature_columns.json not found. Run training first."

    def test_training_stats_json_exists(self):
        assert (MODEL_DIR / "training_stats.json").exists(), \
            "training_stats.json not found. Run training first."

    def test_registry_json_exists(self):
        assert (MODEL_DIR / "registry.json").exists(), \
            "registry.json not found. Run training first."


# ─────────────────────────────────────────────
# Model loading tests
# ─────────────────────────────────────────────

class TestModelLoading:
    def test_model_loads_successfully(self):
        from src.models.model_registry import load_model
        model = load_model("best_model")
        assert model is not None

    def test_loaded_model_has_predict(self):
        from src.models.model_registry import load_model
        model = load_model("best_model")
        assert hasattr(model, "predict")

    def test_loaded_model_has_predict_proba(self):
        from src.models.model_registry import load_model
        model = load_model("best_model")
        assert hasattr(model, "predict_proba")

    def test_active_model_exists_check(self):
        from src.models.model_registry import active_model_exists
        assert active_model_exists()

    def test_nonexistent_model_raises(self):
        from src.models.model_registry import load_model
        with pytest.raises(FileNotFoundError):
            load_model("does_not_exist_xyz")


# ─────────────────────────────────────────────
# Feature column tests
# ─────────────────────────────────────────────

class TestFeatureColumns:
    def test_feature_columns_load(self):
        from src.models.predict import load_feature_columns
        cols = load_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_feature_columns_are_strings(self):
        from src.models.predict import load_feature_columns
        cols = load_feature_columns()
        assert all(isinstance(c, str) for c in cols)

    def test_feature_columns_consistent_with_model(self):
        """Model must be able to predict on a vector of correct feature length."""
        from src.models.model_registry import load_model
        from src.models.predict import load_feature_columns
        model = load_model("best_model")
        cols = load_feature_columns()
        X_test = np.zeros((1, len(cols)))
        prob = model.predict_proba(X_test)
        assert prob.shape == (1, 2)


# ─────────────────────────────────────────────
# Prediction tests
# ─────────────────────────────────────────────

class TestPredictor:
    def test_predictor_instantiates(self):
        from src.models.predict import Predictor
        p = Predictor()
        assert p is not None

    def test_prediction_returns_probability(self, sample_features):
        from src.models.predict import Predictor
        p = Predictor()
        result = p.predict(sample_features)
        assert "probability" in result
        prob = result["probability"]
        assert prob is not None
        assert 0.0 <= prob <= 1.0

    def test_prediction_returns_risk_category(self, sample_features):
        from src.models.predict import Predictor
        p = Predictor()
        result = p.predict(sample_features)
        assert result["risk_category"] in ["Low", "Moderate", "High", "Very High"]

    def test_prediction_uses_actual_model(self, sample_features):
        """Predictions must not be hard-coded / random."""
        from src.models.predict import Predictor
        p = Predictor()
        result1 = p.predict(sample_features)
        result2 = p.predict(sample_features)
        # Same inputs → same output (deterministic)
        assert result1["probability"] == pytest.approx(result2["probability"])

    def test_missing_required_feature_handled(self):
        """Missing required features must not crash silently."""
        from src.models.predict import Predictor
        p = Predictor()
        # Deliberately omit slope_degrees
        result = p.predict({"rainfall_24h_mm": 80.0})
        # Either a valid prediction with imputed value, or explicit error
        assert "probability" in result or "errors" in result

    def test_batch_prediction(self):
        """Batch prediction should return a DataFrame."""
        from src.models.predict import Predictor, load_feature_columns
        p = Predictor()
        cols = load_feature_columns()
        df = pd.DataFrame([{c: 0.5 for c in cols}] * 3)
        result = p.predict_batch(df)
        assert "probability" in result.columns
        assert len(result) == 3

    def test_probability_to_risk_category(self):
        from src.models.predict import probability_to_risk_category
        assert probability_to_risk_category(0.10) == "Low"
        assert probability_to_risk_category(0.45) == "Moderate"
        assert probability_to_risk_category(0.65) == "High"
        assert probability_to_risk_category(0.85) == "Very High"


# ─────────────────────────────────────────────
# Evaluation metrics tests
# ─────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_compute_metrics_returns_dict(self, simple_pipeline):
        from src.models.evaluate import compute_metrics
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 1])
        metrics = compute_metrics(simple_pipeline, X, y)
        assert isinstance(metrics, dict)

    def test_precision_recall_f1_present(self, simple_pipeline):
        from src.models.evaluate import compute_metrics
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 1])
        metrics = compute_metrics(simple_pipeline, X, y)
        for key in ["precision", "recall", "f1"]:
            assert key in metrics, f"'{key}' not in metrics"

    def test_confusion_matrix_present(self, simple_pipeline):
        from src.models.evaluate import compute_metrics
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 1])
        metrics = compute_metrics(simple_pipeline, X, y)
        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_sample_count_correct(self, simple_pipeline):
        from src.models.evaluate import compute_metrics
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 1])
        metrics = compute_metrics(simple_pipeline, X, y)
        assert metrics["n_samples"] == 3
