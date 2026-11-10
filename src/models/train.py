"""
Model training pipeline for LandslideGuard.

Trains multiple classifiers for landslide susceptibility prediction,
selects the best model by PR-AUC on the validation set, and saves
all artefacts to the models/ directory.

Design decisions:
- Random Forest is the primary candidate due to its robustness to
  correlated features and built-in feature importance.
- XGBoost is included as a strong gradient boosting baseline.
- Logistic Regression serves as an interpretable baseline.
- class_weight='balanced' addresses label imbalance without synthetic oversampling.
- Spatial or chronological splits are preferred where data permits.
  For synthetic data, a random stratified split is used with a fixed seed.
- PR-AUC is the primary selection metric because landslide events are rare
  (imbalanced classes) and recall of positive events is critical.

Usage:
    python -m src.models.train
    python -m src.models.train --data-path data/processed/dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly as `python src/models/train.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from src.data_processing.feature_engineering import (
    EXTENDED_FEATURE_COLUMNS,
    select_available_features,
)
from src.models.evaluate import compute_metrics, save_evaluation_report
from src.models.model_registry import save_model
from src.utils.logging_config import get_logger, setup_logging
from src.utils.settings import get_setting

logger = get_logger(__name__)

RANDOM_SEED = int(get_setting("model", "random_seed") or 42)
TEST_SIZE = float(get_setting("model", "test_size") or 0.20)
VAL_SIZE = float(get_setting("model", "val_size") or 0.15)
TARGET_COL = "landslide_occurred"


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_training_data(data_path: str | Path) -> pd.DataFrame:
    """Load processed training data.

    Args:
        data_path: Path to Parquet or CSV file.

    Returns:
        DataFrame with features and target column.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If target column is missing.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}. "
            "Run: python scripts/prepare_data.py"
        )

    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    pos = df[TARGET_COL].sum()
    neg = (df[TARGET_COL] == 0).sum()
    logger.info(
        "Training data loaded: %d rows | %d positives (%.1f%%) | %d negatives",
        len(df), pos, 100 * pos / len(df), neg,
    )
    return df


# ─────────────────────────────────────────────
# Data splitting
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame, feature_cols: list[str]):
    """Create reproducible train / validation / test splits.

    If 'event_date' is present, the test set is the most recent 20%
    chronologically (temporal split).  Otherwise, a stratified random
    split is used.

    Args:
        df: Processed DataFrame with target column.
        feature_cols: List of feature column names.

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Drop rows where target is NaN
    df = df[df[TARGET_COL].notna()].copy()

    if "event_date" in df.columns and df["event_date"].notna().any():
        df = df.sort_values("event_date")
        n = len(df)
        test_cutoff = int(n * (1 - TEST_SIZE))
        val_cutoff = int(test_cutoff * (1 - VAL_SIZE))

        train_df = df.iloc[:val_cutoff]
        val_df = df.iloc[val_cutoff:test_cutoff]
        test_df = df.iloc[test_cutoff:]
        logger.info(
            "Chronological split: train=%d, val=%d, test=%d",
            len(train_df), len(val_df), len(test_df),
        )
    else:
        train_val, test = train_test_split(
            df, test_size=TEST_SIZE, stratify=df[TARGET_COL],
            random_state=RANDOM_SEED,
        )
        train, val = train_test_split(
            train_val, test_size=VAL_SIZE / (1 - TEST_SIZE),
            stratify=train_val[TARGET_COL], random_state=RANDOM_SEED,
        )
        train_df, val_df, test_df = train, val, test
        logger.info(
            "Stratified random split: train=%d, val=%d, test=%d",
            len(train_df), len(val_df), len(test_df),
        )

    # Fill NaN in features with median from training set
    medians = train_df[feature_cols].median()
    for split in [train_df, val_df, test_df]:
        for col in feature_cols:
            if col in split.columns:
                split[col] = split[col].fillna(medians[col])

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────

def build_candidate_pipelines() -> dict[str, Pipeline]:
    """Return candidate model pipelines.

    Each pipeline includes a StandardScaler (required for Logistic Regression;
    harmless for tree models).

    Returns:
        Dict mapping model name → sklearn Pipeline.
    """
    candidates = {}

    # Logistic Regression — interpretable baseline
    candidates["logistic_regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )),
    ])

    # Random Forest — primary model
    candidates["random_forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    # Gradient Boosting — comparison
    candidates["gradient_boosting"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )),
    ])

    # XGBoost — if available
    try:
        from xgboost import XGBClassifier
        pos_weight = 3  # rough class imbalance ratio; will be set dynamically
        candidates["xgboost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                verbosity=0,
            )),
        ])
        logger.info("XGBoost candidate added.")
    except ImportError:
        logger.warning("XGBoost not available; skipping.")

    return candidates


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train_and_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
) -> tuple[str, Pipeline, dict, dict]:
    """Train all candidate models and select the best by PR-AUC.

    Args:
        X_train: Training features array.
        y_train: Training labels array.
        X_val: Validation features array.
        y_val: Validation labels array.
        feature_cols: Feature column names (for logging).

    Returns:
        Tuple (best_model_name, best_pipeline, all_metrics_dict, all_trained_pipelines).
    """
    # Adjust XGBoost scale_pos_weight dynamically
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos = neg / max(pos, 1)

    candidates = build_candidate_pipelines()
    if "xgboost" in candidates:
        candidates["xgboost"].named_steps["clf"].set_params(
            scale_pos_weight=scale_pos
        )

    all_metrics: dict[str, dict] = {}
    best_name = None
    best_pr_auc = -1.0
    best_pipeline = None

    for name, pipeline in candidates.items():
        logger.info("Training: %s …", name)
        pipeline.fit(X_train, y_train)

        metrics = compute_metrics(pipeline, X_val, y_val, split="validation")
        all_metrics[name] = metrics

        pr_auc = metrics.get("pr_auc", 0.0)
        logger.info(
            "%s | val PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f",
            name, pr_auc, metrics.get("roc_auc", 0), metrics.get("f1", 0),
        )

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name
            best_pipeline = pipeline

    logger.info("Best model: %s (PR-AUC=%.4f)", best_name, best_pr_auc)
    return best_name, best_pipeline, all_metrics, candidates


# ─────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────

def run_training(data_path: str | Path | None = None) -> dict:
    """Full training pipeline: load → split → train → evaluate → save.

    Args:
        data_path: Path to processed dataset. Defaults to
                   data/processed/dataset.parquet.

    Returns:
        Dict containing best model name, test metrics, and feature list.
    """
    from src.models.evaluate import save_evaluation_report

    if data_path is None:
        data_path = Path("data/processed/dataset.parquet")

    df = load_training_data(data_path)

    # Determine available features
    feature_cols = select_available_features(df)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, feature_cols)

    best_name, best_pipeline, all_metrics, trained_candidates = train_and_select(
        X_train, y_train, X_val, y_val, feature_cols
    )

    # Final evaluation on held-out test set
    test_metrics = compute_metrics(best_pipeline, X_test, y_test, split="test")
    all_metrics[f"{best_name}_test"] = test_metrics
    logger.info("Test set evaluation: %s", test_metrics)

    # Save models
    save_model(
        best_pipeline,
        "best_model",
        metadata={
            "algorithm": best_name,
            "feature_columns": feature_cols,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_metrics": test_metrics,
            "validation_metrics": all_metrics.get(best_name, {}),
        },
        overwrite=True,
    )

    # Save individual fitted candidates (use trained_candidates, not new unfitted instances)
    for name, pipeline in trained_candidates.items():
        try:
            save_model(pipeline, f"candidate_{name}", metadata={
                "validation_metrics": all_metrics.get(name, {}),
                "feature_columns": feature_cols,
            }, overwrite=True)
        except Exception:
            pass  # Non-critical — best_model is already saved

    # Save feature column list
    feature_path = Path("models/feature_columns.json")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_path, "w") as f:
        json.dump({"feature_columns": feature_cols}, f, indent=2)

    # Save training stats (medians used for NaN fill)
    medians = df[feature_cols].median().to_dict()
    stats_path = Path("models/training_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"feature_medians": medians}, f, indent=2, default=float)

    # Save evaluation report
    save_evaluation_report(all_metrics, feature_cols)

    result = {
        "best_model": best_name,
        "feature_columns": feature_cols,
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
    }

    logger.info("Training pipeline complete.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LandslideGuard models.")
    parser.add_argument(
        "--data-path",
        default="data/processed/dataset.parquet",
        help="Path to processed training data.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    result = run_training(args.data_path)
    print("\nTraining complete.")
    print(f"Best model: {result['best_model']}")
    print(f"Test metrics: {result['test_metrics']}")
