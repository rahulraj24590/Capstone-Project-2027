"""
Model evaluation metrics for LandslideGuard.

Computes and stores comprehensive evaluation metrics including:
- Precision, Recall, F1
- ROC-AUC, PR-AUC
- Confusion matrix
- Brier score (calibration)
- Classification report

All metrics are derived from actual model outputs, not fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running directly as `python src/models/evaluate.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

REPORTS_DIR = Path("reports")


def compute_metrics(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split: str = "test",
    threshold: float = 0.5,
) -> dict:
    """Compute a comprehensive set of classification metrics.

    Args:
        model: Fitted sklearn-compatible classifier with predict_proba.
        X: Feature matrix.
        y: True binary labels.
        split: Label for the data split (e.g., "validation", "test").
        threshold: Probability threshold for binary classification.

    Returns:
        Dict of metric names to values.
    """
    y = np.array(y)

    # Probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler
        scores = model.decision_function(X)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        logger.warning("Model has no predict_proba or decision_function. Using hard predictions.")
        y_prob = model.predict(X).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict = {"split": split, "threshold": threshold}

    # Basic classification metrics
    metrics["precision"] = float(precision_score(y, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y, y_pred, zero_division=0))

    # Curve-based metrics (require both classes present)
    unique_classes = np.unique(y)
    if len(unique_classes) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y, y_prob))

        fpr, tpr, _ = roc_curve(y, y_prob)
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

        prec_curve, rec_curve, _ = precision_recall_curve(y, y_prob)
        metrics["pr_curve"] = {
            "precision": prec_curve.tolist(),
            "recall": rec_curve.tolist(),
        }
    else:
        logger.warning("Only one class in labels; ROC-AUC and PR-AUC cannot be computed.")
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["brier_score"] = None

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Classification report
    metrics["classification_report"] = classification_report(
        y, y_pred, output_dict=True, zero_division=0
    )

    # Sample sizes
    metrics["n_samples"] = int(len(y))
    metrics["n_positive"] = int(y.sum())
    metrics["n_negative"] = int((y == 0).sum())
    metrics["positive_rate"] = float(y.mean())

    logger.info(
        "[%s] Precision=%.4f Recall=%.4f F1=%.4f ROC-AUC=%s PR-AUC=%s",
        split,
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        f"{metrics.get('roc_auc', 'N/A'):.4f}" if metrics.get("roc_auc") is not None else "N/A",
        f"{metrics.get('pr_auc', 'N/A'):.4f}" if metrics.get("pr_auc") is not None else "N/A",
    )

    return metrics


def compute_feature_importance(
    model,
    feature_names: list[str],
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    n_permutations: int = 10,
) -> pd.DataFrame:
    """Compute feature importance using tree-based or permutation methods.

    For tree-based models, uses built-in feature_importances_.
    Falls back to permutation importance when built-in is not available.

    Args:
        model: Fitted model (pipeline or estimator).
        feature_names: Names of feature columns.
        X_test: Test features (for permutation importance).
        y_test: Test labels (for permutation importance).
        n_permutations: Number of permutation repeats.

    Returns:
        DataFrame with columns ['feature', 'importance', 'method'].
    """
    # Unwrap pipeline to get the classifier
    clf = model
    if hasattr(model, "named_steps"):
        # Last step in pipeline
        clf = list(model.named_steps.values())[-1]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        method = "tree_feature_importance"
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
            "method": method,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp

    # Fall back to permutation importance
    if X_test is not None and y_test is not None:
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=n_permutations,
                random_state=42,
                scoring="average_precision",
            )
            df_imp = pd.DataFrame({
                "feature": feature_names,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
                "method": "permutation",
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            return df_imp
        except Exception as e:
            logger.warning("Permutation importance failed: %s", e)

    logger.warning("No feature importance method available for this model.")
    return pd.DataFrame({"feature": feature_names, "importance": [np.nan] * len(feature_names)})


def save_evaluation_report(
    all_metrics: dict,
    feature_cols: list[str],
) -> None:
    """Write model evaluation results to reports/model_evaluation.md.

    Args:
        all_metrics: Dict of split/model name → metrics dict.
        feature_cols: List of feature column names used.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "model_evaluation.md"

    lines = [
        "# LandslideGuard — Model Evaluation Report",
        f"\n**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "\n> All metrics are computed from actual model outputs on held-out data.",
        "\n---\n",
        "## Feature Columns Used\n",
        "```",
        "\n".join(feature_cols),
        "```\n",
        "---\n",
        "## Results by Model / Split\n",
    ]

    for model_split, metrics in all_metrics.items():
        lines.append(f"\n### {model_split}\n")
        lines.append(f"- **Split:** {metrics.get('split', 'unknown')}")
        lines.append(f"- **N samples:** {metrics.get('n_samples', 'N/A')}")
        lines.append(f"- **Positives:** {metrics.get('n_positive', 'N/A')} ({metrics.get('positive_rate', 0):.1%})")
        lines.append(f"- **Precision:** {metrics.get('precision', 'N/A'):.4f}")
        lines.append(f"- **Recall:** {metrics.get('recall', 'N/A'):.4f}")
        lines.append(f"- **F1-Score:** {metrics.get('f1', 'N/A'):.4f}")
        lines.append(f"- **ROC-AUC:** {metrics.get('roc_auc', 'N/A')}")
        lines.append(f"- **PR-AUC:** {metrics.get('pr_auc', 'N/A')}")
        lines.append(f"- **Brier Score:** {metrics.get('brier_score', 'N/A')}")

        cm = metrics.get("confusion_matrix")
        if cm:
            lines.append(f"\n**Confusion Matrix:**")
            lines.append(f"```")
            lines.append(f"TN={cm[0][0]}  FP={cm[0][1]}")
            lines.append(f"FN={cm[1][0]}  TP={cm[1][1]}")
            lines.append(f"```")

    lines.append("\n---\n")
    lines.append("## Important Caveats\n")
    lines.append(
        "- Metrics marked with * are computed on **synthetic development data** "
        "and DO NOT reflect real-world performance.\n"
        "- Model superiority claims require rigorous comparison on authentic, "
        "geographically diverse datasets.\n"
        "- PR-AUC is reported as the primary metric due to class imbalance."
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save raw metrics as JSON
    json_path = REPORTS_DIR / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # Remove non-serialisable curve data before saving
        serialisable = {}
        for k, v in all_metrics.items():
            serialisable[k] = {
                kk: vv for kk, vv in v.items()
                if kk not in ("roc_curve", "pr_curve", "classification_report")
            }
        json.dump(serialisable, f, indent=2, default=float)

    logger.info("Evaluation report saved to %s", report_path)


def run_evaluation(
    model_name: str = "best_model",
    data_path: str | Path = "data/processed/dataset.parquet",
) -> dict:
    """Evaluate a saved model on the held-out chronological test split.

    Args:
        model_name: Name of the registered model to evaluate.
        data_path: Path to the processed dataset.

    Returns:
        Dictionary of computed test metrics.
    """
    from src.models.model_registry import load_model
    from src.models.predict import load_feature_columns
    from src.models.train import load_training_data, split_data

    model = load_model(model_name)
    feature_cols = load_feature_columns()
    df = load_training_data(data_path)
    _, _, X_test, _, _, y_test = split_data(df, feature_cols)

    metrics = compute_metrics(model, X_test, y_test, split="test")
    imp_df = compute_feature_importance(model, feature_cols, X_test, y_test)
    metrics["feature_importance"] = imp_df.to_dict(orient="records")
    return metrics


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    parser = argparse.ArgumentParser(description="Evaluate a saved LandslideGuard model.")
    parser.add_argument("--model", default="best_model", help="Model name (without .pkl).")
    parser.add_argument(
        "--data-path",
        default="data/processed/dataset.parquet",
        help="Path to processed dataset.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    results = run_evaluation(model_name=args.model, data_path=args.data_path)
    print(f"\nModel Evaluation ({args.model} — Test Split):")
    print(f"  Samples:   {results['n_samples']} ({results['n_positive']} positive)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  ROC-AUC:   {results.get('roc_auc')}")
    print(f"  PR-AUC:    {results.get('pr_auc')}")
    print(f"  Brier:     {results.get('brier_score')}")
