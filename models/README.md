# LandslideGuard — Model Artifacts

This directory stores trained model artifacts and associated metadata.

## Files

| File | Description |
|------|-------------|
| `best_model.pkl` | Best performing model (joblib serialised pipeline) |
| `candidate_*.pkl` | Candidate models from training run |
| `feature_columns.json` | Ordered list of feature columns expected by the model |
| `registry.json` | Full model registry with metadata and metrics |
| `training_stats.json` | Dataset statistics at training time |

## Model Format

Models are serialised using `joblib` as sklearn `Pipeline` objects containing:
1. `StandardScaler` (fitted on training data)
2. The classifier (LogisticRegression / RandomForestClassifier / etc.)

## Loading a Model

```python
from src.models.model_registry import load_model
model = load_model("best_model")  # or any candidate name
```

## Important Notes

- **⚠️ Current models are trained on SYNTHETIC data** and do not have real-world predictive validity.
- Never overwrite `best_model.pkl` without first recording the old model's metrics.
- Model version, training date, and feature list are tracked in `registry.json`.
- Feature column order must match `feature_columns.json` exactly at inference time.

## Model Versioning

When retraining with new data:
1. Run `python src/models/train.py`
2. New models are saved with a timestamp prefix.
3. Evaluate all candidates with `python src/models/evaluate.py`
4. Update `best_model.pkl` only after confirming improvement on the test set.
5. Archive the previous best model before overwriting.

## Not in Version Control

Large model files should not be committed to Git. If sharing models:
- Use Git LFS for models < 100MB.
- Use DVC for large model versioning with remote storage.
- Document model checksums in `registry.json`.
