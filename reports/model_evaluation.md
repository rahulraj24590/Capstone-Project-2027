# LandslideGuard — Model Evaluation Report

**Generated:** 2026-11-10T16:16:18.547546+00:00

> All metrics are computed from actual model outputs on held-out data.

---

## Feature Columns Used

```
slope_degrees
elevation_m
rainfall_24h_mm
rainfall_72h_mm
rainfall_1h_mm
vegetation_index
soil_moisture
```

---

## Results by Model / Split


### logistic_regression

- **Split:** validation
- **N samples:** 240
- **Positives:** 61 (25.4%)
- **Precision:** 0.9385
- **Recall:** 1.0000
- **F1-Score:** 0.9683
- **ROC-AUC:** 1.0
- **PR-AUC:** 1.0
- **Brier Score:** 0.011231397399035422

**Confusion Matrix:**
```
TN=175  FP=4
FN=0  TP=61
```

### random_forest

- **Split:** validation
- **N samples:** 240
- **Positives:** 61 (25.4%)
- **Precision:** 0.9242
- **Recall:** 1.0000
- **F1-Score:** 0.9606
- **ROC-AUC:** 0.9989009982599139
- **PR-AUC:** 0.9967387857897292
- **Brier Score:** 0.020268751383366777

**Confusion Matrix:**
```
TN=174  FP=5
FN=0  TP=61
```

### gradient_boosting

- **Split:** validation
- **N samples:** 240
- **Positives:** 61 (25.4%)
- **Precision:** 0.9516
- **Recall:** 0.9672
- **F1-Score:** 0.9593
- **ROC-AUC:** 0.9992673321732759
- **PR-AUC:** 0.9979774324036621
- **Brier Score:** 0.011472083670279435

**Confusion Matrix:**
```
TN=176  FP=3
FN=2  TP=59
```

### xgboost

- **Split:** validation
- **N samples:** 240
- **Positives:** 61 (25.4%)
- **Precision:** 0.9375
- **Recall:** 0.9836
- **F1-Score:** 0.9600
- **ROC-AUC:** 0.999267332173276
- **PR-AUC:** 0.9979665825977302
- **Brier Score:** 0.012551329098641872

**Confusion Matrix:**
```
TN=175  FP=4
FN=1  TP=60
```

### logistic_regression_test

- **Split:** test
- **N samples:** 400
- **Positives:** 103 (25.8%)
- **Precision:** 0.9450
- **Recall:** 1.0000
- **F1-Score:** 0.9717
- **ROC-AUC:** 1.0
- **PR-AUC:** 1.0
- **Brier Score:** 0.008021195835387008

**Confusion Matrix:**
```
TN=291  FP=6
FN=0  TP=103
```

---

## Important Caveats

- Metrics marked with * are computed on **synthetic development data** and DO NOT reflect real-world performance.
- Model superiority claims require rigorous comparison on authentic, geographically diverse datasets.
- PR-AUC is reported as the primary metric due to class imbalance.