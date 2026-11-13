# LandslideGuard — Comprehensive Model Evaluation & Validation Report

**Generated:** November 2026  
**Target:** Academic B.Tech CSE Final-Year Capstone Evaluation  
**Artifact Version:** Integrated Release (Phases 1–9)  
**Primary Metric:** Precision-Recall AUC (PR-AUC) due to Class Imbalance  

---

## 1. Feature Representation & Canonical Input Schema

The trained models in LandslideGuard operate on a strictly aligned 7-dimensional feature space persisted in `models/feature_columns.json`. At inference and explainability time, input feature dictionaries are mapped deterministically to this canonical order:

| Index | Feature Column | Unit / Domain | Source / Extraction Mechanism | Role in Hazard Modeling |
|:---:|:---|:---:|:---|:---|
| 0 | `slope_degrees` | Degrees (0°–90°) | SRTM 30 m DEM / Local fallback | Primary static conditioning factor (gravitational shear stress) |
| 1 | `elevation_m` | Meters above sea level | SRTM 30 m DEM / Open-Elevation | Orographic rainfall barrier & terrain relief proxy |
| 2 | `rainfall_24h_mm` | Millimeters ($mm$) | Open-Meteo Forecast / Archive API | Immediate triggering factor (pore-water pressure spike) |
| 3 | `rainfall_72h_mm` | Millimeters ($mm$) | Open-Meteo Forecast / Archive API | Antecedent moisture & cumulative soil saturation |
| 4 | `rainfall_1h_mm` | Millimeters ($mm$) | Open-Meteo Forecast / Archive API | Flash precipitation intensity |
| 5 | `vegetation_index` | Normalized index (-1 to +1) | Sentinel-2 Level-2A NDVI / Fallback | Root cohesion and surface shear strength proxy |
| 6 | `soil_moisture` | Fraction / Volumetric ($m^3/m^3$) | Open-Meteo 0–7 cm soil moisture | Subsurface saturation status |

---

## 2. Experimental Model Performance Ledger

All metrics below are computed strictly from actual evaluation outputs stored in `reports/metrics.json` and `models/registry.json`.

### 2.1 Validation Split Evaluation ($N = 240$, Chronological Split)
- **Positive Instances:** 61 (25.4%)
- **Negative Instances:** 179 (74.6%)

| Model / Algorithm | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Primary) | Brier Score (Calibration) | Confusion Matrix |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Logistic Regression** (Selected) | **0.9385** | **1.0000** | **0.9683** | **1.0000** | **1.0000** | **0.0112** | TN=175, FP=4<br>FN=0, TP=61 |
| **Random Forest** | 0.9242 | 1.0000 | 0.9606 | 0.9989 | 0.9967 | 0.0203 | TN=174, FP=5<br>FN=0, TP=61 |
| **Gradient Boosting** (sklearn) | 0.9516 | 0.9672 | 0.9593 | 0.9993 | 0.9980 | 0.0115 | TN=176, FP=3<br>FN=2, TP=59 |
| **XGBoost Classifier** | 0.9375 | 0.9836 | 0.9600 | 0.9993 | 0.9980 | 0.0126 | TN=175, FP=4<br>FN=1, TP=60 |

### 2.2 Independent Held-Out Test Evaluation ($N = 400$, Chronological Split)
- **Positive Instances:** 103 (25.75%)
- **Negative Instances:** 297 (74.25%)
- **Evaluated Model:** `models/best_model.pkl` (`Logistic Regression` with `StandardScaler`)

| Metric | Measured Score | Interpretation |
|:---|:---:|:---|
| **Precision** | **0.9450** | 94.5% of predicted landslide hazards were true positive events ($FP = 6$). |
| **Recall (Sensitivity)** | **1.0000** | 100.0% of true landslide events were identified ($FN = 0$). Zero missed events in benchmark. |
| **F1-Score** | **0.9717** | Harmonic mean of precision and recall. |
| **ROC-AUC** | **1.0000** | Area Under Receiver Operating Characteristic curve. |
| **PR-AUC (Primary)** | **1.0000** | Area Under Precision-Recall curve (uninflated by true negative majority). |
| **Brier Score** | **0.0080** | Mean squared difference between predicted probabilities and actual binary outcomes ($BS \rightarrow 0$ indicates near-perfect probabilistic calibration on benchmark). |

**Test Split Confusion Matrix ($N = 400$):**
```
                ┌───────────────────────┬───────────────────────┐
                │ Predicted Negative    │ Predicted Positive    │
┌───────────────┼───────────────────────┼───────────────────────┤
│ True Negative │ TN = 291 (72.75%)     │ FP = 6 (1.50%)        │
├───────────────┼───────────────────────┼───────────────────────┤
│ True Positive │ FN = 0   (0.00%)      │ TP = 103 (25.75%)     │
└───────────────┴───────────────────────┴───────────────────────┘
```

---

## 3. Methodological & Scientific Audit

### 3.1 Training and Testing Separation
The dataset partitioning strategy follows a strict chronological split implemented in `src/models/train.py`:
1. Total dataset: 2,000 records.
2. Training split: 1,360 records (68.0%).
3. Validation split: 240 records (12.0%, `val_size=0.15` of non-test records).
4. Held-out test split: 400 records (20.0%, `test_size=0.20` of total records).

Chronological splitting guarantees that future environmental events are never seen during the training phase, mirroring prospective operational deployment.

### 3.2 Data Leakage Prevention
To prevent information leakage across splits:
- **Feature Transformers:** `StandardScaler` and median imputers are fitted **strictly on the training split** (`src/data_processing/preprocessing.py`). The transformation parameters ($\mu_{train}, \sigma_{train}$) are applied forward to validation and test records without recomputing on the test data.
- **Missing-Value Imputation:** Terrain medians are derived exclusively from training rows and persisted in `models/training_stats.json`.
- **Zero-Imputation Refusal:** Missing rainfall values are **never** silently imputed as `0.0`. Imputing missing rainfall with zero creates severe false-negative bias (lowering risk artificially during sensor outages). If rainfall is missing, the system refuses full ML evaluation and triggers heuristic fallback with an explicit `UNAVAILABLE` flag.

### 3.3 Class Imbalance Analysis
In natural slope environments, landslide occurrences are rare events (typically $< 5\%$ of spatiotemporal observations). In our synthetic development dataset, the class ratio is 25.0% positive : 75.0% negative.
- **Class Weighting:** All estimators are trained with `class_weight='balanced'`, penalizing false negatives by the inverse class frequency:
  $$w_1 = \frac{N}{2 \cdot N_{pos}}, \quad w_0 = \frac{N}{2 \cdot N_{neg}}$$
- **Metric Selection:** In class-imbalanced settings, ROC-AUC can be deceptively optimistic because the False Positive Rate ($FPR = \frac{FP}{FP + TN}$) is dominated by the large number of true negatives ($TN$). Therefore, LandslideGuard uses **PR-AUC (Precision-Recall Area Under Curve)** as the decisive model-selection criterion.

### 3.4 Spatial Data Leakage & Spatial Autocorrelation
> [!IMPORTANT]
> **Spatial Data Leakage Analysis (Tobler's First Law of Geography):**  
> *"Everything is related to everything else, but near things are more related than distant things."*  
> Standard random or chronological train/test splits suffer from **spatial autocorrelation**. When landslide events occurring in the same drainage basin or mountain range are randomly divided between training and testing sets, the model learns the spatial signature (elevation, rock type, localized climate) rather than generalizable physical relationships.
>
> In LandslideGuard:
> 1. The near-perfect evaluation scores ($PR\text{-}AUC = 1.0000$, $Brier = 0.0080$) are an artifact of regularized synthetic data and spatial proximity.
> 2. For genuine operational deployment, **Spatial Block Cross-Validation** or **Leave-One-Region-Out (LORO)** cross-validation (e.g., train on Western Ghats, test on the Himalayas) is required to evaluate true cross-regional transferability.

### 3.5 Probabilistic Calibration (Brier Score)
In early warning systems, thresholding depends on calibrated probabilities. 
- Logistic Regression outputs calibrated probabilities directly via the logistic sigmoid link function:
  $$P(Y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$$
- As shown in the validation ledger, Logistic Regression achieved the lowest Brier Score ($BS = 0.0112$ validation, $0.0080$ test), outperforming uncalibrated Random Forest ($0.0203$) and XGBoost ($0.0126$).

---

## 4. Scientific Honesty & Generalization Statement

1. **Synthetic Nature of Training Data:** All models were trained and tested on `data/processed/dataset.parquet` which carries the explicit metadata tag `is_synthetic = 1`.
2. **Non-Operational Boundary:** The resulting model weights (`best_model.pkl`) demonstrate computational correctness, schema alignment, pipeline stability, and explainability integration. **They do not possess operational validity for life-safety decision-making.**
3. **No Field Certification:** These models have not been calibrated or backtested against official ground-truth catalogs from the Geological Survey of India (GSI) or National Disaster Management Authority (NDMA).
