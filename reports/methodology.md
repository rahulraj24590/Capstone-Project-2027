# LandslideGuard — Methodology Report

**Project:** LandslideGuard: AI-Based Dynamic Landslide Risk Prediction and Early Warning System  
**Type:** Academic Capstone Project — B.Tech Computer Science and Engineering  
**Date:** October 2026  
**Status:** Experimental — not operationally validated

---

## 1. Problem Formulation

### 1.1 What Is Being Predicted?

Landslide susceptibility and dynamic risk are **distinct** tasks:

| Concept | Description | Time Horizon |
|---------|-------------|-------------|
| **Static Susceptibility** | Probability that a terrain unit is prone to landslides based on terrain, geology, and land cover | Long-term (years) |
| **Dynamic Risk Estimate** | Susceptibility adjusted by current or recent environmental conditions, primarily rainfall | Hours to days |
| **Operational Early Warning** | A validated decision to alert stakeholders, based on real-time data and calibrated thresholds | Minutes to hours |

This project implements (1) and (2). It does **not** implement (3) — operational warnings require independent validation by geoscience authorities, real-time sensor integration, and legal accountability frameworks that are beyond the scope of a university capstone.

### 1.2 Target Variable

`landslide_occurred` — Binary:
- `1`: A landslide event was recorded at this location during the observation period.
- `0`: No landslide was recorded (absence evidence, not evidence of absence).

**Class imbalance:** Landslides are rare events. In the synthetic development dataset, approximately 25% of observations are positive. In real-world datasets, positive rates can be far lower.

---

## 2. Dataset Sources

### 2.1 Primary Training Data

**Synthetic Development Dataset (LABELLED SYNTHETIC)**

During development, a clearly labelled synthetic dataset was generated because:
- The NASA Global Landslide Catalog requires a NASA Earthdata account and the download was not automated during initial development.
- Field-collected datasets with all required environmental features are not freely available at the required spatial and temporal resolution.

The synthetic dataset was designed with physically plausible feature distributions:
- Rainfall values sampled from log-normal distributions consistent with Indian monsoon climatology.
- Slope angles from terrain-appropriate ranges for the Indian subcontinent.
- Elevation sampled to reflect realistic hill-slope settings.
- Landslide occurrence correlated with slope and rainfall intensity (with noise) to produce a learnable signal.

**⚠️ All model metrics are computed on synthetic data and do not reflect real-world predictive performance.**

### 2.2 Investigated Real-World Sources

| Source | URL | Status |
|--------|-----|--------|
| NASA Global Landslide Catalog | https://pmm.nasa.gov/data-access/downloads/gpm | Requires Earthdata account |
| Open-Meteo (weather) | https://open-meteo.com/ | Available, no key required |
| SRTM Elevation | https://earthexplorer.usgs.gov/ | Requires account for bulk download |
| SoilGrids | https://soilgrids.org/ | REST API available |
| ESA WorldCover | https://esa-worldcover.org/ | GeoTIFF download available |

---

## 3. Data Preprocessing

### 3.1 Pipeline Steps

1. **Load** CSV or Parquet files via pandas.
2. **Column validation** — check required columns are present; raise an informative error if not.
3. **Coordinate cleaning** — remove rows with latitude outside [-90, 90] or longitude outside [-180, 180]; log count removed.
4. **Duplicate removal** — drop exact duplicates on all columns; near-duplicates (same coordinates, same date) are flagged.
5. **Missing value handling** — configurable per-column strategy (median imputation for numeric features); missing rainfall is **never silently replaced with zero**.
6. **Temporal feature extraction** — month, season, and monsoon flag derived from event timestamp where available.
7. **Feature engineering** — slope class bins, elevation band, rainfall intensity ratio, antecedent saturation index (approximated from 72h/24h ratio), slope×rainfall interaction.
8. **Encoding** — soil type and land cover are label-encoded when present.
9. **Train/validation/test split** — 60/20/20 chronological split; no shuffling before split to prevent temporal leakage.
10. **Feature scaling** — StandardScaler fitted on training data only; applied identically to validation and test sets.

### 3.2 Feature Schema

| Feature | Source | Notes |
|---------|--------|-------|
| `slope_degrees` | DEM-derived | Primary terrain susceptibility indicator |
| `elevation_m` | DEM | Correlated with slope; used as secondary indicator |
| `rainfall_1h_mm` | Rain gauge / satellite | Short-duration intensity |
| `rainfall_24h_mm` | Weather station / satellite | Standard meteorological window |
| `rainfall_72h_mm` | Weather station / satellite | Antecedent moisture proxy |
| `vegetation_index` | Satellite (NDVI) | Vegetation cover reduces surface runoff |
| `soil_moisture` | Model-derived | ERA5-Land or Open-Meteo soil layer |
| `slope_class` | Engineered | Bin of slope_degrees (0–4) |
| `elevation_band` | Engineered | Bin of elevation_m |
| `rainfall_intensity_ratio` | Engineered | rainfall_1h / max(rainfall_24h, 1) |
| `antecedent_saturation_index` | Engineered | rainfall_72h / max(rainfall_24h, 1) |
| `slope_rainfall_interaction` | Engineered | slope_degrees × rainfall_24h_mm |
| `event_month`, `event_season` | Temporal | Month and season of observation |
| `in_monsoon_season` | Temporal | Flag for June–September |
| `soil_type_encoded` | Categorical | Label-encoded soil type |
| `land_cover_encoded` | Categorical | Label-encoded land cover class |

---

## 4. Machine Learning Models

### 4.1 Candidate Models

| Model | Rationale |
|-------|-----------|
| Logistic Regression | Simple, interpretable baseline; tests for linear separability |
| Random Forest | Handles non-linearity and feature interactions; robust to outliers |
| Gradient Boosting | Often strong on tabular data; sequential error correction |
| XGBoost | Regularised boosting; efficient on medium-sized datasets |

### 4.2 Model Selection Process

1. All candidates trained on the same training split.
2. Evaluated on a held-out validation set (no hyperparameter tuning on test set).
3. Best model selected by PR-AUC (primary metric, accounts for class imbalance).
4. Final evaluation on a separate test set (not used for model selection).

### 4.3 Class Imbalance Handling

- `class_weight='balanced'` in sklearn models.
- `scale_pos_weight` in XGBoost.
- PR-AUC used as primary metric (robust to imbalance, unlike accuracy).

---

## 5. Experimental Design

### 5.1 Data Splitting Strategy

```
Full dataset → (60% train) (20% validation) (20% test)
               Sorted by timestamp (chronological)
               No data from future events in past splits
```

Rationale: Chronological splitting prevents temporal leakage where patterns from future events could influence model training.

### 5.2 Spatial Separation

Spatially separated validation (holding out specific geographic regions) is the ideal approach but was not feasible with the synthetic development dataset. In future work with a real dataset, spatial cross-validation using geographic folds should be implemented.

---

## 6. Validation Methodology

### 6.1 Metrics

| Metric | Rationale |
|--------|-----------|
| Precision | Fraction of predicted positives that are correct (false alarm rate) |
| Recall | Fraction of true positives detected (miss rate) |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Discriminative ability across all thresholds |
| PR-AUC | Primary metric; robust to class imbalance |
| Brier Score | Calibration quality of probability estimates |
| Confusion Matrix | Explicit counts of TP, TN, FP, FN |

### 6.2 Calibration

Probability estimates are evaluated with the Brier score. Reliability diagrams (calibration curves) are recommended for operational deployment.

---

## 7. Dynamic Risk Estimation Approach

### 7.1 Three-Tier Framework

```
STATIC SUSCEPTIBILITY
  → Terrain + land cover features only
  → Slow-changing; updated when terrain data changes

DYNAMIC RISK ESTIMATE
  → Susceptibility × current environmental conditions
  → Rainfall (1h, 24h, 72h) is the primary dynamic input
  → Updates with every new weather observation

EXPERIMENTAL ALERT
  → Dynamic risk ≥ configured threshold
  → Subject to cooldown period
  → EXPERIMENTAL ONLY — not operationally validated
```

### 7.2 Heuristic Baseline

When the trained model is unavailable, a weighted linear combination of normalized features is used:

```
score = Σ (w_i × normalized(x_i)) / Σ w_i
```

Weights: slope (0.30), rainfall_24h (0.25), rainfall_72h (0.20), elevation (0.10), soil_moisture (0.10), vegetation_index (0.05).

This baseline is explicitly marked as EXPERIMENTAL and not claimed to have predictive validity.

---

## 8. Alert Policy

### 8.1 Threshold Design

Risk thresholds are **provisional** and were chosen heuristically:

| Category | Probability Range |
|----------|------------------|
| Low | < 0.35 |
| Moderate | 0.35 – 0.55 |
| High | 0.55 – 0.75 |
| Very High | ≥ 0.75 |

These thresholds are **not calibrated against real-world alert outcomes** and must not be used for operational decisions.

### 8.2 Alert Cooldown

- Configurable cooldown period (default: 6 hours) prevents repeated alerts for the same location.
- All alerts are logged with timestamp, location, risk estimate, features used, and data source.
- Alerts require explicit acknowledgement by the user.

### 8.3 Experimental Alerts vs. Official Warnings

This system's alerts are **experimental research outputs** and must not be used as emergency warnings. Official disaster risk management decisions must be made by qualified geoscientists and authorised agencies.

---

## 9. Known Limitations

See `reports/limitations.md` for a full discussion.

Key points:
- All metrics are from synthetic data.
- No spatial validation has been performed.
- The model has not been tested on unseen geographies.
- Heuristic weights in the baseline scorer are not empirically derived.
- Alert thresholds are provisional.

---

## 10. Future Work

1. Acquire the NASA Global Landslide Catalog and re-train on authentic data.
2. Implement spatial cross-validation (leave-one-region-out).
3. Integrate SRTM or Copernicus DEM for real terrain features.
4. Connect Open-Meteo API for live rainfall data.
5. Calibrate thresholds using precision-recall curves on real data.
6. Implement SHAP waterfall plots for individual prediction explanations.
7. Add uncertainty quantification (conformal prediction or calibrated probabilities).
