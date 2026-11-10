# LandslideGuard — Viva Preparation Guide

**Project:** LandslideGuard: AI-Based Dynamic Landslide Risk Prediction and Early Warning System  
**Prepared for:** B.Tech CSE Capstone Project Viva  
**Date:** October 2026

---

## Section 1: Project Architecture Overview

```
LandslideGuard
│
├── app.py                   ← Streamlit dashboard (6 pages)
├── src/
│   ├── data_processing/     ← Preprocessing & feature engineering
│   ├── models/              ← Training, evaluation, prediction, registry
│   ├── risk/                ← Risk engine, thresholds, input validation
│   ├── data_providers/      ← Weather & terrain API integrations
│   ├── geospatial/          ← Maps and spatial analysis
│   ├── database/            ← SQLite schema and queries
│   ├── alerts/              ← Alert manager (cooldowns, acknowledgements)
│   └── utils/               ← Logging, settings, validation helpers
├── scripts/                 ← Data download, generation, preparation
├── tests/                   ← pytest test suite (112 tests, all passing)
├── reports/                 ← Methodology, limitations, evaluation
└── models/                  ← Saved model artifacts
```

**Data flow:**
```
Raw Data → Preprocessing → Feature Engineering → Model Training
                                                      ↓
User Input → Input Validation → Risk Engine → Alert Manager → Dashboard
                                ↗
                         Weather API (Open-Meteo)
```

---

## Section 2: Algorithm Selection — Why Each Was Chosen

### 2.1 Logistic Regression (Baseline)

**Why chosen:**
- Simplest interpretable baseline; establishes a lower bound on performance.
- Coefficients directly interpretable as log-odds.
- Efficient to train; fast inference.

**Limitation for this problem:**
- Assumes linear separability; landslide risk is highly non-linear (threshold effects in rainfall).

### 2.2 Random Forest

**Why chosen:**
- Handles non-linear feature interactions naturally (e.g., slope AND rainfall together).
- Resistant to overfitting through averaging of many trees.
- Provides tree-based feature importance natively.
- Works well on tabular environmental data.

**Limitation:**
- Does not extrapolate well outside training distribution.
- Prediction time scales with number of trees.

### 2.3 Gradient Boosting (sklearn)

**Why chosen:**
- Sequential error correction often produces better calibrated probabilities than Random Forest.
- Explicit regularisation (max_depth, n_estimators, learning_rate) reduces overfitting.

**Limitation:**
- Slower to train than Random Forest; sensitive to hyperparameters.

### 2.4 XGBoost

**Why chosen:**
- Regularised gradient boosting with efficient parallelisation.
- Handles missing values natively (which is critical when environmental features may be absent).
- State-of-the-art performance on many tabular benchmark datasets.

**Limitation:**
- More hyperparameters than simpler models; risk of overfitting on small datasets.

### 2.5 Why Not Deep Learning?

- Dataset size (~2000 records in development) is too small for reliable neural network training.
- Tree-based models are generally preferred over deep learning for structured tabular data.
- Interpretability requirements for a safety-critical application favour simpler models.

---

## Section 3: Dataset Limitations

1. **Synthetic data only** — All model metrics are from a synthetically generated dataset, not real landslide observations. Metrics would change significantly on authentic data.

2. **Positive-only catalog issue** — Real-world landslide catalogs contain only known events. The choice of "negative" examples (non-landslide locations) heavily influences model performance and is an open research problem.

3. **Spatial bias** — NASA Global Landslide Catalog over-represents areas with English-language media coverage and established remote sensing programs.

4. **Feature resolution mismatch** — Rainfall may be available at 25 km resolution while slope is available at 30 m. Spatial aggregation introduces uncertainty.

---

## Section 4: Evaluation Metrics — What They Mean and Why

| Metric | Formula | Why Important for Landslides |
|--------|---------|------------------------------|
| **Precision** | TP/(TP+FP) | Low precision = many false alarms; erodes trust |
| **Recall** | TP/(TP+FN) | Low recall = missed warnings; safety-critical |
| **F1-Score** | Harmonic(P,R) | Balance between precision and recall |
| **ROC-AUC** | Area under TPR vs FPR | Discrimination ability; threshold-independent |
| **PR-AUC** | Area under P vs R | Robust to class imbalance; preferred primary metric |
| **Brier Score** | Mean(p - y)² | Calibration; are probabilities meaningful? |

**Key insight:** Accuracy is a misleading metric when positive rate is low (e.g., 99% accuracy by predicting "no landslide" always). PR-AUC avoids this trap.

**Reported metrics (SYNTHETIC DATA ONLY):**
- Logistic Regression test PR-AUC: ~0.999
- These numbers reflect the simplicity of the synthetic dataset, not real-world performance.

---

## Section 5: Landslide Susceptibility vs. Dynamic Prediction

This is a conceptually important distinction that examiners frequently ask about:

| Aspect | Susceptibility | Dynamic Prediction |
|--------|---------------|-------------------|
| **Question answered** | "Is this terrain prone to landslides?" | "Is a landslide likely in the next few hours?" |
| **Time scale** | Years to decades | Hours to days |
| **Key inputs** | Slope, geology, land cover | + Rainfall intensity, antecedent moisture |
| **Validation** | Historical inventory mapping | Prospective event prediction |
| **Use case** | Land use planning, zoning | Emergency response, evacuation |
| **In this project** | Static model output | Dynamic risk engine |

This project implements both concepts but clearly separates them in the dashboard and documentation.

---

## Section 6: How the Alert Mechanism Works

### Step-by-Step

1. User inputs location coordinates and environmental data (or weather API fetches automatically).
2. `RiskEngine.estimate_risk()` calls the trained ML model (or heuristic baseline if unavailable).
3. Risk probability is classified into Low / Moderate / High / Very High using configured thresholds.
4. If risk_category ≥ configured alert threshold (default: High):
   - Check if location is on cooldown (default: 6 hours since last alert).
   - If NOT on cooldown: create alert record in SQLite database; display in dashboard.
   - If on cooldown: log the trigger but suppress the alert.
5. Alert displayed with: location, time, probability, risk category, data source, features used.
6. User can acknowledge the alert, which logs the acknowledgement with a timestamp.

### Important Caveats
- All alerts are explicitly labelled EXPERIMENTAL.
- No automatic communication to emergency services.
- Thresholds are provisional and not calibrated against real outcomes.

---

## Section 7: Model Limitations

1. **Distribution shift:** A model trained on data from one region may fail in another even with identical feature distributions, if local geology differs.

2. **Feature correlation:** Slope and rainfall are often correlated in certain terrain types, making individual feature importance scores less reliable.

3. **Missing data at inference:** When a feature is missing at prediction time (e.g., no soil moisture reading), the model sees a different feature vector than it was trained on. This is handled by falling back to the heuristic baseline with an explicit warning.

4. **Temporal validity:** Environmental conditions change rapidly. A prediction made 6 hours ago may not reflect current conditions. The system enforces a data freshness check.

5. **Threshold sensitivity:** Small changes in probability threshold dramatically change precision-recall trade-off. Operational threshold selection requires consultation with domain experts.

---

## Section 8: Future Improvements

### Short-term (3–6 months)
1. Acquire NASA Global Landslide Catalog data and re-train models.
2. Connect SRTM DEM API for real terrain extraction.
3. Implement spatial cross-validation (leave-one-district-out for India).
4. Add SHAP waterfall plots for individual predictions in the dashboard.
5. Calibrate probability thresholds using real precision-recall curves.

### Medium-term (6–12 months)
1. Integrate India Meteorological Department (IMD) gridded rainfall data.
2. Implement convolutional neural network on DEM patches for automated slope characterisation.
3. Add uncertainty quantification using conformal prediction.
4. Build mobile-friendly progressive web app interface.
5. Partner with geological survey for ground-truth validation.

### Long-term (Research Directions)
1. Multi-hazard modelling (combine landslide + flood risk).
2. Climate projection integration (CMIP6 scenarios).
3. Federated learning for model training across institutions without data sharing.
4. Explainability methods for spatiotemporal models (Graph Neural Networks).

---

## Section 9: Likely Examiner Questions and Suggested Answers

**Q: Why did you use machine learning instead of a physically-based model?**  
A: Physically-based models (e.g., infinite slope stability analysis) require detailed soil properties (cohesion, friction angle, unit weight) that are not available at regional scales. Data-driven models can be applied wherever historical inventory and proxy environmental data exist. Hybrid approaches combining physics-based constraints with ML are an active research direction.

**Q: How would you validate this in a real-world deployment?**  
A: Prospective validation — deploy the model, let it make predictions, then compare against verified events over a 12–24 month period. This is called "operational validation" and is distinct from retrospective skill on historical data. Key metrics: detection rate, false alarm ratio, and lead time.

**Q: What is the class imbalance problem and how did you handle it?**  
A: Landslides are rare relative to stable terrain conditions. If a model predicts "no landslide" 99% of the time, it gets 99% accuracy on a 1% positive-rate dataset, which is useless. We addressed this with: (1) class_weight='balanced' in model training, (2) PR-AUC as the primary metric instead of accuracy, (3) reporting precision, recall, and F1 separately for each class.

**Q: What is the difference between the heuristic baseline and the ML model?**  
A: The heuristic baseline is a weighted linear combination of normalized features with hand-crafted weights. It requires no training data and always produces a score, even with partial inputs. The ML model learns weights from data, captures non-linear interactions, and has measured performance. The baseline is used as a fallback when the model is unavailable and is always labelled EXPERIMENTAL.

**Q: Why not use a deep learning model?**  
A: Our dataset (~2000 records) is too small for reliable neural network training. Tree-based models (Random Forest, XGBoost) consistently outperform deep learning on small, structured tabular datasets. If we had millions of records (e.g., pixel-level susceptibility mapping across a continent), CNNs on DEM patches would become relevant.

**Q: How do you prevent data leakage?**  
A: Three main measures: (1) Chronological train/test split — no future data in training. (2) StandardScaler fitted only on training data, applied to validation/test. (3) No hyperparameter tuning on the test set — only validation data used for model selection.

**Q: What is SHAP and how does it explain predictions?**  
A: SHAP (SHapley Additive exPlanations) computes the marginal contribution of each feature to an individual prediction, based on game theory (Shapley values from cooperative game theory). A positive SHAP value for "rainfall_24h_mm" means that feature pushed this specific prediction toward "landslide", compared to the average prediction. It satisfies desirable properties like consistency and accuracy that simpler methods (e.g., feature permutation) do not always guarantee.

---

## Section 10: Quick Reference — Key Facts for Viva

| Item | Value |
|------|-------|
| Algorithm (final) | Logistic Regression (best on synthetic data; see caveats) |
| Primary metric | PR-AUC |
| Training data | 1200 synthetic records |
| Validation data | 400 synthetic records |
| Test data | 400 synthetic records |
| Feature count | 17 |
| Risk categories | Low / Moderate / High / Very High |
| Alert threshold | High (≥ 0.55 probability) |
| Cooldown period | 6 hours (configurable) |
| Database | SQLite |
| Frontend | Streamlit |
| Weather API | Open-Meteo (no key required) |
| Test coverage | 112 tests, all passing |
| Deployment | Local (streamlit run app.py) |
