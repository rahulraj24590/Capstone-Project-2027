# LandslideGuard — Known Limitations

**Project:** LandslideGuard: AI-Based Dynamic Landslide Risk Prediction and Early Warning System  
**Type:** Academic Capstone Project — B.Tech Computer Science and Engineering  
**Date:** October 2026

> [!CAUTION]
> This document is required reading before deploying or interpreting any output from this system.
> The limitations described here are fundamental constraints, not implementation bugs.

---

## 1. Dataset Limitations

### 1.1 Synthetic Development Data

All model training and evaluation in this release uses a **synthetic dataset** generated solely to test the application pipeline. This dataset:

- Was constructed using simplified statistical distributions, not real sensor observations.
- Does not represent any specific geographic region.
- Artificially correlates slope and rainfall with landslide occurrence in ways that real terrain may not.
- Produces **inflated model performance metrics** that do not generalize to real-world data.

**Consequence:** Reported metrics (Precision ~0.98, ROC-AUC ~0.999) are not meaningful measures of real-world predictive ability and must not be cited as such.

### 1.2 Positive and Negative Label Bias

Landslide inventories record **known events**. Absence of a record does not mean a location is safe:
- Remote areas may have no reporting infrastructure.
- Historical records may be incomplete for pre-digital periods.
- Small or shallow failures may not be catalogued.

This creates **sampling bias** in any model trained on such data.

### 1.3 Spatial Coverage

The synthetic dataset covers a notional region of the Indian subcontinent (roughly 8°N–35°N, 68°E–97°E). A model trained on this data cannot be expected to generalize to:
- Different geological settings (Himalayas vs. Western Ghats vs. Northeast India).
- Regions with different soil types, rainfall regimes, or land cover.

---

## 2. Class Imbalance

Landslides are rare events. Even in the synthetic dataset, positive cases represent approximately 25% of observations — a ratio far more balanced than the real world. In practice:

- Real datasets may have positive rates below 1%.
- Class-weighted training and PR-AUC metrics partially address this.
- Very high recall may require accepting many false positives.
- The optimal precision-recall trade-off depends on the consequences of false alarms vs. missed warnings in a specific operational context.

---

## 3. Spatial and Temporal Generalization

### 3.1 No Spatial Cross-Validation

Model evaluation used a random (not spatial) train/test split. In practice, nearby locations often share geological and climatic characteristics. Spatially correlated training and test samples can lead to:
- Optimistic accuracy estimates.
- Models that fail on unseen geographic regions even when performing well on the test set.

Proper validation requires **leave-one-region-out cross-validation** or similar spatial separation strategies.

### 3.2 Temporal Generalization

Climate patterns change over time. A model trained on historical data may not capture:
- Changing rainfall patterns due to climate change.
- Land use change (deforestation, construction, terracing).
- Anthropogenic triggers (road cutting, irrigation, reservoir filling).

---

## 4. False Negatives (Missed Warnings)

A false negative occurs when the model predicts Low or Moderate risk but a landslide occurs. The consequences of a false negative in a real early warning context are severe:
- Loss of life.
- Failure to evacuate in time.
- Erosion of trust in the system.

**This system has not been evaluated for false negative rate in a real-world context.** Any operational use would require empirical verification of recall at operationally acceptable thresholds.

---

## 5. False Positives (False Alarms)

A false positive occurs when the model predicts High or Very High risk but no landslide occurs. Frequent false alarms:
- Cause alert fatigue.
- Erode community trust.
- Result in unnecessary evacuations.

The optimal false positive rate depends on local context, community capacity to respond, and economic consequences of evacuation. This system does not provide tools for operationally calibrating this trade-off.

---

## 6. Environmental Data Uncertainty

### 6.1 Rainfall Measurement

- Rain gauge networks are sparse in many high-risk areas.
- Satellite-derived precipitation (GPM, IMERG) has spatial resolution of ~10 km, which may not capture highly localized convective events.
- Model-derived soil moisture (ERA5-Land, Open-Meteo) is a simulation output, not a direct measurement.

### 6.2 Terrain Data

- SRTM DEM has 30m resolution. Local slope in steep terrain can vary significantly at scales smaller than one pixel.
- Elevation data is a static snapshot. Post-event terrain changes (mass movement, erosion) are not reflected.

### 6.3 Feature Availability

Not all features are available for every location at every time step. This system:
- Does not silently substitute zero for missing rainfall (correct behaviour).
- May use partial feature sets in the heuristic baseline (with explicit warnings).
- May produce unreliable estimates when critical features are absent.

---

## 7. Differences Between Susceptibility and Dynamic Forecasting

| Aspect | Susceptibility Mapping | Dynamic Risk Forecasting |
|--------|-----------------------|--------------------------|
| Time horizon | Long-term | Short-term (hours to days) |
| Input features | Terrain, geology, land cover | + Rainfall, soil moisture |
| Data requirements | Static layers | Real-time environmental feeds |
| Validation | Retrospective inventory | Prospective event prediction |
| Output interpretation | Relative terrain tendency | Time-varying risk level |

This system partially implements both but does not have real-time data feeds or prospective validation.

---

## 8. Limitations of the Early Warning Mechanism

The alert system in LandslideGuard is **experimental**:

- Thresholds (Low/Moderate/High/Very High) are heuristically defined, not calibrated against real outcomes.
- Cooldown periods are arbitrary defaults.
- No communication pipeline to first responders or emergency management agencies.
- No fail-safe mechanisms for data feed interruption.
- No accountability or audit trail for alert-based decisions.
- Not compliant with any formal disaster risk management standard.

**Alerts from this system must not be used as emergency warnings.**

---

## 9. Why This Project Cannot Replace Official Authorities

Official geological and disaster management authorities (e.g., GSI, NDMA, IMD, State DMA) have:
- Access to dense observation networks (rain gauges, seismic sensors, GPS).
- Trained geoscientists who can inspect terrain in the field.
- Legal mandate and accountability for issuing public warnings.
- Validated historical performance of their warning systems.
- Standard operating procedures for false alarm management.
- Communication channels to civil authorities and first responders.

This project is an educational demonstration of how machine learning and geospatial data can be combined for risk assessment. It contributes to the research landscape but is not a substitute for institutional expertise.

---

## 10. Model Explainability Limitations

- Feature importance scores indicate which features **correlate** with the model's output, not which features **cause** landslides.
- High SHAP importance for rainfall does not mean rainfall caused a specific landslide.
- Permutation importance can be misleading when features are correlated (as rainfall and soil moisture often are).

---

## 11. Reproducibility

- All model artifacts are saved with metadata including Python version, library versions, and dataset fingerprint.
- Random seeds are fixed for reproducibility.
- However, models trained on different hardware may produce slightly different floating-point results.

---

## Summary Table

| Limitation | Severity | Mitigation in This Project |
|-----------|----------|---------------------------|
| Synthetic training data | **Critical** | All metrics labelled as synthetic |
| No spatial cross-validation | High | Documented; recommended for future work |
| Class imbalance | High | PR-AUC metric, class weighting |
| No real-time data feeds | Moderate | Open-Meteo integration; manual input option |
| Uncalibrated thresholds | Moderate | Explicitly labelled as provisional |
| No operational alert pipeline | Moderate | Dashboard alerts only; not emergency warnings |
| Missing data in some regions | Moderate | Graceful degradation with warnings |
| False negative risk | High | Documented; no operational claim made |
