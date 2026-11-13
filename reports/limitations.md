# LandslideGuard — Known Limitations

**Project:** LandslideGuard: AI-Based Dynamic Landslide Risk Prediction and Early Warning System  
**Type:** Academic Capstone Project — B.Tech Computer Science and Engineering  
**Date:** November 2026

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

This system partially implements both using Open-Meteo environmental feeds and interactive dynamic assessment, but lacks dense real-time field sensor telemetry (inclinometers, piezometers) and prospective event validation.

---

## 8. Limitations of the Early Warning Mechanism

The alert system in LandslideGuard is **experimental**:

- Thresholds (Low/Moderate/High/Very High) are heuristically defined (`0.35 / 0.55 / 0.75`), not calibrated against real outcomes.
- Cooldown periods (`120` minutes default in `config/settings.yaml`) are configurable defaults, not operationally tuned.
- External email and SMS channels (`src/alerts/notifications.py`) are strictly disabled by default and have no live emergency agency dispatch integration.
- No fail-safe hardware/telemetry redundancy for external API outages (degrades to cached snapshots or manual inputs).
- Although technical lifecycle transitions (`new` -> `acknowledged` -> `resolved`) and timestamps are logged in SQLite (`alert_acknowledgements`), there is no institutional governance or legal accountability framework for human decision-making.
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

## 12. Phase 6 Environmental Analytics, Forecasting & Drift Limitations

- **Baseline Archive Dependence:** Anomaly detection and multi-scale/seasonal trends depend on historical ERA5 reanalysis archive availability (`>= 30` valid days for Z-score/quantiles, `>= 60` for Isolation Forest, `>= 15` per season); offline or short baselines yield explicit `Unavailable` states rather than simulated baselines.
- **No Exact Timing Prediction:** 24h/48h/72h risk trajectories propagate forecasted weather inputs through models trained on **synthetic development data**. They indicate directional meteorological sensitivity only and **cannot predict exact landslide failure timing**.
- **Forecast Sensitivity vs. Calibration:** Lead-time uncertainty bands reflect rainfall input sensitivity (`±25%` scaled by horizon) rather than empirically calibrated conformal intervals.
- **Ground-Truth Outcome Latency:** Drift monitoring compares live records against synthetic training baselines and cannot confirm real-world predictive degradation until sufficient verified ground-truth outcomes (`prediction_outcomes`, minimum `10` samples across outcomes) are logged.

---

## 13. Phase 7 Geospatial, Terrain & Satellite Limitations

- **Historical Hotspots are Non-Predictive:** DBSCAN clusters and Kernel Density Estimation delineate spatial concentrations of *past recorded events*. They do **NOT** predict future landslide occurrence or classify an area as a future hotspot without dynamic meteorological triggering, geotechnical slope conditions, and pore-water pressure accumulation.
- **Topographic Wetness Index Requires Specific Catchment Area:** TWI ($ln(a / \tan(\beta))$) strictly requires upslope contributing area per unit contour width ($a$). Point SRTM stencils and isolated tiles lack full hydrological basin routing; TWI calculation is refused rather than fabricated with arbitrary catchment areas.
- **Sentinel-2 Spatial Resolution (10 m):** The 10 m Ground Sample Distance of Sentinel-2 VNIR bands cannot resolve small-scale, shallow translational slides or debris slumps smaller than $100\text{ m}^2$.
- **Cloud & Shadow Contamination:** Optical remote sensing in mountainous monsoonal regions suffers heavy cloud cover and steep terrain cast shadows. Cloud-obscured pixels are masked out, creating observation gaps during peak rainfall periods.
- **Non-Landslide Vegetation Change Confounding:** Drops in NDVI can be caused by seasonal phenology, agricultural harvest, wildfire, forestry timber clearing, or drought. Automated change detection is an **unverified screening indicator** that flags candidate zones requiring high-resolution optical, drone, or ground inspection.

---

## 14. Phase 8 AI Assistant & Computer Vision Limitations

- **AI Assistant Information Bounds:** The AI Assistant only possesses knowledge grounded directly in the project's SQLite database tables, loaded ML model metadata, Open-Meteo telemetry, and project documentation. It cannot access real-time unmonitored local news, unindexed micro-catchments, or unrecorded historical archives. If a location is not in the database, it refuses estimation rather than improvising answers.
- **Superficial Optical Sensing Only (No Subsurface Pore Pressure):** The experimental computer vision module analyzes optical surface reflectance (color and high-frequency edge gradients). It has **zero ability to measure subterranean pore water pressure**, soil water saturation, groundwater table depth, or internal slip-surface shear strength—the primary physical drivers of slope failure.
- **Illumination, Perspective & Shadow Confounding in CV:** Optical photographs of natural terrain suffer from cast shadows, tree branch shadows, lighting variations, and steep perspective distortions. High-contrast shadows may be falsely identified as surface cracks, while genuine fissures in heavily shaded hollows may be missed.
- **Computer Vision is NOT an Operational Early Warning:** An image classification result or elevated visual anomaly score is **NOT proof of an active landslide** and must **NEVER independently trigger an official warning**, evacuation notice, or emergency alert.
- **Multilingual Guidance is Educational, Not Operational Dispatch:** The multilingual safety guide provides verified civil-defense advisories from NDMA and GSI for public awareness. It does not replace real-time localized warnings dispatched by district magistrates, NDRF, SDRF, or state disaster management centers.

---

## 15. Phase 9 Disaster Response & Community Intelligence Limitations

- **Static Road Network vs. Active Disaster Blind Spots:** The OSRM routing engine queries static OpenStreetMap highway topologies. It has **zero awareness of real-time mudflow inundation, severed bridges, fallen trees, rockfall blockages, or emergency police cordons**. Driving routes are strictly informational and must never be navigated during an active disaster without explicit clearance from local traffic police and disaster management authorities (NDRF/SDMA).
- **Crowd-Sourced Information Noise & Subjectivity:** Community hazard submissions depend on public reporting. Reports may contain inaccurate coordinates, misidentified minor soil washouts, exaggerated hazard tiers, or duplicate entries. Strict administrative review (`submitted` $\to$ `under_review` $\to$ `verified` / `rejected`) is mandatory before field dispatch.
- **Strict Data Segregation Invariant:** Community reports are **strictly segregated** in the `community_reports` database table and never automatically enter scientific training sets (`observations`). This prevents subjective or unverified crowd reports from biasing model training or corrupting empirical ground-truth catalogs.
- **Retrospective Event Replay Limitations:** Re-evaluating historical events re-runs the trained pipeline using historical records available for that epoch. It cannot simulate prospective forecasting conditions, real-time telemetry lag, or unobserved micro-catchment precipitation spikes, and must not be cited as operational forecasting proof.
- **PDF Report Snapshot Shelf-Life:** Downloaded PDF risk reports represent static point-in-time snapshots. Mountain weather, antecedent rainfall, and slope saturation evolve rapidly; generated PDFs lose validity within hours and must not be used for protracted emergency planning.
- **Offline Cache Stale Telemetry Risk:** While the offline cache maintains interface availability during mountain grid or telecommunication failures, it presents historical telemetry explicitly stamped with `is_live: False`. In disconnected environments, personnel must rely on primary civil defense radio networks (VHF/wireless) rather than aging cached dashboard data.

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
| Historical hotspot over-interpretation | High | Mandatory anti-hallucination notice; non-predictive warning |
| Satellite change false positives | Moderate | Mandatory unverified indicator label; NDVI disclaimer |
| DEM hydrological data absence | Moderate | Strict TWI refusal without catchment area |
| CV optical surface limit (no subsurface sensing) | **Critical** | Mandatory experimental disclaimer; never triggers alerts |
| CV shadow and illumination false positives | High | Multi-feature heuristics; clearly labeled experimental |
| AI Assistant boundary limitation | Moderate | Strict anti-hallucination refusal for unindexed places |
| Road routing active blockage blind spots | **Critical** | Mandatory routing disclaimer; requires NDRF/SDMA clearance |
| Crowd-sourced report subjectivity | Moderate | Strict multi-stage audit lifecycle; segregation from training data |
| Retrospective replay hindsight bias | Moderate | Explicit non-prospective notice; interval data refusal |
| Static PDF report obsolescence | Low | Timestamped snapshot badges; academic notice watermarked |
