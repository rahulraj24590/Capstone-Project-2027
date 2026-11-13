# LandslideGuard — Methodology Report

**Project:** LandslideGuard: AI-Based Dynamic Landslide Risk Prediction and Early Warning System  
**Type:** Academic Capstone Project — B.Tech Computer Science and Engineering  
**Date:** November 2026  
**Status:** Integrated Prototype (Phases 1–5 Complete) — Experimental, not operationally validated

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
Full dataset (2,000 rows; sorted chronologically where event_date exists)
  ├── Training set:   1,360 samples (68% of total = 85% of non-test split)
  ├── Validation set:   240 samples (12% of total = `val_size: 0.15` of non-test split; used for model selection)
  └── Test set:         400 samples (20% of total = `test_size: 0.20`; held-out final evaluation)
```

*Note:* The active saved pipeline (`models/best_model.pkl`) consumes the 7 core environmental/terrain features listed in `models/feature_columns.json` (`slope_degrees`, `elevation_m`, `rainfall_24h_mm`, `rainfall_72h_mm`, `rainfall_1h_mm`, `vegetation_index`, `soil_moisture`), while `src/data_processing/feature_engineering.py` computes derived indicators (slope classes, saturation indices, interactions) for exploratory and temporal analysis.

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

### 8.2 Alert Cooldown, Escalation & Lifecycle

- Configurable cooldown period (`alerts.cooldown_minutes`, default `120` minutes in `config/settings.yaml`) prevents repeated duplicate notifications for the same location and severity or lower.
- Significant severity escalations (e.g., `High` active within cooldown escalating to `Very High`) bypass cooldown suppression and log `escalated_from`.
- Full lifecycle persisted in SQLite (`new` -> `acknowledged` -> `resolved`) with supporting environmental info, coordinates, model version, data source, and immutable audit log entries in `alert_acknowledgements`.
- External email and SMS notification hooks (`src/alerts/notifications.py`) are strictly disabled by default unless explicitly enabled and configured with environment credentials.

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

## 10. Explainability & Future Scientific Validation

1. **Explainable AI (`src/models/explainability.py`):** Integrated SHAP (`LinearExplainer` for Logistic Regression, `TreeExplainer` for Random Forest / Gradient Boosting / XGBoost), sklearn `permutation_importance`, and native tree importances with strict alignment against `models/feature_columns.json`, bounded caching (`128` entries), and explicit causal humility disclaimers.
2. **Future Operational Scientific Validation:**
   - Acquire the NASA Global Landslide Catalog (with NASA Earthdata credentials) and re-train on authentic inventory records paired with non-landslide terrain sampling.
   - Implement spatial cross-validation (leave-one-region-out across Himalayan, Western Ghats, and Northeast Indian zones).
   - Mount local 30 m SRTM or Copernicus DEM GeoTIFF tiles (`data/raw/`) into the Rasterio terrain extractor (`src/geospatial/terrain.py`).
   - Calibrate thresholds using empirical precision-recall curves and uncertainty quantification (conformal prediction) before any non-experimental field trial.

---

## 11. India State/District Regional Monitoring Architecture

To support authentic India-wide regional exploration (`State/UT -> District -> Location`) without spatial over-claiming:
1. **Authoritative Polygon Resolution (`src/data_providers/india_boundaries.py`):** Joins geoBoundaries IND ADM2 districts cleanly against ADM1 States/UTs via polygon containment (`gpd.sjoin(..., predicate="within")`) plus metric nearest-polygon fallback (`sjoin_nearest`) for island districts such as Lakshadweep, yielding `36` States/UTs and `735` Districts with interior `representative_point()` coordinates.
2. **Coordinate-Level vs. Administrative Scope Separation:** Open-Meteo reanalysis/forecasts and SRTM terrain queries evaluate representative point coordinates inside each district. Folium maps (`build_india_regional_map`) render State & District polygons neutrally and explicitly forbid coloring an entire district polygon high risk from a single point score.
3. **Official vs. Experimental Separation:** Official government feeds (`NDMA SACHET` CAP warnings and `IMD` operational feeds when `IMD_API_KEY` is configured) persist in `official_warnings`, strictly isolated from LandslideGuard's experimental ML `alerts` table.
4. **Zero-Imputation & Freshness Guarantees:** Missing rainfall windows or terrain metrics remain `None` rather than silent zero-fill, and cached responses (`ETag 304` or file cache) are explicitly labeled cached rather than live.

---

## 12. Phase 6 — Intelligent Environmental Analysis, Forecasting & Drift Monitoring

1. **Statistical Rainfall Anomaly Detection (`src/monitoring/environmental_analytics.py`):**
   - Evaluates recent 24-hour rainfall solely against authentic historical daily baselines (Open-Meteo ERA5 archive, minimum `30` valid days required; `>= 60` valid days with variance > 0 for Isolation Forest). Never fabricates baselines or classifies anomalies when insufficient historical archive days exist.
   - Combines empirical Z-scores, percentile rankings (`P90 / P95 / P99`), rolling 7-day statistical mean + standard-deviation envelopes, and unsupervised `IsolationForest` outlier flags into four transparent categories: `Normal rainfall`, `Above-normal rainfall`, `Unusual rainfall`, and `Extreme rainfall` (also noting documented IMD Very Heavy `115.6 mm/24h` reference alongside empirical quantiles).
2. **Multi-Scale Rainfall Trend Analysis:**
   - Computes daily OLS slopes and 7-day rolling averages, ISO weekly aggregations (requiring `>= 5` valid days per week), monthly aggregations (`>= 20` valid days per month), historical percentile benchmarks, and IMD seasonal comparisons (`Winter JF`, `Pre-Monsoon MAM`, `Monsoon JJAS`, `Post-Monsoon OND`; requiring `>= 15` valid days per season).
3. **Experimental 24h / 48h / 72h Risk Trend Forecasting:**
   - Feeds actual forecast hourly precipitation and model-derived surface soil moisture (combined strictly without zero-imputation with trailing 72h historical reanalysis only where overlapping antecedent windows require) through `RiskEngine`.
   - Labels all forecast-derived features explicitly, evaluates symmetric rainfall sensitivity uncertainty bands widened by forecast lead time (`24h -> 48h -> 72h`), marks incomplete forecast horizons strictly `Unavailable`, and states explicitly that exact landslide timing cannot be predicted.
4. **Evidence-Gated Model Drift Monitoring (`src/monitoring/drift_monitor.py`):**
   - Compares production `prediction_records` and `environmental_observations` against reference training distributions (`data/processed/dataset.parquet` / `models/training_stats.json` / `reports/metrics.json`) using Population Stability Index (PSI), two-sample Kolmogorov-Smirnov (`ks_2samp`) tests, missing/imputed-rate deltas, and live ground-truth performance (`prediction_outcomes` table).
   - Enforces minimum sample counts (`>= 20` feature/prediction records, `>= 10` verified ground-truth outcomes) before confirming drift.

---

## 13. Phase 7 — Advanced Geospatial, Terrain & Satellite Intelligence

1. **DEM Terrain Derivatives & Hydrological Flow Routing (`src/geospatial/terrain.py`):**
   - **Slope & Aspect:** Evaluates central finite-difference spatial gradients ($dz/dx, dz/dy$) converted to slope angle $\beta$ (degrees) and compass aspect azimuth (degrees 0–360° from True North).
   - **Terrain Ruggedness Index (TRI):** Implements Riley et al. (1999) 8-neighbor RMS elevation difference to quantify micro-relief ruggedness without edge fabrication.
   - **Laplacian Curvature:** Computes second derivatives $\nabla^2 z$ as a local convexity/concavity proxy.
   - **Hydrological Input Guard & Topographic Wetness Index (TWI):** Implements $TWI = \ln(a / \tan(\beta))$. Strictly enforces input requirements: if specific catchment area $a$ is unavailable, TWI evaluation is refused with a clear explanatory status. When full DEM matrices are available, authentic D8 steepest-descent flow direction routing sorts elevation descending to compute cumulative upslope contributing area $a = (\text{flow\_cells} \cdot dx \cdot dy) / \text{contour\_width}$.
   - **3D Digital Elevation Models:** Constructs interactive 3D terrain surface meshes with Plotly Surface, supporting elevation, slope angle, and roughness draping with vertical exaggeration controls.

2. **Landslide Hotspot Detection & Spatial Clustering (`src/geospatial/hotspot_detection.py`):**
   - **DBSCAN Clustering:** Evaluates spatial clustering using spherical haversine metric on radian coordinates ($\text{eps} = \text{radius\_km} / 6371.0$) with configurable minimum points (`min_samples >= 3`). Outlier events are identified as spatial noise (-1). For each cluster, computes centroid coordinates, bounding box extents, Shapely convex hull polygons, geodesic area ($\text{km}^2$), and event density ($\text{events/km}^2$).
   - **Kernel Density Estimation (KDE):** Evaluates continuous 2D spatial event density surfaces using `scipy.stats.gaussian_kde` over a regular geographic grid covering the observation extents. Computes empirical density percentiles (P75, P90, P95, P99) to identify high-density hotspot zones.
   - **Anti-Hallucination Guard:** Enforces `HOTSPOT_SCIENTIFIC_DISCLAIMER` across all cluster outputs: historical clustering identifies recorded past event density only and does NOT predict future landslide occurrence or hotspot activation without dynamic environmental triggering.
   - **Authentic Data Filtering:** Supports `authentic_only=True` to exclude synthetic development records from real-world hotspot analysis.

3. **Sentinel-2 Satellite Vegetation Monitoring (`src/geospatial/satellite_monitor.py`):**
   - **Spectral Band Verification:** Implements Normalized Difference Vegetation Index: $NDVI = (NIR - Red) / (NIR + Red)$. Strictly requires both Red (Sentinel-2 Band 4 ~665 nm) and NIR (Sentinel-2 Band 8 ~842 nm) spectral bands; refuses computation from RGB-only or single-band imagery.
   - **Cloud Masking:** Leverages Sentinel-2 Scene Classification Layer (SCL) or QA bitmasks to exclude cloud and cloud-shadow pixels from NDVI evaluation and summary statistics.
   - **Vegetation Class Distribution:** Quantifies pixel fractions into Water/Snow ($<0$), Bare Soil/Rock ($[0, 0.2)$), Sparse Vegetation ($[0.2, 0.4)$), Moderate Vegetation ($[0.4, 0.6)$), and Dense Canopy ($\ge 0.6$).
   - **Vegetation Scientific Disclaimer:** Enforces `VEGETATION_DISCLAIMER` stating that NDVI fluctuations alone cannot be interpreted as proof of landslide activity without geotechnical or field corroboration.

4. **Multi-Temporal Satellite Change Detection:**
   - **Dual-Epoch Spectral Difference:** Compares pre-event ($t_1$) and post-event ($t_2$) acquisitions to compute $\Delta NDVI = NDVI_{post} - NDVI_{pre}$.
   - **Dual-Cloud Masking:** Masks pixels obscured by clouds or cloud shadows in either epoch $t_1$ or $t_2$, reporting total unobservable cloud area.
   - **Disturbance Area Quantification:** Classifies change into severe loss ($\Delta NDVI \le -0.25$), moderate loss ($-0.25 < \Delta NDVI \le -0.10$), stable, and regrowth, calculating affected surface area in $\text{km}^2$ from the 10 m Ground Sample Distance.
   - **Verification Indicator Label:** Automatically affixes `CHANGE_DETECTION_VERIFICATION_LABEL` warning that spectral change detection is an unverified screening indicator requiring independent optical or ground corroboration.

---

## 14. Phase 8 — LandslideGuard AI Assistant & Computer Vision

1. **AI Assistant Architecture & Retrieval-Augmented Engine (`src/assistant/knowledge_engine.py`):**
   - **Intent Parsing & Domain Routing:** Employs rule-based semantic and keyword intent classification across 8 project domains: `MONITORING_LOCATIONS`, `DISTRICT_LOOKUP`, `RAINFALL_WEATHER`, `RISK_EXPLANATION`, `ACTIVE_ALERTS`, `HISTORICAL_LANDSLIDES`, `DATA_SOURCES`, `METHODOLOGY_MODEL`, and `SAFETY_GUIDANCE`.
   - **Strict Data Retrieval:** Answers are synthesized directly from active database records:
     - Monitored locations and live telemetry from `LiveRiskMonitor` and `locations` / `environmental_observations` tables.
     - Active alerts, cooldown status, and escalation logs from `AlertManager` and `alerts` / `alert_acknowledgements` tables.
     - Historical landslide records filtered from the `observations` table (`NASA GLC` / national inventories).
     - India administrative boundaries and centroids from the `india_districts` table (735 indexed districts).
     - Model explanations from `explain_prediction` and `explain_global` SHAP attributions.
   - **Anti-Hallucination & Scope Enforcement:** If a queried location or district is absent from the database, the assistant explicitly reports that no records exist rather than inventing values. Questions outside the landslide risk and environmental domain (e.g., finance, pop culture, sports) are politely refused.
   - **Provenance Citations:** Every response explicitly cites its underlying data sources (e.g., `[Source: SQLite table 'alerts']`, `[Source: Open-Meteo API v1]`).

2. **Experimental Terrain Computer Vision Pipeline (`src/geospatial/terrain_cv.py`):**
   - **Input Validation & Safety:** Enforces strict 10 MB payload limits, decodes verified formats (PNG, JPEG, WEBP), and normalizes images to 3-channel RGB grids downscaled to a maximum dimension of 1024 pixels.
   - **Surface Tension Crack Detection:** Converts luminance to grayscale ($Y = 0.299R + 0.587G + 0.114B$) and computes 2D spatial gradients via horizontal and vertical Sobel filters ($G_x, G_y$). Pixels exhibiting both high gradient magnitude ($|\nabla Y| > \mu_{|\nabla|} + 1.4\sigma_{|\nabla|}$) and negative contrast against local background luminance ($\bar{Y}_{15} - Y > 10.0$) are segmented. Morphological closing connects thin linear fissures, which are labeled into candidate components ($\ge 10\text{ px}$) to compute fissure count, maximum span, and a bounded crack severity index $[0.0, 1.0]$.
   - **Bare Soil Exposure & Scar Segmentation:** Masks upper overexposed sky pixels ($Y > 195.0$, $B \ge G - 10$, $y < 0.45h$) and separates healthy vegetative canopy via the Excess Green index ($ExG > 0.03$, $G > 1.04R$, $G > 1.04B$). Remaining non-sky terrain is classified into bare soil/scree based on earth-tone chromatic ratios ($R > 1.10B$, $R \ge 0.85G$, $R > 40$). Computes soil exposure ratio, vegetation cover percentage, and exposure classification (`Low < 15%`, `Moderate 15–40%`, `High > 40%`).
   - **Texture Roughness & Deformation Heuristics:** Computes local gradient standard deviation across exposed ground as a proxy for surface irregularity, stepped scarps, or hummocky terrain deformation.
   - **Composite Anomaly Scoring & Annotated Overlay:** Blends crack severity (45%), soil exposure (35%), and texture irregularity (20%) into a composite visual anomaly score $[0.0, 1.0]$ categorized into 4 tiers (`Normal`, `Low`, `Moderate`, `Significant Visual Anomaly`). Generates an annotated PNG overlay highlighting crack lines in red and bare soil in translucent amber.
   - **Mandatory Scientific Warning:** Enforces `CV_SCIENTIFIC_DISCLAIMER` stating that optical images only reflect superficial ground characteristics and cannot measure subsurface pore water pressure or subterranean slip planes.

3. **Multilingual Safety Guidance (`src/assistant/safety_guidance.py`):**
   - **Verified Civil Defense Standards:** Encapsulates official landslide preparedness, response, and recovery guidelines from the National Disaster Management Authority (NDMA) and Geological Survey of India (GSI).
   - **Supported Languages:** Provides complete parallel translations in English (`en`), Hindi (`hi` — हिन्दी), and Punjabi (`pa` — ਪੰਜਾਬੀ).
   - **Structured Phases:** Categorized into Before (pre-monsoon warning signs, evacuation route planning, emergency grab kits), During (indoor shielding, outdoor evacuation paths, vehicle precautions, fetal ball curl protection), and After (secondary slide avoidance, safe rescue direction, utility line hazards, foundation inspection).
   - **Emergency Helpline Directory:** Integrates All-India 24x7 toll-free emergency numbers (`112` National Emergency, `1078` NDMA Disaster Helpline, `1070` SDMA, `1077` DDMA, `100` Police, `108` Ambulance, `101` Fire).

---

## 15. Phase 9 — Disaster Response and Community Intelligence

1. **Nearby Emergency Facilities Mapping (`src/geospatial/facilities.py`):**
   - **Overpass API Integration:** Queries OpenStreetMap Overpass API (`https://overpass-api.de/api/interpreter`) for critical emergency facilities within a configurable radius: hospitals (`amenity=hospital`), police stations (`amenity=police`), fire and rescue stations (`amenity=fire_station`), and emergency shelters (`social_facility=shelter`).
   - **Curated Regional Fallback Directory:** Incorporates an authentic, pre-compiled directory of regional emergency facilities covering high-risk Indian mountain districts: Wayanad and Idukki (Kerala), Dehradun, Chamoli, Joshimath, Nainital, and Rudraprayag (Uttarakhand), Darjeeling and Kalimpong (West Bengal), Shimla, Mandi, and Kullu (Himachal Pradesh), and Nilgiris (Tamil Nadu).
   - **Proximity Analytics & Sorting:** Computes great-circle Haversine distances in kilometres from any selected station coordinate, sorts facilities ascending by distance, aggregates amenity counts, and isolates the single closest facility across each essential category.
   - **Data Freshness Badges:** Clearly labels results with provenance indicators: `🟢 Live Synchronized (OpenStreetMap API)` or `🟠 Authentic Regional Fallback (Offline Mode)`.

2. **Emergency Routing Engine & Safety Guard (`src/geospatial/routing.py`):**
   - **OSRM Road Network Routing:** Interfaces with the Open Source Routing Machine (`router.project-osrm.org`) driving service to compute real-world road path distance ($\text{km}$), estimated travel duration ($\text{min}$), and full GeoJSON LineString coordinates suitable for interactive map rendering.
   - **Geodesic Mountain Vector Fallback:** In offline scenarios or remote mountain locations lacking tagged OSM road segments, seamlessly computes direct great-circle vectors with an empirical hill road winding coefficient ($1.35\times$) and conservative average transit speed ($30\text{ km/h}$).
   - **Mandatory Safety Disclaimer:** Strictly requires and displays `ROUTING_DISCLAIMER` warning that paths are informational road maps and do **not** account for real-time landslide debris blockages, bridge washouts, or police closures.

3. **Crowd-Sourced Community Incident Reporting (`src/alerts/community_reporting.py`):**
   - **Field Observation Ingestion:** Provides a citizen and field volunteer reporting interface capturing incident description, hazard category (`Rockfall`, `Debris Flow / Mudflow`, `Tension Cracks`, `Road Subsidence / Slope Slump`, `Blocked Drainage`, `Other Hazard`), geographic coordinates, date/time, reporter details, and optional photographic proof.
   - **Input Validation:** Enforces strict bounds checking: coordinate range validation ($[-90..90, -180..180]$), description length ($\ge 10$ characters), category adherence, and a $10\text{ MB}$ maximum photo upload limit.
   - **Administrative Audit Workflow:** Reports enter the system with status `submitted` and follow an auditable lifecycle (`submitted` $\to$ `under_review` $\to$ `verified` / `rejected`), logging review timestamps, administrative notes, and reviewer identities.
   - **Strict Data Segregation Invariant:** Community reports are stored strictly in the isolated `community_reports` SQLite table and **never** automatically merge into scientific training catalogs (`observations`) or trigger automated public alarms.

4. **Historical Landslide Event Replay (`src/monitoring/event_replay.py`):**
   - **Multi-Source Event Exploration:** Retrieves recorded landslide occurrences from the project's authentic historical catalog (`NASA GLC` and national inventories).
   - **Environmental Telemetry Reconstruction:** Queries linked historical observations and antecedent conditions for the event epoch.
   - **Strict Interval Telemetry Refusal (Anti-Hallucination):** When historical hourly rainfall time series are absent, the system explicitly marks the timeline as unavailable rather than fabricating synthetic rainfall curves.
   - **Retrospective Risk Pipeline Evaluation:** Re-evaluates trained susceptibility models on reconstructed epoch features, appending `HISTORICAL_REPLAY_DISCLAIMER` to prevent retrospective bias and clarify that retrospective runs do not constitute prospective forecast validation.

5. **Automated PDF Risk Assessment Reports (`src/utils/pdf_generator.py`):**
   - **Publication-Ready Synthesis:** Uses ReportLab (`5.1.0`) to generate multi-page, publication-quality PDF risk assessment dossiers.
   - **Comprehensive Report Structure:** Integrates station geographic profile, dynamic 1h/24h/72h rainfall and soil moisture telemetry, trained machine-learning risk classifications and probability scores, nearest historical landslides, and nearest emergency response facilities.
   - **Non-Operational Academic Notices:** Prominently embeds official capstone research disclaimers (`PDF_DISCLAIMER_TEXT`), affirming that assessments are experimental outputs not certified for operational life-safety evacuation.

6. **Offline Synchronization & Cache Architecture (`src/monitoring/offline_cache.py`):**
   - **Persistent File-Based Store:** Caches weather telemetry, environmental observations, and emergency facility profiles to `data/cache/offline_sync/`.
   - **Integrity Rule Enforcement:** Every cached record is explicitly stamped with `is_live: False`, `from_offline_cache: True`, and `last_successful_sync` ISO timestamps. Cached information is **never** misrepresented as live data.
   - **Offline Health Dashboard:** Summarizes total cached entries, elapsed time since last synchronization, and upstream provider status.

7. **Authentication & Role-Based Access Control (`src/utils/auth.py`):**
   - **Role Separation:** Enforces two security roles: `admin` (full permissions: station management, community report verification, alert lifecycle governance) and `viewer` (read-only monitoring and reporting).
   - **Constant-Time Verification:** Employs `hmac.compare_digest` with SHA-256 password hashing to resist side-channel timing attacks.
   - **REST API Header Security:** Secures administrative FastAPI endpoints using the `X-Admin-Key` header with constant-time token comparison.
