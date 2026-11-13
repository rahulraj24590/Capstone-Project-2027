# LandslideGuard: AI-Based Dynamic Landslide Risk Prediction and Early Warning System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Academic Capstone Project** — B.Tech Computer Science and Engineering  
> **Important Notice:** This is an experimental research system. It is NOT a replacement for official geological surveys, government disaster-management authorities, or certified early-warning systems. Predictions are model estimates, not operational warnings.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Features](#features)
5. [Technology Stack](#technology-stack)
6. [Architecture](#architecture)
7. [Dataset Requirements](#dataset-requirements)
8. [Installation](#installation)
9. [Configuration](#configuration)
10. [Model Training](#model-training)
11. [Application Startup](#application-startup)
12. [API Usage](#api-usage)
13. [Testing](#testing)
14. [Known Limitations](#known-limitations)

---

## Project Overview

LandslideGuard is an intelligent landslide-risk assessment platform that combines machine learning, environmental data integration, geospatial analysis, and interactive visualization. The system is designed to:

- Assess long-term terrain susceptibility to landslides using static geographic and geological features.
- Estimate dynamic risk by incorporating changing environmental conditions such as rainfall intensity.
- Provide interactive geospatial dashboards for exploring historical landslide observations and risk estimates.
- Generate configurable experimental alerts when estimated risk exceeds defined thresholds.
- Explain model predictions using feature importance and SHAP analysis.

**This system distinguishes between:**
- **Susceptibility:** The long-term tendency of terrain to experience landslides (static).
- **Dynamic Risk:** An estimate incorporating current or recent environmental conditions.
- **Operational Warning:** A validated, threshold-based alert (experimental in this project).

---

## Problem Statement

Landslides are one of the most destructive natural hazards in mountainous and hilly regions globally. India's Western Ghats, Himalayas, and North-Eastern regions experience frequent and deadly landslide events, often triggered by intense rainfall. Traditional monitoring systems rely on manual inspection and are limited in geographic coverage. Machine learning and geospatial technologies offer the potential to build scalable, data-driven risk assessment tools.

---

## Objectives

1. Build a reusable data pipeline for processing landslide-related datasets.
2. Train and evaluate machine learning models for landslide susceptibility classification.
3. Implement a dynamic risk engine that responds to changing environmental inputs.
4. Integrate real-time weather data from Open-Meteo API.
5. Create an interactive geospatial dashboard for risk visualization.
6. Provide model explainability using SHAP and feature importance analysis.
7. Implement a configurable experimental alert management system.
8. Document methodology, limitations, and assumptions clearly.

---

## Features

### Core Features
- **Live Risk Monitoring Dashboard (Phase 1)** — Multi-station continuous surveillance with Open-Meteo reanalysis weather ingestion, SRTM terrain lookup, feed freshness badges, offline fallback, and station telemetry trends.
- **Landslide Susceptibility Modelling** using Random Forest, Gradient Boosting, and XGBoost.
- **Dynamic Risk Assessment** incorporating rainfall intensity, slope, and elevation.
- **Interactive Geospatial Risk Heatmap (Phase 2)** — Multi-layer Folium mapping integrating historical observations (`event_id`, location, date, source, trigger), experimental point risk markers (including `Unavailable`/`Insufficient Data`), kernel-density risk heatmap, location search, Indian State/District boundary zoom & filtering, click-level 7-part location inspector (`inspect_location_details`), RFC 7946 GeoJSON validation/reprojection (`EPSG:4326`), and DEM slope, aspect, terrain roughness (TRI/relief), and curvature analysis (`src/geospatial/`).
- **Advanced Geospatial, DEM Terrain & Satellite Intelligence (Phase 7)** — Rigorous DEM terrain derivatives (Elevation, Slope, Aspect, Riley et al. 1999 Terrain Roughness Index, D8 topological flow routing and authentic Topographic Wetness Index strictly requiring specific catchment area $a$, refusing computation without it), 3D interactive terrain model (`go.Surface` with elevation/slope/roughness drape modes), DBSCAN haversine spatial clustering on radian coordinates (`eps_km=25.0`, `min_samples=3`) & continuous 2D Kernel Density Estimation (KDE) with scientific anti-hallucination disclaimers (*"Do not call an area a future landslide hotspot based only on historical clustering"*), Sentinel-2 Level-2A multi-spectral NDVI monitoring strictly requiring Red (B04) & NIR (B08) spectral bands with SCL/QA cloud masking (*"Do not interpret vegetation changes alone as proof of landslide activity"*), multi-temporal change detection ($t_1 \rightarrow t_2$) with dual-epoch cloud screening and candidate disturbance quantification, and 4 dedicated FastAPI endpoints (`/api/v1/geospatial/hotspots`, `/api/v1/satellite/ndvi`, `/api/v1/satellite/change-detection`, `/api/v1/terrain/derivatives`).
- **AI Assistant, Multilingual Safety & Computer Vision (Phase 8)** — Retrieval-augmented context-aware QA assistant querying authentic SQLite records, trained model metadata, and verified sources with strict anti-hallucination guards (never inventing risk scores or unindexed locations), experimental slope/landscape Computer Vision analyzing surface tension cracks, bare soil exposure ratios, and hummocky terrain deformation with annotated feature overlays and mandatory non-operational disclaimers (*"An image classification result is NOT proof of an active landslide and must NEVER independently trigger an official warning"*), verified NDMA/GSI multilingual disaster preparedness guidance in English, Hindi (हिन्दी), and Punjabi (ਪੰਜਾਬੀ), and 3 dedicated FastAPI endpoints (`/api/v1/assistant/query`, `/api/v1/cv/analyze-image`, `/api/v1/assistant/safety-guidance`).
- **Dynamic Rainfall Analysis & Phase 6 Analytics (`src/monitoring/environmental_analytics.py`)** — Open-Meteo weather provider separating current model snapshots (`weather_code` WMO condition decoding), past reanalysis, and future forecasts; strict coordinate & parameter validation; HTTP `429` rate-limit detection & `bypass_cache` manual refresh control; unit validation & conversion; explicit cache & API connection provenance (cached responses never shown as live); strict `1h / 24h / 48h / 72h` accumulations without zero-imputation; strict incomplete-forecast detection; ERA5 historical-baseline anomaly detection (Z-score, rolling IQR/mean+std thresholds, Isolation Forest when `>= 60` baseline days, tiered `Normal / Above-normal / Unusual / Extreme` classification strictly gated to `>= 30` baseline days); daily/weekly/monthly/historical/seasonal trend analysis; 24h/48h/72h forecast-driven risk estimation with widening horizon uncertainty ribbons and explicit non-timing disclaimers; 5+ interactive Plotly charts; dynamic risk delta explanations; and manual fallback inputs.
- **Explainable AI — SHAP & Feature Importance (Phase 4)** — Global explainability (`LinearExplainer` / `TreeExplainer`, permutation importance with error bars, intrinsic tree importance, cross-method feature contribution comparison, SHAP summary, interactive beeswarm, and impact distribution box plots) plus individual waterfall attributions, positive/negative contribution comparison, missing-input & limitation reporting, zero-input refusal, full model metadata provenance, feature-order alignment, caching, and non-causal Feature Importance vs. SHAP vs. Correlation vs. Causation guidance (`src/models/explainability.py`).
- **Evidence-Gated Model Drift Monitoring (Phase 6)** — Tracks Population Stability Index (PSI) and two-sample KS distribution shifts, training-median imputation & missingness deltas, prediction probability distribution shifts, and ground-truth `prediction_outcomes` performance deltas (`src/monitoring/drift_monitor.py`), never confirming drift without sufficient sample counts and statistical corroboration.
- **Historical Analysis** of landslide occurrences and environmental patterns.
- **Model Explainability** via SHAP values and permutation feature importance.
- **Experimental Alert Management System (Phase 5)** — Configurable experimental alert generation (`Low`, `Moderate`, `High`, `Very High`), full alert lifecycle (`new`, `acknowledged`, `resolved`) with audit timestamps, configurable cooldown suppression with automatic severity escalation override (`src/alerts/alert_manager.py`), category/date filtering dashboard tabs, SQLite indexing & migrations, and disabled-by-default Email/SMS notification provider hooks (`src/alerts/notifications.py`).
- **Weather Integration** via Open-Meteo API (no API key required).
- **FastAPI Backend** for programmatic access to predictions and live station monitoring.
- **SQLite Database** for storing monitored locations, environmental observations, predictions, and alerts.

### Dashboard Pages
1. **Overview** — System status, data provenance badge, data quality indicators, and observed class distribution.
2. **Live Monitoring** — Centralized KPIs (Monitored Locations, Historical Records, Low/Medium/High/Very High Risk counts, Active Experimental Alerts, Latest Weather Update), environmental cards (Rainfall 1h/24h/72h, Temperature, Humidity, Soil Moisture, Elevation, Slope, Weather Condition, Source & Timestamp), Risk by Region & Historical Distribution charts, and separated Recent Activity tabs (Recently Updated Locations, Authentic Historical Records, All Observations, Weather Updates, Risk Assessments, Alerts).
3. **Risk Analysis** — Location-specific experimental risk estimation with SHAP explanations.
4. **Geospatial Map (Phase 2 & Phase 7)** — Geographic filtering, experimental risk heatmap, GeoJSON overlay/export, interactive DBSCAN landslide hotspot clustering, 2D continuous Kernel Density Estimation (KDE), Sentinel-2 MSI NDVI vegetation monitoring, multi-temporal satellite change detection ($t_1 \rightarrow t_2$), interactive 3D terrain surface (`go.Surface`), D8 flow accumulation and authentic TWI calculation, point SRTM elevation inspector, and local GeoTIFF DEM raster inspection (`rasterio` / PIL fallback).
5. **Weather Analysis** — Separated current/historical/forecast Open-Meteo views, `1h/24h/48h/72h` accumulations, ERA5 anomaly detection (Z-score, rolling bounds, Isolation Forest), daily/weekly/monthly/seasonal trends, 24h/48h/72h forecast-driven risk estimation with uncertainty ribbons, dynamic risk delta integration, and manual fallback inputs.
6. **India Regional Monitoring** — 36 Indian States/UTs -> 735 Districts drilldown, map boundaries, live Open-Meteo telemetry, NDMA SACHET official CAP warnings, and IMD status.
7. **Historical Events** — Temporal and spatial analysis of historical landslide inventories.
8. **Model Performance** — Evaluation metrics, confusion matrices, ROC curves, plus evidence-gated Phase 6 Model Drift & Data Quality Monitoring (feature/prediction PSI & KS shifts, missingness deltas, and verified `prediction_outcomes` logging).
9. **AI Explainability** — Model provenance & evaluation summary, global SHAP summary, permutation & tree-based importance, cross-method comparison, beeswarm & distribution plots, SQLite/eval sample waterfall inspector with missing-input & limitation tracking, and Feature Importance vs. SHAP vs. Correlation vs. Causation guide.
10. **Alert Management** — Alert lifecycle (`new`, `acknowledged`, `resolved`), severity escalation override, cooldown rules, and audit history.
11. **AI Assistant & Vision** — Three-tab intelligent interface: LandslideGuard Assistant with suggested question pills, source citations, and context selectors; Experimental Terrain Image Analysis with drag-and-drop upload, sample benchmark scenarios, crack detection, bare soil segmentation, and annotated overlays; and Multilingual Safety Guidance with NDMA/GSI civil defense checklists and All-India emergency hotlines in English, Hindi, and Punjabi.
12. **Data Management** — SQLite table row-count inspector, authentic (`is_synthetic=0`) vs. synthetic dataset breakdown, `dataset_inventory.csv` viewer, boundary/cache artifact check, and CSV exports.
13. **Settings** — Read-only inspector for active `config/settings.yaml` thresholds, alert cooldowns, provider configs, and masked environment variable status.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Dashboard | Streamlit |
| ML Framework | scikit-learn, XGBoost |
| Explainability | SHAP |
| Geospatial | Folium, GeoPandas, Rasterio, OSRM |
| Reporting & PDFs | ReportLab |
| Visualization | Plotly |
| API Backend | FastAPI |
| Database | SQLite (via Python sqlite3) |
| Data Processing | Pandas, NumPy |
| Weather Data | Open-Meteo API |
| Configuration | PyYAML, python-dotenv |

---

## Architecture

```
LandslideGuard/
├── app.py                          # Streamlit dashboard entry point
├── api.py                          # FastAPI backend
├── config/
│   └── settings.yaml               # Application configuration
├── data/
│   ├── raw/                        # Unmodified source data
│   ├── processed/                  # Cleaned, feature-engineered data
│   └── dataset_inventory.csv       # Documented dataset metadata
├── models/                         # Saved model artifacts
├── notebooks/                      # Experimental notebooks
├── scripts/
│   ├── download_data.py            # Dataset acquisition
│   ├── validate_data.py            # Data quality checks
│   └── prepare_data.py             # Full preprocessing pipeline runner
├── src/
│   ├── data_processing/
│   │   ├── preprocessing.py        # Data cleaning and validation
│   │   └── feature_engineering.py  # Feature creation and transformation
│   ├── models/
│   │   ├── train.py                # Model training pipeline
│   │   ├── evaluate.py             # Evaluation metrics
│   │   ├── predict.py              # Inference module
│   │   └── model_registry.py       # Model versioning and loading
│   ├── risk/
│   │   ├── risk_engine.py          # Dynamic risk scoring
│   │   ├── thresholds.py           # Risk category definitions
│   │   └── validation.py           # Input validation
│   ├── data_providers/
│   │   ├── base.py                 # Abstract provider interface
│   │   ├── weather_provider.py     # Open-Meteo integration
│   │   └── terrain_provider.py     # Terrain data utilities
│   ├── geospatial/
│   │   ├── maps.py                 # Folium maps, HeatMap, GeoJSON overlays & popups
│   │   ├── spatial_analysis.py     # Geographic filtering, spatial density & GeoJSON validation
│   │   ├── terrain.py              # SRTM/Rasterio slope, aspect, TRI & TWI terrain tools
│   │   ├── terrain_cv.py           # Phase 8 terrain crack & bare soil computer vision
│   │   ├── facilities.py           # Phase 9 OpenStreetMap Overpass & curated emergency facilities
│   │   └── routing.py              # Phase 9 OSRM road routing & geodesic fallback engine
│   ├── database/
│   │   ├── db.py                   # Database connection, schemas and initialization
│   │   └── models.py               # Table schemas and CRUD helpers
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── live_monitor.py         # Live station monitoring & offline fallback
│   │   ├── rainfall_analysis.py    # Rainfall accumulation & trend analytics
│   │   ├── event_replay.py         # Phase 9 historical event replay & retrospective engine
│   │   └── offline_cache.py        # Phase 9 offline synchronization & persistent disk cache
│   ├── alerts/
│   │   ├── alert_manager.py        # Alert generation and lifecycle management
│   │   ├── notifications.py        # Notification dispatch hooks
│   │   └── community_reporting.py  # Phase 9 crowd-sourced incident reporting & review
│   ├── assistant/
│   │   ├── knowledge_engine.py     # Phase 8 context-aware QA knowledge engine
│   │   └── safety_guidance.py      # Phase 8 multilingual civil defense safety guide
│   └── utils/
│       ├── auth.py                 # Phase 9 RBAC, password hashing & API key security
│       ├── pdf_generator.py        # Phase 9 ReportLab automated PDF assessment reports
│       ├── logging_config.py       # Logging setup
│       └── validation.py           # Input validation utilities
├── tests/                          # pytest test suite
└── reports/                        # Academic documentation
```

---

## Dataset Requirements

The project supports the following data sources:

### Primary: NASA Global Landslide Catalog (GLC)
- **URL:** https://catalog.data.gov/dataset/global-landslide-catalog-export
- **Coverage:** Global, 1988–present
- **Access:** Public CSV download (requires free registration)
- **Features:** Latitude, longitude, date, landslide type, trigger, fatalities
- **Note:** Contains only positive (landslide) observations. Negative samples must be generated.

### Weather: Open-Meteo API
- **URL:** https://open-meteo.com/ (Forecast API + ERA5 Historical Archive API; CC BY 4.0)
- **Access:** Free, no API key required for standard variables
- **Variables:** `precipitation`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `soil_moisture_0_to_7cm` (model-derived volumetric soil water), `weather_code` (WMO 4677), daily `precipitation_sum`, `temperature_2m_max`/`min`
- **Limitations:** Model-derived reanalysis/forecasts (not physical rain-gauge station telemetry); rainfall alone cannot establish landslide causation; subject to free-tier rate limits (`HTTP 429`), handled with retry/backoff, disk caching, and manual fallback inputs. Historical ERA5 data available from 1940; forecast up to 16 days.

### Terrain: SRTM via Open-Elevation API
- **URL:** https://api.open-elevation.com/
- **Access:** Free, no API key required
- **Variables:** Elevation (metres)
- **Note:** Slope must be derived from elevation data.

### Synthetic Development Dataset
When authentic training data is unavailable or insufficient, the project generates a clearly labelled synthetic dataset for pipeline testing. This dataset uses geophysically plausible parameter distributions derived from published research but **must not be used to claim real-world predictive performance**.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip 22+
- Internet connection (for weather data integration)

### Steps

```bash
# 1. Clone repository
git clone <repository-url>
cd LandslideGuard

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env with your settings (optional; no API keys required for core functionality)

# 5. Initialize database
python -c "from src.database.db import initialize_database; initialize_database()"

# 6. Prepare data, cache India State/District boundaries, and train model
python scripts/prepare_data.py
python scripts/download_india_boundaries.py
python -m src.models.train
```

---

## Configuration

Edit `config/settings.yaml` to adjust:
- Risk thresholds and category boundaries
- Alert cooldown periods
- Weather data freshness requirements
- Model version and path
- Logging level

See `config/settings.yaml` for all available options with documentation.

---

## Model Training

```bash
# Train all models and save the best one
python -m src.models.train

# Evaluate saved model
python -m src.models.evaluate

# Results are saved to reports/model_evaluation.md
```

---

## Application Startup

```bash
# Start Streamlit dashboard
streamlit run app.py

# Start FastAPI backend (optional, in a separate terminal)
uvicorn api:app --reload --port 8000
```

---

## Deployment

### Option A: Streamlit Community Cloud (Recommended Free Hosting)
1. Push the repository branch (`landslide-guard` or `main`) to GitHub. Note that `.gitignore` intentionally excludes `.env`, `*.db`, `data/raw/*`, `data/processed/*.parquet`, and `models/*.pkl`.
2. Sign in to [share.streamlit.io](https://share.streamlit.io), select the repository/branch, set **Main file path** to `app.py`, and select Python `3.14` (if selectable) in **Advanced Settings**.
3. *(Optional)* Add environment variables such as `OPENMETEO_API_KEY` or `LANDSLIDEGUARD_ENV` under **App Secrets**. No API keys are required for core functionality.
4. On initial cold boot without local `.parquet` / `.pkl` files, the dashboard renders cleanly with empty-state instructions and heuristic fallback risk estimation; run `scripts/prepare_data.py` + `python -m src.models.train` locally or via Docker if full historical charts and trained model artifacts are desired.

### Option B: Docker Container Deployment
A production-style `Dockerfile` and `.dockerignore` are included for self-contained deployment (automatically initializes SQLite, generates pipeline-test synthetic data, and trains model artifacts during build):
```bash
# Build container image
docker build -t landslideguard:latest .

# Run Streamlit dashboard container on port 8501
docker run -d --name landslideguard -p 8501:8501 landslideguard:latest

# Optional: run FastAPI backend instead of Streamlit UI
docker run -d --name landslideguard-api -p 8000:8000 landslideguard:latest \
  uvicorn api:app --host 0.0.0.0 --port 8000
```

> **Security Note:** Never bake `.env` or secret credentials into Docker images, and never enable external Email/SMS notification hooks in public demos unless explicitly authorized.

---

## API Usage

The FastAPI backend exposes the following endpoints when running:

```
GET   /health                                        System health check
GET   /api/v1/model-info                             Loaded model metadata
POST  /api/v1/predict                                Single-location risk prediction
GET   /api/v1/observations                           Historical observations
GET   /api/v1/alerts                                 Filter alerts by lifecycle status, category & date
POST  /api/v1/alerts/{alert_id}/acknowledge          Acknowledge an alert (with audit timestamp)
POST  /api/v1/alerts/{alert_id}/resolve              Resolve an alert (with audit timestamp)
GET   /api/v1/alerts/audit                           Retrieve alert acknowledgement & resolution audit log
GET   /api/v1/alerts/notification-status             Check Email/SMS notification hook status (disabled by default)
GET   /api/v1/monitoring/locations                   List monitored stations
POST  /api/v1/monitoring/locations                   Register custom station
PATCH /api/v1/monitoring/locations/{location_id}     Pause or resume a station
POST  /api/v1/monitoring/seed-defaults               Register default Indian stations
GET   /api/v1/monitoring/status                      Latest live snapshot across stations
POST  /api/v1/monitoring/refresh                     Refresh station weather, risk & alerts
GET   /api/v1/monitoring/locations/{location_id}/history  Station environmental & risk history
GET   /api/v1/geospatial/risk-geojson                Export assessed risk estimates as GeoJSON
GET   /api/v1/rainfall/analysis                      Separated Open-Meteo rainfall & analytics
POST  /api/v1/rainfall/evaluate-risk                 Dynamic rainfall / manual risk evaluation with delta explanation
GET   /api/v1/explain/global                         Global SHAP summary, permutation & tree importance
POST  /api/v1/explain/predict                        Individual prediction with SHAP waterfall contributions & statements
GET   /api/v1/india/states                           List cached Indian States/UTs (`geoBoundaries ADM1`)
GET   /api/v1/india/districts                        List dependent Indian Districts filtered by State/UT (`geoBoundaries ADM2`)
POST  /api/v1/india/monitor                          Run 10-step India State -> District regional monitoring & experimental RiskEngine + SHAP assessment
GET   /api/v1/india/providers/status                 Retrieve live/cached status & freshness of Geographical Boundaries, Open-Meteo, IMD & NDMA SACHET
GET   /api/v1/analytics/rainfall-intelligence        Phase 6 baseline-backed rainfall anomaly & multi-scale trend intelligence
GET   /api/v1/forecast/risk-trend                    Phase 6 24h/48h/72h forecast-driven risk estimation with uncertainty bounds
GET   /api/v1/model/drift                            Phase 6 evidence-gated feature, missingness, prediction & performance drift report
POST  /api/v1/model/outcomes                         Record verified ground-truth landslide outcome for live drift validation
GET   /api/v1/geospatial/hotspots                    Phase 7 DBSCAN haversine spatial clustering & 2D KDE hotspot detection
POST  /api/v1/satellite/ndvi                         Phase 7 Sentinel-2 MSI NDVI spectral calculation & SCL cloud masking
POST  /api/v1/satellite/change-detection             Phase 7 Multi-temporal satellite change detection & disturbance analysis
POST  /api/v1/terrain/derivatives                    Phase 7 DEM terrain derivatives (Slope, Aspect, TRI, D8 Flow, TWI)
POST  /api/v1/assistant/query                        Phase 8 Context-aware QA assistant querying authentic project data
POST  /api/v1/cv/analyze-image                       Phase 8 Experimental terrain computer vision (cracks, soil, deformation)
GET   /api/v1/assistant/safety-guidance              Phase 8 Multilingual landslide safety guidance (English, Hindi, Punjabi)
GET   /api/v1/response/facilities                    Phase 9 Nearby emergency facilities (OpenStreetMap + curated fallback)
POST  /api/v1/response/route                         Phase 9 Emergency route calculation (OSRM driving + geodesic fallback)
POST  /api/v1/community/reports                      Phase 9 Submit crowd-sourced community hazard report
GET   /api/v1/community/reports                      Phase 9 List community hazard reports with status filtering
PATCH /api/v1/admin/community/reports/{report_id}    Phase 9 Review and update community report status (Admin authenticated)
GET   /api/v1/reports/pdf/{location_id}              Phase 9 Generate & download ReportLab PDF risk assessment report
GET   /api/v1/monitoring/offline-status              Phase 9 Offline sync cache health, sync age & freshness indicators
```

API documentation: http://localhost:8000/docs (Swagger UI when server is running)

### Example prediction request:
```json
POST /api/v1/predict
{
  "latitude": 22.5,
  "longitude": 88.3,
  "rainfall_1h_mm": 25.0,
  "rainfall_24h_mm": 80.0,
  "rainfall_72h_mm": 120.0,
  "slope_degrees": 28.0,
  "elevation_m": 650.0
}
```

---

## Testing

```bash
# Run full test suite (238 automated tests across 16 modules)
python -m pytest tests/ -v --tb=short

# Run specific phase/integration suite
pytest tests/test_end_to_end_integration.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Reports & Capstone Documentation

- [reports/final_audit.md](reports/final_audit.md) — Complete technical audit, security/performance checks & remediation log
- [reports/final_project_report.md](reports/final_project_report.md) — Comprehensive academic B.Tech capstone report across all 9 phases
- [reports/viva_preparation.md](reports/viva_preparation.md) — Complete viva guide with 100 technical, ML, GIS & architecture Q&As
- [reports/methodology.md](reports/methodology.md) — Feature engineering, ML splits, risk engine & explainability methods
- [reports/dataset_report.md](reports/dataset_report.md) — Dataset inventory, schemas & biases
- [reports/model_evaluation.md](reports/model_evaluation.md) — Held-out validation/test metrics & spatial data leakage analysis
- [reports/limitations.md](reports/limitations.md) — Scientific constraints & non-operational disclaimers

---

## Known Limitations

1. **No real-time data by default.** Weather data is fetched on demand from Open-Meteo. For genuine early warning, live data streams would be required.
2. **Synthetic training data.** The default training dataset is synthetic. Models trained on synthetic data cannot claim real-world predictive validity.
3. **No official calibration.** Risk thresholds have not been validated against historical event data from Indian meteorological authorities.
4. **Point-based estimation.** The system estimates risk at discrete point locations. Spatial interpolation is not validated.
5. **Class imbalance.** Landslide events are rare. Models may underestimate false-negative rates in real-world conditions.
6. **Not a replacement for professional assessment.** This system must not be used to make evacuation or emergency management decisions.
7. **GeoPandas optional.** If GDAL binaries are unavailable on Windows, some geospatial features gracefully degrade.

See `reports/limitations.md` for a comprehensive discussion.

---

## Academic Context

This project was developed as a B.Tech Computer Science and Engineering capstone project. The implementation demonstrates:
- End-to-end machine learning pipeline development
- Geospatial data processing and visualization
- REST API design and implementation
- Software engineering best practices
- Scientific integrity in ML claims and documentation

---

## License

MIT License — see LICENSE file for details.

---

*LandslideGuard is an academic research prototype. It is not suitable for operational use in disaster management.*
