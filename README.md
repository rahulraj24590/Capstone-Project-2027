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
- **Landslide Susceptibility Modelling** using Random Forest, Gradient Boosting, and XGBoost.
- **Dynamic Risk Assessment** incorporating rainfall intensity, slope, and elevation.
- **Interactive Risk Map** with Folium-based geospatial visualization.
- **Historical Analysis** of landslide occurrences and environmental patterns.
- **Model Explainability** via SHAP values and permutation feature importance.
- **Experimental Alert Management** with configurable thresholds and cooldown periods.
- **Weather Integration** via Open-Meteo API (no API key required).
- **FastAPI Backend** for programmatic access to predictions.
- **SQLite Database** for storing observations, predictions, and alerts.

### Dashboard Pages
1. **Overview** — System status, data quality indicators, and risk distribution.
2. **Interactive Risk Map** — Geospatial visualization with filtering.
3. **Risk Prediction** — Location-specific risk estimation with explanations.
4. **Historical Analysis** — Temporal and spatial analysis of historical events.
5. **Model Performance** — Evaluation metrics, confusion matrices, ROC curves.
6. **Alerts** — Alert management with acknowledgement and audit history.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Dashboard | Streamlit |
| ML Framework | scikit-learn, XGBoost |
| Explainability | SHAP |
| Geospatial | Folium, GeoPandas |
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
│   │   ├── maps.py                 # Folium map creation
│   │   └── spatial_analysis.py     # Spatial operations
│   ├── database/
│   │   ├── db.py                   # Database connection and initialization
│   │   └── models.py               # Table schemas
│   ├── alerts/
│   │   └── alert_manager.py        # Alert generation and management
│   └── utils/
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
- **URL:** https://open-meteo.com/
- **Access:** Free, no API key required for standard variables
- **Variables:** Precipitation (1h, 24h, sum), temperature, wind speed
- **Note:** Historical data available from 1940; forecast up to 16 days.

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

# 6. Prepare data and train model
python scripts/prepare_data.py
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

## API Usage

The FastAPI backend exposes the following endpoints when running:

```
GET  /health                              System health check
GET  /api/v1/model-info                   Loaded model metadata
POST /api/v1/predict                      Single-location risk prediction
GET  /api/v1/observations                 Historical observations
GET  /api/v1/alerts                       Active alerts
POST /api/v1/alerts/{alert_id}/acknowledge  Acknowledge an alert
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
# Run full test suite
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

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
