# LandslideGuard — Dataset Report

**Project:** LandslideGuard  
**Date:** November 2026
**Status:** Integrated (Synthetic Training Dataset + Live Open-Meteo Forecast & Archive API)

---

## 1. Dataset Status Summary

| Dataset | Status | Notes |
|---------|--------|-------|
| Synthetic Development Dataset | ✅ Generated & Used | Clearly labelled; used for pipeline testing only |
| NASA Global Landslide Catalog | ⛔ Not Obtained | Requires NASA Earthdata account; access steps documented |
| Open-Meteo Weather API | 🔄 Integrated (No key needed) | Live fetch + caching implemented |
| SRTM Elevation Data | ⛔ Not Obtained | Requires USGS EarthExplorer account |
| ESA WorldCover | ⛔ Not Obtained | Large GeoTIFF; requires disk space planning |
| SoilGrids REST API | ⛔ Not Integrated | REST API available; integration documented |

---

## 2. Synthetic Development Dataset

### Purpose
Exclusively for testing the data pipeline, model training code, and dashboard components during development. **Not used to claim real-world predictive performance.**

### Generation Method
`scripts/generate_synthetic_data.py`

### Schema
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `latitude` | float | 8 – 35 | Decimal degrees, N |
| `longitude` | float | 68 – 97 | Decimal degrees, E |
| `timestamp` | datetime | 2018 – 2023 | UTC observation time |
| `slope_degrees` | float | 5 – 70 | Terrain slope |
| `elevation_m` | float | 100 – 3500 | Metres above sea level |
| `rainfall_1h_mm` | float | 0 – 80 | Hourly rainfall |
| `rainfall_24h_mm` | float | 0 – 300 | 24-hour accumulated rainfall |
| `rainfall_72h_mm` | float | 0 – 700 | 72-hour accumulated rainfall |
| `vegetation_index` | float | 0.1 – 0.9 | Normalised NDVI proxy |
| `soil_moisture` | float | 0.1 – 0.9 | Volumetric fraction proxy |
| `soil_type` | string | loam/clay/sandy | Soil texture category |
| `land_cover` | string | forest/agricultural/... | Land cover class |
| `landslide_occurred` | int | 0 or 1 | Target variable |
| `is_synthetic` | int | 1 | Always 1 (provenance label) |

### Statistics
- Total records: 2,000 (`19` columns in `data/processed/dataset.parquet`)
- Positive class (`landslide_occurred=1`): 500 (25.0%)
- Negative class (`landslide_occurred=0`): 1,500 (75.0%)
- Provenance label (`is_synthetic=1`): 2,000 / 2,000 (100% synthetic development data)
- Temporal range: January 2018 – December 2023
- Geographic extent: Indian subcontinent bounding box (8°N–35°N, 68°E–97°E)
- Metadata columns also present: `event_date`, `source_name`, `country`, `landslide_type`, `trigger`, `fatalities`

---

## 3. NASA Global Landslide Catalog

### Access Instructions
1. Register at https://urs.earthdata.nasa.gov/
2. Set environment variables:
   ```
   NASA_EARTHDATA_USERNAME=<your_username>
   NASA_EARTHDATA_PASSWORD=<your_password>
   ```
3. Run: `python scripts/download_data.py --source nasa_glc`

### Documented Characteristics
- Coverage: Global, 2007 – present
- Positive landslide events only (no confirmed negatives)
- Location accuracy varies (point uncertainty up to tens of km)
- Trigger type recorded (rainfall, earthquake, anthropogenic, etc.)
- Data format: CSV/GeoJSON

### Known Biases
- Heavily biased toward English-language news reports and scientific literature.
- Under-reports events in data-sparse regions (sub-Saharan Africa, parts of South Asia).
- Temporal coverage improves after 2015 when satellite-based detection improved.

---

## 4. Open-Meteo Weather API

### Endpoint
`https://api.open-meteo.com/v1/forecast`  
`https://archive-api.open-meteo.com/v1/archive`

### No API Key Required
Free tier supports up to ~10,000 requests/day. No registration required.

### Variables Used
| Variable | Open-Meteo Name | Units |
|----------|----------------|-------|
| Hourly precipitation | `precipitation` | mm/h |
| 24h precipitation sum | `precipitation_sum` (daily) | mm/day |
| 2m temperature | `temperature_2m` | °C |
| Relative humidity | `relative_humidity_2m` | % |
| Soil moisture (0-7cm layer) | `soil_moisture_0_to_7cm` | m³/m³ |

### Important Notes
- `soil_moisture_0_to_7cm` is **model-derived** (ERA5-based), not observed.
- Spatial resolution: ~25 km (ERA5) to ~5 km (ECMWF IFS).
- Not a substitute for dense rain gauge networks in complex terrain.

### Attribution
Weather data by Open-Meteo.com (CC BY 4.0)

---

## 5. India Regional Boundary & Official Alert Sources

- **geoBoundaries IND ADM1 & ADM2 (`data/india_boundaries/`):** Downloaded via `scripts/download_india_boundaries.py` (`36` States/UTs + `735` Districts with spatial containment + nearest-island fallback join and verified `EPSG:4326` representative coordinates inside each district polygon). GeoJSON geometry files are `.gitignore`d while metadata syncs into SQLite (`india_states`, `india_districts`).
- **NDMA SACHET CAP Feed (`src/data_providers/sachet_provider.py`):** Live Common Alerting Protocol JSON (`FetchAllAlertDetails`) with `ETag` conditional caching and RSS XML (`rss_india.xml`) fallback. Official alerts persist into SQLite `official_warnings` strictly separated from experimental ML alerts.
- **IMD Operational API (`src/data_providers/imd_provider.py`):** Supports official documented `api.imd.gov.in/api/v1/` endpoints (`staterainfall`, `districtrainfall`, `districtwarning`, `districtnowcast`, `state_district_rainfall_forecast`) using `.env` credentials (`IMD_API_KEY`), reporting `"Not configured"` cleanly when absent.

---

## 6. Data Inventory CSV

See [dataset_inventory.csv](dataset_inventory.csv) for the machine-readable inventory.

---

## 6. Data Quality Indicators

| Indicator | Current Status |
|-----------|---------------|
| Duplicate records | None detected (synthetic dataset) |
| Invalid coordinates | None (validated on generation) |
| Missing values | < 5% (synthetic, imputed by median) |
| Label quality | Synthetic labels; no real-world validation |
| Temporal coverage | Adequate for pipeline testing |
| Spatial coverage | Notional India bounding box |
