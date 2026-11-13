"""
LandslideGuard — FastAPI Backend

Provides a REST API for programmatic access to:
- System health checks
- Model information
- Single-location risk predictions
- Historical observations
- Alert management

Run with:
    uvicorn api:app --reload --port 8000

API docs available at: http://localhost:8000/docs

IMPORTANT:
- All predictions are EXPERIMENTAL and have not been operationally validated.
- This system must NOT be used as a substitute for official geological or
  disaster-management early-warning services.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.database.db import initialize_database, get_table_counts
from src.utils.logging_config import setup_logging, get_logger
from src.utils.settings import get_setting

# ─────────────────────────────────────────────
# Application bootstrap
# ─────────────────────────────────────────────

setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Ensure database is initialised on startup
initialize_database()

APP_NAME = get_setting("app", "name") or "LandslideGuard"
APP_VERSION = get_setting("app", "version") or "1.0.0"
APP_DESCRIPTION = get_setting("app", "description") or ""


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage FastAPI startup and shutdown lifecycle."""
    logger.info(
        "%s API v%s started. Docs at http://localhost:8000/docs",
        APP_NAME, APP_VERSION,
    )
    yield
    logger.info("%s API shutting down.", APP_NAME)


app = FastAPI(
    title=f"{APP_NAME} API",
    version=APP_VERSION,
    description=(
        f"{APP_DESCRIPTION}\n\n"
        "⚠️ **EXPERIMENTAL SYSTEM** — All risk estimates are research outputs "
        "from a capstone project. They have NOT been validated by any geological "
        "or disaster-management authority."
    ),
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Request body for a single-location risk prediction."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    rainfall_1h_mm: float | None = Field(None, ge=0, description="1-hour rainfall in mm")
    rainfall_24h_mm: float | None = Field(None, ge=0, description="24-hour rainfall in mm")
    rainfall_72h_mm: float | None = Field(None, ge=0, description="72-hour rainfall in mm")
    slope_degrees: float | None = Field(None, ge=0, le=90, description="Slope in degrees")
    elevation_m: float | None = Field(None, ge=-500, le=9000, description="Elevation in metres")
    vegetation_index: float | None = Field(None, ge=-1, le=1, description="NDVI vegetation index")
    soil_moisture: float | None = Field(None, ge=0, le=1, description="Soil moisture (0–1)")


class PredictionResponse(BaseModel):
    """Response body for a risk prediction."""
    probability: float | None
    risk_category: str
    method: str
    is_experimental: bool = True
    data_quality: dict = {}
    warnings: list[str] = []
    errors: list[str] = []
    features_used: dict = {}
    location_name: str | None = None
    estimated_at: str | None = None


class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    app_name: str
    version: str
    database: dict
    model_available: bool
    timestamp: str
    notice: str


class ModelInfoResponse(BaseModel):
    """Active model metadata."""
    name: str | None
    algorithm: str | None
    saved_at: str | None
    feature_columns: list[str] = []
    test_metrics: dict = {}
    is_experimental: bool = True


class AcknowledgeRequest(BaseModel):
    """Request body for acknowledging an alert."""
    note: str | None = None


class MonitoringLocationCreate(BaseModel):
    """Request body to register a monitored station."""
    name: str = Field(..., min_length=1, max_length=120)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    country: str | None = "India"
    state_region: str | None = None
    elevation_m: float | None = Field(None, ge=-500, le=9000)
    slope_degrees: float | None = Field(None, ge=0, le=90)
    fetch_terrain: bool = True
    is_active: bool = True


class MonitoringLocationStatusUpdate(BaseModel):
    """Request body to activate or pause monitoring for a station."""
    is_active: bool


class MonitoringRefreshRequest(BaseModel):
    """Request body to refresh live monitoring stations."""
    location_id: int | None = None
    active_only: bool = True
    generate_alerts: bool = True


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check — verifies database and model availability."""
    from src.models.model_registry import active_model_exists

    db_counts = get_table_counts()
    model_ok = active_model_exists()

    return HealthResponse(
        status="healthy",
        app_name=APP_NAME,
        version=APP_VERSION,
        database=db_counts,
        model_available=model_ok,
        timestamp=datetime.now(timezone.utc).isoformat(),
        notice=(
            "EXPERIMENTAL SYSTEM — All outputs are research prototypes. "
            "Not suitable for operational disaster management."
        ),
    )


# ─────────────────────────────────────────────
# Model information
# ─────────────────────────────────────────────

@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return metadata about the currently loaded model."""
    from src.models.model_registry import get_model_info

    info = get_model_info("best_model")
    if info is None:
        return ModelInfoResponse(
            name=None,
            algorithm=None,
            saved_at=None,
            is_experimental=True,
        )

    meta = info.get("metadata", {})
    return ModelInfoResponse(
        name=info.get("name"),
        algorithm=meta.get("algorithm"),
        saved_at=info.get("saved_at"),
        feature_columns=meta.get("feature_columns", []),
        test_metrics=meta.get("test_metrics", {}),
        is_experimental=True,
    )


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Estimate landslide risk for a single location.

    Requires latitude, longitude, and at least some environmental features.
    Missing optional features are imputed from training medians with a warning.

    ⚠️ All predictions are EXPERIMENTAL.
    """
    from src.risk.risk_engine import get_risk_engine

    features: dict[str, Any] = {
        "latitude": request.latitude,
        "longitude": request.longitude,
    }

    # Add optional features only if provided
    optional_fields = {
        "rainfall_1h_mm": request.rainfall_1h_mm,
        "rainfall_24h_mm": request.rainfall_24h_mm,
        "rainfall_72h_mm": request.rainfall_72h_mm,
        "slope_degrees": request.slope_degrees,
        "elevation_m": request.elevation_m,
        "vegetation_index": request.vegetation_index,
        "soil_moisture": request.soil_moisture,
    }
    for key, val in optional_fields.items():
        if val is not None:
            features[key] = val

    engine = get_risk_engine()
    result = engine.estimate_risk(
        features=features,
        location_name=f"({request.latitude:.4f}, {request.longitude:.4f})",
    )

    return PredictionResponse(
        probability=result.get("probability"),
        risk_category=result.get("risk_category", "Unknown"),
        method=result.get("method", "unknown"),
        is_experimental=result.get("is_experimental", True),
        data_quality=result.get("data_quality", {}),
        warnings=result.get("warnings", []),
        errors=result.get("errors", []),
        features_used=result.get("features_used", {}),
        location_name=result.get("location_name"),
        estimated_at=result.get("estimated_at"),
    )


def _dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe dicts (replacing NaN/NaT with None)."""
    clean_df = df.astype(object).where(pd.notna(df), None)
    return clean_df.to_dict(orient="records")


# ─────────────────────────────────────────────
# Observations
# ─────────────────────────────────────────────

@app.get("/api/v1/observations", tags=["Data"])
async def get_observations(
    limit: int = Query(100, ge=1, le=5000),
    include_synthetic: bool = True,
):
    """Retrieve historical landslide observations from the database.

    Args:
        limit: Maximum number of records to return (default: 100).
        include_synthetic: Include synthetic test data records (default: True).
    """
    from src.database.models import get_observations as db_get_observations

    df = db_get_observations(limit=limit, include_synthetic=include_synthetic)
    if df.empty:
        return {"observations": [], "count": 0, "includes_synthetic": include_synthetic}

    records = _dataframe_to_records(df)
    return {
        "observations": records,
        "count": len(records),
        "includes_synthetic": include_synthetic,
    }


# ─────────────────────────────────────────────
# Alerts
# ─────────────────────────────────────────────

@app.get("/api/v1/alerts", tags=["Alerts"])
async def get_alerts(
    include_acknowledged: bool = False,
    include_resolved: bool = True,
    status: str | None = None,
    risk_category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
):
    """Retrieve alerts from the database with optional lifecycle and severity filters."""
    from src.database.models import get_alerts as db_get_alerts

    df = db_get_alerts(
        include_acknowledged=include_acknowledged,
        include_resolved=include_resolved,
        status=status,
        risk_category=risk_category,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    notice = (
        "EXPERIMENTAL ALERTS — These are NOT official warnings and must "
        "NOT trigger evacuation or emergency response."
    )
    if df.empty:
        return {"alerts": [], "count": 0, "notice": notice}

    records = _dataframe_to_records(df)
    return {"alerts": records, "count": len(records), "notice": notice}


@app.post("/api/v1/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(alert_id: int, request: AcknowledgeRequest | None = None):
    """Acknowledge an alert by ID and record audit timestamp."""
    from src.database.models import acknowledge_alert as db_acknowledge

    note = request.note if request else None
    try:
        success = db_acknowledge(alert_id, note)
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
        return {
            "status": "acknowledged",
            "alert_id": alert_id,
            "acknowledged_at": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to acknowledge alert %d: %s", alert_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/alerts/{alert_id}/resolve", tags=["Alerts"])
async def resolve_alert_endpoint(alert_id: int, request: AcknowledgeRequest | None = None):
    """Mark an alert as resolved and record audit timestamp."""
    from src.database.models import resolve_alert as db_resolve

    note = request.note if request else None
    try:
        success = db_resolve(alert_id, note)
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
        return {
            "status": "resolved",
            "alert_id": alert_id,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve alert %d: %s", alert_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/alerts/audit", tags=["Alerts"])
async def get_alert_audit(alert_id: int | None = None, limit: int = 100):
    """Retrieve alert acknowledgement and resolution audit log."""
    from src.database.models import get_alert_audit_log

    df = get_alert_audit_log(alert_id=alert_id, limit=limit)
    records = _dataframe_to_records(df) if not df.empty else []
    return {"audit_records": records, "count": len(records)}


@app.get("/api/v1/alerts/notification-status", tags=["Alerts"])
async def get_alert_notification_status():
    """Return status of configured external notification channels (disabled by default)."""
    from src.alerts.alert_manager import get_alert_manager

    manager = get_alert_manager()
    return {"channels": manager.get_notification_status()}


# ─────────────────────────────────────────────
# Live Monitoring Endpoints (Phase 1)
# ─────────────────────────────────────────────

@app.get("/api/v1/monitoring/locations", tags=["Live Monitoring"])
async def list_monitoring_locations(active_only: bool = False):
    """List registered monitoring locations."""
    from src.database.models import get_locations

    df = get_locations(active_only=active_only)
    records = _dataframe_to_records(df) if not df.empty else []
    return {
        "locations": records,
        "count": len(records),
    }


@app.post("/api/v1/monitoring/locations", tags=["Live Monitoring"])
async def create_monitoring_location(request: MonitoringLocationCreate):
    """Register or update a monitored location (optionally resolving SRTM terrain)."""
    from src.monitoring import get_live_monitor

    monitor = get_live_monitor()
    try:
        record = monitor.register_location(
            name=request.name,
            latitude=request.latitude,
            longitude=request.longitude,
            country=request.country,
            state_region=request.state_region,
            elevation_m=request.elevation_m,
            slope_degrees=request.slope_degrees,
            fetch_terrain=request.fetch_terrain,
            is_active=request.is_active,
        )
        return {"status": "registered", "location": record}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Failed to register monitoring location: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/v1/monitoring/locations/{location_id}", tags=["Live Monitoring"])
async def toggle_monitoring_location(
    location_id: int,
    request: MonitoringLocationStatusUpdate,
):
    """Activate or pause a monitoring location."""
    from src.monitoring import get_live_monitor

    monitor = get_live_monitor()
    updated = monitor.set_location_active(location_id, request.is_active)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Location {location_id} not found.")
    return {
        "status": "updated",
        "location_id": location_id,
        "is_active": request.is_active,
    }


@app.post("/api/v1/monitoring/seed-defaults", tags=["Live Monitoring"])
async def seed_default_monitoring_stations():
    """Register default landslide-prone Indian monitoring coordinates."""
    from src.monitoring import get_live_monitor

    monitor = get_live_monitor()
    try:
        ids = monitor.seed_default_stations()
        return {"status": "seeded", "location_ids": ids, "count": len(ids)}
    except Exception as e:
        logger.error("Failed to seed default monitoring stations: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/monitoring/status", tags=["Live Monitoring"])
async def get_monitoring_status(active_only: bool = True):
    """Return latest DB-backed live risk snapshot across monitored stations."""
    from src.monitoring import get_live_monitor

    monitor = get_live_monitor()
    try:
        snapshot = monitor.get_live_snapshot(active_only=active_only)
        snapshot["notice"] = (
            "EXPERIMENTAL MONITORING — Risk scores use model-derived Open-Meteo reanalysis "
            "and experimental models; NOT official emergency warnings."
        )
        return snapshot
    except Exception as e:
        logger.error("Failed to fetch monitoring snapshot: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/monitoring/refresh", tags=["Live Monitoring"])
async def refresh_monitoring(request: MonitoringRefreshRequest | None = None):
    """Trigger live weather ingestion, risk inference, and alert evaluation."""
    from src.monitoring import get_live_monitor

    req = request or MonitoringRefreshRequest()
    monitor = get_live_monitor()
    try:
        if req.location_id is not None:
            station = monitor.refresh_location(
                req.location_id, generate_alerts=req.generate_alerts
            )
            return {
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
                "total_stations": 1,
                "stations": [station],
                "is_experimental": True,
            }
        return monitor.refresh_all(
            active_only=req.active_only, generate_alerts=req.generate_alerts
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Monitoring refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/monitoring/locations/{location_id}/history", tags=["Live Monitoring"])
async def get_monitoring_location_history(
    location_id: int,
    limit: int = Query(50, ge=1, le=500),
):
    """Retrieve stored environmental observations and prediction history for a station."""
    from src.monitoring import get_live_monitor

    monitor = get_live_monitor()
    try:
        history = monitor.get_station_history(location_id, limit=limit)
        return {
            "location": history["location"],
            "environmental_history": _dataframe_to_records(history["environmental_history"]),
            "prediction_history": _dataframe_to_records(history["prediction_history"]),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to load history for location %d: %s", location_id, e)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Geospatial GeoJSON Endpoints (Phase 2)
# ─────────────────────────────────────────────

@app.get("/api/v1/geospatial/risk-geojson", tags=["Geospatial"])
async def get_risk_geojson(active_only: bool = True):
    """Export monitored station risk estimates as an RFC 7946 WGS84 GeoJSON FeatureCollection."""
    from src.geospatial.spatial_analysis import records_to_geojson
    from src.monitoring import get_live_monitor

    monitor = get_live_monitor()
    snapshot = monitor.get_live_snapshot(active_only=active_only)
    geojson = records_to_geojson(snapshot.get("stations", []), record_kind="experimental_risk_estimate")
    geojson["crs_note"] = "WGS84 (EPSG:4326) longitude-latitude order per RFC 7946"
    geojson["notice"] = "EXPERIMENTAL RISK ESTIMATES — Not official emergency warnings."
    return geojson


# ─────────────────────────────────────────────
# Dynamic Rainfall Analysis Endpoints (Phase 3)
# ─────────────────────────────────────────────

class RainfallRiskEvaluateRequest(BaseModel):
    """Request body for evaluating dynamic rainfall impact on experimental risk."""

    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    location_id: int | None = None
    location_name: str | None = None
    is_manual: bool = False
    slope_degrees: float | None = Field(None, ge=0.0, le=90.0)
    elevation_m: float | None = Field(None, ge=-500.0, le=9000.0)
    rainfall_1h_mm: float | None = Field(None, ge=0.0, le=500.0)
    rainfall_24h_mm: float | None = Field(None, ge=0.0, le=2000.0)
    rainfall_48h_mm: float | None = Field(None, ge=0.0, le=3500.0)
    rainfall_72h_mm: float | None = Field(None, ge=0.0, le=5000.0)
    soil_moisture: float | None = Field(None, ge=0.0, le=1.0)
    observation_time: str | None = None
    persist: bool = True


@app.get("/api/v1/rainfall/analysis", tags=["Rainfall Analysis"])
async def get_rainfall_analysis(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    past_days: int = Query(3, ge=1, le=30),
    forecast_days: int = Query(3, ge=0, le=7),
    include_baseline: bool = True,
    bypass_cache: bool = False,
):
    """Fetch separated Open-Meteo current/historical/forecast rainfall and analytics."""
    from src.data_providers.weather_provider import WeatherProvider
    from src.monitoring.rainfall_analysis import compute_rainfall_metrics

    provider = WeatherProvider()
    weather = provider.fetch_separated_weather(
        latitude=latitude,
        longitude=longitude,
        past_days=past_days,
        forecast_days=forecast_days,
        include_archive_baseline=include_baseline,
        bypass_cache=bool(bypass_cache),
    )
    if weather is None:
        meta_err = provider.last_fetch_metadata.get("error")
        detail = (
            f"Open-Meteo weather API unavailable ({meta_err}); use manual input fallback."
            if meta_err
            else "Open-Meteo weather API is currently unavailable; use manual input fallback."
        )
        raise HTTPException(status_code=503, detail=detail)

    metrics = compute_rainfall_metrics(
        historical_hourly=weather.get("historical_hourly", {}),
        forecast_hourly=weather.get("forecast_hourly", {}),
        baseline_daily=weather.get("archive_baseline_daily", {}),
    )
    return {
        "latitude": weather["latitude"],
        "longitude": weather["longitude"],
        "timezone": weather.get("timezone"),
        "current": weather.get("current"),
        "historical_hourly": weather.get("historical_hourly"),
        "forecast_hourly": weather.get("forecast_hourly"),
        "daily_historical": weather.get("daily_historical"),
        "daily_forecast": weather.get("daily_forecast"),
        "metrics": metrics,
        "unit_validation": weather.get("unit_validation"),
        "provenance": weather.get("provenance"),
    }


@app.post("/api/v1/rainfall/evaluate-risk", tags=["Rainfall Analysis"])
async def evaluate_rainfall_risk(request: RainfallRiskEvaluateRequest):
    """Recalculate experimental landslide risk from rainfall updates or manual fallback."""
    from src.data_providers.weather_provider import WeatherProvider, parse_api_timestamp
    from src.database.models import get_location
    from src.monitoring.rainfall_analysis import (
        compute_rainfall_metrics,
        evaluate_rainfall_risk_change,
    )

    loc_name = request.location_name
    slope_val = request.slope_degrees
    elev_val = request.elevation_m
    if request.location_id is not None:
        loc_row = get_location(request.location_id)
        if loc_row is None:
            raise HTTPException(status_code=404, detail=f"Location {request.location_id} not found.")
        loc_name = loc_name or loc_row.get("name")
        if slope_val is None:
            slope_val = loc_row.get("slope_degrees")
        if elev_val is None:
            elev_val = loc_row.get("elevation_m")

    obs_time = parse_api_timestamp(request.observation_time) if request.observation_time else None

    if request.is_manual:
        features = {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "slope_degrees": slope_val,
            "elevation_m": elev_val,
            "rainfall_1h_mm": request.rainfall_1h_mm,
            "rainfall_24h_mm": request.rainfall_24h_mm,
            "rainfall_48h_mm": request.rainfall_48h_mm,
            "rainfall_72h_mm": request.rainfall_72h_mm,
            "soil_moisture": request.soil_moisture,
        }
        source_label = "User-Provided Data (Manual Fallback)"
        obs_time = obs_time or datetime.now(timezone.utc)
    else:
        provider = WeatherProvider()
        weather = provider.fetch_separated_weather(
            latitude=request.latitude,
            longitude=request.longitude,
            past_days=3,
            forecast_days=1,
            include_archive_baseline=False,
        )
        if weather is None:
            raise HTTPException(
                status_code=503,
                detail="Open-Meteo weather API unavailable. Submit with is_manual=true for fallback.",
            )
        metrics = compute_rainfall_metrics(
            historical_hourly=weather.get("historical_hourly", {}),
            forecast_hourly=weather.get("forecast_hourly", {}),
        )
        soil_series = [
            v for v in (weather.get("historical_hourly", {}).get("soil_moisture_0_to_7cm") or [])
            if v is not None
        ]
        latest_soil = soil_series[-1] if soil_series else request.soil_moisture
        latest_times = weather.get("historical_hourly", {}).get("time") or []
        if obs_time is None and latest_times:
            obs_time = parse_api_timestamp(latest_times[-1], weather.get("utc_offset_seconds", 0))
        features = {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "slope_degrees": slope_val,
            "elevation_m": elev_val if elev_val is not None else weather.get("elevation_m"),
            "rainfall_1h_mm": request.rainfall_1h_mm if request.rainfall_1h_mm is not None else metrics.get("rainfall_1h_mm"),
            "rainfall_24h_mm": request.rainfall_24h_mm if request.rainfall_24h_mm is not None else metrics.get("rainfall_24h_mm"),
            "rainfall_48h_mm": request.rainfall_48h_mm if request.rainfall_48h_mm is not None else metrics.get("rainfall_48h_mm"),
            "rainfall_72h_mm": request.rainfall_72h_mm if request.rainfall_72h_mm is not None else metrics.get("rainfall_72h_mm"),
            "soil_moisture": latest_soil,
        }
        is_cached = bool(weather.get("is_cached", False))
        source_label = (
            f"Open-Meteo Reanalysis (Cached at {weather.get('cached_at')})"
            if is_cached
            else "Open-Meteo Live Model-Derived Reanalysis"
        )

    return evaluate_rainfall_risk_change(
        features=features,
        observation_time=obs_time,
        location_id=request.location_id,
        location_name=loc_name,
        source_label=source_label,
        is_manual=request.is_manual,
        persist=request.persist,
    )


# ─────────────────────────────────────────────
# Explainable AI (SHAP & Feature Importance) Endpoints (Phase 4)
# ─────────────────────────────────────────────

class ExplainPredictionRequest(BaseModel):
    """Request payload for individual prediction SHAP explainability."""

    model_name: str = "best_model"
    slope_degrees: float | None = Field(None, ge=0.0, le=90.0)
    elevation_m: float | None = Field(None, ge=-500.0, le=9000.0)
    rainfall_1h_mm: float | None = Field(None, ge=0.0, le=500.0)
    rainfall_24h_mm: float | None = Field(None, ge=0.0, le=2000.0)
    rainfall_72h_mm: float | None = Field(None, ge=0.0, le=5000.0)
    vegetation_index: float | None = Field(None, ge=-1.0, le=1.0)
    soil_moisture: float | None = Field(None, ge=0.0, le=1.0)


@app.get("/api/v1/explain/global", tags=["Explainability"])
async def get_global_explainability(
    model_name: str = Query("best_model"),
    max_samples: int = Query(100, ge=10, le=500),
    include_beeswarm: bool = False,
):
    """Compute global feature importance, permutation importance, and SHAP summary."""
    from src.models.explainability import EXPLAINABILITY_CONCEPTS, explain_global

    res = explain_global(model_name=model_name, max_samples=max_samples)
    if not res["available"] and res.get("errors"):
        raise HTTPException(status_code=503, detail="; ".join(res["errors"]))
    payload = {
        "model_name": res["model_name"],
        "model_version": res["model_version"],
        "model_metadata": res.get("model_metadata"),
        "shap_available": res["shap_available"],
        "shap_error": res.get("shap_error"),
        "explainer_type": res.get("explainer_type"),
        "shap_unit": res.get("shap_unit"),
        "sample_count": res["sample_count"],
        "feature_importance": res["feature_importance"],
        "permutation_importance": res["permutation_importance"],
        "shap_summary": res["shap_summary"],
        "method_comparison": res.get("method_comparison", []),
        "concepts": EXPLAINABILITY_CONCEPTS,
        "warnings": res["warnings"],
        "from_cache": res["from_cache"],
        "scientific_disclaimer": res["scientific_disclaimer"],
    }
    if include_beeswarm:
        payload["beeswarm_records"] = res["beeswarm_records"]
    return payload


@app.post("/api/v1/explain/predict", tags=["Explainability"])
async def explain_single_prediction(request: ExplainPredictionRequest):
    """Return individual prediction, input values, SHAP contributions, and plain-text insights."""
    from src.models.explainability import explain_prediction

    features = {
        "slope_degrees": request.slope_degrees,
        "elevation_m": request.elevation_m,
        "rainfall_1h_mm": request.rainfall_1h_mm,
        "rainfall_24h_mm": request.rainfall_24h_mm,
        "rainfall_72h_mm": request.rainfall_72h_mm,
        "vegetation_index": request.vegetation_index,
        "soil_moisture": request.soil_moisture,
    }
    res = explain_prediction(features=features, model_name=request.model_name)
    if not res["available"] and res.get("errors"):
        raise HTTPException(status_code=503, detail="; ".join(res["errors"]))
    return res


# ─────────────────────────────────────────────
# India State / District Real-Time Regional Endpoints
# ─────────────────────────────────────────────

class IndiaMonitorRequest(BaseModel):
    """Request payload for running an India State/District regional assessment."""

    state: str = Field(..., min_length=2, description="Indian State or Union Territory name")
    district: str = Field(..., min_length=2, description="District name inside the selected State/UT")
    include_terrain: bool = True
    include_shap: bool = True
    persist: bool = True
    generate_experimental_alert: bool = False


@app.get("/api/v1/india/states", tags=["India Regional"])
async def list_india_states_api(query: str | None = Query(None, description="Optional case-insensitive state filter")):
    """List available Indian States and Union Territories from cached geoBoundaries ADM1."""
    from src.data_providers.india_boundaries import IndiaBoundaryProvider

    provider = IndiaBoundaryProvider()
    states = provider.list_states(query=query)
    status = provider.get_boundary_status()
    return {
        "country": "India",
        "count": len(states),
        "boundary_status": status,
        "states": states,
    }


@app.get("/api/v1/india/districts", tags=["India Regional"])
async def list_india_districts_api(
    state: str | None = Query(None, description="Parent State/UT name"),
    query: str | None = Query(None, description="Optional case-insensitive district search"),
):
    """List Indian Districts filtered by State/UT and optional search query."""
    from src.data_providers.india_boundaries import IndiaBoundaryProvider

    provider = IndiaBoundaryProvider()
    districts = provider.list_districts(state_name=state, query=query)
    return {
        "country": "India",
        "state": state,
        "count": len(districts),
        "districts": districts,
    }


@app.post("/api/v1/india/monitor", tags=["India Regional"])
async def run_india_regional_monitor(request: IndiaMonitorRequest):
    """Execute end-to-end State -> District regional monitoring & experimental risk assessment."""
    from src.monitoring.india_monitor import IndiaRegionalMonitor

    monitor = IndiaRegionalMonitor()
    try:
        result = monitor.run_regional_assessment(
            state_name=request.state,
            district_name=request.district,
            include_terrain=request.include_terrain,
            include_shap=request.include_shap,
            persist=request.persist,
            generate_experimental_alert=request.generate_experimental_alert,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Return JSON-safe response (omit heavy GeoJSON polygons unless requested separately)
    hist = result.get("historical") or {}
    hist_df = hist.get("observations_df")
    hist_records = _dataframe_to_records(hist_df.head(100)) if isinstance(hist_df, pd.DataFrame) and not hist_df.empty else []

    return {
        "country": result["country"],
        "state": result["state"],
        "district": result["district"],
        "district_meta": result["district_meta"],
        "location_id": result["location_id"],
        "prediction_id": result["prediction_id"],
        "weather": result["weather"],
        "imd": result["imd"],
        "sachet": {k: v for k, v in (result.get("sachet") or {}).items() if k != "all_alerts"},
        "terrain": result["terrain"],
        "historical_summary": {
            "total_count": hist.get("total_count", 0),
            "authentic_count": hist.get("authentic_count", 0),
            "synthetic_count": hist.get("synthetic_count", 0),
            "filter_method": hist.get("filter_method"),
            "sample_records": hist_records,
        },
        "features_used": result["features_used"],
        "risk_result": result["risk_result"],
        "shap_explanation": result["shap_explanation"],
        "experimental_alert": result["experimental_alert"],
        "provider_status": result["provider_status"],
    }


@app.get("/api/v1/india/providers/status", tags=["India Regional"])
async def get_india_provider_status():
    """Return latest persisted external data provider health, caching, and timestamp status."""
    from src.database.models import get_provider_statuses

    statuses = get_provider_statuses()
    return {
        "count": len(statuses),
        "providers": statuses,
    }


# ─────────────────────────────────────────────
# Environmental Intelligence, Risk Forecasting & Drift Endpoints (Phase 6)
# ─────────────────────────────────────────────

class PredictionOutcomeRequest(BaseModel):
    """Ground-truth outcome record used for empirical performance drift tracking."""

    actual_label: int = Field(..., ge=0, le=1)
    prediction_id: int | None = None
    location_id: int | None = None
    outcome_source: str = "Operator Verification"
    observed_at: str | None = None
    notes: str | None = None


@app.get("/api/v1/analytics/rainfall-intelligence", tags=["Phase 6 Analytics"])
async def get_rainfall_intelligence(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    past_days: int = Query(7, ge=1, le=30),
    forecast_days: int = Query(3, ge=0, le=7),
    bypass_cache: bool = False,
):
    """Compute statistical rainfall anomaly detection and daily/weekly/monthly/seasonal trends."""
    from src.data_providers.weather_provider import WeatherProvider
    from src.monitoring.environmental_analytics import (
        analyze_rainfall_trends,
        detect_rainfall_anomalies,
    )
    from src.monitoring.rainfall_analysis import compute_rainfall_metrics

    provider = WeatherProvider()
    weather = provider.fetch_separated_weather(
        latitude=latitude,
        longitude=longitude,
        past_days=past_days,
        forecast_days=forecast_days,
        include_archive_baseline=True,
        bypass_cache=bool(bypass_cache),
    )
    if weather is None:
        raise HTTPException(status_code=503, detail="Open-Meteo weather API unavailable for rainfall intelligence.")

    metrics = compute_rainfall_metrics(
        historical_hourly=weather.get("historical_hourly", {}),
        forecast_hourly=weather.get("forecast_hourly", {}),
        baseline_daily=weather.get("archive_baseline_daily", {}),
    )
    baseline_daily = weather.get("archive_baseline_daily") or {}
    anomaly = detect_rainfall_anomalies(
        recent_24h_mm=metrics.get("rainfall_24h_mm"),
        baseline_daily=baseline_daily,
    )
    trends = analyze_rainfall_trends(
        baseline_daily=baseline_daily,
        daily_historical=weather.get("daily_historical"),
        recent_24h_mm=metrics.get("rainfall_24h_mm"),
    )
    return {
        "latitude": weather["latitude"],
        "longitude": weather["longitude"],
        "rainfall_metrics": metrics,
        "anomaly_detection": anomaly,
        "trend_analysis": trends,
        "provenance": weather.get("provenance"),
    }


@app.get("/api/v1/forecast/risk-trend", tags=["Phase 6 Analytics"])
async def get_forecast_risk_trend(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    slope_degrees: float | None = Query(None, ge=0.0, le=90.0),
    elevation_m: float | None = Query(None, ge=-500.0, le=9000.0),
    location_id: int | None = None,
    bypass_cache: bool = False,
):
    """Compute experimental 24h / 48h / 72h risk trends from Open-Meteo forecast inputs."""
    from src.data_providers.weather_provider import WeatherProvider
    from src.database.models import get_location
    from src.monitoring.environmental_analytics import forecast_risk_trend

    slope_val = slope_degrees
    elev_val = elevation_m
    if location_id is not None:
        loc_row = get_location(location_id)
        if loc_row is None:
            raise HTTPException(status_code=404, detail=f"Location {location_id} not found.")
        if slope_val is None:
            slope_val = loc_row.get("slope_degrees")
        if elev_val is None:
            elev_val = loc_row.get("elevation_m")

    provider = WeatherProvider()
    weather = provider.fetch_separated_weather(
        latitude=latitude,
        longitude=longitude,
        past_days=3,
        forecast_days=4,
        include_archive_baseline=False,
        bypass_cache=bool(bypass_cache),
    )
    if weather is None:
        raise HTTPException(status_code=503, detail="Open-Meteo weather forecast unavailable.")

    return forecast_risk_trend(
        weather_payload=weather,
        terrain_features={"slope_degrees": slope_val, "elevation_m": elev_val},
    )


@app.get("/api/v1/model/drift", tags=["Phase 6 Analytics"])
async def get_model_drift_monitoring(limit: int = Query(300, ge=10, le=2000)):
    """Return evidence-gated feature, missingness, prediction, and outcome drift metrics."""
    from src.monitoring.drift_monitor import compute_model_drift_report

    return compute_model_drift_report(limit=limit)


@app.post("/api/v1/model/outcomes", tags=["Phase 6 Analytics"])
async def record_model_prediction_outcome(request: PredictionOutcomeRequest):
    """Record a verified ground-truth landslide outcome (0 or 1) for drift checks."""
    from src.database.models import insert_prediction_outcome

    try:
        outcome_id = insert_prediction_outcome(
            actual_label=request.actual_label,
            observed_at=request.observed_at,
            prediction_id=request.prediction_id,
            location_id=request.location_id,
            outcome_source=request.outcome_source,
            notes=request.notes,
        )
        return {"status": "recorded", "outcome_id": outcome_id, "actual_label": request.actual_label}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ─────────────────────────────────────────────
# Advanced Geospatial, Terrain & Satellite Endpoints (Phase 7)
# ─────────────────────────────────────────────

class SatelliteNdviRequest(BaseModel):
    """Request schema for spectral NDVI calculation requiring Red and NIR bands."""

    red_band: list[list[float]] = Field(..., description="Sentinel-2 Band 4 Red reflectance (~665nm) 2D array")
    nir_band: list[list[float]] = Field(..., description="Sentinel-2 Band 8 NIR reflectance (~842nm) 2D array")
    cloud_mask: list[list[int | bool]] | None = Field(None, description="Optional SCL categories or cloud boolean mask")
    observation_date: str | None = Field(None, description="ISO-8601 acquisition timestamp")
    resolution_m: float = Field(10.0, ge=1.0, le=1000.0, description="Pixel Ground Sample Distance in metres")


class SatelliteChangeRequest(BaseModel):
    """Request schema for two-date multi-temporal satellite change detection."""

    pre_red_band: list[list[float]] = Field(..., description="Pre-event (t1) Band 4 Red reflectance array")
    pre_nir_band: list[list[float]] = Field(..., description="Pre-event (t1) Band 8 NIR reflectance array")
    pre_date: str = Field(..., description="Pre-event acquisition date (e.g., 2024-05-18)")
    post_red_band: list[list[float]] = Field(..., description="Post-event (t2) Band 4 Red reflectance array")
    post_nir_band: list[list[float]] = Field(..., description="Post-event (t2) Band 8 NIR reflectance array")
    post_date: str = Field(..., description="Post-event acquisition date (e.g., 2024-08-06)")
    pre_cloud_mask: list[list[int | bool]] | None = Field(None, description="Pre-event cloud mask")
    post_cloud_mask: list[list[int | bool]] | None = Field(None, description="Post-event cloud mask")
    resolution_m: float = Field(10.0, ge=1.0, le=1000.0, description="Pixel GSD in metres")


class TerrainDerivativesRequest(BaseModel):
    """Request schema for DEM terrain derivatives computation."""

    elevation_grid: list[list[float]] = Field(..., description="2D digital elevation array in metres")
    pixel_size_x_m: float = Field(30.0, gt=0, description="Cell width in metres")
    pixel_size_y_m: float = Field(30.0, gt=0, description="Cell height in metres")
    specific_catchment_area_m: list[list[float]] | None = Field(None, description="Optional upslope contributing area for TWI")


# ─────────────────────────────────────────────
# Phase 8 AI Assistant & Computer Vision Schemas
# ─────────────────────────────────────────────

class AssistantQueryRequest(BaseModel):
    """Request schema for LandslideGuard AI Assistant context-aware query."""

    query: str = Field(..., min_length=1, description="Natural language question text")
    district: str | None = Field(None, description="Optional target Indian district context")
    state: str | None = Field(None, description="Optional target Indian state/UT context")
    location_id: int | None = Field(None, description="Optional monitored station ID context")
    language: str = Field("en", description="Language preference ('en', 'hi', 'pa')")


class CVAnalysisRequest(BaseModel):
    """Request schema for experimental computer vision terrain image analysis."""

    image_base64: str = Field(..., min_length=16, description="Base64-encoded image bytes (PNG, JPEG, WEBP)")
    filename: str = Field("image.png", description="Uploaded image filename")


# ─────────────────────────────────────────────
# Phase 9 Disaster Response & Community Schemas
# ─────────────────────────────────────────────

class RouteRequest(BaseModel):
    """Request schema for emergency route estimation."""

    start_lat: float = Field(..., ge=-90.0, le=90.0, description="Start latitude")
    start_lon: float = Field(..., ge=-180.0, le=180.0, description="Start longitude")
    end_lat: float = Field(..., ge=-90.0, le=90.0, description="Destination latitude")
    end_lon: float = Field(..., ge=-180.0, le=180.0, description="Destination longitude")
    facility_name: str | None = Field(None, description="Name of target facility")
    facility_category: str | None = Field(None, description="Type of facility")


class CommunityReportRequest(BaseModel):
    """Request schema for crowd-sourced community hazard reporting."""

    location_name: str = Field(..., min_length=2, description="Location or landmark name")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude coordinate")
    incident_date: str = Field(..., description="Date/time of observed hazard (ISO or YYYY-MM-DD)")
    category: str = Field(..., description="Hazard category (Rockfall, Mudflow, Tension Cracks, etc.)")
    description: str = Field(..., min_length=10, description="Detailed description of ground disturbance")
    reporter_name: str | None = Field(None, description="Optional reporter name")
    reporter_contact: str | None = Field(None, description="Optional reporter phone or email")
    state_name: str | None = Field(None, description="State / Union Territory")
    district_name: str | None = Field(None, description="District name")
    severity: str = Field("moderate", pattern="^(low|moderate|high|critical)$", description="Severity tier")
    photo_base64: str | None = Field(None, description="Optional base64-encoded photo")


class ReportReviewRequest(BaseModel):
    """Request schema for administrative review of community reports."""

    status: str = Field(..., pattern="^(submitted|under_review|verified|rejected)$", description="Target status")
    review_notes: str | None = Field(None, description="Administrative audit notes")
    reviewed_by: str = Field("admin", description="Admin username or ID")


@app.get("/api/v1/geospatial/hotspots", tags=["Phase 7 Geospatial & Satellite"])
async def get_geospatial_hotspots(
    eps_km: float = Query(25.0, ge=1.0, le=500.0, description="DBSCAN neighborhood radius in km"),
    min_samples: int = Query(3, ge=2, le=50, description="DBSCAN minimum points per cluster"),
    authentic_only: bool = Query(False, description="Filter out synthetic development records"),
    method: str = Query("both", pattern="^(dbscan|kde|both)$", description="Clustering method"),
):
    """Detect spatial landslide clusters and continuous density hotspots from historical observations."""
    from src.database.models import get_observations
    from src.geospatial.hotspot_detection import detect_landslide_hotspots

    auth_val = bool(authentic_only.default if hasattr(authentic_only, "default") else authentic_only)
    eps_val = float(eps_km.default if hasattr(eps_km, "default") else eps_km)
    min_val = int(min_samples.default if hasattr(min_samples, "default") else min_samples)
    method_val = str(method.default if hasattr(method, "default") else method)

    obs_df = get_observations(limit=2000, include_synthetic=not auth_val)
    if obs_df.empty:
        pq_path = Path(get_setting("paths", "data_processed") or "data/processed") / "dataset.parquet"
        if pq_path.exists():
            obs_df = pd.read_parquet(pq_path)

    return detect_landslide_hotspots(
        obs_df,
        method=method_val,
        eps_km=eps_val,
        min_samples=min_val,
        authentic_only=auth_val,
    )


@app.post("/api/v1/satellite/ndvi", tags=["Phase 7 Geospatial & Satellite"])
async def evaluate_satellite_ndvi(request: SatelliteNdviRequest):
    """Compute Normalized Difference Vegetation Index (NDVI) requiring Red and NIR bands."""
    from src.geospatial.satellite_monitor import compute_ndvi

    try:
        res = compute_ndvi(
            red_band=np.array(request.red_band),
            nir_band=np.array(request.nir_band),
            cloud_mask=np.array(request.cloud_mask) if request.cloud_mask is not None else None,
            observation_date=request.observation_date,
            resolution_m=request.resolution_m,
        )
        return {
            "available": res["available"],
            "sensor": res["sensor"],
            "observation_date": res["observation_date"],
            "resolution_m": res["resolution_m"],
            "crs": res["crs"],
            "total_pixel_count": res["total_pixel_count"],
            "valid_pixel_count": res["valid_pixel_count"],
            "cloud_cover_pct": res["cloud_cover_pct"],
            "statistics": res["statistics"],
            "vegetation_classes": res["vegetation_classes"],
            "disclaimer": res["disclaimer"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/satellite/change-detection", tags=["Phase 7 Geospatial & Satellite"])
async def evaluate_satellite_change_detection(request: SatelliteChangeRequest):
    """Detect multi-temporal satellite vegetation changes (dNDVI) between two acquisition dates."""
    from src.geospatial.satellite_monitor import compute_ndvi, detect_satellite_changes

    try:
        pre_scene = compute_ndvi(
            red_band=np.array(request.pre_red_band),
            nir_band=np.array(request.pre_nir_band),
            cloud_mask=np.array(request.pre_cloud_mask) if request.pre_cloud_mask is not None else None,
            observation_date=request.pre_date,
            resolution_m=request.resolution_m,
        )
        post_scene = compute_ndvi(
            red_band=np.array(request.post_red_band),
            nir_band=np.array(request.post_nir_band),
            cloud_mask=np.array(request.post_cloud_mask) if request.post_cloud_mask is not None else None,
            observation_date=request.post_date,
            resolution_m=request.resolution_m,
        )
        change_report = detect_satellite_changes(pre_scene, post_scene)
        return {
            "available": change_report["available"],
            "temporal_baseline": change_report["temporal_baseline"],
            "resolution_m": change_report["resolution_m"],
            "total_pixels": change_report["total_pixels"],
            "valid_evaluated_pixels": change_report["valid_evaluated_pixels"],
            "cloud_obscured_pct": change_report["cloud_obscured_pct"],
            "analyzed_surface_area_km2": change_report["analyzed_surface_area_km2"],
            "candidate_disturbance_area_km2": change_report["candidate_disturbance_area_km2"],
            "d_ndvi_statistics": change_report["d_ndvi_statistics"],
            "change_classification": change_report["change_classification"],
            "limitations": change_report["limitations"],
            "verification_indicator_label": change_report["verification_indicator_label"],
            "disclaimer": change_report["disclaimer"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/terrain/derivatives", tags=["Phase 7 Geospatial & Satellite"])
async def evaluate_terrain_derivatives(request: TerrainDerivativesRequest):
    """Evaluate DEM terrain derivatives (Elevation, Slope, Aspect, Roughness, Curvature, and TWI)."""
    from src.geospatial.terrain import compute_slope_and_aspect

    try:
        elev_arr = np.array(request.elevation_grid, dtype=float)
        sca_arr = (
            np.array(request.specific_catchment_area_m, dtype=float)
            if request.specific_catchment_area_m is not None
            else None
        )
        stats = compute_slope_and_aspect(
            elev_arr,
            pixel_size_x_m=request.pixel_size_x_m,
            pixel_size_y_m=request.pixel_size_y_m,
            specific_catchment_area_m=sca_arr,
        )
        return {
            "valid_cells": stats["valid_cells"],
            "nodata_cells": stats["nodata_cells"],
            "mean_elevation_m": stats["mean_elevation_m"],
            "elevation_relief_m": stats["elevation_relief_m"],
            "elevation_std_m": stats["elevation_std_m"],
            "mean_slope_deg": stats["mean_slope_deg"],
            "max_slope_deg": stats["max_slope_deg"],
            "mean_aspect_deg": stats["mean_aspect_deg"],
            "mean_tri_m": stats["mean_tri_m"],
            "max_tri_m": stats["max_tri_m"],
            "mean_curvature": stats["mean_curvature"],
            "twi_status": stats["twi_status"],
            "mean_twi": stats["mean_twi"],
            "max_twi": stats["max_twi"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ─────────────────────────────────────────────
# Phase 8 AI Assistant & Computer Vision Endpoints
# ─────────────────────────────────────────────

@app.post("/api/v1/assistant/query", tags=["Phase 8 AI Assistant & Computer Vision"])
async def query_assistant(request: AssistantQueryRequest):
    """Context-aware QA assistant querying authentic database, model metadata, and verified sources."""
    from src.assistant.knowledge_engine import get_knowledge_engine

    try:
        engine = get_knowledge_engine()
        ctx = {
            "district": request.district,
            "state": request.state,
            "location_id": request.location_id,
            "language": request.language,
        }
        res = engine.answer_query(request.query, context=ctx)
        return {
            "query": request.query,
            "answer": res["answer"],
            "intent": res["intent"],
            "retrieved_facts": res.get("retrieved_facts", {}),
            "sources": res.get("sources", []),
            "is_experimental": res.get("is_experimental", True),
            "disclaimer": res.get("disclaimer", ""),
        }
    except Exception as exc:
        logger.error("Assistant query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Assistant processing error: {str(exc)}")


@app.post("/api/v1/cv/analyze-image", tags=["Phase 8 AI Assistant & Computer Vision"])
async def analyze_terrain_image(request: CVAnalysisRequest):
    """Experimental computer vision analysis for tension cracks, soil exposure, and slope deformation."""
    import base64
    from src.geospatial.terrain_cv import get_cv_analyzer

    try:
        raw_b64 = request.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(raw_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 payload. Could not decode image bytes.")

        analyzer = get_cv_analyzer()
        res = analyzer.analyze(img_bytes)

        if not res.get("available"):
            raise HTTPException(status_code=400, detail=res.get("error", "Image analysis failed."))

        return res
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("CV analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Computer vision analysis error: {str(exc)}")


@app.get("/api/v1/assistant/safety-guidance", tags=["Phase 8 AI Assistant & Computer Vision"])
async def get_safety_guidance_endpoint(
    language: str = Query("en", description="Language preference ('en', 'hi', 'pa')")
):
    """Retrieve verified multilingual landslide safety protocols in English, Hindi, or Punjabi."""
    from src.assistant.safety_guidance import get_safety_guidance, format_safety_markdown

    try:
        guidance = get_safety_guidance(language)
        md = format_safety_markdown(guidance)
        return {
            "language": guidance["language"],
            "title": guidance["title"],
            "authority": guidance["authority"],
            "summary": guidance["summary"],
            "before": guidance["before"],
            "during": guidance["during"],
            "after": guidance["after"],
            "emergency_contacts": guidance["emergency_contacts"],
            "formatted_markdown": md,
            "disclaimer": guidance["disclaimer"],
        }
    except Exception as exc:
        logger.error("Failed to retrieve safety guidance: %s", exc)
        raise HTTPException(status_code=500, detail=f"Safety guidance error: {str(exc)}")


# ─────────────────────────────────────────────
# Phase 9 Disaster Response & Community Endpoints
# ─────────────────────────────────────────────

@app.get("/api/v1/response/facilities", tags=["Phase 9 Disaster Response & Community"])
async def get_nearby_emergency_facilities(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Center latitude"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Center longitude"),
    radius_km: float = Query(35.0, ge=1.0, le=200.0, description="Search radius in kilometres"),
):
    """Retrieve nearby emergency facilities (hospitals, police, fire, shelters) via OpenStreetMap or regional fallback."""
    from src.geospatial.facilities import fetch_nearby_facilities

    try:
        return fetch_nearby_facilities(latitude=latitude, longitude=longitude, radius_km=radius_km)
    except Exception as exc:
        logger.error("Facilities query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Emergency facilities query error: {str(exc)}")


# Alias for backwards compatibility & clean testing imports
get_emergency_facilities = get_nearby_emergency_facilities


@app.post("/api/v1/response/route", tags=["Phase 9 Disaster Response & Community"])
async def get_emergency_route(request: RouteRequest):
    """Calculate driving route between monitoring station and facility using OSRM or geodesic fallback."""
    from src.geospatial.routing import calculate_emergency_route

    try:
        return calculate_emergency_route(
            start_lat=request.start_lat,
            start_lon=request.start_lon,
            end_lat=request.end_lat,
            end_lon=request.end_lon,
            facility_name=request.facility_name,
            facility_category=request.facility_category,
        )
    except Exception as exc:
        logger.error("Route calculation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Route calculation error: {str(exc)}")


@app.post("/api/v1/community/reports", tags=["Phase 9 Disaster Response & Community"])
async def post_community_report(request: CommunityReportRequest):
    """Submit a crowd-sourced community landslide hazard report."""
    import base64
    from src.alerts.community_reporting import submit_community_report

    try:
        photo_bytes: bytes | None = None
        if request.photo_base64:
            raw_b64 = request.photo_base64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                photo_bytes = base64.b64decode(raw_b64)
            except Exception:
                raise HTTPException(status_code=400, detail="Could not decode photo_base64.")

        res = submit_community_report(
            location_name=request.location_name,
            latitude=request.latitude,
            longitude=request.longitude,
            incident_date=request.incident_date,
            category=request.category,
            description=request.description,
            reporter_name=request.reporter_name,
            reporter_contact=request.reporter_contact,
            state_name=request.state_name,
            district_name=request.district_name,
            severity=request.severity,
            photo_bytes=photo_bytes,
        )
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("error", "Failed to submit report."))
        return res
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to submit community report: %s", exc)
        raise HTTPException(status_code=500, detail=f"Submission error: {str(exc)}")


@app.get("/api/v1/community/reports", tags=["Phase 9 Disaster Response & Community"])
async def list_community_reports(
    status: str | None = Query(None, description="Optional status filter ('submitted', 'under_review', 'verified', 'rejected')"),
    limit: int = Query(100, ge=1, le=500, description="Max reports to retrieve"),
):
    """List crowd-sourced community hazard reports."""
    from src.database.models import get_community_reports

    try:
        reports = get_community_reports(status=status, limit=limit)
        return {
            "total_reports": len(reports),
            "status_filter": status,
            "reports": reports,
            "segregation_notice": "Community reports are crowd-sourced and strictly segregated from authentic historical catalogs.",
        }
    except Exception as exc:
        logger.error("Failed to list community reports: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to query community reports: {str(exc)}")


@app.patch("/api/v1/admin/community/reports/{report_id}", tags=["Phase 9 Disaster Response & Community"])
async def admin_review_community_report(
    report_id: int,
    request: ReportReviewRequest,
    x_admin_key: str | None = Header(None, alias="X-Admin-Key"),
):
    """Administratively review and update status of a community report (requires X-Admin-Key)."""
    from src.alerts.community_reporting import review_community_report
    from src.utils.auth import verify_admin_key

    if not verify_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-Admin-Key header.")

    res = review_community_report(
        report_id=report_id,
        new_status=request.status,
        review_notes=request.review_notes,
        reviewed_by=request.reviewed_by,
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Update failed."))
    return res


@app.get("/api/v1/reports/pdf/{location_id}", tags=["Phase 9 Disaster Response & Community"])
async def download_location_pdf_report(location_id: int):
    """Generate and download a comprehensive PDF hazard assessment report for a monitored location."""
    from src.database.models import get_location_by_id, get_observations
    from src.monitoring.live_monitor import get_live_monitor
    from src.geospatial.facilities import fetch_nearby_facilities
    from src.utils.pdf_generator import generate_assessment_pdf_report

    loc = get_location_by_id(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location #{location_id} not found in database.")

    lm = get_live_monitor()
    snap = lm.get_live_snapshot(active_only=False)
    stn_snap = next((s for s in snap.get("stations", []) if s.get("location_id") == location_id), None)

    weather_data = stn_snap.get("latest_weather", {}) if stn_snap else {}
    risk_data = stn_snap.get("latest_risk", {}) if stn_snap else {}

    lat = loc["latitude"]
    lon = loc["longitude"]
    fac_res = fetch_nearby_facilities(lat, lon, radius_km=25.0, max_results=4)
    obs = get_observations(include_synthetic=False, limit=5)

    pdf_bytes = generate_assessment_pdf_report(
        location_profile=loc,
        weather_telemetry=weather_data,
        risk_assessment=risk_data,
        historical_events=obs.to_dict(orient="records") if not obs.empty else [],
        nearby_facilities=fac_res.get("facilities", []),
        is_synthetic_benchmark=False,
    )

    filename = f"LandslideGuard_Assessment_Report_Loc_{location_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/v1/monitoring/offline-status", tags=["Phase 9 Disaster Response & Community"])
async def get_offline_monitoring_status():
    """Retrieve offline cache synchronization health, sync age, and freshness indicators."""
    from src.monitoring.offline_cache import get_offline_cache

    try:
        cache = get_offline_cache()
        return cache.get_offline_status()
    except Exception as exc:
        logger.error("Failed to query offline cache status: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))





