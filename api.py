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

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
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
    limit: int = Query(50, ge=1, le=1000),
):
    """Retrieve alerts from the database.

    Args:
        include_acknowledged: Include already-acknowledged alerts.
        limit: Maximum number of alerts to return.
    """
    from src.database.models import get_alerts as db_get_alerts

    df = db_get_alerts(include_acknowledged=include_acknowledged, limit=limit)
    if df.empty:
        return {
            "alerts": [],
            "count": 0,
            "notice": (
                "EXPERIMENTAL ALERTS — These are NOT official warnings and must "
                "NOT trigger evacuation or emergency response."
            ),
        }

    records = _dataframe_to_records(df)
    return {
        "alerts": records,
        "count": len(records),
        "notice": (
            "EXPERIMENTAL ALERTS — These are NOT official warnings and must "
            "NOT trigger evacuation or emergency response."
        ),
    }


@app.post("/api/v1/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(alert_id: int, request: AcknowledgeRequest | None = None):
    """Acknowledge an alert by ID.

    Args:
        alert_id: Alert record ID.
        request: Optional acknowledgement note.
    """
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

