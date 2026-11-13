"""
Final-phase end-to-end integration tests verifying all five LandslideGuard phases work together:
1. Load supported dataset
2. Validate dataset & coordinates
3. Preprocess & engineer features
4. Load trained model (with consistent model version/metadata)
5. Retrieve environmental data (Open-Meteo schema, deterministic offline mock)
6. Compute dynamic experimental risk estimate
7. Visualize location on Folium interactive risk map
8. Explain model output with SHAP + human-readable statements
9. Evaluate & trigger experimental alert under configured thresholds/cooldown
10. Persist prediction, alert, and audit history in SQLite
11. Retrieve & verify persisted assessment and alert lifecycle history
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import folium
import pytest

import src.database.db as db_mod
import src.database.models as models_mod
from src.alerts.alert_manager import AlertManager
from src.data_processing.feature_engineering import engineer_features
from src.data_processing.preprocessing import (
    load_processed,
    preprocess,
    validate_columns,
)
from src.data_providers.weather_provider import WeatherProvider
from src.geospatial.maps import build_landslide_map
from src.models.explainability import explain_prediction
from src.models.model_registry import get_model_info, load_model
from src.models.predict import load_feature_columns
from src.risk.risk_engine import RiskEngine
from src.utils.pdf_generator import generate_assessment_pdf_report
from src.utils.validation import validate_feature_dict


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch):
    """Point db_context at a clean temporary SQLite file."""
    db_file = tmp_path / "integration_test.db"

    @contextmanager
    def _tmp_ctx(_db_path=None):
        conn = db_mod.get_connection(db_file)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(db_mod, "db_context", _tmp_ctx)
    monkeypatch.setattr(models_mod, "db_context", _tmp_ctx)
    db_mod.initialize_database(db_file)
    return db_file


def test_complete_11_step_end_to_end_workflow(isolated_db: Path):
    """Execute the full 11-step integrated LandslideGuard workflow on an isolated DB."""
    # 1. Load supported dataset
    dataset_path = Path("data/processed/dataset.parquet")
    assert dataset_path.exists(), "Processed dataset missing."
    df = load_processed(dataset_path)
    assert len(df) == 2000

    # 2. Validate dataset schema
    required_cols = ["latitude", "longitude", "slope_degrees", "elevation_m", "rainfall_24h_mm"]
    missing = validate_columns(df, required_cols)
    assert missing == [], f"Unexpected missing required columns: {missing}"

    # 3. Preprocess & engineer environmental features
    cleaned_df, artefacts = preprocess(df.head(50), is_training=True)
    engineered_df, feat_stats = engineer_features(cleaned_df)
    assert len(engineered_df) == 50
    assert "categorical_encoders" in artefacts
    assert "rainfall_24h_mean" in feat_stats
    assert "slope_rainfall_interaction" in engineered_df.columns

    # 4. Load supported ML model & inspect consistent metadata
    model = load_model("best_model")
    model_info = get_model_info("best_model")
    feature_cols = load_feature_columns()
    assert model is not None
    assert model_info is not None
    assert feature_cols == model_info["metadata"]["feature_columns"]

    # 5. Retrieve environmental data (deterministic offline Open-Meteo response)
    lat, lon = 10.0889, 77.0595  # Munnar monitoring coordinates
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
    fake_payload = {
        "latitude": lat,
        "longitude": lon,
        "current": {
            "time": now_iso,
            "temperature_2m": 21.5,
            "relative_humidity_2m": 96.0,
            "wind_speed_10m": 12.4,
            "precipitation": 34.0,
            "rain": 34.0,
        },
        "hourly": {
            "time": [now_iso],
            "precipitation": [34.0],
            "soil_moisture_0_to_7cm": [0.78],
        },
        "daily": {
            "time": [now_iso[:10]],
            "precipitation_sum": [185.0],
        },
    }
    provider = WeatherProvider()
    with patch.object(provider, "_get_cached", return_value=fake_payload):
        weather = provider.fetch(lat, lon)
    assert weather is not None
    assert weather["current"]["precipitation_mm"] == 34.0

    # 6. Calculate experimental risk estimate using combined terrain + weather features
    features = {
        "slope_degrees": 48.0,
        "elevation_m": 1600.0,
        "rainfall_1h_mm": float(weather["current"]["precipitation_mm"]),
        "rainfall_24h_mm": 195.0,
        "rainfall_72h_mm": 380.0,
        "vegetation_index": 0.25,
        "soil_moisture": 0.80,
    }
    engine = RiskEngine()
    risk_result = engine.estimate_risk(
        features,
        observation_time=datetime.now(timezone.utc),
    )
    assert risk_result["is_experimental"] is True
    assert risk_result["risk_category"] in ("High", "Very High")
    assert 0.0 <= risk_result["probability"] <= 1.0

    # 7. Visualize location on interactive map
    risk_point = {
        "id": 1,
        "name": "Munnar (Idukki, Kerala)",
        "latitude": lat,
        "longitude": lon,
        "risk_category": risk_result["risk_category"],
        "probability": risk_result["probability"],
        "rainfall_24h_mm": features["rainfall_24h_mm"],
        "rainfall_72h_mm": features["rainfall_72h_mm"],
        "slope_degrees": features["slope_degrees"],
        "elevation_m": features["elevation_m"],
        "data_source": "Open-Meteo Forecast",
        "observed_at": now_iso,
        "is_experimental": True,
    }
    fmap = build_landslide_map(
        observations_df=df.head(40),
        risk_data=[risk_point],
        show_heatmap=True,
    )
    assert isinstance(fmap, folium.Map)
    assert len(fmap.to_json()) > 100

    # 8. Explain the model output with SHAP
    xai_result = explain_prediction(features, model_name="best_model")
    assert xai_result["available"] is True
    assert xai_result["shap_available"] is True
    assert len(xai_result["contributions"]) == len(feature_cols)
    assert len(xai_result["feature_statements"]) >= 1
    assert "causation" in xai_result["scientific_disclaimer"].lower()

    # 9 & 10. Persist station, prediction, and experimental alert in SQLite
    loc_id = models_mod.upsert_location(
        name="Munnar (Idukki, Kerala)",
        latitude=lat,
        longitude=lon,
        state_region="Kerala",
        elevation_m=features["elevation_m"],
        slope_degrees=features["slope_degrees"],
    )
    pred_id = models_mod.save_prediction(
        probability=risk_result["probability"],
        risk_category=risk_result["risk_category"],
        model_name=str(risk_result.get("model_name", "best_model")),
        method=str(risk_result.get("method", "ml_model")),
        input_features=features,
        warnings=risk_result.get("warnings", []),
        location_id=loc_id,
        is_experimental=True,
    )
    assert pred_id > 0

    manager = AlertManager()
    alert = manager.evaluate_and_alert(
        risk_result,
        location_id=loc_id,
        location_name="Munnar (Idukki, Kerala)",
        latitude=lat,
        longitude=lon,
        supporting_env_info=features,
        model_version=str(risk_result.get("model_name", "best_model")),
        data_source="Open-Meteo Forecast",
    )
    assert alert is not None
    assert alert["is_experimental"] is True
    assert alert["status"] == "new"

    # 11. Retrieve assessment & acknowledge/resolve alert history
    assert manager.acknowledge(alert["id"], note="Reviewed in E2E integration test") is True
    assert manager.resolve(alert["id"], note="Cleared in E2E integration test") is True

    persisted_alerts = manager.get_all_alerts()
    assert len(persisted_alerts) == 1
    first_row = persisted_alerts.iloc[0]
    assert first_row["status"] == "resolved"
    env_info = first_row["supporting_env_info"]
    if isinstance(env_info, str):
        env_info = json.loads(env_info)
    assert env_info["rainfall_24h_mm"] == 195.0

    audit = manager.get_audit_log(alert_id=alert["id"])
    assert set(audit["action"].tolist()) == {"acknowledged", "resolved"}
    assert len(audit) == 2

    with sqlite3.connect(isolated_db) as conn:
        pred_count = conn.execute("SELECT COUNT(*) FROM prediction_records").fetchone()[0]
    assert pred_count == 2
    assert int(first_row["prediction_id"]) > 0


def test_canonical_10_step_user_workflow(isolated_db: Path):
    """Verify the exact 10-step end-to-end workflow specified in the audit requirements:

    1. User selects a monitoring region.
    2. Application loads available geographic information.
    3. Application retrieves weather data where supported.
    4. Application validates environmental inputs.
    5. Risk model generates an estimate when sufficient data exists.
    6. Explainability module explains the estimate.
    7. Alert module evaluates configured conditions.
    8. Dashboard displays results (structured view model verification).
    9. SQLite stores applicable records.
    10. Reports can be generated from actual available information (PDF report generation).
    """
    # Step 1: User selects a monitoring region
    selected_region = {
        "state": "Kerala",
        "district": "Wayanad",
        "station_name": "Meppadi (Wayanad, Kerala)",
    }
    assert selected_region["state"] == "Kerala"
    assert selected_region["district"] == "Wayanad"

    # Step 2: Application loads available geographic information
    # Authentic coordinates for Meppadi / Chooralmala disaster zone
    geo_profile = {
        "name": selected_region["station_name"],
        "state_region": selected_region["state"],
        "district": selected_region["district"],
        "latitude": 11.5520,
        "longitude": 76.1280,
        "elevation_m": 880.0,
        "slope_degrees": 42.5,
    }
    assert -90.0 <= geo_profile["latitude"] <= 90.0
    assert -180.0 <= geo_profile["longitude"] <= 180.0
    assert geo_profile["slope_degrees"] > 0
    loc_id = models_mod.upsert_location(
        name=geo_profile["name"],
        latitude=geo_profile["latitude"],
        longitude=geo_profile["longitude"],
        state_region=geo_profile["state_region"],
        elevation_m=geo_profile["elevation_m"],
        slope_degrees=geo_profile["slope_degrees"],
    )
    assert loc_id > 0

    # Step 3: Application retrieves weather data where supported
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
    fake_telemetry = {
        "latitude": geo_profile["latitude"],
        "longitude": geo_profile["longitude"],
        "current": {
            "time": now_iso,
            "temperature_2m": 22.0,
            "relative_humidity_2m": 98.0,
            "precipitation": 45.0,
        },
        "hourly": {
            "time": [now_iso],
            "precipitation": [45.0],
            "soil_moisture_0_to_7cm": [0.82],
        },
        "daily": {
            "time": [now_iso[:10]],
            "precipitation_sum": [240.0],
        },
    }
    weather_provider = WeatherProvider()
    with patch.object(weather_provider, "_get_cached", return_value=fake_telemetry):
        weather = weather_provider.fetch(geo_profile["latitude"], geo_profile["longitude"])
    assert weather is not None
    assert weather["current"]["precipitation_mm"] == 45.0

    # Step 4: Application validates environmental inputs
    raw_env_inputs = {
        "slope_degrees": geo_profile["slope_degrees"],
        "elevation_m": geo_profile["elevation_m"],
        "rainfall_1h_mm": float(weather["current"]["precipitation_mm"]),
        "rainfall_24h_mm": 240.0,
        "rainfall_72h_mm": 410.0,
        "vegetation_index": 0.35,
        "soil_moisture": 0.82,
    }
    required_keys = ["slope_degrees", "elevation_m", "rainfall_24h_mm", "rainfall_72h_mm"]
    validation_res = validate_feature_dict(raw_env_inputs, required_features=required_keys)
    assert validation_res.is_valid is True
    assert validation_res.errors == []
    clean_features = raw_env_inputs

    # Step 5: Risk model generates an estimate when sufficient data exists
    risk_engine = RiskEngine()
    risk_estimate = risk_engine.estimate_risk(
        clean_features,
        observation_time=datetime.now(timezone.utc),
    )
    assert risk_estimate["risk_category"] in ("High", "Very High")
    assert 0.0 <= risk_estimate["probability"] <= 1.0
    assert risk_estimate["is_experimental"] is True

    # Step 6: Explainability module explains the estimate
    xai = explain_prediction(clean_features, model_name="best_model")
    assert xai["available"] is True
    assert len(xai["contributions"]) > 0
    assert len(xai["feature_statements"]) >= 1

    # Step 7: Alert module evaluates configured conditions
    alert_mgr = AlertManager()
    alert_eval = alert_mgr.evaluate_and_alert(
        risk_estimate,
        location_id=loc_id,
        location_name=geo_profile["name"],
        latitude=geo_profile["latitude"],
        longitude=geo_profile["longitude"],
        supporting_env_info=clean_features,
        model_version=str(risk_estimate.get("model_name", "best_model")),
        data_source="Open-Meteo Verified Test Mock",
    )
    assert alert_eval is not None
    assert alert_eval["risk_category"] in ("High", "Very High")
    assert alert_eval["status"] == "new"
    assert alert_eval["is_experimental"] is True

    # Step 8: Dashboard displays results (verify UI view-model construction)
    dashboard_view = {
        "kpi_cards": {
            "location": geo_profile["name"],
            "risk_tier": risk_estimate["risk_category"],
            "probability_pct": f"{risk_estimate['probability'] * 100:.1f}%",
            "active_alert_severity": alert_eval["risk_category"],
            "weather_update": now_iso,
        },
        "environmental_cards": {
            "rainfall_1h": clean_features["rainfall_1h_mm"],
            "rainfall_24h": clean_features["rainfall_24h_mm"],
            "soil_moisture": clean_features["soil_moisture"],
            "slope": clean_features["slope_degrees"],
            "elevation": clean_features["elevation_m"],
        },
        "xai_explanation": {
            "top_factor": xai["top_positive_features"][0] if xai.get("top_positive_features") else None,
            "disclaimer_present": bool(xai.get("scientific_disclaimer")),
        },
    }
    assert dashboard_view["kpi_cards"]["risk_tier"] in ("High", "Very High")
    assert dashboard_view["xai_explanation"]["disclaimer_present"] is True

    # Step 9: SQLite stores applicable records
    loc_id = models_mod.upsert_location(
        name=geo_profile["name"],
        latitude=geo_profile["latitude"],
        longitude=geo_profile["longitude"],
        state_region=geo_profile["state_region"],
        elevation_m=geo_profile["elevation_m"],
        slope_degrees=geo_profile["slope_degrees"],
    )
    pred_rec_id = models_mod.save_prediction(
        probability=risk_estimate["probability"],
        risk_category=risk_estimate["risk_category"],
        model_name=str(risk_estimate.get("model_name", "best_model")),
        method=str(risk_estimate.get("method", "ml_model")),
        input_features=clean_features,
        warnings=risk_estimate.get("warnings", []),
        location_id=loc_id,
        is_experimental=True,
    )
    assert loc_id > 0
    assert pred_rec_id > 0

    # Step 10: Reports can be generated from actual available information (PDF generation)
    sample_facilities = [
        {"name": "Meppadi Community Health Centre", "facility_type": "hospital", "distance_km": 3.2},
        {"name": "Meppadi Police Station", "facility_type": "police", "distance_km": 2.8},
    ]
    sample_historical = [
        {"event_id": "WAYANAD-2024-001", "event_date": "2024-07-30", "location": "Chooralmala", "distance_km": 4.5},
    ]
    pdf_bytes = generate_assessment_pdf_report(
        location_profile=geo_profile,
        weather_telemetry=clean_features,
        risk_assessment=risk_estimate,
        historical_events=sample_historical,
        nearby_facilities=sample_facilities,
        is_synthetic_benchmark=True,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
