"""Unit and API tests for Phase 1 Live Risk Monitoring functionality."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi import HTTPException

import src.database.db as db_mod
import src.database.models as models_mod
from api import (
    MonitoringLocationCreate,
    MonitoringLocationStatusUpdate,
    MonitoringRefreshRequest,
    create_monitoring_location,
    get_monitoring_location_history,
    get_monitoring_status,
    list_monitoring_locations,
    refresh_monitoring,
    seed_default_monitoring_stations,
    toggle_monitoring_location,
)
from src.monitoring.live_monitor import (
    LiveRiskMonitor,
    compute_historical_distribution,
    compute_risk_by_region,
    derive_weather_condition,
)


def _run_coro_sync(coro):
    """Execute a coroutine without creating an asyncio event loop socketpair."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch):
    """Point db_context at a clean temporary SQLite file for each test."""
    db_file = tmp_path / "test_monitoring.db"

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


def _sample_weather_payload(rain_mm: float = 12.0, soil: float = 0.52) -> dict:
    """Build a valid normalized WeatherProvider response matching its real schema."""
    now_iso = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()
    return {
        "latitude": 10.0889,
        "longitude": 77.0595,
        "timezone": "UTC",
        "elevation_m": 1525.0,
        "source": "Open-Meteo ECMWF/ERA5 Reanalysis",
        "data_type": "model_derived_reanalysis",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "hourly": {
            "time": [now_iso] * 72,
            "precipitation": [1.0] * 48 + [2.0] * 23 + [rain_mm],
            "temperature_2m": [21.5] * 72,
            "relative_humidity_2m": [86.0] * 72,
            "wind_speed_10m": [14.2] * 72,
            "soil_moisture_0_to_7cm": [soil] * 72,
        },
    }


def test_location_and_environmental_crud(isolated_db):
    loc_id = models_mod.upsert_location(
        name="Munnar",
        latitude=10.0889,
        longitude=77.0595,
        state_region="Kerala",
        elevation_m=1532.0,
        slope_degrees=26.0,
    )
    assert loc_id > 0

    # Upsert same name (case-insensitive) updates instead of duplicating
    loc_id_2 = models_mod.upsert_location(
        name="munnar",
        latitude=10.09,
        longitude=77.06,
        slope_degrees=28.0,
    )
    assert loc_id_2 == loc_id
    locs = models_mod.get_locations(active_only=False)
    assert len(locs) == 1
    assert float(locs.iloc[0]["slope_degrees"]) == pytest.approx(28.0)

    env_id = models_mod.insert_environmental_observation(
        location_id=loc_id,
        observed_at=datetime.now(timezone.utc).isoformat(),
        rainfall_1h_mm=4.5,
        rainfall_24h_mm=38.0,
        rainfall_72h_mm=85.0,
        temperature_c=20.0,
        humidity_pct=88.0,
        soil_moisture=0.51,
        data_source="Open-Meteo",
        is_observed=False,
    )
    assert env_id > 0
    latest = models_mod.get_latest_environmental_observation(loc_id)
    assert latest is not None
    assert float(latest["rainfall_24h_mm"]) == pytest.approx(38.0)

    assert models_mod.update_location_status(loc_id, False) is True
    assert models_mod.get_locations(active_only=True).empty


def test_validate_and_extract_weather():
    from datetime import timedelta

    assert LiveRiskMonitor.validate_and_extract_weather(None) is None
    assert LiveRiskMonitor.validate_and_extract_weather({}) is None
    assert LiveRiskMonitor.validate_and_extract_weather({"hourly": {"precipitation": [None, None]}}) is None

    extracted = LiveRiskMonitor.validate_and_extract_weather(_sample_weather_payload(rain_mm=8.0, soil=0.6))
    assert extracted is not None
    assert extracted["rainfall_1h_mm"] == pytest.approx(8.0)
    # Last 24h: 23 * 2.0 + 8.0 = 54.0
    assert extracted["rainfall_24h_mm"] == pytest.approx(54.0)
    # Last 72h: 48 * 1.0 + 23 * 2.0 + 8.0 = 102.0
    assert extracted["rainfall_72h_mm"] == pytest.approx(102.0)
    assert extracted["wind_speed_kmh"] == pytest.approx(14.2)
    assert extracted["soil_moisture"] == pytest.approx(0.6)
    assert extracted["is_observed"] is False

    # Future forecast hours should be excluded and separated into forecast_summary;
    # <24h window should leave 24h/72h accumulations as None instead of fabricating sums.
    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    past_t = (now_ist - timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")
    future_t = (now_ist + timedelta(hours=6)).strftime("%Y-%m-%dT%H:00")
    payload_with_future = {
        "source": "Open-Meteo",
        "utc_offset_seconds": 19800,
        "hourly": {
            "time": [past_t, future_t],
            "precipitation": [3.5, 99.0],
            "temperature_2m": [22.0, 10.0],
            "relative_humidity_2m": [80.0, 50.0],
            "wind_speed_10m": [18.4, 42.0],
            "soil_moisture_0_to_7cm": [0.45, 0.99],
        },
    }
    trimmed = LiveRiskMonitor.validate_and_extract_weather(payload_with_future)
    assert trimmed is not None
    assert trimmed["rainfall_1h_mm"] == pytest.approx(3.5)
    assert trimmed["rainfall_24h_mm"] is None
    assert trimmed["rainfall_72h_mm"] is None
    assert trimmed["wind_speed_kmh"] == pytest.approx(18.4)
    assert trimmed["soil_moisture"] == pytest.approx(0.45)
    assert trimmed["forecast_summary"]["is_forecast"] is True
    assert trimmed["forecast_summary"]["forecast_total_rain_mm"] == pytest.approx(99.0)


def test_dashboard_summary_empty_db_missing_dataset_and_distribution(isolated_db):
    import pandas as pd
    from src.monitoring.live_monitor import compute_risk_distribution

    monitor = LiveRiskMonitor()
    empty_summary = monitor.get_dashboard_summary(dataset_df=None, model_info={})
    assert empty_summary["dataset_status"] == "Missing"
    assert empty_summary["total_historical_observations"] == 0
    assert empty_summary["locations_assessed"] == 0
    assert empty_summary["high_risk_count"] == 0
    assert empty_summary["very_high_risk_count"] == 0
    assert empty_summary["overall_freshness"] == "No Stations"

    empty_act = monitor.get_recent_activity(limit=5)
    assert empty_act["recent_observations"].empty
    assert empty_act["recent_predictions"].empty
    assert empty_act["recent_alerts"].empty
    assert empty_act["recent_weather_updates"].empty

    sample_df = pd.DataFrame({"landslide_occurred": [1, 0, 1], "is_synthetic": [1, 1, 1]})
    with_ds = monitor.get_dashboard_summary(
        dataset_df=sample_df,
        model_info={"metadata": {"algorithm": "logistic_regression", "saved_at": "2026-11-12T00:00:00Z"}},
    )
    assert "Synthetic Dev" in with_ds["dataset_status"]
    assert with_ds["total_historical_observations"] == 2
    assert with_ds["model_name"] == "Logistic Regression"
    assert with_ds["model_version"] == "2026-11-12"

    dist = compute_risk_distribution([
        {"risk_category": "Low"},
        {"risk_category": "Moderate"},
        {"risk_category": "High"},
        {"risk_category": "Very High"},
        {"risk_category": "Very High"},
    ])
    assert dist["total"] == 5
    assert dist["counts"]["Low"] == 1
    assert dist["counts"]["Very High"] == 2
    assert dist["percentages"]["Very High"] == pytest.approx(40.0)


def test_base_provider_retry_decorator_valid():
    from src.data_providers.weather_provider import WeatherProvider

    wp = WeatherProvider()
    wp._session.get = MagicMock(side_effect=RuntimeError("mock-abort"))
    wp.max_retries = 1
    with pytest.raises(RuntimeError, match="mock-abort"):
        wp._get("https://example.invalid")


def test_register_location_validation_and_terrain_fallback(isolated_db):
    mock_terrain = MagicMock()
    mock_terrain.get_terrain_features.side_effect = RuntimeError("timeout")
    monitor = LiveRiskMonitor(terrain_provider=mock_terrain)

    with pytest.raises(ValueError):
        monitor.register_location("", 10.0, 77.0)
    with pytest.raises(ValueError):
        monitor.register_location("BadLat", 95.0, 77.0)

    rec = monitor.register_location("Coonoor", 11.353, 76.7959, fetch_terrain=True)
    assert rec["name"] == "Coonoor"
    assert any("Terrain lookup unavailable" in w for w in rec["warnings"])


def test_refresh_location_live_and_offline_fallback(isolated_db):
    mock_weather = MagicMock()
    mock_weather.fetch.return_value = _sample_weather_payload(rain_mm=40.0, soil=0.82)
    monitor = LiveRiskMonitor(weather_provider=mock_weather)

    loc = monitor.register_location(
        "Munnar",
        10.0889,
        77.0595,
        elevation_m=1532.0,
        slope_degrees=38.0,
        fetch_terrain=False,
    )
    loc_id = int(loc["id"])

    live_res = monitor.refresh_location(loc_id, generate_alerts=True)
    assert live_res["feed_status"] == "live"
    assert live_res["freshness_status"] == "Fresh"
    assert live_res["is_experimental"] is True
    assert live_res["is_observed"] is False
    assert live_res["prediction_id"] > 0
    assert live_res["risk_category"] in {"Low", "Moderate", "High", "Very High"}
    if live_res["risk_category"] in {"High", "Very High"}:
        assert live_res["alert_triggered"] is True
        assert isinstance(live_res["alert_id"], int) and live_res["alert_id"] > 0

    # Simulate external weather API outage -> should fall back to cached DB observation
    mock_weather.fetch.return_value = None
    fallback_res = monitor.refresh_location(loc_id, generate_alerts=False)
    assert fallback_res["feed_status"] == "offline_fallback"
    assert fallback_res["freshness_status"] == "Offline Fallback"
    assert any("using cached observation" in w for w in fallback_res["warnings"])

    # Brand new station with unavailable weather and zero cache
    uncached = monitor.register_location(
        "NewStation", 12.0, 76.0, elevation_m=800.0, slope_degrees=15.0, fetch_terrain=False
    )
    unavail_res = monitor.refresh_location(int(uncached["id"]), generate_alerts=False)
    assert unavail_res["feed_status"] == "unavailable"
    assert unavail_res["freshness_status"] == "Unavailable"
    assert any("no cached environmental observation exists" in w for w in unavail_res["warnings"])


def test_monitoring_api_endpoints(isolated_db, monkeypatch):
    mock_weather = MagicMock()
    mock_weather.fetch.return_value = _sample_weather_payload(rain_mm=4.0, soil=0.4)
    mock_terrain = MagicMock()
    mock_terrain.get_terrain_features.return_value = {"elevation_m": 1200.0, "slope_degrees": 22.0}

    monkeypatch.setattr(
        "src.monitoring.get_live_monitor",
        lambda: LiveRiskMonitor(weather_provider=mock_weather, terrain_provider=mock_terrain),
    )

    seeded = _run_coro_sync(seed_default_monitoring_stations())
    assert seeded["count"] == 5

    created = _run_coro_sync(
        create_monitoring_location(
            MonitoringLocationCreate(
                name="Idukki",
                latitude=9.85,
                longitude=76.97,
                state_region="Kerala",
                fetch_terrain=True,
            )
        )
    )
    assert created["status"] == "registered"
    idukki_id = int(created["location"]["id"])

    listed = _run_coro_sync(list_monitoring_locations(active_only=True))
    assert listed["count"] == 6

    refreshed = _run_coro_sync(
        refresh_monitoring(MonitoringRefreshRequest(location_id=idukki_id, generate_alerts=False))
    )
    assert refreshed["total_stations"] == 1
    assert refreshed["stations"][0]["feed_status"] == "live"

    status = _run_coro_sync(get_monitoring_status(active_only=True))
    assert status["total_stations"] == 6
    assert "EXPERIMENTAL MONITORING" in status["notice"]

    hist = _run_coro_sync(get_monitoring_location_history(idukki_id, limit=20))
    assert len(hist["environmental_history"]) == 1
    assert len(hist["prediction_history"]) == 1

    toggled = _run_coro_sync(
        toggle_monitoring_location(idukki_id, MonitoringLocationStatusUpdate(is_active=False))
    )
    assert toggled["is_active"] is False

    with pytest.raises(HTTPException) as exc_info:
        _run_coro_sync(get_monitoring_location_history(999999))
    assert exc_info.value.status_code == 404


def test_dashboard_with_authentic_data(isolated_db):
    mock_weather = MagicMock()
    mock_weather.fetch.return_value = _sample_weather_payload(rain_mm=0.0, soil=0.25)
    monitor = LiveRiskMonitor(weather_provider=mock_weather)

    loc = monitor.register_location(
        "Munnar", 10.0889, 77.0595, state_region="Kerala",
        elevation_m=1532.0, slope_degrees=18.0, fetch_terrain=False,
    )
    monitor.refresh_location(int(loc["id"]), generate_alerts=False)

    models_mod.insert_observation(
        latitude=10.1,
        longitude=77.05,
        event_date="2024-07-30",
        landslide_type="debris flow",
        trigger="downpour",
        country="India",
        state_region="Kerala",
        source_name="NASA GLC",
        fatalities=3,
        is_synthetic=False,
    )
    auth_df = pd.DataFrame([
        {
            "latitude": 10.1,
            "longitude": 77.05,
            "state_region": "Kerala",
            "trigger": "downpour",
            "landslide_occurred": 1,
            "is_synthetic": 0,
        },
        {
            "latitude": 11.6,
            "longitude": 76.1,
            "state_region": "Kerala",
            "trigger": "monsoon",
            "landslide_occurred": 0,
            "is_synthetic": 0,
        },
    ])

    summary = monitor.get_dashboard_summary(dataset_df=auth_df)
    assert summary["authentic_observations_count"] == 1
    assert summary["synthetic_observations_count"] == 0
    assert summary["dataset_status"].startswith("Authentic")
    assert summary["dataset_is_synthetic"] is False
    assert summary["low_risk_count"] + summary["moderate_risk_count"] + summary["high_risk_count"] + summary["very_high_risk_count"] == 1
    assert len(summary["risk_by_region"]) == 1
    assert summary["risk_by_region"][0]["state_region"] == "Kerala"
    assert summary["historical_distribution"]["source_label"] == "Authentic DB Records"
    assert summary["historical_distribution"]["is_synthetic"] is False

    activity = monitor.get_recent_activity(limit=5)
    assert len(activity["recent_authentic_observations"]) == 1
    assert len(activity["recently_updated_locations"]) == 1
    assert activity["recently_updated_locations"][0]["weather_condition"] != "Unavailable"


def test_dashboard_with_synthetic_development_data(isolated_db):
    monitor = LiveRiskMonitor()
    models_mod.insert_observation(
        latitude=27.04,
        longitude=88.26,
        event_date="2025-01-10",
        landslide_type="mudslide",
        trigger="rain",
        country="India",
        state_region="West Bengal",
        source_name="Synthetic Generator",
        fatalities=0,
        is_synthetic=True,
    )
    synth_df = pd.DataFrame([
        {
            "latitude": 27.04,
            "longitude": 88.26,
            "state_region": "West Bengal",
            "trigger": "rain",
            "landslide_occurred": 1,
            "is_synthetic": 1,
        }
    ])

    summary = monitor.get_dashboard_summary(dataset_df=synth_df)
    assert summary["authentic_observations_count"] == 0
    assert summary["synthetic_observations_count"] == 1
    assert summary["dataset_status"].startswith("Synthetic Dev")
    assert summary["dataset_is_synthetic"] is True
    assert summary["historical_distribution"]["is_synthetic"] is True

    activity = monitor.get_recent_activity(limit=5)
    assert activity["recent_authentic_observations"].empty
    assert len(activity["recent_observations"]) == 1


def test_dashboard_with_empty_datasets(isolated_db):
    monitor = LiveRiskMonitor()
    summary = monitor.get_dashboard_summary(dataset_df=pd.DataFrame())
    assert summary["dataset_status"] == "Missing"
    assert summary["total_active_stations"] == 0
    assert summary["low_risk_count"] == 0
    assert summary["moderate_risk_count"] == 0
    assert summary["high_risk_count"] == 0
    assert summary["very_high_risk_count"] == 0
    assert summary["overall_freshness"] == "No Stations"
    assert summary["risk_by_region"] == []
    assert summary["historical_distribution"]["available"] is False
    assert summary["historical_distribution"]["source_label"] == "Unavailable"


def test_missing_environmental_variables_not_imputed_with_zero(isolated_db):
    payload = _sample_weather_payload(rain_mm=3.5)
    for k, seq in payload["hourly"].items():
        payload["hourly"][k] = seq[-10:]
    del payload["hourly"]["soil_moisture_0_to_7cm"]
    payload["hourly"]["temperature_2m"] = [None] * 10
    payload["elevation_m"] = None

    extracted = LiveRiskMonitor.validate_and_extract_weather(payload)
    assert extracted is not None
    assert extracted["rainfall_1h_mm"] == 3.5
    assert extracted["rainfall_24h_mm"] is None
    assert extracted["rainfall_72h_mm"] is None
    assert extracted["soil_moisture"] is None
    assert extracted["temperature_c"] is None
    assert extracted["elevation_m"] is None

    assert derive_weather_condition(None, None, None) == "Unavailable"


def test_invalid_records_and_corrupt_warnings_handled_safely(isolated_db):
    assert LiveRiskMonitor.validate_and_extract_weather({"hourly": {"precipitation": [float("nan"), float("inf")]}}) is None

    monitor = LiveRiskMonitor()
    loc = monitor.register_location("CorruptWarn", 10.5, 77.5, state_region="Tamil Nadu", fetch_terrain=False)
    models_mod.insert_environmental_observation(
        location_id=int(loc["id"]),
        observed_at="2026-01-01T00:00:00+00:00",
        rainfall_1h_mm=0.5,
    )
    with models_mod.db_context() as conn:
        conn.execute(
            """
            INSERT INTO prediction_records
                (location_id, predicted_at, probability, risk_category, model_name, method, is_experimental, input_features, warnings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (int(loc["id"]), "2026-01-01T00:00:00+00:00", 0.25, "Low", "test", "heuristic_baseline", 1, "{}", "{corrupt-json"),
        )
    snap = monitor.get_live_snapshot(active_only=True)
    assert snap["stations"][0]["warnings"] == ["{corrupt-json"]

    region_rows = compute_risk_by_region([
        {"state_region": "Kerala", "risk_category": "Low", "latitude": 10.0, "longitude": 77.0},
        {"state_region": "BadLat", "risk_category": "High", "latitude": 140.0, "longitude": 77.0},
        {"state_region": "NaNLon", "risk_category": "Moderate", "latitude": 10.0, "longitude": float("nan")},
    ])
    assert len(region_rows) == 1
    assert region_rows[0]["state_region"] == "Kerala"

    invalid_df = pd.DataFrame([
        {"latitude": 200.0, "longitude": 77.0, "state_region": "Invalid", "landslide_occurred": 1, "is_synthetic": 0},
        {"latitude": 10.1, "longitude": 77.0, "state_region": "Kerala", "landslide_occurred": 1, "is_synthetic": 0},
    ])
    hist_dist = compute_historical_distribution(dataset_df=invalid_df)
    assert hist_dist["valid_records"] == 1
    assert hist_dist["dropped_invalid_count"] == 1
    assert hist_dist["by_region"][0]["region"] == "Kerala"


def test_api_unavailable_state_dashboard_summary(isolated_db):
    mock_weather = MagicMock()
    mock_weather.fetch.side_effect = RuntimeError("Open-Meteo timeout")
    monitor = LiveRiskMonitor(weather_provider=mock_weather)

    loc = monitor.register_location("Wayanad", 11.6854, 76.1320, state_region="Kerala", fetch_terrain=False)
    res = monitor.refresh_location(int(loc["id"]), generate_alerts=False)
    assert res["feed_status"] == "unavailable"
    assert res["freshness_status"] == "Unavailable"

    summary = monitor.get_dashboard_summary()
    assert summary["total_active_stations"] == 1
    assert summary["latest_weather_update"] is None
    assert summary["snapshot"]["stations"][0]["weather_condition"] == "Unavailable"
