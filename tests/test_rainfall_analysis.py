"""Unit and API tests for Phase 3 Dynamic Rainfall Analysis."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
from fastapi import HTTPException

import src.data_providers.base as base_mod
import src.database.db as db_mod
import src.database.models as models_mod
from api import (
    RainfallRiskEvaluateRequest,
    evaluate_rainfall_risk,
    get_rainfall_analysis,
)
from src.data_providers.base import ProviderError
from src.data_providers.weather_provider import (
    WeatherProvider,
    convert_precipitation_values,
    convert_temperature_values,
    convert_wind_values,
    decode_wmo_weather_code,
    parse_api_timestamp,
)
from src.monitoring.rainfall_analysis import (
    build_daily_series_dataframe,
    build_hourly_series_dataframe,
    compute_rainfall_metrics,
    evaluate_rainfall_risk_change,
)


def _run_coro_sync(coro):
    """Execute a coroutine synchronously without creating an asyncio socketpair."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch):
    """Point db_context at a clean temporary SQLite file."""
    db_file = tmp_path / "test_rainfall.db"

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


def _build_raw_openmeteo_payload(
    past_hours: int = 72,
    forecast_hours: int = 12,
    precip_val: float = 2.0,
    precip_unit: str = "mm",
    wind_unit: str = "km/h",
    missing_indices: list[int] | None = None,
) -> dict:
    """Create a realistic Open-Meteo raw JSON payload with past + future timestamps."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    times: list[str] = []
    precips: list[float | None] = []
    temps: list[float] = []
    rh: list[float] = []
    wind: list[float] = []
    soil: list[float] = []

    for offset in range(-(past_hours - 1), forecast_hours + 1):
        ts = (now + timedelta(hours=offset)).strftime("%Y-%m-%dT%H:%M")
        times.append(ts)
        precips.append(precip_val)
        temps.append(22.0)
        rh.append(85.0)
        wind.append(10.0)
        soil.append(0.42)

    for idx in missing_indices or []:
        if 0 <= idx < len(precips):
            precips[idx] = None

    today_date = now.date()
    return {
        "latitude": 10.0889,
        "longitude": 77.0595,
        "timezone": "UTC",
        "utc_offset_seconds": 0,
        "elevation": 1525.0,
        "current_units": {
            "precipitation": precip_unit,
            "temperature_2m": "°C",
            "wind_speed_10m": wind_unit,
        },
        "current": {
            "time": now.strftime("%Y-%m-%dT%H:%M"),
            "precipitation": 3.5,
            "temperature_2m": 21.4,
            "relative_humidity_2m": 88.0,
            "wind_speed_10m": 12.0,
        },
        "hourly_units": {
            "precipitation": precip_unit,
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": wind_unit,
            "soil_moisture_0_to_7cm": "m³/m³",
        },
        "hourly": {
            "time": times,
            "precipitation": precips,
            "temperature_2m": temps,
            "relative_humidity_2m": rh,
            "wind_speed_10m": wind,
            "soil_moisture_0_to_7cm": soil,
        },
        "daily_units": {
            "precipitation_sum": precip_unit,
            "temperature_2m_max": "°C",
            "temperature_2m_min": "°C",
        },
        "daily": {
            "time": [
                (today_date - timedelta(days=2)).isoformat(),
                (today_date - timedelta(days=1)).isoformat(),
                today_date.isoformat(),
                (today_date + timedelta(days=1)).isoformat(),
            ],
            "precipitation_sum": [18.0, 24.0, 12.0, 40.0],
            "temperature_2m_max": [24.0, 23.0, 22.5, 21.0],
            "temperature_2m_min": [16.0, 15.5, 15.0, 14.5],
        },
    }


def test_parse_openmeteo_separates_current_historical_forecast_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "CACHE_DIR", tmp_path)
    provider = WeatherProvider()
    raw = _build_raw_openmeteo_payload(past_hours=72, forecast_hours=6, precip_val=2.0)

    call_count = {"n": 0}

    def fake_get(url, params=None):
        call_count["n"] += 1
        return raw

    monkeypatch.setattr(provider, "_get", fake_get)

    first = provider.fetch(latitude=10.0889, longitude=77.0595, past_days=3, forecast_days=1)
    assert first is not None
    assert call_count["n"] == 1
    assert first["is_cached"] is False
    assert first["current"]["is_forecast"] is False
    assert first["current"]["precipitation_mm"] == 3.5
    assert first["historical_hourly"]["is_forecast"] is False
    assert first["forecast_hourly"]["is_forecast"] is True
    assert len(first["historical_hourly"]["precipitation"]) == 72
    assert len(first["forecast_hourly"]["precipitation"]) == 6
    assert len(first["daily_historical"]["time"]) == 3
    assert len(first["daily_forecast"]["time"]) == 1
    assert first["unit_validation"]["validated"] is True

    # Second identical call hits disk cache and MUST mark is_cached=True + cached_at
    second = provider.fetch(latitude=10.0889, longitude=77.0595, past_days=3, forecast_days=1)
    assert second is not None
    assert call_count["n"] == 1
    assert second["is_cached"] is True
    assert second["cached_at"] is not None


def test_missing_rainfall_never_replaced_with_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "CACHE_DIR", tmp_path)
    provider = WeatherProvider()
    # 72 past hours, but index 60 (within trailing 24h) is None
    raw = _build_raw_openmeteo_payload(
        past_hours=72,
        forecast_hours=0,
        precip_val=2.5,
        missing_indices=[60],
    )
    monkeypatch.setattr(provider, "_get", lambda url, params=None: raw)

    acc = provider.get_rainfall_accumulations(10.0889, 77.0595)
    assert acc["rainfall_1h_mm"] == 2.5
    assert acc["rainfall_24h_mm"] is None
    assert acc["rainfall_48h_mm"] is None
    assert acc["rainfall_72h_mm"] is None
    assert acc["missing_hours_24h"] == 1


def test_accumulation_windows_24h_48h_72h_strict_sufficiency():
    # Exact 72 valid hours -> all windows populated
    metrics_72 = compute_rainfall_metrics({"precipitation": [1.5] * 72})
    assert metrics_72["rainfall_1h_mm"] == 1.5
    assert metrics_72["rainfall_24h_mm"] == 36.0
    assert metrics_72["rainfall_48h_mm"] == 72.0
    assert metrics_72["rainfall_72h_mm"] == 108.0

    # Only 36 valid hours -> 1h and 24h valid, 48h and 72h strictly None
    metrics_36 = compute_rainfall_metrics({"precipitation": [2.0] * 36})
    assert metrics_36["rainfall_1h_mm"] == 2.0
    assert metrics_36["rainfall_24h_mm"] == 48.0
    assert metrics_36["rainfall_48h_mm"] is None
    assert metrics_36["rainfall_72h_mm"] is None


def test_timestamp_validation_and_timezone_offset():
    dt_utc = parse_api_timestamp("2026-06-15T12:00:00Z")
    assert dt_utc == datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)

    # Naive Open-Meteo timestamp interpreted using +05:30 offset (19800s) -> 06:30 UTC
    dt_ist = parse_api_timestamp("2026-06-15T12:00", utc_offset_seconds=19800)
    assert dt_ist == datetime(2026, 6, 15, 6, 30, tzinfo=timezone.utc)

    assert parse_api_timestamp("not-a-timestamp") is None
    assert parse_api_timestamp(None) is None


def test_unit_conversion_and_validation(tmp_path, monkeypatch):
    assert convert_precipitation_values([1.0, None], "inch") == [25.4, None]
    assert convert_precipitation_values([2.0], "cm") == [20.0]
    assert convert_wind_values([10.0, None], "m/s") == [36.0, None]
    assert convert_temperature_values([68.0, None], "°F") == [20.0, None]

    with pytest.raises(ValueError):
        convert_precipitation_values([1.0], "liters_per_acre")
    with pytest.raises(ValueError):
        convert_wind_values([5.0], "parsecs_per_day")

    # End-to-end provider unit normalization (inch -> mm, m/s -> km/h)
    monkeypatch.setattr(base_mod, "CACHE_DIR", tmp_path)
    provider = WeatherProvider()
    raw = _build_raw_openmeteo_payload(past_hours=24, forecast_hours=0, precip_val=1.0, precip_unit="inch", wind_unit="m/s")
    monkeypatch.setattr(provider, "_get", lambda url, params=None: raw)

    parsed = provider.fetch(10.0, 77.0)
    assert parsed["unit_validation"]["validated"] is True
    assert parsed["hourly"]["precipitation"][0] == pytest.approx(25.4, rel=1e-3)
    assert parsed["hourly"]["wind_speed_10m"][0] == pytest.approx(36.0, rel=1e-3)


def test_api_timeout_triggers_provider_error_and_graceful_none(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "CACHE_DIR", tmp_path)
    provider = WeatherProvider()
    provider.max_retries = 2

    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("simulated read timeout")

    monkeypatch.setattr(provider._session, "get", raise_timeout)
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)

    with pytest.raises(ProviderError, match="timed out"):
        provider._get("https://api.open-meteo.com/v1/forecast")

    # Public fetch gracefully degrades to None instead of crashing
    assert provider.fetch(10.0, 77.0) is None


def test_retry_behavior_succeeds_after_transient_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "CACHE_DIR", tmp_path)
    provider = WeatherProvider()
    provider.max_retries = 3
    attempts = {"count": 0}

    ok_resp = MagicMock()
    ok_resp.raise_for_status.return_value = None
    ok_resp.json.return_value = {"hourly": {}, "daily": {}}

    def flaky_get(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise requests.exceptions.ConnectionError("transient network glitch")
        return ok_resp

    monkeypatch.setattr(provider._session, "get", flaky_get)
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)

    payload = provider._get("https://api.open-meteo.com/v1/forecast")
    assert attempts["count"] == 3
    assert payload == {"hourly": {}, "daily": {}}


def test_rainfall_trend_anomaly_and_dataframes():
    rising_hourly = {"precipitation": [float(i) for i in range(24)]}
    baseline = {"precipitation_sum": [10.0] * 40}
    metrics = compute_rainfall_metrics(rising_hourly, baseline_daily=baseline)

    assert metrics["trend_label"] == "Increasing"
    assert metrics["trend_slope_mm_per_hr"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["baseline_daily_mean_mm"] == 10.0
    assert metrics["anomaly_24h_mm"] == pytest.approx(sum(range(24)) - 10.0, rel=1e-3)
    assert metrics["anomaly_status"] == "Above Historical Daily Mean"

    # DataFrames cleanly tag Past vs Forecast
    raw_parsed = WeatherProvider()._parse_response(
        _build_raw_openmeteo_payload(past_hours=12, forecast_hours=4),
        10.0,
        77.0,
    )
    hdf = build_hourly_series_dataframe(raw_parsed)
    ddf = build_daily_series_dataframe(raw_parsed)
    assert set(hdf["is_forecast"].unique()) == {False, True}
    assert set(ddf["is_forecast"].unique()) == {False, True}


def test_dynamic_risk_integration_manual_and_delta_explanation(isolated_db):
    loc_id = models_mod.upsert_location(
        name="Munnar Test",
        latitude=10.0889,
        longitude=77.0595,
        elevation_m=1525.0,
        slope_degrees=28.0,
    )

    first = evaluate_rainfall_risk_change(
        features={
            "latitude": 10.0889,
            "longitude": 77.0595,
            "slope_degrees": 28.0,
            "elevation_m": 1525.0,
            "rainfall_1h_mm": 1.0,
            "rainfall_24h_mm": 8.0,
            "rainfall_48h_mm": 12.0,
            "rainfall_72h_mm": 15.0,
            "soil_moisture": 0.25,
        },
        observation_time=datetime.now(timezone.utc),
        location_id=loc_id,
        location_name="Munnar Test",
        source_label="Open-Meteo Live Reanalysis",
        is_manual=False,
    )
    assert first["prediction_id"] is not None
    assert first["previous_estimate"]["probability"] is None
    assert first["category_change"].startswith("Initial Assessment")

    # Elevated manual fallback update records delta vs previous prediction
    second = evaluate_rainfall_risk_change(
        features={
            "latitude": 10.0889,
            "longitude": 77.0595,
            "slope_degrees": 28.0,
            "elevation_m": 1525.0,
            "rainfall_1h_mm": 45.0,
            "rainfall_24h_mm": 210.0,
            "rainfall_48h_mm": 310.0,
            "rainfall_72h_mm": 390.0,
            "soil_moisture": 0.78,
        },
        observation_time=datetime.now(timezone.utc),
        location_id=loc_id,
        location_name="Munnar Test",
        source_label="User-Provided Data (Manual Fallback)",
        is_manual=True,
    )
    assert second["is_manual_input"] is True
    assert second["previous_estimate"]["probability"] == first["current_estimate"]["probability"]
    assert second["probability_delta"] is not None
    assert any(f["feature"] == "rainfall_24h_mm" and f["delta"] == 202.0 for f in second["influencing_factors"])
    assert any("Manual Fallback Input" in w for w in second["warnings"])
    assert "causation" in second["scientific_disclaimer"].lower()


def test_rainfall_api_endpoints_and_offline_fallback(isolated_db, monkeypatch):
    # API offline -> 503 with manual fallback instruction
    monkeypatch.setattr(WeatherProvider, "fetch_separated_weather", lambda *a, **kw: None)
    with pytest.raises(HTTPException) as exc_info:
        _run_coro_sync(get_rainfall_analysis(latitude=10.0889, longitude=77.0595))
    assert exc_info.value.status_code == 503

    # Manual fallback evaluation succeeds even when WeatherProvider is offline
    req = RainfallRiskEvaluateRequest(
        latitude=10.0889,
        longitude=77.0595,
        location_name="Offline Fallback Point",
        is_manual=True,
        slope_degrees=30.0,
        elevation_m=1400.0,
        rainfall_1h_mm=20.0,
        rainfall_24h_mm=95.0,
        rainfall_48h_mm=140.0,
        rainfall_72h_mm=180.0,
        soil_moisture=0.62,
        persist=True,
    )
    res = _run_coro_sync(evaluate_rainfall_risk(req))
    assert res["is_manual_input"] is True
    assert res["current_estimate"]["probability"] is not None
    assert res["available_rainfall_mm"]["rainfall_48h_mm"] == 140.0


def test_wmo_weather_condition_decoding_and_missing_field(tmp_path, monkeypatch):
    assert decode_wmo_weather_code(0) == "Clear Sky"
    assert decode_wmo_weather_code(63) == "Rain"
    assert decode_wmo_weather_code(95) == "Thunderstorm"
    assert decode_wmo_weather_code(None) == "Unavailable"

    monkeypatch.setattr(base_mod, "CACHE_DIR", tmp_path)
    provider = WeatherProvider()
    raw = _build_raw_openmeteo_payload(past_hours=6, forecast_hours=2)
    raw["current"]["weather_code"] = 63
    parsed_with_code = provider._parse_response(raw, 10.0, 77.0)
    assert parsed_with_code["current"]["weather_code"] == 63
    assert parsed_with_code["current"]["weather_condition"] == "Rain"

    # Missing weather_code must degrade cleanly to Unavailable, never invented
    raw["current"].pop("weather_code", None)
    parsed_no_code = provider._parse_response(raw, 10.0, 77.0)
    assert parsed_no_code["current"]["weather_code"] is None
    assert parsed_no_code["current"]["weather_condition"] == "Unavailable"


def test_invalid_coordinates_rejected_before_api_or_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "CACHE_DIR", tmp_path)
    provider = WeatherProvider()
    called = {"count": 0}

    def fail_if_called(*args, **kwargs):
        called["count"] += 1
        return {}

    monkeypatch.setattr(provider, "_get", fail_if_called)

    assert provider.fetch(95.0, 77.0) is None
    assert provider.last_fetch_metadata["status"] == "invalid_coordinates"
    assert provider.fetch(10.0, 200.0) is None
    assert provider.fetch(float("nan"), 77.0) is None
    assert provider.fetch_separated_weather(-91.0, 77.0) is None
    acc = provider.get_rainfall_accumulations(100.0, 77.0)
    assert acc["source"] == "invalid_coordinates"
    assert acc["rainfall_24h_mm"] is None
    assert called["count"] == 0


def test_rate_limit_429_handling_and_manual_refresh_bypass_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "CACHE_DIR", tmp_path)
    provider = WeatherProvider()
    provider.max_retries = 1

    # 1. HTTP 429 rate-limit -> ProviderError and graceful None with rate_limited status
    rl_resp = MagicMock()
    rl_resp.status_code = 429
    rl_resp.headers = {"Retry-After": "30"}
    monkeypatch.setattr(provider._session, "get", lambda *a, **kw: rl_resp)

    with pytest.raises(ProviderError, match="Rate limited.*429"):
        provider._get("https://api.open-meteo.com/v1/forecast")

    assert provider.fetch(10.0, 77.0, past_days=1, forecast_days=0) is None
    assert provider.last_fetch_metadata["status"] == "rate_limited"
    assert provider.last_fetch_metadata["rate_limited"] is True

    # 2. Live fetch writes cache; default fetch uses cache; bypass_cache=True triggers live fetch
    raw_v1 = _build_raw_openmeteo_payload(past_hours=6, forecast_hours=2, precip_val=1.5)
    raw_v2 = _build_raw_openmeteo_payload(past_hours=6, forecast_hours=2, precip_val=4.5)
    responses = [raw_v1, raw_v2]
    calls = {"count": 0}

    def fake_get(url, params=None):
        calls["count"] += 1
        return responses[calls["count"] - 1]

    monkeypatch.setattr(provider, "_get", fake_get)

    live_1 = provider.fetch_separated_weather(10.0, 77.0, past_days=1, forecast_days=1, include_archive_baseline=False)
    assert live_1 is not None and live_1["is_cached"] is False
    assert calls["count"] == 1

    cached_read = provider.fetch_separated_weather(10.0, 77.0, past_days=1, forecast_days=1, include_archive_baseline=False)
    assert cached_read is not None and cached_read["is_cached"] is True
    assert calls["count"] == 1  # No extra API call

    manual_refresh = provider.fetch_separated_weather(
        10.0,
        77.0,
        past_days=1,
        forecast_days=1,
        include_archive_baseline=False,
        bypass_cache=True,
    )
    assert manual_refresh is not None and manual_refresh["is_cached"] is False
    assert manual_refresh["provenance"]["connection_status"] == "live"
    assert calls["count"] == 2
    assert manual_refresh["historical_hourly"]["precipitation"][0] == pytest.approx(4.5)


def test_incomplete_forecast_data_flagged_strictly():
    hist_24 = {"precipitation": [2.0] * 24}
    partial_fcst = {"precipitation": [1.0, None, 3.0, 2.0] + [1.0] * 20}
    metrics = compute_rainfall_metrics(hist_24, forecast_hourly=partial_fcst)

    assert metrics["rainfall_24h_mm"] == pytest.approx(48.0)
    assert metrics["forecast_24h_mm"] is None
    assert metrics["forecast_total_mm"] is None
    assert metrics["forecast_is_complete"] is False
    assert metrics["forecast_missing_hours"] == 1
    assert metrics["forecast_valid_hours_count"] == 23
    assert "Incomplete" in metrics["forecast_data_status"]


