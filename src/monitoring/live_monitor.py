"""Live Risk Monitoring service connecting data providers, DB, RiskEngine, and AlertManager.

Handles scheduled or on-demand station refreshes, validates external API responses,
falls back cleanly to cached DB observations when offline, and surfaces explicit
data provenance and freshness metadata.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from src.alerts.alert_manager import AlertManager, get_alert_manager
from src.data_providers.terrain_provider import TerrainProvider
from src.data_providers.weather_provider import WeatherProvider
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting
from src.database.db import get_table_counts
from src.database.models import (
    get_alerts,
    get_environmental_history,
    get_latest_environmental_observation,
    get_latest_prediction,
    get_location,
    get_locations,
    get_observations,
    get_recent_environmental_observations,
    get_recent_predictions,
    get_recent_predictions_with_location,
    insert_environmental_observation,
    save_prediction,
    update_location_status,
    upsert_location,
)
from src.risk.risk_engine import RiskEngine

logger = get_logger("monitoring.live_monitor")

# Verified coordinates of landslide-prone Indian monitoring corridors.
# Note: These provide station coordinates and coarse terrain metadata ONLY;
# no weather measurements or landslide events are fabricated here.
DEFAULT_MONITORING_STATIONS: list[dict[str, Any]] = [
    {
        "name": "Munnar",
        "latitude": 10.0889,
        "longitude": 77.0595,
        "country": "India",
        "state_region": "Kerala",
        "elevation_m": 1532.0,
        "slope_degrees": 26.0,
    },
    {
        "name": "Wayanad",
        "latitude": 11.6854,
        "longitude": 76.1320,
        "country": "India",
        "state_region": "Kerala",
        "elevation_m": 1120.0,
        "slope_degrees": 24.0,
    },
    {
        "name": "Darjeeling",
        "latitude": 27.0410,
        "longitude": 88.2663,
        "country": "India",
        "state_region": "West Bengal",
        "elevation_m": 2042.0,
        "slope_degrees": 29.0,
    },
    {
        "name": "Dehradun",
        "latitude": 30.3165,
        "longitude": 78.0322,
        "country": "India",
        "state_region": "Uttarakhand",
        "elevation_m": 640.0,
        "slope_degrees": 18.0,
    },
    {
        "name": "Joshimath",
        "latitude": 30.5506,
        "longitude": 79.5660,
        "country": "India",
        "state_region": "Uttarakhand",
        "elevation_m": 1875.0,
        "slope_degrees": 31.0,
    },
]

DEFAULT_VEGETATION_INDEX_PROXY = 0.55
DEFAULT_SLOPE_PROXY_DEG = 15.0
DEFAULT_ELEVATION_PROXY_M = 500.0


def _is_finite_number(value: Any) -> bool:
    """Return True if value can be converted to a finite float."""
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _latest_valid(values: Any) -> float | None:
    """Return the last finite float in a sequence, or None."""
    if not isinstance(values, (list, tuple)):
        return None
    for item in reversed(values):
        if _is_finite_number(item):
            return float(item)
    return None


def _numeric_tail_sum(values: Any, window: int) -> float | None:
    """Sum valid non-negative precipitation values in the last `window` slots."""
    if not isinstance(values, (list, tuple)) or not values:
        return None
    tail = [max(0.0, float(v)) for v in values[-window:] if _is_finite_number(v)]
    if not tail:
        return None
    return round(sum(tail), 2)


def _is_valid_coordinate(latitude: Any, longitude: Any) -> bool:
    """Return True if latitude/longitude are finite WGS84 coordinates."""
    if not (_is_finite_number(latitude) and _is_finite_number(longitude)):
        return False
    lat = float(latitude)
    lon = float(longitude)
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def derive_weather_condition(
    rainfall_1h_mm: Any,
    humidity_pct: Any,
    temperature_c: Any,
    weather_code: Any = None,
) -> str:
    """Return an explicit weather condition description or 'Unavailable'."""
    from src.data_providers.weather_provider import decode_wmo_weather_code

    decoded = decode_wmo_weather_code(weather_code)
    if decoded != "Unavailable":
        return decoded

    rain = float(rainfall_1h_mm) if _is_finite_number(rainfall_1h_mm) else None
    hum = float(humidity_pct) if _is_finite_number(humidity_pct) else None
    temp = float(temperature_c) if _is_finite_number(temperature_c) else None

    if rain is None and hum is None and temp is None:
        return "Unavailable"
    if rain is not None:
        if rain >= 15.0:
            return "Heavy Rain (Derived)"
        if rain >= 2.5:
            return "Moderate Rain (Derived)"
        if rain > 0.2:
            return "Light Rain (Derived)"
    if hum is not None and hum >= 85.0:
        return "Humid / Overcast (Derived)"
    if temp is not None and temp <= 5.0:
        return "Cold / Dry (Derived)"
    return "Dry / Clear (Derived)"


def _parse_iso_utc(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp to timezone-aware UTC datetime."""
    if not value or not isinstance(value, str):
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _compute_freshness(
    observed_at: str | None,
    freshness_hours: float,
    feed_status: str | None = None,
) -> tuple[float | None, str]:
    """Compute observation age in hours and human-readable freshness label."""
    if feed_status == "unavailable":
        return None, "Unavailable"
    if not observed_at:
        return None, "Never Synced"

    dt = _parse_iso_utc(observed_at)
    if dt is None:
        return None, "Unknown"

    age_hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
    if feed_status == "offline_fallback":
        return round(age_hours, 2), "Offline Fallback"
    if age_hours <= freshness_hours:
        return round(age_hours, 2), "Fresh"
    return round(age_hours, 2), "Stale"


class LiveRiskMonitor:
    """Coordinates live weather ingestion, terrain metadata, risk inference, and alerts."""

    def __init__(
        self,
        weather_provider: WeatherProvider | None = None,
        terrain_provider: TerrainProvider | None = None,
        risk_engine: RiskEngine | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self.weather_provider = weather_provider or WeatherProvider()
        self.terrain_provider = terrain_provider or TerrainProvider()
        self.risk_engine = risk_engine or RiskEngine()
        self.alert_manager = alert_manager or get_alert_manager()
        self.freshness_hours = float(get_setting("risk", "data_freshness_hours", default=6))

    # ──────────────────────────────────────────
    # Location management
    # ──────────────────────────────────────────

    def seed_default_stations(self) -> list[int]:
        """Idempotently register default Indian landslide-prone monitoring stations."""
        location_ids: list[int] = []
        for station in DEFAULT_MONITORING_STATIONS:
            loc_id = upsert_location(
                name=station["name"],
                latitude=station["latitude"],
                longitude=station["longitude"],
                country=station["country"],
                state_region=station["state_region"],
                elevation_m=station["elevation_m"],
                slope_degrees=station["slope_degrees"],
                is_active=True,
            )
            location_ids.append(loc_id)
        logger.info("Seeded %d default monitoring stations.", len(location_ids))
        return location_ids

    def register_location(
        self,
        name: str,
        latitude: float,
        longitude: float,
        country: str | None = "India",
        state_region: str | None = None,
        elevation_m: float | None = None,
        slope_degrees: float | None = None,
        fetch_terrain: bool = True,
        is_active: bool = True,
    ) -> dict[str, Any]:
        """Validate and register a monitored location, optionally resolving DEM terrain."""
        if not name or not name.strip():
            raise ValueError("Location name must not be empty.")
        if not (-90.0 <= float(latitude) <= 90.0):
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        if not (-180.0 <= float(longitude) <= 180.0):
            raise ValueError("Longitude must be between -180 and 180 degrees.")

        terrain_warnings: list[str] = []
        if fetch_terrain and (elevation_m is None or slope_degrees is None):
            try:
                terrain = self.terrain_provider.get_terrain_features(
                    float(latitude), float(longitude)
                )
                if elevation_m is None and _is_finite_number(terrain.get("elevation_m")):
                    elevation_m = float(terrain["elevation_m"])
                if slope_degrees is None and _is_finite_number(terrain.get("slope_degrees")):
                    slope_degrees = float(terrain["slope_degrees"])
                if terrain.get("slope_warning"):
                    terrain_warnings.append(str(terrain["slope_warning"]))
            except Exception as exc:
                logger.warning("Terrain lookup failed for %s: %s", name, exc)
                terrain_warnings.append(
                    f"Terrain lookup unavailable ({exc}); local elevation/slope defaults may apply."
                )

        loc_id = upsert_location(
            name=name.strip(),
            latitude=float(latitude),
            longitude=float(longitude),
            country=country,
            state_region=state_region,
            elevation_m=elevation_m,
            slope_degrees=slope_degrees,
            is_active=is_active,
        )
        record = get_location(loc_id) or {"id": loc_id, "name": name.strip()}
        record["warnings"] = terrain_warnings
        return record

    def set_location_active(self, location_id: int, is_active: bool) -> bool:
        """Toggle whether a station is included in live refresh runs."""
        return update_location_status(location_id, is_active)

    # ──────────────────────────────────────────
    # External weather validation & extraction
    # ──────────────────────────────────────────

    @staticmethod
    def validate_and_extract_weather(weather_data: Any) -> dict[str, Any] | None:
        """Validate normalized WeatherProvider payload and compute key environmental metrics.

        Returns None if payload is malformed or contains no usable precipitation series.
        """
        if not isinstance(weather_data, dict):
            return None

        hourly = weather_data.get("hourly")
        if not isinstance(hourly, dict):
            return None

        precip = hourly.get("precipitation")
        if not isinstance(precip, list) or not any(_is_finite_number(v) for v in precip):
            return None

        times = hourly.get("time", [])
        utc_offset = int(weather_data.get("utc_offset_seconds") or 0)
        feed_tz = timezone(timedelta(seconds=utc_offset))
        now_utc = datetime.now(timezone.utc)

        # When forecast_days >= 1 is requested, filter out future forecast hours
        cutoff_idx = len(precip)
        if isinstance(times, list) and len(times) == len(precip) and times:
            past_indices: list[int] = []
            for idx, raw_t in enumerate(times):
                if not raw_t:
                    continue
                try:
                    dt = datetime.fromisoformat(str(raw_t).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=feed_tz)
                    if dt.astimezone(timezone.utc) <= now_utc:
                        past_indices.append(idx)
                except ValueError:
                    continue
            if past_indices:
                cutoff_idx = past_indices[-1] + 1

        precip_window = precip[:cutoff_idx]
        if not any(_is_finite_number(v) for v in precip_window):
            return None

        observed_at = None
        if isinstance(times, list) and times:
            for idx in range(min(len(times), cutoff_idx) - 1, -1, -1):
                if idx < len(precip_window) and _is_finite_number(precip_window[idx]) and times[idx]:
                    try:
                        dt = datetime.fromisoformat(str(times[idx]).replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=feed_tz)
                        observed_at = dt.astimezone(timezone.utc).isoformat()
                    except ValueError:
                        observed_at = None
                    break
        if not observed_at:
            observed_at = str(weather_data.get("fetched_at") or now_utc.isoformat())

        rainfall_1h = _latest_valid(precip_window)
        valid_24 = [
            max(0.0, float(v)) for v in precip_window[-24:] if _is_finite_number(v)
        ]
        valid_72 = [
            max(0.0, float(v)) for v in precip_window[-72:] if _is_finite_number(v)
        ]
        rainfall_24h = round(sum(valid_24), 2) if len(valid_24) >= 24 else None
        rainfall_72h = round(sum(valid_72), 2) if len(valid_72) >= 72 else None

        def _slice(key: str):
            seq = hourly.get(key)
            return seq[:cutoff_idx] if isinstance(seq, list) else seq

        temp_c = _latest_valid(_slice("temperature_2m"))
        humidity_pct = _latest_valid(_slice("relative_humidity_2m"))
        wind_speed_kmh = _latest_valid(_slice("wind_speed_10m"))
        soil_moisture = _latest_valid(_slice("soil_moisture_0_to_7cm"))
        if soil_moisture is not None:
            soil_moisture = round(min(max(soil_moisture, 0.0), 1.0), 4)

        elev = weather_data.get("elevation_m")
        elevation_m = float(elev) if _is_finite_number(elev) else None

        # Keep strictly future slots separated as explicit Forecast (never mixed with past reanalysis)
        future_precip = [
            max(0.0, float(v))
            for v in precip[cutoff_idx : cutoff_idx + 24]
            if _is_finite_number(v)
        ]
        forecast_summary = (
            {
                "is_forecast": True,
                "forecast_hours": len(future_precip),
                "forecast_total_rain_mm": round(sum(future_precip), 2),
                "forecast_max_hourly_mm": round(max(future_precip), 2),
            }
            if future_precip
            else None
        )

        return {
            "observed_at": observed_at,
            "rainfall_1h_mm": round(max(0.0, rainfall_1h), 2) if rainfall_1h is not None else None,
            "rainfall_24h_mm": rainfall_24h,
            "rainfall_72h_mm": rainfall_72h,
            "temperature_c": round(temp_c, 2) if temp_c is not None else None,
            "humidity_pct": round(min(max(humidity_pct, 0.0), 100.0), 1)
            if humidity_pct is not None
            else None,
            "wind_speed_kmh": round(max(0.0, wind_speed_kmh), 1)
            if wind_speed_kmh is not None
            else None,
            "soil_moisture": soil_moisture,
            "elevation_m": elevation_m,
            "forecast_summary": forecast_summary,
            "data_source": (
                f"{weather_data.get('source', 'Open-Meteo')} "
                "(model-derived reanalysis, not gauge observation)"
            ),
            "is_observed": False,
        }

    # ──────────────────────────────────────────
    # Live Refresh & Snapshot
    # ──────────────────────────────────────────

    def refresh_location(
        self,
        location_id: int,
        generate_alerts: bool = True,
    ) -> dict[str, Any]:
        """Fetch latest weather for a location, score risk, store history, and evaluate alerts."""
        loc = get_location(location_id)
        if not loc:
            raise ValueError(f"Location {location_id} not found.")

        lat = float(loc["latitude"])
        lon = float(loc["longitude"])
        station_warnings: list[str] = []
        feed_status = "live"

        raw_weather: dict | None = None
        try:
            raw_weather = self.weather_provider.fetch(
                lat, lon, past_days=3, forecast_days=1
            )
        except Exception as exc:
            logger.warning("Weather fetch error for %s: %s", loc["name"], exc)
            raw_weather = None

        extracted = self.validate_and_extract_weather(raw_weather)

        if extracted is not None:
            insert_environmental_observation(
                location_id=location_id,
                observed_at=extracted["observed_at"],
                rainfall_1h_mm=extracted["rainfall_1h_mm"],
                rainfall_24h_mm=extracted["rainfall_24h_mm"],
                rainfall_72h_mm=extracted["rainfall_72h_mm"],
                temperature_c=extracted["temperature_c"],
                humidity_pct=extracted["humidity_pct"],
                wind_speed_kmh=extracted.get("wind_speed_kmh"),
                soil_moisture=extracted["soil_moisture"],
                data_source=extracted["data_source"],
                is_observed=False,
            )
            env = extracted
            station_warnings.append(
                "Weather and soil moisture metrics are model-derived Open-Meteo reanalysis "
                "(not direct physical rain-gauge or in-situ sensor measurements)."
            )
        else:
            cached = get_latest_environmental_observation(location_id)
            if cached is not None:
                feed_status = "offline_fallback"
                env = dict(cached)
                station_warnings.append(
                    "Live Open-Meteo fetch failed or returned invalid data; "
                    f"using cached observation from {cached.get('observed_at', 'unknown time')}."
                )
            else:
                feed_status = "unavailable"
                env = {
                    "observed_at": None,
                    "rainfall_1h_mm": 0.0,
                    "rainfall_24h_mm": 0.0,
                    "rainfall_72h_mm": 0.0,
                    "temperature_c": None,
                    "humidity_pct": None,
                    "soil_moisture": 0.25,
                    "data_source": "unavailable (conservative dry fallback)",
                    "is_observed": False,
                }
                station_warnings.append(
                    "Weather feed unavailable and no cached environmental observation exists; "
                    "using conservative zero-rainfall fallback so station remains visible."
                )

        # Resolve terrain features cleanly with explicit disclosure if proxy used
        slope = loc.get("slope_degrees")
        if not _is_finite_number(slope):
            slope = DEFAULT_SLOPE_PROXY_DEG
            station_warnings.append(
                f"Station slope not measured; using default {DEFAULT_SLOPE_PROXY_DEG:.0f}° proxy."
            )

        elevation = loc.get("elevation_m")
        if not _is_finite_number(elevation):
            elevation = env.get("elevation_m")
        if not _is_finite_number(elevation):
            elevation = DEFAULT_ELEVATION_PROXY_M
            station_warnings.append(
                f"Station elevation unavailable; using default {DEFAULT_ELEVATION_PROXY_M:.0f}m proxy."
            )

        soil_moisture = env.get("soil_moisture")
        if not _is_finite_number(soil_moisture):
            soil_moisture = 0.25
            station_warnings.append(
                "Soil moisture unavailable from weather feed; using 0.25 baseline proxy."
            )

        station_warnings.append(
            f"Vegetation index (NDVI) uses regional proxy ({DEFAULT_VEGETATION_INDEX_PROXY:.2f}); "
            "live satellite NDVI ingestion is not yet connected."
        )

        features = {
            "slope_degrees": float(slope),
            "elevation_m": float(elevation),
            "rainfall_1h_mm": float(env["rainfall_1h_mm"]) if _is_finite_number(env.get("rainfall_1h_mm")) else None,
            "rainfall_24h_mm": float(env["rainfall_24h_mm"]) if _is_finite_number(env.get("rainfall_24h_mm")) else None,
            "rainfall_72h_mm": float(env["rainfall_72h_mm"]) if _is_finite_number(env.get("rainfall_72h_mm")) else None,
            "vegetation_index": DEFAULT_VEGETATION_INDEX_PROXY,
            "soil_moisture": float(soil_moisture),
        }

        obs_dt = _parse_iso_utc(env.get("observed_at"))
        risk_result = self.risk_engine.estimate_risk(
            features=features,
            observation_time=obs_dt,
            location_name=loc["name"],
        )

        combined_warnings = list(
            dict.fromkeys(station_warnings + list(risk_result.get("warnings", [])))
        )

        if risk_result.get("method") == "ml_model":
            model_name = getattr(self.risk_engine._predictor, "model_name", "best_model") or "best_model"
        else:
            model_name = str(risk_result.get("method", "heuristic_baseline"))

        prediction_id = save_prediction(
            probability=risk_result.get("probability"),
            risk_category=risk_result["risk_category"],
            model_name=model_name,
            method=risk_result["method"],
            input_features=features,
            warnings=combined_warnings,
            location_id=location_id,
            is_experimental=True,
        )
        risk_result["prediction_id"] = prediction_id

        alert_record = None
        if generate_alerts:
            try:
                alert_record = self.alert_manager.evaluate_and_alert(
                    risk_result,
                    location_id=location_id,
                    location_name=loc["name"],
                    latitude=lat,
                    longitude=lon,
                    supporting_env_info=features,
                    model_version=model_name,
                    data_source=env.get("data_source"),
                )
            except Exception as exc:
                logger.warning("Alert evaluation failed for %s: %s", loc["name"], exc)
                combined_warnings.append(f"Alert evaluation error: {exc}")

        age_hours, freshness_label = _compute_freshness(
            env.get("observed_at"), self.freshness_hours, feed_status=feed_status
        )

        return {
            "location_id": location_id,
            "name": loc["name"],
            "latitude": lat,
            "longitude": lon,
            "country": loc.get("country"),
            "state_region": loc.get("state_region"),
            "elevation_m": float(elevation),
            "slope_degrees": float(slope),
            "is_active": bool(loc.get("is_active", 1)),
            "feed_status": feed_status,
            "freshness_status": freshness_label,
            "data_age_hours": age_hours,
            "observed_at": env.get("observed_at"),
            "predicted_at": risk_result.get("estimated_at") or datetime.now(timezone.utc).isoformat(),
            "rainfall_1h_mm": features["rainfall_1h_mm"],
            "rainfall_24h_mm": features["rainfall_24h_mm"],
            "rainfall_72h_mm": features["rainfall_72h_mm"],
            "temperature_c": env.get("temperature_c"),
            "humidity_pct": env.get("humidity_pct"),
            "wind_speed_kmh": env.get("wind_speed_kmh"),
            "forecast_summary": env.get("forecast_summary"),
            "soil_moisture": features["soil_moisture"],
            "data_source": env.get("data_source"),
            "is_observed": False,
            "probability": risk_result.get("probability"),
            "risk_category": risk_result["risk_category"],
            "model_name": model_name,
            "method": risk_result["method"],
            "is_experimental": True,
            "prediction_id": prediction_id,
            "alert_triggered": alert_record is not None,
            "alert_id": alert_record.get("id") if alert_record else None,
            "warnings": combined_warnings,
        }

    def refresh_all(
        self,
        active_only: bool = True,
        generate_alerts: bool = True,
    ) -> dict[str, Any]:
        """Refresh all monitored stations and summarize outcomes."""
        loc_df = get_locations(active_only=active_only)
        stations: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        if not loc_df.empty:
            for row in loc_df.to_dict(orient="records"):
                loc_id = int(row["id"])
                try:
                    stations.append(
                        self.refresh_location(loc_id, generate_alerts=generate_alerts)
                    )
                except Exception as exc:
                    logger.error("Failed refreshing station %s (%d): %s", row.get("name"), loc_id, exc)
                    errors.append(
                        {
                            "location_id": loc_id,
                            "name": row.get("name"),
                            "error": str(exc),
                        }
                    )

        return {
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "total_stations": len(stations),
            "live_count": sum(1 for s in stations if s["feed_status"] == "live"),
            "fallback_count": sum(1 for s in stations if s["feed_status"] == "offline_fallback"),
            "unavailable_count": sum(1 for s in stations if s["feed_status"] == "unavailable"),
            "alerts_triggered": sum(1 for s in stations if s.get("alert_triggered")),
            "high_risk_count": sum(
                1 for s in stations if s.get("risk_category") in {"High", "Very High"}
            ),
            "stations": stations,
            "errors": errors,
            "is_experimental": True,
        }

    def get_live_snapshot(self, active_only: bool = True) -> dict[str, Any]:
        """Return latest DB-backed monitoring snapshot across stations without network calls."""
        loc_df = get_locations(active_only=active_only)
        alerts_df = get_alerts(include_acknowledged=False, limit=100)
        active_alert_loc_ids: set[int] = set()
        if not alerts_df.empty and "location_id" in alerts_df.columns:
            active_alert_loc_ids = {
                int(v) for v in alerts_df["location_id"].dropna().tolist()
            }

        stations: list[dict[str, Any]] = []
        if not loc_df.empty:
            for row in loc_df.to_dict(orient="records"):
                loc_id = int(row["id"])
                env = get_latest_environmental_observation(loc_id) or {}
                pred = get_latest_prediction(loc_id) or {}

                warnings_list: list[str] = []
                raw_warnings = pred.get("warnings")
                if isinstance(raw_warnings, str) and raw_warnings:
                    try:
                        parsed = json.loads(raw_warnings)
                        if isinstance(parsed, list):
                            warnings_list = [str(w) for w in parsed]
                    except json.JSONDecodeError:
                        warnings_list = [raw_warnings]

                feed_status = "live" if env.get("observed_at") else "unavailable"
                age_hours, freshness_label = _compute_freshness(
                    env.get("observed_at"),
                    self.freshness_hours,
                    feed_status=feed_status,
                )

                def _finite_or_none(val: Any) -> float | None:
                    return float(val) if _is_finite_number(val) else None

                r1h = _finite_or_none(env.get("rainfall_1h_mm"))
                temp_val = _finite_or_none(env.get("temperature_c"))
                hum_val = _finite_or_none(env.get("humidity_pct"))

                stations.append(
                    {
                        "location_id": loc_id,
                        "name": row["name"],
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "country": row.get("country"),
                        "state_region": row.get("state_region"),
                        "elevation_m": _finite_or_none(row.get("elevation_m")),
                        "slope_degrees": _finite_or_none(row.get("slope_degrees")),
                        "is_active": bool(row.get("is_active", 1)),
                        "feed_status": feed_status,
                        "freshness_status": freshness_label,
                        "data_age_hours": _finite_or_none(age_hours),
                        "observed_at": env.get("observed_at"),
                        "predicted_at": pred.get("predicted_at"),
                        "rainfall_1h_mm": r1h,
                        "rainfall_24h_mm": _finite_or_none(env.get("rainfall_24h_mm")),
                        "rainfall_72h_mm": _finite_or_none(env.get("rainfall_72h_mm")),
                        "temperature_c": temp_val,
                        "humidity_pct": hum_val,
                        "wind_speed_kmh": _finite_or_none(env.get("wind_speed_kmh")),
                        "soil_moisture": _finite_or_none(env.get("soil_moisture")),
                        "weather_condition": derive_weather_condition(
                            r1h,
                            hum_val,
                            temp_val,
                        ),
                        "data_source": env.get("data_source"),
                        "is_observed": bool(env.get("is_observed", 0)),
                        "probability": _finite_or_none(pred.get("probability")),
                        "risk_category": pred.get("risk_category") or "Not Evaluated",
                        "model_name": pred.get("model_name"),
                        "method": pred.get("method"),
                        "is_experimental": bool(pred.get("is_experimental", 1)),
                        "has_active_alert": loc_id in active_alert_loc_ids,
                        "warnings": warnings_list,
                    }
                )

        dist = compute_risk_distribution(stations)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_stations": len(stations),
            "high_risk_count": dist["counts"]["High"] + dist["counts"]["Very High"],
            "high_risk_only_count": dist["counts"]["High"],
            "very_high_risk_count": dist["counts"]["Very High"],
            "risk_distribution": dist,
            "stale_or_unsynced_count": sum(
                1 for s in stations if s.get("freshness_status") != "Fresh"
            ),
            "active_alerts_count": int(len(alerts_df)) if not alerts_df.empty else 0,
            "stations": stations,
            "is_experimental": True,
        }

    def get_dashboard_summary(
        self,
        dataset_df=None,
        model_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compile verified Phase 1 dashboard overview metrics from DB, dataset, and model metadata."""
        snapshot = self.get_live_snapshot(active_only=True)
        stations = snapshot["stations"]
        counts = get_table_counts()
        db_obs_count = max(0, int(counts.get("observations", 0)))

        dataset_rows = 0
        dataset_events = 0
        dataset_status = "Missing"
        is_synth: bool | None = None
        if dataset_df is not None and not getattr(dataset_df, "empty", True):
            dataset_rows = int(len(dataset_df))
            if "landslide_occurred" in dataset_df.columns:
                dataset_events = int(dataset_df["landslide_occurred"].fillna(0).sum())
            is_synth = (
                bool(dataset_df["is_synthetic"].all())
                if "is_synthetic" in dataset_df.columns
                else True
            )
            dataset_status = (
                f"Synthetic Dev ({dataset_rows:,} rows)"
                if is_synth
                else f"Authentic ({dataset_rows:,} rows)"
            )

        all_obs_df = get_observations(include_synthetic=True, limit=10000)
        auth_obs_df = get_observations(include_synthetic=False, limit=10000)
        auth_obs_count = int(len(auth_obs_df)) if not auth_obs_df.empty else 0
        synth_obs_count = max(0, db_obs_count - auth_obs_count)

        historical_total = db_obs_count if db_obs_count > 0 else dataset_events

        assessed_stations = [
            s for s in stations if s.get("risk_category") not in (None, "Not Evaluated", "Unknown")
        ]

        observed_times = [s["observed_at"] for s in stations if s.get("observed_at")]
        latest_weather_update = max(observed_times) if observed_times else None

        freshness_values = [s.get("freshness_status") for s in stations]
        if not stations:
            overall_freshness = "No Stations"
        elif all(v == "Fresh" for v in freshness_values):
            overall_freshness = "Fresh"
        elif any(v == "Fresh" for v in freshness_values):
            overall_freshness = "Partial Stale"
        elif any(v == "Offline Fallback" for v in freshness_values):
            overall_freshness = "Offline Fallback"
        elif any(v == "Stale" for v in freshness_values):
            overall_freshness = "Stale"
        else:
            overall_freshness = "Never Synced"

        meta = (model_info or {}).get("metadata", {}) if isinstance(model_info, dict) else {}
        raw_alg = str(meta.get("algorithm") or "").strip()
        model_name = raw_alg.replace("_", " ").title() if raw_alg else "Heuristic Baseline Only"
        saved_at = str(meta.get("saved_at") or "").strip()
        model_version = saved_at[:10] if saved_at else ("Active" if raw_alg else "Fallback")

        dist_counts = snapshot["risk_distribution"]["counts"]
        return {
            "total_historical_observations": historical_total,
            "db_observations_count": db_obs_count,
            "authentic_observations_count": auth_obs_count,
            "synthetic_observations_count": synth_obs_count,
            "dataset_rows_count": dataset_rows,
            "dataset_events_count": dataset_events,
            "dataset_is_synthetic": is_synth,
            "dataset_status": dataset_status,
            "locations_assessed": len(assessed_stations),
            "total_active_stations": len(stations),
            "low_risk_count": dist_counts["Low"],
            "moderate_risk_count": dist_counts["Moderate"],
            "high_risk_count": snapshot["high_risk_only_count"],
            "very_high_risk_count": snapshot["very_high_risk_count"],
            "combined_high_risk_count": dist_counts["High"] + dist_counts["Very High"],
            "latest_weather_update": latest_weather_update,
            "active_alerts_count": snapshot["active_alerts_count"],
            "model_name": model_name,
            "model_version": model_version,
            "overall_freshness": overall_freshness,
            "risk_distribution": snapshot["risk_distribution"],
            "risk_by_region": compute_risk_by_region(stations),
            "historical_distribution": compute_historical_distribution(
                dataset_df=dataset_df,
                db_obs_df=all_obs_df,
            ),
            "snapshot": snapshot,
        }

    def get_recent_activity(self, limit: int = 10) -> dict[str, Any]:
        """Return actual recent DB records for observations, assessments, alerts, and weather feeds."""
        snapshot = self.get_live_snapshot(active_only=True)
        updated_stations = sorted(
            [s for s in snapshot["stations"] if s.get("observed_at")],
            key=lambda s: str(s.get("observed_at") or ""),
            reverse=True,
        )[:limit]
        return {
            "recent_observations": get_observations(include_synthetic=True, limit=limit),
            "recent_authentic_observations": get_observations(include_synthetic=False, limit=limit),
            "recent_predictions": get_recent_predictions_with_location(limit=limit),
            "recent_alerts": get_alerts(include_acknowledged=True, limit=limit),
            "recent_weather_updates": get_recent_environmental_observations(limit=limit),
            "recently_updated_locations": updated_stations,
        }

    def get_station_history(self, location_id: int, limit: int = 50) -> dict[str, Any]:
        """Return environmental observations and prediction history for one location."""
        loc = get_location(location_id)
        if not loc:
            raise ValueError(f"Location {location_id} not found.")

        env_df = get_environmental_history(location_id, limit=limit)
        pred_df = get_recent_predictions(location_id=location_id, limit=limit)
        return {
            "location": loc,
            "environmental_history": env_df,
            "prediction_history": pred_df,
        }


def compute_risk_distribution(stations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute counts and percentages across standard risk tiers for monitored stations."""
    categories = ["Low", "Moderate", "High", "Very High", "Not Evaluated"]
    counts = {cat: 0 for cat in categories}
    for station in stations:
        cat = str(station.get("risk_category") or "Not Evaluated")
        if cat in counts:
            counts[cat] += 1
        else:
            counts["Not Evaluated"] += 1

    total = len(stations)
    percentages = {
        cat: round((cnt / total) * 100.0, 1) if total > 0 else 0.0
        for cat, cnt in counts.items()
    }
    return {"total": total, "counts": counts, "percentages": percentages}


def compute_risk_by_region(stations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate station counts by state_region and risk category, filtering invalid coordinates."""
    categories = ["Low", "Moderate", "High", "Very High", "Not Evaluated"]
    buckets: dict[str, dict[str, int]] = {}
    for station in stations:
        if "latitude" in station and "longitude" in station:
            if not _is_valid_coordinate(station.get("latitude"), station.get("longitude")):
                continue
        region = str(station.get("state_region") or "").strip() or "Unspecified Region"
        cat = str(station.get("risk_category") or "Not Evaluated")
        if cat not in categories:
            cat = "Not Evaluated"
        row = buckets.setdefault(region, {c: 0 for c in categories})
        row[cat] += 1

    result: list[dict[str, Any]] = []
    for region, counts in buckets.items():
        total = sum(counts.values())
        result.append({"state_region": region, **counts, "total": total})
    result.sort(key=lambda item: (-item["total"], item["state_region"]))
    return result


def compute_historical_distribution(
    dataset_df=None,
    db_obs_df=None,
) -> dict[str, Any]:
    """Compute historical landslide breakdown across authentic DB or dataset records."""
    import pandas as pd

    source_df = None
    source_label = "Unavailable"
    is_synthetic: bool | None = None

    has_db = db_obs_df is not None and not getattr(db_obs_df, "empty", True)
    has_ds = dataset_df is not None and not getattr(dataset_df, "empty", True)

    if has_db:
        source_df = db_obs_df.copy()
        if "is_synthetic" in source_df.columns:
            auth_only = source_df[source_df["is_synthetic"].fillna(1).astype(int) == 0]
            if not auth_only.empty:
                source_df = auth_only
                source_label = "Authentic DB Records"
                is_synthetic = False
            elif bool((source_df["is_synthetic"].fillna(1).astype(int) == 1).all()):
                source_label = "Synthetic DB Records"
                is_synthetic = True
            else:
                source_label = "Mixed DB Records"
                is_synthetic = True
        else:
            source_label = "DB Records"
    elif has_ds:
        source_df = dataset_df.copy()
        if "is_synthetic" in source_df.columns:
            auth_only = source_df[source_df["is_synthetic"].fillna(1).astype(int) == 0]
            if not auth_only.empty:
                source_df = auth_only
                source_label = "Authentic Dataset"
                is_synthetic = False
            else:
                source_label = "Synthetic Development Dataset"
                is_synthetic = True
        else:
            source_label = "Development Dataset"
            is_synthetic = True

    if source_df is None or source_df.empty:
        return {
            "available": False,
            "source_label": "Unavailable",
            "is_synthetic": None,
            "total_records": 0,
            "valid_records": 0,
            "dropped_invalid_count": 0,
            "landslide_events": 0,
            "non_events": 0,
            "by_region": [],
            "by_trigger": [],
        }

    total_records = int(len(source_df))
    if "latitude" in source_df.columns and "longitude" in source_df.columns:
        valid_mask = source_df.apply(
            lambda r: _is_valid_coordinate(r.get("latitude"), r.get("longitude")),
            axis=1,
        )
        valid_df = source_df[valid_mask].copy()
    else:
        valid_df = source_df.copy()

    dropped = max(0, total_records - int(len(valid_df)))
    if valid_df.empty:
        return {
            "available": False,
            "source_label": source_label,
            "is_synthetic": is_synthetic,
            "total_records": total_records,
            "valid_records": 0,
            "dropped_invalid_count": dropped,
            "landslide_events": 0,
            "non_events": 0,
            "by_region": [],
            "by_trigger": [],
        }

    if "landslide_occurred" in valid_df.columns:
        occurred = pd.to_numeric(valid_df["landslide_occurred"], errors="coerce").fillna(0).astype(int)
        valid_df["_occurred"] = (occurred > 0).astype(int)
    else:
        valid_df["_occurred"] = 1

    events_count = int(valid_df["_occurred"].sum())
    non_events_count = int(len(valid_df) - events_count)

    region_col = (
        "state_region"
        if "state_region" in valid_df.columns
        else ("region" if "region" in valid_df.columns else None)
    )
    by_region: list[dict[str, Any]] = []
    if region_col:
        grouped = (
            valid_df.assign(_region=valid_df[region_col].fillna("Unspecified").astype(str).str.strip().replace("", "Unspecified"))
            .groupby("_region", dropna=False)
            .agg(total_records=("_occurred", "count"), landslide_events=("_occurred", "sum"))
            .reset_index()
        )
        grouped["non_events"] = grouped["total_records"] - grouped["landslide_events"]
        grouped = grouped.sort_values(["landslide_events", "total_records", "_region"], ascending=[False, False, True])
        by_region = [
            {
                "region": str(row["_region"]),
                "total_records": int(row["total_records"]),
                "landslide_events": int(row["landslide_events"]),
                "non_events": int(row["non_events"]),
            }
            for _, row in grouped.iterrows()
        ]

    by_trigger: list[dict[str, Any]] = []
    if "trigger" in valid_df.columns:
        trig_df = valid_df[valid_df["_occurred"] == 1].copy()
        if not trig_df.empty:
            trig_series = (
                trig_df["trigger"]
                .fillna("Unspecified")
                .astype(str)
                .str.strip()
                .replace("", "Unspecified")
            )
            trig_counts = trig_series.value_counts()
            by_trigger = [
                {"trigger": str(k), "count": int(v)} for k, v in trig_counts.items()
            ]

    return {
        "available": True,
        "source_label": source_label,
        "is_synthetic": is_synthetic,
        "total_records": total_records,
        "valid_records": int(len(valid_df)),
        "dropped_invalid_count": dropped,
        "landslide_events": events_count,
        "non_events": non_events_count,
        "by_region": by_region,
        "by_trigger": by_trigger,
    }


def get_live_monitor() -> LiveRiskMonitor:
    """Create a configured LiveRiskMonitor instance."""
    return LiveRiskMonitor()
