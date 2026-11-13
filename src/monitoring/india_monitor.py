"""India State & District Regional Monitoring Orchestrator for LandslideGuard.

Connects:
- IndiaBoundaryProvider (geoBoundaries IND ADM1 / ADM2: 36 States/UTs -> 735 Districts)
- WeatherProvider (Open-Meteo coordinate-level current, historical, forecast & accumulations)
- IMDProvider (Official IMD API when IMD_API_KEY is configured)
- SACHETProvider (Official NDMA SACHET CAP JSON / RSS alerts)
- TerrainProvider (SRTM elevation & slope via Open-Elevation)
- RiskEngine & SHAP Explainability
- AlertManager (strictly experimental ML alerts, separated from official warnings)
- SQLite persistence (locations, environmental_observations, prediction_records,
  official_warnings, provider_status)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.alerts.alert_manager import AlertManager, get_alert_manager
from src.data_providers.imd_provider import IMDProvider
from src.data_providers.india_boundaries import IndiaBoundaryProvider, normalize_admin_name
from src.data_providers.sachet_provider import SACHETProvider
from src.data_providers.terrain_provider import TerrainProvider
from src.data_providers.weather_provider import WeatherProvider, parse_api_timestamp
from src.database.models import (
    get_observations,
    get_provider_statuses,
    insert_environmental_observation_dedup,
    save_prediction,
    upsert_location,
    upsert_official_warning,
    upsert_provider_status,
)
from src.geospatial.spatial_analysis import filter_by_bounding_box, haversine_distance
from src.models.explainability import explain_prediction
from src.risk.risk_engine import RiskEngine
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting
from src.utils.validation import validate_coordinates

logger = get_logger("monitoring.india_monitor")


def _compute_freshness(
    timestamp_str: str | None,
    stale_after_hours: float = 6.0,
) -> dict[str, Any]:
    """Compute human-readable freshness state from an ISO timestamp."""
    if not timestamp_str:
        return {
            "timestamp": None,
            "is_valid": False,
            "is_stale": True,
            "age_minutes": None,
            "label": "Unknown timestamp",
        }
    dt = parse_api_timestamp(str(timestamp_str))
    if dt is None:
        return {
            "timestamp": str(timestamp_str),
            "is_valid": False,
            "is_stale": True,
            "age_minutes": None,
            "label": "Invalid timestamp",
        }
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_sec = max(0.0, (now - dt).total_seconds())
    age_min = round(age_sec / 60.0, 1)
    is_stale = (age_sec / 3600.0) > float(stale_after_hours)
    if age_min < 60:
        age_text = f"{age_min:.0f} min ago"
    else:
        age_text = f"{age_sec / 3600.0:.1f} h ago"
    return {
        "timestamp": dt.isoformat(),
        "is_valid": True,
        "is_stale": is_stale,
        "age_minutes": age_min,
        "label": f"{'Stale' if is_stale else 'Fresh'} ({age_text})",
    }


def _finite_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class IndiaRegionalMonitor:
    """Orchestrates end-to-end India State/District environmental monitoring & risk estimation."""

    def __init__(
        self,
        boundary_provider: IndiaBoundaryProvider | None = None,
        weather_provider: WeatherProvider | None = None,
        imd_provider: IMDProvider | None = None,
        sachet_provider: SACHETProvider | None = None,
        terrain_provider: TerrainProvider | None = None,
        risk_engine: RiskEngine | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        cfg = get_setting("india_monitoring") or {}
        b_dir = cfg.get("boundaries_dir")
        self.boundaries = boundary_provider or IndiaBoundaryProvider(boundaries_dir=b_dir)
        self.weather = weather_provider or WeatherProvider()
        self.imd = imd_provider or IMDProvider()
        self.sachet = sachet_provider or SACHETProvider()
        self.terrain = terrain_provider or TerrainProvider()
        self.risk_engine = risk_engine or RiskEngine()
        self.alert_manager = alert_manager or get_alert_manager()

    def filter_regional_observations(
        self,
        state_name: str,
        district_lat: float,
        district_lon: float,
        bbox: list[float] | None = None,
        radius_km: float = 75.0,
    ) -> dict[str, Any]:
        """Load historical landslide observations near the selected State/District."""
        obs_df = get_observations(limit=5000, include_synthetic=True)
        if obs_df is None or obs_df.empty:
            return {
                "observations_df": pd.DataFrame(),
                "total_count": 0,
                "synthetic_count": 0,
                "authentic_count": 0,
                "filter_method": "no_observations_in_db",
            }

        df = obs_df.copy()
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])

        norm_state = normalize_admin_name(state_name).lower()
        state_mask = pd.Series(False, index=df.index)
        if "state_region" in df.columns:
            state_mask = df["state_region"].astype(str).str.strip().str.lower().eq(norm_state)

        # Nearby distance filter from representative district coordinates
        distances = df.apply(
            lambda r: haversine_distance(
                float(district_lat),
                float(district_lon),
                float(r["latitude"]),
                float(r["longitude"]),
            ),
            axis=1,
        )
        df["distance_km"] = distances.round(2)
        nearby_mask = distances <= float(radius_km)

        bbox_mask = pd.Series(False, index=df.index)
        if bbox and len(bbox) == 4:
            in_bbox = filter_by_bounding_box(df, bbox[0], bbox[2], bbox[1], bbox[3])
            if not in_bbox.empty:
                bbox_mask.loc[in_bbox.index] = True

        regional = df[state_mask | nearby_mask | bbox_mask].copy()
        regional = regional.sort_values(by="distance_km", ascending=True)
        syn_count = int(regional["is_synthetic"].fillna(0).astype(int).sum()) if "is_synthetic" in regional.columns and not regional.empty else 0
        tot = int(len(regional))
        return {
            "observations_df": regional,
            "total_count": tot,
            "synthetic_count": syn_count,
            "authentic_count": tot - syn_count,
            "filter_method": f"state='{state_name}' OR distance<={radius_km:.0f}km OR district_bbox",
        }

    def run_regional_assessment(
        self,
        state_name: str,
        district_name: str,
        include_terrain: bool = True,
        include_shap: bool = True,
        persist: bool = True,
        generate_experimental_alert: bool = True,
        stale_after_hours: float = 6.0,
    ) -> dict[str, Any]:
        """Execute full 10-step India State/District regional assessment workflow."""
        now_utc = datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()

        # Step 1: Resolve State -> District coordinates & boundaries
        boundary_status = self.boundaries.get_boundary_status()
        district_meta = self.boundaries.resolve_location(state_name, district_name)
        if district_meta is None:
            raise ValueError(f"District '{district_name}' not found under state '{state_name}'.")

        lat = float(district_meta["rep_lat"])
        lon = float(district_meta["rep_lon"])
        coord_validation = validate_coordinates(lat, lon)
        if not coord_validation.is_valid:
            raise ValueError(
                f"Invalid resolved coordinates ({lat}, {lon}) for {district_name}: "
                + "; ".join(coord_validation.errors)
            )

        canonical_state = district_meta["state_name"]
        canonical_district = district_meta["district_name"]
        bbox = district_meta.get("bbox")
        state_geojson = self.boundaries.get_state_geojson(canonical_state)
        district_geojson = self.boundaries.get_district_geojson(canonical_state, canonical_district)

        # Record Geographical Boundaries status
        if persist:
            upsert_provider_status(
                provider_name="Geographical Boundaries",
                status=boundary_status.get("status", "Unavailable"),
                is_live=False,
                is_cached=True,
                last_success_at=boundary_status.get("last_updated"),
                last_attempt_at=now_iso,
                data_timestamp=boundary_status.get("last_updated"),
                attribution=boundary_status.get("source"),
                message=f"{boundary_status.get('state_count', 0)} states/UTs, {boundary_status.get('district_count', 0)} districts",
            )

        # Step 2: Open-Meteo Coordinate-Level Weather & Accumulations
        separated_weather = None
        accumulations: dict[str, Any] | None = None
        weather_error: str | None = None
        try:
            separated_weather = self.weather.fetch_separated_weather(lat, lon, past_days=3, forecast_days=3)
            accumulations = self.weather.get_rainfall_accumulations(lat, lon)
        except Exception as exc:
            weather_error = str(exc)
            logger.warning("[IndiaMonitor] Open-Meteo error for %s: %s", canonical_district, exc)

        weather_status_str = "Unavailable"
        weather_cached = False
        weather_ts = None
        current_wx: dict[str, Any] = {}
        hist_daily: dict[str, Any] = {}
        fcst_daily: dict[str, Any] = {}
        hist_hourly: dict[str, Any] = {}
        fcst_hourly: dict[str, Any] = {}
        soil_moisture_latest = None

        if separated_weather:
            weather_cached = bool(separated_weather.get("is_cached", False))
            weather_status_str = "Connected (Cached)" if weather_cached else "Connected"
            current_wx = separated_weather.get("current") or {}
            weather_ts = (
                current_wx.get("timestamp_utc")
                or current_wx.get("time")
                or separated_weather.get("fetched_at")
            )
            hist_hourly = separated_weather.get("historical_hourly") or {}
            fcst_hourly = separated_weather.get("forecast_hourly") or {}
            hist_daily = separated_weather.get("daily_historical") or {}
            fcst_daily = separated_weather.get("daily_forecast") or {}

            soil_series = hist_hourly.get("soil_moisture_0_to_7cm") or []
            valid_soils = [s for s in soil_series if _finite_or_none(s) is not None]
            if valid_soils:
                soil_moisture_latest = float(valid_soils[-1])

        weather_freshness = _compute_freshness(weather_ts, stale_after_hours=stale_after_hours)
        if separated_weather and not weather_freshness["is_valid"]:
            weather_status_str = "Degraded (Invalid Timestamp)"

        if persist:
            upsert_provider_status(
                provider_name="Open-Meteo Weather",
                status="Connected" if separated_weather else "Unavailable",
                is_live=bool(separated_weather and not weather_cached),
                is_cached=weather_cached,
                last_success_at=now_iso if separated_weather else None,
                last_attempt_at=now_iso,
                data_timestamp=weather_freshness.get("timestamp") or weather_ts,
                attribution=self.weather.attribution,
                message=weather_error or (
                    f"Point weather at ({lat:.4f}, {lon:.4f}) - cached={weather_cached}"
                    if separated_weather
                    else "Open-Meteo unavailable"
                ),
            )

        # Step 3: Official IMD Integration
        imd_data = self.imd.fetch(state_name=canonical_state, district_name=canonical_district)
        if persist:
            upsert_provider_status(
                provider_name="IMD (India Meteorological Department)",
                status=imd_data.get("status", "Not configured"),
                is_live=bool(imd_data.get("status") == "Connected" and not imd_data.get("is_cached")),
                is_cached=bool(imd_data.get("is_cached", False)),
                last_success_at=now_iso if imd_data.get("status") == "Connected" else None,
                last_attempt_at=now_iso,
                data_timestamp=imd_data.get("last_updated"),
                attribution=imd_data.get("attribution"),
                message=imd_data.get("message") or imd_data.get("status"),
            )

        # Step 4: Official NDMA SACHET CAP Alerts
        sachet_data = self.sachet.fetch(state_name=canonical_state, district_name=canonical_district)
        if persist:
            upsert_provider_status(
                provider_name="NDMA SACHET",
                status=sachet_data.get("status", "Link only"),
                is_live=bool(sachet_data.get("status") == "Connected" and not sachet_data.get("is_cached")),
                is_cached=bool(sachet_data.get("is_cached", False)),
                last_success_at=now_iso if sachet_data.get("status") == "Connected" else None,
                last_attempt_at=now_iso,
                data_timestamp=sachet_data.get("last_updated"),
                attribution=sachet_data.get("attribution"),
                message=sachet_data.get("error") or f"{len(sachet_data.get('regional_alerts', []))} regional alerts ({sachet_data.get('total_india_alerts', 0)} total India)",
            )
            for alert in sachet_data.get("regional_alerts", []):
                upsert_official_warning(
                    provider="NDMA SACHET",
                    external_id=str(alert.get("identifier") or f"sachet_{hash(str(alert))}"),
                    state_name=canonical_state,
                    district_name=canonical_district,
                    headline=alert.get("warning_message") or alert.get("headline"),
                    disaster_type=alert.get("alert_title") or alert.get("disaster_type"),
                    severity=alert.get("severity"),
                    severity_color=alert.get("severity_color"),
                    area_description=alert.get("affected_region") or alert.get("area_description"),
                    effective_from=alert.get("issued_at") or alert.get("effective_from"),
                    effective_until=alert.get("valid_until") or alert.get("effective_until"),
                    issued_at=alert.get("issued_at") or sachet_data.get("last_updated"),
                    sender_org=alert.get("alert_source") or alert.get("sender_org"),
                    source_url=alert.get("official_url") or alert.get("source_url"),
                    raw_payload=alert,
                )

        # Step 5: Historical Landslide Observations in State / Region
        regional_hist = self.filter_regional_observations(
            state_name=canonical_state,
            district_lat=lat,
            district_lon=lon,
            bbox=bbox,
        )

        # Step 6: Terrain lookup (without fabricating missing slope/elevation)
        terrain_data: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "elevation_m": None,
            "slope_degrees": None,
            "terrain_source": self.terrain.attribution,
        }
        if include_terrain:
            try:
                terrain_data = self.terrain.get_terrain_features(lat, lon)
            except Exception as exc:
                logger.warning("[IndiaMonitor] Terrain lookup failed: %s", exc)
                terrain_data["terrain_error"] = str(exc)

        # Step 7: Assemble strict experimental feature dictionary (NEVER zero-fill missing rainfall/terrain)
        acc = accumulations or {}
        r1h = _finite_or_none(current_wx.get("precipitation_mm"))
        if r1h is None:
            r1h = _finite_or_none(current_wx.get("precipitation"))
        if r1h is None:
            r1h = _finite_or_none(acc.get("rainfall_1h_mm"))
        r24h = _finite_or_none(acc.get("rainfall_24h_mm"))
        r72h = _finite_or_none(acc.get("rainfall_72h_mm"))
        temp_c = _finite_or_none(current_wx.get("temperature_2m_c"))
        if temp_c is None:
            temp_c = _finite_or_none(current_wx.get("temperature_2m"))
        humidity_pct = _finite_or_none(current_wx.get("relative_humidity_2m_pct"))
        if humidity_pct is None:
            humidity_pct = _finite_or_none(current_wx.get("relative_humidity_2m"))
        wind_kmh = _finite_or_none(current_wx.get("wind_speed_10m_kmh"))
        if wind_kmh is None:
            wind_kmh = _finite_or_none(current_wx.get("wind_speed_10m"))

        features: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "elevation_m": _finite_or_none(terrain_data.get("elevation_m")),
            "slope_degrees": _finite_or_none(terrain_data.get("slope_degrees")),
            "rainfall_1h_mm": r1h,
            "rainfall_24h_mm": r24h,
            "rainfall_72h_mm": r72h,
            "temperature_c": temp_c,
            "humidity_pct": humidity_pct,
            "wind_speed_kmh": wind_kmh,
            "soil_moisture": _finite_or_none(soil_moisture_latest),
        }

        obs_dt = parse_api_timestamp(str(weather_ts)) if weather_ts else None
        risk_result = self.risk_engine.estimate_risk(
            features=features,
            observation_time=obs_dt,
            location_name=f"{canonical_district}, {canonical_state}",
        )
        if separated_weather and not weather_freshness["is_valid"]:
            risk_result.setdefault("warnings", []).append(
                f"Weather timestamp missing or invalid ({weather_ts!r}); freshness could not be verified."
            )

        # Step 8: SHAP explanation when ML model is compatible & loaded
        shap_explanation: dict[str, Any] | None = None
        if include_shap:
            try:
                shap_explanation = explain_prediction(features, model_name="best_model")
            except Exception as exc:
                logger.warning("[IndiaMonitor] SHAP explanation skipped: %s", exc)
                shap_explanation = {
                    "available": False,
                    "shap_available": False,
                    "human_readable_summary": f"SHAP explanation unavailable ({exc}).",
                }

        # Step 9 & 10: SQLite Persistence (Location, Environmental Observation, Prediction, Experimental Alert)
        location_id: int | None = None
        prediction_id: int | None = None
        experimental_alert: dict[str, Any] | None = None
        if persist:
            location_id = upsert_location(
                name=f"{canonical_district}, {canonical_state}",
                latitude=lat,
                longitude=lon,
                country="India",
                state_region=canonical_state,
                elevation_m=features["elevation_m"],
                slope_degrees=features["slope_degrees"],
                is_active=True,
            )
            obs_ts_to_save = weather_freshness.get("timestamp") or now_iso
            if separated_weather:
                insert_environmental_observation_dedup(
                    location_id=location_id,
                    observed_at=obs_ts_to_save,
                    rainfall_1h_mm=r1h,
                    rainfall_24h_mm=r24h,
                    rainfall_72h_mm=r72h,
                    temperature_c=temp_c,
                    humidity_pct=humidity_pct,
                    wind_speed_kmh=wind_kmh,
                    soil_moisture=features["soil_moisture"],
                    data_source=f"Open-Meteo Coordinate Estimate ({'Cached' if weather_cached else 'Live'})",
                    is_observed=False,
                )
            prediction_id = save_prediction(
                probability=risk_result.get("probability"),
                risk_category=risk_result.get("risk_category", "Unknown"),
                model_name=risk_result.get("method", "unknown"),
                method=risk_result.get("method", "unknown"),
                input_features=features,
                warnings=risk_result.get("warnings"),
                location_id=location_id,
                is_experimental=True,
            )
            if generate_experimental_alert:
                experimental_alert = self.alert_manager.evaluate_and_alert(
                    risk_result=risk_result,
                    location_id=location_id,
                    location_name=f"{canonical_district}, {canonical_state}",
                    latitude=lat,
                    longitude=lon,
                    supporting_env_info=features,
                    model_version=risk_result.get("method", "unknown"),
                    data_source=f"India Regional Monitor ({canonical_state} / {canonical_district})",
                )

        # Build concise point-coordinate weather summary
        weather_summary = {
            "scope_note": (
                f"Point-coordinate Open-Meteo model estimate at ({lat:.4f}°N, {lon:.4f}°E) "
                f"representing {canonical_district}, {canonical_state}. NOT an area-wide station gauge."
            ),
            "status": weather_status_str,
            "is_cached": weather_cached,
            "attribution": self.weather.attribution,
            "timestamp": weather_ts,
            "freshness": weather_freshness,
            "temperature_c": temp_c,
            "humidity_pct": humidity_pct,
            "wind_speed_kmh": wind_kmh,
            "current_precip_mm": r1h,
            "rainfall_24h_mm": r24h,
            "rainfall_48h_mm": _finite_or_none(acc.get("rainfall_48h_mm")),
            "rainfall_72h_mm": r72h,
            "soil_moisture": features["soil_moisture"],
            "historical_hourly": hist_hourly,
            "forecast_hourly": fcst_hourly,
            "daily_historical": hist_daily,
            "daily_forecast": fcst_daily,
            "error": weather_error,
        }

        provider_panel = [
            {
                "provider": "Geographical Boundaries",
                "status": boundary_status.get("status", "Unavailable"),
                "is_live": False,
                "is_cached": True,
                "last_updated": boundary_status.get("last_updated"),
                "attribution": boundary_status.get("source"),
            },
            {
                "provider": "Open-Meteo Weather",
                "status": "Connected" if separated_weather else "Unavailable",
                "is_live": bool(separated_weather and not weather_cached),
                "is_cached": weather_cached,
                "last_updated": weather_freshness.get("timestamp") or weather_ts,
                "attribution": self.weather.attribution,
            },
            {
                "provider": "IMD (India Meteorological Department)",
                "status": imd_data.get("status", "Not configured"),
                "is_live": bool(imd_data.get("status") == "Connected" and not imd_data.get("is_cached")),
                "is_cached": bool(imd_data.get("is_cached", False)),
                "last_updated": imd_data.get("last_updated"),
                "attribution": imd_data.get("attribution"),
            },
            {
                "provider": "NDMA SACHET",
                "status": sachet_data.get("status", "Link only"),
                "is_live": bool(sachet_data.get("status") == "Connected" and not sachet_data.get("is_cached")),
                "is_cached": bool(sachet_data.get("is_cached", False)),
                "last_updated": sachet_data.get("last_updated"),
                "attribution": sachet_data.get("attribution"),
            },
        ]

        return {
            "country": "India",
            "state": canonical_state,
            "district": canonical_district,
            "district_meta": district_meta,
            "state_geojson": state_geojson,
            "district_geojson": district_geojson,
            "location_id": location_id,
            "prediction_id": prediction_id,
            "weather": weather_summary,
            "imd": imd_data,
            "sachet": sachet_data,
            "terrain": terrain_data,
            "historical": regional_hist,
            "features_used": features,
            "risk_result": risk_result,
            "shap_explanation": shap_explanation,
            "experimental_alert": experimental_alert,
            "provider_status": provider_panel,
            "persisted_statuses": get_provider_statuses() if persist else [],
        }
