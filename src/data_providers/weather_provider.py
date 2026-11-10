"""
Open-Meteo weather data provider for LandslideGuard.

Uses the Open-Meteo API (https://open-meteo.com/) to fetch:
- Hourly precipitation
- Historical precipitation accumulations
- Temperature and humidity

Open-Meteo is free, open-source, and does not require an API key for
standard variables. Commercial use requires a subscription.

Attribution: Weather data by Open-Meteo.com (CC BY 4.0)

IMPORTANT: Only fetch variables that the Open-Meteo API actually provides.
Do not claim this provides real-time soil moisture (it provides volumetric
soil water content at depth layers, which is model-derived, not observed).
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.data_providers.base import BaseProvider, ProviderError
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

# Open-Meteo variables
# See: https://open-meteo.com/en/docs
HOURLY_VARS = [
    "precipitation",         # mm/h
    "temperature_2m",        # °C
    "relative_humidity_2m",  # %
    "wind_speed_10m",        # km/h
    "soil_moisture_0_to_7cm",  # model-derived, m³/m³
]

DAILY_VARS = [
    "precipitation_sum",     # mm/day
    "temperature_2m_max",
    "temperature_2m_min",
]


class WeatherProvider(BaseProvider):
    """Open-Meteo weather data provider."""

    name = "Open-Meteo"
    attribution = "Weather data by Open-Meteo.com (CC BY 4.0)"

    def __init__(self) -> None:
        super().__init__()
        cfg = get_setting("providers", "weather") or {}
        self.base_url = cfg.get("base_url", "https://api.open-meteo.com/v1")
        self.timeout_seconds = int(cfg.get("timeout_seconds", 10))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.retry_backoff_factor = int(cfg.get("retry_backoff_factor", 2))
        self.cache_ttl_seconds = int(cfg.get("cache_ttl_seconds", 3600))
        # API key (only needed for commercial Open-Meteo subscription)
        self._api_key = os.environ.get("OPENMETEO_API_KEY", "")

    def fetch(
        self,
        latitude: float,
        longitude: float,
        past_days: int = 3,
        forecast_days: int = 1,
    ) -> dict | None:
        """Fetch weather data for a location.

        Args:
            latitude: Location latitude in decimal degrees.
            longitude: Location longitude in decimal degrees.
            past_days: Days of historical data to retrieve (max 92 for free tier).
            forecast_days: Days of forecast data (max 16).

        Returns:
            Dict with processed weather data or None if unavailable.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(HOURLY_VARS),
            "daily": ",".join(DAILY_VARS),
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "auto",
            "timeformat": "iso8601",
        }
        if self._api_key:
            params["apikey"] = self._api_key

        url = f"{self.base_url}/forecast"
        raw = self._get_cached("openmeteo", {**params, "url": url})

        if raw is None:
            logger.warning("[WeatherProvider] Fetch failed for (%.4f, %.4f).", latitude, longitude)
            return None

        return self._parse_response(raw, latitude, longitude)

    def fetch_historical(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> dict | None:
        """Fetch historical weather data.

        Uses the Open-Meteo Historical API (ERA5-based data from 1940).

        Args:
            latitude: Latitude.
            longitude: Longitude.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            Dict with daily precipitation and temperature, or None.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join(DAILY_VARS),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": "auto",
        }

        url = "https://archive-api.open-meteo.com/v1/archive"
        raw = self._get_cached("openmeteo_hist", {**params, "url": url})

        if raw is None:
            return None

        return self._parse_response(raw, latitude, longitude)

    def get_rainfall_accumulations(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, float | None]:
        """Fetch recent rainfall accumulations for risk estimation.

        Returns accumulations over 1h, 24h, and 72h windows.
        These are derived from hourly precipitation data.

        Data provenance: Open-Meteo ERA5/ECMWF reanalysis (model-derived,
        not direct rain gauge observations).

        Args:
            latitude: Location latitude.
            longitude: Location longitude.

        Returns:
            Dict with keys:
            - rainfall_1h_mm: Most recent hour precipitation (mm).
            - rainfall_24h_mm: Sum over the last 24 hours (mm).
            - rainfall_72h_mm: Sum over the last 72 hours (mm).
            - source: Data source description.
            - fetched_at: ISO timestamp.
            - is_observed: False (model-derived reanalysis).
        """
        data = self.fetch(latitude, longitude, past_days=3, forecast_days=0)
        if data is None:
            return {
                "rainfall_1h_mm": None,
                "rainfall_24h_mm": None,
                "rainfall_72h_mm": None,
                "source": "unavailable",
                "fetched_at": None,
            }

        hourly = data.get("hourly", {})
        precip = hourly.get("precipitation", [])

        if not precip:
            return {
                "rainfall_1h_mm": None,
                "rainfall_24h_mm": None,
                "rainfall_72h_mm": None,
                "source": self.attribution,
                "fetched_at": data.get("fetched_at"),
            }

        # Most recent hour
        rainfall_1h = float(precip[-1]) if precip else None
        # Last 24 hours
        rainfall_24h = float(sum(precip[-24:])) if len(precip) >= 1 else None
        # Last 72 hours
        rainfall_72h = float(sum(precip[-72:])) if len(precip) >= 1 else None

        return {
            "rainfall_1h_mm": round(rainfall_1h, 2) if rainfall_1h is not None else None,
            "rainfall_24h_mm": round(rainfall_24h, 2) if rainfall_24h is not None else None,
            "rainfall_72h_mm": round(rainfall_72h, 2) if rainfall_72h is not None else None,
            "source": f"{self.attribution} (model-derived, NOT direct observations)",
            "fetched_at": data.get("fetched_at"),
            "is_observed": False,
        }

    def _parse_response(
        self,
        raw: dict,
        latitude: float,
        longitude: float,
    ) -> dict:
        """Parse and normalise Open-Meteo API response.

        Args:
            raw: Raw JSON response dict.
            latitude: Requested latitude.
            longitude: Requested longitude.

        Returns:
            Normalised weather data dict.
        """
        result: dict[str, Any] = {
            "latitude": raw.get("latitude", latitude),
            "longitude": raw.get("longitude", longitude),
            "timezone": raw.get("timezone", "UTC"),
            "elevation_m": raw.get("elevation"),
            "source": self.attribution,
            "data_type": "model_derived_reanalysis",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        # Hourly data
        hourly = raw.get("hourly", {})
        if hourly:
            result["hourly"] = {
                "time": hourly.get("time", []),
                "precipitation": [
                    float(v) if v is not None else None
                    for v in hourly.get("precipitation", [])
                ],
                "temperature_2m": [
                    float(v) if v is not None else None
                    for v in hourly.get("temperature_2m", [])
                ],
                "relative_humidity_2m": [
                    float(v) if v is not None else None
                    for v in hourly.get("relative_humidity_2m", [])
                ],
                "soil_moisture_0_to_7cm": [
                    float(v) if v is not None else None
                    for v in hourly.get("soil_moisture_0_to_7cm", [])
                ],
            }

        # Daily data
        daily = raw.get("daily", {})
        if daily:
            result["daily"] = {
                "time": daily.get("time", []),
                "precipitation_sum": [
                    float(v) if v is not None else None
                    for v in daily.get("precipitation_sum", [])
                ],
            }

        return result
