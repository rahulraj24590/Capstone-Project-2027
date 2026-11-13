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

import math
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.data_providers.base import BaseProvider, ProviderError
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting
from src.utils.validation import validate_coordinates

logger = get_logger(__name__)

# Open-Meteo variables
# See: https://open-meteo.com/en/docs
HOURLY_VARS = [
    "precipitation",         # mm/h
    "temperature_2m",        # °C
    "relative_humidity_2m",  # %
    "wind_speed_10m",        # km/h
    "soil_moisture_0_to_7cm",  # model-derived, m³/m³
    "weather_code",          # WMO 4677 weather interpretation code
]

CURRENT_VARS = [
    "precipitation",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "weather_code",
]

DAILY_VARS = [
    "precipitation_sum",     # mm/day
    "temperature_2m_max",
    "temperature_2m_min",
]


def _to_finite_float(val: Any) -> float | None:
    """Convert value to finite float or None (preserving missing values)."""
    if val is None:
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def decode_wmo_weather_code(weather_code: Any) -> str:
    """Decode an Open-Meteo WMO 4677 weather code into a human-readable string."""
    val = _to_finite_float(weather_code)
    if val is None:
        return "Unavailable"
    code = int(round(val))
    wmo_exact = {
        0: "Clear Sky",
        1: "Mainly Clear",
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing Rime Fog",
    }
    if code in wmo_exact:
        return wmo_exact[code]
    if 51 <= code <= 57:
        return "Drizzle"
    if 61 <= code <= 67 or 80 <= code <= 82:
        return "Rain"
    if 71 <= code <= 77 or 85 <= code <= 86:
        return "Snow"
    if 95 <= code <= 99:
        return "Thunderstorm"
    return "Unavailable"


def convert_precipitation_values(
    values: list[Any],
    unit: str | None = "mm",
) -> list[float | None]:
    """Validate and convert precipitation values into millimetres (mm)."""
    norm_unit = (unit or "mm").strip().lower()
    factors = {"mm": 1.0, "millimeter": 1.0, "millimeters": 1.0, "cm": 10.0, "inch": 25.4, "in": 25.4}
    if norm_unit not in factors:
        raise ValueError(f"Unsupported precipitation unit: {unit!r}")
    factor = factors[norm_unit]
    return [
        round(v * factor, 4) if (v := _to_finite_float(raw)) is not None else None
        for raw in values
    ]


def convert_wind_values(
    values: list[Any],
    unit: str | None = "km/h",
) -> list[float | None]:
    """Validate and convert wind speeds into km/h."""
    norm_unit = (unit or "km/h").strip().lower()
    factors = {"km/h": 1.0, "kmh": 1.0, "m/s": 3.6, "ms": 3.6, "mph": 1.60934, "kn": 1.852, "knots": 1.852}
    if norm_unit not in factors:
        raise ValueError(f"Unsupported wind speed unit: {unit!r}")
    factor = factors[norm_unit]
    return [
        round(v * factor, 4) if (v := _to_finite_float(raw)) is not None else None
        for raw in values
    ]


def convert_temperature_values(
    values: list[Any],
    unit: str | None = "°C",
) -> list[float | None]:
    """Validate and convert temperatures into degrees Celsius (°C)."""
    norm_unit = (unit or "°c").strip().lower()
    if norm_unit in {"°c", "c", "celsius"}:
        return [_to_finite_float(raw) for raw in values]
    if norm_unit in {"°f", "f", "fahrenheit"}:
        return [
            round((v - 32.0) * 5.0 / 9.0, 4) if (v := _to_finite_float(raw)) is not None else None
            for raw in values
        ]
    raise ValueError(f"Unsupported temperature unit: {unit!r}")


def parse_api_timestamp(
    ts: str | None,
    utc_offset_seconds: int = 0,
) -> datetime | None:
    """Parse an Open-Meteo ISO-8601 timestamp into a UTC-aware datetime."""
    if not ts or not isinstance(ts, str):
        return None
    cleaned = ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        tz = timezone(timedelta(seconds=int(utc_offset_seconds or 0)))
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)


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

    def _validate_coords(self, latitude: float, longitude: float) -> bool:
        """Validate coordinates before querying Open-Meteo or disk cache."""
        check = validate_coordinates(latitude, longitude)
        if not check.is_valid:
            err_msg = "; ".join(check.errors)
            logger.warning("[WeatherProvider] Invalid coordinates (%r, %r): %s", latitude, longitude, err_msg)
            self._last_fetch_meta = {
                "is_cached": False,
                "cached_at": None,
                "age_seconds": None,
                "status": "invalid_coordinates",
                "rate_limited": False,
                "error": err_msg,
            }
            return False
        return True

    def fetch(
        self,
        latitude: float,
        longitude: float,
        past_days: int = 3,
        forecast_days: int = 1,
        bypass_cache: bool = False,
    ) -> dict | None:
        """Fetch weather data for a location.

        Args:
            latitude: Location latitude in decimal degrees [-90, 90].
            longitude: Location longitude in decimal degrees [-180, 180].
            past_days: Days of historical data to retrieve (max 92 for free tier).
            forecast_days: Days of forecast data (max 16).
            bypass_cache: If True, skip cached responses and query the API live.

        Returns:
            Dict with processed weather data or None if invalid/unavailable.
        """
        if not self._validate_coords(latitude, longitude):
            return None

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(HOURLY_VARS),
            "daily": ",".join(DAILY_VARS),
            "current": ",".join(CURRENT_VARS),
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "auto",
            "timeformat": "iso8601",
        }
        if self._api_key:
            params["apikey"] = self._api_key

        url = f"{self.base_url}/forecast"
        raw = (
            self._get_cached("openmeteo", {**params, "url": url}, bypass_cache=True)
            if bypass_cache
            else self._get_cached("openmeteo", {**params, "url": url})
        )

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
        bypass_cache: bool = False,
    ) -> dict | None:
        """Fetch historical weather data.

        Uses the Open-Meteo Historical API (ERA5-based data from 1940).

        Args:
            latitude: Latitude [-90, 90].
            longitude: Longitude [-180, 180].
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            bypass_cache: If True, bypass cached disk entries.

        Returns:
            Dict with daily precipitation and temperature, or None.
        """
        if not self._validate_coords(latitude, longitude):
            return None
        if start_date > end_date:
            err_msg = f"start_date ({start_date}) must be <= end_date ({end_date})."
            self._last_fetch_meta = {
                "is_cached": False,
                "cached_at": None,
                "age_seconds": None,
                "status": "invalid_parameters",
                "rate_limited": False,
                "error": err_msg,
            }
            logger.warning("[WeatherProvider] %s", err_msg)
            return None

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join(DAILY_VARS),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": "auto",
        }

        url = "https://archive-api.open-meteo.com/v1/archive"
        raw = (
            self._get_cached("openmeteo_hist", {**params, "url": url}, bypass_cache=True)
            if bypass_cache
            else self._get_cached("openmeteo_hist", {**params, "url": url})
        )

        if raw is None:
            return None

        return self._parse_response(raw, latitude, longitude, is_archive=True)

    def get_rainfall_accumulations(
        self,
        latitude: float,
        longitude: float,
        bypass_cache: bool = False,
    ) -> dict[str, Any]:
        """Fetch recent rainfall accumulations for risk estimation.

        Computes 1h, 24h, 48h, and 72h windows strictly from non-future
        reanalysis/past hourly slots. Missing rainfall values are never replaced
        with zero; insufficient windows yield None.
        """
        data = self.fetch(latitude, longitude, past_days=3, forecast_days=0, bypass_cache=bypass_cache)
        if data is None:
            meta = self.last_fetch_metadata
            return {
                "rainfall_1h_mm": None,
                "rainfall_24h_mm": None,
                "rainfall_48h_mm": None,
                "rainfall_72h_mm": None,
                "missing_hours_count": None,
                "source": meta.get("status") or "unavailable",
                "error": meta.get("error"),
                "fetched_at": None,
                "last_successful_update": self.last_update_time,
                "is_cached": False,
                "cached_at": None,
                "is_observed": False,
                "is_forecast": False,
            }

        hist_hourly = data.get("historical_hourly") or data.get("hourly") or {}
        precip = hist_hourly.get("precipitation", [])

        if not precip:
            return {
                "rainfall_1h_mm": None,
                "rainfall_24h_mm": None,
                "rainfall_48h_mm": None,
                "rainfall_72h_mm": None,
                "missing_hours_count": 0,
                "source": self.attribution,
                "fetched_at": data.get("fetched_at"),
                "last_successful_update": data.get("last_successful_update"),
                "is_cached": bool(data.get("is_cached", False)),
                "cached_at": data.get("cached_at"),
                "is_observed": False,
                "is_forecast": False,
            }

        def _strict_window_sum(series: list[float | None], window: int) -> tuple[float | None, int]:
            if len(series) < window:
                missing_in_avail = sum(1 for v in series if v is None)
                return None, missing_in_avail + (window - len(series))
            slice_vals = series[-window:]
            missing = sum(1 for v in slice_vals if v is None or v < 0)
            if missing > 0:
                return None, missing
            return round(float(sum(v for v in slice_vals if v is not None)), 2), 0

        last_val = precip[-1]
        rainfall_1h = round(float(last_val), 2) if (last_val is not None and last_val >= 0) else None
        rainfall_24h, miss_24 = _strict_window_sum(precip, 24)
        rainfall_48h, miss_48 = _strict_window_sum(precip, 48)
        rainfall_72h, miss_72 = _strict_window_sum(precip, 72)

        return {
            "rainfall_1h_mm": rainfall_1h,
            "rainfall_24h_mm": rainfall_24h,
            "rainfall_48h_mm": rainfall_48h,
            "rainfall_72h_mm": rainfall_72h,
            "missing_hours_24h": miss_24,
            "missing_hours_48h": miss_48,
            "missing_hours_72h": miss_72,
            "source": f"{self.attribution} (model-derived, NOT direct gauge observations)",
            "fetched_at": data.get("fetched_at"),
            "last_successful_update": data.get("last_successful_update"),
            "is_cached": bool(data.get("is_cached", False)),
            "cached_at": data.get("cached_at"),
            "is_observed": False,
            "is_forecast": False,
        }

    def fetch_separated_weather(
        self,
        latitude: float,
        longitude: float,
        past_days: int = 3,
        forecast_days: int = 3,
        include_archive_baseline: bool = True,
        bypass_cache: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch weather with separated current, historical, forecast, and baseline blocks."""
        weather = self.fetch(
            latitude=latitude,
            longitude=longitude,
            past_days=past_days,
            forecast_days=forecast_days,
            bypass_cache=bypass_cache,
        )
        weather_meta = self.last_fetch_metadata
        if weather is None:
            return None

        archive_data = None
        if include_archive_baseline:
            end_dt = date.today() - timedelta(days=5)
            start_dt = end_dt - timedelta(days=364)
            archive_data = self.fetch_historical(
                latitude, longitude, start_dt, end_dt, bypass_cache=bypass_cache
            )

        return {
            "latitude": weather["latitude"],
            "longitude": weather["longitude"],
            "timezone": weather["timezone"],
            "utc_offset_seconds": weather["utc_offset_seconds"],
            "elevation_m": weather.get("elevation_m"),
            "current": weather.get("current"),
            "historical_hourly": weather.get("historical_hourly", {}),
            "forecast_hourly": weather.get("forecast_hourly", {}),
            "daily_historical": weather.get("daily_historical", {}),
            "daily_forecast": weather.get("daily_forecast", {}),
            "archive_baseline_daily": (archive_data or {}).get("daily", {}),
            "unit_validation": weather.get("unit_validation", {}),
            "provenance": {
                "provider": self.name,
                "attribution": self.attribution,
                "data_type": weather.get("data_type"),
                "fetched_at": weather.get("fetched_at"),
                "last_successful_update": weather.get("last_successful_update"),
                "is_cached": bool(weather.get("is_cached", False)),
                "cached_at": weather.get("cached_at"),
                "cache_age_seconds": weather.get("cache_age_seconds"),
                "connection_status": weather.get(
                    "connection_status",
                    weather_meta.get("status", "cached" if weather.get("is_cached") else "live"),
                ),
                "rate_limited": bool(weather_meta.get("rate_limited", False)),
                "error": weather_meta.get("error"),
            },
            # Preserve flat keys for callers expecting standard fetch schema
            "hourly": weather.get("hourly", {}),
            "daily": weather.get("daily", {}),
            "fetched_at": weather.get("fetched_at"),
            "is_cached": bool(weather.get("is_cached", False)),
            "cached_at": weather.get("cached_at"),
            "last_successful_update": weather.get("last_successful_update"),
            "connection_status": weather.get("connection_status"),
        }

    def _parse_response(
        self,
        raw: dict,
        latitude: float,
        longitude: float,
        is_archive: bool = False,
    ) -> dict:
        """Parse and normalise Open-Meteo API response with unit and time separation."""
        meta = self.last_fetch_metadata
        utc_offset = int(raw.get("utc_offset_seconds") or 0)
        now_utc = datetime.now(timezone.utc)
        now_local_date = (now_utc + timedelta(seconds=utc_offset)).date().isoformat()

        unit_validation: dict[str, Any] = {
            "validated": True,
            "conversions_applied": [],
            "errors": [],
            "hourly_units": raw.get("hourly_units", {}),
            "daily_units": raw.get("daily_units", {}),
            "current_units": raw.get("current_units", {}),
        }

        result: dict[str, Any] = {
            "latitude": raw.get("latitude", latitude),
            "longitude": raw.get("longitude", longitude),
            "timezone": raw.get("timezone", "UTC"),
            "utc_offset_seconds": utc_offset,
            "elevation_m": raw.get("elevation"),
            "source": self.attribution,
            "data_type": "era5_historical_archive" if is_archive else "model_derived_reanalysis_and_forecast",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "last_successful_update": self.last_update_time,
            "is_cached": bool(meta.get("is_cached", False)),
            "cached_at": meta.get("cached_at"),
            "cache_age_seconds": meta.get("age_seconds"),
            "connection_status": meta.get(
                "status",
                "cached" if meta.get("is_cached") else "live",
            ),
            "unit_validation": unit_validation,
        }

        # Current weather block (Open-Meteo model snapshot, clearly marked not forecast)
        current = raw.get("current", {})
        current_units = raw.get("current_units", {})
        if current:
            c_precip_unit = current_units.get("precipitation", "mm")
            c_temp_unit = current_units.get("temperature_2m", "°C")
            c_wind_unit = current_units.get("wind_speed_10m", "km/h")
            try:
                c_precip = convert_precipitation_values([current.get("precipitation")], c_precip_unit)[0]
                c_temp = convert_temperature_values([current.get("temperature_2m")], c_temp_unit)[0]
                c_wind = convert_wind_values([current.get("wind_speed_10m")], c_wind_unit)[0]
            except ValueError as exc:
                unit_validation["validated"] = False
                unit_validation["errors"].append(str(exc))
                c_precip = _to_finite_float(current.get("precipitation"))
                c_temp = _to_finite_float(current.get("temperature_2m"))
                c_wind = _to_finite_float(current.get("wind_speed_10m"))

            raw_wmo = _to_finite_float(current.get("weather_code"))
            wmo_code_int = int(round(raw_wmo)) if raw_wmo is not None else None
            result["current"] = {
                "time": current.get("time"),
                "timestamp_utc": (
                    dt.isoformat()
                    if (dt := parse_api_timestamp(current.get("time"), utc_offset)) is not None
                    else None
                ),
                "precipitation_mm": c_precip,
                "temperature_2m_c": c_temp,
                "relative_humidity_2m_pct": _to_finite_float(current.get("relative_humidity_2m")),
                "wind_speed_10m_kmh": c_wind,
                "weather_code": wmo_code_int,
                "weather_condition": decode_wmo_weather_code(wmo_code_int),
                "is_forecast": False,
                "is_observed": False,
                "is_cached": result["is_cached"],
                "data_label": "Open-Meteo Current Model Snapshot (NOT direct station gauge)",
            }

        # Hourly data with unit validation & conversion
        hourly = raw.get("hourly", {})
        if hourly:
            h_units = raw.get("hourly_units", {})
            h_precip_unit = h_units.get("precipitation", "mm")
            h_temp_unit = h_units.get("temperature_2m", "°C")
            h_wind_unit = h_units.get("wind_speed_10m", "km/h")

            try:
                precip_vals = convert_precipitation_values(hourly.get("precipitation", []), h_precip_unit)
                if h_precip_unit and h_precip_unit.lower() != "mm":
                    unit_validation["conversions_applied"].append(f"hourly.precipitation: {h_precip_unit} -> mm")
            except ValueError as exc:
                unit_validation["validated"] = False
                unit_validation["errors"].append(str(exc))
                precip_vals = [_to_finite_float(v) for v in hourly.get("precipitation", [])]

            try:
                temp_vals = convert_temperature_values(hourly.get("temperature_2m", []), h_temp_unit)
                if h_temp_unit and h_temp_unit.lower() not in {"°c", "c"}:
                    unit_validation["conversions_applied"].append(f"hourly.temperature_2m: {h_temp_unit} -> °C")
            except ValueError as exc:
                unit_validation["validated"] = False
                unit_validation["errors"].append(str(exc))
                temp_vals = [_to_finite_float(v) for v in hourly.get("temperature_2m", [])]

            try:
                wind_vals = convert_wind_values(hourly.get("wind_speed_10m", []), h_wind_unit)
                if h_wind_unit and h_wind_unit.lower() not in {"km/h", "kmh"}:
                    unit_validation["conversions_applied"].append(f"hourly.wind_speed_10m: {h_wind_unit} -> km/h")
            except ValueError as exc:
                unit_validation["validated"] = False
                unit_validation["errors"].append(str(exc))
                wind_vals = [_to_finite_float(v) for v in hourly.get("wind_speed_10m", [])]

            times = hourly.get("time", [])
            rh_vals = [_to_finite_float(v) for v in hourly.get("relative_humidity_2m", [])]
            soil_vals = [_to_finite_float(v) for v in hourly.get("soil_moisture_0_to_7cm", [])]
            wmo_vals = [
                int(round(fv)) if (fv := _to_finite_float(v)) is not None else None
                for v in hourly.get("weather_code", [])
            ]

            result["hourly"] = {
                "time": times,
                "precipitation": precip_vals,
                "temperature_2m": temp_vals,
                "relative_humidity_2m": rh_vals,
                "wind_speed_10m": wind_vals,
                "soil_moisture_0_to_7cm": soil_vals,
                "weather_code": wmo_vals,
            }

            # Separate strictly past/present hours from future forecast hours
            hist_idx: list[int] = []
            fcst_idx: list[int] = []
            for idx, ts_str in enumerate(times):
                dt_utc = parse_api_timestamp(ts_str, utc_offset)
                if dt_utc is not None and dt_utc > now_utc and not is_archive:
                    fcst_idx.append(idx)
                else:
                    hist_idx.append(idx)

            def _slice_hourly(indices: list[int], is_forecast_flag: bool) -> dict[str, Any]:
                def _pick(seq: list[Any], idx: int) -> Any:
                    return seq[idx] if idx < len(seq) else None

                return {
                    "time": [times[i] for i in indices],
                    "precipitation": [_pick(precip_vals, i) for i in indices],
                    "temperature_2m": [_pick(temp_vals, i) for i in indices],
                    "relative_humidity_2m": [_pick(rh_vals, i) for i in indices],
                    "wind_speed_10m": [_pick(wind_vals, i) for i in indices],
                    "soil_moisture_0_to_7cm": [_pick(soil_vals, i) for i in indices],
                    "weather_code": [_pick(wmo_vals, i) for i in indices],
                    "is_forecast": is_forecast_flag,
                    "is_observed": False,
                }

            result["historical_hourly"] = _slice_hourly(hist_idx, is_forecast_flag=False)
            result["forecast_hourly"] = _slice_hourly(fcst_idx, is_forecast_flag=True)

        # Daily data with unit validation & historical/forecast split
        daily = raw.get("daily", {})
        if daily:
            d_units = raw.get("daily_units", {})
            d_precip_unit = d_units.get("precipitation_sum", "mm")
            d_temp_unit = d_units.get("temperature_2m_max", "°C")
            try:
                d_precip = convert_precipitation_values(daily.get("precipitation_sum", []), d_precip_unit)
            except ValueError as exc:
                unit_validation["validated"] = False
                unit_validation["errors"].append(str(exc))
                d_precip = [_to_finite_float(v) for v in daily.get("precipitation_sum", [])]
            try:
                d_tmax = convert_temperature_values(daily.get("temperature_2m_max", []), d_temp_unit)
                d_tmin = convert_temperature_values(daily.get("temperature_2m_min", []), d_temp_unit)
            except ValueError as exc:
                unit_validation["validated"] = False
                unit_validation["errors"].append(str(exc))
                d_tmax = [_to_finite_float(v) for v in daily.get("temperature_2m_max", [])]
                d_tmin = [_to_finite_float(v) for v in daily.get("temperature_2m_min", [])]

            d_times = daily.get("time", [])
            result["daily"] = {
                "time": d_times,
                "precipitation_sum": d_precip,
                "temperature_2m_max": d_tmax,
                "temperature_2m_min": d_tmin,
            }

            d_hist_idx = [
                i for i, d_str in enumerate(d_times)
                if is_archive or (isinstance(d_str, str) and d_str <= now_local_date)
            ]
            d_fcst_idx = [
                i for i, d_str in enumerate(d_times)
                if not is_archive and isinstance(d_str, str) and d_str > now_local_date
            ]

            def _slice_daily(indices: list[int], is_forecast_flag: bool) -> dict[str, Any]:
                def _pick(seq: list[Any], idx: int) -> Any:
                    return seq[idx] if idx < len(seq) else None

                return {
                    "time": [d_times[i] for i in indices],
                    "precipitation_sum": [_pick(d_precip, i) for i in indices],
                    "temperature_2m_max": [_pick(d_tmax, i) for i in indices],
                    "temperature_2m_min": [_pick(d_tmin, i) for i in indices],
                    "is_forecast": is_forecast_flag,
                    "is_observed": False,
                }

            result["daily_historical"] = _slice_daily(d_hist_idx, is_forecast_flag=False)
            result["daily_forecast"] = _slice_daily(d_fcst_idx, is_forecast_flag=True)

        return result
