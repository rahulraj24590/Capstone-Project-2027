"""Official India Meteorological Department (IMD) API provider.

Strictly implements ONLY documented endpoints from https://api.imd.gov.in/public/api_reference.html:
- /staterainfall                    State-wise daily/weekly/cumulative/monthly rainfall & departure
- /districtrainfall                 District-wise rainfall & departure
- /districtwarning                  District-wise 5-day weather warning codes & colors (Green/Yellow/Orange/Red)
- /districtnowcast                  District-wise 3-hourly severe weather nowcast categories (Cat1-Cat19)
- /current_wx                       Station current weather observation
- /state_district_rainfall_forecast 5-day spatial rainfall distribution forecast

Authentication & Safety:
- Reads ``IMD_API_KEY`` (header ``x-api-key``) and optional ``IMD_AUTH_TOKEN`` from environment variables.
- Never invents credentials or response payloads.
- Reports provider status explicitly: ``"Not configured"``, ``"Connected"``, or ``"Unavailable"``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

from src.data_providers.base import BaseProvider, ProviderError, _cache_key, _read_cache_entry, _write_cache
from src.data_providers.india_boundaries import normalize_admin_name
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

# Documented IMD Rainfall Departure Categories (api_reference.html sections 5 & 8)
RAINFALL_CATEGORY_MAP: dict[str, str] = {
    "LE": "Large Excess (+60% or more)",
    "E": "Excess (+20% to +59%)",
    "N": "Normal (-19% to +19%)",
    "D": "Deficient (-59% to -20%)",
    "LD": "Large Deficient (-99% to -60%)",
    "NR": "No Rain (-100%)",
    "ND": "No Data",
}

# Documented IMD Warning Color Codes (api_reference.html section 6)
WARNING_COLOR_MAP: dict[str, dict[str, str]] = {
    "1": {"label": "No Warning (Green)", "hex": "#008000"},
    "2": {"label": "Watch / Yellow", "hex": "#eab308"},
    "3": {"label": "Alert / Orange", "hex": "#f97316"},
    "4": {"label": "Warning / Red", "hex": "#ef4444"},
}

# Documented IMD District Warning Codes (api_reference.html section 6)
WARNING_CODE_MAP: dict[str, str] = {
    "1": "No Warning",
    "2": "Heavy Rain",
    "3": "Heavy to Very Heavy Rain",
    "4": "Extremely Heavy Rain",
    "5": "Thunderstorm & Lightning",
    "6": "Hailstorm",
    "7": "Gusty Wind",
    "8": "Dust Storm",
    "9": "Heat Wave",
    "10": "Cold Wave",
    "11": "Dense Fog",
    "12": "Frost",
    "13": "Snowfall",
    "14": "Strong Surface Wind",
    "15": "Squall",
}

# Documented IMD Nowcast Categories (api_reference.html section 4)
NOWCAST_CATEGORY_MAP: dict[str, str] = {
    "Cat1": "No Weather",
    "Cat2": "Light rain (< 5 mm/hr)",
    "Cat3": "Light snow (< 5 cm/hr)",
    "Cat4": "Light Thunderstorms (wind < 40 kmph)",
    "Cat5": "Slight dust storm",
    "Cat6": "Low lightning probability (< 30%)",
    "Cat7": "Moderate rain (5-15 mm/hr)",
    "Cat8": "Moderate snow (5-15 cm/hr)",
    "Cat9": "Moderate Thunderstorms (wind 41-61 kmph)",
    "Cat10": "Moderate dust storm",
    "Cat11": "Moderate lightning probability (30-60%)",
    "Cat12": "Heavy rain (> 15 mm/hr)",
    "Cat13": "Heavy snow (> 15 cm/hr)",
    "Cat14": "Severe Thunderstorms (wind 62-87 kmph)",
    "Cat15": "Very Severe Thunderstorms (wind > 87 kmph)",
    "Cat17": "Thunderstorms with Hail",
    "Cat18": "Severe dust storm (> 61 kmph)",
}


def _to_float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace("%", "")
    if not s or s.upper() in {"NA", "N/A", "ND", "NULL", "-"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _match_admin_record(records: list[dict], field_name: str, target_name: str) -> dict | None:
    """Case- and diacritic-insensitive match against IMD administrative records."""
    if not records or not target_name:
        return None
    target = normalize_admin_name(target_name).upper()
    for r in records:
        cand = normalize_admin_name(str(r.get(field_name) or "")).upper()
        if cand == target:
            return r
    for r in records:
        cand = normalize_admin_name(str(r.get(field_name) or "")).upper()
        if target in cand or cand in target:
            return r
    return None


class IMDProvider(BaseProvider):
    """Client for official India Meteorological Department v1 APIs."""

    name = "IMD"
    attribution = "Official weather & warnings by India Meteorological Department (api.imd.gov.in)"

    def __init__(self) -> None:
        super().__init__()
        cfg = get_setting("providers", "imd") or {}
        self.base_url = cfg.get("base_url", "https://api.imd.gov.in/api/v1")
        self.docs_url = cfg.get("docs_url", "https://api.imd.gov.in/public/api_reference.html")
        self.timeout_seconds = int(cfg.get("timeout_seconds", 12))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.retry_backoff_factor = int(cfg.get("retry_backoff_factor", 2))
        self.cache_ttl_seconds = int(cfg.get("cache_ttl_seconds", 1800))

    @property
    def api_key(self) -> str:
        return os.environ.get("IMD_API_KEY", "").strip()

    @property
    def auth_token(self) -> str:
        return os.environ.get("IMD_AUTH_TOKEN", "").strip()

    def is_configured(self) -> bool:
        """Return True only if an IMD API key is present in environment variables."""
        key = self.api_key
        return bool(key and not key.startswith("<your_"))

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        if self.auth_token:
            headers["Authorization"] = (
                self.auth_token
                if self.auth_token.lower().startswith("bearer ")
                else f"Bearer {self.auth_token}"
            )
        return headers

    def _fetch_endpoint(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[Any | None, dict[str, Any]]:
        """Query an IMD endpoint with caching, auth failure short-circuit, and attribution."""
        if not self.is_configured():
            return None, {
                "status": "Not configured",
                "is_cached": False,
                "error": "IMD_API_KEY not configured in environment (.env).",
            }

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        key = _cache_key(f"imd_{endpoint}", {"url": url, **(params or {})})
        cached, cache_meta = _read_cache_entry(key, self.cache_ttl_seconds)
        if cached is not None:
            return cached, {
                "status": "Connected",
                "is_cached": True,
                "cached_at": (cache_meta or {}).get("cached_at"),
                "age_seconds": (cache_meta or {}).get("age_seconds"),
            }

        try:
            resp = self._session.get(
                url,
                params=params or None,
                headers=self._auth_headers(),
                timeout=self.timeout_seconds,
            )
            if resp.status_code in (401, 403):
                msg = f"IMD authentication rejected ({resp.status_code}): check IMD_API_KEY / IP whitelisting."
                logger.warning("[IMD] %s", msg)
                return None, {"status": "Unavailable", "is_cached": False, "error": msg}
            if resp.status_code == 429:
                msg = "IMD API rate limit exceeded (429)."
                logger.warning("[IMD] %s", msg)
                return None, {"status": "Unavailable", "is_cached": False, "error": msg}
            resp.raise_for_status()
            payload = resp.json()
            _write_cache(key, payload)
            now_iso = datetime.now(timezone.utc).isoformat()
            self._last_successful_fetch = datetime.now(timezone.utc)
            return payload, {
                "status": "Connected",
                "is_cached": False,
                "fetched_at": now_iso,
            }
        except requests.RequestException as exc:
            logger.warning("[IMD] Request to %s failed: %s", endpoint, exc)
            return None, {"status": "Unavailable", "is_cached": False, "error": str(exc)}
        except ValueError as exc:
            logger.warning("[IMD] Invalid JSON from %s: %s", endpoint, exc)
            return None, {"status": "Unavailable", "is_cached": False, "error": f"Invalid JSON: {exc}"}

    @staticmethod
    def parse_state_rainfall(raw: Any, state_name: str) -> dict[str, Any] | None:
        records = raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])
        rec = _match_admin_record(records, "State", state_name)
        if not rec:
            return None
        cat_code = str(rec.get("Daily Category") or "").strip()
        return {
            "state": rec.get("State"),
            "date": rec.get("Date"),
            "daily_actual_mm": _to_float_or_none(rec.get("Daily Actual")),
            "daily_normal_mm": _to_float_or_none(rec.get("Daily Normal")),
            "daily_departure_pct": str(rec.get("Daily Departure Per") or "").strip() or None,
            "daily_category_code": cat_code or None,
            "daily_category": RAINFALL_CATEGORY_MAP.get(cat_code, cat_code or None),
            "weekly_actual_mm": _to_float_or_none(rec.get("Weekly Actual")),
            "weekly_normal_mm": _to_float_or_none(rec.get("Weekly Normal")),
            "weekly_departure_pct": str(rec.get("Weekly Departure Per") or "").strip() or None,
            "monthly_actual_mm": _to_float_or_none(rec.get("Monthly Actual") or rec.get("Monthly Acutual")),
            "cumulative_actual_mm": _to_float_or_none(rec.get("Cumulative Actual")),
        }

    @staticmethod
    def parse_district_rainfall(raw: Any, district_name: str) -> dict[str, Any] | None:
        records = raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])
        rec = _match_admin_record(records, "District", district_name)
        if not rec:
            return None
        cat_code = str(rec.get("Daily Category") or "").strip()
        return {
            "obj_id": rec.get("OBJ_ID") or rec.get("Obj_id"),
            "district": rec.get("District"),
            "date": rec.get("Date"),
            "daily_actual_mm": _to_float_or_none(rec.get("Daily Actual")),
            "daily_normal_mm": _to_float_or_none(rec.get("Daily Normal")),
            "daily_departure_pct": str(rec.get("Daily Departure Per") or "").strip() or None,
            "daily_category_code": cat_code or None,
            "daily_category": RAINFALL_CATEGORY_MAP.get(cat_code, cat_code or None),
            "weekly_actual_mm": _to_float_or_none(rec.get("Weekly Actual")),
            "weekly_normal_mm": _to_float_or_none(rec.get("Weekly Normal")),
            "monthly_actual_mm": _to_float_or_none(rec.get("Monthly Actual")),
        }

    @staticmethod
    def parse_district_warning(raw: Any, district_name: str) -> dict[str, Any] | None:
        records = raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])
        rec = _match_admin_record(records, "District", district_name)
        if not rec:
            return None
        c_code = str(rec.get("Day1_Color") or "1").strip()
        color_meta = WARNING_COLOR_MAP.get(c_code, {"label": f"Code {c_code}", "hex": "#64748b"})
        raw_codes = [c.strip() for c in str(rec.get("Day_1") or "1").split(",") if c.strip()]
        descriptions = [WARNING_CODE_MAP.get(c, f"IMD Warning Code {c}") for c in raw_codes]
        return {
            "obj_id": rec.get("Obj_id"),
            "district": rec.get("District"),
            "date_of_issue": rec.get("Date"),
            "utc_time": rec.get("UTC"),
            "warning_category": color_meta["label"],
            "warning_color_hex": color_meta["hex"],
            "day1_codes": raw_codes,
            "warning_description": "; ".join(descriptions),
            "day2_color": WARNING_COLOR_MAP.get(str(rec.get("Day2_Color") or "").strip(), {}).get("label"),
            "day3_color": WARNING_COLOR_MAP.get(str(rec.get("Day3_Color") or "").strip(), {}).get("label"),
        }

    @staticmethod
    def parse_district_nowcast(raw: Any, district_name: str) -> dict[str, Any] | None:
        records = raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])
        rec = _match_admin_record(records, "Station", district_name) or _match_admin_record(records, "District", district_name)
        if not rec:
            return None
        active_cats = []
        for cat_key, desc in NOWCAST_CATEGORY_MAP.items():
            val = rec.get(cat_key)
            if val not in (None, "", "0", 0) and cat_key != "Cat1":
                active_cats.append(f"{cat_key}: {desc}")
        if rec.get("Cat16") not in (None, "", "0", 0):
            active_cats.append(str(rec.get("Cat16")))
        c_code = str(rec.get("color") or "1").strip()
        color_meta = WARNING_COLOR_MAP.get(c_code, {"label": f"Code {c_code}", "hex": "#64748b"})
        return {
            "station": rec.get("Station") or rec.get("District"),
            "date": rec.get("Date"),
            "issued_time": rec.get("time"),
            "valid_upto": rec.get("vupto"),
            "nowcast_color": color_meta["label"],
            "nowcast_color_hex": color_meta["hex"],
            "nowcast_summary": "; ".join(active_cats) if active_cats else NOWCAST_CATEGORY_MAP["Cat1"],
        }

    @staticmethod
    def parse_forecast_5day(raw: Any, district_name: str) -> dict[str, Any] | None:
        records = raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])
        rec = _match_admin_record(records, "District", district_name)
        if not rec:
            return None
        days = []
        for d in range(1, 6):
            dist_lbl = rec.get(f"day{d}_distribution")
            pct_lbl = rec.get(f"day{d}_distribution_percentage")
            if dist_lbl or pct_lbl:
                days.append({
                    "day": d,
                    "distribution": dist_lbl,
                    "stations_pct": pct_lbl,
                    "color": rec.get(f"day{d}_color"),
                })
        return {
            "date_obs": rec.get("date_obs"),
            "district": rec.get("District"),
            "state": rec.get("State"),
            "days": days,
        }

    def fetch(self, state_name: str = "", district_name: str = "") -> dict[str, Any]:
        """Fetch available IMD state/district rainfall, warnings, nowcast & forecast safely."""
        if not self.is_configured():
            return {
                "provider": self.name,
                "status": "Not configured",
                "attribution": self.attribution,
                "docs_url": self.docs_url,
                "is_cached": False,
                "state_rainfall": None,
                "district_rainfall": None,
                "district_warning": None,
                "district_nowcast": None,
                "rainfall_forecast": None,
                "last_updated": None,
                "message": "Set IMD_API_KEY in environment variables (.env) to query api.imd.gov.in.",
            }

        raw_sr, meta_sr = self._fetch_endpoint("staterainfall")
        raw_dr, meta_dr = self._fetch_endpoint("districtrainfall")
        raw_dw, meta_dw = self._fetch_endpoint("districtwarning")
        raw_dn, meta_dn = self._fetch_endpoint("districtnowcast")
        raw_fc, meta_fc = self._fetch_endpoint("state_district_rainfall_forecast")

        metas = [meta_sr, meta_dr, meta_dw, meta_dn, meta_fc]
        any_connected = any(m.get("status") == "Connected" for m in metas)
        any_cached = any(m.get("is_cached") for m in metas if m.get("status") == "Connected")
        first_err = next((m.get("error") for m in metas if m.get("error")), None)

        return {
            "provider": self.name,
            "status": "Connected" if any_connected else (meta_sr.get("status") or "Unavailable"),
            "attribution": self.attribution,
            "docs_url": self.docs_url,
            "is_cached": any_cached,
            "state_rainfall": self.parse_state_rainfall(raw_sr, state_name) if raw_sr is not None else None,
            "district_rainfall": self.parse_district_rainfall(raw_dr, district_name) if raw_dr is not None else None,
            "district_warning": self.parse_district_warning(raw_dw, district_name) if raw_dw is not None else None,
            "district_nowcast": self.parse_district_nowcast(raw_dn, district_name) if raw_dn is not None else None,
            "rainfall_forecast": self.parse_forecast_5day(raw_fc, district_name) if raw_fc is not None else None,
            "last_updated": self.last_update_time,
            "message": first_err,
        }

