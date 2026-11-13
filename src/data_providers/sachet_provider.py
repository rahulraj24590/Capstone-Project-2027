"""Official NDMA SACHET Common Alerting Protocol (CAP) alert provider.

Uses documented public endpoints on https://sachet.ndma.gov.in/:
- JSON Alert Details feed: https://sachet.ndma.gov.in/cap_public_website/FetchAllAlertDetails
- All-India RSS CAP XML fallback: https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml
- Official CAP XML detail link: https://sachet.ndma.gov.in/cap_public_website/FetchXMLFile?identifier=<id>

Maintains strict separation between official NDMA government warnings and
LandslideGuard experimental ML alerts. Supports HTTP ETag caching (If-None-Match / 304)
and falls back cleanly to 'Link only' / 'Unavailable' if endpoints fail.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.data_providers.base import BaseProvider, CACHE_DIR
from src.data_providers.india_boundaries import normalize_admin_name
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

SACHET_PORTAL_URL = "https://sachet.ndma.gov.in/"
SACHET_JSON_URL = "https://sachet.ndma.gov.in/cap_public_website/FetchAllAlertDetails"
SACHET_RSS_URL = "https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml"
SACHET_XML_DETAIL_URL = "https://sachet.ndma.gov.in/cap_public_website/FetchXMLFile?identifier={identifier}"


class SACHETProvider(BaseProvider):
    """Retrieves and normalizes public official alerts from NDMA SACHET."""

    name = "NDMA SACHET"
    attribution = "Official disaster warnings via NDMA SACHET CAP Portal (sachet.ndma.gov.in)"

    def __init__(self) -> None:
        super().__init__()
        cfg = get_setting("providers", "sachet") or {}
        self.base_url = cfg.get("base_url", "https://sachet.ndma.gov.in/cap_public_website")
        self.portal_url = cfg.get("portal_url", SACHET_PORTAL_URL)
        self.timeout_seconds = int(cfg.get("timeout_seconds", 12))
        self.cache_ttl_seconds = int(cfg.get("cache_ttl_seconds", 600))
        self._etag_cache_file = CACHE_DIR / "sachet_etag_cache.json"

    @staticmethod
    def _parse_centroid(raw_centroid: Any) -> tuple[float | None, float | None]:
        """Parse SACHET 'lon,lat' centroid string into (lat, lon)."""
        if not raw_centroid or "," not in str(raw_centroid):
            return None, None
        parts = [p.strip() for p in str(raw_centroid).split(",")]
        if len(parts) != 2:
            return None, None
        try:
            lon = float(parts[0])
            lat = float(parts[1])
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass
        return None, None

    @classmethod
    def parse_json_alerts(cls, payload: list[dict[str, Any]], is_cached: bool = False) -> list[dict[str, Any]]:
        """Normalize raw FetchAllAlertDetails JSON objects."""
        out: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            ident = str(item.get("identifier") or item.get("alert_id_sdma_autoinc") or "").strip()
            lat, lon = cls._parse_centroid(item.get("centroid"))
            area_desc = str(item.get("area_description") or "").strip()
            disaster_type = str(item.get("disaster_type") or "Weather / Hazard Advisory").strip()
            out.append({
                "identifier": ident,
                "alert_source": str(item.get("alert_source") or "NDMA SACHET").strip(),
                "alert_title": disaster_type,
                "warning_message": str(item.get("warning_message") or "").strip(),
                "affected_region": area_desc,
                "severity": str(item.get("severity") or "UNKNOWN").strip(),
                "severity_level": str(item.get("severity_level") or "").strip() or None,
                "severity_color": str(item.get("severity_color") or "").strip().lower() or None,
                "issued_at": str(item.get("effective_start_time") or "").strip() or None,
                "valid_until": str(item.get("effective_end_time") or "").strip() or None,
                "centroid_lat": lat,
                "centroid_lon": lon,
                "area_covered_sqkm": item.get("area_covered"),
                "official_url": SACHET_XML_DETAIL_URL.format(identifier=ident) if ident else SACHET_PORTAL_URL,
                "portal_url": SACHET_PORTAL_URL,
                "is_official": True,
                "is_cached": is_cached,
            })
        return out

    @staticmethod
    def parse_rss_alerts(xml_text: str, is_cached: bool = False) -> list[dict[str, Any]]:
        """Fallback parser for SACHET All-India CAP RSS XML feed."""
        alerts: list[dict[str, Any]] = []
        root = ET.fromstring(xml_text)
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "Official CAP Advisory").strip()
            link = (item.findtext("link") or SACHET_PORTAL_URL).strip()
            pub_date = (item.findtext("pubDate") or "").strip() or None
            desc = (item.findtext("description") or "").strip()
            author = (item.findtext("author") or "NDMA SACHET").strip()
            guid = (item.findtext("guid") or link).strip()
            alerts.append({
                "identifier": guid,
                "alert_source": author,
                "alert_title": title,
                "warning_message": desc,
                "affected_region": title,
                "severity": "ADVISORY",
                "severity_level": None,
                "severity_color": None,
                "issued_at": pub_date,
                "valid_until": None,
                "centroid_lat": None,
                "centroid_lon": None,
                "area_covered_sqkm": None,
                "official_url": link,
                "portal_url": SACHET_PORTAL_URL,
                "is_official": True,
                "is_cached": is_cached,
            })
        return alerts

    @staticmethod
    def filter_alerts_for_region(
        alerts: list[dict[str, Any]],
        state_name: str = "",
        district_name: str = "",
    ) -> list[dict[str, Any]]:
        """Return alerts matching the selected district or state text cleanly."""
        if not alerts:
            return []
        s_norm = normalize_admin_name(state_name).lower()
        d_norm = normalize_admin_name(district_name).lower()
        matched: list[dict[str, Any]] = []
        for a in alerts:
            haystack = " ".join([
                normalize_admin_name(a.get("affected_region")),
                normalize_admin_name(a.get("warning_message")),
                normalize_admin_name(a.get("alert_title")),
            ]).lower()
            if (d_norm and d_norm in haystack) or (s_norm and s_norm in haystack):
                matched.append(a)
        return matched

    def fetch(
        self,
        state_name: str = "",
        district_name: str = "",
    ) -> dict[str, Any]:
        """Fetch live active NDMA SACHET alerts with RSS + cache fallback."""
        headers: dict[str, str] = {}
        cached_entry: dict[str, Any] = {}
        if self._etag_cache_file.exists():
            try:
                cached_entry = json.loads(self._etag_cache_file.read_text(encoding="utf-8"))
                if cached_entry.get("etag"):
                    headers["If-None-Match"] = cached_entry["etag"]
            except Exception:
                cached_entry = {}

        # 1. Try primary official FetchAllAlertDetails JSON endpoint
        try:
            resp = self._session.get(SACHET_JSON_URL, headers=headers, timeout=self.timeout_seconds)
            if resp.status_code == 304 and isinstance(cached_entry.get("alerts"), list):
                all_alerts = self.parse_json_alerts(cached_entry["alerts"], is_cached=True)
                regional = self.filter_alerts_for_region(all_alerts, state_name, district_name)
                return {
                    "provider": self.name,
                    "status": "Connected",
                    "portal_url": self.portal_url,
                    "attribution": self.attribution,
                    "is_cached": True,
                    "last_updated": cached_entry.get("fetched_at"),
                    "total_india_alerts": len(all_alerts),
                    "regional_alerts": regional,
                    "all_alerts": all_alerts,
                    "error": None,
                }
            resp.raise_for_status()
            raw_list = resp.json()
            if not isinstance(raw_list, list):
                raise ValueError("Expected list response from SACHET FetchAllAlertDetails.")
            now_iso = datetime.now(timezone.utc).isoformat()
            self._last_successful_fetch = datetime.now(timezone.utc)
            try:
                self._etag_cache_file.write_text(
                    json.dumps({"etag": resp.headers.get("ETag"), "fetched_at": now_iso, "alerts": raw_list}),
                    encoding="utf-8",
                )
            except Exception:
                pass
            all_alerts = self.parse_json_alerts(raw_list, is_cached=False)
            regional = self.filter_alerts_for_region(all_alerts, state_name, district_name)
            return {
                "provider": self.name,
                "status": "Connected",
                "portal_url": self.portal_url,
                "attribution": self.attribution,
                "is_cached": False,
                "last_updated": now_iso,
                "total_india_alerts": len(all_alerts),
                "regional_alerts": regional,
                "all_alerts": all_alerts,
                "error": None,
            }
        except Exception as json_exc:
            logger.warning("[SACHET] FetchAllAlertDetails failed (%s); trying RSS fallback.", json_exc)

        # 2. Fallback to official CAP RSS feed (with ETag support)
        try:
            r_rss = self._session.get(SACHET_RSS_URL, headers=headers, timeout=self.timeout_seconds)
            r_rss.raise_for_status()
            now_iso = datetime.now(timezone.utc).isoformat()
            self._last_successful_fetch = datetime.now(timezone.utc)
            all_alerts = self.parse_rss_alerts(r_rss.text, is_cached=False)
            regional = self.filter_alerts_for_region(all_alerts, state_name, district_name)
            return {
                "provider": self.name,
                "status": "Connected",
                "portal_url": self.portal_url,
                "attribution": self.attribution,
                "is_cached": False,
                "last_updated": now_iso,
                "total_india_alerts": len(all_alerts),
                "regional_alerts": regional,
                "all_alerts": all_alerts,
                "error": None,
            }
        except Exception as rss_exc:
            logger.warning("[SACHET] RSS fallback also unavailable: %s", rss_exc)

        # 3. Cached fallback or explicit Link-only degraded state
        if isinstance(cached_entry.get("alerts"), list) and cached_entry["alerts"]:
            all_alerts = self.parse_json_alerts(cached_entry["alerts"], is_cached=True)
            return {
                "provider": self.name,
                "status": "Unavailable",
                "portal_url": self.portal_url,
                "attribution": self.attribution,
                "is_cached": True,
                "last_updated": cached_entry.get("fetched_at"),
                "total_india_alerts": len(all_alerts),
                "regional_alerts": self.filter_alerts_for_region(all_alerts, state_name, district_name),
                "all_alerts": all_alerts,
                "error": "Live SACHET feed unreachable; showing stale cached alerts.",
            }

        return {
            "provider": self.name,
            "status": "Link only",
            "portal_url": self.portal_url,
            "attribution": self.attribution,
            "is_cached": False,
            "last_updated": None,
            "total_india_alerts": 0,
            "regional_alerts": [],
            "all_alerts": [],
            "error": "SACHET API unreachable; use official portal link directly.",
        }

