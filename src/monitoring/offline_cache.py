"""
Offline Synchronization & Telemetry Cache Module (Phase 9).

Maintains persistent offline caches for weather observations, India regional
telemetry, and emergency facilities so that LandslideGuard remains functional
during mountain field deployments or upstream API service outages.

INTEGRITY POLICY:
Never presents cached information as current live data. Explicitly tracks
'last_successful_sync', 'cached_timestamp', and 'is_live' flags.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache/offline_sync")


class OfflineSyncCache:
    """Manages file-based synchronization caches for offline monitoring."""

    def __init__(self, cache_dir: Path | str = CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in key)
        return self.cache_dir / f"{safe_key}.json"

    def put(
        self,
        key: str,
        payload: dict[str, Any],
        provider_name: str = "Open-Meteo",
    ) -> None:
        """Store payload in persistent offline cache with sync metadata."""
        now_iso = datetime.now(timezone.utc).isoformat()
        wrapped = {
            "cache_key": key,
            "provider_name": provider_name,
            "cached_at": now_iso,
            "last_successful_sync": now_iso,
            "is_live": False,  # Once saved to cache, future reads are from cache
            "data": payload,
        }
        try:
            with open(self._cache_path(key), "w", encoding="utf-8") as f:
                json.dump(wrapped, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to write offline cache for '%s': %s", key, exc)

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve payload from offline cache.

        Returns None if key is not cached. Returned dict has 'is_live': False
        and explicit sync timestamps.
        """
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                wrapped = json.load(f)
            wrapped["is_live"] = False
            wrapped["from_offline_cache"] = True
            return wrapped
        except Exception as exc:
            logger.warning("Failed to read offline cache for '%s': %s", key, exc)
            return None

    def get_offline_status(self) -> dict[str, Any]:
        """Summarize current offline cache health, last sync times, and cached keys."""
        cached_files = list(self.cache_dir.glob("*.json"))
        records: list[dict[str, Any]] = []

        latest_sync: str | None = None

        for f in cached_files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    item = json.load(fp)
                sync_time = item.get("last_successful_sync")
                if sync_time and (latest_sync is None or sync_time > latest_sync):
                    latest_sync = sync_time
                records.append({
                    "cache_key": item.get("cache_key", f.stem),
                    "provider": item.get("provider_name", "External Provider"),
                    "cached_at": item.get("cached_at"),
                    "last_successful_sync": sync_time,
                })
            except Exception:
                pass

        now_dt = datetime.now(timezone.utc)
        sync_age_str = "No synchronization recorded"
        if latest_sync:
            try:
                sync_dt = datetime.fromisoformat(latest_sync)
                mins = int((now_dt - sync_dt).total_seconds() / 60)
                sync_age_str = f"{mins} min ago" if mins < 60 else f"{mins // 60}h {mins % 60}m ago"
            except Exception:
                sync_age_str = latest_sync

        return {
            "offline_cache_enabled": True,
            "total_cached_records": len(records),
            "last_successful_synchronization": latest_sync or "Unavailable",
            "sync_age": sync_age_str,
            "cached_entries": records,
            "integrity_rule": "Never present cached information as current live data.",
        }


_cache_singleton: OfflineSyncCache | None = None


def get_offline_cache() -> OfflineSyncCache:
    """Retrieve or initialize the singleton OfflineSyncCache."""
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = OfflineSyncCache()
    return _cache_singleton

