"""
Abstract base class for all data providers in LandslideGuard.

All external data sources (weather, terrain, etc.) must inherit from
BaseProvider and implement the required interface.

Design principles:
- Providers must distinguish observed, forecast, and cached data.
- Providers must implement retry logic with exponential backoff.
- API keys must be read from environment variables, never hardcoded.
- Cached responses must be timestamped and not presented as live data.
- Graceful degradation: providers return None and log warnings on failure.
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(".cache/providers")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(prefix: str, params: dict) -> str:
    """Generate a deterministic cache key from a request prefix and params."""
    content = json.dumps(params, sort_keys=True)
    digest = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _read_cache_entry(key: str, ttl_seconds: int) -> tuple[dict | None, dict | None]:
    """Read cached payload together with provenance metadata if fresh."""
    cache_file = CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None, None
    with open(cache_file, "r", encoding="utf-8") as f:
        entry = json.load(f)
    cached_at = entry.get("_cached_at", 0)
    age = time.time() - cached_at
    if age > ttl_seconds:
        logger.debug("Cache expired for key %s (age %.0fs)", key, age)
        return None, None
    logger.debug("Cache hit for key %s (age %.0fs)", key, age)
    cached_iso = datetime.fromtimestamp(cached_at, tz=timezone.utc).isoformat() if cached_at else None
    meta = {
        "is_cached": True,
        "cached_at": cached_iso,
        "age_seconds": round(max(0.0, age), 2),
    }
    return entry.get("data"), meta


def _read_cache(key: str, ttl_seconds: int) -> dict | None:
    """Return cached data payload when fresh, else None."""
    data, _ = _read_cache_entry(key, ttl_seconds)
    return data


def _write_cache(key: str, data: dict) -> None:
    """Write data to cache with a timestamp.

    Args:
        key: Cache key.
        data: Response data to cache.
    """
    cache_file = CACHE_DIR / f"{key}.json"
    entry = {"_cached_at": time.time(), "data": data}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(entry, f)


class ProviderError(Exception):
    """Raised when a data provider request fails."""
    pass


class BaseProvider(abc.ABC):
    """Abstract data provider interface."""

    name: str = "BaseProvider"
    base_url: str = ""
    timeout_seconds: int = 10
    max_retries: int = 3
    retry_backoff_factor: int = 2
    cache_ttl_seconds: int = 3600
    attribution: str = ""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "LandslideGuard/1.0 (academic research)"})
        self._last_successful_fetch: datetime | None = None
        self._last_fetch_meta: dict[str, Any] = {}

    @abc.abstractmethod
    def fetch(self, **kwargs) -> dict | None:
        """Fetch data from the provider.

        Args:
            **kwargs: Provider-specific parameters.

        Returns:
            Response data dict or None if unavailable.
        """
        ...

    def _get(self, url: str, params: dict | None = None) -> dict:
        """Perform a GET request with retry logic.

        Args:
            url: Full URL to fetch.
            params: Query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            ProviderError: If the request fails after all retries.
        """
        @retry(
            retry=retry_if_exception_type((requests.RequestException, ProviderError)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_backoff_factor, min=1, max=60),
            before_sleep=before_sleep_log(logger, log_level=20),
            reraise=True,
        )
        def _attempt():
            resp = self._session.get(
                url, params=params, timeout=self.timeout_seconds
            )
            if resp.status_code == 429:
                headers = getattr(resp, "headers", None) or {}
                retry_after = headers.get("Retry-After") if isinstance(headers, dict) else None
                suffix = f" Retry-After: {retry_after}." if retry_after else ""
                raise ProviderError(
                    f"Rate limited by {self.name} (429).{suffix} "
                    "Consider reducing request frequency."
                )
            resp.raise_for_status()
            return resp.json()

        try:
            data = _attempt()
            self._last_successful_fetch = datetime.now(timezone.utc)
            return data
        except requests.exceptions.ConnectionError:
            raise ProviderError(
                f"Cannot connect to {self.name}. "
                "Check internet connection or service availability."
            )
        except requests.exceptions.Timeout:
            raise ProviderError(
                f"Request to {self.name} timed out after {self.timeout_seconds}s."
            )
        except requests.RequestException as exc:
            raise ProviderError(f"Request to {self.name} failed: {exc}")

    def _get_cached(
        self,
        cache_key_prefix: str,
        params: dict,
        bypass_cache: bool = False,
    ) -> dict | None:
        """Fetch with caching unless ``bypass_cache`` forces a live refresh.

        Args:
            cache_key_prefix: Prefix for the cache key.
            params: Request parameters (also used as cache key input). May
                include a ``"url"`` key to override ``self.base_url``.
            bypass_cache: If True, skip disk cache lookup and request fresh data.

        Returns:
            Cached or freshly fetched data, or None on failure.
        """
        key = _cache_key(cache_key_prefix, params)
        if not bypass_cache:
            cached, cache_meta = _read_cache_entry(key, self.cache_ttl_seconds)
            if cached is not None:
                meta = dict(cache_meta or {"is_cached": True, "cached_at": None, "age_seconds": None})
                meta.setdefault("status", "cached")
                meta.setdefault("rate_limited", False)
                meta.setdefault("error", None)
                self._last_fetch_meta = meta
                return cached

        url = params.get("url") or self.base_url
        request_params = {k: v for k, v in params.items() if k != "url"}
        try:
            data = self._get(url, params=request_params)
            _write_cache(key, data)
            self._last_fetch_meta = {
                "is_cached": False,
                "cached_at": None,
                "age_seconds": 0.0,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "status": "live",
                "rate_limited": False,
                "error": None,
            }
            return data
        except ProviderError as e:
            logger.warning("[%s] Request failed: %s", self.name, e)
            err_str = str(e)
            is_rl = "429" in err_str or "rate limited" in err_str.lower()
            self._last_fetch_meta = {
                "is_cached": False,
                "cached_at": None,
                "age_seconds": None,
                "status": "rate_limited" if is_rl else "error",
                "rate_limited": is_rl,
                "error": err_str,
            }
            return None

    @property
    def last_update_time(self) -> str | None:
        """Return ISO-formatted timestamp of last successful fetch."""
        if self._last_successful_fetch:
            return self._last_successful_fetch.isoformat()
        return None

    @property
    def last_fetch_metadata(self) -> dict[str, Any]:
        """Return provenance metadata for the most recent _get_cached call."""
        return dict(getattr(self, "_last_fetch_meta", {}))

    def health_check(self) -> dict:
        """Check provider availability.

        Returns:
            Dict with 'available' bool and optional 'error' message.
        """
        try:
            self._get(self.base_url)
            return {"available": True, "provider": self.name}
        except ProviderError as e:
            return {"available": False, "provider": self.name, "error": str(e)}


# Alias for backwards compatibility and test imports
DataProvider = BaseProvider
