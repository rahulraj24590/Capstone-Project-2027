"""
Terrain data provider for LandslideGuard.

Fetches elevation data via the Open-Elevation API (SRTM-based).
Slope is derived from elevation using a numerical gradient approximation.

LIMITATION: Open-Elevation provides single-point elevation values, not
DEMs (Digital Elevation Models). Accurate slope computation requires
a local raster DEM. The slope approximation here is based on sampling
neighbouring points, which may be inaccurate at coarse resolutions.

Attribution: Elevation data from SRTM via Open-Elevation API
(https://api.open-elevation.com/)
"""

from __future__ import annotations

import math
import os
from typing import Any

from src.data_providers.base import BaseProvider, ProviderError
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

# Offset in degrees for slope approximation (≈ 1 km at equator)
SLOPE_SAMPLE_OFFSET = 0.009  # degrees ≈ 1 km


class TerrainProvider(BaseProvider):
    """Elevation and slope data provider using Open-Elevation API."""

    name = "Open-Elevation"
    attribution = "Elevation data from SRTM via Open-Elevation API"

    def __init__(self) -> None:
        super().__init__()
        cfg = get_setting("providers", "terrain") or {}
        self.base_url = cfg.get("base_url", "https://api.open-elevation.com/api/v1")
        self.timeout_seconds = int(cfg.get("timeout_seconds", 15))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.retry_backoff_factor = int(cfg.get("retry_backoff_factor", 2))
        self.cache_ttl_seconds = int(cfg.get("cache_ttl_seconds", 86400))  # 24h

    def fetch(self, latitude: float, longitude: float) -> dict | None:
        """Fetch elevation for a single point.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.

        Returns:
            Dict with 'elevation_m' and metadata, or None if unavailable.
        """
        params = {"locations": f"{latitude},{longitude}"}
        url = f"{self.base_url}/lookup"
        raw = self._get_cached("terrain", {"url": url, **params})

        if raw is None:
            return None

        results = raw.get("results", [])
        if not results:
            logger.warning("No elevation data returned for (%.4f, %.4f).", latitude, longitude)
            return None

        elevation = results[0].get("elevation")
        if elevation is None:
            return None

        return {
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": float(elevation),
            "source": self.attribution,
            "resolution_note": "Point elevation from SRTM (~90m resolution)",
        }

    def fetch_batch(self, locations: list[dict]) -> list[dict | None]:
        """Fetch elevation for multiple points in one API call.

        Args:
            locations: List of dicts with 'latitude' and 'longitude' keys.

        Returns:
            List of elevation dicts (None for failed lookups).
        """
        if not locations:
            return []

        loc_str = "|".join(f"{loc['latitude']},{loc['longitude']}" for loc in locations)
        params = {"locations": loc_str}
        url = f"{self.base_url}/lookup"

        try:
            raw = self._get(url, params=params)
        except ProviderError as e:
            logger.warning("[TerrainProvider] Batch fetch failed: %s", e)
            return [None] * len(locations)

        results = raw.get("results", [])
        output = []
        for i, res in enumerate(results):
            elev = res.get("elevation")
            if elev is not None:
                output.append({
                    "latitude": locations[i]["latitude"],
                    "longitude": locations[i]["longitude"],
                    "elevation_m": float(elev),
                    "source": self.attribution,
                })
            else:
                output.append(None)

        return output

    def estimate_slope(
        self,
        latitude: float,
        longitude: float,
        offset_deg: float = SLOPE_SAMPLE_OFFSET,
    ) -> dict | None:
        """Approximate slope by sampling elevation at neighbouring points.

        This is a coarse approximation using the finite difference method.
        For accurate slope, a local DEM raster (e.g., SRTM GeoTIFF) should
        be used with rasterio or gdaldem.

        Args:
            latitude: Centre point latitude.
            longitude: Centre point longitude.
            offset_deg: Angular offset for neighbouring points (degrees).

        Returns:
            Dict with 'slope_degrees' and methodology note, or None.
        """
        locs = [
            {"latitude": latitude, "longitude": longitude},
            {"latitude": latitude + offset_deg, "longitude": longitude},
            {"latitude": latitude, "longitude": longitude + offset_deg},
        ]

        elevations = self.fetch_batch(locs)
        if any(e is None for e in elevations):
            logger.warning("Cannot compute slope: some elevation values missing.")
            return None

        e_centre = elevations[0]["elevation_m"]
        e_north = elevations[1]["elevation_m"]
        e_east = elevations[2]["elevation_m"]

        # Convert offset from degrees to metres (~111km per degree)
        offset_m = offset_deg * 111_000

        dz_north = e_north - e_centre
        dz_east = e_east - e_centre

        gradient_magnitude = math.sqrt(
            (dz_north / offset_m) ** 2 + (dz_east / offset_m) ** 2
        )
        slope_degrees = math.degrees(math.atan(gradient_magnitude))

        return {
            "latitude": latitude,
            "longitude": longitude,
            "slope_degrees": round(slope_degrees, 2),
            "elevation_m": e_centre,
            "source": self.attribution,
            "methodology": (
                "Finite-difference slope approximation using SRTM elevation "
                f"at three points with {offset_deg:.4f}° offset (~{offset_m:.0f}m). "
                "This is a coarse approximation; local DEM analysis is preferred."
            ),
        }

    def get_terrain_features(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Get all available terrain features for a location.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.

        Returns:
            Dict with elevation_m, slope_degrees, and data quality info.
        """
        result: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": None,
            "slope_degrees": None,
            "terrain_source": self.attribution,
        }

        slope_data = self.estimate_slope(latitude, longitude)
        if slope_data:
            result["elevation_m"] = slope_data["elevation_m"]
            result["slope_degrees"] = slope_data["slope_degrees"]
            result["slope_methodology"] = slope_data["methodology"]
        else:
            # Fallback: try elevation-only
            elev_data = self.fetch(latitude, longitude)
            if elev_data:
                result["elevation_m"] = elev_data["elevation_m"]
                result["slope_warning"] = "Slope could not be computed; only elevation available."

        return result
