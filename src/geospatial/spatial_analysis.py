"""
Spatial analysis utilities for LandslideGuard.

Provides spatial aggregation, clustering, and nearest-neighbour queries
for landslide observation datasets.

NOTE: geopandas is used where available. If it is not installed, pure
pandas/numpy implementations are used as fallback.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Check geopandas availability
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
    logger.info("geopandas available for spatial operations.")
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning("geopandas not available; using pure pandas/numpy spatial fallbacks.")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points (km).

    Args:
        lat1, lon1: First point coordinates (decimal degrees).
        lat2, lon2: Second point coordinates (decimal degrees).

    Returns:
        Distance in kilometres.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def filter_by_bounding_box(
    df: pd.DataFrame,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> pd.DataFrame:
    """Filter observations to a geographic bounding box.

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns.
        min_lat, max_lat: Latitude bounds.
        min_lon, max_lon: Longitude bounds.

    Returns:
        Filtered DataFrame.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning("Cannot apply bounding box: missing coordinate columns.")
        return df

    mask = (
        df["latitude"].between(min_lat, max_lat) &
        df["longitude"].between(min_lon, max_lon)
    )
    result = df[mask].copy()
    logger.debug("Bounding box filter: %d → %d rows.", len(df), len(result))
    return result


def compute_spatial_density(
    df: pd.DataFrame,
    grid_size_deg: float = 1.0,
) -> pd.DataFrame:
    """Compute event count density on a regular geographic grid.

    Args:
        df: DataFrame with 'latitude', 'longitude' columns.
        grid_size_deg: Grid cell size in degrees.

    Returns:
        DataFrame with grid cell centres and event counts.
    """
    if df.empty or "latitude" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["lat_bin"] = (df["latitude"] // grid_size_deg) * grid_size_deg + grid_size_deg / 2
    df["lon_bin"] = (df["longitude"] // grid_size_deg) * grid_size_deg + grid_size_deg / 2

    density = (
        df.groupby(["lat_bin", "lon_bin"])
        .size()
        .reset_index(name="event_count")
    )
    return density


def nearest_observations(
    lat: float,
    lon: float,
    df: pd.DataFrame,
    n: int = 5,
    max_km: float = 500.0,
) -> pd.DataFrame:
    """Find the N nearest observations within a maximum distance.

    Args:
        lat: Query latitude.
        lon: Query longitude.
        df: DataFrame with 'latitude' and 'longitude'.
        n: Maximum number of results.
        max_km: Maximum distance threshold (km).

    Returns:
        DataFrame of nearest observations sorted by distance.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["_distance_km"] = df.apply(
        lambda r: haversine_distance(lat, lon, r["latitude"], r["longitude"]),
        axis=1,
    )
    result = df[df["_distance_km"] <= max_km].nsmallest(n, "_distance_km")
    result = result.drop(columns=["_distance_km"])
    return result


def to_geodataframe(df: pd.DataFrame) -> Any:
    """Convert a DataFrame to a GeoDataFrame.

    Falls back to returning the plain DataFrame if geopandas is not available.

    Args:
        df: DataFrame with 'latitude' and 'longitude'.

    Returns:
        GeoDataFrame or plain DataFrame.
    """
    if not GEOPANDAS_AVAILABLE:
        logger.warning("Cannot create GeoDataFrame: geopandas not available.")
        return df

    if df.empty:
        return gpd.GeoDataFrame()

    df = df.copy()
    df["geometry"] = df.apply(
        lambda r: Point(r["longitude"], r["latitude"]), axis=1
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf


def summarise_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observation counts by country.

    Args:
        df: Observations DataFrame.

    Returns:
        DataFrame with country-level event counts.
    """
    if "country" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("country")
        .agg(
            event_count=("latitude", "count"),
            avg_lat=("latitude", "mean"),
            avg_lon=("longitude", "mean"),
        )
        .reset_index()
        .sort_values("event_count", ascending=False)
    )
    return summary
