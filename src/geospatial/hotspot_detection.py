"""
Phase 7 Landslide Hotspot Detection and Spatial Clustering for LandslideGuard.

Provides:
1. DBSCAN density-based spatial clustering on spherical haversine coordinates.
2. Kernel Density Estimation (KDE) continuous density surfaces and percentile hotspot regions.
3. Cluster geometric properties (centroids, bounding boxes, convex hulls, event density).
4. Authentic observation filtering and strict anti-hallucination disclaimers.

SCIENTIFIC INTEGRITY GUARD:
"Do not call an area a future landslide hotspot based only on historical clustering."
Historical clustering identifies spatial concentration of recorded past events only;
it does not predict future landslide activation without dynamic environmental triggering.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Silence Windows loky wmic CPU detection warning
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN

from src.geospatial.spatial_analysis import drop_invalid_coordinates, haversine_distance
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

try:
    from shapely.geometry import MultiPoint, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


HOTSPOT_SCIENTIFIC_DISCLAIMER = (
    "CRITICAL SCIENTIFIC NOTICE: Historical hotspot detection reflects the spatial concentration "
    "and density of recorded past landslide events. It does NOT predict future landslide occurrence "
    "or classify an area as a future landslide hotspot based only on historical clustering. "
    "Future landslide activation requires dynamic triggering factors (such as extreme precipitation, "
    "pore-water pressure increases, and slope destabilization)."
)


def _approximate_polygon_area_sq_km(coords_lat_lon: list[list[float]]) -> float:
    """Calculate approximate geodesic surface area (km²) of a WGS84 polygon using Gauss's area formula."""
    if len(coords_lat_lon) < 3:
        return 0.1
    # Project to approximate local metres using mean latitude
    mean_lat = float(np.mean([c[0] for c in coords_lat_lon]))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = max(1.0, 111_320.0 * math.cos(math.radians(mean_lat)))

    x = [c[1] * m_per_deg_lon for c in coords_lat_lon]
    y = [c[0] * m_per_deg_lat for c in coords_lat_lon]

    # Shoelace formula in square metres
    area_m2 = 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(len(x) - 1)))
    area_km2 = area_m2 / 1_000_000.0
    return max(0.1, round(area_km2, 2))


def detect_landslide_hotspots(
    df: pd.DataFrame | None,
    method: str = "both",  # "dbscan", "kde", or "both"
    eps_km: float = 25.0,
    min_samples: int = 3,
    authentic_only: bool = False,
    kde_grid_size: int = 30,
) -> dict[str, Any]:
    """Detect spatial landslide clusters and continuous density hotspots from historical observations.

    Args:
        df: DataFrame containing at least 'latitude' and 'longitude' columns.
        method: Clustering/density method ('dbscan', 'kde', or 'both').
        eps_km: DBSCAN neighborhood search radius in kilometres.
        min_samples: DBSCAN minimum points to form a core cluster.
        authentic_only: If True, filters out synthetic development records (is_synthetic == 1).
        kde_grid_size: Resolution of regular grid for KDE density surface.

    Returns:
        Structured dictionary with clusters, noise events, KDE density surface, hotspot regions,
        and mandatory anti-hallucination disclaimer.
    """
    cfg = get_setting("geospatial", "hotspot") or {}
    if eps_km <= 0:
        eps_km = float(cfg.get("eps_km", 25.0))
    if min_samples < 2:
        min_samples = int(cfg.get("min_samples", 3))

    if df is None or df.empty:
        return {
            "available": False,
            "reason": "Observation dataset is empty or None.",
            "disclaimer": HOTSPOT_SCIENTIFIC_DISCLAIMER,
        }

    valid_df, dropped_count = drop_invalid_coordinates(df)
    if valid_df.empty:
        return {
            "available": False,
            "reason": f"All {dropped_count} records had invalid or non-finite WGS84 coordinates.",
            "disclaimer": HOTSPOT_SCIENTIFIC_DISCLAIMER,
        }

    if authentic_only and "is_synthetic" in valid_df.columns:
        synth_vals = pd.to_numeric(valid_df["is_synthetic"], errors="coerce").fillna(1).astype(int)
        auth_df = valid_df[synth_vals == 0].copy()
        if auth_df.empty:
            return {
                "available": False,
                "authentic_only": True,
                "reason": (
                    "No authentic historical observations found (all records are synthetic); "
                    "refusing to identify real-world hotspots on development data."
                ),
                "disclaimer": HOTSPOT_SCIENTIFIC_DISCLAIMER,
            }
        valid_df = auth_df

    total_pts = len(valid_df)
    if total_pts < min_samples:
        return {
            "available": False,
            "reason": f"Insufficient observations ({total_pts}) for clustering (minimum required: {min_samples}).",
            "disclaimer": HOTSPOT_SCIENTIFIC_DISCLAIMER,
        }

    lats = valid_df["latitude"].to_numpy(dtype=float)
    lons = valid_df["longitude"].to_numpy(dtype=float)

    # ── 1. DBSCAN Spatial Clustering (Haversine Metric) ──────────────
    clusters: list[dict[str, Any]] = []
    noise_count = 0
    clustered_count = 0

    if method in {"dbscan", "both"}:
        coords_rad = np.radians(np.column_stack([lats, lons]))
        eps_rad = float(eps_km) / 6371.0
        db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", n_jobs=1)
        labels = db.fit_predict(coords_rad)

        valid_df = valid_df.copy()
        valid_df["cluster_id"] = labels

        unique_labels = sorted(set(labels))
        for cid in unique_labels:
            if cid == -1:
                noise_count = int(np.sum(labels == -1))
                continue

            sub = valid_df[labels == cid]
            n_events = len(sub)
            clustered_count += n_events
            sub_lats = sub["latitude"].to_numpy(dtype=float)
            sub_lons = sub["longitude"].to_numpy(dtype=float)

            c_lat = round(float(np.mean(sub_lats)), 5)
            c_lon = round(float(np.mean(sub_lons)), 5)
            bbox = {
                "min_lat": round(float(np.min(sub_lats)), 5),
                "max_lat": round(float(np.max(sub_lats)), 5),
                "min_lon": round(float(np.min(sub_lons)), 5),
                "max_lon": round(float(np.max(sub_lons)), 5),
            }

            # Convex hull boundary
            hull_coords: list[list[float]] = []
            if SHAPELY_AVAILABLE and n_events >= 3:
                try:
                    mp = MultiPoint([(lon, lat) for lat, lon in zip(sub_lats, sub_lons)])
                    hull = mp.convex_hull
                    if isinstance(hull, Polygon) and not hull.is_empty:
                        hull_coords = [[round(lat, 5), round(lon, 5)] for lon, lat in hull.exterior.coords]
                except Exception as exc:
                    logger.debug("Convex hull generation skipped: %s", exc)

            if not hull_coords:
                # Approximate bounding polygon with 4 corners
                hull_coords = [
                    [bbox["min_lat"], bbox["min_lon"]],
                    [bbox["max_lat"], bbox["min_lon"]],
                    [bbox["max_lat"], bbox["max_lon"]],
                    [bbox["min_lat"], bbox["max_lon"]],
                    [bbox["min_lat"], bbox["min_lon"]],
                ]

            area_km2 = _approximate_polygon_area_sq_km(hull_coords)
            density = round(n_events / area_km2, 3)

            # Metadata properties
            dates = sub["event_date"].dropna().astype(str).tolist() if "event_date" in sub.columns else []
            triggers = sub["trigger"].dropna().astype(str).tolist() if "trigger" in sub.columns else []
            types = sub["landslide_type"].dropna().astype(str).tolist() if "landslide_type" in sub.columns else []
            event_ids = (
                sub["id"].dropna().astype(str).tolist()
                if "id" in sub.columns
                else (sub["event_id"].dropna().astype(str).tolist() if "event_id" in sub.columns else [])
            )

            clusters.append({
                "cluster_id": int(cid),
                "point_count": n_events,
                "centroid": {"latitude": c_lat, "longitude": c_lon},
                "bounding_box": bbox,
                "convex_hull": hull_coords,
                "area_sq_km": area_km2,
                "event_density_per_sq_km": density,
                "date_range": {
                    "start": min(dates) if dates else "Unknown",
                    "end": max(dates) if dates else "Unknown",
                },
                "dominant_trigger": pd.Series(triggers).mode().iloc[0] if len(triggers) > 0 else "Unknown",
                "dominant_type": pd.Series(types).mode().iloc[0] if len(types) > 0 else "Unknown",
                "event_ids": event_ids[:50],
                "is_synthetic": bool((sub.get("is_synthetic", pd.Series([0])) == 1).any()),
            })

    # ── 2. Kernel Density Estimation (KDE) ───────────────────────────
    kde_payload: dict[str, Any] | None = None
    hotspot_regions: list[dict[str, Any]] = []

    if method in {"kde", "both"} and total_pts >= 3:
        lat_spread = float(np.ptp(lats))
        lon_spread = float(np.ptp(lons))

        if lat_spread > 1e-4 and lon_spread > 1e-4:
            try:
                # Add 10% geographic buffer around observations
                buf_lat = max(0.2, lat_spread * 0.1)
                buf_lon = max(0.2, lon_spread * 0.1)
                min_lat, max_lat = float(np.min(lats)) - buf_lat, float(np.max(lats)) + buf_lat
                min_lon, max_lon = float(np.min(lons)) - buf_lon, float(np.max(lons)) + buf_lon

                kde = gaussian_kde(np.vstack([lons, lats]))

                g_lon = np.linspace(min_lon, max_lon, kde_grid_size)
                g_lat = np.linspace(min_lat, max_lat, kde_grid_size)
                mesh_lon, mesh_lat = np.meshgrid(g_lon, g_lat)

                eval_pts = np.vstack([mesh_lon.ravel(), mesh_lat.ravel()])
                density_vals = kde(eval_pts).reshape(mesh_lon.shape)

                d_min = float(np.min(density_vals))
                d_max = float(np.max(density_vals))
                p75 = float(np.percentile(density_vals, 75))
                p90 = float(np.percentile(density_vals, 90))
                p95 = float(np.percentile(density_vals, 95))
                p99 = float(np.percentile(density_vals, 99))

                kde_payload = {
                    "grid_size": kde_grid_size,
                    "bounding_box": {
                        "min_lat": round(min_lat, 4),
                        "max_lat": round(max_lat, 4),
                        "min_lon": round(min_lon, 4),
                        "max_lon": round(max_lon, 4),
                    },
                    "density_min": round(d_min, 6),
                    "density_max": round(d_max, 6),
                    "percentiles": {
                        "p75": round(p75, 6),
                        "p90": round(p90, 6),
                        "p95": round(p95, 6),
                        "p99": round(p99, 6),
                    },
                    "lon_axis": [round(float(x), 4) for x in g_lon],
                    "lat_axis": [round(float(y), 4) for y in g_lat],
                    "density_matrix": [[round(float(v), 6) for v in row] for row in density_vals],
                }

                # Top Hotspot Regions (cells with density >= 90th percentile)
                hot_mask = density_vals >= p90
                hot_indices = np.argwhere(hot_mask)
                for r_idx, c_idx in hot_indices[:20]:
                    cell_lat = round(float(g_lat[r_idx]), 4)
                    cell_lon = round(float(g_lon[c_idx]), 4)
                    cell_density = round(float(density_vals[r_idx, c_idx]), 6)
                    tier = "High Density (>=p95)" if cell_density >= p95 else "Elevated Density (>=p90)"
                    if cell_density >= p99:
                        tier = "Extreme Density (>=p99)"
                    hotspot_regions.append({
                        "latitude": cell_lat,
                        "longitude": cell_lon,
                        "density_value": cell_density,
                        "hotspot_tier": tier,
                    })
            except Exception as exc:
                logger.warning("KDE density evaluation failed: %s", exc)

    # ── 3. Regional / Administrative Distribution ─────────────────────
    state_breakdown: dict[str, int] = {}
    if "state_region" in valid_df.columns:
        state_breakdown = {
            str(k): int(v)
            for k, v in valid_df["state_region"].fillna("Unknown").value_counts().head(10).items()
        }

    return {
        "available": True,
        "method": method,
        "parameters": {"eps_km": eps_km, "min_samples": min_samples, "authentic_only": authentic_only},
        "summary": {
            "total_events_analyzed": total_pts,
            "clustered_events_count": clustered_count,
            "noise_events_count": noise_count,
            "cluster_count": len(clusters),
            "highest_density_events_per_km2": max([c["event_density_per_sq_km"] for c in clusters], default=0.0),
        },
        "clusters": clusters,
        "kde_density": kde_payload,
        "hotspot_regions": hotspot_regions,
        "state_breakdown": state_breakdown,
        "disclaimer": HOTSPOT_SCIENTIFIC_DISCLAIMER,
    }

