"""Terrain and DEM analysis utilities for LandslideGuard (Phase 2).

Supports:
1. Local GeoTIFF DEM inspection via optional `rasterio` (when installed and a DEM
   file is supplied), computing Horn/central-difference slope angle and aspect.
2. Point SRTM terrain analysis via `TerrainProvider` (Open-Elevation API) using a
   4-point finite-difference stencil to estimate elevation, slope angle, and aspect.

Scientific integrity rules:
- Documents raster CRS, spatial resolution, and nodata masking behaviour.
- Never fabricates terrain values when both local DEM and remote SRTM are unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from src.data_providers.terrain_provider import TerrainProvider
from src.utils.logging_config import get_logger
from src.utils.validation import validate_coordinates

logger = get_logger(__name__)

try:
    import rasterio  # type: ignore
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None  # type: ignore
    RASTERIO_AVAILABLE = False

try:
    from pyproj import Transformer as _PyprojTransformer  # type: ignore
    PYPROJ_AVAILABLE = True
except ImportError:
    _PyprojTransformer = None  # type: ignore
    PYPROJ_AVAILABLE = False

SRTM_RESOLUTION_DESC = "SRTM 1-arc-second (~30 m nominal at equator)"
DEFAULT_CRS = "EPSG:4326"
NODATA_POLICY = (
    "Masked to NaN before derivative evaluation; edge or nodata-adjacent cells "
    "with undefined gradients remain NaN and are excluded from summary statistics."
)


def transform_point_crs(
    x: float,
    y: float,
    src_crs: str,
    dst_crs: str,
) -> tuple[float, float]:
    """Reproject a coordinate pair from ``src_crs`` to ``dst_crs`` cleanly.

    Uses ``always_xy=True`` so callers consistently pass and receive ``(lon/x, lat/y)``.
    Raises RuntimeError if reprojection is needed but ``pyproj`` is unavailable.
    """
    src_norm = str(src_crs or "").strip()
    dst_norm = str(dst_crs or "").strip()
    if not src_norm or not dst_norm:
        raise ValueError("Both source and destination CRS must be specified.")
    if src_norm.upper() == dst_norm.upper():
        return float(x), float(y)
    if not PYPROJ_AVAILABLE or _PyprojTransformer is None:
        raise RuntimeError(
            f"Cannot transform coordinates from {src_norm} to {dst_norm}: 'pyproj' is not installed."
        )
    transformer = _PyprojTransformer.from_crs(src_norm, dst_norm, always_xy=True)
    tx, ty = transformer.transform(float(x), float(y))
    if not (math.isfinite(tx) and math.isfinite(ty)):
        raise ValueError(f"Coordinate transformation produced non-finite output for ({x}, {y}).")
    return float(tx), float(ty)


def aspect_degrees_to_cardinal(aspect_deg: float | None) -> str | None:
    """Convert azimuth aspect in degrees (0=N, 90=E, 180=S, 270=W) to cardinal label."""
    if aspect_deg is None or not math.isfinite(aspect_deg):
        return None
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((aspect_deg + 22.5) % 360.0) // 45.0)
    return dirs[idx]


def _compute_tri_and_curvature(
    arr: np.ndarray,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Riley (1999) Terrain Ruggedness Index (m) and Laplacian curvature proxy (1/m).

    Cells without a complete 3x3 finite neighborhood are assigned NaN (no fabricated edge values).
    """
    rows, cols = arr.shape
    tri = np.full_like(arr, np.nan, dtype=float)
    curvature = np.full_like(arr, np.nan, dtype=float)
    if rows < 3 or cols < 3:
        return tri, curvature

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            window = arr[r - 1 : r + 2, c - 1 : c + 2]
            if not np.all(np.isfinite(window)):
                continue
            center = window[1, 1]
            diffs_sq = (window - center) ** 2
            # Exclude center cell itself (0 diff) -> divide by 8 neighbors per Riley et al.
            tri[r, c] = math.sqrt(float(np.sum(diffs_sq)) / 8.0)

            d2z_dx2 = (arr[r, c + 1] - 2.0 * center + arr[r, c - 1]) / (pixel_size_x_m ** 2)
            d2z_dy2 = (arr[r - 1, c] - 2.0 * center + arr[r + 1, c]) / (pixel_size_y_m ** 2)
            curvature[r, c] = float(-(d2z_dx2 + d2z_dy2))

    return tri, curvature


def compute_slope_and_aspect(
    elevation_grid: np.ndarray,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    nodata: float | None = None,
    specific_catchment_area_m: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute slope angle, aspect, terrain roughness (TRI), curvature, and TWI derivatives on a DEM grid.

    Args:
        elevation_grid: 2D array of elevation values in metres.
        pixel_size_x_m: East-west cell spacing in metres (> 0).
        pixel_size_y_m: North-south cell spacing in metres (> 0).
        nodata: Optional raster nodata sentinel value to mask.
        specific_catchment_area_m: Optional upstream contributing area per unit contour width (metres).
            Required to calculate Topographic Wetness Index (TWI).

    Returns:
        Dict with ``slope_deg``, ``aspect_deg``, ``tri_m``, ``curvature`` grids, optional ``twi`` grid,
        and summary statistics.
    """
    arr = np.asarray(elevation_grid, dtype=float).copy()
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("Elevation grid must be a 2D array with at least 2x2 cells.")
    if pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
        raise ValueError("Pixel spacing must be strictly positive (metres).")

    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        arr[arr == float(nodata)] = np.nan

    nodata_mask = ~np.isfinite(arr)
    valid_cells = int(np.sum(~nodata_mask))
    nodata_cells = int(np.sum(nodata_mask))

    if valid_cells < 4:
        return {
            "slope_deg": np.full_like(arr, np.nan),
            "aspect_deg": np.full_like(arr, np.nan),
            "tri_m": np.full_like(arr, np.nan),
            "curvature": np.full_like(arr, np.nan),
            "twi": None,
            "twi_status": "Insufficient valid cells (< 4)",
            "valid_cells": valid_cells,
            "nodata_cells": nodata_cells,
            "mean_elevation_m": float(np.nanmean(arr)) if valid_cells > 0 else None,
            "elevation_relief_m": None,
            "elevation_std_m": None,
            "mean_slope_deg": None,
            "max_slope_deg": None,
            "mean_aspect_deg": None,
            "mean_tri_m": None,
            "max_tri_m": None,
            "mean_curvature": None,
        }

    # Note: np.gradient axis=0 increases southwards (top-to-bottom raster rows),
    # so dz_dy_north = -dz_drow.
    dz_drow, dz_dx = np.gradient(arr, pixel_size_y_m, pixel_size_x_m)
    dz_dnorth = -dz_drow

    grad_mag = np.hypot(dz_dx, dz_dnorth)
    slope_deg = np.degrees(np.arctan(grad_mag))

    # Compass azimuth of steepest descent (0 = North, 90 = East, 180 = South, 270 = West)
    # Steepest descent vector points along (-dz_dx, -dz_dnorth)
    aspect_rad = np.arctan2(-dz_dx, -dz_dnorth)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    # Flat terrain has undefined aspect
    aspect_deg = np.where(grad_mag < 1e-6, np.nan, aspect_deg)

    tri_m, curvature = _compute_tri_and_curvature(arr, pixel_size_x_m, pixel_size_y_m)

    # Topographic Wetness Index (TWI = ln(a / tan(beta)))
    twi_res = compute_topographic_wetness_index(slope_deg, specific_catchment_area_m=specific_catchment_area_m)

    valid_elevs = arr[np.isfinite(arr)]
    valid_slopes = slope_deg[np.isfinite(slope_deg)]
    valid_aspects = aspect_deg[np.isfinite(aspect_deg)]
    valid_tri = tri_m[np.isfinite(tri_m)]
    valid_curv = curvature[np.isfinite(curvature)]

    return {
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "tri_m": tri_m,
        "curvature": curvature,
        "twi": twi_res.get("twi"),
        "twi_status": "Computed" if twi_res.get("available") else twi_res.get("reason"),
        "mean_twi": twi_res.get("mean_twi"),
        "max_twi": twi_res.get("max_twi"),
        "valid_cells": valid_cells,
        "nodata_cells": nodata_cells,
        "mean_elevation_m": round(float(np.mean(valid_elevs)), 2),
        "elevation_relief_m": round(float(np.max(valid_elevs) - np.min(valid_elevs)), 2),
        "elevation_std_m": round(float(np.std(valid_elevs)), 2),
        "mean_slope_deg": round(float(np.mean(valid_slopes)), 2) if valid_slopes.size else None,
        "max_slope_deg": round(float(np.max(valid_slopes)), 2) if valid_slopes.size else None,
        "mean_aspect_deg": round(float(np.mean(valid_aspects)), 1) if valid_aspects.size else None,
        "mean_tri_m": round(float(np.mean(valid_tri)), 2) if valid_tri.size else None,
        "max_tri_m": round(float(np.max(valid_tri)), 2) if valid_tri.size else None,
        "mean_curvature": round(float(np.mean(valid_curv)), 6) if valid_curv.size else None,
    }


def compute_topographic_wetness_index(
    slope_deg: np.ndarray,
    specific_catchment_area_m: np.ndarray | None = None,
    min_slope_rad: float = 0.001,
) -> dict[str, Any]:
    """Calculate Topographic Wetness Index (TWI) only when required inputs are available.

    Formula: TWI = ln(a / tan(beta))
    where 'a' is specific catchment area (upslope contributing area per unit contour width, metres)
    and 'beta' is slope in radians.

    Scientific guard:
    Returns available=False with a clear explanatory reason when specific catchment area 'a'
    is not supplied or cannot be calculated. Never fabricates hydrological contributing area.
    """
    if specific_catchment_area_m is None:
        return {
            "available": False,
            "twi": None,
            "mean_twi": None,
            "max_twi": None,
            "min_twi": None,
            "formula": "TWI = ln(a / tan(beta))",
            "reason": (
                "Specific catchment area (upslope contributing area 'a') is required to compute "
                "Topographic Wetness Index (TWI = ln(a / tan(beta))). Calculation refused without required hydrological inputs."
            ),
        }

    slope_arr = np.asarray(slope_deg, dtype=float)
    sca_arr = np.asarray(specific_catchment_area_m, dtype=float)
    if slope_arr.shape != sca_arr.shape:
        raise ValueError(
            f"Shape mismatch between slope grid {slope_arr.shape} and catchment area grid {sca_arr.shape}."
        )

    slope_rad = np.radians(np.clip(slope_arr, 0.0, 89.9))
    slope_rad = np.maximum(slope_rad, min_slope_rad)
    tan_beta = np.tan(slope_rad)

    valid_mask = np.isfinite(slope_arr) & np.isfinite(sca_arr) & (sca_arr > 0)
    twi = np.full_like(slope_arr, np.nan, dtype=float)

    if np.any(valid_mask):
        ratio = sca_arr[valid_mask] / tan_beta[valid_mask]
        ratio = np.maximum(ratio, 1e-6)
        twi[valid_mask] = np.log(ratio)

    valid_vals = twi[np.isfinite(twi)]
    if len(valid_vals) == 0:
        return {
            "available": False,
            "twi": twi,
            "mean_twi": None,
            "max_twi": None,
            "min_twi": None,
            "formula": "TWI = ln(a / tan(beta))",
            "reason": "No valid non-zero positive catchment and slope cells to evaluate TWI.",
        }

    return {
        "available": True,
        "twi": twi,
        "valid_cells": int(len(valid_vals)),
        "mean_twi": round(float(np.mean(valid_vals)), 2),
        "max_twi": round(float(np.max(valid_vals)), 2),
        "min_twi": round(float(np.min(valid_vals)), 2),
        "std_twi": round(float(np.std(valid_vals)), 2),
        "formula": "TWI = ln(a / tan(beta))",
        "reason": "Successfully calculated using authentic slope and upslope contributing area.",
    }


def compute_d8_flow_accumulation(
    elevation_grid: np.ndarray,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> dict[str, Any]:
    """Compute D8 steepest-descent flow direction and accumulation on a DEM grid.

    Returns:
        Dict with 'flow_accumulation_cells', 'specific_catchment_area_m', and summary statistics.
    """
    elev = np.asarray(elevation_grid, dtype=float).copy()
    rows, cols = elev.shape
    if rows < 3 or cols < 3:
        raise ValueError("Elevation grid must have at least 3x3 cells for D8 flow routing.")

    dx = float(pixel_size_x_m)
    dy = float(pixel_size_y_m)
    diag = math.hypot(dx, dy)

    neighbors = [
        (-1, 0, dy),    # North
        (1, 0, dy),     # South
        (0, -1, dx),    # West
        (0, 1, dx),     # East
        (-1, -1, diag), # NW
        (-1, 1, diag),  # NE
        (1, -1, diag),  # SW
        (1, 1, diag),   # SE
    ]

    downstream_r = np.full((rows, cols), -1, dtype=int)
    downstream_c = np.full((rows, cols), -1, dtype=int)

    for r in range(rows):
        for c in range(cols):
            z_c = elev[r, c]
            if not math.isfinite(z_c):
                continue
            max_slope = 0.0
            best_r, best_c = -1, -1
            for dr, dc, dist in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    z_n = elev[nr, nc]
                    if math.isfinite(z_n) and z_n < z_c:
                        slope = (z_c - z_n) / dist
                        if slope > max_slope:
                            max_slope = slope
                            best_r, best_c = nr, nc
            if best_r != -1:
                downstream_r[r, c] = best_r
                downstream_c[r, c] = best_c

    flat_indices = np.argsort(-elev.ravel())
    accum = np.zeros((rows, cols), dtype=float)
    accum[np.isfinite(elev)] = 1.0

    for idx in flat_indices:
        r = idx // cols
        c = idx % cols
        if not math.isfinite(elev[r, c]):
            continue
        nr = downstream_r[r, c]
        nc = downstream_c[r, c]
        if nr != -1 and nc != -1:
            accum[nr, nc] += accum[r, c]

    cell_contour_width = (dx + dy) / 2.0
    cell_area = dx * dy
    specific_catchment_m = (accum * cell_area) / cell_contour_width

    valid_accum = accum[np.isfinite(elev)]
    return {
        "flow_accumulation_cells": accum,
        "specific_catchment_area_m": specific_catchment_m,
        "mean_accumulation": round(float(np.mean(valid_accum)), 2) if len(valid_accum) else None,
        "max_accumulation": round(float(np.max(valid_accum)), 2) if len(valid_accum) else None,
    }


def build_3d_terrain_surface(
    elevation_grid: np.ndarray,
    bounds: tuple[float, float, float, float] | None = None,
    drape_mode: str = "elevation",
    drape_grid: np.ndarray | None = None,
    title: str = "3D Interactive Digital Elevation Model",
    colorscale: str = "Earth",
) -> dict[str, Any]:
    """Generate 3D terrain surface data and Plotly figure specification.

    Args:
        elevation_grid: 2D array of elevation values in metres.
        bounds: Optional (min_lon, min_lat, max_lon, max_lat) in WGS84.
        drape_mode: Color drape layer ('elevation', 'slope', 'roughness', 'custom').
        drape_grid: Optional matching 2D array to drape onto the 3D surface.
        title: Figure title.
        colorscale: Plotly color scale name.

    Returns:
        Dict with 'figure', 'bounds', 'elevation_range', and terrain dimensions.
    """
    import plotly.graph_objects as go

    arr = np.asarray(elevation_grid, dtype=float).copy()
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("Elevation grid must be a 2D array with at least 2x2 cells.")

    rows, cols = arr.shape
    if bounds and len(bounds) == 4:
        min_lon, min_lat, max_lon, max_lat = [float(b) for b in bounds]
        x_vals = np.linspace(min_lon, max_lon, cols)
        y_vals = np.linspace(max_lat, min_lat, rows)  # Rows go from North to South
    else:
        x_vals = np.arange(cols)
        y_vals = np.arange(rows)

    surface_color = arr
    colorbar_title = "Elevation (m)"
    if drape_grid is not None and drape_grid.shape == arr.shape:
        surface_color = drape_grid
        colorbar_title = drape_mode.replace("_", " ").title()
        if drape_mode == "slope":
            colorbar_title += " (°)"
            colorscale = "Viridis"
        elif drape_mode == "roughness":
            colorbar_title += " (TRI m)"
            colorscale = "Plasma"

    surface = go.Surface(
        z=arr,
        x=x_vals,
        y=y_vals,
        surfacecolor=surface_color,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title, len=0.7),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        ),
    )

    fig = go.Figure(data=[surface])
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        autosize=True,
        margin=dict(l=20, r=20, t=50, b=20),
        scene=dict(
            xaxis_title="Longitude (°E)" if bounds else "X (cells)",
            yaxis_title="Latitude (°N)" if bounds else "Y (cells)",
            zaxis_title="Elevation (m)",
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.4),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.9),
                center=dict(x=0, y=0, z=-0.1),
            ),
        ),
    )

    valid_z = arr[np.isfinite(arr)]
    return {
        "figure": fig,
        "rows": rows,
        "cols": cols,
        "bounds": bounds,
        "drape_mode": drape_mode,
        "elevation_min_m": round(float(np.min(valid_z)), 2) if len(valid_z) else None,
        "elevation_max_m": round(float(np.max(valid_z)), 2) if len(valid_z) else None,
        "elevation_relief_m": round(float(np.ptp(valid_z)), 2) if len(valid_z) else None,
    }


def _resolution_to_metres(res_x: float, res_y: float, is_geographic: bool, center_lat: float = 0.0) -> tuple[float, float]:
    """Convert raster cell dimensions to approximate metres."""
    if not is_geographic:
        return abs(float(res_x)), abs(float(res_y))
    lat_m = 111_320.0
    lon_m = max(1.0, 111_320.0 * math.cos(math.radians(center_lat)))
    return abs(float(res_x)) * lon_m, abs(float(res_y)) * lat_m


def inspect_dem_raster(
    dem_path: str | Path,
    sample_lat: float | None = None,
    sample_lon: float | None = None,
) -> dict[str, Any]:
    """Inspect a local GeoTIFF DEM raster with Rasterio if available.

    Always documents CRS, spatial resolution, and nodata handling, and returns an
    explicit `available=False` result if Rasterio or the DEM file is absent.
    """
    path = Path(dem_path)
    if not path.exists() or not path.is_file():
        return {
            "available": False,
            "rasterio_installed": RASTERIO_AVAILABLE,
            "source": str(path),
            "crs": None,
            "resolution": None,
            "nodata_value": None,
            "nodata_policy": NODATA_POLICY,
            "reason": f"DEM raster file not found at '{path}'.",
        }

    if not RASTERIO_AVAILABLE:
        # Fallback to PIL for reading GeoTIFF / TIFF rasters when Rasterio is absent
        try:
            from PIL import Image
            with Image.open(path) as img:
                band = np.array(img, dtype=float)
                if band.ndim != 2:
                    return {
                        "available": False,
                        "rasterio_installed": False,
                        "source": str(path),
                        "crs": None,
                        "resolution": None,
                        "nodata_value": None,
                        "nodata_policy": NODATA_POLICY,
                        "reason": f"Expected a 2D single-band elevation raster, found shape {band.shape}.",
                    }
                tags = getattr(img, "tag_v2", getattr(img, "tag", {}))
                has_georef = any(k in tags for k in (33550, 33922, 34735))
                if not has_georef:
                    return {
                        "available": False,
                        "rasterio_installed": False,
                        "source": str(path),
                        "crs": None,
                        "resolution": None,
                        "nodata_value": None,
                        "nodata_policy": NODATA_POLICY,
                        "reason": "DEM raster has no embedded Coordinate Reference System (CRS) or GeoTIFF tags; refusing unreferenced sampling.",
                    }
                scale = tags.get(33550)
                tiepoint = tags.get(33922)
                res_x = float(scale[0]) if scale and len(scale) >= 1 else 0.0002777777777777778
                res_y = float(scale[1]) if scale and len(scale) >= 2 else 0.0002777777777777778
                crs_str = "EPSG:4326"
                px_m, py_m = _resolution_to_metres(res_x, res_y, is_geographic=True, center_lat=sample_lat or 0.0)
                stats = compute_slope_and_aspect(band, px_m, py_m)

                sampled: dict[str, Any] = {}
                if sample_lat is not None and sample_lon is not None and tiepoint and len(tiepoint) >= 6:
                    origin_x, origin_y = float(tiepoint[3]), float(tiepoint[4])
                    top = origin_y
                    bottom = origin_y - band.shape[0] * res_y
                    left = origin_x
                    right = origin_x + band.shape[1] * res_x
                    if left <= sample_lon <= right and bottom <= sample_lat <= top:
                        col = int((sample_lon - left) / res_x)
                        row = int((top - sample_lat) / res_y)
                        if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
                            elev_val = band[row, col]
                            slope_val = stats["slope_deg"][row, col]
                            aspect_val = stats["aspect_deg"][row, col]
                            tri_val = stats["tri_m"][row, col]
                            curv_val = stats["curvature"][row, col]
                            sampled = {
                                "elevation_m": round(float(elev_val), 2) if math.isfinite(elev_val) else None,
                                "slope_degrees": round(float(slope_val), 2) if math.isfinite(slope_val) else None,
                                "aspect_degrees": round(float(aspect_val), 1) if math.isfinite(aspect_val) else None,
                                "aspect_cardinal": aspect_degrees_to_cardinal(float(aspect_val))
                                if math.isfinite(aspect_val)
                                else None,
                                "roughness_tri_m": round(float(tri_val), 2) if math.isfinite(tri_val) else None,
                                "curvature": round(float(curv_val), 6) if math.isfinite(curv_val) else None,
                            }
                return {
                    "available": True,
                    "rasterio_installed": False,
                    "source": f"GeoTIFF via PIL ({path})",
                    "crs": crs_str,
                    "resolution": {"x": float(res_x), "y": float(res_y), "x_m": round(px_m, 2), "y_m": round(py_m, 2)},
                    "nodata_value": None,
                    "nodata_policy": NODATA_POLICY,
                    "shape": [int(band.shape[0]), int(band.shape[1])],
                    "valid_cells": stats["valid_cells"],
                    "nodata_cells": stats["nodata_cells"],
                    "mean_elevation_m": stats["mean_elevation_m"],
                    "elevation_relief_m": stats["elevation_relief_m"],
                    "elevation_std_m": stats["elevation_std_m"],
                    "mean_slope_deg": stats["mean_slope_deg"],
                    "max_slope_deg": stats["max_slope_deg"],
                    "mean_aspect_deg": stats["mean_aspect_deg"],
                    "mean_tri_m": stats["mean_tri_m"],
                    "max_tri_m": stats["max_tri_m"],
                    "mean_curvature": stats["mean_curvature"],
                    "sampled_point": sampled,
                }
        except Exception as exc:
            logger.debug("PIL raster load failed: %s", exc)

        return {
            "available": False,
            "rasterio_installed": False,
            "source": str(path),
            "crs": None,
            "resolution": None,
            "nodata_value": None,
            "nodata_policy": NODATA_POLICY,
            "reason": (
                "Optional package 'rasterio' is not installed in this environment. "
                "Install rasterio or use Open-Elevation SRTM point lookup instead."
            ),
        }

    with rasterio.open(path) as src:
        band = src.read(1).astype(float)
        if src.crs is None:
            return {
                "available": False,
                "rasterio_installed": True,
                "source": str(path),
                "crs": None,
                "resolution": None,
                "nodata_value": src.nodata,
                "nodata_policy": NODATA_POLICY,
                "reason": "DEM raster has no embedded Coordinate Reference System (CRS); refusing unreferenced sampling.",
            }
        crs_str = str(src.crs)
        is_geo = bool(src.crs.is_geographic)
        res_x, res_y = src.res
        bounds = src.bounds
        center_lat = (bounds.top + bounds.bottom) / 2.0 if is_geo else 0.0
        px_m, py_m = _resolution_to_metres(res_x, res_y, is_geo, center_lat=center_lat)
        stats = compute_slope_and_aspect(band, px_m, py_m, nodata=src.nodata)

        sampled: dict[str, Any] = {}
        if sample_lat is not None and sample_lon is not None:
            sx, sy = transform_point_crs(float(sample_lon), float(sample_lat), DEFAULT_CRS, crs_str)
            if bounds.left <= sx <= bounds.right and bounds.bottom <= sy <= bounds.top:
                row, col = src.index(sx, sy)
                if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
                    elev_val = band[row, col]
                    if src.nodata is not None and elev_val == float(src.nodata):
                        elev_val = float("nan")
                    slope_val = stats["slope_deg"][row, col]
                    aspect_val = stats["aspect_deg"][row, col]
                    tri_val = stats["tri_m"][row, col]
                    curv_val = stats["curvature"][row, col]
                    sampled = {
                        "elevation_m": round(float(elev_val), 2) if math.isfinite(elev_val) else None,
                        "slope_degrees": round(float(slope_val), 2) if math.isfinite(slope_val) else None,
                        "aspect_degrees": round(float(aspect_val), 1) if math.isfinite(aspect_val) else None,
                        "aspect_cardinal": aspect_degrees_to_cardinal(float(aspect_val))
                        if math.isfinite(aspect_val)
                        else None,
                        "roughness_tri_m": round(float(tri_val), 2) if math.isfinite(tri_val) else None,
                        "curvature": round(float(curv_val), 6) if math.isfinite(curv_val) else None,
                    }

        return {
            "available": True,
            "rasterio_installed": True,
            "source": str(path),
            "crs": crs_str,
            "resolution": {"x": float(res_x), "y": float(res_y), "x_m": round(px_m, 2), "y_m": round(py_m, 2)},
            "nodata_value": src.nodata,
            "nodata_policy": NODATA_POLICY,
            "shape": [int(band.shape[0]), int(band.shape[1])],
            "valid_cells": stats["valid_cells"],
            "nodata_cells": stats["nodata_cells"],
            "mean_elevation_m": stats["mean_elevation_m"],
            "elevation_relief_m": stats["elevation_relief_m"],
            "elevation_std_m": stats["elevation_std_m"],
            "mean_slope_deg": stats["mean_slope_deg"],
            "max_slope_deg": stats["max_slope_deg"],
            "mean_aspect_deg": stats["mean_aspect_deg"],
            "mean_tri_m": stats["mean_tri_m"],
            "max_tri_m": stats["max_tri_m"],
            "mean_curvature": stats["mean_curvature"],
            "sampled_point": sampled,
        }


def get_point_terrain_summary(
    latitude: float,
    longitude: float,
    dem_path: str | Path | None = None,
    terrain_provider: TerrainProvider | None = None,
    offset_deg: float = 0.001,
) -> dict[str, Any]:
    """Retrieve terrain characteristics (elevation, slope, aspect) for a WGS84 point.

    Prefers a supplied local GeoTIFF DEM (if Rasterio + file are available), then
    falls back to Open-Elevation SRTM 4-point central finite difference. Returns
    `available=False` with an explicit reason when neither source can supply data.
    """
    coord_check = validate_coordinates(latitude, longitude)
    if not coord_check.is_valid:
        return {
            "available": False,
            "latitude": latitude,
            "longitude": longitude,
            "crs": DEFAULT_CRS,
            "reason": "; ".join(coord_check.errors),
        }

    if dem_path is not None:
        dem_info = inspect_dem_raster(dem_path, sample_lat=latitude, sample_lon=longitude)
        if dem_info.get("available") and dem_info.get("sampled_point"):
            sp = dem_info["sampled_point"]
            return {
                "available": sp.get("elevation_m") is not None,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "elevation_m": sp.get("elevation_m"),
                "slope_degrees": sp.get("slope_degrees"),
                "aspect_degrees": sp.get("aspect_degrees"),
                "aspect_cardinal": sp.get("aspect_cardinal"),
                "source": f"Local DEM GeoTIFF ({dem_info['source']})",
                "crs": dem_info["crs"],
                "resolution": f"{dem_info['resolution']['x_m']}m x {dem_info['resolution']['y_m']}m",
                "nodata_policy": dem_info["nodata_policy"],
            }

    provider = terrain_provider or TerrainProvider()
    points = [
        (latitude, longitude),
        (latitude + offset_deg, longitude),  # North
        (latitude - offset_deg, longitude),  # South
        (latitude, longitude + offset_deg),  # East
        (latitude, longitude - offset_deg),  # West
    ]
    elevs = provider.fetch_batch(points)
    if not elevs or len(elevs) < 5 or any(e is None or not math.isfinite(float(e)) for e in elevs):
        return {
            "available": False,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "elevation_m": None,
            "slope_degrees": None,
            "aspect_degrees": None,
            "aspect_cardinal": None,
            "roughness_tri_m": None,
            "local_relief_m": None,
            "curvature_proxy": None,
            "source": "SRTM / Open-Elevation unavailable",
            "crs": DEFAULT_CRS,
            "resolution": SRTM_RESOLUTION_DESC,
            "nodata_policy": NODATA_POLICY,
            "reason": (
                "Terrain DEM data is currently unavailable (no local GeoTIFF supplied and "
                "Open-Elevation SRTM service returned no valid elevation points). "
                "No terrain values were fabricated."
            ),
        }

    e_c, e_n, e_s, e_e, e_w = [float(v) for v in elevs]
    dy_m = offset_deg * 111_320.0
    dx_m = max(1.0, offset_deg * 111_320.0 * math.cos(math.radians(latitude)))

    dz_dnorth = (e_n - e_s) / (2.0 * dy_m)
    dz_deast = (e_e - e_w) / (2.0 * dx_m)
    grad = math.hypot(dz_deast, dz_dnorth)
    slope_deg = round(math.degrees(math.atan(grad)), 2)

    # 4-point stencil RMS ruggedness (m), local elevation relief (m), and Laplacian curvature (1/m)
    diffs = [e_n - e_c, e_s - e_c, e_e - e_c, e_w - e_c]
    tri_4pt = round(math.sqrt(sum(d * d for d in diffs) / 4.0), 2)
    relief_m = round(max(e_c, e_n, e_s, e_e, e_w) - min(e_c, e_n, e_s, e_e, e_w), 2)
    curv_lap = round(-(((e_e - 2.0 * e_c + e_w) / (dx_m ** 2)) + ((e_n - 2.0 * e_c + e_s) / (dy_m ** 2))), 6)

    if grad < 1e-6:
        aspect_deg = None
        cardinal = "Flat"
    else:
        aspect_deg = round((math.degrees(math.atan2(-dz_deast, -dz_dnorth)) + 360.0) % 360.0, 1)
        cardinal = aspect_degrees_to_cardinal(aspect_deg)

    return {
        "available": True,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "elevation_m": round(e_c, 2),
        "slope_degrees": slope_deg,
        "aspect_degrees": aspect_deg,
        "aspect_cardinal": cardinal,
        "roughness_tri_m": tri_4pt,
        "local_relief_m": relief_m,
        "curvature_proxy": curv_lap,
        "source": provider.attribution,
        "crs": DEFAULT_CRS,
        "resolution": f"{SRTM_RESOLUTION_DESC} (4-point stencil ~{round(dx_m, 0):.0f}m)",
        "nodata_policy": NODATA_POLICY,
    }

