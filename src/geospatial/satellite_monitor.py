"""
Phase 7 Satellite Vegetation Monitoring and Change Detection for LandslideGuard.

Provides:
1. Sentinel-2 MultiSpectral Instrument (MSI) NDVI calculation strictly requiring Red and NIR bands.
2. Cloud masking & quality control using Scene Classification Layer (SCL) or QA bitmasks.
3. Multi-temporal satellite change detection (dNDVI) comparing pre- and post-event acquisitions.
4. Disturbance area quantification, cloud obscuration accounting, and resolution limits.
5. Strict scientific disclaimers & verification indicator labels.

SCIENTIFIC INTEGRITY GUARDS:
1. "Do not interpret vegetation changes alone as proof of landslide activity."
   Vegetation loss can stem from seasonal phenology, agricultural harvest, timber cutting,
   wildfire, or cloud shadows.
2. "Clearly label automated change detection as an indicator requiring verification."
   Automated change detection is an unverified screening indicator requiring ground or optical corroboration.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

VEGETATION_DISCLAIMER = (
    "SCIENTIFIC NOTICE: Vegetation index (NDVI) fluctuations and reductions may result from seasonal phenology, "
    "agricultural cycles, timber harvesting, wildfire, or atmospheric/cloud contamination. "
    "Reductions in vegetation density alone CANNOT be interpreted as proof of landslide activity "
    "without geotechnical, ground-truth, or high-resolution optical corroboration."
)

CHANGE_DETECTION_VERIFICATION_LABEL = (
    "AUTOMATED CHANGE DETECTION INDICATOR: Spectral difference metrics identify potential surface disturbance candidate zones. "
    "This automated output is an unverified screening indicator and requires independent optical or field verification."
)

SENTINEL2_METADATA = {
    "sensor": "Sentinel-2 MultiSpectral Instrument (MSI)",
    "constellation": "Copernicus Sentinel-2A / Sentinel-2B (ESA)",
    "spectral_bands": {
        "B02": {"name": "Blue", "wavelength_nm": 490, "resolution_m": 10},
        "B03": {"name": "Green", "wavelength_nm": 560, "resolution_m": 10},
        "B04": {"name": "Red", "wavelength_nm": 665, "resolution_m": 10},
        "B08": {"name": "NIR (Near-Infrared)", "wavelength_nm": 842, "resolution_m": 10},
        "SCL": {"name": "Scene Classification Layer", "resolution_m": 20},
    },
    "revisit_time_days": 5,
    "license": "Copernicus Open Access Policy (Free, Full, and Open)",
}


def compute_ndvi(
    red_band: np.ndarray,
    nir_band: np.ndarray,
    cloud_mask: np.ndarray | None = None,
    observation_date: str | None = None,
    sensor: str = "Sentinel-2 MSI",
    crs: str = "EPSG:4326",
    resolution_m: float = 10.0,
    bounds: tuple[float, float, float, float] | None = None,
    scale_factor: float = 1.0,
) -> dict[str, Any]:
    """Compute Normalized Difference Vegetation Index (NDVI) strictly requiring Red and NIR bands.

    Formula: NDVI = (NIR - Red) / (NIR + Red)
    Where:
      - Red is Sentinel-2 Band 4 (~665 nm)
      - NIR is Sentinel-2 Band 8 (~842 nm)

    Raises:
        ValueError: If either Red or NIR band is missing, not a 2D array, or dimensions do not match.
    """
    if red_band is None or nir_band is None:
        raise ValueError(
            "NDVI requires both Red (Sentinel-2 B04 ~665nm) and NIR (Sentinel-2 B08 ~842nm) spectral bands. "
            "Cannot compute NDVI from single-band, panchromatic, or RGB-only imagery."
        )

    red = np.asarray(red_band, dtype=float) * float(scale_factor)
    nir = np.asarray(nir_band, dtype=float) * float(scale_factor)

    if red.ndim != 2 or nir.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays for spectral bands; received Red shape {red.shape} and NIR shape {nir.shape}."
        )
    if red.shape != nir.shape:
        raise ValueError(
            f"Dimension mismatch between Red band {red.shape} and NIR band {nir.shape}."
        )

    rows, cols = red.shape
    total_pixels = rows * cols

    # Evaluate NDVI safely
    denom = nir + red
    diff = nir - red
    valid_denom = np.isfinite(denom) & (denom > 1e-6)

    ndvi = np.full_like(red, np.nan, dtype=float)
    ndvi[valid_denom] = diff[valid_denom] / denom[valid_denom]
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # Cloud masking
    cloud_bool_mask = np.zeros_like(red, dtype=bool)
    if cloud_mask is not None:
        c_arr = np.asarray(cloud_mask)
        if c_arr.shape == red.shape:
            if c_arr.dtype == bool:
                cloud_bool_mask = c_arr
            else:
                # Handle Sentinel-2 SCL categories: 3 (cloud shadows), 8 (med cloud), 9 (high cloud), 10 (cirrus)
                cloud_bool_mask = np.isin(c_arr, [3, 8, 9, 10]) | (c_arr > 0 if np.max(c_arr) <= 1 else False)
            ndvi[cloud_bool_mask] = np.nan

    valid_mask = np.isfinite(ndvi)
    valid_count = int(np.sum(valid_mask))
    cloud_count = int(np.sum(cloud_bool_mask))
    cloud_pct = round((cloud_count / max(1, total_pixels)) * 100.0, 2)

    valid_vals = ndvi[valid_mask]
    if valid_count > 0:
        mean_v = round(float(np.mean(valid_vals)), 4)
        min_v = round(float(np.min(valid_vals)), 4)
        max_v = round(float(np.max(valid_vals)), 4)
        std_v = round(float(np.std(valid_vals)), 4)
        p10_v = round(float(np.percentile(valid_vals, 10)), 4)
        p50_v = round(float(np.median(valid_vals)), 4)
        p90_v = round(float(np.percentile(valid_vals, 90)), 4)

        # Classification breakdown (fraction of valid pixels)
        water_snow_pct = round(float(np.mean(valid_vals < 0.0)) * 100.0, 1)
        bare_soil_pct = round(float(np.mean((valid_vals >= 0.0) & (valid_vals < 0.2))) * 100.0, 1)
        sparse_veg_pct = round(float(np.mean((valid_vals >= 0.2) & (valid_vals < 0.4))) * 100.0, 1)
        moderate_veg_pct = round(float(np.mean((valid_vals >= 0.4) & (valid_vals < 0.6))) * 100.0, 1)
        dense_veg_pct = round(float(np.mean(valid_vals >= 0.6)) * 100.0, 1)
    else:
        mean_v, min_v, max_v, std_v, p10_v, p50_v, p90_v = [None] * 7
        water_snow_pct, bare_soil_pct, sparse_veg_pct, moderate_veg_pct, dense_veg_pct = [0.0] * 5

    return {
        "available": True,
        "sensor": sensor,
        "observation_date": observation_date or "Unknown",
        "resolution_m": float(resolution_m),
        "crs": crs,
        "bounds": bounds,
        "dimensions": {"rows": rows, "cols": cols},
        "total_pixel_count": total_pixels,
        "valid_pixel_count": valid_count,
        "cloud_masked_count": cloud_count,
        "cloud_cover_pct": cloud_pct,
        "statistics": {
            "mean_ndvi": mean_v,
            "min_ndvi": min_v,
            "max_ndvi": max_v,
            "std_ndvi": std_v,
            "p10_ndvi": p10_v,
            "median_ndvi": p50_v,
            "p90_ndvi": p90_v,
        },
        "vegetation_classes": {
            "water_or_snow_pct": water_snow_pct,
            "bare_soil_rock_pct": bare_soil_pct,
            "sparse_vegetation_pct": sparse_veg_pct,
            "moderate_vegetation_pct": moderate_veg_pct,
            "dense_canopy_pct": dense_veg_pct,
        },
        "ndvi_grid": ndvi,
        "cloud_mask": cloud_bool_mask,
        "disclaimer": VEGETATION_DISCLAIMER,
    }


def detect_satellite_changes(
    pre_scene: dict[str, Any],
    post_scene: dict[str, Any],
    change_threshold: float = -0.15,
) -> dict[str, Any]:
    """Detect multi-temporal satellite vegetation changes (dNDVI) between two acquisition dates.

    Args:
        pre_scene: Result dict from compute_ndvi representing pre-event acquisition (t1).
        post_scene: Result dict from compute_ndvi representing post-event acquisition (t2).
        change_threshold: Delta NDVI cutoff below which vegetation loss is flagged.

    Returns:
        Structured change detection report with dNDVI matrix, loss area (km²), cloud limitations,
        resolution limitations, and mandatory verification label.
    """
    pre_ndvi = pre_scene.get("ndvi_grid")
    post_ndvi = post_scene.get("ndvi_grid")

    if pre_ndvi is None or post_ndvi is None:
        raise ValueError("Both pre_scene and post_scene must contain computed 'ndvi_grid' matrices.")

    pre_arr = np.asarray(pre_ndvi, dtype=float)
    post_arr = np.asarray(post_ndvi, dtype=float)

    if pre_arr.shape != post_arr.shape:
        raise ValueError(
            f"Shape mismatch between pre-event grid {pre_arr.shape} and post-event grid {post_arr.shape}."
        )

    t1_date = pre_scene.get("observation_date", "t1")
    t2_date = post_scene.get("observation_date", "t2")
    res_m = float(pre_scene.get("resolution_m", 10.0))

    # Dual cloud masking: mask pixel if cloudy in either pre or post acquisition
    pre_cloud = pre_scene.get("cloud_mask", np.zeros_like(pre_arr, dtype=bool))
    post_cloud = post_scene.get("cloud_mask", np.zeros_like(post_arr, dtype=bool))
    dual_cloud = np.asarray(pre_cloud, dtype=bool) | np.asarray(post_cloud, dtype=bool)

    d_ndvi = post_arr - pre_arr
    d_ndvi[dual_cloud] = np.nan

    total_pixels = int(pre_arr.size)
    cloud_obscured_pixels = int(np.sum(dual_cloud))
    cloud_obscured_pct = round((cloud_obscured_pixels / max(1, total_pixels)) * 100.0, 2)

    valid_mask = np.isfinite(d_ndvi)
    valid_pixels = int(np.sum(valid_mask))
    valid_diffs = d_ndvi[valid_mask]

    # Pixel area in m² and km²
    pixel_area_m2 = res_m * res_m
    pixel_area_km2 = pixel_area_m2 / 1_000_000.0

    if valid_pixels > 0:
        mean_d = round(float(np.mean(valid_diffs)), 4)
        median_d = round(float(np.median(valid_diffs)), 4)
        min_d = round(float(np.min(valid_diffs)), 4)
        max_d = round(float(np.max(valid_diffs)), 4)

        # Classified change counts
        severe_loss_mask = valid_mask & (d_ndvi <= -0.25)
        mod_loss_mask = valid_mask & (d_ndvi > -0.25) & (d_ndvi <= -0.10)
        stable_mask = valid_mask & (d_ndvi > -0.10) & (d_ndvi < 0.10)
        growth_mask = valid_mask & (d_ndvi >= 0.10)

        severe_loss_px = int(np.sum(severe_loss_mask))
        mod_loss_px = int(np.sum(mod_loss_mask))
        stable_px = int(np.sum(stable_mask))
        growth_px = int(np.sum(growth_mask))

        severe_area_km2 = round(severe_loss_px * pixel_area_km2, 4)
        mod_area_km2 = round(mod_loss_px * pixel_area_km2, 4)
        analyzed_area_km2 = round(valid_pixels * pixel_area_km2, 4)
    else:
        mean_d, median_d, min_d, max_d = [None] * 4
        severe_loss_px, mod_loss_px, stable_px, growth_px = 0, 0, 0, 0
        severe_area_km2, mod_area_km2, analyzed_area_km2 = 0.0, 0.0, 0.0

    return {
        "available": True,
        "pre_observation_date": t1_date,
        "post_observation_date": t2_date,
        "temporal_baseline": f"{t1_date} → {t2_date}",
        "resolution_m": res_m,
        "total_pixels": total_pixels,
        "valid_evaluated_pixels": valid_pixels,
        "cloud_obscured_pixels": cloud_obscured_pixels,
        "cloud_obscured_pct": cloud_obscured_pct,
        "analyzed_surface_area_km2": analyzed_area_km2,
        "d_ndvi_statistics": {
            "mean_delta_ndvi": mean_d,
            "median_delta_ndvi": median_d,
            "min_delta_ndvi": min_d,
            "max_delta_ndvi": max_d,
        },
        "change_classification": {
            "severe_vegetation_loss_pixels": severe_loss_px,
            "severe_loss_area_km2": severe_area_km2,
            "moderate_vegetation_loss_pixels": mod_loss_px,
            "moderate_loss_area_km2": mod_area_km2,
            "stable_vegetation_pixels": stable_px,
            "vegetation_growth_pixels": growth_px,
        },
        "candidate_disturbance_area_km2": severe_area_km2,
        "limitations": {
            "spatial_resolution": f"Sentinel-2 {res_m:.0f}m Ground Sample Distance cannot detect sub-pixel landslide scars (<100 m²).",
            "cloud_cover": f"Cloud cover obscured {cloud_obscured_pct:.1f}% of the scene across both acquisition dates and was excluded from change detection.",
            "temporal_latency": f"Acquisitions span {t1_date} to {t2_date}; intermediate weather/clearing events are unobserved.",
        },
        "d_ndvi_grid": d_ndvi,
        "verification_indicator_label": CHANGE_DETECTION_VERIFICATION_LABEL,
        "disclaimer": VEGETATION_DISCLAIMER,
    }


def get_sample_sentinel2_scenario(scenario: str = "wayanad_monsoon") -> tuple[dict[str, Any], dict[str, Any]]:
    """Return realistic, documented Sentinel-2 Level-2A multi-band acquisition grids for demonstration.

    Scenario: Western Ghats high-relief forested terrain (e.g. Wayanad / Idukki)
      - Pre-monsoon scene (May 2024): Dense forest canopy (NDVI ~0.75-0.85), low cloud cover.
      - Post-monsoon scene (August 2024): Localized slope scar clearing (NDVI drop to ~0.15)
        and localized cloud edge patch.
    """
    rng = np.random.default_rng(42)
    rows, cols = 30, 30

    # 1. Pre-monsoon scene: Healthy forest canopy
    # Forest NIR is high (~0.60), Red is low (~0.08) -> NDVI ~ 0.76
    pre_red = 0.08 + rng.normal(0.0, 0.01, size=(rows, cols))
    pre_nir = 0.60 + rng.normal(0.0, 0.02, size=(rows, cols))
    pre_red = np.clip(pre_red, 0.02, 0.20)
    pre_nir = np.clip(pre_nir, 0.40, 0.85)
    pre_cloud = np.zeros((rows, cols), dtype=bool)
    # A few random cloud pixels in pre scene
    pre_cloud[2:4, 25:28] = True

    pre_scene = compute_ndvi(
        red_band=pre_red,
        nir_band=pre_nir,
        cloud_mask=pre_cloud,
        observation_date="2024-05-18",
        sensor="Sentinel-2B MSI L2A",
        bounds=(76.10, 11.45, 76.13, 11.48),
        resolution_m=10.0,
    )

    # 2. Post-monsoon scene: Localized slope disturbance / scar + cloud patch
    post_red = pre_red.copy()
    post_nir = pre_nir.copy()
    post_cloud = np.zeros((rows, cols), dtype=bool)

    # Simulate localized landslide scar in rows 12..18, cols 10..16:
    # Vegetation stripped exposing bare soil / rock: Red increases to ~0.25, NIR drops to ~0.28 -> NDVI drops to ~0.05
    for r in range(12, 19):
        for c in range(10, 17):
            if (r - 15) ** 2 + (c - 13) ** 2 <= 10:
                post_red[r, c] = 0.25 + rng.normal(0.0, 0.02)
                post_nir[r, c] = 0.28 + rng.normal(0.0, 0.02)

    # Post-monsoon cloud patch in top-left corner
    post_cloud[0:6, 0:7] = True

    post_scene = compute_ndvi(
        red_band=post_red,
        nir_band=post_nir,
        cloud_mask=post_cloud,
        observation_date="2024-08-06",
        sensor="Sentinel-2A MSI L2A",
        bounds=(76.10, 11.45, 76.13, 11.48),
        resolution_m=10.0,
    )

    return pre_scene, post_scene

