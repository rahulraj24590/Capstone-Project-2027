"""
Unit and integration tests for Phase 7 Advanced Geospatial and Satellite Intelligence.

Covers:
1. DEM-based terrain analysis (Slope, Aspect, TRI Roughness, Curvature).
2. Topographic Wetness Index (TWI) input requirement guards & D8 flow accumulation.
3. 3D interactive terrain model mesh generation.
4. Raster loading, CRS handling, missing DEM, and unreferenced raster rejection.
5. Landslide hotspot detection (DBSCAN haversine clustering & KDE density surface).
6. Anti-hallucination disclaimers (historical clustering is non-predictive).
7. Sentinel-2 NDVI spectral band enforcement (Red + NIR mandatory) and cloud masking.
8. Multi-temporal satellite change detection (dNDVI), area calculation, and verification indicator.
9. Phase 7 FastAPI REST endpoints.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api
from src.geospatial.hotspot_detection import (
    HOTSPOT_SCIENTIFIC_DISCLAIMER,
    detect_landslide_hotspots,
)
from src.geospatial.satellite_monitor import (
    CHANGE_DETECTION_VERIFICATION_LABEL,
    VEGETATION_DISCLAIMER,
    compute_ndvi,
    detect_satellite_changes,
    get_sample_sentinel2_scenario,
)
from src.geospatial.spatial_analysis import validate_geojson
from src.geospatial.terrain import (
    build_3d_terrain_surface,
    compute_d8_flow_accumulation,
    compute_slope_and_aspect,
    compute_topographic_wetness_index,
    get_point_terrain_summary,
    inspect_dem_raster,
)


@pytest.fixture
def sample_elevation_grid() -> np.ndarray:
    """Create a 10x10 synthetic planar inclined plane with slope."""
    r = np.arange(10, dtype=float).reshape(10, 1)
    c = np.arange(10, dtype=float).reshape(1, 10)
    # Elevation drops north-to-south and west-to-east
    return 1000.0 - (r * 15.0) - (c * 10.0)


# ── 1. DEM Terrain Analysis & Topographic Wetness Index (TWI) ─────────

def test_dem_terrain_derivatives_and_twi_guard(sample_elevation_grid):
    # 1. Compute standard derivatives
    stats = compute_slope_and_aspect(sample_elevation_grid, pixel_size_x_m=30.0, pixel_size_y_m=30.0)
    assert stats["valid_cells"] == 100
    assert stats["mean_slope_deg"] is not None
    assert 20.0 < stats["mean_slope_deg"] < 40.0
    assert stats["mean_tri_m"] is not None
    assert stats["mean_aspect_deg"] is not None

    # 2. Strict TWI refusal when catchment area is NOT provided
    assert stats["twi"] is None
    assert "required" in stats["twi_status"].lower()

    twi_refusal = compute_topographic_wetness_index(stats["slope_deg"], specific_catchment_area_m=None)
    assert twi_refusal["available"] is False
    assert twi_refusal["twi"] is None
    assert "refused" in twi_refusal["reason"].lower()

    # 3. D8 flow accumulation calculates upslope contributing area
    d8_flow = compute_d8_flow_accumulation(sample_elevation_grid, pixel_size_x_m=30.0, pixel_size_y_m=30.0)
    assert d8_flow["flow_accumulation_cells"].shape == (10, 10)
    assert d8_flow["specific_catchment_area_m"].shape == (10, 10)
    assert d8_flow["max_accumulation"] >= 10.0

    # 4. Authentic TWI calculated when specific catchment area IS supplied
    twi_computed = compute_topographic_wetness_index(
        stats["slope_deg"],
        specific_catchment_area_m=d8_flow["specific_catchment_area_m"],
    )
    assert twi_computed["available"] is True
    assert twi_computed["mean_twi"] is not None
    assert twi_computed["max_twi"] > twi_computed["min_twi"]

    # 5. Dimension checks & edge cases
    with pytest.raises(ValueError, match="at least 2x2"):
        compute_slope_and_aspect(np.array([[100.0]]), 30.0, 30.0)

    with pytest.raises(ValueError, match="strictly positive"):
        compute_slope_and_aspect(sample_elevation_grid, -10.0, 30.0)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_topographic_wetness_index(np.ones((4, 4)), specific_catchment_area_m=np.ones((5, 5)))


# ── 2. 3D Terrain Visualization ──────────────────────────────────────

def test_3d_terrain_surface_builder(sample_elevation_grid):
    # Standard elevation drape
    surf_elev = build_3d_terrain_surface(
        sample_elevation_grid,
        bounds=(76.0, 10.0, 76.1, 10.1),
        drape_mode="elevation",
    )
    assert surf_elev["rows"] == 10
    assert surf_elev["cols"] == 10
    assert len(surf_elev["figure"].data) == 1
    assert surf_elev["elevation_min_m"] is not None
    assert surf_elev["elevation_max_m"] > surf_elev["elevation_min_m"]

    # Slope drape
    slope_mat = np.full((10, 10), 25.0)
    surf_slope = build_3d_terrain_surface(
        sample_elevation_grid,
        drape_mode="slope",
        drape_grid=slope_mat,
    )
    assert surf_slope["drape_mode"] == "slope"

    with pytest.raises(ValueError, match="at least 2x2"):
        build_3d_terrain_surface(np.array([100.0, 200.0]))


# ── 3. Raster Loading, CRS Handling & Missing DEM ─────────────────────

def test_raster_loading_crs_handling_and_missing_dem(tmp_path):
    # 1. Missing DEM file -> returns available=False gracefully
    missing_path = tmp_path / "non_existent_dem.tif"
    res_miss = inspect_dem_raster(missing_path)
    assert res_miss["available"] is False
    assert "not found" in res_miss["reason"].lower()

    # 2. Unreferenced TIFF (no GeoTIFF CRS tags) -> refuses unreferenced sampling
    unreferenced_tif = tmp_path / "plain_image.tif"
    arr = np.random.uniform(500, 1500, size=(20, 20)).astype(np.float32)
    Image.fromarray(arr).save(unreferenced_tif)

    res_unref = inspect_dem_raster(unreferenced_tif)
    assert res_unref["available"] is False
    assert "crs" in res_unref["reason"].lower() or "tags" in res_unref["reason"].lower()

    # 3. GeoTIFF with ModelPixelScale and ModelTiepoint tags
    geotiff_path = tmp_path / "valid_geotiff.tif"
    im = Image.fromarray(arr)
    # Add GeoTIFF tags: 33550 (ModelPixelScale: [scale_x, scale_y, 0]), 33922 (ModelTiepoint: [0,0,0, lon, lat, 0])
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (0.000277, 0.000277, 0.0)
    ifd[33922] = (0.0, 0.0, 0.0, 76.5, 11.5, 0.0)
    im.save(geotiff_path, tiffinfo=ifd)

    res_geo = inspect_dem_raster(geotiff_path, sample_lat=11.498, sample_lon=76.502)
    assert res_geo["available"] is True
    assert res_geo["crs"] == "EPSG:4326"
    assert res_geo["mean_elevation_m"] is not None

    # 4. Point SRTM inspection coordinate boundary validation
    invalid_pt = get_point_terrain_summary(95.0, 78.0)
    assert invalid_pt["available"] is False
    assert "latitude" in invalid_pt["reason"].lower()


# ── 4. Landslide Hotspot Detection (DBSCAN & KDE) ──────────────────────

def test_landslide_hotspot_dbscan_clustering_and_disclaimer():
    # Observations with 2 distinct geographic clusters + 1 outlier
    # Cluster 0: Uttarakhand foothills (~30.1°N, 78.2°E)
    # Cluster 1: Kerala Western Ghats (~10.5°N, 76.8°E)
    # Outlier: Eastern Ghats (~20.0°N, 85.0°E)
    df = pd.DataFrame({
        "id": [f"evt-{i}" for i in range(7)],
        "latitude": [30.10, 30.12, 30.11, 10.50, 10.52, 10.51, 20.00],
        "longitude": [78.20, 78.21, 78.19, 76.80, 76.81, 76.79, 85.00],
        "event_date": ["2023-08-01", "2023-08-02", "2023-08-03", "2024-07-15", "2024-07-16", "2024-07-17", "2022-01-01"],
        "trigger": ["Monsoon", "Monsoon", "Continuous Rain", "Downpour", "Downpour", "Downpour", "Earthquake"],
        "landslide_type": ["Debris Flow", "Debris Flow", "Rockfall", "Mudslide", "Mudslide", "Mudslide", "Rockfall"],
        "is_synthetic": [0, 0, 0, 0, 0, 0, 0],
    })

    hotspots = detect_landslide_hotspots(df, eps_km=25.0, min_samples=3)
    assert hotspots["available"] is True
    assert hotspots["summary"]["cluster_count"] == 2
    assert hotspots["summary"]["noise_events_count"] == 1
    assert len(hotspots["clusters"]) == 2

    # Check cluster properties
    c0 = hotspots["clusters"][0]
    assert c0["point_count"] == 3
    assert c0["event_density_per_sq_km"] > 0
    assert "start" in c0["date_range"]
    assert len(c0["convex_hull"]) >= 3

    # Anti-hallucination disclaimer check
    assert HOTSPOT_SCIENTIFIC_DISCLAIMER in hotspots["disclaimer"]
    assert "does not predict future" in hotspots["disclaimer"].lower()

    # Authentic-only filter test
    df_synth = df.copy()
    df_synth["is_synthetic"] = 1
    hot_auth = detect_landslide_hotspots(df_synth, authentic_only=True)
    assert hot_auth["available"] is False
    assert "synthetic" in hot_auth["reason"].lower()

    # Insufficient data check
    short_df = df.iloc[:2]
    hot_short = detect_landslide_hotspots(short_df, min_samples=3)
    assert hot_short["available"] is False
    assert "insufficient" in hot_short["reason"].lower()


def test_kernel_density_estimation_hotspots():
    df = pd.DataFrame({
        "latitude": [30.10, 30.12, 30.11, 30.13, 10.50, 10.52, 10.51, 10.53],
        "longitude": [78.20, 78.21, 78.19, 78.22, 76.80, 76.81, 76.79, 76.82],
    })
    res = detect_landslide_hotspots(df, method="kde", kde_grid_size=20)
    assert res["available"] is True
    assert res["kde_density"] is not None
    assert "percentiles" in res["kde_density"]
    assert res["kde_density"]["percentiles"]["p99"] > 0
    assert len(res["hotspot_regions"]) > 0


# ── 5. Satellite Vegetation Monitoring (NDVI) ─────────────────────────

def test_satellite_ndvi_spectral_band_enforcement_and_cloud_masking():
    # 1. Refusal on missing NIR band
    with pytest.raises(ValueError, match="both Red.*and NIR"):
        compute_ndvi(red_band=np.ones((5, 5)), nir_band=None)

    with pytest.raises(ValueError, match="both Red.*and NIR"):
        compute_ndvi(red_band=None, nir_band=np.ones((5, 5)))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        compute_ndvi(red_band=np.ones((5, 5)), nir_band=np.ones((6, 6)))

    # 2. Healthy vegetation: high NIR, low Red -> high positive NDVI
    red = np.full((10, 10), 0.08)
    nir = np.full((10, 10), 0.60)
    # Expected NDVI = (0.60 - 0.08) / (0.60 + 0.08) = 0.52 / 0.68 ≈ 0.7647
    res = compute_ndvi(red_band=red, nir_band=nir, observation_date="2024-05-18")
    assert res["available"] is True
    assert pytest.approx(res["statistics"]["mean_ndvi"], abs=0.01) == 0.7647
    assert res["vegetation_classes"]["dense_canopy_pct"] == 100.0
    assert VEGETATION_DISCLAIMER in res["disclaimer"]

    # 3. Cloud Masking
    cloud_mask = np.zeros((10, 10), dtype=bool)
    cloud_mask[0:2, 0:2] = True  # 4 pixels cloudy
    res_cloud = compute_ndvi(red_band=red, nir_band=nir, cloud_mask=cloud_mask)
    assert res_cloud["cloud_masked_count"] == 4
    assert res_cloud["cloud_cover_pct"] == 4.0
    assert res_cloud["valid_pixel_count"] == 96
    # Masked pixels are NaN in ndvi_grid
    assert np.isnan(res_cloud["ndvi_grid"][0, 0])

    # 4. SCL categorical cloud mask handling (e.g. SCL=9 for high cloud)
    scl = np.full((10, 10), 4)  # 4 = vegetation
    scl[0, 0] = 9               # 9 = cloud
    res_scl = compute_ndvi(red_band=red, nir_band=nir, cloud_mask=scl)
    assert res_scl["cloud_masked_count"] == 1
    assert np.isnan(res_scl["ndvi_grid"][0, 0])


# ── 6. Satellite Change Detection ─────────────────────────────────────

def test_satellite_change_detection_and_verification_label():
    pre_scene, post_scene = get_sample_sentinel2_scenario()
    assert pre_scene["available"] is True
    assert post_scene["available"] is True

    change = detect_satellite_changes(pre_scene, post_scene)
    assert change["available"] is True
    assert "2024-05-18 → 2024-08-06" == change["temporal_baseline"]

    # Localized simulated slope failure in sample scenario
    assert change["change_classification"]["severe_vegetation_loss_pixels"] > 0
    assert change["candidate_disturbance_area_km2"] > 0.0
    assert change["cloud_obscured_pct"] > 0.0

    # Verification indicator label & disclaimer presence
    assert CHANGE_DETECTION_VERIFICATION_LABEL in change["verification_indicator_label"]
    assert "unverified screening indicator" in change["verification_indicator_label"].lower()
    assert "10m" in change["limitations"]["spatial_resolution"]

    # Shape mismatch refusal
    small_scene = compute_ndvi(red_band=np.ones((5, 5)), nir_band=np.ones((5, 5)) * 0.5)
    with pytest.raises(ValueError, match="Shape mismatch"):
        detect_satellite_changes(pre_scene, small_scene)


# ── 7. GeoJSON Validation & CRS Handling ──────────────────────────────

def test_invalid_geometries_and_geojson_crs_handling():
    # Valid GeoJSON
    valid_gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [76.5, 11.5]},
                "properties": {"name": "Test Point"},
            }
        ],
    }
    assert validate_geojson(valid_gj)["type"] == "FeatureCollection"

    # Out-of-bounds coordinates -> raises ValueError
    invalid_coords_gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [999.0, 11.5]},
                "properties": {},
            }
        ],
    }
    with pytest.raises(ValueError, match="invalid or out-of-range"):
        validate_geojson(invalid_coords_gj)

    # Non-WGS84 CRS specification -> raises ValueError
    projected_gj = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32643"}},
        "features": [],
    }
    with pytest.raises(ValueError, match="Unsupported non-WGS84"):
        validate_geojson(projected_gj)


# ── 8. Phase 7 FastAPI REST Endpoints ─────────────────────────────────

def _run_coro_sync(coro):
    """Execute a coroutine synchronously without creating an asyncio event loop."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


def test_phase7_fastapi_endpoints():
    from api import (
        SatelliteChangeRequest,
        SatelliteNdviRequest,
        TerrainDerivativesRequest,
        evaluate_satellite_change_detection,
        evaluate_satellite_ndvi,
        evaluate_terrain_derivatives,
        get_geospatial_hotspots,
    )
    from fastapi import HTTPException

    # 1. GET /api/v1/geospatial/hotspots
    data_hot = _run_coro_sync(get_geospatial_hotspots(eps_km=30.0, min_samples=2))
    assert data_hot["available"] is True
    assert "disclaimer" in data_hot
    assert "CRITICAL SCIENTIFIC NOTICE" in data_hot["disclaimer"]

    # 2. POST /api/v1/satellite/ndvi
    red_5x5 = [[0.08] * 5 for _ in range(5)]
    nir_5x5 = [[0.60] * 5 for _ in range(5)]
    data_ndvi = _run_coro_sync(
        evaluate_satellite_ndvi(
            SatelliteNdviRequest(
                red_band=red_5x5,
                nir_band=nir_5x5,
                observation_date="2024-05-18",
                resolution_m=10.0,
            )
        )
    )
    assert data_ndvi["available"] is True
    assert data_ndvi["statistics"]["mean_ndvi"] > 0.70
    assert "disclaimer" in data_ndvi

    # 3. POST /api/v1/satellite/ndvi (dimension mismatch -> 400 Bad Request)
    with pytest.raises(HTTPException) as exc_info:
        _run_coro_sync(
            evaluate_satellite_ndvi(
                SatelliteNdviRequest(
                    red_band=[[0.08] * 4 for _ in range(4)],
                    nir_band=nir_5x5,
                    observation_date="2024-05-18",
                    resolution_m=10.0,
                )
            )
        )
    assert exc_info.value.status_code == 400

    # 4. POST /api/v1/satellite/change-detection
    data_change = _run_coro_sync(
        evaluate_satellite_change_detection(
            SatelliteChangeRequest(
                pre_red_band=red_5x5,
                pre_nir_band=nir_5x5,
                pre_date="2024-05-18",
                post_red_band=[[0.25] * 5 for _ in range(5)],
                post_nir_band=[[0.30] * 5 for _ in range(5)],
                post_date="2024-08-06",
                resolution_m=10.0,
            )
        )
    )
    assert data_change["available"] is True
    assert data_change["change_classification"]["severe_vegetation_loss_pixels"] == 25
    assert "verification_indicator_label" in data_change

    # 5. POST /api/v1/terrain/derivatives
    elev_4x4 = [
        [1000.0, 990.0, 980.0, 970.0],
        [985.0,  975.0, 965.0, 955.0],
        [970.0,  960.0, 950.0, 940.0],
        [955.0,  945.0, 935.0, 925.0],
    ]
    data_terr = _run_coro_sync(
        evaluate_terrain_derivatives(
            TerrainDerivativesRequest(
                elevation_grid=elev_4x4,
                pixel_size_x_m=30.0,
                pixel_size_y_m=30.0,
            )
        )
    )
    assert data_terr["valid_cells"] == 16
    assert data_terr["mean_slope_deg"] > 0
    assert "required" in data_terr["twi_status"].lower()

