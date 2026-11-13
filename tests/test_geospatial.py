"""Unit and integration tests for Phase 2 Geospatial Risk Heatmap & Terrain Analysis."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from api import get_risk_geojson
from src.geospatial.maps import (
    MAP_RISK_PALETTE,
    add_geojson_layer,
    add_india_region_layers,
    add_risk_heatmap,
    add_risk_markers,
    build_landslide_map,
    create_base_map,
)
from src.geospatial.spatial_analysis import (
    GEOPANDAS_AVAILABLE,
    apply_geographic_filters,
    drop_invalid_coordinates,
    ensure_wgs84_geodataframe,
    inspect_location_details,
    records_to_geojson,
    validate_geojson,
)
from src.geospatial.terrain import (
    PYPROJ_AVAILABLE,
    aspect_degrees_to_cardinal,
    compute_slope_and_aspect,
    get_point_terrain_summary,
    inspect_dem_raster,
    transform_point_crs,
)


def _run_coro_sync(coro):
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


def test_geojson_validation_and_export():
    valid_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [77.0595, 10.0889]},
                "properties": {"name": "Munnar", "risk_category": "High"},
            }
        ],
    }
    normalized = validate_geojson(valid_fc)
    assert normalized["type"] == "FeatureCollection"
    assert len(normalized["features"]) == 1

    # Single Feature normalizes cleanly to FeatureCollection
    single = validate_geojson(valid_fc["features"][0])
    assert len(single["features"]) == 1

    # Out-of-bounds WGS84 longitude rejected
    with pytest.raises(ValueError, match="invalid or out-of-range"):
        validate_geojson({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [205.0, 10.0]},
                    "properties": {},
                }
            ],
        })

    exported = records_to_geojson(
        [
            {"latitude": 10.0889, "longitude": 77.0595, "name": "Munnar", "probability": 0.78},
            {"latitude": 999.0, "longitude": 77.0, "name": "Invalid"},
        ]
    )
    assert len(exported["features"]) == 1
    assert exported["features"][0]["geometry"]["coordinates"] == [77.0595, 10.0889]


def test_invalid_coordinate_rejection_and_empty_dataset():
    df = pd.DataFrame({
        "latitude": [10.0, 95.0, None, -120.0, 27.04],
        "longitude": [77.0, 77.0, 77.0, 80.0, 200.0],
        "state_region": ["Kerala", "Bad", "Bad", "Bad", "Bad"],
    })
    valid, dropped = drop_invalid_coordinates(df)
    assert len(valid) == 1
    assert dropped == 4

    empty_filtered = apply_geographic_filters(pd.DataFrame())
    assert empty_filtered.empty

    empty_map = build_landslide_map(observations_df=pd.DataFrame(), risk_data=[])
    html = empty_map.get_root().render()
    assert "Provisional Risk Legend" in html
    assert "Green:" in html


def test_geographic_filters():
    df = pd.DataFrame({
        "latitude": [10.08, 11.68, 30.55],
        "longitude": [77.05, 76.13, 79.56],
        "state_region": ["Kerala", "Kerala", "Uttarakhand"],
        "risk_category": ["High", "Low", "Very High"],
        "source_name": ["NASA GLC", "Synthetic", "Open-Meteo"],
        "event_date": ["2024-07-15", "2023-01-10", "2025-08-20"],
    })
    filtered = apply_geographic_filters(
        df,
        states_regions=["Kerala"],
        min_lat=9.0,
        max_lat=12.5,
        min_lon=75.0,
        max_lon=78.0,
        risk_categories=["High", "Very High"],
        date_range=(date(2024, 1, 1), date(2024, 12, 31)),
        data_sources=["NASA GLC"],
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["state_region"] == "Kerala"


def test_risk_markers_heatmap_and_map_rendering():
    risk_entries = [
        {
            "location_name": "Munnar",
            "latitude": 10.0889,
            "longitude": 77.0595,
            "probability": 0.82,
            "risk_category": "Very High",
            "rainfall_24h_mm": 120.5,
            "elevation_m": 1532.0,
            "slope_degrees": 31.0,
            "aspect_degrees": 135.0,
            "data_source": "Open-Meteo Reanalysis",
            "observed_at": "2026-11-12T10:00:00+00:00",
            "model_name": "Logistic Regression",
            "model_version": "2026-11-12",
            "method": "ml_model",
            "is_experimental": True,
        },
        {
            "location_name": "BadPoint",
            "latitude": 150.0,
            "longitude": 77.0,
            "probability": 0.5,
            "risk_category": "Moderate",
        },
        {
            "location_name": "Wayanad",
            "latitude": 11.6854,
            "longitude": 76.1320,
            "probability": 0.42,
            "risk_category": "Moderate",
            "method": "ml_model",
        },
        {
            "location_name": "Dehradun",
            "latitude": 30.3165,
            "longitude": 78.0322,
            "probability": 0.21,
            "risk_category": "Low",
            "method": "ml_model",
        },
    ]

    assert MAP_RISK_PALETTE["Low"] == "#2ecc71"
    assert MAP_RISK_PALETTE["Moderate"] == "#f1c40f"
    assert MAP_RISK_PALETTE["High"] == "#e67e22"
    assert MAP_RISK_PALETTE["Very High"] == "#e74c3c"

    obs_df = pd.DataFrame({
        "latitude": [10.10],
        "longitude": [77.06],
        "event_date": ["2024-07-30"],
        "country": ["India"],
        "state_region": ["Kerala"],
        "landslide_type": ["mudslide"],
        "trigger": ["downpour"],
        "source_name": ["Synthetic Test"],
        "is_synthetic": [1],
    })

    gj = records_to_geojson([risk_entries[0]])
    m = build_landslide_map(
        observations_df=obs_df,
        risk_data=risk_entries,
        center_lat=15.0,
        center_lon=77.0,
        zoom=5,
        show_heatmap=True,
        geojson_data=gj,
    )
    html = m.get_root().render()
    assert "Munnar" in html
    assert "Logistic Regression (2026-11-12)" in html
    assert "Open-Meteo Reanalysis" in html
    assert "Experimental Risk Heatmap" in html
    assert "Yellow:" in html and "Orange:" in html

    # Empty heatmap should plot 0 points without inventing coordinates
    m2 = create_base_map()
    m2, meta = add_risk_heatmap(m2, [{"latitude": 10.0, "longitude": 77.0, "probability": None}])
    assert meta["points_plotted"] == 0
    assert meta["skipped_points"] == 1
    assert meta["sufficient_spatial_coverage"] is False

    m3 = create_base_map()
    add_risk_markers(m3, risk_entries)
    add_geojson_layer(m3, gj)
    m3_html = m3.get_root().render()
    assert "geo_json_" in m3_html and "Munnar" in m3_html


def test_terrain_slope_aspect_and_missing_dem_handling(tmp_path: Path):
    # South-facing inclined plane: elevation increases northwards (row 0 = top/North)
    # Row 0 = 130m, Row 1 = 120m, Row 2 = 110m with 10m spacing -> slope = 45 deg, aspect = 180 (South)
    grid = np.array([
        [130.0, 130.0, 130.0],
        [120.0, 120.0, 120.0],
        [110.0, 110.0, -9999.0],
    ])
    res = compute_slope_and_aspect(grid, pixel_size_x_m=10.0, pixel_size_y_m=10.0, nodata=-9999.0)
    assert res["valid_cells"] == 8
    assert res["nodata_cells"] == 1
    # Interior unmasked cell [1, 0] has exact gradient slope 45.0 deg facing South (180 deg)
    assert res["slope_deg"][1, 0] == pytest.approx(45.0, abs=1e-3)
    assert res["aspect_deg"][1, 0] == pytest.approx(180.0, abs=1e-3)
    assert aspect_degrees_to_cardinal(180.0) == "S"

    # Missing DEM file should fail transparently without inventing values
    missing_info = inspect_dem_raster(tmp_path / "missing_dem.tif")
    assert missing_info["available"] is False
    assert "not found" in missing_info["reason"].lower()

    # Point SRTM terrain calculation via mocked Open-Elevation 5-point batch
    mock_tp = MagicMock()
    mock_tp.attribution = "Open-Elevation SRTM"
    # c, n, s, e, w
    mock_tp.fetch_batch.return_value = [1500.0, 1525.0, 1475.0, 1500.0, 1500.0]
    pt = get_point_terrain_summary(10.0889, 77.0595, terrain_provider=mock_tp)
    assert pt["available"] is True
    assert pt["elevation_m"] == 1500.0
    assert pt["slope_degrees"] > 0
    assert pt["aspect_cardinal"] == "S"

    # Unavailable Open-Elevation -> available=False, zero invented numbers
    mock_tp_down = MagicMock()
    mock_tp_down.fetch_batch.return_value = None
    unavail = get_point_terrain_summary(10.0889, 77.0595, terrain_provider=mock_tp_down)
    assert unavail["available"] is False
    assert unavail["elevation_m"] is None
    assert "No terrain values were fabricated" in unavail["reason"]


def test_geospatial_geojson_api_endpoint(monkeypatch):
    mock_monitor = MagicMock()
    mock_monitor.get_live_snapshot.return_value = {
        "stations": [
            {"location_id": 1, "name": "Munnar", "latitude": 10.0889, "longitude": 77.0595, "probability": 0.72}
        ]
    }
    monkeypatch.setattr("src.monitoring.get_live_monitor", lambda: mock_monitor)

    fc = _run_coro_sync(get_risk_geojson(active_only=True))
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    assert fc["features"][0]["geometry"]["coordinates"] == [77.0595, 10.0889]


def test_phase2_district_search_authentic_filters_and_location_inspector():
    df = pd.DataFrame([
        {
            "event_id": "GLC-101",
            "latitude": 10.09,
            "longitude": 77.06,
            "state_region": "Kerala",
            "district": "Idukki",
            "location_name": "Munnar Gap",
            "source_name": "NASA GLC",
            "trigger": "downpour",
            "landslide_type": "debris flow",
            "event_date": "2024-07-30",
            "is_synthetic": 0,
        },
        {
            "event_id": "SYN-202",
            "latitude": 11.68,
            "longitude": 76.13,
            "state_region": "Kerala",
            "district": "Wayanad",
            "location_name": "Vythiri",
            "source_name": "Synthetic Dev",
            "trigger": "rain",
            "landslide_type": "mudslide",
            "event_date": "2024-08-05",
            "is_synthetic": 1,
        },
    ])

    by_district = apply_geographic_filters(df, districts=["Idukki"], authentic_only=True)
    assert len(by_district) == 1
    assert by_district.iloc[0]["event_id"] == "GLC-101"

    by_search = apply_geographic_filters(df, search_query="vythiri")
    assert len(by_search) == 1
    assert by_search.iloc[0]["event_id"] == "SYN-202"

    stations = [
        {
            "name": "Munnar",
            "latitude": 10.0889,
            "longitude": 77.0595,
            "elevation_m": 1532.0,
            "slope_degrees": 28.0,
            "rainfall_1h_mm": 6.4,
            "rainfall_24h_mm": 88.2,
            "rainfall_72h_mm": 165.0,
            "soil_moisture": 0.412,
            "data_source": "Open-Meteo ECMWF",
            "freshness_status": "Fresh",
            "observed_at": "2026-11-12T10:00:00+00:00",
            "predicted_at": "2026-11-12T10:05:00+00:00",
            "data_age_hours": 0.2,
            "probability": 0.78,
            "risk_category": "Very High",
            "model_name": "Logistic Regression",
            "model_version": "2026-11-12",
            "method": "ml_model",
        },
        {
            "name": "UnscoredPeak",
            "latitude": 27.04,
            "longitude": 88.26,
            "probability": None,
            "risk_category": "Unavailable",
        },
    ]

    # Map rendering handles unavailable markers and missing boundary overlays cleanly
    assert MAP_RISK_PALETTE["Unavailable"] == "#95a5a6"
    assert MAP_RISK_PALETTE["Insufficient Data"] == "#95a5a6"
    m = build_landslide_map(
        observations_df=by_district,
        risk_data=stations,
        show_heatmap=True,
        show_risk_markers=True,
        cluster_risk_markers=True,
        state_geojson=None,
        district_geojson=None,
    )
    html = m.get_root().render()
    assert "GLC-101" in html
    assert "Authentic Landslide Event" in html
    assert "Unavailable / Insufficient Data" in html

    # Click inspector returns all 7 components for matched station & nearby events
    details = inspect_location_details(10.09, 77.06, stations=stations, observations_df=df, radius_km=25.0)
    assert details["valid"] is True
    assert details["coordinates"]["crs"] == "EPSG:4326"
    assert details["rainfall"]["rainfall_24h_mm"] == 88.2
    assert details["terrain"]["elevation_m"] == 1532.0
    assert details["historical_events"]["authentic_count"] == 1
    assert details["risk_estimate"]["risk_category"] == "Very High"
    assert details["data_freshness"]["freshness_status"] == "Fresh"
    assert "24h rainfall is 88.2 mm" in details["risk_explanation"]

    # Far away empty region returns honest Unavailable state without fabrication
    empty_details = inspect_location_details(20.5, 78.9, stations=[], observations_df=pd.DataFrame(), radius_km=25.0)
    assert empty_details["valid"] is True
    assert empty_details["rainfall"]["available"] is False
    assert empty_details["terrain"]["available"] is False
    assert empty_details["risk_estimate"]["risk_category"] == "Unavailable"
    assert "No model risk assessment is available" in empty_details["risk_explanation"]


def test_phase2_crs_validation_invalid_geometries_and_terrain_roughness():
    # Reject unclosed / <4-vertex polygon ring & projected CRS GeoJSON
    with pytest.raises(ValueError, match="invalid or out-of-range"):
        validate_geojson({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[77.0, 10.0], [77.1, 10.0], [77.1, 10.1]]]},
                    "properties": {},
                }
            ],
        })

    with pytest.raises(ValueError, match="Unsupported non-WGS84 GeoJSON CRS"):
        validate_geojson({
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:32643"}},
            "features": [],
        })

    if PYPROJ_AVAILABLE:
        lon_wgs, lat_wgs = transform_point_crs(725000.0, 1115000.0, "EPSG:32643", "EPSG:4326")
        assert 76.0 <= lon_wgs <= 78.5
        assert 9.5 <= lat_wgs <= 11.0

    if GEOPANDAS_AVAILABLE:
        import geopandas as gpd
        from shapely.geometry import Point

        unreferenced = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(77.0, 10.0)])
        with pytest.raises(ValueError, match="no Coordinate Reference System"):
            ensure_wgs84_geodataframe(unreferenced)

        utm_gdf = gpd.GeoDataFrame({"name": ["Munnar"]}, geometry=[Point(725000.0, 1115000.0)], crs="EPSG:32643")
        wgs_gdf = ensure_wgs84_geodataframe(utm_gdf)
        assert str(wgs_gdf.crs).upper() == "EPSG:4326"
        assert 76.0 <= float(wgs_gdf.geometry.iloc[0].x) <= 78.5

    # Terrain roughness (TRI) + relief + curvature on 3x3 DEM array
    dem_3x3 = np.array([
        [100.0, 105.0, 110.0],
        [100.0, 112.0, 110.0],
        [100.0, 105.0, 110.0],
    ])
    t_res = compute_slope_and_aspect(dem_3x3, pixel_size_x_m=30.0, pixel_size_y_m=30.0)
    assert t_res["mean_tri_m"] is not None and t_res["mean_tri_m"] > 0
    assert t_res["elevation_relief_m"] == 12.0
    assert t_res["mean_curvature"] is not None

    # Point SRTM stencil also computes roughness_tri_m & local_relief_m
    mock_tp = MagicMock()
    mock_tp.attribution = "Open-Elevation SRTM"
    mock_tp.fetch_batch.return_value = [1500.0, 1520.0, 1480.0, 1510.0, 1490.0]
    pt = get_point_terrain_summary(10.0889, 77.0595, terrain_provider=mock_tp)
    assert pt["roughness_tri_m"] > 0
    assert pt["local_relief_m"] == 40.0
    assert pt["curvature_proxy"] is not None

    # add_india_region_layers handles empty boundary overlays safely
    m = create_base_map()
    add_india_region_layers(m, state_geojson=None, district_geojson=None, state_name="Kerala", district_name="Idukki")
    assert "folium" in m.get_root().render().lower()
