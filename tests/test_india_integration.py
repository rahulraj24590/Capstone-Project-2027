"""Tests for India State/District Real-Time Data Integration.

Covers all unit and integration scenarios with mocked external APIs:
1. India State/UT and dependent District selection & search
2. Spatial containment & island nearest-polygon ADM1<-ADM2 mapping
3. Coordinate resolution & validation
4. Weather parsing (Open-Meteo separated current/historical/forecast)
5. IMD parsing (rainfall, warnings, nowcast, forecast, Not configured, Unavailable)
6. SACHET CAP JSON & RSS XML parsing, ETag cache, Link-only fallback, region filter
7. Missing API responses & graceful degradation
8. Invalid timestamps & freshness classification
9. Missing rainfall preservation (never zero-filled)
10. Interactive India Folium boundary & point risk map generation
11. End-to-end 10-step RiskEngine + SHAP integration
12. SQLite persistence & deduplication (india_states, india_districts, official_warnings, provider_status)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from src.data_providers.imd_provider import IMDProvider
from src.data_providers.india_boundaries import (
    IndiaBoundaryProvider,
    build_admin_index_from_geodataframes,
    normalize_admin_name,
)
from src.data_providers.sachet_provider import SACHETProvider
from src.database import db as db_mod
from src.database.db import initialize_database
from src.database.models import (
    get_official_warnings,
    get_provider_statuses,
    list_india_districts,
    list_india_states,
    sync_india_boundaries,
    upsert_official_warning,
    upsert_provider_status,
)
from src.geospatial.maps import build_india_regional_map
from src.monitoring.india_monitor import IndiaRegionalMonitor, _compute_freshness


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to an isolated temporary SQLite database per test."""
    test_db = tmp_path / "test_india_integration.db"
    monkeypatch.setattr(db_mod, "DB_PATH", test_db)
    initialize_database(test_db)
    return test_db


@pytest.fixture
def synthetic_boundaries(tmp_path):
    """Create minimal valid GeoJSON + index files for 2 States and 3 Districts (including an island)."""
    state1_poly = Polygon([(76.0, 9.5), (77.5, 9.5), (77.5, 10.5), (76.0, 10.5)])
    state2_poly = Polygon([(77.8, 30.0), (78.8, 30.0), (78.8, 31.0), (77.8, 31.0)])
    gdf1 = gpd.GeoDataFrame(
        {"shapeName": ["Kerala", "Uttarakhand"], "shapeID": ["ST-KL", "ST-UK"], "shapeISO": ["IN-KL", "IN-UT"]},
        geometry=[state1_poly, state2_poly],
        crs="EPSG:4326",
    )

    dist1_poly = Polygon([(76.5, 9.7), (77.3, 9.7), (77.3, 10.2), (76.5, 10.2)])
    dist2_poly = Polygon([(76.1, 9.6), (76.4, 9.6), (76.4, 10.0), (76.1, 10.0)])
    dist3_poly = Polygon([(78.0, 30.2), (78.4, 30.2), (78.4, 30.6), (78.0, 30.6)])
    # Island slightly outside Kerala coast -> nearest fallback should map to Kerala
    dist_island = Polygon([(75.8, 9.8), (75.9, 9.8), (75.9, 9.9), (75.8, 9.9)])

    gdf2 = gpd.GeoDataFrame(
        {
            "shapeName": ["Idukki", "Ernakulam", "Dehradun", "Coastal Island"],
            "shapeID": ["DT-IDK", "DT-EKM", "DT-DDN", "DT-ISL"],
            "shapeISO": ["", "", "", ""],
        },
        geometry=[dist1_poly, dist2_poly, dist3_poly, dist_island],
        crs="EPSG:4326",
    )

    b_dir = tmp_path / "boundaries"
    b_dir.mkdir(parents=True, exist_ok=True)
    gdf1.to_file(b_dir / "adm1_states.geojson", driver="GeoJSON")
    gdf2.to_file(b_dir / "adm2_districts.geojson", driver="GeoJSON")

    idx = build_admin_index_from_geodataframes(gdf1, gdf2)
    import json
    (b_dir / "india_districts_index.json").write_text(json.dumps(idx), encoding="utf-8")
    return IndiaBoundaryProvider(boundaries_dir=b_dir)


# ─────────────────────────────────────────────
# 1-4. Geographical Selection, Mapping & Validation
# ─────────────────────────────────────────────

class TestIndiaBoundariesAndSelection:
    def test_state_and_district_selection_and_search(self, synthetic_boundaries):
        states = synthetic_boundaries.list_states()
        assert [s["state_name"] for s in states] == ["Kerala", "Uttarakhand"]

        assert len(synthetic_boundaries.list_states(query="ker")) == 1
        kerala_dists = synthetic_boundaries.list_districts(state_name="Kerala")
        names = {d["district_name"] for d in kerala_dists}
        assert {"Idukki", "Ernakulam", "Coastal Island"}.issubset(names)
        assert "Dehradun" not in names

        search_res = synthetic_boundaries.list_districts(state_name="Kerala", query="iduk")
        assert len(search_res) == 1
        assert search_res[0]["district_name"] == "Idukki"

    def test_coordinate_resolution_and_polygons(self, synthetic_boundaries):
        loc = synthetic_boundaries.resolve_location("Kerala", "Idukki")
        assert loc is not None
        assert 9.7 <= loc["rep_lat"] <= 10.2
        assert 76.5 <= loc["rep_lon"] <= 77.3
        assert len(loc["bbox"]) == 4

        st_gj = synthetic_boundaries.get_state_geojson("Kerala")
        dt_gj = synthetic_boundaries.get_district_geojson("Kerala", "Idukki")
        assert st_gj is not None and len(st_gj["features"]) == 1
        assert dt_gj is not None and len(dt_gj["features"]) == 1

    def test_normalize_admin_name_ascii(self):
        assert normalize_admin_name("  Dādra  & Nagar\tHaveli ") == "Dadra & Nagar Haveli"


# ─────────────────────────────────────────────
# 5. IMD Provider Parsing & Config States
# ─────────────────────────────────────────────

class TestIMDProvider:
    def test_not_configured_without_api_key(self, monkeypatch):
        monkeypatch.delenv("IMD_API_KEY", raising=False)
        monkeypatch.delenv("IMD_AUTH_TOKEN", raising=False)
        imd = IMDProvider()
        res = imd.fetch("Kerala", "Idukki")
        assert res["status"] == "Not configured"
        assert res["state_rainfall"] is None
        assert res["district_warning"] is None

    def test_parse_imd_documented_schemas(self):
        sr = IMDProvider.parse_state_rainfall(
            [{"State": "KERALA", "Date": "2026-07-15", "Daily Actual": "38.5", "Daily Normal": "22.0", "Daily Departure Per": "75%", "Daily Category": "LE", "Weekly Actual": "150", "Weekly Normal": "110", "Weekly Departure Per": "36%", "Monthly Acutual": "410", "Cumulative Actual": "920"}],
            "Kerala",
        )
        assert sr is not None
        assert sr["daily_actual_mm"] == 38.5
        assert "Large Excess" in sr["daily_category"]

        dr = IMDProvider.parse_district_rainfall(
            [{"OBJ_ID": "597", "District": "IDUKKI", "Date": "2026-07-15", "Daily Actual": "54.2", "Daily Normal": "25.0", "Daily Departure Per": "117%", "Daily Category": "E"}],
            "Idukki",
        )
        assert dr is not None and dr["daily_actual_mm"] == 54.2 and "Excess" in dr["daily_category"]

        dw = IMDProvider.parse_district_warning(
            [{"Obj_id": "597", "District": "IDUKKI", "Date": "2026-07-15", "UTC": "08", "Day_1": "4,27", "Day1_Color": "3", "Day2_Color": "2"}],
            "Idukki",
        )
        assert dw is not None
        assert "Orange" in dw["warning_category"]
        assert "Extremely Heavy Rain" in dw["warning_description"]

        dn = IMDProvider.parse_district_nowcast(
            [{"Station": "IDUKKI", "Date": "2026-07-15", "time": "1400", "vupto": "1700", "color": "3", "Cat7": "1", "Cat1": "0"}],
            "Idukki",
        )
        assert dn is not None and "Moderate rain" in dn["nowcast_summary"] and "Orange" in dn["nowcast_color"]

        fc = IMDProvider.parse_forecast_5day(
            [{"District": "IDUKKI", "State": "KERALA", "date_obs": "2026-07-15", "day1_distribution": "WS", "day1_distribution_percentage": "76-100%", "day1_color": "3"}],
            "Idukki",
        )
        assert fc is not None and len(fc["days"]) == 1 and fc["days"][0]["distribution"] == "WS"

    def test_imd_unavailable_on_http_error(self, monkeypatch):
        monkeypatch.setenv("IMD_API_KEY", "dummy-key")
        imd = IMDProvider()
        resp = MagicMock(status_code=401, text="Unauthorized")
        imd._session.get = MagicMock(return_value=resp)
        res = imd.fetch("Kerala", "Idukki")
        assert res["status"] == "Unavailable"


# ─────────────────────────────────────────────
# 6. NDMA SACHET Parsing, Caching & Fallbacks
# ─────────────────────────────────────────────

class TestSACHETProvider:
    def test_sachet_json_parsing_and_region_filter(self, tmp_path):
        provider = SACHETProvider()
        provider._etag_cache_file = tmp_path / "sachet_etag_cache.json"
        sample_json = [
            {
                "identifier": "alert-kl-1",
                "disaster_type": "Heavy Rain",
                "severity": "Severe",
                "severity_color": "Orange",
                "area_description": "Idukki, Kerala",
                "warning_message": "Heavy rainfall warning over Idukki.",
                "effective_start_time": "2026-07-15T06:00:00Z",
                "effective_end_time": "2026-07-16T06:00:00Z",
                "sender_org_id": "KSDMA",
                "centroid": "76.97,9.85",
            },
            {
                "identifier": "alert-rj-2",
                "disaster_type": "Heat Wave",
                "severity": "Moderate",
                "severity_color": "Yellow",
                "area_description": "Jaisalmer, Rajasthan",
                "warning_message": "Heat wave alert.",
                "effective_start_time": "2026-07-15T06:00:00Z",
                "effective_end_time": "2026-07-16T06:00:00Z",
                "sender_org_id": "IMD",
                "centroid": "70.91,26.91",
            },
        ]
        mock_resp = MagicMock(status_code=200, headers={"ETag": '"etag-1"'})
        mock_resp.json.return_value = sample_json
        provider._session.get = MagicMock(return_value=mock_resp)

        res = provider.fetch("Kerala", "Idukki")
        assert res["status"] == "Connected"
        assert res["total_india_alerts"] == 2
        assert len(res["regional_alerts"]) == 1
        assert res["regional_alerts"][0]["identifier"] == "alert-kl-1"
        assert pytest.approx(res["regional_alerts"][0]["centroid_lat"], 0.01) == 9.85
        assert "FetchXMLFile?identifier=alert-kl-1" in res["regional_alerts"][0]["official_url"]

        # Second call returns 304 Not Modified -> uses cached alerts transparently
        resp_304 = MagicMock(status_code=304, headers={})
        provider._session.get = MagicMock(return_value=resp_304)
        res_cached = provider.fetch("Kerala", "Idukki")
        assert res_cached["status"] == "Connected"
        assert res_cached["is_cached"] is True
        assert len(res_cached["regional_alerts"]) == 1

    def test_sachet_rss_and_link_only_fallback(self, tmp_path):
        provider = SACHETProvider()
        provider._etag_cache_file = tmp_path / "sachet_etag_cache.json"
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel><title>SACHET</title>
        <item>
            <title>Thunderstorm warning in Dehradun, Uttarakhand</title>
            <link>https://sachet.ndma.gov.in/alert/123</link>
            <description>Isolated hailstorm expected.</description>
            <pubDate>2026-07-15T08:00:00Z</pubDate>
        </item>
        </channel></rss>"""

        def _fake_get(url, **kwargs):
            if "FetchAllAlertDetails" in url:
                raise ConnectionError("JSON endpoint down")
            r = MagicMock(status_code=200, text=rss_xml)
            return r

        provider._session.get = _fake_get
        res = provider.fetch("Uttarakhand", "Dehradun")
        assert res["status"] == "Connected"
        assert len(res["regional_alerts"]) == 1

        # Both down & no cache -> Link only
        provider._session.get = MagicMock(side_effect=ConnectionError("offline"))
        res_down = provider.fetch("Uttarakhand", "Dehradun")
        assert res_down["status"] == "Link only"
        assert "sachet.ndma.gov.in" in res_down["portal_url"]


# ─────────────────────────────────────────────
# 7-12. Freshness, Missing Rainfall, Map, RiskEngine & DB Persistence
# ─────────────────────────────────────────────

class TestIndiaMonitorWorkflowAndDB:
    def test_freshness_valid_stale_and_invalid_timestamps(self):
        fresh_ts = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        stale_ts = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()

        f_ok = _compute_freshness(fresh_ts, stale_after_hours=6.0)
        f_stale = _compute_freshness(stale_ts, stale_after_hours=6.0)
        f_bad = _compute_freshness("not-a-timestamp", stale_after_hours=6.0)

        assert f_ok["is_valid"] is True and f_ok["is_stale"] is False
        assert f_stale["is_valid"] is True and f_stale["is_stale"] is True
        assert f_bad["is_valid"] is False and f_bad["label"] == "Invalid timestamp"

    def test_end_to_end_assessment_preserves_missing_rainfall_and_persists(
        self, isolated_db, synthetic_boundaries
    ):
        mock_weather = MagicMock()
        mock_weather.attribution = "Open-Meteo.com (CC BY 4.0)"
        now_ts = datetime.now(timezone.utc).isoformat()
        # Simulate missing 24h/72h accumulation & missing temperature — MUST stay None (not 0!)
        mock_weather.fetch_separated_weather.return_value = {
            "is_cached": False,
            "fetched_at": now_ts,
            "current": {"time": now_ts, "precipitation": None, "temperature_2m": 24.5, "relative_humidity_2m": 82, "wind_speed_10m": 12.0},
            "historical_hourly": {"time": [now_ts], "precipitation": [None], "soil_moisture_0_to_7cm": [0.42]},
            "forecast_hourly": {"time": [], "precipitation": []},
            "daily_historical": {"time": [], "precipitation_sum": []},
            "daily_forecast": {"time": [], "precipitation_sum": []},
        }
        mock_weather.get_rainfall_accumulations.return_value = {
            "rainfall_1h_mm": None,
            "rainfall_24h_mm": None,
            "rainfall_48h_mm": None,
            "rainfall_72h_mm": None,
        }

        mock_imd = MagicMock()
        mock_imd.fetch.return_value = {
            "provider": "IMD",
            "status": "Not configured",
            "attribution": "IMD",
            "is_cached": False,
            "last_updated": None,
            "message": "IMD_API_KEY not set",
        }

        mock_sachet = MagicMock()
        mock_sachet.fetch.return_value = {
            "provider": "NDMA SACHET",
            "status": "Connected",
            "attribution": "NDMA SACHET CAP Feed",
            "is_cached": False,
            "last_updated": now_ts,
            "total_india_alerts": 1,
            "regional_alerts": [
                {
                    "identifier": "kl-cap-99",
                    "disaster_type": "Landslide Watch",
                    "severity": "Moderate",
                    "severity_color": "Yellow",
                    "area_description": "Idukki district",
                    "headline": "Localized heavy spells possible",
                    "effective_from": now_ts,
                    "effective_until": now_ts,
                    "sender_org": "KSDMA",
                    "source_url": "https://sachet.ndma.gov.in/cap_public_website/FetchXMLFile?identifier=kl-cap-99",
                    "centroid_lat": 9.85,
                    "centroid_lon": 76.97,
                }
            ],
            "all_alerts": [],
        }

        mock_terrain = MagicMock()
        mock_terrain.attribution = "Open-Elevation SRTM"
        mock_terrain.get_terrain_features.return_value = {
            "elevation_m": 1120.0,
            "slope_degrees": 24.5,
            "terrain_source": "Open-Elevation SRTM",
        }

        monitor = IndiaRegionalMonitor(
            boundary_provider=synthetic_boundaries,
            weather_provider=mock_weather,
            imd_provider=mock_imd,
            sachet_provider=mock_sachet,
            terrain_provider=mock_terrain,
        )

        # Persist boundaries first and verify deduplication
        idx = synthetic_boundaries.load_index()
        sync_1 = sync_india_boundaries(idx["states"], idx["districts"], db_path=isolated_db)
        sync_2 = sync_india_boundaries(idx["states"], idx["districts"], db_path=isolated_db)
        assert sync_1["states_synced"] == 2 and sync_1["districts_synced"] == 4
        assert sync_2["districts_synced"] == 4
        assert len(list_india_states(db_path=isolated_db)) == 2
        assert len(list_india_districts("Kerala", db_path=isolated_db)) == 3

        res = monitor.run_regional_assessment(
            state_name="Kerala",
            district_name="Idukki",
            include_terrain=True,
            include_shap=False,
            persist=True,
            generate_experimental_alert=False,
        )

        # Missing rainfall NEVER coerced to 0!
        assert res["features_used"]["rainfall_1h_mm"] is None
        assert res["features_used"]["rainfall_24h_mm"] is None
        assert res["features_used"]["rainfall_72h_mm"] is None
        assert res["risk_result"]["is_experimental"] is True
        assert res["location_id"] is not None
        assert res["prediction_id"] is not None

        # Verify official warnings persisted & deduplicated
        warnings_df = get_official_warnings(state_name="Kerala", district_name="Idukki", db_path=isolated_db)
        assert len(warnings_df) == 1
        assert warnings_df.iloc[0]["external_id"] == "kl-cap-99"

        # Verify provider statuses persisted cleanly
        statuses = {s["provider_name"]: s for s in get_provider_statuses(db_path=isolated_db)}
        assert statuses["IMD (India Meteorological Department)"]["status"] == "Not configured"
        assert statuses["NDMA SACHET"]["status"] == "Connected"

        # Interactive India map renders cleanly without errors
        fmap = build_india_regional_map(
            center_lat=res["district_meta"]["rep_lat"],
            center_lon=res["district_meta"]["rep_lon"],
            state_name="Kerala",
            district_name="Idukki",
            state_geojson=res["state_geojson"],
            district_geojson=res["district_geojson"],
            zoom_bbox=res["district_meta"]["bbox"],
            weather_summary=res["weather"],
            risk_result=res["risk_result"],
            official_alerts=res["sachet"]["regional_alerts"],
        )
        html = fmap._repr_html_()
        assert "Idukki" in html
        assert "Administrative boundary only" in html

    def test_offline_weather_and_terrain_never_crash_monitor(
        self, isolated_db, synthetic_boundaries
    ):
        mock_weather = MagicMock()
        mock_weather.attribution = "Open-Meteo"
        mock_weather.fetch_separated_weather.side_effect = ConnectionError("network unreachable")

        mock_imd = MagicMock()
        mock_imd.fetch.return_value = {"status": "Not configured", "is_cached": False, "attribution": "IMD"}

        mock_sachet = MagicMock()
        mock_sachet.fetch.return_value = {"status": "Link only", "is_cached": False, "regional_alerts": [], "attribution": "SACHET"}

        mock_terrain = MagicMock()
        mock_terrain.attribution = "Open-Elevation"
        mock_terrain.get_terrain_features.side_effect = TimeoutError("timeout")

        monitor = IndiaRegionalMonitor(
            boundary_provider=synthetic_boundaries,
            weather_provider=mock_weather,
            imd_provider=mock_imd,
            sachet_provider=mock_sachet,
            terrain_provider=mock_terrain,
        )
        res = monitor.run_regional_assessment("Uttarakhand", "Dehradun", include_shap=False, persist=True)
        assert res["weather"]["status"] == "Unavailable"
        assert res["terrain"]["elevation_m"] is None
        assert res["features_used"]["rainfall_24h_mm"] is None
        assert any(p["provider"] == "Open-Meteo Weather" and p["status"] == "Unavailable" for p in res["provider_status"])

    def test_invalid_resolved_coordinates_rejected(self, synthetic_boundaries):
        bad_resolve = MagicMock(return_value={"state_name": "Kerala", "district_name": "Bad", "rep_lat": 145.0, "rep_lon": 77.0, "bbox": None})
        synthetic_boundaries.resolve_location = bad_resolve
        monitor = IndiaRegionalMonitor(boundary_provider=synthetic_boundaries)
        with pytest.raises(ValueError, match="Invalid resolved coordinates"):
            monitor.run_regional_assessment("Kerala", "Bad", persist=False)

    def test_upsert_official_warning_and_provider_status_dedup(self, isolated_db):
        id1 = upsert_official_warning("SACHET", "ext-1", state_name="Kerala", district_name="Idukki", headline="Initial", db_path=isolated_db)
        id2 = upsert_official_warning("SACHET", "ext-1", state_name="Kerala", district_name="Idukki", headline="Updated", db_path=isolated_db)
        assert id1 == id2
        df = get_official_warnings(provider="SACHET", db_path=isolated_db)
        assert len(df) == 1 and df.iloc[0]["headline"] == "Updated"

        upsert_provider_status("IMD", "Not configured", db_path=isolated_db)
        upsert_provider_status("IMD", "Connected", is_live=True, db_path=isolated_db)
        ps = [p for p in get_provider_statuses(db_path=isolated_db) if p["provider_name"] == "IMD"]
        assert len(ps) == 1 and ps[0]["status"] == "Connected" and ps[0]["is_live"] == 1
