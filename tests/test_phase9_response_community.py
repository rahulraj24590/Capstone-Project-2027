"""
Test Suite for Phase 9: Disaster Response & Community Intelligence.

Tests:
1. Emergency Facilities Mapping & Curated Fallback (Wayanad, Chamoli, Darjeeling, etc.)
2. Emergency Route Calculation (OSRM with Geodesic Fallback & Mandatory Safety Disclaimers)
3. Crowd-Sourced Community Incident Reporting (Validation, Review Workflow, Strict Data Segregation)
4. Historical Event Replay (Retrospective Assessment, Interval Data Refusal, Anti-Hallucination)
5. Automated PDF Hazard Assessment Report Generation (ReportLab, Styling, Disclaimers)
6. Offline Synchronization & Telemetry Cache (Freshness Badges, Non-Live Cached Enforcement)
7. Authentication & Role-Based Access Control (Admin vs. Viewer Permissions, HMAC Verification)
8. Phase 9 FastAPI REST API Endpoints.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.alerts.community_reporting import (
    COMMUNITY_DATA_NOTICE,
    REPORT_CATEGORIES,
    REPORT_STATUSES,
    submit_community_report,
    review_community_report,
    validate_report_submission,
)
from src.database.db import get_table_counts
from src.database.models import (
    get_community_reports,
    get_community_report_by_id,
)
from src.geospatial.facilities import (
    CURATED_EMERGENCY_FACILITIES,
    fetch_nearby_facilities,
    haversine_distance_km,
)
from src.geospatial.routing import (
    ROUTING_DISCLAIMER,
    calculate_emergency_route,
    _haversine_km,
)
from src.monitoring.event_replay import (
    HISTORICAL_REPLAY_DISCLAIMER,
    list_replayable_events,
    replay_historical_event,
)
from src.monitoring.offline_cache import OfflineSyncCache, get_offline_cache
from src.utils.auth import (
    ADMIN_API_KEY,
    DEFAULT_ADMIN_USER,
    authenticate_user,
    verify_admin_key,
    can_manage_locations,
    can_review_community_reports,
    can_manage_alerts,
)
from src.utils.pdf_generator import (
    PDF_DISCLAIMER_TEXT,
    generate_assessment_pdf_report,
)


def _run_coro_sync(coro):
    """Run an async coroutine synchronously."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise RuntimeError("Coroutine did not complete synchronously.")


# ─────────────────────────────────────────────────────────────
# 1. Emergency Facilities Mapping Tests
# ─────────────────────────────────────────────────────────────

class TestEmergencyFacilities:
    def test_haversine_distance_accuracy(self):
        # Distance between Munnar (10.0889, 77.0595) and Devikulam (10.0630, 77.1020) ~ 5.5 km
        d = haversine_distance_km(10.0889, 77.0595, 10.0630, 77.1020)
        assert 4.0 < d < 7.0

    def test_fetch_nearby_facilities_curated_fallback(self):
        # Using Wayanad coordinates (11.6080, 76.0820)
        # Mock requests.get to force fallback to curated database
        with patch("requests.get", side_effect=Exception("Network unreachable")):
            res = fetch_nearby_facilities(11.6080, 76.0820, radius_km=30.0, max_results=10)

        assert res["available"] is True
        assert "Curated Indian Emergency Facilities" in res["data_source"]
        assert len(res["facilities"]) >= 3
        assert "nearest" in res
        assert "hospital" in res["nearest"]
        assert "police" in res["nearest"]
        assert "disclaimer" in res
        assert "NDRF/SDMA" in res["disclaimer"]

    def test_facility_categorization_and_sorting(self):
        # Chamoli / Joshimath coordinates (30.5564, 79.5667)
        with patch("requests.get", side_effect=Exception("Timeout")):
            res = fetch_nearby_facilities(30.5564, 79.5667, radius_km=50.0)

        assert res["available"] is True
        facs = res["facilities"]
        # Distances should be sorted ascending
        distances = [f["distance_km"] for f in facs]
        assert distances == sorted(distances)

        categories = {f["category"] for f in facs}
        assert "hospital" in categories or "shelter" in categories

    def test_empty_facilities_for_ocean_coordinates(self):
        # Point in middle of the Indian Ocean
        with patch("requests.get", side_effect=Exception("Offline")):
            res = fetch_nearby_facilities(-10.0, 70.0, radius_km=10.0)
        assert res["available"] is True
        assert len(res["facilities"]) == 0
        assert res["counts"]["total"] == 0


# ─────────────────────────────────────────────────────────────
# 2. Emergency Routing & Safety Constraints Tests
# ─────────────────────────────────────────────────────────────

class TestEmergencyRouting:
    def test_routing_disclaimer_mandatory_text(self):
        assert "INFORMATIONAL ROUTE ONLY" in ROUTING_DISCLAIMER
        assert "landslide blockages" in ROUTING_DISCLAIMER
        assert "NDRF/SDMA" in ROUTING_DISCLAIMER

    def test_geodesic_fallback_when_osrm_unreachable(self):
        # Start at Meppadi (11.5510, 76.1260) to Wayanad Memorial Hospital (11.6080, 76.0820)
        with patch("requests.get", side_effect=Exception("OSRM server down")):
            route = calculate_emergency_route(
                start_lat=11.5510,
                start_lon=76.1260,
                end_lat=11.6080,
                end_lon=76.0820,
                facility_name="Wayanad District Hospital",
                facility_category="hospital",
            )

        assert route["success"] is True
        assert route["is_fallback"] is True
        assert "Geodesic" in route["routing_engine"]
        assert route["distance_km"] > 0
        assert route["duration_minutes"] > 0
        assert route["disclaimer"] == ROUTING_DISCLAIMER
        assert "geometry_geojson" in route
        coords = route["geometry_geojson"]["coordinates"]
        assert len(coords) >= 2
        # Start coordinate lon, lat
        assert math.isclose(coords[0][0], 76.1260, abs_tol=1e-3)
        assert math.isclose(coords[0][1], 11.5510, abs_tol=1e-3)

    def test_osrm_response_parsing(self):
        # Mock successful OSRM response
        fake_osrm_payload = {
            "routes": [
                {
                    "distance": 14200.0,  # 14.2 km
                    "duration": 1800.0,   # 30 min
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[76.126, 11.551], [76.100, 11.580], [76.082, 11.608]],
                    },
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_osrm_payload

        with patch("requests.get", return_value=mock_resp):
            route = calculate_emergency_route(11.551, 76.126, 11.608, 76.082)

        assert route["success"] is True
        assert route["is_fallback"] is False
        assert route["distance_km"] == 14.2
        assert route["duration_minutes"] == 30.0
        assert "OSRM" in route["routing_engine"]


# ─────────────────────────────────────────────────────────────
# 3. Crowd-Sourced Community Incident Reporting Tests
# ─────────────────────────────────────────────────────────────

class TestCommunityIncidentReporting:
    def test_validation_rules(self):
        # Empty location
        v, err = validate_report_submission("", 11.5, 76.0, "2026-11-13", "Rockfall", "Rockfall blocking road")
        assert v is False
        assert "Location name cannot be empty" in err

        # Invalid coordinates
        v, err = validate_report_submission("Pass", 120.0, 76.0, "2026-11-13", "Rockfall", "Rockfall blocking road")
        assert v is False
        assert "out of valid geographic range" in err

        # Invalid category
        v, err = validate_report_submission("Pass", 11.5, 76.0, "2026-11-13", "Alien Invasion", "Massive debris sliding")
        assert v is False
        assert "Invalid category" in err

        # Description too short
        v, err = validate_report_submission("Pass", 11.5, 76.0, "2026-11-13", "Rockfall", "Rocks")
        assert v is False
        assert "at least 10 characters" in err

        # Oversized photo (> 10 MB)
        huge_bytes = b"0" * (10 * 1024 * 1024 + 10)
        v, err = validate_report_submission("Pass", 11.5, 76.0, "2026-11-13", "Rockfall", "Massive rockfall on highway", huge_bytes)
        assert v is False
        assert "exceeds 10 MB" in err

        # Valid payload
        v, err = validate_report_submission("Munnar Ghat", 10.08, 77.06, "2026-11-13 14:00", "Rockfall", "Boulder rolling down slope onto NH-85")
        assert v is True
        assert err is None

    def test_submit_and_strict_database_segregation(self):
        # Verify that submitting community reports does NOT pollute the scientific observations table
        init_counts = get_table_counts()
        obs_before = init_counts.get("observations", 0)

        res = submit_community_report(
            location_name="Wayanad Test Ghat Road",
            latitude=11.5510,
            longitude=76.1260,
            incident_date="2026-11-13 15:30",
            category="Debris Flow / Mudflow",
            description="Active mud slurry and minor soil displacement observed along tea estate road.",
            reporter_name="Local Field Observer",
            reporter_contact="+91-9988776655",
            state_name="Kerala",
            district_name="Wayanad",
            severity="moderate",
        )

        assert res["success"] is True
        report_id = res["report_id"]
        assert report_id is not None
        assert res["status"] == "submitted"
        assert "COMMUNITY DATA NOTICE" in res["notice"]

        # Invariant: observations table count must remain strictly unchanged!
        post_counts = get_table_counts()
        assert post_counts.get("observations") == obs_before

        # Community report should exist in community_reports table
        report = get_community_report_by_id(report_id)
        assert report is not None
        assert report["location_name"] == "Wayanad Test Ghat Road"
        assert report["category"] == "Debris Flow / Mudflow"
        assert report["status"] == "submitted"

    def test_report_review_lifecycle(self):
        # Create a report for review
        res = submit_community_report(
            location_name="Kalimpong Ridge Road",
            latitude=27.0660,
            longitude=88.4720,
            incident_date="2026-11-13 10:00",
            category="Tension Cracks",
            description="Crown tension cracks extending 15 meters along the shoulder.",
            reporter_name="Geotech Volunteer",
        )
        report_id = res["report_id"]

        # 1. Invalid status update
        rev_bad = review_community_report(report_id, "approved")  # 'approved' is not allowed
        assert rev_bad["success"] is False
        assert "Invalid status" in rev_bad["error"]

        # 2. Transition: submitted -> under_review
        rev1 = review_community_report(report_id, "under_review", review_notes="Dispatching local survey team")
        assert rev1["success"] is True
        assert rev1["new_status"] == "under_review"

        r1 = get_community_report_by_id(report_id)
        assert r1["status"] == "under_review"
        assert "Dispatching" in r1["review_notes"]

        # 3. Transition: under_review -> verified
        rev2 = review_community_report(report_id, "verified", review_notes="Confirmed by field inspector", reviewed_by="Dr. Sharma")
        assert rev2["success"] is True
        assert rev2["new_status"] == "verified"

        r2 = get_community_report_by_id(report_id)
        assert r2["status"] == "verified"
        assert r2["reviewed_by"] == "Dr. Sharma"


# ─────────────────────────────────────────────────────────────
# 4. Historical Event Replay Tests
# ─────────────────────────────────────────────────────────────

class TestHistoricalEventReplay:
    def test_list_replayable_events(self):
        events = list_replayable_events(limit=10)
        assert isinstance(events, list)
        assert len(events) > 0
        ev0 = events[0]
        assert "id" in ev0
        assert "event_date" in ev0
        assert "latitude" in ev0
        assert "longitude" in ev0

    def test_replay_nonexistent_event(self):
        res = replay_historical_event(99999999)
        assert res["available"] is False
        assert "not found" in res["error"]
        assert res["disclaimer"] == HISTORICAL_REPLAY_DISCLAIMER

    def test_replay_valid_event_and_anti_hallucination_rule(self):
        events = list_replayable_events(limit=1)
        target_id = events[0]["id"]

        res = replay_historical_event(target_id)
        assert res["available"] is True
        assert "event_profile" in res
        assert "rainfall_timeline" in res
        assert "retrospective_assessment" in res
        assert res["disclaimer"] == HISTORICAL_REPLAY_DISCLAIMER

        # Check anti-hallucination guard: If interval telemetry doesn't exist,
        # status must explicitly declare it unavailable and never fabricate numbers!
        tl = res["rainfall_timeline"]
        if not tl["has_hourly_series"]:
            assert "Unavailable" in tl["status"]
            assert len(tl["data_points"]) == 0

        # Retrospective evaluation notice check
        retro = res["retrospective_assessment"]
        if retro.get("evaluation_status") == "Success":
            assert "Retrospective score evaluated" in retro["retrospective_notice"]
        else:
            assert "Unavailable" in retro.get("evaluation_status") or "Failed" in retro.get("evaluation_status")


# ─────────────────────────────────────────────────────────────
# 5. Automated PDF Assessment Report Generator Tests
# ─────────────────────────────────────────────────────────────

class TestAssessmentPDFReport:
    def test_generate_pdf_with_minimal_profile(self):
        loc = {
            "name": "Kottayam Hills",
            "latitude": 9.5915,
            "longitude": 76.5222,
            "elevation_m": 420.0,
            "slope_degrees": 18.5,
            "state_region": "Kerala",
        }
        pdf_bytes = generate_assessment_pdf_report(location_profile=loc)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF-")

    def test_generate_pdf_comprehensive_sections(self):
        loc = {
            "name": "Wayanad Highland Monitoring Station",
            "latitude": 11.6080,
            "longitude": 76.0820,
            "elevation_m": 780.0,
            "slope_degrees": 28.5,
            "state_region": "Kerala",
        }
        telemetry = {
            "rainfall_1h_mm": 12.4,
            "rainfall_24h_mm": 115.8,
            "rainfall_72h_mm": 210.5,
            "temperature_c": 21.2,
            "relative_humidity_pct": 94.0,
            "soil_moisture": 0.42,
        }
        risk = {
            "risk_category": "High",
            "probability": 0.84,
            "model_name": "Random Forest Ensemble",
            "is_experimental": True,
        }
        facilities = [
            {"name": "Wayanad Memorial Hospital", "category": "hospital", "distance_km": 3.2, "phone": "04936-202245"},
            {"name": "Kalpetta Fire Station", "category": "fire_station", "distance_km": 4.1, "phone": "101"},
        ]
        history = [
            {"event_date": "2024-07-30", "trigger": "Heavy Monsoon Downpour", "landslide_type": "Debris Flow", "fatalities": 350},
        ]

        pdf_bytes = generate_assessment_pdf_report(
            location_profile=loc,
            weather_telemetry=telemetry,
            risk_assessment=risk,
            historical_events=history,
            nearby_facilities=facilities,
            is_synthetic_benchmark=False,
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 2500
        assert pdf_bytes.startswith(b"%PDF-")
        # Ensure EOF marker exists
        assert b"%%EOF" in pdf_bytes[-1024:]


# ─────────────────────────────────────────────────────────────
# 6. Offline Synchronization & Cache Tests
# ─────────────────────────────────────────────────────────────

class TestOfflineSyncCache:
    def test_offline_cache_put_get_and_live_flag(self, tmp_path):
        cache = OfflineSyncCache(cache_dir=tmp_path)
        payload = {"temperature": 24.5, "rainfall_24h": 45.0, "station": "Nilgiris-01"}

        cache.put("station_nilgiris", payload, provider_name="Open-Meteo Telemetry")

        retrieved = cache.get("station_nilgiris")
        assert retrieved is not None
        assert retrieved["data"]["temperature"] == 24.5
        assert retrieved["data"]["station"] == "Nilgiris-01"
        assert retrieved["is_live"] is False
        assert retrieved["from_offline_cache"] is True
        assert "last_successful_sync" in retrieved

    def test_offline_cache_status_summary(self, tmp_path):
        cache = OfflineSyncCache(cache_dir=tmp_path)
        cache.put("station_1", {"metric": 10})
        cache.put("station_2", {"metric": 20})

        status = cache.get_offline_status()
        assert status["offline_cache_enabled"] is True
        assert status["total_cached_records"] == 2
        assert status["last_successful_synchronization"] != "Unavailable"
        assert "Never present cached information as current live data" in status["integrity_rule"]


# ─────────────────────────────────────────────────────────────
# 7. Authentication & Role-Based Access Control Tests
# ─────────────────────────────────────────────────────────────

class TestAuthAndRBAC:
    def test_admin_authentication(self):
        ok, role, err = authenticate_user(DEFAULT_ADMIN_USER, "landslideguard2026")
        assert ok is True
        assert role == "admin"
        assert err is None

    def test_viewer_authentication(self):
        ok, role, err = authenticate_user("viewer", "guest")
        assert ok is True
        assert role == "viewer"
        assert err is None

    def test_invalid_credentials(self):
        ok, role, err = authenticate_user("admin", "wrongpassword")
        assert ok is False
        assert role is None
        assert "Invalid username or password" in err

    def test_admin_api_key_verification(self):
        assert verify_admin_key(ADMIN_API_KEY) is True
        assert verify_admin_key(f"Bearer {ADMIN_API_KEY}") is True
        assert verify_admin_key("invalid-secret-token") is False
        assert verify_admin_key(None) is False

    def test_rbac_permission_functions(self):
        assert can_manage_locations("admin") is True
        assert can_manage_locations("viewer") is False
        assert can_manage_locations(None) is False

        assert can_review_community_reports("admin") is True
        assert can_review_community_reports("viewer") is False

        assert can_manage_alerts("admin") is True
        assert can_manage_alerts("viewer") is False


# ─────────────────────────────────────────────────────────────
# 8. Phase 9 FastAPI REST API Endpoints Tests
# ─────────────────────────────────────────────────────────────

class TestPhase9FastAPIEndpoints:
    def test_emergency_facilities_endpoint(self):
        from api import get_emergency_facilities

        with patch("requests.get", side_effect=Exception("Offline")):
            res = _run_coro_sync(get_emergency_facilities(latitude=11.6080, longitude=76.0820, radius_km=30.0))

        assert res["available"] is True
        assert "facilities" in res
        assert len(res["facilities"]) >= 3
        assert "nearest" in res

    def test_emergency_route_endpoint(self):
        from api import get_emergency_route, RouteRequest

        with patch("requests.get", side_effect=Exception("Offline")):
            req = RouteRequest(start_lat=11.5510, start_lon=76.1260, end_lat=11.6080, end_lon=76.0820)
            res = _run_coro_sync(get_emergency_route(req))

        assert res["success"] is True
        assert res["is_fallback"] is True
        assert res["disclaimer"] == ROUTING_DISCLAIMER

    def test_community_reports_post_and_list_endpoints(self):
        from api import (
            post_community_report,
            list_community_reports,
            CommunityReportRequest,
        )

        req = CommunityReportRequest(
            location_name="Api Test Pass",
            latitude=30.3240,
            longitude=78.0410,
            incident_date="2026-11-13 12:00",
            category="Road Subsidence / Slope Slump",
            description="Shoulder erosion and asphalt cracking along Dehradun-Mussoorie road.",
            reporter_name="Field Patrol",
            severity="high",
        )
        post_res = _run_coro_sync(post_community_report(req))
        assert post_res["success"] is True
        rep_id = post_res["report_id"]

        list_res = _run_coro_sync(list_community_reports(status=None, limit=50))
        assert list_res["total_reports"] >= 1
        ids = [r["id"] for r in list_res["reports"]]
        assert rep_id in ids

    def test_admin_community_report_review_endpoint(self):
        from api import (
            admin_review_community_report,
            ReportReviewRequest,
        )
        from fastapi import HTTPException

        # 1. Reject without admin key
        req = ReportReviewRequest(status="verified", review_notes="Field verified", reviewed_by="Admin Officer")
        with pytest.raises(HTTPException) as exc_info:
            _run_coro_sync(admin_review_community_report(report_id=1, request=req, x_admin_key=None))
        assert exc_info.value.status_code == 401

        # 2. Reject with invalid admin key
        with pytest.raises(HTTPException) as exc_info:
            _run_coro_sync(admin_review_community_report(report_id=1, request=req, x_admin_key="wrong-key"))
        assert exc_info.value.status_code == 401

        # 3. Accept with valid admin key
        res = _run_coro_sync(admin_review_community_report(report_id=1, request=req, x_admin_key=ADMIN_API_KEY))
        assert res["success"] is True
        assert res["new_status"] == "verified"

    def test_download_location_pdf_report_endpoint(self):
        from api import download_location_pdf_report

        res = _run_coro_sync(download_location_pdf_report(location_id=1))
        assert res.media_type == "application/pdf"
        assert res.body.startswith(b"%PDF-")
        assert len(res.body) > 1000

    def test_offline_monitoring_status_endpoint(self):
        from api import get_offline_monitoring_status

        res = _run_coro_sync(get_offline_monitoring_status())
        assert res["offline_cache_enabled"] is True
        assert "integrity_rule" in res
