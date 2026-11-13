"""
Emergency Route Visualization & Estimation Module (Phase 9).

Queries Open Source Routing Machine (OSRM) for road routes between monitoring points
and emergency facilities (hospitals, shelters), with graceful geodesic fallback
when offline or in remote terrain without road coverage.

MANDATORY SAFETY CONSTRAINT:
Routes are strictly informational and generated from static street maps. They do NOT
account for active landslide blockages, bridge failures, or police closures.
"""

from __future__ import annotations

import math
from typing import Any

import requests

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

OSRM_ROUTE_API = "https://router.project-osrm.org/route/v1/driving"

ROUTING_DISCLAIMER = (
    "INFORMATIONAL ROUTE ONLY: This route estimate is generated using open road network data. "
    "It does NOT account for active landslide blockages, washed-out bridges, rockfalls, "
    "mudflows, or emergency police road closures. Never attempt this route during an active "
    "disaster without prior clearance from local traffic police and disaster management authorities (NDRF/SDMA)."
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2.0) ** 2
    )
    return float(r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def calculate_emergency_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    facility_name: str | None = None,
    facility_category: str | None = None,
) -> dict[str, Any]:
    """Calculate driving route between two points using OSRM or geodesic fallback.

    Args:
        start_lat: Origin latitude.
        start_lon: Origin longitude.
        end_lat: Destination facility latitude.
        end_lon: Destination facility longitude.
        facility_name: Name of target facility.
        facility_category: Category (hospital, shelter, etc.).

    Returns:
        Structured dictionary containing route GeoJSON, distance in km,
        travel duration in minutes, routing engine provenance, and safety disclaimer.
    """
    direct_dist_km = _haversine_km(start_lat, start_lon, end_lat, end_lon)

    # 1. Attempt OSRM driving route API
    url = f"{OSRM_ROUTE_API}/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "LandslideGuard-Capstone/1.0"},
            timeout=3.5,
        )
        if resp.status_code == 200:
            data = resp.json()
            routes = data.get("routes", [])
            if routes:
                best = routes[0]
                dist_km = round(best.get("distance", 0.0) / 1000.0, 2)
                dur_min = round(best.get("duration", 0.0) / 60.0, 1)
                geometry = best.get("geometry", {})
                coords = geometry.get("coordinates", [])

                return {
                    "success": True,
                    "available": True,
                    "facility_name": facility_name or "Emergency Facility",
                    "facility_category": facility_category or "facility",
                    "origin": {"latitude": start_lat, "longitude": start_lon},
                    "destination": {"latitude": end_lat, "longitude": end_lon},
                    "distance_km": dist_km,
                    "estimated_duration_min": dur_min,
                    "duration_minutes": dur_min,
                    "route_geojson": {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": {
                            "facility_name": facility_name,
                            "distance_km": dist_km,
                            "duration_min": dur_min,
                        },
                    },
                    "geometry_geojson": geometry,
                    "waypoints_count": len(coords),
                    "routing_engine": "OSRM (Open Source Routing Machine)",
                    "is_road_network": True,
                    "is_fallback": False,
                    "is_estimated_fallback": False,
                    "disclaimer": ROUTING_DISCLAIMER,
                }
    except Exception as exc:
        logger.info("OSRM routing unavailable (%s); using geodesic emergency vector fallback.", exc)

    # 2. Geodesic Fallback: direct line with mountain road winding factor (1.35x) and average speed 30 km/h
    est_road_km = round(direct_dist_km * 1.35, 2)
    est_duration_min = round((est_road_km / 30.0) * 60.0, 1)
    fallback_geom = {
        "type": "LineString",
        "coordinates": [[start_lon, start_lat], [end_lon, end_lat]],
    }

    return {
        "success": True,
        "available": True,
        "facility_name": facility_name or "Emergency Facility",
        "facility_category": facility_category or "facility",
        "origin": {"latitude": start_lat, "longitude": start_lon},
        "destination": {"latitude": end_lat, "longitude": end_lon},
        "distance_km": est_road_km,
        "direct_distance_km": round(direct_dist_km, 2),
        "estimated_duration_min": est_duration_min,
        "duration_minutes": est_duration_min,
        "route_geojson": {
            "type": "Feature",
            "geometry": fallback_geom,
            "properties": {
                "facility_name": facility_name,
                "distance_km": est_road_km,
                "duration_min": est_duration_min,
                "note": "Direct geodesic vector (road network offline)",
            },
        },
        "geometry_geojson": fallback_geom,
        "waypoints_count": 2,
        "routing_engine": "Geodesic Great-Circle Fallback (Winding Factor 1.35x)",
        "is_road_network": False,
        "is_fallback": True,
        "is_estimated_fallback": True,
        "disclaimer": ROUTING_DISCLAIMER,
    }
