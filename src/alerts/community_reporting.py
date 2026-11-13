"""
Community Incident Reporting Module (Phase 9).

Allows residents, local volunteers, and field observers to submit crowd-sourced
landslide hazard reports:
- Incident description and visual category (Rockfall, Mudflow, Tension Cracks, Road Subsidence).
- Location name, district, and coordinates.
- Timestamp of observation.
- Optional uploaded photograph.

AUDIT & VERIFICATION POLICY:
Community submissions enter the system with status 'submitted'. They are subject to
administrative review ('under_review' -> 'verified' / 'rejected') and must NEVER
automatically merge into authentic scientific training datasets or official warning logs.
"""

from __future__ import annotations

import base64
from typing import Any

from src.database.models import (
    insert_community_report,
    update_community_report_status,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

REPORT_CATEGORIES: list[str] = [
    "Rockfall",
    "Debris Flow / Mudflow",
    "Tension Cracks",
    "Road Subsidence / Slope Slump",
    "Blocked Drainage",
    "Other Hazard",
]

REPORT_SEVERITIES: list[str] = ["low", "moderate", "high", "critical"]
REPORT_STATUSES: list[str] = ["submitted", "under_review", "verified", "rejected"]

COMMUNITY_DATA_NOTICE: str = (
    "COMMUNITY DATA NOTICE: Community hazard reports are crowd-sourced field observations. "
    "They undergo administrative review and do NOT automatically constitute verified historical "
    "landslide catalog records, nor do they independently trigger official evacuation warnings."
)


def validate_report_submission(
    location_name: str,
    latitude: float,
    longitude: float,
    incident_date: str,
    category: str,
    description: str,
    photo_bytes: bytes | None = None,
) -> tuple[bool, str | None]:
    """Validate report submission parameters before saving to database."""
    if not location_name or not location_name.strip():
        return False, "Location name cannot be empty."

    try:
        lat = float(latitude)
        lon = float(longitude)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return False, f"Coordinates ({lat}, {lon}) are out of valid geographic range [-90..90, -180..180]."
    except (ValueError, TypeError):
        return False, "Invalid coordinate format; latitude and longitude must be numeric."

    if not incident_date or not incident_date.strip():
        return False, "Incident date and time is required."

    if category not in REPORT_CATEGORIES:
        return False, f"Invalid category '{category}'. Allowed: {', '.join(REPORT_CATEGORIES)}."

    if not description or len(description.strip()) < 10:
        return False, "Description must be at least 10 characters detailing the observed hazard."

    if photo_bytes is not None and len(photo_bytes) > 10 * 1024 * 1024:
        return False, "Attached photograph exceeds 10 MB maximum upload limit."

    return True, None


def submit_community_report(
    location_name: str,
    latitude: float,
    longitude: float,
    incident_date: str,
    category: str,
    description: str,
    reporter_name: str | None = None,
    reporter_contact: str | None = None,
    state_name: str | None = None,
    district_name: str | None = None,
    severity: str = "moderate",
    photo_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Validate and insert a new community hazard report."""
    valid, err = validate_report_submission(
        location_name, latitude, longitude, incident_date, category, description, photo_bytes
    )
    if not valid:
        return {"success": False, "error": err, "report_id": None}

    photo_b64: str | None = None
    if photo_bytes:
        photo_b64 = base64.b64encode(photo_bytes).decode("ascii")

    try:
        report_id = insert_community_report(
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            incident_date=incident_date,
            category=category,
            description=description,
            reporter_name=reporter_name,
            reporter_contact=reporter_contact,
            state_name=state_name,
            district_name=district_name,
            severity=severity,
            photo_base64=photo_b64,
        )
        return {
            "success": True,
            "report_id": report_id,
            "status": "submitted",
            "message": f"Community report #{report_id} successfully submitted and queued for review.",
            "notice": COMMUNITY_DATA_NOTICE,
        }
    except Exception as exc:
        logger.error("Failed to insert community report: %s", exc)
        return {"success": False, "error": str(exc), "report_id": None}


def review_community_report(
    report_id: int,
    new_status: str,
    review_notes: str | None = None,
    reviewed_by: str = "admin",
) -> dict[str, Any]:
    """Update report review status ('submitted', 'under_review', 'verified', 'rejected')."""
    s = new_status.lower().strip()
    if s not in REPORT_STATUSES:
        return {"success": False, "error": f"Invalid status '{new_status}'. Allowed: {REPORT_STATUSES}"}

    try:
        ok = update_community_report_status(
            report_id=report_id,
            status=s,
            review_notes=review_notes,
            reviewed_by=reviewed_by,
        )
        if not ok:
            return {"success": False, "error": f"Report #{report_id} not found."}
        return {
            "success": True,
            "report_id": report_id,
            "new_status": s,
            "reviewed_by": reviewed_by,
            "message": f"Report #{report_id} status updated to '{s}'.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}

