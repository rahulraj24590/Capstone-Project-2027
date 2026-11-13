"""
Database CRUD operations for LandslideGuard.

Provides typed helper functions for inserting and querying records.
All database access goes through the db_context context manager.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.database.db import db_context, DB_PATH
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Observations
# ─────────────────────────────────────────────

def insert_observation(
    latitude: float,
    longitude: float,
    event_date: str | None = None,
    landslide_type: str | None = None,
    trigger: str | None = None,
    country: str | None = None,
    state_region: str | None = None,
    source_name: str | None = None,
    fatalities: int | None = None,
    is_synthetic: bool = False,
) -> int:
    """Insert a landslide observation record.

    Args:
        latitude: Event latitude.
        longitude: Event longitude.
        event_date: ISO date string.
        landslide_type: Type classification (e.g., 'debris flow').
        trigger: Triggering factor (e.g., 'rain').
        country: Country name.
        state_region: State or region name.
        source_name: Data source name.
        fatalities: Number of fatalities (if known).
        is_synthetic: True if this is synthetic test data.

    Returns:
        Row ID of inserted record.
    """
    with db_context() as conn:
        cursor = conn.execute(
            """
            INSERT INTO observations
                (event_date, latitude, longitude, landslide_type, trigger,
                 fatalities, country, state_region, source_name, is_synthetic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_date, latitude, longitude, landslide_type, trigger,
             fatalities, country, state_region, source_name, int(is_synthetic)),
        )
        return cursor.lastrowid


def get_observations(
    limit: int = 500,
    include_synthetic: bool = True,
) -> pd.DataFrame:
    """Retrieve historical observations.

    Args:
        limit: Maximum number of rows to return.
        include_synthetic: If False, exclude synthetic records.

    Returns:
        DataFrame of observations.
    """
    with db_context() as conn:
        query = "SELECT * FROM observations"
        if not include_synthetic:
            query += " WHERE is_synthetic = 0"
        query += " ORDER BY event_date DESC LIMIT ?"
        rows = conn.execute(query, (int(limit),)).fetchall()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def bulk_insert_observations(df: pd.DataFrame, is_synthetic: bool) -> int:
    """Bulk insert observations from a DataFrame.

    Args:
        df: DataFrame with observation columns.
        is_synthetic: Mark all records as synthetic.

    Returns:
        Number of records inserted.
    """
    records = []
    for _, row in df.iterrows():
        records.append((
            str(row.get("event_date", "")) or None,
            float(row["latitude"]),
            float(row["longitude"]),
            str(row.get("landslide_type", "")) or None,
            str(row.get("trigger", "")) or None,
            int(row["fatalities"]) if pd.notna(row.get("fatalities")) else None,
            str(row.get("country", "")) or None,
            str(row.get("state_region", "")) or None,
            str(row.get("source_name", "")) or None,
            int(is_synthetic),
        ))

    with db_context() as conn:
        conn.executemany(
            """
            INSERT INTO observations
                (event_date, latitude, longitude, landslide_type, trigger,
                 fatalities, country, state_region, source_name, is_synthetic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
    logger.info("Inserted %d observations (synthetic=%s).", len(records), is_synthetic)
    return len(records)


# ─────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────

def save_prediction(
    probability: float | None,
    risk_category: str,
    model_name: str,
    method: str,
    input_features: dict,
    warnings: list[str] | None = None,
    location_id: int | None = None,
    is_experimental: bool = True,
) -> int:
    """Save a prediction result to the database.

    Args:
        probability: Estimated probability.
        risk_category: Risk category string.
        model_name: Model identifier.
        method: 'ml_model' or 'heuristic_baseline'.
        input_features: Dict of feature values used.
        warnings: List of warning messages.
        location_id: Optional linked location ID.
        is_experimental: Always True until independently validated.

    Returns:
        Prediction record ID.
    """
    with db_context() as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_records
                (location_id, predicted_at, probability, risk_category,
                 model_name, method, is_experimental, input_features, warnings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location_id,
                datetime.now(timezone.utc).isoformat(),
                probability,
                risk_category,
                model_name,
                method,
                int(is_experimental),
                json.dumps(input_features),
                json.dumps(warnings or []),
            ),
        )
        return cursor.lastrowid


# ─────────────────────────────────────────────
# Alerts
# ─────────────────────────────────────────────

def create_alert(
    risk_category: str,
    probability: float | None,
    location_id: int | None = None,
    prediction_id: int | None = None,
    cooldown_minutes: int = 120,
    location_name: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    supporting_env_info: dict | str | None = None,
    model_version: str | None = None,
    data_source: str | None = None,
    status: str = "new",
    escalated_from: str | None = None,
    created_at: datetime | None = None,
) -> int:
    """Create a new experimental alert record inside an atomic transaction."""
    from datetime import timedelta

    now = created_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_iso = now.isoformat()
    cooldown_ends = (now + timedelta(minutes=int(cooldown_minutes))).isoformat()

    if supporting_env_info is None:
        env_json = None
    elif isinstance(supporting_env_info, str):
        env_json = supporting_env_info
    else:
        env_json = json.dumps(supporting_env_info, default=str)

    with db_context() as conn:
        # Backfill location details from locations table when location_id is linked
        if location_id is not None and (location_name is None or latitude is None or longitude is None):
            loc_row = conn.execute(
                "SELECT name, latitude, longitude FROM locations WHERE id = ?",
                (int(location_id),),
            ).fetchone()
            if loc_row:
                location_name = location_name or loc_row["name"]
                if latitude is None:
                    latitude = loc_row["latitude"]
                if longitude is None:
                    longitude = loc_row["longitude"]

        cursor = conn.execute(
            """
            INSERT INTO alerts
                (location_id, prediction_id, location_name, latitude, longitude,
                 risk_category, probability, supporting_env_info, model_version,
                 data_source, status, created_at, is_experimental, escalated_from,
                 cooldown_ends_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                location_id,
                prediction_id,
                location_name,
                float(latitude) if latitude is not None else None,
                float(longitude) if longitude is not None else None,
                risk_category,
                probability,
                env_json,
                model_version,
                data_source,
                status,
                now_iso,
                escalated_from,
                cooldown_ends,
            ),
        )
        return cursor.lastrowid


def get_alert_by_id(alert_id: int) -> dict | None:
    """Retrieve a single alert record joined with its location metadata."""
    with db_context() as conn:
        row = conn.execute(
            """
            SELECT a.*,
                   COALESCE(a.location_name, l.name) AS resolved_location_name,
                   COALESCE(a.latitude, l.latitude) AS resolved_latitude,
                   COALESCE(a.longitude, l.longitude) AS resolved_longitude
            FROM alerts a
            LEFT JOIN locations l ON l.id = a.location_id
            WHERE a.id = ?
            """,
            (int(alert_id),),
        ).fetchone()
    if not row:
        return None
    rec = dict(row)
    rec["location_name"] = rec.pop("resolved_location_name", None)
    rec["latitude"] = rec.pop("resolved_latitude", None)
    rec["longitude"] = rec.pop("resolved_longitude", None)
    return rec


def get_alerts(
    include_acknowledged: bool = False,
    limit: int = 100,
    include_resolved: bool = True,
    status: str | None = None,
    risk_category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    location_id: int | None = None,
) -> pd.DataFrame:
    """Retrieve alerts from the database with optional lifecycle, category, and date filters."""
    clauses: list[str] = []
    params: list[Any] = []

    if status is not None:
        clauses.append("a.status = ?")
        params.append(status)
    else:
        if not include_acknowledged:
            clauses.append("a.is_acknowledged = 0 AND a.is_resolved = 0")
        elif not include_resolved:
            clauses.append("a.is_resolved = 0")

    if risk_category is not None:
        clauses.append("a.risk_category = ?")
        params.append(risk_category)

    if start_date is not None:
        clauses.append("substr(a.created_at, 1, 10) >= ?")
        params.append(start_date[:10])

    if end_date is not None:
        clauses.append("substr(a.created_at, 1, 10) <= ?")
        params.append(end_date[:10])

    if location_id is not None:
        clauses.append("a.location_id = ?")
        params.append(int(location_id))

    query = """
        SELECT a.*,
               COALESCE(a.location_name, l.name, 'Unlinked Location') AS resolved_location_name,
               COALESCE(a.latitude, l.latitude) AS resolved_latitude,
               COALESCE(a.longitude, l.longitude) AS resolved_longitude
        FROM alerts a
        LEFT JOIN locations l ON l.id = a.location_id
    """
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY a.created_at DESC, a.id DESC LIMIT ?"
    params.append(int(limit))

    with db_context() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()

    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        rec = dict(r)
        rec["location_name"] = rec.pop("resolved_location_name", None)
        rec["latitude"] = rec.pop("resolved_latitude", None)
        rec["longitude"] = rec.pop("resolved_longitude", None)
        records.append(rec)
    return pd.DataFrame(records)


def acknowledge_alert(alert_id: int, note: str | None = None) -> bool:
    """Mark an alert as acknowledged and record an audit timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    with db_context() as conn:
        cursor = conn.execute(
            """
            UPDATE alerts
            SET is_acknowledged = 1,
                acknowledged_at = ?,
                status = CASE WHEN is_resolved = 1 THEN 'resolved' ELSE 'acknowledged' END
            WHERE id = ?
            """,
            (now, int(alert_id)),
        )
        if cursor.rowcount == 0:
            logger.warning("Alert %d not found; cannot acknowledge.", alert_id)
            return False
        conn.execute(
            "INSERT INTO alert_acknowledgements (alert_id, action, acknowledged_at, note) VALUES (?, 'acknowledged', ?, ?)",
            (int(alert_id), now, note),
        )
    logger.info("Alert %d acknowledged.", alert_id)
    return True


def resolve_alert(alert_id: int, note: str | None = None) -> bool:
    """Mark an alert as resolved and record an audit timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    with db_context() as conn:
        cursor = conn.execute(
            """
            UPDATE alerts
            SET is_resolved = 1,
                resolved_at = ?,
                resolution_note = ?,
                is_acknowledged = 1,
                acknowledged_at = COALESCE(acknowledged_at, ?),
                status = 'resolved'
            WHERE id = ?
            """,
            (now, note, now, int(alert_id)),
        )
        if cursor.rowcount == 0:
            logger.warning("Alert %d not found; cannot resolve.", alert_id)
            return False
        conn.execute(
            "INSERT INTO alert_acknowledgements (alert_id, action, acknowledged_at, note) VALUES (?, 'resolved', ?, ?)",
            (int(alert_id), now, note),
        )
    logger.info("Alert %d resolved.", alert_id)
    return True


def get_alert_audit_log(alert_id: int | None = None, limit: int = 100) -> pd.DataFrame:
    """Retrieve acknowledgement and resolution audit events."""
    with db_context() as conn:
        if alert_id is not None:
            rows = conn.execute(
                """
                SELECT aa.*, a.risk_category, COALESCE(a.location_name, l.name, 'Unlinked') AS location_name
                FROM alert_acknowledgements aa
                JOIN alerts a ON a.id = aa.alert_id
                LEFT JOIN locations l ON l.id = a.location_id
                WHERE aa.alert_id = ?
                ORDER BY aa.acknowledged_at DESC, aa.id DESC
                LIMIT ?
                """,
                (int(alert_id), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT aa.*, a.risk_category, COALESCE(a.location_name, l.name, 'Unlinked') AS location_name
                FROM alert_acknowledgements aa
                JOIN alerts a ON a.id = aa.alert_id
                LEFT JOIN locations l ON l.id = a.location_id
                ORDER BY aa.acknowledged_at DESC, aa.id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def get_active_cooldown_alerts(
    location_id: int | None,
    now: datetime | None = None,
) -> list[dict]:
    """Return non-resolved alerts for a location still inside their cooldown window."""
    if location_id is None:
        return []
    ref_now = now or datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)
    now_iso = ref_now.isoformat()

    with db_context() as conn:
        rows = conn.execute(
            """
            SELECT id, risk_category, probability, status, created_at, cooldown_ends_at
            FROM alerts
            WHERE location_id = ?
              AND is_resolved = 0
              AND cooldown_ends_at > ?
            ORDER BY created_at DESC, id DESC
            """,
            (int(location_id), now_iso),
        ).fetchall()
    return [dict(r) for r in rows]


def is_on_cooldown(
    location_id: int | None,
    risk_category: str,
    now: datetime | None = None,
) -> bool:
    """Check if a location is within the alert cooldown period for this severity or higher."""
    if location_id is None:
        return False

    severity_rank = {"Low": 1, "Moderate": 2, "High": 3, "Very High": 4}
    target_rank = severity_rank.get(risk_category, 0)
    active = get_active_cooldown_alerts(location_id, now=now)
    for rec in active:
        if severity_rank.get(rec.get("risk_category", ""), 0) >= target_rank:
            return True
    return False


# ─────────────────────────────────────────────
# Locations & Environmental Observations
# ─────────────────────────────────────────────

def insert_location(
    name: str,
    latitude: float,
    longitude: float,
    country: str | None = "India",
    state_region: str | None = None,
    elevation_m: float | None = None,
    slope_degrees: float | None = None,
    is_active: bool = True,
) -> int:
    """Insert a new monitoring location into the locations table.

    Returns:
        Newly created location ID.
    """
    with db_context() as conn:
        cursor = conn.execute(
            """
            INSERT INTO locations
                (name, latitude, longitude, country, state_region,
                 elevation_m, slope_degrees, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                float(latitude),
                float(longitude),
                country,
                state_region,
                float(elevation_m) if elevation_m is not None else None,
                float(slope_degrees) if slope_degrees is not None else None,
                int(is_active),
            ),
        )
        return cursor.lastrowid


def upsert_location(
    name: str,
    latitude: float,
    longitude: float,
    country: str | None = "India",
    state_region: str | None = None,
    elevation_m: float | None = None,
    slope_degrees: float | None = None,
    is_active: bool = True,
) -> int:
    """Create a monitoring location or update an existing one matched by name (case-insensitive)."""
    clean_name = name.strip()
    with db_context() as conn:
        existing = conn.execute(
            "SELECT id FROM locations WHERE LOWER(name) = LOWER(?) LIMIT 1",
            (clean_name,),
        ).fetchone()
        if existing:
            loc_id = int(existing["id"])
            conn.execute(
                """
                UPDATE locations
                SET latitude = ?,
                    longitude = ?,
                    country = COALESCE(?, country),
                    state_region = COALESCE(?, state_region),
                    elevation_m = COALESCE(?, elevation_m),
                    slope_degrees = COALESCE(?, slope_degrees),
                    is_active = ?
                WHERE id = ?
                """,
                (
                    float(latitude),
                    float(longitude),
                    country,
                    state_region,
                    float(elevation_m) if elevation_m is not None else None,
                    float(slope_degrees) if slope_degrees is not None else None,
                    int(is_active),
                    loc_id,
                ),
            )
            return loc_id

    return insert_location(
        name=clean_name,
        latitude=latitude,
        longitude=longitude,
        country=country,
        state_region=state_region,
        elevation_m=elevation_m,
        slope_degrees=slope_degrees,
        is_active=is_active,
    )


def get_locations(active_only: bool = True) -> pd.DataFrame:
    """Retrieve monitored locations from the database."""
    with db_context() as conn:
        query = "SELECT * FROM locations"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY id ASC"
        rows = conn.execute(query).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def get_location(location_id: int) -> dict | None:
    """Retrieve a single location record by ID."""
    with db_context() as conn:
        row = conn.execute(
            "SELECT * FROM locations WHERE id = ?",
            (int(location_id),),
        ).fetchone()
    return dict(row) if row else None


# Convenience alias
get_location_by_id = get_location


def update_location_status(location_id: int, is_active: bool) -> bool:
    """Activate or deactivate a monitoring location."""
    with db_context() as conn:
        cursor = conn.execute(
            "UPDATE locations SET is_active = ? WHERE id = ?",
            (int(is_active), int(location_id)),
        )
        return cursor.rowcount > 0


# Convenience alias
set_location_active = update_location_status


def insert_environmental_observation(
    location_id: int,
    observed_at: str,
    rainfall_1h_mm: float | None = None,
    rainfall_24h_mm: float | None = None,
    rainfall_72h_mm: float | None = None,
    temperature_c: float | None = None,
    humidity_pct: float | None = None,
    soil_moisture: float | None = None,
    data_source: str | None = None,
    is_observed: bool = False,
    wind_speed_kmh: float | None = None,
) -> int:
    """Persist an environmental weather/soil observation for a monitored location."""
    with db_context() as conn:
        cursor = conn.execute(
            """
            INSERT INTO environmental_observations
                (location_id, observed_at, rainfall_1h_mm, rainfall_24h_mm,
                 rainfall_72h_mm, temperature_c, humidity_pct, wind_speed_kmh,
                 soil_moisture, data_source, is_observed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(location_id),
                observed_at,
                float(rainfall_1h_mm) if rainfall_1h_mm is not None else None,
                float(rainfall_24h_mm) if rainfall_24h_mm is not None else None,
                float(rainfall_72h_mm) if rainfall_72h_mm is not None else None,
                float(temperature_c) if temperature_c is not None else None,
                float(humidity_pct) if humidity_pct is not None else None,
                float(wind_speed_kmh) if wind_speed_kmh is not None else None,
                float(soil_moisture) if soil_moisture is not None else None,
                data_source,
                int(is_observed),
            ),
        )
        return cursor.lastrowid


def get_latest_environmental_observation(location_id: int) -> dict | None:
    """Fetch the most recent environmental observation for a location."""
    with db_context() as conn:
        row = conn.execute(
            """
            SELECT * FROM environmental_observations
            WHERE location_id = ?
            ORDER BY observed_at DESC, id DESC
            LIMIT 1
            """,
            (int(location_id),),
        ).fetchone()
    return dict(row) if row else None


def get_environmental_history(location_id: int, limit: int = 50) -> pd.DataFrame:
    """Retrieve recent environmental observations for a location (oldest to newest for plotting)."""
    with db_context() as conn:
        rows = conn.execute(
            """
            SELECT * FROM environmental_observations
            WHERE location_id = ?
            ORDER BY observed_at DESC, id DESC
            LIMIT ?
            """,
            (int(location_id), int(limit)),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def get_latest_prediction(location_id: int) -> dict | None:
    """Fetch the most recent prediction record for a location."""
    with db_context() as conn:
        row = conn.execute(
            """
            SELECT * FROM prediction_records
            WHERE location_id = ?
            ORDER BY predicted_at DESC, id DESC
            LIMIT 1
            """,
            (int(location_id),),
        ).fetchone()
    return dict(row) if row else None


def get_recent_predictions(
    location_id: int | None = None,
    limit: int = 50,
) -> pd.DataFrame:
    """Retrieve recent prediction records optionally filtered by location."""
    with db_context() as conn:
        if location_id is not None:
            rows = conn.execute(
                """
                SELECT * FROM prediction_records
                WHERE location_id = ?
                ORDER BY predicted_at DESC, id DESC
                LIMIT ?
                """,
                (int(location_id), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM prediction_records
                ORDER BY predicted_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def get_recent_environmental_observations(limit: int = 20) -> pd.DataFrame:
    """Retrieve latest environmental observations across locations with station names."""
    with db_context() as conn:
        rows = conn.execute(
            """
            SELECT e.*, COALESCE(l.name, 'Unknown') AS location_name
            FROM environmental_observations e
            LEFT JOIN locations l ON l.id = e.location_id
            ORDER BY e.observed_at DESC, e.id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def get_recent_predictions_with_location(limit: int = 100) -> pd.DataFrame:
    """Retrieve recent prediction records joined with location names."""
    with db_context() as conn:
        rows = conn.execute(
            """
            SELECT p.*, COALESCE(l.name, 'Ad-hoc / Unlinked') AS location_name
            FROM prediction_records p
            LEFT JOIN locations l ON l.id = p.location_id
            ORDER BY p.predicted_at DESC, p.id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# ─────────────────────────────────────────────
# India States, Districts, Official Warnings & Provider Status
# ─────────────────────────────────────────────

def _norm_key(value: str | None) -> str:
    """Normalize State/District names consistently with IndiaBoundaryProvider."""
    return " ".join(str(value or "").strip().lower().split())


def sync_india_boundaries(
    states: list[dict[str, Any]],
    districts: list[dict[str, Any]],
    boundary_source: str = "geoBoundaries IND ADM1/ADM2",
    db_path: Path | None = None,
) -> dict[str, int]:
    """Upsert India States/UTs and Districts into SQLite without duplication."""
    now_iso = datetime.now(timezone.utc).isoformat()
    state_count = 0
    district_count = 0

    with db_context(db_path) as conn:
        state_id_by_norm: dict[str, int] = {}
        for st in states:
            name = str(st.get("state_name") or "").strip()
            norm = _norm_key(st.get("normalized_name") or name)
            if not name or not norm:
                continue
            shape_id = str(st.get("shape_id") or "").strip() or None
            iso_code = str(st.get("iso_code") or st.get("shape_iso") or "").strip() or None
            bbox_json = json.dumps(st.get("bbox")) if st.get("bbox") is not None else None
            conn.execute(
                """
                INSERT INTO india_states
                    (shape_id, state_name, normalized_name, iso_code,
                     centroid_lat, centroid_lon, bbox_json, boundary_source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(normalized_name) DO UPDATE SET
                    shape_id = COALESCE(excluded.shape_id, india_states.shape_id),
                    state_name = excluded.state_name,
                    iso_code = COALESCE(excluded.iso_code, india_states.iso_code),
                    centroid_lat = excluded.centroid_lat,
                    centroid_lon = excluded.centroid_lon,
                    bbox_json = excluded.bbox_json,
                    boundary_source = excluded.boundary_source,
                    updated_at = excluded.updated_at
                """,
                (
                    shape_id,
                    name,
                    norm,
                    iso_code,
                    st.get("centroid_lat"),
                    st.get("centroid_lon"),
                    bbox_json,
                    boundary_source,
                    now_iso,
                ),
            )
            row = conn.execute(
                "SELECT id FROM india_states WHERE normalized_name = ?",
                (norm,),
            ).fetchone()
            if row:
                state_id_by_norm[norm] = int(row["id"])
            state_count += 1

        for dist in districts:
            st_name = str(dist.get("state_name") or "").strip()
            dist_name = str(dist.get("district_name") or "").strip()
            norm_st = _norm_key(dist.get("normalized_state") or st_name)
            norm_dist = _norm_key(dist.get("normalized_district") or dist_name)
            lat = dist.get("centroid_lat") if dist.get("centroid_lat") is not None else dist.get("rep_lat")
            lon = dist.get("centroid_lon") if dist.get("centroid_lon") is not None else dist.get("rep_lon")
            shape_id = str(dist.get("shape_id") or "").strip() or None
            if not st_name or not dist_name or not norm_st or not norm_dist or lat is None or lon is None:
                continue

            state_id = state_id_by_norm.get(norm_st)
            if state_id is None:
                row = conn.execute(
                    "SELECT id FROM india_states WHERE normalized_name = ?",
                    (norm_st,),
                ).fetchone()
                if row:
                    state_id = int(row["id"])
                    state_id_by_norm[norm_st] = state_id

            bbox_json = json.dumps(dist.get("bbox")) if dist.get("bbox") is not None else None
            conn.execute(
                """
                INSERT INTO india_districts
                    (shape_id, state_id, state_name, district_name,
                     normalized_state, normalized_district,
                     centroid_lat, centroid_lon, bbox_json, boundary_source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(normalized_state, normalized_district) DO UPDATE SET
                    shape_id = COALESCE(excluded.shape_id, india_districts.shape_id),
                    state_id = COALESCE(excluded.state_id, india_districts.state_id),
                    state_name = excluded.state_name,
                    district_name = excluded.district_name,
                    centroid_lat = excluded.centroid_lat,
                    centroid_lon = excluded.centroid_lon,
                    bbox_json = excluded.bbox_json,
                    boundary_source = excluded.boundary_source,
                    updated_at = excluded.updated_at
                """,
                (
                    shape_id,
                    state_id,
                    st_name,
                    dist_name,
                    norm_st,
                    norm_dist,
                    float(lat),
                    float(lon),
                    bbox_json,
                    boundary_source,
                    now_iso,
                ),
            )
            district_count += 1

    return {"states_synced": state_count, "districts_synced": district_count}


def list_india_states(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Retrieve persisted India States/UTs ordered alphabetically."""
    with db_context(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM india_states ORDER BY state_name ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def list_india_districts(
    state_name: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Retrieve persisted India districts optionally filtered by State/UT."""
    with db_context(db_path) as conn:
        if state_name:
            rows = conn.execute(
                """
                SELECT * FROM india_districts
                WHERE normalized_state = ?
                ORDER BY district_name ASC
                """,
                (_norm_key(state_name),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM india_districts ORDER BY state_name ASC, district_name ASC"
            ).fetchall()
    return [dict(r) for r in rows]


def upsert_official_warning(
    provider: str,
    external_id: str,
    state_name: str | None = None,
    district_name: str | None = None,
    headline: str | None = None,
    disaster_type: str | None = None,
    severity: str | None = None,
    severity_color: str | None = None,
    area_description: str | None = None,
    effective_from: str | None = None,
    effective_until: str | None = None,
    issued_at: str | None = None,
    sender_org: str | None = None,
    source_url: str | None = None,
    raw_payload: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> int:
    """Persist or update an official warning record deduplicated by (provider, external_id)."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    raw_json = json.dumps(raw_payload, default=str) if raw_payload is not None else None
    with db_context(db_path) as conn:
        conn.execute(
            """
            INSERT INTO official_warnings
                (provider, external_id, state_name, district_name, headline,
                 disaster_type, severity, severity_color, area_description,
                 effective_from, effective_until, issued_at, sender_org,
                 source_url, raw_payload_json, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, external_id) DO UPDATE SET
                state_name = COALESCE(excluded.state_name, official_warnings.state_name),
                district_name = COALESCE(excluded.district_name, official_warnings.district_name),
                headline = excluded.headline,
                disaster_type = excluded.disaster_type,
                severity = excluded.severity,
                severity_color = excluded.severity_color,
                area_description = excluded.area_description,
                effective_from = excluded.effective_from,
                effective_until = excluded.effective_until,
                issued_at = excluded.issued_at,
                sender_org = excluded.sender_org,
                source_url = excluded.source_url,
                raw_payload_json = COALESCE(excluded.raw_payload_json, official_warnings.raw_payload_json),
                fetched_at = excluded.fetched_at
            """,
            (
                provider.strip(),
                str(external_id).strip(),
                state_name,
                district_name,
                headline,
                disaster_type,
                severity,
                severity_color,
                area_description,
                effective_from,
                effective_until,
                issued_at,
                sender_org,
                source_url,
                raw_json,
                fetched_at,
            ),
        )
        row = conn.execute(
            "SELECT id FROM official_warnings WHERE provider = ? AND external_id = ?",
            (provider.strip(), str(external_id).strip()),
        ).fetchone()
    return int(row["id"]) if row else 0


def get_official_warnings(
    provider: str | None = None,
    state_name: str | None = None,
    district_name: str | None = None,
    limit: int = 50,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Retrieve persisted official warnings ordered most recently fetched first."""
    clauses: list[str] = []
    params: list[Any] = []
    if provider:
        clauses.append("LOWER(provider) = LOWER(?)")
        params.append(provider.strip())
    if state_name:
        clauses.append("(LOWER(state_name) = LOWER(?) OR LOWER(COALESCE(area_description, '')) LIKE ?)")
        params.extend([state_name.strip(), f"%{state_name.strip().lower()}%"])
    if district_name:
        clauses.append("(LOWER(district_name) = LOWER(?) OR LOWER(COALESCE(area_description, '')) LIKE ?)")
        params.extend([district_name.strip(), f"%{district_name.strip().lower()}%"])

    sql = "SELECT * FROM official_warnings"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY fetched_at DESC, id DESC LIMIT ?"
    params.append(int(limit))

    with db_context(db_path) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def upsert_provider_status(
    provider_name: str,
    status: str,
    is_live: bool = False,
    is_cached: bool = False,
    last_success_at: str | None = None,
    last_attempt_at: str | None = None,
    data_timestamp: str | None = None,
    attribution: str | None = None,
    message: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Upsert external provider health & freshness telemetry in SQLite."""
    attempt_ts = last_attempt_at or datetime.now(timezone.utc).isoformat()
    with db_context(db_path) as conn:
        conn.execute(
            """
            INSERT INTO provider_status
                (provider_name, status, is_live, is_cached, last_success_at,
                 last_attempt_at, data_timestamp, attribution, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider_name) DO UPDATE SET
                status = excluded.status,
                is_live = excluded.is_live,
                is_cached = excluded.is_cached,
                last_success_at = COALESCE(excluded.last_success_at, provider_status.last_success_at),
                last_attempt_at = excluded.last_attempt_at,
                data_timestamp = COALESCE(excluded.data_timestamp, provider_status.data_timestamp),
                attribution = COALESCE(excluded.attribution, provider_status.attribution),
                message = excluded.message
            """,
            (
                provider_name.strip(),
                status.strip(),
                int(bool(is_live)),
                int(bool(is_cached)),
                last_success_at,
                attempt_ts,
                data_timestamp,
                attribution,
                message,
            ),
        )


def get_provider_statuses(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Retrieve health and freshness telemetry for all recorded providers."""
    with db_context(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM provider_status ORDER BY provider_name ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def insert_environmental_observation_dedup(
    location_id: int,
    observed_at: str,
    rainfall_1h_mm: float | None = None,
    rainfall_24h_mm: float | None = None,
    rainfall_72h_mm: float | None = None,
    temperature_c: float | None = None,
    humidity_pct: float | None = None,
    soil_moisture: float | None = None,
    data_source: str | None = None,
    is_observed: bool = False,
    wind_speed_kmh: float | None = None,
    db_path: Path | None = None,
) -> int:
    """Insert an environmental observation only if (location_id, observed_at, data_source) is new."""
    with db_context(db_path) as conn:
        existing = conn.execute(
            """
            SELECT id FROM environmental_observations
            WHERE location_id = ? AND observed_at = ? AND COALESCE(data_source, '') = COALESCE(?, '')
            LIMIT 1
            """,
            (int(location_id), observed_at, data_source),
        ).fetchone()
        if existing:
            return int(existing["id"])

    return insert_environmental_observation(
        location_id=location_id,
        observed_at=observed_at,
        rainfall_1h_mm=rainfall_1h_mm,
        rainfall_24h_mm=rainfall_24h_mm,
        rainfall_72h_mm=rainfall_72h_mm,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        soil_moisture=soil_moisture,
        data_source=data_source,
        is_observed=is_observed,
        wind_speed_kmh=wind_speed_kmh,
    )


def insert_prediction_outcome(
    actual_label: int,
    observed_at: str | None = None,
    prediction_id: int | None = None,
    location_id: int | None = None,
    outcome_source: str = "Operator Verification",
    notes: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Persist a verified ground-truth landslide outcome (0 or 1) for drift checks."""
    label_int = int(actual_label)
    if label_int not in (0, 1):
        raise ValueError(f"actual_label must be 0 or 1; got {actual_label!r}")

    obs_ts = observed_at or datetime.now(timezone.utc).isoformat()
    with db_context(db_path) as conn:
        if prediction_id is not None and location_id is None:
            pred_row = conn.execute(
                "SELECT location_id FROM prediction_records WHERE id = ?",
                (int(prediction_id),),
            ).fetchone()
            if pred_row:
                location_id = pred_row["location_id"]

        cursor = conn.execute(
            """
            INSERT INTO prediction_outcomes
                (prediction_id, location_id, actual_label, outcome_source, observed_at, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(prediction_id) if prediction_id is not None else None,
                int(location_id) if location_id is not None else None,
                label_int,
                outcome_source,
                obs_ts,
                notes,
            ),
        )
        return cursor.lastrowid


def get_prediction_outcomes(
    limit: int = 500,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Retrieve labelled outcomes joined with prediction probabilities when linked."""
    with db_context(db_path) as conn:
        rows = conn.execute(
            """
            SELECT o.*,
                   p.probability AS predicted_probability,
                   p.risk_category AS predicted_category,
                   p.model_name AS model_name,
                   p.predicted_at AS predicted_at
            FROM prediction_outcomes o
            LEFT JOIN prediction_records p ON p.id = o.prediction_id
            ORDER BY o.observed_at DESC, o.id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# ─────────────────────────────────────────────
# Community Incident Reports (Phase 9)
# ─────────────────────────────────────────────

def insert_community_report(
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
    photo_base64: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Insert a new community-submitted hazard/incident report.

    Status defaults to 'submitted'. Community reports are strictly segregated
    and never automatically merged into authentic historical inventories.
    """
    valid_categories = {
        "Rockfall", "Debris Flow / Mudflow", "Tension Cracks",
        "Road Subsidence / Slope Slump", "Blocked Drainage", "Other Hazard"
    }
    cat = category if category in valid_categories else "Other Hazard"
    sev = severity if severity in ("low", "moderate", "high", "critical") else "moderate"

    with db_context(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO community_reports (
                reporter_name, reporter_contact, location_name, state_name, district_name,
                latitude, longitude, incident_date, category, description,
                severity, photo_base64, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'submitted')
            """,
            (
                reporter_name,
                reporter_contact,
                location_name.strip(),
                state_name,
                district_name,
                float(latitude),
                float(longitude),
                str(incident_date),
                cat,
                description.strip(),
                sev,
                photo_base64,
            ),
        )
        return cur.lastrowid


def get_community_reports(
    status: str | None = None,
    limit: int = 200,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Retrieve community hazard reports, optionally filtered by review status."""
    query = "SELECT * FROM community_reports"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status.lower().strip())
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(int(limit))

    with db_context(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_community_report_by_id(
    report_id: int,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    """Retrieve a single community report by its primary key ID."""
    with db_context(db_path) as conn:
        row = conn.execute("SELECT * FROM community_reports WHERE id = ?", (int(report_id),)).fetchone()
    return dict(row) if row else None


def update_community_report_status(
    report_id: int,
    status: str,
    review_notes: str | None = None,
    reviewed_by: str | None = None,
    db_path: Path | None = None,
) -> bool:
    """Update community report status ('submitted', 'under_review', 'verified', 'rejected')."""
    valid_statuses = {"submitted", "under_review", "verified", "rejected"}
    s = status.lower().strip()
    if s not in valid_statuses:
        raise ValueError(f"Invalid report status '{status}'. Must be one of {valid_statuses}")

    now_iso = datetime.now(timezone.utc).isoformat()
    with db_context(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE community_reports
            SET status = ?, review_notes = ?, reviewed_by = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (s, review_notes, reviewed_by, now_iso, int(report_id)),
        )
        return cur.rowcount > 0

