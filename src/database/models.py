"""
Database CRUD operations for LandslideGuard.

Provides typed helper functions for inserting and querying records.
All database access goes through the db_context context manager.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
) -> int:
    """Create a new alert record.

    Args:
        risk_category: Alert risk category.
        probability: Estimated probability.
        location_id: Optional linked location.
        prediction_id: Optional linked prediction.
        cooldown_minutes: Minutes before another alert for same location.

    Returns:
        Alert record ID.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    cooldown_ends = (now + timedelta(minutes=cooldown_minutes)).isoformat()

    with db_context() as conn:
        cursor = conn.execute(
            """
            INSERT INTO alerts
                (location_id, prediction_id, risk_category, probability,
                 is_experimental, cooldown_ends_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (location_id, prediction_id, risk_category, probability, cooldown_ends),
        )
        return cursor.lastrowid


def get_alerts(
    include_acknowledged: bool = False,
    limit: int = 100,
) -> pd.DataFrame:
    """Retrieve alerts from the database.

    Args:
        include_acknowledged: If True, include acknowledged alerts.
        limit: Maximum rows to return.

    Returns:
        DataFrame of alerts.
    """
    with db_context() as conn:
        query = "SELECT * FROM alerts"
        if not include_acknowledged:
            query += " WHERE is_acknowledged = 0"
        query += " ORDER BY created_at DESC LIMIT ?"
        rows = conn.execute(query, (int(limit),)).fetchall()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def acknowledge_alert(alert_id: int, note: str | None = None) -> bool:
    """Mark an alert as acknowledged.

    Args:
        alert_id: Alert record ID.
        note: Optional acknowledgement note.

    Returns:
        True if the alert existed and was acknowledged, False if not found.
    """
    now = datetime.now(timezone.utc).isoformat()
    with db_context() as conn:
        cursor = conn.execute(
            "UPDATE alerts SET is_acknowledged=1, acknowledged_at=? WHERE id=?",
            (now, alert_id),
        )
        if cursor.rowcount == 0:
            logger.warning("Alert %d not found; cannot acknowledge.", alert_id)
            return False
        conn.execute(
            "INSERT INTO alert_acknowledgements (alert_id, acknowledged_at, note) VALUES (?, ?, ?)",
            (alert_id, now, note),
        )
    logger.info("Alert %d acknowledged.", alert_id)
    return True


def is_on_cooldown(location_id: int | None, risk_category: str) -> bool:
    """Check if a location is within the alert cooldown period.

    Args:
        location_id: Location ID (or None for unlinked alerts).
        risk_category: Risk category to check.

    Returns:
        True if a recent unacknowledged alert for this location exists.
    """
    if location_id is None:
        return False

    now = datetime.now(timezone.utc).isoformat()
    with db_context() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM alerts
            WHERE location_id = ? AND risk_category = ?
              AND is_acknowledged = 0
              AND cooldown_ends_at > ?
            """,
            (location_id, risk_category, now),
        ).fetchone()

    return (row[0] if row else 0) > 0
