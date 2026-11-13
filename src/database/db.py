"""
SQLite database module for LandslideGuard.

Provides connection management and schema initialisation.

Tables:
  - observations: Historical landslide event records
  - locations: Named monitoring locations
  - environmental_observations: Weather and terrain measurements
  - prediction_records: Stored model predictions
  - alerts: Generated alert records
  - alert_acknowledgements: Alert acknowledgement audit log
  - model_metadata: Trained model version information
  - synthetic_data_flag: Marks records as synthetic (for display purposes)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

DB_PATH = Path(get_setting("paths", "database") or "data/landslide_guard.db")


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and foreign key enforcement.

    Args:
        db_path: Path to the SQLite database file. Defaults to DB_PATH.

    Returns:
        Open sqlite3.Connection.
    """
    resolved_path = Path(DB_PATH if db_path is None else db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def db_context(db_path: str | Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager providing a database connection with auto-commit/rollback.

    Args:
        db_path: Path to the SQLite database file.

    Yields:
        Open sqlite3.Connection.
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_CREATE_TABLES_SQL = """
-- Historical landslide observations
CREATE TABLE IF NOT EXISTS observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_date      TEXT,
    latitude        REAL NOT NULL,
    longitude       REAL NOT NULL,
    landslide_type  TEXT,
    trigger         TEXT,
    fatalities      INTEGER,
    injuries        INTEGER,
    country         TEXT,
    state_region    TEXT,
    source_name     TEXT,
    source_url      TEXT,
    is_synthetic    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Named monitoring locations
CREATE TABLE IF NOT EXISTS locations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    latitude        REAL NOT NULL,
    longitude       REAL NOT NULL,
    country         TEXT,
    state_region    TEXT,
    elevation_m     REAL,
    slope_degrees   REAL,
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Environmental observations
CREATE TABLE IF NOT EXISTS environmental_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id     INTEGER REFERENCES locations(id),
    observed_at     TEXT NOT NULL,
    rainfall_1h_mm  REAL,
    rainfall_24h_mm REAL,
    rainfall_72h_mm REAL,
    temperature_c   REAL,
    humidity_pct    REAL,
    wind_speed_kmh  REAL,
    soil_moisture   REAL,
    data_source     TEXT,
    is_observed     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Prediction records
CREATE TABLE IF NOT EXISTS prediction_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id     INTEGER REFERENCES locations(id),
    predicted_at    TEXT NOT NULL,
    probability     REAL,
    risk_category   TEXT,
    model_name      TEXT,
    method          TEXT,
    is_experimental INTEGER NOT NULL DEFAULT 1,
    input_features  TEXT,
    warnings        TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Alerts
CREATE TABLE IF NOT EXISTS alerts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id         INTEGER REFERENCES locations(id),
    prediction_id       INTEGER REFERENCES prediction_records(id),
    location_name       TEXT,
    latitude            REAL,
    longitude           REAL,
    risk_category       TEXT NOT NULL,
    probability         REAL,
    supporting_env_info TEXT,
    model_version       TEXT,
    data_source         TEXT,
    status              TEXT NOT NULL DEFAULT 'new',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    is_experimental     INTEGER NOT NULL DEFAULT 1,
    is_acknowledged     INTEGER NOT NULL DEFAULT 0,
    acknowledged_at     TEXT,
    is_resolved         INTEGER NOT NULL DEFAULT 0,
    resolved_at         TEXT,
    resolution_note     TEXT,
    escalated_from      TEXT,
    cooldown_ends_at    TEXT
);

-- Alert lifecycle audit log (acknowledgements & resolutions)
CREATE TABLE IF NOT EXISTS alert_acknowledgements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id        INTEGER NOT NULL REFERENCES alerts(id),
    action          TEXT NOT NULL DEFAULT 'acknowledged',
    acknowledged_at TEXT NOT NULL DEFAULT (datetime('now')),
    note            TEXT
);

-- Model metadata
CREATE TABLE IF NOT EXISTS model_metadata (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL UNIQUE,
    algorithm       TEXT,
    trained_at      TEXT,
    dataset_version TEXT,
    feature_columns TEXT,
    test_pr_auc     REAL,
    test_roc_auc    REAL,
    test_f1         REAL,
    is_active       INTEGER NOT NULL DEFAULT 0,
    notes           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- India States / Union Territories (Phase: India Regional Data Integration)
CREATE TABLE IF NOT EXISTS india_states (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    shape_id            TEXT UNIQUE,
    state_name          TEXT NOT NULL,
    normalized_name     TEXT NOT NULL UNIQUE,
    iso_code            TEXT,
    centroid_lat        REAL,
    centroid_lon        REAL,
    bbox_json           TEXT,
    boundary_source     TEXT,
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- India Districts mapped to parent State/UT
CREATE TABLE IF NOT EXISTS india_districts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    shape_id            TEXT UNIQUE,
    state_id            INTEGER REFERENCES india_states(id),
    state_name          TEXT NOT NULL,
    district_name       TEXT NOT NULL,
    normalized_state    TEXT NOT NULL,
    normalized_district TEXT NOT NULL,
    centroid_lat        REAL NOT NULL,
    centroid_lon        REAL NOT NULL,
    bbox_json           TEXT,
    boundary_source     TEXT,
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(normalized_state, normalized_district)
);

-- Official External Warnings (NDMA SACHET CAP / IMD — strictly separate from experimental alerts)
CREATE TABLE IF NOT EXISTS official_warnings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    provider            TEXT NOT NULL,
    external_id         TEXT NOT NULL,
    state_name          TEXT,
    district_name       TEXT,
    headline            TEXT,
    disaster_type       TEXT,
    severity            TEXT,
    severity_color      TEXT,
    area_description    TEXT,
    effective_from      TEXT,
    effective_until     TEXT,
    issued_at           TEXT,
    sender_org          TEXT,
    source_url          TEXT,
    raw_payload_json    TEXT,
    fetched_at          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(provider, external_id)
);

-- External Data Provider Health & Freshness Status
CREATE TABLE IF NOT EXISTS provider_status (
    provider_name       TEXT PRIMARY KEY,
    status              TEXT NOT NULL,
    is_live             INTEGER DEFAULT 0,
    is_cached           INTEGER DEFAULT 0,
    last_success_at     TEXT,
    last_attempt_at     TEXT,
    data_timestamp      TEXT,
    attribution         TEXT,
    message             TEXT
);

-- Ground-Truth Outcome Feedback for Model Drift Monitoring (Phase 6)
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id   INTEGER REFERENCES prediction_records(id),
    location_id     INTEGER REFERENCES locations(id),
    actual_label    INTEGER NOT NULL,
    outcome_source  TEXT,
    observed_at     TEXT NOT NULL,
    notes           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Community Incident Reports (Phase 9 Disaster Response & Community Intelligence)
CREATE TABLE IF NOT EXISTS community_reports (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    reporter_name       TEXT,
    reporter_contact    TEXT,
    location_name       TEXT NOT NULL,
    state_name          TEXT,
    district_name       TEXT,
    latitude            REAL NOT NULL,
    longitude           REAL NOT NULL,
    incident_date       TEXT NOT NULL,
    category            TEXT NOT NULL,
    description         TEXT NOT NULL,
    severity            TEXT NOT NULL DEFAULT 'moderate',
    photo_base64        TEXT,
    status              TEXT NOT NULL DEFAULT 'submitted',
    review_notes        TEXT,
    reviewed_by         TEXT,
    reviewed_at         TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_observations_date ON observations(event_date);
CREATE INDEX IF NOT EXISTS idx_observations_location ON observations(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(is_acknowledged);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON prediction_records(predicted_at);
CREATE INDEX IF NOT EXISTS idx_india_districts_state ON india_districts(normalized_state, normalized_district);
CREATE INDEX IF NOT EXISTS idx_official_warnings_region ON official_warnings(provider, state_name, district_name);
CREATE INDEX IF NOT EXISTS idx_prediction_outcomes_pred_id ON prediction_outcomes(prediction_id);
CREATE INDEX IF NOT EXISTS idx_community_reports_status ON community_reports(status);
CREATE INDEX IF NOT EXISTS idx_community_reports_coords ON community_reports(latitude, longitude);
"""


def initialize_database(db_path: str | Path = DB_PATH) -> None:
    """Create all tables and indexes if they do not exist.

    Idempotent — safe to call multiple times and upgrades existing schemas cleanly.

    Args:
        db_path: Path to the SQLite database file.
    """
    with db_context(db_path) as conn:
        conn.executescript(_CREATE_TABLES_SQL)

        env_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(environmental_observations)").fetchall()
        }
        if "wind_speed_kmh" not in env_cols:
            conn.execute(
                "ALTER TABLE environmental_observations ADD COLUMN wind_speed_kmh REAL"
            )

        alert_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(alerts)").fetchall()
        }
        alert_migrations = {
            "location_name": "TEXT",
            "latitude": "REAL",
            "longitude": "REAL",
            "supporting_env_info": "TEXT",
            "model_version": "TEXT",
            "data_source": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'new'",
            "is_resolved": "INTEGER NOT NULL DEFAULT 0",
            "resolved_at": "TEXT",
            "resolution_note": "TEXT",
            "escalated_from": "TEXT",
        }
        for col_name, col_def in alert_migrations.items():
            if col_name not in alert_cols:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {col_name} {col_def}")

        # Keep legacy rows consistent with Phase 5 lifecycle statuses
        conn.execute(
            """
            UPDATE alerts
            SET status = CASE
                WHEN is_resolved = 1 THEN 'resolved'
                WHEN is_acknowledged = 1 THEN 'acknowledged'
                ELSE 'new'
            END
            WHERE status IS NULL OR status = ''
               OR (is_resolved = 1 AND status != 'resolved')
               OR (is_resolved = 0 AND is_acknowledged = 1 AND status = 'new')
            """
        )

        ack_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(alert_acknowledgements)").fetchall()
        }
        if "action" not in ack_cols:
            conn.execute(
                "ALTER TABLE alert_acknowledgements ADD COLUMN action TEXT NOT NULL DEFAULT 'acknowledged'"
            )

        # Create Phase 5 indexes after column migrations succeed
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
            CREATE INDEX IF NOT EXISTS idx_alerts_category ON alerts(risk_category);
            CREATE INDEX IF NOT EXISTS idx_alerts_cooldown_lookup
                ON alerts(location_id, risk_category, status, cooldown_ends_at);
            CREATE INDEX IF NOT EXISTS idx_alerts_ack_alert_id
                ON alert_acknowledgements(alert_id);
            """
        )
    logger.info("Database initialised at %s", db_path)


def get_table_counts(db_path: str | Path = DB_PATH) -> dict[str, int]:
    """Return row counts for each table.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Dict of table name → row count.
    """
    tables = [
        "observations", "locations", "environmental_observations",
        "prediction_records", "prediction_outcomes", "alerts", "alert_acknowledgements", "model_metadata",
        "india_states", "india_districts", "official_warnings", "provider_status", "community_reports",
    ]
    counts = {}
    with db_context(db_path) as conn:
        for table in tables:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = row[0] if row else 0
            except Exception:
                counts[table] = -1  # Table may not exist
    return counts
