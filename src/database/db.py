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


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and foreign key enforcement.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Open sqlite3.Connection.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def db_context(db_path: str | Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
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
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id     INTEGER REFERENCES locations(id),
    prediction_id   INTEGER REFERENCES prediction_records(id),
    risk_category   TEXT NOT NULL,
    probability     REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    is_experimental INTEGER NOT NULL DEFAULT 1,
    is_acknowledged INTEGER NOT NULL DEFAULT 0,
    acknowledged_at TEXT,
    cooldown_ends_at TEXT
);

-- Alert acknowledgements audit log
CREATE TABLE IF NOT EXISTS alert_acknowledgements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id        INTEGER NOT NULL REFERENCES alerts(id),
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

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_observations_date ON observations(event_date);
CREATE INDEX IF NOT EXISTS idx_observations_location ON observations(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(is_acknowledged);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON prediction_records(predicted_at);
"""


def initialize_database(db_path: str | Path = DB_PATH) -> None:
    """Create all tables and indexes if they do not exist.

    Idempotent — safe to call multiple times.

    Args:
        db_path: Path to the SQLite database file.
    """
    with db_context(db_path) as conn:
        conn.executescript(_CREATE_TABLES_SQL)
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
        "prediction_records", "alerts", "alert_acknowledgements", "model_metadata",
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
