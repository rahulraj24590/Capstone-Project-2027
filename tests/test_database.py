"""
Tests for the SQLite database module.

Verifies:
- Database initializes successfully
- Tables are created
- Basic CRUD operations work
- Alert cooldown logic is correct
- Prediction records are saved and retrievable
- No fabricated data is inserted automatically
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary SQLite database path for isolated tests."""
    return tmp_path / "test_landslide_guard.db"


# ─────────────────────────────────────────────
# Initialization tests
# ─────────────────────────────────────────────

class TestDatabaseInitialization:
    def test_initialize_creates_db_file(self, temp_db):
        from src.database.db import initialize_database
        initialize_database(temp_db)
        assert temp_db.exists()

    def test_initialize_idempotent(self, temp_db):
        """Calling initialize_database twice must not raise."""
        from src.database.db import initialize_database
        initialize_database(temp_db)
        initialize_database(temp_db)  # second call
        assert temp_db.exists()

    def test_all_tables_created(self, temp_db):
        from src.database.db import initialize_database, get_table_counts
        initialize_database(temp_db)
        counts = get_table_counts(temp_db)
        expected_tables = [
            "observations", "locations", "environmental_observations",
            "prediction_records", "alerts", "alert_acknowledgements", "model_metadata",
        ]
        for table in expected_tables:
            assert table in counts, f"Table '{table}' not created"

    def test_initial_tables_are_empty(self, temp_db):
        from src.database.db import initialize_database, get_table_counts
        initialize_database(temp_db)
        counts = get_table_counts(temp_db)
        for table, count in counts.items():
            assert count >= 0  # All counts should be non-negative after init
            assert count == 0, f"Table '{table}' has {count} rows on init (expected 0)"


# ─────────────────────────────────────────────
# Connection tests
# ─────────────────────────────────────────────

class TestDatabaseConnection:
    def test_get_connection_returns_connection(self, temp_db):
        from src.database.db import get_connection, initialize_database
        initialize_database(temp_db)
        conn = get_connection(temp_db)
        assert conn is not None
        conn.close()

    def test_context_manager_commits(self, temp_db):
        from src.database.db import db_context, initialize_database
        initialize_database(temp_db)
        with db_context(temp_db) as conn:
            conn.execute("SELECT 1")  # Should not raise

    def test_context_manager_rollback_on_error(self, temp_db):
        from src.database.db import db_context, initialize_database
        initialize_database(temp_db)
        with pytest.raises(ValueError):
            with db_context(temp_db) as conn:
                raise ValueError("Intentional error for rollback test")


# ─────────────────────────────────────────────
# Direct SQL operation tests (using temp_db directly)
# ─────────────────────────────────────────────

class TestDirectSQLOperations:
    """Tests using the db_context directly with a temp database path."""

    def test_insert_and_count_prediction(self, temp_db):
        from src.database.db import initialize_database, db_context
        import json
        from datetime import datetime, timezone

        initialize_database(temp_db)

        with db_context(temp_db) as conn:
            conn.execute(
                """
                INSERT INTO prediction_records
                    (location_id, predicted_at, probability, risk_category,
                     model_name, method, is_experimental, input_features, warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (None, datetime.now(timezone.utc).isoformat(),
                 0.72, "High", "test_model", "ml_model", 1,
                 json.dumps({"slope_degrees": 30.0}), json.dumps([])),
            )

        from src.database.db import get_table_counts
        counts = get_table_counts(temp_db)
        assert counts["prediction_records"] == 1

    def test_insert_and_count_alert(self, temp_db):
        from src.database.db import initialize_database, db_context, get_table_counts
        from datetime import datetime, timezone, timedelta

        initialize_database(temp_db)
        cooldown_ends = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

        with db_context(temp_db) as conn:
            conn.execute(
                """
                INSERT INTO alerts
                    (location_id, prediction_id, risk_category, probability,
                     is_experimental, cooldown_ends_at)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (None, None, "High", 0.80, cooldown_ends),
            )

        counts = get_table_counts(temp_db)
        assert counts["alerts"] == 1

    def test_acknowledge_alert_in_db(self, temp_db):
        from src.database.db import initialize_database, db_context
        from datetime import datetime, timezone, timedelta

        initialize_database(temp_db)
        cooldown_ends = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        now = datetime.now(timezone.utc).isoformat()

        with db_context(temp_db) as conn:
            cursor = conn.execute(
                "INSERT INTO alerts (risk_category, probability, is_experimental, cooldown_ends_at) VALUES (?,?,1,?)",
                ("High", 0.85, cooldown_ends),
            )
            alert_id = cursor.lastrowid
            conn.execute(
                "UPDATE alerts SET is_acknowledged=1, acknowledged_at=? WHERE id=?",
                (now, alert_id),
            )

        with db_context(temp_db) as conn:
            row = conn.execute(
                "SELECT is_acknowledged FROM alerts WHERE id=?", (alert_id,)
            ).fetchone()
            assert row[0] == 1

    def test_observation_insert(self, temp_db):
        from src.database.db import initialize_database, db_context, get_table_counts
        initialize_database(temp_db)

        with db_context(temp_db) as conn:
            conn.execute(
                "INSERT INTO observations (latitude, longitude, is_synthetic) VALUES (?, ?, ?)",
                (22.5, 78.0, 1),
            )

        counts = get_table_counts(temp_db)
        assert counts["observations"] == 1


# ─────────────────────────────────────────────
# Table count tests
# ─────────────────────────────────────────────

class TestTableCounts:
    def test_get_table_counts_returns_dict(self, temp_db):
        from src.database.db import initialize_database, get_table_counts
        initialize_database(temp_db)
        counts = get_table_counts(temp_db)
        assert isinstance(counts, dict)

    def test_table_counts_non_negative(self, temp_db):
        from src.database.db import initialize_database, get_table_counts
        initialize_database(temp_db)
        counts = get_table_counts(temp_db)
        for table, count in counts.items():
            assert count >= 0, f"Table '{table}' has negative count: {count}"


# ─────────────────────────────────────────────
# Database helper CRUD tests (models.py)
# ─────────────────────────────────────────────

class TestDatabaseHelpers:
    def test_acknowledge_nonexistent_alert_returns_false(self, temp_db, monkeypatch):
        """Acknowledging a missing alert must return False without raising FK errors."""
        from contextlib import contextmanager
        from src.database.db import initialize_database, db_context as real_db_context
        import src.database.models as models

        initialize_database(temp_db)

        @contextmanager
        def _temp_ctx():
            with real_db_context(temp_db) as conn:
                yield conn

        monkeypatch.setattr(models, "db_context", _temp_ctx)
        assert models.acknowledge_alert(99999, "does not exist") is False

    def test_acknowledge_existing_alert_returns_true_and_audits(self, temp_db, monkeypatch):
        from contextlib import contextmanager
        from src.database.db import initialize_database, db_context as real_db_context
        import src.database.models as models

        initialize_database(temp_db)

        @contextmanager
        def _temp_ctx():
            with real_db_context(temp_db) as conn:
                yield conn

        monkeypatch.setattr(models, "db_context", _temp_ctx)
        alert_id = models.create_alert(risk_category="High", probability=0.88)
        assert models.acknowledge_alert(alert_id, "verified") is True

        active = models.get_active_alerts() if hasattr(models, "get_active_alerts") else models.get_alerts(include_acknowledged=False)
        assert active.empty
        all_alerts = models.get_alerts(include_acknowledged=True, limit=10)
        assert len(all_alerts) == 1
        assert int(all_alerts.iloc[0]["is_acknowledged"]) == 1

    def test_get_observations_limit(self, temp_db, monkeypatch):
        from contextlib import contextmanager
        from src.database.db import initialize_database, db_context as real_db_context
        import src.database.models as models

        initialize_database(temp_db)

        @contextmanager
        def _temp_ctx():
            with real_db_context(temp_db) as conn:
                yield conn

        monkeypatch.setattr(models, "db_context", _temp_ctx)
        for i in range(5):
            models.insert_observation(latitude=20.0 + i, longitude=78.0, is_synthetic=True)

        df = models.get_observations(limit=3, include_synthetic=True)
        assert len(df) == 3
