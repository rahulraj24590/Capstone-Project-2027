"""Comprehensive unit and integration tests for Phase 5 Alert Management System."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

import src.database.db as db_mod
import src.database.models as models_mod
from api import (
    AcknowledgeRequest,
    acknowledge_alert,
    get_alert_audit,
    get_alert_notification_status,
    get_alerts,
    resolve_alert_endpoint,
)
from src.alerts.alert_manager import AlertManager
from src.alerts.notifications import NotificationDispatcher


def _run_coro_sync(coro):
    """Execute a coroutine synchronously without creating an asyncio socketpair."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch):
    """Point db_context at a clean temporary SQLite database."""
    db_file = tmp_path / "test_alerts_p5.db"

    @contextmanager
    def _tmp_ctx(_db_path=None):
        conn = db_mod.get_connection(db_file)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(db_mod, "db_context", _tmp_ctx)
    monkeypatch.setattr(models_mod, "db_context", _tmp_ctx)
    db_mod.initialize_database(db_file)
    return db_file


def test_alert_generation_fields_and_threshold_boundaries(isolated_db):
    loc_id = models_mod.upsert_location(
        name="Munnar Alert Test",
        latitude=10.0889,
        longitude=77.0595,
        state_region="Kerala",
        elevation_m=1525.0,
        slope_degrees=32.0,
    )
    mgr = AlertManager()

    # Low threshold check -> suppressed under default High trigger
    low_res = {"probability": 0.25, "risk_category": "Low", "method": "ml_model", "features_used": {"rainfall_24h_mm": 5.0}}
    assert mgr.evaluate_and_alert(low_res, location_id=loc_id, location_name="Munnar Alert Test") is None

    # Exact boundary below High (0.5499 -> Moderate) suppressed under High trigger
    mod_res = {"probability": 0.5499, "risk_category": "Moderate", "method": "ml_model", "features_used": {"rainfall_24h_mm": 40.0}}
    assert mgr.evaluate_and_alert(mod_res, location_id=loc_id, location_name="Munnar Alert Test", trigger_category="High") is None

    # Configurable lower trigger allows Moderate explicitly
    mod_alert = mgr.evaluate_and_alert(
        mod_res,
        location_id=loc_id,
        location_name="Munnar Alert Test",
        model_version="best_model [lr]",
        data_source="Open-Meteo Test",
        trigger_category="Moderate",
    )
    assert mod_alert is not None
    assert mod_alert["id"] > 0
    assert mod_alert["location_id"] == loc_id
    assert mod_alert["latitude"] == pytest.approx(10.0889)
    assert mod_alert["longitude"] == pytest.approx(77.0595)
    assert mod_alert["risk_category"] == "Moderate"
    assert mod_alert["probability"] == pytest.approx(0.5499)
    assert mod_alert["supporting_env_info"] == {"rainfall_24h_mm": 40.0}
    assert mod_alert["model_version"] == "best_model [lr]"
    assert mod_alert["data_source"] == "Open-Meteo Test"
    assert mod_alert["status"] == "new"
    assert mod_alert["is_experimental"] is True
    assert "EXPERIMENTAL" in mod_alert["notice"]


def test_cooldown_suppression_and_severity_escalation(isolated_db):
    loc_id = models_mod.upsert_location(
        name="Wayanad Cooldown Test",
        latitude=11.6854,
        longitude=76.1320,
        elevation_m=950.0,
        slope_degrees=29.0,
    )
    mgr = AlertManager()
    t0 = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)

    high_res = {"probability": 0.65, "risk_category": "High", "method": "ml_model", "features_used": {"rainfall_24h_mm": 90.0}}
    first = mgr.evaluate_and_alert(
        high_res, location_id=loc_id, location_name="Wayanad Cooldown Test", cooldown_minutes=120, now=t0
    )
    assert first is not None
    assert first["is_escalation"] is False

    # Same severity within cooldown (t0 + 30 min) MUST be suppressed
    dup = mgr.evaluate_and_alert(
        high_res, location_id=loc_id, location_name="Wayanad Cooldown Test", cooldown_minutes=120, now=t0 + timedelta(minutes=30)
    )
    assert dup is None

    # Significant escalation High -> Very High within cooldown MUST be allowed
    vhigh_res = {"probability": 0.88, "risk_category": "Very High", "method": "ml_model", "features_used": {"rainfall_24h_mm": 180.0}}
    esc = mgr.evaluate_and_alert(
        vhigh_res, location_id=loc_id, location_name="Wayanad Cooldown Test", cooldown_minutes=120, now=t0 + timedelta(minutes=45)
    )
    assert esc is not None
    assert esc["risk_category"] == "Very High"
    assert esc["is_escalation"] is True
    assert esc["escalated_from"] == "High"

    # Repeated Very High right after escalation MUST be suppressed while inside cooldown
    dup_vh = mgr.evaluate_and_alert(
        vhigh_res, location_id=loc_id, location_name="Wayanad Cooldown Test", cooldown_minutes=120, now=t0 + timedelta(minutes=60)
    )
    assert dup_vh is None

    # After cooldown expires (t0 + 180 min), alert generation resumes
    after = mgr.evaluate_and_alert(
        high_res, location_id=loc_id, location_name="Wayanad Cooldown Test", cooldown_minutes=120, now=t0 + timedelta(minutes=180)
    )
    assert after is not None


def test_pure_cooldown_policy_helper():
    policy = AlertManager.evaluate_cooldown_policy(
        location_id=1,
        new_category="Very High",
        active_cooldown_alerts=[{"risk_category": "High", "cooldown_ends_at": "2099-01-01T00:00:00+00:00"}],
    )
    assert policy["suppressed"] is False
    assert policy["is_escalation"] is True
    assert policy["escalated_from"] == "High"

    suppressed = AlertManager.evaluate_cooldown_policy(
        location_id=1,
        new_category="High",
        active_cooldown_alerts=[{"risk_category": "Very High", "cooldown_ends_at": "2099-01-01T00:00:00+00:00"}],
    )
    assert suppressed["suppressed"] is True


def test_alert_lifecycle_acknowledge_resolve_history_and_persistence(isolated_db, monkeypatch):
    loc_id = models_mod.upsert_location(name="Darjeeling Lifecycle", latitude=27.041, longitude=88.2663)
    mgr = AlertManager()
    t0 = datetime(2026, 7, 2, 8, 0, tzinfo=timezone.utc)

    created = mgr.evaluate_and_alert(
        {"probability": 0.82, "risk_category": "Very High", "method": "ml_model", "features_used": {"rainfall_24h_mm": 140.0}},
        location_id=loc_id,
        location_name="Darjeeling Lifecycle",
        now=t0,
    )
    alert_id = created["id"]

    assert len(mgr.get_active_alerts()) == 1
    assert len(mgr.get_high_priority_alerts()) == 1

    # Acknowledge transitions status new -> acknowledged and creates audit entry
    assert mgr.acknowledge(alert_id, "Checked telemetry") is True
    assert len(mgr.get_active_alerts()) == 0
    ack_df = mgr.get_acknowledged_alerts()
    assert len(ack_df) == 1
    assert ack_df.iloc[0]["status"] == "acknowledged"
    assert ack_df.iloc[0]["acknowledged_at"] is not None

    # Resolve transitions status acknowledged -> resolved and logs audit entry
    assert mgr.resolve(alert_id, "Rain cleared") is True
    assert len(mgr.get_acknowledged_alerts()) == 0
    res_df = mgr.get_resolved_alerts()
    assert len(res_df) == 1
    assert res_df.iloc[0]["status"] == "resolved"
    assert res_df.iloc[0]["resolution_note"] == "Rain cleared"

    # Full history + category/date filtering + audit log verification
    hist_df = mgr.get_all_alerts(risk_category="Very High", start_date="2026-07-01", end_date="2026-07-03")
    assert len(hist_df) == 1
    assert mgr.get_all_alerts(risk_category="Moderate").empty

    audit_df = mgr.get_audit_log(alert_id=alert_id)
    assert len(audit_df) == 2
    assert set(audit_df["action"]) == {"acknowledged", "resolved"}

    # Re-initializing database is idempotent and preserves persisted records
    db_mod.initialize_database(isolated_db)
    fresh_mgr = AlertManager()
    persisted = fresh_mgr.get_alert(alert_id)
    assert persisted is not None
    assert persisted["status"] == "resolved"
    assert persisted["latitude"] == pytest.approx(27.041)


def test_missing_risk_estimates_and_disabled_notifications(isolated_db):
    mgr = AlertManager(dispatcher=NotificationDispatcher())

    # None / NaN / malformed risk estimates safely produce None without exceptions
    assert mgr.evaluate_and_alert({"probability": None, "risk_category": "Unknown"}) is None
    assert mgr.evaluate_and_alert({"probability": float("nan"), "risk_category": "High"}) is None
    assert mgr.evaluate_and_alert({}) is None

    # External Email and SMS channels stay strictly disabled by default
    statuses = mgr.get_notification_status()
    assert all(s["enabled"] is False for s in statuses)
    dispatch_res = mgr.dispatcher.dispatch({"id": 1, "risk_category": "High"})
    assert all(r["dispatched"] is False and r["reason"] == "disabled_by_default" for r in dispatch_res)


def test_phase5_alert_api_endpoints(isolated_db):
    loc_id = models_mod.upsert_location(name="API Station", latitude=10.1, longitude=77.1)
    mgr = AlertManager()
    a1 = mgr.evaluate_and_alert(
        {"probability": 0.90, "risk_category": "Very High", "method": "ml_model", "features_used": {"rainfall_24h_mm": 160.0}},
        location_id=loc_id,
        location_name="API Station",
    )
    aid = a1["id"]

    active_resp = _run_coro_sync(get_alerts(status="new"))
    assert active_resp["count"] == 1
    assert active_resp["alerts"][0]["is_experimental"] == 1

    ack_resp = _run_coro_sync(acknowledge_alert(aid, AcknowledgeRequest(note="API Ack")))
    assert ack_resp["status"] == "acknowledged"

    res_resp = _run_coro_sync(resolve_alert_endpoint(aid, AcknowledgeRequest(note="API Resolved")))
    assert res_resp["status"] == "resolved"

    audit_resp = _run_coro_sync(get_alert_audit(alert_id=aid))
    assert audit_resp["count"] == 2

    notif_resp = _run_coro_sync(get_alert_notification_status())
    assert all(ch["enabled"] is False for ch in notif_resp["channels"])

    with pytest.raises(HTTPException) as exc_info:
        _run_coro_sync(resolve_alert_endpoint(999999))
    assert exc_info.value.status_code == 404

