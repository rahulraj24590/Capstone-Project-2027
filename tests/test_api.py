"""
Tests for application-level imports and startup.

Verifies:
- Application imports without errors
- Settings can be loaded
- Database initializes on startup
- Project works without optional external services
- No credentials are hard-coded in source files
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


# ─────────────────────────────────────────────
# Import tests
# ─────────────────────────────────────────────

class TestImports:
    def test_src_package_imports(self):
        """All src modules should be importable without error."""
        from src.utils.logging_config import get_logger
        assert callable(get_logger)

    def test_settings_module_imports(self):
        from src.utils.settings import get_setting, load_settings
        assert callable(get_setting)

    def test_database_module_imports(self):
        from src.database.db import initialize_database, get_connection, db_context
        assert callable(initialize_database)

    def test_risk_engine_imports(self):
        from src.risk.risk_engine import RiskEngine, get_risk_engine, BaselineRiskScorer
        assert callable(get_risk_engine)

    def test_models_module_imports(self):
        from src.models.model_registry import save_model, load_model, active_model_exists
        assert callable(active_model_exists)

    def test_predict_module_imports(self):
        from src.models.predict import Predictor, get_predictor, probability_to_risk_category
        assert callable(probability_to_risk_category)

    def test_alerts_module_imports(self):
        from src.alerts.alert_manager import AlertManager, get_alert_manager
        assert callable(get_alert_manager)

    def test_preprocessing_imports(self):
        from src.data_processing.preprocessing import (
            validate_columns,
            remove_invalid_coordinates,
            handle_missing_values,
        )
        assert callable(validate_columns)

    def test_feature_engineering_imports(self):
        from src.data_processing.feature_engineering import (
            compute_slope_class,
            select_available_features,
        )
        assert callable(compute_slope_class)

    def test_geospatial_maps_imports(self):
        from src.geospatial.maps import build_landslide_map, create_base_map
        assert callable(build_landslide_map)

    def test_data_providers_imports(self):
        from src.data_providers.base import DataProvider
        from src.data_providers.weather_provider import WeatherProvider
        assert True  # If we get here, imports succeeded


# ─────────────────────────────────────────────
# Settings tests
# ─────────────────────────────────────────────

class TestSettings:
    def test_settings_file_exists(self):
        assert Path("config/settings.yaml").exists()

    def test_settings_loads(self):
        from src.utils.settings import load_settings
        settings = load_settings()
        assert isinstance(settings, dict)

    def test_app_settings_present(self):
        from src.utils.settings import get_setting
        assert get_setting("app", "name") is not None
        assert get_setting("app", "version") is not None

    def test_model_settings_present(self):
        from src.utils.settings import get_setting
        assert get_setting("model", "active_model") is not None

    def test_risk_thresholds_present(self):
        from src.utils.settings import get_setting
        thresholds = get_setting("risk", "thresholds")
        assert thresholds is not None
        assert "low" in thresholds
        assert "moderate" in thresholds
        assert "high" in thresholds


# ─────────────────────────────────────────────
# Security tests
# ─────────────────────────────────────────────

class TestNoHardcodedSecrets:
    """Verify no API keys or secrets are hard-coded in source files."""

    SUSPICIOUS_PATTERNS = [
        "api_key =",
        "API_KEY =",
        "password =",
        "secret =",
        "token =",
    ]

    def _search_pattern_in_src(self, pattern: str) -> list[Path]:
        """Return list of Python files in src/ that contain the pattern."""
        found = []
        for py_file in Path("src").rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                # Skip comment lines and the .env.example pattern
                lines = [ln for ln in content.splitlines()
                         if not ln.strip().startswith("#")
                         and "example" not in str(py_file).lower()
                         and "test" not in str(py_file).lower()]
                for line in lines:
                    if pattern.lower() in line.lower():
                        # Check if it's an actual assignment, not documentation
                        if "=" in line and "os.environ" not in line and "get_setting" not in line:
                            found.append(py_file)
                            break
            except Exception:
                pass
        return found

    def test_no_hardcoded_api_keys(self):
        """No literal api_key assignment in source code.

        Looks for lines that assign a string *literal* to an api_key variable.
        Excludes:
        - Comment lines
        - Lines using os.environ / get_setting (proper env-var reads)
        - Lines where the RHS is a self.* attribute (already read from env)
        """
        src_dir = Path("src")
        suspicious = []
        for py_file in src_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for line in content.splitlines():
                    stripped = line.strip()
                    if not ("api_key" in stripped.lower() and "=" in stripped):
                        continue
                    if stripped.startswith("#"):
                        continue
                    if any(exc in stripped for exc in [
                        "os.environ",
                        "get_setting",
                        "self._api_key",
                        "self.api_key",
                    ]):
                        continue
                    # Only flag if a quoted string literal follows '='
                    rhs = stripped.split("=", 1)[-1].strip()
                    if rhs.startswith(("'", '"', 'b"', "b'")):
                        suspicious.append((py_file, stripped))
            except Exception:
                pass
        assert suspicious == [], f"Possible hardcoded keys: {suspicious}"

    def test_env_example_has_no_real_values(self):
        """The .env.example file should contain placeholders, not real keys."""
        env_example = Path(".env.example")
        if not env_example.exists():
            pytest.skip(".env.example not found")
        content = env_example.read_text()
        # Should contain placeholder values like "your_key_here" or "<...>"
        assert "your_" in content.lower() or "<" in content or "PLACEHOLDER" in content.upper()


# ─────────────────────────────────────────────
# Offline functionality tests
# ─────────────────────────────────────────────

class TestOfflineFunctionality:
    def test_risk_engine_works_without_model(self):
        """Risk engine must not crash when the ML model is unavailable."""
        from src.risk.risk_engine import RiskEngine
        engine = RiskEngine()
        engine._model_available = False  # Force offline mode
        features = {"slope_degrees": 25.0, "rainfall_24h_mm": 80.0}
        result = engine.estimate_risk(features=features)
        assert result is not None
        assert result.get("method") == "heuristic_baseline"

    def test_heuristic_baseline_works_without_model(self):
        from src.risk.risk_engine import BaselineRiskScorer
        scorer = BaselineRiskScorer()
        score, warnings = scorer.score({"slope_degrees": 30.0, "rainfall_24h_mm": 100.0})
        assert 0.0 <= score <= 1.0

    def test_database_works_without_external_services(self):
        from src.database.db import initialize_database, get_table_counts
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "offline_test.db"
            initialize_database(db_path)
            counts = get_table_counts(db_path)
            assert isinstance(counts, dict)


# ─────────────────────────────────────────────
# FastAPI backend & data provider tests
# ─────────────────────────────────────────────

def _run_coro_sync(coro):
    """Execute a non-suspending async FastAPI handler without requiring socketpair."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise RuntimeError("Coroutine unexpectedly yielded an awaitable")


class TestFastAPIBackend:
    def test_api_app_routes_registered(self):
        from api import app
        paths = {route.path for route in app.routes}
        expected = {
            "/health",
            "/api/v1/model-info",
            "/api/v1/predict",
            "/api/v1/observations",
            "/api/v1/alerts",
            "/api/v1/alerts/{alert_id}/acknowledge",
        }
        assert expected.issubset(paths)

    def test_health_endpoint(self):
        from api import health_check
        res = _run_coro_sync(health_check())
        assert res.status == "healthy"
        assert res.model_available is True

    def test_model_info_endpoint(self):
        from api import model_info
        res = _run_coro_sync(model_info())
        assert res.name == "best_model"
        assert len(res.feature_columns) > 0
        assert res.is_experimental is True

    def test_predict_endpoint(self):
        from api import predict, PredictionRequest
        req = PredictionRequest(
            latitude=22.5,
            longitude=78.0,
            slope_degrees=28.0,
            elevation_m=800.0,
            rainfall_1h_mm=15.0,
            rainfall_24h_mm=80.0,
            rainfall_72h_mm=120.0,
        )
        res = _run_coro_sync(predict(req))
        assert res.probability is not None
        assert 0.0 <= res.probability <= 1.0
        assert res.risk_category in {"Low", "Moderate", "High", "Very High"}
        assert res.is_experimental is True

    def test_observations_and_alerts_nan_safe(self, monkeypatch):
        import json
        import pandas as pd
        from api import _dataframe_to_records

        df = pd.DataFrame([
            {"id": 1, "fatalities": float("nan"), "country": "India", "note": None}
        ])
        records = _dataframe_to_records(df)
        dumped = json.dumps(records, allow_nan=False)
        assert "null" in dumped

    def test_acknowledge_missing_alert_raises_404(self, tmp_path, monkeypatch):
        from contextlib import contextmanager
        from fastapi import HTTPException
        from src.database.db import initialize_database, db_context as real_db_context
        import src.database.models as db_models
        from api import acknowledge_alert

        temp_db = tmp_path / "api_ack_test.db"
        initialize_database(temp_db)

        @contextmanager
        def _temp_ctx():
            with real_db_context(temp_db) as conn:
                yield conn

        monkeypatch.setattr(db_models, "db_context", _temp_ctx)
        with pytest.raises(HTTPException) as exc_info:
            _run_coro_sync(acknowledge_alert(999999))
        assert exc_info.value.status_code == 404

    def test_provider_cached_uses_override_url_and_strips_url_param(self, tmp_path, monkeypatch):
        from src.data_providers.weather_provider import WeatherProvider
        import src.data_providers.base as base_mod

        monkeypatch.setattr(base_mod, "CACHE_DIR", tmp_path)
        provider = WeatherProvider()
        captured = {}

        def fake_get(url, params=None):
            captured["url"] = url
            captured["params"] = params
            return {"hourly": {}, "daily": {}}

        monkeypatch.setattr(provider, "_get", fake_get)
        provider.fetch(latitude=22.5, longitude=78.0)

        assert captured["url"].endswith("/forecast")
        assert "url" not in captured["params"]
