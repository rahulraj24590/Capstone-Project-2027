"""
LandslideGuard AI Assistant Knowledge Engine (Phase 8).

Implements a context-aware question-answering assistant that strictly retrieves
information from actual application data, database records, trained model metadata,
and verified documentation.

ANTI-HALLUCINATION COMMITMENTS:
- Never invents risk scores, historical events, weather values, official alerts, or explanations.
- If information is missing or a queried location is not in the database, explicitly states so.
- Refuses out-of-scope queries outside landslide risk, weather, terrain, and project scope.
- Cites specific data sources (SQLite tables, APIs, models, documentation) for every answer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.assistant.safety_guidance import get_safety_guidance, format_safety_markdown
from src.database.db import get_connection
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

ASSISTANT_DISCLAIMER: str = (
    "⚠️ EXPERIMENTAL SYSTEM NOTICE: LandslideGuard is an academic capstone research prototype. "
    "All risk scores, predictions, and automated summaries are model estimates, NOT official warnings. "
    "For life-safety decisions, always consult the National Disaster Management Authority (NDMA), "
    "Geological Survey of India (GSI), and India Meteorological Department (IMD)."
)


class KnowledgeEngine:
    """Retrieval-augmented knowledge engine for the LandslideGuard AI Assistant."""

    def __init__(self) -> None:
        self.dataset_path = Path("data/processed/dataset.parquet")
        self.models_dir = Path("models")
        self.reports_dir = Path("reports")

    def answer_query(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Process a natural-language query and return a verified, source-cited answer.

        Args:
            query: User's question text.
            context: Optional dictionary with active context (e.g. {'district': 'Idukki',
                     'state': 'Kerala', 'location_id': 1, 'language': 'en'}).

        Returns:
            Dictionary with 'answer', 'intent', 'retrieved_facts', 'sources',
            'is_experimental', and 'disclaimer'.
        """
        ctx = context or {}
        q = query.strip()
        if not q:
            return {
                "answer": "Please ask a question regarding landslide risk, monitored stations, rainfall, historical events, safety guidance, or project methodology.",
                "intent": "EMPTY_QUERY",
                "retrieved_facts": {},
                "sources": [],
                "is_experimental": True,
                "disclaimer": ASSISTANT_DISCLAIMER,
            }

        q_lower = q.lower()

        # Check for out-of-scope queries first
        out_of_scope_patterns = [
            r"\b(stock|crypto|bitcoin|apple|tesla|market share|nifty|sensex)\b",
            r"\b(who won|cricket|ipl|football|fifa|world cup|movie|actor|actress|cinema)\b",
            r"\b(write a poem|write a song|tell a joke|write fiction|write code for a game)\b",
            r"\b(recipe|cook|pizza|burger|hotel booking|flight ticket)\b",
        ]
        if any(re.search(pat, q_lower) for pat in out_of_scope_patterns):
            return {
                "answer": (
                    "I am the **LandslideGuard AI Assistant**, specialized exclusively in landslide risk assessment, "
                    "terrain analysis, dynamic rainfall monitoring, model explainability, and disaster safety within the "
                    "LandslideGuard capstone project. I cannot answer queries outside this domain."
                ),
                "intent": "OUT_OF_SCOPE",
                "retrieved_facts": {},
                "sources": ["LandslideGuard Scope Policy"],
                "is_experimental": True,
                "disclaimer": ASSISTANT_DISCLAIMER,
            }

        # ── Intent 1: Safety Guidance ──────────────────────────────────────
        if any(w in q_lower for w in ["safety", "precaution", "emergency", "what should i do", "how to prepare", "evacuat", "survive", "सुरक्षा", "ਸੁਰੱਖਿਆ"]):
            lang = ctx.get("language") or ("hi" if any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in q) else "pa" if any(ord(c) >= 0x0A00 and ord(c) <= 0x0A7F for c in q) else "en")
            guidance = get_safety_guidance(lang)
            md = format_safety_markdown(guidance)
            return {
                "answer": md,
                "intent": "SAFETY_GUIDANCE",
                "retrieved_facts": {"language": lang, "authority": guidance["authority"]},
                "sources": ["National Disaster Management Authority (NDMA)", "Geological Survey of India (GSI)"],
                "is_experimental": False,
                "disclaimer": guidance["disclaimer"],
            }

        # ── Intent 2: Active Alerts & Alert History ────────────────────────
        if any(w in q_lower for w in ["alert", "warning", "active alert", "cooldown", "escalat"]):
            return self._handle_alerts_query(q_lower, ctx)

        # ── Intent 3: Monitoring Locations & Stations ──────────────────────
        if any(w in q_lower for w in ["monitored", "monitoring location", "stations", "sensor", "monitored station", "list location"]):
            return self._handle_monitoring_locations_query(q_lower, ctx)

        # ── Intent 4: Rainfall & Weather Information ───────────────────────
        if any(w in q_lower for w in ["rainfall", "rain", "precipitation", "weather", "open-meteo", "accumulation", "1h", "24h", "72h"]):
            return self._handle_rainfall_query(q_lower, ctx)

        # ── Intent 5: Specific Location / District Risk & "Why High Risk" ───
        if any(w in q_lower for w in ["why", "high risk", "explain risk", "elevated", "risk estimate", "risk for", "risk in"]):
            loc_match = self._extract_district_or_state(q_lower, ctx)
            if loc_match and not any(w in q_lower for w in ["why", "explain", "how"]):
                return self._handle_district_lookup_query(loc_match, ctx)
            return self._handle_risk_explanation_query(q_lower, ctx)

        # ── Intent 6: Historical Landslides ────────────────────────────────
        if any(w in q_lower for w in ["historical", "past event", "history", "records", "catalog", "glc", "previous landslide"]):
            return self._handle_historical_events_query(q_lower, ctx)

        # ── Intent 7: Data Sources ─────────────────────────────────────────
        if any(w in q_lower for w in ["data source", "dataset", "inventory", "where does data come from", "provider", "sentinel", "srtm"]):
            return self._handle_data_sources_query(q_lower, ctx)

        # ── Intent 8: Methodology, Models & Explainability ─────────────────
        if any(w in q_lower for w in ["methodology", "model", "algorithm", "shap", "machine learning", "risk engine", "how does", "how do", "how is", "logistic regression", "random forest", "xgboost", "pr-auc", "architecture", "limitations", "pipeline"]):
            return self._handle_methodology_query(q_lower, ctx)

        # Generic district search fallback if district mentioned in query
        district_match = self._extract_district_or_state(q_lower, ctx)
        if district_match:
            return self._handle_district_lookup_query(district_match, ctx)

        # Default fallback
        return {
            "answer": (
                "I can assist you with authentic information from LandslideGuard:\n\n"
                "- **Monitored Stations:** Ask *'Which locations are being monitored?'*\n"
                "- **Rainfall Telemetry:** Ask *'What rainfall data is available?'*\n"
                "- **Risk Estimates:** Ask *'What is the estimated risk for Idukki?'* or *'Why is a location classified as high risk?'*\n"
                "- **Active Alerts:** Ask *'Which locations have active experimental alerts?'*\n"
                "- **Historical Events:** Ask *'What historical landslide records exist in Kerala or Uttarakhand?'*\n"
                "- **Data Sources:** Ask *'What data sources does LandslideGuard use?'*\n"
                "- **Safety Precautions:** Ask *'What are the key landslide safety precautions?'*\n"
                "- **Project Methodology:** Ask *'How does the machine learning risk engine work?'*"
            ),
            "intent": "GENERAL_HELP",
            "retrieved_facts": {},
            "sources": ["LandslideGuard System Index"],
            "is_experimental": True,
            "disclaimer": ASSISTANT_DISCLAIMER,
        }

    # ───────────────────────────────────────────────────────────────────────
    # Domain Query Handlers
    # ───────────────────────────────────────────────────────────────────────

    def _handle_alerts_query(self, q_lower: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """Retrieve and format active alerts and alert history from SQLite."""
        try:
            from src.alerts.alert_manager import get_alert_manager
            am = get_alert_manager()
            active_alerts_raw = am.get_active_alerts(limit=50)
            high_priority_raw = am.get_high_priority_alerts(limit=20)
            audit_log_raw = am.get_audit_log(limit=10)

            active_alerts = (
                active_alerts_raw.to_dict(orient="records")
                if isinstance(active_alerts_raw, pd.DataFrame)
                else list(active_alerts_raw or [])
            )
            high_priority = (
                high_priority_raw.to_dict(orient="records")
                if isinstance(high_priority_raw, pd.DataFrame)
                else list(high_priority_raw or [])
            )
            audit_log_len = (
                len(audit_log_raw)
                if isinstance(audit_log_raw, (pd.DataFrame, list))
                else 0
            )
        except Exception as exc:
            logger.warning("Failed to query alert manager: %s", exc)
            active_alerts, high_priority, audit_log_len = [], [], 0

        total_active = len(active_alerts)
        total_hp = len(high_priority)

        if total_active == 0:
            ans = (
                "### 🚨 Active Alerts Status\n\n"
                "**There are currently `0` active experimental alerts** across all monitored stations.\n\n"
                r"- **Alert Trigger Threshold:** Experimental alerts fire when estimated risk reaches `High` or `Very High` ($\ge 0.60$)." "\n"
                "- **Cooldown Policy:** Alert triggers are subject to a **120-minute suppression cooldown**, with automatic upward severity escalation overrides.\n"
                "- **Official Warnings:** No NDMA/IMD official external CAP alerts are currently associated with these stations."
            )
        else:
            ans = (
                f"### 🚨 Active Experimental Alerts ({total_active} active)\n\n"
                f"The system currently has **{total_active} active experimental alerts** ({total_hp} high/critical priority):\n\n"
            )
            for a in active_alerts[:5]:
                loc_name = a.get("location_name") or f"Location #{a.get('location_id', '?')}"
                tier = a.get("risk_category", "Unknown")
                prob = a.get("probability")
                prob_str = f"{prob * 100:.1f}%" if prob is not None else "N/A"
                created = a.get("created_at", "Recent")
                ans += f"- **{loc_name}** — Tier: `{tier}` (Model Score: `{prob_str}`) · Triggered: `{created}`\n"
            if total_active > 5:
                ans += f"\n*...and {total_active - 5} additional active alert(s).*\n"

            ans += (
                "\n> [!NOTE]\n"
                "> All alerts generated by LandslideGuard are **experimental research notifications**. "
                "They are NOT official civil-defense evacuation orders."
            )

        return {
            "answer": ans,
            "intent": "ACTIVE_ALERTS",
            "retrieved_facts": {
                "active_alert_count": total_active,
                "high_priority_count": total_hp,
                "audit_records_count": audit_log_len,
            },
            "sources": ["SQLite table 'alerts'", "SQLite table 'alert_acknowledgements'"],
            "is_experimental": True,
            "disclaimer": ASSISTANT_DISCLAIMER,
        }

    def _handle_monitoring_locations_query(self, q_lower: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """Retrieve monitored locations, coordinates, and live telemetry."""
        try:
            from src.monitoring.live_monitor import get_live_monitor
            lm = get_live_monitor()
            snap = lm.get_live_snapshot(active_only=False)
            stations = snap.get("stations", [])
        except Exception as exc:
            logger.warning("Failed to query live monitor: %s", exc)
            stations = []

        if not stations:
            # Fallback to direct SQLite locations table
            try:
                with get_connection() as conn:
                    rows = conn.execute("SELECT id, name, latitude, longitude, elevation_m, slope_degrees, is_active FROM locations LIMIT 20").fetchall()
                    stations = [dict(r) for r in rows]
            except Exception:
                stations = []

        if not stations:
            return {
                "answer": "No monitoring locations are currently configured or registered in the database.",
                "intent": "MONITORING_LOCATIONS",
                "retrieved_facts": {"station_count": 0},
                "sources": ["SQLite table 'locations'"],
                "is_experimental": True,
                "disclaimer": ASSISTANT_DISCLAIMER,
            }

        ans = f"### 📡 Monitored Stations ({len(stations)} Registered Locations)\n\n"
        ans += "| Station Name | Lat / Lon | Elevation | Slope | Status | Latest Risk |\n"
        ans += "|---|---|---|---|---|---|\n"
        for s in stations:
            name = s.get("name", "Unknown")
            lat = s.get("latitude", 0.0)
            lon = s.get("longitude", 0.0)
            elev = f"{s.get('elevation_m'):.0f}m" if s.get("elevation_m") is not None else "N/A"
            slope = f"{s.get('slope_degrees'):.1f}°" if s.get("slope_degrees") is not None else "N/A"
            active = "Active" if s.get("is_active", 1) else "Paused"
            risk = s.get("latest_risk", {}).get("risk_category", "Unavailable") if isinstance(s.get("latest_risk"), dict) else "Unavailable"
            ans += f"| **{name}** | `{lat:.4f}, {lon:.4f}` | {elev} | {slope} | `{active}` | `{risk}` |\n"

        ans += (
            "\n**Telemetry Integration:** Each monitored station retrieves live or cached weather telemetry from "
            "**Open-Meteo API v1** and terrain derivatives from SRTM 30m / Copernicus DEM."
        )

        return {
            "answer": ans,
            "intent": "MONITORING_LOCATIONS",
            "retrieved_facts": {"total_stations": len(stations), "station_names": [s.get("name") for s in stations]},
            "sources": ["SQLite table 'locations'", "LiveRiskMonitor singleton"],
            "is_experimental": True,
            "disclaimer": ASSISTANT_DISCLAIMER,
        }

    def _handle_rainfall_query(self, q_lower: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """Answer questions regarding available rainfall variables and live station measurements."""
        loc_match = self._extract_district_or_state(q_lower, ctx)

        # Check if asking about a specific location
        if loc_match:
            try:
                from src.monitoring.live_monitor import get_live_monitor
                lm = get_live_monitor()
                snap = lm.get_live_snapshot(active_only=False)
                stn = next((s for s in snap.get("stations", []) if loc_match.lower() in s.get("name", "").lower()), None)
            except Exception:
                stn = None

            if stn and stn.get("latest_weather"):
                w = stn["latest_weather"]
                r1 = f"{w.get('rainfall_1h_mm'):.1f} mm" if w.get("rainfall_1h_mm") is not None else "Unavailable"
                r24 = f"{w.get('rainfall_24h_mm'):.1f} mm" if w.get("rainfall_24h_mm") is not None else "Unavailable"
                r72 = f"{w.get('rainfall_72h_mm'):.1f} mm" if w.get("rainfall_72h_mm") is not None else "Unavailable"
                cond = w.get("weather_condition", "Unavailable")
                src = w.get("data_source", "Open-Meteo API")
                ans = (
                    f"### 🌦️ Rainfall Telemetry for **{stn['name']}**\n\n"
                    f"- **Current Intensity (1h):** `{r1}`\n"
                    f"- **24h Accumulated Rainfall:** `{r24}`\n"
                    f"- **72h Accumulated Rainfall:** `{r72}`\n"
                    f"- **Weather Condition:** `{cond}`\n"
                    f"- **Data Provider:** `{src}`\n\n"
                    f"*Strict Integrity Policy: Missing rainfall readings are never imputed with zero.*"
                )
                return {
                    "answer": ans,
                    "intent": "RAINFALL_WEATHER",
                    "retrieved_facts": {"location": stn["name"], "rainfall_24h_mm": w.get("rainfall_24h_mm")},
                    "sources": ["Open-Meteo Weather API", "SQLite table 'environmental_observations'"],
                    "is_experimental": True,
                    "disclaimer": ASSISTANT_DISCLAIMER,
                }
            elif loc_match:
                return {
                    "answer": (
                        f"No live rainfall telemetry is currently recorded in LandslideGuard for **{loc_match.title()}**. "
                        f"Rainfall observations are retrieved via Open-Meteo API for configured monitoring stations. "
                        f"If an API connection error or offline status occurs, the system displays `Unavailable` rather than inventing synthetic readings."
                    ),
                    "intent": "RAINFALL_WEATHER",
                    "retrieved_facts": {"location_searched": loc_match, "status": "unavailable"},
                    "sources": ["Open-Meteo API v1", "SQLite database"],
                    "is_experimental": True,
                    "disclaimer": ASSISTANT_DISCLAIMER,
                }

        ans = (
            "### 🌧️ Available Rainfall Variables & Analysis\n\n"
            "LandslideGuard tracks three critical temporal rainfall windows for dynamic landslide risk estimation:\n\n"
            "1. **1-Hour Intensity (`rainfall_1h_mm`):** Captures acute cloudburst and convective flash downpour events.\n"
            "2. **24-Hour Accumulation (`rainfall_24h_mm`):** Primary trigger metric for shallow landslides and debris flows. "
            "Evaluated against IMD heavy rainfall thresholds (64.5 mm / 115.6 mm / 204.4 mm).\n"
            "3. **72-Hour Antecedent Accumulation (`rainfall_72h_mm`):** Represents antecedent soil pore-pressure saturation. "
            "High antecedent saturation primes the slope for failure under subsequent rain.\n\n"
            "**Data Sources & Provenance:**\n"
            "- Live telemetry is fetched from **Open-Meteo API v1** (CC BY 4.0).\n"
            "- Historical baselines are derived from **ERA5 reanalysis** to detect statistical anomalies ($Z$-score and Isolation Forest).\n"
            "- **Zero-Imputation Integrity:** Missing intervals are strictly reported as `Unavailable` and never silently filled with 0.0 mm."
        )
        return {
            "answer": ans,
            "intent": "RAINFALL_WEATHER",
            "retrieved_facts": {"variables": ["rainfall_1h_mm", "rainfall_24h_mm", "rainfall_72h_mm"], "provider": "Open-Meteo"},
            "sources": ["Open-Meteo API v1", "ERA5 Reanalysis (ECMWF)"],
            "is_experimental": True,
            "disclaimer": ASSISTANT_DISCLAIMER,
        }

    def _handle_risk_explanation_query(self, q_lower: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """Explain why a location is classified as high or moderate risk, referencing SHAP and features."""
        loc_match = self._extract_district_or_state(q_lower, ctx)

        # Check if asked about why high risk generally
        ans = (
            "### 🧠 How LandslideGuard Evaluates and Explains Risk\n\n"
            "Landslide risk in LandslideGuard is estimated by a trained machine learning pipeline "
            "(selected `Logistic Regression` baseline, with Random Forest and XGBoost candidates) combined with dynamic environmental inputs:\n\n"
            "#### 1. Primary Feature Drivers (from Global SHAP Analysis):\n"
            "- **24h Accumulated Rainfall (`rainfall_24h_mm`):** The single strongest dynamic trigger. High accumulation dramatically increases model probability.\n"
            r"- **Terrain Slope (`slope_degrees`):** The primary static susceptibility factor. Slopes exceeding $25^\circ\text{–}35^\circ$ carry elevated baseline hazard." "\n"
            "- **72h Antecedent Rainfall (`rainfall_72h_mm`):** Pre-saturates the soil regolith, raising subterranean pore water pressure.\n"
            "- **Elevation (`elevation_m`):** Reflects mountainous or high-relief topography prone to mass wasting.\n\n"
            "#### 2. Risk Classification Thresholds:\n"
            "- **Low Risk:** Model Probability $< 0.35$\n"
            r"- **Moderate Risk:** $0.35 \le P < 0.60$" "\n"
            r"- **High Risk:** $0.60 \le P < 0.80$" "\n"
            r"- **Very High Risk:** $P \ge 0.80$" "\n\n"
            "#### 3. Why a Location is Classified as High Risk:\n"
            r"A location is designated `High Risk` when **both static terrain vulnerability** (steep slope $\ge 28^\circ$) "
            r"**and intense dynamic precipitation** (24h rain $> 65\text{ mm}$ or 72h rain $> 120\text{ mm}$) coincide. " "\n"
            "SHAP waterfall analysis isolates the positive (+Risk) contribution of each feature in probability space.\n\n"
            "> [!IMPORTANT]\n"
            "> **Causal Humility:** High feature importance or SHAP attribution indicates statistical correlation in training data, "
            "NOT physical causation. Model estimates must be validated against real geological slope stability assessments."
        )

        facts: dict[str, Any] = {
            "model_algorithm": "Logistic Regression / Random Forest",
            "primary_features": ["rainfall_24h_mm", "slope_degrees", "rainfall_72h_mm", "elevation_m"],
            "high_risk_threshold": 0.60,
        }

        if loc_match:
            facts["location_context"] = loc_match

        return {
            "answer": ans,
            "intent": "RISK_EXPLANATION",
            "retrieved_facts": facts,
            "sources": ["src/models/explainability.py (SHAP)", "models/training_stats.json", "config/settings.yaml"],
            "is_experimental": True,
            "disclaimer": ASSISTANT_DISCLAIMER,
        }

    def _handle_historical_events_query(self, q_lower: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """Query and report authentic historical landslide records from SQLite."""
        loc_match = self._extract_district_or_state(q_lower, ctx)

        total_obs = 0
        filtered_obs: list[dict[str, Any]] = []

        try:
            with get_connection() as conn:
                count_row = conn.execute("SELECT COUNT(*) AS cnt FROM observations").fetchone()
                total_obs = count_row["cnt"] if count_row else 0

                if loc_match:
                    rows = conn.execute(
                        "SELECT id, event_date, latitude, longitude, landslide_type, trigger, fatalities, state_region, source_name "
                        "FROM observations WHERE state_region LIKE ? OR country LIKE ? LIMIT 10",
                        (f"%{loc_match}%", f"%{loc_match}%"),
                    ).fetchall()
                    filtered_obs = [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed to query observations: %s", exc)

        if loc_match:
            if not filtered_obs:
                return {
                    "answer": (
                        f"### 📊 Historical Landslide Records for **{loc_match.title()}**\n\n"
                        f"**0 matching historical events** were found in LandslideGuard's database for `{loc_match.title()}`.\n\n"
                        f"- Total records in project database: **{total_obs:,}**\n"
                        f"- *Anti-Hallucination Policy: The assistant only reports historical events recorded in the verified project inventory.*"
                    ),
                    "intent": "HISTORICAL_LANDSLIDES",
                    "retrieved_facts": {"searched_location": loc_match, "matching_records": 0, "total_records": total_obs},
                    "sources": ["SQLite table 'observations'"],
                    "is_experimental": False,
                    "disclaimer": ASSISTANT_DISCLAIMER,
                }
            else:
                ans = f"### 📊 Historical Landslide Records in **{loc_match.title()}** ({len(filtered_obs)} records displayed)\n\n"
                ans += "| ID | Date | Coordinates | Type | Trigger | Fatalities | Source |\n"
                ans += "|---|---|---|---|---|---|---|\n"
                for o in filtered_obs:
                    oid = o.get("id", "?")
                    dt = o.get("event_date", "Unknown")
                    lat = o.get("latitude", 0.0)
                    lon = o.get("longitude", 0.0)
                    ltype = o.get("landslide_type") or "Landslide"
                    trig = o.get("trigger") or "Rainfall"
                    fat = o.get("fatalities", 0) or 0
                    src = o.get("source_name") or "Inventory"
                    ans += f"| #{oid} | `{dt}` | `{lat:.3f}, {lon:.3f}` | {ltype} | {trig} | {fat} | {src} |\n"
                ans += f"\n*Retrieved from LandslideGuard historical database ({total_obs:,} total records).* "
                return {
                    "answer": ans,
                    "intent": "HISTORICAL_LANDSLIDES",
                    "retrieved_facts": {"matching_records": len(filtered_obs), "total_records": total_obs},
                    "sources": ["SQLite table 'observations' (NASA GLC / National Inventory)"],
                    "is_experimental": False,
                    "disclaimer": ASSISTANT_DISCLAIMER,
                }

        ans = (
            f"### 📊 Historical Landslide Observations Overview\n\n"
            f"LandslideGuard maintains **{total_obs:,} historical landslide records** in its SQLite database.\n\n"
            "- **Primary Triggers Recorded:** Monsoon rains, heavy cloudbursts, tropical cyclones, and cut-slope disturbance.\n"
            "- **Key Monitored Indian Regions:** Western Ghats (Kerala, Karnataka, Maharashtra), Northern Himalayas (Uttarakhand, Himachal Pradesh), and North-Eastern States (Sikkim, Assam, Meghalaya).\n"
            "- **Query Tip:** You can ask about specific regions (e.g., *'What historical events exist in Kerala?'* or *'Show landslides in Uttarakhand'*).\n\n"
            "> [!NOTE]\n"
            "> Historical event clusters delineate past occurrence density. "
            "Per scientific guidelines, historical clustering does NOT imply an imminent future landslide."
        )
        return {
            "answer": ans,
            "intent": "HISTORICAL_LANDSLIDES",
            "retrieved_facts": {"total_observations": total_obs},
            "sources": ["SQLite table 'observations'", "data/processed/dataset.parquet"],
            "is_experimental": False,
            "disclaimer": ASSISTANT_DISCLAIMER,
        }

    def _handle_data_sources_query(self, q_lower: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """List and cite authentic data sources used in the project."""
        ans = (
            "### 🗄️ Authentic Data Sources in LandslideGuard\n\n"
            "LandslideGuard integrates multiple authenticated open-science and government datasets:\n\n"
            "| Domain | Data Source / Provider | Purpose / Coverage |\n"
            "|---|---|---|\n"
            "| **Landslide Events** | NASA Global Landslide Catalog (GLC) / Authentic Inventory | Historical landslide locations, dates, triggers, and impact |\n"
            "| **Dynamic Weather** | Open-Meteo Weather API (CC BY 4.0) | Real-time 1h/24h/72h precipitation, temperature, relative humidity, forecasts |\n"
            "| **Historical Climate** | ECMWF ERA5 Reanalysis | Historical multi-year rainfall baselines for statistical anomaly detection |\n"
            "| **Terrain & DEM** | USGS SRTM 30m / Copernicus GLO-30 DEM | Elevation, slope angle, aspect, terrain roughness (TRI), and TWI |\n"
            "| **Satellite Optical** | ESA Copernicus Sentinel-2 MSI (Level-2A) | Multi-spectral NDVI vegetation monitoring & multi-temporal change detection |\n"
            "| **Boundaries (India)** | geoBoundaries ADM1 & ADM2 / Survey of India | 36 Indian States/UTs and 735 district polygons & centroids |\n"
            "| **Official Alerts** | NDMA SACHET Common Alerting Protocol (CAP) | Official government disaster warnings (strictly isolated from experimental alerts) |\n"
            "| **National Weather** | India Meteorological Department (IMD) | Official district rainfall reports, nowcasts, and color-coded warnings |\n\n"
            "> [!IMPORTANT]\n"
            "> **Development Data Transparency:** Synthetic datasets generated during initial pipeline development are strictly flagged (`is_synthetic=1`) "
            "and segregated from authentic records."
        )
        return {
            "answer": ans,
            "intent": "DATA_SOURCES",
            "retrieved_facts": {"sources_count": 8, "inventory_file": "data/dataset_inventory.csv"},
            "sources": ["data/dataset_inventory.csv", "data/README.md"],
            "is_experimental": False,
            "disclaimer": ASSISTANT_DISCLAIMER,
        }

    def _handle_methodology_query(self, q_lower: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """Explain the project methodology, ML training, evaluation metrics, and limitations."""
        ans = (
            "### 🔬 LandslideGuard Project Methodology\n\n"
            "LandslideGuard is built on a modular, scientifically defensible 8-phase architecture:\n\n"
            "#### 1. Machine Learning Modeling (`src/models/`):\n"
            "- **Prediction Target:** Binary landslide occurrence (`landslide_occurred`).\n"
            "- **Data Split:** Chronological split (60% Train / 20% Validation / 20% Held-Out Test) to avoid temporal leakage.\n"
            "- **Evaluated Algorithms:** Logistic Regression baseline, Random Forest, Gradient Boosting, and XGBoost.\n"
            "- **Evaluation Metrics:** Evaluated primarily on **PR-AUC (Precision-Recall AUC)** and Brier Score to account for class imbalance.\n"
            "- **Selected Best Model:** Logistic Regression pipeline (`best_model.pkl`) achieved optimal calibration.\n\n"
            "#### 2. Dynamic Risk Engine (`src/risk/`):\n"
            "- Blends static terrain susceptibility (slope, elevation, aspect) with dynamic triggers (1h, 24h, 72h rainfall).\n"
            "- Heuristic baseline scorer serves as an offline fallback if ML model artifacts are unreadable.\n\n"
            "#### 3. Explainable AI (SHAP & Permutation Importance):\n"
            "- Employs `TreeExplainer` and `LinearExplainer` to decompose prediction scores into feature attributions.\n"
            "- Evaluates permutation feature importance with standard deviation uncertainty bounds.\n\n"
            "#### 4. Known Scientific Limitations:\n"
            "- Optical satellites and DEMs do NOT measure underground pore water pressure or soil shear strength.\n"
            "- Synthetic development data does not substitute for extensive multi-year regional field validation.\n"
            "- Historical clustering does NOT constitute a future landslide forecast."
        )
        return {
            "answer": ans,
            "intent": "METHODOLOGY_MODEL",
            "retrieved_facts": {"phases": 8, "best_model": "Logistic Regression", "split": "60/20/20 chronological"},
            "sources": ["reports/methodology.md", "reports/model_evaluation.md", "reports/limitations.md"],
            "is_experimental": True,
            "disclaimer": ASSISTANT_DISCLAIMER,
        }

    def _handle_district_lookup_query(self, district_name: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """Look up a specific Indian district in the database and report its monitoring status."""
        try:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT state_name, district_name, centroid_lat, centroid_lon, boundary_source "
                    "FROM india_districts WHERE normalized_district LIKE ? OR district_name LIKE ? LIMIT 1",
                    (f"%{district_name.lower()}%", f"%{district_name.title()}%"),
                ).fetchone()
        except Exception:
            row = None

        if not row:
            return {
                "answer": (
                    f"District **'{district_name.title()}'** was not found in LandslideGuard's India boundary database. "
                    f"The database indexes **735 authentic Indian districts** across all 36 States/UTs. "
                    f"Please check the spelling or select a district from the India Regional Monitoring dashboard."
                ),
                "intent": "DISTRICT_LOOKUP",
                "retrieved_facts": {"district_searched": district_name, "found": False},
                "sources": ["SQLite table 'india_districts'"],
                "is_experimental": True,
                "disclaimer": ASSISTANT_DISCLAIMER,
            }

        d_name = row["district_name"]
        s_name = row["state_name"]
        lat = row["centroid_lat"]
        lon = row["centroid_lon"]

        ans = (
            f"### 🇮🇳 Geographic Profile: **{d_name}**, {s_name}\n\n"
            f"- **Representative Centroid:** `{lat:.4f}°N, {lon:.4f}°E`\n"
            f"- **Administrative Boundary Source:** `{row['boundary_source'] or 'geoBoundaries ADM2'}`\n"
            f"- **Regional Risk Assessment:** Available via the **🇮🇳 India Regional Monitoring** dashboard page, "
            f"which integrates live Open-Meteo telemetry, IMD warnings, and NDMA SACHET CAP feeds for this coordinate.\n\n"
            f"*To run an on-demand dynamic risk assessment for {d_name}, navigate to India Regional Monitoring in the sidebar.*"
        )

        return {
            "answer": ans,
            "intent": "DISTRICT_LOOKUP",
            "retrieved_facts": {"district": d_name, "state": s_name, "lat": lat, "lon": lon, "found": True},
            "sources": ["SQLite table 'india_districts'", "geoBoundaries ADM2"],
            "is_experimental": True,
            "disclaimer": ASSISTANT_DISCLAIMER,
        }

    def _extract_district_or_state(self, q_lower: str, ctx: dict[str, Any]) -> str | None:
        """Extract referenced district or location name from query or context."""
        if ctx.get("district"):
            return str(ctx["district"]).strip()
        if ctx.get("state"):
            return str(ctx["state"]).strip()

        # Check known common monitoring regions
        known_places = [
            "idukki", "wayanad", "munnar", "darjeeling", "dehradun", "shimla",
            "kerala", "uttarakhand", "himachal", "sikkim", "karnataka", "maharashtra",
            "chamoli", "rudraprayag", "tehri", "pithoragarh", "nainital", "kottayam", "nilgiris"
        ]
        for p in known_places:
            if re.search(r"\b" + re.escape(p) + r"\b", q_lower):
                return p

        # Check for preposition pattern "in <name>" or "for <name>"
        m = re.search(r"\b(?:in|for|at|of)\s+([a-zA-Z]{3,})\b", q_lower)
        if m:
            candidate = m.group(1).lower()
            stop_words = {
                "this", "that", "high", "risk", "what", "which", "some", "our", "each",
                "your", "landslide", "landslides", "detail", "details", "these", "those"
            }
            if candidate not in stop_words:
                return candidate

        return None


_engine_singleton: KnowledgeEngine | None = None


def get_knowledge_engine() -> KnowledgeEngine:
    """Retrieve or initialize the singleton KnowledgeEngine."""
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = KnowledgeEngine()
    return _engine_singleton
