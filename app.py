"""
LandslideGuard — Main Streamlit Dashboard

Twelve-page interactive dashboard for:
1. Overview
2. Live Monitoring
3. Risk Analysis
4. Geospatial Map
5. Weather Analysis
6. India Regional Monitoring
7. Historical Events
8. Model Performance
9. AI Explainability
10. Alert Management
11. Data Management
12. Settings

Run with:
    streamlit run app.py

IMPORTANT:
- All model outputs are EXPERIMENTAL and have not been operationally validated.
- This system must NOT be used as a substitute for official geological or
  disaster-management early-warning services.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database.db import initialize_database, get_table_counts
from src.utils.logging_config import setup_logging, get_logger
from src.utils.settings import get_setting

# ─────────────────────────────────────────────
# Application bootstrap
# ─────────────────────────────────────────────

setup_logging(log_level="WARNING")
logger = get_logger(__name__)

# Ensure database is initialised on startup
initialize_database()

APP_NAME = get_setting("app", "name") or "LandslideGuard"
APP_VERSION = get_setting("app", "version") or "1.0.0"
APP_DESCRIPTION = get_setting("app", "description") or ""

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title=f"{APP_NAME} — AI Landslide Risk System",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top right, #162238 0%, #0b0f19 45%, #080b12 100%);
}

/* Main container breathing room */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

/* Sidebar styling & pill navigation */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1523 0%, #131b2e 100%);
    border-right: 1px solid #222d44;
}

section[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.35rem;
}

section[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 0.55rem 0.85rem !important;
    margin: 0 !important;
    transition: all 0.18s ease;
    cursor: pointer;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(56, 189, 248, 0.08);
    border-color: rgba(56, 189, 248, 0.25);
}

section[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(56, 189, 248, 0.18), rgba(56, 189, 248, 0.05));
    border-color: rgba(56, 189, 248, 0.45);
    box-shadow: inset 3px 0 0 #38bdf8;
}

/* KPI Metric cards */
[data-testid="stMetric"],
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #131b2e 0%, #172136 100%);
    border: 1px solid #24314a;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.22);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

[data-testid="stMetric"]:hover,
[data-testid="metric-container"]:hover {
    border-color: #38bdf866;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
}

/* Header & Cards */
.page-hero {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}

.section-header {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f1f5f9;
    margin: 0;
}

.section-sub {
    color: #94a3b8;
    font-size: 0.92rem;
    margin-top: 0.2rem;
}

.status-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    border: 1px solid #2a3954;
    background: #131b2e;
    color: #cbd5e1;
}

.pill-amber {
    background: rgba(245, 158, 11, 0.12);
    border-color: rgba(245, 158, 11, 0.4);
    color: #fbbf24;
}

.pill-blue {
    background: rgba(56, 189, 248, 0.12);
    border-color: rgba(56, 189, 248, 0.4);
    color: #38bdf8;
}

.pill-green {
    background: rgba(46, 204, 113, 0.12);
    border-color: rgba(46, 204, 113, 0.4);
    color: #4ade80;
}

.experimental-banner {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.14) 0%, rgba(180, 83, 9, 0.08) 100%);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0 1.2rem 0;
    color: #fde68a;
    font-size: 0.85rem;
    line-height: 1.45;
}

.glass-card {
    background: linear-gradient(145deg, #131b2e 0%, #162036 100%);
    border: 1px solid #24314a;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.8rem;
}

.tier-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
    margin: 0.6rem 0 0.9rem 0;
}

.tier-chip {
    border-radius: 8px;
    padding: 0.45rem 0.6rem;
    text-align: center;
    font-size: 0.78rem;
    border: 1px solid #24314a;
    background: #101727;
    color: #94a3b8;
}

.sidebar-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid #24314a;
    border-radius: 10px;
    padding: 0.8rem 0.95rem;
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* Form & tab polish */
[data-testid="stForm"] {
    background: rgba(19, 27, 46, 0.65);
    border: 1px solid #24314a;
    border-radius: 14px;
    padding: 1.25rem;
}

button[data-baseweb="tab"] {
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cached data & helper functions
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_dataset() -> pd.DataFrame | None:
    data_path = Path("data/processed/dataset.parquet")
    if not data_path.exists():
        return None
    try:
        return pd.read_parquet(data_path)
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        return None


@st.cache_resource
def load_model_info() -> dict:
    try:
        registry_path = Path("models/registry.json")
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
            models = registry.get("models", [])
            best = next((m for m in models if m["name"] == "best_model"), None)
            return best or {}
    except Exception:
        pass
    return {}


@st.cache_data(ttl=60)
def load_metrics() -> dict:
    try:
        p = Path("reports/metrics.json")
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _style_fig(fig: go.Figure, height: int = 300, showlegend: bool | None = None) -> go.Figure:
    """Apply consistent dark-theme Plotly layout styling."""
    layout_kwargs: dict = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        height=height,
        margin=dict(t=36, b=24, l=16, r=16),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    if showlegend is not None:
        layout_kwargs["showlegend"] = showlegend
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)")
    return fig


def render_page_header(title: str, subtitle: str = "", pills: list[tuple[str, str]] | None = None) -> None:
    """Render a consistent hero header with status pills."""
    pills_html = ""
    if pills:
        chips = "".join(f'<span class="pill {cls}">{label}</span>' for label, cls in pills)
        pills_html = f'<div class="status-pills">{chips}</div>'
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="page-hero"><div><h1 class="section-header">{title}</h1>{sub_html}</div>{pills_html}</div>',
        unsafe_allow_html=True,
    )


def experimental_banner(extra: str = "") -> None:
    msg = (
        "⚠️ <b>EXPERIMENTAL SYSTEM</b> — All risk estimates are research outputs "
        "from a capstone project. They have <b>NOT</b> been validated by any geological "
        "or disaster-management authority and must <b>NOT</b> be used for real evacuation "
        "or emergency decisions. " + extra
    )
    st.markdown(f'<div class="experimental-banner">{msg}</div>', unsafe_allow_html=True)


def _render_individual_shap_panel(
    exp_res: dict,
    *,
    compact: bool = False,
    show_feature_table: bool = False,
    show_concepts: bool = True,
) -> None:
    """Render individual prediction SHAP waterfall, pos/neg breakdown, missing inputs & limitations."""
    from src.models.explainability import EXPLAINABILITY_CONCEPTS

    st.caption(
        f"**Model Version:** `{exp_res.get('model_version', 'Unavailable')}` · "
        f"**Explainer:** `{exp_res.get('explainer_type') or 'Unavailable'}` · "
        f"**Cached:** `{exp_res.get('from_cache', False)}`"
    )

    if not exp_res.get("available"):
        for err in exp_res.get("errors") or ["Prediction explanation is unavailable for this input or model."]:
            st.error(err)
        return

    missing_feats = exp_res.get("missing_features") or []
    if missing_feats:
        st.warning(
            f"**Missing / Unavailable Inputs ({len(missing_feats)}):** `{', '.join(missing_feats)}` "
            "— filled with training medians for model continuity (not site measurements)."
        )

    if exp_res.get("shap_available") and exp_res.get("contributions"):
        contribs = exp_res["contributions"]
        ordered = list(reversed(contribs))
        base_v = exp_res.get("base_value") or 0.0
        wf_labels = ["Baseline"] + [c["feature"] for c in ordered] + ["Model Output"]
        wf_measures = ["absolute"] + ["relative"] * len(ordered) + ["total"]
        wf_values = [base_v] + [c["shap_value"] for c in ordered] + [0.0]

        fig_wf = go.Figure(
            go.Waterfall(
                name="SHAP Attribution",
                orientation="h",
                measure=wf_measures,
                y=wf_labels,
                x=wf_values,
                connector={"line": {"color": "#64748b", "width": 1}},
                increasing={"marker": {"color": "#ef4444"}},
                decreasing={"marker": {"color": "#22c55e"}},
                totals={"marker": {"color": "#38bdf8"}},
            )
        )
        fig_wf.update_layout(
            title=f"SHAP Waterfall Attribution ({exp_res.get('shap_unit', 'score')} units)",
            xaxis_title="Cumulative Model Output",
        )
        _style_fig(fig_wf, height=340)

        # Signed Feature Contribution Comparison Bar Chart
        c_df = pd.DataFrame(contribs).sort_values("shap_value", ascending=True)
        bar_colors = ["#ef4444" if v > 0 else "#22c55e" for v in c_df["shap_value"]]
        fig_cmp = go.Figure(
            go.Bar(
                x=c_df["shap_value"],
                y=c_df["feature"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.4f}" for v in c_df["shap_value"]],
                textposition="auto",
            )
        )
        fig_cmp.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
        fig_cmp.update_layout(
            title="Feature Contribution Comparison (+ Risk Up vs - Risk Down)",
            xaxis_title=f"SHAP Contribution ({exp_res.get('shap_unit', 'score')})",
        )
        _style_fig(fig_cmp, height=340)

        if compact:
            st.plotly_chart(fig_wf, width="stretch")
        else:
            w_col, c_col = st.columns(2)
            with w_col:
                st.plotly_chart(fig_wf, width="stretch")
            with c_col:
                st.plotly_chart(fig_cmp, width="stretch")

        p_col, n_col = st.columns(2)
        with p_col:
            st.markdown("**Pushing Risk Higher (+SHAP)**")
            pos = exp_res.get("positive_contributors", [])
            if pos:
                for c in pos[:4]:
                    st.markdown(f"- `{c['feature']}` ({c['feature_value']}): **{c['shap_value']:+.4f}**")
            else:
                st.caption("None for this observation.")
        with n_col:
            st.markdown("**Pushing Risk Lower (-SHAP)**")
            neg = exp_res.get("negative_contributors", [])
            if neg:
                for c in neg[:4]:
                    st.markdown(f"- `{c['feature']}` ({c['feature_value']}): **{c['shap_value']:+.4f}**")
            else:
                st.caption("None for this observation.")

        if show_feature_table:
            tbl_rows = [
                {
                    "Feature": c["feature"],
                    "Category": c.get("feature_category", "Model Input"),
                    "Input Value": round(float(c["feature_value"]), 4),
                    "Input Status": "Missing (Training Median Filled)" if c.get("is_imputed") else "Provided",
                    "SHAP Value": round(float(c["shap_value"]), 5),
                }
                for c in contribs
            ]
            st.dataframe(pd.DataFrame(tbl_rows), width="stretch", hide_index=True)

        st.markdown("**Plain-Language Interpretation**")
        st.info(exp_res["human_readable_summary"])
        for stmt in exp_res.get("feature_statements", []):
            st.caption(f"• {stmt}")
    else:
        st.warning(exp_res.get("shap_error") or "SHAP attribution is unavailable for this model or input.")

    limitations = exp_res.get("explanation_limitations") or []
    if limitations:
        with st.expander("⚠️ Explanation Limitations & Scientific Scope", expanded=False):
            for lim in limitations:
                st.markdown(f"- {lim}")

    if show_concepts:
        with st.expander("📘 Feature Importance vs SHAP vs Correlation vs Causation", expanded=False):
            for item in EXPLAINABILITY_CONCEPTS:
                st.markdown(f"**{item['concept']}:** {item['definition']}")
            st.caption(exp_res["scientific_disclaimer"])


# Prediction scenario presets (coordinates & plausible environmental scenarios)
SCENARIO_PRESETS: dict[str, dict[str, float]] = {
    "Custom Input": {},
    "Severe Monsoon Slope (High Risk Demo)": {
        "pred_lat": 10.0889, "pred_lon": 77.0595,
        "pred_slope": 38.0, "pred_elev": 1450.0,
        "pred_rf1": 28.0, "pred_rf24": 145.0, "pred_rf72": 290.0,
        "pred_veg": 0.22, "pred_moist": 0.78,
    },
    "Moderate Wet Hillside": {
        "pred_lat": 27.0410, "pred_lon": 88.2663,
        "pred_slope": 24.0, "pred_elev": 1100.0,
        "pred_rf1": 10.0, "pred_rf24": 55.0, "pred_rf72": 115.0,
        "pred_veg": 0.42, "pred_moist": 0.50,
    },
    "Dry Low-Elevation Plains (Low Risk Demo)": {
        "pred_lat": 22.5726, "pred_lon": 88.3639,
        "pred_slope": 5.0, "pred_elev": 45.0,
        "pred_rf1": 1.0, "pred_rf24": 8.0, "pred_rf72": 18.0,
        "pred_veg": 0.68, "pred_moist": 0.24,
    },
    "Munnar, Western Ghats (10.09°N, 77.06°E)": {
        "pred_lat": 10.0889, "pred_lon": 77.0595,
        "pred_slope": 31.0, "pred_elev": 1530.0,
        "pred_rf1": 18.0, "pred_rf24": 92.0, "pred_rf72": 185.0,
        "pred_veg": 0.35, "pred_moist": 0.64,
    },
    "Wayanad, Western Ghats (11.69°N, 76.13°E)": {
        "pred_lat": 11.6854, "pred_lon": 76.1320,
        "pred_slope": 29.0, "pred_elev": 980.0,
        "pred_rf1": 20.0, "pred_rf24": 110.0, "pred_rf72": 220.0,
        "pred_veg": 0.32, "pred_moist": 0.68,
    },
    "Darjeeling, Eastern Himalayas (27.04°N, 88.27°E)": {
        "pred_lat": 27.0410, "pred_lon": 88.2663,
        "pred_slope": 34.0, "pred_elev": 2040.0,
        "pred_rf1": 16.0, "pred_rf24": 85.0, "pred_rf72": 175.0,
        "pred_veg": 0.38, "pred_moist": 0.60,
    },
    "Dehradun Foothills (30.32°N, 78.03°E)": {
        "pred_lat": 30.3165, "pred_lon": 78.0322,
        "pred_slope": 22.0, "pred_elev": 680.0,
        "pred_rf1": 12.0, "pred_rf24": 60.0, "pred_rf72": 125.0,
        "pred_veg": 0.45, "pred_moist": 0.48,
    },
}

REGION_BOUNDS: dict[str, tuple[float, float, float, float] | None] = {
    "All Regions": None,
    "Western Ghats (8°–21°N, 72°–78°E)": (8.0, 21.0, 72.0, 78.0),
    "Himalayan Foothills (27°–35°N, 72°–89°E)": (27.0, 35.0, 72.0, 89.0),
    "North-East India (22°–29°N, 89°–97°E)": (22.0, 29.0, 89.0, 97.0),
}


def _apply_scenario_preset() -> None:
    preset_name = st.session_state.get("pred_preset", "Custom Input")
    vals = SCENARIO_PRESETS.get(preset_name, {})
    for k, v in vals.items():
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Sidebar navigation & live status
# ─────────────────────────────────────────────

PAGES = [
    "🏠 Overview",
    "📡 Live Monitoring",
    "🔮 Risk Analysis",
    "🗺️ Geospatial Map",
    "🌦️ Weather Analysis",
    "🇮🇳 India Regional Monitoring",
    "📊 Historical Events",
    "🤖 Model Performance",
    "🧠 AI Explainability",
    "🚨 Alert Management",
    "💬 AI Assistant & Vision",
    "🛡️ Disaster Response & Community",
    "🔐 Admin Panel",
    "🗄️ Data Management",
    "⚙️ Settings",
]

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 0.6rem 0 0.8rem 0;">
        <div style="font-size:2.4rem; line-height:1;">⛰️</div>
        <div style="font-size:1.3rem; font-weight:700; color:#f1f5f9; margin-top:0.3rem;">{APP_NAME}</div>
        <div style="margin-top:6px;">
            <span class="pill pill-blue">v{APP_VERSION} · Research Prototype</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio("Navigation", PAGES, label_visibility="collapsed", key="nav_page")

    st.divider()

    df_side = load_dataset()
    model_side = load_model_info()
    counts_side = get_table_counts()
    alg_side = (model_side.get("metadata", {}).get("algorithm", "") or "").replace("_", " ").title() or "Not Trained"

    try:
        from src.database.models import get_alerts as _db_get_alerts
        active_alerts_n = len(_db_get_alerts(include_acknowledged=False, limit=100))
    except Exception:
        active_alerts_n = 0

    if df_side is not None:
        n_total = len(df_side)
        n_pos = int(df_side["landslide_occurred"].sum()) if "landslide_occurred" in df_side.columns else 0
        is_synth = bool(df_side.get("is_synthetic", pd.Series([0])).all()) if "is_synthetic" in df_side.columns else False
        source_badge = '<span style="color:#fbbf24;">⚠️ Synthetic Dev Data</span>' if is_synth else '<span style="color:#4ade80;">✅ Authentic Data</span>'
        alert_color = "#f87171" if active_alerts_n > 0 else "#4ade80"
        st.markdown(f"""
        <div class="sidebar-card">
            <div style="font-weight:600; color:#e2e8f0; margin-bottom:0.35rem;">System Snapshot</div>
            📦 Records: <b style="color:#f1f5f9;">{n_total:,}</b> ({n_pos:,} events)<br>
            🤖 Model: <b style="color:#38bdf8;">{alg_side}</b><br>
            🚨 Active Alerts: <b style="color:{alert_color};">{active_alerts_n}</b><br>
            🗄️ Predictions Logged: <b style="color:#f1f5f9;">{counts_side.get("prediction_records", 0)}</b><br>
            {source_badge}
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<div style="font-size:0.72rem; color:#64748b; text-align:center;">B.Tech CSE Capstone Project<br>LandslideGuard · 2025–26</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    df = load_dataset()
    model_info = load_model_info()
    db_counts = get_table_counts()
    alg = model_info.get("metadata", {}).get("algorithm", "N/A") if model_info else "N/A"
    alg_label = alg.replace("_", " ").title() if alg != "N/A" else "Not Trained"

    dataset_pill = (
        ("📦 No Dataset Loaded", "pill")
        if df is None or df.empty
        else (
            ("📦 Synthetic Dev Dataset", "pill")
            if ("is_synthetic" in df.columns and bool(df["is_synthetic"].all()))
            else ("✅ Authentic Dataset", "pill-green")
        )
    )
    render_page_header(
        "⛰️ LandslideGuard Dashboard",
        APP_DESCRIPTION,
        pills=[
            ("⚠️ Experimental Prototype", "pill-amber"),
            (f"🤖 {alg_label}", "pill-blue"),
            dataset_pill,
        ],
    )
    experimental_banner()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n = len(df) if df is not None else 0
        st.metric("📦 Total Records", f"{n:,}")
    with c2:
        n_pos = int(df["landslide_occurred"].sum()) if df is not None and "landslide_occurred" in df.columns else 0
        pos_pct = f"{(n_pos / len(df) * 100):.1f}% rate" if df is not None and len(df) > 0 else None
        st.metric("🔴 Landslide Events", f"{n_pos:,}", delta=pos_pct, delta_color="off")
    with c3:
        st.metric("🤖 Active Model", alg_label)
    with c4:
        meta = model_info.get("metadata", {}) if model_info else {}
        tm = meta.get("test_metrics", {})
        f1 = tm.get("f1", None)
        st.metric("📈 Test F1 Score", f"{f1:.4f}" if f1 is not None else "N/A",
                  help="Computed on synthetic development test split only")
    with c5:
        st.metric("🚨 Total DB Alerts", db_counts.get("alerts", 0),
                  delta=f"{active_alerts_n} active", delta_color="off")

    st.divider()

    col_left, col_right = st.columns([1, 1.2])
    with col_left:
        st.markdown("### 📊 Observed Class Distribution")
        if df is not None and "landslide_occurred" in df.columns:
            cats = df["landslide_occurred"].map({0: "No Event", 1: "Landslide Event"})
            cat_counts = cats.value_counts().reset_index()
            cat_counts.columns = ["Outcome", "Count"]
            color_map = {"No Event": "#2ecc71", "Landslide Event": "#e74c3c"}
            fig = px.pie(
                cat_counts, values="Count", names="Outcome",
                color="Outcome", color_discrete_map=color_map, hole=0.52,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            _style_fig(fig, height=320)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No processed dataset available (`data/processed/dataset.parquet`). Run `python scripts/prepare_data.py` to generate development data.")

    with col_right:
        st.markdown("### 🗺️ Geographic Coverage")
        if df is not None and "latitude" in df.columns:
            valid_df = df.dropna(subset=["latitude", "longitude"])
            valid_df = valid_df[valid_df["latitude"].between(-90, 90) & valid_df["longitude"].between(-180, 180)]
            if not valid_df.empty:
                sample_df = valid_df.head(600).copy()
                sample_df["Event"] = sample_df["landslide_occurred"].map({0: "No Event", 1: "Landslide"})
                fig_map = px.scatter_map(
                    sample_df, lat="latitude", lon="longitude",
                    color="Event",
                    color_discrete_map={"No Event": "#2ecc71", "Landslide": "#e74c3c"},
                    hover_data=[c for c in ["slope_degrees", "rainfall_24h_mm", "elevation_m"] if c in sample_df.columns],
                    center={"lat": 22.0, "lon": 83.0},
                    zoom=3.4, height=320, map_style="carto-darkmatter", opacity=0.68,
                )
                fig_map.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=8, b=8, l=8, r=8),
                    legend=dict(bgcolor="rgba(19,27,46,0.75)", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(fig_map, width="stretch")
            else:
                st.info("No valid coordinate rows found in dataset.")
        else:
            st.info("Geographic preview unavailable until dataset is loaded (`python scripts/prepare_data.py`).")

    st.divider()
    st.markdown("### 🔬 Data Quality & Feature Summary")
    if df is not None:
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            missing_pct = df.isnull().mean().mean() * 100
            st.metric("Missing Cells", f"{missing_pct:.1f}%")
        with q2:
            st.metric("Duplicate Rows", int(df.duplicated().sum()))
        with q3:
            is_synthetic = ("is_synthetic" in df.columns and df["is_synthetic"].all())
            st.metric("Provenance", "⚠️ Synthetic" if is_synthetic else "🟢 Authentic")
        with q4:
            if "event_date" in df.columns:
                try:
                    dates = pd.to_datetime(df["event_date"], errors="coerce")
                    yr_range = f"{int(dates.dt.year.min())}–{int(dates.dt.year.max())}"
                    st.metric("Temporal Span", yr_range)
                except Exception:
                    st.metric("Temporal Span", "N/A")
    else:
        st.info("Data quality summary unavailable — no processed dataset loaded.")

    st.divider()
    st.markdown("### 🤖 Active Model Card")
    if model_info:
        meta = model_info.get("metadata", {})
        tm = meta.get("test_metrics", {})
        mi1, mi2, mi3, mi4 = st.columns(4)
        with mi1:
            st.metric("Algorithm", meta.get("algorithm", "N/A").replace("_", " ").title())
            st.caption(f"Saved: {model_info.get('saved_at', 'N/A')[:19].replace('T', ' ')} UTC")
        with mi2:
            train_s = meta.get("train_samples")
            test_s = meta.get("test_samples")
            st.metric("Features", len(meta.get("feature_columns", [])))
            st.caption(f"Train: {train_s:,} · Test: {test_s:,}" if isinstance(train_s, int) and isinstance(test_s, int) else "")
        with mi3:
            prec = tm.get("precision")
            rec = tm.get("recall")
            st.metric("Test Precision", f"{prec:.4f}" if prec is not None else "N/A")
            st.caption(f"Test Recall: {rec:.4f}" if rec is not None else "")
        with mi4:
            pr_auc = tm.get("pr_auc")
            roc_auc = tm.get("roc_auc")
            st.metric("Test PR-AUC", f"{pr_auc:.4f}" if pr_auc is not None else "N/A")
            st.caption(f"Test ROC-AUC: {roc_auc:.4f}" if roc_auc is not None else "")
        st.caption("⚠️ Metrics above are computed on **synthetic development data** and do NOT reflect real-world predictive performance.")
    else:
        st.info("No trained model found. Run: `python -m src.models.train`")


# ═══════════════════════════════════════════════════════════════
# PAGE: LIVE RISK MONITORING (PHASE 1)
# ═══════════════════════════════════════════════════════════════

elif page == "📡 Live Monitoring":
    from src.monitoring import get_live_monitor
    from src.geospatial.maps import build_landslide_map
    from streamlit_folium import st_folium

    monitor = get_live_monitor()

    render_page_header(
        "📡 Live Risk Monitoring Dashboard",
        "Continuous station surveillance combining Open-Meteo model reanalysis weather feeds, SRTM terrain attributes, and experimental RiskEngine inference.",
        pills=[("Open-Meteo Reanalysis", "pill-blue"), ("Experimental Monitoring", "pill-amber")],
    )
    experimental_banner(
        "Environmental feeds are <b>model-derived reanalysis</b> (not physical rain-gauge telemetry) and "
        "satellite NDVI currently uses a regional proxy (0.55) until satellite ingestion is connected."
    )

    # Control bar
    cb1, cb2, cb3, cb4 = st.columns([1.4, 1.2, 1.3, 1.4])
    with cb1:
        refresh_clicked = st.button("🔄 Refresh All Stations", type="primary", width="stretch", key="live_refresh_all")
    with cb2:
        seed_clicked = st.button("📍 Load Default Stations", width="stretch", key="live_seed_defaults")
    with cb3:
        auto_choice = st.selectbox(
            "Auto-Refresh",
            ["Off", "Every 60s", "Every 300s"],
            index=0,
            key="live_auto_interval",
        )
    with cb4:
        gen_alerts = st.toggle("Evaluate Alerts on Refresh", value=True, key="live_gen_alerts")

    if seed_clicked:
        with st.spinner("Registering default landslide-prone Indian monitoring stations…"):
            ids = monitor.seed_default_stations()
        st.success(f"Registered/verified {len(ids)} monitoring stations (Munnar, Wayanad, Darjeeling, Dehradun, Joshimath).")

    if refresh_clicked:
        with st.spinner("Fetching Open-Meteo feeds and scoring active stations…"):
            summary = monitor.refresh_all(active_only=True, generate_alerts=gen_alerts)
        if summary["total_stations"] == 0:
            st.warning("No active monitoring stations found. Click 'Load Default Stations' or add a custom station below.")
        else:
            msg = (
                f"Refreshed {summary['total_stations']} station(s): "
                f"{summary['live_count']} live, {summary['fallback_count']} offline fallback, "
                f"{summary['unavailable_count']} unavailable · "
                f"{summary['alerts_triggered']} alert(s) triggered."
            )
            if summary["fallback_count"] > 0 or summary["unavailable_count"] > 0:
                st.warning(msg)
            else:
                st.success(msg)

    # Register / manage custom stations
    with st.expander("➕ Register Custom Monitoring Station", expanded=False):
        with st.form("register_station_form"):
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                new_name = st.text_input("Station Name", placeholder="e.g. Idukki Ridge")
                new_state = st.text_input("State / Region", value="Kerala")
            with rc2:
                new_lat = st.number_input("Latitude (°N)", -90.0, 90.0, 9.8500, format="%.4f")
                new_lon = st.number_input("Longitude (°E)", -180.0, 180.0, 76.9700, format="%.4f")
            with rc3:
                new_elev = st.number_input("Elevation (m, optional if auto)", -500.0, 9000.0, 0.0, step=10.0)
                new_slope = st.number_input("Slope (°, optional if auto)", 0.0, 90.0, 0.0, step=0.5)
            auto_terrain = st.checkbox("Auto-fetch SRTM elevation & slope if left 0", value=True)
            add_submit = st.form_submit_button("✅ Register Station", width="stretch")

        if add_submit:
            if not new_name.strip():
                st.error("Please enter a valid station name.")
            else:
                try:
                    with st.spinner("Registering station and resolving terrain metadata…"):
                        created = monitor.register_location(
                            name=new_name.strip(),
                            latitude=float(new_lat),
                            longitude=float(new_lon),
                            state_region=new_state.strip() or None,
                            elevation_m=float(new_elev) if new_elev != 0.0 else None,
                            slope_degrees=float(new_slope) if new_slope > 0.0 else None,
                            fetch_terrain=auto_terrain,
                        )
                    st.success(f"Registered '{created['name']}' (ID {created.get('id')}).")
                    for w in created.get("warnings", []):
                        st.warning(w)
                except Exception as exc:
                    st.error(f"Could not register station: {exc}")

    interval_map = {"Off": None, "Every 60s": 60, "Every 300s": 300}
    run_every_sec = interval_map.get(auto_choice)

    def _render_live_snapshot(trigger_auto_refresh: bool = False) -> None:
        if trigger_auto_refresh:
            with st.spinner("Auto-refreshing live stations…"):
                monitor.refresh_all(active_only=True, generate_alerts=gen_alerts)

        dataset_live = load_dataset()
        model_live = load_model_info()
        summary = monitor.get_dashboard_summary(
            dataset_df=dataset_live,
            model_info=model_live,
        )
        snapshot = summary["snapshot"]
        stations = snapshot.get("stations", [])

        # ── 1. Dashboard Overview ─────────────────────────────────────
        st.markdown("### 📌 Dashboard Overview KPIs")
        ov1, ov2, ov3, ov4, ov5 = st.columns(5)
        with ov1:
            st.metric(
                "Monitored Locations",
                f"{summary['total_active_stations']}",
                delta=f"{summary['locations_assessed']} assessed" if summary["total_active_stations"] else "Unavailable",
                delta_color="off",
            )
        with ov2:
            st.metric(
                "Historical Landslide Records",
                f"{summary['total_historical_observations']:,}",
                delta=f"{summary.get('authentic_observations_count', 0):,} authentic DB"
                if summary["total_historical_observations"]
                else "Unavailable",
                delta_color="off",
                help="Count from SQLite observations table (or dataset positive events if DB table empty).",
            )
        with ov3:
            st.metric(
                "High Risk Locations",
                f"{summary['high_risk_count']}",
                delta=f"{summary['very_high_risk_count']} Very High"
                if summary["total_active_stations"]
                else "Unavailable",
                delta_color="off",
            )
        with ov4:
            st.metric(
                "Medium Risk Locations",
                f"{summary.get('moderate_risk_count', 0)}",
                help="Monitored locations currently categorized as Moderate (Medium) experimental risk.",
            )
        with ov5:
            st.metric(
                "Low Risk Locations",
                f"{summary.get('low_risk_count', 0)}",
                help="Monitored locations currently categorized as Low experimental risk.",
            )

        ov6, ov7, ov8, ov9, ov10 = st.columns(5)
        latest_wx = summary.get("latest_weather_update")
        latest_wx_str = latest_wx[:16].replace("T", " ") + " UTC" if latest_wx else "Unavailable"
        freshness_val = summary["overall_freshness"]
        fresh_color = {
            "Fresh": "#4ade80",
            "Partial Stale": "#fbbf24",
            "Offline Fallback": "#f59e0b",
            "Stale": "#f87171",
        }.get(freshness_val, "#94a3b8")

        with ov6:
            st.metric("Active Experimental Alerts", f"{summary['active_alerts_count']}")
        with ov7:
            st.markdown(
                f'<div class="glass-card"><small style="color:#94a3b8;">Latest Weather Update</small>'
                f'<br><b style="color:#f1f5f9;">{latest_wx_str}</b></div>',
                unsafe_allow_html=True,
            )
        with ov8:
            st.markdown(
                f'<div class="glass-card"><small style="color:#94a3b8;">Model & Version</small>'
                f'<br><b style="color:#38bdf8;">{summary["model_name"]}</b>'
                f' <small style="color:#94a3b8;">({summary["model_version"]})</small></div>',
                unsafe_allow_html=True,
            )
        with ov9:
            st.markdown(
                f'<div class="glass-card"><small style="color:#94a3b8;">Dataset Status</small>'
                f'<br><b style="color:#f1f5f9;">{summary["dataset_status"]}</b></div>',
                unsafe_allow_html=True,
            )
        with ov10:
            st.markdown(
                f'<div class="glass-card"><small style="color:#94a3b8;">Data Freshness</small>'
                f'<br><b style="color:{fresh_color};">● {freshness_val}</b></div>',
                unsafe_allow_html=True,
            )

        if not stations:
            st.info(
                "No monitoring stations configured yet. Click **📍 Load Default Stations** above "
                "or register a custom coordinate in **➕ Register Custom Monitoring Station**, then click **🔄 Refresh All Stations**."
            )
        else:
            # ── 2. Station Status Table & Map + Environmental Cards ─────
            st.markdown("### 📊 Monitored Station Status")
            table_rows = []
            for s in stations:
                prob = s.get("probability")
                table_rows.append({
                    "ID": s["location_id"],
                    "Station": s["name"],
                    "Region": s.get("state_region") or "—",
                    "Risk Tier": s.get("risk_category", "Not Evaluated"),
                    "Prob.": f"{prob:.3f}" if prob is not None else "—",
                    "Feed Freshness": s.get("freshness_status", "Never Synced"),
                    "Age (h)": f"{s['data_age_hours']:.1f}" if s.get("data_age_hours") is not None else "—",
                    "Rain 1h (mm)": f"{s['rainfall_1h_mm']:.1f}" if s.get("rainfall_1h_mm") is not None else "—",
                    "Rain 24h (mm)": f"{s['rainfall_24h_mm']:.1f}" if s.get("rainfall_24h_mm") is not None else "—",
                    "Rain 72h (mm)": f"{s['rainfall_72h_mm']:.1f}" if s.get("rainfall_72h_mm") is not None else "—",
                    "Wind (km/h)": f"{s['wind_speed_kmh']:.1f}" if s.get("wind_speed_kmh") is not None else "—",
                    "Temp (°C)": f"{s['temperature_c']:.1f}" if s.get("temperature_c") is not None else "—",
                    "Active Alert": "🚨 Yes" if s.get("has_active_alert") else "No",
                })
            st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

            map_col, detail_col = st.columns([1.25, 1.0])
            with map_col:
                st.markdown("### 🗺️ Live Station Risk Map")
                risk_markers = [
                    {
                        "latitude": s["latitude"],
                        "longitude": s["longitude"],
                        "risk_category": s["risk_category"],
                        "probability": s["probability"],
                        "method": s.get("method") or "unscored",
                        "is_experimental": True,
                        "location_name": s["name"],
                    }
                    for s in stations
                ]
                avg_lat = sum(s["latitude"] for s in stations) / len(stations)
                avg_lon = sum(s["longitude"] for s in stations) / len(stations)
                live_map = build_landslide_map(
                    observations_df=None,
                    risk_data=risk_markers,
                    center_lat=avg_lat,
                    center_lon=avg_lon,
                )
                st_folium(live_map, width=None, height=390, returned_objects=[])

            with detail_col:
                st.markdown("### 🔎 Selected Station & Controls")
                station_map = {f"{s['name']} (ID {s['location_id']})": s for s in stations}
                selected_label = st.selectbox("Select Station", list(station_map.keys()), key="live_selected_station")
                sel = station_map[selected_label]
                sel_id = int(sel["location_id"])

                sb1, sb2 = st.columns(2)
                with sb1:
                    if st.button("🔄 Refresh Station", key="live_refresh_one", width="stretch"):
                        with st.spinner(f"Refreshing {sel['name']}…"):
                            one_res = monitor.refresh_location(sel_id, generate_alerts=gen_alerts)
                        st.session_state[f"live_forecast_{sel_id}"] = one_res.get("forecast_summary")
                        st.success(f"{sel['name']}: {one_res['risk_category']} ({one_res['freshness_status']})")
                with sb2:
                    if st.button("⏸️ Pause Station", key="live_pause_one", width="stretch"):
                        monitor.set_location_active(sel_id, False)
                        st.info(f"Paused monitoring for {sel['name']}.")
                        st.rerun()

                cat_colors = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c", "Very High": "#9b59b6"}
                c_col = cat_colors.get(sel.get("risk_category", ""), "#64748b")
                p_str = f"{sel['probability']:.4f}" if sel.get("probability") is not None else "N/A"
                st.markdown(
                    f'<div class="glass-card" style="border-left:4px solid {c_col};">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<div><b style="font-size:1.1rem; color:#f1f5f9;">{sel["name"]}</b>'
                    f'<br><small style="color:#94a3b8;">{sel.get("state_region") or "India"} · '
                    f'Elev {sel.get("elevation_m") or "?"}m · Slope {sel.get("slope_degrees") or "?"}°</small></div>'
                    f'<div style="text-align:right;"><b style="color:{c_col}; font-size:1.15rem;">{sel["risk_category"]}</b>'
                    f'<br><small style="color:#cbd5e1;">Prob: {p_str} · {sel["freshness_status"]}</small></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                if sel.get("warnings"):
                    with st.expander(f"⚠️ Provenance & Warnings ({len(sel['warnings'])})", expanded=False):
                        for w in sel["warnings"]:
                            st.caption(f"• {w}")

            # ── Environmental Monitoring Cards ────────────────────────
            st.markdown(f"### 🌦️ Environmental Monitoring Cards — {sel['name']}")
            ec1, ec2, ec3, ec4, ec5 = st.columns(5)
            with ec1:
                v = sel.get("rainfall_1h_mm")
                st.metric("Current Rain (1h)", f"{v:.1f} mm" if v is not None else "Unavailable")
            with ec2:
                v = sel.get("rainfall_24h_mm")
                st.metric(
                    "24h Accumulated Rain",
                    f"{v:.1f} mm" if v is not None else "Unavailable",
                    help="Shown only when >=24 hourly past reanalysis slots are present.",
                )
            with ec3:
                v = sel.get("rainfall_72h_mm")
                st.metric(
                    "72h Accumulated Rain",
                    f"{v:.1f} mm" if v is not None else "Unavailable",
                    help="Shown only when >=72 hourly past reanalysis slots are present.",
                )
            with ec4:
                v = sel.get("temperature_c")
                st.metric("Temperature", f"{v:.1f} °C" if v is not None else "Unavailable")
            with ec5:
                v = sel.get("humidity_pct")
                st.metric("Relative Humidity", f"{v:.0f}%" if v is not None else "Unavailable")

            ec6, ec7, ec8, ec9, ec10 = st.columns(5)
            with ec6:
                v = sel.get("soil_moisture")
                st.metric(
                    "Soil Moisture (0–7cm)",
                    f"{v:.3f} m³/m³" if v is not None else "Unavailable",
                    help="Open-Meteo topsoil volumetric water content when present.",
                )
            with ec7:
                v = sel.get("elevation_m")
                st.metric("Elevation", f"{v:,.0f} m" if v is not None else "Unavailable")
            with ec8:
                v = sel.get("slope_degrees")
                st.metric("Terrain Slope", f"{v:.1f}°" if v is not None else "Unavailable")
            with ec9:
                wx_cond = sel.get("weather_condition") or "Unavailable"
                st.metric("Weather Condition", wx_cond)
            with ec10:
                v = sel.get("wind_speed_kmh")
                st.metric("Wind Speed", f"{v:.1f} km/h" if v is not None else "Unavailable")

            obs_type_label = (
                "Direct Observation" if sel.get("is_observed") else "Model-Derived Reanalysis (Not Forecast / Not Physical Gauge)"
            )
            src_str = sel.get("data_source") or "Unavailable (not yet synced)"
            ts_str = sel.get("observed_at") or "Unavailable"
            st.caption(f"📡 **Source:** {src_str} | **Type:** {obs_type_label} | **Last Updated:** `{ts_str}`")

            fc_summary = st.session_state.get(f"live_forecast_{sel_id}")
            if isinstance(fc_summary, dict):
                st.info(
                    f"🔮 **Upcoming Forecast Preview (Next {fc_summary['forecast_hours']}h — FORECAST ONLY, Not Observed):** "
                    f"Total expected precipitation {fc_summary['forecast_total_rain_mm']:.1f} mm · "
                    f"Peak hourly {fc_summary['forecast_max_hourly_mm']:.1f} mm/h."
                )

            # ── 3. Risk Distribution & Historical Risk Trend ──────────
            st.markdown("### 📈 Risk Distribution & Historical Trends")
            rd_col, reg_col = st.columns(2)
            with rd_col:
                dist = summary["risk_distribution"]
                pie_labels = [k for k, v in dist["counts"].items() if v > 0]
                pie_values = [dist["counts"][k] for k in pie_labels]
                if pie_values:
                    pie_df = pd.DataFrame({"Category": pie_labels, "Locations": pie_values})
                    fig_pie = px.pie(
                        pie_df,
                        names="Category",
                        values="Locations",
                        hole=0.45,
                        color="Category",
                        color_discrete_map={
                            "Low": "#2ecc71",
                            "Moderate": "#f39c12",
                            "High": "#e74c3c",
                            "Very High": "#9b59b6",
                            "Not Evaluated": "#64748b",
                        },
                        title="Station Risk Category Distribution (Low / Medium / High)",
                    )
                    _style_fig(fig_pie, height=300)
                    st.plotly_chart(fig_pie, width="stretch")
                else:
                    st.info("No stations available to plot risk distribution.")

            with reg_col:
                region_rows = summary.get("risk_by_region") or []
                if region_rows:
                    reg_df = pd.DataFrame(region_rows)
                    tier_cols = [c for c in ["Low", "Moderate", "High", "Very High", "Not Evaluated"] if c in reg_df.columns]
                    reg_melt = reg_df.melt(
                        id_vars=["state_region"],
                        value_vars=tier_cols,
                        var_name="Risk Category",
                        value_name="Stations",
                    )
                    reg_melt = reg_melt[reg_melt["Stations"] > 0]
                    if not reg_melt.empty:
                        fig_reg = px.bar(
                            reg_melt,
                            x="state_region",
                            y="Stations",
                            color="Risk Category",
                            barmode="stack",
                            title="Risk Distribution by Region",
                            labels={"state_region": "State / Region", "Stations": "Monitored Locations"},
                            color_discrete_map={
                                "Low": "#2ecc71",
                                "Moderate": "#f39c12",
                                "High": "#e74c3c",
                                "Very High": "#9b59b6",
                                "Not Evaluated": "#64748b",
                            },
                        )
                        _style_fig(fig_reg, height=300)
                        st.plotly_chart(fig_reg, width="stretch")
                    else:
                        st.info("No evaluated regional stations available yet.")
                else:
                    st.info("Regional risk distribution is unavailable without monitored locations.")

            hist_col, tr_col = st.columns(2)
            with hist_col:
                hist_dist = summary.get("historical_distribution") or {}
                if hist_dist.get("available"):
                    hist_src_tag = hist_dist.get("source_label", "Dataset")
                    if hist_dist.get("by_region"):
                        hr_df = pd.DataFrame(hist_dist["by_region"]).head(12)
                        fig_hist_dist = px.bar(
                            hr_df,
                            x="region",
                            y=["landslide_events", "non_events"],
                            barmode="stack",
                            title=f"Historical Landslide Distribution by Region ({hist_src_tag})",
                            labels={"region": "Region", "value": "Records", "variable": "Outcome"},
                        )
                        _style_fig(fig_hist_dist, height=300)
                        st.plotly_chart(fig_hist_dist, width="stretch")
                    elif hist_dist.get("by_trigger"):
                        ht_df = pd.DataFrame(hist_dist["by_trigger"]).head(10)
                        fig_hist_dist = px.bar(
                            ht_df,
                            x="trigger",
                            y="count",
                            title=f"Historical Landslide Trigger Distribution ({hist_src_tag})",
                            labels={"trigger": "Trigger", "count": "Events"},
                        )
                        _style_fig(fig_hist_dist, height=300)
                        st.plotly_chart(fig_hist_dist, width="stretch")
                    else:
                        summary_counts_df = pd.DataFrame([
                            {"Outcome": "Landslide Events", "Records": hist_dist["landslide_events"]},
                            {"Outcome": "Non-Events", "Records": hist_dist["non_events"]},
                        ])
                        fig_hist_dist = px.bar(
                            summary_counts_df,
                            x="Outcome",
                            y="Records",
                            title=f"Historical Landslide Outcomes ({hist_src_tag})",
                        )
                        _style_fig(fig_hist_dist, height=300)
                        st.plotly_chart(fig_hist_dist, width="stretch")
                else:
                    st.info("Historical landslide distribution is unavailable (no dataset or DB observations loaded).")

            with tr_col:
                from src.database.models import get_recent_predictions_with_location

                pred_trend_df = get_recent_predictions_with_location(limit=120)
                if not pred_trend_df.empty and pred_trend_df["probability"].notna().any():
                    plot_tr = pred_trend_df.sort_values("predicted_at").copy()
                    fig_tr = px.line(
                        plot_tr,
                        x="predicted_at",
                        y="probability",
                        color="location_name",
                        markers=True,
                        title="Experimental Risk Probability Trend (Logged Assessments)",
                        labels={
                            "predicted_at": "Timestamp (UTC)",
                            "probability": "Risk Probability",
                            "location_name": "Location",
                        },
                    )
                    fig_tr.update_yaxes(range=[0, 1])
                    _style_fig(fig_tr, height=300)
                    st.plotly_chart(fig_tr, width="stretch")
                else:
                    st.info("No historical risk assessments logged yet. Refresh stations to record assessment history.")

            # Selected station rainfall vs probability detail
            try:
                hist = monitor.get_station_history(sel_id, limit=40)
                env_hist = hist["environmental_history"]
                if not env_hist.empty:
                    st.markdown(f"#### 🌧️ Rainfall Accumulation Trend — {sel['name']}")
                    plot_e = env_hist.sort_values("observed_at").copy()
                    fig_e = go.Figure()
                    fig_e.add_trace(go.Bar(
                        x=plot_e["observed_at"], y=plot_e["rainfall_24h_mm"],
                        name="24h Rain (mm)", marker_color="#38bdf8",
                    ))
                    fig_e.add_trace(go.Scatter(
                        x=plot_e["observed_at"], y=plot_e["rainfall_72h_mm"],
                        name="72h Rain (mm)", line=dict(color="#f59e0b", width=2),
                    ))
                    _style_fig(fig_e, height=260)
                    st.plotly_chart(fig_e, width="stretch")
                else:
                    st.info(f"No rainfall trend history recorded yet for {sel['name']}. Refresh stations to capture hourly weather snapshots.")
            except Exception as exc:
                st.warning(f"Could not render station history: {exc}")

        # ── 4. Recent Activity ────────────────────────────────────────
        st.markdown("### 🕒 Recent Monitoring Activity")
        st.caption(
            "Historical landslide observations (recorded past events) are kept explicitly separate from "
            "model-generated experimental risk assessments and weather reanalysis updates."
        )
        activity = monitor.get_recent_activity(limit=10)
        act_t1, act_t2, act_t3, act_t4, act_t5, act_t6 = st.tabs([
            "🕒 Recently Updated Locations",
            "✅ Authentic Historical Records",
            "📦 All Observations (Incl. Synthetic)",
            "🌦️ Latest Weather Observations",
            "🔮 Experimental Risk Assessments",
            "🚨 Alert History",
        ])
        with act_t1:
            upd_locs = activity.get("recently_updated_locations") or []
            if upd_locs:
                upd_rows = [
                    {
                        "Station": s["name"],
                        "Region": s.get("state_region") or "—",
                        "Last Updated (UTC)": str(s.get("observed_at") or "—")[:19].replace("T", " "),
                        "Freshness": s.get("freshness_status", "Unknown"),
                        "Weather Condition": s.get("weather_condition", "Unavailable"),
                        "Rain 24h (mm)": f"{s['rainfall_24h_mm']:.1f}" if s.get("rainfall_24h_mm") is not None else "—",
                        "Risk Tier": s.get("risk_category", "Not Evaluated"),
                    }
                    for s in upd_locs
                ]
                st.dataframe(pd.DataFrame(upd_rows), width="stretch", hide_index=True)
            else:
                st.info("No monitored locations have been synced with weather observations yet.")
        with act_t2:
            auth_act = activity.get("recent_authentic_observations")
            if auth_act is not None and not auth_act.empty:
                cols = [c for c in ["event_date", "country", "state_region", "landslide_type", "trigger", "fatalities", "source_name"] if c in auth_act.columns]
                st.dataframe(auth_act[cols], width="stretch", hide_index=True)
            else:
                st.info(
                    "No authentic (`is_synthetic = 0`) historical landslide records are currently loaded in SQLite. "
                    "Import verified NASA GLC / Bhukosh / GSI event inventories into SQLite to populate authentic historical tracking."
                )
        with act_t3:
            obs_act = activity["recent_observations"]
            if not obs_act.empty:
                cols = [c for c in ["event_date", "country", "state_region", "landslide_type", "trigger", "fatalities", "is_synthetic", "source_name"] if c in obs_act.columns]
                st.dataframe(obs_act[cols], width="stretch", hide_index=True)
            else:
                st.info("No landslide observations found in SQLite yet.")
        with act_t4:
            wx_act = activity["recent_weather_updates"]
            if not wx_act.empty:
                cols = [c for c in ["observed_at", "location_name", "rainfall_1h_mm", "rainfall_24h_mm", "rainfall_72h_mm", "temperature_c", "humidity_pct", "soil_moisture", "wind_speed_kmh", "is_observed", "data_source"] if c in wx_act.columns]
                st.dataframe(wx_act[cols], width="stretch", hide_index=True)
            else:
                st.info("No weather observations synced yet.")
        with act_t5:
            pred_act = activity["recent_predictions"]
            if not pred_act.empty:
                cols = [c for c in ["predicted_at", "location_name", "risk_category", "probability", "model_name", "method", "is_experimental"] if c in pred_act.columns]
                st.dataframe(pred_act[cols], width="stretch", hide_index=True)
            else:
                st.info("No experimental risk assessments logged yet.")
        with act_t6:
            alert_act = activity["recent_alerts"]
            if not alert_act.empty:
                cols = [c for c in ["created_at", "location_id", "risk_category", "probability", "status", "is_acknowledged", "is_experimental"] if c in alert_act.columns]
                st.dataframe(alert_act[cols], width="stretch", hide_index=True)
            else:
                st.info("No experimental alerts generated yet.")

    if run_every_sec is not None:
        @st.fragment(run_every=run_every_sec)
        def _auto_live_fragment() -> None:
            _render_live_snapshot(trigger_auto_refresh=True)

        _auto_live_fragment()
    else:
        _render_live_snapshot(trigger_auto_refresh=False)


# ═══════════════════════════════════════════════════════════════
# PAGE 2: INTERACTIVE RISK MAP
# ═══════════════════════════════════════════════════════════════

elif page == "🗺️ Geospatial Map":
    from src.geospatial.maps import build_landslide_map
    from src.geospatial.spatial_analysis import (
        apply_geographic_filters,
        drop_invalid_coordinates,
        inspect_location_details,
        records_to_geojson,
        validate_geojson,
    )
    from src.geospatial.hotspot_detection import detect_landslide_hotspots, HOTSPOT_SCIENTIFIC_DISCLAIMER
    from src.geospatial.satellite_monitor import (
        compute_ndvi,
        detect_satellite_changes,
        get_sample_sentinel2_scenario,
        VEGETATION_DISCLAIMER,
        CHANGE_DETECTION_VERIFICATION_LABEL,
    )
    from src.geospatial.terrain import (
        DEFAULT_CRS,
        NODATA_POLICY,
        RASTERIO_AVAILABLE,
        SRTM_RESOLUTION_DESC,
        build_3d_terrain_surface,
        compute_d8_flow_accumulation,
        compute_slope_and_aspect,
        compute_topographic_wetness_index,
        get_point_terrain_summary,
        inspect_dem_raster,
    )
    from src.monitoring import get_live_monitor

    render_page_header(
        "🗺️ Advanced Geospatial Risk Heatmap & Terrain Analysis",
        "Interactive Folium surveillance integrating historical observations, experimental point risk estimates, kernel-density risk heatmap, Indian State/District boundaries, click-level risk explanations, GeoJSON layers, and DEM terrain derivatives.",
        pills=[("EPSG:4326 WGS84", "pill-blue"), ("GeoJSON + HeatMap", "pill-green")],
    )
    experimental_banner(
        "Authentic historical/synthetic landslide markers represent past dataset entries, while risk markers & heatmaps represent "
        "<b>experimental model estimates</b> (evaluated strictly at verified coordinates with no unvalidated wide-area interpolation)."
    )

    df = load_dataset()
    model_info_geo = load_model_info()
    model_meta_geo = model_info_geo.get("metadata", {}) if isinstance(model_info_geo, dict) else {}
    model_ver_geo = str(model_meta_geo.get("saved_at") or "")[:10] or "Active"
    model_name_geo = str(model_meta_geo.get("algorithm") or "risk_model").replace("_", " ").title()

    monitor_geo = get_live_monitor()
    live_snap_geo = monitor_geo.get_live_snapshot(active_only=True)
    raw_stations_geo = live_snap_geo.get("stations", [])
    for st_entry in raw_stations_geo:
        st_entry["model_version"] = model_ver_geo
        st_entry.setdefault("model_name", model_name_geo)

    # Also include authentic SQLite historical observations if present alongside processed dataset
    from src.database.models import get_observations as _geo_get_obs
    db_obs_geo = _geo_get_obs(include_synthetic=True, limit=2000)

    if df is None or df.empty:
        if not db_obs_geo.empty:
            df_geo = db_obs_geo.copy()
        else:
            st.warning("Processed dataset and SQLite observations are currently unavailable or empty. Historical dataset points are omitted, while live monitored risk markers and terrain tools remain active.")
            df_geo = pd.DataFrame()
    else:
        df_geo = df.copy()
        if not db_obs_geo.empty and "is_synthetic" in db_obs_geo.columns:
            auth_db = db_obs_geo[db_obs_geo["is_synthetic"].fillna(1).astype(int) == 0]
            if not auth_db.empty:
                df_geo = pd.concat([auth_db, df_geo], ignore_index=True)

    # Optional authoritative India boundary loader (gracefully handles missing boundary files)
    boundary_provider = None
    admin_states: list[dict[str, Any]] = []
    boundary_status_info: dict[str, Any] = {"status": "Unavailable"}
    try:
        from src.data_providers.india_boundaries import IndiaBoundariesProvider
        boundary_provider = IndiaBoundariesProvider()
        boundary_status_info = boundary_provider.get_boundary_status()
        admin_states = boundary_provider.list_states()
    except Exception:
        boundary_provider = None

    admin_state_names = [s["state_name"] for s in admin_states]

    # Collect dynamic filter domain options cleanly from real available columns
    state_options = sorted(
        {
            str(x)
            for x in (
                (df_geo["state_region"].dropna().tolist() if "state_region" in df_geo.columns else [])
                + [s.get("state_region") for s in raw_stations_geo if s.get("state_region")]
                + admin_state_names
            )
            if str(x).strip()
        }
    )
    source_options = sorted(
        {
            str(x)
            for x in (
                (df_geo["source_name"].dropna().tolist() if "source_name" in df_geo.columns else [])
                + [s.get("data_source") for s in raw_stations_geo if s.get("data_source")]
            )
            if str(x).strip()
        }
    )

    with st.expander("🔍 Location Search, Geographic Filters, Layer Controls & GeoJSON", expanded=True):
        sbar1, sbar2, sbar3 = st.columns([1.4, 1.0, 1.0])
        with sbar1:
            geo_search_query = st.text_input(
                "🔎 Location Search (Station, District, State, Trigger or Source)",
                value="",
                placeholder="e.g. Munnar, Idukki, Kerala, Darjeeling…",
                key="geo_search_q",
            )
        with sbar2:
            admin_focus_state = st.selectbox(
                "Boundary Focus State / UT",
                ["All States / Regions"] + admin_state_names,
                key="geo_focus_state",
                help="Select a State/UT to populate Districts and optionally overlay verified EPSG:4326 boundaries.",
            )
        with sbar3:
            district_choices: list[str] = ["All Districts"]
            if boundary_provider is not None and admin_focus_state != "All States / Regions":
                district_choices += [
                    d["district_name"] for d in boundary_provider.list_districts(state_name=admin_focus_state)
                ]
            elif "district" in df_geo.columns and not df_geo.empty:
                district_choices += sorted(df_geo["district"].dropna().astype(str).unique().tolist())
            admin_focus_district = st.selectbox(
                "District Filter / Zoom",
                district_choices,
                key="geo_focus_district",
            )

        fc1, fc2, fc3, fc4 = st.columns([1.15, 1.2, 1.1, 1.0])
        with fc1:
            st.markdown("**Layer Visibility**")
            show_obs = st.checkbox("Historical Observations", value=True, key="geo_show_obs")
            show_risk_markers = st.checkbox("Model Risk Markers", value=True, key="geo_show_risk")
            show_heatmap = st.checkbox("Experimental Risk Heatmap", value=True, key="geo_show_heat")
            show_boundaries = st.checkbox("Admin Boundaries (EPSG:4326)", value=True, key="geo_show_bounds")
            show_hotspots = st.checkbox("Historical Hotspots (DBSCAN + KDE)", value=False, key="geo_show_hotspots")
            cluster_obs = st.checkbox("Cluster Historical Markers", value=True, key="geo_cluster_obs")
            cluster_risk = st.checkbox("Cluster Risk Markers", value=False, key="geo_cluster_risk")
        with fc2:
            region_choice = st.selectbox("Region Bounds Preset", list(REGION_BOUNDS.keys()), key="geo_region_preset")
            preset_bounds = REGION_BOUNDS.get(region_choice)
            default_lat_rng = (float(preset_bounds[0]), float(preset_bounds[1])) if preset_bounds else (6.0, 38.0)
            default_lon_rng = (float(preset_bounds[2]), float(preset_bounds[3])) if preset_bounds else (68.0, 98.0)
            lat_bounds = st.slider("Latitude Range (°N)", -90.0, 90.0, default_lat_rng, step=0.5, key="geo_lat_rng")
            lon_bounds = st.slider("Longitude Range (°E)", -180.0, 180.0, default_lon_rng, step=0.5, key="geo_lon_rng")
        with fc3:
            sel_states = st.multiselect("State / Region Multi-Filter", state_options, default=[], key="geo_states", help="Leave empty for all states/regions.")
            sel_risks = st.multiselect(
                "Risk Category (Model Estimates)",
                ["Low", "Moderate", "High", "Very High", "Not Evaluated", "Unavailable", "Insufficient Data", "Unknown"],
                default=["Low", "Moderate", "High", "Very High", "Not Evaluated", "Unavailable", "Insufficient Data", "Unknown"],
                key="geo_risks",
            )
            sel_sources = st.multiselect("Data Source", source_options, default=[], key="geo_sources", help="Leave empty for all sources.")
            authentic_only_geo = st.checkbox("Authentic Historical Events Only", value=False, key="geo_auth_only")
        with fc4:
            show_events = st.multiselect(
                "Observation Outcome",
                ["Landslide Events", "Non-events"],
                default=["Landslide Events", "Non-events"],
                key="geo_outcome",
            )
            date_filter_range = None
            if "event_date" in df_geo.columns and not df_geo.empty:
                parsed_dates = pd.to_datetime(df_geo["event_date"], errors="coerce").dropna()
                if not parsed_dates.empty:
                    min_d = parsed_dates.min().date()
                    max_d = parsed_dates.max().date()
                    picked = st.date_input("Observation Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d, key="geo_date_rng")
                    if isinstance(picked, (tuple, list)) and len(picked) == 2:
                        date_filter_range = (picked[0], picked[1])
            max_points = st.slider("Max Observation Markers", 50, 2000, 500, step=50, key="geo_max_pts")

        uploaded_geojson = st.file_uploader(
            "Optional GeoJSON Boundary / Layer Overlay (.geojson / .json, WGS84 EPSG:4326)",
            type=["geojson", "json"],
            key="geo_upload",
        )

    parsed_geojson_overlay = None
    max_geojson_bytes = 10 * 1024 * 1024  # 10 MB upload cap
    if uploaded_geojson is not None:
        try:
            raw_bytes = uploaded_geojson.getvalue()
            if len(raw_bytes) > max_geojson_bytes:
                raise ValueError(
                    f"Uploaded file size ({len(raw_bytes) / (1024 * 1024):.2f} MB) exceeds the 10 MB limit."
                )
            raw_gj = json.loads(raw_bytes.decode("utf-8"))
            parsed_geojson_overlay = validate_geojson(raw_gj)
            st.success(f"Loaded validated GeoJSON layer ({len(parsed_geojson_overlay['features'])} feature(s), EPSG:4326).")
        except Exception as exc:
            st.error(f"Invalid GeoJSON file rejected: {exc}")

    # Resolve optional State / District boundary polygons & bounding box cleanly (handle missing boundaries)
    state_gj_layer = None
    district_gj_layer = None
    zoom_bbox_geo = None
    focus_state_arg = admin_focus_state if admin_focus_state != "All States / Regions" else None
    focus_district_arg = admin_focus_district if admin_focus_district != "All Districts" else None

    if focus_state_arg:
        if boundary_provider is not None and boundary_status_info.get("status") == "Loaded":
            for st_rec in admin_states:
                if st_rec["state_name"].lower() == focus_state_arg.lower():
                    zoom_bbox_geo = st_rec.get("bbox")
                    break
            if show_boundaries:
                state_gj_layer = boundary_provider.get_state_geojson(focus_state_arg)
            if focus_district_arg:
                dist_meta = boundary_provider.resolve_location(focus_state_arg, focus_district_arg)
                if dist_meta and dist_meta.get("bbox"):
                    zoom_bbox_geo = dist_meta["bbox"]
                if show_boundaries:
                    district_gj_layer = boundary_provider.get_district_geojson(focus_state_arg, focus_district_arg)
            if show_boundaries and state_gj_layer is None and district_gj_layer is None:
                st.info(f"Administrative boundary polygons are unavailable locally for '{focus_state_arg}'. Rendering map markers without boundary overlay.")
        else:
            st.info("Local administrative boundary GeoJSON files are unavailable; filtering relies on coordinates and metadata.")

    # Auto-search Indian district match if user searched a district name not in station list
    effective_states_filter = list(sel_states) if sel_states else None
    if focus_state_arg and not effective_states_filter:
        effective_states_filter = [focus_state_arg]

    effective_lat_min, effective_lat_max = float(lat_bounds[0]), float(lat_bounds[1])
    effective_lon_min, effective_lon_max = float(lon_bounds[0]), float(lon_bounds[1])
    if zoom_bbox_geo and focus_district_arg:
        effective_lat_min = max(effective_lat_min, float(zoom_bbox_geo[0]) - 0.5)
        effective_lon_min = max(effective_lon_min, float(zoom_bbox_geo[1]) - 0.5)
        effective_lat_max = min(effective_lat_max, float(zoom_bbox_geo[2]) + 0.5)
        effective_lon_max = min(effective_lon_max, float(zoom_bbox_geo[3]) + 0.5)
        if effective_lat_min > effective_lat_max or effective_lon_min > effective_lon_max:
            effective_lat_min, effective_lat_max = float(zoom_bbox_geo[0]) - 0.5, float(zoom_bbox_geo[2]) + 0.5
            effective_lon_min, effective_lon_max = float(zoom_bbox_geo[1]) - 0.5, float(zoom_bbox_geo[3]) + 0.5

    # Filter historical observations safely
    obs_filtered = df_geo.copy()
    if not obs_filtered.empty:
        if "Landslide Events" not in show_events and "landslide_occurred" in obs_filtered.columns:
            obs_filtered = obs_filtered[obs_filtered["landslide_occurred"] == 0]
        if "Non-events" not in show_events and "landslide_occurred" in obs_filtered.columns:
            obs_filtered = obs_filtered[obs_filtered["landslide_occurred"] == 1]

        obs_filtered, dropped_obs_coords = drop_invalid_coordinates(obs_filtered)
        obs_filtered = apply_geographic_filters(
            obs_filtered,
            states_regions=effective_states_filter if not focus_district_arg else None,
            districts=[focus_district_arg] if (focus_district_arg and "district" in obs_filtered.columns) else None,
            min_lat=effective_lat_min,
            max_lat=effective_lat_max,
            min_lon=effective_lon_min,
            max_lon=effective_lon_max,
            date_range=date_filter_range,
            data_sources=sel_sources or None,
            search_query=geo_search_query or None,
            authentic_only=authentic_only_geo,
        ).head(max_points)
    else:
        dropped_obs_coords = 0

    # Filter model risk points safely (no synthetic random points generated)
    stations_df = pd.DataFrame(raw_stations_geo) if raw_stations_geo else pd.DataFrame()
    if not stations_df.empty:
        stations_filtered_df = apply_geographic_filters(
            stations_df,
            states_regions=effective_states_filter if not focus_district_arg else None,
            districts=[focus_district_arg] if (focus_district_arg and "district" in stations_df.columns) else None,
            min_lat=effective_lat_min,
            max_lat=effective_lat_max,
            min_lon=effective_lon_min,
            max_lon=effective_lon_max,
            risk_categories=sel_risks or None,
            data_sources=sel_sources or None,
            search_query=geo_search_query or None,
        )
        risk_points_filtered = stations_filtered_df.to_dict(orient="records")
    else:
        risk_points_filtered = []

    valid_heat_points = [
        r for r in risk_points_filtered
        if r.get("probability") is not None
        and isinstance(r.get("probability"), (int, float))
        and math.isfinite(float(r["probability"]))
        and str(r.get("risk_category") or "") not in {"Unavailable", "Insufficient Data", "Not Evaluated"}
    ]

    # Map metrics bar
    gm1, gm2, gm3, gm4 = st.columns(4)
    with gm1:
        st.metric("Historical Points Plotted", f"{len(obs_filtered):,}" if show_obs else "0")
    with gm2:
        st.metric("Assessed Risk Points", f"{len(risk_points_filtered)}" if show_risk_markers else "0")
    with gm3:
        st.metric("Valid Heatmap Points", f"{len(valid_heat_points)}" if show_heatmap else "0")
    with gm4:
        st.metric("Invalid Coords Dropped", f"{dropped_obs_coords}")

    if show_heatmap and len(valid_heat_points) == 0:
        st.info(
            "No evaluated risk points with numeric probabilities exist inside the active filters to render a heatmap. "
            "Refresh stations on **📡 Live Monitoring** first — no synthetic geographic points are fabricated."
        )
    elif show_heatmap and 0 < len(valid_heat_points) < 3:
        st.caption(
            f"ℹ️ Heatmap uses {len(valid_heat_points)} verified coordinate(s). Kernel-density visualization does NOT interpolate across unmeasured wide gaps."
        )

    # Legend Banner (Section 6)
    st.markdown(
        '<div style="display:flex; flex-wrap:wrap; gap:0.65rem; align-items:center; margin:0.4rem 0 0.8rem 0;">'
        '<span class="pill" style="border-color:#2ecc71;color:#2ecc71;">🟢 Green: Low (&lt; 0.35)</span>'
        '<span class="pill" style="border-color:#f1c40f;color:#f1c40f;">🟡 Yellow: Moderate (0.35–0.55)</span>'
        '<span class="pill" style="border-color:#e67e22;color:#e67e22;">🟠 Orange: High (0.55–0.75)</span>'
        '<span class="pill" style="border-color:#e74c3c;color:#e74c3c;">🔴 Red: Very High (&ge; 0.75)</span>'
        '<span class="pill">⚪ Gray: Historical / Unavailable / Insufficient Data</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Compute center safely
    if zoom_bbox_geo:
        center_lat = float((zoom_bbox_geo[0] + zoom_bbox_geo[2]) / 2.0)
        center_lon = float((zoom_bbox_geo[1] + zoom_bbox_geo[3]) / 2.0)
    elif risk_points_filtered and show_risk_markers:
        center_lat = float(sum(r["latitude"] for r in risk_points_filtered) / len(risk_points_filtered))
        center_lon = float(sum(r["longitude"] for r in risk_points_filtered) / len(risk_points_filtered))
    elif not obs_filtered.empty and show_obs:
        center_lat = float(obs_filtered["latitude"].mean())
        center_lon = float(obs_filtered["longitude"].mean())
    else:
        center_lat = float((lat_bounds[0] + lat_bounds[1]) / 2.0)
        center_lon = float((lon_bounds[0] + lon_bounds[1]) / 2.0)

    map_obs = obs_filtered if show_obs else pd.DataFrame()
    map_risk = risk_points_filtered if (show_risk_markers or show_heatmap) else []
    clicked_map_state: dict[str, Any] = {}

    try:
        import streamlit_folium as stf
        from src.geospatial.hotspot_detection import detect_landslide_hotspots
        hotspots_geo = None
        if show_hotspots and not map_obs.empty:
            hotspots_geo = detect_landslide_hotspots(
                map_obs,
                method="both",
                eps_km=25.0,
                min_samples=3,
                authentic_only=authentic_only_geo,
            )

        m = build_landslide_map(
            observations_df=map_obs,
            risk_data=map_risk,
            center_lat=center_lat,
            center_lon=center_lon,
            show_heatmap=show_heatmap and len(valid_heat_points) > 0,
            geojson_data=parsed_geojson_overlay,
            cluster_observations=cluster_obs,
            show_risk_markers=show_risk_markers,
            cluster_risk_markers=cluster_risk,
            state_geojson=state_gj_layer,
            district_geojson=district_gj_layer,
            state_name=focus_state_arg,
            district_name=focus_district_arg,
            zoom_bbox=zoom_bbox_geo,
            hotspots_data=hotspots_geo,
        )
        clicked_map_state = stf.st_folium(
            m,
            width=None,
            height=540,
            returned_objects=["last_object_clicked", "last_clicked"],
            key="main_map",
        ) or {}
    except ImportError:
        if not map_obs.empty:
            color_col = "landslide_occurred" if "landslide_occurred" in map_obs.columns else None
            fig = px.scatter_map(
                map_obs, lat="latitude", lon="longitude",
                color=map_obs[color_col].astype(str) if color_col else None,
                color_discrete_map={"0": "#2ecc71", "1": "#e74c3c"},
                zoom=4, height=540, map_style="carto-darkmatter",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=5, b=5, l=5, r=5))
            st.plotly_chart(fig, width="stretch")

    # ── 2. Clicked / Selected Location Deep Inspector ───────────────
    st.markdown("### 📍 Location Inspector (Click Any Map Point or Pick a Location)")
    raw_click = (
        clicked_map_state.get("last_object_clicked")
        or clicked_map_state.get("last_clicked")
    )
    if isinstance(raw_click, dict) and raw_click.get("lat") is not None and raw_click.get("lng") is not None:
        st.session_state["geo_inspected_lat"] = float(raw_click["lat"])
        st.session_state["geo_inspected_lon"] = float(raw_click["lng"])

    station_quick_map = {"Use Clicked / Custom Coordinate": None}
    for s in raw_stations_geo:
        station_quick_map[f"{s['name']} ({s['latitude']:.4f}, {s['longitude']:.4f})"] = s

    iq1, iq2, iq3 = st.columns([1.4, 1.0, 1.0])
    with iq1:
        quick_pick = st.selectbox("Inspect Monitored Station or Map Click", list(station_quick_map.keys()), key="geo_quick_inspect")
        picked_st = station_quick_map[quick_pick]
    default_inspect_lat = (
        float(picked_st["latitude"])
        if picked_st
        else float(st.session_state.get("geo_inspected_lat", center_lat))
    )
    default_inspect_lon = (
        float(picked_st["longitude"])
        if picked_st
        else float(st.session_state.get("geo_inspected_lon", center_lon))
    )
    with iq2:
        st.caption("Selected Latitude (°N)")
        inspect_lat = st.number_input("Inspect Lat", -90.0, 90.0, value=max(-90.0, min(90.0, default_inspect_lat)), format="%.4f", label_visibility="collapsed", key="geo_insp_lat_in")
    with iq3:
        st.caption("Selected Longitude (°E)")
        inspect_lon = st.number_input("Inspect Lon", -180.0, 180.0, value=max(-180.0, min(180.0, default_inspect_lon)), format="%.4f", label_visibility="collapsed", key="geo_insp_lon_in")

    insp = inspect_location_details(
        inspect_lat,
        inspect_lon,
        stations=raw_stations_geo,
        observations_df=df_geo,
        radius_km=50.0,
    )
    if insp.get("valid"):
        ic_c = insp["coordinates"]
        ic_r = insp["rainfall"]
        ic_t = insp["terrain"]
        ic_rk = insp["risk_estimate"]
        ic_f = insp["data_freshness"]
        ic_h = insp["historical_events"]

        im1, im2, im3, im4, im5 = st.columns(5)
        with im1:
            st.metric("Coordinates (EPSG:4326)", f"{ic_c['latitude']:.4f}, {ic_c['longitude']:.4f}", delta=f"Nearest stn: {ic_c['nearest_station_km']} km" if ic_c.get("nearest_station_km") is not None else None, delta_color="off")
        with im2:
            r24 = ic_r.get("rainfall_24h_mm")
            r1 = ic_r.get("rainfall_1h_mm")
            st.metric("Rainfall (24h / 1h)", f"{r24:.1f} / {r1:.1f} mm" if (r24 is not None and r1 is not None) else (f"{r1:.1f} mm (1h)" if r1 is not None else "Unavailable"))
        with im3:
            el = ic_t.get("elevation_m")
            sl = ic_t.get("slope_degrees")
            st.metric("Terrain (Elev / Slope)", f"{el:,.0f}m / {sl:.1f}°" if (el is not None and sl is not None) else (f"{el:,.0f}m" if el is not None else "Unavailable"))
        with im4:
            p_val = ic_rk.get("probability")
            st.metric("Experimental Risk", f"{ic_rk['risk_category']}", delta=f"p={p_val:.3f}" if p_val is not None else "Unavailable", delta_color="off")
        with im5:
            st.metric("Data Freshness", str(ic_f["freshness_status"]), delta=f"Nearby events: {ic_h['count']}", delta_color="off")

        st.info(f"**Risk Explanation:** {insp['risk_explanation']}")
        if ic_h["nearest_events"]:
            with st.expander(f"📜 Historical Events Within {ic_h['radius_km']:.0f} km ({ic_h['count']} — {ic_h['authentic_count']} authentic, {ic_h['synthetic_count']} synthetic)", expanded=False):
                st.dataframe(pd.DataFrame(ic_h["nearest_events"]), width="stretch", hide_index=True)

    # GeoJSON Export row
    exp1, exp2 = st.columns(2)
    with exp1:
        risk_gj = records_to_geojson(risk_points_filtered, record_kind="experimental_risk_estimate")
        st.download_button(
            "⬇️ Export Filtered Risk Estimates (.geojson)",
            data=json.dumps(risk_gj, indent=2),
            file_name="landslide_guard_risk_estimates.geojson",
            mime="application/geo+json",
            width="stretch",
            disabled=len(risk_gj["features"]) == 0,
        )
    with exp2:
        obs_gj = records_to_geojson(
            obs_filtered.head(500).to_dict(orient="records") if not obs_filtered.empty else [],
            record_kind="historical_observation",
        )
        st.download_button(
            "⬇️ Export Filtered Observations (.geojson)",
            data=json.dumps(obs_gj, indent=2),
            file_name="landslide_guard_observations.geojson",
            mime="application/geo+json",
            width="stretch",
            disabled=len(obs_gj["features"]) == 0,
        )

    # ── 3. Advanced Geospatial, Terrain & Satellite Intelligence (Phase 7) ──
    st.divider()
    st.markdown("### 🛰️ Advanced Geospatial, Terrain & Satellite Intelligence (Phase 7)")
    st.caption(
        "Interactive spatial clustering, Kernel Density hotspots, Sentinel-2 NDVI vegetation monitoring, "
        "multi-temporal change detection, 3D digital elevation models, and hydrological Topographic Wetness Index (TWI)."
    )

    tab_hotspots, tab_satellite, tab_change, tab_3d, tab_twi, tab_pt_dem, tab_tif_dem = st.tabs([
        "🎯 Landslide Hotspots (DBSCAN & KDE)",
        "🛰️ Sentinel-2 Vegetation (NDVI)",
        "🔄 Satellite Change Detection",
        "🏔️ 3D Terrain Model",
        "💧 DEM Derivatives & TWI",
        "📍 Point SRTM Inspector",
        "🗺️ Local GeoTIFF DEM (Rasterio)",
    ])

    with tab_hotspots:
        st.markdown("#### 🎯 Spatial Landslide Clustering & Hotspot Detection")
        st.warning(f"**Anti-Hallucination Guard**: {HOTSPOT_SCIENTIFIC_DISCLAIMER}")

        hc1, hc2, hc3, hc4 = st.columns([1.2, 1.0, 1.0, 1.0])
        with hc1:
            h_eps = st.slider("DBSCAN Neighborhood (km)", 5.0, 100.0, 25.0, 5.0, key="hot_eps_slider")
        with hc2:
            h_min = st.slider("Min Cluster Points", 2, 20, 3, 1, key="hot_min_slider")
        with hc3:
            h_method = st.selectbox("Clustering Method", ["both", "dbscan", "kde"], key="hot_method_sel")
        with hc4:
            h_auth = st.checkbox("Authentic Observations Only", value=authentic_only_geo, key="hot_auth_chk")

        active_df_hot = obs_filtered if not obs_filtered.empty else df_geo
        hot_res = detect_landslide_hotspots(
            active_df_hot,
            method=h_method,
            eps_km=float(h_eps),
            min_samples=int(h_min),
            authentic_only=bool(h_auth),
        )

        if hot_res.get("available"):
            sum_h = hot_res["summary"]
            hk1, hk2, hk3, hk4, hk5 = st.columns(5)
            with hk1:
                st.metric("Total Events Analyzed", f"{sum_h['total_events_analyzed']:,}")
            with hk2:
                st.metric("DBSCAN Clusters", f"{sum_h['cluster_count']}")
            with hk3:
                st.metric("Clustered Events", f"{sum_h['clustered_events_count']:,}")
            with hk4:
                st.metric("Noise Outliers", f"{sum_h['noise_events_count']:,}")
            with hk5:
                st.metric("Max Event Density", f"{sum_h['highest_density_events_per_km2']:.2f} /km²")

            if hot_res.get("clusters"):
                st.markdown("##### 📍 Identified Historical Clusters (DBSCAN)")
                c_rows = []
                for cl in hot_res["clusters"]:
                    c_rows.append({
                        "Cluster ID": f"#{cl['cluster_id']}",
                        "Events": cl["point_count"],
                        "Density (events/km²)": cl["event_density_per_sq_km"],
                        "Centroid Lat/Lon": f"({cl['centroid']['latitude']:.4f}, {cl['centroid']['longitude']:.4f})",
                        "Est. Area (km²)": cl["area_sq_km"],
                        "Dominant Trigger": cl["dominant_trigger"],
                        "Landslide Type": cl["dominant_type"],
                        "Date Span": f"{cl['date_range']['start']} to {cl['date_range']['end']}",
                    })
                st.dataframe(pd.DataFrame(c_rows), width="stretch", hide_index=True)

            if hot_res.get("kde_density"):
                st.markdown("##### 🌊 Kernel Density Estimation (KDE) Density Surface & Hotspot Tiers")
                kde_d = hot_res["kde_density"]
                fig_kde = go.Figure(
                    data=go.Contour(
                        z=kde_d["density_matrix"],
                        x=kde_d["lon_axis"],
                        y=kde_d["lat_axis"],
                        colorscale="Hot",
                        reversescale=True,
                        contours=dict(coloring="heatmap", showlabels=True),
                        colorbar=dict(title="Event Density"),
                    )
                )
                fig_kde.update_layout(
                    title=f"KDE Historical Event Density Surface (75th p={kde_d['percentiles']['p75']:.5f}, 90th p={kde_d['percentiles']['p90']:.5f})",
                    xaxis_title="Longitude (°E)",
                    yaxis_title="Latitude (°N)",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=420,
                )
                st.plotly_chart(fig_kde, width="stretch")

                if hot_res.get("hotspot_regions"):
                    with st.expander(f"📍 Top Hotspot Coordinates (>= 90th Percentile Density, {len(hot_res['hotspot_regions'])} cells)"):
                        st.dataframe(pd.DataFrame(hot_res["hotspot_regions"]), width="stretch", hide_index=True)
        else:
            st.warning(hot_res.get("reason", "Hotspot detection unavailable."))

    with tab_satellite:
        st.markdown("#### 🛰️ Sentinel-2 Vegetation Index Monitoring (NDVI)")
        st.info(f"**Vegetation Science Disclaimer**: {VEGETATION_DISCLAIMER}")

        pre_demo, post_demo = get_sample_sentinel2_scenario()
        sat_mode = st.radio(
            "Sentinel-2 Acquisition Scenario",
            ["Pre-Monsoon Healthy Canopy (Sentinel-2B May 2024)", "Post-Monsoon Disturbed Canopy (Sentinel-2A August 2024)"],
            horizontal=True,
            key="sat_scene_radio",
        )
        active_scene = pre_demo if "Pre-Monsoon" in sat_mode else post_demo

        sk1, sk2, sk3, sk4, sk5 = st.columns(5)
        stats_s = active_scene["statistics"]
        with sk1:
            st.metric("Mean NDVI", f"{stats_s['mean_ndvi']:.4f}")
        with sk2:
            st.metric("Median NDVI", f"{stats_s['median_ndvi']:.4f}")
        with sk3:
            st.metric("Min / Max NDVI", f"{stats_s['min_ndvi']:.2f} / {stats_s['max_ndvi']:.2f}")
        with sk4:
            st.metric("Cloud Obscured %", f"{active_scene['cloud_cover_pct']:.1f}%")
        with sk5:
            st.metric("Resolution / Sensor", f"{active_scene['resolution_m']:.0f}m · S2 MSI")

        col_ndvi_map, col_ndvi_classes = st.columns([1.3, 1.0])
        with col_ndvi_map:
            fig_ndvi = go.Figure(
                data=go.Heatmap(
                    z=active_scene["ndvi_grid"],
                    colorscale="YlGn",
                    zmin=0.0,
                    zmax=1.0,
                    colorbar=dict(title="NDVI"),
                )
            )
            fig_ndvi.update_layout(
                title=f"NDVI Grid ({active_scene['observation_date']})",
                xaxis_title="Eastings (pixels)",
                yaxis_title="Northings (pixels)",
                margin=dict(l=20, r=20, t=40, b=20),
                height=350,
            )
            st.plotly_chart(fig_ndvi, width="stretch")

        with col_ndvi_classes:
            classes = active_scene["vegetation_classes"]
            fig_bar = px.bar(
                x=list(classes.keys()),
                y=list(classes.values()),
                labels={"x": "Class", "y": "Coverage (% of Valid Pixels)"},
                title="Canopy / Surface Breakdown (%)",
                color=list(classes.keys()),
                color_discrete_sequence=["#3498db", "#e67e22", "#f1c40f", "#2ecc71", "#1b5e20"],
            )
            fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=350)
            st.plotly_chart(fig_bar, width="stretch")

    with tab_change:
        st.markdown("#### 🔄 Multi-Temporal Satellite Change Detection (dNDVI)")
        st.warning(f"**Verification Label**: {CHANGE_DETECTION_VERIFICATION_LABEL}")

        change_res = detect_satellite_changes(pre_demo, post_demo)
        ck1, ck2, ck3, ck4, ck5 = st.columns(5)
        with ck1:
            st.metric("Baseline Period", change_res["temporal_baseline"])
        with ck2:
            st.metric("Candidate Disturbance", f"{change_res['candidate_disturbance_area_km2']:.4f} km²")
        with ck3:
            st.metric("Severe Loss Pixels", f"{change_res['change_classification']['severe_vegetation_loss_pixels']}")
        with ck4:
            st.metric("Dual Cloud Obscured", f"{change_res['cloud_obscured_pct']:.1f}%")
        with ck5:
            st.metric("Analyzed Area", f"{change_res['analyzed_surface_area_km2']:.3f} km²")

        col_dmap, col_limits = st.columns([1.3, 1.0])
        with col_dmap:
            fig_d = go.Figure(
                data=go.Heatmap(
                    z=change_res["d_ndvi_grid"],
                    colorscale="RdYlGn",
                    zmid=0.0,
                    zmin=-0.6,
                    zmax=0.4,
                    colorbar=dict(title="Δ NDVI"),
                )
            )
            fig_d.update_layout(
                title="Δ NDVI Spectral Change Map (Red = Canopy Loss)",
                xaxis_title="Eastings (pixels)",
                yaxis_title="Northings (pixels)",
                margin=dict(l=20, r=20, t=40, b=20),
                height=350,
            )
            st.plotly_chart(fig_d, width="stretch")

        with col_limits:
            st.markdown("##### ⚠️ Operational Remote Sensing Limitations")
            lims = change_res["limitations"]
            st.markdown(f"- **Resolution**: {lims['spatial_resolution']}")
            st.markdown(f"- **Cloud Cover**: {lims['cloud_cover']}")
            st.markdown(f"- **Temporal Latency**: {lims['temporal_latency']}")
            st.caption("Change detection flags candidate zones for high-resolution optical / drone / field verification.")

    with tab_3d:
        st.markdown("#### 🏔️ 3D Interactive Digital Elevation Model (DEM)")
        st.caption("Interactive 3D mesh surface rendered with Plotly Surface. Supports draping slope, roughness, and elevation palettes.")

        t3d_col1, t3d_col2 = st.columns([1.0, 1.0])
        with t3d_col1:
            drape_choice = st.selectbox(
                "3D Surface Drape Mode",
                ["elevation", "slope", "roughness"],
                format_func=lambda x: f"Drape {x.title()}",
                key="t3d_drape_sel",
            )
        with t3d_col2:
            colorscale_3d = st.selectbox("Color Palette", ["Earth", "Viridis", "Plasma", "Turbo"], key="t3d_color_sel")

        rng_3d = np.random.default_rng(101)
        x_3d = np.linspace(-3, 3, 30)
        y_3d = np.linspace(-3, 3, 30)
        xx, yy = np.meshgrid(x_3d, y_3d)
        demo_elev = 1200.0 + 800.0 * np.exp(-(xx**2 + yy**2) / 4.0) + 150.0 * np.sin(2 * xx) + rng_3d.normal(0, 10, size=xx.shape)

        t_stats_3d = compute_slope_and_aspect(demo_elev, 30.0, 30.0)
        drape_mat = demo_elev
        if drape_choice == "slope":
            drape_mat = t_stats_3d["slope_deg"]
        elif drape_choice == "roughness":
            drape_mat = t_stats_3d["tri_m"]

        surface_3d = build_3d_terrain_surface(
            demo_elev,
            drape_mode=drape_choice,
            drape_grid=drape_mat,
            colorscale=colorscale_3d,
            title=f"3D High-Relief Terrain Model (Drape: {drape_choice.title()})",
        )
        st.plotly_chart(surface_3d["figure"], width="stretch")

        k3d1, k3d2, k3d3, k3d4 = st.columns(4)
        with k3d1:
            st.metric("Elevation Min", f"{surface_3d['elevation_min_m']:.1f} m")
        with k3d2:
            st.metric("Elevation Max", f"{surface_3d['elevation_max_m']:.1f} m")
        with k3d3:
            st.metric("Terrain Relief", f"{surface_3d['elevation_relief_m']:.1f} m")
        with k3d4:
            st.metric("Mean Slope", f"{t_stats_3d['mean_slope_deg']:.1f}°")

    with tab_twi:
        st.markdown("#### 💧 Topographic Wetness Index (TWI) & Hydrological Flow Routing")
        st.latex(r"TWI = \ln\left(\frac{a}{\tan(\beta)}\right)")
        st.caption(
            "Where *a* is the specific catchment area (upslope contributing area per unit contour width in metres) "
            "and *β* is the local slope angle in radians. **Scientific Guard**: TWI is refused without required hydrological inputs."
        )

        tw1, tw2 = st.columns(2)
        with tw1:
            st.markdown("##### ❌ Case 1: TWI Evaluation Without Catchment Area")
            twi_refused = compute_topographic_wetness_index(t_stats_3d["slope_deg"], specific_catchment_area_m=None)
            st.error(f"**Refusal Enforced**: {twi_refused['reason']}")
            st.metric("TWI Status", "Refused / Unavailable (Zero Hallucination)")

        with tw2:
            st.markdown("##### ✅ Case 2: D8 Hydrological Routing & Authentic TWI")
            d8_res = compute_d8_flow_accumulation(demo_elev, 30.0, 30.0)
            twi_computed = compute_topographic_wetness_index(t_stats_3d["slope_deg"], specific_catchment_area_m=d8_res["specific_catchment_area_m"])
            st.success(f"**Calculated Safely**: {twi_computed['reason']}")
            st.metric("Mean TWI", f"{twi_computed['mean_twi']:.2f}", delta=f"Max: {twi_computed['max_twi']:.2f}")

        fig_twi = go.Figure(
            data=go.Heatmap(
                z=twi_computed["twi"],
                colorscale="Blues",
                colorbar=dict(title="TWI"),
            )
        )
        fig_twi.update_layout(
            title="Topographic Wetness Index Grid (High = Moisture Accumulation Hollows)",
            xaxis_title="Eastings (cells)",
            yaxis_title="Northings (cells)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=340,
        )
        st.plotly_chart(fig_twi, width="stretch")

    with tab_pt_dem:
        t1, t2, t3 = st.columns([1.2, 1.0, 1.0])
        with t1:
            station_choices_t = {"Inspected / Custom Coordinate": None}
            for s in raw_stations_geo:
                station_choices_t[f"{s['name']} ({s['latitude']:.2f}, {s['longitude']:.2f})"] = s
            picked_station_t = st.selectbox("Select Station or Inspected Coordinate", list(station_choices_t.keys()), key="terrain_station_pick")
            picked_obj_t = station_choices_t[picked_station_t]
        with t2:
            t_lat = st.number_input(
                "Latitude (°N)", -90.0, 90.0,
                value=float(picked_obj_t["latitude"]) if picked_obj_t else float(inspect_lat),
                format="%.4f", key="terrain_lat_in",
            )
        with t3:
            t_lon = st.number_input(
                "Longitude (°E)", -180.0, 180.0,
                value=float(picked_obj_t["longitude"]) if picked_obj_t else float(inspect_lon),
                format="%.4f", key="terrain_lon_in",
            )

        if st.button("⛰️ Compute DEM Elevation, Slope, Aspect & Roughness", key="terrain_inspect_btn"):
            with st.spinner("Querying Open-Elevation SRTM DEM stencil…"):
                terrain_res = get_point_terrain_summary(float(t_lat), float(t_lon))
            if terrain_res.get("available"):
                tr1, tr2, tr3, tr4, tr5 = st.columns(5)
                with tr1:
                    st.metric("Elevation", f"{terrain_res['elevation_m']:.1f} m")
                with tr2:
                    st.metric("Slope Angle", f"{terrain_res['slope_degrees']:.2f}°")
                with tr3:
                    asp = terrain_res.get("aspect_degrees")
                    card = terrain_res.get("aspect_cardinal") or ""
                    st.metric("Aspect", f"{asp:.1f}° ({card})" if asp is not None else card or "Flat")
                with tr4:
                    tri_v = terrain_res.get("roughness_tri_m")
                    rel_v = terrain_res.get("local_relief_m")
                    st.metric("Roughness (TRI)", f"{tri_v:.2f} m" if tri_v is not None else "Unavailable", delta=f"Relief: {rel_v:.1f}m" if rel_v is not None else None, delta_color="off")
                with tr5:
                    st.metric("CRS", terrain_res.get("crs", DEFAULT_CRS))
                st.caption(f"Source: {terrain_res.get('source')} · Resolution: {terrain_res.get('resolution')} · Curvature Proxy: `{terrain_res.get('curvature_proxy')}`")
            else:
                st.warning(terrain_res.get("reason", "Terrain DEM data is currently unavailable."))

    with tab_tif_dem:
        dem_path_input = st.text_input(
            "Local SRTM GeoTIFF Path (.tif)",
            value="data/raw/srtm_dem.tif",
            key="dem_path_in",
        )
        if not RASTERIO_AVAILABLE:
            st.info(
                "Optional package `rasterio` is not installed in this Python environment, and local GeoTIFF DEMs are absent unless provided. "
                "Local GeoTIFF reading, CRS transformation, and Rasterio slope/aspect/TRI calculations activate automatically when `rasterio` and a valid `.tif` file exist; "
                "all terrain functions never fabricate fallback raster values."
            )
        if st.button("🔍 Inspect Local DEM Raster", key="dem_inspect_btn"):
            dem_res = inspect_dem_raster(dem_path_input, sample_lat=float(inspect_lat), sample_lon=float(inspect_lon))
            if dem_res.get("available"):
                dr1, dr2, dr3, dr4, dr5 = st.columns(5)
                with dr1:
                    st.metric("Mean Elevation", f"{dem_res['mean_elevation_m']} m")
                with dr2:
                    st.metric("Mean / Max Slope", f"{dem_res['mean_slope_deg']}° / {dem_res['max_slope_deg']}°")
                with dr3:
                    st.metric("Mean Roughness (TRI)", f"{dem_res.get('mean_tri_m')} m" if dem_res.get("mean_tri_m") is not None else "Unavailable")
                with dr4:
                    st.metric("Local Relief", f"{dem_res.get('elevation_relief_m')} m" if dem_res.get("elevation_relief_m") is not None else "Unavailable")
                with dr5:
                    st.metric("CRS / Valid Cells", f"{dem_res['crs']} ({dem_res['valid_cells']:,})")
                st.json(dem_res)
            else:
                st.warning(dem_res.get("reason", "DEM raster unavailable."))


# ═══════════════════════════════════════════════════════════════
# PAGE 3: DYNAMIC RAINFALL ANALYSIS
# ═══════════════════════════════════════════════════════════════

elif page == "🌦️ Weather Analysis":
    from src.data_providers.weather_provider import WeatherProvider, parse_api_timestamp
    from src.database.models import get_locations
    from src.monitoring.rainfall_analysis import (
        SCIENTIFIC_DISCLAIMER,
        build_daily_series_dataframe,
        build_hourly_series_dataframe,
        compute_rainfall_metrics,
        evaluate_rainfall_risk_change,
    )

    render_page_header(
        "🌧️ Dynamic Rainfall Analysis",
        "Analyze Open-Meteo current, historical reanalysis, and forecast rainfall separately, with strict accumulation windows and dynamic risk integration.",
        pills=[
            ("Open-Meteo API", "pill-blue"),
            ("Forecasts != Observations", "pill-amber"),
            ("No Zero-Imputation", "pill-green"),
        ],
    )
    experimental_banner(SCIENTIFIC_DISCLAIMER)

    locs_df = get_locations(active_only=True)
    station_map: dict[str, dict] = {}
    if not locs_df.empty:
        for _, row in locs_df.iterrows():
            station_map[f"{row['name']} ({row.get('state_region') or 'India'})"] = dict(row)

    mode_col1, mode_col2, mode_col3, mode_col4, mode_col5 = st.columns([1.6, 0.85, 0.85, 1.0, 1.0])
    with mode_col1:
        selected_station_label = st.selectbox(
            "Select Monitored Station or Custom Coordinate",
            ["Custom Coordinate"] + list(station_map.keys()),
            key="rain_station_select",
        )
    if selected_station_label != "Custom Coordinate":
        st_rec = station_map[selected_station_label]
        default_lat = float(st_rec["latitude"])
        default_lon = float(st_rec["longitude"])
        default_loc_id = int(st_rec["id"])
        default_loc_name = str(st_rec["name"])
        default_slope = float(st_rec["slope_degrees"]) if pd.notna(st_rec.get("slope_degrees")) else None
        default_elev = float(st_rec["elevation_m"]) if pd.notna(st_rec.get("elevation_m")) else None
    else:
        default_lat = 10.0889
        default_lon = 77.0595
        default_loc_id = None
        default_loc_name = "Custom Coordinate"
        default_slope = 25.0
        default_elev = 1200.0

    with mode_col2:
        rain_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=default_lat, format="%.4f", key="rain_lat")
    with mode_col3:
        rain_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=default_lon, format="%.4f", key="rain_lon")
    with mode_col4:
        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
        fetch_clicked = st.button("🌦️ Fetch Weather", type="primary", width="stretch", key="rain_fetch_btn")
    with mode_col5:
        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
        refresh_clicked = st.button(
            "🔄 Manual Refresh",
            help="Bypass local cache and query Open-Meteo live.",
            width="stretch",
            key="rain_manual_refresh_btn",
        )

    opt1, opt2, opt3 = st.columns([1, 1, 1.4])
    with opt1:
        past_days = st.slider("Past Reanalysis Days", min_value=1, max_value=7, value=3, key="rain_past_days")
    with opt2:
        forecast_days = st.slider("Forecast Horizon Days", min_value=0, max_value=7, value=3, key="rain_fcst_days")
    with opt3:
        include_baseline = st.checkbox(
            "Include 365-Day ERA5 Historical Baseline (for Anomaly)",
            value=True,
            key="rain_include_baseline",
        )

    if fetch_clicked or refresh_clicked:
        bypass_cache_flag = bool(refresh_clicked)
        spinner_msg = (
            "Manual live refresh from Open-Meteo (bypassing local cache)..."
            if bypass_cache_flag
            else "Fetching Open-Meteo rainfall reanalysis, forecast, and ERA5 archive..."
        )
        with st.spinner(spinner_msg):
            provider = WeatherProvider()
            fetched = provider.fetch_separated_weather(
                latitude=rain_lat,
                longitude=rain_lon,
                past_days=past_days,
                forecast_days=forecast_days,
                include_archive_baseline=include_baseline,
                bypass_cache=bypass_cache_flag,
            )
            fetch_meta = provider.last_fetch_metadata
            st.session_state["rain_weather_payload"] = fetched
            st.session_state["rain_weather_loc_meta"] = {
                "location_id": default_loc_id,
                "location_name": default_loc_name,
                "slope_degrees": default_slope,
                "elevation_m": default_elev,
            }
            if provider.last_update_time:
                st.session_state["rain_last_successful_update"] = provider.last_update_time
            if fetched is None:
                err_detail = fetch_meta.get("error") or (
                    "Open-Meteo API request failed or timed out after retries."
                )
                status_tag = fetch_meta.get("status", "error")
                if status_tag == "rate_limited":
                    st.session_state["rain_connection_status"] = "Rate Limited (HTTP 429)"
                elif status_tag == "invalid_coordinates":
                    st.session_state["rain_connection_status"] = "Invalid Coordinates"
                else:
                    st.session_state["rain_connection_status"] = "Connection Error / Unavailable"
                st.session_state["rain_fetch_error"] = (
                    f"{err_detail} Use the Manual Environmental Input Fallback below."
                )
            else:
                st.session_state["rain_fetch_error"] = None
                st.session_state["rain_connection_status"] = (
                    "Serving Timestamped Cache (Not Live)"
                    if fetched.get("is_cached")
                    else "Connected (Live Open-Meteo API)"
                )

    weather_payload = st.session_state.get("rain_weather_payload")
    fetch_error = st.session_state.get("rain_fetch_error")
    conn_status = st.session_state.get("rain_connection_status", "Not Queried Yet")
    last_api_update = st.session_state.get("rain_last_successful_update")
    loc_meta = st.session_state.get("rain_weather_loc_meta") or {
        "location_id": default_loc_id,
        "location_name": default_loc_name,
        "slope_degrees": default_slope,
        "elevation_m": default_elev,
    }

    st_c1, st_c2, st_c3 = st.columns(3)
    st_c1.metric("API Connection Status", conn_status)
    st_c2.metric(
        "Last Successful API Update",
        str(last_api_update)[:19].replace("T", " ") + " UTC" if last_api_update else "Unavailable",
    )
    st_c3.metric(
        "Active Telemetry Mode",
        "Cached Replay" if (weather_payload and weather_payload.get("is_cached")) else ("Live API" if weather_payload else "Manual Fallback Ready"),
    )

    if fetch_error:
        st.error(fetch_error)

    if weather_payload is not None:
        prov = weather_payload.get("provenance", {})
        is_cached = bool(weather_payload.get("is_cached", False))
        unit_val = weather_payload.get("unit_validation", {})

        # Cache / Freshness / Provider Attribution banner (never show cached as live)
        if is_cached:
            st.warning(
                f"**CACHED DATA (NOT LIVE):** Served from local cache captured at "
                f"`{weather_payload.get('cached_at') or 'unknown'}` "
                f"(age: {prov.get('cache_age_seconds', '?')}s). "
                f"Last live update: `{weather_payload.get('last_successful_update') or last_api_update or 'N/A'}`. "
                f"Provider: {prov.get('attribution', 'Open-Meteo')}."
            )
        else:
            st.info(
                f"**LIVE OPEN-METEO FETCH** · Provider: {prov.get('attribution', 'Open-Meteo')} · "
                f"Fetched UTC: `{weather_payload.get('fetched_at')}` · "
                f"Unit validation: `{'PASSED' if unit_val.get('validated', True) else 'WARNINGS'}`"
            )

        # ── Separated Current Weather Block ───────────────────────────
        current_blk = weather_payload.get("current") or {}
        st.markdown("### 🌡️ Current Weather Snapshot (Model Snapshot — Not Forecast, Not Direct Gauge)")
        if current_blk:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Condition", str(current_blk.get("weather_condition") or "Unavailable"))
            c2.metric("Current Precip", f"{current_blk['precipitation_mm']:.2f} mm" if current_blk.get("precipitation_mm") is not None else "N/A")
            c3.metric("Temperature", f"{current_blk['temperature_2m_c']:.1f} °C" if current_blk.get("temperature_2m_c") is not None else "N/A")
            c4.metric("Rel. Humidity", f"{current_blk['relative_humidity_2m_pct']:.0f} %" if current_blk.get("relative_humidity_2m_pct") is not None else "N/A")
            c5.metric("Wind Speed", f"{current_blk['wind_speed_10m_kmh']:.1f} km/h" if current_blk.get("wind_speed_10m_kmh") is not None else "N/A")
            c6.metric("Snapshot Time", str(current_blk.get("time") or "N/A"))
        else:
            st.caption("Current snapshot block not included in this API response.")

        # ── Strict Accumulation, Trend & Anomaly KPIs ─────────────────
        metrics = compute_rainfall_metrics(
            historical_hourly=weather_payload.get("historical_hourly", {}),
            forecast_hourly=weather_payload.get("forecast_hourly", {}),
            baseline_daily=weather_payload.get("archive_baseline_daily", {}),
        )
        st.markdown("### 📊 Rainfall Accumulations, Trend & Anomaly (Strict Temporal Sufficiency)")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("1h Rainfall", f"{metrics['rainfall_1h_mm']:.2f} mm" if metrics["rainfall_1h_mm"] is not None else "N/A")
        m2.metric(
            "24h Accumulated",
            f"{metrics['rainfall_24h_mm']:.2f} mm" if metrics["rainfall_24h_mm"] is not None else "Insufficient",
            help=f"Missing/unavailable slots in 24h window: {metrics['missing_hours_24h']}",
        )
        m3.metric(
            "48h Accumulated",
            f"{metrics['rainfall_48h_mm']:.2f} mm" if metrics["rainfall_48h_mm"] is not None else "Insufficient",
            help=f"Missing/unavailable slots in 48h window: {metrics['missing_hours_48h']}",
        )
        m4.metric(
            "72h Accumulated",
            f"{metrics['rainfall_72h_mm']:.2f} mm" if metrics["rainfall_72h_mm"] is not None else "Insufficient",
            help=f"Missing/unavailable slots in 72h window: {metrics['missing_hours_72h']}",
        )
        m5.metric(
            "24h Trend",
            metrics["trend_label"],
            delta=(
                f"{metrics['trend_slope_mm_per_hr']:+.3f} mm/h"
                if metrics["trend_slope_mm_per_hr"] is not None
                else None
            ),
        )
        m6.metric(
            "24h Baseline Anomaly",
            f"{metrics['anomaly_24h_mm']:+.2f} mm" if metrics["anomaly_24h_mm"] is not None else "N/A",
            delta=metrics["anomaly_status"] if metrics["anomaly_24h_mm"] is not None else None,
            help=(
                f"Compared against ERA5 {metrics['baseline_days_count']}-day daily mean "
                f"({metrics['baseline_daily_mean_mm']} mm/day)."
                if metrics["baseline_daily_mean_mm"] is not None
                else "Insufficient archive baseline (>=30 days required)."
            ),
        )
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric(
            "Reanalysis Coverage",
            f"{metrics['valid_hours_total']} valid hrs",
            delta=f"{metrics['missing_hours_total']} missing hrs" if metrics["missing_hours_total"] else "Zero missing hours",
        )
        fc2.metric(
            "Forecast Next 24h (Predicted)",
            f"{metrics['forecast_24h_mm']:.2f} mm" if metrics["forecast_24h_mm"] is not None else "Incomplete / Unavailable",
        )
        fc3.metric(
            "Forecast Status",
            metrics.get("forecast_data_status", "Unavailable"),
            delta=f"Total: {metrics['forecast_total_mm']:.2f} mm" if metrics.get("forecast_total_mm") is not None else "Not summed if incomplete",
        )

        # ── Five Interactive Plotly Charts ────────────────────────────
        hourly_df = build_hourly_series_dataframe(weather_payload)
        daily_df = build_daily_series_dataframe(weather_payload)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("#### 1. Hourly Precipitation (Past Reanalysis vs Forecast)")
            if not hourly_df.empty:
                fig_hr = px.bar(
                    hourly_df,
                    x="time",
                    y="precipitation_mm",
                    color="series_type",
                    color_discrete_map={
                        "Past Reanalysis (Model-Derived)": "#38bdf8",
                        "Forecast (Predicted)": "#a855f7",
                    },
                    labels={"time": "Timestamp", "precipitation_mm": "Precipitation (mm/h)", "series_type": "Data Category"},
                )
                fig_hr.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.4)",
                    font_color="#e2e8f0",
                    height=330,
                    margin=dict(l=10, r=10, t=20, b=20),
                )
                st.plotly_chart(fig_hr, width="stretch")
            else:
                st.info("No hourly precipitation data available.")

        with ch2:
            st.markdown("#### 2. Daily Rainfall (Past/Current Date vs Forecast)")
            if not daily_df.empty:
                fig_dy = px.bar(
                    daily_df,
                    x="date",
                    y="precipitation_sum_mm",
                    color="series_type",
                    color_discrete_map={
                        "Past / Current-Date (Reanalysis)": "#22c55e",
                        "Forecast (Predicted)": "#f59e0b",
                    },
                    labels={"date": "Date", "precipitation_sum_mm": "Daily Rainfall (mm)", "series_type": "Data Category"},
                )
                fig_dy.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.4)",
                    font_color="#e2e8f0",
                    height=330,
                    margin=dict(l=10, r=10, t=20, b=20),
                )
                st.plotly_chart(fig_dy, width="stretch")
            else:
                st.info("No daily precipitation data available.")

        ch3, ch4 = st.columns(2)
        with ch3:
            st.markdown("#### 3. Strict Rainfall Accumulation Windows (1h / 24h / 48h / 72h)")
            acc_rows = [
                {"Window": "1h", "Accumulation (mm)": metrics["rainfall_1h_mm"]},
                {"Window": "24h", "Accumulation (mm)": metrics["rainfall_24h_mm"]},
                {"Window": "48h", "Accumulation (mm)": metrics["rainfall_48h_mm"]},
                {"Window": "72h", "Accumulation (mm)": metrics["rainfall_72h_mm"]},
            ]
            acc_df = pd.DataFrame([r for r in acc_rows if r["Accumulation (mm)"] is not None])
            if not acc_df.empty:
                fig_acc = px.bar(
                    acc_df,
                    x="Window",
                    y="Accumulation (mm)",
                    text="Accumulation (mm)",
                    color="Accumulation (mm)",
                    color_continuous_scale="Blues",
                )
                fig_acc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.4)",
                    font_color="#e2e8f0",
                    height=320,
                    margin=dict(l=10, r=10, t=20, b=20),
                )
                st.plotly_chart(fig_acc, width="stretch")
            else:
                st.info("No complete accumulation window is available without missing values.")

        with ch4:
            st.markdown("#### 4. Historical Reanalysis vs Forecast Cumulative Curve")
            if not hourly_df.empty and hourly_df["precipitation_mm"].notna().any():
                cum_df = hourly_df.copy()
                cum_df["precip_valid"] = cum_df["precipitation_mm"].fillna(0.0)
                cum_df["cumulative_mm"] = cum_df.groupby("series_type")["precip_valid"].cumsum()
                fig_cmp = px.line(
                    cum_df,
                    x="time",
                    y="cumulative_mm",
                    color="series_type",
                    color_discrete_map={
                        "Past Reanalysis (Model-Derived)": "#38bdf8",
                        "Forecast (Predicted)": "#a855f7",
                    },
                    labels={"time": "Timestamp", "cumulative_mm": "Segment Cumulative (mm)", "series_type": "Data Category"},
                )
                fig_cmp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.4)",
                    font_color="#e2e8f0",
                    height=320,
                    margin=dict(l=10, r=10, t=20, b=20),
                )
                st.plotly_chart(fig_cmp, width="stretch")
            else:
                st.info("Insufficient hourly records to compare historical vs forecast trajectories.")

        st.markdown("#### 5. Trailing 24-Hour Rainfall Trend Visualization")
        past_only = hourly_df[~hourly_df["is_forecast"]].tail(24).copy() if not hourly_df.empty else pd.DataFrame()
        if not past_only.empty and past_only["precipitation_mm"].notna().sum() >= 6:
            valid_past = past_only.dropna(subset=["precipitation_mm"]).reset_index(drop=True)
            fig_tr = go.Figure()
            fig_tr.add_trace(
                go.Bar(
                    x=valid_past["time"],
                    y=valid_past["precipitation_mm"],
                    name="Hourly Precipitation (mm)",
                    marker_color="#38bdf8",
                )
            )
            slope = metrics["trend_slope_mm_per_hr"] or 0.0
            mean_y = float(valid_past["precipitation_mm"].mean())
            x_idx = list(range(len(valid_past)))
            x_mid = (len(valid_past) - 1) / 2.0
            trend_line = [max(0.0, mean_y + slope * (xi - x_mid)) for xi in x_idx]
            fig_tr.add_trace(
                go.Scatter(
                    x=valid_past["time"],
                    y=trend_line,
                    mode="lines",
                    name=f"OLS Trend ({metrics['trend_label']}: {slope:+.3f} mm/h)",
                    line=dict(color="#f97316", width=3),
                )
            )
            fig_tr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                font_color="#e2e8f0",
                height=310,
                margin=dict(l=10, r=10, t=20, b=20),
            )
            st.plotly_chart(fig_tr, width="stretch")
        else:
            st.info("Need at least 6 valid hourly observations in the trailing 24h window to render trend.")

        # ── Dynamic Risk Integration from Live/Cached Rainfall ────────
        st.divider()
        st.markdown("### ⚡ Dynamic Risk Integration (Open-Meteo Rainfall -> Risk Engine)")
        if st.button("🔄 Recalculate Experimental Risk from Rainfall Data", key="rain_eval_live_btn"):
            hist_blk = weather_payload.get("historical_hourly") or {}
            soil_vals = [v for v in (hist_blk.get("soil_moisture_0_to_7cm") or []) if v is not None]
            latest_soil = soil_vals[-1] if soil_vals else None
            hist_times = hist_blk.get("time") or []
            obs_dt = (
                parse_api_timestamp(hist_times[-1], weather_payload.get("utc_offset_seconds", 0))
                if hist_times
                else None
            )
            live_features = {
                "latitude": rain_lat,
                "longitude": rain_lon,
                "slope_degrees": loc_meta.get("slope_degrees"),
                "elevation_m": loc_meta.get("elevation_m") or weather_payload.get("elevation_m"),
                "rainfall_1h_mm": metrics.get("rainfall_1h_mm"),
                "rainfall_24h_mm": metrics.get("rainfall_24h_mm"),
                "rainfall_48h_mm": metrics.get("rainfall_48h_mm"),
                "rainfall_72h_mm": metrics.get("rainfall_72h_mm"),
                "soil_moisture": latest_soil,
            }
            src_desc = (
                f"Open-Meteo Reanalysis (CACHED at {weather_payload.get('cached_at')})"
                if is_cached
                else "Open-Meteo Live Model-Derived Reanalysis"
            )
            st.session_state["rain_risk_delta"] = evaluate_rainfall_risk_change(
                features=live_features,
                observation_time=obs_dt,
                location_id=loc_meta.get("location_id"),
                location_name=loc_meta.get("location_name"),
                source_label=src_desc,
                is_manual=False,
                persist=True,
            )

        # ── Phase 6: Anomaly Detection, Multi-Scale Trends & 24h/48h/72h Risk Trend Forecast ──
        from src.monitoring.environmental_analytics import (
            FORECAST_TIMING_DISCLAIMER,
            analyze_rainfall_trends,
            detect_rainfall_anomalies,
            forecast_risk_trend,
        )

        st.divider()
        st.markdown("### 🔬 Phase 6 — Statistical Rainfall Anomaly Detection & Multi-Scale Trends")
        baseline_daily_blk = weather_payload.get("archive_baseline_daily") or {}
        anom = detect_rainfall_anomalies(
            recent_24h_mm=metrics.get("rainfall_24h_mm"),
            baseline_daily=baseline_daily_blk,
        )
        anom_colors = {
            "Normal rainfall": "#22c55e",
            "Above-normal rainfall": "#eab308",
            "Unusual rainfall": "#f97316",
            "Extreme rainfall": "#ef4444",
        }
        anom_cls = anom["classification"]
        anom_col = anom_colors.get(anom_cls, "#64748b")

        an_c1, an_c2, an_c3, an_c4 = st.columns(4)
        with an_c1:
            st.markdown(
                f'<div style="background:{anom_col}22; border:2px solid {anom_col}; border-radius:10px; '
                f'padding:0.75rem; text-align:center;"><b style="color:{anom_col}; font-size:1.05rem;">{anom_cls}</b>'
                f'<br><small style="color:#94a3b8;">Anomaly Tier ({anom["baseline_days_count"]} archive days)</small></div>',
                unsafe_allow_html=True,
            )
        with an_c2:
            st.metric("Empirical Z-Score", f"{anom['z_score']:+.2f} σ" if anom["z_score"] is not None else "Unavailable")
        with an_c3:
            st.metric(
                "7d Rolling Upper Threshold",
                f"{anom['rolling_upper_threshold_mm']:.1f} mm" if anom["rolling_upper_threshold_mm"] is not None else "Unavailable",
                delta=(
                    "Above threshold" if anom.get("above_rolling_threshold") is True
                    else ("Within envelope" if anom.get("above_rolling_threshold") is False else None)
                ),
            )
        with an_c4:
            if anom["isolation_forest_available"]:
                iso_lbl = "Outlier Flagged" if anom["isolation_forest_anomaly"] else "Inlier (Normal)"
                st.metric("Isolation Forest", iso_lbl, delta=f"score={anom['isolation_forest_score']:+.3f}" if anom.get("isolation_forest_score") is not None else None)
            else:
                st.metric("Isolation Forest", "Unavailable", help="Requires >= 60 valid baseline days with variance.")

        for rsn in anom.get("reasons", []):
            st.caption(f"• {rsn}")

        hist_anom_series = anom.get("history_series") or []
        if hist_anom_series:
            anom_df = pd.DataFrame(hist_anom_series)
            fig_anom = go.Figure()
            fig_anom.add_trace(go.Bar(
                x=anom_df["date"], y=anom_df["precipitation_mm"],
                name="Daily Historical (mm)", marker_color="#38bdf8", opacity=0.65,
            ))
            fig_anom.add_trace(go.Scatter(
                x=anom_df["date"], y=anom_df["rolling_mean_mm"],
                mode="lines", name="7d Rolling Mean", line=dict(color="#22c55e", width=2),
            ))
            fig_anom.add_trace(go.Scatter(
                x=anom_df["date"], y=anom_df["rolling_upper_threshold_mm"],
                mode="lines", name="Rolling Upper Envelope (+2σ)", line=dict(color="#f97316", dash="dash", width=2),
            ))
            outliers = anom_df[anom_df["is_statistical_outlier"]]
            if not outliers.empty:
                fig_anom.add_trace(go.Scatter(
                    x=outliers["date"], y=outliers["precipitation_mm"],
                    mode="markers", name="Historical Outlier Day",
                    marker=dict(color="#ef4444", size=8, symbol="diamond"),
                ))
            if anom.get("recent_24h_mm") is not None:
                fig_anom.add_hline(
                    y=anom["recent_24h_mm"],
                    line_dash="dot", line_color="#a855f7",
                    annotation_text=f"Current 24h: {anom['recent_24h_mm']:.1f} mm",
                )
            fig_anom.update_layout(
                title="Historical Baseline Rainfall, Rolling Threshold Envelope & Detected Anomalies",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.4)",
                font_color="#e2e8f0", height=320, margin=dict(l=10, r=10, t=35, b=20),
            )
            st.plotly_chart(fig_anom, width="stretch")

        trends = analyze_rainfall_trends(
            baseline_daily=baseline_daily_blk,
            daily_historical=weather_payload.get("daily_historical"),
            recent_24h_mm=metrics.get("rainfall_24h_mm"),
        )
        t_tab_daily, t_tab_weekly, t_tab_monthly, t_tab_seasonal = st.tabs([
            "📅 Daily Trend",
            "📆 Weekly Trends",
            "🗓️ Monthly Trends",
            "🌦️ Seasonal & Historical Comparison",
        ])
        with t_tab_daily:
            d_recs = trends["daily_trend"]["records"]
            if d_recs:
                d_df = pd.DataFrame(d_recs)
                fig_td = go.Figure()
                fig_td.add_trace(go.Bar(x=d_df["date"], y=d_df["precipitation_mm"], name="Daily Rainfall (mm)", marker_color="#38bdf8", opacity=0.6))
                fig_td.add_trace(go.Scatter(x=d_df["date"], y=d_df["rolling_7d_mean_mm"], mode="lines", name="7d Rolling Mean", line=dict(color="#f59e0b", width=2.5)))
                fig_td.update_layout(title=f"Daily Rainfall Trend ({trends['daily_trend']['status']})", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.4)", font_color="#e2e8f0", height=290, margin=dict(l=10, r=10, t=35, b=20))
                st.plotly_chart(fig_td, width="stretch")
            else:
                st.info("Insufficient daily archive coverage to render multi-day trends.")

        with t_tab_weekly:
            w_recs = trends["weekly_trend"]["records"]
            if w_recs:
                w_df = pd.DataFrame(w_recs)
                fig_wk = px.bar(w_df, x="week", y="total_mm", text="valid_days", title=f"Weekly Rainfall Totals (Complete Weeks >=5d; Trend: {trends['weekly_trend']['status']})", color_discrete_sequence=["#22c55e"])
                fig_wk.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.4)", font_color="#e2e8f0", height=290, margin=dict(l=10, r=10, t=35, b=20))
                st.plotly_chart(fig_wk, width="stretch")
            else:
                st.info("Insufficient complete weeks (>=5 valid days/week required).")

        with t_tab_monthly:
            m_recs = trends["monthly_trend"]["records"]
            if m_recs:
                m_df = pd.DataFrame(m_recs)
                fig_mo = px.bar(m_df, x="month", y="total_mm", title=f"Monthly Rainfall Totals (>=20 valid days/month; Trend: {trends['monthly_trend']['status']})", color_discrete_sequence=["#a855f7"])
                fig_mo.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.4)", font_color="#e2e8f0", height=290, margin=dict(l=10, r=10, t=35, b=20))
                st.plotly_chart(fig_mo, width="stretch")
            else:
                st.info("Insufficient complete months (>=20 valid days/month required; enable 365-day baseline).")

        with t_tab_seasonal:
            hc = trends["historical_comparison"]
            if hc.get("baseline_available"):
                hc1, hc2, hc3, hc4 = st.columns(4)
                hc1.metric("Archive Daily Mean", f"{hc['archive_mean_daily_mm']:.2f} mm")
                hc2.metric("Archive Median / P90", f"{hc['archive_p50_mm']:.1f} / {hc['archive_p90_mm']:.1f} mm")
                hc3.metric("Trailing 7d Sum", f"{hc['recent_7d_mm']:.1f} mm" if hc.get("recent_7d_mm") is not None else "N/A")
                hc4.metric("Recent 24h vs Mean", f"{hc['recent_vs_archive_ratio']:.2f}x" if hc.get("recent_vs_archive_ratio") is not None else "N/A")
            s_df = pd.DataFrame(trends["seasonal_comparison"])
            if not s_df.empty:
                st.dataframe(s_df, width="stretch", hide_index=True)

        st.divider()
        st.markdown("### 🔮 Phase 6 — Experimental 24h / 48h / 72h Risk Trend Forecast")
        st.warning(FORECAST_TIMING_DISCLAIMER)
        fcst_risk = forecast_risk_trend(
            weather_payload=weather_payload,
            terrain_features={
                "slope_degrees": loc_meta.get("slope_degrees"),
                "elevation_m": loc_meta.get("elevation_m") or weather_payload.get("elevation_m"),
            },
        )
        hz = fcst_risk.get("horizons", {})
        h1, h2, h3, h4 = st.columns(4)
        for col_w, h_key in ((h1, "24h"), (h2, "48h"), (h3, "72h")):
            h_rec = hz.get(h_key, {})
            p_val = h_rec.get("probability")
            if p_val is not None:
                lo_p = h_rec.get("lower_probability")
                hi_p = h_rec.get("upper_probability")
                band_txt = f"Band: {lo_p*100:.0f}%–{hi_p*100:.0f}% ({h_rec.get('risk_category')})" if lo_p is not None else h_rec.get("risk_category")
                col_w.metric(f"+{h_key} Forecast Risk", f"{p_val*100:.1f}%", delta=band_txt)
            else:
                col_w.metric(f"+{h_key} Forecast Risk", "Unavailable", help=h_rec.get("status", "Incomplete forecast"))
        h4.metric(
            "Model Provenance",
            fcst_risk.get("model_name", "unknown"),
            delta="Synthetic Dev Model" if fcst_risk.get("model_is_synthetic_dev") else "Validated",
        )

        traj_valid = [t for t in (fcst_risk.get("trajectory") or []) if t.get("probability") is not None]
        if traj_valid:
            t_df = pd.DataFrame(traj_valid)
            fig_fr = go.Figure()
            fig_fr.add_trace(go.Scatter(
                x=t_df["horizon_hours"], y=t_df["upper_probability"],
                mode="lines", line=dict(width=0), showlegend=False, name="Upper Uncertainty",
            ))
            fig_fr.add_trace(go.Scatter(
                x=t_df["horizon_hours"], y=t_df["lower_probability"],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(168, 85, 247, 0.20)", name="Forecast Rainfall Sensitivity Band",
            ))
            fig_fr.add_trace(go.Scatter(
                x=t_df["horizon_hours"], y=t_df["probability"],
                mode="lines+markers", name="Nominal Experimental Risk Score",
                line=dict(color="#a855f7", width=3),
            ))
            fig_fr.add_hline(y=0.55, line_dash="dash", line_color="#f97316", annotation_text="Provisional High Threshold (0.55)")
            fig_fr.update_yaxes(range=[0.0, 1.05], title="Experimental Risk Score")
            fig_fr.update_xaxes(title="Forecast Lead Time (Hours Ahead)")
            fig_fr.update_layout(
                title="Forecast-Driven Risk Trend & Lead-Time Uncertainty Envelope (NOT Timing Prediction)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.4)",
                font_color="#e2e8f0", height=320, margin=dict(l=10, r=10, t=35, b=20),
            )
            st.plotly_chart(fig_fr, width="stretch")

            with st.expander("Forecast-Derived Inputs & Model Limitations", expanded=False):
                st.dataframe(
                    t_df[["horizon_hours", "forecast_time", "risk_category", "probability", "lower_probability", "upper_probability", "rolling_24h_mm", "rolling_72h_mm", "method", "forecast_derived_inputs"]],
                    width="stretch",
                    hide_index=True,
                )
                for lim in fcst_risk.get("limitations", []):
                    st.caption(f"• {lim}")
        else:
            st.info(f"24h/48h/72h Risk Trend unavailable: {fcst_risk.get('status')}")
    else:
        st.info("Click **Fetch Open-Meteo Data** above to load live rainfall analytics, or use **Manual Environmental Input Fallback** below.")

    # ── Manual Environmental Input Fallback ───────────────────────────
    st.divider()
    with st.expander("📝 Manual Environmental Input Fallback (User-Provided Data — Offline Safe)", expanded=bool(fetch_error)):
        st.warning("**User-Provided Data Notice:** Manual inputs below are NOT live weather telemetry and are labelled `User-Provided Data (Manual Fallback)` in all records.")
        with st.form("manual_rainfall_fallback_form"):
            mf1, mf2, mf3, mf4 = st.columns(4)
            with mf1:
                m_rf1 = st.number_input("1h Rainfall (mm)", min_value=0.0, max_value=500.0, value=12.0, step=1.0)
                m_slope = st.number_input("Slope (degrees)", min_value=0.0, max_value=90.0, value=float(default_slope or 25.0), step=1.0)
            with mf2:
                m_rf24 = st.number_input("24h Accumulated Rainfall (mm)", min_value=0.0, max_value=2000.0, value=65.0, step=5.0)
                m_elev = st.number_input("Elevation (m)", min_value=-500.0, max_value=9000.0, value=float(default_elev or 1200.0), step=50.0)
            with mf3:
                m_rf48 = st.number_input("48h Accumulated Rainfall (mm)", min_value=0.0, max_value=3500.0, value=95.0, step=5.0)
                m_soil = st.slider("Soil Moisture (m³/m³)", min_value=0.0, max_value=1.0, value=0.45, step=0.02)
            with mf4:
                m_rf72 = st.number_input("72h Accumulated Rainfall (mm)", min_value=0.0, max_value=5000.0, value=130.0, step=5.0)
            manual_submit = st.form_submit_button("🧮 Evaluate Experimental Risk (Manual User Input)", type="primary")

        if manual_submit:
            manual_features = {
                "latitude": rain_lat,
                "longitude": rain_lon,
                "slope_degrees": m_slope,
                "elevation_m": m_elev,
                "rainfall_1h_mm": m_rf1,
                "rainfall_24h_mm": m_rf24,
                "rainfall_48h_mm": m_rf48,
                "rainfall_72h_mm": m_rf72,
                "soil_moisture": m_soil,
            }
            st.session_state["rain_risk_delta"] = evaluate_rainfall_risk_change(
                features=manual_features,
                observation_time=datetime.now(timezone.utc),
                location_id=default_loc_id,
                location_name=default_loc_name,
                source_label="User-Provided Data (Manual Fallback)",
                is_manual=True,
                persist=True,
            )

    # ── Risk Change Explanation Panel ─────────────────────────────────
    delta_res = st.session_state.get("rain_risk_delta")
    if delta_res:
        st.markdown("### 🧠 Risk Change Explanation")
        prev_est = delta_res["previous_estimate"]
        curr_est = delta_res["current_estimate"]
        fresh_info = delta_res["data_freshness"]

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric(
            "Previous Risk Estimate",
            prev_est["risk_category"] or "None",
            delta=f"{prev_est['probability']*100:.1f}%" if prev_est["probability"] is not None else "No prior record",
        )
        rc2.metric(
            "Current Risk Estimate",
            curr_est["risk_category"],
            delta=(
                f"{delta_res['probability_delta']*100:+.1f}% vs prior"
                if delta_res["probability_delta"] is not None
                else (f"{curr_est['probability']*100:.1f}%" if curr_est["probability"] is not None else None)
            ),
        )
        rc3.metric("Category Change", delta_res["category_change"])
        rc4.metric("Data Freshness", fresh_info["status"])

        st.caption(
            f"**Input Provenance:** `{delta_res['source_label']}` · "
            f"**Manual Input:** `{delta_res['is_manual_input']}` · "
            f"**Method:** `{curr_est.get('method')}` · "
            f"{delta_res['scientific_disclaimer']}"
        )

        if delta_res.get("errors"):
            for err_msg in delta_res["errors"]:
                st.error(err_msg)
        if delta_res["warnings"]:
            for w_msg in delta_res["warnings"]:
                st.warning(w_msg)

        factors = delta_res.get("influencing_factors", [])
        if factors:
            st.dataframe(pd.DataFrame(factors), width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4: RISK PREDICTION
# ═══════════════════════════════════════════════════════════════

elif page == "🔮 Risk Analysis":
    render_page_header(
        "🔮 Dynamic Risk Prediction",
        "Estimate point landslide risk from terrain and rainfall features with SHAP feature attribution.",
        pills=[("Live Open-Meteo Ready", "pill-blue"), ("SHAP Explainability", "pill-green")],
    )
    experimental_banner("Predictions use a model trained on synthetic data. The actual risk at any real location is unknown.")

    from src.risk.risk_engine import get_risk_engine
    from src.risk.thresholds import RISK_COLORS, RISK_EMOJI

    # Initialize prediction widget defaults cleanly in session_state
    _pred_defaults = {
        "pred_lat": 22.5,
        "pred_lon": 78.0,
        "pred_slope": 25.0,
        "pred_elev": 800.0,
        "pred_rf1": 15.0,
        "pred_rf24": 60.0,
        "pred_rf72": 120.0,
        "pred_veg": 0.35,
        "pred_moist": 0.45,
        "pred_preset": "Custom Input",
    }
    for _k, _v in _pred_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # Quick scenario bar (uses selectbox on_change + optional live weather/terrain pull)
    pcol1, pcol2 = st.columns([2.2, 1])
    with pcol1:
        st.selectbox(
            "Quick Scenario / Location Preset",
            list(SCENARIO_PRESETS.keys()),
            key="pred_preset",
            on_change=_apply_scenario_preset,
            help="Select a preset to auto-populate terrain and rainfall values below.",
        )
    with pcol2:
        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
        live_fetch = st.button("🌦️ Pull Live Open-Meteo Rainfall", width="stretch",
                               help="Fetches recent 1h/24h/72h model-derived precipitation for the current latitude/longitude.")

    if live_fetch:
        try:
            from src.data_providers.weather_provider import WeatherProvider
            wp = WeatherProvider()
            rain = wp.get_rainfall_accumulations(st.session_state["pred_lat"], st.session_state["pred_lon"])
            if rain.get("rainfall_24h_mm") is not None:
                st.session_state["pred_rf1"] = float(rain["rainfall_1h_mm"] or 0.0)
                st.session_state["pred_rf24"] = float(rain["rainfall_24h_mm"] or 0.0)
                st.session_state["pred_rf72"] = float(rain["rainfall_72h_mm"] or 0.0)
                st.success(
                    f"Updated rainfall from {rain.get('source', 'Open-Meteo')}: "
                    f"1h={rain['rainfall_1h_mm']}mm, 24h={rain['rainfall_24h_mm']}mm, 72h={rain['rainfall_72h_mm']}mm"
                )
            else:
                st.warning("Live weather fetch unavailable (offline or rate-limited). Using current form inputs.")
        except Exception as exc:
            st.warning(f"Live weather lookup skipped: {exc}")

    st.markdown("### Input Environmental & Terrain Features")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📍 Location & Terrain**")
            latitude = st.number_input("Latitude (°)", -90.0, 90.0, step=0.001, format="%.4f", key="pred_lat")
            longitude = st.number_input("Longitude (°)", -180.0, 180.0, step=0.001, format="%.4f", key="pred_lon")
            slope = st.slider("Slope Angle (°)", 0.0, 90.0, step=0.5, key="pred_slope")
            elevation = st.number_input("Elevation (m)", 0.0, 9000.0, step=10.0, key="pred_elev")
        with col2:
            st.markdown("**🌧️ Rainfall & Vegetation**")
            rf_1h = st.number_input("Rainfall — 1h (mm)", 0.0, 500.0, step=1.0, key="pred_rf1")
            rf_24h = st.number_input("Rainfall — 24h (mm)", 0.0, 2000.0, step=1.0, key="pred_rf24")
            rf_72h = st.number_input("Rainfall — 72h (mm)", 0.0, 5000.0, step=1.0, key="pred_rf72")
            veg = st.slider("NDVI Vegetation Index", -1.0, 1.0, step=0.01, key="pred_veg")
        with col3:
            st.markdown("**💧 Soil & Observation Time**")
            moisture = st.slider("Soil Moisture (0–1)", 0.0, 1.0, step=0.01, key="pred_moist")
            land_cover = st.selectbox("Land Cover (context)", ["Forest", "Shrubland", "Grassland", "Cropland", "Bare"])
            soil_type = st.selectbox("Soil Type (context)", ["Clay", "Sandy Loam", "Loam", "Silty Clay", "Rock"])
            obs_date = st.date_input("Observation Date", value=datetime.now().date())
            obs_hour = st.slider("Observation Hour (UTC)", 0, 23, datetime.now().hour)

        submitted = st.form_submit_button("🔍 Estimate Risk", width="stretch", type="primary")

    if submitted:
        obs_time = datetime.combine(obs_date, _dt.time(obs_hour), tzinfo=timezone.utc)
        features = {
            "latitude": latitude, "longitude": longitude,
            "slope_degrees": slope, "elevation_m": elevation,
            "rainfall_1h_mm": rf_1h, "rainfall_24h_mm": rf_24h,
            "rainfall_72h_mm": rf_72h, "vegetation_index": veg,
            "soil_moisture": moisture,
        }

        with st.spinner("Computing risk estimate…"):
            engine = get_risk_engine()
            result = engine.estimate_risk(
                features=features,
                observation_time=obs_time,
                location_name=f"({latitude:.4f}, {longitude:.4f})",
            )
        st.session_state["last_prediction_result"] = result
        st.session_state["last_prediction_features"] = features
        st.session_state["last_prediction_time"] = obs_time.strftime("%Y-%m-%d %H:00 UTC")

    if "last_prediction_result" in st.session_state:
        result = st.session_state["last_prediction_result"]
        feat_snapshot = st.session_state["last_prediction_features"]
        prob = result.get("probability")
        category = result.get("risk_category", "Unknown")
        method = result.get("method", "unknown")
        color = RISK_COLORS.get(category, "#95a5a6")
        emoji = RISK_EMOJI.get(category, "⚪")

        st.divider()
        st.markdown("### 📊 Risk Estimate Result")

        r1, r2, r3 = st.columns([1.15, 1, 1.35])
        with r1:
            st.markdown(
                f'<div style="background:{color}1a; border:2px solid {color}; border-radius:14px; '
                f'padding:1.35rem; text-align:center; box-shadow: 0 6px 20px {color}22;">'
                f'<div style="font-size:2.4rem; line-height:1.1;">{emoji}</div>'
                f'<div style="font-size:1.55rem; font-weight:700; color:{color}; margin-top:0.25rem;">{category}</div>'
                f'<div style="font-size:0.82rem; color:#94a3b8; margin-top:0.2rem;">Provisional Risk Tier</div></div>',
                unsafe_allow_html=True,
            )
        with r2:
            st.metric("Probability Score", f"{prob:.4f}" if prob is not None else "N/A",
                      help="Output probability from the active classifier")
            st.metric("Method", method.replace("_", " ").title())
            dq = result.get("data_quality", {})
            st.metric("Feature Completeness", f"{dq.get('completeness_pct', 'N/A')}%")
        with r3:
            if prob is not None:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%", "font": {"size": 30, "color": "#f1f5f9"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 35], "color": "#133325"},
                            {"range": [35, 55], "color": "#3d2e07"},
                            {"range": [55, 75], "color": "#451818"},
                            {"range": [75, 100], "color": "#2f1147"},
                        ],
                        "threshold": {
                            "line": {"color": "#f8fafc", "width": 2},
                            "thickness": 0.78,
                            "value": round(prob * 100, 1),
                        },
                    },
                    domain={"x": [0.05, 0.95], "y": [0, 1]},
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                    height=200, margin=dict(t=16, b=8, l=24, r=24),
                )
                st.plotly_chart(fig_gauge, width="stretch")

        # 4-Tier Threshold Indicator Strip
        tiers = [
            ("Low", "< 35%", "#2ecc71"),
            ("Moderate", "35% – 55%", "#f39c12"),
            ("High", "55% – 75%", "#e74c3c"),
            ("Very High", "≥ 75%", "#8e44ad"),
        ]
        tier_html = '<div class="tier-strip">'
        for t_name, t_rng, t_col in tiers:
            active_style = (
                f"background:{t_col}24; border:1px solid {t_col}; color:#f8fafc; font-weight:700;"
                if t_name == category else ""
            )
            tier_html += f'<div class="tier-chip" style="{active_style}">{t_name}<br><small>{t_rng}</small></div>'
        tier_html += "</div>"
        st.markdown(tier_html, unsafe_allow_html=True)

        if result.get("warnings"):
            with st.expander(f"⚠️ Validation & Data Quality Notes ({len(result['warnings'])})", expanded=False):
                for w in result["warnings"]:
                    st.warning(w)
        if result.get("errors"):
            for e in result["errors"]:
                st.error(e)

        st.divider()
        ex_left, ex_right = st.columns([1.2, 1])

        with ex_left:
            st.markdown("### 🧠 SHAP Individual Prediction Explanation")
            try:
                from src.models.explainability import explain_prediction

                df_bg = load_dataset()
                exp_res = explain_prediction(
                    features=feat_snapshot,
                    model_name="best_model",
                    background_df=df_bg.head(120) if df_bg is not None else None,
                    use_cache=True,
                )
                _render_individual_shap_panel(
                    exp_res,
                    compact=True,
                    show_feature_table=False,
                    show_concepts=True,
                )
            except Exception as ex:
                st.warning(f"Explainability module error: {ex}")

        with ex_right:
            st.markdown("### ⚖️ Input Profile vs Dataset Median")
            try:
                from src.models.predict import load_feature_columns, load_training_stats
                feat_cols = load_feature_columns()
                medians = load_training_stats().get("feature_medians", {})
                comp_rows = []
                for c in feat_cols:
                    u_val = feat_snapshot.get(c)
                    m_val = medians.get(c)
                    if u_val is not None and m_val is not None:
                        diff_pct = ((float(u_val) - float(m_val)) / max(abs(float(m_val)), 1e-6)) * 100
                        comp_rows.append({
                            "Feature": c,
                            "Input": round(float(u_val), 3),
                            "Dataset Median": round(float(m_val), 3),
                            "vs Median": f"{diff_pct:+.0f}%",
                        })
                if comp_rows:
                    st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True, height=300)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# PAGE 4: HISTORICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════

elif page == "📊 Historical Events":
    render_page_header(
        "📊 Historical & Pattern Analysis",
        "Explore temporal trends, environmental distributions, and correlations across the dataset.",
        pills=[("Synthetic Dev Dataset", "pill-amber")],
    )
    experimental_banner("Analysis uses synthetic development data, not real landslide records.")

    df = load_dataset()
    if df is None:
        st.error("Dataset not found.")
        st.stop()

    is_synthetic = ("is_synthetic" in df.columns and df["is_synthetic"].all())
    if is_synthetic:
        st.warning("⚠️ All charts use **synthetic test data**. They illustrate pipeline analytics but do not represent actual landslide statistics.", icon="⚠️")

    events = df[df["landslide_occurred"] == 1].copy() if "landslide_occurred" in df.columns else df.copy()
    tab_temp, tab_env, tab_corr = st.tabs([
        "📅 Temporal & Seasonal Trends",
        "🌧️ Environmental Distributions",
        "🔗 Correlations & Statistics",
    ])

    with tab_temp:
        ha1, ha2 = st.columns(2)
        with ha1:
            st.markdown("#### 📅 Events by Year")
            if "event_date" in events.columns:
                try:
                    events["year"] = pd.to_datetime(events["event_date"], errors="coerce").dt.year
                    yr_cnt = events.groupby("year").size().reset_index(name="Count")
                    fig_yr = px.bar(yr_cnt, x="year", y="Count", color="Count", color_continuous_scale="Reds")
                    _style_fig(fig_yr, height=310)
                    fig_yr.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_yr, width="stretch")
                except Exception:
                    st.info("Date parsing failed.")

        with ha2:
            st.markdown("#### 🗓️ Seasonal Pattern (Events by Month)")
            if "event_date" in events.columns:
                try:
                    events["month"] = pd.to_datetime(events["event_date"], errors="coerce").dt.month
                    month_cnt = events.groupby("month").size().reset_index(name="Count")
                    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                    month_cnt["Month"] = month_cnt["month"].map(month_names)
                    fig_mon = px.bar(
                        month_cnt, x="Month", y="Count", color="Count",
                        color_continuous_scale="Reds",
                        category_orders={"Month": list(month_names.values())},
                    )
                    _style_fig(fig_mon, height=310)
                    fig_mon.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_mon, width="stretch")
                    st.caption("Synthetic data replicates a Jun–Sep monsoon bias by design; do not cite as real seasonal evidence.")
                except Exception:
                    pass

    with tab_env:
        ha3, ha4 = st.columns(2)
        with ha3:
            st.markdown("#### 🌧️ 24h Rainfall Distribution")
            if "rainfall_24h_mm" in df.columns and "landslide_occurred" in df.columns:
                plot_r = df.head(2000).copy()
                plot_r["Outcome"] = plot_r["landslide_occurred"].map({0: "No Event", 1: "Landslide"})
                fig_rain = px.histogram(
                    plot_r, x="rainfall_24h_mm", color="Outcome",
                    color_discrete_map={"No Event": "#2ecc71", "Landslide": "#e74c3c"},
                    barmode="overlay", nbins=40, opacity=0.72,
                    labels={"rainfall_24h_mm": "24h Rainfall (mm)"},
                )
                _style_fig(fig_rain, height=310)
                st.plotly_chart(fig_rain, width="stretch")

        with ha4:
            st.markdown("#### ⛰️ Slope Angle Distribution")
            if "slope_degrees" in df.columns and "landslide_occurred" in df.columns:
                fig_slope = px.box(
                    df, x=df["landslide_occurred"].map({0: "No Event", 1: "Landslide"}),
                    y="slope_degrees",
                    color=df["landslide_occurred"].map({0: "No Event", 1: "Landslide"}),
                    color_discrete_map={"No Event": "#2ecc71", "Landslide": "#e74c3c"},
                    labels={"x": "Outcome", "slope_degrees": "Slope (°)"},
                )
                _style_fig(fig_slope, height=310, showlegend=False)
                st.plotly_chart(fig_slope, width="stretch")

        st.markdown("#### 🌱 Vegetation Index vs Slope (Landslide Events)")
        if "vegetation_index" in events.columns and "slope_degrees" in events.columns:
            fig_veg = px.scatter(
                events.head(500), x="slope_degrees", y="vegetation_index",
                color="rainfall_24h_mm" if "rainfall_24h_mm" in events.columns else None,
                color_continuous_scale="Reds", opacity=0.72,
                labels={"slope_degrees": "Slope (°)", "vegetation_index": "NDVI", "rainfall_24h_mm": "24h Rain (mm)"},
            )
            _style_fig(fig_veg, height=320)
            st.plotly_chart(fig_veg, width="stretch")

    with tab_corr:
        num_cols = [c for c in ["slope_degrees", "elevation_m", "rainfall_1h_mm", "rainfall_24h_mm",
                                 "rainfall_72h_mm", "vegetation_index", "soil_moisture", "landslide_occurred"]
                    if c in df.columns]
        if len(num_cols) >= 2:
            corr = df[num_cols].corr(numeric_only=True).round(2)
            fig_corr = px.imshow(
                corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Feature Correlation Matrix (Pearson)",
            )
            _style_fig(fig_corr, height=400)
            st.plotly_chart(fig_corr, width="stretch")

        st.markdown("#### 🔢 Summary Statistics")
        stat_cols = [c for c in num_cols if c != "landslide_occurred"]
        if stat_cols:
            st.dataframe(df[stat_cols].describe().T.style.format("{:.2f}"), width="stretch")


# ═══════════════════════════════════════════════════════════════
# PAGE 5: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════

elif page == "🤖 Model Performance":
    metrics = load_metrics()
    model_info = load_model_info()
    best_alg = (model_info.get("metadata", {}).get("algorithm", "") or "").replace("_", " ").title() or "Unknown"

    render_page_header(
        "🤖 Model Performance & Diagnostics",
        "Validation benchmark across candidate models and held-out chronological test evaluation.",
        pills=[(f"🏆 Selected: {best_alg}", "pill-blue"), ("Synthetic Dev Data", "pill-amber")],
    )
    experimental_banner("All metrics are from evaluation on <b>synthetic development data</b>.")

    if not metrics:
        st.error("No metrics found. Run: `python -m src.models.train`")
        st.stop()

    st.markdown("### 📊 Candidate Model Comparison (Validation Split)")
    val_results = {k: v for k, v in metrics.items() if "_test" not in k and isinstance(v, dict) and "split" in v}
    if val_results:
        rows = []
        for name, m in val_results.items():
            rows.append({
                "Model": name.replace("_", " ").title(),
                "Precision": round(m.get("precision", 0), 4),
                "Recall": round(m.get("recall", 0), 4),
                "F1": round(m.get("f1", 0), 4),
                "ROC-AUC": round(m.get("roc_auc") or 0, 4),
                "PR-AUC": round(m.get("pr_auc") or 0, 4),
                "Brier Score": round(m.get("brier_score") or 0, 4),
            })
        comp_df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
        mc1, mc2 = st.columns([1.25, 1])
        with mc1:
            st.dataframe(
                comp_df.style
                .highlight_max(subset=["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], color="#1a4731")
                .highlight_min(subset=["Brier Score"], color="#1a4731")
                .format("{:.4f}", subset=["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier Score"]),
                width="stretch", hide_index=True,
            )
        with mc2:
            melted = comp_df.melt(id_vars="Model", value_vars=["PR-AUC", "ROC-AUC", "F1"],
                                  var_name="Metric", value_name="Score")
            fig_comp = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group",
                              color_discrete_sequence=["#38bdf8", "#4ade80", "#f59e0b"],
                              title="Validation Metrics by Candidate")
            fig_comp.update_yaxes(range=[0.85, 1.02])
            _style_fig(fig_comp, height=260)
            st.plotly_chart(fig_comp, width="stretch")

    st.divider()
    st.markdown(f"### 🏆 Best Model ({best_alg}) — Held-Out Test Split")
    test_key = next((k for k in metrics if "_test" in k), None)
    if test_key:
        tm = metrics[test_key]
        t1, t2, t3, t4, t5 = st.columns(5)
        with t1:
            st.metric("Precision", f"{tm.get('precision', 0):.4f}")
        with t2:
            st.metric("Recall", f"{tm.get('recall', 0):.4f}")
        with t3:
            st.metric("F1-Score", f"{tm.get('f1', 0):.4f}")
        with t4:
            st.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.4f}" if tm.get("roc_auc") else "N/A")
        with t5:
            st.metric("PR-AUC", f"{tm.get('pr_auc', 0):.4f}" if tm.get("pr_auc") else "N/A")

        cm = tm.get("confusion_matrix")
        roc = tm.get("roc_curve") or (
            model_info.get("metadata", {}).get("test_metrics", {}).get("roc_curve")
            if model_info else None
        )
        if cm or roc:
            p1, p2 = st.columns([1, 1])
            with p1:
                if cm:
                    fig_cm = px.imshow(
                        [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
                        x=["Predicted: No Event", "Predicted: Event"],
                        y=["Actual: No Event", "Actual: Event"],
                        text_auto=True, color_continuous_scale="Blues", zmin=0,
                        title="Confusion Matrix (Test Set)",
                    )
                    _style_fig(fig_cm, height=330)
                    st.plotly_chart(fig_cm, width="stretch")
            with p2:
                if roc:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=roc["fpr"], y=roc["tpr"], mode="lines",
                        name=f"ROC (AUC={tm.get('roc_auc', 0):.3f})",
                        line=dict(color="#38bdf8", width=2.5),
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(color="#64748b", dash="dash"), name="Random",
                    ))
                    fig_roc.update_layout(title="ROC Curve (Test Set)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                    _style_fig(fig_roc, height=330)
                    st.plotly_chart(fig_roc, width="stretch")

    st.divider()
    st.markdown("### 🔑 Feature Importance")
    try:
        from src.models.model_registry import load_model, active_model_exists
        from src.models.predict import load_feature_columns
        if active_model_exists():
            model = load_model("best_model")
            feat_cols = load_feature_columns()
            clf = model.named_steps.get("clf", model)
            if hasattr(clf, "feature_importances_"):
                imp, label = clf.feature_importances_, "Tree Feature Importance"
            elif hasattr(clf, "coef_"):
                imp, label = abs(clf.coef_[0]), "Logistic Regression |Standardized Coefficient|"
            else:
                imp = None
            if imp is not None:
                imp_df = pd.DataFrame({"Feature": feat_cols, "Importance": imp}).sort_values("Importance", ascending=True)
                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#1e3a5f", "#38bdf8"],
                    title=f"{label} ({best_alg})",
                )
                _style_fig(fig_imp, height=340)
                fig_imp.update_coloraxes(showscale=False)
                st.plotly_chart(fig_imp, width="stretch")
    except Exception as e:
        st.info(f"Feature importance not available: {e}")

    st.divider()
    st.markdown("### 🛡️ Phase 6 — Model Drift & Data Quality Monitoring")
    from src.database.models import insert_prediction_outcome
    from src.monitoring.drift_monitor import compute_model_drift_report

    drift = compute_model_drift_report(limit=300)
    dq_sum = drift["data_quality_summary"]
    overall_drift = drift["overall_status"]

    if drift["drift_confirmed"]:
        st.error(f"**Drift Status: {overall_drift}** — Indicators: {', '.join(drift['confirmed_indicators'])}")
    elif "Insufficient" in overall_drift:
        st.info(f"**Drift Status: {overall_drift}** — Log at least {dq_sum['min_feature_samples_required']} predictions or {dq_sum['min_outcome_samples_required']} verified outcomes before inferring drift.")
    elif "Watchlist" in overall_drift:
        st.warning(f"**Drift Status: {overall_drift}** — Signals: {', '.join(drift['watchlist_signals'])}")
    else:
        st.success(f"**Drift Status: {overall_drift}**")

    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric("Prediction Records Analyzed", dq_sum["prediction_records_analyzed"])
    dm2.metric("Feature Vectors Evaluated", dq_sum["feature_vectors_analyzed"])
    dm3.metric(
        "Avg Feature Completeness",
        f"{dq_sum['avg_feature_completeness_pct']:.1f}%" if dq_sum["avg_feature_completeness_pct"] is not None else "Unavailable",
    )
    dm4.metric("Verified Outcomes Logged", dq_sum["labelled_outcomes_count"])

    st.caption(f"**Reference Baseline:** `{drift['reference_source']}` · {drift['methodology_note']}")

    dtab_feat, dtab_miss, dtab_pred, dtab_perf = st.tabs([
        "📐 Feature Distribution Drift (PSI / KS)",
        "🧩 Missing Data Monitor",
        "📈 Prediction Score Shift",
        "🎯 Labelled Outcome Performance",
    ])

    with dtab_feat:
        fd_df = pd.DataFrame(drift["feature_drift"])
        if not fd_df.empty:
            st.dataframe(fd_df, width="stretch", hide_index=True)
            valid_psi = fd_df.dropna(subset=["psi"])
            if not valid_psi.empty:
                fig_psi = px.bar(
                    valid_psi, x="psi", y="feature", orientation="h", color="status",
                    title="Feature Population Stability Index (PSI vs Training Reference)",
                )
                fig_psi.add_vline(x=0.10, line_dash="dot", line_color="#eab308", annotation_text="Moderate (0.10)")
                fig_psi.add_vline(x=0.25, line_dash="dash", line_color="#ef4444", annotation_text="Significant (0.25)")
                _style_fig(fig_psi, height=300)
                st.plotly_chart(fig_psi, width="stretch")

    with dtab_miss:
        ms_df = pd.DataFrame(drift["missingness_drift"])
        if not ms_df.empty:
            st.dataframe(ms_df, width="stretch", hide_index=True)
            fig_ms = px.bar(
                ms_df, x="feature", y="current_missing_rate", color="status",
                title="Current Missing / Imputed Feature Rate (Threshold Delta: +10%)",
            )
            _style_fig(fig_ms, height=280)
            st.plotly_chart(fig_ms, width="stretch")

    with dtab_pred:
        pd_info = drift["prediction_drift"]
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Current Mean Score", f"{pd_info['current_mean_probability']:.3f}" if pd_info["current_mean_probability"] is not None else "N/A")
        pc2.metric("Reference Mean Score", f"{pd_info['reference_mean_probability']:.3f}" if pd_info["reference_mean_probability"] is not None else "N/A")
        pc3.metric("Prediction PSI", f"{pd_info['psi']:.4f}" if pd_info["psi"] is not None else pd_info["status"])
        if pd_info.get("current_probabilities") and pd_info.get("reference_probabilities"):
            comp_hist = pd.DataFrame(
                [{"probability": p, "Cohort": "Reference Training"} for p in pd_info["reference_probabilities"]]
                + [{"probability": p, "Cohort": "Recent Production"} for p in pd_info["current_probabilities"]]
            )
            fig_pd = px.histogram(
                comp_hist, x="probability", color="Cohort", barmode="overlay", histnorm="probability density", nbins=20,
                title="Prediction Probability Distribution: Reference vs Production",
            )
            _style_fig(fig_pd, height=290)
            st.plotly_chart(fig_pd, width="stretch")

    with dtab_perf:
        perf_d = drift["performance_drift"]
        st.info(f"**Labelled Evaluation Status:** {perf_d['status']}")
        if perf_d.get("live_metrics"):
            lm = perf_d["live_metrics"]
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Live Accuracy", f"{lm['accuracy']:.3f}")
            lc2.metric("Live F1", f"{lm['f1']:.3f}", delta=f"{perf_d['f1_delta']:+.3f} vs test" if perf_d["f1_delta"] is not None else None)
            lc3.metric("Live Brier", f"{lm['brier_score']:.4f}", delta=f"{perf_d['brier_delta']:+.4f} vs test" if perf_d["brier_delta"] is not None else None)
            lc4.metric("ROC-AUC", f"{lm['roc_auc']:.3f}" if lm.get("roc_auc") is not None else "Single Class")

        with st.expander("➕ Record Verified Ground-Truth Outcome (for Live Performance Drift)", expanded=False):
            with st.form("log_outcome_form"):
                oc1, oc2, oc3 = st.columns([0.9, 0.9, 1.4])
                with oc1:
                    out_pred_id = st.number_input("Prediction ID (from SQLite)", min_value=1, step=1, value=1, key="out_pred_id")
                with oc2:
                    out_label = st.selectbox("Observed Outcome", [0, 1], format_func=lambda x: "1 — Landslide Occurred" if x == 1 else "0 — No Event", key="out_label")
                with oc3:
                    out_src = st.text_input("Verification Source", "Field / News Verification", key="out_src")
                out_note = st.text_input("Notes (optional)", key="out_note")
                if st.form_submit_button("💾 Save Verified Outcome", width="stretch"):
                    try:
                        oid = insert_prediction_outcome(
                            actual_label=int(out_label),
                            prediction_id=int(out_pred_id),
                            outcome_source=out_src or "Operator Verification",
                            notes=out_note or None,
                        )
                        st.success(f"Recorded verified outcome #{oid} for prediction #{int(out_pred_id)}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not save outcome (verify prediction ID exists): {exc}")

    st.divider()
    with st.expander("⚠️ Known Modeling Caveats & Limitations", expanded=False):
        st.markdown("""
- All metrics computed on **synthetic development data** — do not reflect real-world performance.
- Synthetic data is separable by construction → inflated precision/recall/AUC metrics.
- No spatial cross-validation without authentic geo-referenced inventories.
- Class imbalance handled with `class_weight='balanced'` (`scale_pos_weight` for XGBoost).
""")


# ═══════════════════════════════════════════════════════════════
# PAGE 8: EXPLAINABLE AI (SHAP & GLOBAL IMPORTANCE)
# ═══════════════════════════════════════════════════════════════

elif page == "🧠 AI Explainability":
    from src.models.explainability import (
        EXPLAINABILITY_CONCEPTS,
        NON_CAUSAL_DISCLAIMER,
        explain_global,
        explain_prediction,
    )
    from src.models.model_registry import list_models

    render_page_header(
        "🧠 Explainable AI (SHAP & Feature Importance)",
        "Inspect global feature sensitivity, SHAP summary & beeswarm impact distributions, and sample-level waterfall attributions from trained model artifacts.",
        pills=[("SHAP Attribution", "pill-blue"), ("Permutation Importance", "pill-green"), ("Non-Causal", "pill-amber")],
    )
    experimental_banner(NON_CAUSAL_DISCLAIMER)

    reg_models = list_models()
    model_names = [m["name"] for m in reg_models] if reg_models else ["best_model"]
    if "best_model" in model_names:
        model_names = ["best_model"] + [m for m in model_names if m != "best_model"]

    xai_c1, xai_c2, xai_c3 = st.columns([1.4, 1.0, 0.8])
    with xai_c1:
        selected_model_name = st.selectbox(
            "Registered Model Artifact",
            model_names,
            key="xai_model_select",
            help="Select active best_model or any trained candidate model in models/registry.json.",
        )
    with xai_c2:
        sample_budget = st.slider(
            "Evaluation Representative Sample Size",
            min_value=40,
            max_value=250,
            value=120,
            step=20,
            key="xai_sample_size",
        )
    with xai_c3:
        n_perm = st.slider(
            "Permutation Repeats",
            min_value=3,
            max_value=15,
            value=6,
            step=1,
            key="xai_perm_repeats",
        )

    @st.cache_data(ttl=600, show_spinner=False)
    def _cached_global_xai(m_name: str, max_samp: int, perms: int) -> dict:
        return explain_global(
            model_name=m_name,
            max_samples=max_samp,
            n_permutations=perms,
            use_cache=True,
        )

    with st.spinner(f"Computing global explainability & SHAP summary for '{selected_model_name}'..."):
        global_res = _cached_global_xai(selected_model_name, sample_budget, n_perm)

    if not global_res.get("available"):
        for err_msg in global_res.get("errors", ["Global model explanation unavailable."]):
            st.error(err_msg)
    else:
        m_meta = global_res.get("model_metadata") or {}
        st.info(
            f"**Model Version:** `{global_res['model_version']}` · "
            f"**Explainer:** `{global_res.get('explainer_type') or 'Unavailable'}` · "
            f"**SHAP Units:** `{global_res.get('shap_unit') or 'N/A'}` · "
            f"**Representative Sample:** `{global_res['sample_count']} / {global_res.get('total_eval_rows', 0)}` held-out test rows"
        )
        with st.expander("📄 Model Information, Training Provenance & Evaluation Metrics", expanded=True):
            mi1, mi2, mi3, mi4 = st.columns(4)
            mi1.metric("Model Name", str(m_meta.get("model_name", selected_model_name)))
            mi2.metric("Algorithm", str(m_meta.get("algorithm", "Unavailable")))
            mi3.metric("Training Date", str(m_meta.get("training_date", "Unavailable")))
            eval_m = m_meta.get("evaluation_metrics") or {}
            test_m = eval_m.get("test") or {}
            val_m = eval_m.get("validation") or {}
            active_m = test_m if test_m else val_m
            active_split_lbl = "Test PR-AUC" if test_m else "Val PR-AUC"
            mi4.metric(
                active_split_lbl,
                f"{active_m['pr_auc']:.4f}" if active_m.get("pr_auc") is not None else "Unavailable",
                delta=f"F1={active_m['f1']:.4f}" if active_m.get("f1") is not None else None,
            )
            st.markdown(
                f"- **Model Version:** `{m_meta.get('model_version', global_res['model_version'])}`\n"
                f"- **Training Dataset Source:** `{m_meta.get('training_dataset_source', 'Unavailable')}`\n"
                f"- **Validation Approach:** {m_meta.get('validation_approach', 'Unavailable')}"
            )
            if test_m or val_m:
                metric_rows = []
                for split_lbl, m_dict in [("Validation", val_m), ("Test (Held-Out)", test_m)]:
                    if m_dict:
                        metric_rows.append({
                            "Split": split_lbl,
                            "Precision": round(float(m_dict["precision"]), 4) if m_dict.get("precision") is not None else None,
                            "Recall": round(float(m_dict["recall"]), 4) if m_dict.get("recall") is not None else None,
                            "F1": round(float(m_dict["f1"]), 4) if m_dict.get("f1") is not None else None,
                            "ROC-AUC": round(float(m_dict["roc_auc"]), 4) if m_dict.get("roc_auc") is not None else None,
                            "PR-AUC": round(float(m_dict["pr_auc"]), 4) if m_dict.get("pr_auc") is not None else None,
                            "Brier Score": round(float(m_dict["brier_score"]), 4) if m_dict.get("brier_score") is not None else None,
                        })
                if metric_rows:
                    st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
            feat_descs = m_meta.get("feature_descriptors") or []
            if feat_descs:
                st.caption(
                    "**Active Trained Features Only:** "
                    + ", ".join(f"`{fd['feature']}` ({fd['category']})" for fd in feat_descs)
                )

        for warn_msg in global_res.get("warnings", []):
            st.warning(warn_msg)

        # ── Global Importance Charts Row ──────────────────────────────
        shap_sum = global_res.get("shap_summary", [])
        perm_list = global_res.get("permutation_importance", [])
        int_list = global_res.get("feature_importance", [])
        comp_rows = global_res.get("method_comparison", [])

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 1. Global SHAP Summary Ranking (Mean |SHAP|)")
            if global_res.get("shap_available") and shap_sum:
                ss_df = pd.DataFrame(shap_sum).sort_values("mean_abs_shap", ascending=True)
                fig_ss = px.bar(
                    ss_df,
                    x="mean_abs_shap",
                    y="feature",
                    color="feature_category" if "feature_category" in ss_df.columns else None,
                    orientation="h",
                    labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": "Feature", "feature_category": "Domain"},
                    title="Most Influential Trained Features (Mean |SHAP|)",
                )
                _style_fig(fig_ss, height=340)
                st.plotly_chart(fig_ss, width="stretch")
            else:
                st.warning(global_res.get("shap_error") or "SHAP summary unavailable for this model.")

        with g2:
            st.markdown("#### 2. Permutation Feature Importance Ranking")
            if perm_list:
                pm_df = pd.DataFrame(perm_list).sort_values("importance_mean", ascending=True)
                fig_pm = go.Figure(
                    go.Bar(
                        x=pm_df["importance_mean"],
                        y=pm_df["feature"],
                        orientation="h",
                        error_x=dict(type="data", array=pm_df["importance_std"], visible=True),
                        marker_color="#22c55e",
                    )
                )
                fig_pm.update_layout(
                    title=f"Permutation Importance (Mean ± SD drop in {pm_df['scoring'].iloc[0]})",
                    xaxis_title="Importance Score",
                )
                _style_fig(fig_pm, height=340)
                st.plotly_chart(fig_pm, width="stretch")
            else:
                st.info(global_res.get("perm_error") or "Permutation importance was unavailable for this dataset split.")

        g3, g4 = st.columns(2)
        with g3:
            int_title = (
                "Tree-Based Split Importance"
                if global_res.get("intrinsic_method") == "tree_feature_importance"
                else "Intrinsic Model Importance"
            )
            st.markdown(f"#### 3. {int_title}")
            if int_list:
                in_df = pd.DataFrame(int_list).sort_values("importance", ascending=True)
                fig_in = px.bar(
                    in_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="feature_category" if "feature_category" in in_df.columns else None,
                    labels={"importance": "Intrinsic Weight", "feature": "Feature", "feature_category": "Domain"},
                    title=f"Built-In Weights ({global_res.get('intrinsic_method')})",
                )
                _style_fig(fig_in, height=340)
                st.plotly_chart(fig_in, width="stretch")
            else:
                st.info("Intrinsic built-in feature weights are unavailable for this estimator type.")

        with g4:
            st.markdown("#### 4. Feature Contribution Comparison Across Methods")
            if comp_rows:
                cmp_df = pd.DataFrame(comp_rows)
                melt_cols = [
                    c for c in ("shap_share_pct", "permutation_share_pct", "intrinsic_share_pct")
                    if c in cmp_df.columns and cmp_df[c].notna().any()
                ]
                if melt_cols:
                    long_cmp = cmp_df.melt(
                        id_vars=["feature"],
                        value_vars=melt_cols,
                        var_name="Method",
                        value_name="Relative Share (%)",
                    )
                    long_cmp["Method"] = long_cmp["Method"].map({
                        "shap_share_pct": "SHAP (|SHAP| %)",
                        "permutation_share_pct": "Permutation (%)",
                        "intrinsic_share_pct": "Tree / Intrinsic (%)",
                    })
                    fig_mc = px.bar(
                        long_cmp,
                        x="Relative Share (%)",
                        y="feature",
                        color="Method",
                        barmode="group",
                        orientation="h",
                        title="Normalized Method Agreement (% Share Across Trained Features)",
                    )
                    _style_fig(fig_mc, height=340)
                    st.plotly_chart(fig_mc, width="stretch")
                else:
                    st.info("Insufficient importance scores for cross-method comparison.")
            else:
                st.info("Cross-method comparison unavailable.")

        # ── Beeswarm & Distribution Row ───────────────────────────────
        beeswarm = global_res.get("beeswarm_records", [])
        if global_res.get("shap_available") and beeswarm:
            bw_df = pd.DataFrame(beeswarm)
            feat_order = [r["feature"] for r in reversed(global_res["shap_summary"])]

            b1, b2 = st.columns(2)
            with b1:
                st.markdown("#### 5. SHAP Beeswarm Plot (Sample Impact & Feature Value)")
                rng_jit = np.random.default_rng(42)
                y_pos_map = {feat: idx for idx, feat in enumerate(feat_order)}
                bw_df["y_jitter"] = (
                    bw_df["feature"].map(y_pos_map)
                    + rng_jit.uniform(-0.24, 0.24, size=len(bw_df))
                )
                fig_bee = px.scatter(
                    bw_df,
                    x="shap_value",
                    y="y_jitter",
                    color="feature_value_norm",
                    color_continuous_scale=["#3b82f6", "#e2e8f0", "#ef4444"],
                    hover_data={"feature": True, "feature_value": True, "shap_value": ":.4f", "y_jitter": False, "feature_value_norm": False},
                    labels={
                        "shap_value": f"SHAP Value ({global_res.get('shap_unit', 'impact')})",
                        "feature_value_norm": "Normalized Value",
                    },
                    title="Red = High Feature Value · Blue = Low Feature Value",
                )
                fig_bee.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
                fig_bee.update_yaxes(
                    tickvals=list(range(len(feat_order))),
                    ticktext=feat_order,
                    title_text="",
                )
                _style_fig(fig_bee, height=380)
                st.plotly_chart(fig_bee, width="stretch")

            with b2:
                st.markdown("#### 6. Feature Impact Distribution (SHAP Spread)")
                fig_dist = px.box(
                    bw_df,
                    x="shap_value",
                    y="feature",
                    category_orders={"feature": list(reversed(feat_order))},
                    points="outliers",
                    color="feature",
                    title="Distribution of SHAP Contributions Across Samples",
                    labels={"shap_value": f"SHAP Value ({global_res.get('shap_unit', 'impact')})", "feature": ""},
                )
                fig_dist.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
                fig_dist.update_layout(showlegend=False)
                _style_fig(fig_dist, height=380)
                st.plotly_chart(fig_dist, width="stretch")

            # Ranking comparison table
            with st.expander("📋 Combined Feature Importance Ranking Table", expanded=False):
                merged = pd.DataFrame(global_res["shap_summary"]).rename(
                    columns={"rank": "shap_rank", "mean_abs_shap": "mean_abs_shap"}
                )
                if perm_list:
                    p_sub = pd.DataFrame(perm_list)[["feature", "importance_mean", "importance_std", "rank"]].rename(
                        columns={"rank": "perm_rank", "importance_mean": "perm_importance_mean"}
                    )
                    merged = merged.merge(p_sub, on="feature", how="left")
                if int_list:
                    i_sub = pd.DataFrame(int_list)[["feature", "importance"]].rename(
                        columns={"importance": "intrinsic_importance"}
                    )
                    merged = merged.merge(i_sub, on="feature", how="left")
                st.dataframe(merged, width="stretch", hide_index=True)

        # ── Sample & Stored Prediction Waterfall Inspector ────────────
        st.divider()
        st.markdown("#### 🔍 Individual Prediction Waterfall & Contribution Inspector")
        from src.database.models import get_recent_predictions_with_location
        import json as _json

        df_eval_full = load_dataset()
        saved_preds_df = get_recent_predictions_with_location(limit=50)
        source_options = ["Evaluation Dataset Record"]
        if not saved_preds_df.empty and "input_features" in saved_preds_df.columns:
            source_options.append("Stored Prediction Record (SQLite)")

        insp_mode = st.radio(
            "Select Prediction Source to Explain",
            source_options,
            horizontal=True,
            key="xai_inspector_source",
        )

        selected_features: dict | None = None
        if insp_mode == "Stored Prediction Record (SQLite)":
            pred_labels: dict[str, dict] = {}
            for _, prow in saved_preds_df.iterrows():
                raw_in = prow.get("input_features")
                parsed_in = {}
                if isinstance(raw_in, str) and raw_in.strip():
                    try:
                        parsed_in = _json.loads(raw_in)
                    except Exception:
                        parsed_in = {}
                elif isinstance(raw_in, dict):
                    parsed_in = raw_in
                p_str = f"{float(prow['probability']) * 100:.1f}%" if pd.notna(prow.get("probability")) else "N/A"
                loc_tag = prow.get("location_name", "Ad-hoc")
                lbl = f"#{int(prow['id'])} ({loc_tag}) — {prow.get('risk_category', 'Unknown')} ({p_str}) @ {str(prow.get('predicted_at', ''))[:19]}"
                pred_labels[lbl] = parsed_in
            chosen_pred_lbl = st.selectbox("Select Saved Assessment", list(pred_labels.keys()), key="xai_saved_pred_select")
            selected_features = pred_labels.get(chosen_pred_lbl, {})
        elif df_eval_full is not None and not df_eval_full.empty:
            max_idx = min(len(df_eval_full) - 1, 200)
            chosen_row = st.slider("Inspect Evaluation Record Index", 0, max_idx, 0, key="xai_row_idx")
            selected_features = df_eval_full.iloc[chosen_row].to_dict()

        if selected_features is not None:
            pt_exp = explain_prediction(
                features=selected_features,
                model_name=selected_model_name,
                background_df=df_eval_full.head(100) if df_eval_full is not None else None,
                use_cache=True,
            )
            if pt_exp.get("available"):
                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("Predicted Risk Category", pt_exp["risk_category"])
                ic2.metric("Model Score (Probability)", f"{pt_exp['probability']*100:.1f}%")
                ic3.metric("SHAP Cache Status", "Hit (Cached)" if pt_exp["from_cache"] else "Freshly Computed")
                _render_individual_shap_panel(
                    pt_exp,
                    compact=False,
                    show_feature_table=True,
                    show_concepts=False,
                )
            else:
                for err_msg in pt_exp.get("errors") or ["Explanation unavailable for selected prediction."]:
                    st.error(err_msg)

    # ── Conceptual Distinction Panel (Section 4) ──────────────────────
    st.divider()
    st.markdown("### 📚 Understanding Explainable AI Concepts & Causation Limits")
    c_cols = st.columns(2)
    for idx, item in enumerate(EXPLAINABILITY_CONCEPTS):
        with c_cols[idx % 2]:
            st.markdown(f"""
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px 14px; margin-bottom:10px;">
                <div style="font-weight:700; color:#38bdf8; margin-bottom:4px;">{item['concept']}</div>
                <div style="font-size:0.9rem; color:#cbd5e1;">{item['definition']}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 9: ALERTS
# ═══════════════════════════════════════════════════════════════

elif page == "🚨 Alert Management":
    from src.alerts.alert_manager import get_alert_manager
    from src.database.models import get_locations
    manager = get_alert_manager()

    render_page_header(
        "🚨 Experimental Alert Management System",
        "Evaluate dynamic risk conditions against experimental thresholds, suppress duplicates with escalation override, manage alert lifecycles, and inspect audit logs.",
        pills=[("Experimental Only", "pill-amber"), ("Email/SMS Disabled", "pill-blue")],
    )
    experimental_banner(
        "These alerts are EXPERIMENTAL research outputs. Thresholds are provisional and "
        "not calibrated as official warnings. Must NOT trigger evacuation or emergency response."
    )

    ALERT_SEVERITY_COLORS = {
        "Low": "#22c55e",
        "Moderate": "#eab308",
        "High": "#f97316",
        "Very High": "#ef4444",
        "Unknown": "#94a3b8",
    }

    trigger = get_setting("alerts", "trigger_category") or "High"
    cooldown = int(get_setting("alerts", "cooldown_minutes") or 120)

    # ── Filters bar ───────────────────────────────────────────────────
    f1, f2, f3 = st.columns([1.2, 1.0, 1.0])
    with f1:
        filter_cat = st.selectbox(
            "Filter by Severity Category",
            ["All Categories", "Low", "Moderate", "High", "Very High"],
            key="alert_filter_cat",
        )
    with f2:
        use_date_filter = st.checkbox("Filter by Creation Date Range", value=False, key="alert_use_dates")
    with f3:
        chan_statuses = manager.get_notification_status()
        chan_label = ", ".join(
            f"{c['channel'].upper()}: {'ON' if c['enabled'] else 'OFF'}"
            for c in chan_statuses
        )
        st.caption(f"External Notification Channels: **{chan_label}** (disabled by default)")

    start_date_str = None
    end_date_str = None
    if use_date_filter:
        d1, d2 = st.columns(2)
        with d1:
            s_dt = st.date_input("Start Date", value=_dt.date.today() - _dt.timedelta(days=30), key="alert_start_dt")
            start_date_str = s_dt.isoformat()
        with d2:
            e_dt = st.date_input("End Date", value=_dt.date.today(), key="alert_end_dt")
            end_date_str = e_dt.isoformat()

    cat_arg = None if filter_cat == "All Categories" else filter_cat

    active_df = manager.get_active_alerts(risk_category=cat_arg, start_date=start_date_str, end_date=end_date_str)
    ack_df = manager.get_acknowledged_alerts(risk_category=cat_arg, start_date=start_date_str, end_date=end_date_str)
    res_df = manager.get_resolved_alerts(risk_category=cat_arg, start_date=start_date_str, end_date=end_date_str)
    high_df = manager.get_high_priority_alerts(limit=200, include_resolved=False)
    all_df = manager.get_all_alerts(limit=200, risk_category=cat_arg, start_date=start_date_str, end_date=end_date_str)

    if cat_arg and isinstance(high_df, pd.DataFrame) and not high_df.empty:
        high_df = high_df[high_df["risk_category"] == cat_arg]

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Active (New)", len(active_df) if isinstance(active_df, pd.DataFrame) else 0)
    k2.metric("High Priority", len(high_df) if isinstance(high_df, pd.DataFrame) else 0)
    k3.metric("Acknowledged", len(ack_df) if isinstance(ack_df, pd.DataFrame) else 0)
    k4.metric("Resolved", len(res_df) if isinstance(res_df, pd.DataFrame) else 0)
    k5.metric("Trigger Tier", trigger)
    k6.metric("Cooldown Window", f"{cooldown} min")

    st.divider()
    with st.expander("🔔 Evaluate Conditions & Generate Experimental Alert", expanded=False):
        locs_df = get_locations(active_only=True)
        loc_options: dict[str, dict | None] = {"Custom Location (Unlinked)": None}
        if isinstance(locs_df, pd.DataFrame) and not locs_df.empty:
            for _, row in locs_df.iterrows():
                loc_options[f"{row['name']} (ID {int(row['id'])})"] = dict(row)

        with st.form("alert_form"):
            sel_col, trig_col = st.columns([1.6, 1.0])
            with sel_col:
                chosen_loc_lbl = st.selectbox("Link Monitored Station (for Cooldown/Coords)", list(loc_options.keys()), key="alert_station_sel")
            with trig_col:
                override_trigger = st.selectbox(
                    "Experimental Trigger Tier",
                    ["High", "Very High", "Moderate", "Low"],
                    index=["High", "Very High", "Moderate", "Low"].index(trigger) if trigger in ["High", "Very High", "Moderate", "Low"] else 0,
                    key="alert_trig_override",
                )

            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                a_slope = st.slider("Slope (°)", 0.0, 90.0, 40.0, key="a_slope")
                a_rf24 = st.number_input("24h Rainfall (mm)", 0.0, 2000.0, 150.0, key="a_rf24")
                a_lat = st.number_input("Latitude", -90.0, 90.0, 10.0889, format="%.4f", key="a_lat")
            with ac2:
                a_rf72 = st.number_input("72h Rainfall (mm)", 0.0, 5000.0, 300.0, key="a_rf72")
                a_elev = st.number_input("Elevation (m)", 0.0, 9000.0, 1200.0, key="a_elev")
                a_lon = st.number_input("Longitude", -180.0, 180.0, 77.0595, format="%.4f", key="a_lon")
            with ac3:
                a_moist = st.slider("Soil Moisture", 0.0, 1.0, 0.75, key="a_moist")
                a_veg = st.slider("Vegetation Index", -1.0, 1.0, 0.15, key="a_veg")
                a_loc = st.text_input("Custom Location Label", "Test Location", key="a_loc")
            generate = st.form_submit_button("⚡ Evaluate & Generate Experimental Alert", type="primary", width="stretch")

        if generate:
            linked_rec = loc_options.get(chosen_loc_lbl)
            use_loc_id = int(linked_rec["id"]) if linked_rec else None
            use_loc_name = str(linked_rec["name"]) if linked_rec else (a_loc or "Custom Location")
            use_lat = float(linked_rec["latitude"]) if linked_rec else float(a_lat)
            use_lon = float(linked_rec["longitude"]) if linked_rec else float(a_lon)

            features = {
                "slope_degrees": a_slope,
                "rainfall_24h_mm": a_rf24,
                "rainfall_72h_mm": a_rf72,
                "elevation_m": a_elev,
                "soil_moisture": a_moist,
                "vegetation_index": a_veg,
                "latitude": use_lat,
                "longitude": use_lon,
            }
            with st.spinner("Evaluating risk, cooldown policy, and escalation rules…"):
                result = manager.check_location(
                    features=features,
                    location_id=use_loc_id,
                    location_name=use_loc_name,
                    latitude=use_lat,
                    longitude=use_lon,
                    data_source="Manual Alert Evaluator Form",
                    trigger_category=override_trigger,
                )

            risk_r = result.get("risk_result", {})
            prob = risk_r.get("probability")
            cat = risk_r.get("risk_category", "Unknown")
            alert = result.get("alert")
            c_col = ALERT_SEVERITY_COLORS.get(cat, "#94a3b8")

            ag1, ag2, ag3 = st.columns(3)
            with ag1:
                st.markdown(
                    f'<div style="background:{c_col}22; border:2px solid {c_col}; border-radius:10px; '
                    f'padding:1rem; text-align:center;"><b style="color:{c_col}; font-size:1.3rem;">{cat}</b>'
                    f'<br><small style="color:#94a3b8;">Risk Category</small></div>',
                    unsafe_allow_html=True,
                )
            with ag2:
                st.metric("Probability", f"{prob:.4f}" if prob is not None else "N/A")
            with ag3:
                if alert:
                    esc_txt = f" (Escalated from {alert['escalated_from']})" if alert.get("is_escalation") else ""
                    st.success(f"✅ Experimental Alert #{alert.get('id')} Created{esc_txt}")
                else:
                    st.info("ℹ️ Suppressed during cooldown or below trigger threshold.")
            if alert:
                st.caption(alert.get("notice", ""))
            active_df = manager.get_active_alerts(risk_category=cat_arg, start_date=start_date_str, end_date=end_date_str)
            high_df = manager.get_high_priority_alerts(limit=200, include_resolved=False)
            all_df = manager.get_all_alerts(limit=200, risk_category=cat_arg, start_date=start_date_str, end_date=end_date_str)

    st.divider()
    tab_active, tab_high, tab_ack, tab_resolved, tab_history = st.tabs([
        "🚨 Active Alerts (New)",
        "🔥 High-Priority",
        "✅ Acknowledged",
        "🏁 Resolved",
        "📜 History & Audit Log",
    ])

    def _render_alert_detail_inspector(df_source: pd.DataFrame, widget_prefix: str) -> None:
        if df_source.empty:
            return
        ids = [int(x) for x in df_source["id"].tolist()]
        sel_id = st.selectbox("Inspect Location & Environmental Details for Alert ID", ids, key=f"{widget_prefix}_inspect_id")
        rec = manager.get_alert(int(sel_id))
        if rec:
            sev_col = ALERT_SEVERITY_COLORS.get(str(rec.get("risk_category")), "#94a3b8")
            st.markdown(
                f"<div style='border-left:4px solid {sev_col}; padding:8px 12px; background:#0f172a; border-radius:6px;'>"
                f"<b>Alert #{rec['id']}</b> · Status: <code>{rec.get('status', 'new')}</code> · "
                f"Category: <b style='color:{sev_col};'>{rec['risk_category']}</b> · "
                f"Location: <b>{rec.get('location_name') or 'Unlinked'}</b> "
                f"(<code>{rec.get('latitude')}, {rec.get('longitude')}</code>)<br>"
                f"<small>Model: <code>{rec.get('model_version') or 'N/A'}</code> · "
                f"Source: <code>{rec.get('data_source') or 'N/A'}</code> · "
                f"Created: <code>{rec.get('created_at')}</code> · Experimental: <code>{bool(rec.get('is_experimental', 1))}</code></small>"
                f"</div>",
                unsafe_allow_html=True,
            )
            env_raw = rec.get("supporting_env_info")
            if env_raw:
                with st.expander("Supporting Environmental Measurements", expanded=False):
                    try:
                        st.json(json.loads(env_raw) if isinstance(env_raw, str) else env_raw)
                    except Exception:
                        st.text(str(env_raw))

    with tab_active:
        if isinstance(active_df, pd.DataFrame) and not active_df.empty:
            st.dataframe(active_df, width="stretch", hide_index=True)
            _render_alert_detail_inspector(active_df, "active")
            st.markdown("#### Lifecycle Actions on Active Alert")
            ak1, ak2, ak3, ak4 = st.columns([1.0, 2.0, 0.9, 0.9])
            active_ids = [int(x) for x in active_df["id"].tolist()]
            with ak1:
                ack_id = st.selectbox("Alert ID", active_ids, key="ack_id")
            with ak2:
                ack_note = st.text_input("Operator Note (optional)", key="ack_note", placeholder="e.g. Reviewed during monitoring check")
            with ak3:
                st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
                if st.button("✅ Acknowledge", key="ack_btn", width="stretch"):
                    if manager.acknowledge(int(ack_id), ack_note or None):
                        st.success(f"Alert {ack_id} acknowledged.")
                        st.rerun()
            with ak4:
                st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
                if st.button("🏁 Resolve", key="res_from_active_btn", width="stretch"):
                    if manager.resolve(int(ack_id), ack_note or None):
                        st.success(f"Alert {ack_id} resolved.")
                        st.rerun()
        else:
            st.info("No active (new) unacknowledged alerts matching current filters.")

    with tab_high:
        if isinstance(high_df, pd.DataFrame) and not high_df.empty:
            st.dataframe(high_df, width="stretch", hide_index=True)
            _render_alert_detail_inspector(high_df, "high")
        else:
            st.info("No open High or Very High experimental alerts.")

    with tab_ack:
        if isinstance(ack_df, pd.DataFrame) and not ack_df.empty:
            st.dataframe(ack_df, width="stretch", hide_index=True)
            ak_r1, ak_r2, ak_r3 = st.columns([1.0, 2.0, 1.0])
            ack_ids = [int(x) for x in ack_df["id"].tolist()]
            with ak_r1:
                res_id = st.selectbox("Acknowledged Alert ID", ack_ids, key="res_ack_id")
            with ak_r2:
                res_note = st.text_input("Resolution Note (optional)", key="res_note", placeholder="e.g. Rainfall subsided below threshold")
            with ak_r3:
                st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
                if st.button("🏁 Mark Resolved", key="res_ack_btn", width="stretch"):
                    if manager.resolve(int(res_id), res_note or None):
                        st.success(f"Alert {res_id} marked resolved.")
                        st.rerun()
        else:
            st.info("No acknowledged alerts waiting for resolution.")

    with tab_resolved:
        if isinstance(res_df, pd.DataFrame) and not res_df.empty:
            st.dataframe(res_df, width="stretch", hide_index=True)
        else:
            st.info("No resolved alerts recorded yet.")

    with tab_history:
        if isinstance(all_df, pd.DataFrame) and not all_df.empty:
            st.markdown("##### Complete Alert History (Experimental)")
            st.dataframe(all_df, width="stretch", hide_index=True)
        else:
            st.info("No alert history matching current filters.")

        st.markdown("##### Audit Log (Acknowledgements & Resolutions)")
        audit_df = manager.get_audit_log(limit=100)
        if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
            st.dataframe(audit_df, width="stretch", hide_index=True)
        else:
            st.info("No acknowledgement or resolution audit events logged yet.")

    with st.expander("📡 External Notification Integration Hooks (Disabled by Default)", expanded=False):
        st.caption(
            "Future Email / SMS notification providers stay strictly disabled unless explicitly enabled in "
            "`config/settings.yaml` and configured with environment credentials."
        )
        st.dataframe(pd.DataFrame(manager.get_notification_status()), width="stretch", hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: INDIA REGIONAL MONITORING (STATE / DISTRICT REAL-TIME DATA)
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🇮🇳 India Regional Monitoring":
    from src.data_providers.india_boundaries import IndiaBoundaryProvider
    from src.geospatial.maps import build_india_regional_map
    from src.monitoring.india_monitor import IndiaRegionalMonitor

    render_page_header(
        "🇮🇳 India State & District Environmental Monitoring",
        "All-India State/UT -> District selection backed by geoBoundaries ADM1/ADM2 polygons, "
        "Open-Meteo coordinate weather, IMD API integration, NDMA SACHET official CAP warnings, "
        "terrain analysis, and experimental RiskEngine + SHAP estimates.",
        pills=[("Country: India", "pill-blue"), ("36 States/UTs • 735 Districts", "pill-green")],
    )
    experimental_banner(
        "Official alerts come strictly from <b>NDMA SACHET</b> and <b>IMD</b> (when configured). "
        "Point-coordinate weather estimates and LandslideGuard ML scores are <b>NOT</b> area-wide official warnings."
    )

    india_cfg = get_setting("india_monitoring") or {}
    default_state = str(india_cfg.get("default_state", "Kerala"))
    default_district = str(india_cfg.get("default_district", "Idukki"))

    boundary_provider = IndiaBoundaryProvider(boundaries_dir=india_cfg.get("boundaries_dir"))
    b_status = boundary_provider.get_boundary_status()

    if b_status.get("state_count", 0) == 0:
        st.warning(
            "India State/District boundary polygons are not cached yet. Click below to download "
            "verified geoBoundaries ADM1 (States/UTs) and ADM2 (Districts) boundaries."
        )
        if st.button("⬇️ Download & Cache India Boundaries", width="content"):
            with st.spinner("Downloading India ADM1/ADM2 boundaries from geoBoundaries..."):
                boundary_provider.download_and_cache()
            st.success("India State & District boundaries cached successfully!")
            st.rerun()

    # ── SECTION A: Geographical Selection (Country -> State/UT -> District -> Location) ──
    st.markdown("### 🗺️ A. India Geographical Selection")
    states_list = boundary_provider.list_states()
    state_names = [s["state_name"] for s in states_list] if states_list else [default_state]
    default_state_idx = state_names.index(default_state) if default_state in state_names else 0

    sel_c1, sel_c2, sel_c3, sel_c4 = st.columns([0.7, 1.3, 1.4, 1.0])
    with sel_c1:
        st.text_input("Country", value="India", disabled=True, key="ind_country")
    with sel_c2:
        selected_state = st.selectbox(
            "State / Union Territory",
            options=state_names,
            index=default_state_idx,
            key="ind_state_select",
            help="Type directly inside the dropdown to search all 36 Indian States & UTs.",
        )
    with sel_c3:
        dist_filter = st.text_input(
            "Filter Districts (optional)",
            value="",
            placeholder="Search district name...",
            key="ind_district_filter",
        )
        districts_list = boundary_provider.list_districts(
            state_name=selected_state,
            query=dist_filter.strip() or None,
        )
        if not districts_list and dist_filter.strip():
            districts_list = boundary_provider.list_districts(state_name=selected_state)
        district_names = [d["district_name"] for d in districts_list] if districts_list else [default_district]
        if selected_state == default_state and default_district in district_names:
            def_dist_idx = district_names.index(default_district)
        else:
            def_dist_idx = 0
        selected_district = st.selectbox(
            "District",
            options=district_names,
            index=def_dist_idx,
            key="ind_district_select",
            help="Dependent on selected State/UT; type to search.",
        )
    with sel_c4:
        st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
        run_assess = st.button("🔄 Refresh Regional Data", type="primary", width="stretch", key="ind_run_btn")

    op1, op2, op3 = st.columns(3)
    with op1:
        include_terrain = st.checkbox("Fetch SRTM Terrain (Elevation & Slope)", value=True, key="ind_terrain_chk")
    with op2:
        include_shap = st.checkbox("Compute SHAP Model Explanation", value=True, key="ind_shap_chk")
    with op3:
        save_to_db = st.checkbox("Persist Assessment & Warnings to SQLite", value=True, key="ind_persist_chk")

    region_cache_key = f"{selected_state}::{selected_district}"
    if (
        run_assess
        or "india_monitor_result" not in st.session_state
        or st.session_state.get("india_monitor_key") != region_cache_key
    ):
        with st.spinner(f"Fetching live environmental & official warning data for {selected_district}, {selected_state}..."):
            monitor = IndiaRegionalMonitor(boundary_provider=boundary_provider)
            try:
                res = monitor.run_regional_assessment(
                    state_name=selected_state,
                    district_name=selected_district,
                    include_terrain=include_terrain,
                    include_shap=include_shap,
                    persist=save_to_db,
                    generate_experimental_alert=save_to_db,
                )
                st.session_state["india_monitor_result"] = res
                st.session_state["india_monitor_key"] = region_cache_key
            except Exception as exc:
                st.error(f"Could not complete regional monitoring for {selected_district}, {selected_state}: {exc}")
                res = None
    else:
        res = st.session_state.get("india_monitor_result")

    if res:
        d_meta = res.get("district_meta") or {}
        wx = res.get("weather") or {}
        imd = res.get("imd") or {}
        sachet = res.get("sachet") or {}
        terrain = res.get("terrain") or {}
        hist = res.get("historical") or {}
        risk_res = res.get("risk_result") or {}
        shap_exp = res.get("shap_explanation") or {}

        # Coordinate strip
        coord_c1, coord_c2, coord_c3, coord_c4 = st.columns(4)
        with coord_c1:
            st.metric("State / UT", res["state"])
        with coord_c2:
            st.metric("District", res["district"])
        with coord_c3:
            st.metric("Representative Latitude", f"{d_meta.get('rep_lat', 0):.4f}°N")
        with coord_c4:
            st.metric("Representative Longitude", f"{d_meta.get('rep_lon', 0):.4f}°E")

        st.caption(
            f"📍 Administrative Boundaries: {b_status.get('state_count', 0)} States/UTs & "
            f"{b_status.get('district_count', 0)} Districts ({b_status.get('source', 'geoBoundaries')}). "
            "Representative coordinates sit inside the selected district polygon."
        )

        # ── SOURCE-STATUS PANEL ──
        st.markdown("#### 🛰️ External Data Provider Health & Freshness")
        status_cols = st.columns(4)
        for idx_p, prov in enumerate(res.get("provider_status", [])[:4]):
            with status_cols[idx_p]:
                st_str = str(prov.get("status", "Unknown"))
                badge = "🟢" if st_str == "Connected" else ("🗂️" if st_str in ("Loaded", "Cached") else ("⚙️" if st_str == "Not configured" else "🟡"))
                cache_note = " (Cached)" if prov.get("is_cached") and not prov.get("is_live") else (" (Live)" if prov.get("is_live") else "")
                st.markdown(f"**{badge} {prov.get('provider')}**")
                st.caption(f"Status: `{st_str}{cache_note}`")
                if prov.get("last_updated"):
                    st.caption(f"Updated: `{str(prov['last_updated'])[:19]}`")

        st.divider()

        # ── MAP + SECTION B WEATHER ──
        map_col, wx_col = st.columns([1.25, 1.0])
        with map_col:
            st.markdown("### 🗺️ Interactive India State & District Map")
            st.caption(
                "Polygon boundaries outline the State (blue dashed) and District (amber). "
                "Polygons are NOT uniformly shaded by risk because weather & experimental ML scores apply only to point coordinates."
            )
            hist_df_map = hist.get("observations_df") if isinstance(hist.get("observations_df"), pd.DataFrame) else pd.DataFrame()
            ind_map = build_india_regional_map(
                center_lat=float(d_meta["rep_lat"]),
                center_lon=float(d_meta["rep_lon"]),
                state_name=res["state"],
                district_name=res["district"],
                state_geojson=res.get("state_geojson"),
                district_geojson=res.get("district_geojson"),
                zoom_bbox=d_meta.get("bbox"),
                historical_df=hist_df_map,
                weather_summary=wx,
                risk_result=risk_res,
                official_alerts=sachet.get("regional_alerts", []),
            )
            try:
                import streamlit_folium as stf
                stf.st_folium(ind_map, width=None, height=470, returned_objects=[], key="india_region_map")
            except ImportError:
                st.info("Install `streamlit-folium` to display interactive Folium boundary maps.")

        with wx_col:
            st.markdown("### 🌦️ B. Coordinate Weather (Open-Meteo)")
            st.info(wx.get("scope_note", "Coordinate-level model estimate."), icon="ℹ️")
            fresh = wx.get("freshness") or {}
            st.caption(
                f"Source: {wx.get('attribution')} | Freshness: `{fresh.get('label', 'N/A')}` | "
                f"Mode: `{'Cached' if wx.get('is_cached') else 'Live'}`"
            )

            w1, w2, w3 = st.columns(3)
            with w1:
                st.metric("Current Rain (1h)", f"{wx['current_precip_mm']:.1f} mm" if wx.get("current_precip_mm") is not None else "Unavailable")
                st.metric("24h Rainfall", f"{wx['rainfall_24h_mm']:.1f} mm" if wx.get("rainfall_24h_mm") is not None else "Unavailable")
            with w2:
                st.metric("Temperature", f"{wx['temperature_c']:.1f} °C" if wx.get("temperature_c") is not None else "Unavailable")
                st.metric("48h Rainfall", f"{wx['rainfall_48h_mm']:.1f} mm" if wx.get("rainfall_48h_mm") is not None else "Unavailable")
            with w3:
                st.metric("Humidity / Wind", f"{wx.get('humidity_pct', 'N/A')}% | {wx.get('wind_speed_kmh', 'N/A')} km/h")
                st.metric("72h Rainfall", f"{wx['rainfall_72h_mm']:.1f} mm" if wx.get("rainfall_72h_mm") is not None else "Unavailable")

            # Past vs Forecast hourly rainfall chart (strictly separated)
            hh = wx.get("historical_hourly") or {}
            fh = wx.get("forecast_hourly") or {}
            rows_wx: list[dict[str, Any]] = []
            for t, p in zip(hh.get("time", []), hh.get("precipitation", [])):
                rows_wx.append({"time": t, "precip_mm": p, "Type": "Past Reanalysis (Non-Forecast)"})
            for t, p in zip(fh.get("time", []), fh.get("precipitation", [])):
                rows_wx.append({"time": t, "precip_mm": p, "Type": "Forecast (Predicted)"})
            if rows_wx:
                wx_chart_df = pd.DataFrame(rows_wx)
                fig_wx = px.bar(
                    wx_chart_df,
                    x="time",
                    y="precip_mm",
                    color="Type",
                    color_discrete_map={
                        "Past Reanalysis (Non-Forecast)": "#38bdf8",
                        "Forecast (Predicted)": "#f59e0b",
                    },
                    labels={"time": "Timestamp", "precip_mm": "Hourly Precip (mm)"},
                )
                _style_fig(fig_wx, height=235)
                st.plotly_chart(fig_wx, width="stretch")

        st.divider()

        # ── SECTION C: OFFICIAL IMD DATA ──
        st.markdown("### 🏛️ C. Official IMD Meteorological Data (`api.imd.gov.in`)")
        imd_status = imd.get("status", "Not configured")
        if imd_status == "Not configured":
            st.warning(
                "⚙️ **IMD API Status: Not configured** — `IMD_API_KEY` is not set in `.env`. "
                "Add your official India Meteorological Department API key (`x-api-key`) to `.env` to enable live IMD state/district rainfall, warnings, and nowcasts.",
            )
        elif imd_status == "Unavailable":
            st.error(f"IMD API currently unavailable: {imd.get('message', 'Connection/Auth error')}")
        else:
            st.success(f"IMD Status: **{imd_status}** (`{imd.get('attribution')}`)")
            i1, i2, i3, i4 = st.columns(4)
            sr = imd.get("state_rainfall") or {}
            dr = imd.get("district_rainfall") or {}
            dw = imd.get("district_warning") or {}
            dn = imd.get("district_nowcast") or {}
            with i1:
                st.metric("IMD State Rain (Daily Actual)", f"{sr.get('daily_actual_mm', 'N/A')} mm")
            with i2:
                st.metric("IMD District Rain (Daily Actual)", f"{dr.get('daily_actual_mm', 'N/A')} mm", delta=dr.get("daily_category"))
            with i3:
                st.metric("IMD District Warning", dw.get("warning_category", "Unavailable"))
                if dw.get("warning_description"):
                    st.caption(dw["warning_description"])
            with i4:
                st.metric("IMD Nowcast Window", dn.get("valid_upto") or "Unavailable", delta=dn.get("nowcast_color"))
                if dn.get("nowcast_summary"):
                    st.caption(dn["nowcast_summary"])

            fcst_5d = (imd.get("rainfall_forecast") or {}).get("days") or []
            if fcst_5d:
                st.markdown("**IMD 5-Day District Rainfall Forecast**")
                st.dataframe(pd.DataFrame(fcst_5d), width="stretch", hide_index=True)

        st.divider()

        # ── SECTION D: REGIONAL HISTORICAL LANDSLIDES, TERRAIN, RISK ENGINE & SHAP ──
        st.markdown("### 🧠 D. Regional Landslides, Terrain & Experimental ML Assessment")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Nearby Landslide Records", f"{hist.get('total_count', 0):,}")
            st.caption(f"Authentic: {hist.get('authentic_count', 0)} | Synthetic: {hist.get('synthetic_count', 0)}")
        with d2:
            elev = terrain.get("elevation_m")
            st.metric("Elevation (SRTM)", f"{elev:.0f} m" if elev is not None else "Unavailable")
        with d3:
            slp = terrain.get("slope_degrees")
            st.metric("Estimated Slope", f"{slp:.1f}°" if slp is not None else "Unavailable")
        with d4:
            prob = risk_res.get("probability")
            st.metric(
                "Experimental ML Score",
                f"{prob:.3f}" if prob is not None else "Unavailable",
                delta=risk_res.get("risk_category", "Unknown"),
            )
            st.caption(f"Method: `{risk_res.get('method', 'none')}` | ⚠️ Research Estimate")

        for w in risk_res.get("warnings", []):
            st.caption(f"⚠️ {w}")

        sh1, sh2 = st.columns([1.1, 1.0])
        with sh1:
            st.markdown("#### 🔍 SHAP Contribution Breakdown")
            if shap_exp and shap_exp.get("shap_available") and shap_exp.get("contributions"):
                contrib_df = pd.DataFrame(shap_exp["contributions"])
                fig_sh = px.bar(
                    contrib_df.sort_values("shap_value"),
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color=contrib_df.sort_values("shap_value")["shap_value"].apply(lambda v: "Increases Risk" if v > 0 else "Decreases Risk"),
                    color_discrete_map={"Increases Risk": "#e74c3c", "Decreases Risk": "#2ecc71"},
                    labels={"shap_value": "SHAP Contribution", "feature": "Feature"},
                )
                _style_fig(fig_sh, height=280)
                st.plotly_chart(fig_sh, width="stretch")
                st.caption(shap_exp.get("human_readable_summary", ""))
            else:
                st.info((shap_exp or {}).get("human_readable_summary", "SHAP explanation unavailable for current input."))

        with sh2:
            st.markdown("#### 📜 Nearby Historical Landslide Records")
            hist_df_show = hist.get("observations_df")
            if isinstance(hist_df_show, pd.DataFrame) and not hist_df_show.empty:
                show_cols = [c for c in ["event_date", "state_region", "landslide_type", "trigger", "distance_km", "is_synthetic"] if c in hist_df_show.columns]
                st.dataframe(hist_df_show[show_cols].head(12), width="stretch", hide_index=True)
            else:
                st.info(f"No historical landslide dataset records matched {res['state']} or within 75 km of {res['district']}.")

        st.divider()

        # ── SECTION E: SEPARATED OFFICIAL WARNINGS vs EXPERIMENTAL ML ALERTS ──
        st.markdown("### 🚨 E. Official External Alerts vs. Experimental ML Alerts")
        al_official, al_exp = st.columns(2)

        with al_official:
            st.markdown("#### 🏛️ Official Alerts (NDMA SACHET CAP / IMD)")
            st.caption(
                f"Portal: [{sachet.get('portal_url', 'https://sachet.ndma.gov.in/')}]({sachet.get('portal_url', 'https://sachet.ndma.gov.in/')}) | "
                f"Status: `{sachet.get('status', 'Unknown')}` | Total All-India Feed Alerts: {sachet.get('total_india_alerts', 0)}"
            )
            reg_alerts = sachet.get("regional_alerts") or []
            if reg_alerts:
                for al in reg_alerts:
                    with st.container(border=True):
                        title_txt = al.get("alert_title") or al.get("disaster_type") or "Official Alert"
                        msg_txt = al.get("warning_message") or al.get("headline") or ""
                        area_txt = al.get("affected_region") or al.get("area_description") or "N/A"
                        from_txt = al.get("issued_at") or al.get("effective_from") or "N/A"
                        until_txt = al.get("valid_until") or al.get("effective_until") or "N/A"
                        url_txt = al.get("official_url") or al.get("source_url")
                        st.markdown(f"**{title_txt}** — Severity: `{al.get('severity', 'Advisory')}`")
                        st.write(msg_txt)
                        st.caption(f"Area: {area_txt} | Valid: {from_txt} -> {until_txt}")
                        if url_txt:
                            st.markdown(f"[Official CAP XML / Source]({url_txt})")
            else:
                st.success(f"No active NDMA SACHET warnings matched `{res['state']}` / `{res['district']}` right now.")
                all_sachet = sachet.get("all_alerts") or []
                if all_sachet:
                    with st.expander(f"View All Active India SACHET Feed Items ({len(all_sachet)})", expanded=False):
                        st.dataframe(
                            pd.DataFrame([
                                {
                                    "Disaster": a.get("alert_title") or a.get("disaster_type"),
                                    "Severity": a.get("severity"),
                                    "Area": a.get("affected_region") or a.get("area_description"),
                                    "Message": str(a.get("warning_message") or a.get("headline") or "")[:120],
                                }
                                for a in all_sachet
                            ]),
                            width="stretch",
                            hide_index=True,
                        )

        with al_exp:
            st.markdown("#### 🧪 Experimental ML Alerts (LandslideGuard Research Prototype)")
            st.caption("Research prototype output only — NEVER an official evacuation or disaster management alert.")
            exp_al = res.get("experimental_alert")
            if exp_al:
                st.warning(
                    f"Experimental Alert Generated (#{exp_al.get('id')}): **{exp_al.get('risk_category')}** "
                    f"(p={exp_al.get('probability', 0):.3f}) for {res['district']}, {res['state']}."
                )
            else:
                st.info(
                    f"No new experimental ML alert triggered at current score (`{risk_res.get('risk_category', 'Unknown')}`). "
                    "Experimental alerts trigger at High / Very High thresholds subject to cooldown rules."
                )


# ═══════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT & COMPUTER VISION (PHASE 8)
# ═══════════════════════════════════════════════════════════════

elif page == "💬 AI Assistant & Vision":
    render_page_header(
        "💬 LandslideGuard AI Assistant & Computer Vision",
        "Context-aware question answering, experimental slope image analysis, and verified multilingual disaster safety guidance.",
        pills=[
            ("🧠 Retrieval Augmented QA", "pill-blue"),
            ("📷 Terrain CV", "pill-amber"),
            ("🌐 Multilingual (EN/HI/PA)", "pill-green"),
            ("⚠️ Non-Operational", "pill-amber"),
        ],
    )
    experimental_banner(
        "The AI assistant retrieves strictly from actual database and model records. "
        "Computer vision image analysis is experimental and must NEVER independently trigger an official warning."
    )

    from src.assistant.knowledge_engine import get_knowledge_engine
    from src.assistant.safety_guidance import get_safety_guidance, format_safety_markdown, get_supported_languages
    from src.geospatial.terrain_cv import get_cv_analyzer, MAX_IMAGE_BYTES

    tab_chat, tab_cv, tab_safety = st.tabs([
        "💬 LandslideGuard Assistant",
        "📷 Terrain Image Analysis (CV)",
        "🌐 Multilingual Safety Guidance",
    ])

    # ─────────────────────────────────────────────────────────────
    # TAB 1: LandslideGuard AI Assistant
    # ─────────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown("### 🤖 Ask LandslideGuard AI Assistant")
        st.caption(
            "Answers questions using authentic project telemetry, SQLite tables, model explanations, "
            "and official data inventories. Anti-hallucination policy: never fabricates unrecorded data."
        )

        # Context Selectors in an expander
        with st.expander("⚙️ Optional Query Context (District, Location & Language)", expanded=False):
            ctx_c1, ctx_c2, ctx_c3 = st.columns(3)
            with ctx_c1:
                ctx_state = st.text_input("Target State / UT", value="", placeholder="e.g. Kerala, Uttarakhand", key="asst_ctx_state")
            with ctx_c2:
                ctx_district = st.text_input("Target District", value="", placeholder="e.g. Idukki, Wayanad", key="asst_ctx_dist")
            with ctx_c3:
                ctx_lang = st.selectbox("Preferred Language", ["en", "hi", "pa"], format_func=lambda x: {"en": "English", "hi": "हिन्दी (Hindi)", "pa": "ਪੰਜਾਬੀ (Punjabi)"}.get(x, x), key="asst_ctx_lang")

        # Suggested Questions
        st.markdown("##### 💡 Suggested Questions")
        suggested_queries = [
            "Which locations have active experimental alerts?",
            "What rainfall data is available?",
            "Why is a location classified as high risk?",
            "What historical landslide records exist in Kerala?",
            "What data sources does LandslideGuard use?",
            "How does the machine learning risk engine work?",
            "What are the key landslide safety precautions?",
        ]

        sq_cols = st.columns(len(suggested_queries[:4]))
        chosen_sq = None
        for idx, sq in enumerate(suggested_queries[:4]):
            if sq_cols[idx].button(sq, key=f"sq_btn_{idx}", width="stretch"):
                chosen_sq = sq

        sq_cols2 = st.columns(len(suggested_queries[4:]))
        for idx, sq in enumerate(suggested_queries[4:]):
            if sq_cols2[idx].button(sq, key=f"sq_btn_{idx+4}", width="stretch"):
                chosen_sq = sq

        if "assistant_messages" not in st.session_state:
            st.session_state.assistant_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! I am the **LandslideGuard AI Assistant**. I can help you explore monitored stations, "
                        "dynamic rainfall telemetry, model risk evaluations, SHAP explanations, historical observations, "
                        "and disaster preparedness guidance.\n\n"
                        "How can I assist your landslide research today?"
                    ),
                    "sources": ["LandslideGuard System Index"],
                    "intent": "GREETING",
                }
            ]

        # Display conversation history
        for msg in st.session_state.assistant_messages:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        st.caption(f"📚 **Data Sources:** `{', '.join(msg['sources'])}`")
                    if msg.get("intent") and msg["intent"] not in ("GREETING", "EMPTY_QUERY"):
                        st.caption(f"🏷️ **Classified Intent:** `{msg['intent']}`")

        # Query input
        user_prompt = st.chat_input("Ask a question about landslide risk, rainfall, alerts, historical events...")
        query_to_run = chosen_sq or user_prompt

        if query_to_run:
            st.session_state.assistant_messages.append({"role": "user", "content": query_to_run})

            engine = get_knowledge_engine()
            query_ctx = {
                "state": ctx_state.strip() or None,
                "district": ctx_district.strip() or None,
                "language": ctx_lang,
            }
            res = engine.answer_query(query_to_run, context=query_ctx)

            st.session_state.assistant_messages.append({
                "role": "assistant",
                "content": res["answer"],
                "sources": res.get("sources", []),
                "intent": res.get("intent", "UNKNOWN"),
                "retrieved_facts": res.get("retrieved_facts", {}),
            })
            st.rerun()

    # ─────────────────────────────────────────────────────────────
    # TAB 2: Computer Vision Terrain Image Analysis
    # ─────────────────────────────────────────────────────────────
    with tab_cv:
        st.markdown("### 📷 Experimental Slope & Landscape Computer Vision")
        st.caption(
            "Upload an optical slope or landscape photograph to analyze superficial ground indicators: "
            "surface tension cracks (high-gradient dark fissures), bare soil exposure / scar segmentation, "
            "and texture irregularity (hummocky slope deformation)."
        )

        cv_c1, cv_c2 = st.columns([1, 1])

        with cv_c1:
            st.markdown("#### 1. Input Image Selection")
            cv_source = st.radio(
                "Image Source",
                ["Load Pre-Configured Test Scenario", "Upload Custom Photograph"],
                horizontal=True,
                key="cv_source_choice",
            )

            cv_bytes: bytes | None = None
            sample_desc = ""

            if cv_source == "Upload Custom Photograph":
                uploaded_file = st.file_uploader(
                    "Choose an image (PNG, JPG, WEBP, max 10MB)",
                    type=["png", "jpg", "jpeg", "webp"],
                    key="cv_file_uploader",
                )
                if uploaded_file is not None:
                    cv_bytes = uploaded_file.getvalue()
            else:
                sample_name = st.selectbox(
                    "Select a benchmark terrain scenario:",
                    [
                        "Scenario A: Vegetated Slope with Tension Crack Fissures",
                        "Scenario B: Steep Bare Soil Scarp & Exposed Debris Scar",
                        "Scenario C: Stable Dense Mountain Forest Canopy",
                    ],
                    key="cv_sample_select",
                )

                def _generate_sample_bytes(key: str) -> bytes:
                    import io
                    from PIL import Image
                    np.random.seed(42)
                    h, w = 320, 480
                    if "Tension Crack" in key:
                        arr = np.zeros((h, w, 3), dtype=np.uint8)
                        arr[:, :] = [45, 140, 45]
                        arr[70:250, 110:370] = [145, 95, 50]
                        for y in range(90, 230):
                            x1 = int(190 + 15 * np.sin(y / 14.0))
                            x2 = int(270 + 20 * np.cos(y / 16.0))
                            if 0 <= x1 < w:
                                arr[y, max(0, x1-1):min(w, x1+2)] = [18, 12, 8]
                            if 0 <= x2 < w:
                                arr[y, max(0, x2-1):min(w, x2+2)] = [15, 10, 6]
                    elif "Bare Soil Scarp" in key:
                        arr = np.zeros((h, w, 3), dtype=np.uint8)
                        arr[:70, :] = [40, 130, 40]
                        for y in range(70, h):
                            for x in range(w):
                                n = int(np.random.randint(-18, 18))
                                arr[y, x] = [
                                    int(np.clip(165 + n, 0, 255)),
                                    int(np.clip(100 + n // 2, 0, 255)),
                                    int(np.clip(55 + n // 3, 0, 255)),
                                ]
                    else:
                        arr = np.zeros((h, w, 3), dtype=np.uint8)
                        for y in range(h):
                            for x in range(w):
                                g = int(np.clip(135 + np.random.randint(-22, 22), 0, 255))
                                arr[y, x] = [int(g * 0.32), g, int(g * 0.30)]
                    bio = io.BytesIO()
                    Image.fromarray(arr).save(bio, format="PNG")
                    return bio.getvalue()

                cv_bytes = _generate_sample_bytes(sample_name)

        with cv_c2:
            st.markdown("#### 2. Computer Vision Feature Analysis")
            if cv_bytes is None:
                st.info("Please upload an image or choose a pre-configured scenario to run analysis.")
            else:
                analyzer = get_cv_analyzer()
                cv_res = analyzer.analyze(cv_bytes)

                if not cv_res.get("available"):
                    st.error(f"Analysis Failed: {cv_res.get('error')}")
                else:
                    ca = cv_res["composite_assessment"]
                    cd = cv_res["crack_detection"]
                    sc = cv_res["surface_composition"]
                    td = cv_res["texture_and_deformation"]

                    st.markdown(
                        f"""
                        <div style="background:#131b2e; border: 1px solid #24314a; border-left: 4px solid {ca['severity_color']}; border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.8rem;">
                            <div style="font-size:0.8rem; color:#94a3b8;">Composite Visual Anomaly Level</div>
                            <div style="font-size:1.3rem; font-weight:700; color:{ca['severity_color']};">{ca['anomaly_tier']}</div>
                            <div style="font-size:0.8rem; color:#cbd5e1; margin-top:2px;">Anomaly Score: <b>{ca['anomaly_score']:.3f}</b> / 1.000</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Crack Candidates", f"{cd['crack_candidate_count']}", f"span ~{cd['max_crack_length_px']:.0f}px")
                    with m2:
                        st.metric("Soil Exposure", f"{sc['soil_exposure_pct']}%", sc["exposure_classification"])
                    with m3:
                        st.metric("Vegetative Cover", f"{sc['vegetation_cover_pct']}%")
                    with m4:
                        st.metric("Texture Irregularity", f"{td['texture_roughness_score']:.3f}")

        if cv_bytes is not None and cv_res.get("available"):
            st.divider()
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown("**Original Landscape / Slope Photograph**")
                st.image(cv_bytes, width="stretch", caption="Input terrain image")
            with img_col2:
                st.markdown("**Annotated Feature Mask (Cracks in Red, Soil in Amber)**")
                import base64
                st.image(
                    base64.b64decode(cv_res["overlay_base64"]),
                    width="stretch",
                    caption="Computer vision segmentation overlay",
                )

            st.markdown("##### 🔍 Detected Visual Indicators")
            for ind in td.get("detected_indicators", []):
                st.markdown(f"- 🔎 {ind}")

            st.warning(
                "⚠️ **EXPERIMENTAL COMPUTER VISION NOTICE:** "
                "Optical images only reflect superficial ground characteristics and cannot measure subsurface pore water pressure, "
                "shear strength, or subterranean slip planes. Visual anomaly classification is NOT proof of an active landslide and "
                "must NEVER independently trigger an official warning or emergency evacuation."
            )

    # ─────────────────────────────────────────────────────────────
    # TAB 3: Multilingual Safety Guidance
    # ─────────────────────────────────────────────────────────────
    with tab_safety:
        st.markdown("### 🌐 Multilingual Landslide Disaster Preparedness & Safety")
        st.caption(
            "Verified civil-defense protocols from the National Disaster Management Authority (NDMA India) "
            "and Geological Survey of India (GSI). Available in English, Hindi, and Punjabi."
        )

        lang_col, _ = st.columns([1, 2])
        with lang_col:
            selected_lang = st.radio(
                "Select Language / भाषा चुनें / ਭਾਸ਼ਾ ਚੁਣੋ:",
                ["en", "hi", "pa"],
                format_func=lambda x: {"en": "🇬🇧 English", "hi": "🇮🇳 हिन्दी (Hindi)", "pa": "🇮🇳 ਪੰਜਾਬੀ (Punjabi)"}.get(x, x),
                horizontal=True,
                key="safety_lang_radio",
            )

        guide = get_safety_guidance(selected_lang)

        st.markdown(f"#### 🛡️ {guide['title']}")
        st.caption(f"**Authority:** {guide['authority']} · {guide['summary']}")

        # Before
        with st.expander(f"🟡 {guide['before']['category']}", expanded=True):
            st.markdown(f"**{guide['before']['warning_signs_heading']}**")
            for ws in guide["before"]["warning_signs"]:
                st.markdown(f"- ⚠️ {ws}")
            st.markdown(f"**{guide['before']['action_items_heading']}**")
            for ai in guide["before"]["action_items"]:
                st.markdown(f"- ✅ {ai}")

        # During
        with st.expander(f"🔴 {guide['during']['category']}", expanded=False):
            st.markdown(f"**{guide['during']['indoor_heading']}**")
            for ia in guide["during"]["indoor_actions"]:
                st.markdown(f"- 🏠 {ia}")
            st.markdown(f"**{guide['during']['outdoor_heading']}**")
            for oa in guide["during"]["outdoor_actions"]:
                st.markdown(f"- 🏃 {oa}")
            st.markdown(f"**{guide['during']['vehicle_heading']}**")
            for va in guide["during"]["vehicle_actions"]:
                st.markdown(f"- 🚗 {va}")
            st.markdown(f"**{guide['during']['trapped_heading']}**")
            for ta in guide["during"]["trapped_actions"]:
                st.markdown(f"- 🛡️ {ta}")

        # After
        with st.expander(f"🟢 {guide['after']['category']}", expanded=False):
            st.markdown(f"**{guide['after']['safety_heading']}**")
            for sr in guide["after"]["safety_rules"]:
                st.markdown(f"- 🔍 {sr}")

        # Emergency Contacts Table
        st.markdown("#### 📞 Emergency Helplines (India)")
        ec_df = pd.DataFrame(guide["emergency_contacts"])
        st.table(ec_df)

        st.info(f"ℹ️ {guide['disclaimer']}")


# ═══════════════════════════════════════════════════════════════
# PAGE: DISASTER RESPONSE & COMMUNITY INTELLIGENCE (PHASE 9)
# ═══════════════════════════════════════════════════════════════

elif page == "🛡️ Disaster Response & Community":
    render_page_header(
        "🛡️ Disaster Response & Community Intelligence",
        "Nearby emergency facilities, OSRM road routing, crowd-sourced hazard incident reporting, historical event replay, and automated PDF reports.",
        pills=[
            ("🏥 Facilities & Routes", "pill-blue"),
            ("📢 Community Reports", "pill-green"),
            ("⏪ Event Replay", "pill-amber"),
            ("📄 PDF Reports", "pill-blue"),
        ],
    )
    experimental_banner(
        "Routes and emergency facilities are for informational preparedness only. "
        "Routes do NOT account for active landslide road blockages. "
        "Community reports undergo review and are strictly segregated from scientific training data."
    )

    from src.geospatial.facilities import fetch_nearby_facilities
    from src.geospatial.routing import calculate_emergency_route, ROUTING_DISCLAIMER
    from src.alerts.community_reporting import (
        submit_community_report,
        REPORT_CATEGORIES,
        REPORT_SEVERITIES,
        COMMUNITY_DATA_NOTICE,
    )
    from src.database.models import get_community_reports, get_locations, get_location_by_id, get_observations
    from src.monitoring.event_replay import list_replayable_events, replay_historical_event, HISTORICAL_REPLAY_DISCLAIMER
    from src.utils.pdf_generator import generate_assessment_pdf_report
    from src.monitoring.offline_cache import get_offline_cache
    from src.monitoring.live_monitor import get_live_monitor

    tab_fac, tab_rep, tab_hist, tab_pdf, tab_offline = st.tabs([
        "🏥 Facilities & Emergency Routing",
        "📢 Community Incident Reporting",
        "⏪ Historical Event Replay",
        "📄 Automated PDF Assessment Reports",
        "📡 Offline Monitoring & Caching",
    ])

    # ─────────────────────────────────────────────────────────────
    # TAB 1: Facilities & Route Visualization
    # ─────────────────────────────────────────────────────────────
    with tab_fac:
        st.markdown("### 🏥 Emergency Facilities & Route Visualization")
        st.caption(
            "Search for nearest hospitals, police stations, fire brigades, and emergency relief shelters "
            "using OpenStreetMap Overpass API (with curated regional Indian fallback). "
            "Generates driving routes via Open Source Routing Machine (OSRM)."
        )

        all_locs_df = get_locations(active_only=False)
        all_locs = all_locs_df.to_dict(orient="records") if (all_locs_df is not None and not all_locs_df.empty) else []
        loc_options = {loc["name"]: loc for loc in all_locs}

        r_c1, r_c2, r_c3 = st.columns([2, 1, 1])
        with r_c1:
            if loc_options:
                sel_loc_name = st.selectbox("Select Origin Monitoring Station", list(loc_options.keys()), key="resp_stn_select")
                sel_stn = loc_options[sel_loc_name]
                stn_lat, stn_lon = float(sel_stn["latitude"]), float(sel_stn["longitude"])
            else:
                st.warning("No monitoring stations found. Please enter coordinates manually.")
                stn_lat, stn_lon = 10.0889, 77.0595
        with r_c2:
            stn_lat = st.number_input("Origin Latitude", value=float(stn_lat), format="%.4f", key="resp_orig_lat")
        with r_c3:
            stn_lon = st.number_input("Origin Longitude", value=float(stn_lon), format="%.4f", key="resp_orig_lon")

        radius_km = st.slider("Search Radius (km)", min_value=10, max_value=80, value=35, step=5, key="resp_radius_slider")

        fac_result = fetch_nearby_facilities(stn_lat, stn_lon, radius_km=radius_km)
        facilities = fac_result.get("facilities", [])

        # Facility summary metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Facilities", f"{fac_result['total_facilities']}", fac_result["freshness_label"])
        with m2:
            nh = fac_result.get("nearest_hospital")
            nh_val = f"{nh['distance_km']} km" if nh else "None"
            st.metric("Nearest Hospital", nh_val, nh["name"][:22] if nh else "Out of radius")
        with m3:
            np_fac = fac_result.get("nearest_police")
            np_val = f"{np_fac['distance_km']} km" if np_fac else "None"
            st.metric("Nearest Police", np_val, np_fac["name"][:22] if np_fac else "Out of radius")
        with m4:
            ns = fac_result.get("nearest_shelter")
            ns_val = f"{ns['distance_km']} km" if ns else "None"
            st.metric("Nearest Shelter", ns_val, ns["name"][:22] if ns else "Out of radius")

        st.caption(f"**Data Provenance:** `{fac_result['data_source']}` · Last updated: `{fac_result['timestamp']}`")

        # Routing Section
        st.markdown("#### 🚗 Route Visualization to Facility")
        if facilities:
            fac_dict = {f"{f['name']} ({f['category'].title()}, {f['distance_km']} km)": f for f in facilities}
            chosen_fac_label = st.selectbox("Select Target Facility for Route Planning", list(fac_dict.keys()), key="resp_target_fac")
            target_fac = fac_dict[chosen_fac_label]

            route_res = calculate_emergency_route(
                start_lat=stn_lat,
                start_lon=stn_lon,
                end_lat=target_fac["latitude"],
                end_lon=target_fac["longitude"],
                facility_name=target_fac["name"],
                facility_category=target_fac["category"],
            )

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.metric("Route Distance", f"{route_res['distance_km']} km", "Estimated road distance")
            with rc2:
                st.metric("Estimated Travel Time", f"{route_res['estimated_duration_min']} min", "Assuming clear road")
            with rc3:
                st.metric("Routing Engine", route_res["routing_engine"], "Road network model")

            # Route Map using Plotly Scattermap
            coords = route_res.get("route_geojson", {}).get("geometry", {}).get("coordinates", [])
            fig_route = go.Figure()
            if coords:
                fig_route.add_trace(go.Scattermap(
                    lat=[c[1] for c in coords],
                    lon=[c[0] for c in coords],
                    mode="lines",
                    line=dict(width=4, color="#ef4444"),
                    name="Driving Route",
                ))

            # Facility markers
            fac_lats = [f["latitude"] for f in facilities]
            fac_lons = [f["longitude"] for f in facilities]
            fac_texts = [f"{f['name']} ({f['category']})" for f in facilities]
            fig_route.add_trace(go.Scattermap(
                lat=fac_lats,
                lon=fac_lons,
                mode="markers",
                marker=dict(size=10, color="#38bdf8"),
                text=fac_texts,
                name="Facilities in Radius",
            ))

            # Origin marker
            fig_route.add_trace(go.Scattermap(
                lat=[stn_lat],
                lon=[stn_lon],
                mode="markers+text",
                marker=dict(size=14, color="#f59e0b"),
                text=["Origin Station"],
                textposition="top right",
                name="Origin",
            ))

            # Target marker
            fig_route.add_trace(go.Scattermap(
                lat=[target_fac["latitude"]],
                lon=[target_fac["longitude"]],
                mode="markers+text",
                marker=dict(size=14, color="#22c55e"),
                text=[target_fac["name"]],
                textposition="top right",
                name="Target Destination",
            ))

            center_lat = (stn_lat + target_fac["latitude"]) / 2.0
            center_lon = (stn_lon + target_fac["longitude"]) / 2.0
            fig_route.update_layout(
                map=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=10),
                margin=dict(l=0, r=0, t=0, b=0),
                height=420,
            )
            st.plotly_chart(fig_route, width="stretch")

            st.warning(f"⚠️ {route_res['disclaimer']}")

    # ─────────────────────────────────────────────────────────────
    # TAB 2: Community Incident Reporting
    # ─────────────────────────────────────────────────────────────
    with tab_rep:
        st.markdown("### 📢 Community Landslide & Hazard Incident Reporting")
        st.caption(
            "Report observed slope disturbances, tension cracks, rockfalls, or road subsidence directly from the field. "
            "Reports enter review status ('submitted') and are verified by administrators. "
            "They are strictly segregated from scientific training data."
        )

        with st.form("community_report_form", clear_on_submit=True):
            st.markdown("#### 1. Submit New Hazard Incident Report")
            f_c1, f_c2 = st.columns(2)
            with f_c1:
                rep_loc_name = st.text_input("Incident Location / Landmark Name *", placeholder="e.g. Near Meppadi Viewpoint, NH 766")
                rep_state = st.text_input("State / UT", placeholder="e.g. Kerala")
                rep_dist = st.text_input("District", placeholder="e.g. Wayanad")
                rep_cat = st.selectbox("Hazard Category *", REPORT_CATEGORIES)
            with f_c2:
                rep_lat = st.number_input("Latitude Coordinate *", value=float(stn_lat), format="%.4f")
                rep_lon = st.number_input("Longitude Coordinate *", value=float(stn_lon), format="%.4f")
                rep_dt = st.text_input("Observation Date & Time *", value=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
                rep_sev = st.selectbox("Observed Hazard Severity", REPORT_SEVERITIES, index=1)

            rep_desc = st.text_area("Detailed Hazard Description *", placeholder="Describe observed ground cracks, falling stones, water accumulation, tree tilting, or road slumping (min 10 characters)...")

            p_c1, p_c2 = st.columns(2)
            with p_c1:
                rep_photo = st.file_uploader("Attach Optional Photo (max 10MB)", type=["png", "jpg", "jpeg", "webp"], key="comm_photo_upload")
            with p_c2:
                rep_name = st.text_input("Reporter Name (Optional)", placeholder="Local Resident / Field Volunteer")
                rep_contact = st.text_input("Contact Info (Optional)", placeholder="Phone / Email for verification")

            submit_btn = st.form_submit_button("📢 Submit Community Report", width="stretch")

            if submit_btn:
                photo_b = rep_photo.getvalue() if rep_photo else None
                res_sub = submit_community_report(
                    location_name=rep_loc_name,
                    latitude=rep_lat,
                    longitude=rep_lon,
                    incident_date=rep_dt,
                    category=rep_cat,
                    description=rep_desc,
                    reporter_name=rep_name,
                    reporter_contact=rep_contact,
                    state_name=rep_state,
                    district_name=rep_dist,
                    severity=rep_sev,
                    photo_bytes=photo_b,
                )
                if res_sub.get("success"):
                    st.success(f"✅ {res_sub['message']}")
                    st.info(f"ℹ️ {res_sub['notice']}")
                else:
                    st.error(f"❌ Submission Failed: {res_sub.get('error')}")

        st.divider()
        st.markdown("#### 📋 Existing Community Hazard Reports")
        status_filter = st.selectbox("Filter Reports by Status", ["All Statuses", "submitted", "under_review", "verified", "rejected"], key="comm_status_filter")
        filter_val = None if status_filter == "All Statuses" else status_filter

        comm_reps = get_community_reports(status=filter_val, limit=50)
        if not comm_reps:
            st.info("No community hazard reports match the selected status filter.")
        else:
            rep_df = pd.DataFrame(comm_reps)[["id", "location_name", "category", "severity", "incident_date", "status", "created_at"]]
            st.dataframe(rep_df, width="stretch")

    # ─────────────────────────────────────────────────────────────
    # TAB 3: Historical Event Replay
    # ─────────────────────────────────────────────────────────────
    with tab_hist:
        st.markdown("### ⏪ Historical Landslide Event Replay")
        st.caption(
            "Explore authentic historical landslide events from the national inventory. "
            "Reconstructs recorded environmental conditions and evaluates retrospective risk. "
            "Anti-hallucination commitment: never fabricates hourly rainfall curves or claims retrospective prediction proof."
        )

        replay_events = list_replayable_events(limit=50, authentic_only=True)
        if not replay_events:
            st.warning("No authentic historical events available for replay.")
        else:
            event_dict = {
                f"#{ev['id']} · {ev.get('event_date', 'Unknown Date')} · {ev.get('state_region', 'Region')} ({ev.get('trigger', 'Rain')})": ev["id"]
                for ev in replay_events
            }
            chosen_event_label = st.selectbox("Select Historical Event to Replay", list(event_dict.keys()), key="hist_replay_select")
            chosen_event_id = event_dict[chosen_event_label]

            replay_data = replay_historical_event(chosen_event_id)
            if not replay_data.get("available"):
                st.error(replay_data.get("error"))
            else:
                ep = replay_data["event_profile"]
                env_c = replay_data["environmental_conditions"]
                tl = replay_data["rainfall_timeline"]
                retro = replay_data["retrospective_assessment"]

                e1, e2, e3, e4 = st.columns(4)
                with e1:
                    st.metric("Event Date", ep["event_date"], ep["state_region"])
                with e2:
                    st.metric("Recorded Trigger", ep["trigger"], ep["landslide_type"][:18])
                with e3:
                    st.metric("Recorded Fatalities", f"{ep['fatalities']}", f"Injuries: {ep['injuries']}")
                with e4:
                    st.metric("Catalog Source", ep["source_name"][:20], "NASA GLC / Inventory")

                st.markdown("##### 🌧️ Historical Environmental & Rainfall Conditions")
                h_c1, h_c2 = st.columns(2)
                with h_c1:
                    st.markdown(f"- **Coordinates:** `{ep['latitude']:.4f}°N, {ep['longitude']:.4f}°E`")
                    st.markdown(f"- **Recorded Slope:** `{env_c.get('recorded_slope') or 'Unavailable in catalog'}`")
                    st.markdown(f"- **Recorded Elevation:** `{env_c.get('recorded_elevation') or 'Unavailable in catalog'}`")
                    st.markdown(f"- **Recorded 24h Rain:** `{env_c.get('recorded_24h_rain') or 'Summary rain only'}`")
                with h_c2:
                    st.markdown(f"- **Hourly Rainfall Series:** `{tl['status']}`")
                    st.markdown(f"- **Data Provenance:** `{env_c['source']}`")

                st.markdown("##### 🤖 Retrospective Model Assessment")
                if retro.get("evaluation_status") == "Success":
                    prob = retro.get("estimated_probability")
                    prob_str = f"{prob * 100:.1f}%" if prob is not None else "N/A"
                    st.success(
                        f"Retrospective Evaluation: **{retro.get('estimated_risk_category')} Risk** "
                        f"(Model Score: `{prob_str}`) · Model: `{retro.get('model_algorithm')}`\n\n"
                        f"*{retro.get('retrospective_notice')}*"
                    )
                else:
                    st.info(f"ℹ️ {retro.get('reason', 'Retrospective evaluation unavailable for this historical event.')}")

                st.caption(f"⚠️ {replay_data['disclaimer']}")

    # ─────────────────────────────────────────────────────────────
    # TAB 4: Automated PDF Assessment Reports
    # ─────────────────────────────────────────────────────────────
    with tab_pdf:
        st.markdown("### 📄 Automated PDF Hazard Assessment Reports")
        st.caption(
            "Generate downloadable, publication-ready PDF risk evaluation reports formatted with ReportLab. "
            "Includes station metrics, live/reanalysis telemetry, ML predictions, nearby facilities, and disclaimers."
        )

        all_locs_pdf_df = get_locations(active_only=False)
        all_locs_pdf = all_locs_pdf_df.to_dict(orient="records") if (all_locs_pdf_df is not None and not all_locs_pdf_df.empty) else []
        if not all_locs_pdf:
            st.warning("No monitoring locations available to generate report.")
        else:
            loc_dict_pdf = {loc["name"]: loc for loc in all_locs_pdf}
            pdf_loc_name = st.selectbox("Select Target Station for PDF Report", list(loc_dict_pdf.keys()), key="pdf_stn_select")
            pdf_stn = loc_dict_pdf[pdf_loc_name]

            lm_pdf = get_live_monitor()
            snap_pdf = lm_pdf.get_live_snapshot(active_only=False)
            stn_snap_pdf = next((s for s in snap_pdf.get("stations", []) if s.get("location_id") == pdf_stn["id"]), None)

            w_pdf = stn_snap_pdf.get("latest_weather", {}) if stn_snap_pdf else {}
            ra_pdf = stn_snap_pdf.get("latest_risk", {}) if stn_snap_pdf else {}

            facs_pdf = fetch_nearby_facilities(pdf_stn["latitude"], pdf_stn["longitude"], radius_km=30.0, max_results=4).get("facilities", [])
            obs_pdf_df = get_observations(include_synthetic=False, limit=4)
            obs_pdf = obs_pdf_df.to_dict(orient="records") if (obs_pdf_df is not None and not obs_pdf_df.empty) else []

            if st.button("📥 Generate Hazard Assessment PDF", key="gen_pdf_btn", width="stretch"):
                with st.spinner("Compiling ReportLab document..."):
                    pdf_bytes = generate_assessment_pdf_report(
                        location_profile=pdf_stn,
                        weather_telemetry=w_pdf,
                        risk_assessment=ra_pdf,
                        historical_events=obs_pdf,
                        nearby_facilities=facs_pdf,
                        is_synthetic_benchmark=False,
                    )
                st.success(f"✅ Generated PDF report ({len(pdf_bytes):,} bytes) successfully.")
                st.download_button(
                    label=f"⬇️ Download {pdf_loc_name} Assessment Report.pdf",
                    data=pdf_bytes,
                    file_name=f"LandslideGuard_Assessment_Report_{pdf_stn['id']}.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

    # ─────────────────────────────────────────────────────────────
    # TAB 5: Offline Monitoring & Cache Health
    # ─────────────────────────────────────────────────────────────
    with tab_offline:
        st.markdown("### 📡 Offline Monitoring & Data Synchronization")
        st.caption(
            "Persistent telemetry caches ensure LandslideGuard remains functional during remote mountain field deployments. "
            "Data freshness badges strictly prevent presenting cached data as live telemetry."
        )

        cache_mgr = get_offline_cache()
        off_status = cache_mgr.get_offline_status()

        o1, o2, o3 = st.columns(3)
        with o1:
            st.metric("Offline Caching", "Enabled" if off_status["offline_cache_enabled"] else "Disabled", "Persistent disk cache")
        with o2:
            st.metric("Total Cached Records", f"{off_status['total_cached_records']}", "Available offline")
        with o3:
            st.metric("Last Synchronization", off_status["sync_age"], off_status["last_successful_synchronization"][:19])

        st.info(f"🛡️ **Integrity Commitment:** {off_status['integrity_rule']}")

        if off_status.get("cached_entries"):
            st.markdown("#### 📂 Cached Telemetry & Facility Records")
            c_df = pd.DataFrame(off_status["cached_entries"])
            st.dataframe(c_df, width="stretch")


# ═══════════════════════════════════════════════════════════════
# PAGE: ADMIN PANEL & GOVERNANCE (PHASE 9)
# ═══════════════════════════════════════════════════════════════

elif page == "🔐 Admin Panel":
    render_page_header(
        "🔐 Administrative Control & Governance",
        "Role-Based Access Control (RBAC) administration: review community incident reports, manage monitoring stations, govern experimental alerts, and inspect system audit logs.",
        pills=[
            ("🔐 RBAC Protected", "pill-amber"),
            ("📋 Audit Logs", "pill-blue"),
            ("⚙️ Station Manager", "pill-green"),
        ],
    )
    experimental_banner(
        "Administrative actions directly modify application state and SQLite database records. "
        "Default evaluation credentials: admin / landslideguard2026."
    )

    from src.utils.auth import authenticate_user, can_manage_locations, can_review_community_reports, can_manage_alerts
    from src.database.models import (
        get_community_reports,
        update_community_report_status,
        get_locations,
        insert_location,
        update_location_status,
        get_alerts,
        acknowledge_alert,
        resolve_alert,
    )
    from src.monitoring.live_monitor import get_live_monitor

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False
        st.session_state.admin_user = None
        st.session_state.admin_role = None

    # Login form if not authenticated as admin
    if not st.session_state.admin_logged_in:
        st.markdown("### 🔐 Administrator Login")
        st.caption("Please authenticate with administrative credentials to access governance controls.")

        with st.form("admin_login_form"):
            l_u = st.text_input("Username", value="admin", placeholder="admin")
            l_p = st.text_input("Password", type="password", placeholder="Enter admin password")
            login_btn = st.form_submit_button("🔑 Login as Administrator", width="stretch")

            if login_btn:
                ok, role, err = authenticate_user(l_u, l_p)
                if ok and role == "admin":
                    st.session_state.admin_logged_in = True
                    st.session_state.admin_user = l_u
                    st.session_state.admin_role = role
                    st.success("✅ Successfully authenticated as Administrator.")
                    st.rerun()
                else:
                    st.error(f"❌ Authentication Failed: {err or 'Invalid credentials'}")
    else:
        # Authenticated Admin Dashboard
        admin_c1, admin_c2 = st.columns([3, 1])
        with admin_c1:
            st.markdown(f"**Logged in as:** `Administrator ({st.session_state.admin_user})` · Role: `ADMIN`")
        with admin_c2:
            if st.button("🚪 Logout", key="admin_logout_btn"):
                st.session_state.admin_logged_in = False
                st.session_state.admin_user = None
                st.session_state.admin_role = None
                st.rerun()

        adm_tab1, adm_tab2, adm_tab3, adm_tab4 = st.tabs([
            "📢 Community Report Review",
            "📡 Station Management",
            "🚨 Alert Governance",
            "📊 System Activity & Counts",
        ])

        # Sub-tab 1: Review Community Reports
        with adm_tab1:
            st.markdown("#### 📢 Audit Community Hazard Reports")
            st.caption("Review submitted community hazard observations, verify coordinates and photos, and update status.")

            status_sel = st.selectbox("Status Filter", ["submitted", "under_review", "verified", "rejected"], key="adm_rep_status")
            reports = get_community_reports(status=status_sel, limit=50)

            if not reports:
                st.info(f"No reports with status '{status_sel}' are currently awaiting action.")
            else:
                for rep in reports:
                    with st.expander(f"Report #{rep['id']} · {rep['location_name']} ({rep['category']}) · {rep['created_at'][:16]}", expanded=True):
                        st.markdown(f"**Hazard Category:** `{rep['category']}` · **Severity:** `{rep['severity']}`")
                        st.markdown(f"**Location:** `{rep['location_name']}` ({rep.get('district_name') or 'N/A'}, {rep.get('state_name') or 'N/A'})")
                        st.markdown(f"**Coordinates:** `{rep['latitude']:.4f}°N, {rep['longitude']:.4f}°E`")
                        st.markdown(f"**Description:** {rep['description']}")
                        if rep.get("reporter_name"):
                            st.caption(f"Reported by: {rep['reporter_name']} ({rep.get('reporter_contact') or 'No contact'})")

                        if rep.get("photo_base64"):
                            import base64
                            try:
                                st.image(base64.b64decode(rep["photo_base64"]), caption="Attached Citizen Photo", width=300)
                            except Exception:
                                st.caption("Could not render photo.")

                        notes_val = st.text_input("Administrative Audit Notes", value=rep.get("review_notes") or "", key=f"notes_{rep['id']}")
                        b_c1, b_c2, b_c3 = st.columns(3)
                        with b_c1:
                            if st.button("Mark Under Review", key=f"btn_ur_{rep['id']}"):
                                update_community_report_status(rep["id"], "under_review", notes_val, st.session_state.admin_user)
                                st.success(f"Report #{rep['id']} set to under_review.")
                                st.rerun()
                        with b_c2:
                            if st.button("✅ Verify Report", key=f"btn_ver_{rep['id']}"):
                                update_community_report_status(rep["id"], "verified", notes_val, st.session_state.admin_user)
                                st.success(f"Report #{rep['id']} marked as verified.")
                                st.rerun()
                        with b_c3:
                            if st.button("❌ Reject Report", key=f"btn_rej_{rep['id']}"):
                                update_community_report_status(rep["id"], "rejected", notes_val, st.session_state.admin_user)
                                st.warning(f"Report #{rep['id']} rejected.")
                                st.rerun()

        # Sub-tab 2: Station Management
        with adm_tab2:
            st.markdown("#### 📡 Manage Monitored Stations")
            all_stns = get_locations(active_only=False)
            if all_stns is not None and not all_stns.empty:
                disp_cols = [c for c in ["id", "name", "latitude", "longitude", "elevation_m", "slope_degrees", "is_active"] if c in all_stns.columns]
                st.dataframe(all_stns[disp_cols], width="stretch")
            else:
                st.info("No registered monitoring stations found.")

            with st.form("new_station_form"):
                st.markdown("##### Register New Monitoring Station")
                ns_c1, ns_c2 = st.columns(2)
                with ns_c1:
                    n_name = st.text_input("Station Name *", placeholder="e.g. Joshimath Lower Slopes")
                    n_lat = st.number_input("Latitude *", value=30.5564, format="%.4f")
                    n_lon = st.number_input("Longitude *", value=79.5667, format="%.4f")
                with ns_c2:
                    n_state = st.text_input("State / Region", value="Uttarakhand")
                    n_elev = st.number_input("Elevation (m)", value=1850.0)
                    n_slope = st.number_input("Slope (degrees)", value=32.0)
                add_stn_btn = st.form_submit_button("➕ Register Station", width="stretch")

                if add_stn_btn and n_name:
                    new_id = insert_location(
                        name=n_name,
                        latitude=n_lat,
                        longitude=n_lon,
                        state_region=n_state,
                        elevation_m=n_elev,
                        slope_degrees=n_slope,
                        is_active=True,
                    )
                    st.success(f"✅ Registered station #{new_id}: {n_name}")
                    st.rerun()

        # Sub-tab 3: Alert Governance
        with adm_tab3:
            st.markdown("#### 🚨 Alert Lifecycle Governance")
            unack_alerts_df = get_alerts(include_acknowledged=False, limit=20)
            unack_alerts = unack_alerts_df.to_dict(orient="records") if (unack_alerts_df is not None and not unack_alerts_df.empty) else []
            if not unack_alerts:
                st.success("No active unacknowledged alerts currently require administrative review.")
            else:
                for a in unack_alerts:
                    with st.expander(f"Alert #{a['id']} · {a.get('location_name', 'Location')} · Tier: {a['risk_category']}", expanded=True):
                        st.markdown(f"**Probability:** `{a.get('probability', 0)*100:.1f}%` · **Triggered:** `{a['created_at']}`")
                        a_c1, a_c2 = st.columns(2)
                        with a_c1:
                            if st.button("Acknowledge", key=f"adm_ack_{a['id']}"):
                                acknowledge_alert(a["id"], note=f"Acknowledged by {st.session_state.admin_user}")
                                st.success(f"Alert #{a['id']} acknowledged.")
                                st.rerun()
                        with a_c2:
                            if st.button("Resolve", key=f"adm_res_{a['id']}"):
                                resolve_alert(a["id"], resolution_note=f"Resolved by {st.session_state.admin_user}")
                                st.success(f"Alert #{a['id']} resolved.")
                                st.rerun()

        # Sub-tab 4: System Activity & Counts
        with adm_tab4:
            st.markdown("#### 📊 System Activity & Database Table Ledger")
            counts = get_table_counts()
            st.dataframe(pd.DataFrame(list(counts.items()), columns=["Table Name", "Row Count"]), width="stretch")


# ═══════════════════════════════════════════════════════════════
# PAGE 11: DATA MANAGEMENT
# ═══════════════════════════════════════════════════════════════

elif page == "🗄️ Data Management":
    render_page_header(
        "🗄️ Data Management & Inventory",
        "Inspect SQLite storage tables, authentic vs. synthetic dataset provenance, regional caches, and CSV exports.",
        pills=[("🗄️ SQLite Backend", "pill-blue"), ("📦 Provenance Audit", "pill-amber")],
    )
    experimental_banner()

    from src.database.models import get_alerts, get_observations, get_recent_predictions_with_location

    db_counts = get_table_counts()
    auth_df = get_observations(include_synthetic=False, limit=10000)
    all_obs_df = get_observations(include_synthetic=True, limit=10000)
    auth_count = int(len(auth_df)) if not auth_df.empty else 0
    synth_count = max(0, int(len(all_obs_df)) - auth_count)

    dm1, dm2, dm3, dm4, dm5 = st.columns(5)
    with dm1:
        st.metric("Locations", f"{db_counts.get('locations', 0):,}")
    with dm2:
        st.metric("Observations (Total)", f"{db_counts.get('observations', 0):,}")
    with dm3:
        st.metric("Authentic DB Records", f"{auth_count:,}", help="Observations with is_synthetic = 0")
    with dm4:
        st.metric("Environmental Logs", f"{db_counts.get('environmental_observations', 0):,}")
    with dm5:
        st.metric("Alerts Logged", f"{db_counts.get('alerts', 0):,}")

    st.markdown("### 📊 SQLite Table Breakdown")
    tbl_rows = [{"Table": k, "Row Count": max(0, int(v))} for k, v in db_counts.items()]
    st.dataframe(pd.DataFrame(tbl_rows), width="stretch", hide_index=True)

    st.markdown("### 📦 Dataset Provenance & Inventory")
    inv_path = Path(__file__).parent / "data" / "dataset_inventory.csv"
    if inv_path.exists():
        try:
            inv_df = pd.read_csv(inv_path)
            st.caption(f"Loaded from `{inv_path}`")
            st.dataframe(inv_df, width="stretch", hide_index=True)
        except Exception as exc:
            st.warning(f"Could not read `dataset_inventory.csv`: {exc}")
    else:
        st.info("`data/dataset_inventory.csv` not found.")

    parquet_df = load_dataset()
    if parquet_df is not None and not parquet_df.empty:
        p_synth = bool(parquet_df["is_synthetic"].all()) if "is_synthetic" in parquet_df.columns else True
        st.info(
            f"**Processed ML Dataset (`data/processed/dataset.parquet`):** "
            f"`{len(parquet_df):,}` rows · `{len(parquet_df.columns)}` columns · "
            f"Provenance: **{'Synthetic Development Data' if p_synth else 'Authentic Verified Data'}** · "
            f"SQLite breakdown: `{auth_count}` authentic vs. `{synth_count}` synthetic observation rows."
        )
    else:
        st.warning("Processed dataset (`data/processed/dataset.parquet`) is currently unavailable.")

    st.markdown("### 🗺️ India Boundary & Cache Artifacts")
    root_dir = Path(__file__).parent
    artifact_checks = [
        ("States GeoJSON", root_dir / "data" / "india_boundaries" / "states.geojson"),
        ("Districts GeoJSON", root_dir / "data" / "india_boundaries" / "districts.geojson"),
        ("India Cache Directory", root_dir / "data" / "india_monitoring_cache"),
        ("SQLite Database", root_dir / (get_setting("database", "path") or "data/landslideguard.db")),
    ]
    art_rows = []
    for label, p in artifact_checks:
        exists = p.exists()
        size_str = (
            f"{(p.stat().st_size / 1024):.1f} KB"
            if exists and p.is_file()
            else (f"{len(list(p.iterdir()))} items" if exists and p.is_dir() else "Missing")
        )
        art_rows.append({"Artifact": label, "Path": str(p.relative_to(root_dir)), "Status": "Available" if exists else "Missing", "Size / Items": size_str})
    st.dataframe(pd.DataFrame(art_rows), width="stretch", hide_index=True)

    st.markdown("### ⬇️ Export Stored Records (CSV)")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if not all_obs_df.empty:
            st.download_button(
                "Download Observations CSV",
                data=all_obs_df.to_csv(index=False).encode("utf-8"),
                file_name="landslideguard_observations.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.button("Observations CSV (Empty)", disabled=True, width="stretch")
    with ex2:
        pred_export_df = get_recent_predictions_with_location(limit=1000)
        if not pred_export_df.empty:
            st.download_button(
                "Download Predictions CSV",
                data=pred_export_df.to_csv(index=False).encode("utf-8"),
                file_name="landslideguard_predictions.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.button("Predictions CSV (Empty)", disabled=True, width="stretch")
    with ex3:
        alert_export_df = get_alerts(include_acknowledged=True, limit=1000)
        if not alert_export_df.empty:
            st.download_button(
                "Download Alerts CSV",
                data=alert_export_df.to_csv(index=False).encode("utf-8"),
                file_name="landslideguard_alerts.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.button("Alerts CSV (Empty)", disabled=True, width="stretch")


# ═══════════════════════════════════════════════════════════════
# PAGE 12: SETTINGS
# ═══════════════════════════════════════════════════════════════

elif page == "⚙️ Settings":
    import os
    from src.utils.settings import load_settings

    render_page_header(
        "⚙️ System Configuration & Provider Settings",
        "Read-only inspector for active YAML runtime thresholds, alert cooldowns, API endpoints, and credential presence.",
        pills=[("⚙️ config/settings.yaml", "pill-blue"), ("🔒 Secrets Masked", "pill-green")],
    )
    experimental_banner()

    cfg = load_settings()
    risk_cfg = cfg.get("risk", {})
    thresholds = risk_cfg.get("thresholds", {})
    alerts_cfg = cfg.get("alerts", {})
    notif_cfg = alerts_cfg.get("notifications", {})
    mon_cfg = cfg.get("monitoring", {})
    prov_cfg = cfg.get("providers", {})

    st.markdown("### 🎚️ Provisional Risk Thresholds & Monitoring Policy")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Low Upper Bound", f"{thresholds.get('low', 0.35):.2f}")
    with s2:
        st.metric("Moderate Upper Bound", f"{thresholds.get('moderate', 0.55):.2f}")
    with s3:
        st.metric("High Upper Bound", f"{thresholds.get('high', 0.75):.2f}")
    with s4:
        st.metric("Alert Cooldown", f"{alerts_cfg.get('cooldown_minutes', 120)} min")
    with s5:
        st.metric("Freshness Window", f"{mon_cfg.get('data_freshness_hours', 6)} h")

    st.info(
        f"**Trigger Category:** `{alerts_cfg.get('trigger_category', 'High')}` · "
        f"**Email Notifications:** `{'Enabled' if notif_cfg.get('email_enabled') else 'Disabled (Default Safe Mode)'}` · "
        f"**SMS Notifications:** `{'Enabled' if notif_cfg.get('sms_enabled') else 'Disabled (Default Safe Mode)'}`"
    )

    st.markdown("### 🌐 External Data Providers")
    prov_rows = []
    for name, section in prov_cfg.items():
        if isinstance(section, dict):
            prov_rows.append({
                "Provider": name,
                "Enabled": bool(section.get("enabled", True)),
                "Base / Feed URL": str(
                    section.get("base_url")
                    or section.get("forecast_url")
                    or section.get("rss_url")
                    or section.get("district_warning_url")
                    or "Local / Built-in"
                ),
                "Timeout (s)": str(section.get("timeout_seconds")) if section.get("timeout_seconds") is not None else "—",
                "Cache TTL": str(section.get("cache_ttl_minutes") or section.get("cache_ttl_hours") or "—"),
            })
    if prov_rows:
        st.dataframe(pd.DataFrame(prov_rows), width="stretch", hide_index=True)

    st.markdown("### 🔐 Environment Variable Presence (Values Never Displayed)")
    env_keys = [
        ("OPEN_METEO_API_KEY", "Optional commercial Open-Meteo key"),
        ("NASA_EARTHDATA_TOKEN", "Optional NASA GPM / SRTM token"),
        ("IMD_API_KEY", "Optional IMD authenticated feed key"),
        ("SMTP_HOST", "Optional alert email relay host"),
        ("TWILIO_ACCOUNT_SID", "Optional alert SMS gateway SID"),
    ]
    env_rows = [
        {
            "Environment Key": key,
            "Purpose": desc,
            "Status": "Configured" if bool(os.environ.get(key)) else "Not Set (Optional / Public Feed)",
        }
        for key, desc in env_keys
    ]
    st.dataframe(pd.DataFrame(env_rows), width="stretch", hide_index=True)
    st.caption("Edit `config/settings.yaml` or `.env` locally and restart the application to apply configuration changes.")
