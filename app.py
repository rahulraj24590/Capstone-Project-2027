"""
LandslideGuard — Main Streamlit Dashboard

Six-page interactive dashboard for:
1. Overview
2. Interactive Risk Map
3. Risk Prediction
4. Historical Analysis
5. Model Performance
6. Alerts

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
    "🗺️ Interactive Risk Map",
    "🔮 Risk Prediction",
    "📊 Historical Analysis",
    "🤖 Model Performance",
    "🚨 Alerts",
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

if page == PAGES[0]:
    df = load_dataset()
    model_info = load_model_info()
    db_counts = get_table_counts()
    alg = model_info.get("metadata", {}).get("algorithm", "N/A") if model_info else "N/A"
    alg_label = alg.replace("_", " ").title() if alg != "N/A" else "Not Trained"

    render_page_header(
        "⛰️ LandslideGuard Dashboard",
        APP_DESCRIPTION,
        pills=[
            ("⚠️ Experimental Prototype", "pill-amber"),
            (f"🤖 {alg_label}", "pill-blue"),
            ("📦 Synthetic Dev Dataset", "pill"),
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
# PAGE 2: INTERACTIVE RISK MAP
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[1]:
    render_page_header(
        "🗺️ Interactive Risk Map",
        "Explore spatial coverage of historical / synthetic observation points across Indian mountain belts.",
        pills=[("Folium Marker Clustering", "pill-blue")],
    )
    experimental_banner()

    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Run: `python scripts/generate_synthetic_data.py`")
        st.stop()

    with st.expander("🔍 Map Filters & Region Presets", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns([1.2, 1.2, 1.1, 0.9])
        with fc1:
            show_events = st.multiselect(
                "Record Types", ["Landslide Events", "Non-events"],
                default=["Landslide Events", "Non-events"],
            )
        with fc2:
            region_choice = st.selectbox("Region Focus", list(REGION_BOUNDS.keys()))
        with fc3:
            if "event_date" in df.columns:
                try:
                    dates = pd.to_datetime(df["event_date"], errors="coerce")
                    min_yr = int(dates.dt.year.min())
                    max_yr = int(dates.dt.year.max())
                    yr_range = st.slider("Year Range", min_yr, max_yr, (min_yr, max_yr))
                except Exception:
                    yr_range = None
            else:
                yr_range = None
        with fc4:
            max_points = st.slider("Max Markers", 100, 2000, 500, step=100)

    plot_df = df.copy()
    if "Landslide Events" not in show_events and "landslide_occurred" in plot_df.columns:
        plot_df = plot_df[plot_df["landslide_occurred"] == 0]
    if "Non-events" not in show_events and "landslide_occurred" in plot_df.columns:
        plot_df = plot_df[plot_df["landslide_occurred"] == 1]

    bounds = REGION_BOUNDS.get(region_choice)
    if bounds:
        lat_min, lat_max, lon_min, lon_max = bounds
        plot_df = plot_df[
            plot_df["latitude"].between(lat_min, lat_max)
            & plot_df["longitude"].between(lon_min, lon_max)
        ]

    if yr_range and "event_date" in plot_df.columns:
        try:
            dates = pd.to_datetime(plot_df["event_date"], errors="coerce")
            plot_df = plot_df[dates.dt.year.between(yr_range[0], yr_range[1])]
        except Exception:
            pass

    plot_df = plot_df.dropna(subset=["latitude", "longitude"])
    plot_df = plot_df[plot_df["latitude"].between(-90, 90) & plot_df["longitude"].between(-180, 180)].head(max_points)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Plotted Points", f"{len(plot_df):,}", delta=f"of {len(df):,} total", delta_color="off")
    with s2:
        ne = int(plot_df["landslide_occurred"].sum()) if "landslide_occurred" in plot_df.columns and not plot_df.empty else 0
        st.metric("Events Shown", f"{ne:,}")
    with s3:
        lat_rng = f"{plot_df['latitude'].min():.1f}°–{plot_df['latitude'].max():.1f}°" if not plot_df.empty else "N/A"
        st.metric("Latitude Span", lat_rng)
    with s4:
        lon_rng = f"{plot_df['longitude'].min():.1f}°–{plot_df['longitude'].max():.1f}°" if not plot_df.empty else "N/A"
        st.metric("Longitude Span", lon_rng)

    center_lat = float(plot_df["latitude"].mean()) if bounds and not plot_df.empty else 22.5
    center_lon = float(plot_df["longitude"].mean()) if bounds and not plot_df.empty else 82.0

    try:
        from src.geospatial.maps import build_landslide_map
        import streamlit_folium as stf
        m = build_landslide_map(observations_df=plot_df, center_lat=center_lat, center_lon=center_lon)
        stf.st_folium(m, width=None, height=540, returned_objects=[], key="main_map")
    except ImportError:
        if not plot_df.empty:
            color_col = "landslide_occurred" if "landslide_occurred" in plot_df.columns else None
            fig = px.scatter_map(
                plot_df, lat="latitude", lon="longitude",
                color=plot_df[color_col].astype(str) if color_col else None,
                color_discrete_map={"0": "#2ecc71", "1": "#e74c3c"},
                zoom=4, height=540, map_style="carto-darkmatter",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=5, b=5, l=5, r=5))
            st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════
# PAGE 3: RISK PREDICTION
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[2]:
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
            st.markdown("### 🧠 SHAP Feature Contributions")
            try:
                from src.models.model_registry import load_model, active_model_exists
                from src.models.predict import load_feature_columns, load_training_stats

                if active_model_exists():
                    model = load_model("best_model")
                    feat_cols = load_feature_columns()
                    stats = load_training_stats()
                    medians = stats.get("feature_medians", {})
                    feat_vector = np.array([[
                        float(feat_snapshot[c]) if feat_snapshot.get(c) is not None else float(medians.get(c, 0.0))
                        for c in feat_cols
                    ]])
                    scaler = model.named_steps.get("scaler") if hasattr(model, "named_steps") else None
                    clf = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model

                    df_bg = load_dataset()
                    if df_bg is not None and all(c in df_bg.columns for c in feat_cols):
                        bg_raw = df_bg[feat_cols].fillna(medians).head(100).values.astype(float)
                    else:
                        bg_raw = np.array([[float(medians.get(c, 0.0)) for c in feat_cols]])

                    X_input = scaler.transform(feat_vector) if scaler is not None else feat_vector
                    X_bg = scaler.transform(bg_raw) if scaler is not None else bg_raw

                    if hasattr(clf, "coef_"):
                        try:
                            import shap
                            explainer = shap.LinearExplainer(clf, X_bg)
                            shap_values = explainer(X_input)
                            shap_df = pd.DataFrame({
                                "Feature": feat_cols,
                                "SHAP Value": shap_values.values[0],
                            }).sort_values("SHAP Value", key=abs, ascending=True)
                            fig_shap = px.bar(
                                shap_df, x="SHAP Value", y="Feature", orientation="h",
                                color="SHAP Value",
                                color_continuous_scale=["#2ecc71", "#e2e8f0", "#e74c3c"],
                                color_continuous_midpoint=0,
                                title="Log-Odds Push Relative to Dataset Baseline",
                            )
                            _style_fig(fig_shap, height=320)
                            st.plotly_chart(fig_shap, width="stretch")
                            st.caption("Red bars push predicted risk higher; green bars push risk lower.")
                        except Exception:
                            coefs = abs(clf.coef_[0])
                            imp_df = pd.DataFrame({"Feature": feat_cols, "Weight (abs)": coefs}).sort_values("Weight (abs)", ascending=True)
                            fig_imp = px.bar(imp_df, x="Weight (abs)", y="Feature", orientation="h",
                                             title="Logistic Regression |Coefficients|",
                                             color="Weight (abs)", color_continuous_scale=["#1a4731", "#e74c3c"])
                            _style_fig(fig_imp, height=320)
                            st.plotly_chart(fig_imp, width="stretch")
                    elif hasattr(clf, "feature_importances_"):
                        imp_df = pd.DataFrame({"Feature": feat_cols, "Importance": clf.feature_importances_}).sort_values("Importance", ascending=True)
                        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                                         color="Importance", color_continuous_scale=["#1a4731", "#e74c3c"])
                        _style_fig(fig_imp, height=320)
                        st.plotly_chart(fig_imp, width="stretch")
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

elif page == PAGES[3]:
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

elif page == PAGES[4]:
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
    with st.expander("⚠️ Known Modeling Caveats & Limitations", expanded=False):
        st.markdown("""
- All metrics computed on **synthetic development data** — do not reflect real-world performance.
- Synthetic data is separable by construction → inflated precision/recall/AUC metrics.
- No spatial cross-validation without authentic geo-referenced inventories.
- Class imbalance handled with `class_weight='balanced'` (`scale_pos_weight` for XGBoost).
""")


# ═══════════════════════════════════════════════════════════════
# PAGE 6: ALERTS
# ═══════════════════════════════════════════════════════════════

elif page == PAGES[5]:
    from src.alerts.alert_manager import get_alert_manager
    manager = get_alert_manager()

    render_page_header(
        "🚨 Experimental Alert Management",
        "Evaluate dynamic risk conditions against alert thresholds, inspect active alerts, and log acknowledgements.",
        pills=[("Experimental Only", "pill-amber")],
    )
    experimental_banner(
        "These alerts are EXPERIMENTAL. They are NOT official warnings and must NOT "
        "trigger evacuation or emergency response."
    )

    trigger = get_setting("alerts", "trigger_category") or "High"
    cooldown = get_setting("alerts", "cooldown_minutes") or 120
    max_alerts = get_setting("alerts", "max_active_alerts") or 50

    active_df = manager.get_active_alerts()
    all_df = manager.get_all_alerts(limit=50)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Active Unacknowledged", len(active_df) if isinstance(active_df, pd.DataFrame) else 0)
    with k2:
        st.metric("Logged Alert History", len(all_df) if isinstance(all_df, pd.DataFrame) else 0)
    with k3:
        st.metric("Trigger Tier", trigger)
    with k4:
        st.metric("Cooldown Window", f"{cooldown} min")

    st.divider()
    st.markdown("### 🔔 Evaluate Conditions & Trigger Test Alert")
    with st.form("alert_form"):
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            a_slope = st.slider("Slope (°)", 0.0, 90.0, 40.0, key="a_slope")
            a_rf24 = st.number_input("24h Rainfall (mm)", 0.0, 2000.0, 150.0, key="a_rf24")
        with ac2:
            a_rf72 = st.number_input("72h Rainfall (mm)", 0.0, 5000.0, 300.0, key="a_rf72")
            a_elev = st.number_input("Elevation (m)", 0.0, 9000.0, 1200.0, key="a_elev")
        with ac3:
            a_moist = st.slider("Soil Moisture", 0.0, 1.0, 0.75, key="a_moist")
            a_veg = st.slider("Vegetation Index", -1.0, 1.0, 0.15, key="a_veg")
        a_loc = st.text_input("Location Name (optional)", "Test Location")
        generate = st.form_submit_button("⚡ Evaluate & Generate Alert", type="primary", width="stretch")

    if generate:
        features = {
            "slope_degrees": a_slope, "rainfall_24h_mm": a_rf24, "rainfall_72h_mm": a_rf72,
            "elevation_m": a_elev, "soil_moisture": a_moist, "vegetation_index": a_veg,
        }
        with st.spinner("Evaluating risk and checking alert conditions…"):
            result = manager.check_location(features=features, location_name=a_loc or None)

        risk_r = result.get("risk_result", {})
        prob = risk_r.get("probability")
        cat = risk_r.get("risk_category", "Unknown")
        alert = result.get("alert")
        cat_colors = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c", "Very High": "#9b59b6", "Unknown": "#95a5a6"}
        c_col = cat_colors.get(cat, "#95a5a6")

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
                st.success(f"✅ Alert Generated (ID: {alert.get('id', '?')})")
            else:
                st.info("ℹ️ Risk below alert trigger tier or within cooldown.")
        if alert:
            st.caption(alert.get("notice", ""))
        active_df = manager.get_active_alerts()
        all_df = manager.get_all_alerts(limit=50)

    st.divider()
    tab_active, tab_history = st.tabs(["📋 Active Alerts", "📜 Alert History"])

    with tab_active:
        try:
            if isinstance(active_df, pd.DataFrame) and not active_df.empty:
                st.dataframe(active_df, width="stretch", hide_index=True)
                st.markdown("#### ✅ Acknowledge an Active Alert")
                ak1, ak2, ak3 = st.columns([1.2, 2, 1])
                active_ids = [int(x) for x in active_df["id"].tolist()]
                with ak1:
                    ack_id = st.selectbox("Select Alert ID", active_ids, key="ack_id")
                with ak2:
                    ack_note = st.text_input("Acknowledgement Note (optional)", key="ack_note", placeholder="e.g. Reviewed during drill")
                with ak3:
                    st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
                    if st.button("✅ Acknowledge", key="ack_btn", width="stretch"):
                        ok = manager.acknowledge(int(ack_id), ack_note or None)
                        if ok:
                            st.success(f"Alert {ack_id} acknowledged.")
                            st.rerun()
                        else:
                            st.error("Alert not found.")
            else:
                st.info("No active unacknowledged alerts at this time.")
        except Exception as e:
            st.error(f"Error loading alerts: {e}")

    with tab_history:
        try:
            if isinstance(all_df, pd.DataFrame) and not all_df.empty:
                st.dataframe(all_df, width="stretch", hide_index=True)
            else:
                st.info("No alert history recorded yet.")
        except Exception as e:
            st.error(f"Error loading history: {e}")
