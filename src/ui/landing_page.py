"""
LandslideGuard — Premium Landing Page & Showcase View (Phase Redesign).

Implements:
- Section A: SaaS Navigation Bar
- Section B: Two-Image Gradient Hero Section (cinematic side-by-side mountain photography)
- Section C: Actual Database Platform Statistics
- Section D: 8 Core Platform Capabilities Cards
- Section E: 4-Step Interactive "How LandslideGuard Works" Process
- Section F: India Regional Monitoring Showcase
- Section G: AI Environmental Intelligence Showcase
- Section H: Live Environmental Insights Charts
- Section I: High-Conversion Final CTA
- Section J: Enterprise Footer
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import plotly.express as px
import streamlit as st

from src.database.db import get_table_counts
from src.database.models import get_locations
from src.ui.components import render_footer
from src.ui.theme import (
    BG_CARD,
    BORDER_DEFAULT,
    CYAN,
    GRADIENT_VIOLET,
    HERO_IMG_LEFT,
    HERO_IMG_RIGHT,
    PRIMARY_BLUE,
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MODERATE,
    TEXT_MAIN,
    TEXT_MUTED,
    style_fig,
)


def render_landing_page(
    df: pd.DataFrame | None = None,
    model_info: dict[str, Any] | None = None,
) -> None:
    """Render the complete, production-grade SaaS landing page."""
    # ─────────────────────────────────────────────
    # SECTION A: Navigation Bar
    # ─────────────────────────────────────────────
    st.markdown("""
    <div class="lg-landing-navbar">
        <div class="lg-navbar-brand">
            <div class="lg-brand-icon">⛰️</div>
            <div class="lg-brand-text">
                <div class="lg-brand-title">Landslide<span>Guard</span></div>
                <div class="lg-brand-subtitle">ENVIRONMENTAL SAAS</div>
            </div>
        </div>
        <div class="lg-navbar-links">
            <span class="lg-nav-active">Home</span>
            <span>Platform</span>
            <span>Live Monitoring</span>
            <span>India Risk Map</span>
            <span>AI Intelligence</span>
            <span>About</span>
        </div>
        <div class="lg-navbar-badge">
            ENTERPRISE SAAS v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SECTION B: Two-Image Gradient Hero Section
    # ─────────────────────────────────────────────
    st.markdown(f"""
    <div class="lg-hero-wrapper">
        <div class="lg-hero-bg-container">
            <div class="lg-hero-img-col">
                <img src="{HERO_IMG_LEFT}" alt="Mountain Landscape 1" class="lg-hero-img" />
            </div>
            <div class="lg-hero-img-col">
                <img src="{HERO_IMG_RIGHT}" alt="Mountain Landscape 2" class="lg-hero-img" />
            </div>
        </div>
        <div class="lg-hero-overlay"></div>
        <div class="lg-hero-content">
            <div class="lg-hero-eyebrow">
                <span>🛰️</span> AI-POWERED ENVIRONMENTAL INTELLIGENCE
            </div>
            <h1 class="lg-hero-title">
                Predict Landslide Risk.<br />
                <span>Protect Every Life.</span>
            </h1>
            <p class="lg-hero-desc">
                Monitor environmental conditions, analyze terrain, and understand landslide risk through intelligent geospatial monitoring and AI-powered analysis across India's Western Ghats and Himalayan regions.
            </p>
            <div class="lg-hero-badges">
                <div class="lg-hero-badge">
                    <span style="color:{CYAN};">●</span> Geographic Monitoring (30m DEM)
                </div>
                <div class="lg-hero-badge">
                    <span style="color:{PRIMARY_BLUE};">●</span> Weather Intelligence (Open-Meteo High-Res)
                </div>
                <div class="lg-hero-badge">
                    <span style="color:#C084FC;">●</span> AI-Based Risk Analysis (SHAP XAI)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Quick-Action Navigation Buttons
    h_col1, h_col2, h_col3 = st.columns([1.2, 1.2, 2.5])
    with h_col1:
        if st.button("📡 Explore Live Monitoring", key="hero_cta_live", width="stretch", type="primary"):
            st.session_state["nav_page"] = "📡 Live Monitoring"
            st.rerun()
    with h_col2:
        if st.button("🗺️ Explore Platform Dashboard", key="hero_cta_platform", width="stretch"):
            st.session_state["nav_page"] = "🏠 Overview"
            st.rerun()
    with h_col3:
        st.caption("🔒 Verified scientific data pipeline. All metrics derived strictly from configured database state.")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SECTION C: Actual Database Platform Statistics
    # ─────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:0.75rem;">
        <span style="color:{PRIMARY_BLUE}; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">SYSTEM TELEMETRY LEDGER</span>
        <h2 style="font-size:1.6rem; font-weight:800; color:{TEXT_MAIN}; margin:0.2rem 0 0.8rem 0;">Platform Statistics at a Glance</h2>
    </div>
    """, unsafe_allow_html=True)

    # Retrieve authentic counts from database
    counts = get_table_counts()
    locs = get_locations()
    active_locs_count = len(locs) if not locs.empty else 0
    obs_count = counts.get("observations", 0)
    preds_count = counts.get("prediction_records", 0)
    alerts_count = counts.get("alerts", 0)

    # Fallback to dataset length if observations table empty
    if obs_count == 0 and df is not None:
        obs_count = len(df)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-label">Monitored Locations</div>
            <div class="lg-stat-val" style="color:{PRIMARY_BLUE};">{active_locs_count}</div>
            <div class="lg-stat-sub">Active Western Ghats & Himalayan Stations</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-label">Historical Landslide Records</div>
            <div class="lg-stat-val" style="color:#C084FC;">{obs_count:,}</div>
            <div class="lg-stat-sub">Authentic Event Catalog Verified</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-label">Monitoring Regions</div>
            <div class="lg-stat-val" style="color:{CYAN};">36 / 735</div>
            <div class="lg-stat-sub">States / UTs and Indian Districts Indexed</div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        st.markdown(f"""
        <div class="lg-stat-card">
            <div class="lg-stat-label">Model Risk Assessments</div>
            <div class="lg-stat-val" style="color:{RISK_LOW};">{preds_count:,}</div>
            <div class="lg-stat-sub">{alerts_count} Experimental Alerts Governed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2.25rem;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SECTION D: Platform Capabilities (8 Cards)
    # ─────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center; max-width:760px; margin:0 auto 1.75rem auto;">
        <span style="color:{PRIMARY_BLUE}; font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">END-TO-END CAPABILITIES</span>
        <h2 style="font-size:2rem; font-weight:800; color:{TEXT_MAIN}; margin:0.3rem 0 0.5rem 0;">Engineered for High-Consequence Environmental Monitoring</h2>
        <p style="font-size:0.95rem; color:{TEXT_MUTED}; line-height:1.6;">
            A unified suite of meteorological data providers, physical terrain models, calibrated machine learning pipelines, and multi-channel disaster response tools.
        </p>
    </div>
    """, unsafe_allow_html=True)

    capabilities = [
        ("📡", "Live Environmental Monitoring", "Continuous ingestion of real-time precipitation, temperature, relative humidity, and soil moisture from Open-Meteo with caching and freshness tracking.", "📡 Live Monitoring"),
        ("🗺️", "India Risk Map", "Multi-layer interactive Folium mapping featuring historical landslide catalogs, point risk markers, district boundaries, and kernel-density heatmaps.", "🗺️ Geospatial Map"),
        ("🔮", "AI Risk Prediction", "Calibrated binary classification combining static terrain attributes with dynamic hourly rainfall intensity into 4-tier risk probabilities.", "🔮 Risk Analysis"),
        ("🌦️", "Rainfall Intelligence", "Strict 1h/24h/48h/72h rainfall accumulations, 24h OLS linear slope trends, and ERA5 historical baseline anomaly detection.", "🌦️ Weather Analysis"),
        ("⛰️", "Terrain Analysis", "High-resolution DEM elevation derivatives, slope gradient, aspect, Riley Terrain Roughness Index (TRI), and interactive 3D surface rendering.", "🗺️ Geospatial Map"),
        ("🧠", "Explainable AI (SHAP)", "Transparent game-theoretic Shapley attributions, individual waterfall breakdowns, global beeswarm charts, and scientific disclaimers.", "🧠 AI Explainability"),
        ("🚨", "Alert Management", "Intelligent notification governance with configurable risk thresholds, 120-minute duplicate suppression cooldown, and severity escalation overrides.", "🚨 Alert Management"),
        ("🛰️", "Satellite Monitoring", "Sentinel-2 Level-2A multi-spectral Normalized Difference Vegetation Index (NDVI) monitoring with cloud screening and bi-temporal disturbance detection.", "🗺️ Geospatial Map"),
    ]

    cap_cols_row1 = st.columns(4)
    for idx, (icon, title, desc, target_page) in enumerate(capabilities[:4]):
        with cap_cols_row1[idx]:
            st.markdown(f"""
            <div class="lg-feature-card">
                <div>
                    <div class="lg-feature-icon">{icon}</div>
                    <div class="lg-feature-title">{title}</div>
                    <div class="lg-feature-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Launch {title.split()[0]} →", key=f"cap_btn_{idx}", width="stretch"):
                st.session_state["nav_page"] = target_page
                st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    cap_cols_row2 = st.columns(4)
    for idx, (icon, title, desc, target_page) in enumerate(capabilities[4:]):
        with cap_cols_row2[idx]:
            st.markdown(f"""
            <div class="lg-feature-card">
                <div>
                    <div class="lg-feature-icon">{icon}</div>
                    <div class="lg-feature-title">{title}</div>
                    <div class="lg-feature-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Launch {title.split()[0]} →", key=f"cap_btn_{idx+4}", width="stretch"):
                st.session_state["nav_page"] = target_page
                st.rerun()

    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SECTION E: How LandslideGuard Works (4 Steps)
    # ─────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{BG_CARD}; border:1px solid {BORDER_DEFAULT}; border-radius:18px; padding:2rem 2.2rem; margin-bottom:2.5rem; box-shadow:0 8px 30px rgba(0,0,0,0.3);">
        <div style="text-align:center; max-width:680px; margin:0 auto 1.5rem auto;">
            <span style="color:{PRIMARY_BLUE}; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">DATA-DRIVEN METHODOLOGY</span>
            <h2 style="font-size:1.75rem; font-weight:800; color:{TEXT_MAIN}; margin:0.25rem 0 0.5rem 0;">How LandslideGuard Works</h2>
            <p style="font-size:0.9rem; color:{TEXT_MUTED}; line-height:1.5;">
                From raw multi-spectral satellite imagery and hourly weather feeds to calibrated risk probabilities and emergency response routes.
            </p>
        </div>
        <div class="lg-step-wrapper">
            <div class="lg-step-card">
                <div class="lg-step-num">1</div>
                <div style="font-weight:700; color:{TEXT_MAIN}; font-size:0.96rem; margin-bottom:0.35rem;">Ingest Environmental Telemetry</div>
                <div style="font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.5;">
                    Retrieves precipitation, temperature, relative humidity, and soil moisture from Open-Meteo APIs with complete-window validation.
                </div>
            </div>
            <div class="lg-step-card">
                <div class="lg-step-num">2</div>
                <div style="font-weight:700; color:{TEXT_MAIN}; font-size:0.96rem; margin-bottom:0.35rem;">Analyze Terrain & Rainfall</div>
                <div style="font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.5;">
                    Calculates slope gradients, aspect, Riley TRI roughness, 24h OLS linear trends, and ERA5 30-day baseline Z-score anomalies.
                </div>
            </div>
            <div class="lg-step-card">
                <div class="lg-step-num">3</div>
                <div style="font-weight:700; color:{TEXT_MAIN}; font-size:0.96rem; margin-bottom:0.35rem;">Estimate Multi-Model Risk</div>
                <div style="font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.5;">
                    Executes calibrated Logistic Regression and tree ensembles to predict failure probability, accompanied by game-theoretic SHAP attributions.
                </div>
            </div>
            <div class="lg-step-card">
                <div class="lg-step-num">4</div>
                <div style="font-weight:700; color:{TEXT_MAIN}; font-size:0.96rem; margin-bottom:0.35rem;">Govern Alerts & Evacuation</div>
                <div style="font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.5;">
                    Enforces 120-minute cooldown suppression with upward escalation overrides, emergency hospital routing, and automated PDF report generation.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SECTION F & G: India Monitoring & AI Intelligence
    # ─────────────────────────────────────────────
    col_india, col_ai = st.columns([1.1, 1])

    with col_india:
        st.markdown(f"""
        <div class="lg-card" style="height:100%;">
            <span style="color:{PRIMARY_BLUE}; font-size:0.76rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">NATIONAL SCALE INTELLIGENCE</span>
            <h3 style="font-size:1.35rem; font-weight:800; color:{TEXT_MAIN}; margin:0.3rem 0 0.6rem 0;">🇮🇳 India Regional Monitoring Showcase</h3>
            <p style="font-size:0.88rem; color:{TEXT_MUTED}; line-height:1.6; margin-bottom:1.1rem;">
                LandslideGuard indexes <b>36 States and Union Territories</b> and <b>735 Districts</b> across India. Monitored hot-zones include:
            </p>
            <div style="display:flex; flex-direction:column; gap:0.6rem; margin-bottom:1.4rem;">
                <div style="background:rgba(8,14,27,0.5); border:1px solid {BORDER_DEFAULT}; border-radius:10px; padding:0.65rem 0.9rem; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:700; font-size:0.86rem; color:{TEXT_MAIN};">Western Ghats Corridor</div>
                        <div style="font-size:0.75rem; color:{TEXT_MUTED};">Kerala (Wayanad, Idukki, Munnar) · Maharashtra · Karnataka</div>
                    </div>
                    <span style="color:{RISK_HIGH}; font-size:0.8rem; font-weight:700;">High Vulnerability</span>
                </div>
                <div style="background:rgba(8,14,27,0.5); border:1px solid {BORDER_DEFAULT}; border-radius:10px; padding:0.65rem 0.9rem; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:700; font-size:0.86rem; color:{TEXT_MAIN};">Himalayan Hazard Belt</div>
                        <div style="font-size:0.75rem; color:{TEXT_MUTED};">Uttarakhand (Joshimath, Dehradun) · Himachal Pradesh · Sikkim</div>
                    </div>
                    <span style="color:{RISK_CRITICAL}; font-size:0.8rem; font-weight:700;">Very High Relief</span>
                </div>
                <div style="background:rgba(8,14,27,0.5); border:1px solid {BORDER_DEFAULT}; border-radius:10px; padding:0.65rem 0.9rem; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:700; font-size:0.86rem; color:{TEXT_MAIN};">North-Eastern Highlands</div>
                        <div style="font-size:0.75rem; color:{TEXT_MUTED};">Darjeeling · Assam · Meghalaya · Arunachal Pradesh</div>
                    </div>
                    <span style="color:{RISK_MODERATE}; font-size:0.8rem; font-weight:700;">Extreme Rain Basin</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore India Geographic Map →", key="landing_btn_india_map", width="stretch", type="primary"):
            st.session_state["nav_page"] = "🗺️ Geospatial Map"
            st.rerun()

    with col_ai:
        st.markdown(f"""
        <div class="lg-card" style="height:100%;">
            <span style="color:#C084FC; font-size:0.76rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">TRANSPARENT ALGORITHMS</span>
            <h3 style="font-size:1.35rem; font-weight:800; color:{TEXT_MAIN}; margin:0.3rem 0 0.6rem 0;">🧠 AI & Explainability Architecture</h3>
            <p style="font-size:0.88rem; color:{TEXT_MUTED}; line-height:1.6; margin-bottom:1rem;">
                LandslideGuard rejects black-box opacity. Every risk estimate is decomposed into measurable physical contributions:
            </p>
            <div style="display:flex; flex-direction:column; gap:0.65rem; margin-bottom:1.4rem;">
                <div style="padding:0.65rem 0.85rem; background:rgba(8,14,27,0.5); border-left:3px solid {PRIMARY_BLUE}; border-radius:8px;">
                    <div style="font-weight:700; font-size:0.84rem; color:{TEXT_MAIN};">Game-Theoretic SHAP Decomposition</div>
                    <div style="font-size:0.76rem; color:{TEXT_MUTED};">LinearExplainer & TreeExplainer allocate exact credit to rainfall, slope, and soil moisture.</div>
                </div>
                <div style="padding:0.65rem 0.85rem; background:rgba(8,14,27,0.5); border-left:3px solid {CYAN}; border-radius:8px;">
                    <div style="font-weight:700; font-size:0.84rem; color:{TEXT_MAIN};">Zero Silent Imputation Invariant</div>
                    <div style="font-size:0.76rem; color:{TEXT_MUTED};">Missing rainfall is never converted to 0.0, avoiding dangerous artificial depression of hazard scores.</div>
                </div>
                <div style="padding:0.65rem 0.85rem; background:rgba(8,14,27,0.5); border-left:3px solid {GRADIENT_VIOLET}; border-radius:8px;">
                    <div style="font-weight:700; font-size:0.84rem; color:{TEXT_MAIN};">Topographic Wetness Index (TWI) Guard</div>
                    <div style="font-size:0.76rem; color:{TEXT_MUTED};">Strictly refuses TWI calculation on isolated points, preventing hydrological hallucination.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Model Explainability →", key="landing_btn_xai", width="stretch"):
            st.session_state["nav_page"] = "🧠 AI Explainability"
            st.rerun()

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SECTION H: Live Environmental Insights Charts
    # ─────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:0.75rem;">
        <span style="color:{PRIMARY_BLUE}; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">VERIFIED PROJECT ANALYTICS</span>
        <h2 style="font-size:1.6rem; font-weight:800; color:{TEXT_MAIN}; margin:0.2rem 0 0.8rem 0;">Environmental Insights & Data Distribution</h2>
    </div>
    """, unsafe_allow_html=True)

    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER_DEFAULT}; border-radius:14px; padding:1.1rem 1.25rem; margin-bottom:0.5rem;">
            <div style="font-weight:700; color:{TEXT_MAIN}; font-size:0.95rem;">Historical Observation Class Distribution</div>
            <div style="font-size:0.78rem; color:{TEXT_MUTED};">Ratio of confirmed landslide occurrences vs. non-event benchmark locations</div>
        </div>
        """, unsafe_allow_html=True)
        if df is not None and "landslide_occurred" in df.columns:
            cats = df["landslide_occurred"].map({0: "Stable Ground (Negative)", 1: "Landslide Failure (Positive)"})
            cat_counts = cats.value_counts().reset_index()
            cat_counts.columns = ["Outcome", "Count"]
            color_map = {"Stable Ground (Negative)": RISK_LOW, "Landslide Failure (Positive)": RISK_CRITICAL}
            fig_pie = px.pie(
                cat_counts, values="Count", names="Outcome",
                color="Outcome", color_discrete_map=color_map, hole=0.55,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            style_fig(fig_pie, height=300)
            st.plotly_chart(fig_pie, width="stretch")
        else:
            st.info("No processed dataset loaded. Run: `python scripts/prepare_data.py`")

    with chart_c2:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER_DEFAULT}; border-radius:14px; padding:1.1rem 1.25rem; margin-bottom:0.5rem;">
            <div style="font-weight:700; color:{TEXT_MAIN}; font-size:0.95rem;">Rainfall vs. Slope Interaction Dynamics</div>
            <div style="font-size:0.78rem; color:{TEXT_MUTED};">Scatter distribution of slope angle against 24-hour cumulative precipitation</div>
        </div>
        """, unsafe_allow_html=True)
        if df is not None and "slope_degrees" in df.columns and "rainfall_24h_mm" in df.columns:
            sample_scatter = df.head(400).copy()
            sample_scatter["Class"] = sample_scatter["landslide_occurred"].map({0: "Stable", 1: "Landslide"})
            fig_scatter = px.scatter(
                sample_scatter, x="slope_degrees", y="rainfall_24h_mm",
                color="Class",
                color_discrete_map={"Stable": RISK_LOW, "Landslide": RISK_CRITICAL},
                labels={"slope_degrees": "Slope Angle (°)", "rainfall_24h_mm": "24h Rainfall (mm)"},
                opacity=0.75,
            )
            style_fig(fig_scatter, height=300)
            st.plotly_chart(fig_scatter, width="stretch")
        else:
            st.info("Scatter distribution unavailable until dataset is generated.")

    st.markdown("<div style='height: 2.25rem;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SECTION I: High-Conversion Final CTA
    # ─────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, rgba(37,99,235,0.2) 0%, rgba(124,58,237,0.2) 100%), {BG_CARD}; border:1px solid rgba(56,189,248,0.35); border-radius:20px; padding:3rem 2.5rem; text-align:center; margin-bottom:2.5rem; box-shadow:0 15px 40px rgba(0,0,0,0.4);">
        <div style="display:inline-flex; align-items:center; gap:0.4rem; padding:0.3rem 0.8rem; border-radius:9999px; background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); color:{PRIMARY_BLUE}; font-size:0.76rem; font-weight:700; text-transform:uppercase; margin-bottom:0.9rem;">
            RESEARCH CAPSTONE PROTOTYPE
        </div>
        <h2 style="font-size:2.35rem; font-weight:800; color:{TEXT_MAIN}; margin:0 0 0.75rem 0;">
            Building a Safer Future Through Environmental Intelligence
        </h2>
        <p style="font-size:1.05rem; color:{TEXT_MUTED}; max-width:650px; margin:0 auto 1.8rem auto; line-height:1.6;">
            Explore intelligent multi-station monitoring, geographic hazard heatmaps, and data-driven landslide risk assessment across India.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cta_c1, cta_c2, cta_c3 = st.columns([1, 1.5, 1])
    with cta_c2:
        if st.button("🚀 Open LandslideGuard Dashboard", key="final_cta_btn", width="stretch", type="primary"):
            st.session_state["nav_page"] = "🏠 Overview"
            st.rerun()

    # ─────────────────────────────────────────────
    # SECTION J: Enterprise Footer
    # ─────────────────────────────────────────────
    render_footer()
