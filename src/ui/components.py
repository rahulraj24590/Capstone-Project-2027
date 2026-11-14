"""
LandslideGuard — Reusable UI Component System (Premium Edition).

Provides standardized, accessible, and responsive components:
- Top Navigation Bar (breadcrumbs, active region, system status)
- Executive KPI Cards & Risk Indicators
- Alert Badges & Provenance Chips
- Section Headings & Action Bars
- Enterprise Footer
"""

from __future__ import annotations

import streamlit as st
from src.ui.theme import (
    BORDER_DEFAULT,
    PRIMARY_BLUE,
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MODERATE,
    RISK_UNAVAILABLE,
    TEXT_DIM,
    TEXT_MAIN,
    TEXT_MUTED,
)


def render_topbar(
    current_page: str,
    selected_region: str = "National Subcontinent (India)",
    is_live_sync: bool = True,
) -> None:
    """Render the top horizontal SaaS navigation bar strictly matching Section 3."""
    topbar_html = f"""
    <div class="lg-topbar">
        <div class="lg-topbar-left">
            <div class="lg-topbar-brand">
                <div class="lg-topbar-icon">⛰️</div>
                <div class="lg-topbar-text">
                    <div class="lg-topbar-title">Landslide<span>Guard</span></div>
                    <div class="lg-topbar-sub">ENVIRONMENTAL SAAS</div>
                </div>
            </div>
            <div class="lg-collapse-btn" title="Toggle Sidebar">☰</div>
        </div>
        <div class="lg-topbar-center">
            <span class="lg-topbar-navitem {'active' if 'Home' in current_page else ''}">Home</span>
            <span class="lg-topbar-navitem">Platform ▾</span>
            <span class="lg-topbar-navitem {'active' if 'Live' in current_page else ''}">Live Monitoring</span>
            <span class="lg-topbar-navitem {'active' if 'Map' in current_page else ''}">India Risk Map</span>
            <span class="lg-topbar-navitem {'active' if 'AI' in current_page or 'Model' in current_page else ''}">AI Intelligence</span>
            <span class="lg-topbar-navitem">About</span>
        </div>
        <div class="lg-topbar-right">
            <div class="lg-topbar-search">
                <span style="color:{PRIMARY_BLUE};">🔍</span> <span>Search location...</span>
            </div>
            <div class="lg-topbar-action-icon" title="Notifications">
                <span>🔔</span>
                <span class="lg-badge-unread">3</span>
            </div>
            <div class="lg-topbar-action-icon" title="Toggle Theme">
                <span>🌙</span>
            </div>
            <div class="lg-user-profile" title="Administrator Profile">
                <span>👤</span>
            </div>
        </div>
    </div>
    """
    st.markdown(topbar_html, unsafe_allow_html=True)


def make_sparkline_svg(
    points: list[float],
    stroke_color: str = "#2196F3",
    height: int = 24,
    width: int = 120,
) -> str:
    """Generate responsive inline SVG sparkline curve with subtle gradient fill."""
    if not points or len(points) < 2:
        return ""
    min_y = min(points)
    max_y = max(points)
    span = (max_y - min_y) if (max_y - min_y) > 0 else 1.0
    pad = 3
    coords = []
    dx = width / (len(points) - 1)
    for i, p in enumerate(points):
        x = i * dx
        y = height - pad - ((p - min_y) / span) * (height - 2 * pad)
        coords.append(f"{x:.1f},{y:.1f}")
    polyline_pts = " ".join(coords)
    first_x, first_y = coords[0].split(",")
    last_x, last_y = coords[-1].split(",")
    area_path = f"M {first_x},{height} L {first_x},{first_y} " + " ".join([f"L {c}" for c in coords[1:]]) + f" L {last_x},{height} Z"
    grad_id = f"spark_grad_{abs(hash(polyline_pts)) % 100000}"
    return f"""
    <svg width="100%" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none" style="overflow:visible; display:block;">
        <defs>
            <linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="{stroke_color}" stop-opacity="0.32" />
                <stop offset="100%" stop-color="{stroke_color}" stop-opacity="0.0" />
            </linearGradient>
        </defs>
        <path d="{area_path}" fill="url(#{grad_id})" />
        <polyline fill="none" stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="{polyline_pts}" />
    </svg>
    """


def render_risk_badge(risk_category: str | None) -> str:
    """Generate HTML markup for a standardized risk category badge."""
    cat = (risk_category or "Unavailable").strip().title()
    if cat in ("Critical", "Very High"):
        color = RISK_CRITICAL
        bg = "rgba(239, 68, 68, 0.14)"
        border = "rgba(239, 68, 68, 0.35)"
        icon = "🔴"
    elif cat == "High":
        color = RISK_HIGH
        bg = "rgba(249, 115, 22, 0.14)"
        border = "rgba(249, 115, 22, 0.35)"
        icon = "🟠"
    elif cat in ("Moderate", "Medium"):
        color = RISK_MODERATE
        bg = "rgba(234, 179, 8, 0.14)"
        border = "rgba(234, 179, 8, 0.35)"
        icon = "🟡"
    elif cat == "Low":
        color = RISK_LOW
        bg = "rgba(34, 197, 94, 0.14)"
        border = "rgba(34, 197, 94, 0.35)"
        icon = "🟢"
    else:
        color = RISK_UNAVAILABLE
        bg = "rgba(100, 116, 139, 0.14)"
        border = "rgba(100, 116, 139, 0.35)"
        icon = "⚪"

    return f"""
    <span style="display:inline-flex; align-items:center; gap:0.35rem; padding:0.22rem 0.65rem; border-radius:9999px; font-size:0.75rem; font-weight:700; background:{bg}; border:1px solid {border}; color:{color};">
        <span>{icon}</span> {cat}
    </span>
    """


def render_page_header(
    title: str,
    subtitle: str = "",
    pills: list[tuple[str, str]] | None = None,
) -> None:
    """Render a consistent page header with subtitle and pill tags."""
    pills_html = ""
    if pills:
        chips = []
        for text, style in pills:
            if style == "pill-blue":
                c = PRIMARY_BLUE
                bg = "rgba(56, 189, 248, 0.12)"
                b = "rgba(56, 189, 248, 0.35)"
            elif style == "pill-amber":
                c = RISK_MODERATE
                bg = "rgba(234, 179, 8, 0.12)"
                b = "rgba(234, 179, 8, 0.35)"
            elif style == "pill-green":
                c = RISK_LOW
                bg = "rgba(34, 197, 94, 0.12)"
                b = "rgba(34, 197, 94, 0.35)"
            else:
                c = TEXT_MUTED
                bg = "rgba(148, 163, 184, 0.1)"
                b = BORDER_DEFAULT
            chips.append(
                f'<span style="display:inline-flex; align-items:center; padding:0.25rem 0.65rem; border-radius:9999px; font-size:0.75rem; font-weight:600; background:{bg}; border:1px solid {b}; color:{c};">{text}</span>'
            )
        pills_html = f'<div style="display:flex; flex-wrap:wrap; gap:0.45rem;">{"".join(chips)}</div>'

    sub_html = f'<div style="color:{TEXT_MUTED}; font-size:0.92rem; margin-top:0.25rem; line-height:1.5;">{subtitle}</div>' if subtitle else ""
    header_html = f"""
    <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-end; gap:0.75rem; margin-bottom:1.25rem; border-bottom:1px solid {BORDER_DEFAULT}; padding-bottom:0.85rem;">
        <div>
            <h1 style="font-size:1.85rem; font-weight:800; letter-spacing:-0.025em; color:{TEXT_MAIN}; margin:0;">{title}</h1>
            {sub_html}
        </div>
        {pills_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def experimental_banner(extra: str = "") -> None:
    """Render the academic research non-operational safety notice."""
    msg = (
        "⚠️ <b>ACADEMIC RESEARCH PLATFORM</b> — All risk estimates and early-warning alerts are experimental "
        "model outputs from a B.Tech capstone project. They have <b>NOT</b> been operationally certified by "
        "statutory geological authorities (GSI, IMD, NDMA) and must <b>NOT</b> be used for civil evacuation "
        "or life-safety emergency decisions. " + extra
    )
    banner_html = f"""
    <div style="background:linear-gradient(90deg, rgba(234, 179, 8, 0.12) 0%, rgba(180, 83, 9, 0.06) 100%); border:1px solid rgba(234, 179, 8, 0.35); border-left:4px solid {RISK_MODERATE}; border-radius:10px; padding:0.75rem 1.1rem; margin:0.4rem 0 1.25rem 0; color:#FDE68A; font-size:0.84rem; line-height:1.5;">
        {msg}
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    delta: str = "",
    delta_color: str = "normal",
    icon: str = "",
) -> None:
    """Render a modern SaaS KPI card with dark navy elevation."""
    delta_html = ""
    if delta:
        d_color = RISK_LOW if delta_color in ("normal", "green") else (RISK_CRITICAL if delta_color in ("inverse", "red") else TEXT_MUTED)
        delta_html = f'<div style="font-size:0.75rem; color:{d_color}; margin-top:0.35rem; font-weight:600;">{delta}</div>'
    sub_html = f'<div style="font-size:0.75rem; color:{TEXT_DIM}; margin-top:0.25rem;">{subtext}</div>' if subtext else ""
    icon_html = f'<div style="font-size:1.4rem; opacity:0.85;">{icon}</div>' if icon else ""

    html = f"""
    <div class="lg-stat-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div class="lg-stat-label">{label}</div>
            {icon_html}
        </div>
        <div class="lg-stat-val">{value}</div>
        {delta_html}
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_empty_state(title: str, message: str, icon: str = "🔍") -> None:
    """Render a sleek empty-state card when data or search results are not found."""
    html = f"""
    <div class="lg-card" style="text-align:center; padding:2.5rem 1.5rem;">
        <div style="font-size:2.5rem; margin-bottom:0.75rem;">{icon}</div>
        <div style="font-size:1.1rem; font-weight:700; color:{TEXT_MAIN}; margin-bottom:0.4rem;">{title}</div>
        <div style="font-size:0.88rem; color:{TEXT_MUTED}; max-width:480px; margin:0 auto;">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_footer() -> None:
    """Render the enterprise footer across the website."""
    footer_html = f"""
    <div class="lg-footer">
        <div class="lg-footer-grid">
            <div>
                <div style="font-size:1.15rem; font-weight:800; color:{TEXT_MAIN}; margin-bottom:0.6rem;">
                    ⛰️ LandslideGuard
                </div>
                <div style="font-size:0.84rem; color:{TEXT_MUTED}; line-height:1.6; max-width:320px;">
                    AI-powered environmental intelligence platform combining static terrain susceptibility with dynamic meteorological forecasting and explainable machine learning.
                </div>
            </div>
            <div>
                <div class="lg-footer-title">Platform Capabilities</div>
                <ul class="lg-footer-links">
                    <li>Dynamic Risk Assessment</li>
                    <li>Geospatial Folium Mapping</li>
                    <li>Rainfall Accumulation (1h/24h/72h)</li>
                    <li>SHAP Model Explainability</li>
                    <li>State & District Telemetry</li>
                </ul>
            </div>
            <div>
                <div class="lg-footer-title">Verified Data Sources</div>
                <ul class="lg-footer-links">
                    <li>Open-Meteo High-Resolution (CC BY 4.0)</li>
                    <li>NASA Global Landslide Catalog</li>
                    <li>Copernicus & SRTM 30m DEM</li>
                    <li>OpenStreetMap & OSRM Engine</li>
                    <li>NDMA SACHET Disaster Feeds</li>
                </ul>
            </div>
            <div>
                <div class="lg-footer-title">Academic Governance</div>
                <ul class="lg-footer-links">
                    <li>B.Tech CSE Capstone 2025–26</li>
                    <li>Zero Silent Imputation Invariant</li>
                    <li>Strict Data Segregation Invariant</li>
                    <li>Open-Source MIT License</li>
                    <li>Documentation & Viva Guide</li>
                </ul>
            </div>
        </div>
        <div class="lg-footer-disclaimer">
            <b>NON-OPERATIONAL RESEARCH NOTICE:</b> LandslideGuard is an academic computer science capstone project. In India, official landslide advisories are issued exclusively by the Geological Survey of India (GSI), National Disaster Management Authority (NDMA), and India Meteorological Department (IMD).
            <div style="margin-top:0.4rem; color:{TEXT_DIM};">
                © 2026 LandslideGuard Environmental Intelligence Platform. All rights reserved.
            </div>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

