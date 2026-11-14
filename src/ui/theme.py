"""
LandslideGuard — Design System & Theme Engine (Premium Edition).

Provides:
- Design tokens (colors, gradients, typography, shadows, borders).
- Scoped custom CSS injection for Streamlit dark-mode SaaS styling.
- Consistent Plotly figure formatting with design token alignment.
- Reusable UI helper functions.
"""

from __future__ import annotations

from typing import Any
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# 1. Design Tokens (Premium Dark Alpine Intelligence)
# ─────────────────────────────────────────────

# Backgrounds
BG_DARK = "#060C19"
BG_SECONDARY = "#0A1427"
BG_SIDEBAR = "#091326"
BG_NAV = "#0A1427"
BG_CARD = "#101E34"
BG_CARD_ELEVATED = "#152641"
BG_CARD_HOVER = "#152641"

# Primary & Accents
PRIMARY_BLUE = "#2196F3"
GRADIENT_BLUE = "#2196F3"
GRADIENT_VIOLET = "#8B5CF6"
CYAN = "#38BDF8"
VIOLET = "#8B5CF6"
GREEN = "#22C55E"

# Typography Colors
TEXT_MAIN = "#F8FAFC"
TEXT_MUTED = "#A8B6CC"
TEXT_DIM = "#64748B"

# Borders & Accents
BORDER_DEFAULT = "#263A59"
BORDER_GLOW = "rgba(33, 150, 243, 0.4)"
BORDER_FOCUS = "#2196F3"

# Risk Indicator Colors
RISK_LOW = "#22C55E"
RISK_MODERATE = "#FACC15"
RISK_HIGH = "#F97316"
RISK_CRITICAL = "#EF4444"
RISK_UNAVAILABLE = "#64748B"

# High-resolution mountain photography assets for the two-image hero and headers
HERO_IMG_LEFT = "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1600&q=80"
HERO_IMG_RIGHT = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1600&q=80"
WEATHER_BG_RAIN = "https://images.unsplash.com/photo-1515694346937-94d85e41e6f0?auto=format&fit=crop&w=800&q=80"


# ─────────────────────────────────────────────
# 2. Comprehensive CSS Injection
# ─────────────────────────────────────────────

def inject_theme_css() -> None:
    """Inject modern SaaS dark theme CSS into the Streamlit DOM."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Canvas */
    html, body, [class*="css"], .stMarkdown, .stText {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: {TEXT_MAIN};
    }}

    code, kbd, samp, pre {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    .stApp {{
        background: radial-gradient(circle at 85% 15%, #111e36 0%, {BG_SECONDARY} 40%, {BG_DARK} 100%);
        background-attachment: fixed;
    }}

    /* ─────────────────────────────────────────────
       1. Application & Viewport Container
       ───────────────────────────────────────────── */
    [data-testid="stAppViewContainer"],
    .stAppViewContainer {{
        height: 100vh !important;
        height: 100dvh !important;
        overflow: hidden !important;
    }}

    /* ─────────────────────────────────────────────
       2. Section 1: Streamlit Native Toolbar Header
       Pinned at the top with glassmorphic backdrop.
       Leaves interactive controls accessible and visible.
       ───────────────────────────────────────────── */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    .stAppHeader {{
        background: rgba(8, 14, 27, 0.95) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid {BORDER_DEFAULT} !important;
        height: 3.75rem !important;
        min-height: 3.75rem !important;
        z-index: 99999 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
    }}

    /* Native toolbar controls (Deploy button, running man, ⋮ menu) */
    [data-testid="stToolbar"] {{
        display: flex !important;
        align-items: center !important;
        margin-right: 1.25rem !important;
    }}

    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a,
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] a,
    [data-testid="stToolbar"],
    [data-testid="stHeaderActionElements"] {{
        pointer-events: auto !important;
    }}

    /* ─────────────────────────────────────────────
       3. Section 2: Main Scrolling Container (stMain)
       Takes up the viewport and scrolls smoothly.
       ───────────────────────────────────────────── */
    section[data-testid="stMain"],
    .stMain {{
        flex: 1 1 auto !important;
        height: 100vh !important;
        height: 100dvh !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        position: relative !important;
    }}

    /* ─────────────────────────────────────────────
       4. Sections 3, 4, 5: Main Content Block Container
       Generous 6rem (96px) top padding ensures that custom
       navigation bars, logos, brand titles, and breadcrumbs
       start safely 36px below the 60px Streamlit header.
       ───────────────────────────────────────────── */
    div[data-testid="stMainBlockContainer"],
    .stMainBlockContainer,
    .block-container {{
        padding-top: 6rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2.25rem !important;
        padding-right: 2.25rem !important;
        max-width: 1460px !important;
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        box-sizing: border-box !important;
    }}

    @media (max-width: 1024px) {{
        div[data-testid="stMainBlockContainer"],
        .stMainBlockContainer,
        .block-container {{
            padding-top: 6rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }}
    }}

    @media (max-width: 640px) {{
        div[data-testid="stMainBlockContainer"],
        .stMainBlockContainer,
        .block-container {{
            padding-top: 6.25rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
    }}

    /* ─────────────────────────────────────────────
       5. Sidebar Alignment
       ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {BG_SIDEBAR} !important;
        border-right: 1px solid {BORDER_DEFAULT} !important;
        height: 100vh !important;
        height: 100dvh !important;
    }}

    [data-testid="stSidebarHeader"] {{
        background: transparent !important;
        padding-top: 0.75rem !important;
        padding-bottom: 0.5rem !important;
    }}

    [data-testid="stSidebarUserContent"] {{
        padding-top: 0.25rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] {{
        gap: 0.35rem;
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] label {{
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.85rem !important;
        margin: 0 !important;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: rgba(56, 189, 248, 0.08) !important;
        border-color: rgba(56, 189, 248, 0.25) !important;
        transform: translateX(3px);
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.18) 0%, rgba(37, 99, 235, 0.08) 100%) !important;
        border-color: rgba(56, 189, 248, 0.45) !important;
        box-shadow: inset 3px 0 0 {PRIMARY_BLUE}, 0 2px 10px rgba(56, 189, 248, 0.12) !important;
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] label span {{
        color: {TEXT_MAIN} !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }}

    /* ─────────────────────────────────────────────
       6. Landing Page Navbar Component
       Structured flex container with explicit alignment,
       contained icon, safe line-height, and responsive wrapping.
       ───────────────────────────────────────────── */
    .lg-landing-navbar {{
        min-height: 72px;
        box-sizing: border-box;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        background: rgba(21, 34, 56, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 16px;
        padding: 0.85rem 1.75rem;
        margin-top: 0;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: visible;
    }}

    .lg-navbar-brand {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin: 0;
        padding: 0;
    }}

    .lg-brand-icon {{
        width: 44px;
        height: 44px;
        min-width: 44px;
        min-height: 44px;
        border-radius: 10px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        line-height: 1;
        flex-shrink: 0;
        box-sizing: border-box;
        overflow: hidden;
    }}

    .lg-brand-text {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        margin: 0;
        padding: 0;
    }}

    .lg-brand-title {{
        font-size: 1.25rem;
        font-weight: 800;
        line-height: 1.3;
        letter-spacing: -0.02em;
        color: {TEXT_MAIN};
        margin: 0;
        padding: 0;
        display: block;
    }}

    .lg-brand-title span {{
        color: {PRIMARY_BLUE};
    }}

    .lg-brand-subtitle {{
        font-size: 0.70rem;
        font-weight: 700;
        color: {TEXT_MUTED};
        letter-spacing: 0.06em;
        text-transform: uppercase;
        line-height: 1.2;
        margin-top: 2px;
        margin-bottom: 0;
        padding: 0;
        display: block;
    }}

    .lg-navbar-links {{
        display: flex;
        align-items: center;
        gap: 1.6rem;
        font-size: 0.88rem;
        font-weight: 600;
        color: {TEXT_MUTED};
        line-height: 1;
        margin: 0;
        padding: 0;
    }}

    .lg-nav-active {{
        color: {PRIMARY_BLUE};
        font-weight: 700;
    }}

    .lg-navbar-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.45rem 0.95rem;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: {PRIMARY_BLUE};
        line-height: 1;
        box-sizing: border-box;
        white-space: nowrap;
    }}

    @media (max-width: 900px) {{
        .lg-landing-navbar {{
            padding: 0.75rem 1.25rem;
            gap: 0.75rem;
        }}
        .lg-navbar-links {{
            display: none;
        }}
    }}

    /* ─────────────────────────────────────────────
       7. Top Navigation Bar (Section 3 Match)
       ───────────────────────────────────────────── */
    .lg-topbar {{
        min-height: 60px;
        box-sizing: border-box;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        background: {BG_NAV};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 14px;
        padding: 0.65rem 1.4rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        position: relative;
    }}

    .lg-topbar-left {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }}

    .lg-topbar-brand {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }}

    .lg-topbar-icon {{
        width: 36px;
        height: 36px;
        min-width: 36px;
        min-height: 36px;
        border-radius: 10px;
        background: rgba(33, 150, 243, 0.12);
        border: 1px solid rgba(33, 150, 243, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        line-height: 1;
        flex-shrink: 0;
        box-sizing: border-box;
        overflow: hidden;
    }}

    .lg-topbar-text {{
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .lg-topbar-title {{
        font-size: 1.15rem;
        font-weight: 800;
        color: {TEXT_MAIN};
        line-height: 1.25;
        margin: 0;
        padding: 0;
    }}

    .lg-topbar-title span {{
        color: {PRIMARY_BLUE};
    }}

    .lg-topbar-sub {{
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: {TEXT_MUTED};
        line-height: 1.1;
        margin-top: 2px;
    }}

    .lg-collapse-btn {{
        background: rgba(6, 12, 25, 0.6);
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 8px;
        color: {TEXT_MUTED};
        padding: 0.3rem 0.55rem;
        font-size: 0.9rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .lg-topbar-center {{
        display: flex;
        align-items: center;
        gap: 1.4rem;
        font-size: 0.86rem;
        font-weight: 600;
        color: {TEXT_MUTED};
    }}

    .lg-topbar-navitem {{
        color: {TEXT_MUTED};
        cursor: pointer;
        transition: color 0.18s ease;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }}

    .lg-topbar-navitem:hover,
    .lg-topbar-navitem.active {{
        color: {PRIMARY_BLUE};
        font-weight: 700;
    }}

    .lg-topbar-right {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }}

    .lg-topbar-search {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        background: rgba(6, 12, 25, 0.7);
        padding: 0.35rem 0.85rem;
        border-radius: 8px;
        border: 1px solid {BORDER_DEFAULT};
        min-width: 170px;
    }}

    .lg-topbar-action-icon {{
        position: relative;
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: rgba(6, 12, 25, 0.6);
        border: 1px solid {BORDER_DEFAULT};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        color: {TEXT_MAIN};
        cursor: pointer;
    }}

    .lg-badge-unread {{
        position: absolute;
        top: -4px;
        right: -4px;
        background: {RISK_CRITICAL};
        color: #ffffff;
        font-size: 0.65rem;
        font-weight: 800;
        border-radius: 9999px;
        padding: 0.1rem 0.35rem;
        line-height: 1;
        box-shadow: 0 0 6px {RISK_CRITICAL};
    }}

    .lg-user-profile {{
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, {PRIMARY_BLUE} 0%, {GRADIENT_VIOLET} 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }}

    @media (max-width: 1100px) {{
        .lg-topbar-center {{
            display: none;
        }}
    }}

    @media (max-width: 768px) {{
        .lg-topbar {{
            padding: 0.65rem 1rem;
            gap: 0.65rem;
        }}
        .lg-topbar-search {{
            display: none;
        }}
    }}

    .lg-status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {RISK_LOW};
        box-shadow: 0 0 8px {RISK_LOW};
        animation: lg-pulse 2s infinite;
    }}

    @keyframes lg-pulse {{
        0% {{ opacity: 0.6; transform: scale(0.95); }}
        50% {{ opacity: 1; transform: scale(1.1); }}
        100% {{ opacity: 0.6; transform: scale(0.95); }}
    }}

    /* ─────────────────────────────────────────────
       Mission Control Header Component (Section 5)
       ───────────────────────────────────────────── */
    .lg-mission-header {{
        position: relative;
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, rgba(10, 20, 39, 0.88) 0%, rgba(6, 12, 25, 0.94) 100%),
                    url('https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-position: center 30%;
        border: 1px solid {BORDER_DEFAULT};
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.45);
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.5rem;
        flex-wrap: wrap;
    }}

    .lg-mission-header-left {{
        max-width: 650px;
    }}

    .lg-mission-eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {CYAN};
        margin-bottom: 0.4rem;
    }}

    .lg-mission-title {{
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: {TEXT_MAIN};
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }}

    .lg-gradient-text {{
        background: linear-gradient(135deg, {PRIMARY_BLUE} 0%, {GRADIENT_VIOLET} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .lg-mission-subtitle {{
        font-size: 0.88rem;
        color: {TEXT_MUTED};
        line-height: 1.5;
        margin: 0;
    }}

    .lg-mission-header-right {{
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.6rem;
    }}

    .lg-telemetry-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: {RISK_LOW};
    }}

    .lg-sync-time {{
        font-size: 0.76rem;
        color: {TEXT_MUTED};
        font-weight: 500;
    }}

    /* ─────────────────────────────────────────────
       5 KPI Cards Component (Section 6)
       ───────────────────────────────────────────── */
    .lg-kpi-card {{
        background: linear-gradient(145deg, {BG_CARD} 0%, {BG_CARD_ELEVATED} 100%);
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        min-height: 155px;
    }}

    .lg-kpi-card:hover {{
        transform: translateY(-2px);
        border-color: {BORDER_GLOW};
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4), 0 0 16px rgba(33, 150, 243, 0.12);
    }}

    .lg-kpi-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }}

    .lg-kpi-icon-box {{
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
    }}

    .lg-kpi-title {{
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: {TEXT_MUTED};
    }}

    .lg-kpi-val {{
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {TEXT_MAIN};
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }}

    .lg-kpi-trend {{
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }}

    .lg-kpi-sparkline {{
        margin-top: 0.5rem;
        width: 100%;
        height: 24px;
        overflow: hidden;
    }}

    /* ─────────────────────────────────────────────
       Weather Card (Section 9)
       ───────────────────────────────────────────── */
    .lg-weather-card {{
        background: linear-gradient(180deg, rgba(16, 30, 52, 0.88) 0%, rgba(10, 20, 39, 0.95) 100%),
                    url('https://images.unsplash.com/photo-1515694346937-94d85e41e6f0?auto=format&fit=crop&w=800&q=80');
        background-size: cover;
        background-position: center;
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 16px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 1rem;
    }}

    .lg-weather-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }}

    .lg-weather-title {{
        font-size: 1.05rem;
        font-weight: 800;
        color: {TEXT_MAIN};
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .lg-weather-location {{
        font-size: 0.82rem;
        color: {CYAN};
        font-weight: 600;
    }}

    .lg-weather-temp {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {TEXT_MAIN};
        line-height: 1;
        margin: 0.4rem 0 0.85rem 0;
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
    }}

    .lg-weather-condition {{
        font-size: 0.88rem;
        color: {TEXT_MUTED};
        font-weight: 600;
    }}

    .lg-weather-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-bottom: 0.85rem;
    }}

    .lg-weather-stat-item {{
        background: rgba(6, 12, 25, 0.6);
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 8px;
        padding: 0.45rem 0.65rem;
    }}

    .lg-weather-stat-label {{
        font-size: 0.70rem;
        color: {TEXT_MUTED};
        margin-bottom: 0.15rem;
    }}

    .lg-weather-stat-val {{
        font-size: 0.85rem;
        font-weight: 700;
        color: {TEXT_MAIN};
    }}

    .lg-weather-forecast {{
        background: rgba(33, 150, 243, 0.08);
        border: 1px solid rgba(33, 150, 243, 0.25);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        font-size: 0.78rem;
        color: {CYAN};
        line-height: 1.4;
        margin-bottom: 0.75rem;
    }}

    .lg-weather-footer {{
        font-size: 0.72rem;
        color: {TEXT_DIM};
        border-top: 1px solid rgba(38, 58, 89, 0.6);
        padding-top: 0.6rem;
    }}

    /* ─────────────────────────────────────────────
       Recent Alerts Card (Section 10)
       ───────────────────────────────────────────── */
    .lg-alerts-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 16px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
        margin-bottom: 1rem;
    }}

    .lg-alerts-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.85rem;
    }}

    .lg-alerts-title {{
        font-size: 1.05rem;
        font-weight: 800;
        color: {TEXT_MAIN};
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .lg-alerts-viewall {{
        font-size: 0.78rem;
        color: {PRIMARY_BLUE};
        font-weight: 700;
        text-decoration: none;
        cursor: pointer;
    }}

    .lg-alert-item {{
        background: rgba(6, 12, 25, 0.55);
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        transition: background 0.18s ease;
    }}

    .lg-alert-item:hover {{
        background: rgba(16, 30, 52, 0.8);
        border-color: {BORDER_GLOW};
    }}

    .lg-alert-item-left {{
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }}

    .lg-alert-item-title {{
        font-size: 0.82rem;
        font-weight: 700;
        color: {TEXT_MAIN};
    }}

    .lg-alert-item-desc {{
        font-size: 0.74rem;
        color: {TEXT_MUTED};
        line-height: 1.3;
    }}

    .lg-alert-item-right {{
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.25rem;
    }}

    .lg-alert-item-time {{
        font-size: 0.70rem;
        color: {TEXT_DIM};
    }}

    /* ─────────────────────────────────────────────
       Map Card & Floating Legend (Section 8)
       ───────────────────────────────────────────── */
    .lg-map-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
        position: relative;
        margin-bottom: 1rem;
    }}

    .lg-map-card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }}

    .lg-map-title {{
        font-size: 1.1rem;
        font-weight: 800;
        color: {TEXT_MAIN};
    }}

    .lg-map-subtitle {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
    }}

    .lg-map-controls-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
        background: rgba(6, 12, 25, 0.6);
        border: 1px solid {BORDER_DEFAULT};
        font-size: 0.75rem;
        color: {CYAN};
        font-weight: 600;
    }}

    .lg-risk-legend-floating {{
        position: absolute;
        bottom: 25px;
        right: 25px;
        background: rgba(6, 12, 25, 0.92);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 10px;
        padding: 0.65rem 0.95rem;
        z-index: 999;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }}

    .lg-risk-legend-item {{
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.74rem;
        color: {TEXT_MAIN};
        font-weight: 600;
    }}

    .lg-risk-legend-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }}

    /* ─────────────────────────────────────────────
       Analytics Cards (Section 11)
       ───────────────────────────────────────────── */
    .lg-analytics-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    .lg-analytics-title {{
        font-size: 0.95rem;
        font-weight: 800;
        color: {TEXT_MAIN};
        margin-bottom: 0.2rem;
    }}

    .lg-analytics-sub {{
        font-size: 0.76rem;
        color: {TEXT_MUTED};
        margin-bottom: 0.6rem;
    }}

    /* Card Containers */
    .lg-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    }}

    .lg-card:hover {{
        border-color: {BORDER_GLOW};
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35), 0 0 15px rgba(56, 189, 248, 0.08);
    }}

    /* Two-Image Gradient Hero Section */
    .lg-hero-wrapper {{
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-bottom: 2rem;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.45);
    }}

    .lg-hero-bg-container {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        height: 480px;
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 1;
        overflow: hidden;
    }}

    .lg-hero-img-col {{
        position: relative;
        height: 100%;
        overflow: hidden;
    }}

    .lg-hero-img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    .lg-hero-wrapper:hover .lg-hero-img {{
        transform: scale(1.04);
    }}

    /* Dark Blue to Violet Cinematic Gradient Overlay */
    .lg-hero-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, 
            rgba(8, 14, 27, 0.94) 0%, 
            rgba(13, 23, 40, 0.82) 35%, 
            rgba(37, 99, 235, 0.45) 70%, 
            rgba(124, 58, 237, 0.45) 100%
        );
        backdrop-filter: blur(2px);
        -webkit-backdrop-filter: blur(2px);
        z-index: 2;
    }}

    .lg-hero-content {{
        position: relative;
        z-index: 3;
        padding: 3.5rem 3rem;
        max-width: 820px;
    }}

    .lg-hero-eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: {PRIMARY_BLUE};
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1.1rem;
    }}

    .lg-hero-title {{
        font-size: 2.85rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.03em;
        color: {TEXT_MAIN};
        margin: 0 0 1rem 0;
    }}

    .lg-hero-title span {{
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 60%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .lg-hero-desc {{
        font-size: 1.08rem;
        line-height: 1.6;
        color: {TEXT_MUTED};
        margin-bottom: 1.8rem;
        max-width: 680px;
    }}

    .lg-hero-badges {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.75rem;
    }}

    .lg-hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8rem;
        color: #CBD5E1;
        background: rgba(19, 31, 50, 0.7);
        border: 1px solid {BORDER_DEFAULT};
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
    }}

    /* Stat Cards */
    .lg-stat-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .lg-stat-card:hover {{
        border-color: {BORDER_GLOW};
        transform: translateY(-2px);
    }}

    .lg-stat-label {{
        font-size: 0.82rem;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }}

    .lg-stat-val {{
        font-size: 1.95rem;
        font-weight: 800;
        color: {TEXT_MAIN};
        line-height: 1.1;
    }}

    .lg-stat-sub {{
        font-size: 0.76rem;
        color: {TEXT_DIM};
        margin-top: 0.35rem;
    }}

    /* Feature Grid Cards */
    .lg-feature-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 14px;
        padding: 1.5rem 1.35rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    }}

    .lg-feature-card:hover {{
        border-color: {BORDER_GLOW};
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3), 0 0 12px rgba(56, 189, 248, 0.12);
        transform: translateY(-3px);
    }}

    .lg-feature-icon {{
        font-size: 1.75rem;
        margin-bottom: 0.8rem;
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .lg-feature-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {TEXT_MAIN};
        margin-bottom: 0.45rem;
    }}

    .lg-feature-desc {{
        font-size: 0.85rem;
        line-height: 1.55;
        color: {TEXT_MUTED};
        flex-grow: 1;
        margin-bottom: 1rem;
    }}

    /* Process Flow Step Cards */
    .lg-step-wrapper {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
        position: relative;
    }}

    .lg-step-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 14px;
        padding: 1.25rem 1.15rem;
        position: relative;
        transition: border-color 0.2s ease;
    }}

    .lg-step-card:hover {{
        border-color: {BORDER_GLOW};
    }}

    .lg-step-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: {GRADIENT_BLUE};
        color: #fff;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }}

    /* Streamlit Metric Overrides */
    [data-testid="stMetric"],
    [data-testid="metric-container"] {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER_DEFAULT} !important;
        border-radius: 12px !important;
        padding: 0.9rem 1.1rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.22) !important;
        transition: border-color 0.2s ease, transform 0.2s ease !important;
    }}

    [data-testid="stMetric"]:hover,
    [data-testid="metric-container"]:hover {{
        border-color: {BORDER_GLOW} !important;
        transform: translateY(-2px) !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {TEXT_MAIN} !important;
        font-weight: 700 !important;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
        color: {TEXT_MAIN} !important;
        border: 1px solid {BORDER_DEFAULT} !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.1rem !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        border-color: {PRIMARY_BLUE} !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
        transform: translateY(-1px) !important;
    }}

    .stButton > button[kind="primary"],
    .stButton > button:focus:not(:focus-visible) {{
        background: linear-gradient(135deg, {GRADIENT_BLUE} 0%, {GRADIENT_VIOLET} 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35) !important;
    }}

    /* Segmented Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem !important;
        background: {BG_SECONDARY} !important;
        padding: 0.35rem 0.5rem !important;
        border-radius: 12px !important;
        border: 1px solid {BORDER_DEFAULT} !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: {TEXT_MUTED} !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {TEXT_MAIN} !important;
        background: rgba(56, 189, 248, 0.08) !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: {BG_CARD} !important;
        color: {PRIMARY_BLUE} !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
    }}

    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* Inputs, Number Inputs, Selectboxes */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER_DEFAULT} !important;
        border-radius: 10px !important;
        color: {TEXT_MAIN} !important;
        font-size: 0.88rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stSelectbox div[data-baseweb="select"]:focus-within > div {{
        border-color: {PRIMARY_BLUE} !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
    }}

    /* File Uploader */
    [data-testid="stFileUploaderDropzone"] {{
        background: {BG_CARD} !important;
        border: 2px dashed rgba(56, 189, 248, 0.35) !important;
        border-radius: 14px !important;
        padding: 1.5rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }}

    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {PRIMARY_BLUE} !important;
        background: rgba(56, 189, 248, 0.06) !important;
    }}

    /* Dataframes & Tables */
    [data-testid="stDataFrame"],
    .stDataFrame {{
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid {BORDER_DEFAULT} !important;
        background: {BG_CARD} !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER_DEFAULT} !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        margin-bottom: 0.85rem !important;
    }}

    [data-testid="stExpander"] summary {{
        font-weight: 600 !important;
        color: {TEXT_MAIN} !important;
        padding: 0.75rem 1rem !important;
    }}

    [data-testid="stExpander"] summary:hover {{
        color: {PRIMARY_BLUE} !important;
    }}

    /* Chat Messages for AI Assistant */
    [data-testid="stChatMessage"] {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER_DEFAULT} !important;
        border-radius: 14px !important;
        padding: 0.9rem 1.2rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background: rgba(37, 99, 235, 0.12) !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
    }}

    [data-testid="stChatInput"] {{
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        background: {BG_SECONDARY} !important;
    }}

    /* Forms */
    [data-testid="stForm"] {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER_DEFAULT} !important;
        border-radius: 14px !important;
        padding: 1.4rem !important;
    }}

    /* Footer */
    .lg-footer {{
        background: {BG_SECONDARY};
        border-top: 1px solid {BORDER_DEFAULT};
        border-radius: 16px 16px 0 0;
        padding: 2.5rem 2rem 1.5rem 2rem;
        margin-top: 3.5rem;
    }}

    .lg-footer-grid {{
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        gap: 2rem;
        margin-bottom: 2rem;
    }}

    .lg-footer-title {{
        font-size: 0.92rem;
        font-weight: 700;
        color: {TEXT_MAIN};
        margin-bottom: 0.8rem;
        letter-spacing: 0.02em;
    }}

    .lg-footer-links {{
        list-style: none;
        padding: 0;
        margin: 0;
        font-size: 0.84rem;
        color: {TEXT_MUTED};
        line-height: 2;
    }}

    .lg-footer-disclaimer {{
        border-top: 1px solid {BORDER_DEFAULT};
        padding-top: 1.25rem;
        font-size: 0.76rem;
        color: {TEXT_DIM};
        line-height: 1.5;
        text-align: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. Plotly Theme Styler
# ─────────────────────────────────────────────

def style_fig(fig: go.Figure, height: int = 320, showlegend: bool | None = None) -> go.Figure:
    """Format Plotly charts to strictly conform with LandslideGuard's dark SaaS palette."""
    layout_kwargs: dict[str, Any] = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, family="Plus Jakarta Sans, sans-serif", size=12),
        height=height,
        margin=dict(t=36, b=24, l=20, r=20),
        legend=dict(bgcolor="rgba(16, 30, 52, 0.8)", font=dict(color=TEXT_MAIN)),
    )
    if showlegend is not None:
        layout_kwargs["showlegend"] = showlegend
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        gridcolor="rgba(38, 58, 89, 0.45)",
        zerolinecolor="rgba(38, 58, 89, 0.65)",
        tickfont=dict(color=TEXT_MUTED),
    )
    fig.update_yaxes(
        gridcolor="rgba(38, 58, 89, 0.45)",
        zerolinecolor="rgba(38, 58, 89, 0.65)",
        tickfont=dict(color=TEXT_MUTED),
    )
    return fig

