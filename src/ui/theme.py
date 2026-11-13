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
# 1. Design Tokens
# ─────────────────────────────────────────────

# Backgrounds
BG_DARK = "#080E1B"
BG_SECONDARY = "#0D1728"
BG_SIDEBAR = "#101A2B"
BG_CARD = "#131F32"
BG_CARD_HOVER = "#18263E"

# Primary & Gradients
PRIMARY_BLUE = "#38BDF8"
GRADIENT_BLUE = "#2563EB"
GRADIENT_VIOLET = "#7C3AED"
CYAN = "#22D3EE"

# Typography Colors
TEXT_MAIN = "#F8FAFC"
TEXT_MUTED = "#94A3B8"
TEXT_DIM = "#64748B"

# Borders & Accents
BORDER_DEFAULT = "rgba(148, 163, 184, 0.15)"
BORDER_GLOW = "rgba(56, 189, 248, 0.35)"
BORDER_FOCUS = "rgba(56, 189, 248, 0.65)"

# Risk Indicator Colors
RISK_LOW = "#22C55E"
RISK_MODERATE = "#EAB308"
RISK_HIGH = "#F97316"
RISK_CRITICAL = "#EF4444"
RISK_UNAVAILABLE = "#64748B"

# High-resolution mountain photography assets for the two-image hero
HERO_IMG_LEFT = "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1600&q=80"
HERO_IMG_RIGHT = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1600&q=80"


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
       Streamlit Top Header & Toolbar Configuration
       Streamlit's native header is 3.75rem (60px) high and positioned absolutely.
       Making the background transparent with pointer-events: none ensures it never
       blocks clicks or casts an opaque dark strip over application content, while
       re-enabling pointer-events on buttons preserves Deploy and menu interactions.
       ───────────────────────────────────────────── */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    .stAppHeader {{
        background: transparent !important;
        height: 3.75rem !important;
        min-height: 3.75rem !important;
        pointer-events: none !important;
        z-index: 99999 !important;
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
       Main Content Container — Robust Global Spacing
       Streamlit's native header height is 3.75rem (60px). Setting padding-top
       to 6rem (96px) guarantees a clean, unclipped 2.25rem (36px) margin below
       the toolbar across all Windows display scaling factors (100%, 125%, 150%)
       and prevents any cutoff of topbars, breadcrumbs, or headings.
       ───────────────────────────────────────────── */
    div[data-testid="stMainBlockContainer"],
    .stMainBlockContainer,
    .block-container {{
        padding-top: 6rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2.25rem !important;
        padding-right: 2.25rem !important;
        max-width: 1460px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}

    @media (max-width: 1024px) {{
        div[data-testid="stMainBlockContainer"],
        .stMainBlockContainer,
        .block-container {{
            padding-top: 6.25rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }}
    }}

    @media (max-width: 640px) {{
        div[data-testid="stMainBlockContainer"],
        .stMainBlockContainer,
        .block-container {{
            padding-top: 6.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
    }}

    /* ─────────────────────────────────────────────
       Sidebar Layout & Vertical Alignment
       ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {BG_SIDEBAR} !important;
        border-right: 1px solid {BORDER_DEFAULT} !important;
    }}

    [data-testid="stSidebarHeader"] {{
        background: transparent !important;
        padding-top: 0.5rem !important;
    }}

    [data-testid="stSidebarUserContent"] {{
        padding-top: 0.4rem !important;
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

    /* Top Navigation Bar Component */
    .lg-topbar {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.85rem;
        background: rgba(19, 31, 50, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {BORDER_DEFAULT};
        border-radius: 14px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }}

    .lg-topbar-left {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.85rem;
    }}

    .lg-breadcrumb {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
        color: {TEXT_MUTED};
    }}

    .lg-breadcrumb-active {{
        color: {PRIMARY_BLUE};
        font-weight: 600;
    }}

    .lg-topbar-right {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.75rem;
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
        legend=dict(bgcolor="rgba(19, 31, 50, 0.7)", font=dict(color=TEXT_MAIN)),
    )
    if showlegend is not None:
        layout_kwargs["showlegend"] = showlegend
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.10)",
        zerolinecolor="rgba(148, 163, 184, 0.18)",
        tickfont=dict(color=TEXT_MUTED),
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.10)",
        zerolinecolor="rgba(148, 163, 184, 0.18)",
        tickfont=dict(color=TEXT_MUTED),
    )
    return fig

