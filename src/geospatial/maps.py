"""
Geospatial map generation for LandslideGuard.

Creates interactive Folium maps for:
- Historical landslide observations
- Risk estimates per location
- Risk category overlays

All map layers are clearly attributed.
Coordinates are validated before display.
"""

from __future__ import annotations

from typing import Any

import folium
import pandas as pd
from folium.plugins import MarkerCluster

from src.risk.thresholds import RISK_COLORS, RISK_EMOJI
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting
from src.utils.validation import validate_coordinates

logger = get_logger(__name__)

MAP_CFG = get_setting("map") or {}
DEFAULT_LAT = float(MAP_CFG.get("default_lat", 22.5))
DEFAULT_LON = float(MAP_CFG.get("default_lon", 82.0))
DEFAULT_ZOOM = int(MAP_CFG.get("default_zoom", 5))


# ─────────────────────────────────────────────
# Base map factory
# ─────────────────────────────────────────────

def create_base_map(
    center_lat: float = DEFAULT_LAT,
    center_lon: float = DEFAULT_LON,
    zoom: int = DEFAULT_ZOOM,
) -> folium.Map:
    """Create a base Folium map centred on India.

    Args:
        center_lat: Map centre latitude.
        center_lon: Map centre longitude.
        zoom: Initial zoom level.

    Returns:
        Folium Map object.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
    )

    # Primary tile layer — OpenStreetMap (free, no API key required)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    ).add_to(m)

    # Satellite imagery fallback
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Esri Satellite",
        attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m



# ─────────────────────────────────────────────
# Observation markers
# ─────────────────────────────────────────────

def add_observation_markers(
    m: folium.Map,
    df: pd.DataFrame,
    cluster: bool = True,
) -> folium.Map:
    """Add historical landslide observation markers to the map.

    Args:
        m: Folium Map to add markers to.
        df: DataFrame with 'latitude', 'longitude', and optional metadata columns.
        cluster: If True, use MarkerCluster for dense datasets.

    Returns:
        Modified Folium Map.
    """
    if df.empty:
        logger.info("No observations to plot.")
        return m

    # Filter to valid coordinates
    valid = df[
        df["latitude"].between(-90, 90) &
        df["longitude"].between(-180, 180) &
        df["latitude"].notna() &
        df["longitude"].notna()
    ].copy()

    if valid.empty:
        logger.warning("No valid coordinates in observation DataFrame.")
        return m

    layer = MarkerCluster(name="Historical Observations") if cluster else folium.FeatureGroup(name="Historical Observations")

    for _, row in valid.iterrows():
        is_synthetic = bool(row.get("is_synthetic", False))
        label = "⚠️ SYNTHETIC DATA" if is_synthetic else "📍 Landslide Event"
        color = "gray" if is_synthetic else "red"
        icon = "info-sign" if is_synthetic else "warning-sign"

        popup_lines = [
            f"<b>{label}</b>",
            f"📅 Date: {row.get('event_date', 'Unknown')}",
            f"🌍 Location: ({row.get('latitude', 0):.4f}, {row.get('longitude', 0):.4f})",
        ]
        if row.get("country"):
            popup_lines.append(f"🗺️ Country: {row['country']}")
        if row.get("landslide_type"):
            popup_lines.append(f"🪨 Type: {row['landslide_type']}")
        if row.get("trigger"):
            popup_lines.append(f"💧 Trigger: {row['trigger']}")
        if row.get("fatalities") and pd.notna(row["fatalities"]):
            popup_lines.append(f"❗ Fatalities: {int(row['fatalities'])}")
        if row.get("source_name"):
            popup_lines.append(f"📊 Source: {row['source_name']}")
        if is_synthetic:
            popup_lines.append(
                "<br><i style='color:gray'>This is synthetic test data, "
                "not a real event.</i>"
            )

        popup_html = "<br>".join(popup_lines)

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{label} | {row.get('event_date', 'Unknown')}",
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
        ).add_to(layer)

    layer.add_to(m)
    logger.info("Added %d observation markers to map.", len(valid))
    return m


# ─────────────────────────────────────────────
# Risk estimate markers
# ─────────────────────────────────────────────

def add_risk_markers(
    m: folium.Map,
    risk_data: list[dict],
) -> folium.Map:
    """Add risk estimate markers to the map.

    Args:
        m: Folium Map.
        risk_data: List of risk estimate dicts from RiskEngine.

    Returns:
        Modified Folium Map.
    """
    layer = folium.FeatureGroup(name="Risk Estimates")

    for entry in risk_data:
        lat = entry.get("latitude") or (entry.get("features_used", {}) or {}).get("latitude")
        lon = entry.get("longitude") or (entry.get("features_used", {}) or {}).get("longitude")

        if lat is None or lon is None:
            continue

        result = validate_coordinates(lat, lon)
        if not result.is_valid:
            continue

        category = entry.get("risk_category", "Unknown")
        prob = entry.get("probability")
        method = entry.get("method", "unknown")
        is_experimental = entry.get("is_experimental", True)

        color_map = {
            "Low": "green",
            "Moderate": "orange",
            "High": "red",
            "Very High": "purple",
        }
        marker_color = color_map.get(category, "gray")
        emoji = RISK_EMOJI.get(category, "⚪")

        popup_lines = [
            f"<b>{emoji} Risk: {category}</b>",
            f"📍 ({float(lat):.4f}, {float(lon):.4f})",
        ]
        if prob is not None:
            popup_lines.append(f"📊 Probability: {prob:.3f}")
        popup_lines.append(f"🤖 Method: {method}")
        if entry.get("estimated_at"):
            popup_lines.append(f"🕐 Estimated: {entry['estimated_at'][:19]}")
        if is_experimental:
            popup_lines.append(
                "<br><i style='color:darkorange'><b>⚠️ EXPERIMENTAL ESTIMATE</b><br>"
                "Not validated. Not an official warning.</i>"
            )
        if entry.get("warnings"):
            popup_lines.append(f"<br>⚠️ Warnings: {len(entry['warnings'])}")

        popup_html = "<br>".join(popup_lines)

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=10,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{emoji} {category} | prob={prob:.3f}" if prob is not None else f"{emoji} {category}",
        ).add_to(layer)

    layer.add_to(m)
    return m


# ─────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────

def add_risk_legend(m: folium.Map) -> folium.Map:
    """Add a risk category legend to the map.

    Args:
        m: Folium Map.

    Returns:
        Modified Folium Map.
    """
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; right: 30px;
        background: rgba(30,30,30,0.92);
        color: #fff;
        padding: 14px 18px;
        border-radius: 10px;
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
        z-index: 9999;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        min-width: 180px;
    ">
        <b style="font-size:14px;">🗺️ Risk Legend</b><hr style="margin:6px 0;border-color:#555">
        <div><span style="color:#2ecc71;font-size:18px;">●</span> Low Risk</div>
        <div><span style="color:#f39c12;font-size:18px;">●</span> Moderate Risk</div>
        <div><span style="color:#e74c3c;font-size:18px;">●</span> High Risk</div>
        <div><span style="color:#8e44ad;font-size:18px;">●</span> Very High Risk</div>
        <div><span style="color:#95a5a6;font-size:18px;">●</span> Unknown</div>
        <hr style="margin:6px 0;border-color:#555">
        <div style="color:#f39c12;font-size:11px;">
            ⚠️ Experimental estimates only.<br>Not official warnings.
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ─────────────────────────────────────────────
# Full map builder
# ─────────────────────────────────────────────

def build_landslide_map(
    observations_df: pd.DataFrame | None = None,
    risk_data: list[dict] | None = None,
    center_lat: float = DEFAULT_LAT,
    center_lon: float = DEFAULT_LON,
) -> folium.Map:
    """Build a complete landslide risk map.

    Args:
        observations_df: Historical observations DataFrame.
        risk_data: List of risk estimate dicts.
        center_lat: Map center latitude.
        center_lon: Map center longitude.

    Returns:
        Completed Folium Map ready for rendering.
    """
    m = create_base_map(center_lat, center_lon)

    if observations_df is not None and not observations_df.empty:
        m = add_observation_markers(m, observations_df)

    if risk_data:
        m = add_risk_markers(m, risk_data)

    m = add_risk_legend(m)
    return m
