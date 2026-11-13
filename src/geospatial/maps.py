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

import math
from typing import Any

import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

from src.geospatial.spatial_analysis import validate_geojson
from src.risk.thresholds import RISK_EMOJI
from src.utils.logging_config import get_logger
from src.utils.settings import get_setting
from src.utils.validation import validate_coordinates

logger = get_logger(__name__)

# Phase 2 standardized provisional risk legend colors (Green -> Yellow -> Orange -> Red)
MAP_RISK_PALETTE: dict[str, str] = {
    "Low": "#2ecc71",           # Green
    "Moderate": "#f1c40f",      # Yellow
    "High": "#e67e22",          # Orange
    "Very High": "#e74c3c",     # Red
    "Unknown": "#95a5a6",       # Gray
    "Not Evaluated": "#95a5a6", # Gray
    "Unavailable": "#95a5a6",   # Gray
    "Insufficient Data": "#95a5a6",  # Gray
}

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

    for idx, row in valid.iterrows():
        is_synthetic = bool(row.get("is_synthetic", False))
        label = "⚠️ SYNTHETIC DEV DATA" if is_synthetic else "📍 Authentic Landslide Event"
        color = "gray" if is_synthetic else "red"
        icon = "info-sign" if is_synthetic else "warning-sign"
        event_id = row.get("event_id") or row.get("id") or f"row-{idx}"

        popup_lines = [
            f"<b>{label}</b>",
            f"🆔 Event ID: <code>{event_id}</code>",
            f"📍 Coordinates (EPSG:4326): ({float(row['latitude']):.4f}, {float(row['longitude']):.4f})",
            f"📅 Event Date: {row.get('event_date') or row.get('observed_at') or 'Unknown'}",
        ]
        loc_parts = [
            str(row.get(col))
            for col in ("location_name", "district", "state_region", "country")
            if row.get(col) and pd.notna(row.get(col))
        ]
        if loc_parts:
            popup_lines.append(f"🗺️ Location: {', '.join(dict.fromkeys(loc_parts))}")
        if row.get("landslide_type") and pd.notna(row.get("landslide_type")):
            popup_lines.append(f"🪨 Recorded Type: {row['landslide_type']}")
        if row.get("trigger") and pd.notna(row.get("trigger")):
            popup_lines.append(f"💧 Trigger: {row['trigger']}")
        if row.get("fatalities") is not None and pd.notna(row.get("fatalities")):
            popup_lines.append(f"❗ Fatalities: {int(row['fatalities'])}")

        rain_parts = []
        for k, lbl in [("rainfall_1h_mm", "1h"), ("rainfall_24h_mm", "24h"), ("rainfall_72h_mm", "72h")]:
            val = row.get(k)
            if val is not None and pd.notna(val):
                rain_parts.append(f"{lbl}: {float(val):.1f}mm")
        if rain_parts:
            popup_lines.append(f"🌧️ Rainfall: {', '.join(rain_parts)}")

        terrain_parts = []
        for k, lbl in [("elevation_m", "Elev"), ("slope_degrees", "Slope"), ("aspect_degrees", "Aspect")]:
            val = row.get(k)
            if val is not None and pd.notna(val):
                unit = "m" if k == "elevation_m" else "°"
                terrain_parts.append(f"{lbl}: {float(val):.1f}{unit}")
        if terrain_parts:
            popup_lines.append(f"⛰️ Terrain: {', '.join(terrain_parts)}")

        if row.get("source_name") and pd.notna(row.get("source_name")):
            popup_lines.append(f"📊 Data Source: {row['source_name']}")
        if is_synthetic:
            popup_lines.append(
                "<br><i style='color:gray'>SYNTHETIC TEST DATA — Historical/synthetic record, not a live prediction.</i>"
            )
        else:
            popup_lines.append(
                "<br><i style='color:#2c3e50'>Historical observation record (distinct from ML model prediction).</i>"
            )

        popup_html = "<br>".join(popup_lines)

        folium.Marker(
            location=[float(row["latitude"]), float(row["longitude"])],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{label} | {row.get('event_date', 'Unknown')}",
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
        ).add_to(layer)

    layer.add_to(m)
    logger.info("Added %d observation markers to map.", len(valid))
    return m


# ─────────────────────────────────────────────
# Risk estimate markers & heatmap (Phase 2)
# ─────────────────────────────────────────────

def add_risk_markers(
    m: folium.Map,
    risk_data: list[dict],
    cluster: bool = False,
) -> folium.Map:
    """Add experimental risk estimate markers to the map with rich location popups."""
    layer = (
        MarkerCluster(name="Experimental Risk Estimates (Clustered)")
        if cluster
        else folium.FeatureGroup(name="Experimental Risk Estimates")
    )

    for entry in risk_data:
        feats = entry.get("features_used", {}) or {}
        lat = entry.get("latitude") if entry.get("latitude") is not None else feats.get("latitude")
        lon = entry.get("longitude") if entry.get("longitude") is not None else feats.get("longitude")

        if lat is None or lon is None:
            continue

        result = validate_coordinates(lat, lon)
        if not result.is_valid:
            continue

        category = str(entry.get("risk_category") or "Unavailable")
        prob = entry.get("probability")
        has_valid_prob = False
        prob_f: float | None = None
        if prob is not None and not isinstance(prob, bool):
            try:
                cand = float(prob)
                if math.isfinite(cand):
                    prob_f = cand
                    has_valid_prob = True
            except (TypeError, ValueError):
                pass
        is_unavailable = (not has_valid_prob) or category in {
            "Unavailable",
            "Insufficient Data",
            "Not Evaluated",
            "Unknown",
        }
        method = entry.get("method", "unknown")
        is_experimental = entry.get("is_experimental", True)

        marker_color = MAP_RISK_PALETTE.get(category, "#95a5a6")
        emoji = RISK_EMOJI.get(category, "⚪")
        title = entry.get("location_name") or entry.get("name") or "Assessed Location"

        popup_lines = [
            f"<b>{emoji} {title} — {category}</b>",
            f"📍 Coordinates (EPSG:4326): ({float(lat):.4f}, {float(lon):.4f})",
        ]
        if has_valid_prob and prob_f is not None:
            popup_lines.append(f"📊 Estimated Risk Probability: <b>{prob_f:.3f}</b>")
        else:
            popup_lines.append("📊 Estimated Risk Probability: <i>Unavailable / Insufficient Data</i>")

        rain_parts = []
        for k, lbl in [("rainfall_1h_mm", "1h"), ("rainfall_24h_mm", "24h"), ("rainfall_72h_mm", "72h")]:
            val = entry.get(k) if entry.get(k) is not None else feats.get(k)
            if val is not None and not isinstance(val, bool) and math.isfinite(float(val)):
                rain_parts.append(f"{lbl}: {float(val):.1f}mm")
        if rain_parts:
            popup_lines.append(f"🌧️ Rainfall: {', '.join(rain_parts)}")

        terrain_parts = []
        for k, lbl in [("elevation_m", "Elev"), ("slope_degrees", "Slope"), ("aspect_degrees", "Aspect")]:
            val = entry.get(k) if entry.get(k) is not None else feats.get(k)
            if val is not None and not isinstance(val, bool) and math.isfinite(float(val)):
                unit = "m" if k == "elevation_m" else "°"
                terrain_parts.append(f"{lbl}: {float(val):.1f}{unit}")
        if terrain_parts:
            popup_lines.append(f"⛰️ Terrain: {', '.join(terrain_parts)}")

        if entry.get("data_source"):
            popup_lines.append(f"📡 Data Source: {entry['data_source']}")

        freshness = entry.get("freshness_status")
        if freshness:
            popup_lines.append(f"🔄 Data Freshness: <b>{freshness}</b>")

        ts = entry.get("observed_at") or entry.get("estimated_at") or entry.get("predicted_at")
        if ts:
            popup_lines.append(f"🕐 Timestamp: {str(ts)[:19]}")

        model_ver = entry.get("model_version")
        model_name = entry.get("model_name") or method
        model_str = f"{model_name} ({model_ver})" if model_ver else str(model_name)
        popup_lines.append(f"🤖 Model / Method: {model_str}")

        if rain_parts or terrain_parts:
            basis = "; ".join([x for x in [", ".join(rain_parts), ", ".join(terrain_parts)] if x])
            popup_lines.append(f"💡 Contributing Inputs: {basis}")

        if is_experimental:
            popup_lines.append(
                "<br><i style='color:#d35400'><b>⚠️ EXPERIMENTAL MODEL ESTIMATE</b><br>"
                "Provisional score only. Not validated and not an official warning.</i>"
            )
        if entry.get("warnings"):
            popup_lines.append(f"⚠️ Warnings: {len(entry['warnings'])}")

        popup_html = "<br>".join(popup_lines)

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=9 if is_unavailable else 10,
            color=marker_color,
            weight=2,
            dash_array="4, 4" if is_unavailable else None,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.45 if is_unavailable else 0.78,
            popup=folium.Popup(popup_html, max_width=340),
            tooltip=f"{emoji} {title} ({category} | p={prob_f:.3f})"
            if has_valid_prob and prob_f is not None
            else f"{emoji} {title} ({category} / Unavailable)",
        ).add_to(layer)

    layer.add_to(m)
    return m


def add_risk_heatmap(
    m: folium.Map,
    risk_data: list[dict],
    radius: int = 28,
    blur: int = 20,
    min_opacity: float = 0.35,
) -> tuple[folium.Map, dict[str, Any]]:
    """Add an experimental risk heatmap strictly from verified coordinates and model probabilities.

    Does NOT generate random points or interpolate across unmeasured wide gaps.
    """
    heat_points: list[list[float]] = []
    skipped = 0
    unavailable_points = 0

    for entry in risk_data:
        feats = entry.get("features_used", {}) or {}
        lat = entry.get("latitude") if entry.get("latitude") is not None else feats.get("latitude")
        lon = entry.get("longitude") if entry.get("longitude") is not None else feats.get("longitude")
        prob = entry.get("probability")
        cat = str(entry.get("risk_category") or "")

        if lat is None or lon is None:
            skipped += 1
            continue

        coord_check = validate_coordinates(lat, lon)
        if not coord_check.is_valid:
            skipped += 1
            continue

        if prob is None or cat in {"Unavailable", "Insufficient Data", "Not Evaluated"}:
            unavailable_points += 1
            skipped += 1
            continue

        try:
            prob_f = float(prob)
        except (TypeError, ValueError):
            skipped += 1
            continue

        if not math.isfinite(prob_f) or not (0.0 <= prob_f <= 1.0):
            skipped += 1
            continue

        heat_points.append([float(lat), float(lon), max(0.05, prob_f)])

    if heat_points:
        HeatMap(
            heat_points,
            name="Experimental Risk Heatmap (Kernel Density)",
            radius=radius,
            blur=blur,
            min_opacity=min_opacity,
            max_zoom=10,
            gradient={
                0.25: "#2ecc71",  # Low (Green)
                0.50: "#f1c40f",  # Moderate (Yellow)
                0.75: "#e67e22",  # High (Orange)
                1.00: "#e74c3c",  # Very High (Red)
            },
        ).add_to(m)

    meta = {
        "points_plotted": len(heat_points),
        "unavailable_points": unavailable_points,
        "skipped_points": skipped,
        "sufficient_spatial_coverage": len(heat_points) >= 3,
        "method": (
            "Folium weighted kernel-density HeatMap evaluated strictly at verified "
            "location coordinates weighted by experimental model probabilities [0, 1]. "
            "Does not interpolate terrain risk across unmeasured regions."
        ),
    }
    return m, meta


def add_geojson_layer(
    m: folium.Map,
    geojson_data: dict[str, Any],
    name: str = "GeoJSON Overlay (EPSG:4326)",
) -> folium.Map:
    """Validate and add an RFC 7946 GeoJSON overlay layer to the Folium map."""
    validated = validate_geojson(geojson_data)
    group = folium.FeatureGroup(name=name)

    def _style_fn(feature: dict[str, Any]) -> dict[str, Any]:
        props = feature.get("properties") or {}
        cat = str(props.get("risk_category") or "")
        col = MAP_RISK_PALETTE.get(cat, "#38bdf8")
        return {"color": col, "weight": 2.2, "fillColor": col, "fillOpacity": 0.25}

    folium.GeoJson(
        validated,
        name=name,
        style_function=_style_fn,
    ).add_to(group)
    group.add_to(m)
    return m


# ─────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────

def add_risk_legend(m: folium.Map) -> folium.Map:
    """Add standardized Phase 2 risk category legend to the map."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; right: 30px;
        background: rgba(19,27,46,0.94);
        color: #f1f5f9;
        padding: 14px 18px;
        border: 1px solid #2a3954;
        border-radius: 10px;
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
        z-index: 9999;
        box-shadow: 0 2px 10px rgba(0,0,0,0.5);
        min-width: 210px;
    ">
        <b style="font-size:14px;">🗺️ Provisional Risk Legend</b>
        <hr style="margin:6px 0;border-color:#2a3954">
        <div><span style="color:#2ecc71;font-size:16px;">●</span> <b>Green:</b> Low (&lt; 0.35)</div>
        <div><span style="color:#f1c40f;font-size:16px;">●</span> <b>Yellow:</b> Moderate (0.35–0.55)</div>
        <div><span style="color:#e67e22;font-size:16px;">●</span> <b>Orange:</b> High (0.55–0.75)</div>
        <div><span style="color:#e74c3c;font-size:16px;">●</span> <b>Red:</b> Very High (&ge; 0.75)</div>
        <div><span style="color:#95a5a6;font-size:16px;">●</span> <b>Gray:</b> Historical / Unscored</div>
        <hr style="margin:6px 0;border-color:#2a3954">
        <div style="color:#fbbf24;font-size:11px;line-height:1.35;">
            ⚠️ Provisional experimental categories.<br>
            Historical markers &ne; live model predictions.
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
    zoom: int = DEFAULT_ZOOM,
    show_heatmap: bool = False,
    geojson_data: dict[str, Any] | None = None,
    cluster_observations: bool = True,
    show_risk_markers: bool = True,
    cluster_risk_markers: bool = False,
    state_geojson: dict[str, Any] | None = None,
    district_geojson: dict[str, Any] | None = None,
    state_name: str | None = None,
    district_name: str | None = None,
    zoom_bbox: list[float] | tuple[float, float, float, float] | None = None,
    hotspots_data: dict[str, Any] | None = None,
) -> folium.Map:
    """Build a complete multi-layer landslide risk map.

    Args:
        observations_df: Historical / synthetic observations DataFrame.
        risk_data: List of risk estimate dicts.
        center_lat: Map center latitude.
        center_lon: Map center longitude.
        zoom: Initial Folium map zoom.
        show_heatmap: Overlay kernel-density heatmap of valid experimental risk scores.
        geojson_data: Optional validated GeoJSON overlay dict.
        cluster_observations: Use MarkerCluster for historical observation points.
        show_risk_markers: Render risk estimate CircleMarkers when ``risk_data`` is supplied.
        cluster_risk_markers: Use MarkerCluster for dense risk marker layers.
        state_geojson: Optional State/UT GeoJSON boundary polygon (EPSG:4326).
        district_geojson: Optional District GeoJSON boundary polygon (EPSG:4326).
        state_name: Optional label for the state boundary layer.
        district_name: Optional label for the district boundary layer.
        zoom_bbox: Optional ``[min_lat, min_lon, max_lat, max_lon]`` viewport bounds.
        hotspots_data: Optional spatial clustering and KDE density results from detect_landslide_hotspots.

    Returns:
        Completed Folium Map ready for rendering.
    """
    m = create_base_map(center_lat, center_lon, zoom=zoom)

    if state_geojson or district_geojson:
        m = add_india_region_layers(
            m,
            state_geojson=state_geojson,
            district_geojson=district_geojson,
            state_name=state_name,
            district_name=district_name,
        )

    if observations_df is not None and not observations_df.empty:
        m = add_observation_markers(m, observations_df, cluster=cluster_observations)

    if risk_data:
        if show_risk_markers:
            m = add_risk_markers(m, risk_data, cluster=cluster_risk_markers)
        if show_heatmap:
            m, _ = add_risk_heatmap(m, risk_data)

    if hotspots_data and hotspots_data.get("available"):
        m = add_hotspot_clusters_layer(m, hotspots_data)
        if hotspots_data.get("hotspot_regions"):
            m = add_kde_heatmap_layer(m, hotspots_data)

    if geojson_data:
        m = add_geojson_layer(m, geojson_data)

    if zoom_bbox:
        m = fit_map_to_bbox(m, zoom_bbox)

    folium.LayerControl(collapsed=False).add_to(m)
    m = add_risk_legend(m)
    return m


# ─────────────────────────────────────────────
# India State / District Regional Map Layering
# ─────────────────────────────────────────────

def add_india_region_layers(
    m: folium.Map,
    state_geojson: dict[str, Any] | None = None,
    district_geojson: dict[str, Any] | None = None,
    state_name: str | None = None,
    district_name: str | None = None,
) -> folium.Map:
    """Overlay verified State/UT outline and highlighted District polygon (EPSG:4326).

    Administrative polygons are rendered neutrally so a single station's
    experimental risk score is never falsely extrapolated across an entire district.
    """
    if state_geojson and state_geojson.get("features"):
        st_label = f"State Boundary: {state_name or 'Selected State/UT'}"
        st_group = folium.FeatureGroup(name=st_label)
        folium.GeoJson(
            validate_geojson(state_geojson),
            name=st_label,
            style_function=lambda _: {
                "color": "#38bdf8",
                "weight": 2.0,
                "dashArray": "5, 5",
                "fillColor": "#38bdf8",
                "fillOpacity": 0.04,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["shapeName"],
                aliases=["State / UT:"],
                sticky=False,
            ) if _geojson_has_prop(state_geojson, "shapeName") else None,
        ).add_to(st_group)
        st_group.add_to(m)

    if district_geojson and district_geojson.get("features"):
        dist_label = f"District Boundary: {district_name or 'Selected District'}"
        dist_group = folium.FeatureGroup(name=dist_label)
        folium.GeoJson(
            validate_geojson(district_geojson),
            name=dist_label,
            style_function=lambda _: {
                "color": "#f59e0b",
                "weight": 3.0,
                "fillColor": "#f59e0b",
                "fillOpacity": 0.10,
            },
            popup=folium.Popup(
                f"<b>{district_name or 'District'} ({state_name or 'India'})</b><br>"
                "Administrative boundary only (geoBoundaries).<br>"
                "<i>Point weather & experimental ML estimates apply to specific "
                "coordinates, NOT uniformly across the entire polygon.</i>",
                max_width=320,
            ),
            tooltip=folium.GeoJsonTooltip(
                fields=["shapeName"],
                aliases=["District:"],
                sticky=False,
            ) if _geojson_has_prop(district_geojson, "shapeName") else None,
        ).add_to(dist_group)
        dist_group.add_to(m)

    return m


def _geojson_has_prop(geojson_data: dict[str, Any], prop_key: str) -> bool:
    """Check whether all features carry a tooltip property before binding GeoJsonTooltip."""
    feats = geojson_data.get("features") or []
    return bool(feats) and all(
        isinstance(f.get("properties"), dict) and prop_key in f["properties"]
        for f in feats
    )


def fit_map_to_bbox(m: folium.Map, bbox: list[float] | tuple[float, float, float, float] | None) -> folium.Map:
    """Zoom Folium map cleanly to [min_lat, min_lon, max_lat, max_lon] bounding box."""
    if not bbox or len(bbox) != 4:
        return m
    try:
        min_lat, min_lon, max_lat, max_lon = [float(x) for x in bbox]
        if all(math.isfinite(v) for v in (min_lat, min_lon, max_lat, max_lon)):
            if -90.0 <= min_lat <= max_lat <= 90.0 and -180.0 <= min_lon <= max_lon <= 180.0:
                m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    except (TypeError, ValueError):
        pass
    return m


def build_india_regional_map(
    center_lat: float,
    center_lon: float,
    state_name: str,
    district_name: str,
    state_geojson: dict[str, Any] | None = None,
    district_geojson: dict[str, Any] | None = None,
    zoom_bbox: list[float] | None = None,
    historical_df: pd.DataFrame | None = None,
    weather_summary: dict[str, Any] | None = None,
    risk_result: dict[str, Any] | None = None,
    official_alerts: list[dict[str, Any]] | None = None,
) -> folium.Map:
    """Build interactive Folium map for an India State -> District assessment."""
    m = create_base_map(center_lat=center_lat, center_lon=center_lon, zoom=8)
    m = add_india_region_layers(
        m,
        state_geojson=state_geojson,
        district_geojson=district_geojson,
        state_name=state_name,
        district_name=district_name,
    )

    if historical_df is not None and not historical_df.empty:
        m = add_observation_markers(m, historical_df, cluster=True)

    # Representative coordinate point marker (Open-Meteo + optional experimental ML point score)
    coord_check = validate_coordinates(center_lat, center_lon)
    if coord_check.is_valid:
        pt_group = folium.FeatureGroup(name="Representative Point Assessment")
        prob = (risk_result or {}).get("probability")
        category = (risk_result or {}).get("risk_category", "Not Evaluated")
        marker_color = MAP_RISK_PALETTE.get(category, "#38bdf8")

        lines = [
            f"<b>📍 {district_name}, {state_name}</b>",
            f"Representative Coords: ({center_lat:.4f}°N, {center_lon:.4f}°E)",
        ]
        if weather_summary:
            lines.append(
                f"🌡️ Temp: {weather_summary.get('temperature_c', 'N/A')} °C | "
                f"🌧️ Current: {weather_summary.get('current_precip_mm', 'N/A')} mm"
            )
            lines.append(
                f"💧 Rain 24h/72h: {weather_summary.get('rainfall_24h_mm', 'N/A')} / "
                f"{weather_summary.get('rainfall_72h_mm', 'N/A')} mm"
            )
        if prob is not None and math.isfinite(float(prob)):
            lines.append(
                f"⚠️ Experimental Point Risk: <b>{category}</b> ({float(prob):.3f})"
            )
        lines.append(
            "<br><i style='color:#d35400'>Coordinate-level estimate only; does NOT "
            "represent uniform risk across the entire district polygon.</i>"
        )
        folium.CircleMarker(
            location=[float(center_lat), float(center_lon)],
            radius=11,
            color="#0f172a",
            weight=2,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.92,
            popup=folium.Popup("<br>".join(lines), max_width=340),
            tooltip=f"{district_name} representative point ({category})",
        ).add_to(pt_group)
        pt_group.add_to(m)

    # Official SACHET / IMD alert markers where valid centroids are provided
    if official_alerts:
        alert_group = folium.FeatureGroup(name="Official External Alerts (NDMA SACHET)")
        for alert in official_alerts:
            a_lat = alert.get("centroid_lat")
            a_lon = alert.get("centroid_lon")
            if a_lat is None or a_lon is None:
                continue
            c_ok = validate_coordinates(a_lat, a_lon)
            if not c_ok.is_valid:
                continue
            sev = str(alert.get("severity") or "Advisory")
            d_type = str(alert.get("alert_title") or alert.get("disaster_type") or "Weather/Hazard Alert")
            area = str(alert.get("affected_region") or alert.get("area_description") or "")
            msg = str(alert.get("warning_message") or alert.get("headline") or "")
            popup_html = (
                f"<b>🏛️ Official Alert ({alert.get('alert_source') or alert.get('provider') or 'NDMA SACHET'})</b><br>"
                f"Hazard: {d_type}<br>"
                f"Severity: <b>{sev}</b><br>"
                f"Area: {area}<br>"
                f"Summary: {msg[:240]}"
            )
            folium.Marker(
                location=[float(a_lat), float(a_lon)],
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"Official {alert.get('provider', 'SACHET')}: {d_type} ({sev})",
                icon=folium.Icon(color="red" if "extreme" in sev.lower() or "red" in str(alert.get("severity_color", "")).lower() else "orange", icon="exclamation-sign"),
            ).add_to(alert_group)
        alert_group.add_to(m)

    if zoom_bbox:
        m = fit_map_to_bbox(m, zoom_bbox)

    folium.LayerControl(collapsed=False).add_to(m)
    m = add_risk_legend(m)
    return m


# ─────────────────────────────────────────────
# Phase 7 Hotspot Clusters & KDE Density Layers
# ─────────────────────────────────────────────

def add_hotspot_clusters_layer(
    m: folium.Map,
    hotspots_data: dict[str, Any],
) -> folium.Map:
    """Overlay historical DBSCAN spatial clusters and convex hull polygons (EPSG:4326)."""
    clusters = hotspots_data.get("clusters", [])
    if not clusters:
        return m

    cluster_group = folium.FeatureGroup(name=f"Historical Hotspot Clusters ({len(clusters)})")

    for c in clusters:
        cid = c["cluster_id"]
        cnt = c["point_count"]
        density = c["event_density_per_sq_km"]
        cent = c["centroid"]
        hull = c.get("convex_hull", [])
        trig = c.get("dominant_trigger", "Unknown")
        dates = c.get("date_range", {})

        popup_html = (
            f"<b>🎯 Historical Cluster #{cid}</b><br>"
            f"Observed Events: <b>{cnt}</b><br>"
            f"Event Density: <b>{density:.2f} events/km²</b><br>"
            f"Area: ~{c.get('area_sq_km', 0.1):.1f} km²<br>"
            f"Dominant Trigger: {trig}<br>"
            f"Temporal Range: {dates.get('start', 'N/A')} to {dates.get('end', 'N/A')}<br>"
            "<i style='color:#c0392b; font-size:11px'>Historical concentration only. "
            "Does NOT predict future activation without dynamic environmental triggers.</i>"
        )

        if hull and len(hull) >= 3:
            folium.Polygon(
                locations=hull,
                color="#8e44ad",
                weight=2,
                fill=True,
                fill_color="#8e44ad",
                fill_opacity=0.18,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"Cluster #{cid} ({cnt} events, {density:.1f}/km²)",
            ).add_to(cluster_group)

        folium.CircleMarker(
            location=[cent["latitude"], cent["longitude"]],
            radius=7,
            color="#4a154b",
            weight=2,
            fill=True,
            fill_color="#9b59b6",
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"Cluster #{cid} Centroid",
        ).add_to(cluster_group)

    cluster_group.add_to(m)
    return m


def add_kde_heatmap_layer(
    m: folium.Map,
    hotspots_data: dict[str, Any],
) -> folium.Map:
    """Overlay Kernel Density Estimation (KDE) top hotspot zones (density >= 90th percentile)."""
    hotspots = hotspots_data.get("hotspot_regions", [])
    if not hotspots:
        return m

    kde_group = folium.FeatureGroup(name=f"KDE High-Density Zones ({len(hotspots)})")

    for pt in hotspots:
        lat = pt["latitude"]
        lon = pt["longitude"]
        tier = pt.get("hotspot_tier", "Elevated Density")
        val = pt.get("density_value", 0.0)
        is_extreme = "Extreme" in tier or "p99" in tier

        folium.CircleMarker(
            location=[lat, lon],
            radius=9 if is_extreme else 6,
            color="#c0392b" if is_extreme else "#d35400",
            weight=1,
            fill=True,
            fill_color="#e74c3c" if is_extreme else "#e67e22",
            fill_opacity=0.55 if is_extreme else 0.40,
            tooltip=f"KDE Hotspot: {tier} ({val:.5f})",
            popup=folium.Popup(
                f"<b>Density Zone: {tier}</b><br>Coordinates: ({lat:.4f}°N, {lon:.4f}°E)<br>"
                f"KDE Density: {val:.6f}<br>"
                "<i>Historical event concentration indicator.</i>",
                max_width=280,
            ),
        ).add_to(kde_group)

    kde_group.add_to(m)
    return m

