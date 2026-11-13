"""
Spatial analysis utilities for LandslideGuard.

Provides spatial aggregation, clustering, and nearest-neighbour queries
for landslide observation datasets.

NOTE: geopandas is used where available. If it is not installed, pure
pandas/numpy implementations are used as fallback.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Check geopandas availability
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
    logger.info("geopandas available for spatial operations.")
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning("geopandas not available; using pure pandas/numpy spatial fallbacks.")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points (km).

    Args:
        lat1, lon1: First point coordinates (decimal degrees).
        lat2, lon2: Second point coordinates (decimal degrees).

    Returns:
        Distance in kilometres.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def filter_by_bounding_box(
    df: pd.DataFrame,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> pd.DataFrame:
    """Filter observations to a geographic bounding box.

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns.
        min_lat, max_lat: Latitude bounds.
        min_lon, max_lon: Longitude bounds.

    Returns:
        Filtered DataFrame.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning("Cannot apply bounding box: missing coordinate columns.")
        return df

    mask = (
        df["latitude"].between(min_lat, max_lat) &
        df["longitude"].between(min_lon, max_lon)
    )
    result = df[mask].copy()
    logger.debug("Bounding box filter: %d → %d rows.", len(df), len(result))
    return result


def compute_spatial_density(
    df: pd.DataFrame,
    grid_size_deg: float = 1.0,
) -> pd.DataFrame:
    """Compute event count density on a regular geographic grid.

    Args:
        df: DataFrame with 'latitude', 'longitude' columns.
        grid_size_deg: Grid cell size in degrees.

    Returns:
        DataFrame with grid cell centres and event counts.
    """
    if df.empty or "latitude" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["lat_bin"] = (df["latitude"] // grid_size_deg) * grid_size_deg + grid_size_deg / 2
    df["lon_bin"] = (df["longitude"] // grid_size_deg) * grid_size_deg + grid_size_deg / 2

    density = (
        df.groupby(["lat_bin", "lon_bin"])
        .size()
        .reset_index(name="event_count")
    )
    return density


def nearest_observations(
    lat: float,
    lon: float,
    df: pd.DataFrame,
    n: int = 5,
    max_km: float = 500.0,
    include_distance: bool = False,
) -> pd.DataFrame:
    """Find the N nearest observations within a maximum distance.

    Args:
        lat: Query latitude.
        lon: Query longitude.
        df: DataFrame with 'latitude' and 'longitude'.
        n: Maximum number of results.
        max_km: Maximum distance threshold (km).
        include_distance: If True, retain ``distance_km`` column in output.

    Returns:
        DataFrame of nearest observations sorted by distance.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    valid, _ = drop_invalid_coordinates(df)
    if valid.empty:
        return pd.DataFrame()

    valid["_distance_km"] = valid.apply(
        lambda r: haversine_distance(float(lat), float(lon), float(r["latitude"]), float(r["longitude"])),
        axis=1,
    )
    result = valid[valid["_distance_km"] <= max_km].nsmallest(n, "_distance_km").copy()
    if include_distance:
        result = result.rename(columns={"_distance_km": "distance_km"})
        result["distance_km"] = result["distance_km"].round(2)
    else:
        result = result.drop(columns=["_distance_km"])
    return result


def ensure_wgs84_geodataframe(
    gdf: Any,
    source_crs: str | None = None,
) -> Any:
    """Validate and normalize a GeoDataFrame to WGS84 (EPSG:4326) with valid WGS84 bounds.

    Rejects unreferenced GeoDataFrames unless an explicit ``source_crs`` is provided,
    reprojects projected layers via GeoPandas/PyProj, and validates bounds/geometries.
    """
    if not GEOPANDAS_AVAILABLE:
        raise RuntimeError("geopandas is required to validate or reproject GeoDataFrames.")
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Expected a geopandas.GeoDataFrame instance.")
    if gdf.empty:
        return gdf.set_crs("EPSG:4326", allow_override=True) if gdf.crs is None else gdf.to_crs("EPSG:4326")

    working = gdf.copy()
    if working.crs is None:
        if not source_crs:
            raise ValueError("GeoDataFrame has no Coordinate Reference System (CRS); supply source_crs explicitly.")
        working = working.set_crs(source_crs)

    if str(working.crs).upper() != "EPSG:4326":
        working = working.to_crs("EPSG:4326")

    if working.geometry.isna().any() or working.geometry.is_empty.any():
        raise ValueError("GeoDataFrame contains empty or null geometries.")

    minx, miny, maxx, maxy = working.total_bounds
    if not (
        math.isfinite(minx)
        and math.isfinite(miny)
        and math.isfinite(maxx)
        and math.isfinite(maxy)
        and -180.0 <= minx <= maxx <= 180.0
        and -90.0 <= miny <= maxy <= 90.0
    ):
        raise ValueError(
            f"GeoDataFrame bounds {(minx, miny, maxx, maxy)} fall outside valid WGS84 (EPSG:4326) extents."
        )
    return working


def to_geodataframe(df: pd.DataFrame) -> Any:
    """Convert a DataFrame to a GeoDataFrame.

    Falls back to returning the plain DataFrame if geopandas is not available.

    Args:
        df: DataFrame with 'latitude' and 'longitude'.

    Returns:
        GeoDataFrame or plain DataFrame.
    """
    if not GEOPANDAS_AVAILABLE:
        logger.warning("Cannot create GeoDataFrame: geopandas not available.")
        return df

    if df.empty:
        return gpd.GeoDataFrame()

    df = df.copy()
    df["geometry"] = df.apply(
        lambda r: Point(r["longitude"], r["latitude"]), axis=1
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf


def summarise_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observation counts by country.

    Args:
        df: Observations DataFrame.

    Returns:
        DataFrame with country-level event counts.
    """
    if "country" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("country")
        .agg(
            event_count=("latitude", "count"),
            avg_lat=("latitude", "mean"),
            avg_lon=("longitude", "mean"),
        )
        .reset_index()
        .sort_values("event_count", ascending=False)
    )
    return summary


# ─────────────────────────────────────────────
# Geographic filtering & GeoJSON utilities (Phase 2)
# ─────────────────────────────────────────────

def drop_invalid_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Filter DataFrame to rows with finite WGS84 coordinates.

    Returns:
        Tuple of (valid_df, dropped_count).
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy(), 0
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df.iloc[0:0].copy(), len(df)

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    mask = lat.notna() & lon.notna() & lat.between(-90.0, 90.0) & lon.between(-180.0, 180.0)
    valid = df[mask].copy()
    valid["latitude"] = lat[mask].astype(float)
    valid["longitude"] = lon[mask].astype(float)
    return valid, int((~mask).sum())


def apply_geographic_filters(
    df: pd.DataFrame,
    states_regions: list[str] | None = None,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    risk_categories: list[str] | None = None,
    date_range: tuple[Any, Any] | None = None,
    data_sources: list[str] | None = None,
    districts: list[str] | None = None,
    search_query: str | None = None,
    authentic_only: bool = False,
) -> pd.DataFrame:
    """Apply geographic, administrative, text-search, temporal, risk-tier, and source filters safely."""
    valid, _ = drop_invalid_coordinates(df)
    if valid.empty:
        return valid

    if min_lat is not None and max_lat is not None and min_lat <= max_lat:
        valid = valid[valid["latitude"].between(float(min_lat), float(max_lat))]
    if min_lon is not None and max_lon is not None and min_lon <= max_lon:
        valid = valid[valid["longitude"].between(float(min_lon), float(max_lon))]

    if states_regions and "state_region" in valid.columns:
        allowed_states = {str(s).strip().lower() for s in states_regions if str(s).strip()}
        if allowed_states:
            valid = valid[valid["state_region"].fillna("Unknown").astype(str).str.strip().str.lower().isin(allowed_states)]

    if districts and "district" in valid.columns:
        allowed_districts = {str(d).strip().lower() for d in districts if str(d).strip()}
        if allowed_districts:
            valid = valid[valid["district"].fillna("").astype(str).str.strip().str.lower().isin(allowed_districts)]

    if authentic_only and "is_synthetic" in valid.columns:
        synth_vals = pd.to_numeric(valid["is_synthetic"], errors="coerce").fillna(1).astype(int)
        valid = valid[synth_vals == 0]

    if risk_categories and "risk_category" in valid.columns:
        allowed_risks = {str(r) for r in risk_categories}
        valid = valid[valid["risk_category"].fillna("Unknown").astype(str).isin(allowed_risks)]

    if data_sources:
        source_col = (
            "source_name"
            if "source_name" in valid.columns
            else ("data_source" if "data_source" in valid.columns else None)
        )
        if source_col:
            allowed_src = {str(s) for s in data_sources}
            valid = valid[valid[source_col].fillna("Unknown").astype(str).isin(allowed_src)]

    if search_query and str(search_query).strip():
        needle = str(search_query).strip().lower()
        search_cols = [
            c
            for c in [
                "name",
                "location_name",
                "district",
                "state_region",
                "country",
                "source_name",
                "data_source",
                "trigger",
                "landslide_type",
            ]
            if c in valid.columns
        ]
        if search_cols:
            mask = pd.Series(False, index=valid.index)
            for col in search_cols:
                mask = mask | valid[col].fillna("").astype(str).str.lower().str.contains(needle, regex=False)
            valid = valid[mask]

    if date_range and len(date_range) == 2 and date_range[0] is not None and date_range[1] is not None:
        date_col = (
            "event_date"
            if "event_date" in valid.columns
            else ("observed_at" if "observed_at" in valid.columns else None)
        )
        if date_col:
            parsed = pd.to_datetime(valid[date_col], errors="coerce", utc=True).dt.date
            start_d, end_d = date_range[0], date_range[1]
            valid = valid[parsed.notna() & (parsed >= start_d) & (parsed <= end_d)]

    return valid.copy()


def _check_coords_pair(coords: Any) -> bool:
    """Return True if [lon, lat, ...] is finite and within WGS84 bounds."""
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return False
    lon, lat = coords[0], coords[1]
    if isinstance(lon, bool) or isinstance(lat, bool):
        return False
    try:
        lon_f, lat_f = float(lon), float(lat)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(lon_f)
        and math.isfinite(lat_f)
        and -180.0 <= lon_f <= 180.0
        and -90.0 <= lat_f <= 90.0
    )


def _is_valid_polygon_ring(ring: Any) -> bool:
    """Validate an RFC 7946 LinearRing (>= 4 valid WGS84 positions with matching closure)."""
    if not isinstance(ring, list) or len(ring) < 4:
        return False
    if not all(_check_coords_pair(c) for c in ring):
        return False
    return float(ring[0][0]) == float(ring[-1][0]) and float(ring[0][1]) == float(ring[-1][1])


def _validate_geometry_coords(geom: dict[str, Any]) -> bool:
    """Validate GeoJSON geometry structure and coordinate arrays against WGS84 bounds."""
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Point":
        return _check_coords_pair(coords)
    if gtype == "MultiPoint":
        return isinstance(coords, list) and len(coords) > 0 and all(_check_coords_pair(c) for c in coords)
    if gtype == "LineString":
        return isinstance(coords, list) and len(coords) >= 2 and all(_check_coords_pair(c) for c in coords)
    if gtype == "MultiLineString":
        return (
            isinstance(coords, list)
            and len(coords) > 0
            and all(isinstance(line, list) and len(line) >= 2 and all(_check_coords_pair(c) for c in line) for line in coords)
        )
    if gtype == "Polygon":
        return isinstance(coords, list) and len(coords) > 0 and all(_is_valid_polygon_ring(ring) for ring in coords)
    if gtype == "MultiPolygon":
        return (
            isinstance(coords, list)
            and len(coords) > 0
            and all(
                isinstance(poly, list) and len(poly) > 0 and all(_is_valid_polygon_ring(ring) for ring in poly)
                for poly in coords
            )
        )
    if gtype == "GeometryCollection":
        geoms = geom.get("geometries")
        return isinstance(geoms, list) and len(geoms) > 0 and all(isinstance(g, dict) and _validate_geometry_coords(g) for g in geoms)
    return False


def validate_geojson(data: Any) -> dict[str, Any]:
    """Validate and normalize GeoJSON into a WGS84 FeatureCollection.

    Raises:
        ValueError: If GeoJSON structure, CRS declaration, or WGS84 coordinates are invalid.
    """
    if not isinstance(data, dict):
        raise ValueError("GeoJSON payload must be a JSON object.")

    crs_member = data.get("crs")
    if crs_member is not None:
        crs_text = str(crs_member).upper()
        if "4326" not in crs_text and "CRS84" not in crs_text:
            raise ValueError(
                f"Unsupported non-WGS84 GeoJSON CRS {crs_member!r}; reproject layer to EPSG:4326 before loading."
            )

    gtype = data.get("type")
    if gtype == "Feature":
        features = [data]
    elif gtype == "FeatureCollection":
        features = data.get("features")
        if not isinstance(features, list):
            raise ValueError("GeoJSON FeatureCollection requires a 'features' list.")
    else:
        raise ValueError(f"Unsupported GeoJSON type: {gtype!r}. Expected Feature or FeatureCollection.")

    valid_features: list[dict[str, Any]] = []
    for idx, feat in enumerate(features):
        if not isinstance(feat, dict) or feat.get("type") != "Feature":
            raise ValueError(f"Feature at index {idx} is malformed.")
        geom = feat.get("geometry")
        if not isinstance(geom, dict) or not _validate_geometry_coords(geom):
            raise ValueError(f"Feature at index {idx} contains invalid or out-of-range WGS84 coordinates.")
        valid_features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": feat.get("properties") if isinstance(feat.get("properties"), dict) else {},
        })

    return {
        "type": "FeatureCollection",
        "crs_note": "WGS84 (EPSG:4326) longitude-latitude coordinate order per RFC 7946",
        "features": valid_features,
    }


def records_to_geojson(
    records: list[dict[str, Any]],
    record_kind: str = "risk_estimate",
) -> dict[str, Any]:
    """Serialize validated coordinate records into an RFC 7946 GeoJSON FeatureCollection."""
    features: list[dict[str, Any]] = []
    for rec in records:
        try:
            lat = float(rec["latitude"])
            lon = float(rec["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(lat) and math.isfinite(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue

        props: dict[str, Any] = {"record_kind": record_kind}
        for k, v in rec.items():
            if k in {"latitude", "longitude"}:
                continue
            if isinstance(v, float) and not math.isfinite(v):
                props[k] = None
            elif pd.isna(v) if not isinstance(v, (list, dict, tuple)) else False:
                props[k] = None
            else:
                props[k] = v

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _finite_float(val: Any) -> float | None:
    """Return finite float or None."""
    if val is None or isinstance(val, bool):
        return None
    try:
        num = float(val)
        return num if math.isfinite(num) else None
    except (TypeError, ValueError):
        return None


def inspect_location_details(
    latitude: float,
    longitude: float,
    stations: list[dict[str, Any]] | None = None,
    observations_df: pd.DataFrame | None = None,
    terrain_summary: dict[str, Any] | None = None,
    radius_km: float = 50.0,
) -> dict[str, Any]:
    """Assemble full click-inspector details for a selected/clicked map location.

    Only reports values actually present on matched stations, DEM summaries, or historical records.
    Missing values stay None / 'Unavailable'.
    """
    lat_f = _finite_float(latitude)
    lon_f = _finite_float(longitude)
    if lat_f is None or lon_f is None or not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return {
            "valid": False,
            "reason": f"Invalid WGS84 coordinates ({latitude}, {longitude}).",
        }

    matched_station: dict[str, Any] | None = None
    nearest_station_km: float | None = None
    for st in stations or []:
        s_lat = _finite_float(st.get("latitude"))
        s_lon = _finite_float(st.get("longitude"))
        if s_lat is None or s_lon is None:
            continue
        dist = haversine_distance(lat_f, lon_f, s_lat, s_lon)
        if nearest_station_km is None or dist < nearest_station_km:
            nearest_station_km = round(dist, 2)
            matched_station = st

    station_within_radius = matched_station if (nearest_station_km is not None and nearest_station_km <= radius_km) else None

    # 1. Coordinates
    coords_info = {
        "latitude": round(lat_f, 5),
        "longitude": round(lon_f, 5),
        "crs": "EPSG:4326",
        "matched_station": (station_within_radius or {}).get("name") or (station_within_radius or {}).get("location_name"),
        "nearest_station_km": nearest_station_km,
    }

    # 2. Available rainfall information
    src_st = station_within_radius or {}
    r1 = _finite_float(src_st.get("rainfall_1h_mm"))
    r24 = _finite_float(src_st.get("rainfall_24h_mm"))
    r72 = _finite_float(src_st.get("rainfall_72h_mm"))
    rainfall_available = any(v is not None for v in (r1, r24, r72))
    rainfall_info = {
        "available": rainfall_available,
        "rainfall_1h_mm": r1,
        "rainfall_24h_mm": r24,
        "rainfall_72h_mm": r72,
        "temperature_c": _finite_float(src_st.get("temperature_c")),
        "humidity_pct": _finite_float(src_st.get("humidity_pct")),
        "data_source": src_st.get("data_source") or "Unavailable",
    }

    # 3. Terrain information (prefer explicit DEM terrain_summary if available, else station terrain metadata)
    ts = terrain_summary if (isinstance(terrain_summary, dict) and terrain_summary.get("available")) else {}
    elev = _finite_float(ts.get("elevation_m")) if ts else _finite_float(src_st.get("elevation_m"))
    slope = _finite_float(ts.get("slope_degrees")) if ts else _finite_float(src_st.get("slope_degrees"))
    aspect = _finite_float(ts.get("aspect_degrees")) if ts else _finite_float(src_st.get("aspect_degrees"))
    roughness = _finite_float(ts.get("roughness_tri_m")) if ts else None
    terrain_info = {
        "available": any(v is not None for v in (elev, slope, aspect, roughness)),
        "elevation_m": elev,
        "slope_degrees": slope,
        "aspect_degrees": aspect,
        "aspect_cardinal": ts.get("aspect_cardinal") if ts else None,
        "roughness_tri_m": roughness,
        "source": ts.get("source") if ts else ("Station metadata" if (elev is not None or slope is not None) else "Unavailable"),
    }

    # 4. Historical event information
    near_df = nearest_observations(lat_f, lon_f, observations_df, n=5, max_km=radius_km, include_distance=True)
    nearest_events: list[dict[str, Any]] = []
    auth_cnt = 0
    synth_cnt = 0
    if not near_df.empty:
        for idx, row in near_df.iterrows():
            is_synth = bool(row.get("is_synthetic", False))
            if is_synth:
                synth_cnt += 1
            else:
                auth_cnt += 1
            nearest_events.append({
                "event_id": row.get("event_id") or row.get("id") or f"row-{idx}",
                "event_date": str(row.get("event_date") or row.get("observed_at") or "Unknown"),
                "location": ", ".join(
                    str(x) for x in [row.get("district"), row.get("state_region"), row.get("country")]
                    if x and pd.notna(x)
                ) or "Unknown",
                "trigger": str(row.get("trigger") or "Unknown"),
                "landslide_type": str(row.get("landslide_type") or "Unknown"),
                "source": str(row.get("source_name") or "Unknown"),
                "distance_km": float(row["distance_km"]),
                "is_synthetic": is_synth,
            })
    historical_info = {
        "available": len(nearest_events) > 0,
        "radius_km": radius_km,
        "count": len(nearest_events),
        "authentic_count": auth_cnt,
        "synthetic_count": synth_cnt,
        "nearest_events": nearest_events,
    }

    # 5. Model-generated risk estimate
    prob = _finite_float(src_st.get("probability"))
    risk_cat = str(src_st.get("risk_category") or "Unavailable")
    risk_available = prob is not None and risk_cat not in {"Unavailable", "Not Evaluated", "Unknown", "Insufficient Data"}
    risk_info = {
        "available": risk_available,
        "risk_category": risk_cat if station_within_radius else "Unavailable",
        "probability": prob if station_within_radius else None,
        "model_name": src_st.get("model_name"),
        "model_version": src_st.get("model_version"),
        "method": src_st.get("method"),
        "is_experimental": True,
    }

    # 6. Data freshness
    freshness_info = {
        "freshness_status": src_st.get("freshness_status") or "Unavailable",
        "observed_at": src_st.get("observed_at") or "Unavailable",
        "predicted_at": src_st.get("predicted_at") or "Unavailable",
        "data_age_hours": _finite_float(src_st.get("data_age_hours")),
    }

    # 7. Explanation of the displayed risk (derived strictly from recorded station factors/warnings)
    if risk_available:
        reasons: list[str] = []
        if r24 is not None:
            reasons.append(f"24h rainfall is {r24:.1f} mm" + (f" (72h: {r72:.1f} mm)" if r72 is not None else ""))
        elif r1 is not None:
            reasons.append(f"1h rainfall is {r1:.1f} mm")
        if slope is not None:
            reasons.append(f"terrain slope is {slope:.1f}°")
        if elev is not None:
            reasons.append(f"elevation is {elev:,.0f} m")
        if src_st.get("soil_moisture") is not None:
            sm = _finite_float(src_st.get("soil_moisture"))
            if sm is not None:
                reasons.append(f"topsoil moisture is {sm:.3f} m³/m³")
        expl_text = (
            f"Experimental {risk_cat} score (p={prob:.3f} via {src_st.get('model_name') or src_st.get('method') or 'risk engine'}) "
            + ("reflecting: " + "; ".join(reasons) + "." if reasons else "using available station telemetry.")
        )
        warns = src_st.get("warnings") or []
        if warns:
            expl_text += f" Caveats ({len(warns)}): {warns[0]}"
    else:
        expl_text = (
            f"No model risk assessment is available within {radius_km:.0f} km of ({lat_f:.4f}, {lon_f:.4f}). "
            "Use Point SRTM Inspection or Risk Analysis to evaluate this coordinate."
        )

    return {
        "valid": True,
        "coordinates": coords_info,
        "rainfall": rainfall_info,
        "terrain": terrain_info,
        "historical_events": historical_info,
        "risk_estimate": risk_info,
        "data_freshness": freshness_info,
        "risk_explanation": expl_text,
    }
