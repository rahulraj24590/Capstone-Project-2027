"""Authoritative India State/UT (ADM1) and District (ADM2) boundary provider.

Source attribution (from geoBoundaries live metadata):
- ADM1 (36 States & Union Territories): DataMeet India Community / Election Commission of India (CC BY 2.5 IN)
- ADM2 (735 Districts): Pathways Data Pvt. Ltd. / Local Government Directory (lgdirectory.gov.in, ODbL 1.0)

Stores boundaries and the spatial State -> District index locally under
``data/india_boundaries/`` to minimize remote downloads and enable offline selection.
"""

from __future__ import annotations

import json
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests

from src.utils.logging_config import get_logger
from src.utils.settings import get_setting

logger = get_logger(__name__)

DEFAULT_BOUNDARIES_DIR = Path(
    get_setting("india_monitoring", "boundaries_dir") or "data/india_boundaries"
)
ADM1_META_URL = (
    get_setting("india_monitoring", "geoboundaries_adm1_url")
    or "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/"
)
ADM2_META_URL = (
    get_setting("india_monitoring", "geoboundaries_adm2_url")
    or "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"
)
BOUNDARY_ATTRIBUTION = (
    get_setting("india_monitoring", "boundary_attribution")
    or "geoBoundaries IND ADM1 (DataMeet/ECI, CC BY 2.5 IN) & ADM2 (Pathways/lgdirectory.gov.in, ODbL 1.0)"
)


def normalize_admin_name(name: str | None) -> str:
    """Normalize Unicode diacritics (e.g. 'Mahārāshtra' -> 'Maharashtra') for clean search."""
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.split()).strip()


def build_admin_index_from_geodataframes(
    gdf_adm1: gpd.GeoDataFrame,
    gdf_adm2: gpd.GeoDataFrame,
    adm1_source: str = "DataMeet / Election Commission of India (via geoBoundaries)",
    adm2_source: str = "Pathways Data / lgdirectory.gov.in (via geoBoundaries)",
) -> dict[str, Any]:
    """Spatially join ADM2 districts into ADM1 states/UTs and extract centroids & bboxes."""
    if gdf_adm1.empty or gdf_adm2.empty:
        raise ValueError("ADM1 and ADM2 GeoDataFrames must both be non-empty.")

    adm1 = gdf_adm1.copy()
    adm2 = gdf_adm2.copy()
    if adm1.crs is None:
        adm1 = adm1.set_crs("EPSG:4326")
    elif str(adm1.crs).upper() != "EPSG:4326":
        adm1 = adm1.to_crs("EPSG:4326")
    if adm2.crs is None:
        adm2 = adm2.set_crs("EPSG:4326")
    elif str(adm2.crs).upper() != "EPSG:4326":
        adm2 = adm2.to_crs("EPSG:4326")

    # Compute representative points guaranteed inside each polygon
    adm1_pts = adm1.geometry.representative_point()
    states_list: list[dict[str, Any]] = []
    state_by_raw: dict[str, dict[str, Any]] = {}
    for idx, row in adm1.iterrows():
        raw_name = str(row.get("shapeName") or "").strip()
        disp_name = normalize_admin_name(raw_name)
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        pt = adm1_pts.loc[idx]
        rec = {
            "state_name": disp_name,
            "official_name": raw_name,
            "search_name": disp_name.lower(),
            "shape_iso": str(row.get("shapeISO") or "").strip(),
            "shape_id": str(row.get("shapeID") or "").strip(),
            "centroid_lat": round(float(pt.y), 6),
            "centroid_lon": round(float(pt.x), 6),
            "bbox": [
                round(float(bounds[1]), 6),  # min_lat
                round(float(bounds[0]), 6),  # min_lon
                round(float(bounds[3]), 6),  # max_lat
                round(float(bounds[2]), 6),  # max_lon
            ],
            "source": adm1_source,
        }
        states_list.append(rec)
        state_by_raw[raw_name] = rec

    states_list.sort(key=lambda s: s["state_name"])

    # Spatial containment join from ADM2 representative points to ADM1 polygons
    adm2_rep_pts = adm2.geometry.representative_point()
    pts_gdf = gpd.GeoDataFrame(
        {
            "district_raw": adm2["shapeName"].astype(str),
            "shape_id": adm2.get("shapeID", "").astype(str),
            "shape_iso": adm2.get("shapeISO", "").astype(str),
        },
        geometry=adm2_rep_pts,
        crs="EPSG:4326",
    )
    adm1_lookup = adm1[["shapeName", "geometry"]].rename(columns={"shapeName": "state_raw"})
    joined = gpd.sjoin(pts_gdf, adm1_lookup, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    unmatched = joined[joined["state_raw"].isna()]
    if not unmatched.empty:
        adm1_m = adm1_lookup.to_crs("EPSG:3857")
        pts_m = pts_gdf.loc[unmatched.index].to_crs("EPSG:3857")
        nearest = gpd.sjoin_nearest(pts_m, adm1_m, how="left")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        for idx, nrow in nearest.iterrows():
            joined.loc[idx, "state_raw"] = nrow["state_raw"]

    districts_list: list[dict[str, Any]] = []
    for idx, jrow in joined.iterrows():
        raw_d = str(jrow["district_raw"]).strip()
        disp_d = normalize_admin_name(raw_d)
        raw_s = str(jrow.get("state_raw") or "Unknown").strip()
        state_meta = state_by_raw.get(raw_s)
        disp_s = state_meta["state_name"] if state_meta else normalize_admin_name(raw_s)
        geom = adm2.loc[idx, "geometry"]
        bounds = geom.bounds
        pt = adm2_rep_pts.loc[idx]
        districts_list.append(
            {
                "state_name": disp_s,
                "state_official_name": raw_s,
                "district_name": disp_d,
                "official_name": raw_d,
                "search_name": disp_d.lower(),
                "shape_id": str(jrow.get("shape_id") or "").strip(),
                "rep_lat": round(float(pt.y), 6),
                "rep_lon": round(float(pt.x), 6),
                "bbox": [
                    round(float(bounds[1]), 6),
                    round(float(bounds[0]), 6),
                    round(float(bounds[3]), 6),
                    round(float(bounds[2]), 6),
                ],
                "source": adm2_source,
            }
        )

    districts_list.sort(key=lambda d: (d["state_name"], d["district_name"]))

    return {
        "country": "India",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "attribution": BOUNDARY_ATTRIBUTION,
        "state_count": len(states_list),
        "district_count": len(districts_list),
        "states": states_list,
        "districts": districts_list,
    }


class IndiaBoundaryProvider:
    """Manages cached India State/UT & District polygons and coordinate resolution."""

    def __init__(self, boundaries_dir: str | Path | None = None) -> None:
        self.boundaries_dir = Path(boundaries_dir) if boundaries_dir else DEFAULT_BOUNDARIES_DIR
        self.boundaries_dir.mkdir(parents=True, exist_ok=True)
        self.adm1_path = self.boundaries_dir / "adm1_states.geojson"
        self.adm2_path = self.boundaries_dir / "adm2_districts.geojson"
        self.index_path = self.boundaries_dir / "india_districts_index.json"
        self._index_cache: dict[str, Any] | None = None
        self._adm1_gj: dict[str, Any] | None = None
        self._adm2_gj: dict[str, Any] | None = None

    def download_and_cache(
        self,
        timeout_seconds: int = 30,
        force: bool = False,
    ) -> dict[str, Any]:
        """Download ADM1 and ADM2 geoBoundaries GeoJSONs and cache index locally."""
        if (
            not force
            and self.adm1_path.exists()
            and self.adm2_path.exists()
            and self.index_path.exists()
        ):
            return self.get_boundary_status()

        session = requests.Session()
        session.headers.update({"User-Agent": "LandslideGuard/1.0 (academic research)"})

        meta1 = session.get(ADM1_META_URL, timeout=timeout_seconds).json()
        meta2 = session.get(ADM2_META_URL, timeout=timeout_seconds).json()

        u1 = meta1.get("simplifiedGeometryGeoJSON") or meta1.get("gjDownloadURL")
        u2 = meta2.get("simplifiedGeometryGeoJSON") or meta2.get("gjDownloadURL")
        if not u1 or not u2:
            raise RuntimeError("geoBoundaries metadata did not return GeoJSON download URLs.")

        r1 = session.get(u1, timeout=max(timeout_seconds, 45))
        r1.raise_for_status()
        r2 = session.get(u2, timeout=max(timeout_seconds, 60))
        r2.raise_for_status()

        # Write atomically
        self.adm1_path.write_bytes(r1.content)
        self.adm2_path.write_bytes(r2.content)

        gdf1 = gpd.read_file(self.adm1_path)
        gdf2 = gpd.read_file(self.adm2_path)
        index_payload = build_admin_index_from_geodataframes(
            gdf1,
            gdf2,
            adm1_source=f"{meta1.get('boundarySource', 'geoBoundaries ADM1')} ({meta1.get('boundaryLicense', '')})".strip(),
            adm2_source=f"{meta2.get('boundarySource', 'geoBoundaries ADM2')} ({meta2.get('boundaryLicense', '')})".strip(),
        )
        self.index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
        self._index_cache = index_payload
        self._adm1_gj = None
        self._adm2_gj = None
        logger.info(
            "Cached India boundaries: %d states/UTs, %d districts.",
            index_payload["state_count"],
            index_payload["district_count"],
        )
        return self.get_boundary_status()

    def load_index(self, auto_download: bool = False) -> dict[str, Any] | None:
        """Load cached administrative index from disk, optionally downloading if absent."""
        if self._index_cache is not None:
            return self._index_cache
        if self.index_path.exists():
            self._index_cache = json.loads(self.index_path.read_text(encoding="utf-8"))
            return self._index_cache
        if auto_download:
            try:
                self.download_and_cache()
                return self._index_cache
            except Exception as exc:
                logger.warning("Auto-download of India boundaries failed: %s", exc)
        return None

    def get_boundary_status(self) -> dict[str, Any]:
        """Report local availability of administrative index & boundary polygons."""
        idx = self.load_index(auto_download=False)
        polygons_ready = self.adm1_path.exists() and self.adm2_path.exists()
        if idx is None:
            return {
                "status": "Unavailable",
                "state_count": 0,
                "district_count": 0,
                "polygons_cached": polygons_ready,
                "source": BOUNDARY_ATTRIBUTION,
                "last_updated": None,
            }
        return {
            "status": "Loaded",
            "state_count": int(idx.get("state_count", len(idx.get("states", [])))),
            "district_count": int(idx.get("district_count", len(idx.get("districts", [])))),
            "polygons_cached": polygons_ready,
            "source": idx.get("attribution", BOUNDARY_ATTRIBUTION),
            "last_updated": idx.get("generated_at"),
        }

    def list_states(self, query: str | None = None) -> list[dict[str, Any]]:
        """Return States & Union Territories sorted alphabetically, optionally filtered by query."""
        idx = self.load_index(auto_download=False)
        if not idx:
            return []
        states = idx.get("states", [])
        if query:
            q = normalize_admin_name(query).lower()
            states = [
                s for s in states
                if q in s["search_name"] or q in normalize_admin_name(s.get("official_name")).lower()
            ]
        return list(states)

    def list_districts(
        self,
        state_name: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return Districts sorted alphabetically, optionally filtered by parent State and search query."""
        idx = self.load_index(auto_download=False)
        if not idx:
            return []
        dists = idx.get("districts", [])
        if state_name:
            target_s = normalize_admin_name(state_name).lower()
            dists = [
                d for d in dists
                if d["state_name"].lower() == target_s
                or normalize_admin_name(d.get("state_official_name")).lower() == target_s
            ]
        if query:
            q = normalize_admin_name(query).lower()
            dists = [
                d for d in dists
                if q in d["search_name"] or q in normalize_admin_name(d.get("official_name")).lower()
            ]
        return list(dists)

    def resolve_location(
        self,
        state_name: str,
        district_name: str,
    ) -> dict[str, Any] | None:
        """Resolve exact district metadata (coordinates, bbox, official name) under a state."""
        dists = self.list_districts(state_name=state_name)
        target_d = normalize_admin_name(district_name).lower()
        for d in dists:
            if d["district_name"].lower() == target_d:
                return dict(d)
        for d in dists:
            if target_d in d["search_name"]:
                return dict(d)
        return None

    def get_state_geojson(self, state_name: str) -> dict[str, Any] | None:
        """Return single-feature GeoJSON FeatureCollection for the selected State/UT polygon."""
        if not self.adm1_path.exists():
            return None
        if self._adm1_gj is None:
            self._adm1_gj = json.loads(self.adm1_path.read_text(encoding="utf-8"))
        target = normalize_admin_name(state_name).lower()
        matched = [
            feat for feat in self._adm1_gj.get("features", [])
            if normalize_admin_name((feat.get("properties") or {}).get("shapeName")).lower() == target
        ]
        if not matched:
            return None
        return {"type": "FeatureCollection", "features": matched}

    def get_district_geojson(self, state_name: str, district_name: str) -> dict[str, Any] | None:
        """Return single-feature GeoJSON FeatureCollection for the selected District polygon."""
        if not self.adm2_path.exists():
            return None
        resolved = self.resolve_location(state_name, district_name)
        if not resolved:
            return None
        if self._adm2_gj is None:
            self._adm2_gj = json.loads(self.adm2_path.read_text(encoding="utf-8"))
        target_id = resolved.get("shape_id")
        target_name = normalize_admin_name(district_name).lower()
        feats = self._adm2_gj.get("features", [])
        matched = []
        if target_id:
            matched = [
                f for f in feats
                if str((f.get("properties") or {}).get("shapeID") or "") == str(target_id)
            ]
        if not matched:
            matched = [
                f for f in feats
                if normalize_admin_name((f.get("properties") or {}).get("shapeName")).lower() == target_name
            ]
        if not matched:
            return None
        return {"type": "FeatureCollection", "features": matched}

