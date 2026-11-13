"""Download and cache India State/UT (ADM1) and District (ADM2) boundaries.

Populates data/india_boundaries/{adm1_states.geojson, adm2_districts.geojson,
india_districts_index.json} and synchronizes the SQLite india_states and
india_districts tables cleanly without duplication.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_providers.india_boundaries import IndiaBoundaryProvider
from src.database.db import initialize_database
from src.database.models import sync_india_boundaries, upsert_provider_status
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and cache India administrative boundaries.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached files exist.")
    args = parser.parse_args()

    initialize_database()
    provider = IndiaBoundaryProvider()
    status = provider.download_and_cache(force=args.force)
    idx = provider.load_index(auto_download=False) or {}

    sync_res = sync_india_boundaries(
        states=idx.get("states", []),
        districts=idx.get("districts", []),
        boundary_source=status.get("source", "geoBoundaries IND ADM1/ADM2"),
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    upsert_provider_status(
        provider_name="Geographical Boundaries",
        status=status.get("status", "Cached"),
        is_live=bool(args.force),
        is_cached=True,
        last_success_at=now_iso,
        data_timestamp=status.get("last_updated"),
        attribution=status.get("source"),
        message=f"{sync_res['states_synced']} states/UTs and {sync_res['districts_synced']} districts synchronized.",
    )
    print(
        f"Boundaries ready: {status.get('state_count')} states/UTs, "
        f"{status.get('district_count')} districts "
        f"(DB synced: {sync_res['states_synced']} states, {sync_res['districts_synced']} districts)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

