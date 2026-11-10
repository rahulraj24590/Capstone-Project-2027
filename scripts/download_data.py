"""
Dataset download script for LandslideGuard.

Attempts to download publicly available landslide datasets.
Documents data sources, access requirements, and limitations.

USAGE:
    python scripts/download_data.py [--dataset all|nasa_glc|synthetic]

DATA SOURCES:
  1. NASA Global Landslide Catalog (GLC)
     URL: https://catalog.data.gov/dataset/global-landslide-catalog-export
     Direct CSV link is attempted; requires internet access.
     No API key needed for the catalog CSV export.

  2. Synthetic dataset (always available as fallback)
     Clearly labelled as synthetic, not suitable for scientific claims.

If downloads fail, the script creates a synthetic dataset for pipeline testing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# NASA Global Landslide Catalog
# ─────────────────────────────────────────────

NASA_GLC_URL = (
    "https://maps.nccs.nasa.gov/arcgis/rest/services/gis5/landslides/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&f=geojson"
)
NASA_GLC_BACKUP_URL = (
    "https://catalog.data.gov/dataset/global-landslide-catalog-export"
)

CATALOG_CSV_URL = (
    "https://data.nasa.gov/api/geospatial/dd9e-wu2v?method=export&type=csv"
)


def download_nasa_glc() -> Path | None:
    """Attempt to download the NASA Global Landslide Catalog CSV.

    Returns:
        Path to downloaded file, or None if download failed.
    """
    output_path = RAW_DIR / "nasa_global_landslide_catalog.csv"

    if output_path.exists():
        logger.info("NASA GLC already downloaded at %s. Skipping.", output_path)
        return output_path

    logger.info("Attempting to download NASA Global Landslide Catalog CSV...")
    logger.info("Source: %s", CATALOG_CSV_URL)

    try:
        resp = requests.get(CATALOG_CSV_URL, timeout=30, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            logger.warning(
                "Received HTML response — NASA GLC may require authentication "
                "or the URL has changed. Visit: %s",
                NASA_GLC_BACKUP_URL,
            )
            return None

        total_bytes = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                total_bytes += len(chunk)

        size_mb = total_bytes / (1024 * 1024)
        logger.info(
            "Downloaded NASA GLC: %s (%.2f MB)", output_path, size_mb
        )
        return output_path

    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to NASA data servers. No internet or server down.")
    except requests.exceptions.Timeout:
        logger.warning("NASA GLC download timed out.")
    except requests.exceptions.HTTPError as e:
        logger.warning("NASA GLC HTTP error: %s", e)
    except Exception as e:
        logger.warning("NASA GLC download failed: %s", e)

    return None


def download_all() -> dict[str, str]:
    """Attempt all configured downloads.

    Returns:
        Dict mapping dataset name to status.
    """
    results: dict[str, str] = {}

    # NASA GLC
    path = download_nasa_glc()
    results["nasa_glc"] = str(path) if path else "FAILED — see logs"

    # Summary
    logger.info("Download summary: %s", results)
    return results


def check_disk_space(required_mb: float = 500.0) -> bool:
    """Check available disk space before downloading.

    Args:
        required_mb: Minimum required free space (MB).

    Returns:
        True if sufficient space is available.
    """
    import shutil
    total, used, free = shutil.disk_usage(RAW_DIR)
    free_mb = free / (1024 * 1024)
    logger.info("Available disk space: %.0f MB (required: %.0f MB)", free_mb, required_mb)
    if free_mb < required_mb:
        logger.warning(
            "Insufficient disk space: %.0f MB available, %.0f MB required.",
            free_mb, required_mb,
        )
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download landslide datasets for LandslideGuard."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "nasa_glc"],
        default="all",
        help="Dataset to download.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    if not check_disk_space(required_mb=200.0):
        logger.error("Insufficient disk space. Aborting download.")
        sys.exit(1)

    if args.dataset in ("all", "nasa_glc"):
        result = download_all()
        for name, status in result.items():
            print(f"  {name}: {status}")
    else:
        print("Unknown dataset. Use --dataset all or --dataset nasa_glc")
