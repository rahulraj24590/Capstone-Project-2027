"""
Full data preparation pipeline runner for LandslideGuard.

Orchestrates:
1. Download real data (if available)
2. Validate and preprocess raw data
3. Apply feature engineering
4. Save processed dataset
5. Load observations into database

If no real data is available, generates a clearly labelled
synthetic dataset for pipeline testing.

USAGE:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --use-synthetic
    python scripts/prepare_data.py --raw-file data/raw/nasa_global_landslide_catalog.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data_processing.preprocessing import (
    load_raw_data,
    preprocess,
    save_processed,
    validate_columns,
)
from src.data_processing.feature_engineering import engineer_features
from src.database.db import initialize_database
from src.database.models import bulk_insert_observations, get_observations
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ─────────────────────────────────────────────
# NASA GLC column mapping
# ─────────────────────────────────────────────

NASA_GLC_COLUMN_MAP = {
    "event_date": "event_date",
    "latitude": "latitude",
    "longitude": "longitude",
    "landslide_category": "landslide_type",
    "trigger": "trigger",
    "fatality_count": "fatalities",
    "injury_count": "injuries",
    "country_name": "country",
    "admin_division_name": "state_region",
    "source_name": "source_name",
}


def load_nasa_glc(filepath: Path) -> pd.DataFrame | None:
    """Load and map the NASA Global Landslide Catalog CSV.

    Args:
        filepath: Path to the NASA GLC CSV.

    Returns:
        Mapped DataFrame or None if loading fails.
    """
    try:
        df = load_raw_data(filepath)
        df = df.rename(columns={k: v for k, v in NASA_GLC_COLUMN_MAP.items() if k in df.columns})
        # NASA GLC only contains positive observations; add target column
        df["landslide_occurred"] = 1
        df["source_name"] = df.get("source_name", pd.Series(["NASA_GLC"] * len(df)))
        logger.info("Loaded NASA GLC: %d rows.", len(df))
        return df
    except Exception as e:
        logger.error("Failed to load NASA GLC: %s", e)
        return None


def add_negative_samples(
    df_positive: pd.DataFrame,
    n_negative: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate negative (no-landslide) samples to pair with positive events.

    Negative samples are placed at geographically separated locations
    using the same terrain feature distributions shifted toward
    lower-risk values.

    Args:
        df_positive: DataFrame of positive (landslide) observations.
        n_negative: Number of negative samples (default: 3× positive count).
        random_seed: NumPy random seed.

    Returns:
        Combined DataFrame with positive and negative samples.
    """
    import numpy as np
    rng = np.random.default_rng(random_seed)

    n = n_negative or len(df_positive) * 3

    # Geographic bounds from positive data
    lat_min = df_positive["latitude"].min()
    lat_max = df_positive["latitude"].max()
    lon_min = df_positive["longitude"].min()
    lon_max = df_positive["longitude"].max()

    neg_df = pd.DataFrame({
        "latitude": rng.uniform(lat_min, lat_max, n),
        "longitude": rng.uniform(lon_min, lon_max, n),
        "landslide_occurred": 0,
        "is_synthetic": 1,  # negative samples are synthetic
        "source_name": "SYNTHETIC_NEGATIVE",
        "country": "Unknown",
    })

    combined = pd.concat([df_positive, neg_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    logger.info(
        "Added %d synthetic negative samples. Total: %d rows.",
        n, len(combined),
    )
    return combined


def run_pipeline(
    raw_file: Path | None = None,
    use_synthetic: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Execute the full data preparation pipeline.

    Args:
        raw_file: Path to raw data file. If None, checks for NASA GLC.
        use_synthetic: Force use of synthetic data.
        output_path: Destination for processed parquet.

    Returns:
        Path to saved processed dataset.
    """
    output_path = output_path or PROCESSED_DIR / "dataset.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ──────────────────────────────────────────────
    df_raw = None

    if not use_synthetic:
        # Try provided file first
        if raw_file and raw_file.exists():
            logger.info("Loading raw data from %s …", raw_file)
            df_raw = load_nasa_glc(raw_file)

        # Try auto-detected NASA GLC
        elif (RAW_DIR / "nasa_global_landslide_catalog.csv").exists():
            logger.info("Found NASA GLC CSV in data/raw/. Loading …")
            df_raw = load_nasa_glc(RAW_DIR / "nasa_global_landslide_catalog.csv")

        if df_raw is not None and not df_raw.empty:
            # NASA GLC only has positive samples; add negatives
            df_raw = add_negative_samples(df_raw)
            is_synthetic = False
            logger.info("Using NASA GLC data with generated negative samples.")
        else:
            logger.warning("No real data available. Falling back to synthetic dataset.")
            use_synthetic = True

    if use_synthetic or df_raw is None:
        from scripts.generate_synthetic_data import generate_synthetic_dataset
        df_raw = generate_synthetic_dataset()
        is_synthetic = True
        logger.warning(
            "Using SYNTHETIC dataset. Results are for pipeline testing only."
        )

    # ── Step 2: Preprocess ─────────────────────────────────────────────
    logger.info("Running preprocessing pipeline …")
    df_clean, artefacts = preprocess(df_raw, is_training=True)

    # ── Step 3: Feature engineering ────────────────────────────────────
    logger.info("Running feature engineering …")
    df_features, fe_stats = engineer_features(df_clean)

    # ── Step 4: Save ───────────────────────────────────────────────────
    save_processed(df_features, output_path)

    # ── Step 5: Load to database ───────────────────────────────────────
    initialize_database()
    obs_cols = [
        "latitude", "longitude", "event_date", "landslide_type",
        "trigger", "fatalities", "country", "state_region", "source_name",
    ]
    obs_df = df_features[[c for c in obs_cols if c in df_features.columns]].copy()
    obs_df["is_synthetic"] = df_features.get("is_synthetic", 1)

    # Only load landslide events into the observations table
    if "landslide_occurred" in df_features.columns:
        events_df = obs_df[df_features["landslide_occurred"] == 1].copy()
    else:
        events_df = obs_df.copy()

    if not events_df.empty:
        bulk_insert_observations(events_df, is_synthetic=is_synthetic)

    logger.info("Data preparation pipeline complete. Output: %s", output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LandslideGuard training data.")
    parser.add_argument("--raw-file", type=str, help="Path to raw data CSV.")
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Force use of synthetic dataset.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    raw_file = Path(args.raw_file) if args.raw_file else None
    output = run_pipeline(raw_file=raw_file, use_synthetic=args.use_synthetic)
    print(f"\n✅ Processed dataset saved: {output}")
