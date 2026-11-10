"""
Synthetic dataset generator for LandslideGuard.

Generates a clearly labelled synthetic dataset for pipeline testing
and dashboard demonstration.

IMPORTANT:
- This dataset uses plausible statistical distributions, not real observations.
- It must NEVER be used to claim real-world predictive performance.
- All generated records are flagged with is_synthetic=True.
- The label distribution is set artificially and does not reflect
  actual landslide frequency in any region.

Methodology:
- Positive (landslide) samples have higher slope, higher rainfall,
  and lower vegetation index on average — consistent with published
  landslide susceptibility literature (not derived from this data).
- Negative (no-landslide) samples have lower slope and rainfall.
- Gaussian noise is added to avoid perfect separability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.logging_config import get_logger, setup_logging
from src.utils.settings import get_setting

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_CFG = get_setting("synthetic_data") or {}
N_SAMPLES = int(SYNTHETIC_CFG.get("n_samples", 2000))
LAT_RANGE = SYNTHETIC_CFG.get("lat_range", [8.0, 35.0])
LON_RANGE = SYNTHETIC_CFG.get("lon_range", [72.0, 97.0])
RANDOM_SEED = int(SYNTHETIC_CFG.get("random_seed", 42))
POSITIVE_FRACTION = float(SYNTHETIC_CFG.get("positive_fraction", 0.25))

# Land types and soil types for variety
LAND_COVER_TYPES = ["Forest", "Shrubland", "Grassland", "Cropland", "Bare"]
SOIL_TYPES = ["Clay", "Sandy Loam", "Loam", "Silty Clay", "Rock"]
LANDSLIDE_TYPES = ["debris flow", "rockfall", "translational slide", "rotational slide"]
TRIGGERS = ["rain", "earthquake", "rain", "rain", "construction"]  # rain is most common


def generate_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    positive_fraction: float = POSITIVE_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic landslide dataset.

    Args:
        n_samples: Total number of samples.
        positive_fraction: Fraction of samples that are landslide events.
        random_seed: NumPy random seed.

    Returns:
        DataFrame with synthetic data, clearly marked.
    """
    rng = np.random.default_rng(random_seed)
    n_positive = int(n_samples * positive_fraction)
    n_negative = n_samples - n_positive

    logger.info(
        "Generating synthetic dataset: %d total (%d positive, %d negative).",
        n_samples, n_positive, n_negative,
    )

    # ── Positive samples (landslide occurred) ─────────────────────────
    pos = {
        "latitude": rng.uniform(LAT_RANGE[0], LAT_RANGE[1], n_positive),
        "longitude": rng.uniform(LON_RANGE[0], LON_RANGE[1], n_positive),
        # Steeper slopes for positive events
        "slope_degrees": np.clip(rng.normal(32, 8, n_positive), 5, 80),
        # Higher elevations (hilly terrain)
        "elevation_m": np.clip(rng.normal(900, 400, n_positive), 100, 5000),
        # Higher rainfall for positive events
        "rainfall_1h_mm": np.clip(rng.exponential(18, n_positive), 0, 200),
        "rainfall_24h_mm": np.clip(rng.exponential(80, n_positive), 0, 800),
        "rainfall_72h_mm": np.clip(rng.exponential(150, n_positive), 0, 2000),
        # Lower vegetation for positive events (degraded land)
        "vegetation_index": np.clip(rng.normal(0.25, 0.15, n_positive), -0.1, 0.8),
        # Higher soil moisture for positive events
        "soil_moisture": np.clip(rng.normal(0.65, 0.15, n_positive), 0.1, 1.0),
        "landslide_occurred": np.ones(n_positive, dtype=int),
        "is_synthetic": np.ones(n_positive, dtype=int),
    }

    # ── Negative samples (no landslide) ───────────────────────────────
    neg = {
        "latitude": rng.uniform(LAT_RANGE[0], LAT_RANGE[1], n_negative),
        "longitude": rng.uniform(LON_RANGE[0], LON_RANGE[1], n_negative),
        # Gentler slopes
        "slope_degrees": np.clip(rng.normal(12, 8, n_negative), 0, 50),
        # Lower elevations (plains)
        "elevation_m": np.clip(rng.normal(350, 300, n_negative), 0, 3000),
        # Lower rainfall
        "rainfall_1h_mm": np.clip(rng.exponential(5, n_negative), 0, 100),
        "rainfall_24h_mm": np.clip(rng.exponential(25, n_negative), 0, 300),
        "rainfall_72h_mm": np.clip(rng.exponential(50, n_negative), 0, 600),
        # Higher vegetation (stable land)
        "vegetation_index": np.clip(rng.normal(0.55, 0.15, n_negative), 0.0, 1.0),
        # Lower soil moisture
        "soil_moisture": np.clip(rng.normal(0.35, 0.15, n_negative), 0.0, 0.9),
        "landslide_occurred": np.zeros(n_negative, dtype=int),
        "is_synthetic": np.ones(n_negative, dtype=int),
    }

    df_pos = pd.DataFrame(pos)
    df_neg = pd.DataFrame(neg)
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Add categorical features
    df["land_cover"] = rng.choice(LAND_COVER_TYPES, size=len(df))
    df["soil_type"] = rng.choice(SOIL_TYPES, size=len(df))

    # Add temporal features (spread across 10 years, monsoon-weighted)
    # Positive events more likely in monsoon months (June–September)
    months = np.empty(len(df), dtype=int)
    pos_mask = df["landslide_occurred"] == 1
    months[pos_mask] = rng.choice(
        [6, 7, 8, 9, 6, 7, 8, 9, 1, 2, 3, 4, 5, 10, 11, 12],
        size=pos_mask.sum(),
    )
    months[~pos_mask] = rng.integers(1, 13, size=(~pos_mask).sum())

    # Create event_date strings using the monsoon-weighted months
    years_for_dates = rng.integers(2014, 2024, size=len(df))
    days = rng.integers(1, 28, size=len(df))
    df["event_date"] = pd.to_datetime(
        dict(year=years_for_dates, month=months, day=days)
    ).dt.strftime("%Y-%m-%d")

    # Add metadata fields
    df["source_name"] = "SYNTHETIC_TEST_DATA"
    df["country"] = "India"
    df["landslide_type"] = np.where(
        df["landslide_occurred"] == 1,
        rng.choice(LANDSLIDE_TYPES, size=len(df)),
        None,
    )
    df["trigger"] = np.where(
        df["landslide_occurred"] == 1,
        rng.choice(TRIGGERS, size=len(df)),
        None,
    )
    df["fatalities"] = np.where(
        df["landslide_occurred"] == 1,
        rng.integers(0, 50, size=len(df)),
        0,
    )

    # Ensure rainfall_72h >= rainfall_24h >= rainfall_1h (approximately)
    df["rainfall_24h_mm"] = np.maximum(df["rainfall_24h_mm"], df["rainfall_1h_mm"])
    df["rainfall_72h_mm"] = np.maximum(df["rainfall_72h_mm"], df["rainfall_24h_mm"])

    logger.info(
        "Synthetic dataset generated: %d rows, positive rate=%.2f%%",
        len(df), df["landslide_occurred"].mean() * 100,
    )

    return df


def save_synthetic_dataset(df: pd.DataFrame) -> Path:
    """Save the synthetic dataset to data/processed/dataset.parquet.

    Args:
        df: Synthetic DataFrame.

    Returns:
        Path to saved file.
    """
    output_path = PROCESSED_DIR / "dataset.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Saved synthetic dataset to %s (%d rows).", output_path, len(df))

    # Also save a CSV copy for inspection
    csv_path = PROCESSED_DIR / "dataset_sample.csv"
    df.head(100).to_csv(csv_path, index=False)
    logger.info("Saved first 100 rows to %s for inspection.", csv_path)

    # Write provenance note
    note_path = PROCESSED_DIR / "SYNTHETIC_DATA_WARNING.txt"
    with open(note_path, "w") as f:
        f.write(
            "WARNING: SYNTHETIC TEST DATA\n"
            "================================\n\n"
            "The dataset in this directory is SYNTHETIC and was generated\n"
            "by scripts/generate_synthetic_data.py for pipeline testing only.\n\n"
            "It does NOT represent real landslide observations.\n"
            "It MUST NOT be used to claim real-world predictive performance.\n\n"
            f"Generated: {pd.Timestamp.now().isoformat()}\n"
            f"N samples: {len(df)}\n"
            f"Positive fraction: {df['landslide_occurred'].mean():.4f}\n"
        )

    return output_path


if __name__ == "__main__":
    setup_logging()
    df = generate_synthetic_dataset()
    path = save_synthetic_dataset(df)
    print(f"\n✅ Synthetic dataset saved to: {path}")
    print(f"   Rows: {len(df)}")
    print(f"   Positive: {df['landslide_occurred'].sum()} ({df['landslide_occurred'].mean():.1%})")
    print(f"\n⚠️  THIS IS SYNTHETIC DATA — NOT suitable for scientific claims.")
