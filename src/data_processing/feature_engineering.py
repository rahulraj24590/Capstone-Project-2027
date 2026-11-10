"""
Feature engineering for LandslideGuard.

Derives additional features from raw observations that are physically
meaningful for landslide susceptibility and dynamic risk estimation.

All derived features are documented with their data requirements and
limitations. Features that cannot be reliably derived are not computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Terrain-derived features
# ─────────────────────────────────────────────

def compute_slope_class(df: pd.DataFrame) -> pd.DataFrame:
    """Classify slope into discrete stability classes.

    Classification based on common geotechnical literature:
    - Flat (0–5°): Very low instability
    - Gentle (5–15°): Low instability
    - Moderate (15–30°): Moderate instability
    - Steep (30–45°): High instability
    - Very steep (45°+): Very high instability

    Args:
        df: DataFrame containing 'slope_degrees'.

    Returns:
        DataFrame with 'slope_class' (int 0–4) column added.
    """
    if "slope_degrees" not in df.columns:
        logger.warning("Cannot compute slope_class: 'slope_degrees' missing.")
        return df

    df = df.copy()
    bins = [0, 5, 15, 30, 45, 91]
    labels = [0, 1, 2, 3, 4]
    df["slope_class"] = pd.cut(
        df["slope_degrees"].clip(0, 90),
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(float)

    logger.debug("Computed slope_class for %d rows.", len(df))
    return df


def compute_elevation_band(df: pd.DataFrame) -> pd.DataFrame:
    """Classify elevation into broad topographic bands.

    Args:
        df: DataFrame containing 'elevation_m'.

    Returns:
        DataFrame with 'elevation_band' column.
    """
    if "elevation_m" not in df.columns:
        logger.warning("Cannot compute elevation_band: 'elevation_m' missing.")
        return df

    df = df.copy()
    bins = [-500, 0, 200, 500, 1000, 2000, 4000, 9001]
    labels = [0, 1, 2, 3, 4, 5, 6]
    df["elevation_band"] = pd.cut(
        df["elevation_m"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(float)

    return df


# ─────────────────────────────────────────────
# Rainfall-derived features
# ─────────────────────────────────────────────

def compute_rainfall_intensity_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio of short-term to medium-term rainfall.

    A high ratio of 1-hour to 24-hour rainfall indicates an intense
    convective event rather than prolonged saturation.

    LIMITATION: Only computed when both columns are non-null and
    rainfall_24h_mm > 0 to avoid division by zero.

    Args:
        df: DataFrame with rainfall columns.

    Returns:
        DataFrame with 'rainfall_intensity_ratio' column.
    """
    if "rainfall_1h_mm" not in df.columns or "rainfall_24h_mm" not in df.columns:
        return df

    df = df.copy()
    safe_24h = df["rainfall_24h_mm"].replace(0, np.nan)
    df["rainfall_intensity_ratio"] = df["rainfall_1h_mm"] / safe_24h
    # Cap at 1.0 — ratio > 1 indicates data inconsistency
    df["rainfall_intensity_ratio"] = df["rainfall_intensity_ratio"].clip(0, 1)
    logger.debug("Computed rainfall_intensity_ratio.")
    return df


def compute_antecedent_saturation_index(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate antecedent soil saturation from multi-day rainfall.

    Uses a weighted sum of rainfall accumulations as a proxy for
    antecedent moisture. This is NOT equivalent to a physics-based
    soil moisture model.

    Weights (approximate, based on memory effect literature):
    - 72h rainfall contributes most to pre-event saturation.
    - 24h rainfall reflects recent event intensity.

    Args:
        df: DataFrame with rainfall columns.

    Returns:
        DataFrame with 'antecedent_saturation_index' column.
    """
    if "rainfall_72h_mm" not in df.columns:
        return df

    df = df.copy()
    asi = df.get("rainfall_72h_mm", pd.Series(np.nan, index=df.index))
    if "rainfall_24h_mm" in df.columns:
        asi = asi * 0.6 + df["rainfall_24h_mm"] * 0.4

    df["antecedent_saturation_index"] = asi
    logger.debug("Computed antecedent_saturation_index.")
    return df


def compute_rainfall_anomaly(
    df: pd.DataFrame,
    mean_24h: float | None = None,
    std_24h: float | None = None,
) -> pd.DataFrame:
    """Compute standardised 24-hour rainfall anomaly.

    Measures how extreme the current rainfall is relative to the
    dataset distribution. Requires sufficient data to estimate
    meaningful mean and standard deviation.

    Args:
        df: DataFrame with 'rainfall_24h_mm'.
        mean_24h: Pre-computed mean (use training set statistics for test set).
        std_24h: Pre-computed standard deviation.

    Returns:
        DataFrame with 'rainfall_24h_anomaly' column.
    """
    if "rainfall_24h_mm" not in df.columns:
        return df

    df = df.copy()
    col = df["rainfall_24h_mm"].dropna()
    if col.empty:
        return df

    mean = mean_24h if mean_24h is not None else col.mean()
    std = std_24h if std_24h is not None else col.std()

    if std == 0 or np.isnan(std):
        logger.warning("Rainfall std is 0 or NaN; anomaly set to 0.")
        df["rainfall_24h_anomaly"] = 0.0
    else:
        df["rainfall_24h_anomaly"] = (df["rainfall_24h_mm"] - mean) / std

    return df


# ─────────────────────────────────────────────
# Temporal features
# ─────────────────────────────────────────────

def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from 'event_date' column.

    Derives:
    - event_month (1–12): For seasonal pattern analysis.
    - event_season (DJF/MAM/JJA/SON): Northern-hemisphere meteorological seasons.
    - monsoon_season (bool): June–September in South Asian context.

    Args:
        df: DataFrame with 'event_date' as datetime column.

    Returns:
        DataFrame with temporal feature columns added.
    """
    if "event_date" not in df.columns:
        return df

    df = df.copy()
    dt = pd.to_datetime(df["event_date"], errors="coerce")

    df["event_month"] = dt.dt.month
    df["event_year"] = dt.dt.year

    def get_season(month):
        if pd.isna(month):
            return np.nan
        m = int(month)
        if m in (12, 1, 2):
            return 0  # DJF
        elif m in (3, 4, 5):
            return 1  # MAM
        elif m in (6, 7, 8):
            return 2  # JJA
        else:
            return 3  # SON

    df["event_season"] = df["event_month"].map(get_season)

    # South Asian monsoon: June–September
    df["in_monsoon_season"] = df["event_month"].isin([6, 7, 8, 9]).astype(float)
    df.loc[dt.isna(), "in_monsoon_season"] = np.nan

    logger.debug("Computed temporal features.")
    return df


# ─────────────────────────────────────────────
# Interaction features
# ─────────────────────────────────────────────

def compute_rainfall_slope_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Compute slope × rainfall interaction term.

    This multiplicative feature captures the joint effect of steep terrain
    and high rainfall — a commonly cited combination in landslide literature.
    It is a feature-engineering proxy, NOT a physics-based model.

    Args:
        df: DataFrame with 'slope_degrees' and 'rainfall_24h_mm'.

    Returns:
        DataFrame with 'slope_rainfall_interaction' column.
    """
    if "slope_degrees" not in df.columns or "rainfall_24h_mm" not in df.columns:
        return df

    df = df.copy()
    df["slope_rainfall_interaction"] = df["slope_degrees"] * df["rainfall_24h_mm"]
    return df


# ─────────────────────────────────────────────
# Full feature engineering pipeline
# ─────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    training_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run all feature engineering steps.

    Args:
        df: Preprocessed DataFrame.
        training_stats: Statistics from training set (for test/inference sets).
                        If None, statistics are computed from df (training mode).

    Returns:
        Tuple of (feature-engineered DataFrame, computed statistics dict).
    """
    stats: dict = training_stats or {}

    df = compute_slope_class(df)
    df = compute_elevation_band(df)
    df = compute_rainfall_intensity_ratio(df)
    df = compute_antecedent_saturation_index(df)
    df = compute_temporal_features(df)
    df = compute_rainfall_slope_interaction(df)

    # Compute or apply rainfall anomaly statistics
    if "rainfall_24h_mm" in df.columns:
        mean_24h = stats.get("rainfall_24h_mean")
        std_24h = stats.get("rainfall_24h_std")
        df = compute_rainfall_anomaly(df, mean_24h=mean_24h, std_24h=std_24h)
        if not training_stats:
            col_vals = df["rainfall_24h_mm"].dropna()
            stats["rainfall_24h_mean"] = float(col_vals.mean()) if not col_vals.empty else 0.0
            stats["rainfall_24h_std"] = float(col_vals.std()) if not col_vals.empty else 1.0

    logger.info("Feature engineering complete. Columns: %s", list(df.columns))
    return df, stats


# ─────────────────────────────────────────────
# Feature column list (used by model pipeline)
# ─────────────────────────────────────────────

CORE_FEATURE_COLUMNS = [
    "slope_degrees",
    "elevation_m",
    "rainfall_24h_mm",
    "rainfall_72h_mm",
]

EXTENDED_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + [
    "rainfall_1h_mm",
    "vegetation_index",
    "soil_moisture",
    "slope_class",
    "elevation_band",
    "rainfall_intensity_ratio",
    "antecedent_saturation_index",
    "slope_rainfall_interaction",
    "event_month",
    "event_season",
    "in_monsoon_season",
    "soil_type_encoded",
    "land_cover_encoded",
]


def select_available_features(
    df: pd.DataFrame,
    desired: list[str] = None,
) -> list[str]:
    """Return subset of desired feature columns that are present in df.

    Args:
        df: DataFrame to check.
        desired: List of feature column names to include (default: EXTENDED).

    Returns:
        List of available feature column names.
    """
    if desired is None:
        desired = EXTENDED_FEATURE_COLUMNS
    available = [c for c in desired if c in df.columns]
    missing = [c for c in desired if c not in df.columns]
    if missing:
        logger.info("Feature columns not available (will be skipped): %s", missing)
    return available
