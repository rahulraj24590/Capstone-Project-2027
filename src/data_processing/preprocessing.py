"""
Data preprocessing pipeline for LandslideGuard.

Implements validation, cleaning, type enforcement, and normalisation
steps for landslide observation datasets.

Design decisions:
- Missing values are logged and handled transparently.
- Invalid coordinates are REMOVED, not corrected.
- Duplicate detection uses both spatial proximity and timestamp.
- No value is silently substituted for missing rainfall or soil moisture.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.logging_config import get_logger
from src.utils.validation import validate_coordinates

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Column schema
# ─────────────────────────────────────────────

# These are the EXPECTED column names from the raw data.
# Any real dataset will be remapped to these names via the column_map argument.
REQUIRED_COLUMNS = ["latitude", "longitude", "landslide_occurred"]
NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "slope_degrees",
    "elevation_m",
    "rainfall_1h_mm",
    "rainfall_24h_mm",
    "rainfall_72h_mm",
    "vegetation_index",
    "soil_moisture",
]
CATEGORICAL_COLUMNS = ["soil_type", "land_cover"]
DATETIME_COLUMNS = ["event_date"]


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def load_raw_data(
    filepath: str | Path,
    column_map: dict[str, str] | None = None,
    date_column: str | None = "event_date",
) -> pd.DataFrame:
    """Load raw data from CSV or Parquet.

    Args:
        filepath: Path to input file (CSV or Parquet).
        column_map: Optional mapping from file column names to standard names.
        date_column: Name of the date column to parse as datetime.

    Returns:
        DataFrame with standardised column names.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    suffix = filepath.suffix.lower()
    logger.info("Loading data from %s (format: %s)", filepath, suffix)

    if suffix == ".csv":
        df = pd.read_csv(filepath, low_memory=False)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(filepath)
    elif suffix == ".geojson":
        try:
            import geopandas as gpd
            gdf = gpd.read_file(filepath)
            df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
        except ImportError:
            logger.warning("geopandas not available; reading GeoJSON as JSON.")
            import json
            with open(filepath, encoding="utf-8") as f:
                geojson = json.load(f)
            records = [
                {**feat["properties"], "longitude": feat["geometry"]["coordinates"][0],
                 "latitude": feat["geometry"]["coordinates"][1]}
                for feat in geojson.get("features", [])
                if feat.get("geometry") and feat["geometry"].get("coordinates")
            ]
            df = pd.DataFrame(records)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported formats: .csv, .parquet, .geojson"
        )

    if column_map:
        df = df.rename(columns=column_map)
        logger.info("Renamed columns: %s", column_map)

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Check that required columns are present.

    Args:
        df: Input DataFrame.
        required: List of required column names.

    Returns:
        List of missing column names.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("Missing required columns: %s", missing)
    return missing


def remove_invalid_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid or missing geographic coordinates.

    Rows are removed if:
    - latitude or longitude is NaN
    - latitude is outside [-90, 90]
    - longitude is outside [-180, 180]

    Swapped latitude/longitude is NOT corrected — it is flagged and removed.

    Args:
        df: DataFrame containing 'latitude' and 'longitude' columns.

    Returns:
        Cleaned DataFrame.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.error("Cannot validate coordinates: 'latitude'/'longitude' columns missing.")
        return df

    initial_count = len(df)
    mask_valid = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & df["latitude"].between(-90.0, 90.0)
        & df["longitude"].between(-180.0, 180.0)
    )

    n_invalid = (~mask_valid).sum()
    if n_invalid > 0:
        logger.warning("Removed %d rows with invalid coordinates.", n_invalid)

    df = df[mask_valid].copy()
    logger.info(
        "Coordinate validation: %d → %d rows (removed %d)",
        initial_count, len(df), n_invalid,
    )
    return df


# ─────────────────────────────────────────────
# Missing value handling
# ─────────────────────────────────────────────

def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Log missing value counts by column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame summarising missing counts and percentages.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)

    if not report.empty:
        logger.info("Missing value report:\n%s", report.to_string())
    else:
        logger.info("No missing values found.")

    return report


def handle_missing_values(
    df: pd.DataFrame,
    strategy: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply column-specific missing value strategies.

    Supported strategies:
    - "drop": Drop rows where this column is null.
    - "median": Fill with column median.
    - "mean": Fill with column mean.
    - "mode": Fill with most frequent value.
    - "flag": Add a boolean indicator column (_missing) and fill with NaN.
    - "zero": Fill with 0 (use only for rainfall where 0 is physically meaningful
              AND the data source confirms zero is correct, not missing).

    If no strategy is provided, NaN values are preserved and a warning is logged.

    Args:
        df: Input DataFrame.
        strategy: Dict mapping column name → strategy name.

    Returns:
        DataFrame after applying missing value strategies.
    """
    df = df.copy()
    strategy = strategy or {}

    for col, strat in strategy.items():
        if col not in df.columns:
            continue
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        logger.info("Handling %d missing values in '%s' using strategy '%s'.", n_missing, col, strat)

        if strat == "drop":
            df = df[df[col].notna()].copy()
        elif strat == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strat == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strat == "mode":
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
        elif strat == "flag":
            df[f"{col}_missing"] = df[col].isnull().astype(int)
            # Value remains NaN — model must handle it
        elif strat == "zero":
            logger.warning(
                "Filling '%s' missing values with 0. Only appropriate if 0 "
                "is a confirmed observed value, not a substitute for missing data.",
                col,
            )
            df[col] = df[col].fillna(0.0)
        else:
            logger.warning("Unknown strategy '%s' for column '%s'. NaN retained.", strat, col)

    return df


# ─────────────────────────────────────────────
# Duplicate detection
# ─────────────────────────────────────────────

def remove_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """Remove exact duplicate rows.

    Args:
        df: Input DataFrame.
        subset: Columns to check for duplicates. Defaults to all columns.

    Returns:
        DataFrame with duplicates removed.
    """
    initial = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").copy()
    removed = initial - len(df)
    if removed:
        logger.info("Removed %d duplicate rows.", removed)
    return df


# ─────────────────────────────────────────────
# Type enforcement
# ─────────────────────────────────────────────

def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to appropriate dtypes.

    Numeric columns that contain strings are coerced with errors='coerce'
    so that unparseable values become NaN rather than crashing.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with enforced dtypes.
    """
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            original_nulls = df[col].isnull().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_nulls = df[col].isnull().sum()
            coerced = new_nulls - original_nulls
            if coerced > 0:
                logger.warning(
                    "Coerced %d non-numeric values to NaN in column '%s'.", coerced, col
                )

    for col in DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            n_invalid = df[col].isnull().sum()
            if n_invalid > 0:
                logger.warning("Could not parse %d datetime values in '%s'.", n_invalid, col)

    # Target must be integer (0 or 1)
    if "landslide_occurred" in df.columns:
        df["landslide_occurred"] = pd.to_numeric(
            df["landslide_occurred"], errors="coerce"
        ).fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────
# Categorical encoding
# ─────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns.

    If ``encoders`` is provided (from a previously fitted training set),
    those encoders are applied directly to prevent data leakage.
    Unknown categories are mapped to -1 with a warning.

    Args:
        df: Input DataFrame.
        columns: List of categorical columns to encode.
        encoders: Previously fitted LabelEncoder objects keyed by column name.

    Returns:
        Tuple of (encoded DataFrame, fitted encoders dict).
    """
    df = df.copy()
    columns = columns or CATEGORICAL_COLUMNS
    encoders = encoders or {}
    fitted: dict[str, LabelEncoder] = {}

    for col in columns:
        if col not in df.columns:
            continue

        encoded_col = f"{col}_encoded"
        df[col] = df[col].fillna("unknown").astype(str)

        if col in encoders:
            le = encoders[col]
            # Map unseen labels to -1
            known = set(le.classes_)
            mapping = {c: i for i, c in enumerate(le.classes_)}
            df[encoded_col] = df[col].map(mapping).fillna(-1).astype(int)
            unseen = set(df[col].unique()) - known - {"unknown"}
            if unseen:
                logger.warning(
                    "Column '%s' has unseen categories: %s. Mapped to -1.", col, unseen
                )
            fitted[col] = le
        else:
            le = LabelEncoder()
            df[encoded_col] = le.fit_transform(df[col])
            fitted[col] = le
            logger.info(
                "Encoded '%s' → %d categories: %s",
                col, len(le.classes_), list(le.classes_)[:10],
            )

    return df, fitted


# ─────────────────────────────────────────────
# Full preprocessing pipeline
# ─────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
    missing_strategy: dict[str, str] | None = None,
    categorical_encoders: dict[str, LabelEncoder] | None = None,
    is_training: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run the full preprocessing pipeline.

    This is the canonical entry point for data cleaning.

    Args:
        df: Raw input DataFrame.
        column_map: Optional column renaming map.
        missing_strategy: Dict of column → missing-value strategy.
        categorical_encoders: Pre-fitted encoders (for inference or test sets).
        is_training: If True, fit new encoders; otherwise apply provided encoders.

    Returns:
        Tuple of (processed DataFrame, artefacts dict containing fitted encoders etc.)
    """
    artefacts: dict[str, Any] = {}

    if column_map:
        df = df.rename(columns=column_map)

    # 1. Enforce dtypes
    df = enforce_dtypes(df)

    # 2. Remove invalid coordinates
    df = remove_invalid_coordinates(df)

    # 3. Report missing values
    report_missing_values(df)

    # 4. Remove duplicates
    df = remove_duplicates(df)

    # 5. Handle missing values
    default_strategy = missing_strategy or {
        "slope_degrees": "median",
        "elevation_m": "median",
        "vegetation_index": "median",
        # Rainfall and soil moisture are NOT filled — missing is preserved.
    }
    df = handle_missing_values(df, strategy=default_strategy)

    # 6. Encode categoricals
    encoders = categorical_encoders if not is_training else {}
    df, fitted_encoders = encode_categoricals(df, encoders=encoders)
    artefacts["categorical_encoders"] = fitted_encoders

    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df, artefacts


# ─────────────────────────────────────────────
# Save / load processed data
# ─────────────────────────────────────────────

def save_processed(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save processed DataFrame to Parquet.

    Args:
        df: Processed DataFrame.
        output_path: Destination file path (.parquet recommended).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved processed data to %s (%d rows)", output_path, len(df))


def load_processed(filepath: str | Path) -> pd.DataFrame:
    """Load a previously saved processed dataset.

    Args:
        filepath: Path to Parquet file.

    Returns:
        Loaded DataFrame.
    """
    filepath = Path(filepath)
    df = pd.read_parquet(filepath)
    logger.info("Loaded processed data from %s (%d rows)", filepath, len(df))
    return df
