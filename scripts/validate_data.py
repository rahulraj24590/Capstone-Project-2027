"""
Data validation script for LandslideGuard.

Validates a processed or raw dataset and produces a data quality report.

USAGE:
    python scripts/validate_data.py
    python scripts/validate_data.py --file data/processed/dataset.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data_processing.preprocessing import (
    validate_columns,
    report_missing_values,
    remove_invalid_coordinates,
)
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["latitude", "longitude", "landslide_occurred"]
NUMERIC_COLUMNS = [
    "slope_degrees", "elevation_m",
    "rainfall_1h_mm", "rainfall_24h_mm", "rainfall_72h_mm",
]


def validate_file(filepath: Path) -> dict:
    """Run full validation on a data file.

    Args:
        filepath: Path to CSV or Parquet file.

    Returns:
        Dict containing validation results.
    """
    results: dict = {"file": str(filepath), "valid": True, "issues": [], "warnings": []}

    if not filepath.exists():
        results["valid"] = False
        results["issues"].append(f"File not found: {filepath}")
        return results

    # Load
    try:
        if filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        results["valid"] = False
        results["issues"].append(f"Failed to load file: {e}")
        return results

    results["n_rows"] = len(df)
    results["n_columns"] = len(df.columns)
    results["columns"] = list(df.columns)

    # Required columns
    missing_cols = validate_columns(df, REQUIRED_COLUMNS)
    if missing_cols:
        results["valid"] = False
        results["issues"].append(f"Missing required columns: {missing_cols}")

    # Coordinate validation
    if "latitude" in df.columns and "longitude" in df.columns:
        invalid_coords = (
            (~df["latitude"].between(-90, 90)) |
            (~df["longitude"].between(-180, 180)) |
            df["latitude"].isna() |
            df["longitude"].isna()
        ).sum()
        if invalid_coords > 0:
            results["warnings"].append(f"{invalid_coords} rows with invalid coordinates.")

    # Missing value report
    missing_report = report_missing_values(df)
    if not missing_report.empty:
        results["missing_values"] = missing_report.to_dict()

    # Class distribution
    if "landslide_occurred" in df.columns:
        vc = df["landslide_occurred"].value_counts()
        results["class_distribution"] = vc.to_dict()
        pos_rate = df["landslide_occurred"].mean()
        results["positive_rate"] = float(pos_rate)
        if pos_rate < 0.01:
            results["warnings"].append(
                f"Very low positive rate ({pos_rate:.2%}). "
                "Model will face severe class imbalance."
            )

    # Synthetic flag
    if "is_synthetic" in df.columns:
        n_synthetic = int(df["is_synthetic"].sum())
        results["n_synthetic"] = n_synthetic
        if n_synthetic == len(df):
            results["warnings"].append(
                "ALL records are synthetic. "
                "Results must not be used for scientific claims."
            )

    # Numeric range checks
    range_issues = []
    if "slope_degrees" in df.columns:
        bad_slope = (df["slope_degrees"] < 0) | (df["slope_degrees"] > 90)
        if bad_slope.any():
            range_issues.append(f"slope_degrees: {bad_slope.sum()} out-of-range values.")
    if "elevation_m" in df.columns:
        bad_elev = df["elevation_m"] < -500
        if bad_elev.any():
            range_issues.append(f"elevation_m: {bad_elev.sum()} suspiciously low values.")

    if range_issues:
        results["warnings"].extend(range_issues)

    return results


def print_validation_report(results: dict) -> None:
    """Print human-readable validation report."""
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"File: {results['file']}")
    print(f"Rows: {results.get('n_rows', 'N/A')}")
    print(f"Columns: {results.get('n_columns', 'N/A')}")
    print(f"Valid: {'✅ YES' if results['valid'] else '❌ NO'}")

    if results.get("class_distribution"):
        print(f"\nClass Distribution: {results['class_distribution']}")
        print(f"Positive Rate: {results.get('positive_rate', 0):.2%}")

    if results.get("n_synthetic") is not None:
        print(f"Synthetic Records: {results['n_synthetic']}")

    if results.get("issues"):
        print("\n❌ ISSUES:")
        for issue in results["issues"]:
            print(f"   - {issue}")

    if results.get("warnings"):
        print("\n⚠️  WARNINGS:")
        for warning in results["warnings"]:
            print(f"   - {warning}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a LandslideGuard dataset.")
    parser.add_argument(
        "--file",
        default="data/processed/dataset.parquet",
        help="Path to data file.",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    results = validate_file(Path(args.file))
    print_validation_report(results)

    sys.exit(0 if results["valid"] else 1)
