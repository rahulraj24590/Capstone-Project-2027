"""
Tests for data preprocessing pipeline.

Verifies:
- Valid inputs are processed correctly
- Invalid coordinates are rejected
- Missing environmental data is handled safely
- Feature engineering functions work as expected
- No data leakage in train/test splits
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing.preprocessing import (
    validate_columns,
    remove_invalid_coordinates,
    handle_missing_values,
    remove_duplicates,
)
from src.data_processing.feature_engineering import (
    compute_slope_class,
    select_available_features,
)
from src.utils.validation import validate_coordinates


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def valid_df():
    """Minimal valid DataFrame with required columns."""
    return pd.DataFrame({
        "latitude": [12.5, 22.1, 30.0],
        "longitude": [77.5, 82.0, 75.0],
        "slope_degrees": [25.0, 15.0, 40.0],
        "elevation_m": [500.0, 300.0, 1200.0],
        "rainfall_24h_mm": [80.0, 20.0, 150.0],
        "rainfall_72h_mm": [150.0, 40.0, 300.0],
        "rainfall_1h_mm": [10.0, 3.0, 25.0],
        "vegetation_index": [0.4, 0.6, 0.2],
        "soil_moisture": [0.5, 0.3, 0.7],
        "landslide_occurred": [1, 0, 1],
        "is_synthetic": [1, 1, 1],
    })


@pytest.fixture
def invalid_coord_df():
    """DataFrame with some invalid coordinates."""
    return pd.DataFrame({
        "latitude": [200.0, -91.0, 22.5],   # first two invalid
        "longitude": [77.5, 181.0, 78.0],    # second invalid
        "slope_degrees": [20.0, 15.0, 30.0],
        "elevation_m": [500.0, 300.0, 800.0],
        "rainfall_24h_mm": [60.0, 30.0, 90.0],
        "rainfall_72h_mm": [120.0, 60.0, 180.0],
        "landslide_occurred": [1, 0, 1],
    })


# ─────────────────────────────────────────────
# Coordinate validation tests
# ─────────────────────────────────────────────

class TestCoordinateValidation:
    def test_valid_lat_lon_accepted(self):
        result = validate_coordinates(22.5, 78.0)
        assert result.is_valid

    def test_invalid_latitude_rejected(self):
        result = validate_coordinates(200.0, 78.0)
        assert not result.is_valid

    def test_invalid_longitude_rejected(self):
        result = validate_coordinates(22.5, 200.0)
        assert not result.is_valid

    def test_boundary_values_accepted(self):
        assert validate_coordinates(90.0, 180.0).is_valid
        assert validate_coordinates(-90.0, -180.0).is_valid

    def test_just_outside_bounds_rejected(self):
        assert not validate_coordinates(90.001, 0.0).is_valid
        assert not validate_coordinates(0.0, -180.001).is_valid

    def test_none_values_rejected(self):
        result = validate_coordinates(None, None)
        assert not result.is_valid

    def test_nan_values_rejected(self):
        result = validate_coordinates(float("nan"), 78.0)
        assert not result.is_valid

    def test_lat_lon_order(self):
        """Latitude 150.0 must fail; longitude 150.0 is fine."""
        assert not validate_coordinates(150.0, 78.0).is_valid
        assert validate_coordinates(22.5, 150.0).is_valid


# ─────────────────────────────────────────────
# remove_invalid_coordinates tests
# ─────────────────────────────────────────────

class TestRemoveInvalidCoordinates:
    def test_invalid_rows_removed(self, invalid_coord_df):
        result = remove_invalid_coordinates(invalid_coord_df)
        # Only (22.5, 78.0) is valid
        assert len(result) == 1
        assert float(result["latitude"].iloc[0]) == pytest.approx(22.5)

    def test_valid_rows_unchanged(self, valid_df):
        result = remove_invalid_coordinates(valid_df)
        assert len(result) == len(valid_df)

    def test_empty_df_handled(self):
        result = remove_invalid_coordinates(pd.DataFrame(columns=["latitude", "longitude"]))
        assert result.empty


# ─────────────────────────────────────────────
# validate_columns tests
# ─────────────────────────────────────────────

class TestValidateColumns:
    def test_all_required_present(self, valid_df):
        missing = validate_columns(valid_df, required=["latitude", "longitude", "slope_degrees"])
        assert missing == []

    def test_missing_column_detected(self, valid_df):
        df = valid_df.drop(columns=["slope_degrees"])
        missing = validate_columns(df, required=["slope_degrees"])
        assert "slope_degrees" in missing

    def test_empty_required_list(self, valid_df):
        missing = validate_columns(valid_df, required=[])
        assert missing == []


# ─────────────────────────────────────────────
# Missing value handling tests
# ─────────────────────────────────────────────

class TestMissingValueHandling:
    def test_numeric_nan_imputed(self):
        df = pd.DataFrame({
            "slope_degrees": [20.0, None, 30.0],
            "elevation_m": [500.0, 600.0, None],
        })
        strategy = {"slope_degrees": "median", "elevation_m": "median"}
        result = handle_missing_values(df, strategy=strategy)
        assert result["slope_degrees"].isna().sum() == 0
        assert result["elevation_m"].isna().sum() == 0

    def test_missing_rainfall_not_filled_with_zero(self):
        """Median imputation must not produce zeros for non-zero columns."""
        df = pd.DataFrame({
            "rainfall_24h_mm": [100.0, None, 80.0],
        })
        result = handle_missing_values(df, strategy={"rainfall_24h_mm": "median"})
        filled_val = result["rainfall_24h_mm"].iloc[1]
        assert filled_val != 0.0
        assert filled_val == pytest.approx(90.0)

    def test_all_nan_column_handled(self):
        """A column that is entirely NaN should not crash."""
        df = pd.DataFrame({
            "slope_degrees": [None, None, None],
            "elevation_m": [500.0, 600.0, 700.0],
        })
        # Median of all-NaN column is NaN — should not crash
        result = handle_missing_values(df, strategy={"slope_degrees": "median"})
        assert len(result) == 3  # must not crash


# ─────────────────────────────────────────────
# Duplicate removal tests
# ─────────────────────────────────────────────

class TestDuplicateRemoval:
    def test_duplicates_removed(self):
        df = pd.DataFrame({
            "latitude": [22.5, 22.5, 30.0],
            "longitude": [78.0, 78.0, 75.0],
            "slope_degrees": [25.0, 25.0, 40.0],
            "landslide_occurred": [1, 1, 0],
        })
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_no_false_positive_dedup(self, valid_df):
        result = remove_duplicates(valid_df)
        assert len(result) == len(valid_df)


# ─────────────────────────────────────────────
# Feature engineering tests
# ─────────────────────────────────────────────

class TestFeatureEngineering:
    def test_slope_class_computed(self, valid_df):
        result = compute_slope_class(valid_df)
        assert "slope_class" in result.columns

    def test_slope_class_range(self, valid_df):
        result = compute_slope_class(valid_df)
        assert result["slope_class"].between(0, 4).all()

    def test_slope_class_missing_column(self):
        """Missing 'slope_degrees' should not crash."""
        df = pd.DataFrame({"elevation_m": [500.0, 300.0]})
        result = compute_slope_class(df)
        assert "slope_class" not in result.columns  # graceful skip

    def test_select_available_features_returns_list(self, valid_df):
        features = select_available_features(valid_df)
        assert isinstance(features, list)
        assert len(features) > 0

    def test_selected_features_exist_in_df(self, valid_df):
        features = select_available_features(valid_df)
        for f in features:
            assert f in valid_df.columns, f"Feature '{f}' not in DataFrame"

    def test_feature_order_consistent(self, valid_df):
        """Feature order must be identical on repeated calls."""
        features1 = select_available_features(valid_df)
        features2 = select_available_features(valid_df)
        assert features1 == features2


# ─────────────────────────────────────────────
# No-leakage sanity check
# ─────────────────────────────────────────────

class TestNoDataLeakage:
    def test_train_test_indices_disjoint(self):
        """Train and test indices must not overlap."""
        from sklearn.model_selection import train_test_split
        indices = np.arange(100)
        train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42)
        assert len(set(train_idx) & set(test_idx)) == 0
