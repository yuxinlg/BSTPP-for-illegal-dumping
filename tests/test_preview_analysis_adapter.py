"""Focused checks for the provisional cbg-park-seasonal analysis adapter."""

import numpy as np
import pandas as pd

from batch_park_fits import (
    ANALYSIS_END,
    ANALYSIS_START,
    ANALYSIS_TOTAL_DAYS,
    COLUMN_NAMES,
    DEFAULT_SPATIAL_SIGMA_PRIOR_SCALE_M,
    SPATIAL_SIGMA_SENSITIVITY_M,
    SPATIAL_WINDOW_M,
    get_park_fit_kwargs,
    legacy_citywide_standardize_covariates,
)


def test_legacy_standardization_uses_all_supplied_rows_and_population_scale():
    values = np.arange(5 * len(COLUMN_NAMES), dtype=float).reshape(5, -1)
    covariates = pd.DataFrame(values, columns=COLUMN_NAMES)

    standardized, record = legacy_citywide_standardize_covariates(covariates)

    expected_mean = values.mean(axis=0)
    expected_scale = values.var(axis=0) ** 0.5
    np.testing.assert_allclose(
        standardized[COLUMN_NAMES].to_numpy(),
        (values - expected_mean) / expected_scale,
    )
    np.testing.assert_allclose(record["mean"], expected_mean)
    np.testing.assert_allclose(record["scale"], expected_scale)
    assert record["row_count"] == len(covariates)
    assert record["columns"] == COLUMN_NAMES


def test_preview_defaults_record_real_unit_spatial_choices():
    settings = get_park_fit_kwargs("tacony_box")

    assert settings["spatial_window"] == SPATIAL_WINDOW_M == 1_000.0
    assert settings["spatial_sigma_prior_scale_m"] == (
        DEFAULT_SPATIAL_SIGMA_PRIOR_SCALE_M
    ) == 250.0
    assert SPATIAL_SIGMA_SENSITIVITY_M == (50.0, 100.0, 250.0)


def test_preview_calendar_horizon_includes_the_2024_leap_day():
    assert ANALYSIS_START == pd.Timestamp("2021-01-01")
    assert ANALYSIS_END == pd.Timestamp("2025-01-01")
    assert ANALYSIS_TOTAL_DAYS == 1_461
