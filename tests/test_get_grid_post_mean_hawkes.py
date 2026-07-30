"""get_grid_post_mean for plain Hawkes covariates (pre-3f CF).

Must not depend on a prior plotting call mutating spatial_cov with a
post_mean column. Derive covariate effects from b_0 or from w @ X_s,
map to computational cells via prepared support/refinement geometry with
intersection-area weighting, and leave model inputs unchanged.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from shapely.geometry import box as shapely_box

from bstpp.main import Hawkes_Model
from bstpp.spatial_grid_helpers import get_grid_post_mean

T_DAYS = 30.0
A = np.array([[0.0, 1.0], [0.0, 1.0]])


def _cov_layer():
    # Two covariate polygons covering the unit square; left/right halves.
    return gpd.GeoDataFrame({
        "cov": [1.0, -1.0],
        "geometry": [shapely_box(0.0, 0.0, 0.5, 1.0),
                     shapely_box(0.5, 0.0, 1.0, 1.0)],
    })


def _events():
    return pd.DataFrame({
        "X": [0.25, 0.75, 0.25, 0.75],
        "Y": [0.25, 0.25, 0.75, 0.75],
        "T": [1.0, 2.0, 3.0, 4.0],
    })


def test_plain_hawkes_get_grid_post_mean_without_prior_plot():
    cov = _cov_layer()
    m = Hawkes_Model(
        _events(), A, T_DAYS, cox_background=False,
        spatial_cov=cov, cov_names=["cov"],
        a_0=dist.Normal(0, 5),
        alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0),
        sigmax_2=dist.HalfNormal(0.25),
        w=dist.Normal(0, 1),
    )
    # Synthetic posterior: constant w so b_0 = cov * w.
    w_true = 2.0
    m.samples = {
        "w": np.full((5, 1), w_true, dtype=np.float32),
        "a_0": np.zeros(5, dtype=np.float32),
        "alpha": np.full(5, 0.3, dtype=np.float32),
        "beta": np.ones(5, dtype=np.float32),
        "sigmax_2": np.full(5, 0.1, dtype=np.float32),
    }
    assert "b_0" not in m.samples
    assert "post_mean" not in m.spatial_cov.columns
    assert "int_df" not in m.args

    cov_before = m.spatial_cov.copy(deep=True)
    comp_before = m.comp_grid.copy(deep=True)
    args_spatial = np.array(m.args["spatial_cov"], copy=True)

    result = get_grid_post_mean(m, include_cov=True)

    assert list(result.columns) == [
        "grid_row", "grid_col", "post_mean", "comp_grid_id", "geometry"]
    assert len(result) == m.args["n_xy"] ** 2
    # Interior of each half (exclude the x=0.5 straddling column, which is
    # correctly area-weighted across both covariate polygons).
    left = result[result.geometry.centroid.x < 0.45]["post_mean"].to_numpy()
    right = result[result.geometry.centroid.x > 0.55]["post_mean"].to_numpy()
    assert len(left) > 0 and len(right) > 0
    assert np.allclose(left, w_true * 1.0, atol=1e-5)
    assert np.allclose(right, w_true * (-1.0), atol=1e-5)

    # No mutation of prepared / covariate / grid state.
    assert "post_mean" not in m.spatial_cov.columns
    pd.testing.assert_frame_equal(m.spatial_cov.drop(columns="geometry"),
                                  cov_before.drop(columns="geometry"))
    assert m.comp_grid["comp_grid_id"].tolist() == comp_before["comp_grid_id"].tolist()
    np.testing.assert_array_equal(np.asarray(m.args["spatial_cov"]), args_spatial)


def test_plain_hawkes_get_grid_post_mean_uses_b0_when_present():
    cov = _cov_layer()
    m = Hawkes_Model(
        _events(), A, T_DAYS, cox_background=False,
        spatial_cov=cov, cov_names=["cov"],
        a_0=dist.Normal(0, 5),
        alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0),
        sigmax_2=dist.HalfNormal(0.25),
        w=dist.Normal(0, 1),
    )
    b0 = np.array([[1.5, -2.5]] * 4, dtype=np.float32)
    m.samples = {
        "b_0": b0,
        "w": np.ones((4, 1), dtype=np.float32),
        "a_0": np.zeros(4, dtype=np.float32),
        "alpha": np.full(4, 0.3, dtype=np.float32),
        "beta": np.ones(4, dtype=np.float32),
        "sigmax_2": np.full(4, 0.1, dtype=np.float32),
    }
    result = get_grid_post_mean(m, include_cov=True)
    left = result[result.geometry.centroid.x < 0.45]["post_mean"].to_numpy()
    right = result[result.geometry.centroid.x > 0.55]["post_mean"].to_numpy()
    assert len(left) > 0 and len(right) > 0
    assert np.allclose(left, 1.5, atol=1e-5)
    assert np.allclose(right, -2.5, atol=1e-5)
    assert "post_mean" not in m.spatial_cov.columns
