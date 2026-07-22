"""Phase 3b seam tests: ModelData / PreparedDomain (/ PreparedPartitions).

Behavior preservation is gated by the golden pins and the full suite; the
tests here pin the seam INTERFACE: the objects exist, carry the documented
fields, agree exactly with the legacy args adapter view, and are immutable
where declared frozen.

RED verification: this file fails collection before the 3b-1 commit
(bstpp.preparation does not exist).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import dataclasses
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpyro.distributions as dist
import pytest

from bstpp.main import Hawkes_Model
from bstpp.preparation import (
    ModelData, PreparedDomain, PreparedPartitions, prepare_domain,
    prepare_partitions, N_XY, T_INTERNAL)
from bstpp.utils import aligned_difference_pairs

A_RECT = np.array([[10.0, 30.0], [5.0, 15.0]])
T_DAYS = 200.0
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))


def _interior_data(n=30, seed=7):
    r = np.random.RandomState(seed)
    return pd.DataFrame({
        "X": r.uniform(10.5, 29.5, n),
        "Y": r.uniform(5.5, 14.5, n),
        "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n)),
    })


def _triangle_gdf(crs=None):
    g = gpd.GeoDataFrame(geometry=[Polygon([(10, 5), (30, 5), (30, 15)])])
    if crs is not None:
        g = g.set_crs(crs)
    return g


def _triangle_data(n=30, seed=7):
    data = _interior_data(n, seed)
    data["X"] = np.random.RandomState(seed + 1).uniform(15.0, 29.5, n)
    data["Y"] = 5.0 + (data["X"] - 10.0) / 2.0 - 1.0
    return data


# ------------------------------------------------------------ prepare_domain

def test_prepare_domain_rectangle():
    dom = prepare_domain(A_RECT)
    assert dom.bounds is A_RECT          # rectangle domain: A_ IS the input
    assert dom.area_ratio == 1
    assert not dom.is_polygon
    assert dom.crs is None
    np.testing.assert_allclose(np.asarray(dom.axis_scales), [20.0, 10.0])


def test_prepare_domain_polygon():
    tri = _triangle_gdf(crs="EPSG:2272")
    dom = prepare_domain(tri)
    np.testing.assert_allclose(dom.bounds, A_RECT)
    # triangle covers half its bounding rectangle
    assert dom.area_ratio == pytest.approx(0.5)
    assert dom.is_polygon
    assert dom.crs == tri.crs
    assert dom.domain is tri


def test_prepare_domain_geographic_warning_preserved():
    lonlat = gpd.GeoDataFrame(
        geometry=[Polygon([(-75.2, 39.9), (-75.1, 39.9), (-75.1, 40.0)])]
    ).set_crs("EPSG:4326")
    with pytest.warns(UserWarning, match="anisotropic on the ground"):
        prepare_domain(lonlat)


def test_prepare_domain_is_frozen():
    dom = prepare_domain(A_RECT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        dom.area_ratio = 2.0


# ------------------------------------------------ constructor integration

def test_model_exposes_seam_objects_consistent_with_args():
    data = _interior_data()
    m = Hawkes_Model(data, A_RECT, T_DAYS, cox_background=False, **PRIORS)
    assert isinstance(m.model_data, ModelData)
    assert isinstance(m.prepared_domain, PreparedDomain)
    # ModelData carries the inputs as supplied
    assert m.model_data.events is m.data
    assert m.model_data.domain is A_RECT
    assert m.model_data.horizon_days == T_DAYS
    # legacy args entries are the adapter view of the SAME objects
    assert m.args["A_"] is m.prepared_domain.bounds
    assert m.args["A_area"] == m.prepared_domain.area_ratio
    assert m.args["axis_scales"] is m.prepared_domain.axis_scales


def test_model_seam_objects_polygon_domain():
    tri = _triangle_gdf()
    m = Hawkes_Model(_triangle_data(), tri, T_DAYS, cox_background=False,
                     excitation_support="rectangle",
                     **PRIORS)
    assert m.prepared_domain.is_polygon
    assert m.prepared_domain.domain is tri
    np.testing.assert_allclose(m.args["A_"], A_RECT)
    assert m.args["A_area"] == pytest.approx(0.5)


def test_model_data_is_frozen():
    m = Hawkes_Model(_interior_data(), A_RECT, T_DAYS, cox_background=False,
                     **PRIORS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.model_data.horizon_days = 999.0


# --------------------------------------------------------- PreparedPartitions

def test_prepare_partitions_rectangle():
    parts = prepare_partitions(prepare_domain(A_RECT), T_DAYS, 0.0)
    assert (parts.n_t, parts.n_s, parts.n_xy) == (50, 24, 25)
    assert len(parts.comp_grid) == N_XY ** 2
    np.testing.assert_array_equal(parts.spatial_grid_cells,
                                  np.arange(N_XY ** 2))
    # W rows sum to the internal temporal cell width (exact-overlap contract)
    np.testing.assert_allclose(
        np.asarray(parts.season_overlap).sum(axis=1),
        T_INTERNAL / parts.n_t, rtol=1e-6)
    # comp_grid tiles the REAL bounding rectangle
    np.testing.assert_allclose(parts.comp_grid.total_bounds,
                               [10.0, 5.0, 30.0, 15.0])


def test_prepare_partitions_polygon_selects_in_domain_cells():
    parts = prepare_partitions(prepare_domain(_triangle_gdf()), T_DAYS, 0.0)
    n_cells = len(parts.spatial_grid_cells)
    # triangle covers half the rectangle: strictly fewer than all cells,
    # far more than none
    assert 0 < n_cells < N_XY ** 2
    assert parts.int_df is None and parts.cov_gdf is None


def test_model_partitions_adapter_identity():
    m = Hawkes_Model(_interior_data(), A_RECT, T_DAYS, cox_background=False,
                     **PRIORS)
    parts = m.prepared_partitions
    assert isinstance(parts, PreparedPartitions)
    # args entries and model attributes ARE the seam object's fields
    assert m.args["x_t"] is parts.x_t
    assert m.args["x_a"] is parts.x_a
    assert m.args["season_overlap"] is parts.season_overlap
    assert m.args["season_idx_of_t"] is parts.season_idx_of_t
    assert m.args["spatial_grid_cells"] is parts.spatial_grid_cells
    assert m.comp_grid is parts.comp_grid
    assert (m.args["n_t"], m.args["n_s"], m.args["n_xy"]) == (50, 24, 25)


def test_covariate_partitions_adapter_identity_plain_hawkes():
    from shapely.geometry import box
    cov = gpd.GeoDataFrame(
        {"v": [1.0, 2.0]},
        geometry=[box(10, 5, 20, 15), box(20, 5, 30, 15)])
    m = Hawkes_Model(_interior_data(), A_RECT, T_DAYS, cox_background=False,
                     spatial_cov=cov, cov_names=["v"], **PRIORS)
    parts = m.prepared_partitions
    assert m.args["spatial_cov"] is parts.cov_values
    assert m.args["cov_area"] is parts.cov_area
    assert parts.cov_gdf is m.spatial_cov
    # standardized design matrix: mean 0, unit variance over cells
    np.testing.assert_allclose(parts.cov_values.mean(axis=0), 0.0, atol=1e-12)


# ------------------------------------------------- binding invariant 10.b.7

def test_pair_construction_takes_no_partition_argument():
    """10.b.7 (binding on all future work): excitation pairs are a function
    of the global event set and cutoff/window only. This is the negative
    constraint -- adding any tile/partition/cell parameter to
    aligned_difference_pairs must fail here and force a recorded decision."""
    import inspect
    params = set(inspect.signature(aligned_difference_pairs).parameters)
    assert params == {"t", "x", "y", "window", "spatial_window",
                      "axis_scales"}
