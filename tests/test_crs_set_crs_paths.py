"""Post-hoc coverage for ``e706107`` CRS alignment (set_crs, no attr assign).

``e706107`` replaced deprecated ``GeoDataFrame.crs = ...`` with ``set_crs``
(or a no-op when CRS already matches). These tests pin the changed paths;
they are not a historical RED for that commit.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import geopandas as gpd
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from bstpp.data_contracts import DataContractError
from bstpp.main import Hawkes_Model
from bstpp.preparation import prepare_domain, prepare_partitions

A_RECT = np.array([[10.0, 30.0], [5.0, 15.0]])
T_DAYS = 200.0
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(0.25),
)
COV_GRID = (10.0, 10.0)


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


def _cov_layer(crs=None):
    g = gpd.GeoDataFrame(
        {"v": [1.0, 2.0]},
        geometry=[box(10, 5, 20, 15), box(20, 5, 30, 15)],
    )
    if crs is not None:
        g = g.set_crs(crs)
    return g


def _tabular_cov_df():
    return pd.DataFrame({
        "X": [15.0, 25.0],
        "Y": [10.0, 10.0],
        "v": [1.0, 2.0],
    })


def test_prepared_domain_polygon_crs_none():
    """CRS-less polygon domain: prepare_partitions must not require a CRS."""
    tri = _triangle_gdf(crs=None)
    assert tri.crs is None
    dom = prepare_domain(tri)
    assert dom.is_polygon
    assert dom.crs is None
    parts = prepare_partitions(dom, T_DAYS, 0.0)
    assert parts.comp_grid.crs is None
    assert len(parts.spatial_grid_cells) > 0


def test_prepared_domain_polygon_with_crs():
    tri = _triangle_gdf(crs="EPSG:2272")
    dom = prepare_domain(tri)
    assert dom.crs == tri.crs
    parts = prepare_partitions(dom, T_DAYS, 0.0)
    assert parts.comp_grid.crs == tri.crs


def test_covariate_attach_when_partition_and_cov_crs_already_match():
    """Matching CRS must not emit deprecated GeoDataFrame.crs assignment."""
    tri = _triangle_gdf(crs="EPSG:2272")
    cov = _cov_layer(crs="EPSG:2272")
    data = _triangle_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        m = Hawkes_Model(
            data, tri, T_DAYS, cox_background=True,
            excitation_support="rectangle",
            spatial_cov=cov, cov_names=["v"],
            data_contracts="reject", **PRIORS,
        )
    assert m.prepared_partitions.comp_grid.crs == tri.crs
    assert m.prepared_partitions.support_cells.crs == tri.crs
    assert m.points.crs == tri.crs
    assert "num_cov" in m.args


def test_array_domain_crs_none_unchanged():
    data = _interior_data()
    m = Hawkes_Model(
        data, A_RECT, T_DAYS, cox_background=False,
        data_contracts="reject", **PRIORS,
    )
    assert m.prepared_domain.crs is None
    assert m.prepared_domain.is_polygon is False


def test_tabular_spatial_cov_crs_path_no_crs_override_warning():
    tri = _triangle_gdf(crs="EPSG:2272")
    data = _triangle_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        m = Hawkes_Model(
            data, tri, T_DAYS, cox_background=False,
            excitation_support="rectangle",
            spatial_cov=_tabular_cov_df(), cov_names=["v"],
            cov_grid_size=COV_GRID,
            spatial_cov_crs="EPSG:2272",
            data_contracts="reject", **PRIORS,
        )
    assert m.spatial_cov.crs == tri.crs
    assert "num_cov" in m.args


def test_public_crs_mismatch_rejected_without_allow_override():
    """Public contract rejects CRS mismatch; do not rely on allow_override."""
    tri = _triangle_gdf(crs="EPSG:2272")
    cov = _cov_layer(crs="EPSG:3857")
    data = _triangle_data()
    with pytest.raises(DataContractError, match="crs_mismatch"):
        Hawkes_Model(
            data, tri, T_DAYS, cox_background=True,
            excitation_support="rectangle",
            spatial_cov=cov, cov_names=["v"],
            data_contracts="reject", **PRIORS,
        )


def test_attach_covariate_partitions_mismatch_is_loud_invariant():
    """CRS mismatch must not be concealed via set_crs(..., allow_override=True).

    Public validate_covariates already rejects mismatches. If a mismatched
    pair still reaches attach_covariate_partitions, fail loudly instead of
    silently rewriting partition CRS.
    """
    from bstpp.preparation import attach_covariate_partitions

    tri = _triangle_gdf(crs="EPSG:2272")
    dom = prepare_domain(tri)
    parts = prepare_partitions(dom, T_DAYS, 0.0)
    assert parts.comp_grid.crs == tri.crs
    cov = _cov_layer(crs="EPSG:3857")
    cov = cov.copy()
    cov["cov_ind"] = np.arange(len(cov))
    with pytest.raises((RuntimeError, ValueError), match="CRS|crs"):
        attach_covariate_partitions(
            parts, dom, cov, ["v"], None, "cox_hawkes")
