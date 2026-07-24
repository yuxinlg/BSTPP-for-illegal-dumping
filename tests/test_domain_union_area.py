"""Phase 3c: PreparedDomain area is the set-union of domain rows (SC).

Positive-area overlaps must not be double-counted. Disjoint multi-row and
single-row behavior is preserved (sum(area) == union.area). Parenting and
polygon mass already use the union; area_ratio / A_area must match.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import geopandas as gpd
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from bstpp.main import Hawkes_Model
from bstpp.preparation import prepare_domain, prepare_partitions

T_DAYS = 30.0
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(40.0),
)


def _events_in(geom, n=8, seed=0):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = geom.bounds
    xs, ys = [], []
    while len(xs) < n:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        from shapely.geometry import Point
        if geom.covers(Point(x, y)):
            xs.append(x)
            ys.append(y)
    return pd.DataFrame({
        "X": xs, "Y": ys,
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def test_overlapping_rows_use_union_area_not_sum():
    a1 = box(0, 0, 100, 100)
    a2 = box(50, 50, 150, 150)
    gdf = gpd.GeoDataFrame(geometry=[a1, a2])
    union = unary_union([a1, a2])
    assert float(gdf.area.sum()) > float(union.area)

    dom = prepare_domain(gdf)
    rect_area = float(
        (dom.bounds[0, 1] - dom.bounds[0, 0])
        * (dom.bounds[1, 1] - dom.bounds[1, 0]))
    assert dom.area_ratio == pytest.approx(float(union.area) / rect_area)
    assert dom.area_ratio != pytest.approx(float(gdf.area.sum()) / rect_area)


def test_row_order_and_duplicate_overlap_do_not_change_prepared_domain():
    a1 = box(0, 0, 100, 100)
    a2 = box(50, 50, 150, 150)
    g1 = gpd.GeoDataFrame(geometry=[a1, a2])
    g2 = gpd.GeoDataFrame(geometry=[a2, a1])
    g3 = gpd.GeoDataFrame(geometry=[a1, a2, a1])  # duplicate overlapping row

    d1, d2, d3 = prepare_domain(g1), prepare_domain(g2), prepare_domain(g3)
    assert d1.area_ratio == pytest.approx(d2.area_ratio)
    assert d1.area_ratio == pytest.approx(d3.area_ratio)
    np.testing.assert_allclose(d1.bounds, d2.bounds)
    np.testing.assert_allclose(d1.bounds, d3.bounds)
    assert d1.union_geometry.equals(d2.union_geometry)
    assert d1.union_geometry.equals(d3.union_geometry)


def test_disjoint_multipolygon_rows_unchanged():
    a1 = box(0, 0, 50, 50)
    a2 = box(100, 100, 150, 150)
    gdf = gpd.GeoDataFrame(geometry=[a1, a2])
    union = unary_union([a1, a2])
    assert float(gdf.area.sum()) == pytest.approx(float(union.area))

    dom = prepare_domain(gdf)
    rect_area = float(
        (dom.bounds[0, 1] - dom.bounds[0, 0])
        * (dom.bounds[1, 1] - dom.bounds[1, 0]))
    assert dom.area_ratio == pytest.approx(float(gdf.area.sum()) / rect_area)
    assert dom.area_ratio == pytest.approx(float(union.area) / rect_area)


def test_model_A_area_and_support_use_union_canonical_geometry():
    a1 = box(0, 0, 100, 100)
    a2 = box(50, 50, 150, 150)
    gdf = gpd.GeoDataFrame(geometry=[a1, a2])
    union = unary_union([a1, a2])
    data = _events_in(union, n=6, seed=1)

    m = Hawkes_Model(
        data, gdf, T_DAYS, cox_background=False,
        excitation_support="rectangle", **PRIORS,
    )
    assert m.args["A_area"] == pytest.approx(m.prepared_domain.area_ratio)
    rect_area = float(
        (m.prepared_domain.bounds[0, 1] - m.prepared_domain.bounds[0, 0])
        * (m.prepared_domain.bounds[1, 1] - m.prepared_domain.bounds[1, 0]))
    assert m.args["A_area"] == pytest.approx(float(union.area) / rect_area)

    # Support clipping must be against the same union geometry / area ratio.
    parts = prepare_partitions(m.prepared_domain, T_DAYS, 0.0)
    support_area = float(parts.support_cells["area"].sum())
    assert support_area == pytest.approx(m.prepared_domain.area_ratio, rel=1e-6)


# ---------- characterization: independent unions match PreparedDomain ------
def _geom_cases():
    from shapely.geometry import Polygon
    return {
        "single": gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]),
        "disjoint": gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)]),
        "overlap": gpd.GeoDataFrame(
            geometry=[box(0, 0, 2, 2), box(1, 1, 3, 3)]),
        "mixed": gpd.GeoDataFrame(geometry=[
            Polygon([(0, 0), (2, 0), (1, 2)]),
            box(1.5, 0.5, 2.5, 1.5),
        ]),
    }


def _independent_unions(gdf):
    """Mirror the pre-plumbing downstream recomputation sites."""
    from bstpp.excitation_support import domain_polygon_geometry
    from bstpp.polygon_mass import _geometry_sha256

    series_union = (
        gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all")
        else gdf.geometry.unary_union)
    return {
        "prepared": prepare_domain(gdf).union_geometry,
        "series_union_all": series_union,
        "excitation_recompute": domain_polygon_geometry(gdf),
        "hash_prepared": _geometry_sha256(prepare_domain(gdf).union_geometry),
        "hash_series": _geometry_sha256(series_union),
        "hash_excitation": _geometry_sha256(domain_polygon_geometry(gdf)),
    }


@pytest.mark.parametrize("case", ["single", "disjoint", "overlap", "mixed"])
def test_independent_domain_unions_match_prepared_wkb_and_hash(case):
    """Pre-plumbing characterization: downstream unary_union / union_all
    recomputes currently equal PreparedDomain.union_geometry (equals + WKB
    + geometry hash). Plumbing must preserve this identity numerically.
    """
    gdf = _geom_cases()[case]
    u = _independent_unions(gdf)
    canon = u["prepared"]
    assert canon is not None
    for name in ("series_union_all", "excitation_recompute"):
        other = u[name]
        assert canon.equals(other), f"{case}: equals failed vs {name}"
        assert canon.wkb == other.wkb, f"{case}: WKB mismatch vs {name}"
        assert float(canon.area) == pytest.approx(float(other.area))
    assert u["hash_prepared"] == u["hash_series"] == u["hash_excitation"]


@pytest.mark.parametrize("case", ["single", "disjoint", "overlap", "mixed"])
def test_prepare_partitions_support_area_matches_canonical_union(case):
    gdf = _geom_cases()[case]
    dom = prepare_domain(gdf)
    parts = prepare_partitions(dom, T_DAYS, 0.0)
    support_area = float(parts.support_cells["area"].sum())
    assert support_area == pytest.approx(dom.area_ratio, rel=1e-6)
    # Absolute union area in real units recovers from area_ratio * rect.
    rect_area = float(
        (dom.bounds[0, 1] - dom.bounds[0, 0])
        * (dom.bounds[1, 1] - dom.bounds[1, 0]))
    assert support_area * rect_area == pytest.approx(
        float(dom.union_geometry.area), rel=1e-6)


def test_excitation_support_domain_geom_matches_prepared_union():
    """Rectangle-mode GeoDataFrame: ExcitationSupport.domain_geom must match
    PreparedDomain.union_geometry (equals + WKB + hash). After plumbing it
    must be the same object (no independent recompute).
    """
    from bstpp.polygon_mass import _geometry_sha256

    gdf = _geom_cases()["overlap"]
    union = unary_union(list(gdf.geometry.values))
    data = _events_in(union, n=5, seed=3)
    m = Hawkes_Model(
        data, gdf, T_DAYS, cox_background=False,
        excitation_support="rectangle", **PRIORS,
    )
    canon = m.prepared_domain.union_geometry
    geom = m.excitation_support.domain_geom
    assert canon is geom, "excitation support must reuse canonical union_geometry"
    assert canon.equals(geom)
    assert canon.wkb == geom.wkb
    assert _geometry_sha256(canon) == _geometry_sha256(geom)
