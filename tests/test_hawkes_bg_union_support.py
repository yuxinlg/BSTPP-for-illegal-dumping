"""Plain-Hawkes no-cov background must sample from PreparedDomain.union_geometry.

Pre-3f / D-30: overlapping domain rows must not duplicate physical area or
inflate background density. The no-covariate sampler must use the same
authoritative union geometry / A_area atoms as other background consumers,
not raw ``self.A`` row geometries.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import geopandas as gpd
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from shapely.geometry import Point, box
from shapely.ops import unary_union

from bstpp.main import Hawkes_Model
from bstpp.likelihood import background_masses

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
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if geom.covers(Point(x, y)):
            xs.append(x)
            ys.append(y)
    return pd.DataFrame({
        "X": xs, "Y": ys,
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _hawkes(domain):
    union = (
        domain.geometry.union_all() if hasattr(domain.geometry, "union_all")
        else unary_union(domain.geometry))
    data = _events_in(union, n=8, seed=1)
    return Hawkes_Model(
        data, domain, T_DAYS, cox_background=False,
        excitation_support="rectangle", **PRIORS)


def test_overlapping_rows_do_not_inflate_background_rate():
    """Overlap must not double-count area in the Poisson mean."""
    a1 = box(0, 0, 100, 100)
    a2 = box(50, 50, 150, 150)
    overlapping = gpd.GeoDataFrame(geometry=[a1, a2])
    union_only = gpd.GeoDataFrame(geometry=[unary_union([a1, a2])])

    m_over = _hawkes(overlapping)
    m_union = _hawkes(union_only)
    assert float(m_over.args["A_area"]) == pytest.approx(float(m_union.args["A_area"]))

    a0 = 2.0
    T_int = float(m_over.args["T"])
    expected = float(np.asarray(
        background_masses(np.exp(a0), np.asarray([m_over.args["A_area"]]), T_int)
    ).sum())
    # Row-sum areas would be strictly larger than the union.
    row_sum_area = float(overlapping.area.sum()) / float(
        (m_over.args["A_"][0, 1] - m_over.args["A_"][0, 0])
        * (m_over.args["A_"][1, 1] - m_over.args["A_"][1, 0]))
    assert row_sum_area > float(m_over.args["A_area"])

    rates_over = np.asarray(background_masses(
        np.exp(a0),
        # Probe the sampler's chosen atoms indirectly via many draws' mean.
        np.asarray([m_over.args["A_area"]]),
        T_int,
    ))
    assert float(rates_over.sum()) == pytest.approx(expected)

    rng = np.random.default_rng(0)
    counts = [len(m_over._sim_hawkes_bg({"a_0": a0}, rng=rng)) for _ in range(40)]
    # Mean count must track the UNION rate, not the inflated row-sum rate.
    inflated = float(np.asarray(
        background_masses(np.exp(a0), np.asarray([row_sum_area]), T_int)
    ).sum())
    mean_c = float(np.mean(counts))
    assert abs(mean_c - expected) < abs(mean_c - inflated)
    assert mean_c == pytest.approx(expected, rel=0.25)


def test_sampled_points_always_inside_union_overlapping():
    a1 = box(0, 0, 100, 100)
    a2 = box(50, 50, 150, 150)
    domain = gpd.GeoDataFrame(geometry=[a1, a2])
    m = _hawkes(domain)
    union = m.prepared_domain.union_geometry
    s = m._sim_hawkes_bg({"a_0": 4.0}, rng=np.random.default_rng(2))
    assert len(s) > 50
    inside = np.array([
        union.buffer(1e-9).covers(Point(x, y)) for x, y in s[:, :2]
    ])
    assert inside.all(), f"{(~inside).sum()}/{len(s)} points outside union"


def test_single_polygon_background_samples_inside():
    poly = box(0, 0, 100, 80)
    m = _hawkes(gpd.GeoDataFrame(geometry=[poly]))
    s = m._sim_hawkes_bg({"a_0": 3.5}, rng=np.random.default_rng(3))
    assert len(s) > 20
    inside = np.array([
        poly.buffer(1e-9).covers(Point(x, y)) for x, y in s[:, :2]
    ])
    assert inside.all()


def test_disjoint_polygons_sample_inside_union_and_conserve_area():
    a1 = box(0, 0, 50, 50)
    a2 = box(100, 100, 150, 150)
    domain = gpd.GeoDataFrame(geometry=[a1, a2])
    m = _hawkes(domain)
    union = m.prepared_domain.union_geometry
    assert float(m.args["A_area"]) == pytest.approx(
        float(a1.area + a2.area) / float(
            (m.args["A_"][0, 1] - m.args["A_"][0, 0])
            * (m.args["A_"][1, 1] - m.args["A_"][1, 0])))

    s = m._sim_hawkes_bg({"a_0": 4.0}, rng=np.random.default_rng(4))
    assert len(s) > 50
    inside = np.array([
        union.buffer(1e-9).covers(Point(x, y)) for x, y in s[:, :2]
    ])
    assert inside.all()
    # Both components should receive points under a large draw.
    in1 = np.array([a1.buffer(1e-9).covers(Point(x, y)) for x, y in s[:, :2]])
    in2 = np.array([a2.buffer(1e-9).covers(Point(x, y)) for x, y in s[:, :2]])
    assert in1.any() and in2.any()


def test_rectangle_array_background_unchanged_regime():
    """Array-rectangle plain Hawkes still samples via the computation grid."""
    A = np.array([[0.0, 1.0], [0.0, 1.0]])
    rng = np.random.default_rng(5)
    data = pd.DataFrame({
        "X": rng.uniform(0.05, 0.95, 12),
        "Y": rng.uniform(0.05, 0.95, 12),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 12)),
    })
    m = Hawkes_Model(data, A, T_DAYS, cox_background=False, **PRIORS)
    assert not m.prepared_domain.is_polygon
    s = m._sim_hawkes_bg({"a_0": 3.0}, rng=np.random.default_rng(6))
    assert len(s) > 10
    assert np.all((s[:, 0] >= 0.0) & (s[:, 0] <= 1.0))
    assert np.all((s[:, 1] >= 0.0) & (s[:, 1] <= 1.0))
    # Grid-based areas still sum to the unit rectangle mass.
    A_ = m.args["A_"]
    areas = (m.A.area / ((A_[0, 1] - A_[0, 0]) * (A_[1, 1] - A_[1, 0]))).values
    assert float(areas.sum()) == pytest.approx(1.0, rel=1e-9)
