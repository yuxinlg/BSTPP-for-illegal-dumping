"""Public simulate() must filter against PreparedDomain.union_geometry.

Pre-3f follow-up (audit of tip 938e32b): internal plain-Hawkes background
sampling already uses the canonical union, but both public Hawkes and LGCP
``simulate()`` paths previously finished with ``points.sjoin(self.A)``. With
overlapping input rows that join duplicates a physical point once per matching
row. Membership in the canonical union must prevent duplication — not
``drop_duplicates``.
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

import bstpp
from bstpp.main import Hawkes_Model, LGCP_Model

T_DAYS = 30.0
HAWKES_PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(40.0),
)
HAWKES_PARAMS = dict(
    a_0=np.float32(0.0),
    alpha=np.float32(0.0),  # no offspring — isolate spatial filter
    beta=np.float32(1.0),
    sigmax_2=np.float32(20.0 ** 2),
)

_SEASONAL_DECODER = os.path.join(
    os.path.dirname(bstpp.__file__), "decoders", "decoder_1d_T24_circ_small_l8")
needs_decoder = pytest.mark.skipif(
    not os.path.isfile(_SEASONAL_DECODER),
    reason="seasonal decoder artifact absent",
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


def _union(domain):
    if hasattr(domain.geometry, "union_all"):
        return domain.geometry.union_all()
    return unary_union(domain.geometry)


def _hawkes(domain):
    data = _events_in(_union(domain), n=8, seed=1)
    return Hawkes_Model(
        data, domain, T_DAYS, cox_background=False,
        excitation_support="rectangle", **HAWKES_PRIORS)


def _inject_bg_point(model, xy_t, monkeypatch):
    """Force public simulate to see exactly one controlled background event."""
    arr = np.asarray(xy_t, dtype=np.float64).reshape(1, 3)

    def _bg(*_a, **_k):
        return arr.copy()

    def _off(bg, *_a, **_k):
        return np.asarray(bg, dtype=np.float64)

    if isinstance(model, Hawkes_Model):
        monkeypatch.setattr(model, "_sim_hawkes_bg", _bg)
        monkeypatch.setattr(model, "_sim_cox", _bg)
        monkeypatch.setattr(model, "_sim_offspring", _off)
    else:
        # LGCP_Model.simulate uses _sim_cox only (no offspring cascade).
        monkeypatch.setattr(model, "_sim_cox", _bg)


def test_hawkes_simulate_overlap_returns_point_once(monkeypatch):
    a1 = box(0, 0, 100, 100)
    a2 = box(50, 50, 150, 150)
    domain = gpd.GeoDataFrame(geometry=[a1, a2])
    m = _hawkes(domain)
    # Interior of the overlap region — matches both input rows under sjoin(A).
    _inject_bg_point(m, [75.0, 75.0, 5.0], monkeypatch)
    out = m.simulate(parameters=dict(HAWKES_PARAMS))
    assert list(out.columns) == ["X", "Y", "T", "geometry"]
    assert len(out) == 1
    assert float(out.iloc[0]["X"]) == pytest.approx(75.0)
    assert float(out.iloc[0]["Y"]) == pytest.approx(75.0)


@needs_decoder
def test_lgcp_simulate_overlap_returns_point_once(monkeypatch):
    a1 = box(0, 0, 100, 100)
    a2 = box(50, 50, 150, 150)
    domain = gpd.GeoDataFrame(geometry=[a1, a2])
    data = _events_in(_union(domain), n=8, seed=2)
    m = LGCP_Model(data, domain, T_DAYS, a_0=dist.Normal(0, 5))
    _inject_bg_point(m, [75.0, 75.0, 5.0], monkeypatch)
    # Minimal decoded fields so simulate does not require posterior samples.
    n_t, n_s, n_xy = m.args["n_t"], m.args["n_s"], m.args["n_xy"]
    params = {
        "a_0": np.float32(0.0),
        "f_t": np.zeros(n_t, dtype=np.float32),
        "f_a": np.zeros(n_s, dtype=np.float32),
        "f_xy": np.zeros(n_xy ** 2, dtype=np.float32),
    }
    out = m.simulate(parameters=params)
    assert list(out.columns) == ["X", "Y", "T", "geometry"]
    assert len(out) == 1
    assert float(out.iloc[0]["X"]) == pytest.approx(75.0)


def test_hawkes_simulate_includes_boundary_of_union(monkeypatch):
    a1 = box(0, 0, 100, 100)
    a2 = box(50, 50, 150, 150)
    domain = gpd.GeoDataFrame(geometry=[a1, a2])
    m = _hawkes(domain)
    # Exact corner of the union (also a corner of a1).
    _inject_bg_point(m, [0.0, 0.0, 5.0], monkeypatch)
    out = m.simulate(parameters=dict(HAWKES_PARAMS))
    assert len(out) == 1
    assert m.prepared_domain.union_geometry.covers(out.geometry.iloc[0])


def test_hawkes_simulate_single_and_disjoint_polygons(monkeypatch):
    single = gpd.GeoDataFrame(geometry=[box(0, 0, 80, 80)])
    m1 = _hawkes(single)
    _inject_bg_point(m1, [10.0, 10.0, 4.0], monkeypatch)
    out1 = m1.simulate(parameters=dict(HAWKES_PARAMS))
    assert len(out1) == 1

    disjoint = gpd.GeoDataFrame(geometry=[box(0, 0, 40, 40), box(80, 80, 120, 120)])
    m2 = _hawkes(disjoint)
    _inject_bg_point(m2, [90.0, 90.0, 4.0], monkeypatch)
    out2 = m2.simulate(parameters=dict(HAWKES_PARAMS))
    assert len(out2) == 1
    assert m2.prepared_domain.union_geometry.covers(out2.geometry.iloc[0])


def test_hawkes_simulate_rectangle_array_unchanged(monkeypatch):
    """Array/rectangle path keeps one retained interior event (bounds filter)."""
    A = np.array([[0.0, 100.0], [0.0, 100.0]])
    data = pd.DataFrame({
        "X": np.linspace(10, 90, 8),
        "Y": np.linspace(10, 90, 8),
        "T": np.linspace(1, T_DAYS - 1, 8),
    })
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="rectangle", **HAWKES_PRIORS)
    assert not m.prepared_domain.is_polygon
    _inject_bg_point(m, [50.0, 50.0, 5.0], monkeypatch)
    out = m.simulate(parameters=dict(HAWKES_PARAMS))
    assert list(out.columns) == ["X", "Y", "T", "geometry"]
    assert len(out) == 1
    assert float(out.iloc[0]["X"]) == pytest.approx(50.0)


def test_hawkes_simulate_all_returned_inside_union_and_one_per_event(monkeypatch):
    a1 = box(0, 0, 100, 100)
    a2 = box(50, 50, 150, 150)
    domain = gpd.GeoDataFrame(geometry=[a1, a2])
    m = _hawkes(domain)
    # Two points in the overlap + one outside the union.
    events = np.array([
        [75.0, 75.0, 3.0],
        [80.0, 80.0, 4.0],
        [200.0, 200.0, 5.0],  # outside
    ], dtype=np.float64)

    def _bg(*_a, **_k):
        return events.copy()

    def _off(bg, *_a, **_k):
        return np.asarray(bg, dtype=np.float64)

    monkeypatch.setattr(m, "_sim_hawkes_bg", _bg)
    monkeypatch.setattr(m, "_sim_offspring", _off)
    out = m.simulate(parameters=dict(HAWKES_PARAMS))
    union = m.prepared_domain.union_geometry
    assert len(out) == 2
    assert all(union.covers(g) for g in out.geometry)
    # Exactly one returned row per in-domain generated event (no index inflation).
    assert out.index.is_unique or len(out) == out.geometry.nunique(dropna=False)
    xy = np.asarray(out[["X", "Y"]], dtype=np.float64)
    assert len(np.unique(np.round(xy, decimals=9), axis=0)) == 2
