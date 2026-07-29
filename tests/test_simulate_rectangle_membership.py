"""Public simulate() on rectangle/array domains must not duplicate grid-edge points.

Pre-3f follow-up on tip 04799d9: polygon simulate already filters via
PreparedDomain.union_geometry, but rectangle/array domains still used
``points.sjoin(self.A)`` where ``self.A`` is the 25x25 computational grid.
A physical point on an internal grid edge or vertex matches two or four cells
and is therefore duplicated. Membership must be a unique, boundary-inclusive
test against the authoritative prepared rectangle bounds -- not
``drop_duplicates``, and not treating computational cells as output regions.
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
    alpha=np.float32(0.0),
    beta=np.float32(1.0),
    sigmax_2=np.float32(20.0 ** 2),
)

_SEASONAL_DECODER = os.path.join(
    os.path.dirname(bstpp.__file__), "decoders", "decoder_1d_T24_circ_small_l8")
needs_decoder = pytest.mark.skipif(
    not os.path.isfile(_SEASONAL_DECODER),
    reason="seasonal decoder artifact absent",
)


def _rect_model():
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
    return m


def _inject_bg_point(model, xy_t, monkeypatch):
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
        monkeypatch.setattr(model, "_sim_cox", _bg)


def _inject_bg_events(model, events, monkeypatch):
    events = np.asarray(events, dtype=np.float64)

    def _bg(*_a, **_k):
        return events.copy()

    def _off(bg, *_a, **_k):
        return np.asarray(bg, dtype=np.float64)

    monkeypatch.setattr(model, "_sim_hawkes_bg", _bg)
    monkeypatch.setattr(model, "_sim_cox", _bg)
    monkeypatch.setattr(model, "_sim_offspring", _off)


def _internal_grid_lines(model):
    """Interior cell-boundary coordinates of the 25x25 computational grid."""
    cg = model.comp_grid
    xs = sorted({float(v) for v in cg.bounds.minx} | {float(v) for v in cg.bounds.maxx})
    ys = sorted({float(v) for v in cg.bounds.miny} | {float(v) for v in cg.bounds.maxy})
    # Drop outer rectangle edges; keep strictly internal lines.
    x0, x1 = float(model.prepared_domain.bounds[0, 0]), float(model.prepared_domain.bounds[0, 1])
    y0, y1 = float(model.prepared_domain.bounds[1, 0]), float(model.prepared_domain.bounds[1, 1])
    xs_int = [x for x in xs if x0 < x < x1]
    ys_int = [y for y in ys if y0 < y < y1]
    assert len(xs_int) >= 1 and len(ys_int) >= 1
    return xs_int, ys_int


def test_hawkes_rectangle_internal_grid_edge_once(monkeypatch):
    m = _rect_model()
    xs_int, ys_int = _internal_grid_lines(m)
    x_edge, y_mid = xs_int[len(xs_int) // 2], 50.0
    # Confirm the legacy join multiplies (defect witness).
    probe = gpd.GeoDataFrame(
        {"X": [x_edge], "Y": [y_mid], "T": [5.0],
         "geometry": [Point(x_edge, y_mid)]})
    assert len(probe.sjoin(m.A[["geometry"]])) >= 2

    _inject_bg_point(m, [x_edge, y_mid, 5.0], monkeypatch)
    out = m.simulate(parameters=dict(HAWKES_PARAMS))
    assert list(out.columns) == ["X", "Y", "T", "geometry"]
    assert len(out) == 1
    assert float(out.iloc[0]["X"]) == pytest.approx(x_edge)
    assert float(out.iloc[0]["Y"]) == pytest.approx(y_mid)


def test_hawkes_rectangle_internal_grid_vertex_once(monkeypatch):
    m = _rect_model()
    xs_int, ys_int = _internal_grid_lines(m)
    x_v, y_v = xs_int[len(xs_int) // 2], ys_int[len(ys_int) // 2]
    probe = gpd.GeoDataFrame(
        {"X": [x_v], "Y": [y_v], "T": [5.0], "geometry": [Point(x_v, y_v)]})
    assert len(probe.sjoin(m.A[["geometry"]])) >= 4

    _inject_bg_point(m, [x_v, y_v, 5.0], monkeypatch)
    out = m.simulate(parameters=dict(HAWKES_PARAMS))
    assert len(out) == 1
    assert float(out.iloc[0]["X"]) == pytest.approx(x_v)
    assert float(out.iloc[0]["Y"]) == pytest.approx(y_v)


@pytest.mark.parametrize(
    "xy",
    [
        (0.0, 50.0),    # left edge
        (100.0, 50.0),  # right edge
        (50.0, 0.0),    # bottom edge
        (50.0, 100.0),  # top edge
        (0.0, 0.0),     # corner
        (100.0, 100.0), # corner
    ],
)
def test_hawkes_rectangle_outer_boundary_and_corner_once(monkeypatch, xy):
    m = _rect_model()
    x, y = xy
    _inject_bg_point(m, [x, y, 5.0], monkeypatch)
    out = m.simulate(parameters=dict(HAWKES_PARAMS))
    assert len(out) == 1
    assert float(out.iloc[0]["X"]) == pytest.approx(x)
    assert float(out.iloc[0]["Y"]) == pytest.approx(y)


def test_hawkes_rectangle_outside_excluded(monkeypatch):
    m = _rect_model()
    _inject_bg_point(m, [150.0, 50.0, 5.0], monkeypatch)
    out = m.simulate(parameters=dict(HAWKES_PARAMS))
    assert len(out) == 0


def test_hawkes_rectangle_interior_once(monkeypatch):
    m = _rect_model()
    _inject_bg_point(m, [33.0, 67.0, 5.0], monkeypatch)
    out = m.simulate(parameters=dict(HAWKES_PARAMS))
    assert len(out) == 1
    assert float(out.iloc[0]["X"]) == pytest.approx(33.0)


def test_hawkes_rectangle_one_row_per_retained_event(monkeypatch):
    m = _rect_model()
    xs_int, ys_int = _internal_grid_lines(m)
    x_v, y_v = xs_int[0], ys_int[0]
    events = np.array([
        [25.0, 25.0, 3.0],       # interior
        [x_v, 40.0, 4.0],        # internal edge
        [x_v, y_v, 5.0],         # internal vertex
        [0.0, 0.0, 6.0],         # outer corner
        [200.0, 200.0, 7.0],     # outside
    ], dtype=np.float64)
    _inject_bg_events(m, events, monkeypatch)
    out = m.simulate(parameters=dict(HAWKES_PARAMS))
    assert len(out) == 4
    assert list(out.columns) == ["X", "Y", "T", "geometry"]
    # Exactly one output row per retained generated event (no join inflation).
    assert len(out) == out.geometry.nunique(dropna=False) or out.index.is_unique
    xy = np.asarray(out[["X", "Y"]], dtype=np.float64)
    assert len(np.unique(np.round(xy, decimals=9), axis=0)) == 4


@needs_decoder
def test_lgcp_rectangle_internal_grid_edge_once(monkeypatch):
    A = np.array([[0.0, 100.0], [0.0, 100.0]])
    data = pd.DataFrame({
        "X": np.linspace(10, 90, 8),
        "Y": np.linspace(10, 90, 8),
        "T": np.linspace(1, T_DAYS - 1, 8),
    })
    m = LGCP_Model(data, A, T_DAYS, a_0=dist.Normal(0, 5))
    assert not m.prepared_domain.is_polygon
    xs_int, _ = _internal_grid_lines(m)
    x_edge = xs_int[len(xs_int) // 2]
    _inject_bg_point(m, [x_edge, 50.0, 5.0], monkeypatch)
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


def _poly_hawkes(domain):
    union = (
        domain.geometry.union_all()
        if hasattr(domain.geometry, "union_all")
        else domain.geometry.unary_union)
    rng = np.random.default_rng(1)
    minx, miny, maxx, maxy = union.bounds
    xs, ys = [], []
    while len(xs) < 8:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if union.covers(Point(x, y)):
            xs.append(x)
            ys.append(y)
    data = pd.DataFrame({
        "X": xs, "Y": ys,
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 8)),
    })
    return Hawkes_Model(
        data, domain, T_DAYS, cox_background=False,
        excitation_support="rectangle", **HAWKES_PRIORS)


def test_no_regression_polygon_single_and_disjoint(monkeypatch):
    """Polygon / single-row / disjoint cases remain one-row-per-event."""
    single = gpd.GeoDataFrame(geometry=[box(0, 0, 80, 80)])
    m1 = _poly_hawkes(single)
    _inject_bg_point(m1, [10.0, 10.0, 4.0], monkeypatch)
    assert len(m1.simulate(parameters=dict(HAWKES_PARAMS))) == 1

    disjoint = gpd.GeoDataFrame(
        geometry=[box(0, 0, 40, 40), box(80, 80, 120, 120)])
    m2 = _poly_hawkes(disjoint)
    _inject_bg_point(m2, [90.0, 90.0, 4.0], monkeypatch)
    assert len(m2.simulate(parameters=dict(HAWKES_PARAMS))) == 1

    overlap = gpd.GeoDataFrame(
        geometry=[box(0, 0, 100, 100), box(50, 50, 150, 150)])
    m3 = _poly_hawkes(overlap)
    _inject_bg_point(m3, [75.0, 75.0, 5.0], monkeypatch)
    assert len(m3.simulate(parameters=dict(HAWKES_PARAMS))) == 1
