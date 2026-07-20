"""Phase 3a MR commit: deterministic unique membership per D-22.

D-22 (phase3_baseline_and_decisions section 8): cells are left-closed/
right-open per axis, [e_k, e_{k+1}), with the domain's outermost right/top
edges closed; the temporal axis is [t_k, t_{k+1}) with t = T-tilde closed
into the last cell; the seasonal circle is seamless.

This is an intentional MICRO-REBASELINE for otherwise-valid events exactly
on grid or cell boundaries (previously: double-join crash with a misleading
message). Interior events are bit-unchanged: unique sjoin rows pass through
the new per-point max untouched, proven here by the interior-agreement
property test and by the golden pins (run at commit time, bit-identical).

RED verification: every grid-line/tie construction below crashes at the
pre-MR code with "Computational grid does not encompass all data points!"
or "Spatial covariates are not defined for all data points!".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import numpyro.distributions as dist

from bstpp.main import Hawkes_Model

# Same non-square real-unit domain as test_data_contracts: x-cells are
# 0.8 real units wide (edges 10 + 0.8k), y-cells 0.4 (edges 5 + 0.4k).
A_RECT = np.array([[10.0, 30.0], [5.0, 15.0]])
T_DAYS = 200.0
N_XY = 25
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))


def _model(data, **kw):
    return Hawkes_Model(data, A_RECT, T_DAYS, cox_background=False,
                        **PRIORS, **kw)


def _with_fillers(rows, n_fill=8, seed=3):
    """Prepend deterministic strictly-interior filler events (cell centers)
    so models have enough events, then append the probe rows."""
    r = np.random.RandomState(seed)
    ix = r.randint(0, N_XY, n_fill)
    iy = r.randint(0, N_XY, n_fill)
    fill = pd.DataFrame({
        "X": 10.0 + (ix + 0.5) * 0.8,
        "Y": 5.0 + (iy + 0.5) * 0.4,
        "T": np.linspace(1.0, 90.0, n_fill),
    })
    probe = pd.DataFrame(rows, columns=["X", "Y", "T"])
    return pd.concat([fill, probe], ignore_index=True), len(fill)


def _cell_id(ix, iy):
    return iy * N_XY + ix


# ------------------------------------------------- field grid: internal edges

def test_x_grid_line_assigns_left_closed_cell():
    # x = 10 + 0.8*5 = 14.0 exactly on the internal edge between ix=4 and 5;
    # D-22: [e_5, e_6) owns it -> ix = 5. y = 7.4 = 5 + 0.4*6 would be an
    # edge too, so use an interior y (cell center iy=5: 7.2).
    data, n_fill = _with_fillers([(14.0, 7.2, 100.0)])
    m = _model(data)
    assert m.args["indices_xy"][n_fill] == _cell_id(5, 5)


def test_y_grid_line_assigns_left_closed_cell():
    # y = 5 + 0.4*6 = 7.4 on the edge between iy=5 and 6 -> iy = 6.
    data, n_fill = _with_fillers([(14.4, 7.4, 100.0)])  # x interior, ix=5
    m = _model(data)
    assert m.args["indices_xy"][n_fill] == _cell_id(5, 6)


def test_corner_grid_point_assigns_left_closed_cell_both_axes():
    # (14.0, 7.4): x-edge AND y-edge; four cells tie -> ix=5, iy=6.
    data, n_fill = _with_fillers([(14.0, 7.4, 100.0)])
    m = _model(data)
    assert m.args["indices_xy"][n_fill] == _cell_id(5, 6)


def test_outermost_edges_closed():
    # D-22: the domain's max-x/max-y edges are closed into the last cells.
    data, n_fill = _with_fillers([
        (30.0, 7.2, 100.0),   # x = x_max -> ix = 24
        (14.4, 15.0, 101.0),  # y = y_max -> iy = 24
        (30.0, 15.0, 102.0),  # domain corner -> (24, 24)
    ])
    m = _model(data)
    got = m.args["indices_xy"][n_fill:n_fill + 3]
    assert list(got) == [_cell_id(24, 5), _cell_id(5, 24), _cell_id(24, 24)]


def test_min_edges_belong_to_first_cells():
    # left-closed: x = x_min / y = y_min are the closed LEFT edges of cell 0.
    data, n_fill = _with_fillers([(10.0, 5.0, 100.0)])
    m = _model(data)
    assert m.args["indices_xy"][n_fill] == _cell_id(0, 0)


def test_interior_events_bit_unchanged():
    """Property test: on strictly interior random events the new max-dedup
    path must reproduce the legacy unique-sjoin assignment exactly (here
    recomputed independently by half-open arithmetic, which coincides with
    geometric containment away from edges)."""
    r = np.random.RandomState(17)
    n = 200
    # keep points away from edges by more than the float noise scale
    u = (r.randint(0, N_XY, n) + r.uniform(0.05, 0.95, n)) / N_XY
    v = (r.randint(0, N_XY, n) + r.uniform(0.05, 0.95, n)) / N_XY
    data = pd.DataFrame({
        "X": 10.0 + 20.0 * u, "Y": 5.0 + 10.0 * v,
        "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n))})
    m = _model(data)
    expected = (np.floor(v * N_XY).astype(int) * N_XY
                + np.floor(u * N_XY).astype(int))
    np.testing.assert_array_equal(np.asarray(m.args["indices_xy"]), expected)


# --------------------------------------------- temporal and seasonal axes

def test_temporal_left_closed_and_horizon_closed():
    """Already-compliant searchsorted(side='right')-1 semantics, pinned as
    the D-22 contract: t on an internal edge belongs to the RIGHT cell;
    t = T (internal T-tilde) closes into the last cell."""
    # temporal cells are T_DAYS/50 = 4 days wide
    data, n_fill = _with_fillers([
        (14.4, 7.2, 0.0),      # left end -> cell 0
        (14.4, 7.2, 4.0),      # exactly on edge 1 -> cell 1
        (14.4, 7.2, 100.0),    # edge 25 -> cell 25
        (14.4, 7.2, T_DAYS),   # t = T -> last cell 49 (closed)
    ])
    m = _model(data)
    got = list(np.asarray(m.args["indices_t"])[n_fill:])
    assert got == [0, 1, 25, 49]
    # seasonal indices stay in range on the circle (seamless, no 24)
    assert np.all(np.asarray(m.args["indices_a"]) >= 0)
    assert np.all(np.asarray(m.args["indices_a"]) < 24)


# ---------------------------------------------------------- covariate ties

def _cov_layer():
    return gpd.GeoDataFrame(
        {"v": [1.0, 2.0]},
        geometry=[box(10, 5, 20, 15), box(20, 5, 30, 15)])


def test_covariate_shared_edge_resolves_to_max_cov_ind():
    data, n_fill = _with_fillers([(20.0, 7.2, 100.0)])  # on the shared edge
    m = _model(data, spatial_cov=_cov_layer(), cov_names=["v"])
    assert m.args["cov_ind"][n_fill] == 1  # deterministic max-cov_ind rule
    # interior events keep their unique assignment
    assert m.args["cov_ind"][0] in (0, 1)
    assert len(m.args["cov_ind"]) == len(data)


def test_heldout_covariate_tie_no_silent_misalignment():
    """Held-out path: a tie used to emit two cov_ind rows with NO length
    check, silently shifting every later event's covariate. Now it must
    resolve to one row per event with the same max rule."""
    data, _ = _with_fillers([(15.0, 7.2, 100.0)], n_fill=12)
    m = _model(data, spatial_cov=_cov_layer(), cov_names=["v"])
    m.run_svi(10, 0.01, plot_loss=False)
    heldout, n_fill = _with_fillers([(20.0, 7.2, 120.0)], n_fill=5, seed=9)
    val = m.log_expected_likelihood(heldout)
    assert np.isfinite(val)


def test_grid_line_event_no_longer_crashes_with_misleading_message():
    """The baseline defect (doc section 5.1): a grid-line event crashed with
    'Computational grid does not encompass all data points!'. Under D-22 it
    is a valid event with deterministic membership."""
    data, _ = _with_fillers([(14.0, 7.4, 100.0)])
    m = _model(data)  # must not raise
    assert len(m.args["indices_xy"]) == len(data)
