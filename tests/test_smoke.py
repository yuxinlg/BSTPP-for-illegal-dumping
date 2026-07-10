"""Smoke tests pinning the audit fixes. Each test maps to a prior fix:

- test_single_factor_site      -> duplicate likelihood factor removed (one factor/model)
- test_*_traces                -> LGCP loglik repair, b_0 site, models construct + finite loglik
- test_window_default          -> window/coords constructor fix + excitation-integral truncation
- test_A_derivation            -> seasonal coord A derived from T; supplied A validated
- test_simulate_runs           -> _sim_offspring range() fix, reversed uniform bounds, no 'A', full window

Everything is seeded. Cox/LGCP tests skip with a clear message if the seasonal
decoder artifact is missing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import bstpp

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import jax
import numpyro.distributions as dist
from numpyro import handlers
import pytest

import bstpp
from bstpp.main import Hawkes_Model, LGCP_Model

# --- seasonal decoder presence -> skip marker for cox/lgcp tests ---
_SEASONAL_DECODER = os.path.join(os.path.dirname(bstpp.__file__), "decoders", "decoder_1d_T24_circ_small_l8")
HAS_SEASONAL_DECODER = os.path.isfile(_SEASONAL_DECODER)
needs_decoder = pytest.mark.skipif(
    not HAS_SEASONAL_DECODER,
    reason="seasonal decoder artifact 'bstpp/decoders/decoder_1d_T24_circ_small_l8' is absent",
)

# --- synthetic dataset: ~60 events, unit square, T ascending over ~2.5 years, NO 'A' column ---
T_DAYS = 2.5 * 365.0  # 912.5 -> a partial final year exists
_rng = np.random.RandomState(0)
_N = 60
DATA = pd.DataFrame({
    "X": _rng.uniform(0.05, 0.95, _N),
    "Y": _rng.uniform(0.05, 0.95, _N),
    "T": np.sort(_rng.uniform(0, T_DAYS, _N)),
})
A_RECT = np.array([[0.0, 1.0], [0.0, 1.0]])          # rectangle spec (fast, for trace tests)
A_GDF = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]})  # single polygon (clean sjoin for simulate)
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))


def make_hawkes(cox, A=A_RECT, **extra):
    return Hawkes_Model(DATA, A, T_DAYS, cox_background=cox, **PRIORS, **extra)


# --- covariate dataset: ~40 events on the unit square + a 2x2 covariate grid ---
_cov_rng = np.random.RandomState(2)
_N_COV = 40
COV_DATA = pd.DataFrame({
    "X": _cov_rng.uniform(0.05, 0.95, _N_COV),
    "Y": _cov_rng.uniform(0.05, 0.95, _N_COV),
    "T": np.sort(_cov_rng.uniform(0, T_DAYS, _N_COV)),
})
# 2x2 grid of covariate polygons over the unit square, single covariate column.
_COV_CELLS = [box(x, y, x + 0.5, y + 0.5) for y in (0.0, 0.5) for x in (0.0, 0.5)]
COV_GDF = gpd.GeoDataFrame({"cov": [0.5, -1.0, 1.5, -0.5], "geometry": _COV_CELLS})


def make_hawkes_cov():
    return Hawkes_Model(COV_DATA, A_RECT, T_DAYS, cox_background=False,
                        spatial_cov=COV_GDF, cov_names=["cov"], **PRIORS)


def make_lgcp(A=A_RECT):
    return LGCP_Model(DATA, A, T_DAYS, a_0=dist.Normal(0, 5))


def _trace(model):
    return handlers.trace(handlers.seed(model.model, jax.random.PRNGKey(0))).get_trace(model.args)


def _factor_sites(tr):
    return [n for n, s in tr.items() if s.get("type") == "sample" and type(s["fn"]).__name__ == "Unit"]


def _loglik(tr):
    return float(np.asarray(tr["loglik"]["value"]))


# ---- duplicate-factor fix: exactly one factor site per model ----
def test_single_factor_site():
    tr = _trace(make_hawkes(cox=False))
    assert _factor_sites(tr) == ["loglik_factor"]


# ---- models construct and trace with finite loglik ----
def test_hawkes_traces():
    tr = _trace(make_hawkes(cox=False))
    assert np.isfinite(_loglik(tr))


@needs_decoder
def test_cox_hawkes_traces():
    tr = _trace(make_hawkes(cox=True))
    assert np.isfinite(_loglik(tr))


@needs_decoder
def test_lgcp_traces():
    tr = _trace(make_lgcp())
    assert np.isfinite(_loglik(tr))


# ---- window/coords constructor fix + excitation-integral truncation ----
def test_window_default():
    m = make_hawkes(cox=False)                 # no window kwarg -> no KeyError
    assert "coords" in m.args

    def excite(window):
        mm = make_hawkes(cox=False, window=window)
        return float(np.asarray(_trace(mm)["Itot_excite"]["value"]))

    Ie_default = excite(None)                  # default window == T
    Ie_untrunc = excite(1e9)                   # explicitly untruncated
    Ie_small = excite(2.0)                      # truncated
    assert abs(Ie_default - Ie_untrunc) < 1e-6  # default matches the full integral
    assert Ie_small < Ie_default                # truncation strictly decreases it


# ---- seasonal coord A derived from T; supplied A validated ----
def test_A_derivation():
    m = make_hawkes(cox=False)                 # DATA has no 'A' -> fit path works
    assert "a_events" in m.args
    bad = DATA.copy()
    bad["A"] = (DATA["T"].values + 100.0) % 365.0   # inconsistent with T
    with pytest.raises(ValueError):
        Hawkes_Model(bad, A_RECT, T_DAYS, cox_background=False, **PRIORS)


# ---- simulation fixes: range() offspring, correct uniform bounds, no 'A', full window ----
def test_simulate_runs():
    m = make_hawkes(cox=False, A=A_GDF)

    # _sim_offspring must iterate (range regression), not crash on a scalar Poisson draw
    np.random.seed(0)
    bg = np.array([[0.5, 0.5, 10.0], [0.3, 0.7, 20.0], [0.6, 0.4, 30.0]])
    off = m._sim_offspring(bg.copy(), {"alpha": 0.3, "beta": 1.0, "sigmax_2": 0.25})
    assert off.ndim == 2 and off.shape[1] == 3

    # full simulate: X,Y,T only, all T in [0, T_DAYS], events in the final partial year
    np.random.seed(1)
    params = {"a_0": 2.0, "f_t": np.zeros(m.args["n_t"]), "f_a": np.zeros(m.args["n_s"]),
              "f_xy": np.zeros(m.args["n_xy"] ** 2), "alpha": 0.05, "beta": 1.0, "sigmax_2": 0.25}
    sim = m.simulate(params)
    assert "A" not in sim.columns
    assert {"X", "Y", "T"}.issubset(sim.columns)
    Tr = sim["T"].values
    assert (Tr >= 0).all() and (Tr <= T_DAYS + 1e-6).all()
    assert (Tr > 2 * 365.0).any()              # events in the 2.0-2.5 year partial window


# ---- per-cell covariate background (fork regression: mu_xyt was indexed at events
#      before contracting against cell areas -> shape error) ----
def test_hawkes_covariate_background():
    m = make_hawkes_cov()
    args = m.args
    assert args["spatial_cov"].shape == (4, 1)     # 4 covariate cells, 1 covariate
    assert len(args["cov_area"]) == 4
    assert len(args["cov_ind"]) == _N_COV          # events index into the cell vector

    tr = _trace(m)

    # (a) trace succeeds with finite loglik
    assert np.isfinite(_loglik(tr))

    # (b) mu_xyt is per cell (4,), not per event
    mu_xyt = np.asarray(tr["mu_xyt"]["value"])
    assert mu_xyt.shape == (4,)
    assert mu_xyt.shape != (_N_COV,)

    # (c) Itot_txy_back == exp(a_0 + b_0) @ cov_area * T, recomputed from traced a_0 and w
    a_0_val = float(np.asarray(tr["a_0"]["value"]))
    w_val = np.asarray(tr["w"]["value"])
    b_0_vals = np.asarray(args["spatial_cov"]) @ w_val
    expected = np.exp(a_0_val + b_0_vals) @ np.asarray(args["cov_area"]) * args["T"]
    got = float(np.asarray(tr["Itot_txy_back"]["value"]))
    assert np.isclose(got, expected, rtol=1e-5)


# ---- log_expected_likelihood must rebuild excitation pairs on the held-out events
#      (training pairs were silently reused, segment_sum dropping OOR indices) ----
_test_rng = np.random.RandomState(3)
_N_TEST = 25
TEST_DATA = pd.DataFrame({
    "X": _test_rng.uniform(0.05, 0.95, _N_TEST),
    "Y": _test_rng.uniform(0.05, 0.95, _N_TEST),
    "T": np.sort(_test_rng.uniform(0, T_DAYS, _N_TEST)),
})
_LEL_PARAMS = {k: np.float32(v) for k, v in
               dict(a_0=1.0, alpha=0.3, beta=2.0, sigmax_2=0.1).items()}


def _build_test_args(train_model, test_data):
    """Mirror log_expected_likelihood's test_args construction (incl. pair rebuild)."""
    from bstpp.utils import aligned_difference_pairs
    ta, points = train_model._scale_xyt(test_data, train_model.args.copy(),
                                        train_model.comp_grid)
    if train_model.args['model'] in ('hawkes', 'cox_hawkes'):
        coords, t_vals, x_vals, y_vals = aligned_difference_pairs(
            ta['t_events'], ta['xy_events'][0], ta['xy_events'][1],
            train_model.args['window'],
            spatial_window=train_model.args.get('spatial_window'))
        ta['coords'], ta['t_vals'], ta['x_vals'], ta['y_vals'] = coords, t_vals, x_vals, y_vals
    for k in ['batch_size', 'num_samples', 'num_warmup', 'num_chains', 'thinning']:
        ta.pop(k, None)
    return ta


def _loglik_at(model, args, params):
    seeded = handlers.substitute(handlers.seed(model.model, jax.random.PRNGKey(0)), params)
    return float(np.asarray(handlers.trace(seeded).get_trace(args)["loglik"]["value"]))


def test_log_expected_likelihood_rebuilds_pairs():
    train = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False, **PRIORS)   # 60-event train
    fresh = Hawkes_Model(TEST_DATA, A_RECT, T_DAYS, cox_background=False, **PRIORS)  # 25-event test

    # loglik from the train model on rebuilt test_args must match a model built
    # directly on the test data (both at the same fixed parameters).
    ll_train = _loglik_at(train, _build_test_args(train, TEST_DATA), _LEL_PARAMS)
    ll_fresh = _loglik_at(fresh, fresh.args, _LEL_PARAMS)
    assert np.isfinite(ll_train)
    assert abs(ll_train - ll_fresh) < 1e-4

    # events beyond the training horizon [0, T] are rejected
    bad = TEST_DATA.copy()
    bad.loc[bad.index[-1], "T"] = T_DAYS * 1.5
    with pytest.raises(ValueError):
        train.log_expected_likelihood(bad)
