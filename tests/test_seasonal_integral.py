"""Exact seasonal-diagonal time integral (overlap matrix) — likelihood + simulator.

Guards the change from the midpoint rule (exp(a_0 + f_t + f_a[season_idx_of_t]))
to the closed form (season_overlap @ exp(f_a)) shared by the likelihood and _sim_cox:

  (a) overlap matrix W is (n_t, n_s), non-negative, rows sum to the internal cell width;
  (b) exact vs a 1000x-refined brute-force reference (offsets 0 and 37), rtol 1e-4;
  (c) the new value is closer to the reference than the OLD midpoint rule (improvement);
  (d) simulator Ig == likelihood Itot_time (the breakpoint sum equals the overlap
      contraction) to rtol 1e-5, and the real _sim_cox uses that same Ig;
  (e) scripts/recover_test.py still imports and its season_idx_of_t usage runs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import jax
import numpyro.distributions as dist
from numpyro import handlers
import pytest

import bstpp
from bstpp.main import Hawkes_Model

_SEASONAL_DECODER = os.path.join(os.path.dirname(bstpp.__file__), "decoders", "decoder_1d_T24_circ_small_l8")
needs_decoder = pytest.mark.skipif(
    not os.path.isfile(_SEASONAL_DECODER),
    reason="seasonal decoder artifact absent",
)

T_DAYS = 2.5 * 365.0
_rng = np.random.RandomState(0)
_N = 40
DATA = pd.DataFrame({
    "X": _rng.uniform(0.05, 0.95, _N),
    "Y": _rng.uniform(0.05, 0.95, _N),
    "T": np.sort(_rng.uniform(0, T_DAYS, _N)),
})
A_RECT = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))


def _model(offset_seasonal=0):
    # season_overlap is built for every model in __init__, so a plain (decoder-free)
    # Hawkes model is enough to exercise the overlap matrix and the integral formulas.
    return Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False,
                        offset_seasonal=offset_seasonal, **PRIORS)


def _exact_Itot_time(args, a_0, f_t, f_a):
    """The new closed form: season_overlap @ exp(f_a), contracted with exp(a_0 + f_t)."""
    W = np.asarray(args["season_overlap"])
    return float(np.sum(np.exp(a_0 + f_t) * (W @ np.exp(f_a))))


def _old_midpoint_Itot_time(args, a_0, f_t, f_a):
    """The OLD midpoint rule, recomputed inline via season_idx_of_t."""
    sidx = np.asarray(args["season_idx_of_t"])
    rate_time = np.exp(a_0 + f_t + f_a[sidx])
    return float(rate_time.sum() / args["n_t"] * args["T"])


def _reference_Itot_time(model, a_0, f_t, f_a, refine=10000):
    """Brute-force integral on a refine-x uniform grid (midpoint field assignment).

    The reference carries its OWN O(1/refine) discretization error at seasonal
    straddles; 10000x refinement drives that well below the 1e-4 tolerance so the
    comparison genuinely tests the exactness of the closed form."""
    args = model.args
    n_t, n_s, T_int = args["n_t"], args["n_s"], args["T"]
    offset, S, realT = args["offset_seasonal"], model.S, model.T
    n_total = n_t * refine
    dw = T_int / n_total
    mid = (np.arange(n_total) + 0.5) * dw
    t_cell = np.clip((mid / (T_int / n_t)).astype(int), 0, n_t - 1)
    day = mid * (realT / T_int)
    s_cell = np.clip(((day + offset) % S / S * n_s).astype(int), 0, n_s - 1)
    return float(np.sum(np.exp(a_0 + f_t[t_cell] + f_a[s_cell]) * dw))


def _breakpoint_Ig(model, a_0, f_t, f_a):
    """Mirror _sim_cox's exact breakpoint-partition time integral, returning Ig."""
    args = model.args
    n_t, T_int = args["n_t"], args["T"]
    n_s, offset = args["n_s"], args["offset_seasonal"]
    edges = np.arange(n_t + 1) * (T_int / n_t)
    h_day = model.S / n_s
    m_lo = int(np.ceil((offset) / h_day - 1e-9))
    m_hi = int(np.floor((model.T + offset) / h_day + 1e-9))
    cross_int = (np.arange(m_lo, m_hi + 1) * h_day - offset) * (T_int / model.T)
    bp = np.unique(np.clip(np.concatenate([edges, cross_int]), 0.0, T_int))
    seg_lo, seg_len = bp[:-1], np.diff(bp)
    mid = seg_lo + 0.5 * seg_len
    t_cell = np.clip((mid / (T_int / n_t)).astype(int), 0, n_t - 1)
    s_cell = np.clip(((mid * (model.T / T_int) + offset) % model.S / model.S * n_s).astype(int),
                     0, n_s - 1)
    g = np.exp(a_0 + f_t[t_cell] + f_a[s_cell])
    return float((g * seg_len).sum())


def test_overlap_matrix_properties():
    for offset in (0, 37):
        args = _model(offset).args
        W = np.asarray(args["season_overlap"])
        assert W.shape == (args["n_t"], args["n_s"])
        assert (W >= 0).all()
        assert np.allclose(W.sum(axis=1), args["T"] / args["n_t"], rtol=1e-6)


def test_exact_matches_reference_and_beats_midpoint():
    for offset in (0, 37):
        model = _model(offset)
        args = model.args
        n_t, n_s = args["n_t"], args["n_s"]
        rng = np.random.RandomState(100 + offset)
        for _ in range(3):
            a_0 = float(rng.normal(0, 1))
            f_t = rng.normal(0, 0.7, n_t)
            f_a = rng.normal(0, 0.7, n_s)

            ref = _reference_Itot_time(model, a_0, f_t, f_a)
            new = _exact_Itot_time(args, a_0, f_t, f_a)
            old = _old_midpoint_Itot_time(args, a_0, f_t, f_a)

            # (b) exactness
            assert np.isclose(new, ref, rtol=1e-4)
            # (c) the change is an improvement, not a lateral move
            assert abs(old - ref) > abs(new - ref)


def test_sim_likelihood_integral_identity():
    # (d) breakpoint sum (simulator) == overlap contraction (likelihood), rtol 1e-5.
    for offset in (0, 37):
        model = _model(offset)
        rng = np.random.RandomState(7 + offset)
        for _ in range(3):
            a_0 = float(rng.normal())
            f_t = rng.normal(0, 0.7, model.args["n_t"])
            f_a = rng.normal(0, 0.7, model.args["n_s"])
            Ig = _breakpoint_Ig(model, a_0, f_t, f_a)
            Itot_time = _exact_Itot_time(model.args, a_0, f_t, f_a)
            assert np.isclose(Ig, Itot_time, rtol=1e-5)


@needs_decoder
def test_real_sim_cox_uses_exact_integral():
    # exercise the actual _sim_cox: its Poisson mean is Ig * Ih; confirm the Ig it uses
    # matches both the breakpoint mirror and the model-traced Itot_time.
    model = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=True, **PRIORS)
    tr = handlers.trace(handlers.seed(model.model, jax.random.PRNGKey(0))).get_trace(model.args)
    a_0 = float(np.asarray(tr["a_0"]["value"]))
    f_t = np.asarray(tr["f_t"]["value"])
    f_a = np.asarray(tr["f_a"]["value"])
    f_xy = np.asarray(tr["f_xy"]["value"])
    Itot_time = float(np.asarray(tr["Itot_time"]["value"]))

    Ig_bp = _breakpoint_Ig(model, a_0, f_t, f_a)
    assert np.isclose(Ig_bp, Itot_time, rtol=1e-5)

    # capture the real Poisson mean (Ig * Ih) from inside _sim_cox
    captured = {}
    real_poisson = np.random.poisson

    def spy(lam, *a, **k):
        captured["lam"] = float(lam)
        return real_poisson(lam, *a, **k)

    params = {"a_0": a_0, "f_t": f_t, "f_a": f_a, "f_xy": f_xy}
    np.random.seed(0)
    orig = np.random.poisson
    try:
        np.random.poisson = spy
        bg = model._sim_cox(params)
    finally:
        np.random.poisson = orig
    assert bg.ndim == 2 and bg.shape[1] == 3
    # STRUCTURAL COUNT TARGET: the Poisson mean must equal the likelihood's own
    # compensator, Itot_time * Itot_xy, both read from the trace. This test
    # previously MIRRORED the duplicated self-join construction to recover Ig
    # (see test_sim_cox_array_domain_support_regression for the defect and its
    # quantification); it now asserts the mathematical invariant. Test-edit
    # sign-off: Terhi, alongside the dedicated regression test.
    Itot_xy = float(np.asarray(tr["Itot_xy"]["value"]))
    assert np.isclose(captured["lam"], Itot_time * Itot_xy, rtol=1e-5)


def test_recover_test_imports_and_season_idx_runs():
    # (e) scripts/recover_test.py imports (no signature change) and its
    # season_idx_of_t consumer still runs.
    import recover_test
    model = _model()
    sidx = model.args["season_idx_of_t"]
    n_t, n_s = model.args["n_t"], model.args["n_s"]
    val = recover_test.intercept_combination(
        1.0, np.zeros(n_t), np.arange(n_s, dtype=float), np.zeros(25 ** 2), sidx)
    assert np.isfinite(val)


@needs_decoder
def test_sim_cox_array_domain_support_regression():
    """REGRESSION (array-domain spatial support): _sim_cox's no-covariate cell
    support must be the constructor's integration arrays -- one row per
    in-domain cell -- never a geometric self-join.

    HISTORY: the old branch built geo_df = comp_grid.sjoin(self.A); for an
    ARRAY domain self.A IS the comp grid, so the join returned cell-neighbor
    pairs. Exact geometry predicts 5329 rows (degrees 4/6/9 for corner/edge/
    interior, mean 8.53); the float-constructed grid actually produced 4761
    rows with degree histogram {4: 36, 6: 228, 9: 361} because ~11% of
    adjacencies fail at the ulp level (cell edges built as j*w + w vs
    (j+1)*w) -- i.e. the sampling weights were FLOAT-NOISE-DEPENDENT, the
    Poisson mean was inflated 7.62x for a flat field (4761/625), and under a
    flat field a full-degree interior cell carried 9/4 = 2.25x the weight of
    a corner cell. GeoDataFrame domains were unaffected (their sjoin is
    against a genuine polygon and deduplicated by np.unique).

    With a ZERO spatial field the mathematical targets are exact:
    the cell-probability vector is uniform over exactly n_xy^2 cells, and the
    Poisson mean is Ig * Ih = T_internal * 1.
    """
    model = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=True, **PRIORS)
    n_cells = model.args["n_xy"] ** 2
    fi = np.asarray(model.args["integration_field_indices"])
    assert len(fi) == n_cells and len(np.unique(fi)) == n_cells

    params = {"a_0": 0.0,
              "f_t": np.zeros(model.args["n_t"], np.float32),
              "f_a": np.zeros(model.args["n_s"], np.float32),
              "f_xy": np.zeros(n_cells, np.float32)}
    captured = {}
    real_poisson, real_choice = np.random.poisson, np.random.choice

    def spy_poisson(lam, *a, **k):
        captured["lam"] = float(lam)
        return real_poisson(lam, *a, **k)

    def spy_choice(n, *a, **k):
        captured["p"] = np.asarray(k["p"])
        captured["support"] = n
        return real_choice(n, *a, **k)

    np.random.seed(3)
    try:
        np.random.poisson, np.random.choice = spy_poisson, spy_choice
        bg = model._sim_cox(params, rng=np.random.default_rng(3))
    finally:
        np.random.poisson, np.random.choice = real_poisson, real_choice

    assert bg.ndim == 2 and bg.shape[1] == 3
    # support: exactly one row per cell
    assert captured["support"] == n_cells, \
        f"cell support has {captured['support']} rows, expected {n_cells}"
    # zero field: uniform cell probabilities
    np.testing.assert_allclose(captured["p"], np.full(n_cells, 1.0 / n_cells),
                               rtol=1e-5)
    # zero field: Poisson mean = Ig * Ih = T_internal * 1
    assert np.isclose(captured["lam"], float(model.args["T"]), rtol=1e-5), \
        f"Poisson mean {captured['lam']:.4f}, mathematical target {model.args['T']}"
