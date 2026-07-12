"""Machine-checkable identities from the BSTPP guide, Section 2.6.

Coverage map (guide identity -> test / existing test):

- (I1)  Ig == Itot_time            -> test_seasonal_integral.py (existing)
- (I2)  overlap row sums           -> test_overlap_matrix_properties (existing)
- (I3)  pair-set equality          -> test_pairs.py (existing)
- (I4)  thinning <-> compensator   -> test_offspring_thinning_matches_compensator_mass (HERE)
- (I5)  single likelihood factor   -> test_single_factor_site (existing)
- (I6)  derived seasonal coord     -> test_A_derivation (existing)
- (I7)  held-out pair rebuild      -> test_log_expected_likelihood_rebuilds_pairs (existing)
- (I8)  special-case reduction     -> test_alpha_zero_reduces_cox_hawkes_to_lgcp,
                                      test_window_at_horizon_recovers_untruncated_loglik (HERE)
- (I9)  refinement invariance      -> test_covariate_refinement_invariance (HERE)
- (I10) unit invariance            -> test_spatial_affine_unit_invariance (HERE)
- (I11) conservation E[n] = E[Lam] -> test_simulated_count_matches_compensator (HERE)

These are behavior pins for the architecture rewrite: they test the mathematical
contract (what any correct implementation must output), not internals, so they
must survive any renaming or restructuring. A failure here is a finding about
the code or the spec -- never "fix" it by editing the test or loosening a
tolerance without a documented reason.

Scoping notes (deliberate, documented restrictions):
- (I8): the f_a = 0 / n_s = 1 reduction to the upstream reference model is not
  testable in-repo (no upstream likelihood available); the alpha = 0 (-> LGCP)
  and w = T (-> untruncated) clauses are tested.
- (I9): the temporal/seasonal/spatial field partitions are pinned by the
  pretrained decoders (n_t=50, n_s=24, 25x25), so refinement is only exercisable
  on the covariate partition {A_m}, which is data-given.
- (I10): the data contract fixes time in days (seasonal period 365 is not a
  parameter), so time-unit rescaling is not a supported input transformation;
  the test covers affine rescaling of the spatial coordinates, under which the
  internal representation -- and hence the internal-unit log-likelihood --
  must be exactly invariant.
- (I11): the compensator's spatial excitation mass is exact only over a
  rectangular domain (guide eq. 27), so the test uses the unit-box GeoDataFrame;
  a non-rectangular domain would make this a bounded approximation, not an
  identity. Known second-order gap: _sim_offspring discards out-of-domain
  offspring only at the end of simulate(), so spatially discarded events can
  still parent in-domain descendants the compensator never charges. With the
  small kernels used here that bias is far below the CLT tolerance; if this
  test ever fails marginally, investigate that ordering first -- do not widen
  the tolerance.
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
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import handlers
import pytest

import bstpp
from bstpp.main import Hawkes_Model, LGCP_Model

# ---- shared fixtures, matching tests/test_smoke.py conventions ----

_SEASONAL_DECODER = os.path.join(os.path.dirname(bstpp.__file__), "decoders",
                                 "decoder_1d_T24_circ_small_l8")
needs_decoder = pytest.mark.skipif(
    not os.path.isfile(_SEASONAL_DECODER),
    reason="seasonal decoder artifact 'bstpp/decoders/decoder_1d_T24_circ_small_l8' is absent",
)

T_DAYS = 2.5 * 365.0
_rng = np.random.RandomState(0)
_N = 60
DATA = pd.DataFrame({
    "X": _rng.uniform(0.05, 0.95, _N),
    "Y": _rng.uniform(0.05, 0.95, _N),
    "T": np.sort(_rng.uniform(0, T_DAYS, _N)),
})
A_RECT = np.array([[0.0, 1.0], [0.0, 1.0]])
A_GDF = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]})
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))

_cov_rng = np.random.RandomState(2)
_N_COV = 40
COV_DATA = pd.DataFrame({
    "X": _cov_rng.uniform(0.05, 0.95, _N_COV),
    "Y": _cov_rng.uniform(0.05, 0.95, _N_COV),
    "T": np.sort(_cov_rng.uniform(0, T_DAYS, _N_COV)),
})
_COV_CELLS = [box(x, y, x + 0.5, y + 0.5) for y in (0.0, 0.5) for x in (0.0, 0.5)]
COV_GDF = gpd.GeoDataFrame({"cov": [0.5, -1.0, 1.5, -0.5], "geometry": _COV_CELLS})


def _loglik_at(model, args, params):
    seeded = handlers.substitute(handlers.seed(model.model, jax.random.PRNGKey(0)), params)
    tr = handlers.trace(seeded).get_trace(args)
    return float(np.asarray(tr["loglik"]["value"])), tr


# =====================================================================
# (I4) Thinning-compensator consistency.
# _sim_offspring (bstpp/main.py) discards candidate offspring at lag > window;
# the likelihood charges alpha * F_beta(min(T - t_j, window)) per parent
# (bstpp/inference_functions.py, temp_part). Lemma 2.4: the per-parent expected
# offspring count is m = alpha * F_beta(window), computed HERE through the same
# t_trig.compute_integral the likelihood uses, so simulator and compensator are
# tied to one expression. For a single immigrant far from the horizon, the
# expected number of total descendants of the cascade is m / (1 - m).
# =====================================================================

def test_offspring_thinning_matches_compensator_mass():
    model = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False, **PRIORS)
    win = 1.5                            # internal units; << typical lags so truncation bites
    model.set_window(win)
    par = dict(alpha=0.6, beta=3.0, sigmax_2=0.01)   # internal units

    # The compensator's per-parent mass, via the likelihood's own trigger object:
    m = par["alpha"] * float(np.asarray(
        model.args["t_trig"].compute_integral({"beta": par["beta"]}, win)))
    assert 0.0 < m < 1.0                  # subcritical; F(win) < 1 so truncation is active
    expected_descendants = m / (1.0 - m)

    np.random.seed(7)
    R = 4000
    totals = np.empty(R)
    immigrant = np.array([[0.5, 0.5, 0.0]])          # (x, y, t) internal
    for r in range(R):
        out = model._sim_offspring(immigrant.copy(), par)
        totals[r] = len(out) - 1
        # Hard thinning rule: every offspring has some earlier event within (0, win].
        t = out[:, 2]
        for k in range(1, len(out)):
            dts = t[k] - t[:k]
            assert np.any((dts > 0) & (dts <= win + 1e-9)), \
                "offspring with no admissible parent lag <= window"

    mean, se = totals.mean(), totals.std(ddof=1) / np.sqrt(R)
    assert abs(mean - expected_descendants) < 5 * se, (
        f"E[descendants] = {mean:.4f} +/- {se:.4f} vs m/(1-m) = "
        f"{expected_descendants:.4f} with m = alpha*F_beta(w) = {m:.4f}")


# =====================================================================
# (I8a) Special-case reduction: alpha = 0 collapses the Cox-Hawkes
# log-likelihood to the LGCP log-likelihood at identical latents.
# =====================================================================

@needs_decoder
def test_alpha_zero_reduces_cox_hawkes_to_lgcp():
    lgcp = LGCP_Model(DATA, A_RECT, T_DAYS, a_0=dist.Normal(0, 5))
    ch = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background="cox", **PRIORS)

    tr = handlers.trace(handlers.seed(lgcp.model, jax.random.PRNGKey(1))).get_trace(lgcp.args)
    shared = {k: tr[k]["value"] for k in ("a_0", "z_temporal", "z_seasonal", "z_spatial")}

    ll_lgcp, _ = _loglik_at(lgcp, lgcp.args, shared)
    ll_ch0, tr_ch = _loglik_at(
        ch, ch.args,
        {**shared, "alpha": np.float32(0.0), "beta": np.float32(1.0),
         "sigmax_2": np.float32(0.05)})

    assert float(np.asarray(tr_ch["Itot_excite"]["value"])) == pytest.approx(0.0, abs=1e-6)
    # float32 pipeline; scale-aware tolerance
    assert ll_ch0 == pytest.approx(ll_lgcp, rel=2e-5, abs=1e-3), (
        f"cox_hawkes(alpha=0) loglik {ll_ch0} != lgcp loglik {ll_lgcp}")


# =====================================================================
# (I8b) Special-case reduction: window = T (default) equals any window
# beyond the horizon, for the FULL log-likelihood (event term via the pair
# set AND compensator via min(T - t_j, w)); extends test_window_default,
# which pins Itot_excite only.
# =====================================================================

def test_window_at_horizon_recovers_untruncated_loglik():
    params = {k: np.float32(v) for k, v in
              dict(a_0=0.5, alpha=0.3, beta=2.0, sigmax_2=0.1).items()}
    m_default = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False, **PRIORS)
    m_beyond = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False,
                            window=1e9, **PRIORS)
    ll_default, _ = _loglik_at(m_default, m_default.args, params)
    ll_beyond, _ = _loglik_at(m_beyond, m_beyond.args, params)
    assert np.isfinite(ll_default)
    assert ll_default == pytest.approx(ll_beyond, rel=1e-6, abs=1e-4)


# =====================================================================
# (I9) Refinement invariance, covariate partition: splitting every covariate
# cell into two halves carrying the same covariate value must change no
# likelihood quantity. Exercises cov_ind (event side), cov_area and
# spatial_cov (compensator side, Itot_txy_back = mu_xyt @ cov_area * T).
# The uniform 2-way split duplicates each covariate value exactly once, so
# the cell-wise standardization (mean/var over cells) is also unchanged and
# both standardize_cov settings must be invariant.
# =====================================================================

def _split_each_cell_in_two(gdf):
    rows = []
    for _, row in gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        midx = 0.5 * (minx + maxx)
        for g in (box(minx, miny, midx, maxy), box(midx, miny, maxx, maxy)):
            rows.append({"cov": row["cov"], "geometry": g})
    return gpd.GeoDataFrame(rows)


@pytest.mark.parametrize("standardize", [False, True])
def test_covariate_refinement_invariance(standardize):
    params = {k: np.float32(v) for k, v in
              dict(a_0=0.2, alpha=0.3, beta=2.0, sigmax_2=0.1).items()}
    params["w"] = np.float32([0.7])

    def build(cov_gdf):
        return Hawkes_Model(COV_DATA, A_RECT, T_DAYS, cox_background=False,
                            spatial_cov=cov_gdf, cov_names=["cov"],
                            standardize_cov=standardize, **PRIORS)

    base = build(COV_GDF)
    refined = build(_split_each_cell_in_two(COV_GDF))

    ll_base, tr_base = _loglik_at(base, base.args, params)
    ll_ref, tr_ref = _loglik_at(refined, refined.args, params)

    itb_base = float(np.asarray(tr_base["Itot_txy_back"]["value"]))
    itb_ref = float(np.asarray(tr_ref["Itot_txy_back"]["value"]))
    assert itb_ref == pytest.approx(itb_base, rel=1e-5)
    assert ll_ref == pytest.approx(ll_base, rel=1e-5, abs=1e-4)


# =====================================================================
# (I10) Unit invariance, spatial: an affine change of spatial units in the
# input data (and domain) must leave the internal representation -- and the
# internal-unit log-likelihood -- exactly invariant, because _scale_xyt
# normalizes to the unit square. Any discrepancy is a conversion defect.
# =====================================================================

def test_spatial_affine_unit_invariance():
    cx, cy, dx, dy = 3.0, 0.5, -1.0, 2.0        # positive scales, arbitrary offsets
    data2 = DATA.copy()
    data2["X"] = cx * DATA["X"] + dx
    data2["Y"] = cy * DATA["Y"] + dy
    a2 = np.array([[cx * 0.0 + dx, cx * 1.0 + dx],
                   [cy * 0.0 + dy, cy * 1.0 + dy]])

    m1 = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False, **PRIORS)
    m2 = Hawkes_Model(data2, a2, T_DAYS, cox_background=False, **PRIORS)

    assert np.allclose(np.asarray(m1.args["t_events"]), np.asarray(m2.args["t_events"]),
                       rtol=0, atol=1e-6)
    assert np.allclose(np.asarray(m1.args["xy_events"]), np.asarray(m2.args["xy_events"]),
                       rtol=0, atol=1e-6)
    # pair sets identical as sets (construction order is outside the contract, I3)
    p1 = set(map(tuple, np.asarray(m1.args["coords"]).reshape(-1, 2).tolist()))
    p2 = set(map(tuple, np.asarray(m2.args["coords"]).reshape(-1, 2).tolist()))
    assert p1 == p2

    params = {k: np.float32(v) for k, v in
              dict(a_0=0.5, alpha=0.3, beta=2.0, sigmax_2=0.1).items()}
    ll1, _ = _loglik_at(m1, m1.args, params)
    ll2, _ = _loglik_at(m2, m2.args, params)
    assert ll1 == pytest.approx(ll2, rel=1e-6, abs=1e-4)


# =====================================================================
# (I11) Conservation: under fixed parameters, E[n_simulated] = E[Lambda(theta)]
# (martingale identity; the excitation compensator depends on the realized
# events, so each replicate is paired with its own compensator evaluation).
# The compensator is assembled from the SAME ingredients the likelihood uses
# (t_trig / sp_trig compute_integral, A_ bounds, A_area) and cross-checked
# once against the traced Itot_txy on the training events.
# =====================================================================

def _compensator(model, t_events, xy_events, params):
    args = model.args
    T = args["T"]
    win = args.get("window", T)
    temp = params["alpha"] * np.asarray(
        args["t_trig"].compute_integral({"beta": params["beta"]},
                                        jnp.minimum(T - t_events, win)))
    A_ = args["A_"]
    x_min, x_max = A_[0]
    y_min, y_max = A_[1]
    sp_limits = jnp.stack((x_max - xy_events[0], xy_events[0] - x_min,
                           y_max - xy_events[1], xy_events[1] - y_min)).reshape(2, 2, -1)
    sp = np.asarray(args["sp_trig"].compute_integral({"sigmax_2": params["sigmax_2"]},
                                                     sp_limits))
    lam_bg = float(np.exp(params["a_0"])) * T * args["A_area"]
    return lam_bg + float(np.sum(temp * sp))


def test_simulated_count_matches_compensator():
    model = Hawkes_Model(DATA, A_GDF, T_DAYS, cox_background=False, **PRIORS)
    truth = dict(a_0=0.4, alpha=0.3, beta=2.0, sigmax_2=0.01)   # internal units

    # Guard: the assembled compensator must equal the model's own Itot_txy
    # on the training events, at the same parameters.
    _, tr = _loglik_at(model, model.args, {k: np.float32(v) for k, v in truth.items()})
    lam_train = _compensator(model, np.asarray(model.args["t_events"]),
                             np.asarray(model.args["xy_events"]), truth)
    assert lam_train == pytest.approx(float(np.asarray(tr["Itot_txy"]["value"])),
                                      rel=2e-5), \
        "helper compensator diverges from the model's Itot_txy -- fix the helper, not the model"

    np.random.seed(11)
    rng = np.random.default_rng(5)
    R = 30
    diffs = np.empty(R)
    counts = np.empty(R)
    for r in range(R):
        sim = model.simulate(parameters=dict(truth), rng=rng)
        counts[r] = len(sim)
        ta, _ = model._scale_xyt(pd.DataFrame(sim[["X", "Y", "T"]]),
                                 model.args.copy(), model.comp_grid)
        diffs[r] = counts[r] - _compensator(model, np.asarray(ta["t_events"]),
                                            np.asarray(ta["xy_events"]), truth)

    mean = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(R)
    assert counts.mean() > 20, "degenerate simulation size; check truth parameters"
    assert abs(mean) < 5 * se + 1e-6, (
        f"E[n - Lambda] = {mean:.3f} +/- {se:.3f} over R={R} "
        f"(mean count {counts.mean():.1f}) -- conservation violated")
