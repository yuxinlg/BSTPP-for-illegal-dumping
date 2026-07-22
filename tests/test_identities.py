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
- (I10) unit covariance            -> test_spatial_similarity_covariance,
                                      test_spatial_kernel_family_is_real_unit_not_internal (HERE)
- (I11) conservation E[n] = E[Lam] -> test_simulated_count_matches_compensator (HERE)
- (I4-ws)/(I11-ws) finite spatial_window variants -> (HERE); impossible before
  the three-leg real-unit box symmetry (event-side-only truncation)
- (I12) box invariance             -> test_trigger_legs_invariant_to_bounding_rectangle (HERE)

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
  parameter), so time-unit rescaling is not a supported input transformation.
  Spatially the identity is now COVARIANCE, not invariance: the trigger is a
  real-unit object, so the loglik is preserved under similarity transformations
  with sigmax_2 -> c^2 sigmax_2, while the internal REPRESENTATION stays
  invariant under any axis-wise affine map. Fixed-parameter invariance under
  NON-uniform rescaling -- the old identity -- is now asserted to FAIL: that
  invariance was the aspect-ratio kernel defect.
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
# non-square domain for the real-unit trigger contract tests: 4:1 aspect
_ns_rng = np.random.RandomState(9)
DATA_NS = pd.DataFrame({
    "X": _ns_rng.uniform(10.2, 13.8, _N),
    "Y": _ns_rng.uniform(20.05, 20.95, _N),
    "T": np.sort(_ns_rng.uniform(0, T_DAYS, _N)),
})
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
    # NOTE: since the rectangle clip (Prop 1.1(ii)), the per-parent law is
    # m = alpha * F_beta(w) * Q_rect(parent location); this configuration puts
    # the immigrant at the center with sd 0.1, so Q_rect = 1 to ~1e-5 and the
    # temporal-mass identity below is unchanged. The spatial factor is
    # exercised by test_offspring_cascade_* and the finite-ws I4 test.
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
# (I10, restated) Unit COVARIANCE, spatial: the spatial trigger is a REAL-unit
# object (isotropic Gaussian in the units of the input X/Y columns), so the
# model is covariant under SIMILARITY transformations of the spatial inputs --
# uniform scaling c and translation, with sigmax_2 -> c^2 * sigmax_2 -- and
# deliberately NOT invariant under axis-wise (non-uniform) rescaling at fixed
# parameter values. The old identity (internal loglik invariant under ANY
# axis-wise affine map at FIXED sigmax_2) was precisely the anisotropy defect:
# the kernel family absorbed the bounding-box shape (consolidation doc,
# Prop. "aniso"). Its replacement here is a signed-off test edit, part of the
# real-unit trigger contract change.
# =====================================================================

def test_spatial_similarity_covariance():
    """(I10) Internal representation is invariant under any axis-wise affine
    map; the LOGLIK is invariant under a similarity transformation with the
    covariant parameter map sigmax_2 -> c^2 sigmax_2."""
    c, dx, dy = 3.0, -1.0, 2.0                  # uniform scale, arbitrary offsets
    data2 = DATA.copy()
    data2["X"] = c * DATA["X"] + dx
    data2["Y"] = c * DATA["Y"] + dy
    a2 = np.array([[c * 0.0 + dx, c * 1.0 + dx],
                   [c * 0.0 + dy, c * 1.0 + dy]])

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

    params1 = {k: np.float32(v) for k, v in
               dict(a_0=0.5, alpha=0.3, beta=2.0, sigmax_2=0.1).items()}
    params2 = dict(params1, sigmax_2=np.float32(c * c * 0.1))   # covariant map
    ll1, _ = _loglik_at(m1, m1.args, params1)
    ll2, _ = _loglik_at(m2, m2.args, params2)
    assert ll1 == pytest.approx(ll2, rel=1e-5, abs=1e-4)


def test_spatial_kernel_family_is_real_unit_not_internal():
    """(I10, negative control) Under a NON-uniform axis rescaling at FIXED
    sigmax_2 the loglik must CHANGE: no single real-unit isotropic kernel can
    equal itself across differently-shaped bounding boxes. The historical
    internal-unit kernel made this comparison exactly equal -- that equality
    WAS the aspect-ratio defect -- so this test is RED on the pre-fix code by
    construction.

    The internal representation (event coords, pair set) must still be
    invariant: only the kernel family is unit-bearing, not the ingestion."""
    cx, cy, dx, dy = 3.0, 0.5, -1.0, 2.0        # non-uniform scales
    data2 = DATA.copy()
    data2["X"] = cx * DATA["X"] + dx
    data2["Y"] = cy * DATA["Y"] + dy
    a2 = np.array([[cx * 0.0 + dx, cx * 1.0 + dx],
                   [cy * 0.0 + dy, cy * 1.0 + dy]])

    m1 = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False, **PRIORS)
    m2 = Hawkes_Model(data2, a2, T_DAYS, cox_background=False, **PRIORS)

    assert np.allclose(np.asarray(m1.args["xy_events"]), np.asarray(m2.args["xy_events"]),
                       rtol=0, atol=1e-6)
    p1 = set(map(tuple, np.asarray(m1.args["coords"]).reshape(-1, 2).tolist()))
    p2 = set(map(tuple, np.asarray(m2.args["coords"]).reshape(-1, 2).tolist()))
    assert p1 == p2

    params = {k: np.float32(v) for k, v in
              dict(a_0=0.5, alpha=0.3, beta=2.0, sigmax_2=0.1).items()}
    ll1, _ = _loglik_at(m1, m1.args, params)
    ll2, _ = _loglik_at(m2, m2.args, params)
    assert abs(float(ll1) - float(ll2)) > 1.0, (
        f"loglik identical across differently-shaped boxes at fixed sigmax_2 "
        f"({float(ll1):.4f} vs {float(ll2):.4f}): the kernel family is still "
        "internal-unit (aspect-ratio defect)")


# =====================================================================
# (I11) Conservation: under fixed parameters, E[n_simulated] = E[Lambda(theta)]
# (martingale identity; the excitation compensator depends on the realized
# events, so each replicate is paired with its own compensator evaluation).
# The compensator is assembled from the SAME ingredients the likelihood uses
# (t_trig / sp_trig compute_integral, A_ bounds, A_area) and cross-checked
# once against the traced Itot_txy on the training events.
# =====================================================================

def _compensator(model, t_events, xy_events, params):
    # Background assembled here; excitation via the PRODUCTION atom (which has
    # its own independent-reference unit test in test_likelihood_atoms.py, so
    # this test does not share a formula with the code under test blindly --
    # and the traced-Itot_txy guard below cross-checks the full assembly).
    from bstpp.likelihood import rectangular_excitation_compensator
    args = model.args
    T = args["T"]
    win = args.get("window", T)
    # Internal bounds + axis_scales, matching the production call site. (The
    # pre-change helper passed A_ as the bounds, which only coincided with the
    # internal rectangle because these tests use the unit box.)
    exc = float(rectangular_excitation_compensator(
        jnp.float32(params["alpha"]), jnp.asarray(t_events), jnp.asarray(xy_events),
        T, win, (args["x_min"], args["x_max"], args["y_min"], args["y_max"]),
        {"beta": jnp.float32(params["beta"])},
        {"sigmax_2": jnp.float32(params["sigmax_2"])},
        args["t_trig"], args["sp_trig"],
        axis_scales=args["axis_scales"],
        spatial_window=args.get("spatial_window")))
    lam_bg = float(np.exp(params["a_0"])) * T * args["A_area"]
    return lam_bg + exc


def test_simulated_count_matches_compensator():
    model = Hawkes_Model(DATA, A_GDF, T_DAYS, cox_background=False,
                         excitation_support="rectangle", **PRIORS)
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



def test_offspring_cascade_discards_outside_rectangle_before_parenting():
    """Prop 1.1(ii) at the cascade level: offspring falling outside the
    bounding rectangle X (the region the compensator charges, eq. 27) must be
    discarded BEFORE they can parent -- so _sim_offspring's own output may
    never contain an out-of-rectangle event. Pre-fix, out-of-rectangle
    offspring stayed in the cascade (only simulate()'s final sjoin removed
    them, after parenting), so hidden events excited observed ones.
    NOTE (honest severity record): the resulting E[n - Lambda] bias is
    SECOND-ORDER -- a boundary-heavy conservation stress test (sigmax_2=0.09,
    alpha=0.55, R=40) could NOT detect it, which is why this regression pins
    the structural property directly rather than a CLT statistic.
    """
    model = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False, **PRIORS)
    model.set_window(1e9)
    par = dict(alpha=0.7, beta=2.0, sigmax_2=0.09)   # sd 0.3: heavy leakage
    A_ = np.asarray(model.args["A_"])
    corners = np.array([[0.03, 0.03, 0.0], [0.97, 0.97, 5.0], [0.02, 0.95, 10.0]])
    np.random.seed(31)
    added = []
    for r in range(200):
        out = model._sim_offspring(corners.copy(), par)
        added.append(out[len(corners):])
    added = np.concatenate(added)
    assert len(added) > 50, "stress config produced too few offspring to be informative"
    inside = ((added[:, 0] >= A_[0, 0]) & (added[:, 0] <= A_[0, 1]) &
              (added[:, 1] >= A_[1, 0]) & (added[:, 1] <= A_[1, 1]))
    n_out = int((~inside).sum())
    assert n_out == 0, (
        f"{n_out}/{len(added)} cascade events lie outside the rectangle -- "
        "out-of-domain offspring are parenting before the clip")


def test_offspring_displacements_isotropic_in_real_units():
    """Real-unit trigger contract at the simulator leg: on a strongly
    NON-SQUARE box (4:1 aspect) the cascade's per-axis position variances
    around a center immigrant must be EQUAL -- the offspring displacement is
    N(0, sigma^2 I) in real coordinates. Pre-fix the internal-isotropic draw
    was rescaled per axis by the box spans, giving a real-space variance
    ratio of (Lx/Ly)^2 = 16 here, so this test is RED on the pre-fix code.

    Generation mixing is harmless: every cascade event's position is the
    immigrant plus an iid sum of displacement vectors, so the per-axis
    variance ratio equals the per-displacement ratio at every generation.
    sd = 0.02 real units against a 4 x 1 box keeps boundary clipping
    negligible, so the rectangle clip (Prop 1.1(ii)) does not distort the
    ratio."""
    A_ns = np.array([[10.0, 14.0], [20.0, 21.0]])       # Lx = 4, Ly = 1
    model = Hawkes_Model(DATA_NS, A_ns, T_DAYS, cox_background=False, **PRIORS)
    model.set_window(1e9)
    par = dict(alpha=0.4, beta=2.0, sigmax_2=0.0004)     # sd 0.02 REAL units

    center = np.array([[12.0, 20.5, 0.0]])
    np.random.seed(53)
    disp = []
    for r in range(1200):
        out = model._sim_offspring(center.copy(), par)
        d = out[1:, :2] - center[0, :2]
        if len(d):
            disp.append(d)
    disp = np.concatenate(disp)
    assert len(disp) > 400, "config produced too few offspring to be informative"
    vx, vy = disp[:, 0].var(ddof=1), disp[:, 1].var(ddof=1)
    ratio = vx / vy
    assert 0.6 < ratio < 1.67, (
        f"per-axis real-space variance ratio {ratio:.2f} (n={len(disp)}) -- "
        "offspring displacements are not isotropic in real units "
        "(internal-unit kernel rescaled by the box spans gives ratio ~16 here)")


def test_offspring_thinning_matches_compensator_mass_finite_spatial_window():
    """(I4-ws) With a finite REAL-unit spatial_window the per-parent cascade
    mean must be m/(1-m) with m = alpha * F_beta(w) * BoxMass(ws), computed
    through the PRODUCTION compensator atom on a single center event. The
    center immigrant with ws << distance-to-boundary keeps every retained
    cascade member deep in the interior, so m is location-constant to ~1e-6
    and the geometric cascade law applies. This identity was IMPOSSIBLE
    before the three-leg symmetry: the event side truncated at ws while the
    compensator charged the full rectangle mass and the simulator never
    thinned spatially."""
    from bstpp.likelihood import rectangular_excitation_compensator
    model = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False, **PRIORS)
    win, sw = 1.5, 0.15                       # sw REAL units (== internal on the unit box)
    model.set_window(win, spatial_window=sw)
    par = dict(alpha=0.5, beta=3.0, sigmax_2=0.01)   # sd 0.1 real: BoxMass(ws) ~ 0.75

    m = float(rectangular_excitation_compensator(
        jnp.float32(par["alpha"]), jnp.float32([0.0]),
        jnp.float32([[0.5], [0.5]]), 1e9, win, (0.0, 1.0, 0.0, 1.0),
        {"beta": jnp.float32(par["beta"])},
        {"sigmax_2": jnp.float32(par["sigmax_2"])},
        model.args["t_trig"], model.args["sp_trig"],
        axis_scales=model.args["axis_scales"], spatial_window=sw))
    assert 0.0 < m < 1.0
    expected = m / (1.0 - m)

    np.random.seed(41)
    R = 4000
    totals = np.empty(R)
    immigrant = np.array([[0.5, 0.5, 0.0]])
    for r in range(R):
        totals[r] = len(model._sim_offspring(immigrant.copy(), par)) - 1
    mean, se = totals.mean(), totals.std(ddof=1) / np.sqrt(R)
    assert abs(mean - expected) < 5 * se, (
        f"E[descendants] = {mean:.4f} +/- {se:.4f} vs m/(1-m) = {expected:.4f} "
        f"with m = alpha*F_beta(w)*BoxMass(ws) = {m:.4f}")


def test_simulated_count_matches_compensator_finite_spatial_window():
    """(I11-ws) Conservation on the unit box with spatial_window = 0.2 (real
    units): holds only because all three legs -- pair set, compensator, and
    offspring thinning -- share the real-unit box semantics. IMPOSSIBLE
    before the symmetry fix."""
    model = Hawkes_Model(DATA, A_GDF, T_DAYS, cox_background=False,
                         excitation_support="rectangle", **PRIORS)
    model.set_window(float(model.args["T"]), spatial_window=0.2)
    truth = dict(a_0=0.4, alpha=0.3, beta=2.0, sigmax_2=0.01)

    _, tr = _loglik_at(model, model.args, {k: np.float32(v) for k, v in truth.items()})
    lam_train = _compensator(model, np.asarray(model.args["t_events"]),
                             np.asarray(model.args["xy_events"]), truth)
    assert lam_train == pytest.approx(float(np.asarray(tr["Itot_txy"]["value"])),
                                      rel=2e-5)

    np.random.seed(43)
    rng = np.random.default_rng(47)
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
    assert counts.mean() > 20
    assert abs(mean) < 5 * se + 1e-6, (
        f"E[n - Lambda] = {mean:.3f} +/- {se:.3f} with spatial_window=0.2 -- "
        "the three-leg real-unit box symmetry is violated")


# =====================================================================
# (I12) Box invariance: the TRIGGER legs of the model are invariant to the
# choice of bounding rectangle. The same real-coordinate dataset ingested
# under two differently-shaped rectangles must yield (a) the same excitation
# pair set under a real-unit spatial_window, with the same real-unit
# displacements; (b) the same real-unit kernel densities at the pairs (the
# internal-measure atom values divided by the per-box Jacobian); (c) the same
# excitation compensator when the ws-square around every event is interior to
# both rectangles (all limits clip to the scalar ws); (d) byte-identical
# offspring cascades in real coordinates under a shared seed. Pre-fix this
# identity FAILS at (a) already: the internal-Euclidean window kept different
# pair sets in differently-shaped boxes. The background legs are exactly
# affine-invariant by construction (piecewise-constant fields; no metric
# object), so the trigger legs are the entire content of box invariance.
# =====================================================================

def test_trigger_legs_invariant_to_bounding_rectangle():
    from bstpp.likelihood import (real_spatial_trigger_values,
                                  rectangular_excitation_compensator)
    rng = np.random.RandomState(17)
    n = 50
    data = pd.DataFrame({
        "X": rng.uniform(10.3, 13.7, n),
        "Y": rng.uniform(20.2, 20.8, n),          # margin >= 0.2 to box1 edges
        "T": np.sort(rng.uniform(0, T_DAYS, n)),
    })
    box1 = np.array([[10.0, 14.0], [20.0, 21.0]])      # 4:1 aspect
    box2 = np.array([[9.0, 15.0], [19.5, 21.5]])       # 3:1 aspect, padded
    win, sw = 5.0, 0.15                                 # ws < every edge margin
    par = dict(alpha=0.4, beta=2.0, sigmax_2=0.02)      # sd ~0.14: BoxMass(ws) ~ 0.5

    models = [Hawkes_Model(data, A, T_DAYS, cox_background=False,
                           window=win, spatial_window=sw, **PRIORS)
              for A in (box1, box2)]

    # (a) same pair set, same REAL displacements
    reals = []
    for m in models:
        sx, sy = np.asarray(m.args["axis_scales"])
        c = np.asarray(m.args["coords"]).reshape(-1, 2)
        order = np.lexsort((c[:, 1], c[:, 0]))
        reals.append((c[order],
                      np.asarray(m.args["x_vals"])[order] * sx,
                      np.asarray(m.args["y_vals"])[order] * sy,
                      order))
    assert reals[0][0].shape == reals[1][0].shape and np.array_equal(reals[0][0], reals[1][0]), \
        "pair sets differ across bounding rectangles"
    np.testing.assert_allclose(reals[0][1], reals[1][1], rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(reals[0][2], reals[1][2], rtol=2e-5, atol=1e-6)

    # (b) same real-unit kernel densities: atom values / per-box Jacobian
    dens = []
    for m, (c, _, _, order) in zip(models, reals):
        sx, sy = np.asarray(m.args["axis_scales"])
        v = np.asarray(real_spatial_trigger_values(
            m.args["sp_trig"], {"sigmax_2": jnp.float32(par["sigmax_2"])},
            m.args["coords"], m.args["x_vals"], m.args["y_vals"],
            m.args["axis_scales"]))
        dens.append(v[order] / (sx * sy))
    np.testing.assert_allclose(dens[0], dens[1], rtol=2e-5)

    # (c) same excitation compensator (every limit clips to the scalar ws)
    comps = []
    for m in models:
        comps.append(float(rectangular_excitation_compensator(
            jnp.float32(par["alpha"]), jnp.asarray(m.args["t_events"]),
            jnp.asarray(m.args["xy_events"]), m.args["T"], win,
            (m.args["x_min"], m.args["x_max"], m.args["y_min"], m.args["y_max"]),
            {"beta": jnp.float32(par["beta"])},
            {"sigmax_2": jnp.float32(par["sigmax_2"])},
            m.args["t_trig"], m.args["sp_trig"],
            axis_scales=m.args["axis_scales"], spatial_window=sw)))
    assert comps[0] == pytest.approx(comps[1], rel=1e-6)

    # (d) byte-identical offspring cascades in real coordinates, shared seed
    immigrant = np.array([[12.0, 20.5, 0.0]])
    outs = []
    for m in models:
        np.random.seed(97)
        outs.append(m._sim_offspring(immigrant.copy(), par))
    assert np.array_equal(outs[0], outs[1]), \
        "offspring cascades differ across bounding rectangles under a shared seed"
    assert len(outs[0]) > 1, "cascade produced no offspring; seed/config uninformative"


@pytest.mark.parametrize("cox", [False, "cox"])
def test_simulate_fully_reproducible_with_generator(cox):
    """RNG unification: one Generator drives every draw, so two simulate()
    calls with identically seeded fresh Generators are byte-identical --
    including the pure-Hawkes background path, whose sample_points draw was
    historically unseeded. (rng=None preserves the legacy behavior.)"""
    if cox == "cox" and not os.path.isfile(_SEASONAL_DECODER):
        pytest.skip("seasonal decoder artifact absent")
    model = Hawkes_Model(DATA, A_GDF, T_DAYS, cox_background=cox,
                         excitation_support="rectangle", **PRIORS)
    if cox == "cox":
        tr = handlers.trace(handlers.seed(model.model,
                                          jax.random.PRNGKey(3))).get_trace(model.args)
        truth = {k: np.asarray(tr[k]["value"]) for k in
                 ("a_0", "z_temporal", "z_seasonal", "z_spatial")}
        truth.update(alpha=np.float32(0.25), beta=np.float32(2.0),
                     sigmax_2=np.float32(0.02))
    else:
        truth = dict(a_0=0.4, alpha=0.3, beta=2.0, sigmax_2=0.02)
    a = model.simulate(parameters=dict(truth), rng=np.random.default_rng(17))
    b = model.simulate(parameters=dict(truth), rng=np.random.default_rng(17))
    np.testing.assert_array_equal(np.asarray(a[["X", "Y", "T"]], dtype=np.float64),
                                  np.asarray(b[["X", "Y", "T"]], dtype=np.float64))


def test_geographic_coordinate_warning():
    """Data contract warning: a declared CRS is authoritative
    (crs.is_geographic decides both ways); the bounds heuristic covers array
    domains and CRS-less GeoDataFrames. Degree-like domains trigger a
    UserWarning -- in lon/lat the real-unit isotropic kernel is anisotropic
    on the ground by cos(latitude). Metric-looking domains (unit box, the
    non-square test boxes) must NOT warn: the test suite itself is the
    false-positive guard."""
    import warnings as _w
    geo_box = np.array([[-75.25, -75.15], [39.90, 40.00]])   # Philadelphia-ish
    geo_data = pd.DataFrame({
        "X": np.linspace(-75.24, -75.16, 20),
        "Y": np.linspace(39.91, 39.99, 20),
        "T": np.sort(np.linspace(1.0, T_DAYS - 1.0, 20)),
    })
    # heuristic path (array domain, no CRS available)
    with pytest.warns(UserWarning, match="geographic"):
        Hawkes_Model(geo_data, geo_box, T_DAYS, cox_background=False, **PRIORS)
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)   # any UserWarning -> failure
        Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False, **PRIORS)

    # CRS path: is_geographic fires regardless of the bounds heuristic
    # (unit box near the origin would never trip it)
    geo_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    with pytest.warns(UserWarning, match="geographic"):
        Hawkes_Model(DATA, geo_gdf, T_DAYS, cox_background=False,
                     excitation_support="rectangle", **PRIORS)

    # CRS path, negative: a PROJECTED CRS suppresses the warning even on
    # degree-like bounds (the CRS is authoritative over the heuristic)
    proj_gdf = gpd.GeoDataFrame(
        {"geometry": [box(-75.25, 39.90, -75.15, 40.00)]}, crs="EPSG:2272")
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        Hawkes_Model(geo_data, proj_gdf, T_DAYS, cox_background=False,
                     excitation_support="rectangle", **PRIORS)
