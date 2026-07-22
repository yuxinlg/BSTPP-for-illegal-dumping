"""Focused tests for the OP-9 backend shootout (scripts/, experiment-only).

Synthetic correctness cases required by the shootout spec: axis-aligned
rectangle (analytic erf product), triangle, concave polygon, polygon with a
hole, multipolygon, and rectangle degeneracy against the production
compensator's spatial mass (Spatial_Symmetric_Gaussian.compute_integral).
Also: cutoff-square intersection + analytic fast path, NaN-safe masked
padding, lookup-table C1/knot-side behavior, bounds, extrapolation
prohibition, and gradient-vs-FD checks.

These tests run in float64 (the shootout's accuracy precision). The autouse
fixture restores the global x64 flag afterwards so the rest of the suite
keeps its default precision.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box as shapely_box

REPO = Path(__file__).resolve().parents[1]

import sys

_spec = importlib.util.spec_from_file_location(
    "polygon_mass_backend_shootout",
    REPO / "scripts" / "polygon_mass_backend_shootout.py")
shoot = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = shoot  # dataclasses resolve __module__ via sys.modules
_spec.loader.exec_module(shoot)


@pytest.fixture(autouse=True)
def _x64_for_this_module():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def quad_mass(poly, x, y, sigma, h=5.0, q=16, ws=None):
    """Backend-A mass via the real preprocessing + jitted evaluation path."""
    prep = shoot.prepare_quadrature(poly, np.asarray(x, float),
                                    np.asarray(y, float), h, ws)
    masses_fn, _, _ = shoot.make_quad_eval(q, ws)
    return np.asarray(masses_fn(np.log(sigma), jnp.asarray(prep.ev_xy),
                                jnp.asarray(prep.panels),
                                jnp.asarray(prep.mask),
                                jnp.asarray(prep.inside_flag)))


def erf_product(x, y, bounds, sigma):
    from scipy.special import erf
    x_min, y_min, x_max, y_max = bounds
    s2 = sigma * np.sqrt(2.0)
    return (0.5 * (erf((x_max - x) / s2) + erf((x - x_min) / s2))
            * 0.5 * (erf((y_max - y) / s2) + erf((y - y_min) / s2)))


RECT = shapely_box(0.0, 0.0, 300.0, 150.0)
TRIANGLE = Polygon([(0, 0), (200, 0), (60, 170)])
CONCAVE = Polygon([(0, 0), (200, 0), (200, 200), (120, 200),
                   (120, 80), (0, 80)])  # L-shape
HOLED = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)],
                holes=[[(60, 60), (140, 60), (140, 140), (60, 140)]])
MULTI = MultiPolygon([shapely_box(0, 0, 80, 80), shapely_box(150, 150, 260, 240)])

EVENTS = [(50.0, 40.0), (150.0, 75.0), (10.0, 10.0), (199.0, 100.0)]
SIGMAS = [10.0, 50.0, 200.0]


def test_rectangle_analytic():
    x, y = np.array([50.0, 150.0, 299.0]), np.array([40.0, 75.0, 1.0])
    for s in SIGMAS:
        got = quad_mass(RECT, x, y, s)
        want = erf_product(x, y, (0.0, 0.0, 300.0, 150.0), s)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


@pytest.mark.parametrize("poly", [TRIANGLE, CONCAVE, HOLED, MULTI],
                         ids=["triangle", "concave", "hole", "multipolygon"])
def test_synthetic_polygons_vs_oracle(poly):
    x, y = zip(*EVENTS)
    for s in SIGMAS:
        got = quad_mass(poly, x, y, s)
        want = np.array([shoot.oracle_mass(poly, sx, sy, s)
                         for sx, sy in EVENTS])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


def test_rectangle_degeneracy_vs_production_compensator():
    """When A is a rectangle, backend A must reproduce the mass the CURRENT
    rectangle compensator charges (trigger.compute_integral on real-unit
    edge distances, likelihood.py:158-166 semantics, no window clip)."""
    from bstpp.trigger import Spatial_Symmetric_Gaussian
    trig = Spatial_Symmetric_Gaussian({})
    x, y = np.array([50.0, 250.0]), np.array([40.0, 149.0])
    for s in SIGMAS:
        lim = jnp.stack([jnp.stack([300.0 - x, x - 0.0]),
                         jnp.stack([150.0 - y, y - 0.0])])
        want = np.asarray(trig.compute_integral({"sigmax_2": s**2}, lim))
        got = quad_mass(RECT, x, y, s)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


def test_cutoff_square_intersection_and_fast_path():
    """A ∩ C_j: near-boundary event uses clipped panels; deep-interior event
    takes the analytic erf^2 fast path (zero panels)."""
    poly = shapely_box(0.0, 0.0, 1000.0, 1000.0)
    ws = 100.0
    x = np.array([30.0, 500.0])   # first: C_j crosses boundary; second: inside
    y = np.array([30.0, 500.0])
    prep = shoot.prepare_quadrature(poly, x, y, 5.0, ws)
    assert prep.inside_flag.tolist() == [0.0, 1.0]
    assert prep.mask[1].sum() == 0  # fast-path event carries no panels
    masses_fn, _, _ = shoot.make_quad_eval(16, ws)
    for s in [10.0, 50.0]:
        got = np.asarray(masses_fn(np.log(s), jnp.asarray(prep.ev_xy),
                                   jnp.asarray(prep.panels),
                                   jnp.asarray(prep.mask),
                                   jnp.asarray(prep.inside_flag)))
        want = np.array([shoot.oracle_mass(poly, sx, sy, s, ws)
                         for sx, sy in zip(x, y)])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)
        # fast path is exactly the retained-square mass
        from scipy.special import erf
        np.testing.assert_allclose(got[1], erf(ws / (np.sqrt(2) * s)) ** 2,
                                   rtol=0, atol=1e-12)


def test_masked_padding_is_nan_safe():
    """Padded (all-zero) panels must contribute exactly zero and never NaN,
    in both the value and the gradient."""
    prep = shoot.prepare_quadrature(TRIANGLE, np.array([50.0]),
                                    np.array([30.0]), 5.0, None)
    panels = np.concatenate([prep.panels,
                             np.zeros((1, 64, 4))], axis=1)  # extra padding
    mask = np.concatenate([prep.mask, np.zeros((1, 64))], axis=1)
    masses_fn, d_fn, vg_fn = shoot.make_quad_eval(8, None)
    args = (jnp.asarray(prep.ev_xy), jnp.asarray(panels), jnp.asarray(mask),
            jnp.asarray(prep.inside_flag))
    m_pad = np.asarray(masses_fn(np.log(20.0), *args))
    m_ref = quad_mass(TRIANGLE, [50.0], [30.0], 20.0, h=5.0, q=8)
    np.testing.assert_allclose(m_pad, m_ref, rtol=0, atol=1e-12)
    v, g = vg_fn(np.log(20.0), *args, jnp.ones(1))
    assert np.isfinite(v) and np.isfinite(g)


def test_quad_gradient_matches_fd():
    x, y = np.array([50.0, 150.0]), np.array([40.0, 75.0])
    prep = shoot.prepare_quadrature(CONCAVE, x, y, 5.0, None)
    _, d_fn, _ = shoot.make_quad_eval(16, None)
    args = (jnp.asarray(prep.ev_xy), jnp.asarray(prep.panels),
            jnp.asarray(prep.mask), jnp.asarray(prep.inside_flag))
    for s in [20.0, 100.0]:
        g = np.asarray(d_fn(np.log(s), *args))
        h = 1e-5
        m1 = quad_mass(CONCAVE, x, y, s * np.exp(h), h=5.0, q=16)
        m0 = quad_mass(CONCAVE, x, y, s * np.exp(-h), h=5.0, q=16)
        np.testing.assert_allclose(g, (m1 - m0) / (2 * h), rtol=1e-6, atol=1e-9)


# ------------------------------------------------------- lookup tables ------
def _small_table(K=24):
    x, y = np.array([50.0, 150.0, 10.0]), np.array([40.0, 75.0, 10.0])
    prep = shoot.build_table(CONCAVE, x, y, K, None, tag="testtmp")
    if prep.npz_path is not None and prep.npz_path.exists():
        prep.npz_path.unlink()  # test artifact, keep results dir clean
    return prep


def test_table_matches_oracle_at_knots_and_is_c1():
    prep = _small_table()
    masses_fn, d_fn, _ = shoot.make_table_eval()
    lk, vj, sj = (jnp.asarray(prep.log_knots), jnp.asarray(prep.values),
                  jnp.asarray(prep.slopes))
    # exact reproduction at knots
    for j in [0, 5, len(prep.log_knots) - 1]:
        got = np.asarray(masses_fn(prep.log_knots[j], lk, vj, sj))
        np.testing.assert_allclose(got, prep.values[:, j], rtol=0, atol=1e-12)
    # value and derivative continuity immediately to both sides of a knot
    for j in [1, 10, len(prep.log_knots) - 2]:
        eps = 1e-9
        lo = np.asarray(masses_fn(prep.log_knots[j] - eps, lk, vj, sj))
        hi = np.asarray(masses_fn(prep.log_knots[j] + eps, lk, vj, sj))
        np.testing.assert_allclose(lo, hi, rtol=0, atol=1e-7)
        dlo = np.asarray(d_fn(prep.log_knots[j] - eps, lk, vj, sj))
        dhi = np.asarray(d_fn(prep.log_knots[j] + eps, lk, vj, sj))
        np.testing.assert_allclose(dlo, dhi, rtol=0, atol=1e-5)


def test_table_bounds_and_offknot_accuracy():
    prep = _small_table(K=40)
    masses_fn, _, _ = shoot.make_table_eval()
    lk, vj, sj = (jnp.asarray(prep.log_knots), jnp.asarray(prep.values),
                  jnp.asarray(prep.slopes))
    rng = np.random.default_rng(3)
    sig = np.exp(rng.uniform(prep.log_knots[0], prep.log_knots[-1], 12))
    for s in sig:
        got = np.asarray(masses_fn(np.log(s), lk, vj, sj))
        assert (got >= -1e-12).all() and (got <= 1 + 1e-12).all()
        want = np.array([shoot.oracle_mass(CONCAVE, sx, sy, s)
                         for sx, sy in [(50, 40), (150, 75), (10, 10)]])
        np.testing.assert_allclose(got, want, rtol=0, atol=shoot.TAU_ABS)


def test_table_extrapolation_prohibited():
    prep = _small_table()
    with pytest.raises(ValueError, match="extrapolation is prohibited"):
        shoot.validate_sigma_in_range(9.0, prep.log_knots)
    with pytest.raises(ValueError, match="extrapolation is prohibited"):
        shoot.validate_sigma_in_range(501.0, prep.log_knots)
    # the jitted kernel is loud too: NaN, never a silently clamped value
    masses_fn, _, _ = shoot.make_table_eval()
    out = np.asarray(masses_fn(np.log(5.0), jnp.asarray(prep.log_knots),
                               jnp.asarray(prep.values),
                               jnp.asarray(prep.slopes)))
    assert np.isnan(out).all()


def test_oracle_selftest_preserved():
    """The §13 oracle's own rectangle self-test still passes unchanged."""
    shoot._diag.self_test()


# ---------------- finite-cutoff hybrid-table confirmation (OP-9 residual) ---
# Geometries sized for the preregistered representative w_s = 400 m (see
# results/polygon_mass_backend_shootout/decision.md and REPRESENTATIVE_WS).
# Case A: large rectangle — boundary-near event (C_j crosses dA) + deep
# interior event (C_j ⊂ A → analytic erf^2 fast path).
# Case B: large concave L — nonrectangular geometry with a boundary-near
# event whose cutoff square clips the re-entrant corner.
WS_CONFIRM = shoot.REPRESENTATIVE_WS  # 400.0 m; preregistered, not 3e
RECT_WS = shapely_box(0.0, 0.0, 2000.0, 2000.0)
CONCAVE_WS = Polygon([(0, 0), (2000, 0), (2000, 900), (900, 900),
                      (900, 2000), (0, 2000)])  # L-shape, metres
# (poly, x, y, labels): boundary-near, interior-fast-path, concave-boundary
_CONFIRM_CASES = [
    (RECT_WS,
     np.array([50.0, 1000.0]),
     np.array([50.0, 1000.0]),
     ("rect_boundary", "rect_interior")),
    (CONCAVE_WS,
     np.array([850.0]),
     np.array([850.0]),
     ("concave_boundary",)),
]
# Documented numerical allowance for the finite-cutoff bound
# 0 <= M <= erf(w_s/(√2 σ))^2 <= 1. Knot values from the quad builder obey
# the bound to ~1e-12; between knots cubic Hermite can overshoot by ~1e-9
# (observed) to ~1e-6 (margin). This is NOT a relaxation of TAU_ABS /
# TAU_DERIV — those oracle gates stay at 5.39e-4.
_BOUND_ROUNDING = 1e-6
_CONT_EPS = 1e-9  # knot-side probe for C1 continuity


def _confirm_sigmas(log_knots: np.ndarray) -> np.ndarray:
    """Off-knot evaluation grid: midpoints, 10 seeded log-uniform, knot sides."""
    mid = np.exp(0.5 * (log_knots[:-1] + log_knots[1:]))
    # subsample of midpoints (keep the test small / deterministic)
    mid_picks = mid[[0, len(mid) // 4, len(mid) // 2, 3 * len(mid) // 4, -1]]
    rng = np.random.default_rng(shoot.OFFGRID_SEED)
    off = np.exp(rng.uniform(log_knots[0], log_knots[-1], 10))
    picks_idx = np.array([1, len(log_knots) // 2, len(log_knots) - 2])
    sides = np.concatenate([
        np.exp(log_knots[picks_idx] - _CONT_EPS),
        np.exp(log_knots[picks_idx] + _CONT_EPS),
    ])
    return np.unique(np.concatenate([mid_picks, off, sides]))


def test_finite_cutoff_hybrid_table_confirmation():
    """Confirm the selected hybrid backend at finite spatial_window=400 m.

    Offline: K=64 log-spaced knots on [10, 500] m; float64 fixed-node
    quadrature (h=20 m, GL-16) supplies knot values AND AD knot slopes on
    A ∩ C_j. Online: existing cubic Hermite evaluator. Oracle: §13 float64
    exact-intersection wrapper + centered FD (h and h/2) for dM/dlog σ.

    Precision separation: the autouse fixture enables jax_enable_x64 only
    for this module (restored after), so table construction is float64;
    the float32 online check casts table arrays / query values explicitly
    and does NOT flip the global x64 flag for unrelated production code.
    """
    import time
    from scipy.special import erf

    t0 = time.perf_counter()
    assert jax.config.jax_enable_x64 is True
    assert WS_CONFIRM == 400.0

    K = 64
    h_panel, gl_order = 20.0, 16
    masses_fn, d_fn, _ = shoot.make_table_eval()

    max_abs_err = 0.0
    max_deriv_err = 0.0
    max_deriv_floor = 0.0
    signed_errs = []
    f32_max_abs_err = 0.0
    f32_max_deriv_err = 0.0
    case_records = []

    for poly, x, y, labels in _CONFIRM_CASES:
        prep_q = shoot.prepare_quadrature(poly, x, y, h_panel, WS_CONFIRM)
        # geometric roles: boundary events have clipped panels; interior uses
        # the analytic fast path (zero panels, inside_flag=1)
        for i, lab in enumerate(labels):
            if "interior" in lab:
                assert prep_q.inside_flag[i] == 1.0
                assert prep_q.mask[i].sum() == 0
            else:
                assert prep_q.inside_flag[i] == 0.0
                assert prep_q.mask[i].sum() > 0

        table = shoot.build_quad_table(
            poly, x, y, K, WS_CONFIRM, h=h_panel, gl_order=gl_order)
        assert table.values.dtype == np.float64
        assert table.slopes.dtype == np.float64
        sigmas = _confirm_sigmas(table.log_knots)
        for s in sigmas:
            shoot.validate_sigma_in_range(s, table.log_knots)

        lk = jnp.asarray(table.log_knots)
        vj = jnp.asarray(table.values)
        sj = jnp.asarray(table.slopes)

        # ---- value / bound / finiteness vs oracle (off-knot) ----
        for s in sigmas:
            got = np.asarray(masses_fn(np.log(s), lk, vj, sj))
            assert np.isfinite(got).all()
            assert not np.isnan(got).any()
            retained = float(erf(WS_CONFIRM / (np.sqrt(2.0) * s)) ** 2)
            assert retained <= 1.0 + _BOUND_ROUNDING
            assert (got >= -_BOUND_ROUNDING).all()
            assert (got <= retained + _BOUND_ROUNDING).all()
            want = np.array([shoot.oracle_mass(poly, sx, sy, s, WS_CONFIRM)
                             for sx, sy in zip(x, y)])
            err = got - want
            signed_errs.append(err)
            max_abs_err = max(max_abs_err, float(np.abs(err).max()))
            np.testing.assert_allclose(got, want, rtol=0, atol=shoot.TAU_ABS)

        # ---- derivative vs converged FD; distinguish FD floor ----
        deriv_sigmas = np.unique(np.concatenate([
            np.exp(0.5 * (table.log_knots[:-1] + table.log_knots[1:]))[
                [0, len(table.log_knots) // 2 - 1, -1]],
            np.exp(table.log_knots[[len(table.log_knots) // 2]]),
        ]))
        for s in deriv_sigmas:
            shoot.validate_sigma_in_range(s, table.log_knots)
            dgot = np.asarray(d_fn(np.log(s), lk, vj, sj))
            assert np.isfinite(dgot).all()
            for i, (sx, sy) in enumerate(zip(x, y)):
                d_ora, d_unc = shoot.oracle_dmass_dlogsigma(
                    poly, sx, sy, s, WS_CONFIRM)
                # half-step check already inside oracle_dmass_dlogsigma
                max_deriv_floor = max(max_deriv_floor, float(d_unc))
                derr = float(abs(dgot[i] - d_ora))
                # backend error after removing FD uncertainty (report both)
                max_deriv_err = max(max_deriv_err, derr)
                assert derr <= shoot.TAU_DERIV + d_unc

        # ---- C1 continuity across representative interior knots ----
        for j in [1, K // 2, K - 2]:
            lo = np.asarray(masses_fn(table.log_knots[j] - _CONT_EPS, lk, vj, sj))
            hi = np.asarray(masses_fn(table.log_knots[j] + _CONT_EPS, lk, vj, sj))
            np.testing.assert_allclose(lo, hi, rtol=0, atol=1e-7)
            dlo = np.asarray(d_fn(table.log_knots[j] - _CONT_EPS, lk, vj, sj))
            dhi = np.asarray(d_fn(table.log_knots[j] + _CONT_EPS, lk, vj, sj))
            np.testing.assert_allclose(dlo, dhi, rtol=0, atol=1e-5)

        # ---- analytic interior fast path vs oracle and erf^2 ----
        # At knots the builder stores the analytic retained-square mass
        # exactly; off-knot Hermite stays within TAU_ABS of both the oracle
        # and erf(w_s/(√2 σ))^2 (the retained-square closed form).
        for i, lab in enumerate(labels):
            if "interior" not in lab:
                continue
            for j, s in enumerate(np.exp(table.log_knots)):
                analytic = float(erf(WS_CONFIRM / (np.sqrt(2.0) * s)) ** 2)
                ora_i = shoot.oracle_mass(poly, float(x[i]), float(y[i]),
                                          s, WS_CONFIRM)
                np.testing.assert_allclose(table.values[i, j], analytic,
                                           rtol=0, atol=1e-12)
                np.testing.assert_allclose(ora_i, analytic, rtol=0, atol=1e-12)
            for s in [20.0, 100.0, 200.0]:
                got_i = float(np.asarray(masses_fn(np.log(s), lk, vj, sj))[i])
                ora_i = shoot.oracle_mass(poly, float(x[i]), float(y[i]),
                                          s, WS_CONFIRM)
                analytic = float(erf(WS_CONFIRM / (np.sqrt(2.0) * s)) ** 2)
                np.testing.assert_allclose(got_i, ora_i, rtol=0,
                                           atol=shoot.TAU_ABS)
                np.testing.assert_allclose(got_i, analytic, rtol=0,
                                           atol=shoot.TAU_ABS)
                np.testing.assert_allclose(ora_i, analytic, rtol=0, atol=1e-12)

        # ---- float32 online evaluation (tables built f64, cast for eval) ----
        lk32 = jnp.asarray(table.log_knots, dtype=jnp.float32)
        vj32 = jnp.asarray(table.values, dtype=jnp.float32)
        sj32 = jnp.asarray(table.slopes, dtype=jnp.float32)
        for s in sigmas:
            got32 = np.asarray(masses_fn(np.float32(np.log(s)), lk32, vj32, sj32))
            assert np.isfinite(got32).all()
            want = np.array([shoot.oracle_mass(poly, sx, sy, s, WS_CONFIRM)
                             for sx, sy in zip(x, y)])
            f32_max_abs_err = max(f32_max_abs_err,
                                  float(np.abs(got32 - want).max()))
            np.testing.assert_allclose(got32, want, rtol=0, atol=shoot.TAU_ABS)
        for s in deriv_sigmas:
            dgot32 = np.asarray(d_fn(np.float32(np.log(s)), lk32, vj32, sj32))
            for i, (sx, sy) in enumerate(zip(x, y)):
                d_ora, d_unc = shoot.oracle_dmass_dlogsigma(
                    poly, sx, sy, s, WS_CONFIRM)
                derr = float(abs(dgot32[i] - d_ora))
                f32_max_deriv_err = max(f32_max_deriv_err, derr)
                assert derr <= shoot.TAU_DERIV + d_unc

        case_records.append({
            "labels": labels,
            "n_events": len(x),
            "fast_path_events": int(prep_q.inside_flag.sum()),
            "build_s": table.build_seconds,
        })

    # no silent extrapolation: outside the knot span the kernel is NaN
    table0 = shoot.build_quad_table(
        RECT_WS, np.array([1000.0]), np.array([1000.0]), K, WS_CONFIRM,
        h=h_panel, gl_order=gl_order)
    with pytest.raises(ValueError, match="extrapolation is prohibited"):
        shoot.validate_sigma_in_range(9.0, table0.log_knots)
    with pytest.raises(ValueError, match="extrapolation is prohibited"):
        shoot.validate_sigma_in_range(501.0, table0.log_knots)
    out_lo = np.asarray(masses_fn(
        np.log(5.0), jnp.asarray(table0.log_knots),
        jnp.asarray(table0.values), jnp.asarray(table0.slopes)))
    assert np.isnan(out_lo).all()

    elapsed = time.perf_counter() - t0
    # surface measured errors for the decision-record / agent summary
    print(
        f"\n[finite-cutoff hybrid confirmation] "
        f"ws={WS_CONFIRM} K={K} h={h_panel} GL={gl_order}\n"
        f"  cases={case_records}\n"
        f"  max_abs_mass_err={max_abs_err:.6e} "
        f"(tau={shoot.TAU_ABS})\n"
        f"  max_abs_dM_dlogsigma_err={max_deriv_err:.6e} "
        f"(tau={shoot.TAU_DERIV}, fd_floor_max={max_deriv_floor:.6e})\n"
        f"  signed_err_minmax="
        f"({np.min(np.concatenate(signed_errs)):.6e}, "
        f"{np.max(np.concatenate(signed_errs)):.6e})\n"
        f"  float32_online max_abs_err={f32_max_abs_err:.6e} "
        f"max_deriv_err={f32_max_deriv_err:.6e}\n"
        f"  runtime_s={elapsed:.3f} PASS\n"
    )
    # stash for optional decision.md consumers reading capsys
    assert max_abs_err <= shoot.TAU_ABS
    assert max_deriv_err <= shoot.TAU_DERIV
    assert f32_max_abs_err <= shoot.TAU_ABS
    assert f32_max_deriv_err <= shoot.TAU_DERIV
