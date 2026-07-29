"""Production-path polygon mass table must meet PRODUCTION_TAU_ABS = 1e-5.

Pre-3f follow-up (audit of tip 938e32b): the K=64 confirmation in
``test_polygon_mass_backend_shootout.py`` builds tables via the experiment-only
shootout ``build_quad_table`` (AD slopes). Users install tables from
``bstpp.polygon_mass.prepare_polygon_mass_table`` (host NumPy/SciPy float64,
central finite-difference knot slopes). This module gates the production
path only.

Does **not** import or call the shootout table builder. Oracle comparisons
use the independent §13 quadrature oracle from the shootout script as a
reference — not as a table construction path. Derivative error is measured
and reported separately under OP-12 and is not conflated with
``PRODUCTION_TAU_ABS``.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import Polygon, box as shapely_box

from bstpp import polygon_mass as pm

# Independent oracle only (not the shootout table builder).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from scripts import polygon_mass_backend_shootout as shoot  # noqa: E402

SIGMA_MIN = 10.0
SIGMA_MAX = 500.0
PANEL_H_M = 20.0
GL_ORDER = 16
WS = 400.0
RECT = shapely_box(0.0, 0.0, 2000.0, 2000.0)
CONCAVE = Polygon([
    (0, 0), (2000, 0), (2000, 900), (900, 900), (900, 2000), (0, 2000),
])
# (poly, x, y): boundary-near + deep-interior (uncut / cut reuse same events)
_CASES = [
    (RECT, np.array([50.0, 1000.0]), np.array([50.0, 1000.0])),
    (CONCAVE, np.array([850.0]), np.array([850.0])),
]
_BOUND_ROUNDING = 1e-6


def _offknot_sigmas(log_knots: np.ndarray) -> np.ndarray:
    mid = np.exp(0.5 * (log_knots[:-1] + log_knots[1:]))
    picks = mid[[0, len(mid) // 4, len(mid) // 2, 3 * len(mid) // 4, -1]]
    rng = np.random.default_rng(11)
    off = np.exp(rng.uniform(log_knots[0], log_knots[-1], 8))
    return np.unique(np.concatenate([picks, off]))


def _build_production_table(poly, x, y, *, ws):
    """Exclusive production preparation path."""
    return pm.prepare_polygon_mass_table(
        poly, x, y,
        min_sigma=SIGMA_MIN,
        max_sigma=SIGMA_MAX,
        spatial_window=ws,
        panel_h_m=PANEL_H_M,
        gl_order=GL_ORDER,
    )


@pytest.mark.parametrize("ws,label", [(None, "uncut"), (WS, "finite_ws")])
def test_production_prepare_path_meets_tau_abs(ws, label):
    """K=64 / panel_h_m=20 / gl_order=16 via prepare_polygon_mass_table."""
    # Do not enable process-global x64 for this production gate.
    x64_before = bool(jax.config.read("jax_enable_x64"))
    masses_fn, d_fn, _ = pm.make_table_eval()

    max_abs_err = 0.0
    max_deriv_err = 0.0
    f32_max_abs_err = 0.0

    for poly, x, y in _CASES:
        table = _build_production_table(poly, x, y, ws=ws)
        assert table.values.dtype == np.float64
        assert table.slopes.dtype == np.float64
        assert table.provenance["slope_method"] == pm.SLOPE_METHOD
        assert int(table.provenance["n_knots"]) == 64
        assert float(table.h_panel) == pytest.approx(PANEL_H_M)
        assert int(table.gl_order) == GL_ORDER

        # Endpoint / bound behavior at knots.
        assert np.isfinite(table.values).all()
        assert (table.values >= -_BOUND_ROUNDING).all()
        assert (table.values <= 1.0 + _BOUND_ROUNDING).all()

        lk = jnp.asarray(table.log_knots)
        vj = jnp.asarray(table.values)
        sj = jnp.asarray(table.slopes)
        sigmas = _offknot_sigmas(table.log_knots)

        for s in sigmas:
            pm.validate_sigma_in_range(s, table.log_knots)
            got = np.asarray(masses_fn(np.log(s), lk, vj, sj))
            assert np.isfinite(got).all()
            assert (got >= -_BOUND_ROUNDING).all()
            assert (got <= 1.0 + _BOUND_ROUNDING).all()
            want = np.array([
                shoot.oracle_mass(poly, float(sx), float(sy), float(s), ws)
                for sx, sy in zip(x, y)
            ])
            err = float(np.max(np.abs(got - want)))
            max_abs_err = max(max_abs_err, err)

            # Supported online float32 path (no global x64 toggle).
            got_f32 = np.asarray(pm.hermite_polygon_masses(
                jnp.asarray(np.log(s), dtype=jnp.float32), table, dtype=jnp.float32))
            assert np.isfinite(got_f32).all()
            f32_err = float(np.max(np.abs(got_f32 - want)))
            f32_max_abs_err = max(f32_max_abs_err, f32_err)

        # Derivative measured separately (OP-12); do not fold into tau_abs.
        deriv_sigmas = np.unique(np.concatenate([
            np.exp(0.5 * (table.log_knots[:-1] + table.log_knots[1:]))[
                [0, len(table.log_knots) // 2 - 1, -1]],
            np.exp(table.log_knots[[len(table.log_knots) // 2]]),
        ]))
        for s in deriv_sigmas:
            dgot = np.asarray(d_fn(np.log(s), lk, vj, sj))
            for i, (sx, sy) in enumerate(zip(x, y)):
                d_ora, _d_unc = shoot.oracle_dmass_dlogsigma(
                    poly, float(sx), float(sy), float(s), ws)
                max_deriv_err = max(max_deriv_err, float(abs(dgot[i] - d_ora)))

    print(
        f"\n[production prepare_polygon_mass_table {label}] "
        f"max_abs_value_err={max_abs_err:.6e} "
        f"(PRODUCTION_TAU_ABS={pm.PRODUCTION_TAU_ABS}); "
        f"max_deriv_err={max_deriv_err:.6e} "
        f"(TAU_DERIV={pm.TAU_DERIV}, OP-12 provisional); "
        f"f32_max_abs_err={f32_max_abs_err:.6e}; "
        f"jax_enable_x64_before={x64_before} "
        f"after={bool(jax.config.read('jax_enable_x64'))}\n"
    )
    assert bool(jax.config.read("jax_enable_x64")) == x64_before
    assert max_abs_err <= pm.PRODUCTION_TAU_ABS, (
        f"production value gate FAILED: max_abs_err={max_abs_err:.6e} > "
        f"PRODUCTION_TAU_ABS={pm.PRODUCTION_TAU_ABS}")
    # Float32 online path must also meet the value gate against the oracle.
    assert f32_max_abs_err <= pm.PRODUCTION_TAU_ABS, (
        f"float32 online value gate FAILED: {f32_max_abs_err:.6e}")
    # Report derivative separately; soft-check against provisional TAU_DERIV
    # without reclassifying as a PRODUCTION_TAU_ABS failure.
    if max_deriv_err > pm.TAU_DERIV:
        print(
            f"[OP-12] production-path derivative err {max_deriv_err:.6e} "
            f"exceeds provisional TAU_DERIV={pm.TAU_DERIV} "
            f"(value gate still passed)\n"
        )
