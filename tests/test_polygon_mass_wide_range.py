"""3d confirmation: hybrid Hermite tables across a full [min_sigma, max_sigma]
range that includes the polygon-mode 5 km default (beyond the shootout's
validated 500 m upper end).

Gates value and dM/dlog(sigma) against the §13 oracle (exact-intersection
wrapper when a finite spatial_window is set). Must pass before the 5 km
default is accepted in production.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from bstpp import polygon_mass as pm

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "polygon_mass_diagnostic",
    REPO / "scripts" / "polygon_mass_diagnostic.py")
_diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diag)


@pytest.fixture(autouse=True)
def _restore_x64_flag():
    prev = bool(jax.config.jax_enable_x64)
    yield
    jax.config.update("jax_enable_x64", prev)


def _oracle_mass(poly, sx, sy, sigma, ws=None):
    if ws is not None:
        poly = poly.intersection(shapely_box(sx - ws, sy - ws, sx + ws, sy + ws))
        if poly.is_empty:
            return 0.0
    return float(_diag.gaussian_polygon_mass(
        poly, sx, sy, sigma, _diag.MAX_STEP_SIGMA, _diag.GL_ORDER))


def _oracle_dmass_dlogsigma(poly, sx, sy, sigma, ws=None, h=1e-3):
    def dm(step):
        return (_oracle_mass(poly, sx, sy, sigma * np.exp(step), ws)
                - _oracle_mass(poly, sx, sy, sigma * np.exp(-step), ws)) / (2 * step)
    d1, d2 = dm(h), dm(h / 2)
    return d2, abs(d1 - d2) / 3.0


# Domain large enough that sigma up to 5 km is a meaningful integral, not a
# near-certain mass-1 fill of a tiny box. Events near boundary and interior.
_WIDE_RECT = shapely_box(0.0, 0.0, 8000.0, 8000.0)
_WIDE_L = Polygon([
    (0, 0), (8000, 0), (8000, 3000), (3000, 3000),
    (3000, 8000), (0, 8000),
])
_WIDE_EVENTS = [
    # (poly, x, y, label)
    (_WIDE_RECT, np.array([50.0, 4000.0]), np.array([50.0, 4000.0]), "rect"),
    (_WIDE_L, np.array([2800.0, 100.0]), np.array([2800.0, 100.0]), "L"),
]

# Full configured default-like range in metres (CRS units for EPSG:26918).
_SIGMA_MIN = 10.0
_SIGMA_MAX = 5000.0  # 5 km
_WS = 400.0          # fixed finite cutoff; representative, not a 3e recommendation


def _eval_grid(log_knots: np.ndarray) -> np.ndarray:
    """Off-knot and near-knot sigma grid spanning the full table range."""
    sig = np.exp(log_knots)
    mids = np.sqrt(sig[:-1] * sig[1:])
    rng = np.random.default_rng(1234)
    rand = np.exp(rng.uniform(log_knots[0], log_knots[-1], size=12))
    # include values above the old 500 m validation ceiling
    above = np.array([600.0, 1000.0, 2000.0, 3500.0, 4800.0])
    above = above[(above > _SIGMA_MIN) & (above < _SIGMA_MAX)]
    near = np.concatenate([
        np.exp(log_knots[[0, len(log_knots) // 2, -1]]) * (1 + 1e-3),
        np.exp(log_knots[[len(log_knots) // 2]]) * (1 - 1e-3),
    ])
    near = near[(near >= _SIGMA_MIN) & (near <= _SIGMA_MAX)]
    return np.unique(np.concatenate([mids[::max(1, len(mids)//8)],
                                     rand, above, near]))


@pytest.mark.parametrize("ws", [None, _WS], ids=["uncut", "ws400"])
def test_wide_range_hybrid_table_confirmation(ws):
    """Value+grad gate on [10 m, 5 km] before accepting the 5 km default."""
    assert pm.knot_count(_SIGMA_MIN, _SIGMA_MAX) > pm.VALIDATED_K

    masses_fn, d_fn, _ = pm.make_table_eval()
    max_abs_err = 0.0
    max_deriv_err = 0.0
    max_fd_floor = 0.0

    for poly, x, y, label in _WIDE_EVENTS:
        table = pm.build_quad_table(
            poly, x, y, _SIGMA_MIN, _SIGMA_MAX, ws=ws,
            h_panel=pm.DEFAULT_PANEL_H_M, gl_order=pm.DEFAULT_GL_ORDER)
        assert table.values.dtype == np.float64
        assert table.slopes.dtype == np.float64
        assert table.n_knots == pm.knot_count(_SIGMA_MIN, _SIGMA_MAX)
        np.testing.assert_allclose(np.exp(table.log_knots[0]), _SIGMA_MIN)
        np.testing.assert_allclose(np.exp(table.log_knots[-1]), _SIGMA_MAX)

        # spacing at least as dense as validated
        dlog = np.diff(table.log_knots)
        assert np.max(dlog) <= pm.VALIDATED_DLOG + 1e-12

        lk = jnp.asarray(table.log_knots)
        vj = jnp.asarray(table.values)
        sj = jnp.asarray(table.slopes)

        for s in _eval_grid(table.log_knots):
            pm.validate_sigma_in_range(s, table.log_knots)
            got = np.asarray(masses_fn(np.log(s), lk, vj, sj))
            want = np.array([_oracle_mass(poly, float(xi), float(yi), s, ws)
                             for xi, yi in zip(x, y)])
            err = np.max(np.abs(got - want))
            max_abs_err = max(max_abs_err, float(err))
            assert err <= pm.TAU_ABS, (
                f"{label} ws={ws} sigma={s}: mass err {err} > {pm.TAU_ABS}")

            dgot = np.asarray(d_fn(np.log(s), lk, vj, sj))
            for i, (xi, yi) in enumerate(zip(x, y)):
                dwant, floor = _oracle_dmass_dlogsigma(
                    poly, float(xi), float(yi), s, ws)
                derr = abs(float(dgot[i]) - dwant)
                max_deriv_err = max(max_deriv_err, derr)
                max_fd_floor = max(max_fd_floor, floor)
                assert derr <= pm.TAU_DERIV, (
                    f"{label} ws={ws} sigma={s} event={i}: "
                    f"deriv err {derr} > {pm.TAU_DERIV}")

        # no extrapolation
        with pytest.raises(ValueError, match="outside table-supported"):
            pm.validate_sigma_in_range(_SIGMA_MIN * 0.99, table.log_knots)
        with pytest.raises(ValueError, match="outside table-supported"):
            pm.validate_sigma_in_range(_SIGMA_MAX * 1.01, table.log_knots)

        # float32 online cast (construction remains f64; global x64 restored)
        lk32 = jnp.asarray(table.log_knots, dtype=jnp.float32)
        vj32 = jnp.asarray(table.values, dtype=jnp.float32)
        sj32 = jnp.asarray(table.slopes, dtype=jnp.float32)
        for s in (50.0, 500.0, 2000.0, 4500.0):
            got32 = np.asarray(masses_fn(
                np.float32(np.log(s)), lk32, vj32, sj32))
            want = np.array([_oracle_mass(poly, float(xi), float(yi), s, ws)
                             for xi, yi in zip(x, y)])
            assert np.max(np.abs(got32 - want)) <= pm.TAU_ABS

    print(
        f"\n[wide-range hybrid confirmation ws={ws}] "
        f"K={pm.knot_count(_SIGMA_MIN, _SIGMA_MAX)} "
        f"max_abs_err={max_abs_err:.6e} max_deriv_err={max_deriv_err:.6e} "
        f"fd_floor={max_fd_floor:.6e}"
    )


def test_knot_count_preserves_validated_spacing():
    assert pm.knot_count(10.0, 500.0) == pm.VALIDATED_K
    assert pm.knot_count(10.0, 5000.0) > pm.VALIDATED_K
    # denser or equal log step on a wider range
    k = pm.knot_count(10.0, 5000.0)
    dlog = (np.log(5000.0) - np.log(10.0)) / (k - 1)
    assert dlog <= pm.VALIDATED_DLOG + 1e-15
