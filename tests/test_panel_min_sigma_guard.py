"""Panel size must resolve configured min_sigma (pre-3f CF + IV/API).

CRS-less / geographic domains treat panel_h_m as domain-coordinate units.
Default panel_h_m=20 with min_sigma≈0.02 yields ratio 1000 and silently
under-resolves the smallest sigma. Production guard:

    effective_panel_h / min_sigma <= MAX_PANEL_TO_MIN_SIGMA_RATIO (= 8)

Reject before the expensive build. PRODUCTION_TAU_ABS remains 1e-5.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import Polygon

from bstpp import polygon_mass as pm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from scripts import polygon_mass_backend_shootout as shoot  # noqa: E402

MIN_SIGMA = 0.02
MAX_SIGMA = 0.2


def _unit_octagon() -> Polygon:
    # CRS-less unit-scale domain (coordinate units, not metres).
    cx, cy, r = 0.5, 0.5, 0.45
    ang = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + np.pi / 8.0
    return Polygon([
        (cx + r * np.cos(a), cy + r * np.sin(a)) for a in ang
    ])


def test_crsless_default_panel_rejected_for_small_min_sigma():
    poly = _unit_octagon()
    x = np.array([0.5, 0.35])
    y = np.array([0.5, 0.55])
    # Default panel_h_m=20 on CRS-less domain => effective_panel=20,
    # ratio = 20/0.02 = 1000 >> 8.
    with pytest.raises(ValueError, match="panel|min_sigma|ratio|panel_h_m") as ei:
        pm.prepare_polygon_mass_table(
            poly, x, y,
            min_sigma=MIN_SIGMA,
            max_sigma=MAX_SIGMA,
        )
    msg = str(ei.value)
    assert "20" in msg or "panel" in msg.lower()
    assert "0.02" in msg or "min_sigma" in msg.lower()
    assert str(pm.MAX_PANEL_TO_MIN_SIGMA_RATIO) in msg or "8" in msg


def test_crsless_valid_small_panel_meets_production_tau_abs():
    poly = _unit_octagon()
    x = np.array([0.5, 0.35])
    y = np.array([0.5, 0.55])
    # panel / min_sigma = 0.1 / 0.02 = 5 <= 8
    panel = 0.1
    x64_before = bool(jax.config.read("jax_enable_x64"))
    table = pm.prepare_polygon_mass_table(
        poly, x, y,
        min_sigma=MIN_SIGMA,
        max_sigma=MAX_SIGMA,
        panel_h_m=panel,
    )
    assert float(table.h_panel) == pytest.approx(panel)
    assert bool(jax.config.read("jax_enable_x64")) == x64_before

    masses_fn, _, _ = pm.make_table_eval()
    lk = jnp.asarray(table.log_knots)
    vj = jnp.asarray(table.values)
    sj = jnp.asarray(table.slopes)

    # Knots + representative off-knot midpoints.
    knot_sigmas = np.exp(table.log_knots)
    mid = np.exp(0.5 * (table.log_knots[:-1] + table.log_knots[1:]))
    offs = mid[[0, len(mid) // 2, -1]]
    sigmas = np.unique(np.concatenate([knot_sigmas, offs]))

    max_abs_err = 0.0
    for s in sigmas:
        got = np.asarray(masses_fn(np.log(s), lk, vj, sj))
        want = np.array([
            shoot.oracle_mass(poly, float(sx), float(sy), float(s), None)
            for sx, sy in zip(x, y)
        ])
        max_abs_err = max(max_abs_err, float(np.max(np.abs(got - want))))

    assert max_abs_err <= pm.PRODUCTION_TAU_ABS, (
        f"max_abs_err={max_abs_err:.6e} > PRODUCTION_TAU_ABS="
        f"{pm.PRODUCTION_TAU_ABS}")
    assert pm.PRODUCTION_TAU_ABS == 1e-5
    assert bool(jax.config.read("jax_enable_x64")) == x64_before

    # Install-time measured residual uses elevated-GL host quadrature; its
    # own bound vs this shapely §13 oracle is BUDGET_REFERENCE_ORACLE_BOUND.
    residual = pm.measure_polygon_mass_table_residual(
        table, domain_geom=poly, event_x_real=x, event_y_real=y,
        spatial_window=None)
    assert residual <= pm.PRODUCTION_TAU_ABS
    assert residual <= max_abs_err + pm.BUDGET_REFERENCE_ORACLE_BOUND
    assert pm.BUDGET_REFERENCE_GL_ORDER == 32
    assert pm.BUDGET_REFERENCE_ORACLE_BOUND == 1e-6
