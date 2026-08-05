"""B1: polygon mass-table install trusts table build settings + accuracy budget.

At tip ``cd62288``, ``build_excitation_support`` compared a supplied table's
``h_panel`` / ``gl_order`` to ``DEFAULT_PANEL_H_M`` / ``DEFAULT_GL_ORDER``.
That rejects valid tables the panel-ratio guard itself instructed the user
to build, and does not assert the production accuracy budget.

These tests are the RED discrimination suite for the CF repair: table
provenance is authoritative; acceptance is the ``PRODUCTION_TAU_ABS`` budget
(via the panel/min_sigma resolution gate that protects it).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import geopandas as gpd
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from shapely.geometry import box

from bstpp import polygon_mass as pm
from bstpp.config import NumericalConfigError, panel_ratio_invariant_clause
from bstpp.main import Hawkes_Model
from bstpp.polygon_mass import (
    DEFAULT_GL_ORDER,
    DEFAULT_PANEL_H_M,
    MAX_PANEL_TO_MIN_SIGMA_RATIO,
    PRODUCTION_TAU_ABS,
    build_quad_table,
    prepare_polygon_mass_table,
)

T_DAYS = 30.0
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(0.5, 0.5),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(0.25),
)

# CRS-less unit domain: default panel_h_m=20 is far too coarse for small σ.
MIN_SIGMA_SMALL = 0.05
MAX_SIGMA_SMALL = 0.5
# Panel at the resolve-guard ceiling (ratio == 8) — sufficient for DEFAULT_GL_ORDER
# under the measured residual gate; gl_order=8 needs a finer panel.
PANEL_GUIDED = MAX_PANEL_TO_MIN_SIGMA_RATIO * MIN_SIGMA_SMALL  # 0.4
PANEL_GL8 = PANEL_GUIDED / 2.0  # ratio 4; meets PRODUCTION_TAU_ABS at gl=8

# Default-path domain in metres-like units with large min_sigma so defaults OK.
A_DEFAULT = np.array([[0.0, 200.0], [0.0, 200.0]])
MIN_SIGMA_DEFAULT = 5.0
MAX_SIGMA_DEFAULT = 40.0


def _unit_gdf():
    return gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)])


def _events_unit(n=6, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "X": rng.uniform(0.1, 0.9, n),
        "Y": rng.uniform(0.1, 0.9, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _events_default(n=6, seed=1):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "X": rng.uniform(20.0, 180.0, n),
        "Y": rng.uniform(20.0, 180.0, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _prepare_guided(data, *, gl_order=DEFAULT_GL_ORDER, panel_h_m=PANEL_GUIDED):
    geom = _unit_gdf().geometry.union_all()
    return prepare_polygon_mass_table(
        geom,
        data["X"].to_numpy(dtype=float),
        data["Y"].to_numpy(dtype=float),
        min_sigma=MIN_SIGMA_SMALL,
        max_sigma=MAX_SIGMA_SMALL,
        panel_h_m=panel_h_m,
        gl_order=gl_order,
        crs=None,
    )


def _hawkes_polygon(data, table, *, A=None, min_sigma=None, max_sigma=None):
    return Hawkes_Model(
        data,
        _unit_gdf() if A is None else A,
        T_DAYS,
        cox_background=False,
        excitation_support="polygon",
        min_sigma=MIN_SIGMA_SMALL if min_sigma is None else min_sigma,
        max_sigma=MAX_SIGMA_SMALL if max_sigma is None else max_sigma,
        mass_table=table,
        **PRIORS,
    )


# ---------------------------------------------------------------------------
# 1. The trap itself
# ---------------------------------------------------------------------------

def test_crsless_guided_panel_installs_after_ratio_guard():
    data = _events_unit()
    geom = _unit_gdf().geometry.union_all()
    # A-26 / D-40: one identity for this invariant, at build time too.
    with pytest.raises(NumericalConfigError) as ei:
        prepare_polygon_mass_table(
            geom,
            data["X"].to_numpy(dtype=float),
            data["Y"].to_numpy(dtype=float),
            min_sigma=MIN_SIGMA_SMALL,
            max_sigma=MAX_SIGMA_SMALL,
            # defaults: panel_h_m=20
        )
    assert str(ei.value).startswith(panel_ratio_invariant_clause(
        panel_h_m=DEFAULT_PANEL_H_M, min_sigma=MIN_SIGMA_SMALL,
        ratio_ceil=MAX_PANEL_TO_MIN_SIGMA_RATIO, tau_abs=PRODUCTION_TAU_ABS))
    table = _prepare_guided(data)
    assert float(table.h_panel) == pytest.approx(PANEL_GUIDED)
    # Must construct. Today: h_panel mismatch vs build setting 20.0.
    m = _hawkes_polygon(data, table)
    assert m.excitation_support.mode == "polygon"
    # Ownership copy at install (pre-3f B4): not the caller object.
    assert m.excitation_support.mass_table is not table
    np.testing.assert_allclose(
        m.excitation_support.mass_table.values, table.values)


# ---------------------------------------------------------------------------
# 2. Non-default gl_order
# ---------------------------------------------------------------------------

def test_nondefault_gl_order_installs_when_budget_met():
    data = _events_unit()
    # gl_order=8 at the ratio ceiling fails the measured residual gate;
    # a finer panel is required (Commit C option i).
    table = _prepare_guided(data, gl_order=8, panel_h_m=PANEL_GL8)
    assert int(table.gl_order) == 8
    m = _hawkes_polygon(data, table)
    assert int(m.excitation_support.mass_table.gl_order) == 8
    assert m.excitation_provenance["measured_max_abs_residual"] <= PRODUCTION_TAU_ABS


def test_gl_order_8_at_ratio_ceiling_fails_measured_budget():
    """Ratio surrogate alone is not valid across gl_order (Commit C)."""
    data = _events_unit()
    table = _prepare_guided(data, gl_order=8, panel_h_m=PANEL_GUIDED)
    assert float(table.h_panel) / MIN_SIGMA_SMALL <= MAX_PANEL_TO_MIN_SIGMA_RATIO
    with pytest.raises(ValueError, match="PRODUCTION_TAU_ABS|measured|residual") as ei:
        _hawkes_polygon(data, table)
    msg = str(ei.value)
    assert "PRODUCTION_TAU_ABS" in msg or str(PRODUCTION_TAU_ABS) in msg
    assert "BUDGET_REFERENCE_GL_ORDER" in msg or "residual" in msg.lower()


# ---------------------------------------------------------------------------
# 3. Budget violation rejected (must not regress)
# ---------------------------------------------------------------------------

def test_table_too_coarse_for_model_min_sigma_rejected():
    """Bypass prepare's ratio guard via build_quad_table; install must refuse."""
    data = _events_unit()
    geom = _unit_gdf().geometry.union_all()
    coarse = build_quad_table(
        geom,
        data["X"].to_numpy(dtype=np.float64),
        data["Y"].to_numpy(dtype=np.float64),
        MIN_SIGMA_SMALL,
        MAX_SIGMA_SMALL,
        h_panel=DEFAULT_PANEL_H_M,  # 20; ratio = 400 >> 8
        gl_order=DEFAULT_GL_ORDER,
    )
    assert float(coarse.h_panel) / MIN_SIGMA_SMALL > MAX_PANEL_TO_MIN_SIGMA_RATIO
    # A-26 / D-40: the constructor raises the canonical clause verbatim, and
    # the identity is NumericalConfigError -- the old bare ValueError could not
    # distinguish this from the install-site error it was supposed to pin.
    with pytest.raises(NumericalConfigError) as ei:
        _hawkes_polygon(data, coarse)
    msg = str(ei.value)
    assert msg == panel_ratio_invariant_clause(
        panel_h_m=DEFAULT_PANEL_H_M, min_sigma=MIN_SIGMA_SMALL,
        ratio_ceil=MAX_PANEL_TO_MIN_SIGMA_RATIO, tau_abs=PRODUCTION_TAU_ABS)
    assert "PRODUCTION_TAU_ABS" in msg
    assert str(MIN_SIGMA_SMALL) in msg


# ---------------------------------------------------------------------------
# 4. Default path unchanged
# ---------------------------------------------------------------------------

def test_default_panel_gl_order_still_installs_when_budget_met():
    data = _events_default()
    from tests._polygon_prepare_helpers import prepare_table_for_model
    table = prepare_table_for_model(
        data, A_DEFAULT,
        min_sigma=MIN_SIGMA_DEFAULT, max_sigma=MAX_SIGMA_DEFAULT)
    assert float(table.h_panel) == pytest.approx(DEFAULT_PANEL_H_M)
    assert int(table.gl_order) == DEFAULT_GL_ORDER
    m = Hawkes_Model(
        data, A_DEFAULT, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=MIN_SIGMA_DEFAULT, max_sigma=MAX_SIGMA_DEFAULT,
        mass_table=table, **PRIORS,
    )
    assert m.excitation_support.mass_table is not table
    np.testing.assert_allclose(
        m.excitation_support.mass_table.values, table.values)


# ---------------------------------------------------------------------------
# 5. Same rule at held-out scoring and set_window
# ---------------------------------------------------------------------------

def test_heldout_guided_panel_table_accepted():
    train = _events_unit(seed=2)
    test = _events_unit(seed=3)
    train_table = _prepare_guided(train)
    m = _hawkes_polygon(train, train_table)
    m.samples = {
        "a_0": np.array([0.0], dtype=np.float32),
        "alpha": np.array([0.3], dtype=np.float32),
        "beta": np.array([2.0], dtype=np.float32),
        "sigmax_2": np.array([0.1], dtype=np.float32),
    }
    test_table = _prepare_guided(test)
    # Must score. Today: held-out support install fails h_panel vs 20.
    ll = m.log_expected_likelihood(test, mass_table=test_table)
    assert np.isfinite(ll)


def test_heldout_nondefault_gl_order_accepted():
    train = _events_unit(seed=4)
    test = _events_unit(seed=5)
    m = _hawkes_polygon(
        train, _prepare_guided(train, gl_order=8, panel_h_m=PANEL_GL8))
    m.samples = {
        "a_0": np.array([0.0], dtype=np.float32),
        "alpha": np.array([0.3], dtype=np.float32),
        "beta": np.array([2.0], dtype=np.float32),
        "sigmax_2": np.array([0.1], dtype=np.float32),
    }
    ll = m.log_expected_likelihood(
        test, mass_table=_prepare_guided(test, gl_order=8, panel_h_m=PANEL_GL8))
    assert np.isfinite(ll)


def test_heldout_coarse_table_rejected():
    train = _events_default()
    from tests._polygon_prepare_helpers import prepare_table_for_model
    m = Hawkes_Model(
        train, A_DEFAULT, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=MIN_SIGMA_DEFAULT, max_sigma=MAX_SIGMA_DEFAULT,
        mass_table=prepare_table_for_model(
            train, A_DEFAULT,
            min_sigma=MIN_SIGMA_DEFAULT, max_sigma=MAX_SIGMA_DEFAULT),
        **PRIORS,
    )
    m.samples = {
        "a_0": np.array([0.0], dtype=np.float32),
        "alpha": np.array([0.3], dtype=np.float32),
        "beta": np.array([2.0], dtype=np.float32),
        "sigmax_2": np.array([25.0], dtype=np.float32),
    }
    test = _events_default(seed=9)
    # Coarse relative to a *small* held-out model would need matching sigma
    # bounds; instead tamper a same-range table's h_panel upward.
    good = prepare_table_for_model(
        test, A_DEFAULT,
        min_sigma=MIN_SIGMA_DEFAULT, max_sigma=MAX_SIGMA_DEFAULT)
    import dataclasses
    coarse = dataclasses.replace(good, h_panel=float(good.h_panel) * 100.0)
    # A-26 / D-40: held-out scoring is a third entry path for the same
    # invariant and now carries the same identity and canonical clause.
    # It still validates against module defaults, not the model's
    # NumericalConfig -- OP-19, routed to WP5 with the ExcitationSupport seam.
    with pytest.raises(NumericalConfigError) as ei:
        m.log_expected_likelihood(test, mass_table=coarse)
    assert str(ei.value).startswith(panel_ratio_invariant_clause(
        panel_h_m=float(coarse.h_panel), min_sigma=MIN_SIGMA_DEFAULT,
        ratio_ceil=MAX_PANEL_TO_MIN_SIGMA_RATIO, tau_abs=PRODUCTION_TAU_ABS))


def test_set_window_accepts_guided_replacement_table():
    data = _events_unit(seed=6)
    table = _prepare_guided(data)
    m = _hawkes_polygon(data, table)
    # Spatial change requires a replacement table for the new window.
    table2 = prepare_polygon_mass_table(
        _unit_gdf().geometry.union_all(),
        data["X"].to_numpy(dtype=float),
        data["Y"].to_numpy(dtype=float),
        min_sigma=MIN_SIGMA_SMALL,
        max_sigma=MAX_SIGMA_SMALL,
        spatial_window=0.5,
        panel_h_m=PANEL_GL8,
        gl_order=8,
        crs=None,
    )
    m.set_window(spatial_window=0.5, mass_table=table2)
    assert m.args["spatial_window"] == pytest.approx(0.5)
    assert int(m.excitation_support.mass_table.gl_order) == 8


# ---------------------------------------------------------------------------
# 6. No silent rebuild on rejection
# ---------------------------------------------------------------------------

def test_rejected_table_does_not_construct_or_rebuild():
    data = _events_unit(seed=7)
    geom = _unit_gdf().geometry.union_all()
    coarse = build_quad_table(
        geom,
        data["X"].to_numpy(dtype=np.float64),
        data["Y"].to_numpy(dtype=np.float64),
        MIN_SIGMA_SMALL,
        MAX_SIGMA_SMALL,
        h_panel=DEFAULT_PANEL_H_M,
        gl_order=DEFAULT_GL_ORDER,
    )
    # A-26 / OP-17 generalized: this rejection is the panel-ratio invariant,
    # so it is pinned by identity and by the canonical clause, not by a bare
    # ValueError that any failure in the install path would have satisfied.
    ratio_clause = panel_ratio_invariant_clause(
        panel_h_m=DEFAULT_PANEL_H_M, min_sigma=MIN_SIGMA_SMALL,
        ratio_ceil=MAX_PANEL_TO_MIN_SIGMA_RATIO, tau_abs=PRODUCTION_TAU_ABS)
    with pytest.raises(NumericalConfigError) as ei:
        _hawkes_polygon(data, coarse)
    assert str(ei.value) == ratio_clause
    # No constructed model exists to inspect; ensure prepare is not invoked
    # as a silent fallback by patching it to explode if called during ctor.
    calls = {"n": 0}
    real_prepare = pm.prepare_polygon_mass_table

    def _boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("silent prepare_polygon_mass_table during install")

    pm.prepare_polygon_mass_table = _boom  # type: ignore[assignment]
    try:
        with pytest.raises(NumericalConfigError) as ei:
            _hawkes_polygon(data, coarse)
        assert str(ei.value) == ratio_clause
        assert calls["n"] == 0
    finally:
        pm.prepare_polygon_mass_table = real_prepare  # type: ignore[assignment]
