"""set_window sentinel semantics: omit vs explicit None (pre-3f).

``_UNSET`` means leave unchanged; ``None`` means clear / restore the
untruncated default. ``set_window()`` is a true no-op. Mutations remain
transactional; omitted ``mass_table`` never triggers silent rebuild.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest

from bstpp.main import Hawkes_Model, _UNSET
from tests._polygon_prepare_helpers import prepare_table_for_model
from tests.test_phase3e_cutoffs import (
    _assert_window_state_unchanged,
    _observable_window_state,
)

T_DAYS = 30.0
A = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(0.25),
)


def _data(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(0.05, 0.95, n),
        "Y": rng.uniform(0.05, 0.95, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _rect_model(**kw):
    defaults = dict(
        cox_background=False,
        window=25.0,
        spatial_window=0.4,
        design_mean_lag_days=2.0,
        design_sigma=0.1,
    )
    defaults.update(kw)
    return Hawkes_Model(_data(), A, T_DAYS, **defaults, **PRIORS)


def test_set_window_empty_is_noop_preserving_identities():
    m = _rect_model()
    before = _observable_window_state(m)
    support_id = id(m.excitation_support)
    m.set_window()
    _assert_window_state_unchanged(before, m)
    assert id(m.excitation_support) == support_id


def test_set_window_temporal_only_leaves_spatial_unchanged():
    m = _rect_model()
    m.set_window(12.0)
    assert m.args["window"] == pytest.approx(12.0)
    assert m.args["spatial_window"] == pytest.approx(0.4)
    assert m.cutoff_provenance.spatial.spatial_window == pytest.approx(0.4)
    assert m.cutoff_provenance.spatial.selection == "physical"


def test_set_window_spatial_only_leaves_temporal_unchanged():
    m = _rect_model()
    m.set_window(spatial_window=0.25)
    assert m.args["window"] == pytest.approx(25.0)
    assert m.args["spatial_window"] == pytest.approx(0.25)
    assert m.cutoff_provenance.temporal.window_internal == pytest.approx(25.0)


def test_set_window_explicit_none_clears_spatial():
    m = _rect_model()
    m.set_window(spatial_window=None)
    assert m.args["spatial_window"] is None
    assert m.cutoff_provenance.spatial.spatial_window is None
    assert m.cutoff_provenance.spatial.selection == "default_untruncated"
    assert m.args["window"] == pytest.approx(25.0)


def test_set_window_explicit_none_restores_untruncated_temporal():
    m = _rect_model()
    m.set_window(window=None)
    # Full observation-horizon setting (T_INTERNAL), not a computational
    # truncation; omitted mass at design is N/A.
    from bstpp.preparation import T_INTERNAL
    assert m.args["window"] == pytest.approx(float(T_INTERNAL))
    assert m.cutoff_provenance.temporal.selection == "default_untruncated"
    assert m.cutoff_provenance.temporal.omitted_mass_at_design is None
    assert m.args["spatial_window"] == pytest.approx(0.4)


def test_set_window_retains_design_scales_and_recomputes_omission():
    m = _rect_model(design_mean_lag_days=2.0, design_sigma=0.1)
    assert m.cutoff_provenance.temporal.design_mean_lag_days == pytest.approx(2.0)
    assert m.cutoff_provenance.spatial.design_sigma == pytest.approx(0.1)

    m.set_window(10.0, spatial_window=0.2)
    assert m.cutoff_provenance.temporal.design_mean_lag_days == pytest.approx(2.0)
    assert m.cutoff_provenance.spatial.design_sigma == pytest.approx(0.1)
    assert m.cutoff_provenance.temporal.omitted_mass_at_design is not None
    assert m.cutoff_provenance.spatial.omitted_mass_at_design is not None
    # Realized omission must match the design scales at the new cutoffs.
    from bstpp.cutoffs import spatial_omitted_mass, temporal_omitted_mass
    beta_int = m.cutoff_provenance.temporal.design_beta_internal
    assert m.cutoff_provenance.temporal.omitted_mass_at_design == pytest.approx(
        temporal_omitted_mass(10.0, beta_int))
    assert m.cutoff_provenance.spatial.omitted_mass_at_design == pytest.approx(
        spatial_omitted_mass(0.2, 0.1))


def test_set_window_unset_sentinel_is_private():
    assert _UNSET is not None
    # Distinct from None so explicit clear remains expressible.
    assert _UNSET is not None
    assert _UNSET is not False


def test_polygon_temporal_only_reuses_table_when_spatial_omitted():
    A_poly = np.array([[0.0, 200.0], [0.0, 200.0]])
    rng = np.random.default_rng(1)
    data = pd.DataFrame({
        "X": rng.uniform(20, 180, 6),
        "Y": rng.uniform(20, 180, 6),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 6)),
    })
    table = prepare_table_for_model(
        data, A_poly, min_sigma=5.0, max_sigma=40.0, spatial_window=50.0)
    m = Hawkes_Model(
        data, A_poly, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        window=25.0, spatial_window=50.0,
        mass_table=table,
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    before_table = m.excitation_support.mass_table
    m.set_window(12.0)  # omit spatial_window → keep 50; reuse table
    assert m.args["spatial_window"] == pytest.approx(50.0)
    assert m.excitation_support.mass_table is before_table


def test_polygon_spatial_change_requires_explicit_replacement():
    A_poly = np.array([[0.0, 200.0], [0.0, 200.0]])
    rng = np.random.default_rng(2)
    data = pd.DataFrame({
        "X": rng.uniform(20, 180, 6),
        "Y": rng.uniform(20, 180, 6),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 6)),
    })
    table = prepare_table_for_model(
        data, A_poly, min_sigma=5.0, max_sigma=40.0, spatial_window=50.0)
    m = Hawkes_Model(
        data, A_poly, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        window=25.0, spatial_window=50.0,
        mass_table=table,
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    before = _observable_window_state(m)
    with pytest.raises(ValueError, match="mass_table"):
        m.set_window(spatial_window=40.0)
    _assert_window_state_unchanged(before, m)


def test_polygon_incompatible_replacement_rolls_back_everything():
    A_poly = np.array([[0.0, 200.0], [0.0, 200.0]])
    rng = np.random.default_rng(3)
    data = pd.DataFrame({
        "X": rng.uniform(20, 180, 6),
        "Y": rng.uniform(20, 180, 6),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 6)),
    })
    table = prepare_table_for_model(
        data, A_poly, min_sigma=5.0, max_sigma=40.0, spatial_window=50.0)
    m = Hawkes_Model(
        data, A_poly, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        window=25.0, spatial_window=50.0,
        mass_table=table,
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    before = _observable_window_state(m)
    bad = prepare_table_for_model(
        data, A_poly, min_sigma=5.0, max_sigma=40.0, spatial_window=50.0)
    with pytest.raises(ValueError):
        m.set_window(spatial_window=40.0, mass_table=bad)
    _assert_window_state_unchanged(before, m)


def test_set_window_mass_table_rebuilds_numerical_config():
    """D-35: successful mass_table= install updates NumericalConfig (WP1.4b)."""
    from bstpp.polygon_mass import prepare_polygon_mass_table
    from bstpp.preparation import prepare_domain
    from shapely.geometry import box as shapely_box

    A_poly = np.array([[0.0, 200.0], [0.0, 200.0]])
    rng = np.random.default_rng(4)
    data = pd.DataFrame({
        "X": rng.uniform(20, 180, 6),
        "Y": rng.uniform(20, 180, 6),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 6)),
    })
    table = prepare_table_for_model(
        data, A_poly, min_sigma=5.0, max_sigma=40.0, spatial_window=50.0)
    m = Hawkes_Model(
        data, A_poly, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        window=25.0, spatial_window=50.0,
        mass_table=table,
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    assert m.numerical_config.panel_h_m == pytest.approx(float(table.h_panel))
    assert m.numerical_config.gl_order == int(table.gl_order)

    dom = prepare_domain(A_poly)
    geom = shapely_box(0.0, 0.0, 200.0, 200.0)
    new_table = prepare_polygon_mass_table(
        geom,
        data["X"].to_numpy(dtype=float),
        data["Y"].to_numpy(dtype=float),
        min_sigma=5.0,
        max_sigma=40.0,
        spatial_window=40.0,
        panel_h_m=10.0,
        gl_order=8,
        crs=dom.crs,
    )
    m.set_window(spatial_window=40.0, mass_table=new_table)
    assert m.numerical_config.panel_h_m == pytest.approx(10.0)
    assert m.numerical_config.gl_order == 8
    assert m.excitation_support.mass_table.h_panel == pytest.approx(10.0)
    assert int(m.excitation_support.mass_table.gl_order) == 8


def test_set_window_rejected_mass_table_leaves_numerical_config():
    """D-35: rejected mass_table= install leaves NumericalConfig unchanged."""
    A_poly = np.array([[0.0, 200.0], [0.0, 200.0]])
    rng = np.random.default_rng(5)
    data = pd.DataFrame({
        "X": rng.uniform(20, 180, 6),
        "Y": rng.uniform(20, 180, 6),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 6)),
    })
    table = prepare_table_for_model(
        data, A_poly, min_sigma=5.0, max_sigma=40.0, spatial_window=50.0)
    m = Hawkes_Model(
        data, A_poly, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        window=25.0, spatial_window=50.0,
        mass_table=table,
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    before_cfg = m.numerical_config
    before_panel = before_cfg.panel_h_m
    before_gl = before_cfg.gl_order
    bad = prepare_table_for_model(
        data, A_poly, min_sigma=5.0, max_sigma=40.0, spatial_window=50.0)
    with pytest.raises(ValueError):
        m.set_window(spatial_window=40.0, mass_table=bad)
    assert m.numerical_config is before_cfg
    assert m.numerical_config.panel_h_m == pytest.approx(before_panel)
    assert m.numerical_config.gl_order == before_gl
