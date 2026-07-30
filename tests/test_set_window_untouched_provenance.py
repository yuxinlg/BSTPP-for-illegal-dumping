"""set_window must preserve untouched-axis cutoff provenance (pre-3f CF+API).

An ``_UNSET`` axis keeps its existing TemporalCutoffRecord /
SpatialCutoffRecord verbatim (selection, tolerance, design scale, cutoff,
omitted mass). Only an explicitly supplied axis is re-resolved. Do not
reconstruct an omitted axis as a physical selection.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest

from bstpp.cutoffs import TemporalCutoffRecord, SpatialCutoffRecord
from bstpp.main import Hawkes_Model
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


def _records_equal(a, b):
    return a == b


def test_spatial_only_preserves_tolerance_selected_temporal_record():
    m = Hawkes_Model(
        _data(), A, T_DAYS, cox_background=False,
        temporal_cutoff_tol=1e-3,
        design_mean_lag_days=2.0,
        spatial_cutoff_tol=1e-3,
        design_sigma=0.1,
        **PRIORS,
    )
    before_t = m.cutoff_provenance.temporal
    assert before_t.selection == "tolerance"
    assert before_t.requested_tol == pytest.approx(1e-3)
    assert before_t.design_mean_lag_days == pytest.approx(2.0)
    assert before_t.omitted_mass_at_design == pytest.approx(1e-3)

    m.set_window(spatial_window=0.25)
    after_t = m.cutoff_provenance.temporal
    assert _records_equal(after_t, before_t)
    assert isinstance(after_t, TemporalCutoffRecord)
    assert after_t.selection == "tolerance"
    assert after_t.requested_tol == pytest.approx(1e-3)
    assert after_t.design_mean_lag_days == pytest.approx(2.0)
    assert after_t.window_internal == pytest.approx(before_t.window_internal)
    assert after_t.omitted_mass_at_design == pytest.approx(1e-3)
    assert m.cutoff_provenance.spatial.selection == "physical"
    assert m.args["spatial_window"] == pytest.approx(0.25)


def test_spatial_only_preserves_default_untruncated_temporal_record():
    m = Hawkes_Model(
        _data(), A, T_DAYS, cox_background=False,
        spatial_window=0.4,
        **PRIORS,
    )
    before_t = m.cutoff_provenance.temporal
    assert before_t.selection == "default_untruncated"
    assert before_t.omitted_mass_at_design is None

    m.set_window(spatial_window=0.25)
    assert _records_equal(m.cutoff_provenance.temporal, before_t)
    assert m.cutoff_provenance.temporal.selection == "default_untruncated"
    assert m.cutoff_provenance.temporal.omitted_mass_at_design is None
    assert m.args["spatial_window"] == pytest.approx(0.25)


def test_temporal_only_preserves_tolerance_selected_spatial_record():
    m = Hawkes_Model(
        _data(), A, T_DAYS, cox_background=False,
        temporal_cutoff_tol=1e-3,
        design_mean_lag_days=2.0,
        spatial_cutoff_tol=1e-3,
        design_sigma=0.1,
        **PRIORS,
    )
    before_s = m.cutoff_provenance.spatial
    assert before_s.selection == "tolerance"
    assert before_s.requested_tol == pytest.approx(1e-3)
    assert before_s.design_sigma == pytest.approx(0.1)
    assert before_s.omitted_mass_at_design == pytest.approx(1e-3)

    m.set_window(12.0)
    after_s = m.cutoff_provenance.spatial
    assert _records_equal(after_s, before_s)
    assert isinstance(after_s, SpatialCutoffRecord)
    assert after_s.selection == "tolerance"
    assert after_s.requested_tol == pytest.approx(1e-3)
    assert after_s.design_sigma == pytest.approx(0.1)
    assert after_s.spatial_window == pytest.approx(before_s.spatial_window)
    assert after_s.omitted_mass_at_design == pytest.approx(1e-3)
    assert m.cutoff_provenance.temporal.selection == "physical"
    assert m.args["window"] == pytest.approx(12.0)


def test_temporal_only_preserves_default_untruncated_spatial_record():
    m = Hawkes_Model(
        _data(), A, T_DAYS, cox_background=False,
        window=25.0,
        **PRIORS,
    )
    before_s = m.cutoff_provenance.spatial
    assert before_s.selection == "default_untruncated"
    assert before_s.spatial_window is None
    assert before_s.omitted_mass_at_design is None

    m.set_window(12.0)
    assert _records_equal(m.cutoff_provenance.spatial, before_s)
    assert m.cutoff_provenance.spatial.selection == "default_untruncated"
    assert m.cutoff_provenance.spatial.omitted_mass_at_design is None
    assert m.args["window"] == pytest.approx(12.0)


def test_explicit_none_still_clears_or_restores_untruncated():
    m = Hawkes_Model(
        _data(), A, T_DAYS, cox_background=False,
        window=25.0, spatial_window=0.4,
        design_mean_lag_days=2.0, design_sigma=0.1,
        **PRIORS,
    )
    m.set_window(spatial_window=None)
    assert m.args["spatial_window"] is None
    assert m.cutoff_provenance.spatial.selection == "default_untruncated"
    assert m.cutoff_provenance.spatial.omitted_mass_at_design is None

    m.set_window(window=None)
    from bstpp.preparation import T_INTERNAL
    assert m.args["window"] == pytest.approx(float(T_INTERNAL))
    assert m.cutoff_provenance.temporal.selection == "default_untruncated"
    assert m.cutoff_provenance.temporal.omitted_mass_at_design is None


def test_zero_argument_set_window_remains_noop():
    m = Hawkes_Model(
        _data(), A, T_DAYS, cox_background=False,
        temporal_cutoff_tol=1e-3,
        design_mean_lag_days=2.0,
        spatial_cutoff_tol=1e-3,
        design_sigma=0.1,
        **PRIORS,
    )
    before = _observable_window_state(m)
    temporal_id = id(m.cutoff_provenance.temporal)
    spatial_id = id(m.cutoff_provenance.spatial)
    m.set_window()
    _assert_window_state_unchanged(before, m)
    assert id(m.cutoff_provenance.temporal) == temporal_id
    assert id(m.cutoff_provenance.spatial) == spatial_id


def test_polygon_temporal_only_reuses_table_and_preserves_spatial_provenance():
    A_poly = np.array([[0.0, 200.0], [0.0, 200.0]])
    rng = np.random.default_rng(11)
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
        temporal_cutoff_tol=1e-3,
        design_mean_lag_days=2.0,
        spatial_window=50.0,
        mass_table=table,
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    before_s = m.cutoff_provenance.spatial
    before_table = m.excitation_support.mass_table
    m.set_window(12.0)
    assert m.excitation_support.mass_table is before_table
    assert _records_equal(m.cutoff_provenance.spatial, before_s)
    assert m.cutoff_provenance.temporal.selection == "physical"


def test_polygon_incompatible_replacement_still_rolls_back():
    A_poly = np.array([[0.0, 200.0], [0.0, 200.0]])
    rng = np.random.default_rng(12)
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
