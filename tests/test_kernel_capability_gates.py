"""Kernel-specific cutoff/scale API gates (pre-3f CF + API).

Tolerance / mean_lag / design-scale mathematics apply only to the exact
builtin kernels that own those formulas:

- Temporal_Exponential: mean_lag_days, temporal_cutoff_tol,
  design_mean_lag_days, temporal leg of shared cutoff_tol, exponential
  omitted-mass calculations.
- Spatial_Symmetric_Gaussian: mandatory sigmax_2, min_sigma/max_sigma,
  prior truncation, spatial_cutoff_tol, design_sigma, spatial leg of
  shared cutoff_tol, Gaussian omitted-mass calculations.

Temporal_Power_Law.beta is a shape parameter, not a mean lag. Custom
rectangle spatial triggers use their own get_par_names() priors and an
explicit spatial_window; they must not require an irrelevant sigmax_2.
Polygon mode retains the existing exact-type Spatial_Symmetric_Gaussian
gate. Gates are per-axis: a supported axis may use its tolerance API even
when the other axis is a custom/unsupported kernel.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest

from bstpp.main import Hawkes_Model
from bstpp.trigger import (
    Spatial_Symmetric_Gaussian,
    Temporal_Exponential,
    Temporal_Power_Law,
    Trigger,
)
from tests._polygon_prepare_helpers import prepare_table_for_model
from tests.test_phase3d_excitation_support import _NonGaussianSpatial

T_DAYS = 30.0
A = np.array([[0.0, 1.0], [0.0, 1.0]])


class _CustomSpatialRho(Trigger):
    """Minimal non-Gaussian rectangle spatial trigger with parameter ``rho``."""

    def get_par_names(self):
        return ["rho"]

    def compute_trigger(self, pars, pairs_and_dxdy):
        coords, dx_vals, dy_vals = pairs_and_dxdy
        r2 = dx_vals ** 2 + dy_vals ** 2
        return coords, jnp.exp(-pars["rho"] * r2)

    def compute_integral(self, pars, limits):
        # Cheap positive mass for construction/smoke; not a scientific kernel.
        n = limits.shape[-1]
        return jnp.full((n,), 0.5)

    def simulate_trigger(self, pars, rng=None):
        gen = rng if rng is not None else np.random
        return np.asarray(gen.normal(scale=0.1, size=2))


def _data(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(0.05, 0.95, n),
        "Y": rng.uniform(0.05, 0.95, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _base_priors(**extra):
    p = dict(
        a_0=dist.Normal(0, 5),
        alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0),
        gamma=dist.HalfNormal(1.0),
        sigmax_2=dist.HalfNormal(0.25),
    )
    p.update(extra)
    return p


# ---- Temporal_Power_Law rejects exponential-only scale/tolerance APIs ----

def test_power_law_rejects_mean_lag_days():
    # Omit legacy beta so the failure under test is the capability gate, not
    # the mean_lag_days/beta mutual-exclusion check.
    priors = dict(
        a_0=dist.Normal(0, 5),
        alpha=dist.Beta(2, 2),
        gamma=dist.HalfNormal(1.0),
        sigmax_2=dist.HalfNormal(0.25),
    )
    with pytest.raises(TypeError, match="Temporal_Exponential|mean_lag_days"):
        Hawkes_Model(
            _data(), A, T_DAYS, cox_background=False,
            temporal_trig=Temporal_Power_Law,
            mean_lag_days=dist.HalfNormal(2.0),
            **priors,
        )


def test_power_law_rejects_temporal_cutoff_tol():
    with pytest.raises(TypeError, match="Temporal_Exponential|temporal_cutoff_tol"):
        Hawkes_Model(
            _data(), A, T_DAYS, cox_background=False,
            temporal_trig=Temporal_Power_Law,
            temporal_cutoff_tol=1e-3,
            design_mean_lag_days=2.0,
            **_base_priors(),
        )


def test_power_law_rejects_design_mean_lag_days_alone():
    with pytest.raises(TypeError, match="Temporal_Exponential|design_mean_lag"):
        Hawkes_Model(
            _data(), A, T_DAYS, cox_background=False,
            temporal_trig=Temporal_Power_Law,
            design_mean_lag_days=2.0,
            window=25.0,
            **_base_priors(),
        )


def test_power_law_rejects_shared_cutoff_tol():
    with pytest.raises(TypeError, match="cutoff_tol"):
        Hawkes_Model(
            _data(), A, T_DAYS, cox_background=False,
            temporal_trig=Temporal_Power_Law,
            cutoff_tol=1e-3,
            design_mean_lag_days=2.0,
            design_sigma=0.1,
            **_base_priors(),
        )


def test_power_law_accepts_explicit_physical_cutoff_and_legacy_priors():
    m = Hawkes_Model(
        _data(), A, T_DAYS, cox_background=False,
        temporal_trig=Temporal_Power_Law,
        window=25.0,
        spatial_window=0.4,
        **_base_priors(),
    )
    assert type(m.args["t_trig"]) is Temporal_Power_Law
    assert m.args["window"] == pytest.approx(25.0)
    assert m.cutoff_provenance.temporal.selection == "physical"
    assert m.cutoff_provenance.temporal.omitted_mass_at_design is None
    assert "beta" in m.args["priors"]
    assert "gamma" in m.args["priors"]


# ---- Custom rectangle spatial trigger: rho, no sigmax_2 ----

def test_custom_spatial_accepts_rho_without_sigmax_2():
    priors = dict(
        a_0=dist.Normal(0, 5),
        alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0),
        rho=dist.HalfNormal(1.0),
    )
    m = Hawkes_Model(
        _data(), A, T_DAYS, cox_background=False,
        spatial_trig=_CustomSpatialRho,
        spatial_window=0.35,
        **priors,
    )
    assert type(m.args["sp_trig"]) is _CustomSpatialRho
    assert "sigmax_2" not in m.args["priors"]
    assert "rho" in m.args["priors"]
    assert m.args["spatial_window"] == pytest.approx(0.35)
    assert m.cutoff_provenance.spatial.selection == "physical"
    assert m.cutoff_provenance.spatial.omitted_mass_at_design is None


def test_custom_spatial_rejects_gaussian_only_settings():
    priors = dict(
        a_0=dist.Normal(0, 5),
        alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0),
        rho=dist.HalfNormal(1.0),
    )
    with pytest.raises(TypeError, match="Spatial_Symmetric_Gaussian|sigmax_2|min_sigma|design_sigma|spatial_cutoff"):
        Hawkes_Model(
            _data(), A, T_DAYS, cox_background=False,
            spatial_trig=_CustomSpatialRho,
            spatial_window=0.35,
            min_sigma=0.05,
            max_sigma=0.5,
            **priors,
        )
    with pytest.raises(TypeError, match="Spatial_Symmetric_Gaussian|design_sigma|spatial_cutoff"):
        Hawkes_Model(
            _data(), A, T_DAYS, cox_background=False,
            spatial_trig=_CustomSpatialRho,
            spatial_cutoff_tol=1e-3,
            design_sigma=0.1,
            **priors,
        )


# ---- Per-axis gates (not a global both-defaults-only check) ----

def test_supported_temporal_tol_with_custom_spatial():
    priors = dict(
        a_0=dist.Normal(0, 5),
        alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0),
        rho=dist.HalfNormal(1.0),
    )
    m = Hawkes_Model(
        _data(), A, T_DAYS, cox_background=False,
        temporal_trig=Temporal_Exponential,
        spatial_trig=_CustomSpatialRho,
        temporal_cutoff_tol=1e-3,
        design_mean_lag_days=2.0,
        spatial_window=0.35,
        **priors,
    )
    assert m.cutoff_provenance.temporal.selection == "tolerance"
    assert m.cutoff_provenance.temporal.omitted_mass_at_design == pytest.approx(1e-3)
    assert m.cutoff_provenance.spatial.selection == "physical"
    assert m.cutoff_provenance.spatial.omitted_mass_at_design is None


def test_supported_spatial_tol_with_power_law_temporal():
    m = Hawkes_Model(
        _data(), A, T_DAYS, cox_background=False,
        temporal_trig=Temporal_Power_Law,
        spatial_trig=Spatial_Symmetric_Gaussian,
        window=25.0,
        spatial_cutoff_tol=1e-3,
        design_sigma=0.1,
        **_base_priors(),
    )
    assert m.cutoff_provenance.temporal.selection == "physical"
    assert m.cutoff_provenance.temporal.omitted_mass_at_design is None
    assert m.cutoff_provenance.spatial.selection == "tolerance"
    assert m.cutoff_provenance.spatial.omitted_mass_at_design == pytest.approx(1e-3)


def test_shared_cutoff_tol_rejected_when_either_axis_unsupported():
    priors = dict(
        a_0=dist.Normal(0, 5),
        alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0),
        rho=dist.HalfNormal(1.0),
    )
    with pytest.raises(TypeError, match="cutoff_tol"):
        Hawkes_Model(
            _data(), A, T_DAYS, cox_background=False,
            temporal_trig=Temporal_Exponential,
            spatial_trig=_CustomSpatialRho,
            cutoff_tol=1e-3,
            design_mean_lag_days=2.0,
            design_sigma=0.1,
            **priors,
        )
    with pytest.raises(TypeError, match="cutoff_tol"):
        Hawkes_Model(
            _data(), A, T_DAYS, cox_background=False,
            temporal_trig=Temporal_Power_Law,
            spatial_trig=Spatial_Symmetric_Gaussian,
            cutoff_tol=1e-3,
            design_mean_lag_days=2.0,
            design_sigma=0.1,
            **_base_priors(),
        )


def test_polygon_still_rejects_non_exact_gaussian_spatial_trigger():
    """Unchanged polygon exact-type gate (regression pin)."""
    A_poly = np.array([[0.0, 200.0], [0.0, 200.0]])
    data = pd.DataFrame({
        "X": [50.0, 100.0], "Y": [50.0, 150.0], "T": [1.0, 2.0]})
    priors = dict(
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    table = prepare_table_for_model(
        data, A_poly, min_sigma=5.0, max_sigma=40.0)
    with pytest.raises(TypeError, match="Spatial_Symmetric_Gaussian"):
        Hawkes_Model(
            data, A_poly, T_DAYS, cox_background=False,
            excitation_support="polygon",
            min_sigma=5.0, max_sigma=40.0,
            spatial_trig=_NonGaussianSpatial,
            mass_table=table,
            **priors,
        )
