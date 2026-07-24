"""Phase 3e: human-unit temporal interface + computational cutoffs (OP-6).

Settled decisions (phase3_baseline_and_decisions section 10.e / OP-6):
- Public temporal scale is mean_lag_days (beta sample site stays internal).
- Spatial sigmax_2 / spatial_window already real-unit -- not reconverted.
- Windows are computational cutoffs of infinite-support kernels (D-13/D-14).
- Spatial geometry is the per-axis square (D-21); tail formulas:
    eps_t = exp(-w/beta),  eps_s = 1 - erf(w_s/(sqrt(2)*sigma))**2
- epsilon-based auto-selection with physical override; physical wins when
  both are supplied; both tolerance and realized cutoff live in provenance.
- Default tolerances subsume the old rules of thumb: w = 5*beta, w_s = 4*sigma.

Acceptance (section 12.2): fixed-cutoff traces/pins unchanged; new unit,
tail-mass, cutoff-consistency, and provenance tests mandatory.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from scipy import special as sp_special

from bstpp.cutoffs import (
    DEFAULT_SPATIAL_SCALE_MULTIPLE,
    DEFAULT_TEMPORAL_SCALE_MULTIPLE,
    DEFAULT_SPATIAL_TOL,
    DEFAULT_TEMPORAL_TOL,
    days_to_internal,
    internal_to_days,
    resolve_computational_cutoffs,
    scale_temporal_prior_to_internal,
    spatial_cutoff_from_tol,
    spatial_omitted_mass,
    temporal_cutoff_from_tol,
    temporal_omitted_mass,
)
from bstpp.main import Hawkes_Model
from bstpp.preparation import T_INTERNAL

T_DAYS = 100.0
PRIORS_INTERNAL = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(0.25),
)


def _data(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(0.1, 0.9, n),
        "Y": rng.uniform(0.1, 0.9, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


# ------------------------------------------------------ default tolerances --
def test_default_tolerances_match_rule_of_thumb_multiples():
    """Default eps subsume w=5*beta and w_s=4*sigma (docstring rules)."""
    assert DEFAULT_TEMPORAL_SCALE_MULTIPLE == 5.0
    assert DEFAULT_SPATIAL_SCALE_MULTIPLE == 4.0
    assert DEFAULT_TEMPORAL_TOL == pytest.approx(math.exp(-5.0))
    assert DEFAULT_SPATIAL_TOL == pytest.approx(
        1.0 - math.erf(4.0 / math.sqrt(2.0)) ** 2)


# ----------------------------------------------------------- tail formulas --
def test_temporal_omitted_mass_and_inverse():
    beta, eps = 2.5, 1e-3
    w = temporal_cutoff_from_tol(eps, beta)
    assert w == pytest.approx(beta * math.log(1.0 / eps))
    assert temporal_omitted_mass(w, beta) == pytest.approx(eps)


def test_spatial_omitted_mass_and_inverse_square_geometry():
    """Independent SciPy reference for the D-21 square formula (not disc)."""
    sigma, eps = 0.3, 0.01
    ws = spatial_cutoff_from_tol(eps, sigma)
    # SciPy erfinv path -- must not share production helpers.
    retained = math.sqrt(1.0 - eps)
    expected_ws = math.sqrt(2.0) * sigma * float(sp_special.erfinv(retained))
    assert ws == pytest.approx(expected_ws)
    got_eps = 1.0 - sp_special.erf(ws / (math.sqrt(2.0) * sigma)) ** 2
    assert spatial_omitted_mass(ws, sigma) == pytest.approx(eps)
    assert got_eps == pytest.approx(eps)
    # Disc formula must NOT be used (retired with D-21); at eps=0.01 the
    # two formulas disagree by several percentage points.
    disc = math.exp(-(ws ** 2) / (2.0 * sigma ** 2))
    assert abs(disc - eps) > 5e-3


# -------------------------------------------------------- days <-> internal --
def test_days_internal_round_trip():
    days = 7.0
    internal = days_to_internal(days, T_DAYS)
    assert internal == pytest.approx(days * T_INTERNAL / T_DAYS)
    assert internal_to_days(internal, T_DAYS) == pytest.approx(days)


def test_mean_lag_days_prior_converts_to_internal_beta_site():
    """Public prior mean_lag_days (days) -> sampled site still named beta."""
    lag_days = dist.HalfNormal(5.0)
    m = Hawkes_Model(
        _data(), np.array([[0.0, 1.0], [0.0, 1.0]]), T_DAYS,
        cox_background=False,
        mean_lag_days=lag_days,
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        sigmax_2=dist.HalfNormal(0.25),
    )
    assert "beta" in m.args["priors"]
    assert "mean_lag_days" not in m.args["priors"]
    # HalfNormal scale transforms by the days->internal factor.
    scale = T_INTERNAL / T_DAYS
    assert float(m.args["priors"]["beta"].scale) == pytest.approx(5.0 * scale)


def test_mean_lag_days_and_beta_together_is_error():
    with pytest.raises(ValueError, match="mean_lag_days.*beta"):
        Hawkes_Model(
            _data(), np.array([[0.0, 1.0], [0.0, 1.0]]), T_DAYS,
            cox_background=False,
            mean_lag_days=dist.HalfNormal(2.0),
            **PRIORS_INTERNAL,
        )


def test_scale_temporal_prior_helper_matches_halfnormal_scale():
    prior = dist.HalfNormal(3.0)
    scaled = scale_temporal_prior_to_internal(prior, T_DAYS)
    assert float(scaled.scale) == pytest.approx(3.0 * T_INTERNAL / T_DAYS)


# ---------------------------------------------- resolve_computational_cutoffs --
def test_resolve_physical_override_wins_over_tolerance():
    prov = resolve_computational_cutoffs(
        horizon_days=T_DAYS,
        temporal_cutoff_days=20.0,
        temporal_cutoff_tol=DEFAULT_TEMPORAL_TOL,
        design_mean_lag_days=2.0,
        spatial_window=0.5,
        spatial_cutoff_tol=DEFAULT_SPATIAL_TOL,
        design_sigma=0.1,
    )
    assert prov.temporal.selection == "physical"
    assert prov.temporal.window_days == pytest.approx(20.0)
    assert prov.temporal.window_internal == pytest.approx(
        days_to_internal(20.0, T_DAYS))
    # Realized omitted mass at the design scale (not the requested tol).
    assert prov.temporal.omitted_mass_at_design == pytest.approx(
        temporal_omitted_mass(prov.temporal.window_internal,
                              days_to_internal(2.0, T_DAYS)))
    assert prov.temporal.requested_tol == pytest.approx(DEFAULT_TEMPORAL_TOL)
    assert prov.spatial.selection == "physical"
    assert prov.spatial.spatial_window == pytest.approx(0.5)
    assert prov.spatial.omitted_mass_at_design == pytest.approx(
        spatial_omitted_mass(0.5, 0.1))


def test_resolve_tolerance_derives_cutoffs_from_design_scales():
    eps_t, eps_s = 1e-3, 1e-4
    beta_days, sigma = 2.0, 0.2
    prov = resolve_computational_cutoffs(
        horizon_days=T_DAYS,
        temporal_cutoff_tol=eps_t,
        design_mean_lag_days=beta_days,
        spatial_cutoff_tol=eps_s,
        design_sigma=sigma,
    )
    assert prov.temporal.selection == "tolerance"
    beta_int = days_to_internal(beta_days, T_DAYS)
    assert prov.temporal.window_internal == pytest.approx(
        temporal_cutoff_from_tol(eps_t, beta_int))
    assert prov.temporal.omitted_mass_at_design == pytest.approx(eps_t)
    assert prov.spatial.selection == "tolerance"
    assert prov.spatial.spatial_window == pytest.approx(
        spatial_cutoff_from_tol(eps_s, sigma))
    assert prov.spatial.omitted_mass_at_design == pytest.approx(eps_s)


def test_resolve_default_untruncated_when_nothing_specified():
    prov = resolve_computational_cutoffs(horizon_days=T_DAYS)
    assert prov.temporal.selection == "default_untruncated"
    assert prov.temporal.window_internal == pytest.approx(float(T_INTERNAL))
    assert prov.spatial.selection == "default_untruncated"
    assert prov.spatial.spatial_window is None


def test_resolve_tolerance_requires_design_scale():
    with pytest.raises(ValueError, match="design_mean_lag_days"):
        resolve_computational_cutoffs(
            horizon_days=T_DAYS, temporal_cutoff_tol=1e-3)
    with pytest.raises(ValueError, match="design_sigma"):
        resolve_computational_cutoffs(
            horizon_days=T_DAYS, spatial_cutoff_tol=1e-4)


# ---------------------------------------------------- model wiring / API ----
def test_legacy_fixed_cutoffs_unchanged_and_report_provenance():
    """BP gate: explicit internal window + real spatial_window bit-stable."""
    m = Hawkes_Model(
        _data(), np.array([[0.0, 1.0], [0.0, 1.0]]), T_DAYS,
        cox_background=False, window=25.0, spatial_window=0.4,
        **PRIORS_INTERNAL,
    )
    assert m.args["window"] == pytest.approx(25.0)
    assert m.args["spatial_window"] == pytest.approx(0.4)
    assert m.cutoff_provenance.temporal.selection == "physical"
    assert m.cutoff_provenance.temporal.window_internal == pytest.approx(25.0)
    assert m.cutoff_provenance.spatial.selection == "physical"
    assert m.cutoff_provenance.spatial.geometry == "per_axis_square"


def test_hawkes_tolerance_api_sets_windows_and_provenance():
    m = Hawkes_Model(
        _data(), np.array([[0.0, 1.0], [0.0, 1.0]]), T_DAYS,
        cox_background=False,
        mean_lag_days=dist.HalfNormal(2.0),
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        sigmax_2=dist.HalfNormal(0.04),
        temporal_cutoff_tol=DEFAULT_TEMPORAL_TOL,
        spatial_cutoff_tol=DEFAULT_SPATIAL_TOL,
        design_mean_lag_days=2.0,
        design_sigma=0.2,
    )
    beta_int = days_to_internal(2.0, T_DAYS)
    assert m.args["window"] == pytest.approx(
        temporal_cutoff_from_tol(DEFAULT_TEMPORAL_TOL, beta_int))
    assert m.args["spatial_window"] == pytest.approx(
        spatial_cutoff_from_tol(DEFAULT_SPATIAL_TOL, 0.2))
    # 4*sigma rule of thumb recovered at the default spatial tol.
    assert m.args["spatial_window"] == pytest.approx(4.0 * 0.2)
    assert m.cutoff_provenance.temporal.selection == "tolerance"
    assert m.cutoff_provenance.spatial.selection == "tolerance"


def test_temporal_cutoff_days_physical_override():
    m = Hawkes_Model(
        _data(), np.array([[0.0, 1.0], [0.0, 1.0]]), T_DAYS,
        cox_background=False,
        temporal_cutoff_days=10.0,
        **PRIORS_INTERNAL,
    )
    assert m.args["window"] == pytest.approx(days_to_internal(10.0, T_DAYS))
    assert m.cutoff_provenance.temporal.window_days == pytest.approx(10.0)


def test_window_and_temporal_cutoff_days_together_is_error():
    with pytest.raises(ValueError, match="temporal_cutoff_days.*window"):
        Hawkes_Model(
            _data(), np.array([[0.0, 1.0], [0.0, 1.0]]), T_DAYS,
            cox_background=False,
            window=25.0, temporal_cutoff_days=10.0,
            **PRIORS_INTERNAL,
        )


def test_tolerance_selected_cutoffs_agree_across_three_legs():
    """D-14: pair mask, compensator clip, and simulator thinning share windows."""
    from bstpp.utils import within_real_box_window

    design_sigma = 0.15
    m = Hawkes_Model(
        _data(n=20, seed=3), np.array([[0.0, 1.0], [0.0, 1.0]]), T_DAYS,
        cox_background=False,
        spatial_cutoff_tol=DEFAULT_SPATIAL_TOL,
        design_sigma=design_sigma,
        **PRIORS_INTERNAL,
    )
    ws = m.args["spatial_window"]
    assert ws == pytest.approx(4.0 * design_sigma)
    # Compensator / args share the same scalar.
    assert m.cutoff_provenance.spatial.spatial_window == pytest.approx(ws)
    # Shared predicate accepts the boundary point of the square.
    assert bool(within_real_box_window(
        np.array([ws]), np.array([0.0]), ws)[0])
    assert not bool(within_real_box_window(
        np.array([ws * 1.01]), np.array([0.0]), ws)[0])


# -------------------------------- invalid tolerance validation (any precedence) --
@pytest.mark.parametrize("bad_tol", [-1.0, 0.0, 1.0, float("nan"), float("inf")])
def test_invalid_temporal_tol_rejected_even_when_physical_wins(bad_tol):
    """Supplied tol must be finite and in (0, 1) even if physical cutoff wins."""
    with pytest.raises(ValueError, match="tol|tolerance"):
        resolve_computational_cutoffs(
            horizon_days=T_DAYS,
            window_internal=25.0,
            temporal_cutoff_tol=bad_tol,
            design_mean_lag_days=2.0,
        )


@pytest.mark.parametrize("bad_tol", [-0.5, 0.0, 1.0, float("nan")])
def test_invalid_spatial_tol_rejected_even_when_physical_wins(bad_tol):
    with pytest.raises(ValueError, match="tol|tolerance"):
        resolve_computational_cutoffs(
            horizon_days=T_DAYS,
            spatial_window=0.5,
            spatial_cutoff_tol=bad_tol,
            design_sigma=0.1,
        )


def test_invalid_shared_cutoff_tol_rejected_with_physical_override():
    with pytest.raises(ValueError, match="tol|tolerance"):
        resolve_computational_cutoffs(
            horizon_days=T_DAYS,
            window_internal=25.0,
            spatial_window=0.5,
            cutoff_tol=-1.0,
            design_mean_lag_days=2.0,
            design_sigma=0.1,
        )


def test_hawkes_rejects_invalid_tol_with_physical_window():
    with pytest.raises(ValueError, match="tol|tolerance"):
        Hawkes_Model(
            _data(), np.array([[0.0, 1.0], [0.0, 1.0]]), T_DAYS,
            cox_background=False,
            window=25.0,
            temporal_cutoff_tol=-1.0,
            **PRIORS_INTERNAL,
        )
