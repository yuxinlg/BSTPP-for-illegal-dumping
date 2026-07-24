"""Computational cutoffs and temporal unit conversion (Phase 3e / OP-6).

Scientific kernels have infinite support (D-13). Finite temporal/spatial
windows are computational cutoffs (D-14) that remove near-zero offspring
mass; they apply consistently to pair construction, compensator, and
simulator. Spatial cutoff geometry is the per-axis real-unit square (D-21).

Public temporal scale is ``mean_lag_days``; the numpyro sample site remains
``beta`` in internal units (T_INTERNAL). Spatial ``sigmax_2`` /
``spatial_window`` are already real-unit and are not reconverted here.

Tail formulas (retained mass complementary):
    eps_t = exp(-w / beta)
    eps_s = 1 - erf(w_s / (sqrt(2) * sigma))**2

Default tolerances subsume the historical rules of thumb
``w >= 5 * beta`` and ``w_s >= 4 * sigma``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

from numpyro.distributions import Distribution, TransformedDistribution
from numpyro.distributions.transforms import AffineTransform
from scipy.special import erfinv as sp_erfinv

from .preparation import T_INTERNAL

# Historical docstring rules of thumb, now expressed as default omitted-mass
# tolerances (section 10.e / OP-6).
DEFAULT_TEMPORAL_SCALE_MULTIPLE = 5.0
DEFAULT_SPATIAL_SCALE_MULTIPLE = 4.0
DEFAULT_TEMPORAL_TOL = float(math.exp(-DEFAULT_TEMPORAL_SCALE_MULTIPLE))
DEFAULT_SPATIAL_TOL = float(
    1.0 - math.erf(DEFAULT_SPATIAL_SCALE_MULTIPLE / math.sqrt(2.0)) ** 2
)


def days_to_internal(days: float, horizon_days: float) -> float:
    """Convert a real-day duration to internal time units ([0, T_INTERNAL])."""
    days = float(days)
    horizon_days = float(horizon_days)
    if not (math.isfinite(days) and days >= 0):
        raise ValueError(f"days must be finite and >= 0; got {days}")
    if not (math.isfinite(horizon_days) and horizon_days > 0):
        raise ValueError(
            f"horizon_days must be finite and > 0; got {horizon_days}")
    return days * (T_INTERNAL / horizon_days)


def internal_to_days(internal: float, horizon_days: float) -> float:
    """Convert an internal-time duration to real days."""
    internal = float(internal)
    horizon_days = float(horizon_days)
    if not (math.isfinite(internal) and internal >= 0):
        raise ValueError(f"internal must be finite and >= 0; got {internal}")
    if not (math.isfinite(horizon_days) and horizon_days > 0):
        raise ValueError(
            f"horizon_days must be finite and > 0; got {horizon_days}")
    return internal * (horizon_days / T_INTERNAL)


def scale_temporal_prior_to_internal(
    prior_days: Distribution, horizon_days: float
) -> Distribution:
    """Map a days-unit temporal-scale prior onto the internal ``beta`` site.

    ``beta_internal = mean_lag_days * (T_INTERNAL / horizon_days)``.
    Scale families (HalfNormal, …) keep their concrete type so priors stay
    inspectable; other distributions use an affine transform.
    """
    if not isinstance(prior_days, Distribution):
        raise TypeError(
            "mean_lag_days prior must be a numpyro Distribution; "
            f"got {type(prior_days)!r}")
    factor = T_INTERNAL / float(horizon_days)
    if not (math.isfinite(factor) and factor > 0):
        raise ValueError(f"invalid temporal scale factor {factor}")
    # HalfNormal(scale_days) -> HalfNormal(scale_days * factor).
    if prior_days.__class__.__name__ == "HalfNormal":
        return prior_days.__class__(prior_days.scale * factor)
    return TransformedDistribution(
        prior_days, AffineTransform(0.0, factor))


def temporal_omitted_mass(window: float, beta: float) -> float:
    """Omitted temporal offspring mass eps_t = exp(-w / beta)."""
    window = float(window)
    beta = float(beta)
    if not (math.isfinite(window) and window >= 0):
        raise ValueError(f"window must be finite and >= 0; got {window}")
    if not (math.isfinite(beta) and beta > 0):
        raise ValueError(f"beta must be finite and > 0; got {beta}")
    return float(math.exp(-window / beta))


def _validate_cutoff_tol(eps: float, *, name: str) -> float:
    """Require a finite omitted-mass tolerance in (0, 1)."""
    eps = float(eps)
    if not (math.isfinite(eps) and 0.0 < eps < 1.0):
        raise ValueError(f"{name} must be finite and in (0, 1); got {eps}")
    return eps


def temporal_cutoff_from_tol(eps: float, beta: float) -> float:
    """w = beta * ln(1/eps) for target omitted mass eps in (0, 1)."""
    eps = _validate_cutoff_tol(eps, name="temporal tol")
    beta = float(beta)
    if not (math.isfinite(beta) and beta > 0):
        raise ValueError(f"beta must be finite and > 0; got {beta}")
    return float(beta * math.log(1.0 / eps))


def spatial_omitted_mass(spatial_window: float, sigma: float) -> float:
    """Omitted spatial mass for the D-21 per-axis square cutoff.

    eps_s = 1 - erf(w_s / (sqrt(2) * sigma))**2.
    Never mix with the retired disc formula exp(-w_s^2 / (2 sigma^2)).
    """
    spatial_window = float(spatial_window)
    sigma = float(sigma)
    if not (math.isfinite(spatial_window) and spatial_window >= 0):
        raise ValueError(
            f"spatial_window must be finite and >= 0; got {spatial_window}")
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma must be finite and > 0; got {sigma}")
    z = spatial_window / (math.sqrt(2.0) * sigma)
    return float(1.0 - math.erf(z) ** 2)


def spatial_cutoff_from_tol(eps: float, sigma: float) -> float:
    """w_s = sqrt(2) * sigma * erfinv(sqrt(1 - eps)) for the square cutoff."""
    eps = _validate_cutoff_tol(eps, name="spatial tol")
    sigma = float(sigma)
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma must be finite and > 0; got {sigma}")
    retained = math.sqrt(1.0 - eps)
    # numpy<2 has no np.erfinv; scipy is already a runtime dependency.
    return float(math.sqrt(2.0) * sigma * float(sp_erfinv(retained)))


@dataclass(frozen=True)
class TemporalCutoffRecord:
    selection: str  # physical | tolerance | default_untruncated
    window_internal: float
    window_days: float
    requested_tol: Optional[float]
    design_mean_lag_days: Optional[float]
    design_beta_internal: Optional[float]
    omitted_mass_at_design: Optional[float]
    geometry: str = "horizon_clip"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpatialCutoffRecord:
    selection: str  # physical | tolerance | default_untruncated
    spatial_window: Optional[float]
    requested_tol: Optional[float]
    design_sigma: Optional[float]
    omitted_mass_at_design: Optional[float]
    geometry: str = "per_axis_square"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CutoffProvenance:
    temporal: TemporalCutoffRecord
    spatial: SpatialCutoffRecord
    default_temporal_tol: float = DEFAULT_TEMPORAL_TOL
    default_spatial_tol: float = DEFAULT_SPATIAL_TOL

    def to_dict(self) -> dict[str, Any]:
        return {
            "temporal": self.temporal.to_dict(),
            "spatial": self.spatial.to_dict(),
            "default_temporal_tol": self.default_temporal_tol,
            "default_spatial_tol": self.default_spatial_tol,
        }


def resolve_computational_cutoffs(
    *,
    horizon_days: float,
    # Physical overrides (temporal: days; spatial: real length).
    temporal_cutoff_days: Optional[float] = None,
    window_internal: Optional[float] = None,
    spatial_window: Optional[float] = None,
    # Tolerance-based selection (OP-6).
    temporal_cutoff_tol: Optional[float] = None,
    spatial_cutoff_tol: Optional[float] = None,
    cutoff_tol: Optional[float] = None,
    # Design scales for eps -> cutoff (latent scales are not known at build).
    design_mean_lag_days: Optional[float] = None,
    design_sigma: Optional[float] = None,
) -> CutoffProvenance:
    """Resolve temporal/spatial computational cutoffs and provenance.

    Precedence per axis (OP-6): physical override wins over tolerance; when
    both are supplied the physical cutoff is used and the requested tol is
    retained in provenance alongside the realized omitted mass at the design
    scale. When nothing is specified, legacy untruncated defaults are kept
    (temporal window = T_INTERNAL, spatial_window = None) so fixed-cutoff
    traces/pins remain bit-stable.
    """
    if temporal_cutoff_days is not None and window_internal is not None:
        raise ValueError(
            "Pass temporal_cutoff_days or window, not both "
            "(days vs legacy internal units).")

    eps_t = temporal_cutoff_tol if temporal_cutoff_tol is not None else cutoff_tol
    eps_s = spatial_cutoff_tol if spatial_cutoff_tol is not None else cutoff_tol
    # Validate every supplied tolerance even when a physical cutoff wins
    # precedence and the tol is only retained in provenance.
    if eps_t is not None:
        _validate_cutoff_tol(eps_t, name="temporal cutoff tol")
    if eps_s is not None:
        _validate_cutoff_tol(eps_s, name="spatial cutoff tol")

    # ---- temporal ----
    design_beta_internal: Optional[float] = None
    if design_mean_lag_days is not None:
        design_beta_internal = days_to_internal(
            float(design_mean_lag_days), horizon_days)

    if temporal_cutoff_days is not None:
        w_days = float(temporal_cutoff_days)
        if not (math.isfinite(w_days) and w_days > 0):
            raise ValueError(
                f"temporal_cutoff_days must be finite and > 0; got {w_days}")
        w_int = days_to_internal(w_days, horizon_days)
        omitted = (
            temporal_omitted_mass(w_int, design_beta_internal)
            if design_beta_internal is not None else None)
        temporal = TemporalCutoffRecord(
            selection="physical",
            window_internal=w_int,
            window_days=w_days,
            requested_tol=float(eps_t) if eps_t is not None else None,
            design_mean_lag_days=(
                float(design_mean_lag_days)
                if design_mean_lag_days is not None else None),
            design_beta_internal=design_beta_internal,
            omitted_mass_at_design=omitted,
        )
    elif window_internal is not None:
        w_int = float(window_internal)
        if not (math.isfinite(w_int) and w_int > 0):
            raise ValueError(
                f"window must be finite and > 0; got {w_int}")
        omitted = (
            temporal_omitted_mass(w_int, design_beta_internal)
            if design_beta_internal is not None else None)
        temporal = TemporalCutoffRecord(
            selection="physical",
            window_internal=w_int,
            window_days=internal_to_days(w_int, horizon_days),
            requested_tol=float(eps_t) if eps_t is not None else None,
            design_mean_lag_days=(
                float(design_mean_lag_days)
                if design_mean_lag_days is not None else None),
            design_beta_internal=design_beta_internal,
            omitted_mass_at_design=omitted,
        )
    elif eps_t is not None:
        if design_mean_lag_days is None or design_beta_internal is None:
            raise ValueError(
                "temporal_cutoff_tol requires design_mean_lag_days "
                "(cutoffs are fixed at construction; beta is latent).")
        w_int = temporal_cutoff_from_tol(float(eps_t), design_beta_internal)
        temporal = TemporalCutoffRecord(
            selection="tolerance",
            window_internal=w_int,
            window_days=internal_to_days(w_int, horizon_days),
            requested_tol=float(eps_t),
            design_mean_lag_days=float(design_mean_lag_days),
            design_beta_internal=design_beta_internal,
            omitted_mass_at_design=float(eps_t),
        )
    else:
        temporal = TemporalCutoffRecord(
            selection="default_untruncated",
            window_internal=float(T_INTERNAL),
            window_days=float(horizon_days),
            requested_tol=None,
            design_mean_lag_days=(
                float(design_mean_lag_days)
                if design_mean_lag_days is not None else None),
            design_beta_internal=design_beta_internal,
            omitted_mass_at_design=None,
        )

    # ---- spatial ----
    if spatial_window is not None:
        ws = float(spatial_window)
        if not (math.isfinite(ws) and ws > 0):
            raise ValueError(
                f"spatial_window must be finite and > 0; got {ws}")
        omitted = (
            spatial_omitted_mass(ws, float(design_sigma))
            if design_sigma is not None else None)
        spatial = SpatialCutoffRecord(
            selection="physical",
            spatial_window=ws,
            requested_tol=float(eps_s) if eps_s is not None else None,
            design_sigma=float(design_sigma) if design_sigma is not None else None,
            omitted_mass_at_design=omitted,
        )
    elif eps_s is not None:
        if design_sigma is None:
            raise ValueError(
                "spatial_cutoff_tol requires design_sigma "
                "(cutoffs are fixed at construction; sigma is latent).")
        ws = spatial_cutoff_from_tol(float(eps_s), float(design_sigma))
        spatial = SpatialCutoffRecord(
            selection="tolerance",
            spatial_window=ws,
            requested_tol=float(eps_s),
            design_sigma=float(design_sigma),
            omitted_mass_at_design=float(eps_s),
        )
    else:
        spatial = SpatialCutoffRecord(
            selection="default_untruncated",
            spatial_window=None,
            requested_tol=None,
            design_sigma=float(design_sigma) if design_sigma is not None else None,
            omitted_mass_at_design=None,
        )

    return CutoffProvenance(temporal=temporal, spatial=spatial)
