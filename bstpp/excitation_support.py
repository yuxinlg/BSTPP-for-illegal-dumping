"""Excitation support mode, sigma bounds, and prior truncation (Phase 3d).

One support object feeds both the offspring parenting predicate and the
excitation compensator charge region (D-18). Polygon mode uses the hybrid
Hermite mass table; rectangle mode retains the closed-form erf product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.truncated import (
    LeftTruncatedDistribution,
    RightTruncatedDistribution,
    TwoSidedTruncatedDistribution,
)
from jax.scipy.special import log_ndtr
from numpyro.distributions.util import promote_shapes, validate_sample
from shapely.geometry import Point
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

from .config import (
    NumericalConfig,
    raise_polygon_min_sigma_violation,
    raise_rectangle_bounds_violation,
    raise_support_mode_violation,
    validate_sigma_pair,
)
from .polygon_mass import (
    BUDGET_REFERENCE_GL_ORDER,
    BUDGET_REFERENCE_ORACLE_BOUND,
    PRODUCTION_TAU_ABS,
    PolygonMassTable,
    validate_polygon_mass_table,
)

DEFAULT_MAX_SIGMA_KM = 5.0
ExcitationSupportMode = Literal["rectangle", "polygon"]

_TRUNCATED_NORMAL_TYPES = (
    TwoSidedTruncatedDistribution,
    LeftTruncatedDistribution,
    RightTruncatedDistribution,
)


def metres_to_crs_units(metres: float, crs) -> float:
    """Convert a length in metres into the CRS's axis units.

    ``crs.axis_info[0].unit_conversion_factor`` is metres per CRS unit
    (1.0 for metre, ~0.3048 for US survey foot). Does not assume the CRS
    unit is a metre.
    """
    if crs is None:
        raise ValueError(
            "Cannot convert metres to CRS units without a CRS; pass an "
            "explicit length in domain-coordinate units instead.")
    if getattr(crs, "is_geographic", False):
        raise ValueError(
            "Cannot convert metres into a geographic CRS (lon/lat). "
            "Project to a metric CRS or pass an explicit length in "
            "domain-coordinate units.")
    info = getattr(crs, "axis_info", None)
    if not info:
        raise ValueError(f"CRS {crs!r} has no axis_info for unit conversion")
    factor = float(info[0].unit_conversion_factor)
    if not (np.isfinite(factor) and factor > 0):
        raise ValueError(
            f"CRS unit_conversion_factor must be finite and positive; got {factor}")
    return float(metres) / factor


def resolve_excitation_support_mode(
    *,
    is_polygon_domain: bool,
    excitation_support: Optional[str],
) -> ExcitationSupportMode:
    """OP-2: nonrectangular domains require an explicit mode choice."""
    if excitation_support is not None:
        if excitation_support not in ("rectangle", "polygon"):
            raise_support_mode_violation(
                support_mode=excitation_support,
                remediation=(
                    "Pass excitation_support='rectangle' or "
                    "excitation_support='polygon'."))
        return excitation_support  # type: ignore[return-value]
    if is_polygon_domain:
        raise ValueError(
            "Nonrectangular domain requires an explicit excitation_support "
            "('rectangle' or 'polygon'). Silent defaults are not allowed "
            "(Phase 3d / OP-2).")
    return "rectangle"


def resolve_sigma_bounds(
    *,
    mode: ExcitationSupportMode,
    min_sigma: Optional[float],
    max_sigma: Optional[float],
    crs,
) -> tuple[Optional[float], Optional[float], dict]:
    """Resolve (min_sigma, max_sigma_real) in domain CRS units.

    Rectangle mode: omitted bounds leave the prior unchanged (both None).
    Polygon mode: min_sigma required (explicit, finite, positive, domain
    units); max_sigma defaults to 5 km via projected CRS conversion when
    omitted.

    This function owns the two ARGUMENT invariants (rectangle both-or-neither,
    polygon-requires-``min_sigma``): it is the last place that still knows
    which argument the caller omitted. It renders the canonical clauses from
    ``config`` rather than restating them, and delegates the resolved-bound
    invariants to ``config.validate_sigma_pair`` (D-40 σ/mode families).
    """
    meta: dict[str, Any] = {
        "min_sigma_user": min_sigma,
        "max_sigma_user": max_sigma,
        "max_sigma_default_km": DEFAULT_MAX_SIGMA_KM,
    }

    # Mode is validated here rather than assumed. The `else` below is the
    # polygon branch: without this, an invalid mode was silently treated as
    # polygon and resolution proceeded. Unreachable from the model paths,
    # which gate on resolve_excitation_support_mode first; direct callers of
    # this function had no such gate.
    if mode not in ("rectangle", "polygon"):
        raise_support_mode_violation(support_mode=mode)

    if mode == "rectangle":
        if min_sigma is None and max_sigma is None:
            meta["bounds_active"] = False
            return None, None, meta
        if min_sigma is None or max_sigma is None:
            raise_rectangle_bounds_violation(
                min_sigma=min_sigma, max_sigma=max_sigma)
        lo, hi = float(min_sigma), float(max_sigma)
        validate_sigma_pair(lo, hi)
        meta.update(bounds_active=True, min_sigma=lo, max_sigma_real=hi,
                    max_sigma_units="domain", max_sigma_source="user")
        return lo, hi, meta

    # polygon mode
    if min_sigma is None:
        raise_polygon_min_sigma_violation()
    lo = float(min_sigma)
    if max_sigma is None:
        hi = metres_to_crs_units(DEFAULT_MAX_SIGMA_KM * 1000.0, crs)
        meta["max_sigma_source"] = "default_5km"
        meta["max_sigma_units"] = (
            None if crs is None else str(crs.axis_info[0].unit_name))
    else:
        hi = float(max_sigma)
        meta["max_sigma_source"] = "user"
        meta["max_sigma_units"] = "domain"
    validate_sigma_pair(lo, hi)
    meta.update(bounds_active=True, min_sigma=lo, max_sigma_real=hi)
    return lo, hi, meta


# ------------------------------------------------------------------ priors --
class TruncatedLogNormal(Distribution):
    """LogNormal truncated to a closed interval on the value scale.

    ``support`` is ``interval(low, high)`` so NUTS/SVI bijectors cannot
    propose outside the interval (no clipping, no NaNs).
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "low": constraints.positive,
        "high": constraints.positive,
    }
    reparametrized_params = ["loc", "scale", "low", "high"]

    def __init__(self, loc, scale, low, high, *, validate_args=None):
        self.loc, self.scale, self.low, self.high = promote_shapes(
            loc, scale, low, high)
        self._base = dist.LogNormal(self.loc, self.scale)
        batch_shape = jnp.shape(self.loc)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)
        if self._validate_args and jnp.any(self.low >= self.high):
            raise ValueError(
                "TruncatedLogNormal requires low < high; "
                f"got low={self.low}, high={self.high}")

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def _log_z(self):
        """log(F(high) - F(low)) in dtype-safe log space on the log-value axis.

        LogNormal CDF mass equals Normal CDF mass on ``log(bounds)``. Prefer
        the survival representation when the interval sits in the right half
        so float32 does not saturate both CDFs to 1 and yield ``log(0)``.
        Replaces the float32-inert ``clip(..., 1e-300)`` guard.
        """
        # Standardized Normal bounds for Z = log(X), X ~ LogNormal(loc, scale).
        a = (jnp.log(self.high) - self.loc) / self.scale
        b = (jnp.log(self.low) - self.loc) / self.scale
        # Φ(a) - Φ(b) = Φ̄(b) - Φ̄(a); use SF when the interval is right of 0.
        use_sf = (0.5 * (a + b)) > 0
        log_p_hi = jnp.where(use_sf, log_ndtr(-b), log_ndtr(a))
        log_p_lo = jnp.where(use_sf, log_ndtr(-a), log_ndtr(b))
        # log(p_hi - p_lo) = log_p_hi + log(1 - exp(log_p_lo - log_p_hi)).
        # Near-zero mass differences need log(-expm1(x)); plain log1p(-exp(x))
        # loses precision when the truncated interval is narrow.
        x = log_p_lo - log_p_hi  # <= 0 when the interval has positive mass
        log_one_m_exp = jnp.where(
            x > -jnp.log(2.0),
            jnp.log(-jnp.expm1(x)),
            jnp.log1p(-jnp.exp(x)),
        )
        return log_p_hi + log_one_m_exp

    def sample(self, key, sample_shape=()):
        # Truncate on the log scale, then map back to the declared positive
        # interval [low, high] (samplers that return log-space draws disagree
        # with support / log_prob).
        tn = dist.TruncatedNormal(
            self.loc, self.scale,
            low=jnp.log(self.low), high=jnp.log(self.high))
        return jnp.exp(tn.sample(key, sample_shape))

    @validate_sample
    def log_prob(self, value):
        return self._base.log_prob(value) - self._log_z()

    def tree_flatten(self):
        return (self.loc, self.scale, self.low, self.high), None

    @classmethod
    def tree_unflatten(cls, aux, params):
        return cls(*params)


def truncate_sigmax_2_prior(
    prior: Distribution,
    min_sigma: float,
    max_sigma: float,
) -> Distribution:
    """Exact truncation of named prior types onto [min_sigma^2, max_sigma^2].

    Supported: HalfNormal, TruncatedNormal (Normal base only; one- or
    two-sided wrappers), LogNormal, TruncatedLogNormal.
    Pre-truncated wrappers whose base_dist is not Normal are rejected.
    Other bases raise a clear unsupported-prior error.
    """
    low = float(min_sigma) ** 2
    high = float(max_sigma) ** 2
    if not (high > low > 0):
        raise ValueError(
            f"invalid truncation interval on sigmax_2: [{low}, {high}]")

    if isinstance(prior, TruncatedLogNormal):
        new_low = jnp.maximum(prior.low, low)
        new_high = jnp.minimum(prior.high, high)
        if jnp.any(new_high <= new_low):
            raise ValueError(
                "sigmax_2 TruncatedLogNormal support does not overlap "
                f"[{low}, {high}]")
        return TruncatedLogNormal(prior.loc, prior.scale, new_low, new_high)

    if isinstance(prior, dist.LogNormal):
        return TruncatedLogNormal(prior.loc, prior.scale, low, high)

    if isinstance(prior, dist.HalfNormal):
        return dist.TruncatedNormal(0.0, prior.scale, low=low, high=high)

    if isinstance(prior, _TRUNCATED_NORMAL_TYPES):
        # Generic truncated wrappers also wrap Cauchy / StudentT / etc.
        # Accept this branch only for a genuine Normal base -- never infer
        # Normality from loc/scale alone (CF of the declared TruncatedNormal
        # adapter; imposing truncation on a prior remains SC / D-28).
        base = prior.base_dist
        if not isinstance(base, dist.Normal):
            raise TypeError(
                "sigmax_2 prior truncation accepts a pre-truncated wrapper "
                "only when its base_dist is numpyro.distributions.Normal "
                f"(got base_dist={type(base).__name__}). "
                "Truncated Cauchy, Student-t, and other non-Normal families "
                "are not silently converted to TruncatedNormal. Supported "
                "families remain HalfNormal, LogNormal, TruncatedNormal, and "
                "TruncatedLogNormal.")
        # Pinned NumPyro one-sided wrappers omit the unused bound attribute:
        # LeftTruncatedDistribution has .low only; RightTruncatedDistribution
        # has .high only. Treat a missing bound as +/- infinity.
        prior_low = prior.low if hasattr(prior, "low") else -jnp.inf
        prior_high = prior.high if hasattr(prior, "high") else jnp.inf
        new_low = jnp.maximum(prior_low, low)
        new_high = jnp.minimum(prior_high, high)
        # Broadcast bounds to the Normal base batch shape (scalar requested
        # bounds vs already-batched wrapper lows/highs).
        new_low, new_high, loc, scale = promote_shapes(
            new_low, new_high, base.loc, base.scale)
        if (not jnp.all(jnp.isfinite(new_low) & jnp.isfinite(new_high))
                or jnp.any(new_high <= new_low)):
            raise ValueError(
                "sigmax_2 TruncatedNormal support does not overlap "
                f"[{low}, {high}]")
        return dist.TruncatedNormal(
            loc, scale, low=new_low, high=new_high)

    raise TypeError(
        "sigmax_2 prior truncation supports HalfNormal, LogNormal, "
        "TruncatedNormal, and TruncatedLogNormal only (exact normalization, "
        "sampling, interval support, batching, and JIT). "
        f"Got {type(prior).__name__}. Supply one of the supported types, or "
        "request an explicitly tested extension.")


# ----------------------------------------------------------- support object --
@dataclass
class ExcitationSupport:
    """Single object for parenting eligibility and compensator charge (D-18)."""
    mode: ExcitationSupportMode
    bounds: np.ndarray
    domain_geom: Any
    spatial_window: float | None
    min_sigma: float | None
    max_sigma_real: float | None
    mass_table: PolygonMassTable | None = None
    provenance: dict = field(default_factory=dict)

    def candidate_in_support(self, x: float, y: float) -> bool:
        """Whether an offspring at real (x, y) may become a parent."""
        if self.mode == "rectangle":
            A_ = self.bounds
            return bool(A_[0, 0] <= x <= A_[0, 1]
                        and A_[1, 0] <= y <= A_[1, 1])
        return bool(self.domain_geom.covers(Point(x, y)))


def domain_polygon_geometry(domain_gdf) -> Any:
    """Unary union of a GeoDataFrame domain as a shapely geometry.

    Prefer ``PreparedDomain.union_geometry`` at call sites that already have
    a prepared domain; this helper remains for characterization and for
    callers that only hold a raw GeoDataFrame.
    """
    return unary_union(list(domain_gdf.geometry.values))


def build_excitation_support(
    *,
    mode: ExcitationSupportMode,
    bounds: np.ndarray,
    domain_gdf,
    is_polygon_domain: bool,
    crs,
    spatial_window: float | None,
    min_sigma: Optional[float],
    max_sigma: Optional[float],
    event_x_real: np.ndarray,
    event_y_real: np.ndarray,
    mass_table: PolygonMassTable | None = None,
    union_geometry: Any | None = None,
    numerical_config: NumericalConfig | None = None,
) -> ExcitationSupport:
    """Construct the support object; validate a supplied table in polygon mode.

    Polygon Hermite tables are prepared explicitly with
    ``bstpp.polygon_mass.prepare_polygon_mass_table`` and supplied here for
    compatibility validation and installation — this function never builds
    tables. For polygon domains, ``union_geometry`` must be the canonical
    ``PreparedDomain.union_geometry`` (no independent unary_union here).

    Table ``h_panel`` / ``gl_order`` are authoritative. Acceptance is the
    measured residual against ``production_tau_abs`` from ``numerical_config``
    when supplied (else module ``PRODUCTION_TAU_ABS``), plus the panel-ratio
    prefilter. Build settings are chosen only at
    ``prepare_polygon_mass_table``; this install path has no ``panel_h_m`` /
    ``gl_order`` parameters.
    """
    if mode not in ("rectangle", "polygon"):
        raise_support_mode_violation(support_mode=mode)

    lo, hi, bound_meta = resolve_sigma_bounds(
        mode=mode, min_sigma=min_sigma, max_sigma=max_sigma, crs=crs)

    if is_polygon_domain:
        if union_geometry is None:
            raise ValueError(
                "union_geometry is required for polygon domains; pass "
                "PreparedDomain.union_geometry")
        geom = union_geometry
    else:
        A_ = np.asarray(bounds, dtype=float)
        geom = shapely_box(A_[0, 0], A_[1, 0], A_[0, 1], A_[1, 1])

    table = mass_table
    builder_meta: dict[str, Any] = {}
    budget_ratio: float | None = None
    measured_residual: float | None = None
    if mode == "polygon":
        assert lo is not None and hi is not None
        if table is None:
            raise ValueError(
                "Polygon excitation_support requires a prepared mass_table "
                "from bstpp.polygon_mass.prepare_polygon_mass_table(...); "
                "silent Hermite table construction is not allowed.")
        budget_ratio, measured_residual = validate_polygon_mass_table(
            table,
            domain_geom=geom,
            event_x_real=event_x_real,
            event_y_real=event_y_real,
            spatial_window=spatial_window,
            sigma_min=lo,
            sigma_max=hi,
            numerical_config=numerical_config,
        )
        # Ownership: copy caller-supplied tables once at install. Reuse the
        # same owned object on temporal-only set_window rebuilds.
        if not getattr(table, "_owned", False):
            table = table.copy()
        builder_meta = dict(table.provenance)

    if numerical_config is not None:
        ref_gl = int(numerical_config.budget_reference_gl_order)
        oracle_bound = float(numerical_config.budget_reference_oracle_bound)
        tau_abs = float(numerical_config.production_tau_abs)
    else:
        ref_gl = int(BUDGET_REFERENCE_GL_ORDER)
        oracle_bound = float(BUDGET_REFERENCE_ORACLE_BOUND)
        tau_abs = float(PRODUCTION_TAU_ABS)

    prov = {
        "excitation_support": mode,
        "spatial_window": spatial_window,
        **bound_meta,
        "builder": builder_meta,
        "table_id": None if table is None else table.provenance.get("table_id"),
        "geometry_sha256": None if table is None else table.geometry_sha256,
        "n_knots": None if table is None else table.n_knots,
        "knot_sigma_min": None if table is None else table.sigma_min,
        "knot_sigma_max": None if table is None else table.sigma_max,
        "table_h_panel": None if table is None else float(table.h_panel),
        "table_gl_order": None if table is None else int(table.gl_order),
        "panel_min_sigma_ratio": budget_ratio,
        "measured_max_abs_residual": measured_residual,
        "BUDGET_REFERENCE_GL_ORDER": (
            None if table is None else ref_gl),
        "BUDGET_REFERENCE_ORACLE_BOUND": (
            None if table is None else oracle_bound),
        "PRODUCTION_TAU_ABS": (
            None if table is None else tau_abs),
    }
    return ExcitationSupport(
        mode=mode,
        bounds=np.asarray(bounds, dtype=float),
        domain_geom=geom,
        spatial_window=spatial_window,
        min_sigma=lo,
        max_sigma_real=hi,
        mass_table=table,
        provenance=prov,
    )
