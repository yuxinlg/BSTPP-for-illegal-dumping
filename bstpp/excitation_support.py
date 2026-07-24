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
from numpyro.distributions.util import promote_shapes, validate_sample
from shapely.geometry import Point
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

from .polygon_mass import (
    DEFAULT_GL_ORDER,
    DEFAULT_PANEL_H_M,
    PolygonMassTable,
    build_quad_table,
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
            raise ValueError(
                "excitation_support must be 'rectangle' or 'polygon', "
                f"got {excitation_support!r}")
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
    """
    meta: dict[str, Any] = {
        "min_sigma_user": min_sigma,
        "max_sigma_user": max_sigma,
        "max_sigma_default_km": DEFAULT_MAX_SIGMA_KM,
    }

    if mode == "rectangle":
        if min_sigma is None and max_sigma is None:
            meta["bounds_active"] = False
            return None, None, meta
        if min_sigma is None or max_sigma is None:
            raise ValueError(
                "In rectangle mode, min_sigma and max_sigma must both be "
                "supplied or both omitted (omitted leaves the sigmax_2 "
                "prior unchanged).")
        lo, hi = float(min_sigma), float(max_sigma)
        _validate_sigma_pair(lo, hi)
        meta.update(bounds_active=True, min_sigma=lo, max_sigma_real=hi,
                    max_sigma_units="domain", max_sigma_source="user")
        return lo, hi, meta

    # polygon mode
    if min_sigma is None:
        raise ValueError(
            "Polygon excitation_support requires an explicit finite positive "
            "min_sigma in domain-coordinate units (no default).")
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
    _validate_sigma_pair(lo, hi)
    meta.update(bounds_active=True, min_sigma=lo, max_sigma_real=hi)
    return lo, hi, meta


def _validate_sigma_pair(lo: float, hi: float) -> None:
    if not (np.isfinite(lo) and lo > 0):
        raise ValueError(
            f"min_sigma must be finite and positive; got {lo}")
    if not (np.isfinite(hi) and hi > lo):
        raise ValueError(
            f"require min_sigma < max_sigma; got min_sigma={lo}, max_sigma={hi}")


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

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def _log_z(self):
        return jnp.log(
            jnp.clip(self._base.cdf(self.high) - self._base.cdf(self.low),
                     a_min=1e-300))

    def sample(self, key, sample_shape=()):
        tn = dist.TruncatedNormal(
            self.loc, self.scale,
            low=jnp.log(self.low), high=jnp.log(self.high))
        return tn.sample(key, sample_shape)

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

    Supported: HalfNormal, TruncatedNormal, LogNormal, TruncatedLogNormal.
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
        new_low = jnp.maximum(prior.low, low)
        new_high = jnp.minimum(prior.high, high)
        if jnp.any(new_high <= new_low):
            raise ValueError(
                "sigmax_2 TruncatedNormal support does not overlap "
                f"[{low}, {high}]")
        return dist.TruncatedNormal(
            prior.loc, prior.scale, low=new_low, high=new_high)

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
    panel_h_m: float = DEFAULT_PANEL_H_M,
    gl_order: int = DEFAULT_GL_ORDER,
    mass_table: PolygonMassTable | None = None,
    union_geometry: Any | None = None,
) -> ExcitationSupport:
    """Construct the support object; build Hermite tables in polygon mode.

    For polygon domains, ``union_geometry`` must be the canonical
    ``PreparedDomain.union_geometry`` (no independent unary_union here).
    """
    if mode not in ("rectangle", "polygon"):
        raise ValueError(
            f"excitation_support must be 'rectangle' or 'polygon', got {mode!r}")

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
    if mode == "polygon":
        assert lo is not None and hi is not None
        if crs is not None and not getattr(crs, "is_geographic", False):
            h_panel = metres_to_crs_units(panel_h_m, crs)
        else:
            # CRS-less synthetic domains: treat panel_h_m as domain units
            h_panel = float(panel_h_m)
        if table is None:
            table = build_quad_table(
                geom, event_x_real, event_y_real, lo, hi,
                ws=spatial_window, h_panel=h_panel, gl_order=gl_order,
                extra_provenance={
                    "excitation_support": mode,
                    "min_sigma": lo,
                    "max_sigma_real": hi,
                    **{k: v for k, v in bound_meta.items()
                       if str(k).startswith("max_sigma")},
                },
            )
        else:
            validate_polygon_mass_table(
                table,
                domain_geom=geom,
                event_x_real=event_x_real,
                event_y_real=event_y_real,
                spatial_window=spatial_window,
                sigma_min=lo,
                sigma_max=hi,
                h_panel=h_panel,
                gl_order=gl_order,
            )
        builder_meta = dict(table.provenance)

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
