"""Frozen configuration objects for Phase 3f (behind the args adapter).

WP1 introduces ``NumericalConfig`` only. Objects are ``@dataclass(frozen=True)``
with all validation in ``__post_init__``, constructed through a single factory
per type (A-23; D-35). No Pydantic dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NoReturn, Optional

from .cutoffs import DEFAULT_SPATIAL_TOL, DEFAULT_TEMPORAL_TOL
from .polygon_mass import (
    BUDGET_REFERENCE_GL_ORDER,
    BUDGET_REFERENCE_ORACLE_BOUND,
    DEFAULT_GL_ORDER,
    DEFAULT_PANEL_H_M,
    MAX_PANEL_TO_MIN_SIGMA_RATIO,
    PRODUCTION_TAU_ABS,
)

ExcitationSupportMode = Literal["rectangle", "polygon"]


class NumericalConfigError(ValueError):
    """Named error identifying a violated ``NumericalConfig`` constraint.

    Subclasses ``ValueError``, so callers catching ``ValueError`` are
    unaffected by an invariant migrating to this identity (D-40).
    """


# --------------------------------------------------- D-40 single sourcing --
# One invariant, one error identity, one canonical clause, independent of entry
# path. A site may append its own remediation clause; it may not restate the
# invariant. Raised message text is ASCII (D-40): a non-ASCII character can
# raise UnicodeEncodeError while the traceback carrying it is printed to a
# cp1252 console, i.e. an error path that fails while failing.

def panel_ratio_invariant_clause(
    *,
    panel_h_m: float,
    min_sigma: float,
    ratio_ceil: float,
    tau_abs: float,
) -> str:
    """Render the canonical panel/``min_sigma`` invariant clause (D-40).

    Every site enforcing ``panel_h_m / min_sigma <= ratio_ceil`` renders this
    exact text for a given violation, byte for byte, whichever entry path
    reached it. This is the string tests match on.
    """
    h = float(panel_h_m)
    s = float(min_sigma)
    return (
        "panel_h_m / min_sigma exceeds max_panel_to_min_sigma_ratio: "
        f"panel_h_m={h}, min_sigma={s}, ratio={h / s}, "
        f"max_panel_to_min_sigma_ratio={float(ratio_ceil)}. This resolution "
        "prefilter is necessary but not sufficient for "
        f"PRODUCTION_TAU_ABS={float(tau_abs)}, which is enforced by a measured "
        "residual at mass-table install."
    )


def raise_panel_ratio_violation(
    *,
    panel_h_m: float,
    min_sigma: float,
    ratio_ceil: float,
    tau_abs: float,
    remediation: str = "",
) -> NoReturn:
    """Raise the single error identity for the panel/``min_sigma`` invariant.

    ``remediation`` is appended after the canonical clause and is the only
    part a call site may vary — build time and install time are different
    situations with different useful advice (D-40).
    """
    msg = panel_ratio_invariant_clause(
        panel_h_m=panel_h_m, min_sigma=min_sigma,
        ratio_ceil=ratio_ceil, tau_abs=tau_abs)
    if remediation:
        msg = f"{msg} {remediation}"
    raise NumericalConfigError(msg)


# ------------------------------------- D-40 \supsd: the sigma/mode families --
# D-40 requires one invariant, one identity, one canonical clause. The five
# sigma/mode invariants have more than one owner, and the reason is that they
# attach to DIFFERENT QUANTITIES:
#
#   * rectangle both-or-neither and polygon-requires-min_sigma are ARGUMENT
#     invariants -- they test which argument was omitted. Defaulting destroys
#     that distinction: resolved rectangle bounds are both-None or both-float
#     by construction, and a resolved polygon min_sigma is never None. Only
#     resolve_sigma_bounds, which still sees the user's arguments, can express
#     them. A resolved-bound validator cannot.
#   * min_sigma positivity and min_sigma < max_sigma are RESOLVED-BOUND
#     invariants. min_sigma < max_sigma is only meaningful after defaulting:
#     in polygon mode with max_sigma omitted there is no user-supplied pair to
#     compare. NumericalConfig owns these and validates the right quantity.
#   * support-mode validity is a MODE invariant, upstream of both.
#   * builder-requires-max_sigma (CI-6, A-28) is a BUILDER-ARGUMENT invariant.
#     It reads like CI-2 and is not: the resolver DEFAULTS an omitted polygon
#     max_sigma and NumericalConfig accepts None for it, so the requirement is
#     the mass-table builder's alone, where max_sigma is the top knot of the
#     log-sigma grid rather than a prior bound. Same test as always -- a
#     different quantity is a different invariant.
#
# So there is more than one owner, but exactly one identity and one canonical
# clause per invariant, produced here and rendered byte for byte wherever the
# violation is detected. A site may append its own remediation; it may not
# restate the invariant. All clause text is ASCII (D-40).

def rectangle_bounds_invariant_clause(
    *, min_sigma: object, max_sigma: object) -> str:
    """Render the canonical rectangle both-or-neither clause (CI-1)."""
    return (
        "support-mode compatibility (rectangle): min_sigma and max_sigma must "
        f"both be supplied or both omitted; got min_sigma={min_sigma!r}, "
        f"max_sigma={max_sigma!r}. Omitting both leaves the sigmax_2 prior "
        "unchanged.")


def raise_rectangle_bounds_violation(
    *, min_sigma: object, max_sigma: object, remediation: str = "",
) -> NoReturn:
    """Raise the single identity for the rectangle both-or-neither invariant."""
    msg = rectangle_bounds_invariant_clause(
        min_sigma=min_sigma, max_sigma=max_sigma)
    if remediation:
        msg = f"{msg} {remediation}"
    raise NumericalConfigError(msg)


def polygon_min_sigma_invariant_clause() -> str:
    """Render the canonical polygon-requires-min_sigma clause (CI-2)."""
    return (
        "support-mode compatibility (polygon): min_sigma is required and has "
        "no default; supply an explicit finite positive min_sigma in "
        "domain-coordinate units.")


def raise_polygon_min_sigma_violation(*, remediation: str = "") -> NoReturn:
    """Raise the single identity for the polygon-requires-min_sigma invariant."""
    msg = polygon_min_sigma_invariant_clause()
    if remediation:
        msg = f"{msg} {remediation}"
    raise NumericalConfigError(msg)


def builder_max_sigma_invariant_clause() -> str:
    """Render the canonical builder-requires-``max_sigma`` clause (CI-6).

    CI-6 is a DISTINCT invariant, not a member of the CI-2 family, and the reason
    is the D-40 sigma/mode test above: owners of one invariant validate the
    same QUANTITY. CI-2's quantity is ``min_sigma``; this one's is ``max_sigma``.
    They are also not the same claim. CI-2 holds package-wide -- nothing
    anywhere defaults ``min_sigma``. "Polygon requires ``max_sigma``" is FALSE
    at the model boundary: ``resolve_sigma_bounds`` defaults an omitted polygon
    ``max_sigma`` to ``DEFAULT_MAX_SIGMA_KM`` through the projected CRS, and
    ``NumericalConfig`` accepts ``max_sigma=None`` outright (the asymmetry
    A-27 froze). Rendering CI-2's clause here would assert a package-wide
    requirement that does not exist.

    The requirement is the BUILDER's alone, because ``max_sigma`` there is not
    a prior bound but the top knot of the table's log-sigma grid
    (``log_knots(sigma_min, sigma_max)``), and the table prohibits
    extrapolation past it. The builder cannot borrow the resolver's default
    either: that default needs a projected CRS, and ``crs`` is optional at the
    builder.
    """
    return (
        "mass-table build range (polygon): max_sigma is required by the "
        "mass-table builder and has no default there; supply an explicit "
        "finite max_sigma in domain-coordinate units.")


def raise_builder_max_sigma_violation(*, remediation: str = "") -> NoReturn:
    """Raise the single identity for the builder-requires-max_sigma invariant."""
    msg = builder_max_sigma_invariant_clause()
    if remediation:
        msg = f"{msg} {remediation}"
    raise NumericalConfigError(msg)


def min_sigma_positive_invariant_clause(*, min_sigma: float) -> str:
    """Render the canonical min_sigma positivity clause (CI-3)."""
    return f"min_sigma must be finite and positive; got {float(min_sigma)}"


def raise_min_sigma_positive_violation(
    *, min_sigma: float, remediation: str = "") -> NoReturn:
    """Raise the single identity for the min_sigma positivity invariant."""
    msg = min_sigma_positive_invariant_clause(min_sigma=min_sigma)
    if remediation:
        msg = f"{msg} {remediation}"
    raise NumericalConfigError(msg)


def sigma_order_invariant_clause(
    *, min_sigma: float, max_sigma: float) -> str:
    """Render the canonical sigma-bound ordering clause (CI-4)."""
    return (
        "sigma-bound coherence requires min_sigma < max_sigma; got "
        f"min_sigma={float(min_sigma)}, max_sigma={float(max_sigma)}")


def raise_sigma_order_violation(
    *, min_sigma: float, max_sigma: float, remediation: str = "") -> NoReturn:
    """Raise the single identity for the sigma-bound ordering invariant."""
    msg = sigma_order_invariant_clause(
        min_sigma=min_sigma, max_sigma=max_sigma)
    if remediation:
        msg = f"{msg} {remediation}"
    raise NumericalConfigError(msg)


def support_mode_invariant_clause(*, support_mode: object) -> str:
    """Render the canonical excitation-support-mode clause (CI-5)."""
    return (
        "excitation support mode must be 'rectangle' or 'polygon'; got "
        f"{support_mode!r}")


def raise_support_mode_violation(
    *, support_mode: object, remediation: str = "") -> NoReturn:
    """Raise the single identity for the support-mode validity invariant."""
    msg = support_mode_invariant_clause(support_mode=support_mode)
    if remediation:
        msg = f"{msg} {remediation}"
    raise NumericalConfigError(msg)


def validate_sigma_pair(min_sigma: float, max_sigma: float) -> None:
    """The single implementation of the resolved-bound invariants (CI-3, CI-4).

    Every site that checks a resolved sigma pair calls this, so the predicate
    and the identity have one spelling. Callers that must coerce their inputs
    do so before calling; this function does not coerce, because argument-type
    discipline is A-23's invariant and not one of these two (see OP-20).
    """
    if not (math.isfinite(min_sigma) and min_sigma > 0.0):
        raise_min_sigma_positive_violation(min_sigma=min_sigma)
    if not (math.isfinite(max_sigma) and max_sigma > min_sigma):
        raise_sigma_order_violation(min_sigma=min_sigma, max_sigma=max_sigma)


def _require_int(name: str, value: object) -> int:
    """Reject bool and non-ints; ``bool`` is an ``int`` subclass."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise NumericalConfigError(
            f"{name} must be an int (bool rejected); got {value!r} "
            f"({type(value).__name__})")
    return value


def _require_real(name: str, value: object) -> float:
    """Reject bool, str, and non-numeric types; accept only int/float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericalConfigError(
            f"{name} must be a real number (int or float; bool and str "
            f"rejected); got {value!r} ({type(value).__name__})")
    return float(value)


@dataclass(frozen=True)
class NumericalConfig:
    """Computational / numerical settings (OP-13 / A-21; A-23 frozen dataclass).

    Holds mass-table builder settings, the Commit~C measured-budget policy, and
    package cutoff tolerance defaults. Support mode and σ bounds are accepted
    for coherence validation (model commitments remain owned elsewhere).

    ``min_sigma`` / ``max_sigma`` are the **resolved** bounds, not the bounds
    the user supplied. Every production caller passes the output of
    ``excitation_support.resolve_sigma_bounds``: in polygon mode with
    ``max_sigma`` omitted, this object stores the defaulted 5 km value in CRS
    units, never ``None`` (``main.py`` construction and ``set_window``).
    The distinction is load-bearing — the argument invariants (rectangle
    both-or-neither, polygon-requires-``min_sigma``) cannot be evaluated here,
    because defaulting has already erased which argument was omitted. See the
    D-40 σ/mode family note above. An earlier version of this docstring claimed
    these fields were user-supplied; it was false, and A-27 records it.

    Immutable after construction. Construct only via :meth:`create`.
    """

    panel_h_m: float
    gl_order: int
    default_temporal_tol: float
    default_spatial_tol: float
    production_tau_abs: float
    budget_reference_gl_order: int
    budget_reference_oracle_bound: float
    max_panel_to_min_sigma_ratio: float
    support_mode: ExcitationSupportMode
    min_sigma: Optional[float]
    max_sigma: Optional[float]

    def __post_init__(self) -> None:
        mode = self.support_mode
        if mode not in ("rectangle", "polygon"):
            raise_support_mode_violation(support_mode=mode)

        # Type discipline before any numeric use (WP1.4a / A-23 reason 3).
        h = _require_real("panel_h_m", self.panel_h_m)
        if not (math.isfinite(h) and h > 0.0):
            raise NumericalConfigError(
                f"panel_h_m must be finite and > 0; got {self.panel_h_m!r}")

        gl = _require_int("gl_order", self.gl_order)
        if gl < 1:
            raise NumericalConfigError(
                f"gl_order must be an integer >= 1; got {self.gl_order!r}")

        tau = _require_real("production_tau_abs", self.production_tau_abs)
        ref_gl = _require_int(
            "budget_reference_gl_order", self.budget_reference_gl_order)
        oracle = _require_real(
            "budget_reference_oracle_bound", self.budget_reference_oracle_bound)
        ratio_ceil = _require_real(
            "max_panel_to_min_sigma_ratio", self.max_panel_to_min_sigma_ratio)

        # Commit C measured-budget policy is frozen into the object (D-35).
        if tau != float(PRODUCTION_TAU_ABS):
            raise NumericalConfigError(
                "production_tau_abs must equal PRODUCTION_TAU_ABS="
                f"{PRODUCTION_TAU_ABS} (A-21 / Commit C measured budget); "
                f"got {self.production_tau_abs!r}")
        if ref_gl != int(BUDGET_REFERENCE_GL_ORDER):
            raise NumericalConfigError(
                "budget_reference_gl_order must equal "
                f"BUDGET_REFERENCE_GL_ORDER={BUDGET_REFERENCE_GL_ORDER} "
                f"(Commit C); got {self.budget_reference_gl_order!r}")
        if oracle != float(BUDGET_REFERENCE_ORACLE_BOUND):
            raise NumericalConfigError(
                "budget_reference_oracle_bound must equal "
                f"BUDGET_REFERENCE_ORACLE_BOUND={BUDGET_REFERENCE_ORACLE_BOUND}; "
                f"got {self.budget_reference_oracle_bound!r}")
        if gl > ref_gl:
            raise NumericalConfigError(
                f"gl_order={gl} exceeds budget_reference_gl_order={ref_gl}; "
                "the Commit C residual reference must be an elevated "
                "Gauss-Legendre order")

        if not (math.isfinite(ratio_ceil) and ratio_ceil > 0.0):
            raise NumericalConfigError(
                "max_panel_to_min_sigma_ratio must be finite and > 0; "
                f"got {self.max_panel_to_min_sigma_ratio!r}")

        for name, eps in (
            ("default_temporal_tol", self.default_temporal_tol),
            ("default_spatial_tol", self.default_spatial_tol),
        ):
            e = _require_real(name, eps)
            if not (math.isfinite(e) and 0.0 < e < 1.0):
                raise NumericalConfigError(
                    f"{name} must be finite and in (0, 1); got {eps!r}")

        # The argument invariants below are unreachable from every public model
        # path: resolve_sigma_bounds runs first and rejects the same inputs, so
        # this object only ever sees a resolved pair. They are retained as the
        # guard for direct NumericalConfig.create callers, and they render the
        # same canonical clause the resolver does, so the identity is the same
        # whichever way the violation arrives (D-40).
        lo, hi = self.min_sigma, self.max_sigma
        if mode == "rectangle":
            if lo is None and hi is None:
                pass
            elif lo is None or hi is None:
                raise_rectangle_bounds_violation(min_sigma=lo, max_sigma=hi)
            else:
                validate_sigma_pair(
                    _require_real("min_sigma", lo),
                    _require_real("max_sigma", hi),
                )
        else:
            # polygon
            if lo is None:
                raise_polygon_min_sigma_violation()
            lo_f = _require_real("min_sigma", lo)
            if hi is None:
                # Reachable only by direct construction: resolve_sigma_bounds
                # defaults max_sigma before this object is built, and rejects
                # when it cannot (no CRS). Behaviour frozen as it stands.
                if not (math.isfinite(lo_f) and lo_f > 0.0):
                    raise_min_sigma_positive_violation(min_sigma=lo_f)
            else:
                validate_sigma_pair(
                    lo_f, _require_real("max_sigma", hi))
            # Panel/min_sigma prefilter (necessary but not sufficient for tau).
            # D-40: the clause is rendered by panel_ratio_invariant_clause, not
            # restated here, so the constructor and set_window paths cannot
            # drift apart again.
            if h / lo_f > ratio_ceil:
                raise_panel_ratio_violation(
                    panel_h_m=h, min_sigma=lo_f,
                    ratio_ceil=ratio_ceil, tau_abs=tau)

    @classmethod
    def create(
        cls,
        *,
        panel_h_m: float = DEFAULT_PANEL_H_M,
        gl_order: int = DEFAULT_GL_ORDER,
        support_mode: ExcitationSupportMode = "rectangle",
        min_sigma: Optional[float] = None,
        max_sigma: Optional[float] = None,
        default_temporal_tol: float = DEFAULT_TEMPORAL_TOL,
        default_spatial_tol: float = DEFAULT_SPATIAL_TOL,
        production_tau_abs: float = PRODUCTION_TAU_ABS,
        budget_reference_gl_order: int = BUDGET_REFERENCE_GL_ORDER,
        budget_reference_oracle_bound: float = BUDGET_REFERENCE_ORACLE_BOUND,
        max_panel_to_min_sigma_ratio: float = MAX_PANEL_TO_MIN_SIGMA_RATIO,
    ) -> NumericalConfig:
        """Single factory for ``NumericalConfig`` (D-35 / A-23).

        Passes arguments through unchanged; all type and value validation runs
        in ``__post_init__``. Do not construct via the dataclass initializer
        from call sites outside this factory. Do not coerce here (WP1.4a).
        """
        return cls(
            panel_h_m=panel_h_m,
            gl_order=gl_order,
            default_temporal_tol=default_temporal_tol,
            default_spatial_tol=default_spatial_tol,
            production_tau_abs=production_tau_abs,
            budget_reference_gl_order=budget_reference_gl_order,
            budget_reference_oracle_bound=budget_reference_oracle_bound,
            max_panel_to_min_sigma_ratio=max_panel_to_min_sigma_ratio,
            support_mode=support_mode,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
        )
