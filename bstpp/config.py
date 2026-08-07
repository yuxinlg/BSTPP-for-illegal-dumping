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

def ascii_safe(text: str) -> str:
    """Escape any non-ASCII that reached a clause through INTERPOLATION (OP-22).

    D-40 requires raised messages to be ASCII. The literal halves of every
    clause are swept statically, but a caller-supplied value reaches the
    message through ``{value!r}`` and is not the clause's to control -- a
    non-ASCII ``support_mode`` or ``min_sigma`` produced a non-ASCII message,
    which can raise UnicodeEncodeError while the traceback carrying it is
    printed to a cp1252 console.

    Applied once to the assembled clause rather than per slot: ``ascii()`` on an
    individual field would also re-quote the plain ``{name}`` slots, changing
    the text of messages that are already correct. For ASCII input this returns
    the string unchanged, so no existing message moves.

    Public since A-37: ``data_contracts.enforce`` is a second D-40 owner of the
    encoding corollary and must render the identical escaping. A second
    spelling of this one line elsewhere in the package is the split D-40 exists
    to forbid, so it is imported rather than copied.
    """
    return text.encode("ascii", "backslashreplace").decode("ascii")


# Private spelling retained as the in-module name (same pattern as
# _require_real / require_config_real): one implementation, two names, never
# two implementations.
_ascii_safe = ascii_safe


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
    return _ascii_safe(
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
    return _ascii_safe(
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
    return _ascii_safe(
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
    return _ascii_safe(
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
    return _ascii_safe(f"min_sigma must be finite and positive; got {float(min_sigma)}")


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
    return _ascii_safe(
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
    return _ascii_safe(
        "excitation support mode must be 'rectangle' or 'polygon'; got "
        f"{support_mode!r}")


def raise_support_mode_violation(
    *, support_mode: object, remediation: str = "") -> NoReturn:
    """Raise the single identity for the support-mode validity invariant."""
    msg = support_mode_invariant_clause(support_mode=support_mode)
    if remediation:
        msg = f"{msg} {remediation}"
    raise NumericalConfigError(msg)


def standardize_cov_invariant_clause(*, standardize_cov: object) -> str:
    """Render the canonical ``standardize_cov`` enumerated-value clause (CI-9).

    One clause, two branches on the SAME text: the legacy-boolean prefix is
    kept because OP-3/OP-4 settled that booleans are rejected *explicitly* and
    never silently reinterpreted, so a migrating user must be told what
    happened to their argument rather than only what the accepted values are.
    """
    legacy = ("standardize_cov no longer accepts booleans; "
              if isinstance(standardize_cov, bool) else "")
    return _ascii_safe(
        f"{legacy}standardize_cov must be None (off, default) or "
        "'domain_area' (area-weighted over |C_c intersect A|); got "
        f"{standardize_cov!r}")


def raise_standardize_cov_violation(
    *, standardize_cov: object, remediation: str = "") -> NoReturn:
    """Raise the single identity for the ``standardize_cov`` value invariant.

    ``ValueError``, not ``NumericalConfigError``: the quantity belongs to
    ``ModelConfig``, which has not landed, and borrowing another object's
    error type would assert an ownership D-40 has not assigned. The type is
    also what every existing caller already catches, so only the site and the
    text move.
    """
    msg = standardize_cov_invariant_clause(standardize_cov=standardize_cov)
    if remediation:
        msg = f"{msg} {remediation}"
    raise ValueError(msg)


def validate_standardize_cov(standardize_cov: object) -> None:
    """CI-9: the enumerated value is valid, whatever else does or does not run.

    Called at construction unconditionally AND from the covariate leg. Two
    sites, one clause, byte-for-byte (D-40) -- the second is not redundant:
    ``attach_covariate_partitions`` is public and reachable without going
    through a model constructor.
    """
    if standardize_cov is None:
        return
    # `type(...) is str` rather than a `!=` comparison: `!=` on an ndarray
    # returns an array and the `if` then raises "truth value ambiguous" -- a
    # ValueError with the wrong message, which is exactly the split identity
    # D-40 forbids. Accept-by-construction, reject everything else.
    if type(standardize_cov) is str and standardize_cov == 'domain_area':
        return
    raise_standardize_cov_violation(standardize_cov=standardize_cov)


def cox_background_invariant_clause(*, cox_background: object) -> str:
    """Render the canonical ``cox_background`` boolean clause (CI-10).

    One clause, one branch on the same text: the old default ``'cox'`` gets a
    replacement named for it, because it is the value an upgrading user is
    most likely to be passing -- they copied it out of the signature.
    """
    replacement = (" use cox_background=True for the same model;"
                   if cox_background == "cox" else "")
    return _ascii_safe(
        "cox_background selects the background form and must be a bool "
        f"(True = Gaussian-process background, False = plain hawkes);{replacement} "
        f"got {cox_background!r}")


def raise_cox_background_violation(
    *, cox_background: object, remediation: str = "") -> NoReturn:
    """Raise the single identity for the ``cox_background`` type invariant.

    ``ValueError`` for the same reason ``standardize_cov`` uses one: the
    quantity belongs to ``ModelConfig``, which has not landed, and borrowing
    ``NumericalConfigError`` would assert an ownership D-40 has not assigned.
    """
    msg = cox_background_invariant_clause(cox_background=cox_background)
    if remediation:
        msg = f"{msg} {remediation}"
    raise ValueError(msg)


def validate_cox_background(cox_background: object) -> None:
    """CI-10: a boolean argument is a bool, not whatever happens to be truthy.

    The argument was consumed as ``if cox_background:`` with nothing checking
    it, so the accept set was every object and the branch was taken by
    TRUTHINESS -- ``'false'`` and ``'nonsense'`` both selected the Gaussian-
    process background. That is CI-9's defect with the sign flipped: there a
    bad value was accepted and IGNORED, here it was accepted and ACTED ON, in
    the direction opposite to what it said.

    ``np.bool_`` is accepted, and the reason is a CPython fact rather than a
    preference. D-42 accepts ``np.float64`` because it is a ``float``
    subclass; ``bool`` CANNOT BE SUBCLASSED in CPython, so ``np.bool_`` has no
    way to opt into the same treatment. Rejecting it would punish numpy for a
    language restriction rather than for being the wrong quantity.
    """
    if type(cox_background) is bool:
        return
    # Deferred: `config` is imported by modules numpy-free at import time, and
    # this is the only site in the module that needs an array type. The check
    # is on the SCALAR type, never on an array's truthiness -- `bool(arr)` on
    # a multi-element array raises its own ValueError with the wrong message,
    # which is the split identity D-40 forbids.
    import numpy as np
    if isinstance(cox_background, np.bool_):
        return
    raise_cox_background_violation(cox_background=cox_background)


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


# ------------------------------------------- CI-7 / CI-8: argument types --
# D-42. One policy for what a config-owned numeric argument may be, applied at
# every factory rather than decided separately at each. Two invariants, not
# one, by the D-40 owners-by-quantity test: CI-7's quantity is a REAL argument,
# CI-8's is an INTEGRAL one, and their accept sets differ (a float is a valid
# real and an invalid gl_order).
#
# The clause text of both is UNCHANGED from the `_require_real` / `_require_int`
# messages it replaces, deliberately: the invariants existed and were correctly
# enforced here, so this commit gives them a name, a single source and reach --
# not new wording. Existing pins on the text stay green, which is why they are
# evidence that the identity moved rather than the message.

def config_real_invariant_clause(*, name: str, value: object) -> str:
    """Render the canonical config real-argument clause (CI-7)."""
    return _ascii_safe(
        f"{name} must be a real number (int or float; bool and str "
        f"rejected); got {value!r} ({type(value).__name__})")


def raise_config_real_violation(
    *, name: str, value: object, remediation: str = "") -> NoReturn:
    """Raise the single identity for the config real-argument invariant."""
    msg = config_real_invariant_clause(name=name, value=value)
    if remediation:
        msg = f"{msg} {remediation}"
    raise NumericalConfigError(msg)


def config_integral_invariant_clause(*, name: str, value: object) -> str:
    """Render the canonical config integral-argument clause (CI-8)."""
    return _ascii_safe(
        f"{name} must be an int (bool rejected); got {value!r} "
        f"({type(value).__name__})")


def raise_config_integral_violation(
    *, name: str, value: object, remediation: str = "") -> NoReturn:
    """Raise the single identity for the config integral-argument invariant."""
    msg = config_integral_invariant_clause(name=name, value=value)
    if remediation:
        msg = f"{msg} {remediation}"
    raise NumericalConfigError(msg)


def require_config_real(name: str, value: object) -> float:
    """Validate a config-owned REAL argument; return it as a ``float`` (CI-7).

    ``bool`` is rejected explicitly because it is an ``int`` subclass, and
    ``np.float64`` is accepted because it is a ``float`` subclass -- the
    asymmetry A-23 reason 3 already had, preserved here rather than silently
    changed. The returned value is the coerced one; callers that use it get
    normalisation for free, callers that keep the original do not (A-34).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise_config_real_violation(name=name, value=value)
    return float(value)


def require_config_integral(name: str, value: object) -> int:
    """Validate a config-owned INTEGRAL argument; return it as an ``int``
    (CI-8).

    Rejecting non-ints matters more here than for reals: a bare ``int()`` on a
    quadrature order accepts ``16.7`` and silently truncates it to 16, and
    accepts ``True`` as the order 1 -- an accuracy change with no error.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise_config_integral_violation(name=name, value=value)
    return int(value)


# Private spellings retained as the in-module names; they are now one
# implementation behind the CI-7 / CI-8 raisers rather than a second spelling
# of the same check (D-40).
_require_int = require_config_integral
_require_real = require_config_real


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

        # A-34 / D-42: normalise on store. Until now _require_real returned
        # float(value) and NOTHING wrote it back, so a frozen object designated
        # the single source of numeric policy held whatever type the caller
        # happened to type -- int for an int argument, np.float64 for a
        # np.float64 one. Validate a coerced copy, store the original.
        # CI-7 now bounds the input to {int, float, np.float64}, so every
        # accepted value coerces to the same float64 and this write-back cannot
        # change a number; it only makes the stored type honest.
        object.__setattr__(self, "panel_h_m", h)
        object.__setattr__(self, "gl_order", gl)
        object.__setattr__(self, "production_tau_abs", tau)
        object.__setattr__(self, "budget_reference_gl_order", ref_gl)
        object.__setattr__(self, "budget_reference_oracle_bound", oracle)
        object.__setattr__(self, "max_panel_to_min_sigma_ratio", ratio_ceil)
        object.__setattr__(
            self, "default_temporal_tol", float(self.default_temporal_tol))
        object.__setattr__(
            self, "default_spatial_tol", float(self.default_spatial_tol))
        if self.min_sigma is not None:
            object.__setattr__(self, "min_sigma", float(self.min_sigma))
        if self.max_sigma is not None:
            object.__setattr__(self, "max_sigma", float(self.max_sigma))

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
