"""Frozen configuration objects for Phase 3f (behind the args adapter).

WP1 introduces ``NumericalConfig`` only. Objects are ``@dataclass(frozen=True)``
with all validation in ``__post_init__``, constructed through a single factory
per type (A-23; D-35). No Pydantic dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

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
    """Named error identifying a violated ``NumericalConfig`` constraint."""


@dataclass(frozen=True)
class NumericalConfig:
    """Computational / numerical settings (OP-13 / A-21; A-23 frozen dataclass).

    Holds mass-table builder settings, the Commit~C measured-budget policy, and
    package cutoff tolerance defaults. Support mode and σ bounds are accepted
    for coherence validation (model commitments remain owned elsewhere).

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
            raise NumericalConfigError(
                f"support_mode must be 'rectangle' or 'polygon'; got {mode!r}")

        h = float(self.panel_h_m)
        if not (math.isfinite(h) and h > 0.0):
            raise NumericalConfigError(
                f"panel_h_m must be finite and > 0; got {self.panel_h_m!r}")

        try:
            gl = int(self.gl_order)
        except (TypeError, ValueError) as exc:
            raise NumericalConfigError(
                f"gl_order must be an integer >= 1; got {self.gl_order!r}"
            ) from exc
        if gl != self.gl_order or gl < 1:
            raise NumericalConfigError(
                f"gl_order must be an integer >= 1; got {self.gl_order!r}")

        # Commit C measured-budget policy is frozen into the object (D-35).
        if float(self.production_tau_abs) != float(PRODUCTION_TAU_ABS):
            raise NumericalConfigError(
                "production_tau_abs must equal PRODUCTION_TAU_ABS="
                f"{PRODUCTION_TAU_ABS} (A-21 / Commit C measured budget); "
                f"got {self.production_tau_abs!r}")
        if int(self.budget_reference_gl_order) != int(BUDGET_REFERENCE_GL_ORDER):
            raise NumericalConfigError(
                "budget_reference_gl_order must equal "
                f"BUDGET_REFERENCE_GL_ORDER={BUDGET_REFERENCE_GL_ORDER} "
                f"(Commit C); got {self.budget_reference_gl_order!r}")
        if float(self.budget_reference_oracle_bound) != float(
                BUDGET_REFERENCE_ORACLE_BOUND):
            raise NumericalConfigError(
                "budget_reference_oracle_bound must equal "
                f"BUDGET_REFERENCE_ORACLE_BOUND={BUDGET_REFERENCE_ORACLE_BOUND}; "
                f"got {self.budget_reference_oracle_bound!r}")
        if gl > int(self.budget_reference_gl_order):
            raise NumericalConfigError(
                f"gl_order={gl} exceeds budget_reference_gl_order="
                f"{self.budget_reference_gl_order}; the Commit C residual "
                "reference must be an elevated Gauss-Legendre order")

        ratio_ceil = float(self.max_panel_to_min_sigma_ratio)
        if not (math.isfinite(ratio_ceil) and ratio_ceil > 0.0):
            raise NumericalConfigError(
                "max_panel_to_min_sigma_ratio must be finite and > 0; "
                f"got {self.max_panel_to_min_sigma_ratio!r}")

        for name, eps in (
            ("default_temporal_tol", self.default_temporal_tol),
            ("default_spatial_tol", self.default_spatial_tol),
        ):
            e = float(eps)
            if not (math.isfinite(e) and 0.0 < e < 1.0):
                raise NumericalConfigError(
                    f"{name} must be finite and in (0, 1); got {eps!r}")

        lo, hi = self.min_sigma, self.max_sigma
        if mode == "rectangle":
            if lo is None and hi is None:
                pass
            elif lo is None or hi is None:
                raise NumericalConfigError(
                    "support-mode compatibility (rectangle): min_sigma and "
                    "max_sigma must both be supplied or both omitted")
            else:
                self._validate_sigma_pair(float(lo), float(hi))
        else:
            # polygon
            if lo is None:
                raise NumericalConfigError(
                    "support-mode compatibility (polygon): min_sigma is "
                    "required (finite, positive; no default)")
            lo_f = float(lo)
            if hi is None:
                if not (math.isfinite(lo_f) and lo_f > 0.0):
                    raise NumericalConfigError(
                        f"min_sigma must be finite and positive; got {lo!r}")
            else:
                self._validate_sigma_pair(lo_f, float(hi))
            # Panel/min_sigma prefilter (necessary but not sufficient for tau).
            ratio = h / lo_f
            if ratio > ratio_ceil:
                raise NumericalConfigError(
                    "panel_h_m / min_sigma exceeds "
                    f"max_panel_to_min_sigma_ratio={ratio_ceil}: "
                    f"panel_h_m={h}, min_sigma={lo_f}, ratio={ratio}. "
                    "PRODUCTION_TAU_ABS measured residual is enforced at "
                    "mass-table install against "
                    f"BUDGET_REFERENCE_GL_ORDER="
                    f"{self.budget_reference_gl_order}."
                )

    @staticmethod
    def _validate_sigma_pair(lo: float, hi: float) -> None:
        if not (math.isfinite(lo) and lo > 0.0):
            raise NumericalConfigError(
                f"min_sigma must be finite and positive; got {lo}")
        if not (math.isfinite(hi) and hi > lo):
            raise NumericalConfigError(
                f"σ-bound coherence requires min_sigma < max_sigma; "
                f"got min_sigma={lo}, max_sigma={hi}")

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

        All validation runs in ``__post_init__``. Do not construct via the
        dataclass initializer from call sites outside this factory.
        """
        return cls(
            panel_h_m=float(panel_h_m),
            gl_order=int(gl_order),
            default_temporal_tol=float(default_temporal_tol),
            default_spatial_tol=float(default_spatial_tol),
            production_tau_abs=float(production_tau_abs),
            budget_reference_gl_order=int(budget_reference_gl_order),
            budget_reference_oracle_bound=float(budget_reference_oracle_bound),
            max_panel_to_min_sigma_ratio=float(max_panel_to_min_sigma_ratio),
            support_mode=support_mode,
            min_sigma=None if min_sigma is None else float(min_sigma),
            max_sigma=None if max_sigma is None else float(max_sigma),
        )
