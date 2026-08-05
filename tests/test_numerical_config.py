"""Lane B rows for NumericalConfig construction (Phase 3f WP1.3).

Valid configs construct; each invalid combination raises NumericalConfigError
naming the violated constraint. Validation lives only in __post_init__ behind
NumericalConfig.create (A-23 / D-35).
"""

from __future__ import annotations

import dataclasses

import pytest

from bstpp.config import (
    NumericalConfig,
    NumericalConfigError,
    min_sigma_positive_invariant_clause,
    panel_ratio_invariant_clause,
    polygon_min_sigma_invariant_clause,
    rectangle_bounds_invariant_clause,
    sigma_order_invariant_clause,
    support_mode_invariant_clause,
)
from bstpp.cutoffs import DEFAULT_SPATIAL_TOL, DEFAULT_TEMPORAL_TOL
from bstpp.polygon_mass import (
    BUDGET_REFERENCE_GL_ORDER,
    BUDGET_REFERENCE_ORACLE_BOUND,
    DEFAULT_GL_ORDER,
    DEFAULT_PANEL_H_M,
    MAX_PANEL_TO_MIN_SIGMA_RATIO,
    PRODUCTION_TAU_ABS,
)


def test_valid_rectangle_default_constructs():
    cfg = NumericalConfig.create(support_mode="rectangle")
    assert cfg.panel_h_m == DEFAULT_PANEL_H_M
    assert cfg.gl_order == DEFAULT_GL_ORDER
    assert cfg.production_tau_abs == PRODUCTION_TAU_ABS
    assert cfg.budget_reference_gl_order == BUDGET_REFERENCE_GL_ORDER
    assert cfg.budget_reference_oracle_bound == BUDGET_REFERENCE_ORACLE_BOUND
    assert cfg.max_panel_to_min_sigma_ratio == MAX_PANEL_TO_MIN_SIGMA_RATIO
    assert cfg.default_temporal_tol == DEFAULT_TEMPORAL_TOL
    assert cfg.default_spatial_tol == DEFAULT_SPATIAL_TOL
    assert cfg.min_sigma is None and cfg.max_sigma is None


def test_valid_rectangle_with_sigma_bounds_constructs():
    cfg = NumericalConfig.create(
        support_mode="rectangle", min_sigma=1.0, max_sigma=10.0)
    assert cfg.min_sigma == 1.0
    assert cfg.max_sigma == 10.0


def test_valid_polygon_guided_panel_constructs():
    min_s = 0.05
    panel = MAX_PANEL_TO_MIN_SIGMA_RATIO * min_s
    cfg = NumericalConfig.create(
        panel_h_m=panel,
        gl_order=DEFAULT_GL_ORDER,
        support_mode="polygon",
        min_sigma=min_s,
        max_sigma=5.0,
    )
    assert cfg.panel_h_m == panel
    assert cfg.support_mode == "polygon"


def test_immutable_after_construction():
    # A-26 / OP-17 generalized: no bare pytest.raises in this file. A frozen
    # dataclass rejects assignment with FrozenInstanceError specifically;
    # raises(Exception) would also have passed on an AttributeError from a
    # typo in the attribute name.
    cfg = NumericalConfig.create(support_mode="rectangle")
    with pytest.raises(dataclasses.FrozenInstanceError, match="panel_h_m"):
        cfg.panel_h_m = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"panel_h_m": 0.0}, "panel_h_m"),
        ({"panel_h_m": float("nan")}, "panel_h_m"),
        ({"panel_h_m": "250"}, "panel_h_m|real number|str"),
        ({"gl_order": 0}, "gl_order"),
        ({"gl_order": 32.7}, "gl_order|int"),
        ({"gl_order": True}, "gl_order|bool"),
        ({"gl_order": 64}, "gl_order|budget_reference"),
        ({"production_tau_abs": 5.39e-4}, "production_tau_abs|PRODUCTION_TAU_ABS"),
        (
            {"budget_reference_gl_order": 16},
            "budget_reference_gl_order|BUDGET_REFERENCE",
        ),
        (
            {"budget_reference_oracle_bound": 1e-3},
            "budget_reference_oracle_bound",
        ),
        ({"default_temporal_tol": 0.0}, "default_temporal_tol"),
        ({"default_spatial_tol": 1.0}, "default_spatial_tol"),
        ({"default_temporal_tol": "0.01"}, "default_temporal_tol|real number|str"),
        (
            {
                "support_mode": "polygon",
                "min_sigma": 0.05,
                "max_sigma": 5.0,
                "panel_h_m": 20.0,
            },
            "panel_h_m / min_sigma|max_panel_to_min_sigma_ratio",
        ),
    ],
)
def test_invalid_configs_raise_named_error(kwargs, match):
    with pytest.raises(NumericalConfigError, match=match):
        NumericalConfig.create(**kwargs)


# A-27 / D-40 σ-mode refinement. The five σ/mode rows used to live in the
# alternation table above, matching "rectangle|both", "polygon|min_sigma",
# "min_sigma < max_sigma|sigma-bound" and "support_mode". Every one of those
# alternations passed both before and after the identity unification, so they
# asserted nothing about the property they were named for -- the same finding
# e-1 made about the panel-ratio tests. They now pin the canonical clause by
# equality, so a site that restates the invariant fails here.

@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {"support_mode": "hexagon"},  # type: ignore[arg-type]
            support_mode_invariant_clause(support_mode="hexagon"),
        ),
        (
            {"support_mode": "rectangle", "min_sigma": 1.0},
            rectangle_bounds_invariant_clause(min_sigma=1.0, max_sigma=None),
        ),
        (
            {"support_mode": "rectangle", "max_sigma": 1.0},
            rectangle_bounds_invariant_clause(min_sigma=None, max_sigma=1.0),
        ),
        (
            {"support_mode": "rectangle", "min_sigma": 5.0, "max_sigma": 1.0},
            sigma_order_invariant_clause(min_sigma=5.0, max_sigma=1.0),
        ),
        (
            {"support_mode": "rectangle", "min_sigma": 0.0, "max_sigma": 1.0},
            min_sigma_positive_invariant_clause(min_sigma=0.0),
        ),
        (
            {"support_mode": "polygon"},
            polygon_min_sigma_invariant_clause(),
        ),
        (
            {"support_mode": "polygon", "min_sigma": 0.0, "max_sigma": 5.0,
             "panel_h_m": 0.2},
            min_sigma_positive_invariant_clause(min_sigma=0.0),
        ),
        (
            {"support_mode": "polygon", "min_sigma": 5.0, "max_sigma": 1.0,
             "panel_h_m": 0.2},
            sigma_order_invariant_clause(min_sigma=5.0, max_sigma=1.0),
        ),
        # Polygon with max_sigma omitted: the config accepts a None upper
        # bound (it is handed a resolved pair in production and never sees
        # this), but min_sigma positivity still applies. See the Lane B row
        # test_lane_b_polygon_default_max_sigma_without_crs_is_rejected.
        (
            {"support_mode": "polygon", "min_sigma": 0.0, "panel_h_m": 0.2},
            min_sigma_positive_invariant_clause(min_sigma=0.0),
        ),
        # A-23 argument-type discipline, not one of the five (OP-20): pinned by
        # equality here for the same reason.
        (
            {"support_mode": "polygon", "min_sigma": "0.05", "max_sigma": 5.0,
             "panel_h_m": 0.2},
            "min_sigma must be a real number (int or float; bool and str "
            "rejected); got '0.05' (str)",
        ),
    ],
)
def test_sigma_mode_invariants_render_the_canonical_clause(kwargs, expected):
    with pytest.raises(NumericalConfigError) as ei:
        NumericalConfig.create(**kwargs)
    assert str(ei.value) == expected
    str(ei.value).encode("ascii")  # D-40: raised messages are ASCII


def test_panel_ratio_clause_is_ascii_and_single_sourced():
    """A-26 / D-40: the config renders the canonical clause, it does not restate it."""
    ratio_ceil = MAX_PANEL_TO_MIN_SIGMA_RATIO
    clause = panel_ratio_invariant_clause(
        panel_h_m=20.0, min_sigma=0.05, ratio_ceil=ratio_ceil,
        tau_abs=PRODUCTION_TAU_ABS)
    clause.encode("ascii")  # raises UnicodeEncodeError if D-40's ASCII rule broke

    with pytest.raises(NumericalConfigError) as ei:
        NumericalConfig.create(
            support_mode="polygon", min_sigma=0.05, max_sigma=5.0,
            panel_h_m=20.0)
    assert str(ei.value) == clause, (
        "the config must raise the canonical clause verbatim, with no "
        "site-specific text of its own")


def test_factory_rejects_silent_coercion():
    """WP1.4a: create must not truncate floats or parse strings."""
    with pytest.raises(NumericalConfigError, match="gl_order"):
        NumericalConfig.create(gl_order=32.7)
    with pytest.raises(NumericalConfigError, match="panel_h_m"):
        NumericalConfig.create(panel_h_m="250")
    with pytest.raises(NumericalConfigError, match="gl_order|bool"):
        NumericalConfig.create(gl_order=True)


def test_factory_is_sole_public_construction_path():
    """create() is the single factory; __post_init__ is the sole validator."""
    assert callable(NumericalConfig.create)
    cfg = NumericalConfig.create(support_mode="rectangle")
    assert isinstance(cfg, NumericalConfig)
