"""Lane B rows for NumericalConfig construction (Phase 3f WP1.3).

Valid configs construct; each invalid combination raises NumericalConfigError
naming the violated constraint. Validation lives only in __post_init__ behind
NumericalConfig.create (A-23 / D-35).
"""

from __future__ import annotations

import pytest

from bstpp.config import NumericalConfig, NumericalConfigError
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
    cfg = NumericalConfig.create(support_mode="rectangle")
    with pytest.raises(Exception):
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
        ({"support_mode": "hexagon"}, "support_mode"),  # type: ignore[arg-type]
        (
            {"support_mode": "rectangle", "min_sigma": 1.0},
            "rectangle|both",
        ),
        (
            {"support_mode": "rectangle", "min_sigma": 5.0, "max_sigma": 1.0},
            "min_sigma < max_sigma|σ-bound",
        ),
        (
            {"support_mode": "polygon"},
            "polygon|min_sigma",
        ),
        (
            {
                "support_mode": "polygon",
                "min_sigma": 0.05,
                "max_sigma": 5.0,
                "panel_h_m": 20.0,
            },
            "panel_h_m / min_sigma|max_panel_to_min_sigma_ratio",
        ),
        (
            {"support_mode": "polygon", "min_sigma": "0.05", "max_sigma": 5.0,
             "panel_h_m": 0.2},
            "min_sigma|real number|str",
        ),
    ],
)
def test_invalid_configs_raise_named_error(kwargs, match):
    with pytest.raises(NumericalConfigError, match=match):
        NumericalConfig.create(**kwargs)


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
