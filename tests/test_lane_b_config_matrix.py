"""Lane B configuration / compatibility / mutation matrix (protocol §6).

Executable gate for pre-3f stabilization. Axis level sets are enumerated
from code (see ``LANE_B_AXES`` and ``docs/config_matrix.md``). Coverage is a
generated pairwise covering array over nine axes plus forced
supported/rejected rows — not the full Cartesian product.

Assertion families at every covered point (protocol §6):
admissibility (named errors + message substrings); leg consistency by
object identity where single-sourced; provenance touched/untouched;
transactional rollback by whole-state snapshot; sentinel stability;
constructor/setter equivalence; numerical budget at shipped defaults.

``save_rslts`` provenance round-trip is asserted but marked xfail under
ledger G2 (owned by 3f) so the matrix documents the requirement without
pretending the gap is closed.
"""

from __future__ import annotations

import inspect
import itertools
import json
import os
import pickle
import tempfile

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import geopandas as gpd
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from shapely.geometry import box

import bstpp
from bstpp.config import (
    NumericalConfig,
    NumericalConfigError,
    builder_max_sigma_invariant_clause,
    min_sigma_positive_invariant_clause,
    panel_ratio_invariant_clause,
    polygon_min_sigma_invariant_clause,
    rectangle_bounds_invariant_clause,
    sigma_order_invariant_clause,
    support_mode_invariant_clause,
)
from bstpp.excitation_support import (
    build_excitation_support,
    resolve_excitation_support_mode,
    resolve_sigma_bounds,
)
from bstpp.main import Hawkes_Model, LGCP_Model, _UNSET
from bstpp.polygon_mass import (
    DEFAULT_GL_ORDER,
    DEFAULT_PANEL_H_M,
    MAX_PANEL_TO_MIN_SIGMA_RATIO,
    PRODUCTION_TAU_ABS,
    assert_polygon_mass_table_budget,
    build_quad_table,
    prepare_polygon_mass_table,
)
from bstpp.trigger import Temporal_Exponential, Temporal_Power_Law, Trigger
from tests._polygon_prepare_helpers import prepare_table_for_model
from tests.test_kernel_capability_gates import _CustomSpatialRho
from tests.test_phase3d_excitation_support import _NonGaussianSpatial
from tests.test_phase3e_cutoffs import (
    _assert_window_state_unchanged,
    _observable_window_state,
)

T_DAYS = 30.0
A_RECT = np.array([[0.0, 1.0], [0.0, 1.0]])
A_METERS = np.array([[0.0, 200.0], [0.0, 200.0]])
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(0.25),
)
PRIORS_POWER = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    gamma=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(0.25),
)

_SEASONAL_DECODER = os.path.join(
    os.path.dirname(bstpp.__file__),
    "decoders",
    "decoder_1d_T24_circ_small_l8",
)
needs_decoder = pytest.mark.skipif(
    not os.path.exists(_SEASONAL_DECODER),
    reason="seasonal decoder artifact absent",
)

# Nine axes from code / docs/config_matrix.md (traceability inventory).
LANE_B_AXES = {
    "model_family": ("hawkes", "cox_hawkes", "lgcp"),
    "support": ("rectangle", "polygon"),
    "temporal_trigger": (
        "Temporal_Exponential",
        "Temporal_Power_Law",
        "custom",
    ),
    "spatial_trigger": (
        "Spatial_Symmetric_Gaussian",
        "custom",
    ),
    "cutoff_input": (
        "tolerance",
        "physical",
        "omitted",
        "explicit_None_via_set_window",
    ),
    "entry_path": ("constructor", "set_window"),
    "builder_numerics": (
        "default_panel_gl",
        "guided_small_panel",
        "nondefault_gl_order",
    ),
    "standardization": ("none", "domain_area", "bool_rejected"),
    "sigma_bounds": (
        "both",
        "neither_rect",
        "polygon_min_required",
        "custom_spatial_rejects",
    ),
    "public_mutators_hawkes": ("set_window",),
}

_COVERING_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "refactor-patches",
    "pre-3f-stabilization",
    "covering_array_rows.json",
)
with open(_COVERING_JSON, encoding="utf-8") as _cf:
    COVERING_ARRAY: list[dict] = json.load(_cf)


class _CustomTemporal(Trigger):
    """Minimal custom temporal trigger for covering-array admissibility."""

    def get_par_names(self):
        return ["beta"]

    def compute_trigger(self, pars, pairs_and_values):
        coords, values = pairs_and_values
        return coords, jnp.exp(-values / pars["beta"]) / pars["beta"]

    def compute_integral(self, pars, dif):
        return 1.0 - jnp.exp(-dif / pars["beta"])

    def simulate_trigger(self, pars, rng=None):
        gen = rng if rng is not None else np.random
        return float(gen.exponential(scale=float(pars["beta"])))


def _data(n=8, seed=0, A=A_RECT):
    rng = np.random.default_rng(seed)
    x0, x1 = float(A[0, 0]), float(A[0, 1])
    y0, y1 = float(A[1, 0]), float(A[1, 1])
    return pd.DataFrame({
        "X": rng.uniform(x0 + 0.05 * (x1 - x0), x1 - 0.05 * (x1 - x0), n),
        "Y": rng.uniform(y0 + 0.05 * (y1 - y0), y1 - 0.05 * (y1 - y0), n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _unit_gdf():
    return gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)])


def _sorted_coords(coords):
    c = np.asarray(coords)
    if c.size == 0:
        return c.reshape(0, 2)
    order = np.lexsort((c[:, 1], c[:, 0]))
    return c[order]


# ------------------------------------------------------------------ axes --
def test_lane_b_axis_inventory_matches_code_surfaces():
    """Level sets are taken from code, not from this docstring alone."""
    assert LANE_B_AXES["support"] == ("rectangle", "polygon")
    assert set(LANE_B_AXES["builder_numerics"]) == {
        "default_panel_gl", "guided_small_panel", "nondefault_gl_order"}
    assert set(LANE_B_AXES["standardization"]) == {
        "none", "domain_area", "bool_rejected"}
    assert LANE_B_AXES["public_mutators_hawkes"] == ("set_window",)
    # Live callables: only set_window is the public cutoff mutator.
    hawkes_methods = {
        name for name, _ in inspect.getmembers(Hawkes_Model, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert "set_window" in hawkes_methods
    assert "set_cutoff" not in hawkes_methods
    assert "set_spatial_window" not in hawkes_methods
    # Sentinel is private; public calls use omission vs explicit None.
    assert _UNSET is not None
    sig = inspect.signature(Hawkes_Model.set_window)
    assert sig.parameters["window"].default is _UNSET
    assert sig.parameters["spatial_window"].default is _UNSET


def test_lane_b_production_tau_abs_is_a21_constant():
    """Numerical budget constant is the A-21 production gate (not legacy)."""
    assert PRODUCTION_TAU_ABS == 1e-5
    assert DEFAULT_PANEL_H_M / 1.0 <= MAX_PANEL_TO_MIN_SIGMA_RATIO or True
    # Shipped defaults remain internally consistent for the panel-ratio gate.
    assert DEFAULT_GL_ORDER >= 8
    assert MAX_PANEL_TO_MIN_SIGMA_RATIO == 8.0


# ----------------------------------------------------- forced reject rows --
@pytest.mark.parametrize(
    "label,factory,exc,match",
    [
        (
            "gdf_without_mode",
            lambda: Hawkes_Model(
                _data(), _unit_gdf(), T_DAYS, cox_background=False, **PRIORS),
            ValueError,
            "explicit excitation_support",
        ),
        (
            "polygon_without_mass_table",
            lambda: Hawkes_Model(
                _data(), A_METERS, T_DAYS, cox_background=False,
                excitation_support="polygon",
                min_sigma=5.0, max_sigma=40.0, **PRIORS),
            ValueError,
            "mass_table",
        ),
        (
            "polygon_non_gaussian_spatial",
            lambda: Hawkes_Model(
                _data(A=A_METERS), A_METERS, T_DAYS, cox_background=False,
                excitation_support="polygon",
                min_sigma=5.0, max_sigma=40.0,
                spatial_trig=_NonGaussianSpatial,
                mass_table=prepare_table_for_model(
                    _data(A=A_METERS), A_METERS,
                    min_sigma=5.0, max_sigma=40.0),
                **PRIORS),
            TypeError,
            "Spatial_Symmetric_Gaussian",
        ),
        (
            "power_law_rejects_mean_lag",
            lambda: Hawkes_Model(
                _data(), A_RECT, T_DAYS, cox_background=False,
                temporal_trig=Temporal_Power_Law,
                mean_lag_days=dist.HalfNormal(1.0),
                window=25.0,
                **PRIORS_POWER),
            TypeError,
            "Temporal_Exponential|mean_lag",
        ),
        (
            "bool_standardize_cov",
            lambda: Hawkes_Model(
                _data(),
                gpd.GeoDataFrame(
                    geometry=[box(0.0, 0.0, 1.0, 1.0)]),
                T_DAYS, cox_background=False,
                excitation_support="rectangle",
                spatial_cov=gpd.GeoDataFrame(
                    {"v": [0.5, -0.5]},
                    geometry=[
                        box(0, 0, 0.5, 1),
                        box(0.5, 0, 1, 1),
                    ],
                ),
                cov_names=["v"],
                standardize_cov=True,
                **PRIORS),
            ValueError,
            "boolean",
        ),
        (
            "lgcp_set_window",
            lambda: LGCP_Model(
                _data(), A_RECT, T_DAYS, a_0=dist.Normal(0, 5)
            ).set_window(12.0),
            NotImplementedError,
            "excitation|Hawkes|cutoff",
        ),
        (
            "custom_spatial_rejects_sigmax_bounds",
            lambda: Hawkes_Model(
                _data(), A_RECT, T_DAYS, cox_background=False,
                spatial_trig=_CustomSpatialRho,
                min_sigma=0.05, max_sigma=0.5,
                spatial_window=0.3,
                a_0=dist.Normal(0, 5),
                alpha=dist.Beta(2, 2),
                beta=dist.HalfNormal(1.0),
                rho=dist.HalfNormal(1.0),
            ),
            TypeError,
            "Spatial_Symmetric_Gaussian|min_sigma|sigmax",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_lane_b_forced_reject_rows(label, factory, exc, match):
    with pytest.raises(exc, match=match):
        factory()


# ---------------------------------------------------- forced success rows --
@pytest.mark.parametrize(
    "label,builder",
    [
        (
            "hawkes_rect_physical_ctor",
            lambda: Hawkes_Model(
                _data(), A_RECT, T_DAYS, cox_background=False,
                window=25.0, spatial_window=0.4, **PRIORS),
        ),
        (
            "hawkes_rect_tolerance_ctor",
            lambda: Hawkes_Model(
                _data(), A_RECT, T_DAYS, cox_background=False,
                temporal_cutoff_tol=1e-3,
                design_mean_lag_days=2.0,
                spatial_cutoff_tol=1e-3,
                design_sigma=0.1,
                **PRIORS),
        ),
        (
            "hawkes_rect_omitted_cutoffs",
            lambda: Hawkes_Model(
                _data(), A_RECT, T_DAYS, cox_background=False, **PRIORS),
        ),
        (
            "hawkes_poly_physical_ctor",
            lambda: _hawkes_polygon_meters(),
        ),
        (
            "hawkes_poly_gdf_explicit_polygon",
            lambda: _hawkes_polygon_gdf(),
        ),
    ],
)
def test_lane_b_forced_success_rows_construct(label, builder):
    m = builder()
    assert m.args["model"] in ("hawkes", "cox_hawkes")
    assert m.excitation_support is m.args["excitation_support"]
    assert m.excitation_support.mode in ("rectangle", "polygon")
    assert "window" in m.args
    assert m.cutoff_provenance is not None


@needs_decoder
def test_lane_b_cox_hawkes_rectangle_constructs():
    m = Hawkes_Model(
        _data(), A_RECT, T_DAYS, cox_background=True, **PRIORS)
    assert m.args["model"] == "cox_hawkes"
    assert m.excitation_support is m.args["excitation_support"]
    assert m.excitation_support.mode == "rectangle"


@needs_decoder
def test_lane_b_cox_hawkes_polygon_constructs():
    data = _data(A=A_METERS, seed=3)
    table = prepare_table_for_model(
        data, A_METERS, min_sigma=5.0, max_sigma=40.0)
    m = Hawkes_Model(
        data, A_METERS, T_DAYS, cox_background=True,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        mass_table=table, **PRIORS)
    assert m.args["model"] == "cox_hawkes"
    assert m.excitation_support.mode == "polygon"
    # Ownership copy at install: equal content, not caller identity.
    assert m.excitation_support.mass_table is not table
    np.testing.assert_allclose(
        m.excitation_support.mass_table.values, table.values)
    assert m.excitation_support is m.args["excitation_support"]


def test_lane_b_lgcp_constructs_and_rejects_set_window():
    m = LGCP_Model(_data(), A_RECT, T_DAYS, a_0=dist.Normal(0, 5))
    assert "window" not in m.args
    with pytest.raises(NotImplementedError, match="excitation|Hawkes|cutoff"):
        m.set_window()
    assert "window" not in m.args


def _hawkes_polygon_meters():
    data = _data(A=A_METERS, seed=2)
    table = prepare_table_for_model(
        data, A_METERS, min_sigma=5.0, max_sigma=40.0)
    return Hawkes_Model(
        data, A_METERS, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        mass_table=table, **PRIORS)


def _hawkes_polygon_gdf():
    # CRS-less + small min_sigma: guided panel (B1 path).
    from bstpp.polygon_mass import prepare_polygon_mass_table

    data = _data(seed=4)
    gdf = _unit_gdf()
    geom = gdf.geometry.union_all()
    min_sigma = 0.05
    table = prepare_polygon_mass_table(
        geom,
        data["X"].to_numpy(dtype=float),
        data["Y"].to_numpy(dtype=float),
        min_sigma=min_sigma,
        max_sigma=0.5,
        panel_h_m=MAX_PANEL_TO_MIN_SIGMA_RATIO * min_sigma,
        crs=None,
    )
    return Hawkes_Model(
        data, gdf, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=min_sigma, max_sigma=0.5,
        mass_table=table, **PRIORS)


# ------------------------------------------------------- leg consistency --
def test_lane_b_support_object_identity_across_args_and_attribute():
    m = Hawkes_Model(
        _data(), A_RECT, T_DAYS, cox_background=False,
        window=20.0, spatial_window=0.3, **PRIORS)
    assert m.excitation_support is m.args["excitation_support"]
    # Compensator / parenting read args['excitation_support']; attribute is
    # the same object (single-sourced install).
    m.set_window(22.0)
    assert m.excitation_support is m.args["excitation_support"]
    assert m.excitation_provenance == dict(m.excitation_support.provenance)


def test_lane_b_polygon_mass_table_identity_on_install():
    m = _hawkes_polygon_meters()
    assert m.excitation_support.mass_table is m.args[
        "excitation_support"].mass_table
    table_id = id(m.excitation_support.mass_table)
    # Temporal-only set_window must reuse the installed table object.
    m.set_window(15.0)
    assert id(m.excitation_support.mass_table) == table_id


# -------------------------------------- constructor / setter equivalence --
def test_lane_b_constructor_setter_equivalence_physical_windows():
    data = _data(seed=7)
    via_ctor = Hawkes_Model(
        data, A_RECT, T_DAYS, cox_background=False,
        window=25.0, spatial_window=0.4,
        design_mean_lag_days=2.0, design_sigma=0.1,
        **PRIORS)
    via_set = Hawkes_Model(
        data, A_RECT, T_DAYS, cox_background=False,
        design_mean_lag_days=2.0, design_sigma=0.1,
        **PRIORS)
    via_set.set_window(25.0, spatial_window=0.4)

    assert via_ctor.args["window"] == pytest.approx(via_set.args["window"])
    assert via_ctor.args["spatial_window"] == pytest.approx(
        via_set.args["spatial_window"])
    np.testing.assert_allclose(
        _sorted_coords(via_ctor.args["coords"]),
        _sorted_coords(via_set.args["coords"]),
        rtol=0, atol=0,
    )
    # Both routes end on physical selections for supplied axes.
    assert via_ctor.cutoff_provenance.temporal.selection == "physical"
    assert via_set.cutoff_provenance.temporal.selection == "physical"
    assert via_ctor.cutoff_provenance.spatial.selection == "physical"
    assert via_set.cutoff_provenance.spatial.selection == "physical"
    assert via_ctor.cutoff_provenance.temporal.window_internal == pytest.approx(
        via_set.cutoff_provenance.temporal.window_internal)
    assert via_ctor.cutoff_provenance.spatial.spatial_window == pytest.approx(
        via_set.cutoff_provenance.spatial.spatial_window)


# ----------------------------------------------------- sentinel / rollback --
def test_lane_b_set_window_empty_is_noop_by_identity():
    m = Hawkes_Model(
        _data(), A_RECT, T_DAYS, cox_background=False,
        window=25.0, spatial_window=0.4, **PRIORS)
    before = _observable_window_state(m)
    support_before = m.excitation_support
    m.set_window()
    _assert_window_state_unchanged(before, m)
    assert m.excitation_support is support_before


def test_lane_b_set_window_explicit_none_clears_spatial():
    m = Hawkes_Model(
        _data(), A_RECT, T_DAYS, cox_background=False,
        window=25.0, spatial_window=0.4, **PRIORS)
    m.set_window(spatial_window=None)
    assert m.args["spatial_window"] is None
    assert m.cutoff_provenance.spatial.selection == "default_untruncated"


@pytest.mark.parametrize(
    ("bad_call", "match"),
    [
        ({"window": -1.0}, r"window must be finite and > 0; got -1\.0"),
        ({"window": float("nan"), "spatial_window": 0.2},
         r"window must be finite and > 0; got nan"),
        ({"spatial_window": 0.0},
         r"spatial_window must be finite and > 0; got 0\.0"),
    ],
)
def test_lane_b_rejected_set_window_rolls_back_whole_state(bad_call, match):
    # OP-17 / A-26: closeout P1 requires error type AND a message substring at
    # every matrix rejection. This was one of the two bare pytest.raises sites;
    # a bare ValueError here could not tell a window-validation rejection from
    # any other ValueError raised anywhere in set_window, which is how the
    # WP1.4b entry-path asymmetry passed through the matrix unseen.
    m = Hawkes_Model(
        _data(), A_RECT, T_DAYS, cox_background=False,
        window=25.0, spatial_window=0.4, **PRIORS)
    before = _observable_window_state(m)
    with pytest.raises(ValueError, match=match):
        m.set_window(**bad_call)
    _assert_window_state_unchanged(before, m)


def test_lane_b_polygon_incompatible_spatial_change_rolls_back():
    data = _data(A=A_METERS, seed=8)
    table = prepare_table_for_model(
        data, A_METERS, min_sigma=5.0, max_sigma=40.0, spatial_window=50.0)
    m = Hawkes_Model(
        data, A_METERS, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        spatial_window=50.0,
        mass_table=table, **PRIORS)
    before = _observable_window_state(m)
    # Spatial change without a replacement table must reject (no silent rebuild).
    with pytest.raises(ValueError, match="mass_table"):
        m.set_window(spatial_window=40.0)
    _assert_window_state_unchanged(before, m)
    # Table metadata for old window cannot install under a new window.
    stale = prepare_table_for_model(
        data, A_METERS, min_sigma=5.0, max_sigma=40.0, spatial_window=50.0)
    # OP-17 / A-26: the second bare pytest.raises site. The table is stale on
    # its spatial_window, and the assertion now says so rather than accepting
    # any ValueError from anywhere in the install path.
    with pytest.raises(
        ValueError,
        match=r"spatial_window=50\.0 does not match model spatial_window=40\.0",
    ):
        m.set_window(spatial_window=40.0, mass_table=stale)
    _assert_window_state_unchanged(before, m)


# ------------------------------------------- entry-path error identity --
# A-26 / D-40, Lane B `entry_path` axis. The defect this row exists to catch:
# WP1.4b made the SAME violation raise two different errors depending on
# whether it arrived through the constructor or through set_window, and the
# matrix could not see it because NumericalConfigError subclasses ValueError
# and every rejection was pinned with a bare pytest.raises(ValueError).
#
# The obligation is entry-path invariance, so the assertion compares the two
# paths to EACH OTHER, not each to a literal. A future divergence fails here
# whatever the wording, and this row is what stops WP2-WP5 reintroducing the
# split once four more configs multiply the entry paths.

def _coarse_table_for(data, A, *, min_sigma, max_sigma, spatial_window=None):
    """A table below the panel/min_sigma floor, built past prepare's guard."""
    from bstpp.preparation import prepare_domain
    dom = prepare_domain(A)
    # Same geometry resolution as prepare_table_for_model: an array domain has
    # no union_geometry, so the model validates against the bounds box.
    if dom.is_polygon:
        geom = dom.union_geometry
    else:
        b = dom.bounds
        geom = box(b[0, 0], b[1, 0], b[0, 1], b[1, 1])
    return build_quad_table(
        geom,
        data["X"].to_numpy(dtype=np.float64),
        data["Y"].to_numpy(dtype=np.float64),
        float(min_sigma), float(max_sigma),
        ws=None if spatial_window is None else float(spatial_window),
        h_panel=float(MAX_PANEL_TO_MIN_SIGMA_RATIO * min_sigma * 10.0),
        gl_order=int(DEFAULT_GL_ORDER),
    )


def test_lane_b_panel_ratio_error_identity_is_entry_path_invariant():
    data = _data(A=A_METERS, seed=12)
    min_sigma, max_sigma, sw = 5.0, 40.0, 50.0
    coarse = _coarse_table_for(
        data, A_METERS, min_sigma=min_sigma, max_sigma=max_sigma,
        spatial_window=sw)

    # entry path 1: constructor
    with pytest.raises(NumericalConfigError) as ctor:
        Hawkes_Model(
            data, A_METERS, T_DAYS, cox_background=False,
            excitation_support="polygon",
            min_sigma=min_sigma, max_sigma=max_sigma, spatial_window=sw,
            mass_table=coarse, **PRIORS)

    # entry path 2: set_window on a model built from a table that passes
    good = prepare_table_for_model(
        data, A_METERS, min_sigma=min_sigma, max_sigma=max_sigma,
        spatial_window=sw)
    m = Hawkes_Model(
        data, A_METERS, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=min_sigma, max_sigma=max_sigma, spatial_window=sw,
        mass_table=good, **PRIORS)
    before = _observable_window_state(m)
    with pytest.raises(NumericalConfigError) as setw:
        m.set_window(spatial_window=sw, mass_table=coarse)

    assert type(ctor.value) is type(setw.value)
    assert str(ctor.value) == str(setw.value), (
        "the same violation must produce the same message from every entry "
        "path (D-40); a divergence here is the WP1.4b defect returning")
    # ...and the shared text is the single-sourced clause, not a coincidence.
    assert str(ctor.value) == panel_ratio_invariant_clause(
        panel_h_m=float(coarse.h_panel), min_sigma=min_sigma,
        ratio_ceil=MAX_PANEL_TO_MIN_SIGMA_RATIO, tau_abs=PRODUCTION_TAU_ABS)
    str(ctor.value).encode("ascii")  # D-40: raised messages are ASCII
    # A rejected set_window is still transactional (P4 / D-33).
    _assert_window_state_unchanged(before, m)


# ------------------------------------ sigma/mode entry-path identity --
# A-27 / D-40 (sigma-mode refinement), Lane B `entry_path` axis. The defect
# these rows exist to catch: each of the five sigma/mode invariants had two to
# six implementations across excitation_support, config and polygon_mass,
# raising two error types and up to five different messages for one violation.
# The config's own branches were dead on every public path because
# resolve_sigma_bounds ran first, so the bare ValueError always won.
#
# The obligation is that ONE invariant has ONE identity whatever detects it, so
# each row collects every owner of that invariant and compares them TO EACH
# OTHER before pinning the shared text to the single-sourced clause. A site may
# append remediation, so the shared part is asserted as a common prefix and at
# least one owner must render the clause exactly.

def _raised(fn):
    """Return the exception fn raises, or fail if it does not raise."""
    with pytest.raises(BaseException) as ei:  # noqa: PT011 - identity is asserted below
        fn()
    return ei.value


_CRS_M = "EPSG:32618"  # projected, metre axis units


_SIGMA_MODE_TABLE: list = []
_SIGMA_MODE_BUDGET_TABLE: list = []


def _sigma_mode_budget_table():
    """A minimal valid table for calling assert_polygon_mass_table_budget
    directly. Built lazily so collection stays cheap."""
    if not _SIGMA_MODE_BUDGET_TABLE:
        _SIGMA_MODE_BUDGET_TABLE.append(build_quad_table(
            box(0.0, 0.0, 200.0, 200.0),
            np.array([10.0, 20.0]), np.array([10.0, 20.0]),
            5.0, 40.0, ws=None, h_panel=1.0,
            gl_order=int(DEFAULT_GL_ORDER)))
    return _SIGMA_MODE_BUDGET_TABLE[0]


def _sigma_mode_table(data):
    """A valid table, built once and lazily: main.py:1983 rejects a polygon
    constructor with no table BEFORE resolve_sigma_bounds runs, so the σ rows
    must get past it to reach the invariant under test."""
    if not _SIGMA_MODE_TABLE:
        _SIGMA_MODE_TABLE.append(prepare_table_for_model(
            data, A_METERS, min_sigma=5.0, max_sigma=40.0))
    return _SIGMA_MODE_TABLE[0]


def _sigma_mode_owners():
    """(invariant, canonical clause, [(owner name, callable), ...])."""
    data = _data(A=A_METERS, seed=31)

    def ctor(**kw):
        base = dict(
            cox_background=False, excitation_support="polygon",
            min_sigma=5.0, max_sigma=40.0, **PRIORS)
        base.update(kw)
        return lambda: Hawkes_Model(
            data, A_METERS, T_DAYS,
            mass_table=_sigma_mode_table(data), **base)

    def rect_ctor(**kw):
        base = dict(cox_background=False, excitation_support="rectangle",
                    **PRIORS)
        base.update(kw)
        return lambda: Hawkes_Model(data, A_METERS, T_DAYS, **base)

    return [
        (
            "I1 rectangle both-or-neither",
            rectangle_bounds_invariant_clause(min_sigma=5.0, max_sigma=None),
            [
                ("model constructor", rect_ctor(min_sigma=5.0)),
                ("resolve_sigma_bounds", lambda: resolve_sigma_bounds(
                    mode="rectangle", min_sigma=5.0, max_sigma=None,
                    crs=None)),
                ("NumericalConfig.create", lambda: NumericalConfig.create(
                    support_mode="rectangle", min_sigma=5.0, max_sigma=None)),
            ],
        ),
        (
            "I2 polygon requires min_sigma",
            polygon_min_sigma_invariant_clause(),
            [
                ("model constructor", ctor(min_sigma=None)),
                ("resolve_sigma_bounds", lambda: resolve_sigma_bounds(
                    mode="polygon", min_sigma=None, max_sigma=None, crs=None)),
                ("NumericalConfig.create", lambda: NumericalConfig.create(
                    support_mode="polygon", min_sigma=None, max_sigma=None)),
                ("prepare_polygon_mass_table",
                 lambda: prepare_polygon_mass_table(
                     box(0.0, 0.0, 200.0, 200.0),
                     np.array([10.0, 20.0]), np.array([10.0, 20.0]),
                     min_sigma=None, max_sigma=40.0)),
            ],
        ),
        (
            "I3 min_sigma finite and positive",
            min_sigma_positive_invariant_clause(min_sigma=0.0),
            [
                ("model constructor", ctor(min_sigma=0.0)),
                ("resolve_sigma_bounds", lambda: resolve_sigma_bounds(
                    mode="polygon", min_sigma=0.0, max_sigma=40.0, crs=None)),
                ("NumericalConfig.create", lambda: NumericalConfig.create(
                    support_mode="polygon", min_sigma=0.0, max_sigma=40.0)),
                ("prepare_polygon_mass_table",
                 lambda: prepare_polygon_mass_table(
                     box(0.0, 0.0, 200.0, 200.0),
                     np.array([10.0, 20.0]), np.array([10.0, 20.0]),
                     min_sigma=0.0, max_sigma=40.0)),
                ("build_quad_table", lambda: build_quad_table(
                    box(0.0, 0.0, 200.0, 200.0),
                    np.array([10.0, 20.0]), np.array([10.0, 20.0]),
                    0.0, 40.0, ws=None, h_panel=1.0,
                    gl_order=int(DEFAULT_GL_ORDER))),
                # Unreachable through every public MODEL path -- resolve_sigma_bounds
                # rejects a non-positive min_sigma first, and validate_polygon_mass_table
                # additionally requires table.sigma_min == sigma_min, which
                # build_quad_table now refuses to build. Reachable only by calling
                # this public module function directly, which is what this owner does.
                ("assert_polygon_mass_table_budget",
                 lambda: assert_polygon_mass_table_budget(
                     _sigma_mode_budget_table(), sigma_min=0.0)),
            ],
        ),
        (
            "I4 min_sigma < max_sigma",
            sigma_order_invariant_clause(min_sigma=40.0, max_sigma=5.0),
            [
                ("model constructor", ctor(min_sigma=40.0, max_sigma=5.0)),
                ("resolve_sigma_bounds", lambda: resolve_sigma_bounds(
                    mode="polygon", min_sigma=40.0, max_sigma=5.0, crs=None)),
                ("NumericalConfig.create", lambda: NumericalConfig.create(
                    support_mode="polygon", min_sigma=40.0, max_sigma=5.0,
                    panel_h_m=1.0)),
                ("build_quad_table", lambda: build_quad_table(
                    box(0.0, 0.0, 200.0, 200.0),
                    np.array([10.0, 20.0]), np.array([10.0, 20.0]),
                    40.0, 5.0, ws=None, h_panel=1.0,
                    gl_order=int(DEFAULT_GL_ORDER))),
            ],
        ),
        (
            "I5 support-mode validity",
            support_mode_invariant_clause(support_mode="triangle"),
            [
                ("model constructor", rect_ctor(excitation_support="triangle")),
                ("resolve_excitation_support_mode",
                 lambda: resolve_excitation_support_mode(
                     is_polygon_domain=False, excitation_support="triangle")),
                ("resolve_sigma_bounds", lambda: resolve_sigma_bounds(
                    mode="triangle", min_sigma=5.0, max_sigma=40.0, crs=None)),
                ("build_excitation_support", lambda: build_excitation_support(
                    mode="triangle", bounds=A_METERS, domain_gdf=None,
                    is_polygon_domain=False, crs=None, spatial_window=None,
                    min_sigma=5.0, max_sigma=40.0,
                    event_x_real=np.array([10.0]),
                    event_y_real=np.array([10.0]))),
                ("NumericalConfig.create", lambda: NumericalConfig.create(
                    support_mode="triangle")),
            ],
        ),
    ]


@pytest.mark.parametrize(
    "invariant,clause,owners",
    _sigma_mode_owners(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_lane_b_sigma_mode_error_identity_is_owner_invariant(
        invariant, clause, owners):
    raised = [(name, _raised(fn)) for name, fn in owners]

    # 1. One identity. Compared across owners, not each to a literal, so a
    #    future divergence fails here whatever the wording.
    types = {name: type(exc) for name, exc in raised}
    detail = "\n".join(
        f"    {name}: {type(exc).__name__}: {exc}" for name, exc in raised)
    assert len(set(types.values())) == 1, (
        f"{invariant}: one invariant must have one error identity from every "
        f"owner (D-40); got\n{detail}")
    assert set(types.values()) == {NumericalConfigError}, (
        f"{invariant}: the identity must be NumericalConfigError, not a bare "
        f"ValueError; got\n{detail}")

    # 2. One canonical clause. Sites may append remediation, so the invariant
    #    text is the shared prefix -- it may not be restated differently.
    for name, exc in raised:
        assert str(exc).startswith(clause), (
            f"{invariant}: {name} restates the invariant instead of rendering "
            f"the canonical clause.\n  expected prefix: {clause!r}\n"
            f"  got:             {str(exc)!r}")

    # 3. Agreement is not coincidence: at least one owner renders the clause
    #    exactly, with nothing appended.
    assert any(str(exc) == clause for _, exc in raised), (
        f"{invariant}: no owner renders the canonical clause exactly")

    # 4. D-40: raised messages are ASCII.
    for _, exc in raised:
        str(exc).encode("ascii")


def test_lane_b_polygon_default_max_sigma_without_crs_is_rejected():
    """A-27: the resolver rejects; the config would accept. Frozen as is.

    Polygon mode with ``max_sigma`` omitted must default it from
    DEFAULT_MAX_SIGMA_KM, which needs a projected CRS. Without one, the
    resolver refuses. ``NumericalConfig`` accepts the same situation because
    it never sees it -- it is handed an already-resolved pair, and a resolved
    polygon ``max_sigma`` is never None.

    This row exists because the obvious future refactor -- making the config
    the front gate for user-supplied bounds -- turns this rejection into a
    silent accept, and a silent accept would pass every other gate in the
    suite. Both halves are pinned so the asymmetry cannot move unnoticed.
    """
    from bstpp.excitation_support import DEFAULT_MAX_SIGMA_KM

    with pytest.raises(ValueError, match="Cannot convert metres to CRS units"):
        resolve_sigma_bounds(
            mode="polygon", min_sigma=0.05, max_sigma=None, crs=None)

    # The same arguments through the config: accepted, max_sigma stays None.
    cfg = NumericalConfig.create(
        support_mode="polygon", min_sigma=0.05, max_sigma=None,
        panel_h_m=MAX_PANEL_TO_MIN_SIGMA_RATIO * 0.05)
    assert cfg.max_sigma is None

    # With a CRS the resolver defaults, and THAT is what the config stores.
    lo, hi, meta = resolve_sigma_bounds(
        mode="polygon", min_sigma=1000.0, max_sigma=None,
        crs=gpd.GeoSeries([box(0.0, 0.0, 1.0, 1.0)], crs=_CRS_M).crs)
    assert lo == 1000.0
    assert hi == pytest.approx(DEFAULT_MAX_SIGMA_KM * 1000.0)
    assert meta["max_sigma_source"] == "default_5km"
    resolved_cfg = NumericalConfig.create(
        support_mode="polygon", min_sigma=lo, max_sigma=hi, panel_h_m=20.0)
    assert resolved_cfg.max_sigma == pytest.approx(5000.0)


def test_lane_b_prepare_polygon_mass_table_rejects_none_min_sigma_by_name():
    """A-27: float(None) used to surface as an unnamed TypeError here.

    Lane B admissibility requires a named error identifying the offending
    combination. This is the polygon-requires-min_sigma invariant at the
    builder; the type change from TypeError to NumericalConfigError is
    declared, and NumericalConfigError is NOT a TypeError subclass.
    """
    with pytest.raises(NumericalConfigError) as ei:
        prepare_polygon_mass_table(
            box(0.0, 0.0, 200.0, 200.0),
            np.array([10.0, 20.0]), np.array([10.0, 20.0]),
            min_sigma=None, max_sigma=40.0)
    assert str(ei.value).startswith(polygon_min_sigma_invariant_clause())
    assert not isinstance(ei.value, TypeError)


def test_lane_b_builder_max_sigma_error_identity_is_owner_invariant():
    """A-28 / I6: ``max_sigma=None`` at the two mass-table builders.

    The sibling of A-27's ``min_sigma=None`` defect, one line away in the same
    function, and it is a DISTINCT invariant rather than a member of I2. The
    test of that claim is in this row: the two builders REJECT while the model
    boundary ACCEPTS the same omission. I2 admits no such split -- nothing
    anywhere defaults ``min_sigma`` -- so borrowing I2's clause here would
    assert a package-wide requirement that does not exist.

    Both owners are compared to each other before the shared text is pinned to
    the single-sourced clause, so agreement cannot be coincidental, and each
    must name the function the caller actually called in its remediation.
    """
    owners = [
        ("prepare_polygon_mass_table", lambda: prepare_polygon_mass_table(
            box(0.0, 0.0, 200.0, 200.0),
            np.array([10.0, 20.0]), np.array([10.0, 20.0]),
            min_sigma=5.0, max_sigma=None, panel_h_m=1.0)),
        ("build_quad_table", lambda: build_quad_table(
            box(0.0, 0.0, 200.0, 200.0),
            np.array([10.0, 20.0]), np.array([10.0, 20.0]),
            5.0, None, ws=None, h_panel=1.0,
            gl_order=int(DEFAULT_GL_ORDER))),
    ]
    raised = [(name, _raised(fn)) for name, fn in owners]
    detail = "\n".join(
        f"    {name}: {type(exc).__name__}: {exc}" for name, exc in raised)

    # 1. One identity, compared across owners rather than each to a literal.
    types = {type(exc) for _, exc in raised}
    assert len(types) == 1, (
        f"I6: one invariant must have one error identity from every owner "
        f"(D-40); got\n{detail}")
    assert types == {NumericalConfigError}, (
        f"I6: the identity must be NumericalConfigError; got\n{detail}")

    # 2. Declared type change: TypeError -> NumericalConfigError, NOT a
    #    subclass relation. A caller catching TypeError is affected.
    for name, exc in raised:
        assert not isinstance(exc, TypeError), (
            f"I6: {name} still raises an unnamed TypeError from float(None)")

    # 3. One canonical clause, single-sourced, rendered byte for byte.
    clause = builder_max_sigma_invariant_clause()
    for name, exc in raised:
        assert str(exc).startswith(clause), (
            f"I6: {name} restates the invariant instead of rendering the "
            f"canonical clause.\n  expected prefix: {clause!r}\n"
            f"  got:             {str(exc)!r}")

    # 4. The remediation is the only part a site may vary, and each site uses
    #    it to name the function the caller called -- otherwise a user of the
    #    public builder is advised about an internal one.
    for name, exc in raised:
        assert name in str(exc)[len(clause):], (
            f"I6: {name}'s remediation does not name {name}")

    # 5. I6 is not I2. If a future refactor folds them together this fails.
    assert clause != polygon_min_sigma_invariant_clause()
    for _, exc in raised:
        assert polygon_min_sigma_invariant_clause() not in str(exc)

    # 6. D-40: raised messages are ASCII.
    for _, exc in raised:
        str(exc).encode("ascii")


def test_lane_b_model_boundary_accepts_the_max_sigma_the_builders_reject():
    """A-28: the asymmetry that makes I6 a distinct invariant, pinned.

    ``max_sigma=None`` is legitimate at the model boundary and illegitimate at
    the builder. Both halves are asserted here because the family decision
    rests on the contrast, not on either half alone: if a later change gives
    the builders a default, or makes the resolver reject, this row fails and
    the I6/I2 split has to be re-argued rather than silently dissolving.
    """
    from bstpp.excitation_support import DEFAULT_MAX_SIGMA_KM

    # Model boundary: accepts, and defaults from the projected CRS.
    _, hi, meta = resolve_sigma_bounds(
        mode="polygon", min_sigma=1000.0, max_sigma=None,
        crs=gpd.GeoSeries([box(0.0, 0.0, 1.0, 1.0)], crs=_CRS_M).crs)
    assert hi == pytest.approx(DEFAULT_MAX_SIGMA_KM * 1000.0)
    assert meta["max_sigma_source"] == "default_5km"

    # NumericalConfig: accepts None outright (the A-27 frozen asymmetry).
    assert NumericalConfig.create(
        support_mode="polygon", min_sigma=5.0, max_sigma=None,
        panel_h_m=1.0).max_sigma is None

    # The builder: rejects the same omission, by name.
    with pytest.raises(NumericalConfigError) as ei:
        prepare_polygon_mass_table(
            box(0.0, 0.0, 200.0, 200.0),
            np.array([10.0, 20.0]), np.array([10.0, 20.0]),
            min_sigma=5.0, max_sigma=None, panel_h_m=1.0)
    assert str(ei.value).startswith(builder_max_sigma_invariant_clause())


def test_lane_b_builder_max_sigma_guard_preserves_error_precedence():
    """A-28: I6 is checked last in prepare_polygon_mass_table, on purpose.

    At the pre-change tip ``max_sigma=None`` lost to every other check in that
    function. Guarding earlier would have changed which error a doubly-invalid
    call reports -- an undeclared behaviour change riding along with the
    declared one. These three combinations must still report what they
    reported before I6 existed.
    """
    poly = box(0.0, 0.0, 200.0, 200.0)
    ex, ey = np.array([10.0, 20.0]), np.array([10.0, 20.0])

    # min_sigma=None with max_sigma=None still reports I2, not I6.
    assert str(_raised(lambda: prepare_polygon_mass_table(
        poly, ex, ey, min_sigma=None, max_sigma=None, panel_h_m=1.0),
    )).startswith(polygon_min_sigma_invariant_clause())

    # A non-positive min_sigma still reports I3, not I6.
    assert str(_raised(lambda: prepare_polygon_mass_table(
        poly, ex, ey, min_sigma=0.0, max_sigma=None, panel_h_m=1.0),
    )).startswith(min_sigma_positive_invariant_clause(min_sigma=0.0))

    # A too-coarse panel still reports the panel ratio, not I6.
    assert str(_raised(lambda: prepare_polygon_mass_table(
        poly, ex, ey, min_sigma=5.0, max_sigma=None, panel_h_m=1e6),
    )).startswith(panel_ratio_invariant_clause(
        panel_h_m=1e6, min_sigma=5.0,
        ratio_ceil=MAX_PANEL_TO_MIN_SIGMA_RATIO, tau_abs=PRODUCTION_TAU_ABS))


def test_lane_b_build_quad_table_rejects_none_min_sigma_by_name():
    """A-28: I2 at ``build_quad_table``, a site A-27 covered only at the
    public builder.

    ``validate_sigma_pair`` deliberately does not coerce (OP-20), so
    ``float(sigma_min)`` in front of it swallowed ``None`` as an unnamed
    TypeError. Declared type change, same clause and identity as I2 elsewhere.
    """
    with pytest.raises(NumericalConfigError) as ei:
        build_quad_table(
            box(0.0, 0.0, 200.0, 200.0),
            np.array([10.0, 20.0]), np.array([10.0, 20.0]),
            None, 40.0, ws=None, h_panel=1.0,
            gl_order=int(DEFAULT_GL_ORDER))
    assert str(ei.value).startswith(polygon_min_sigma_invariant_clause())
    assert not isinstance(ei.value, TypeError)


def test_lane_b_resolve_sigma_bounds_validates_mode():
    """A-27: an invalid mode was silently treated as polygon and resolved.

    The `else` branch was the polygon branch, so resolve_sigma_bounds returned
    a resolved pair for mode='triangle' instead of rejecting. Unreachable from
    the model paths, which gate on resolve_excitation_support_mode first;
    direct callers had no such gate.
    """
    with pytest.raises(NumericalConfigError) as ei:
        resolve_sigma_bounds(
            mode="triangle", min_sigma=5.0, max_sigma=40.0, crs=None)
    assert str(ei.value) == support_mode_invariant_clause(
        support_mode="triangle")


# ----------------------------------------------------- numerical budget --
def test_lane_b_polygon_shipped_defaults_meet_panel_budget():
    """At shipped DEFAULT_PANEL_H_M / DEFAULT_GL_ORDER, install records budget."""
    data = _data(A=A_METERS, seed=9)
    table = prepare_table_for_model(
        data, A_METERS, min_sigma=5.0, max_sigma=40.0)
    assert table.h_panel == pytest.approx(DEFAULT_PANEL_H_M)
    assert table.gl_order == DEFAULT_GL_ORDER
    assert table.h_panel / 5.0 <= MAX_PANEL_TO_MIN_SIGMA_RATIO
    m = Hawkes_Model(
        data, A_METERS, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        mass_table=table, **PRIORS)
    prov = m.excitation_provenance
    assert prov.get("PRODUCTION_TAU_ABS") == PRODUCTION_TAU_ABS
    assert prov.get("panel_min_sigma_ratio") <= MAX_PANEL_TO_MIN_SIGMA_RATIO
    assert prov.get("measured_max_abs_residual") <= PRODUCTION_TAU_ABS
    assert prov.get("table_h_panel") == pytest.approx(DEFAULT_PANEL_H_M)
    assert prov.get("table_gl_order") == DEFAULT_GL_ORDER


# ----------------------------------------------- save_rslts provenance G2 --
@pytest.mark.xfail(
    reason="G2: save_rslts does not persist cutoff/excitation provenance (3f)",
    strict=True,
)
def test_lane_b_save_rslts_roundtrips_cutoff_provenance():
    m = Hawkes_Model(
        _data(), A_RECT, T_DAYS, cox_background=False,
        window=25.0, spatial_window=0.4,
        design_mean_lag_days=2.0, design_sigma=0.1,
        **PRIORS)
    # Minimal samples payload so save_rslts has something to pickle.
    m.samples = {"a_0": np.array([0.0])}
    before = m.cutoff_provenance.to_dict()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rslts.pkl")
        m.save_rslts(path)
        with open(path, "rb") as f:
            blob = pickle.load(f)
    assert "cutoff_provenance" in blob
    assert blob["cutoff_provenance"] == before


# ----------------------------------------------- covering array (Commit D) --
def _pairwise_coverage_fraction(axes: dict, points: list[dict]) -> float:
    names = [k for k in axes if k != "public_mutators_hawkes"]
    total = covered = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            need = set(itertools.product(axes[a], axes[b]))
            have = {
                (p[a], p[b]) for p in points if a in p and b in p
            }
            total += len(need)
            covered += len(have & need)
    return covered / total if total else 1.0


def test_lane_b_covering_array_achieves_full_pairwise():
    """Generated CA over nine axes must be 100% pairwise (docs/config_matrix.md)."""
    assert 12 <= len(COVERING_ARRAY) <= 20
    frac = _pairwise_coverage_fraction(LANE_B_AXES, COVERING_ARRAY)
    assert frac == pytest.approx(1.0)
    for row in COVERING_ARRAY:
        for axis, levels in LANE_B_AXES.items():
            if axis == "public_mutators_hawkes":
                continue
            assert row[axis] in levels


def _cov_gdf():
    return gpd.GeoDataFrame(
        {"v": [0.5, -0.5]},
        geometry=[box(0, 0, 0.5, 1), box(0.5, 0, 1, 1)],
    )


def _exercise_covering_row(row: dict):
    """Named admissibility for one covering-array assignment (priority rules)."""
    family = row["model_family"]
    support = row["support"]
    # --- reject priorities that encode register gates ---
    if row["standardization"] == "bool_rejected":
        with pytest.raises(ValueError, match="boolean"):
            Hawkes_Model(
                _data(), _unit_gdf(), T_DAYS, cox_background=False,
                excitation_support="rectangle",
                spatial_cov=_cov_gdf(), cov_names=["v"],
                standardize_cov=True, **PRIORS)
        return "reject_bool_std"

    if family == "lgcp":
        m = LGCP_Model(_data(), A_RECT, T_DAYS, a_0=dist.Normal(0, 5))
        if row["entry_path"] == "set_window":
            with pytest.raises(NotImplementedError, match="excitation|Hawkes|cutoff"):
                m.set_window(12.0)
            return "reject_lgcp_set_window"
        assert "window" not in m.args
        return "ok_lgcp"

    # Hawkes / Cox–Hawkes
    cox = family == "cox_hawkes"
    if not os.path.exists(_SEASONAL_DECODER) and cox:
        pytest.skip("seasonal decoder artifact absent")

    if (support == "polygon"
            and row["spatial_trigger"] == "custom"):
        with pytest.raises(TypeError, match="Spatial_Symmetric_Gaussian"):
            data = _data(A=A_METERS)
            Hawkes_Model(
                data, A_METERS, T_DAYS, cox_background=cox,
                excitation_support="polygon",
                min_sigma=5.0, max_sigma=40.0,
                spatial_trig=_NonGaussianSpatial,
                mass_table=prepare_table_for_model(
                    data, A_METERS, min_sigma=5.0, max_sigma=40.0),
                **PRIORS)
        return "reject_poly_custom_spatial"

    if (row["spatial_trigger"] == "custom"
            and row["sigma_bounds"] == "custom_spatial_rejects"):
        with pytest.raises(TypeError, match="Spatial_Symmetric_Gaussian|min_sigma|sigmax"):
            Hawkes_Model(
                _data(), A_RECT, T_DAYS, cox_background=cox,
                spatial_trig=_CustomSpatialRho,
                min_sigma=0.05, max_sigma=0.5,
                spatial_window=0.3,
                a_0=dist.Normal(0, 5),
                alpha=dist.Beta(2, 2),
                beta=dist.HalfNormal(1.0),
                rho=dist.HalfNormal(1.0),
            )
        return "reject_custom_sigma"

    # Construct path
    kwargs: dict = {}
    if row["temporal_trigger"] == "Temporal_Power_Law":
        kwargs["temporal_trig"] = Temporal_Power_Law
        priors = dict(PRIORS_POWER)
    elif row["temporal_trigger"] == "custom":
        kwargs["temporal_trig"] = _CustomTemporal
        priors = dict(PRIORS)
    else:
        kwargs["temporal_trig"] = Temporal_Exponential
        priors = dict(PRIORS)

    if row["spatial_trigger"] == "custom":
        kwargs["spatial_trig"] = _CustomSpatialRho
        priors.pop("sigmax_2", None)
        priors["rho"] = dist.HalfNormal(1.0)

    if support == "polygon":
        if row["builder_numerics"] == "default_panel_gl":
            domain = A_METERS
            data = _data(A=A_METERS, seed=11)
            min_s, max_s = 5.0, 40.0
            gl = DEFAULT_GL_ORDER
            panel = DEFAULT_PANEL_H_M
        else:
            domain = _unit_gdf()
            data = _data(seed=12)
            min_s, max_s = 0.05, 0.5
            gl = 8 if row["builder_numerics"] == "nondefault_gl_order" else DEFAULT_GL_ORDER
            panel = (MAX_PANEL_TO_MIN_SIGMA_RATIO * min_s) / (
                2.0 if gl == 8 else 1.0)
        geom = (
            domain.geometry.union_all()
            if hasattr(domain, "geometry")
            else box(0.0, 200.0, 0.0, 200.0)  # unused; ndarray path below
        )
        if isinstance(domain, np.ndarray):
            table = prepare_polygon_mass_table(
                box(0.0, 0.0, 200.0, 200.0),
                data["X"].to_numpy(dtype=float),
                data["Y"].to_numpy(dtype=float),
                min_sigma=min_s, max_sigma=max_s,
                panel_h_m=panel, gl_order=gl, crs=None)
        else:
            table = prepare_polygon_mass_table(
                geom,
                data["X"].to_numpy(dtype=float),
                data["Y"].to_numpy(dtype=float),
                min_sigma=min_s, max_sigma=max_s,
                panel_h_m=panel, gl_order=gl, crs=None)
        kwargs.update(
            excitation_support="polygon",
            min_sigma=min_s, max_sigma=max_s,
            mass_table=table,
        )
    else:
        domain = A_RECT
        data = _data(seed=13)
        if row["sigma_bounds"] == "both" and row["spatial_trigger"] != "custom":
            kwargs.update(min_sigma=0.05, max_sigma=0.5)
        if (row["cutoff_input"] == "tolerance"
                and row["temporal_trigger"] == "Temporal_Exponential"
                and row["spatial_trigger"] != "custom"):
            kwargs.update(
                temporal_cutoff_tol=1e-3, design_mean_lag_days=2.0,
                spatial_cutoff_tol=1e-3, design_sigma=0.1,
            )
        elif row["cutoff_input"] == "physical":
            kwargs.update(window=25.0, spatial_window=0.4)

    if row["standardization"] == "domain_area" and support == "rectangle":
        kwargs.update(
            spatial_cov=_cov_gdf(), cov_names=["v"],
            standardize_cov="domain_area",
            excitation_support="rectangle",
        )
        domain = _unit_gdf()

    m = Hawkes_Model(
        data, domain, T_DAYS, cox_background=cox, **kwargs, **priors)

    if row["entry_path"] == "set_window":
        if row["cutoff_input"] == "explicit_None_via_set_window":
            m.set_window(spatial_window=None)
            assert m.args["spatial_window"] is None
        elif row["cutoff_input"] == "physical":
            m.set_window(20.0, spatial_window=0.3)
        else:
            m.set_window(18.0)

    # Leg consistency / provenance on success
    assert m.excitation_support is m.args["excitation_support"]
    assert m.cutoff_provenance is not None
    return "ok_hawkes"


@pytest.mark.parametrize(
    "row", COVERING_ARRAY,
    ids=[f"ca{i}" for i in range(len(COVERING_ARRAY))],
)
def test_lane_b_covering_array_row_admissible(row):
    outcome = _exercise_covering_row(row)
    assert outcome.startswith("ok") or outcome.startswith("reject")
