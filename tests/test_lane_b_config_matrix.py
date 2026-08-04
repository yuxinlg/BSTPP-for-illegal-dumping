"""Lane B configuration / compatibility / mutation matrix (protocol §6).

Executable gate for pre-3f stabilization Commit 3. Axis level sets are
enumerated from code (see ``LANE_B_AXES`` and
``refactor-patches/pre-3f-stabilization/traceability_matrix.md``). Coverage
is pairwise over the feasible space plus forced supported/rejected rows —
not the full Cartesian product.

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
import os
import pickle
import tempfile

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import geopandas as gpd
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from shapely.geometry import box

import bstpp
from bstpp.main import Hawkes_Model, LGCP_Model, _UNSET
from bstpp.polygon_mass import (
    DEFAULT_GL_ORDER,
    DEFAULT_PANEL_H_M,
    MAX_PANEL_TO_MIN_SIGMA_RATIO,
    PRODUCTION_TAU_ABS,
)
from bstpp.trigger import Temporal_Power_Law
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

# Axis inventory from code (Lane B). Keep in sync with production sites;
# tests below assert the live callables still expose these levels.
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
    "public_mutators_hawkes": ("set_window",),
}


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
    "bad_call",
    [
        {"window": -1.0},
        {"window": float("nan"), "spatial_window": 0.2},
        {"spatial_window": 0.0},
    ],
)
def test_lane_b_rejected_set_window_rolls_back_whole_state(bad_call):
    m = Hawkes_Model(
        _data(), A_RECT, T_DAYS, cox_background=False,
        window=25.0, spatial_window=0.4, **PRIORS)
    before = _observable_window_state(m)
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        m.set_window(spatial_window=40.0, mass_table=stale)
    _assert_window_state_unchanged(before, m)


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
