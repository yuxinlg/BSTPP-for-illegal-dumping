"""Phase 3a data-contract adversarial tests (IV commit).

Each contract in docs/phase3_baseline_and_decisions.tex section 10.a gets a
rejection test (mode='reject'), and the report-only default gets tests that
legacy behavior is preserved while the defect is surfaced loudly. Membership
REBASELINE tests (D-22 deterministic unique membership) belong to the
separate MR commit, not here: grid-line and covariate ties are asserted only
as report diagnostics via the standalone validators.

RED verification: this file fails collection at the frozen tip 476c2a0
(bstpp.data_contracts does not exist), and every constructor-mode test fails
without the data_contracts wiring in Point_Process_Model.__init__.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
import numpyro.distributions as dist
import pytest

from bstpp.main import Hawkes_Model
from bstpp.data_contracts import (
    DataContractError, validate_events, validate_covariates)

# Non-square real-unit domain (x: 10..30, y: 5..15) so real/internal
# conversions are exercised; horizon 200 days.
A_RECT = np.array([[10.0, 30.0], [5.0, 15.0]])
T_DAYS = 200.0
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))


def _interior_data(n=30, seed=7):
    r = np.random.RandomState(seed)
    return pd.DataFrame({
        "X": r.uniform(10.5, 29.5, n),
        "Y": r.uniform(5.5, 14.5, n),
        "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n)),
    })


def _model(data, A=A_RECT, mode="report", **kw):
    return Hawkes_Model(data, A, T_DAYS, cox_background=False,
                        excitation_support="rectangle",
                        data_contracts=mode, **PRIORS, **kw)


def _triangle_gdf(crs=None):
    # (10,5)-(30,5)-(30,15): interior is y < 5 + (x-10)/2, bounding box A_RECT
    g = gpd.GeoDataFrame(geometry=[Polygon([(10, 5), (30, 5), (30, 15)])])
    if crs is not None:
        g = g.set_crs(crs)
    return g


def _triangle_data(n=30, seed=7):
    """Events strictly inside the triangle: x in (15, 29.5), y one unit
    below the hypotenuse (>= 6.5 > bottom edge 5)."""
    data = _interior_data(n, seed)
    data["X"] = np.random.RandomState(seed + 1).uniform(15.0, 29.5, n)
    data["Y"] = 5.0 + (data["X"] - 10.0) / 2.0 - 1.0
    return data


def _checks_by_name(checks):
    return {c.name: c for c in checks}


# ---------------------------------------------------------------- valid data

def test_valid_interior_data_report_ok_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # any contract warning fails
        m = _model(_interior_data())
    assert m.data_contract_report.ok
    assert m.data_contract_report.mode == "report"


def test_valid_interior_data_constructs_under_reject():
    m = _model(_interior_data(), mode="reject")
    assert m.data_contract_report.ok


def test_default_mode_is_reject():
    """Post-dry-run flip (reviewer sign-off 2026-07-20): constructing with
    invalid data and NO data_contracts argument must reject."""
    data = _interior_data()
    data.loc[3, "X"] = np.nan
    with pytest.raises(DataContractError, match="event_coordinates_nonfinite"):
        Hawkes_Model(data, A_RECT, T_DAYS, cox_background=False, **PRIORS)


def test_default_mode_reject_heldout_path():
    m = Hawkes_Model(_interior_data(), A_RECT, T_DAYS, cox_background=False,
                     **PRIORS)
    assert m.data_contract_report.mode == "reject"
    m.run_svi(10, 0.01, plot_loss=False)
    with_nan = _interior_data(n=10, seed=11)
    with_nan.loc[len(with_nan)] = {"X": np.nan, "Y": 7.0, "T": 50.0}
    with pytest.raises(DataContractError, match="nonfinite X/Y/T"):
        m.log_expected_likelihood(with_nan)


# ------------------------------------------------------ nonfinite coordinates

def test_nan_coordinate_rejected():
    data = _interior_data()
    data.loc[3, "X"] = np.nan
    with pytest.raises(DataContractError, match="event_coordinates_nonfinite"):
        _model(data, mode="reject")


def test_nonnumeric_coordinate_rejected():
    data = _interior_data()
    data["Y"] = data["Y"].astype(object)
    data.loc[5, "Y"] = "not-a-number"
    with pytest.raises(DataContractError, match="event_coordinates_nonfinite"):
        _model(data, mode="reject")


def test_inf_coordinate_rejected():
    data = _interior_data()
    data.loc[2, "T"] = np.inf
    with pytest.raises(DataContractError, match="event_coordinates_nonfinite"):
        _model(data, mode="reject")


def test_nan_coordinate_report_warns_then_legacy_crash():
    """Report mode surfaces the defect loudly but preserves the legacy
    failure (misleading grid-membership crash), bit-for-bit no new behavior."""
    data = _interior_data()
    data.loc[3, "X"] = np.nan
    with pytest.warns(UserWarning, match="event_coordinates_nonfinite"):
        with pytest.raises(Exception, match="encompass"):
            _model(data, mode="report")


def test_missing_event_column_rejected():
    data = _interior_data().drop(columns=["Y"])
    with pytest.raises(DataContractError, match="event_columns_missing"):
        _model(data, mode="reject")


# ------------------------------------------------------------- temporal range

def test_time_beyond_horizon_rejected():
    data = _interior_data()
    data.loc[len(data) - 1, "T"] = T_DAYS + 1.0
    with pytest.raises(DataContractError, match="event_time_out_of_range"):
        _model(data, mode="reject")


def test_time_beyond_horizon_report_warns_but_legacy_accepts():
    data = _interior_data()
    data.loc[len(data) - 1, "T"] = T_DAYS + 1.0
    with pytest.warns(UserWarning, match="event_time_out_of_range"):
        m = _model(data, mode="report")
    # legacy silent acceptance preserved: all events still ingested
    assert len(m.args["t_events"]) == len(data)


# -------------------------------------------------------- spatial containment

def test_event_outside_rectangle_rejected():
    data = _interior_data()
    data.loc[0, "X"] = 31.0
    with pytest.raises(DataContractError, match="event_outside_domain"):
        _model(data, mode="reject")


def test_event_outside_polygon_inside_rectangle_rejected():
    """The baseline defect: in-rectangle out-of-polygon events are silently
    accepted (likelihood credit without compensator debit). Reject mode
    refuses them."""
    tri = _triangle_gdf()
    data = _triangle_data()
    data.loc[0, ["X", "Y"]] = (12.0, 12.0)  # in A_RECT, outside triangle
    with pytest.raises(DataContractError, match="event_outside_domain"):
        _model(data, A=tri, mode="reject")


def test_event_outside_polygon_report_warns_then_fails_at_membership():
    """UPDATED at 3c-2 (real requirement change, flagged in that commit):
    the 3a-era version of this test pinned report mode's legacy SILENT
    acceptance of out-of-domain events. Since 3c-2 the event membership
    join runs against the clipped support C_c ∩ A, so an out-of-domain
    event has no supported field cell: report mode still surfaces the
    defect by name first, then construction fails loudly at membership
    (D-3 fail-fast) instead of silently charging a bounding-rectangle
    cell outside A."""
    tri = _triangle_gdf()
    data = _triangle_data()
    data.loc[0, ["X", "Y"]] = (12.0, 12.0)
    with pytest.warns(UserWarning, match="event_outside_domain"):
        with pytest.raises(Exception, match="encompass"):
            _model(data, A=tri, mode="report")


def test_polygon_boundary_point_is_inside_D4():
    """D-4: a point exactly on the boundary of A is IN the domain -- no
    violation in reject mode, and it is reported as a diagnostic."""
    tri = _triangle_gdf()
    data = _triangle_data()
    data.loc[0, ["X", "Y"]] = (20.0, 10.0)  # exactly on the hypotenuse
    m = _model(data, A=tri, mode="reject")  # must not raise
    names = [c.name for c in m.data_contract_report.diagnostics]
    assert "event_on_domain_boundary" in names


def test_hole_and_multipolygon_containment():
    """Holes and multipolygon parts follow the same D-3/D-4 contract."""
    part1 = Polygon([(10, 5), (18, 5), (18, 15), (10, 15)],
                    holes=[[(12, 8), (14, 8), (14, 10), (12, 10)]])
    part2 = box(22, 5, 30, 15)
    dom = gpd.GeoDataFrame(geometry=[part1, part2])
    data = pd.DataFrame({
        "X": [11.0, 13.0, 20.0, 25.0],
        "Y": [6.0, 9.0, 10.0, 10.0],
        "T": [10.0, 20.0, 30.0, 40.0],
    })
    checks = _checks_by_name(validate_events(data, dom, T_DAYS))
    # row 1 in the hole, row 2 in the gap between parts; rows 0 and 3 valid
    assert list(checks["event_outside_domain"].indices) == [1, 2]


# --------------------------------------------------- grid-line tie diagnostics

def test_grid_line_events_are_diagnostics_each_axis_and_corner():
    """Points exactly on internal 25x25 grid lines are VALID events whose
    membership the D-22 MR commit makes deterministic; here they must be
    surfaced as diagnostics (they are the section-14 tie inventory)."""
    # internal edges: x = 10 + 0.8k, y = 5 + 0.4k
    data = pd.DataFrame({
        "X": [14.0, 12.3, 14.8],   # rows 0, 2 on x-edges (k=5, 6)
        "Y": [7.3, 7.4, 7.4],      # rows 1, 2 on y-edges (k=6); row 2 = corner
        "T": [10.0, 20.0, 30.0],
    })
    checks = _checks_by_name(validate_events(data, A_RECT, T_DAYS))
    assert "event_outside_domain" not in checks
    assert list(checks["event_on_grid_line_x"].indices) == [0, 2]
    assert list(checks["event_on_grid_line_y"].indices) == [1, 2]


def test_polygon_domain_grid_line_events_are_diagnostics():
    """Grid edges for a polygon domain come from its bounding rectangle
    (here exactly A_RECT for the triangle), so interior events on internal
    edges must be surfaced just as in the array-domain case."""
    # x = 14.0 is the k=5 internal x-edge (10 + 5/25*20); y = 7.0 is the
    # k=5 internal y-edge (5 + 5/25*10). Both points are strictly inside
    # the triangle (y < 5 + (x-10)/2) and off the other axis's edges.
    data = pd.DataFrame({
        "X": [14.0, 20.0], "Y": [6.0, 7.0], "T": [10.0, 20.0]})
    checks = _checks_by_name(validate_events(data, _triangle_gdf(), T_DAYS))
    assert "event_outside_domain" not in checks
    assert list(checks["event_on_grid_line_x"].indices) == [0]
    assert list(checks["event_on_grid_line_y"].indices) == [1]


def test_domain_edge_events_are_not_grid_line_ties():
    """Events on the OUTER rectangle edges join a single cell today and are
    covered by the D-22 outermost-edge-closed rule; they must not appear in
    the internal grid-line diagnostic."""
    data = pd.DataFrame({
        "X": [10.0, 30.0], "Y": [7.3, 7.4], "T": [10.0, 20.0]})
    checks = _checks_by_name(validate_events(data, A_RECT, T_DAYS))
    assert "event_on_grid_line_x" not in checks
    assert "event_outside_domain" not in checks


# ------------------------------------------------------------------- geometry

def test_invalid_domain_geometry_rejected():
    bowtie = Polygon([(10, 5), (30, 15), (30, 5), (10, 15)])  # self-intersecting
    dom = gpd.GeoDataFrame(geometry=[bowtie])
    with pytest.raises(DataContractError, match="domain_geometry_invalid"):
        _model(_interior_data(), A=dom, mode="reject")


def test_invalid_rectangle_domain_rejected():
    bad = np.array([[30.0, 10.0], [5.0, 15.0]])  # x1 < x0
    with pytest.raises(DataContractError, match="domain_rectangle_invalid"):
        _model(_interior_data(), A=bad, mode="reject")


# ----------------------------------------------------------------- covariates

def _cov_layer(vals=(1.0, 2.0), crs=None):
    g = gpd.GeoDataFrame(
        {"v": list(vals)},
        geometry=[box(10, 5, 20, 15), box(20, 5, 30, 15)])
    if crs is not None:
        g = g.set_crs(crs)
    return g


def test_nan_covariate_rejected():
    cov = _cov_layer(vals=(1.0, np.nan))
    data = _interior_data()
    with pytest.raises(DataContractError, match="covariate_values_nonfinite"):
        _model(data, mode="reject", spatial_cov=cov, cov_names=["v"])


def test_nan_covariate_report_warns_but_legacy_accepts():
    cov = _cov_layer(vals=(1.0, np.nan))
    data = _interior_data()
    with pytest.warns(UserWarning, match="covariate_values_nonfinite"):
        m = _model(data, mode="report", spatial_cov=cov, cov_names=["v"])
    assert "num_cov" in m.args  # legacy construction completed


def test_missing_covariate_coverage_rejected():
    cov = gpd.GeoDataFrame({"v": [1.0]}, geometry=[box(10, 5, 20, 15)])
    data = _interior_data()
    data.loc[0, "X"] = 25.0  # right half: no covariate polygon
    with pytest.raises(DataContractError, match="event_missing_covariate"):
        _model(data, mode="reject", spatial_cov=cov, cov_names=["v"])


def test_covariate_membership_tie_is_diagnostic():
    cov = _cov_layer()
    pts = np.array([[20.0, 7.0], [15.0, 7.0]])  # row 0 on the shared edge
    checks = _checks_by_name(
        validate_covariates(cov, ["v"], A_RECT, points_xy=pts))
    assert "event_missing_covariate" not in checks
    assert list(checks["covariate_membership_tie"].indices) == [0]


def test_invalid_covariate_geometry_rejected():
    cov = gpd.GeoDataFrame(
        {"v": [1.0]},
        geometry=[Polygon([(10, 5), (30, 15), (30, 5), (10, 15)])])
    with pytest.raises(DataContractError, match="covariate_geometry_invalid"):
        _model(_interior_data(), mode="reject", spatial_cov=cov,
               cov_names=["v"])


def test_crs_mismatch_rejected():
    tri = _triangle_gdf(crs="EPSG:2272")
    cov = _cov_layer(crs="EPSG:3857")
    data = _triangle_data()
    with pytest.raises(DataContractError, match="crs_mismatch"):
        _model(data, A=tri, mode="reject", spatial_cov=cov, cov_names=["v"])


# ------------------------------------------------------------------- horizon
@pytest.mark.parametrize("bad_T", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_horizon_rejected(bad_T):
    """T_max / horizon_days must be finite and positive before construction."""
    data = _interior_data(n=5)
    # Keep event times valid against a positive horizon so the failure is the
    # horizon contract itself, not event_time_out_of_range.
    if np.isfinite(bad_T) and bad_T <= 0.0:
        data = data.copy()
        data["T"] = 0.0
    with pytest.raises(DataContractError, match="horizon_invalid"):
        Hawkes_Model(
            data, A_RECT, bad_T, cox_background=False,
            excitation_support="rectangle",
            data_contracts="reject", **PRIORS,
        )


def test_validate_events_rejects_nonfinite_horizon_directly():
    checks = _checks_by_name(
        validate_events(_interior_data(n=3), A_RECT, float("nan")))
    assert "horizon_invalid" in checks
    assert checks["horizon_invalid"].kind == "violation"


# ------------------------------------------- held-out path: dropna made loud

def _fitted_model():
    m = _model(_interior_data())
    m.run_svi(10, 0.01, plot_loss=False)
    return m


def test_heldout_nan_rows_warn_and_match_legacy_drop():
    m = _fitted_model()
    heldout = _interior_data(n=10, seed=11)
    with_nan = heldout.copy()
    with_nan.loc[len(with_nan)] = {"X": np.nan, "Y": 7.0, "T": 50.0}
    with pytest.warns(UserWarning, match="nonfinite X/Y/T"):
        got = m.log_expected_likelihood(with_nan)
    clean = m.log_expected_likelihood(heldout)
    assert got == pytest.approx(clean, rel=1e-6)


def test_heldout_nan_rows_rejected_in_reject_mode():
    m = _model(_interior_data(), mode="reject")
    m.run_svi(10, 0.01, plot_loss=False)
    with_nan = _interior_data(n=10, seed=11)
    with_nan.loc[len(with_nan)] = {"X": np.nan, "Y": 7.0, "T": 50.0}
    with pytest.raises(DataContractError, match="nonfinite X/Y/T"):
        m.log_expected_likelihood(with_nan)
