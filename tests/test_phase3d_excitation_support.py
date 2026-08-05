"""Phase 3d: excitation support modes, sigma bounds, polygon Hermite mass.

Semantic / property tests (MR/SC). Rectangle-mode traces on array domains
must remain unchanged; polygon mode is the new path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro import handlers
from pyproj.crs import CRS
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from bstpp.config import (
    NumericalConfigError,
    polygon_min_sigma_invariant_clause,
)
from bstpp.excitation_support import (
    TruncatedLogNormal,
    metres_to_crs_units,
    resolve_sigma_bounds,
    truncate_sigmax_2_prior,
)
from bstpp.main import Hawkes_Model
from bstpp.polygon_mass import PolygonMassTable, knot_count
from bstpp.trigger import Spatial_Symmetric_Gaussian
from tests._polygon_prepare_helpers import prepare_table_for_model


class _NonGaussianSpatial(Spatial_Symmetric_Gaussian):
    """Subclass that overrides the event kernel (still Gaussian compensator).

    Used to prove polygon mode must reject anything other than the exact
    built-in Spatial_Symmetric_Gaussian class: isinstance would incorrectly
    admit this while the Hermite mass backend remains Gaussian-only.
    """

    def compute_trigger(self, pars, pairs_and_dxdy):
        coords, dx_vals, dy_vals = pairs_and_dxdy
        return coords, dx_vals * 0 + 1.0

T_DAYS = 30.0
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(0.25),
)


def _rect_data(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(0.1, 0.9, n),
        "Y": rng.uniform(0.1, 0.9, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _trace_loglik(model):
    tr = handlers.trace(handlers.seed(model.model, jax.random.PRNGKey(0))).get_trace(
        model.args)
    return float(np.asarray(tr["loglik"]["value"]))


# ---------------------------------------------------------------- OP-2 -----
def test_polygon_domain_requires_explicit_support_mode():
    gdf = gpd.GeoDataFrame({"geometry": [shapely_box(0, 0, 1, 1)]})
    with pytest.raises(ValueError, match="explicit excitation_support"):
        Hawkes_Model(_rect_data(), gdf, T_DAYS, cox_background=False, **PRIORS)


def test_array_domain_defaults_to_rectangle_mode():
    m = Hawkes_Model(_rect_data(), np.array([[0.0, 1.0], [0.0, 1.0]]),
                     T_DAYS, cox_background=False, **PRIORS)
    assert m.excitation_support.mode == "rectangle"
    assert m.excitation_support.mass_table is None
    assert m.args["priors"]["sigmax_2"].__class__.__name__ == "HalfNormal"


# -------------------------------------------------------------- bounds -----
def test_metres_to_crs_units_does_not_assume_metres():
    m_crs = CRS.from_epsg(26918)
    ft_crs = CRS.from_epsg(2272)
    assert metres_to_crs_units(5000.0, m_crs) == pytest.approx(5000.0)
    assert metres_to_crs_units(5000.0, ft_crs) == pytest.approx(
        5000.0 / ft_crs.axis_info[0].unit_conversion_factor)
    with pytest.raises(ValueError, match="geographic"):
        metres_to_crs_units(5000.0, CRS.from_epsg(4326))


def test_polygon_mode_requires_min_sigma_and_defaults_max_sigma():
    crs = CRS.from_epsg(26918)
    # A-27 / D-40: this was `pytest.raises(ValueError, match="min_sigma")`,
    # which discriminated nothing -- NumericalConfigError subclasses
    # ValueError and "min_sigma" appears in every candidate message, so it
    # passed identically before and after the identity unification. It now
    # pins the type exactly and the canonical clause by equality.
    with pytest.raises(NumericalConfigError) as ei:
        resolve_sigma_bounds(mode="polygon", min_sigma=None, max_sigma=None,
                             crs=crs)
    assert type(ei.value) is NumericalConfigError
    assert str(ei.value) == polygon_min_sigma_invariant_clause()
    str(ei.value).encode("ascii")  # D-40: raised messages are ASCII
    lo, hi, meta = resolve_sigma_bounds(
        mode="polygon", min_sigma=10.0, max_sigma=None, crs=crs)
    assert lo == 10.0
    assert hi == pytest.approx(5000.0)
    assert meta["max_sigma_source"] == "default_5km"


def test_rectangle_mode_omitted_bounds_leave_prior_unchanged():
    lo, hi, meta = resolve_sigma_bounds(
        mode="rectangle", min_sigma=None, max_sigma=None, crs=None)
    assert lo is None and hi is None and meta["bounds_active"] is False


def test_truncate_halfnormal_and_lognormal_have_interval_support():
    from numpyro.distributions.truncated import TwoSidedTruncatedDistribution

    hn = truncate_sigmax_2_prior(dist.HalfNormal(0.25), 0.1, 2.0)
    assert isinstance(hn, TwoSidedTruncatedDistribution)
    assert float(hn.low) == pytest.approx(0.01)
    assert float(hn.high) == pytest.approx(4.0)

    ln = truncate_sigmax_2_prior(dist.LogNormal(0.0, 0.5), 0.1, 2.0)
    assert isinstance(ln, TruncatedLogNormal)
    # support is interval — NUTS bijector cannot propose outside
    s = ln.support
    assert float(s.lower_bound) == pytest.approx(0.01)
    assert float(s.upper_bound) == pytest.approx(4.0)


def test_unsupported_prior_raises_clear_error():
    with pytest.raises(TypeError, match="supports HalfNormal"):
        truncate_sigmax_2_prior(dist.Gamma(2.0, 2.0), 0.1, 1.0)


# ------------------------------------ rectangle degeneracy / mode agree ----
def test_rectangle_modes_agree_on_array_domain():
    """On an axis-aligned rectangle, rectangle and polygon modes agree."""
    A = np.array([[0.0, 300.0], [0.0, 150.0]])
    rng = np.random.default_rng(1)
    data = pd.DataFrame({
        "X": rng.uniform(20, 280, 8),
        "Y": rng.uniform(10, 140, 8),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 8)),
    })
    priors = dict(
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0),
        sigmax_2=dist.HalfNormal(40.0),
    )
    m_rect = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="rectangle",
        min_sigma=5.0, max_sigma=80.0, **priors)
    table = prepare_table_for_model(
        data, A, min_sigma=5.0, max_sigma=80.0)
    m_poly = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=80.0,
        mass_table=table, **priors)

    params = dict(a_0=0.0, alpha=0.3, beta=2.0, sigmax_2=np.float32(20.0 ** 2))

    def itot_excite(m):
        tr = handlers.trace(handlers.seed(
            handlers.substitute(m.model, data=params),
            jax.random.PRNGKey(0))).get_trace(m.args)
        return float(np.asarray(tr["Itot_excite"]["value"]))

    r = itot_excite(m_rect)
    p = itot_excite(m_poly)
    assert abs(r - p) <= 1e-5, f"rect={r} poly={p}"


def test_polygon_parenting_discards_outside_A():
    """D-18: polygon mode refuses parenting outside A (L-shape)."""
    L = Polygon([(0, 0), (200, 0), (200, 80), (80, 80), (80, 200), (0, 200)])
    gdf = gpd.GeoDataFrame({"geometry": [L]})
    # one interior seed event
    data = pd.DataFrame({"X": [40.0], "Y": [40.0], "T": [1.0]})
    table = prepare_table_for_model(
        data, gdf, min_sigma=5.0, max_sigma=40.0)
    m = Hawkes_Model(
        data, gdf, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        mass_table=table,
        **PRIORS)
    support = m.excitation_support
    assert support.candidate_in_support(40.0, 40.0)
    assert not support.candidate_in_support(150.0, 150.0)  # outside L
    assert support.candidate_in_support(150.0, 40.0)       # inside L arm


def test_polygon_table_export_reload_roundtrip():
    A = np.array([[0.0, 200.0], [0.0, 200.0]])
    data = pd.DataFrame({
        "X": [50.0, 100.0], "Y": [50.0, 150.0], "T": [1.0, 2.0]})
    table = prepare_table_for_model(
        data, A, min_sigma=5.0, max_sigma=40.0)
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        mass_table=table, **PRIORS)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "table.npz"
        m.export_polygon_mass_table(path)
        loaded = PolygonMassTable.load_npz(path)
    np.testing.assert_allclose(loaded.values, m.excitation_support.mass_table.values)
    assert loaded.n_knots == knot_count(5.0, 40.0)


def test_rectangle_mode_trace_unchanged_without_bounds():
    """Array-domain rectangle mode with no sigma bounds: prior class unchanged
    and a seeded trace remains finite (pin suite covers bit equality)."""
    A = np.array([[0.0, 1.0], [0.0, 1.0]])
    m = Hawkes_Model(_rect_data(), A, T_DAYS, cox_background=False, **PRIORS)
    assert isinstance(m.args["priors"]["sigmax_2"], dist.HalfNormal)
    ll = _trace_loglik(m)
    assert np.isfinite(ll)


def test_wider_max_sigma_increases_knot_count():
    assert knot_count(10.0, 500.0) == 64
    assert knot_count(10.0, 5000.0) > 64


# -------------------------------- polygon spatial-trigger compatibility ----
def test_polygon_mode_rejects_non_exact_gaussian_spatial_trigger():
    """Polygon Hermite mass integrates Spatial_Symmetric_Gaussian only.

    A subclass that overrides the event kernel must be rejected: otherwise
    the event term and compensator evaluate different spatial kernels.
    Exact-type check required (not isinstance).
    """
    A = np.array([[0.0, 200.0], [0.0, 200.0]])
    data = pd.DataFrame({
        "X": [50.0, 100.0], "Y": [50.0, 150.0], "T": [1.0, 2.0]})
    priors = dict(
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    # Missing-table preflight precedes trigger checks; supply a table so the
    # exact-type gate is the failure under test.
    table = prepare_table_for_model(
        data, A, min_sigma=5.0, max_sigma=40.0)
    with pytest.raises(TypeError, match="Spatial_Symmetric_Gaussian"):
        Hawkes_Model(
            data, A, T_DAYS, cox_background=False,
            excitation_support="polygon",
            min_sigma=5.0, max_sigma=40.0,
            spatial_trig=_NonGaussianSpatial,
            mass_table=table,
            **priors,
        )


def test_rectangle_mode_still_accepts_custom_spatial_trigger():
    """Rectangle mode keeps the existing custom-trigger interface."""
    A = np.array([[0.0, 200.0], [0.0, 200.0]])
    data = pd.DataFrame({
        "X": [50.0, 100.0], "Y": [50.0, 150.0], "T": [1.0, 2.0]})
    priors = dict(
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="rectangle",
        spatial_trig=_NonGaussianSpatial,
        **priors,
    )
    assert type(m.args["sp_trig"]) is _NonGaussianSpatial
    assert m.excitation_support.mode == "rectangle"
    assert m.excitation_support.mass_table is None


def test_polygon_mode_accepts_exact_builtin_gaussian():
    A = np.array([[0.0, 200.0], [0.0, 200.0]])
    data = pd.DataFrame({
        "X": [50.0, 100.0], "Y": [50.0, 150.0], "T": [1.0, 2.0]})
    priors = dict(
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    table = prepare_table_for_model(
        data, A, min_sigma=5.0, max_sigma=40.0)
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        spatial_trig=Spatial_Symmetric_Gaussian,
        mass_table=table,
        **priors,
    )
    assert type(m.args["sp_trig"]) is Spatial_Symmetric_Gaussian
    assert m.excitation_support.mass_table is not None


# --------------------- post-hoc regression for 0eeaebd (float64 snapshot) --
def test_xy_events_real_preserves_float64_against_float32_axis_scales():
    """Post-hoc regression for ``0eeaebd`` (parent ``ce5508f`` fails).

    ``PreparedDomain.axis_scales`` is a JAX float32 array. Mapping internal
    ``xy_events`` back through those scales changes at least one original
    float64 coordinate on this offset/scale domain. Polygon preparation and
    ``_event_xy_real`` must use the ingestion float64 snapshot instead.
    """
    # Domain chosen so float64 (x-ox)/(x1-ox) * float32(axis_scales[0]) + ox
    # differs from x (see parent ce5508f affine fallback).
    A = np.array(
        [[2607925.96103126, 2637613.72396247],
         [4757272.11199708, 4947053.35197739]],
        dtype=np.float64,
    )
    x0 = np.float64(2617742.1695774216)
    y0 = np.float64(4800000.0)
    assert A[0, 0] < x0 < A[0, 1] and A[1, 0] < y0 < A[1, 1]

    from bstpp.preparation import prepare_domain

    dom = prepare_domain(A)
    assert dom.axis_scales.dtype == np.dtype("float32")
    A_ = np.asarray(dom.bounds, dtype=np.float64)
    scales = np.asarray(dom.axis_scales, dtype=float)
    xi = (float(x0) - A_[0, 0]) / (A_[0, 1] - A_[0, 0])
    x_roundtrip = xi * scales[0] + A_[0, 0]
    assert float(x_roundtrip) != float(x0)

    rng = np.random.default_rng(0)
    n = 8
    xs = np.concatenate(
        [[x0], rng.uniform(A[0, 0] + 500.0, A[0, 1] - 500.0, n - 1)]
    ).astype(np.float64)
    ys = np.concatenate(
        [[y0], rng.uniform(A[1, 0] + 500.0, A[1, 1] - 500.0, n - 1)]
    ).astype(np.float64)
    data = pd.DataFrame({
        "X": xs,
        "Y": ys,
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })
    min_sigma, max_sigma = 5.0, 40.0
    table = prepare_table_for_model(
        data, A, min_sigma=min_sigma, max_sigma=max_sigma)
    # Table identity was built from the original ordered float64 coordinates.
    np.testing.assert_array_equal(
        np.asarray(data["X"], dtype=np.float64), xs)
    np.testing.assert_array_equal(
        np.asarray(data["Y"], dtype=np.float64), ys)

    priors = dict(
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(40.0),
    )
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=min_sigma, max_sigma=max_sigma,
        mass_table=table,
        **priors,
    )
    xr, yr = m._event_xy_real()
    np.testing.assert_array_equal(np.asarray(xr, dtype=np.float64), xs)
    np.testing.assert_array_equal(np.asarray(yr, dtype=np.float64), ys)
    assert "xy_events_real" in m.args

    # A representable float64 tweak changes event identity and must reject.
    xs_bad = xs.copy()
    xs_bad[0] = np.nextafter(xs_bad[0], xs_bad[0] + 1.0)
    assert xs_bad[0] != xs[0]
    data_bad = data.copy()
    data_bad["X"] = xs_bad
    with pytest.raises(ValueError, match="events_sha256|event"):
        Hawkes_Model(
            data_bad, A, T_DAYS, cox_background=False,
            excitation_support="polygon",
            min_sigma=min_sigma, max_sigma=max_sigma,
            mass_table=table,
            **priors,
        )
