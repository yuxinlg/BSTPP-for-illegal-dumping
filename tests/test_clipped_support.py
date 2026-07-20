"""Phase 3c-1: no-covariate background support is |C_c ∩ A| (D-6, SC).

Baseline defect (phase3_baseline_and_decisions §5.5, §10.c): on a polygon
domain A the no-covariate Cox background charged every in-domain field cell
its FULL internal area 1/n_xy^2 -- boundary cells straddling ∂A were
overcharged, and cells merely TOUCHING A (zero-area intersection) were
charged a full cell -- while the sampler drew locations over the full cell
and relied on simulate()'s A-filter to discard the outside points, so the
charged mass and the sampled support disagreed.

3c-1 semantic change (D-6): the support object holds the clipped
geometries C_c ∩ A with exact normalized intersection areas, and the SAME
object feeds the likelihood integration arrays and the background sampler
(10.c clipped-geometry reuse); background points outside A are never drawn.
The rectangle regime is unchanged: array domains keep the exact uniform
1/n_xy^2 areas bit-identically (golden-pin gate), and a rectangle supplied
as a polygon must degenerate to the same support within tolerance
(rectangle degeneracy acceptance, §10.d).

RED (pre-3c-1): the clipped-area property tests and both sampler-support
tests fail (uniform full areas; notch samples); the rectangle-regime tests
pass and are retained as the unchanged-regime pins.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
import jax
import numpyro.distributions as dist
from numpyro import handlers
import pytest

import bstpp
from bstpp.main import LGCP_Model
from bstpp.preparation import (prepare_domain, prepare_partitions,
                               finalize_integration_arrays, N_XY)

_SEASONAL_DECODER = os.path.join(os.path.dirname(bstpp.__file__), "decoders",
                                 "decoder_1d_T24_circ_small_l8")
needs_decoder = pytest.mark.skipif(
    not os.path.isfile(_SEASONAL_DECODER),
    reason="seasonal decoder artifact 'bstpp/decoders/decoder_1d_T24_circ_small_l8' is absent",
)

T_DAYS = 2.5 * 365.0

# Triangle with the unit square as bounding rectangle: the hypotenuse cuts a
# long run of boundary cells into proper sub-cell intersections (generic
# clipped-area case; no grid alignment anywhere).
TRI_POLY = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
TRI_GDF = gpd.GeoDataFrame({"geometry": [TRI_POLY]})

# L-shape whose notch corner (0.48, 0.48) lies ON the 25x25 grid lines
# (cell edges at multiples of 1/25 = 0.04): the cells bordering the notch
# from outside touch A along an edge only -- zero-area intersections that
# the baseline sjoin support charged at full 1/625.
L_POLY = box(0.0, 0.0, 1.0, 0.48).union(box(0.0, 0.48, 0.48, 1.0))
L_GDF = gpd.GeoDataFrame({"geometry": [L_POLY]})

# Non-grid-aligned L (notch at 0.5): cells straddling the notch edges have
# proper partial intersections -- the sampler-support probe domain, mirroring
# tests/test_lgcp_sim.py.
L_HALF_POLY = box(0.0, 0.0, 1.0, 0.5).union(box(0.0, 0.5, 0.5, 1.0))
L_HALF_GDF = gpd.GeoDataFrame({"geometry": [L_HALF_POLY]})


def _no_cov_integration_arrays(domain_input):
    """Seam-level pipeline: domain -> partitions -> no-covariate arrays."""
    dom = prepare_domain(domain_input)
    parts = prepare_partitions(dom, T_DAYS, offset_seasonal=0.0)
    finalize_integration_arrays(parts, "lgcp")
    return parts


def _events_inside(poly, n=50, seed=4):
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.98)
        if poly.buffer(-1e-6).contains(gpd.points_from_xy([x], [y])[0]):
            pts.append((x, y))
    return pd.DataFrame({"X": [p[0] for p in pts],
                         "Y": [p[1] for p in pts],
                         "T": np.sort(rng.uniform(0, T_DAYS, n))})


# ---------------------------------------------------------------------------
# Clipped-area semantics (seam level, D-6)
# ---------------------------------------------------------------------------

def test_polygon_areas_are_exact_cell_domain_intersections():
    """Each no-covariate integration area equals |C_c ∩ A| (normalized to
    the internal unit square), computed independently here with shapely on
    the triangle domain. RED pre-3c-1: every area is the full 1/625."""
    parts = _no_cov_integration_arrays(TRI_GDF)
    fi = np.asarray(parts.integration_field_indices)
    areas = np.asarray(parts.integration_areas, dtype=np.float64)
    cell_geoms = parts.comp_grid.set_index("comp_grid_id").geometry
    expected = np.array([cell_geoms.loc[c].intersection(TRI_POLY).area
                         for c in fi])
    # float32 storage of exact double-precision intersection areas
    np.testing.assert_allclose(areas, expected, rtol=2e-6, atol=0.0)
    # the hypotenuse must actually produce partial cells for this test to bite
    assert (expected < 1.0 / N_XY**2 * (1 - 1e-9)).any()


def test_polygon_support_total_mass_is_domain_area():
    """sum of integration areas = |A| / |A_rect| -- the clipped support
    charges the polygon, not the union of full cells. RED pre-3c-1: the
    triangle sum is n_in_domain_cells/625 > 0.5."""
    parts = _no_cov_integration_arrays(TRI_GDF)
    total = float(np.sum(np.asarray(parts.integration_areas, dtype=np.float64)))
    assert total == pytest.approx(TRI_POLY.area, rel=2e-6)


def test_touch_only_cells_carry_no_support():
    """Cells that touch A along an edge only (zero-area intersection; the
    grid-aligned L notch) are excluded from the support instead of being
    charged a full 1/625. All retained areas are strictly positive. RED
    pre-3c-1: the touching cells appear with full area."""
    parts = _no_cov_integration_arrays(L_GDF)
    fi = np.asarray(parts.integration_field_indices)
    areas = np.asarray(parts.integration_areas, dtype=np.float64)
    assert (areas > 0.0).all()
    cell_geoms = parts.comp_grid.set_index("comp_grid_id").geometry
    zero_area = [c for c in fi
                 if cell_geoms.loc[c].intersection(L_POLY).area <= 1e-12]
    assert zero_area == [], (
        f"cells {zero_area} touch the domain with zero intersection area "
        f"but are charged background mass")
    assert float(areas.sum()) == pytest.approx(L_POLY.area, rel=2e-6)


# ---------------------------------------------------------------------------
# Unchanged rectangle regime (golden-pin companions; pass pre- AND post-3c-1)
# ---------------------------------------------------------------------------

def test_array_rectangle_regime_bit_unchanged():
    """Array-domain support is the full grid with EXACT uniform float32
    1/n_xy^2 areas -- the regime covered by the four golden pin configs;
    3c-1 must not perturb it by a single bit."""
    parts = _no_cov_integration_arrays(np.array([[0.0, 1.0], [0.0, 1.0]]))
    np.testing.assert_array_equal(np.asarray(parts.integration_field_indices),
                                  np.arange(N_XY**2))
    areas = np.asarray(parts.integration_areas)
    assert areas.dtype == np.float32
    np.testing.assert_array_equal(
        areas, np.full(N_XY**2, 1.0 / N_XY**2, dtype=np.float32))


def test_rectangle_polygon_degeneracy():
    """Rectangle degeneracy acceptance (§10.d): a rectangle supplied as a
    polygon GeoDataFrame yields the same support as the array rectangle
    within tolerance -- all 625 cells, uniform areas."""
    parts = _no_cov_integration_arrays(gpd.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 1.0, 1.0)]}))
    np.testing.assert_array_equal(np.asarray(parts.integration_field_indices),
                                  np.arange(N_XY**2))
    np.testing.assert_allclose(np.asarray(parts.integration_areas, dtype=np.float64),
                               np.full(N_XY**2, 1.0 / N_XY**2), rtol=2e-6)


# ---------------------------------------------------------------------------
# Sampler support (model level): background points never leave A
# ---------------------------------------------------------------------------

def _z_truth(model, key=3):
    tr = handlers.trace(handlers.seed(model.model,
                                      jax.random.PRNGKey(key))).get_trace(model.args)
    return {k: np.asarray(tr[k]["value"]) for k in
            ("a_0", "z_temporal", "z_seasonal", "z_spatial")}


@needs_decoder
def test_sim_cox_background_supported_on_A():
    """_sim_cox itself (not simulate()'s downstream A-filter) draws every
    no-covariate background point inside A: the sampler uses the clipped
    geometries, so points outside A never exist to be filtered. RED
    pre-3c-1: boundary cells are sampled over the full cell and the notch
    receives points with overwhelming probability."""
    data = _events_inside(L_HALF_POLY)
    m = LGCP_Model(data, L_HALF_GDF, T_DAYS, a_0=dist.Normal(0, 5))
    truth = _z_truth(m)
    truth["a_0"] = np.float32(truth["a_0"] + 1.5)
    truth = m._decode_field_parameters(dict(truth))
    s = m._sim_cox(truth, rng=np.random.default_rng(7))
    assert len(s) > 100, "draw too small to probe the notch"
    inside = np.array([L_HALF_POLY.buffer(1e-9).contains(p)
                       for p in gpd.points_from_xy(s[:, 0], s[:, 1])])
    assert inside.all(), (
        f"{(~inside).sum()}/{len(s)} background points outside A: sampler "
        f"support is not the clipped geometry")


@needs_decoder
def test_sim_cox_count_rate_matches_clipped_compensator():
    """The Poisson count mean of the background draw equals Ig * Ih with Ih
    the CLIPPED spatial integral: simulated counts and the likelihood
    compensator charge the same mass (I1 on the polygon regime). Monte
    Carlo check with a seeded generator, 5-sigma band. RED pre-3c-1: the
    sampler's Ih overcharges by the unclipped boundary mass (~7% on this
    L), then simulate()'s filter silently discards the excess."""
    from bstpp.likelihood import seasonal_time_integral, spatial_refinement_integral
    data = _events_inside(L_HALF_POLY)
    m = LGCP_Model(data, L_HALF_GDF, T_DAYS, a_0=dist.Normal(0, 5))
    truth = m._decode_field_parameters(_z_truth(m))
    _, Ig = seasonal_time_integral(truth["a_0"], truth["f_t"], truth["f_a"],
                                   m.args["season_overlap"])
    fi = np.asarray(m.args["integration_field_indices"])
    cell_geoms = m.comp_grid.set_index("comp_grid_id").geometry
    clipped = np.array([cell_geoms.loc[c].intersection(L_HALF_POLY).area
                        for c in fi])
    Ih_clipped = float(spatial_refinement_integral(
        np.asarray(truth["f_xy"], dtype=np.float64), fi, clipped))
    mean = float(Ig) * Ih_clipped
    rng = np.random.default_rng(123)
    counts = [len(m._sim_cox(truth, rng=rng)) for _ in range(200)]
    z = (np.mean(counts) - mean) / np.sqrt(mean / len(counts))
    assert abs(z) < 5.0, (
        f"background count mean {np.mean(counts):.2f} vs clipped-compensator "
        f"mean {mean:.2f} (z={z:.1f}): sampler and compensator charge "
        f"different mass")
