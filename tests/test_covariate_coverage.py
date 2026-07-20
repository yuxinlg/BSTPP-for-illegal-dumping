"""Phase 3c coverage contract (IV; D-7, doc section 10.c).

Each required covariate layer must cover the model domain A exactly once:

- a GAP (a region of A no covariate polygon covers) is a violation --
  legacy behavior silently treated gaps as zero-valued covariate regions
  (baseline doc section 5.6), and since the 3c-3 refinement C_c ∩ A_m ∩ A a
  gap silently removes background mass from the compensator instead;
- a positive-area pairwise OVERLAP between covariate polygons (within A) is
  a violation -- membership ties are resolvable (D-22 max-id) but the
  refinement double-charges the overlapped region;
- sub-tolerance pieces (normalized area <= 1e-10, the sliver threshold the
  refinement drops) are SLIVER diagnostics, never violations.

The checks EXPORT the actual offending geometries (ContractCheck.geometry)
-- not merely failing ids -- per the 10.c coverage-contract row.

RED (pre-3c-4): the standalone-validator tests fail on import (the function
does not exist) and the constructor tests fail because gapped/overlapping
layers construct silently.
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
from shapely.geometry import box
import numpyro.distributions as dist
import pytest

from bstpp.main import Hawkes_Model
from bstpp.data_contracts import (DataContractError,
                                  validate_covariate_coverage)

T_DAYS = 200.0
A_RECT = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))


def _data(n=20, seed=5, lo=0.05, hi=0.45):
    r = np.random.RandomState(seed)
    return pd.DataFrame({"X": r.uniform(lo, hi, n),
                         "Y": r.uniform(lo, hi, n),
                         "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n))})


def _model(data, cov, mode="reject"):
    return Hawkes_Model(data, A_RECT, T_DAYS, cox_background=False,
                        spatial_cov=cov, cov_names=["v"],
                        data_contracts=mode, **PRIORS)


TILING = gpd.GeoDataFrame(
    {"v": [0.5, -1.0, 1.5, -0.5]},
    geometry=[box(0, 0, 0.5, 0.5), box(0.5, 0, 1, 0.5),
              box(0, 0.5, 0.5, 1), box(0.5, 0.5, 1, 1)])

# upper-right quadrant missing: gap area 0.25
GAPPED = gpd.GeoDataFrame(
    {"v": [0.5, -1.0, 1.5]},
    geometry=[box(0, 0, 0.5, 0.5), box(0.5, 0, 1, 0.5), box(0, 0.5, 0.5, 1)])

# second polygon overlaps the first on [0.4,0.5]x[0,1]: overlap area 0.1
OVERLAPPING = gpd.GeoDataFrame(
    {"v": [0.5, -1.0]},
    geometry=[box(0, 0, 0.5, 1), box(0.4, 0, 1, 1)])


# ------------------------------------------------------ standalone validator

def test_exact_tiling_has_no_coverage_findings():
    checks = validate_covariate_coverage(TILING, A_RECT)
    assert [c for c in checks if c.kind == "violation"] == []


def test_gap_violation_exports_the_gap_geometry():
    checks = validate_covariate_coverage(GAPPED, A_RECT)
    gaps = [c for c in checks if c.name == "covariate_gap"]
    assert len(gaps) == 1 and gaps[0].kind == "violation"
    assert gaps[0].geometry is not None
    assert gaps[0].geometry.area == pytest.approx(0.25, rel=1e-9)


def test_overlap_violation_exports_the_overlap_geometry_and_rows():
    checks = validate_covariate_coverage(OVERLAPPING, A_RECT)
    ovl = [c for c in checks if c.name == "covariate_overlap"]
    assert len(ovl) == 1 and ovl[0].kind == "violation"
    assert ovl[0].geometry is not None
    assert ovl[0].geometry.area == pytest.approx(0.1, rel=1e-9)
    assert set(ovl[0].indices.tolist()) == {0, 1}


def test_polygon_domain_gap_is_relative_to_A():
    """A layer that covers the DOMAIN exactly passes even when it does not
    tile the bounding rectangle: A is authoritative (D-7), coverage is of
    A, not A_rect."""
    dom = gpd.GeoDataFrame(geometry=[box(0, 0, 0.5, 1).union(box(0.5, 0, 1, 0.5))])
    cov = gpd.GeoDataFrame(
        {"v": [1.0, 2.0]},
        geometry=[box(0, 0, 0.5, 1), box(0.5, 0, 1, 0.5)])
    checks = validate_covariate_coverage(cov, dom)
    assert [c for c in checks if c.kind == "violation"] == []
    # and the same layer against the FULL rectangle domain has a gap
    checks = validate_covariate_coverage(cov, A_RECT)
    assert any(c.name == "covariate_gap" and c.kind == "violation"
               for c in checks)


def test_shared_edges_are_not_overlaps():
    """Exact tilings share boundary segments; zero-area intersections must
    not be flagged."""
    checks = validate_covariate_coverage(TILING, A_RECT)
    assert not any(c.name == "covariate_overlap" for c in checks)


def test_sub_tolerance_gap_is_a_sliver_diagnostic():
    """A gap below the 1e-10 normalized sliver threshold (the refinement
    drops such pieces) is exported as a diagnostic, never a violation."""
    eps = 5e-12
    cov = gpd.GeoDataFrame(
        {"v": [1.0, 2.0]},
        geometry=[box(0, 0, 0.5 - eps, 1), box(0.5, 0, 1, 1)])
    checks = validate_covariate_coverage(cov, A_RECT)
    assert not any(c.kind == "violation" for c in checks)
    slivers = [c for c in checks if c.name == "covariate_sliver"]
    assert len(slivers) == 1 and slivers[0].geometry is not None


def test_invalid_geometry_skips_coverage_analysis():
    """REGRESSION: GEOS set operations raise TopologyException on invalid
    inputs, so an invalid (bowtie) covariate geometry must SKIP coverage
    analysis -- the invalid-geometry violation is validate_covariates'
    finding and must surface as the named DataContractError, not a
    GEOSException from this validator (same skip pattern as
    validate_events)."""
    from shapely.geometry import Polygon
    bowtie = gpd.GeoDataFrame(
        {"v": [1.0]},
        geometry=[Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])])
    assert validate_covariate_coverage(bowtie, A_RECT) == []
    bowtie_dom = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])])
    assert validate_covariate_coverage(TILING, bowtie_dom) == []


# ------------------------------------------------------- constructor wiring

def test_gapped_layer_rejected_by_default():
    """RED pre-3c-4: a gapped covariate layer constructs silently (gap =
    silent zero-covariate region pre-3c-3; silently uncharged region
    post-3c-3)."""
    with pytest.raises(DataContractError, match="covariate_gap"):
        _model(_data(), GAPPED)


def test_overlapping_layer_rejected_by_default():
    data = _data()
    with pytest.raises(DataContractError, match="covariate_overlap"):
        _model(data, OVERLAPPING)


def test_gapped_layer_report_mode_warns_and_constructs():
    """Report mode surfaces the coverage violation loudly, exports the
    geometry on the stored report, and keeps legacy construction (events
    all lie in covered cells here)."""
    with pytest.warns(UserWarning, match="covariate_gap"):
        m = _model(_data(), GAPPED, mode="report")
    gaps = [c for c in m.data_contract_report.checks
            if c.name == "covariate_gap"]
    assert len(gaps) == 1 and gaps[0].geometry is not None


def test_exact_tiling_constructs_under_reject():
    m = _model(_data(), TILING)
    assert m.data_contract_report.ok
