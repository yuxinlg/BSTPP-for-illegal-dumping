"""Ingestion-contract pins supplementing the landed CRS-authoritative
detection (Phase 2d).

The Phase 2c series landed CRS-authoritative geographic detection with tests
for both CRS directions in test_identities.py::test_geographic_coordinate_warning.
Two gaps remain, closed here WITHOUT touching the historical test:

1. The landed positive-CRS assertion is VACUOUS: it matches 'geographic',
   which geopandas' own 'Geometry is in a geographic CRS' warning -- emitted
   by the constructor's A.area computation on any geographic-CRS domain --
   satisfies even with the contract warning deleted. Mutation-checked:
   disabling the CRS-branch warn leaves that test green. The pin here matches
   the contract message body ('anisotropic on the ground') instead, so it
   fails when the contract warning does.

2. The CRS-less GeoDataFrame fallback (heuristic path) was untested: the
   landed heuristic directions use an array domain only.

The contract comment gains the shared-linear-unit requirement (one projected
unit on both axes; no X-in-meters / Y-in-feet mixtures -- a projected CRS
guarantees it, raw arrays and CRS-less GeoDataFrames are on the user's
honor). Comment/tests only: no detection logic changes in this commit.
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

T_DAYS = 2.5 * 365.0
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
CONTRACT_MATCH = "anisotropic on the ground"


def _data_in(x0, x1, y0, y1, n=20, seed=5):
    r = np.random.RandomState(seed)
    return pd.DataFrame({
        "X": r.uniform(x0 + 0.01 * (x1 - x0), x1 - 0.01 * (x1 - x0), n),
        "Y": r.uniform(y0 + 0.01 * (y1 - y0), y1 - 0.01 * (y1 - y0), n),
        "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n)),
    })


def test_geographic_crs_contract_warning_is_nonvacuous():
    """The CONTRACT warning (not merely some warning containing
    'geographic') must fire on a geographic-CRS domain, including where the
    bounds heuristic is blind (near-origin box, |lon|,|lat| < 5). Matching
    the message body makes this pin fail if the contract warning is removed
    -- unlike the landed match='geographic', which geopandas' area warning
    satisfies vacuously (mutation-checked)."""
    A = gpd.GeoDataFrame({"geometry": [box(0.1, 0.1, 0.5, 0.5)]},
                         crs="EPSG:4326")
    with pytest.warns(UserWarning, match=CONTRACT_MATCH):
        Hawkes_Model(_data_in(0.1, 0.5, 0.1, 0.5), A, T_DAYS,
                     cox_background=False, **PRIORS)


def test_crsless_gdf_falls_back_to_heuristic():
    """No CRS metadata -> the bounds heuristic still guards GeoDataFrame
    domains: a CRS-less Philadelphia-like degree box warns (contract
    message), a CRS-less metric unit box constructs warning-free. (The
    array-domain heuristic directions remain pinned by
    test_geographic_coordinate_warning.)"""
    A_deg = gpd.GeoDataFrame({"geometry": [box(-75.25, 39.90, -75.15, 40.00)]})
    assert A_deg.crs is None
    with pytest.warns(UserWarning, match=CONTRACT_MATCH):
        Hawkes_Model(_data_in(-75.25, -75.15, 39.90, 40.00), A_deg, T_DAYS,
                     cox_background=False, **PRIORS)
    A_unit = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]})
    assert A_unit.crs is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Hawkes_Model(_data_in(0.0, 1.0, 0.0, 1.0), A_unit, T_DAYS,
                     cox_background=False, **PRIORS)
