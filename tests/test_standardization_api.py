"""Phase 3c standardization reporting + explicit "domain_area" convenience.

Doc section 10.c / 7.3 (class API), bounded by OP-3/OP-4 (both "not 3c"):
the default standardize_cov=True count-weighted behavior is BIT-UNCHANGED
and the full method-string API is NOT settled here. What 3c lands:

- REPORTING (D-10): the model always records whether/how it standardized
  -- model.standardization = {"method": "count"|"none"|"domain_area",
  "columns": ..., "mean": ..., "scale": ...} with mean/scale invertible
  (X ≈ standardized * scale + mean), None for method "none".
- The one narrow explicit convenience (D-11, exact intersection areas
  exist): standardize_cov="domain_area" weights mean and variance by the
  clipped covariate areas |C_c ∩ A|, so cells with no domain mass do not
  influence the standardization. New path, fail-loud: zero total weight or
  a zero-variance column raises.
- Any other string is rejected loudly (no silent fallback).

RED (pre-commit): every test fails -- model.standardization does not
exist, "domain_area" raises nothing/TypeError, and invalid strings are
silently truthy (legacy bool coercion).
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
import numpyro.distributions as dist
import pytest

from bstpp.main import Hawkes_Model

T_DAYS = 200.0
TRI_POLY = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
TRI_GDF = gpd.GeoDataFrame({"geometry": [TRI_POLY]})
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))

COV = gpd.GeoDataFrame(
    {"v": [0.5, -1.0, 1.5, -0.5]},
    geometry=[box(0, 0, 0.5, 0.5), box(0.5, 0, 1, 0.5),
              box(0, 0.5, 0.5, 1), box(0.5, 0.5, 1, 1)])
# clipped |C_c ∩ A| weights on the triangle y <= x (see test_clipped_support)
W_TRI = np.array([0.125, 0.25, 0.0, 0.125])


def _data(n=15, seed=4):
    r = np.random.RandomState(seed)
    pts = []
    while len(pts) < n:
        x, y = r.uniform(0.05, 0.95), r.uniform(0.05, 0.95)
        if y < x - 0.03:
            pts.append((x, y))
    return pd.DataFrame({"X": [p[0] for p in pts],
                         "Y": [p[1] for p in pts],
                         "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n))})


def _model(standardize=True, cov=COV, domain=TRI_GDF):
    kw = {} if standardize == "DEFAULT" else {"standardize_cov": standardize}
    return Hawkes_Model(_data(), domain, T_DAYS, cox_background=False,
                        excitation_support="rectangle",
                        spatial_cov=cov, cov_names=["v"], **PRIORS, **kw)


def test_default_count_standardization_is_reported_and_bit_unchanged():
    """The unspecified default remains the legacy count-weighted z-score
    (OP-3 not flipped in 3c), now REPORTED as method 'count' with
    invertible mean/scale."""
    m = _model("DEFAULT")
    rep = m.standardization
    assert rep["method"] == "count"
    assert rep["columns"] == ["v"]
    X = COV[["v"]].values.astype(float)
    legacy = (X - X.mean(axis=0)) / (X.var(axis=0) ** 0.5)
    np.testing.assert_array_equal(np.asarray(m.args["spatial_cov"]), legacy)
    np.testing.assert_allclose(
        np.asarray(m.args["spatial_cov"]) * rep["scale"] + rep["mean"], X,
        rtol=1e-12)


def test_off_preserves_values_and_reports_none():
    m = _model(False)
    rep = m.standardization
    assert rep["method"] == "none"
    assert rep["mean"] is None and rep["scale"] is None
    np.testing.assert_array_equal(np.asarray(m.args["spatial_cov"]),
                                  COV[["v"]].values)


def test_domain_area_weights_by_clipped_areas():
    """standardize_cov='domain_area': mean/variance weighted by |C_c ∩ A|,
    independently recomputed here from the analytic triangle weights."""
    m = _model("domain_area")
    rep = m.standardization
    assert rep["method"] == "domain_area"
    X = COV["v"].values.astype(float)
    mean = (W_TRI * X).sum() / W_TRI.sum()
    var = (W_TRI * (X - mean) ** 2).sum() / W_TRI.sum()
    expected = (X - mean) / var ** 0.5
    np.testing.assert_allclose(np.asarray(m.args["spatial_cov"])[:, 0],
                               expected, rtol=1e-12)
    np.testing.assert_allclose(rep["mean"], [mean], rtol=1e-12)
    np.testing.assert_allclose(rep["scale"], [var ** 0.5], rtol=1e-12)


def test_domain_area_zero_weight_cell_has_no_influence():
    """A covariate cell with no domain mass (the upper-left quadrant on the
    triangle) must not influence the standardization: perturbing its value
    leaves every other cell's standardized value unchanged."""
    cov2 = COV.copy()
    cov2.loc[2, "v"] = 999.0
    a = np.asarray(_model("domain_area").args["spatial_cov"])[:, 0]
    b = np.asarray(_model("domain_area", cov=cov2).args["spatial_cov"])[:, 0]
    keep = [0, 1, 3]
    np.testing.assert_allclose(b[keep], a[keep], rtol=1e-12)


def test_domain_area_constant_column_fails_loud():
    """New path, fail-loud (unlike the legacy z-score's silent NaN): a
    column constant over the positive-weight cells raises."""
    cov2 = COV.copy()
    cov2["v"] = [2.0, 2.0, 5.0, 2.0]  # varies only on the zero-weight cell
    with pytest.raises(ValueError, match="zero variance"):
        _model("domain_area", cov=cov2)


def test_unknown_method_string_rejected():
    """Any string other than 'domain_area' is rejected loudly -- the legacy
    bool coercion silently treated truthy strings as True."""
    with pytest.raises(ValueError, match="standardize_cov"):
        _model("area")
