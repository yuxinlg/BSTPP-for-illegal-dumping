"""Pre-3f standardization API (OP-3/OP-4 settled).

Accepted values: None (default, off) and ``"domain_area"``.
Legacy booleans are rejected explicitly — never silently reinterpreted.
Arbitrary user-supplied weights remain deferred.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import geopandas as gpd
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

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


def _model(standardize="DEFAULT", cov=COV, domain=TRI_GDF):
    kw = {} if standardize == "DEFAULT" else {"standardize_cov": standardize}
    return Hawkes_Model(_data(), domain, T_DAYS, cox_background=False,
                        excitation_support="rectangle",
                        spatial_cov=cov, cov_names=["v"], **PRIORS, **kw)


def test_default_is_off_and_preserves_values():
    m = _model("DEFAULT")
    rep = m.standardization
    assert rep["method"] == "none"
    assert rep["mean"] is None and rep["scale"] is None
    np.testing.assert_array_equal(np.asarray(m.args["spatial_cov"]),
                                  COV[["v"]].values)


def test_explicit_none_preserves_values():
    m = _model(None)
    rep = m.standardization
    assert rep["method"] == "none"
    np.testing.assert_array_equal(np.asarray(m.args["spatial_cov"]),
                                  COV[["v"]].values)


def test_boolean_true_rejected():
    with pytest.raises(ValueError, match="boolean"):
        _model(True)


def test_boolean_false_rejected():
    with pytest.raises(ValueError, match="boolean"):
        _model(False)


def test_domain_area_weights_by_clipped_areas():
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
    cov2 = COV.copy()
    cov2.loc[2, "v"] = 999.0
    a = np.asarray(_model("domain_area").args["spatial_cov"])[:, 0]
    b = np.asarray(_model("domain_area", cov=cov2).args["spatial_cov"])[:, 0]
    keep = [0, 1, 3]
    np.testing.assert_allclose(b[keep], a[keep], rtol=1e-12)


def test_domain_area_constant_column_fails_loud():
    cov2 = COV.copy()
    cov2["v"] = [2.0, 2.0, 5.0, 2.0]
    with pytest.raises(ValueError, match="zero variance"):
        _model("domain_area", cov=cov2)


def test_unknown_method_string_rejected():
    with pytest.raises(ValueError, match="standardize_cov"):
        _model("area")
