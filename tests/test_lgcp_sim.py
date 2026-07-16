"""LGCP simulation-path coverage (Phase 2d).

The Phase 2b simulator rewiring introduced _sample_cells as a Hawkes_Model
method while _sim_cox -- its caller -- lives on Point_Process_Model, so
LGCP_Model.simulate() raised AttributeError; independently, the z->f decode
block lived only in Hawkes_Model.simulate(), so the LGCP path also raised
KeyError('f_t') on any z-only parameter dict (e.g. mcmc.get_samples() output,
which carries sample sites, not deterministics). Every prior simulation test
exercised Hawkes_Model only -- including its Cox configurations -- which is
how a broken base-class path survived a 49-test suite. These tests close that
gap and pin:

- test_lgcp_simulate_runs_and_is_reproducible -> the AttributeError regression,
  plus RNG unification on the LGCP path (byte-identical under a shared seed)
- test_lgcp_simulate_decodes_z_parameters     -> the KeyError regression: a
  z-only dict must simulate, and identically to its pre-decoded f-dict twin
- test_lgcp_simulate_clips_to_polygon         -> the simulate()-side A-filter
  that _sim_cox's docstring promises, now kept by LGCP_Model.simulate() too
  (declared behavior change: previously out-of-polygon points were returned)
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
import jax
import numpyro.distributions as dist
from numpyro import handlers
import pytest

import bstpp
from bstpp.main import LGCP_Model
from bstpp.decode_fields import (decode_temporal_field, decode_seasonal_field,
                                 decode_spatial_field)

_SEASONAL_DECODER = os.path.join(os.path.dirname(bstpp.__file__), "decoders",
                                 "decoder_1d_T24_circ_small_l8")
needs_decoder = pytest.mark.skipif(
    not os.path.isfile(_SEASONAL_DECODER),
    reason="seasonal decoder artifact 'bstpp/decoders/decoder_1d_T24_circ_small_l8' is absent",
)

T_DAYS = 2.5 * 365.0
_rng = np.random.RandomState(0)
_N = 60
DATA = pd.DataFrame({
    "X": _rng.uniform(0.05, 0.95, _N),
    "Y": _rng.uniform(0.05, 0.95, _N),
    "T": np.sort(_rng.uniform(0, T_DAYS, _N)),
})
A_GDF = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]})
# Non-rectangular domain: an L-shape whose bounding rectangle is the unit
# square, so grid cells straddling the notch get sampled over the FULL cell
# by _sim_cox and must be clipped by simulate()'s A-filter.
L_SHAPE = gpd.GeoDataFrame({"geometry": [
    box(0, 0, 1, 0.5).union(box(0, 0.5, 0.5, 1))]})
# Events kept inside the L so ingestion is clean.
_l_rng = np.random.RandomState(4)
_pts = []
while len(_pts) < 50:
    x, y = _l_rng.uniform(0.02, 0.98), _l_rng.uniform(0.02, 0.98)
    if y <= 0.48 or x <= 0.48:
        _pts.append((x, y))
L_DATA = pd.DataFrame({
    "X": [p[0] for p in _pts],
    "Y": [p[1] for p in _pts],
    "T": np.sort(_l_rng.uniform(0, T_DAYS, len(_pts))),
})


def _z_truth(model, key=3):
    """Fixed LGCP latents (z sites only) from a seeded model trace."""
    tr = handlers.trace(handlers.seed(model.model,
                                      jax.random.PRNGKey(key))).get_trace(model.args)
    return {k: np.asarray(tr[k]["value"]) for k in
            ("a_0", "z_temporal", "z_seasonal", "z_spatial")}


def _decode(model, truth):
    """Reference z->f decode through the decode_fields atoms directly."""
    args = model.args
    out = {"a_0": truth["a_0"]}
    out["f_t"] = np.asarray(decode_temporal_field(
        truth["z_temporal"], args["decoder_params_temporal"],
        args["hidden_dim_temporal"], args["n_t"]))[:args["n_t"]]
    out["f_a"] = np.asarray(decode_seasonal_field(
        truth["z_seasonal"], args["decoder_params_seasonal"],
        args["hidden_dim1_seasonal"], args["hidden_dim2_seasonal"],
        args["n_s"]))[:args["n_s"]]
    out["f_xy"] = np.asarray(decode_spatial_field(
        truth["z_spatial"], args["decoder_params_spatial"],
        args["hidden_dim1_spatial"], args["hidden_dim2_spatial"],
        args["n_xy"], args["sp_var_mu"]))
    return out


@needs_decoder
def test_lgcp_simulate_runs_and_is_reproducible():
    """REGRESSION (base-class layering): LGCP_Model.simulate() must run --
    it reaches _sample_cells through Point_Process_Model._sim_cox, so the
    primitive must live on the base class. RED pre-fix:
    AttributeError: 'LGCP_Model' object has no attribute '_sample_cells'.
    Also pins RNG unification on this path: identically seeded fresh
    Generators give byte-identical simulations."""
    m = LGCP_Model(DATA, A_GDF, T_DAYS, a_0=dist.Normal(0, 5))
    truth = _z_truth(m)
    a = m.simulate(parameters=dict(truth), rng=np.random.default_rng(17))
    b = m.simulate(parameters=dict(truth), rng=np.random.default_rng(17))
    assert {"X", "Y", "T"}.issubset(a.columns)
    assert len(a) > 0, "seeded LGCP simulation produced no events; config uninformative"
    Tr = a["T"].values
    assert (Tr >= 0).all() and (Tr <= T_DAYS + 1e-6).all()
    np.testing.assert_array_equal(np.asarray(a[["X", "Y", "T"]], dtype=np.float64),
                                  np.asarray(b[["X", "Y", "T"]], dtype=np.float64))


@needs_decoder
def test_lgcp_simulate_decodes_z_parameters():
    """REGRESSION (missing decode): a z-only parameter dict -- the shape of
    mcmc.get_samples() output -- must simulate, and must equal the simulation
    from its pre-decoded f-dict twin under the same seed (the decode is
    deterministic, so the RNG stream is untouched). RED pre-fix:
    KeyError: 'f_t'."""
    m = LGCP_Model(DATA, A_GDF, T_DAYS, a_0=dist.Normal(0, 5))
    truth = _z_truth(m)
    f_dict = _decode(m, truth)
    a = m.simulate(parameters=dict(truth), rng=np.random.default_rng(11))
    b = m.simulate(parameters=dict(f_dict), rng=np.random.default_rng(11))
    np.testing.assert_array_equal(np.asarray(a[["X", "Y", "T"]], dtype=np.float64),
                                  np.asarray(b[["X", "Y", "T"]], dtype=np.float64))


@needs_decoder
def test_lgcp_simulate_clips_to_polygon():
    """simulate()'s A-filter, promised by _sim_cox's docstring ('boundary
    cells are sampled over the FULL cell and then clipped by simulate()'s
    A-filter'), must hold on the LGCP path: every returned point lies in the
    domain polygon. Uses an L-shaped domain whose bounding rectangle is the
    unit square, so unclipped full-cell sampling puts points in the notch
    with high probability. Declared behavior change: pre-fix (modulo the
    AttributeError) out-of-polygon points were returned."""
    m = LGCP_Model(L_DATA, L_SHAPE, T_DAYS, a_0=dist.Normal(0, 5))
    truth = _z_truth(m)
    # a_0 high enough for a dense draw so the notch is well probed
    truth["a_0"] = np.float32(truth["a_0"] + 1.0)
    s = m.simulate(parameters=dict(truth), rng=np.random.default_rng(7))
    assert len(s) > 50, "clip test draw too small to probe the notch"
    poly = L_SHAPE.geometry.iloc[0]
    inside = np.array([poly.buffer(1e-9).contains(g) for g in s.geometry])
    assert inside.all(), (
        f"{(~inside).sum()}/{len(s)} simulated LGCP points fall outside the "
        f"domain polygon; simulate()'s A-filter is not applied on this path")
