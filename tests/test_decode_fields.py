"""Unit tests for bstpp/decode_fields.py.

The decode functions are compared against direct vae_functions application
(the pre-extraction expression) and checked for the calibration-gain contract
and differentiability w.r.t. z -- the decode path sits inside NUTS/SVI.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

import bstpp
from bstpp.main import Hawkes_Model
from bstpp.decode_fields import (decode_temporal_field, decode_seasonal_field,
                                 decode_spatial_field)
from bstpp.vae_functions import (vae_decoder_temporal, vae_decoder_seasonal,
                                 vae_decoder_spatial)

_SEASONAL_DECODER = os.path.join(os.path.dirname(bstpp.__file__), "decoders",
                                 "decoder_1d_T24_circ_small_l8")
needs_decoder = pytest.mark.skipif(
    not os.path.isfile(_SEASONAL_DECODER),
    reason="seasonal decoder artifact is absent")


@pytest.fixture(scope="module")
def cox_args():
    T_DAYS = 2.5 * 365.0
    rng = np.random.RandomState(0)
    data = pd.DataFrame({"X": rng.uniform(0.05, 0.95, 40),
                         "Y": rng.uniform(0.05, 0.95, 40),
                         "T": np.sort(rng.uniform(0, T_DAYS, 40))})
    m = Hawkes_Model(data, np.array([[0., 1.], [0., 1.]]), T_DAYS,
                     cox_background=True,
                     a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
                     beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
    return m.args


@needs_decoder
def test_decode_fields_match_direct_decoder_application(cox_args):
    a = cox_args
    rng = np.random.default_rng(6)

    z_t = jnp.asarray(rng.normal(0, 1, a["z_dim_temporal"]).astype(np.float32))
    got = decode_temporal_field(z_t, a["decoder_params_temporal"],
                                a["hidden_dim_temporal"], a["n_t"])
    want = vae_decoder_temporal(a["hidden_dim_temporal"], a["n_t"])[1](
        a["decoder_params_temporal"], z_t)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    z_s = jnp.asarray(rng.normal(0, 1, a["z_dim_seasonal"]).astype(np.float32))
    got = decode_seasonal_field(z_s, a["decoder_params_seasonal"],
                                a["hidden_dim1_seasonal"],
                                a["hidden_dim2_seasonal"], a["n_s"])
    want = vae_decoder_seasonal(a["hidden_dim1_seasonal"],
                                a["hidden_dim2_seasonal"], a["n_s"])[1](
        a["decoder_params_seasonal"], z_s)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    z_xy = jnp.asarray(rng.normal(0, 1, a["z_dim_spatial"]).astype(np.float32))
    got = decode_spatial_field(z_xy, a["decoder_params_spatial"],
                               a["hidden_dim1_spatial"], a["hidden_dim2_spatial"],
                               a["n_xy"], a["sp_var_mu"])
    want = jnp.exp(a["sp_var_mu"]) * vae_decoder_spatial(
        a["hidden_dim1_spatial"], a["hidden_dim2_spatial"], a["n_xy"])[1](
        a["decoder_params_spatial"], z_xy)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@needs_decoder
def test_spatial_calibration_gain_contract(cox_args):
    """f(sp_var_mu) = exp(sp_var_mu - mu0) * f(mu0): the gain is a pure scalar
    multiplier paired with the decoder, applied in exactly one place."""
    a = cox_args
    z = jnp.asarray(np.random.default_rng(7).normal(
        0, 1, a["z_dim_spatial"]).astype(np.float32))
    f2 = decode_spatial_field(z, a["decoder_params_spatial"],
                              a["hidden_dim1_spatial"], a["hidden_dim2_spatial"],
                              a["n_xy"], 2.0)
    f0 = decode_spatial_field(z, a["decoder_params_spatial"],
                              a["hidden_dim1_spatial"], a["hidden_dim2_spatial"],
                              a["n_xy"], 0.0)
    np.testing.assert_allclose(np.asarray(f2), np.exp(2.0) * np.asarray(f0),
                               rtol=2e-6)


@needs_decoder
def test_decode_fields_differentiable_in_z(cox_args):
    a = cox_args
    z = jnp.zeros(a["z_dim_spatial"], dtype=jnp.float32)
    g = jax.grad(lambda zz: jnp.sum(decode_spatial_field(
        zz, a["decoder_params_spatial"], a["hidden_dim1_spatial"],
        a["hidden_dim2_spatial"], a["n_xy"], a["sp_var_mu"]) ** 2))(z)
    assert np.all(np.isfinite(np.asarray(g)))
