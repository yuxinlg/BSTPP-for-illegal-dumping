"""Unit tests for bstpp/likelihood.py atoms.

Each atom is tested against an INDEPENDENT NumPy/SciPy reference
implementation on random inputs (never against the code path it was lifted
from), plus finite-gradient checks -- these atoms live inside NUTS/SVI, so
differentiability is part of the contract, not an implementation detail.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.special import erf
import jax
import jax.numpy as jnp

from bstpp.likelihood import (aggregate_pair_trigger_values,
                              rectangular_excitation_compensator,
                              seasonal_time_integral)
from bstpp.trigger import Temporal_Exponential, Spatial_Symmetric_Gaussian


def test_aggregate_pair_trigger_values_matches_numpy_reference():
    rng = np.random.default_rng(0)
    n, P = 25, 300
    i = rng.integers(0, n, P)
    j = rng.integers(0, n, P)
    coords = np.stack([i, j], axis=1)
    tv = rng.uniform(0.1, 2.0, P).astype(np.float32)
    sv = rng.uniform(0.1, 2.0, P).astype(np.float32)

    ref = np.zeros(n, dtype=np.float64)
    np.add.at(ref, i, tv.astype(np.float64) * sv.astype(np.float64))

    out = np.asarray(aggregate_pair_trigger_values(jnp.asarray(coords),
                                                   jnp.asarray(tv),
                                                   jnp.asarray(sv), n))
    np.testing.assert_allclose(out, ref, rtol=2e-6, atol=2e-6)

    # pair order irrelevant (I3 restated at the atom level)
    perm = rng.permutation(P)
    out_perm = np.asarray(aggregate_pair_trigger_values(
        jnp.asarray(coords[perm]), jnp.asarray(tv[perm]), jnp.asarray(sv[perm]), n))
    np.testing.assert_allclose(out_perm, out, rtol=2e-6, atol=2e-6)

    # differentiable w.r.t. kernel values, finite gradient
    g = jax.grad(lambda t: jnp.sum(aggregate_pair_trigger_values(
        jnp.asarray(coords), t, jnp.asarray(sv), n)) ** 2)(jnp.asarray(tv))
    assert np.all(np.isfinite(np.asarray(g)))


def _reference_compensator(alpha, t, x, y, T, win, bounds, beta, sigmax_2):
    """Independent NumPy/SciPy implementation of the rectangular compensator."""
    x_min, x_max, y_min, y_max = bounds
    temp = alpha * (1.0 - np.exp(-np.minimum(T - t, win) / beta))
    s = np.sqrt(2.0 * sigmax_2)
    mass_x = 0.5 * (erf((x_max - x) / s) + erf((x - x_min) / s))
    mass_y = 0.5 * (erf((y_max - y) / s) + erf((y - y_min) / s))
    return float(np.sum(temp * mass_x * mass_y))


def test_rectangular_excitation_compensator_matches_reference():
    rng = np.random.default_rng(1)
    n = 40
    T, win = 50.0, 7.5
    bounds = (0.0, 1.0, 0.0, 1.0)
    t = np.sort(rng.uniform(0, T, n)).astype(np.float32)
    x = rng.uniform(0, 1, n).astype(np.float32)
    y = rng.uniform(0, 1, n).astype(np.float32)
    alpha, beta, sigmax_2 = 0.4, 2.5, 0.03

    out = float(rectangular_excitation_compensator(
        jnp.float32(alpha), jnp.asarray(t), jnp.stack([jnp.asarray(x), jnp.asarray(y)]),
        T, win, bounds, {"beta": jnp.float32(beta)}, {"sigmax_2": jnp.float32(sigmax_2)},
        Temporal_Exponential({}), Spatial_Symmetric_Gaussian({})))
    ref = _reference_compensator(alpha, t.astype(np.float64), x.astype(np.float64),
                                 y.astype(np.float64), T, win, bounds, beta, sigmax_2)
    np.testing.assert_allclose(out, ref, rtol=2e-5)

    # gradients w.r.t. all continuous kernel parameters must be finite
    def f(a, b, s2):
        return rectangular_excitation_compensator(
            a, jnp.asarray(t), jnp.stack([jnp.asarray(x), jnp.asarray(y)]),
            T, win, bounds, {"beta": b}, {"sigmax_2": s2},
            Temporal_Exponential({}), Spatial_Symmetric_Gaussian({}))
    grads = jax.grad(f, argnums=(0, 1, 2))(jnp.float32(alpha), jnp.float32(beta),
                                           jnp.float32(sigmax_2))
    assert all(np.isfinite(float(g)) for g in grads)


def test_compensator_window_beyond_horizon_is_untruncated():
    """min(T - t, win) = T - t when win >= T: I8b restated at the atom level."""
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 50.0, 30)).astype(np.float32)
    xy = jnp.stack([jnp.asarray(rng.uniform(0, 1, 30).astype(np.float32)),
                    jnp.asarray(rng.uniform(0, 1, 30).astype(np.float32))])
    common = dict(rectangular_bounds=(0.0, 1.0, 0.0, 1.0),
                  temporal_parameters={"beta": jnp.float32(2.0)},
                  spatial_parameters={"sigmax_2": jnp.float32(0.05)},
                  temporal_trigger=Temporal_Exponential({}),
                  spatial_trigger=Spatial_Symmetric_Gaussian({}))
    a = rectangular_excitation_compensator(jnp.float32(0.3), jnp.asarray(t), xy,
                                           50.0, 50.0, **common)
    b = rectangular_excitation_compensator(jnp.float32(0.3), jnp.asarray(t), xy,
                                           50.0, 1e9, **common)
    np.testing.assert_allclose(float(a), float(b), rtol=1e-6)


def test_seasonal_time_integral_matches_numpy_reference():
    rng = np.random.default_rng(3)
    n_t, n_s = 50, 24
    W = rng.uniform(0, 0.1, (n_t, n_s)).astype(np.float32)
    f_t = rng.normal(0, 0.5, n_t).astype(np.float32)
    f_a = rng.normal(0, 0.5, n_s).astype(np.float32)
    a_0 = np.float32(0.3)

    mass, total = seasonal_time_integral(jnp.float32(a_0), jnp.asarray(f_t),
                                         jnp.asarray(f_a), jnp.asarray(W))
    ref_mass = W.astype(np.float64) @ np.exp(f_a.astype(np.float64))
    ref_total = float(np.sum(np.exp(np.float64(a_0) + f_t.astype(np.float64)) * ref_mass))
    np.testing.assert_allclose(np.asarray(mass), ref_mass, rtol=2e-6)
    np.testing.assert_allclose(float(total), ref_total, rtol=2e-6)

    # gradients w.r.t. all field inputs finite
    g = jax.grad(lambda a, ft, fa: seasonal_time_integral(a, ft, fa, jnp.asarray(W))[1],
                 argnums=(0, 1, 2))(jnp.float32(a_0), jnp.asarray(f_t), jnp.asarray(f_a))
    assert all(np.all(np.isfinite(np.asarray(x))) for x in g)
