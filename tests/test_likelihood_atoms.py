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
import jax
import jax.numpy as jnp

from bstpp.likelihood import aggregate_pair_trigger_values


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
