"""Tests for the O(n log n + P) rewrite of aligned_difference_pairs.

- test_equivalence_vs_dense : pairwise-identical (as sets) to the old dense n^2
  implementation across sizes, tie/duplicate timestamps, and spatial windows.
- test_likelihood_invariance: the Hawkes 'loglik' is unchanged when the pairs are
  built by the new vs the old implementation.
- test_scale_smoke        : 20k events run in seconds (no (n, n) allocation) and
  emit a pair count consistent with the window fraction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import handlers
import pytest

from bstpp.utils import aligned_difference_pairs
from bstpp.main import Hawkes_Model


# --- private copy of the ORIGINAL dense implementation, as the reference oracle ---
def _dense_pairs(t, x, y, window, spatial_window=None, axis_scales=(1.0, 1.0)):
    window = float(window)
    if spatial_window is not None:
        spatial_window = float(spatial_window)

    t = jnp.reshape(jnp.asarray(t), (t.shape[0], 1))
    x = jnp.reshape(jnp.asarray(x), (x.shape[0], 1))
    y = jnp.reshape(jnp.asarray(y), (y.shape[0], 1))

    t_diff = t - t.T
    x_diff = x - x.T
    y_diff = y - y.T

    t_mask = (t_diff > 0) & (t_diff <= window)
    if spatial_window is not None:
        # REAL-unit per-axis box truncation (semantics change from internal
        # Euclidean, signed off with the real-unit trigger contract -- see
        # aligned_difference_pairs / within_real_box_window)
        spatial_dist = jnp.maximum(jnp.abs(x_diff) * float(axis_scales[0]),
                                   jnp.abs(y_diff) * float(axis_scales[1]))
        mask = t_mask & (spatial_dist <= spatial_window)
    else:
        mask = t_mask

    indices = jnp.where(mask)
    coords = jnp.stack(indices, axis=-1)
    return coords, t_diff[indices], x_diff[indices], y_diff[indices]


def _canonical(coords, t_vals, x_vals, y_vals):
    """Sort pairs by (i, j) so two set-equal results become elementwise comparable."""
    coords = np.asarray(coords).reshape(-1, 2)
    t_vals = np.asarray(t_vals).reshape(-1)
    x_vals = np.asarray(x_vals).reshape(-1)
    y_vals = np.asarray(y_vals).reshape(-1)
    if coords.shape[0] == 0:
        return coords, t_vals, x_vals, y_vals
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order], t_vals[order], x_vals[order], y_vals[order]


def _assert_same(a, b):
    ca, ta, xa, ya = _canonical(*a)
    cb, tb, xb, yb = _canonical(*b)
    assert ca.shape == cb.shape
    assert np.array_equal(ca, cb)
    assert np.allclose(ta, tb, atol=1e-5)
    assert np.allclose(xa, xb, atol=1e-5)
    assert np.allclose(ya, yb, atol=1e-5)


def test_equivalence_vs_dense():
    rng = np.random.RandomState(7)
    cases = []
    for n in (1, 2, 50, 300):
        # continuous times
        cases.append(rng.uniform(0, 10, n))
        # duplicated timestamps -> exercises the strict dt > 0 tie rule
        dup = rng.randint(0, max(1, n // 3) + 1, n).astype(float)
        cases.append(np.sort(dup))
    assert len(cases) >= 8

    n_checks = 0
    for t in cases:
        n = t.shape[0]
        x = rng.uniform(0, 1, n)
        y = rng.uniform(0, 1, n)
        trange = float(t.max() - t.min()) if n else 0.0
        for window in (max(trange * 0.3, 0.5), trange + 5.0):   # smaller & larger than range
            # exercise the real-unit box mask with non-trivial, UNEQUAL scales
            for spatial_window, scales in ((None, (1.0, 1.0)), (0.2, (1.0, 1.0)),
                                           (0.5, (4.0, 1.0))):
                new = aligned_difference_pairs(t, x, y, window,
                                               spatial_window=spatial_window,
                                               axis_scales=scales)
                old = _dense_pairs(t, x, y, window, spatial_window=spatial_window,
                                   axis_scales=scales)
                _assert_same(new, old)
                n_checks += 1
    assert n_checks >= 10   # >= 10 random datasets/config combinations compared


def test_tie_rule_excludes_equal_times():
    t = np.array([0.0, 1.0, 1.0, 2.0])
    x = np.zeros(4); y = np.zeros(4)
    coords, tv, _, _ = aligned_difference_pairs(t, x, y, window=5.0)
    coords = np.asarray(coords)
    # no self pairs, and equal-time pair (1,2) must be absent in either direction
    assert not any((i == j) for i, j in coords)
    assert (1, 2) not in {tuple(c) for c in coords}
    assert (2, 1) not in {tuple(c) for c in coords}
    assert (np.asarray(tv) > 0).all()


def test_likelihood_invariance():
    T = 2.5 * 365.0
    rng = np.random.RandomState(0)
    n = 40
    data = pd.DataFrame({
        "X": rng.uniform(0.05, 0.95, n),
        "Y": rng.uniform(0.05, 0.95, n),
        "T": np.sort(rng.uniform(0, T, n)),
    })
    A = np.array([[0.0, 1.0], [0.0, 1.0]])
    priors = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
                  beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
    m = Hawkes_Model(data, A, T, cox_background=False, **priors)

    params = {k: np.float32(v) for k, v in
              dict(a_0=1.0, alpha=0.3, beta=2.0, sigmax_2=0.1).items()}
    t_ev = m.args["t_events"]
    xy = m.args["xy_events"]
    window = m.args["window"]

    def loglik_with(pairs_fn):
        coords, t_vals, x_vals, y_vals = pairs_fn(t_ev, xy[0], xy[1], window)
        args = dict(m.args)
        args.update(coords=coords, t_vals=t_vals, x_vals=x_vals, y_vals=y_vals)
        seeded = handlers.substitute(handlers.seed(m.model, jax.random.PRNGKey(0)), params)
        tr = handlers.trace(seeded).get_trace(args)
        return float(np.asarray(tr["loglik"]["value"]))

    ll_new = loglik_with(aligned_difference_pairs)
    ll_old = loglik_with(_dense_pairs)
    assert np.isfinite(ll_new)
    assert abs(ll_new - ll_old) < 1e-5


def test_scale_smoke():
    n = 20_000
    rng = np.random.RandomState(1)
    t = np.sort(rng.uniform(0, 1.0, n))
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    f = 0.01                      # window covers ~1% of the [0, 1] range
    window = f * 1.0

    start = time.time()
    coords, t_vals, x_vals, y_vals = aligned_difference_pairs(t, x, y, window)
    elapsed = time.time() - start
    assert elapsed < 20.0         # seconds, not the minutes an (n, n) build would take

    P = np.asarray(coords).shape[0]
    assert P == np.asarray(t_vals).shape[0]
    assert P < n * n              # obviously not the dense count

    # Expected ordered close-pair count for n uniform points: one per unordered
    # pair within the window. P(|t_a - t_b| <= w) = 2f - f^2 for w = f*R, R = 1.
    p = 2 * f - f * f
    M = n * (n - 1) / 2.0
    expected = M * p
    sigma = np.sqrt(M * p * (1 - p))
    assert abs(P - expected) < 3 * sigma
