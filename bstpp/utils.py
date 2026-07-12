# general libraries
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

# JAX
import jax
import jax.numpy as jnp
from jax import random, lax, jit, ops
#from jax.experimental import stax

from functools import partial


#@title

def dist_euclid(x, z):
    x = jnp.array(x) 
    z = jnp.array(z)
    if len(x.shape)==1:
        x = x.reshape(x.shape[0], 1)
    if len(z.shape)==1:
        z = x.reshape(x.shape[0], 1)
    n_x, m = x.shape
    n_z, m_z = z.shape
    assert m == m_z
    delta = jnp.zeros((n_x,n_z))
    for d in jnp.arange(m):
        x_d = x[:,d]
        z_d = z[:,d]
        delta += (x_d[:,jnp.newaxis] - z_d)**2
    return jnp.sqrt(delta)


def exp_sq_kernel(x, z, var, length, noise, jitter=1.0e-6):
    dist = dist_euclid(x, z)
    deltaXsq = jnp.power(dist/ length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k

    
def aligned_difference_pairs(t, x, y, window, spatial_window=None):
    """Emit the (receiver i, source j) excitation pairs with 0 < t[i]-t[j] <= window.

    Contract (identical to the former dense n x n implementation):
      - returns (coords, t_vals, x_vals, y_vals) as jnp arrays;
      - coords is (P, 2) with coords[:, 0] = i (receiver / later event) and
        coords[:, 1] = j (source / earlier event), in the ORIGINAL event order;
      - t_vals = t[i] - t[j], strictly > 0 and <= window (equal times excluded);
      - x_vals = x[i] - x[j], y_vals = y[i] - y[j];
      - if spatial_window is not None, only pairs with
        sqrt(x_vals**2 + y_vals**2) <= spatial_window are kept.

    The pair ORDERING within the returned arrays is NOT part of the contract:
    the likelihood aggregates them with segment_sum on coords[:, 0], which is
    order-invariant. This is why a sort-based construction is a drop-in.

    Construction is O(n log n + P) in time and O(n + P) in memory (P = number of
    emitted pairs): sort by time, then for each event use searchsorted to find the
    contiguous block of earlier events within the window, instead of materializing
    the dense (n, n) difference matrices.
    """
    window = float(window)
    if spatial_window is not None:
        spatial_window = float(spatial_window)

    t = np.asarray(t).reshape(-1)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = t.shape[0]

    if n == 0:
        return (jnp.zeros((0, 2), dtype=jnp.int32), jnp.zeros((0,)),
                jnp.zeros((0,)), jnp.zeros((0,)))

    order = np.argsort(t, kind='stable')
    ts = t[order]

    # For receiver at sorted position k, valid sources are the sorted positions
    # [lo[k], hi[k]): ts[j] >= ts[k]-window (dt <= window, side='left' keeps
    # dt == window) and ts[j] < ts[k] (strictly earlier, so dt > 0 and ties are
    # excluded because side='left' places equal times at/after hi[k]).
    lo = np.searchsorted(ts, ts - window, side='left')
    hi = np.searchsorted(ts, ts, side='left')
    counts = np.maximum(hi - lo, 0)

    P = int(counts.sum())
    if P == 0:
        return (jnp.zeros((0, 2), dtype=jnp.int32), jnp.zeros((0,)),
                jnp.zeros((0,)), jnp.zeros((0,)))

    rows = np.repeat(np.arange(n), counts)              # receiver positions (sorted)
    starts = np.repeat(lo, counts)                       # block start per emitted pair
    offs = np.arange(P) - np.repeat(np.cumsum(counts) - counts, counts)
    srcs = starts + offs                                 # source positions (sorted)

    i_idx = order[rows]                                  # receiver, ORIGINAL order
    j_idx = order[srcs]                                  # source, ORIGINAL order

    t_vals = t[i_idx] - t[j_idx]
    x_vals = x[i_idx] - x[j_idx]
    y_vals = y[i_idx] - y[j_idx]

    if spatial_window is not None:
        keep = np.sqrt(x_vals**2 + y_vals**2) <= spatial_window
        i_idx, j_idx = i_idx[keep], j_idx[keep]
        t_vals, x_vals, y_vals = t_vals[keep], x_vals[keep], y_vals[keep]

    coords = np.stack((i_idx, j_idx), axis=-1)
    return (jnp.asarray(coords), jnp.asarray(t_vals),
            jnp.asarray(x_vals), jnp.asarray(y_vals))
