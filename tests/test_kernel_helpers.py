"""dist_euclid / exp_sq_kernel helper repairs (pre-3f CF).

dist_euclid must reshape one-dimensional z from z (not x). exp_sq_kernel
must support rectangular cross covariance (n_x != n_z) and add diagonal
noise/jitter only for a true same-input covariance; equal-length but
distinct x and z remain a cross kernel.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np

from bstpp.utils import dist_euclid, exp_sq_kernel


def test_dist_euclid_different_length_1d_matches_numpy_oracle():
    x = np.array([0.0, 1.0, 2.0])
    z = np.array([0.5, 1.5])
    got = np.asarray(dist_euclid(x, z))
    # Column-vector interpretation of 1-D inputs.
    x2 = x.reshape(-1, 1)
    z2 = z.reshape(-1, 1)
    want = np.sqrt(((x2[:, None, :] - z2[None, :, :]) ** 2).sum(axis=-1))
    assert got.shape == (3, 2)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_exp_sq_kernel_rectangular_cross_covariance():
    x = np.array([[0.0], [1.0], [2.0]])
    z = np.array([[0.5], [1.5]])
    var, length, noise = 1.2, 0.8, 0.05
    k = np.asarray(exp_sq_kernel(x, z, var, length, noise, jitter=1e-6))
    assert k.shape == (3, 2)
    dist = np.sqrt(((x[:, None, :] - z[None, :, :]) ** 2).sum(axis=-1))
    want = var * np.exp(-0.5 * (dist / length) ** 2)
    # Cross kernel: no diagonal noise/jitter term.
    np.testing.assert_allclose(k, want, rtol=1e-6, atol=1e-6)


def test_exp_sq_kernel_equal_length_distinct_inputs_is_cross():
    x = np.array([[0.0], [1.0]])
    z = np.array([[2.0], [3.0]])
    var, length, noise = 1.0, 1.0, 0.1
    k = np.asarray(exp_sq_kernel(x, z, var, length, noise, jitter=1e-3))
    assert k.shape == (2, 2)
    dist = np.sqrt(((x[:, None, :] - z[None, :, :]) ** 2).sum(axis=-1))
    want = var * np.exp(-0.5 * (dist / length) ** 2)
    np.testing.assert_allclose(k, want, rtol=1e-6, atol=1e-6)
    # Must not have added (noise+jitter) on the diagonal of a cross kernel.
    assert not np.allclose(k[0, 0], want[0, 0] + noise + 1e-3)


def test_exp_sq_kernel_same_input_includes_diagonal_noise():
    x = np.array([[0.0], [1.0], [2.0]])
    var, length, noise, jitter = 0.9, 1.1, 0.07, 1e-4
    k = np.asarray(exp_sq_kernel(x, x, var, length, noise, jitter=jitter))
    dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1))
    want = var * np.exp(-0.5 * (dist / length) ** 2)
    want = want + (noise + jitter) * np.eye(3)
    np.testing.assert_allclose(k, want, rtol=1e-6, atol=1e-6)
