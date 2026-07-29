"""TruncatedLogNormal must sample on the declared positive interval (CF).

Pre-3f audit: ``sample`` returned a TruncatedNormal draw on the log scale
without exponentiating, so draws disagreed with ``support`` / ``log_prob``.
Choosing to truncate a user prior remains an SC (D-28); this file pins the
sampler/density consistency of the already-declared truncated LogNormal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer.initialization import init_to_median

from bstpp.excitation_support import TruncatedLogNormal, truncate_sigmax_2_prior


def _d(loc=0.0, scale=0.5, low=0.01, high=4.0, dtype=jnp.float32):
    return TruncatedLogNormal(
        jnp.asarray(loc, dtype=dtype),
        jnp.asarray(scale, dtype=dtype),
        jnp.asarray(low, dtype=dtype),
        jnp.asarray(high, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_sample_scalar_and_batched_within_bounds(dtype):
    if dtype is jnp.float64 and not jax.config.read("jax_enable_x64"):
        pytest.skip("float64 not enabled")
    d = _d(dtype=dtype)
    key = random.PRNGKey(0)
    scalar = d.sample(key)
    assert scalar.shape == ()
    assert jnp.isfinite(scalar)
    assert float(d.low) <= float(scalar) <= float(d.high)

    batched = TruncatedLogNormal(
        jnp.asarray([0.0, 0.5], dtype=dtype),
        jnp.asarray([0.5, 0.4], dtype=dtype),
        jnp.asarray([0.01, 0.05], dtype=dtype),
        jnp.asarray([4.0, 9.0], dtype=dtype),
    )
    draws = batched.sample(random.PRNGKey(1), sample_shape=(32,))
    assert draws.shape == (32, 2)
    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all(draws >= batched.low))
    assert bool(jnp.all(draws <= batched.high))


def test_sample_shape_respected():
    d = _d()
    draws = d.sample(random.PRNGKey(2), sample_shape=(7, 3))
    assert draws.shape == (7, 3)
    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all(draws >= d.low))
    assert bool(jnp.all(draws <= d.high))


@pytest.mark.parametrize(
    "low,high",
    [
        (0.01, 4.0),          # typical sigma^2 truncation interval
        (1e-3, 1e-2),         # narrow, both bounds << 1
        (1.0, 100.0),         # wide, above unit
        (0.25, 0.30),         # modest width (float32 ULP-honest)
    ],
)
def test_supported_truncation_intervals_finite_in_bounds(low, high):
    d = TruncatedLogNormal(0.0, 1.0, low, high)
    draws = d.sample(random.PRNGKey(3), sample_shape=(64,))
    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all(draws >= d.low))
    assert bool(jnp.all(draws <= d.high))


def test_truncate_sigmax_2_lognormal_and_retruncate_sample_ok():
    """Both truncate_sigmax_2_prior branches that yield TruncatedLogNormal."""
    ln = truncate_sigmax_2_prior(dist.LogNormal(0.0, 0.5), 0.1, 2.0)
    assert isinstance(ln, TruncatedLogNormal)
    d1 = ln.sample(random.PRNGKey(4), sample_shape=(32,))
    assert bool(jnp.all((d1 >= ln.low) & (d1 <= ln.high)))

    narrower = truncate_sigmax_2_prior(ln, 0.2, 1.5)
    assert isinstance(narrower, TruncatedLogNormal)
    d2 = narrower.sample(random.PRNGKey(5), sample_shape=(32,))
    assert bool(jnp.all((d2 >= narrower.low) & (d2 <= narrower.high)))


def test_init_to_median_returns_value_on_support():
    d = _d(loc=-1.0, scale=0.8, low=0.05, high=2.5)
    site = {
        "name": "sigmax_2",
        "type": "sample",
        "fn": d,
        "args": (),
        "kwargs": {
            "rng_key": random.PRNGKey(11),
            "sample_shape": (),
        },
        "value": None,
        "intermediates": [],
        "cond_indep_stack": [],
        "is_observed": False,
    }
    init = init_to_median(num_samples=31)(site)
    assert jnp.isfinite(init)
    assert float(d.low) <= float(init) <= float(d.high)


def test_log_prob_finite_near_both_bounds():
    d = _d(loc=0.0, scale=0.5, low=0.01, high=4.0)
    # Interior points near each bound (avoid exact endpoints where density
    # may still be finite but bijectors use open intervals).
    near_lo = d.low * (1.0 + jnp.asarray(1e-4, dtype=d.low.dtype))
    near_hi = d.high * (1.0 - jnp.asarray(1e-4, dtype=d.high.dtype))
    lp_lo = d.log_prob(near_lo)
    lp_hi = d.log_prob(near_hi)
    assert jnp.isfinite(lp_lo)
    assert jnp.isfinite(lp_hi)
    # Unnormalized LogNormal log_prob minus finite log Z must be finite.
    base = dist.LogNormal(d.loc, d.scale)
    assert float(lp_lo) == pytest.approx(
        float(base.log_prob(near_lo) - d._log_z()), rel=0, abs=1e-5)
    assert float(lp_hi) == pytest.approx(
        float(base.log_prob(near_hi) - d._log_z()), rel=0, abs=1e-5)


def test_log_z_finite_float32_narrow_tail():
    """float32-inert 1e-300 guard must not poison normalization."""
    # Interval deep in the right tail of LogNormal(0, 0.25): CDF mass tiny.
    d = TruncatedLogNormal(
        jnp.float32(0.0), jnp.float32(0.25),
        jnp.float32(20.0), jnp.float32(25.0),
    )
    lz = d._log_z()
    assert jnp.isfinite(lz)
    draws = d.sample(random.PRNGKey(6), sample_shape=(16,))
    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all((draws >= d.low) & (draws <= d.high)))
    assert bool(jnp.all(jnp.isfinite(d.log_prob(draws))))


def test_sample_matches_value_scale_not_log_scale():
    """Regression: pre-fix sample returned log(sigmax_2) (often negative)."""
    d = _d(loc=0.0, scale=0.5, low=0.25, high=4.0)
    draws = np.asarray(d.sample(random.PRNGKey(7), sample_shape=(200,)))
    assert np.all(draws > 0.0)
    # Median of a unit-centered LogNormal truncated to [0.25, 4] is O(1),
    # not O(log) near 0 / negative.
    assert float(np.median(draws)) > 0.2
