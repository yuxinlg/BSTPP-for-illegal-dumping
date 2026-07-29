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


def _scipy_log_z(loc, scale, low, high):
    """High-precision reference: log(F(high) - F(low)) via SciPy float64.

    Uses CDF or survival consistently with the production branch choice so
    far-tail intervals do not underflow to a zero probability difference.
    """
    from scipy.stats import lognorm
    from scipy.special import log_ndtr

    a = (np.log(high) - loc) / scale
    b = (np.log(low) - loc) / scale
    use_sf = 0.5 * (a + b) > 0
    if use_sf:
        log_p_hi = float(log_ndtr(-b))
        log_p_lo = float(log_ndtr(-a))
    else:
        log_p_hi = float(log_ndtr(a))
        log_p_lo = float(log_ndtr(b))
    x = log_p_lo - log_p_hi
    if x > -np.log(2.0):
        log_one_m = float(np.log(-np.expm1(x)))
    else:
        log_one_m = float(np.log1p(-np.exp(x)))
    # Cross-check against ordinary probability arithmetic when it is safe.
    dist = lognorm(s=scale, scale=np.exp(loc))
    p = (dist.sf(low) - dist.sf(high)) if use_sf else (dist.cdf(high) - dist.cdf(low))
    if p > 0.0 and np.isfinite(np.log(p)):
        return float(np.log(p))
    return log_p_hi + log_one_m


@pytest.mark.parametrize(
    "loc,scale,low,high,abs_tol",
    [
        (0.0, 0.5, 0.01, 4.0, 5e-5),       # typical two-sided
        (0.0, 1.0, 1e-6, 1e-3, 5e-5),      # lower-tail mass
        (0.0, 0.5, 3.0, 5.0, 5e-5),        # right half (SF branch)
        (-2.0, 0.5, 0.01, 0.05, 5e-5),     # far left on value scale
        (0.0, 1.0, 0.999, 1.001, 2e-4),    # narrow — cancellation stress
        (0.0, 0.5, 0.25, 0.25001, 2e-3),   # extremely narrow (float32 ULP)
    ],
)
def test_log_z_matches_scipy_reference(loc, scale, low, high, abs_tol):
    d = TruncatedLogNormal(
        jnp.asarray(loc, dtype=jnp.float32),
        jnp.asarray(scale, dtype=jnp.float32),
        jnp.asarray(low, dtype=jnp.float32),
        jnp.asarray(high, dtype=jnp.float32),
    )
    got = float(d._log_z())
    want = _scipy_log_z(loc, scale, low, high)
    assert np.isfinite(got) and np.isfinite(want)
    assert got == pytest.approx(want, rel=0, abs=abs_tol)


def test_log_z_float32_narrow_interval_finite():
    d = TruncatedLogNormal(
        jnp.float32(0.0), jnp.float32(1.0),
        jnp.float32(0.999), jnp.float32(1.001),
    )
    lz = d._log_z()
    assert jnp.isfinite(lz)
    want = _scipy_log_z(0.0, 1.0, 0.999, 1.001)
    assert float(lz) == pytest.approx(want, rel=0, abs=2e-4)


def test_truncate_sigmax_2_halfnormal_and_truncated_normal_branches():
    """Every supported pre-truncated adapter branch used by the prior adapter."""
    hn = truncate_sigmax_2_prior(dist.HalfNormal(1.0), 0.1, 2.0)
    assert isinstance(hn, (dist.Distribution,))
    draws = hn.sample(random.PRNGKey(8), sample_shape=(16,))
    assert bool(jnp.all(jnp.isfinite(draws)))

    tn = truncate_sigmax_2_prior(
        dist.TruncatedNormal(0.0, 1.0, low=0.0, high=10.0), 0.2, 1.5)
    draws2 = tn.sample(random.PRNGKey(9), sample_shape=(16,))
    assert bool(jnp.all(jnp.isfinite(draws2)))

    ln = truncate_sigmax_2_prior(dist.LogNormal(0.0, 0.5), 0.1, 2.0)
    assert isinstance(ln, TruncatedLogNormal)
    tln = truncate_sigmax_2_prior(ln, 0.2, 1.5)
    assert isinstance(tln, TruncatedLogNormal)


@pytest.mark.parametrize(
    "low,high",
    [
        (1.0, 1.0),     # equal
        (2.0, 1.0),     # reversed
    ],
)
def test_invalid_bounds_raise_with_validate_args(low, high):
    with pytest.raises((ValueError, AssertionError)):
        TruncatedLogNormal(0.0, 1.0, low, high, validate_args=True)
