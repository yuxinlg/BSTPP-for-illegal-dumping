"""truncate_sigmax_2_prior: one-sided wrappers and Normal-base gate.

Pre-3f follow-up on tip 04799d9: NumPyro one-sided truncated wrappers omit the
unused bound attribute (LeftTruncatedDistribution has .low only;
RightTruncatedDistribution has .high only). The adapter must treat a missing
bound as +/-inf, intersect with the newly declared sigma^2 interval, and accept
pre-truncated wrappers only when base_dist is genuinely Normal -- never convert
truncated Cauchy / Student-t / other loc-scale families into TruncatedNormal.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.distributions.truncated import (
    LeftTruncatedDistribution,
    RightTruncatedDistribution,
    TwoSidedTruncatedDistribution,
)

from bstpp.excitation_support import truncate_sigmax_2_prior


def test_lower_only_truncated_normal_intersects_sigma_bounds():
    """LeftTruncatedDistribution has .low but no .high under pinned NumPyro."""
    prior = dist.TruncatedNormal(0.0, 1.0, low=0.0)
    assert isinstance(prior, LeftTruncatedDistribution)
    assert hasattr(prior, "low") and not hasattr(prior, "high")

    out = truncate_sigmax_2_prior(prior, 0.1, 2.0)
    # Intersection of [0, +inf) with [0.01, 4.0].
    assert float(out.low) == pytest.approx(0.01)
    assert float(out.high) == pytest.approx(4.0)
    assert isinstance(out.base_dist, dist.Normal)
    assert float(out.base_dist.loc) == pytest.approx(0.0)
    assert float(out.base_dist.scale) == pytest.approx(1.0)

    draws = out.sample(random.PRNGKey(0), sample_shape=(64,))
    assert bool(jnp.all((draws >= out.low) & (draws <= out.high)))
    # log_prob finite inside; -inf / raises outside support via support check
    mid = 0.5 * (float(out.low) + float(out.high))
    assert bool(jnp.isfinite(out.log_prob(mid)))


def test_upper_only_truncated_normal_intersects_sigma_bounds():
    """RightTruncatedDistribution has .high but no .low under pinned NumPyro."""
    prior = dist.TruncatedNormal(0.0, 1.0, high=5.0)
    assert isinstance(prior, RightTruncatedDistribution)
    assert hasattr(prior, "high") and not hasattr(prior, "low")

    out = truncate_sigmax_2_prior(prior, 0.1, 2.0)
    # Intersection of (-inf, 5] with [0.01, 4.0].
    assert float(out.low) == pytest.approx(0.01)
    assert float(out.high) == pytest.approx(4.0)
    assert isinstance(out.base_dist, dist.Normal)
    draws = out.sample(random.PRNGKey(1), sample_shape=(64,))
    assert bool(jnp.all((draws >= out.low) & (draws <= out.high)))
    mid = 0.5 * (float(out.low) + float(out.high))
    assert bool(jnp.isfinite(out.log_prob(mid)))


def test_two_sided_truncated_normal_still_intersects():
    prior = dist.TruncatedNormal(0.0, 1.0, low=0.0, high=10.0)
    assert isinstance(prior, TwoSidedTruncatedDistribution)
    out = truncate_sigmax_2_prior(prior, 0.2, 1.5)
    assert float(out.low) == pytest.approx(0.04)
    assert float(out.high) == pytest.approx(2.25)
    draws = out.sample(random.PRNGKey(2), sample_shape=(32,))
    assert bool(jnp.all((draws >= out.low) & (draws <= out.high)))


def test_existing_bound_tighter_than_requested_is_preserved():
    # Prior already tighter than [0.01, 4.0].
    prior = dist.TruncatedNormal(0.0, 1.0, low=0.5, high=2.0)
    out = truncate_sigmax_2_prior(prior, 0.1, 2.0)
    assert float(out.low) == pytest.approx(0.5)
    assert float(out.high) == pytest.approx(2.0)


def test_requested_bound_tighter_than_existing_wins():
    prior = dist.TruncatedNormal(0.0, 1.0, low=0.0, high=10.0)
    out = truncate_sigmax_2_prior(prior, 0.5, 1.0)
    assert float(out.low) == pytest.approx(0.25)
    assert float(out.high) == pytest.approx(1.0)


def test_empty_equal_intersection_raises():
    prior = dist.TruncatedNormal(0.0, 1.0, low=0.0, high=0.02)
    with pytest.raises(ValueError, match="does not overlap"):
        truncate_sigmax_2_prior(prior, 0.5, 2.0)  # [0.25, 4] ∩ [0, 0.02] empty

    prior2 = dist.TruncatedNormal(0.0, 1.0, low=4.0, high=9.0)
    with pytest.raises(ValueError, match="does not overlap"):
        # Equal-endpoint intersection: prior.low == new high.
        truncate_sigmax_2_prior(prior2, 0.1, 2.0)  # [0.01, 4] ∩ [4, 9] empty


def test_scalar_and_batched_loc_scale():
    scalar = truncate_sigmax_2_prior(
        dist.TruncatedNormal(0.0, 1.0, low=0.0), 0.1, 2.0)
    assert float(scalar.low) == pytest.approx(0.01)

    batched_prior = dist.TruncatedNormal(
        jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0]), low=0.0, high=10.0)
    batched = truncate_sigmax_2_prior(batched_prior, 0.1, 2.0)
    assert batched.batch_shape == (2,)
    # NumPyro may store scalar-like bounds as shape (1,) under batch (2,);
    # effective support must still be [0.01, 4] for every batch element.
    low_b = np.broadcast_to(np.asarray(batched.low, dtype=float), (2,))
    high_b = np.broadcast_to(np.asarray(batched.high, dtype=float), (2,))
    np.testing.assert_allclose(low_b, [0.01, 0.01])
    np.testing.assert_allclose(high_b, [4.0, 4.0])
    draws = batched.sample(random.PRNGKey(3), sample_shape=(8,))
    assert draws.shape[-1] == 2
    assert bool(jnp.all((draws >= batched.low) & (draws <= batched.high)))


def test_samples_and_log_prob_respect_effective_bounds():
    prior = dist.TruncatedNormal(0.0, 1.0, low=0.25, high=3.0)
    out = truncate_sigmax_2_prior(prior, 0.1, 1.0)  # ∩ [0.01, 1.0] -> [0.25, 1.0]
    lo, hi = float(out.low), float(out.high)
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(1.0)

    draws = out.sample(random.PRNGKey(4), sample_shape=(128,))
    assert bool(jnp.all((draws >= lo) & (draws <= hi)))
    interior = jnp.asarray(0.5)
    assert bool(jnp.isfinite(out.log_prob(interior)))
    # Effective support is the intersected interval; points outside are rejected.
    assert bool(out.support(interior))
    assert not bool(out.support(jnp.asarray(2.0)))
    assert not bool(out.support(jnp.asarray(0.1)))


def test_truncated_cauchy_rejected_without_family_change():
    prior = dist.TruncatedCauchy(0.0, 1.0, low=-1.0, high=5.0)
    assert isinstance(prior, TwoSidedTruncatedDistribution)
    assert isinstance(prior.base_dist, dist.Cauchy)
    with pytest.raises(TypeError, match="Normal"):
        truncate_sigmax_2_prior(prior, 0.1, 2.0)


def test_truncated_student_t_rejected_without_family_change():
    prior = dist.TruncatedDistribution(
        dist.StudentT(3.0, 0.0, 1.0), low=-2.0, high=5.0)
    assert isinstance(prior.base_dist, dist.StudentT)
    with pytest.raises(TypeError, match="Normal"):
        truncate_sigmax_2_prior(prior, 0.1, 2.0)


def test_unsupported_base_fails_loudly_not_silently_converted():
    """loc/scale alone must not imply Normality (Cauchy has both)."""
    prior = dist.TruncatedCauchy(1.0, 0.5, low=0.0, high=10.0)
    assert hasattr(prior.base_dist, "loc") and hasattr(prior.base_dist, "scale")
    with pytest.raises(TypeError) as ei:
        truncate_sigmax_2_prior(prior, 0.2, 1.5)
    msg = str(ei.value)
    assert "Normal" in msg
    # Prior remains a truncated Cauchy (adapter must not mutate or convert it).
    assert isinstance(prior.base_dist, dist.Cauchy)
