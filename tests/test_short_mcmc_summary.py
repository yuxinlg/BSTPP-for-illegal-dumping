"""Short MCMC smoke runs must not fail in print_summary (pre-3f CF+API).

A valid short NUTS run that finishes sampling must still return samples when
retained draws are below NumPyro's diagnostic minimum. Skip only the summary
and emit one clear warning; do not change sampling knobs.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest

from bstpp.main import Hawkes_Model

T_DAYS = 20.0
A = np.array([[0.0, 1.0], [0.0, 1.0]])


def _tiny_hawkes():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({
        "X": rng.uniform(0.1, 0.9, 6),
        "Y": rng.uniform(0.1, 0.9, 6),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 6)),
    })
    return Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        a_0=dist.Normal(0, 5),
        alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0),
        sigmax_2=dist.HalfNormal(0.25),
    )


def test_run_mcmc_num_samples_2_returns_samples_without_summary_failure():
    m = _tiny_hawkes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.run_mcmc(
            num_warmup=2, num_samples=2, num_chains=1, thinning=1,
            rng_key=jax.random.PRNGKey(0),
        )
    assert hasattr(m, "samples")
    assert "alpha" in m.samples
    assert np.asarray(m.samples["alpha"]).shape[0] == 2
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("summary" in msg.lower() or "diagnostic" in msg.lower()
               for msg in msgs), msgs


def test_run_mcmc_sufficient_samples_still_prints_summary(capsys):
    m = _tiny_hawkes()
    m.run_mcmc(
        num_warmup=5, num_samples=8, num_chains=1, thinning=1,
        rng_key=jax.random.PRNGKey(1),
    )
    out = capsys.readouterr().out
    assert hasattr(m, "samples")
    assert np.asarray(m.samples["alpha"]).shape[0] == 8
    # NumPyro summary table includes a mean column header.
    assert "mean" in out.lower() or "Mean" in out
