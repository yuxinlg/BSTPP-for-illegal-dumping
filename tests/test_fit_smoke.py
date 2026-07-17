"""End-to-end NUTS fit-path smoke test (pre-SBC gate).

Everything else in the suite operates at trace/gradient/simulation level;
nothing exercised run_mcmc end-to-end (init strategy, mass-matrix
adaptation, posterior-site extraction) post-refactor. SBC is hundreds of
NUTS fits, so a broken fit path should cost this one smoke test, not the
first overnight run. Reference run: scripts/recover_test.py --nuts (green
on this machine before this test landed).

Stage-1 configuration per the SBC runbook: plain Hawkes
(cox_background=False), the fastest NUTS target. Asserts are health checks,
not calibration: finite latent posteriors, finite identified log-background
criterion log(Itot_txy - Itot_excite), and zero divergences (retrievable
because run_mcmc now collects the "diverging" extra field).

Marked slow: a real 50-warmup/50-draw NUTS run, minutes not seconds.
Deselect with -m "not slow".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import bstpp

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import jax
import numpyro.distributions as dist
from numpyro.infer import Predictive
import pytest

from bstpp.main import Hawkes_Model

T_DAYS = 2.5 * 365.0
_rng = np.random.RandomState(0)
_N = 60
DATA = pd.DataFrame({
    "X": _rng.uniform(0.05, 0.95, _N),
    "Y": _rng.uniform(0.05, 0.95, _N),
    "T": np.sort(_rng.uniform(0, T_DAYS, _N)),
})
A_RECT = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))

LATENT_SCALARS = ["a_0", "alpha", "beta", "sigmax_2"]


@pytest.mark.slow
def test_nuts_fit_smoke_plain_hawkes():
    fit = Hawkes_Model(DATA, A_RECT, T_DAYS, cox_background=False, **PRIORS)
    fit.run_mcmc(num_warmup=50, num_samples=50, num_chains=1)

    # posterior-site extraction: all latent scalars present and finite
    for name in LATENT_SCALARS:
        assert name in fit.samples, f"latent site '{name}' missing from posterior"
        vals = np.asarray(fit.samples[name])
        assert vals.shape[0] == 50
        assert np.all(np.isfinite(vals)), f"non-finite posterior for '{name}'"

    # zero divergences (health, not calibration): a diverging short chain on
    # this tiny well-posed target signals a broken fit path, not bad luck
    diverging = np.asarray(fit.mcmc.get_extra_fields()["diverging"])
    assert diverging.sum() == 0, f"{int(diverging.sum())} divergent transitions"

    # identified log-background criterion, same pattern as recover_test.py:
    # deterministics are not collected by MCMC, so replay via Predictive
    pred = Predictive(fit.model, posterior_samples=fit.samples,
                      return_sites=["Itot_txy", "Itot_excite"])
    det = pred(jax.random.PRNGKey(1), args=fit.args)
    itot_txy = np.asarray(det["Itot_txy"])
    itot_excite = np.asarray(det["Itot_excite"])
    background_mass = itot_txy - itot_excite
    assert np.all(background_mass > 0), "excitation mass exceeds total compensator"
    log_background = np.log(background_mass)
    assert np.all(np.isfinite(log_background)), "non-finite log_background criterion"
