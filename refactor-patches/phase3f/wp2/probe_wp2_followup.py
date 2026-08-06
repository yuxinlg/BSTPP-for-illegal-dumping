"""WP2 Step 1b follow-up: three candidates whose guards the first probe did
not actually reach. Recorded because a guard that is not reached reports
ACCEPTED, which is indistinguishable from a guard that does not exist.
"""
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402

import bstpp  # noqa: E402

assert os.path.abspath(bstpp.__file__).startswith(_REPO), bstpp.__file__

from bstpp.main import Hawkes_Model  # noqa: E402
from bstpp.trigger import Trigger  # noqa: E402

T_DAYS = 30.0
A = np.array([[0.0, 200.0], [0.0, 200.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
rng = np.random.default_rng(0)
DATA = pd.DataFrame({"X": rng.uniform(10, 190, 8), "Y": rng.uniform(10, 190, 8),
                     "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 8))})
# Covariate grid covering A, so the standardization leg actually RUNS.
gx, gy = np.meshgrid(np.arange(10.0, 200.0, 20.0), np.arange(10.0, 200.0, 20.0))
COV = pd.DataFrame({"X": gx.ravel(), "Y": gy.ravel(),
                    "c1": rng.normal(size=gx.size)})


class _CustomSpatial(Trigger):
    def get_par_names(self):
        return ["rho"]

    def compute_trigger(self, pars, pav):
        c, v = pav
        return c, np.exp(-v / pars["rho"])

    def compute_integral(self, pars, dif):
        return 1.0 - np.exp(-dif / pars["rho"])

    def simulate_trigger(self, pars, rng=None):
        return 1.0


def fire(label, fn):
    try:
        fn()
    except BaseException as e:  # noqa: BLE001
        print(f"  {label:<44} {type(e).__name__:<22} "
              f"{re.sub(r'\s+', ' ', str(e))[:80]}")
        return
    print(f"  {label:<44} {'ACCEPTED':<22} (no error)")


print("bstpp.__file__ =", bstpp.__file__)
print()
print("1. standardize_cov -- WITH covariates, so the standardization leg runs")
print("   (the first probe passed no covariates, so the guard was never reached")
print("    and reported ACCEPTED -- a false negative)")
for val in [True, False, "nope", "domain_area", None]:
    fire(f"standardize_cov={val!r} (+covariates)",
         lambda v=val: Hawkes_Model(
             DATA, A, T_DAYS, cox_background=False,
             excitation_support="rectangle", spatial_cov=COV,
             cov_names=["c1"], cov_grid_size=(20.0, 20.0),
             standardize_cov=v, **PRIORS))

print()
print("2. design_sigma / spatial tol -- with a REAL custom spatial trigger")
p = dict(PRIORS)
p.pop("sigmax_2")
p["rho"] = dist.HalfNormal(1.0)
fire("design_sigma=10 + custom spatial",
     lambda: Hawkes_Model(DATA, A, T_DAYS, cox_background=False,
                          excitation_support="rectangle",
                          spatial_trig=_CustomSpatial,
                          spatial_window=50.0, design_sigma=10.0, **p))
fire("spatial_cutoff_tol + custom spatial",
     lambda: Hawkes_Model(DATA, A, T_DAYS, cox_background=False,
                          excitation_support="rectangle",
                          spatial_trig=_CustomSpatial,
                          spatial_window=50.0, spatial_cutoff_tol=0.01, **p))
fire("min_sigma + custom spatial",
     lambda: Hawkes_Model(DATA, A, T_DAYS, cox_background=False,
                          excitation_support="rectangle",
                          spatial_trig=_CustomSpatial, spatial_window=50.0,
                          min_sigma=5.0, max_sigma=40.0, **p))

print()
print("3. horizon T -- is it validated as a QUANTITY, or only via the events?")
empty = pd.DataFrame({"X": [], "Y": [], "T": []})
for label, T in [("T=0", 0.0), ("T=-5", -5.0), ("T=inf", float("inf")),
                 ("T=nan", float("nan")), ("T='30'", "30")]:
    fire(f"{label} with events present",
         lambda t=T: Hawkes_Model(DATA, A, t, cox_background=False,
                                  excitation_support="rectangle", **PRIORS))

print()
print("4. the prior-type message: does it describe the actual failure?")
fire("a_0=1.0 (KNOWN name, wrong type)",
     lambda: Hawkes_Model(DATA, A, T_DAYS, cox_background=False,
                          excitation_support="rectangle",
                          **{**PRIORS, "a_0": 1.0}))
