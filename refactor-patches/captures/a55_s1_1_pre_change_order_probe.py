"""S1.1 pre-change characterization: which exception wins when two config
arguments are invalid at once.

Run against the PRE-CHANGE tree (S1.0 tip abe0f8b) before bstpp/ is edited.
ModelConfig relocates three already-landed enforcement clauses -- CI-10
cox_background (Hawkes_Model, main.py:2027), CI-9 standardize_cov
(Point_Process_Model.__init__, main.py:333) and CI-7 sp_var_mu
(main.py:564) -- into one __post_init__. Relocation is BP only if a caller
passing two invalid arguments still sees the SAME exception, so the order the
three fire today has to be measured before it can be preserved.

The last block is the one that is NOT purely descriptive: it measures where
sp_var_mu sits relative to the covariate leg, because ModelConfig is
constructed at main.py:333's position and that moves sp_var_mu earlier
relative to work that is not config validation. Measured, then declared.

Prints bstpp.__file__ (AGENTS.md: a stale bstpp in site-packages shadows the
repo) and its own exit status.
"""
import os
import sys
import traceback

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

# THIS BLOCK IS LOAD-BEARING AND THE FIRST RUN OF THIS PROBE PROVED IT.
# Python puts the SCRIPT's directory on sys.path, not the CWD. This file lives
# in refactor-patches/captures/, so a bare `import bstpp` from the repo root
# resolved to the copy in site-packages -- which is old enough to predate
# A-53's sp_var_mu parameter entirely, so every reading described a different
# object. That is the exact hazard AGENTS.md names ("a stale bstpp in
# site-packages shadows the repo for scripts run from a subdirectory, and the
# capture then describes a different object"). Prepending the repo root fixes
# it; the assertion below makes the failure loud rather than silent, because a
# probe that CAN quietly measure the wrong tree is not evidence.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import numpyro.distributions as dist                                # noqa: E402

import bstpp                                                        # noqa: E402
from bstpp.main import Hawkes_Model, LGCP_Model                     # noqa: E402

_resolved = os.path.abspath(bstpp.__file__)
if not _resolved.startswith(os.path.join(_REPO_ROOT, "bstpp") + os.sep):
    raise SystemExit(
        f"REFUSING TO MEASURE THE WRONG TREE: bstpp resolved to {_resolved}, "
        f"which is outside the repository at {_REPO_ROOT}.")

T_DAYS = 200.0
A = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))

BAD_COX = "cox"          # CI-10: the retired string, rejected since A-50
BAD_STD = True           # CI-9: legacy boolean, rejected explicitly
BAD_SPV = "2.0"          # CI-7: coerces to the right number, wrong type


def _data(n=15, seed=5):
    r = np.random.RandomState(seed)
    return pd.DataFrame({"X": r.uniform(0.05, 0.95, n),
                         "Y": r.uniform(0.05, 0.95, n),
                         "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n))})


def observe(label, build):
    """Record the exception type and message a construction actually raises."""
    print(f"--- {label}")
    try:
        build()
    except BaseException as exc:                      # noqa: BLE001 -- probe
        first = str(exc).splitlines()[0] if str(exc) else ""
        print(f"    RAISED {type(exc).__name__}: {first}")
        return type(exc).__name__, str(exc)
    print("    CONSTRUCTED (no exception)")
    return None, None


def hawkes(**over):
    kw = dict(PRIORS)
    kw.update(over)
    kw.setdefault("cox_background", False)
    return lambda: Hawkes_Model(_data(), A, T_DAYS, **kw)


def lgcp(**over):
    return lambda: LGCP_Model(_data(), A, T_DAYS, a_0=dist.Normal(0, 5), **over)


def main() -> int:
    print(f"bstpp.__file__ : {bstpp.__file__}")
    print(f"interpreter    : {sys.executable}")
    print()

    print("== SINGLE INVALID ARGUMENT (the identity each site raises today) ==")
    observe("cox_background='cox'", hawkes(cox_background=BAD_COX))
    observe("standardize_cov=True", hawkes(standardize_cov=BAD_STD))
    observe("sp_var_mu='2.0'", hawkes(sp_var_mu=BAD_SPV))
    print()

    print("== TWO INVALID ARGUMENTS (the ordering rows S1.1 must preserve) ==")
    observe("cox_background BAD + standardize_cov BAD",
            hawkes(cox_background=BAD_COX, standardize_cov=BAD_STD))
    observe("cox_background BAD + sp_var_mu BAD",
            hawkes(cox_background=BAD_COX, sp_var_mu=BAD_SPV))
    observe("standardize_cov BAD + sp_var_mu BAD",
            hawkes(standardize_cov=BAD_STD, sp_var_mu=BAD_SPV))
    print()

    print("== ALL THREE INVALID ==")
    observe("cox_background + standardize_cov + sp_var_mu all BAD",
            hawkes(cox_background=BAD_COX, standardize_cov=BAD_STD,
                   sp_var_mu=BAD_SPV))
    print()

    print("== LGCP (no cox_background argument on this class) ==")
    observe("LGCP standardize_cov BAD", lgcp(standardize_cov=BAD_STD))
    observe("LGCP sp_var_mu BAD", lgcp(sp_var_mu=BAD_SPV))
    observe("LGCP standardize_cov BAD + sp_var_mu BAD",
            lgcp(standardize_cov=BAD_STD, sp_var_mu=BAD_SPV))
    print()

    print("== VALID CONSTRUCTIONS (the accept set must not move) ==")
    observe("hawkes valid", hawkes())
    observe("cox_hawkes valid", hawkes(cox_background=True))
    observe("lgcp valid", lgcp())
    observe("standardize_cov='domain_area' with no covariates",
            hawkes(standardize_cov="domain_area"))
    observe("sp_var_mu=np.float64(2.5)", hawkes(sp_var_mu=np.float64(2.5)))
    print()

    print("== ORDERING AGAINST NON-CONFIG WORK ==")
    print("    sp_var_mu is validated at main.py:564 today, AFTER the data")
    print("    contract leg (main.py:325) and AFTER prepared-domain work.")
    print("    ModelConfig is constructed at main.py:333's position, which")
    print("    keeps sp_var_mu after the data contracts but moves it ahead of")
    print("    the covariate leg. Both orderings are measured here.")
    bad_events = pd.DataFrame({"X": [0.5, np.nan], "Y": [0.5, 0.5],
                               "T": [1.0, 2.0]})
    observe("bad EVENTS + sp_var_mu BAD (data contracts must still win)",
            lambda: Hawkes_Model(bad_events, A, T_DAYS, cox_background=False,
                                 sp_var_mu=BAD_SPV, **PRIORS))
    observe("bad EVENTS + standardize_cov BAD (data contracts win today)",
            lambda: Hawkes_Model(bad_events, A, T_DAYS, cox_background=False,
                                 standardize_cov=BAD_STD, **PRIORS))
    # main.py:569 fires on a NON-Distribution kwarg. A Distribution-valued
    # unknown kwarg is ACCEPTED at :566 into default_priors, whatever its name,
    # so `nonsense_prior=dist.Normal(0, 1)` constructs. Both rows are kept: the
    # accepting one states the boundary the raising one sits on.
    observe("unknown kwarg that IS a Distribution -- accepted at main.py:566",
            hawkes(nonsense_prior=dist.Normal(0, 1)))
    observe("unknown kwarg that is NOT a Distribution "
            "(the WP9 intake site, main.py:569 bare Exception)",
            hawkes(nonsense_prior=5))
    observe("non-Distribution kwarg + sp_var_mu BAD "
            "(sp_var_mu at :564 fires before the kwarg loop at :565)",
            hawkes(sp_var_mu=BAD_SPV, nonsense_prior=5))
    print()

    print("== THE PAIR THE BRIEF DOES NOT COVER: sp_var_mu vs THE COVARIATE LEG ==")
    print("    The covariate leg (main.py:440-520) runs BETWEEN standardize_cov")
    print("    at :333 and sp_var_mu at :564. Constructing ModelConfig at :333")
    print("    therefore moves sp_var_mu AHEAD of it. No accept set moves --")
    print("    both arguments are invalid in both trees and both raise -- but")
    print("    WHICH error a caller sees changes, so it is measured here")
    print("    pre-change and declared rather than discovered later.")
    cov_bad_names = pd.DataFrame({"X": [0.25, 0.75], "Y": [0.25, 0.75],
                                  "value": [1.0, 2.0]})
    observe("covariates naming a MISSING column, sp_var_mu VALID",
            hawkes(spatial_cov=cov_bad_names, cov_names=["no_such_column"],
                   cov_grid_size=[0.5, 0.5]))
    observe("covariates naming a MISSING column + sp_var_mu BAD "
            "(PRE-CHANGE: the covariate error wins)",
            hawkes(spatial_cov=cov_bad_names, cov_names=["no_such_column"],
                   cov_grid_size=[0.5, 0.5], sp_var_mu=BAD_SPV))
    observe("covariates naming a MISSING column + standardize_cov BAD "
            "(standardize_cov wins in BOTH trees -- :333 already precedes it)",
            hawkes(spatial_cov=cov_bad_names, cov_names=["no_such_column"],
                   cov_grid_size=[0.5, 0.5], standardize_cov=BAD_STD))
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException:                              # noqa: BLE001 -- probe
        traceback.print_exc()
        rc = 1
    print(f"EXIT_STATUS:{rc}")
    sys.exit(rc)
