"""A-50: enumerate what is LEFT in WP2's declared scope, and measure each row.

WHY A PROBE AND NOT A LIST FROM MEMORY. The brief that commissioned this is
explicit that the enumeration IS the check -- that WP2's scope is a set, not a
habit of picking whatever looks adjacent to the last item. A list written from
recall would confirm whatever the writer already believed, which is how CI-9
came to be described as "the smallest item on the list" without anybody having
enumerated the list's remainder.

THE POPULATION, DEFINED BEFORE IT IS COUNTED. WP2's seam is `ModelConfig` and
`PriorConfig` (the work-package outline, recorded at A-43). The population is
therefore: **public-constructor arguments whose quantity belongs to one of
those two objects**, plus the items C1 routed to WP2 explicitly. Excluded, with
the reason attached to each row rather than dropped silently:

  * `ModelData` quantities -- events, domain, horizon, CRS, seasonal offset --
    which are WP4's;
  * `NumericalConfig` fields, which are WP1's (its object invariants are
    CLOSED; only its construction sites reopen, under D-43 clause 2);
  * `ExcitationSupport` / `PolygonMassTable` / cutoff arguments, which are
    WP5's, and WP5 cannot open;
  * `data_contracts` and `spatial_cov_crs`, which the outline assigns to WP8.

WHAT IS MEASURED. For each row in the population, whether an out-of-set value
is REJECTED AT CONSTRUCTION on a path where the leg that consumes it does not
run. That is CI-9's own predicate, applied to the rest of the population
instead of to the one argument that prompted it.

Usage:
    python refactor-patches/phase3f/wp2/probe_a50_wp2_remaining_items.py

Exit status is 0 if every row could be measured and 1 if any row's measurement
itself failed, so a capture records the probe's own verdict (D-41). Findings
do not set the status: this is a census, not a gate.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
import numpyro.distributions as dist                              # noqa: E402

import bstpp                                                      # noqa: E402
from bstpp.main import Hawkes_Model                               # noqa: E402

# A-42: every ad-hoc probe records which bstpp it actually imported, because a
# stale copy in site-packages shadows the repo and the capture then describes a
# different object.
print(f"bstpp.__file__ : {bstpp.__file__}")
print()

T_DAYS = 2.5 * 365.0
_rng = np.random.RandomState(0)
DATA = pd.DataFrame({"X": _rng.uniform(0.05, 0.95, 40),
                     "Y": _rng.uniform(0.05, 0.95, 40),
                     "T": np.sort(_rng.uniform(0, T_DAYS, 40))})
A_BOX = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = {"a_0": dist.Normal(1, 10), "alpha": dist.Beta(20, 60),
          "beta": dist.HalfNormal(2.0),
          "sigmax_2": dist.HalfNormal(0.25)}


def _construct(cls, **over):
    """Build a model with one argument overridden; return the exception or None.

    No covariates are supplied, deliberately: that is the path on which the
    consuming leg does not run, which is the whole of CI-9's predicate.

    Every row measured here is a `Hawkes_Model` row, because every argument in
    the population is one `Hawkes_Model` takes. `cls` stays a parameter rather
    than being inlined so that adding an `LGCP_Model` row later is a change to
    ROWS and not to the measurement.
    """
    kwargs = dict(PRIORS)
    kwargs.update(over)
    try:
        cls(DATA.copy(), A_BOX, T_DAYS, **kwargs)
        return None
    except Exception as exc:                    # noqa: BLE001 -- census
        return exc


#: Each row: (argument, owning object, in-scope?, why, the out-of-set value,
#: and what the package does with it today).
ROWS = [
    # ---- IN SCOPE: ModelConfig / PriorConfig quantities -------------------
    ("cox_background", "ModelConfig", True,
     "selects the background form (cox_hawkes vs hawkes); documented `bool`",
     [("'false'", "false"), ("'nonsense'", "nonsense"), ("0", 0), ("[]", [])]),
    ("sp_var_mu", "ModelConfig", True,
     "fixed log-amplitude multiplier on the spatial decoder; a real",
     [("'2.0'", "2.0"), ("True", True), ("None", None)]),
    ("standardize_cov", "ModelConfig", True,
     "CI-9's own argument; LANDED at A-45 and included here as a CONTROL -- "
     "a census whose known-enforced row does not fire is measuring nothing",
     [("'nonsense'", "nonsense"), ("True", True)]),
    # ---- OUT OF SCOPE, each with its owner ---------------------------------
    ("data_contracts", "WP8 (input metadata)", False,
     "the outline assigns `data_contracts` mode to WP8, not WP2", []),
    ("spatial_cov_crs", "WP8 (input metadata)", False,
     "same row of the outline", []),
    ("offset_seasonal", "WP4 (ModelData)", False,
     "the seasonal time origin is a ModelData fact", []),
    ("excitation_support / min_sigma / max_sigma / mass_table",
     "WP5", False,
     "WP5's seam exactly, and WP5 cannot open (OP-23 has no text)", []),
    ("panel_h_m / gl_order / tolerance defaults", "WP1 (NumericalConfig)",
     False,
     "WP1's object invariants are CLOSED; its construction sites reopen only "
     "under D-43 cl.2, which is the separate C1 item below", []),
]

#: The items C1 routed to WP2 explicitly, with their state as the register
#: records it. Not measured here -- their state is a register fact, not a
#: property of the tree -- but enumerated so the list is the whole list.
C1_ITEMS = [
    ("CI-9 enforcement", "LANDED at A-45", "closed"),
    ("D-43 cl.2 -- panel_h_m bind-time sentinel", "decision LANDED (A-36, "
     "reviewed A-39); implementation OUTSTANDING", "unblocked"),
    ("D-43 cl.2 -- standardize_cov bind-time relocation",
     "BLOCKED by declared gap G-B until OP-28 is answered", "blocked"),
    ("OP-25 -- warnings.warn ASCII sibling",
     "DEFERRED decision (A-39's standing rule bars settling it in a "
     "bookkeeping pass); its criterion needs a count not yet taken",
     "blocked"),
]


def main() -> int:
    failures = 0
    print("WP2 REMAINING-SCOPE CENSUS")
    print("  predicate : is an out-of-set value REJECTED AT CONSTRUCTION on a")
    print("              path where the consuming leg does not run? (CI-9)")
    print("  path      : Hawkes_Model, array domain, NO covariates")
    print()

    print("-- C1's routed items, state from the register ------------------")
    for name, state, verdict in C1_ITEMS:
        print(f"  [{verdict.upper():<9}] {name}")
        print(f"              {state}")
    print()

    print("-- constructor arguments, by owning object ---------------------")
    for arg, owner, in_scope, why, values in ROWS:
        tag = "IN SCOPE " if in_scope else "EXCLUDED "
        print(f"  [{tag}] {arg}   -> {owner}")
        print(f"              {why}")
        for label, value in values:
            try:
                exc = _construct(Hawkes_Model, **{arg: value})
            except Exception as exc_outer:      # noqa: BLE001
                print(f"      {label:<12} MEASUREMENT FAILED: "
                      f"{type(exc_outer).__name__}: {exc_outer}")
                failures += 1
                continue
            if exc is None:
                print(f"      {label:<12} ACCEPTED SILENTLY  <-- finding")
            else:
                print(f"      {label:<12} rejected: "
                      f"{type(exc).__name__}: {str(exc).splitlines()[0][:88]}")
        print()

    # cox_background's shipped DEFAULT, which is the row that makes this one
    # different from CI-9: the package's own default is outside the documented
    # accept set, so an enforcement written naively rejects the default.
    import inspect
    sig = inspect.signature(Hawkes_Model.__init__)
    default = sig.parameters["cox_background"].default
    print("-- the shipped default, measured -------------------------------")
    print(f"  Hawkes_Model.__init__ cox_background default = {default!r} "
          f"({type(default).__name__})")
    print("  docstring declares it `bool`. A bool-only enforcement would "
          "reject the")
    print("  package's own default, so the default is part of the change, "
          "not incidental.")
    print()

    if failures:
        print(f"PROBE_FAIL {failures} row(s) could not be measured")
        print("A49_CENSUS_EXIT:1")
        return 1
    print("A49_CENSUS_EXIT:0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
