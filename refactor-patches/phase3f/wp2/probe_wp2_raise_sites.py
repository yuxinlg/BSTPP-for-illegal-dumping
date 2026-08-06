"""WP2 Step 1b: raise sites and reachability for ModelConfig / PriorConfig
candidate invariants -- RUN, not read.

Two parts:
  A. AST census of every raise site in the constructor path, attributed to the
     candidate config that would own the field it guards.
  B. Runtime reachability: fire each invariant through every entry path that
     can reach it, and record TYPE + message, so divergence in accept/reject
     is visible and not merely divergence in wording.
"""
import ast
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

from bstpp.main import Hawkes_Model, LGCP_Model  # noqa: E402
from bstpp.trigger import (  # noqa: E402
    Spatial_Symmetric_Gaussian, Temporal_Exponential, Temporal_Power_Law,
)

# ------------------------------------------------------------------ part A --
# Attribution keywords -> candidate owner. A raise site matching none is
# reported as UNATTRIBUTED rather than silently dropped.
OWNERS = [
    ("PriorConfig", (
        "prior", "Distribution", "sigmax_2", "mean_lag_days", "beta",
        "get_par_names", "Unknown argument")),
    ("ModelConfig", (
        "model", "cox_background", "offset_seasonal", "sp_var_mu",
        "data_contracts", "standardize", "standardization", "cov_names",
        "cov_grid_size", "spatial_cov_crs", "horizon", "T_max",
        "temporal_trig", "spatial_trig", "trigger", "kernel")),
    ("NumericalConfig (WP1, existing)", (
        "min_sigma", "max_sigma", "panel_h_m", "gl_order", "support_mode",
        "excitation_support", "mass_table", "tau")),
    ("CutoffConfig (no object yet)", (
        "window", "cutoff", "tol", "design_sigma", "design_mean_lag")),
]


def attribute(text: str) -> str:
    for owner, keys in OWNERS:
        if any(k.lower() in text.lower() for k in keys):
            return owner
    return "UNATTRIBUTED"


def census(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            src = ast.unparse(node.exc)
            exc_type = "?"
            if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                exc_type = node.exc.func.id
            elif isinstance(node.exc, ast.Name):
                exc_type = node.exc.id
            rows.append((node.lineno, exc_type, attribute(src),
                         re.sub(r"\s+", " ", src)[:88]))
    return rows


print("bstpp.__file__ =", bstpp.__file__)
print()
print("=" * 92)
print("PART A -- AST census of raise sites in bstpp/main.py, by candidate owner")
print("=" * 92)
rows = census(os.path.join(_REPO, "bstpp", "main.py"))
by_owner = {}
for lineno, exc, owner, src in rows:
    by_owner.setdefault(owner, []).append((lineno, exc, src))
for owner, _ in OWNERS + [("UNATTRIBUTED", ())]:
    hits = by_owner.get(owner, [])
    print(f"\n  {owner}: {len(hits)} raise site(s)")
    for lineno, exc, src in hits:
        print(f"    main.py:{lineno:<5} {exc:<22} {src}")
print()
print(f"  TOTAL raise sites in main.py: {len(rows)}")
types = sorted({r[1] for r in rows})
print(f"  DISTINCT exception types: {len(types)} -> {', '.join(types)}")

# ------------------------------------------------------------------ part B --
T_DAYS = 30.0
A = np.array([[0.0, 200.0], [0.0, 200.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))


def _data(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(10, 190, n), "Y": rng.uniform(10, 190, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n))})


DATA = _data()


def fire(label, fn):
    try:
        fn()
    except BaseException as e:  # noqa: BLE001 - identity is the measurement
        msg = re.sub(r"\s+", " ", str(e))[:96]
        print(f"    {label:<34} {type(e).__name__:<24} {msg}")
        return type(e).__name__
    print(f"    {label:<34} {'ACCEPTED':<24} (no error)")
    return None


def hawkes(**kw):
    base = dict(cox_background=False, excitation_support="rectangle")
    base.update(kw)
    priors = {k: base.pop(k) for k in list(base) if k in
              ("a_0", "alpha", "beta", "sigmax_2", "gamma", "mean_lag_days")}
    p = dict(PRIORS)
    p.update(priors)
    for k in base.pop("_drop_priors", []):
        p.pop(k, None)
    return lambda: Hawkes_Model(DATA, A, T_DAYS, **base, **p)


print()
print("=" * 92)
print("PART B -- runtime reachability per candidate invariant")
print("=" * 92)

print("\n  PriorConfig candidates")
fire("prior not a Distribution", hawkes(a_0=1.0))
fire("unknown kwarg (non-Distribution)", hawkes(nonsense=3))
fire("missing sigmax_2 (Gaussian sp)", hawkes(_drop_priors=["sigmax_2"]))
fire("missing beta (Exponential temporal)", hawkes(_drop_priors=["beta"]))
fire("Power_Law without gamma", hawkes(temporal_trig=Temporal_Power_Law))
fire("mean_lag_days AND beta both", hawkes(mean_lag_days=dist.HalfNormal(1.0)))
fire("mean_lag_days on Power_Law", hawkes(
    temporal_trig=Temporal_Power_Law, gamma=dist.HalfNormal(1.0),
    mean_lag_days=dist.HalfNormal(1.0), _drop_priors=["beta"]))

print("\n  ModelConfig candidates")
fire("T = 0", lambda: Hawkes_Model(DATA, A, 0.0, cox_background=False,
                                   excitation_support="rectangle", **PRIORS))
fire("T negative", lambda: Hawkes_Model(DATA, A, -5.0, cox_background=False,
                                        excitation_support="rectangle", **PRIORS))
fire("T = inf", lambda: Hawkes_Model(DATA, A, float("inf"),
                                     cox_background=False,
                                     excitation_support="rectangle", **PRIORS))
fire("data_contracts bogus mode", hawkes(data_contracts="bogus"))
fire("standardize_cov=True (legacy bool)", hawkes(standardize_cov=True))
fire("standardize_cov bogus str", hawkes(standardize_cov="nope"))
fire("sp_var_mu = str", hawkes(sp_var_mu="big"))
fire("sp_var_mu = None", hawkes(sp_var_mu=None))
fire("offset_seasonal = str", hawkes(offset_seasonal="x"))
fire("excitation_support bogus", hawkes(excitation_support="triangle"))

print("\n  Cross-object: trigger capability gates (which object owns these?)")
fire("mean_lag_days + Power_Law", hawkes(
    temporal_trig=Temporal_Power_Law, gamma=dist.HalfNormal(1.0),
    mean_lag_days=dist.HalfNormal(1.0), _drop_priors=["beta"]))
fire("design_sigma + custom spatial", hawkes(design_sigma=10.0))
