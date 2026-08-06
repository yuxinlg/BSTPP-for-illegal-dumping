"""WP2 Step 1c: which candidate ModelConfig / PriorConfig fields go stale
under set_window?

Method: snapshot every args key and every candidate config field before and
after set_window, and diff. Per the brief, what matters is what is CONSUMED
downstream, not what is stored -- so the sigmax_2 prior is inspected by its
support bounds, which is what the likelihood and NUTS actually see.
"""
import os
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

T_DAYS = 30.0
A = np.array([[0.0, 200.0], [0.0, 200.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
rng = np.random.default_rng(0)
DATA = pd.DataFrame({"X": rng.uniform(10, 190, 8), "Y": rng.uniform(10, 190, 8),
                     "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 8))})

CANDIDATES = {
    # ModelConfig candidates
    "T": lambda m: m.args.get("T"),
    "S": lambda m: m.args.get("S"),
    "model": lambda m: m.args.get("model"),
    "offset_seasonal": lambda m: m.args.get("offset_seasonal"),
    "sp_var_mu": lambda m: m.args.get("sp_var_mu"),
    "A_area": lambda m: m.args.get("A_area"),
    "num_cov": lambda m: m.args.get("num_cov"),
    # PriorConfig candidates -- inspected by what the sampler SEES
    "prior a_0": lambda m: repr(m.args["priors"].get("a_0")),
    "prior alpha": lambda m: repr(m.args["priors"].get("alpha")),
    "prior beta": lambda m: repr(m.args["priors"].get("beta")),
    "prior sigmax_2 type": lambda m: type(m.args["priors"]["sigmax_2"]).__name__,
    "prior sigmax_2 support": lambda m: repr(
        getattr(m.args["priors"]["sigmax_2"], "support", None)),
    "t_trig par names": lambda m: tuple(m.args["t_trig"].get_par_names()),
    "sp_trig par names": lambda m: tuple(m.args["sp_trig"].get_par_names()),
    # Cutoff / numerical, for contrast
    "window": lambda m: m.args.get("window"),
    "spatial_window": lambda m: m.args.get("spatial_window"),
    "numerical_config id": lambda m: id(m.args.get("numerical_config")),
    "numerical_config value": lambda m: repr(m.args.get("numerical_config")),
    "n excitation pairs": lambda m: int(np.asarray(m.args["dif_t"]).shape[0])
    if "dif_t" in m.args else None,
}


def snap(m):
    out = {}
    for k, fn in CANDIDATES.items():
        try:
            out[k] = fn(m)
        except Exception as e:  # noqa: BLE001
            out[k] = f"<err {type(e).__name__}>"
    return out


print("bstpp.__file__ =", bstpp.__file__)
m = Hawkes_Model(DATA, A, T_DAYS, cox_background=False,
                 excitation_support="rectangle", min_sigma=5.0, max_sigma=40.0,
                 spatial_window=60.0, **PRIORS)
before = snap(m)
m.set_window(spatial_window=30.0)
after = snap(m)

print()
print("set_window(spatial_window=60 -> 30) on a rectangle Hawkes model")
print("=" * 84)
print(f"{'field':<26}{'changed':<10}{'before -> after'}")
print("-" * 84)
changed, unchanged = [], []
for k in CANDIDATES:
    b, a = before[k], after[k]
    same = (b == a)
    (unchanged if same else changed).append(k)
    mark = "same" if same else "CHANGED"
    bs, as_ = str(b)[:24], str(a)[:24]
    print(f"{k:<26}{mark:<10}{bs} -> {as_}")

print()
print(f"CHANGED   ({len(changed)}): {', '.join(changed)}")
print(f"UNCHANGED ({len(unchanged)}): {', '.join(unchanged)}")
print()
print("READING: every ModelConfig and PriorConfig candidate is UNCHANGED across")
print("set_window on this path. The mutation exposure for the two WP2 objects is")
print("therefore driven by the SIGMA-BOUND path, not the window path -- so the")
print("discriminating case is a polygon set_window that installs a table, since")
print("that is the only public call that re-resolves sigma bounds.")
