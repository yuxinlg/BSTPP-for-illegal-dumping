"""WP2 Step 1d: OP-20 measured, not argued.

Which input classes does each sanctioned path ACCEPT for a real-valued config
argument? resolve_sigma_bounds coerces with float(); NumericalConfig rejects
with _require_real. The gap is the decision WP2 must settle before either new
config is written, because ModelConfig and PriorConfig raise it again.
"""
import os
import sys
from decimal import Decimal
from fractions import Fraction

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402

import bstpp  # noqa: E402

assert os.path.abspath(bstpp.__file__).startswith(_REPO), bstpp.__file__

from bstpp.config import NumericalConfig  # noqa: E402
from bstpp.excitation_support import resolve_sigma_bounds  # noqa: E402
from bstpp.main import Hawkes_Model  # noqa: E402


class HasFloat:
    def __float__(self):
        return 5.0

    def __repr__(self):
        return "<HasFloat 5.0>"


CLASSES = [
    ("float          ", 5.0),
    ("int            ", 5),
    ("bool True      ", True),
    ("str '5'        ", "5"),
    ("np.float32     ", np.float32(5.0)),
    ("np.float64     ", np.float64(5.0)),
    ("np.int64       ", np.int64(5)),
    ("0-d ndarray    ", np.array(5.0)),
    ("Decimal('5')   ", Decimal("5")),
    ("Fraction(5,1)  ", Fraction(5, 1)),
    ("obj.__float__  ", HasFloat()),
]

T_DAYS = 30.0
A = np.array([[0.0, 200.0], [0.0, 200.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))


def _data(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(10, 190, n), "Y": rng.uniform(10, 190, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n))})


def verdict(fn):
    try:
        fn()
    except BaseException as e:  # noqa: BLE001 - the verdict IS the measurement
        return f"reject({type(e).__name__})"
    return "ACCEPT"


print("bstpp.__file__ =", bstpp.__file__)
print()
print("OP-20: acceptance surface for a real-valued config argument")
print("=" * 78)
hdr = f"{'input class':<17}{'resolve_sigma_bounds':<26}{'NumericalConfig.create':<26}"
print(hdr)
print("-" * 78)

rows = []
for name, val in CLASSES:
    r = verdict(lambda v=val: resolve_sigma_bounds(
        mode="rectangle", min_sigma=v, max_sigma=40.0, crs=None))
    c = verdict(lambda v=val: NumericalConfig.create(
        support_mode="rectangle", min_sigma=v, max_sigma=40.0))
    rows.append((name, val, r, c))
    flag = "  <-- DIVERGES" if (r == "ACCEPT") != (c == "ACCEPT") else ""
    print(f"{name:<17}{r:<26}{c:<26}{flag}")

print()
div = [r for r in rows if (r[2] == "ACCEPT") != (r[3] == "ACCEPT")]
print(f"DIVERGENT INPUT CLASSES: {len(div)}")
for name, _, r, c in div:
    print(f"  {name.strip():<15} resolver={r:<22} config={c}")

print()
print("Reachability through the PUBLIC constructor (what a user can actually do)")
print("-" * 78)
data = _data()
constructed = []
for name, val in CLASSES:
    v = verdict(lambda vv=val: Hawkes_Model(
        data, A, T_DAYS, cox_background=False, excitation_support="rectangle",
        min_sigma=vv, max_sigma=40.0, **PRIORS))
    print(f"{name:<17}Hawkes_Model(min_sigma=...) -> {v}")
    if v == "ACCEPT":
        constructed.append(name.strip())
print()
print(f"CONSTRUCTS A MODEL TODAY: {len(constructed)} of {len(CLASSES)} classes")
print("  " + ", ".join(constructed))
