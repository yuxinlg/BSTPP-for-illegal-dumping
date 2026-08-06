"""OP-20 acceptance census: which input classes construct today, and what type
each config field ends up holding.

Run BEFORE and AFTER each commit of the OP-20 series and diff. This census is
the artifact the declared accept->reject flips are enumerated FROM -- the
brief's counts are not authoritative, the measurement is (A-29 precedent: a
cited figure the probe refutes is a defect in the register, not a rounding
error).

Reproducible on demand and independent of WHEN it runs: no baseline is
embedded, and the output is a pure function of the installed bstpp. A-30's
AST-equivalence artifact failed that property by defaulting its baseline to
HEAD; this one has no baseline at all, and the diff is taken between two saved
runs.

Usage:
    python census_argtypes.py            # human-readable census
    python census_argtypes.py --json     # machine-diffable form
"""
from __future__ import annotations

import argparse
import json
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
from shapely.geometry import box  # noqa: E402

import bstpp  # noqa: E402

assert os.path.abspath(bstpp.__file__).startswith(_REPO), (
    f"census loaded the WRONG bstpp: {bstpp.__file__}")

from bstpp.config import NumericalConfig  # noqa: E402
from bstpp.excitation_support import resolve_sigma_bounds  # noqa: E402
from bstpp.main import Hawkes_Model  # noqa: E402
from bstpp.polygon_mass import (  # noqa: E402
    DEFAULT_GL_ORDER,
    build_quad_table,
    prepare_polygon_mass_table,
)


class HasFloat:
    def __float__(self):
        return 5.0

    def __repr__(self):
        return "<obj.__float__>"


class HasIndex:
    def __index__(self):
        return 16

    def __repr__(self):
        return "<obj.__index__>"


# The ten classes the brief names, plus Fraction (measured divergent during WP2
# Step 1d) and an __index__ object for the integral fields.
REAL_CLASSES = [
    ("int", 5),
    ("float", 5.0),
    ("np.float64", np.float64(5.0)),
    ("np.float32", np.float32(5.0)),
    ("np.int64", np.int64(5)),
    ("ndarray0d", np.array(5.0)),
    ("str", "5"),
    ("bool", True),
    ("Decimal", Decimal("5")),
    ("Fraction", Fraction(5, 1)),
    ("obj.__float__", HasFloat()),
]

# Integral classes are applied per-field at the field's REQUIRED value, because
# three NumericalConfig fields are frozen policy pinned by an equality check
# (production_tau_abs == PRODUCTION_TAU_ABS, budget_reference_gl_order ==
# BUDGET_REFERENCE_GL_ORDER, budget_reference_oracle_bound == ...). Feeding them
# an arbitrary value measures the VALUE gate, not the TYPE gate, and the first
# run of this census did exactly that -- reporting `int` as rejected for
# budget_reference_gl_order, which is true of the value 16 and false of the
# type. Each entry is (label, factory taking the field's required magnitude).
INT_CLASS_MAKERS = [
    ("int", lambda n: int(n)),
    ("float", lambda n: float(n)),
    ("np.int64", lambda n: np.int64(n)),
    ("np.float64", lambda n: np.float64(n)),
    ("str", lambda n: str(n)),
    ("bool", lambda n: True),  # bool cannot carry a magnitude; it is its own case
    ("obj.__index__", lambda n: HasIndex()),
]
INT_PATH_REQUIRED = {
    "NumericalConfig.create/gl_order": 16,
    "NumericalConfig.create/budget_reference_gl_order": 32,
}

T_DAYS = 30.0
A = np.array([[0.0, 200.0], [0.0, 200.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
_rng = np.random.default_rng(0)
DATA = pd.DataFrame({
    "X": _rng.uniform(10, 190, 8),
    "Y": _rng.uniform(10, 190, 8),
    "T": np.sort(_rng.uniform(0.5, T_DAYS - 0.5, 8)),
})
POLY = box(0.0, 0.0, 200.0, 200.0)
EX, EY = np.array([10.0, 20.0]), np.array([10.0, 20.0])

REAL_PATHS = [
    "NumericalConfig.create/min_sigma",
    "NumericalConfig.create/panel_h_m",
    "resolve_sigma_bounds/min_sigma",
    "Hawkes_Model/min_sigma",
    "prepare_polygon_mass_table/min_sigma",
    "build_quad_table/sigma_min",
]
INT_PATHS = [
    "NumericalConfig.create/gl_order",
    "NumericalConfig.create/budget_reference_gl_order",
]


def paths_for_real(v):
    return {
        "NumericalConfig.create/min_sigma": lambda: NumericalConfig.create(
            support_mode="rectangle", min_sigma=v, max_sigma=40.0),
        "NumericalConfig.create/panel_h_m": lambda: NumericalConfig.create(
            support_mode="rectangle", panel_h_m=v),
        "resolve_sigma_bounds/min_sigma": lambda: resolve_sigma_bounds(
            mode="rectangle", min_sigma=v, max_sigma=40.0, crs=None),
        "Hawkes_Model/min_sigma": lambda: Hawkes_Model(
            DATA, A, T_DAYS, cox_background=False,
            excitation_support="rectangle", min_sigma=v, max_sigma=40.0,
            **PRIORS),
        "prepare_polygon_mass_table/min_sigma": lambda:
            prepare_polygon_mass_table(
                POLY, EX, EY, min_sigma=v, max_sigma=40.0, panel_h_m=1.0),
        "build_quad_table/sigma_min": lambda: build_quad_table(
            POLY, EX, EY, v, 40.0, ws=None, h_panel=1.0,
            gl_order=int(DEFAULT_GL_ORDER)),
    }


def int_path(path: str, make):
    """Build the call for one integral path at that field's REQUIRED value."""
    v = make(INT_PATH_REQUIRED[path])
    if path == "NumericalConfig.create/gl_order":
        return lambda: NumericalConfig.create(
            support_mode="rectangle", gl_order=v)
    return lambda: NumericalConfig.create(
        support_mode="rectangle", budget_reference_gl_order=v)


def verdict(fn) -> str:
    try:
        fn()
    except BaseException as e:  # noqa: BLE001 - the verdict IS the measurement
        return f"reject:{type(e).__name__}"
    return "ACCEPT"


STORED_FIELDS = ("panel_h_m", "gl_order", "min_sigma", "max_sigma",
                 "production_tau_abs", "max_panel_to_min_sigma_ratio")


def stored_types(**kw):
    try:
        c = NumericalConfig.create(**kw)
    except BaseException:  # noqa: BLE001
        return None
    return {f: type(getattr(c, f)).__name__ for f in STORED_FIELDS}


STORED_CASES = [
    ("all float args", dict(support_mode="rectangle", min_sigma=5.0,
                            max_sigma=40.0, panel_h_m=20.0, gl_order=16)),
    ("int args", dict(support_mode="rectangle", min_sigma=5, max_sigma=40,
                      panel_h_m=20, gl_order=16)),
    ("np.float64 args", dict(support_mode="rectangle",
                             min_sigma=np.float64(5.0),
                             max_sigma=np.float64(40.0),
                             panel_h_m=np.float64(20.0), gl_order=16)),
]


def build() -> dict:
    census: dict = {"real": {}, "int": {}, "stored_types": {}}
    for cname, val in REAL_CLASSES:
        fns = paths_for_real(val)
        census["real"][cname] = {p: verdict(fns[p]) for p in REAL_PATHS}
    for cname, make in INT_CLASS_MAKERS:
        census["int"][cname] = {
            p: verdict(int_path(p, make)) for p in INT_PATHS}
    for label, kw in STORED_CASES:
        census["stored_types"][label] = stored_types(**kw)
    return census


def _cell(v: str, width: int) -> str:
    text = "ACCEPT" if v == "ACCEPT" else v.split(":", 1)[1]
    return f"{text[:width - 2]:<{width}}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="machine-diffable form for before/after comparison")
    args = ap.parse_args()
    census = build()

    if args.json:
        print(json.dumps(census, indent=1, sort_keys=True))
        return 0

    print(f"bstpp.__file__ = {bstpp.__file__}")
    print()
    print("=" * 108)
    print("ACCEPTANCE CENSUS -- config-owned REAL arguments")
    print("=" * 108)
    print(f"{'input class':<16}" + "".join(
        f"{p.split('/')[0][:20]:<18}" for p in REAL_PATHS))
    print(f"{'':<16}" + "".join(
        f"{'.' + p.split('/')[1][:18]:<18}" for p in REAL_PATHS))
    print("-" * 108)
    for cname, _ in REAL_CLASSES:
        row = census["real"][cname]
        print(f"{cname:<16}" + "".join(_cell(row[p], 18) for p in REAL_PATHS))

    print()
    print("=" * 108)
    print("ACCEPTANCE CENSUS -- config-owned INTEGRAL arguments")
    print("each field probed at its REQUIRED magnitude "
          "(gl_order=16, budget_reference_gl_order=32), so a rejection here "
          "is a TYPE rejection and not the frozen-policy value gate")
    print("=" * 108)
    print(f"{'input class':<16}" + "".join(
        f"{p.split('/')[1][:30]:<34}" for p in INT_PATHS))
    print("-" * 108)
    for cname, _ in INT_CLASS_MAKERS:
        row = census["int"][cname]
        print(f"{cname:<16}" + "".join(_cell(row[p], 34) for p in INT_PATHS))

    print()
    print("=" * 108)
    print("STORED FIELD TYPES -- what the frozen object actually holds")
    print("=" * 108)
    for label, types in census["stored_types"].items():
        print(f"  {label}:")
        if types is None:
            print("    <rejected>")
            continue
        for f, t in types.items():
            flag = "" if t in ("float", "int", "NoneType") else "  <-- NOT a builtin"
            print(f"    {f:<32}{t}{flag}")

    print()
    div = []
    for cname, _ in REAL_CLASSES:
        row = census["real"][cname]
        acc = [p for p in REAL_PATHS if row[p] == "ACCEPT"]
        rej = [p for p in REAL_PATHS if row[p] != "ACCEPT"]
        if acc and rej:
            div.append((cname, acc, rej))
    print("=" * 108)
    print(f"DIVERGENT REAL CLASSES (accepted on some paths, rejected on "
          f"others): {len(div)}")
    print("=" * 108)
    for cname, acc, rej in div:
        print(f"  {cname}")
        print(f"    ACCEPT : {', '.join(p.split('/')[0] for p in acc)}")
        print(f"    reject : {', '.join(p.split('/')[0] for p in rej)}")

    uniform_accept = [c for c, _ in REAL_CLASSES
                      if all(census["real"][c][p] == "ACCEPT" for p in REAL_PATHS)]
    print()
    print(f"UNIFORMLY ACCEPTED on every real path: {len(uniform_accept)} "
          f"-> {', '.join(uniform_accept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
