"""WP1.4e-2 Step 1, probe A: accept/reject cross-product of the two validators.

Determines BY EXECUTION, for each candidate (mode, min_sigma, max_sigma):

  * what ``excitation_support.resolve_sigma_bounds`` does (accept -> resolved
    pair, or reject -> error type + message);
  * what ``config.NumericalConfig.create`` does with the SAME user-supplied
    arguments (i.e. as if the config were the source at construction);
  * what ``NumericalConfig.create`` does with the RESOLVED pair, which is what
    production actually passes today (main.py:2069-2070, 2307-2308).

The third column is the one that says whether the config's sigma branches are
reachable at all on a public path; the second is the one that says whether a
delegation design would change accept/reject.

Read-only. No production code is modified.
"""

from __future__ import annotations

import os
import traceback
from decimal import Decimal

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
from pyproj import CRS

import bstpp

# Provenance: a stale copy of the package exists in site-packages. Every probe
# must prove which one it loaded before it reports anything.
print(f"bstpp.__file__ = {bstpp.__file__}")
assert "BSTPP-refactor" in bstpp.__file__.replace("\\", "/"), (
    "probe loaded the installed bstpp, not the working tree")

from bstpp.config import NumericalConfig, NumericalConfigError  # noqa: E402
from bstpp.excitation_support import resolve_sigma_bounds  # noqa: E402

CRS_M = CRS.from_epsg(32618)  # UTM 18N, metre axis units

# panel_h_m small enough that the panel/min_sigma prefilter never fires for any
# accepted min_sigma below; isolates the five sigma/mode invariants.
PANEL_TINY = 1e-6


class _Floaty:
    """Object with __float__ but not an int/float instance."""

    def __float__(self):
        return 0.05

    def __repr__(self):
        return "_Floaty()"


def _fmt(v):
    return f"{v!r}"


def _outcome(fn):
    try:
        val = fn()
    except BaseException as exc:  # noqa: BLE001 - probe records everything
        tb = traceback.extract_tb(exc.__traceback__)
        # last frame inside bstpp
        site = "?"
        for fr in tb:
            if "bstpp" in fr.filename.replace("\\", "/"):
                site = f"{os.path.basename(fr.filename)}:{fr.lineno}"
        return ("REJECT", type(exc).__name__, str(exc).replace("\n", " "), site)
    return ("ACCEPT", "", repr(val), "")


def _resolve(mode, lo, hi, crs):
    def go():
        a, b, _ = resolve_sigma_bounds(
            mode=mode, min_sigma=lo, max_sigma=hi, crs=crs)
        return (a, b)
    return _outcome(go)


def _config(mode, lo, hi):
    def go():
        cfg = NumericalConfig.create(
            support_mode=mode, min_sigma=lo, max_sigma=hi,
            panel_h_m=PANEL_TINY)
        return (cfg.min_sigma, cfg.max_sigma)
    return _outcome(go)


CASES = [
    # (label, mode, min_sigma, max_sigma, crs)
    ("R both omitted",            "rectangle", None, None, CRS_M),
    ("R min only",                "rectangle", 0.05, None, CRS_M),
    ("R max only",                "rectangle", None, 0.5, CRS_M),
    ("R valid pair",              "rectangle", 0.05, 0.5, CRS_M),
    ("R lo == hi",                "rectangle", 0.5, 0.5, CRS_M),
    ("R lo > hi",                 "rectangle", 0.9, 0.5, CRS_M),
    ("R lo = 0",                  "rectangle", 0.0, 0.5, CRS_M),
    ("R lo < 0",                  "rectangle", -0.1, 0.5, CRS_M),
    ("R lo = nan",                "rectangle", float("nan"), 0.5, CRS_M),
    ("R lo = inf",                "rectangle", float("inf"), 0.5, CRS_M),
    ("R hi = inf",                "rectangle", 0.05, float("inf"), CRS_M),
    ("R hi = nan",                "rectangle", 0.05, float("nan"), CRS_M),
    ("R str lo",                  "rectangle", "0.05", 0.5, CRS_M),
    ("R str hi",                  "rectangle", 0.05, "0.5", CRS_M),
    ("R str both",                "rectangle", "0.05", "0.5", CRS_M),
    ("R bool lo True",            "rectangle", True, 2.0, CRS_M),
    ("R bool lo False",           "rectangle", False, 2.0, CRS_M),
    ("R np.float64",              "rectangle", np.float64(0.05), np.float64(0.5), CRS_M),
    ("R np.float32",              "rectangle", np.float32(0.05), np.float32(0.5), CRS_M),
    ("R np.int64",                "rectangle", np.int64(1), np.int64(2), CRS_M),
    ("R 0-d array",               "rectangle", np.array(0.05), np.array(0.5), CRS_M),
    ("R Decimal",                 "rectangle", Decimal("0.05"), Decimal("0.5"), CRS_M),
    ("R __float__ obj",           "rectangle", _Floaty(), 0.5, CRS_M),
    ("R int pair",                "rectangle", 1, 2, CRS_M),

    ("P min omitted",             "polygon", None, None, CRS_M),
    ("P min omitted, max given",  "polygon", None, 0.5, CRS_M),
    ("P max omitted (default)",   "polygon", 0.05, None, CRS_M),
    ("P valid pair",              "polygon", 0.05, 0.5, CRS_M),
    ("P lo == hi",                "polygon", 0.5, 0.5, CRS_M),
    ("P lo > hi",                 "polygon", 0.9, 0.5, CRS_M),
    ("P lo = 0",                  "polygon", 0.0, 0.5, CRS_M),
    ("P lo < 0",                  "polygon", -0.1, 0.5, CRS_M),
    ("P lo = 0, max omitted",     "polygon", 0.0, None, CRS_M),
    ("P lo = nan, max omitted",   "polygon", float("nan"), None, CRS_M),
    ("P lo = inf, max omitted",   "polygon", float("inf"), None, CRS_M),
    ("P str lo",                  "polygon", "0.05", 0.5, CRS_M),
    ("P str lo, max omitted",     "polygon", "0.05", None, CRS_M),
    ("P str hi",                  "polygon", 0.05, "0.5", CRS_M),
    ("P bool lo True",            "polygon", True, 2.0, CRS_M),
    ("P np.float32",              "polygon", np.float32(0.05), np.float32(0.5), CRS_M),
    ("P np.float64",              "polygon", np.float64(0.05), np.float64(0.5), CRS_M),
    ("P np.int64",                "polygon", np.int64(1), np.int64(2), CRS_M),
    ("P Decimal",                 "polygon", Decimal("0.05"), Decimal("0.5"), CRS_M),
    ("P __float__ obj",           "polygon", _Floaty(), 0.5, CRS_M),
    ("P crs=None, max omitted",   "polygon", 0.05, None, None),
    ("P crs=None, max given",     "polygon", 0.05, 0.5, None),

    # invariant 5: support-mode validity
    ("M bad mode 'triangle'",     "triangle", 0.05, 0.5, CRS_M),
    ("M bad mode 'Rectangle'",    "Rectangle", 0.05, 0.5, CRS_M),
    ("M bad mode None",           None, 0.05, 0.5, CRS_M),
    ("M bad mode, both omitted",  "triangle", None, None, CRS_M),
]


def main() -> None:
    rows = []
    for label, mode, lo, hi, crs in CASES:
        r = _resolve(mode, lo, hi, crs)
        c_user = _config(mode, lo, hi)
        if r[0] == "ACCEPT":
            resolved = eval(r[2])  # tuple repr produced above
            c_res = _config(mode, resolved[0], resolved[1])
        else:
            c_res = ("n/a", "", "resolve rejected first", "")
        rows.append((label, mode, lo, hi, crs is not None, r, c_user, c_res))

    print()
    print("=" * 100)
    print("TABLE 1 - resolve_sigma_bounds vs NumericalConfig.create (same user args)")
    print("=" * 100)
    div_user = []
    for label, mode, lo, hi, has_crs, r, c_user, c_res in rows:
        flag = ""
        if r[0] != c_user[0]:
            flag = "  <<< DIVERGENCE"
            div_user.append((label, mode, lo, hi, r, c_user))
        print(f"\n[{label}]  mode={_fmt(mode)} min={_fmt(lo)} max={_fmt(hi)} crs={'yes' if has_crs else 'None'}{flag}")
        print(f"    resolve : {r[0]:7s} {r[1]:22s} {r[2][:110]}  @{r[3]}")
        print(f"    cfg(usr): {c_user[0]:7s} {c_user[1]:22s} {c_user[2][:110]}  @{c_user[3]}")
        print(f"    cfg(res): {c_res[0]:7s} {c_res[1]:22s} {c_res[2][:110]}  @{c_res[3]}")

    print()
    print("=" * 100)
    print(f"DIVERGENCES (resolve vs config on the SAME user-supplied args): {len(div_user)}")
    print("=" * 100)
    for label, mode, lo, hi, r, c_user in div_user:
        direction = (
            "accept -> reject" if r[0] == "ACCEPT" else "reject -> accept")
        print(f"  {direction:18s} | {label:26s} mode={_fmt(mode)} min={_fmt(lo)} max={_fmt(hi)}")
        print(f"      resolve : {r[0]} {r[1]} {r[2][:100]}")
        print(f"      config  : {c_user[0]} {c_user[1]} {c_user[2][:100]}")

    print()
    print("=" * 100)
    print("REACHABILITY: does the config ever REJECT what resolve ACCEPTED?")
    print("(this is the only way a config sigma branch fires on a public path today)")
    print("=" * 100)
    live = [
        (label, r, c_res) for label, mode, lo, hi, has_crs, r, c_user, c_res in rows
        if r[0] == "ACCEPT" and c_res[0] == "REJECT"
    ]
    if not live:
        print("  NONE - every pair accepted by resolve_sigma_bounds is also")
        print("  accepted by NumericalConfig.create. The config's sigma/mode")
        print("  branches are unreachable through every public path.")
    for label, r, c_res in live:
        print(f"  {label}: resolve->{r[2]}  config->{c_res[1]}: {c_res[2][:120]}")


if __name__ == "__main__":
    main()
