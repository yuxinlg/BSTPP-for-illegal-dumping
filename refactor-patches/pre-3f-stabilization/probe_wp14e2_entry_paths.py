"""WP1.4e-2 Step 1, probe B: which raise site wins on each public entry path.

For each of the five sigma/mode invariants, drives a real public entry path
with a violating input and records, BY EXECUTION:

  * the error type and message actually raised;
  * the exact production file:line that raised it;
  * which raise lines inside ``bstpp/config.py`` were executed at all, via a
    line tracer -- this is the direct evidence for "is the config's branch
    reachable", rather than an argument from call order.

Entry paths covered:
  E1  Hawkes_Model(...)                       rectangle constructor
  E2  Hawkes_Model(...)                       polygon constructor
  E3  set_window(spatial_window=, mass_table=) polygon install
  E4  set_window(window=)                      temporal-only rebuild
  E5  log_expected_likelihood(mass_table=)     held-out scoring
  E6  LGCP_Model(...)                          rectangle constructor
  E7  prepare_polygon_mass_table(...)          public builder

Read-only. No production code is modified.
"""

from __future__ import annotations

import os
import sys
import traceback
from decimal import Decimal

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import bstpp

print(f"bstpp.__file__ = {bstpp.__file__}")
assert "BSTPP-refactor" in bstpp.__file__.replace("\\", "/"), (
    "probe loaded the installed bstpp, not the working tree")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.geometry import box  # noqa: E402

from bstpp.main import Hawkes_Model, LGCP_Model  # noqa: E402
from bstpp.polygon_mass import (  # noqa: E402
    DEFAULT_GL_ORDER,
    MAX_PANEL_TO_MIN_SIGMA_RATIO,
    prepare_polygon_mass_table,
)

CONFIG_PY = os.path.join(os.path.dirname(bstpp.__file__), "config.py")
EXSUP_PY = os.path.join(os.path.dirname(bstpp.__file__), "excitation_support.py")
POLY_PY = os.path.join(os.path.dirname(bstpp.__file__), "polygon_mass.py")

# Raise lines that implement each invariant, per file.
#
# Line numbers move when the source does, so this map is versioned: BEFORE is
# the f70ac7d source (the Step 1 enumeration and the RED capture), AFTER is the
# unified source. Selected by SIGMA_PROBE_STATE so the same probe produces a
# comparable table on both sides of the change instead of tracing stale lines
# and reporting a silently meaningless result.
_BEFORE = (
    {
        133: "I5 support_mode validity            (config)",
        196: "I1 rectangle both-or-neither        (config)",
        207: "I2 polygon requires min_sigma       (config)",
        213: "I3 min_sigma finite/positive        (config, polygon max=None branch)",
        230: "I3 min_sigma finite/positive        (config, _validate_sigma_pair)",
        233: "I4 min_sigma < max_sigma            (config, _validate_sigma_pair)",
    },
    {
        82:  "I5 support_mode validity            (resolve_excitation_support_mode)",
        119: "I1 rectangle both-or-neither        (resolve_sigma_bounds)",
        131: "I2 polygon requires min_sigma       (resolve_sigma_bounds)",
        151: "I3 min_sigma finite/positive        (_validate_sigma_pair)",
        154: "I4 min_sigma < max_sigma            (_validate_sigma_pair)",
        375: "I5 support_mode validity            (build_excitation_support)",
    },
    {
        608: "I3 min_sigma finite/positive        (assert_polygon_mass_table_budget)",
        1045: "I3 min_sigma finite/positive       (prepare_polygon_mass_table)",
    },
)
_AFTER = (
    {
        205: "I3 min_sigma finite/positive        (config, validate_sigma_pair)",
        207: "I4 min_sigma < max_sigma            (config, validate_sigma_pair)",
        265: "I5 support_mode validity            (config)",
        333: "I1 rectangle both-or-neither        (config)",
        342: "I2 polygon requires min_sigma       (config)",
        349: "I3 min_sigma finite/positive        (config, polygon max=None branch)",
    },
    {
        88:  "I5 support_mode validity            (resolve_excitation_support_mode)",
        134: "I5 support_mode validity            (resolve_sigma_bounds)",
        141: "I1 rectangle both-or-neither        (resolve_sigma_bounds)",
        151: "I2 polygon requires min_sigma       (resolve_sigma_bounds)",
        384: "I5 support_mode validity            (build_excitation_support)",
    },
    {
        612: "I3 min_sigma finite/positive        (assert_polygon_mass_table_budget)",
        1057: "I2 polygon requires min_sigma      (prepare_polygon_mass_table)",
        1064: "I3 min_sigma finite/positive       (prepare_polygon_mass_table)",
    },
)
_STATE = os.environ.get("SIGMA_PROBE_STATE", "after").lower()
CONFIG_INVARIANT_LINES, EXSUP_INVARIANT_LINES, POLY_INVARIANT_LINES = (
    _BEFORE if _STATE == "before" else _AFTER)
print(f"SIGMA_PROBE_STATE = {_STATE}")

_TARGETS = {
    CONFIG_PY: CONFIG_INVARIANT_LINES,
    EXSUP_PY: EXSUP_INVARIANT_LINES,
    POLY_PY: POLY_INVARIANT_LINES,
}
_norm = {os.path.normcase(k): v for k, v in _TARGETS.items()}

_hits: list[tuple[str, int]] = []


def _tracer(frame, event, arg):
    fn = os.path.normcase(frame.f_code.co_filename)
    if fn not in _norm:
        return None
    lines = _norm[fn]

    def _line(fr, ev, a):
        if ev == "line" and fr.f_lineno in lines:
            _hits.append((os.path.basename(fr.f_code.co_filename), fr.f_lineno))
        return _line
    return _line


def run(label, fn):
    """Execute fn under the line tracer; return an outcome record."""
    global _hits
    _hits = []
    sys.settrace(_tracer)
    try:
        fn()
        outcome = ("ACCEPT", "", "", "")
    except BaseException as exc:  # noqa: BLE001 - probe records everything
        tb = traceback.extract_tb(exc.__traceback__)
        site = "?"
        for fr in tb:
            p = fr.filename.replace("\\", "/")
            if "/bstpp/" in p:
                site = f"{os.path.basename(fr.filename)}:{fr.lineno}"
        outcome = ("REJECT", type(exc).__name__,
                   str(exc).replace("\n", " "), site)
    finally:
        sys.settrace(None)
    hits = sorted(set(_hits))
    return label, outcome, hits


# --------------------------------------------------------------- fixtures --
T_DAYS = 30.0
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(0.5, 0.5),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(0.25),
)
MIN_S = 0.05
MAX_S = 0.5
PANEL_GUIDED = MAX_PANEL_TO_MIN_SIGMA_RATIO * MIN_S


def _unit_gdf():
    return gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)])


def _events(n=6, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "X": rng.uniform(0.1, 0.9, n),
        "Y": rng.uniform(0.1, 0.9, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _table(data, panel_h_m=PANEL_GUIDED, min_sigma=MIN_S, max_sigma=MAX_S):
    geom = _unit_gdf().geometry.union_all()
    return prepare_polygon_mass_table(
        geom,
        data["X"].to_numpy(dtype=float),
        data["Y"].to_numpy(dtype=float),
        min_sigma=min_sigma, max_sigma=max_sigma,
        panel_h_m=panel_h_m, gl_order=DEFAULT_GL_ORDER, crs=None)


def _rect(**kw):
    # A GeoDataFrame domain is a polygon DOMAIN (OP-2 requires an explicit
    # excitation_support); rectangle SUPPORT on it is the rectangle entry path.
    kwargs = dict(cox_background=False, excitation_support="rectangle", **PRIORS)
    kwargs.update(kw)
    return Hawkes_Model(_events(), _unit_gdf(), T_DAYS, **kwargs)


def _poly(data, table, **kw):
    kwargs = dict(
        cox_background=False, excitation_support="polygon",
        min_sigma=MIN_S, max_sigma=MAX_S, mass_table=table, **PRIORS)
    kwargs.update(kw)
    return Hawkes_Model(data, _unit_gdf(), T_DAYS, **kwargs)


def main() -> None:
    data = _events()
    table = _table(data)
    good_poly = _poly(data, table)
    heldout = _events(n=5, seed=7)
    heldout_table = _table(heldout)

    cases = []

    # ---------------------------------------------------------------- I1 --
    cases.append(run(
        "I1 rectangle both-or-neither | E1 Hawkes rectangle ctor (min only)",
        lambda: _rect(min_sigma=MIN_S)))
    cases.append(run(
        "I1 rectangle both-or-neither | E1 Hawkes rectangle ctor (max only)",
        lambda: _rect(max_sigma=MAX_S)))
    cases.append(run(
        "I1 rectangle both-or-neither | E6 LGCP ctor (no sigma args exist)",
        lambda: LGCP_Model(_events(), _unit_gdf(), T_DAYS,
                           a_0=dist.Normal(0, 5))))

    # ---------------------------------------------------------------- I2 --
    cases.append(run(
        "I2 polygon requires min_sigma | E2 Hawkes polygon ctor",
        lambda: _poly(data, table, min_sigma=None)))
    cases.append(run(
        "I2 polygon requires min_sigma | E7 prepare_polygon_mass_table",
        lambda: _table(data, min_sigma=None)))

    # ---------------------------------------------------------------- I3 --
    for bad, tag in ((0.0, "0.0"), (-1.0, "-1.0"), (float("nan"), "nan"),
                     (float("inf"), "inf")):
        cases.append(run(
            f"I3 min_sigma finite/positive | E2 Hawkes polygon ctor ({tag})",
            lambda b=bad: _poly(data, table, min_sigma=b)))
    cases.append(run(
        "I3 min_sigma finite/positive | E1 Hawkes rectangle ctor (0.0)",
        lambda: _rect(min_sigma=0.0, max_sigma=MAX_S)))
    cases.append(run(
        "I3 min_sigma finite/positive | E7 prepare_polygon_mass_table (0.0)",
        lambda: _table(data, min_sigma=0.0)))

    # ---------------------------------------------------------------- I4 --
    cases.append(run(
        "I4 min_sigma < max_sigma | E1 Hawkes rectangle ctor (lo>hi)",
        lambda: _rect(min_sigma=0.9, max_sigma=0.5)))
    cases.append(run(
        "I4 min_sigma < max_sigma | E1 Hawkes rectangle ctor (lo==hi)",
        lambda: _rect(min_sigma=0.5, max_sigma=0.5)))
    cases.append(run(
        "I4 min_sigma < max_sigma | E2 Hawkes polygon ctor (lo>hi)",
        lambda: _poly(data, table, min_sigma=0.9, max_sigma=0.5)))

    # ---------------------------------------------------------------- I5 --
    cases.append(run(
        "I5 support_mode validity | E1 Hawkes ctor (excitation_support='triangle')",
        lambda: _rect(excitation_support="triangle")))
    cases.append(run(
        "I5 support_mode validity | E1 Hawkes ctor (excitation_support='Rectangle')",
        lambda: _rect(excitation_support="Rectangle")))

    # ------------------------------------------ the string / coercion case --
    cases.append(run(
        "COERCION str min_sigma | E1 Hawkes rectangle ctor ('0.05','0.5')",
        lambda: _rect(min_sigma="0.05", max_sigma="0.5")))
    cases.append(run(
        "COERCION str min_sigma | E2 Hawkes polygon ctor ('0.05')",
        lambda: _poly(data, table, min_sigma="0.05")))
    cases.append(run(
        "COERCION bool min_sigma | E1 Hawkes rectangle ctor (True, 2.0)",
        lambda: _rect(min_sigma=True, max_sigma=2.0)))
    cases.append(run(
        "COERCION np.float32 min_sigma | E2 Hawkes polygon ctor",
        lambda: _poly(data, table, min_sigma=np.float32(MIN_S))))
    cases.append(run(
        "COERCION np.float64 min_sigma | E2 Hawkes polygon ctor",
        lambda: _poly(data, table, min_sigma=np.float64(MIN_S))))
    cases.append(run(
        "COERCION np.float32 min_sigma | E1 Hawkes rectangle ctor",
        lambda: _rect(min_sigma=np.float32(MIN_S), max_sigma=MAX_S)))
    cases.append(run(
        "COERCION Decimal min_sigma | E1 Hawkes rectangle ctor",
        lambda: _rect(min_sigma=Decimal("0.05"), max_sigma=MAX_S)))
    cases.append(run(
        "COERCION str min_sigma | E1 Hawkes rectangle ctor ('0.05','0.5') rerun",
        lambda: _rect(min_sigma="0.05", max_sigma="0.5")))

    # ---------------------- polygon default max_sigma with no CRS (crux) --
    cases.append(run(
        "DEFAULTING polygon max_sigma=None, crs=None | E2 Hawkes polygon ctor",
        lambda: _poly(data, table, max_sigma=None)))

    # ------------------------------------- downstream paths on a good model --
    cases.append(run(
        "BASELINE valid | E2 Hawkes polygon ctor (accepts)",
        lambda: _poly(data, table)))
    cases.append(run(
        "BASELINE valid | E1 Hawkes rectangle ctor, bounds omitted (accepts)",
        lambda: _rect()))
    cases.append(run(
        "BASELINE valid | E1 Hawkes rectangle ctor, bounds supplied (accepts)",
        lambda: _rect(min_sigma=MIN_S, max_sigma=MAX_S)))
    cases.append(run(
        "E4 set_window temporal-only on valid polygon model",
        lambda: _poly(data, table).set_window(window=5.0)))
    cases.append(run(
        "E3 set_window spatial + mass_table on valid polygon model",
        lambda: _poly(data, table).set_window(
            spatial_window=0.4, mass_table=_table(data))))
    cases.append(run(
        "E5 log_expected_likelihood held-out (no fit -> may fail elsewhere)",
        lambda: good_poly.log_expected_likelihood(
            heldout, mass_table=heldout_table)))

    # ------------------------------------------------------------- report --
    print()
    print("=" * 108)
    print("TABLE 2 - entry-path reachability: which site raises, and which "
          "config branch executed")
    print("=" * 108)
    for label, outcome, hits in cases:
        print(f"\n[{label}]")
        print(f"    outcome : {outcome[0]} {outcome[1]}")
        if outcome[0] == "REJECT":
            print(f"    raised@ : {outcome[3]}")
            print(f"    message : {outcome[2][:150]}")
        if hits:
            print("    invariant lines executed:")
            for f, ln in hits:
                tbl = (CONFIG_INVARIANT_LINES if f == "config.py"
                       else EXSUP_INVARIANT_LINES if f == "excitation_support.py"
                       else POLY_INVARIANT_LINES)
                print(f"        {f}:{ln}  {tbl[ln]}")
        else:
            print("    invariant lines executed: NONE")

    print()
    print("=" * 108)
    print("CONFIG-BRANCH REACHABILITY SUMMARY")
    print("=" * 108)
    all_hits = {(f, ln) for _, _, hs in cases for (f, ln) in hs}
    for ln, desc in sorted(CONFIG_INVARIANT_LINES.items()):
        state = "REACHED" if ("config.py", ln) in all_hits else "NEVER EXECUTED"
        print(f"  config.py:{ln:<5} {desc:<62} {state}")
    print()
    for ln, desc in sorted(EXSUP_INVARIANT_LINES.items()):
        state = ("REACHED" if ("excitation_support.py", ln) in all_hits
                 else "never executed")
        print(f"  excitation_support.py:{ln:<5} {desc:<52} {state}")
    print()
    for ln, desc in sorted(POLY_INVARIANT_LINES.items()):
        state = ("REACHED" if ("polygon_mass.py", ln) in all_hits
                 else "never executed")
        print(f"  polygon_mass.py:{ln:<5} {desc:<58} {state}")


if __name__ == "__main__":
    main()
