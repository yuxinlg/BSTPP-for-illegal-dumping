"""A-36 measurement: the silent-collapse census (report only; nothing committed
as a production change by this probe).

PREDICATE. A (entry point, parameter) pair is IN CLASS when an
accepted-but-unintended TYPE for that parameter yields a VALID, DIFFERENT
computation instead of an error. Four generators are tested:

  G1  bool as int      -- ``bool`` is an ``int`` subclass; ``True`` is 1.
  G2  None as default  -- ``None`` collapsing to a meaningful default, so that
                          "not supplied" and "explicitly nothing" agree.
  G3  float narrowing  -- a float silently truncated to an int.
  G4  empty container  -- an empty list/array standing in for an absent one.

ENUMERATION IS BY ENTRY POINT INTO COMPUTATION, NOT BY ARGUMENT NAME. Name
ordering is what missed ``prepare_polygon_mass_table`` at A-34.

Each row is FIRED, not reasoned about. The verdict is one of

  RAISED      -- the call raised; not in class.
  COLLAPSED   -- the call returned/constructed, and the value the computation
                 ran on differs from the value the caller's type suggests.
  SAME        -- the call returned and the computation is identical (the
                 unintended type coerces to the same number).
  N/A         -- the row could not be fired here; stated, never assumed.

THE COVERED COLUMN (A-39). A RAISED verdict is not one thing, and reporting it
as one overstates the protection. Two kinds are distinguished, and the
distinction is DERIVED BY FIRING, not hand-labelled:

  type-discipline    the same NUMERIC VALUE supplied with the intended type is
                     ACCEPTED, so what rejected the bool was the type check.
                     CI-7 / CI-8. Protected: a suite row asserts it.

  interval-accident  the same numeric value with the intended type is ALSO
                     rejected, so nothing about the TYPE was tested -- the bool
                     merely coerced to a number outside a permitted interval.
                     `True -> 1.0` and `False -> 0.0` both fall outside (0, 1),
                     so the tolerance family is covered BY THE SHAPE OF THE
                     INTERVAL and by nothing else.

  UNPROTECTED, and measured: the column `suite_pins_bool` greps tests/ for the
  parameter supplied as a bool literal. Where it is False, WIDENING THE
  INTERVAL TO (0, 1] WOULD MAKE `True` ACCEPTED AND SILENTLY COLLAPSE THE
  PARAMETER TO 1.0, AND NOTHING IN THE SUITE WOULD MOVE. The existing rows
  parametrise bad_tol over [-1.0, 0.0, 1.0, nan, inf] -- all floats, no bool --
  so they would simply be updated for the new interval, and the bool case has
  no row to fail.

Usage:  JAX_PLATFORM_NAME=cpu python probe_a36_silent_collapse_census.py
"""
import os
import sys

# A stale bstpp in site-packages shadows the repo for scripts run from a
# subdirectory; the capture would then describe a different object (AGENTS.md).
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
import warnings

import numpy as np
import pandas as pd
import numpyro.distributions as dist

import bstpp
from bstpp.main import Hawkes_Model
from bstpp.config import NumericalConfig
from bstpp import cutoffs, polygon_mass, utils

warnings.filterwarnings("ignore")

print("bstpp.__file__ =", bstpp.__file__)
print()

T_DAYS = 2.5 * 365.0
rng = np.random.RandomState(0)
N = 60
DATA = pd.DataFrame({"X": rng.uniform(0.05, 0.95, N),
                     "Y": rng.uniform(0.05, 0.95, N),
                     "T": np.sort(rng.uniform(0, T_DAYS, N))})
A = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))

ROWS = []
TESTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests")


def _suite_pins_bool(param: str) -> bool:
    """Does any suite row supply this parameter as a bool literal?

    Mechanical and deliberately crude: it greps for `<param>=True` /
    `=False` across tests/. A crude check that is STATED beats a careful one
    that is asserted, and it errs toward reporting MORE protection than exists
    (a match may be incidental), so a False here is the strong finding.
    """
    needles = (f"{param}=True", f"{param}=False",
               f"{param} = True", f"{param} = False")
    for root, _dirs, files in os.walk(TESTS_DIR):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            try:
                text = open(os.path.join(root, fn), encoding="utf-8").read()
            except OSError:
                continue
            if any(n in text for n in needles):
                return True
    return False


def row(entry, param, generator, probe, read, typed_probe=None):
    """Fire one census row.

    ``read`` maps the constructed object to the value the computation actually
    ran on. ``typed_probe`` re-fires the SAME NUMERIC VALUE with the INTENDED
    type; it is what separates a type-discipline rejection from an
    interval-accident one, and it is only consulted when the row RAISED.
    """
    try:
        obj = probe()
    except Exception as exc:  # noqa: BLE001 -- the verdict IS the exception
        detail = f"{type(exc).__name__}: {str(exc).splitlines()[0][:96]}"
        if typed_probe is None:
            covered = "unclassified (no typed probe supplied)"
        else:
            try:
                typed_probe()
                covered = "type-discipline"
            except Exception:  # noqa: BLE001 -- rejected on value, not type
                pinned = _suite_pins_bool(param.split()[0])
                covered = ("interval-accident "
                           f"[suite_pins_bool={pinned}"
                           f"{'' if pinned else '  <-- UNPROTECTED'}]")
        ROWS.append((entry, param, generator, "RAISED", detail, covered))
        return
    try:
        seen = read(obj) if read is not None else "<constructed>"
    except Exception as exc:  # noqa: BLE001
        seen = f"<unreadable: {type(exc).__name__}: {exc}>"
    ROWS.append((entry, param, generator, "COLLAPSED",
                 f"computed on {seen!r}", "n/a (nothing rejected it)"))


# ---------------------------------------------------------------------------
# Entry point 1: Point_Process_Model.__init__ (reached via Hawkes_Model)
# ---------------------------------------------------------------------------
row("Hawkes_Model.__init__", "T", "G1 bool-as-int",
    lambda: Hawkes_Model(DATA[DATA["T"] < 1.0], A, True,
                         cox_background=False, excitation_support="rectangle",
                         **PRIORS),
    lambda m: (m.T, m.args["T"]))

row("Hawkes_Model.__init__", "offset_seasonal", "G1 bool-as-int",
    lambda: Hawkes_Model(DATA, A, T_DAYS, offset_seasonal=True,
                         cox_background=False, excitation_support="rectangle",
                         **PRIORS),
    lambda m: (m.args["offset_seasonal"],
               float(np.asarray(m.args["season_overlap"]).sum())))

row("Hawkes_Model.__init__", "sp_var_mu", "G1 bool-as-int",
    lambda: Hawkes_Model(DATA, A, T_DAYS, sp_var_mu=True,
                         cox_background="cox", excitation_support="rectangle",
                         **PRIORS),
    lambda m: m.args["sp_var_mu"])

row("Hawkes_Model.__init__", "window", "G1 bool-as-int",
    lambda: Hawkes_Model(DATA, A, T_DAYS, window=True,
                         cox_background=False, excitation_support="rectangle",
                         **PRIORS),
    lambda m: m.args["window"])

row("Hawkes_Model.__init__", "spatial_window", "G1 bool-as-int",
    lambda: Hawkes_Model(DATA, A, T_DAYS, spatial_window=True,
                         cox_background=False, excitation_support="rectangle",
                         **PRIORS),
    lambda m: m.args["spatial_window"])

row("Hawkes_Model.__init__", "temporal_cutoff_days", "G1 bool-as-int",
    lambda: Hawkes_Model(DATA, A, T_DAYS, temporal_cutoff_days=True,
                         cox_background=False, excitation_support="rectangle",
                         **PRIORS),
    lambda m: m.args["window"])

row("Hawkes_Model.__init__", "design_sigma", "G1 bool-as-int",
    lambda: Hawkes_Model(DATA, A, T_DAYS, spatial_cutoff_tol=0.05,
                         design_sigma=True,
                         cox_background=False, excitation_support="rectangle",
                         **PRIORS),
    lambda m: m.args["spatial_window"])

row("Hawkes_Model.__init__", "design_mean_lag_days", "G1 bool-as-int",
    lambda: Hawkes_Model(DATA, A, T_DAYS, temporal_cutoff_tol=0.05,
                         design_mean_lag_days=True,
                         cox_background=False, excitation_support="rectangle",
                         **PRIORS),
    lambda m: m.args["window"])

row("Hawkes_Model.__init__", "cutoff_tol", "G1 bool-as-int",
    lambda: Hawkes_Model(DATA, A, T_DAYS, cutoff_tol=True,
                         design_mean_lag_days=30.0, design_sigma=0.05,
                         cox_background=False, excitation_support="rectangle",
                         **PRIORS),
    lambda m: (m.args["window"], m.args["spatial_window"]),
    typed_probe=lambda: Hawkes_Model(DATA, A, T_DAYS, cutoff_tol=1.0,
                                     design_mean_lag_days=30.0, design_sigma=0.05,
                                     cox_background=False,
                                     excitation_support="rectangle", **PRIORS))

row("Hawkes_Model.__init__", "min_sigma/max_sigma", "G1 bool-as-int",
    lambda: Hawkes_Model(DATA, A, T_DAYS, min_sigma=True, max_sigma=2,
                         cox_background=False, excitation_support="rectangle",
                         **PRIORS),
    lambda m: (m.numerical_config.min_sigma, m.numerical_config.max_sigma),
    typed_probe=lambda: Hawkes_Model(DATA, A, T_DAYS, min_sigma=1.0, max_sigma=2,
                                     cox_background=False,
                                     excitation_support="rectangle", **PRIORS))

# Covariate leg: cov_grid_size and standardize_cov.
COV = None
try:
    import geopandas as gpd
    import shapely

    cells, vals = [], []
    for i in range(4):
        for j in range(4):
            cells.append(shapely.box(i / 4, j / 4, (i + 1) / 4, (j + 1) / 4))
            vals.append(float(i + 2 * j))
    COV = gpd.GeoDataFrame({"c1": vals}, geometry=cells)
except Exception as exc:  # noqa: BLE001
    print("geopandas unavailable, covariate rows are N/A:", exc)

if COV is not None:
    row("Hawkes_Model.__init__", "standardize_cov (no covariates)",
        "G2 None-vs-set (guard never reached)",
        lambda: Hawkes_Model(DATA, A, T_DAYS, standardize_cov="domain_area",
                             cox_background=False,
                             excitation_support="rectangle", **PRIORS),
        lambda m: getattr(m, "standardization", "<no standardization attr>"))

    row("Hawkes_Model.__init__", "standardize_cov=True (no covariates)",
        "G1 bool (guard never reached)",
        lambda: Hawkes_Model(DATA, A, T_DAYS, standardize_cov=True,
                             cox_background=False,
                             excitation_support="rectangle", **PRIORS),
        lambda m: getattr(m, "standardization", "<no standardization attr>"))

    row("Hawkes_Model.__init__", "standardize_cov=True (+covariates)",
        "G1 bool (guard reached)",
        lambda: Hawkes_Model(DATA, A, T_DAYS, spatial_cov=COV,
                             cov_names=["c1"], standardize_cov=True,
                             cox_background=False,
                             excitation_support="rectangle", **PRIORS),
        lambda m: m.standardization,
        typed_probe=lambda: Hawkes_Model(DATA, A, T_DAYS, spatial_cov=COV,
                                         cov_names=["c1"],
                                         standardize_cov="domain_area",
                                         cox_background=False,
                                         excitation_support="rectangle",
                                         **PRIORS))

    row("Hawkes_Model.__init__", "cov_names=[] (+covariates)",
        "G4 empty container",
        lambda: Hawkes_Model(DATA, A, T_DAYS, spatial_cov=COV, cov_names=[],
                             cox_background=False,
                             excitation_support="rectangle", **PRIORS),
        lambda m: getattr(m.args, "get", dict().get)("num_cov", "<absent>"))

# ---------------------------------------------------------------------------
# Entry point 2: run_svi / run_mcmc  (sampler control integers)
# ---------------------------------------------------------------------------
_BASE = Hawkes_Model(DATA, A, T_DAYS, cox_background=False,
                     excitation_support="rectangle", **PRIORS)

row("Point_Process_Model.run_svi", "num_steps", "G1 bool-as-int",
    lambda: _BASE.run_svi(True, 0.1, num_samples=2, plot_loss=False),
    lambda r: "<ran; num_steps=True == 1 step>")

row("Point_Process_Model.run_svi", "lr", "G1 bool-as-int",
    lambda: _BASE.run_svi(2, True, num_samples=2, plot_loss=False),
    lambda r: "<ran; lr=True == 1.0>")

row("Point_Process_Model.run_svi", "num_samples", "G1 bool-as-int",
    lambda: _BASE.run_svi(2, 0.1, num_samples=True, plot_loss=False),
    lambda r: "<ran; num_samples=True == 1 draw>")

# ---------------------------------------------------------------------------
# Entry point 3: NumericalConfig.create  (CI-7 / CI-8 -- expected RAISED)
# ---------------------------------------------------------------------------
row("NumericalConfig.create", "gl_order", "G1 bool-as-int",
    lambda: NumericalConfig.create(support_mode="rectangle", gl_order=True),
    lambda c: c.gl_order,
    typed_probe=lambda: NumericalConfig.create(support_mode="rectangle",
                                               gl_order=1))
row("NumericalConfig.create", "gl_order", "G3 float narrowing",
    lambda: NumericalConfig.create(support_mode="rectangle", gl_order=16.7),
    lambda c: c.gl_order,
    typed_probe=lambda: NumericalConfig.create(support_mode="rectangle",
                                               gl_order=16))
row("NumericalConfig.create", "panel_h_m", "G1 bool-as-int",
    lambda: NumericalConfig.create(support_mode="rectangle", panel_h_m=True),
    lambda c: c.panel_h_m,
    typed_probe=lambda: NumericalConfig.create(support_mode="rectangle",
                                               panel_h_m=1.0))

# ---------------------------------------------------------------------------
# Entry point 4: polygon_mass module-level computation entries
# ---------------------------------------------------------------------------
row("polygon_mass.knot_count", "sigma_min", "G1 bool-as-int",
    lambda: polygon_mass.knot_count(True, 4.0), lambda v: v)
row("polygon_mass.log_knots", "sigma_min", "G1 bool-as-int",
    lambda: polygon_mass.log_knots(True, 4.0), lambda v: (v[0], v[-1], len(v)))
row("polygon_mass.make_quad_eval", "gl_order", "G1 bool-as-int",
    lambda: polygon_mass.make_quad_eval(True, 1.0), lambda v: "<built>")
row("polygon_mass.warn_if_sigma_near_bound", "frac", "G1 bool-as-int",
    lambda: polygon_mass.warn_if_sigma_near_bound(
        np.array([0.25]), 1.0, 0.1, frac=True), lambda v: "<ran>")

# ---------------------------------------------------------------------------
# Entry point 5: cutoffs -- the real-day conversion / tolerance atoms
# ---------------------------------------------------------------------------
row("cutoffs.days_to_internal", "days", "G1 bool-as-int",
    lambda: cutoffs.days_to_internal(True, 365.0), lambda v: v)
row("cutoffs.temporal_omitted_mass", "window", "G1 bool-as-int",
    lambda: cutoffs.temporal_omitted_mass(True, 2.0), lambda v: v)
row("cutoffs.temporal_cutoff_from_tol", "eps", "G1 bool-as-int",
    lambda: cutoffs.temporal_cutoff_from_tol(True, 2.0), lambda v: v,
    typed_probe=lambda: cutoffs.temporal_cutoff_from_tol(1.0, 2.0))
row("cutoffs.spatial_cutoff_from_tol", "eps", "G1 bool-as-int",
    lambda: cutoffs.spatial_cutoff_from_tol(True, 0.1), lambda v: v,
    typed_probe=lambda: cutoffs.spatial_cutoff_from_tol(1.0, 0.1))
row("cutoffs.resolve_computational_cutoffs", "horizon_days", "G1 bool-as-int",
    lambda: cutoffs.resolve_computational_cutoffs(
        horizon_days=True, temporal_cutoff_days=0.5, spatial_window=0.1),
    lambda v: (v[0], v[1]))

# ---------------------------------------------------------------------------
# Entry point 6: utils -- the pair builder (event-indexed state)
# ---------------------------------------------------------------------------
row("utils.aligned_difference_pairs", "window", "G1 bool-as-int",
    lambda: utils.aligned_difference_pairs(
        np.linspace(0, 10, 20), np.linspace(0, 1, 20), np.linspace(0, 1, 20),
        True, None),
    lambda v: f"<{np.asarray(v[0]).shape[-1] if len(v) else 0} pairs>")
row("utils.within_real_box_window", "spatial_window", "G1 bool-as-int",
    lambda: utils.within_real_box_window(
        np.array([0.5]), np.array([0.5]), True), lambda v: np.asarray(v).tolist())

# ---------------------------------------------------------------------------
# Entry point 7: set_window  (the _UNSET sentinel's own class)
# ---------------------------------------------------------------------------
row("Hawkes_Model.set_window", "window", "G1 bool-as-int",
    lambda: (_BASE.set_window(window=True), _BASE)[1],
    lambda m: m.args["window"])

# ---------------------------------------------------------------------------
print()
print("=" * 120)
print(f"{'ENTRY POINT':<38}{'PARAMETER':<34}{'VERDICT':<11}{'COVERED BY':<46}DETAIL")
print("=" * 120)
n_coll = 0
for entry, param, gen, verdict, detail, covered in ROWS:
    if verdict == "COLLAPSED":
        n_coll += 1
    print(f"{entry:<38}{param:<34}{verdict:<11}{covered:<46}{detail}")
print("=" * 120)

n_type = sum(1 for r in ROWS if r[5] == "type-discipline")
n_interval = sum(1 for r in ROWS if r[5].startswith("interval-accident"))
n_unprot = sum(1 for r in ROWS if "UNPROTECTED" in r[5])
n_unclass = sum(1 for r in ROWS if r[5].startswith("unclassified"))
print(f"ROWS_FIRED={len(ROWS)}  COLLAPSED={n_coll}  "
      f"RAISED={sum(1 for r in ROWS if r[3] == 'RAISED')}")
print(f"  RAISED_BY_TYPE_DISCIPLINE={n_type}   (CI-7 / CI-8; a suite row asserts it)")
print(f"  RAISED_BY_INTERVAL_ACCIDENT={n_interval}   "
      f"of which UNPROTECTED={n_unprot}")
print(f"  RAISED_UNCLASSIFIED={n_unclass}   (no typed probe supplied)")
print()
if n_unprot:
    print("COVERAGE NO TEST PROTECTS")
    print("  These rows RAISE today, and the census would overstate the")
    print("  package's protection by counting them beside the CI-7/CI-8 rows.")
    print("  Nothing tested the TYPE: True coerced to 1.0 and 1.0 is outside")
    print("  the permitted interval, so the interval did the work. Widen it to")
    print("  (0, 1] -- a change someone could argue for on its merits -- and")
    print("  True is accepted, the parameter silently collapses to 1.0, AND NO")
    print("  SUITE ROW MOVES: the existing rows parametrise bad_tol over")
    print("  [-1.0, 0.0, 1.0, nan, inf], all floats, so they would simply be")
    print("  updated for the new interval, and the bool case has no row at all.")
    for entry, param, gen, verdict, detail, covered in ROWS:
        if "UNPROTECTED" in covered:
            print(f"    {entry}.{param}")
    print()
print("EXIT:0")
