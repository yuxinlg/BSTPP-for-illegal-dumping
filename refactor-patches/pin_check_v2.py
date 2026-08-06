"""Golden pins for refactor verification: loglik VALUES and GRADIENTS.

Usage: python pin_check.py <repo_path> > pins.json
Compares runs with `diff`. Pure lifts must be bit-identical; commits that
declare an expression change re-baseline with a documented tolerance check.

Gradients are taken with jax.value_and_grad of the traced 'loglik'
deterministic w.r.t. EVERY continuous latent (scalars and z-vectors), because
NUTS/SVI consume gradients: a refactor can preserve values while breaking
differentiability (stray np conversion, dtype promotion, stopped gradient).

SIX CONFIGURATIONS SINCE A-47, NOT FOUR. The last two are OP-24's polygon-mode
pin, landed as a WP5 ENTRY PRECONDITION and deliberately built outside WP5.
They are a FORWARD BASELINE: they certify commits taken AFTER their first
capture and say nothing about any commit before it. The baseline this harness
has always compared against, refactor-patches/baselines-2026-07/pins.json,
carries only the first four and is NOT extended -- its bytes are quoted in
every historical capture's provenance block, and rewriting it would make those
captures describe a file that no longer exists. Pin 5's baseline is a separate,
later-dated file passed with pin_compare.py --baseline.

PROVENANCE GOES TO STDERR, NOT STDOUT (A-38). This script emits no MATCH line
-- it produces the CANDIDATE, and the verdict is
`refactor-patches/pin_compare.py`'s, which is where the path-and-hash
provenance lives. Here stdout IS the artifact and must stay byte-comparable
with the baseline, so anything printed to it would corrupt the very file the
comparison reads. The identifying facts a reader needs about a capture -- WHICH
repo was measured, WHICH bstpp answered the import (a stale copy in
site-packages shadows the repo for scripts run from a subdirectory), and what
the canonical baseline hashed to at the time -- therefore go to stderr, which
the harness already captures beside the JSON as `*_pins_stderr.txt`.
"""
import os, sys, json, hashlib, subprocess
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings; warnings.filterwarnings("ignore")
# Every import below resolves against the tree named on THIS harness's command
# line, so it cannot precede the insert: moved above it, `bstpp` would come
# from whatever is on the default path and the capture would describe another
# object. Ruff's E402 is switched off for this file in pyproject.toml with
# that reason attached (A-48); it is a requirement of the mechanism, not an
# inherited untidiness.
sys.path.insert(0, sys.argv[1])
import geopandas as gpd
import numpy as np, pandas as pd, jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import handlers
from shapely.geometry import Point, Polygon
from bstpp.main import Hawkes_Model, LGCP_Model
from bstpp.polygon_mass import prepare_polygon_mass_table

# ------------------------------------------------- A-38 capture provenance --
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANONICAL_BASELINE = os.path.join(_HERE, "baselines-2026-07", "pins.json")


def _emit_provenance():
    """Describe this capture on stderr. See the module docstring for why."""
    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=sys.argv[1], text=True,
                                  capture_output=True).stdout.strip() or "?"
        except Exception:                       # noqa: BLE001 - never fatal
            return "?"

    # Read from sys.modules rather than adding an `import bstpp` line: the
    # package is already imported via bstpp.main, and a new import below the
    # sys.path.insert would add a ruff E402 to a file whose ten findings are
    # all inherited. Measured: this keeps introduced findings at zero.
    bstpp_file = getattr(sys.modules.get("bstpp"), "__file__", "?")

    print("PIN_CAPTURE_PROVENANCE", file=sys.stderr)
    print(f"  repo_under_test  : {os.path.abspath(sys.argv[1])}", file=sys.stderr)
    print(f"  bstpp.__file__   : {bstpp_file}", file=sys.stderr)
    print(f"  git_rev          : {_git('rev-parse', '--short', 'HEAD')}",
          file=sys.stderr)
    dirty = _git("status", "--porcelain")
    tracked = [ln for ln in dirty.splitlines() if not ln.startswith("??")]
    print(f"  tracked_dirty    : {len(tracked)}", file=sys.stderr)
    for ln in tracked:
        print(f"    {ln}", file=sys.stderr)
    if os.path.isfile(_CANONICAL_BASELINE):
        with open(_CANONICAL_BASELINE, "rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        print(f"  canonical_baseline        : {_CANONICAL_BASELINE}",
              file=sys.stderr)
        print(f"  canonical_baseline_sha256 : {h}", file=sys.stderr)
    else:
        print("  canonical_baseline        : NOT FOUND", file=sys.stderr)
    print("  (this script emits no verdict; compare with "
          "refactor-patches/pin_compare.py)", file=sys.stderr)


_emit_provenance()

T_DAYS = 2.5*365.0; rng = np.random.RandomState(0); N = 60
DATA = pd.DataFrame({"X": rng.uniform(0.05,0.95,N), "Y": rng.uniform(0.05,0.95,N),
                     "T": np.sort(rng.uniform(0,T_DAYS,N))})
A = np.array([[0.,1.],[0.,1.]])
PRIORS = dict(a_0=dist.Normal(0,5), alpha=dist.Beta(2,2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))

def loglik_fn(model):
    def f(params):
        s = handlers.substitute(handlers.seed(model.model, jax.random.PRNGKey(0)), params)
        tr = handlers.trace(s).get_trace(model.args)
        return tr["loglik"]["value"]
    return f

def pin(model, params):
    val, grads = jax.value_and_grad(loglik_fn(model))(params)
    rec = {"loglik": repr(float(val))}
    for k in sorted(grads):
        g = np.asarray(grads[k])
        assert np.all(np.isfinite(g)), f"non-finite gradient at {k}"
        rec["grad_"+k] = repr(g.tolist())
    return rec

out = {}
p = {k: jnp.float32(v) for k,v in dict(a_0=0.5, alpha=0.3, beta=2.0, sigmax_2=0.1).items()}
hw = Hawkes_Model(DATA, A, T_DAYS, cox_background=False, **PRIORS)
out["hawkes"] = pin(hw, p)

ch = Hawkes_Model(DATA, A, T_DAYS, cox_background="cox", **PRIORS)
t2 = handlers.trace(handlers.seed(ch.model, jax.random.PRNGKey(3))).get_trace(ch.args)
lat = {k: t2[k]["value"] for k in ("a_0","z_temporal","z_seasonal","z_spatial","alpha","beta","sigmax_2")}
out["cox_hawkes"] = pin(ch, lat)

lg = LGCP_Model(DATA, A, T_DAYS, a_0=dist.Normal(0,5))
t3 = handlers.trace(handlers.seed(lg.model, jax.random.PRNGKey(3))).get_trace(lg.args)
lat = {k: t3[k]["value"] for k in ("a_0","z_temporal","z_seasonal","z_spatial")}
out["lgcp"] = pin(lg, lat)

# --- NON-SQUARE (4:1) config: the discriminating pin for the real-unit
# spatial-trigger contract. On the unit box the old (internal-isotropic) and
# new (real-isotropic) kernels coincide algebraically, so the three configs
# above CANNOT distinguish them; this one can, because the per-axis affine
# ingestion scales differ (sx=4, sy=1). Same 60-point cloud, stretched 4x in
# X only, so every pair has nonzero displacement on BOTH axes and grad_sigmax_2
# is informative. sigmax_2=0.1 is read in the units the code under test
# defines: internal-square units before the contract, squared REAL units
# after -- the pin MOVING at exactly the contract commit is the fix's
# signature; moving anywhere else is a defect.
A_NS = np.array([[0.,4.],[0.,1.]])
DATA_NS = pd.DataFrame({"X": 4.0*DATA["X"].values, "Y": DATA["Y"].values,
                        "T": DATA["T"].values})
hn = Hawkes_Model(DATA_NS, A_NS, T_DAYS, cox_background=False, **PRIORS)
out["hawkes_nonsquare_4to1"] = pin(hn, dict(p))

# --------------------------------------------------------- OP-24: pin 5 ----
# THE POLYGON REGIME, IN BOTH EXCITATION SUPPORT MODES. Until this
# configuration existed the harness contained zero occurrences of `polygon`,
# `mass_table`, `excitation_support` or `min_sigma`, so PIN_DIFFS was not weak
# evidence about ExcitationSupport / PolygonMassTable / cutoff provenance -- it
# was no evidence at all, about exactly the surfaces WP5 changes (A-36/A-44).
# Built OUTSIDE WP5 on purpose: an instrument built by the package it is meant
# to gate is not a gate.
#
# BOTH MODES ARE EMITTED, over the SAME domain, the SAME events and the SAME
# sigma bounds, so the only difference between the two records is the mode
# switch itself -- which is the surface the six items routed to WP5 all touch.
# Pinning one mode would leave the other unreachable, which is the defect this
# configuration exists to remove, half-fixed.
#
# EVERY CHOICE BELOW IS AN EXPLICIT PIN CHOICE, NOT A PACKAGE DEFAULT. D-29
# forbids a sigma-bound default and is not relaxed to make a pin convenient:
# min_sigma / max_sigma are stated here and reported in the provenance block,
# and panel_h_m is stated because the shipped default (20 m) is ~20x this
# domain and would be rejected by the panel ratio guard.
POLY_MIN_SIGMA = 0.05          # real units; sqrt(sigmax_2)=0.316 lies inside
POLY_MAX_SIGMA = 0.5           # [min^2, max^2] = [0.0025, 0.25] truncates the prior
POLY_PANEL_H = 0.4             # == MAX_PANEL_TO_MIN_SIGMA_RATIO * min_sigma
POLY_N = 60

# NON-UNIT ASPECT (4:1) for the reason hawkes_nonsquare_4to1 exists: on a unit
# box the internal-isotropic and real-isotropic kernels coincide algebraically,
# so a unit-box polygon pin would reintroduce that blind spot in the polygon
# regime. NON-RECTANGULAR (a notch in the top edge) so polygon mass is not the
# rectangle's analytic box mass -- against a plain 4:1 box the two modes would
# agree by construction and the mode-switch pin would be two copies of one
# number. Measured here: the two legs differ by ~0.894 nats.
POLY = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 1.0),
                (2.4, 1.0), (2.4, 0.45), (1.6, 0.45), (1.6, 1.0),
                (0.0, 1.0)])
A_POLY = gpd.GeoDataFrame(geometry=[POLY])

# Rejection sampling on a fixed stream: the cloud must lie INSIDE the notched
# domain, so DATA/DATA_NS cannot be reused.
_rng = np.random.RandomState(0)
_xs, _ys = [], []
while len(_xs) < POLY_N:
    _x = _rng.uniform(0.05, 3.95)
    _y = _rng.uniform(0.05, 0.95)
    if POLY.contains(Point(_x, _y)):
        _xs.append(_x)
        _ys.append(_y)
DATA_POLY = pd.DataFrame({"X": np.array(_xs), "Y": np.array(_ys),
                          "T": np.sort(_rng.uniform(0, T_DAYS, POLY_N))})

MASS_TABLE = prepare_polygon_mass_table(
    POLY, DATA_POLY["X"].to_numpy(dtype=float),
    DATA_POLY["Y"].to_numpy(dtype=float),
    min_sigma=POLY_MIN_SIGMA, max_sigma=POLY_MAX_SIGMA,
    panel_h_m=POLY_PANEL_H, crs=None)


def _mass_table_sha256(tbl):
    """The table is part of the pin's identity (D-27/A-11 integrity clause).

    A pin that silently depends on a regenerable artifact is not reproducible:
    rebuild the table differently and the loglik moves with no record of why.
    Hashing the interpolation data INTO the pinned record makes that a DRIFT.
    """
    h = hashlib.sha256()
    for arr in (tbl.log_knots, tbl.values, tbl.slopes):
        h.update(np.ascontiguousarray(np.asarray(arr), dtype=np.float64).tobytes())
    return h.hexdigest()


_table_sha = _mass_table_sha256(MASS_TABLE)

hp = Hawkes_Model(DATA_POLY, A_POLY, T_DAYS, cox_background=False,
                  excitation_support="polygon",
                  min_sigma=POLY_MIN_SIGMA, max_sigma=POLY_MAX_SIGMA,
                  mass_table=MASS_TABLE, **PRIORS)
rec = pin(hp, dict(p))
rec["mass_table_sha256"] = _table_sha
out["hawkes_notched_4to1_polygon_mode"] = rec

hr = Hawkes_Model(DATA_POLY, A_POLY, T_DAYS, cox_background=False,
                  excitation_support="rectangle",
                  min_sigma=POLY_MIN_SIGMA, max_sigma=POLY_MAX_SIGMA,
                  **PRIORS)
out["hawkes_notched_4to1_rectangle_mode"] = pin(hr, dict(p))

# Pin-5 provenance to STDERR, beside the capture, for the same reason the
# A-38 block goes there: stdout IS the artifact.
print("PIN5_POLYGON_PROVENANCE  (OP-24; explicit choices, not defaults)",
      file=sys.stderr)
print(f"  domain            : notched octagon, bounds {POLY.bounds}, "
      f"area {POLY.area!r}, aspect 4:1, NON-rectangular", file=sys.stderr)
print(f"  events            : {POLY_N}, RandomState(0) rejection-sampled "
      "inside the polygon", file=sys.stderr)
print(f"  min_sigma         : {POLY_MIN_SIGMA}   (real units; explicit, D-29 "
      "forbids a default)", file=sys.stderr)
print(f"  max_sigma         : {POLY_MAX_SIGMA}   (explicit)", file=sys.stderr)
print(f"  panel_h_m         : {POLY_PANEL_H}   (explicit; shipped default 20.0 "
      "fails the panel ratio guard on this domain)", file=sys.stderr)
print(f"  gl_order          : {int(MASS_TABLE.gl_order)}   (shipped default)",
      file=sys.stderr)
print(f"  knots x events    : {MASS_TABLE.n_knots} x {MASS_TABLE.n_events}",
      file=sys.stderr)
print(f"  MASS_TABLE_SHA256 : {_table_sha}", file=sys.stderr)
print("  excitation modes  : polygon AND rectangle, same domain/events/bounds",
      file=sys.stderr)
print("  FORWARD BASELINE  : pin 5 certifies commits AFTER this capture. It "
      "cannot certify b98e91d or 0e78f7d, which predate it.", file=sys.stderr)

print(json.dumps(out, indent=0, sort_keys=True))
