"""Golden pins for refactor verification: loglik VALUES and GRADIENTS.

Usage: python pin_check.py <repo_path> > pins.json
Compares runs with `diff`. Pure lifts must be bit-identical; commits that
declare an expression change re-baseline with a documented tolerance check.

Gradients are taken with jax.value_and_grad of the traced 'loglik'
deterministic w.r.t. EVERY continuous latent (scalars and z-vectors), because
NUTS/SVI consume gradients: a refactor can preserve values while breaking
differentiability (stray np conversion, dtype promotion, stopped gradient).

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
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd, jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import handlers
from bstpp.main import Hawkes_Model, LGCP_Model

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
print(json.dumps(out, indent=0, sort_keys=True))
