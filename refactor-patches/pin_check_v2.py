"""Golden pins for refactor verification: loglik VALUES and GRADIENTS.

Usage: python pin_check.py <repo_path> > pins.json
Compares runs with `diff`. Pure lifts must be bit-identical; commits that
declare an expression change re-baseline with a documented tolerance check.

Gradients are taken with jax.value_and_grad of the traced 'loglik'
deterministic w.r.t. EVERY continuous latent (scalars and z-vectors), because
NUTS/SVI consume gradients: a refactor can preserve values while breaking
differentiability (stray np conversion, dtype promotion, stopped gradient).
"""
import os, sys, json
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd, jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import handlers
from bstpp.main import Hawkes_Model, LGCP_Model

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
print(json.dumps(out, indent=0, sort_keys=True))
