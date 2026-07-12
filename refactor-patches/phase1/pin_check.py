import os, sys, json
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd, jax
import numpyro.distributions as dist
from numpyro import handlers
from bstpp.main import Hawkes_Model, LGCP_Model

T_DAYS = 2.5*365.0; rng = np.random.RandomState(0); N = 60
DATA = pd.DataFrame({"X": rng.uniform(0.05,0.95,N), "Y": rng.uniform(0.05,0.95,N),
                     "T": np.sort(rng.uniform(0,T_DAYS,N))})
A = np.array([[0.,1.],[0.,1.]])
PRIORS = dict(a_0=dist.Normal(0,5), alpha=dist.Beta(2,2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
def tr_at(m, params):
    s = handlers.substitute(handlers.seed(m.model, jax.random.PRNGKey(0)), params)
    return handlers.trace(s).get_trace(m.args)
out = {}
p = {k: np.float32(v) for k,v in dict(a_0=0.5, alpha=0.3, beta=2.0, sigmax_2=0.1).items()}
hw = Hawkes_Model(DATA, A, T_DAYS, cox_background=False, **PRIORS)
t = tr_at(hw, p); out["hawkes"] = {k: repr(np.asarray(t[k]["value"]).tolist()) for k in ("loglik","Itot_excite","Itot_txy")}
ch = Hawkes_Model(DATA, A, T_DAYS, cox_background="cox", **PRIORS)
t2 = handlers.trace(handlers.seed(ch.model, jax.random.PRNGKey(3))).get_trace(ch.args)
lat = {k: t2[k]["value"] for k in ("a_0","z_temporal","z_seasonal","z_spatial","alpha","beta","sigmax_2")}
t = tr_at(ch, lat); out["cox_hawkes"] = {k: repr(np.asarray(t[k]["value"]).tolist()) for k in ("loglik","Itot_time","Itot_xy","Itot_excite")}
lg = LGCP_Model(DATA, A, T_DAYS, a_0=dist.Normal(0,5))
t3 = handlers.trace(handlers.seed(lg.model, jax.random.PRNGKey(3))).get_trace(lg.args)
lat = {k: t3[k]["value"] for k in ("a_0","z_temporal","z_seasonal","z_spatial")}
t = tr_at(lg, lat); out["lgcp"] = {k: repr(np.asarray(t[k]["value"]).tolist()) for k in ("loglik","Itot_time","Itot_xy")}
print(json.dumps(out, indent=0, sort_keys=True))
