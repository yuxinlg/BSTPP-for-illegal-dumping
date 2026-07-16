import os, sys
os.environ.setdefault("JAX_PLATFORM_NAME","cpu")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])  # repo path, "." from repo root
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box
import numpyro.distributions as dist
from bstpp.main import Hawkes_Model
T_DAYS=2.5*365.; r=np.random.RandomState(0); N=60
DATA=pd.DataFrame({"X":r.uniform(.05,.95,N),"Y":r.uniform(.05,.95,N),"T":np.sort(r.uniform(0,T_DAYS,N))})
A=gpd.GeoDataFrame({"geometry":[box(0,0,1,1)]})
PR=dict(a_0=dist.Normal(0,5),alpha=dist.Beta(2,2),beta=dist.HalfNormal(1.),sigmax_2=dist.HalfNormal(.25))
COV=gpd.GeoDataFrame({"cov":[0.5,-1.0,1.5,-0.5],
                      "geometry":[box(x,y,x+.5,y+.5) for y in (0,.5) for x in (0,.5)]})
n_xy=25
zero=dict(f_t=np.zeros(50,np.float32), f_a=np.zeros(24,np.float32),
          f_xy=np.zeros(n_xy**2,np.float32))
configs={
 "cox":   (Hawkes_Model(DATA,A,T_DAYS,cox_background="cox",**PR),
           dict(a_0=np.float32(0.7),alpha=np.float32(.3),beta=np.float32(2.),
                sigmax_2=np.float32(.02),**zero)),
 "plain": (Hawkes_Model(DATA,A,T_DAYS,cox_background=False,**PR),
           dict(a_0=np.float32(0.7),alpha=np.float32(.3),beta=np.float32(2.),
                sigmax_2=np.float32(.02))),
 "cov":   (Hawkes_Model(DATA,A,T_DAYS,cox_background=False,spatial_cov=COV,
                        cov_names=["cov"],**PR),
           dict(a_0=np.float32(0.2),alpha=np.float32(.3),beta=np.float32(2.),
                sigmax_2=np.float32(.02),w=np.float32([0.5]))),
}
R=100; out={}
for name,(m,pars) in configs.items():
    counts=[]; Ts=[]; Xs=[]
    for rep in range(R):
        np.random.seed(1000+rep); g=np.random.default_rng(2000+rep)
        s=m.simulate(parameters=dict(pars),rng=g)
        counts.append(len(s)); Ts.append(np.asarray(s["T"])); Xs.append(np.asarray(s["X"]))
    out[name+"_n"]=np.array(counts)
    out[name+"_T"]=np.concatenate(Ts); out[name+"_X"]=np.concatenate(Xs)
np.savez(sys.argv[2], **out)
print({k:(f"{v.mean():.2f}" if k.endswith("_n") else len(v)) for k,v in out.items()})
