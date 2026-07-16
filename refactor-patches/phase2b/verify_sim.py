"""Simulator verification against THEORETICAL targets (Phase 2b acceptance).

Checks, per review: (1) background count mean AND variance against the
compensator (Poisson index); (2) temporal-segment frequencies against
normalized breakpoint masses; (3) spatial-cell frequencies against normalized
refinement masses; (4) the intentionally-changed ARRAY-domain case against its
mathematical target; (5) pure-Hawkes background per-cell counts against
background_masses. KS pre/post comparisons are SMOKE, not acceptance.
Usage: python verify_sim.py <repo_path>
"""
import os, sys
os.environ.setdefault("JAX_PLATFORM_NAME","cpu")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd, geopandas as gpd, jax
from shapely.geometry import box
import numpyro.distributions as dist
from numpyro import handlers
from bstpp.main import Hawkes_Model
from bstpp.likelihood import (seasonal_time_integral, spatial_refinement_masses,
                              background_masses)

T_DAYS=2.5*365.; r=np.random.RandomState(0); N=60
DATA=pd.DataFrame({"X":r.uniform(.05,.95,N),"Y":r.uniform(.05,.95,N),"T":np.sort(r.uniform(0,T_DAYS,N))})
PR=dict(a_0=dist.Normal(0,5),alpha=dist.Beta(2,2),beta=dist.HalfNormal(1.),sigmax_2=dist.HalfNormal(.25))
fails=[]

def breakpoints(m):
    n_t,T_int=m.args['n_t'],m.args['T']; n_s,off=m.args['n_s'],m.args['offset_seasonal']
    edges=np.arange(n_t+1)*(T_int/n_t); h_day=m.S/n_s
    m_lo=int(np.ceil(off/h_day-1e-9)); m_hi=int(np.floor((m.T+off)/h_day+1e-9))
    cross=(np.arange(m_lo,m_hi+1)*h_day-off)*(T_int/m.T)
    bp=np.unique(np.clip(np.concatenate([edges,cross]),0.,T_int))
    mid=bp[:-1]+.5*np.diff(bp)
    t_cell=np.clip((mid/(T_int/n_t)).astype(int),0,n_t-1)
    s_cell=np.clip(((mid*(m.T/T_int)+off)%m.S/m.S*n_s).astype(int),0,n_s-1)
    return bp,t_cell,s_cell

def cox_block(name, A, R=200):
    m=Hawkes_Model(DATA,A,T_DAYS,cox_background=True,**PR)
    tr=handlers.trace(handlers.seed(m.model,jax.random.PRNGKey(2))).get_trace(m.args)
    pars={"a_0":float(np.asarray(tr["a_0"]["value"])),
          "f_t":np.asarray(tr["f_t"]["value"]),"f_a":np.asarray(tr["f_a"]["value"]),
          "f_xy":np.asarray(tr["f_xy"]["value"])}
    # theoretical rate from the atoms
    _,Ig=seasonal_time_integral(pars["a_0"],pars["f_t"],pars["f_a"],m.args["season_overlap"])
    masses=np.asarray(spatial_refinement_masses(pars["f_xy"],m.args["integration_field_indices"],
                                                m.args["integration_areas"]))
    lam=float(Ig)*float(masses.sum())
    # scale to a manageable size by shifting a_0 if needed
    if lam>400: pars["a_0"]-=np.log(lam/200); \
        lam=float(seasonal_time_integral(pars["a_0"],pars["f_t"],pars["f_a"],m.args["season_overlap"])[1])*float(masses.sum())
    bp,_,_=breakpoints(m)
    g=np.exp(pars["a_0"]+pars["f_t"][np.clip(((bp[:-1]+.5*np.diff(bp))/(m.args['T']/m.args['n_t'])).astype(int),0,m.args['n_t']-1)]
             +pars["f_a"][np.clip((((bp[:-1]+.5*np.diff(bp))*(m.T/m.args['T'])+m.args['offset_seasonal'])%m.S/m.S*m.args['n_s']).astype(int),0,m.args['n_s']-1)])
    w=g*np.diff(bp); p_seg=w/w.sum(); p_cell=masses/masses.sum()
    counts=[]; seg_hist=np.zeros(len(w)); cell_hist=np.zeros(len(masses))
    for rep in range(R):
        np.random.seed(5000+rep); gen=np.random.default_rng(6000+rep)
        bg=m._sim_cox(dict(pars),rng=gen)
        counts.append(len(bg))
        if len(bg):
            seg=np.clip(np.searchsorted(bp,bg[:,2],side="right")-1,0,len(w)-1)
            seg_hist+=np.bincount(seg,minlength=len(w))
            cid=np.clip((bg[:,1]*25).astype(int),0,24)*25+np.clip((bg[:,0]*25).astype(int),0,24)
            cell_hist+=np.bincount(cid,minlength=625)[m.args["integration_field_indices"]] if len(masses)==625 else 0

    def freq_check(hist, probs, n):
        """Frequency check valid in the small-expectation regime: exact Poisson
        two-sided tails with Bonferroni for cells with expected < 5; normal z
        (max over cells, threshold 6) where the approximation holds."""
        from scipy.stats import poisson
        exp_cnt = n * probs
        small = exp_cnt < 5
        worst_z = 0.0 if small.all() else float(np.max(np.abs(
            (hist[~small] - exp_cnt[~small]) / np.sqrt(exp_cnt[~small] * (1 - probs[~small])))))
        if small.any():
            lam_s = np.maximum(exp_cnt[small], 1e-12); obs_s = hist[small]
            pv = np.minimum(poisson.sf(obs_s - 1, lam_s), poisson.cdf(obs_s, lam_s)) * 2
            worst_p = float(pv.min()); n_tests = int(small.sum())
        else:
            worst_p, n_tests = 1.0, 0
        ok = worst_z < 6 and (n_tests == 0 or worst_p > 0.001 / max(n_tests, 1))
        return ok, worst_z, worst_p
    counts=np.array(counts,float); n_tot=counts.sum()
    z=(counts.mean()-lam)/np.sqrt(counts.var(ddof=1)/R)
    idx=counts.var(ddof=1)/counts.mean()
    zi=(idx-1)/np.sqrt(2/(R-1))
    seg_ok, zseg, pseg = freq_check(seg_hist, p_seg, n_tot)
    ok=abs(z)<4 and abs(zi)<4 and seg_ok
    line=(f"[cox/{name}] mean {counts.mean():.1f} vs lam {lam:.1f} (z={z:+.2f}); "
          f"var/mean {idx:.3f} (z={zi:+.2f}); seg max-z {zseg:.2f} min-p {pseg:.3g}")
    if len(masses)==625:
        cell_ok, zc, pc = freq_check(cell_hist, p_cell, n_tot)
        line+=f"; cell max-z {zc:.2f} min-p {pc:.3g}"; ok=ok and cell_ok
    print(("PASS " if ok else "FAIL ")+line)
    if not ok: fails.append(name)

cox_block("gdf-box", gpd.GeoDataFrame({"geometry":[box(0,0,1,1)]}))
cox_block("ARRAY-domain (the intentionally changed case)", np.array([[0.,1.],[0.,1.]]))

# pure-Hawkes covariate background: per-cell counts vs background_masses
COV=gpd.GeoDataFrame({"cov":[0.5,-1.0,1.5,-0.5],
                      "geometry":[box(x,y,x+.5,y+.5) for y in (0,.5) for x in (0,.5)]})
m=Hawkes_Model(DATA,np.array([[0.,1.],[0.,1.]]),T_DAYS,cox_background=False,
               spatial_cov=COV,cov_names=["cov"],**PR)
b0=np.float32([.3,-.2,.5,-.4]); a0=0.2
rates=np.asarray(background_masses(np.exp(a0+b0),np.asarray(m.args["cov_area"]),m.args["T"]))
R=300; per_cell=np.zeros((R,4))
for rep in range(R):
    np.random.seed(7000+rep)
    bg=m._sim_hawkes_bg({"a_0":a0,"b_0":b0})
    if len(bg):
        cell=(bg[:,1]>=.5).astype(int)*2+(bg[:,0]>=.5).astype(int)
        per_cell[rep]=np.bincount(cell,minlength=4)
zc=(per_cell.mean(0)-rates)/np.sqrt(per_cell.var(0,ddof=1)/R)
ok=np.all(np.abs(zc)<4)
print(("PASS " if ok else "FAIL ")+f"[hawkes-bg/cov] per-cell mean {np.round(per_cell.mean(0),2).tolist()} vs rates {np.round(rates,2).tolist()}; z {np.round(zc,2).tolist()}")
if not ok: fails.append("hawkes-bg")
print("VERIFY_SIM:", "ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(0 if not fails else 1)
