"""Simulate-and-recover harness for the cox-Hawkes model (plumbing check, NOT SBC).

Default mode is IN-CLASS: the truth's latent GP fields are drawn from the model's OWN
generative prior (z ~ N(0, I), decoded by the same VAE decoders and sp_var_mu multiplier
the likelihood uses), so the truth is exactly representable by the fit. This discriminates
between two explanations for the alpha/a_0 miss seen with hand-set fields:
  (a) background MISSPECIFICATION -- truth fields outside the decoder's range are absorbed
      as excitation; or
  (b) a residual code bug in the simulator/likelihood pair.
If recovery is good in-class, (a) is confirmed and the harness is done. If alpha still
inflates in-class UNDER NUTS, that indicates a code bug to find (see pass criteria).

The truth parameters are NOT tuned to make results pass -- the point is measurement.

Inference note: SVI with an AutoMultivariateNormal guide is itself an approximation whose
posterior spread is typically UNDERSTATED, so CI-coverage conclusions are provisional
under SVI; the --nuts path is the reference, and the SBC follow-up (many replicates +
rank histograms) must use NUTS for the same reason.

Out-of-class finding (--out-of-class mode, retained deliberately): "Under background
misspecification the excitation share is inflated; estimates of the self-excited fraction
on real data are upper bounds to the extent the VAE background under-fits."

Usage:
  python scripts/recover_test.py                 # in-class, SVI (num_steps=20000)
  python scripts/recover_test.py --nuts          # in-class, NUTS (reference)
  python scripts/recover_test.py --out-of-class  # hand-set fields (robustness check)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import jax
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import Predictive
from bstpp.main import Hawkes_Model

T_DAYS = 2.5 * 365.0
WINDOW = 5.0                       # consistent excitation window in simulation + fit
A_GDF = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]})
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(2.0), sigmax_2=dist.HalfNormal(0.25))
LATENT = ["a_0", "alpha", "beta", "sigmax_2", "z_temporal", "z_seasonal", "z_spatial"]
DETERMINISTIC = ["f_t", "f_a", "f_xy", "Itot_excite", "Itot_txy"]


def build_model(data):
    return Hawkes_Model(data, A_GDF, T_DAYS, cox_background=True, window=WINDOW, **PRIORS)


def summarize(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), sd=float(x.std()),
               lo=float(np.quantile(x, 0.05)), hi=float(np.quantile(x, 0.95)))


def covered(s, truth):
    return s["lo"] <= truth <= s["hi"]


def make_placeholder(seed):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"X": rng.uniform(0.05, 0.95, 50),
                         "Y": rng.uniform(0.05, 0.95, 50),
                         "T": np.sort(rng.uniform(0, T_DAYS, 50))})


def build_truth_in_class(model, seed):
    rng = np.random.default_rng(seed)
    return {
        "a_0": 2.0, "alpha": 0.3, "beta": 1.0, "sigmax_2": 0.01,
        "z_temporal": rng.standard_normal(model.args["z_dim_temporal"]),
        "z_seasonal": rng.standard_normal(model.args["z_dim_seasonal"]),
        "z_spatial": rng.standard_normal(model.args["z_dim_spatial"]),
    }


def build_truth_out_of_class(model):
    # hand-set fields NOT drawn from the decoder -> deliberately misrepresentable
    n_t, n_s, n_xy = model.args["n_t"], model.args["n_s"], model.args["n_xy"]
    return {
        "a_0": 2.0, "alpha": 0.3, "beta": 1.0, "sigmax_2": 0.05,
        "f_t": np.zeros(n_t),
        "f_a": 0.8 * np.sin(2 * np.pi * np.arange(n_s) / n_s),
        "f_xy": np.zeros(n_xy ** 2),
    }


def simulate_target_events(gen, truth, lo=800, hi=2500, target=1200):
    """simulate(); adjust a_0 ONLY (the count knob) to land the event count in [lo, hi].

    E[N] is proportional to exp(a_0), so a_0 += log(target / N) retargets in one jump; a
    couple of refinements absorb Poisson/branching noise. alpha/beta/sigmax_2 are untouched.
    (The in-class VAE fields carry an exp(sp_var_mu) amplitude, so the count at a fixed a_0
    is much larger than in the flat-field out-of-class case -- hence a_0 usually drops.)
    """
    np.random.seed(123)
    sim = gen.simulate(truth)          # mutates truth: adds decoded f_t/f_a/f_xy (in-class)
    for _ in range(6):
        n = len(sim)
        if lo <= n <= hi:
            break
        truth["a_0"] += 2.0 if n == 0 else float(np.log(target / n))
        sim = gen.simulate(truth)
    return sim


def posterior_dict(fit, use_nuts, num_steps, num_warmup, num_samples, lr=0.01):
    if use_nuts:
        fit.run_mcmc(num_warmup=num_warmup, num_samples=num_samples, num_chains=1)
        latent = {k: np.asarray(v) for k, v in fit.samples.items()}
        pred = Predictive(fit.model, posterior_samples=fit.samples, return_sites=DETERMINISTIC)
        det = pred(jax.random.PRNGKey(1), args=fit.args)
        return {**latent, **{k: np.asarray(v) for k, v in det.items()}}
    fit.run_svi(lr=lr, num_steps=num_steps, num_samples=num_samples, plot_loss=False)
    return {k: np.asarray(v) for k, v in fit.samples.items()}


def truth_integrals(fit, truth):
    """Trace the model's own formulas AT the truth params on the simulated events
    (same window, same truncated integral); return (Itot_excite, Itot_txy).

    Both plug-in targets derive from these: the excitation share Ie/Itxy, and the
    identified log-integral background target log(Itxy - Ie) = log(Itot_time*Itot_xy)."""
    fixed = {k: truth[k] for k in LATENT}
    tr = handlers.trace(handlers.substitute(handlers.seed(fit.model, jax.random.PRNGKey(2)),
                                            fixed)).get_trace(fit.args)
    Ie = float(np.asarray(tr["Itot_excite"]["value"]))
    Itxy = float(np.asarray(tr["Itot_txy"]["value"]))
    return Ie, Itxy


def intercept_combination(a_0, f_t, f_a, f_xy, season_idx):
    """a_0 + mean(f_t) + mean(f_a[season_idx_of_t]) + spatial-mean(f_xy).

    DIAGNOSTIC ONLY -- approximately identified. This is a MEAN of logs, but the
    likelihood pins only LOGS of integrals: non-constant tilts between fields that
    preserve int exp(f) leave this combination free to drift (verified empirically:
    components drifted by -0.76/+0.25/+0.48/-0.44 while log(Itot_time*Itot_xy)
    covered). The pass/fail criterion uses the log-integral target instead."""
    return (np.asarray(a_0) + np.asarray(f_t).mean()
            + np.asarray(f_a)[season_idx].mean() + np.asarray(f_xy).mean())


def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-of-class", action="store_true", help="hand-set (misrepresentable) fields")
    ap.add_argument("--nuts", action="store_true", help="reference inference via NUTS")
    ap.add_argument("--num-steps", type=int, default=20000, help="SVI steps")
    ap.add_argument("--num-warmup", type=int, default=300, help="NUTS warmup")
    ap.add_argument("--num-samples", type=int, default=500, help="posterior samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    in_class = not args.out_of_class

    gen = build_model(make_placeholder(args.seed))
    truth = build_truth_in_class(gen, args.seed) if in_class else build_truth_out_of_class(gen)
    sim = simulate_target_events(gen, truth)
    print(f"mode={'in-class' if in_class else 'out-of-class'}  "
          f"inference={'NUTS' if args.nuts else 'SVI'}  window={WINDOW}")
    print(f"simulated {len(sim)} events (a_0 retargeted to {truth['a_0']:.3f}); "
          f"columns={list(sim.columns)}")
    if "A" in sim.columns:
        raise SystemExit("FATAL: simulate() emitted an 'A' column")

    fit = build_model(pd.DataFrame(sim[["X", "Y", "T"]]))
    post = posterior_dict(fit, args.nuts, args.num_steps, args.num_warmup, args.num_samples)

    # crash/NaN guard -> nonzero exit
    for k in ["alpha", "beta", "sigmax_2", "a_0"]:
        if not np.all(np.isfinite(post[k])):
            raise SystemExit(f"FATAL: non-finite posterior for {k}")

    results = []  # (label, passed)
    print("\nparam       truth      mean       sd        90% CI              covered?")
    print("-" * 74)
    for name in ["alpha", "beta", "sigmax_2", "a_0"]:
        s = summarize(post[name])
        cov = covered(s, truth[name])
        if name != "a_0":                      # a_0 judged via the identified combination
            results.append((name, cov))
        print(f"{name:9s}  {truth[name]:7.3f}  {s['mean']:8.3f}  {s['sd']:7.3f}  "
              f"[{s['lo']:8.3f}, {s['hi']:8.3f}]   {'yes' if cov else 'NO'}")

    if in_class:
        Ie_true, Itxy_true = truth_integrals(fit, truth)

        # Identified background target: log(Itot_txy - Itot_excite) = log(Itot_time*Itot_xy).
        # (Replaces the mean-log intercept combination as the pass/fail criterion;
        # that combination is only approximately identified -- see its docstring.)
        logback_true = float(np.log(Itxy_true - Ie_true))
        logback_post = np.log(np.asarray(post["Itot_txy"]) - np.asarray(post["Itot_excite"]))
        s = summarize(logback_post)
        cov = covered(s, logback_true)
        results.append(("log_background", cov))
        print(f"{'log_bg':9s}  {logback_true:7.3f}  {s['mean']:8.3f}  {s['sd']:7.3f}  "
              f"[{s['lo']:8.3f}, {s['hi']:8.3f}]   {'yes' if cov else 'NO'}   (identified log-integral)")

        # Mean-log combination retained as a printed diagnostic (no pass/fail).
        sidx = fit.args["season_idx_of_t"]
        icc_true = float(intercept_combination(truth["a_0"], truth["f_t"], truth["f_a"],
                                               truth["f_xy"], sidx))
        icc_post = np.array([
            intercept_combination(post["a_0"][i], post["f_t"][i], post["f_a"][i],
                                  post["f_xy"][i], sidx)
            for i in range(len(post["a_0"]))
        ])
        si = summarize(icc_post)
        print(f"{'a0+fbar':9s}  {icc_true:7.3f}  {si['mean']:8.3f}  {si['sd']:7.3f}  "
              f"[{si['lo']:8.3f}, {si['hi']:8.3f}]   ----   (mean-log diagnostic, no pass/fail)")

        share_true = Ie_true / Itxy_true
        share_post = summarize(np.asarray(post["Itot_excite"]) / np.asarray(post["Itot_txy"]))
        share_cov = covered(share_post, share_true)
        results.append(("exc_share", share_cov))
        print(f"{'exc_share':9s}  {share_true:7.3f}  {share_post['mean']:8.3f}  {share_post['sd']:7.3f}  "
              f"[{share_post['lo']:8.3f}, {share_post['hi']:8.3f}]   {'yes' if share_cov else 'NO'}   (plug-in truth)")

        f_a_mean = np.asarray(post["f_a"]).mean(axis=0)
        f_t_mean = np.asarray(post["f_t"]).mean(axis=0)
        f_xy_mean = np.asarray(post["f_xy"]).mean(axis=0)
        c_fa = pearson(f_a_mean, truth["f_a"])
        c_ft = pearson(f_t_mean, truth["f_t"])
        c_fxy = pearson(f_xy_mean, truth["f_xy"])
        fa_pass = c_fa > 0.7
        results.append(("corr_f_a>0.7", fa_pass))
        print(f"\nfield recovery (corr posterior-mean vs truth): "
              f"f_a={c_fa:.3f}  f_t={c_ft:.3f}  f_xy={c_fxy:.3f}")
    else:
        share_post = summarize(np.asarray(post["Itot_excite"]) / np.asarray(post["Itot_txy"]))
        print(f"\nexc_share posterior: mean={share_post['mean']:.3f} "
              f"90% CI [{share_post['lo']:.3f}, {share_post['hi']:.3f}] "
              f"(no plug-in truth: fields are not in the decoder's range)")

    print("\n" + "-" * 74)
    for label, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")

    if in_class and args.nuts:
        alpha_ok = dict(results).get("alpha", False)
        if not alpha_ok and args.num_samples >= 200:
            print("\n*** alpha FAILS in-class under NUTS (converged chain) -> this indicates a "
                  "CODE BUG in the simulator/likelihood pair, not a methodology issue. Stop and "
                  "investigate; do NOT adjust the truth to make it pass. ***")
        elif not alpha_ok:
            print("\n(alpha not covered, but this NUTS chain is too short to have adapted -- NOT "
                  "diagnostic. Increase --num-warmup/--num-samples before concluding anything.)")
    print("\n(single-replicate plumbing check; calibration requires SBC under NUTS.)")


if __name__ == "__main__":
    main()
