"""Stage-1 SBC harness: plain Hawkes (cox_background=False) on the unit box.

Design source: docs/sbc_runbook.md (the accepted design; read it first). This
module implements stage 1 of the staged plan -- four scalar parameters, the
fastest NUTS target, exercising pairs, the excitation compensator, and the
constant background in isolation. Stages 2 (LGCP) and 3 (cox-Hawkes) are
reserved subcommand values that fail loudly until implemented.

What SBC tests here: prior -> simulate -> NUTS fit -> posterior inverts, i.e.
the rank of each pre-registered truth functional among L thinned posterior
draws is uniform on {0..L} over replicates (Talts et al. 2018). The simulator
and likelihood share one set of atom expressions (Phase 2b), so a calibration
failure cannot be blamed on simulator/likelihood drift; this harness tests the
one layer no other test covers -- end-to-end Bayesian self-consistency, where
the double-factor bug lived.

PRE-REGISTERED (restated from the runbook; do not re-litigate mid-run):

* Ranked functionals: `alpha`, `beta`, `sigmax_2` directly, plus the
  identified background functional log_background = log(Itot_txy -
  Itot_excite), computed by tracing the SAME deterministic sites on the truth
  and posterior sides. Stage-1 note: with a constant background,
  log_background = a_0 + log(T_internal * A_area) EXACTLY, so its rank equals
  a_0's rank -- a_0 is fully identified in stage 1 (the integral-preserving
  field tilts that break its identification only exist in stages 2-3). It is
  ranked through the traced integrals anyway so the machinery is the one the
  later stages inherit unchanged. `exc_share` = Itot_excite/Itot_txy is
  recorded as a SUPPLEMENTARY diagnostic (reported, not gating); it is a
  valid rank target because SBC rank uniformity holds for any measurable
  f(theta, y), data-dependent functionals included.
* Priors: tight, matched by construction -- ONE dict of numpyro
  distributions is sampled for the truth draw and passed verbatim to the
  fitting model, which samples its sites from the same objects. NO post-hoc
  rejection of oversized/undersized simulations: the event count is bounded
  through the prior itself (`check` subcommand verifies the claim
  empirically before any overnight run). A degenerate replicate (< 2 events)
  aborts the run loudly -- stop-and-report, never skip, because silent
  skipping is rejection by another name and distorts the ranks.
* Divergent replicates are NOT dropped (same rejection logic). Per-replicate
  divergence counts are recorded (collectable since the extra_fields change)
  and summarized in the report; a material divergence rate is itself a
  finding to investigate before trusting the ranks.
* NUTS rng: every replicate receives an independent, reproducible MCMC key
  from the same SeedSequence tree as its prior and simulation streams. Reusing
  one fixed posterior-sampling stream across replicates would condition the
  rank ensemble on one set of Monte Carlo randomness and invalidate the iid
  discrete-uniform reference used by the report.
* MCMC dependence: NUTS returns an unthinned chain (`num_samples` raw draws).
  For every ranked functional the harness estimates the minimum ESS across 19
  empirical-quantile indicator functions, following Talts et al. Algorithm 2.
  The replicate aborts rather than being skipped if any primary ESS is below
  `min_ess_ratio * rank_draws`; otherwise the raw chain is uniformly thinned
  to exactly `rank_draws` states. Thus L=127 is earned, not assumed from a
  fixed thinning factor.
* Scope: rectangular unit-box domain, spatial_window=None, temporal
  window=None (-> full window T, the exact untruncated likelihood; simulator
  offspring beyond T are filtered, so likelihood and simulator agree exactly
  in this regime -- the conservation-structural regime of Phase 2b).
* Known small effects on record (runbook): out-of-domain-parenting
  second-order gap, kept negligible by the sigmax_2 prior concentrated on
  small kernels; the unseeded background draw concern is moot here (the
  plain-Hawkes background simulator drives every draw off the provided
  Generator, and SBC needs correct distributions, not per-replicate
  reproducibility anyway).
* Uniformity instrument: sup-distance between the empirical CDF of ranks and
  the discrete-uniform CDF, with a Monte Carlo p-value under exact discrete
  uniformity (M simulated rank sets; envelope by simulation, not by an
  asymptotic band -- auditable and exact up to MC error). Decision rule:
  each of the 4 primary functionals must have p >= 0.01 (family-wise false
  alarm ~4% at R in the designed range). Rank histograms are saved for
  reading shape (U-shape = overconfident, hump = underconfident, slope =
  bias), but the ECDF statistic is the test.

UNIT ANNOTATION (REQUIRED; runbook "Priors" section). This config knowingly
mixes two unit systems:

    sigmax_2       SQUARED REAL units of the input X/Y columns
                   (real-unit trigger contract)
    beta, window   INTERNAL rescaled-time units (data time -> [0, 50];
                   temporal conversion layer is Phase 3)
    a_0            internal-unit background intercept (background contract)

On THIS unit-box domain real and internal spatial units coincide, so the
numbers are unit-neutral -- but the annotation stays, because this config
transplanted to a real domain is NOT.

Resumability: per-replicate results are appended to replicates.jsonl as each
fit completes; config.json records the config hash, prior spec, unit
annotation, and provenance. A resumed run must hash-match the stored config
(raising otherwise -- mixing configs in one rank pool is invalid); raising
`replicates` on a matching config is allowed, since replicate r's seeds
depend only on (master_seed, r).

Usage:
  python scripts/sbc.py check  --draws 200          # independent prior-predictive budget check
  python scripts/sbc.py run    --replicates 150     # the real run (overnight, resumable)
  python scripts/sbc.py report                      # ranks -> ECDF test + histograms (+ --plot)
"""
import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import jax
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import Predictive

from bstpp.main import Hawkes_Model

T_DAYS = 2.5 * 365.0
A_RECT = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIOR_CHECK_MASTER_SEED = 20260717
LATENT = ["a_0", "alpha", "beta", "sigmax_2"]
PRIMARY = ["alpha", "beta", "sigmax_2", "log_background"]
DETERMINISTIC = ["Itot_txy", "Itot_excite"]

UNIT_ANNOTATION = {
    "sigmax_2": "SQUARED REAL units of the input X/Y columns (real-unit trigger contract)",
    "beta": "INTERNAL rescaled-time units (data time -> [0, 50]; Phase 3 conversion layer)",
    "a_0": "internal-unit background intercept (background contract unchanged)",
    "note": ("unit-box domain: real == internal spatially, numbers unit-neutral HERE; "
             "the same config on a real domain is NOT"),
}


def stage1_priors():
    """The pre-registered stage-1 priors -- used identically for truth draws and fits.

    Tightness rationale (verify with `check` before believing it):
      a_0 ~ N(0.8, 0.35): background count exp(a_0)*T_int*|A| = exp(a_0)*50, so
        ~55-225 background events across +-2sd; with the alpha prior's cascade
        multiplier the total stays in roughly 50-500.
      alpha ~ Beta(2, 6): mean 0.25, q99 ~ 0.63 -> cascade multiplier
        1/(1-alpha) <= ~2.7 with high prior probability; keeps simulations
        comfortably subcritical (no runaway cascades to bound post hoc).
      beta ~ LogNormal(0, 0.5): median 1 internal unit, q99 ~ 3.2 << window=50,
        and bounded away from 0 (kinder NUTS geometry than a HalfNormal mode
        at zero).
      sigmax_2 ~ LogNormal(log 0.005, 0.5): kernel sd ~ 0.07 real units on the
        unit box, q99 sd ~ 0.13 -- the small-kernel concentration that keeps
        the pre-registered out-of-domain-parenting gap negligible.
    """
    return dict(
        a_0=dist.Normal(0.8, 0.35),
        alpha=dist.Beta(2.0, 6.0),
        beta=dist.LogNormal(0.0, 0.5),
        sigmax_2=dist.LogNormal(float(np.log(0.005)), 0.5),
    )


def prior_spec(priors):
    """Serializable record of the prior family + parameters, derived from the
    distribution objects themselves (no dual maintenance)."""
    spec = {}
    for name, d in priors.items():
        spec[name] = {"dist": type(d).__name__}
        for arg in d.arg_constraints:
            spec[name][arg] = float(np.asarray(getattr(d, arg)))
    return spec


@dataclasses.dataclass(frozen=True)
class SBCConfig:
    replicates: int = 150
    master_seed: int = 0
    num_warmup: int = 300
    num_samples: int = 508     # raw post-warmup transitions; MCMC thinning is always 1
    rank_draws: int = 127      # L after ESS-qualified uniform thinning
    min_ess_ratio: float = 0.95
    out_dir: str = os.path.join("results", "sbc_stage1")

    def identity(self, priors):
        """The fields whose change invalidates pooling ranks across runs.
        `replicates` is deliberately excluded: extending R on a matching
        config is valid because replicate seeds depend only on
        (master_seed, r)."""
        return {
            "stage": 1,
            "model": "hawkes",
            "domain": [[0.0, 1.0], [0.0, 1.0]],
            "T_days": T_DAYS,
            "master_seed": self.master_seed,
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "rank_draws": self.rank_draws,
            "min_ess_ratio": self.min_ess_ratio,
            "priors": prior_spec(priors),
            "implementation": implementation_identity(),
        }

    def config_hash(self, priors):
        blob = json.dumps(self.identity(priors), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _git(*cmd):
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", *cmd], capture_output=True, text=True,
                             cwd=repo_root)
        return out.stdout if out.returncode == 0 else None
    except OSError:
        return None


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def implementation_identity():
    """Implementation/environment fields that must not change on resume."""
    import numpyro
    diff = _git("diff", "--binary", "HEAD", "--", "bstpp", "scripts/sbc.py")
    return {
        "commit": (_git("rev-parse", "HEAD") or "").strip() or None,
        "tracked_code_diff_sha256": (
            hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff is not None else None
        ),
        "sbc_script_sha256": _file_sha256(os.path.abspath(__file__)),
        "jax": jax.__version__,
        "numpyro": numpyro.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def provenance():
    return {
        **implementation_identity(),
        "branch": (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "caveat": "MACHINE-LOCAL results; never compare against runs from another machine",
    }


def make_placeholder(seed=0, n=50):
    """Constructor fodder for the generator model; the grids and unit maps it
    builds are data-independent, so one generator serves all replicates."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"X": rng.uniform(0.05, 0.95, n),
                         "Y": rng.uniform(0.05, 0.95, n),
                         "T": np.sort(rng.uniform(0, T_DAYS, n))})


def build_model(data, priors):
    return Hawkes_Model(data, A_RECT, T_DAYS, cox_background=False, **priors)


def replicate_seeds(master_seed, r):
    """Independent, reproducible per-replicate streams via SeedSequence
    spawning (not ad-hoc arithmetic on the master seed)."""
    ss = np.random.SeedSequence([int(master_seed), int(r)])
    c_prior, c_sim, c_legacy, c_mcmc, c_tie = ss.spawn(5)
    return {
        "prior_key": jax.random.PRNGKey(int(c_prior.generate_state(1)[0]) % (2 ** 31)),
        "sim_rng": np.random.default_rng(c_sim),
        "legacy_seed": int(c_legacy.generate_state(1)[0]) % (2 ** 31),
        "mcmc_key": jax.random.PRNGKey(int(c_mcmc.generate_state(1)[0]) % (2 ** 31)),
        "tie_rng": np.random.default_rng(c_tie),
    }


def draw_truth(key, priors):
    truth = {}
    for name in LATENT:
        key, sub = jax.random.split(key)
        truth[name] = float(priors[name].sample(sub))
    return truth


def simulate_replicate(gen, truth, seeds):
    # np.random is seeded too, purely as a belt-and-braces for any legacy
    # np.random draw outside the Generator-driven path; the plain-Hawkes
    # simulator drives every draw off the provided Generator.
    np.random.seed(seeds["legacy_seed"])
    sim = gen.simulate(dict(truth), rng=seeds["sim_rng"])
    events = pd.DataFrame(sim[["X", "Y", "T"]]).sort_values("T").reset_index(drop=True)
    return events


def truth_deterministics(fit, truth):
    """Trace the model's own deterministic sites AT the truth on the observed
    events -- the same expressions the posterior side replays, so truth and
    posterior functionals cannot drift apart (recover_test pattern)."""
    fixed = {k: truth[k] for k in LATENT}
    tr = handlers.trace(handlers.substitute(
        handlers.seed(fit.model, jax.random.PRNGKey(0)), fixed)).get_trace(fit.args)
    return {k: float(np.asarray(tr[k]["value"])) for k in DETERMINISTIC}


def posterior_deterministics(fit):
    pred = Predictive(fit.model, posterior_samples=fit.samples,
                      return_sites=DETERMINISTIC)
    det = pred(jax.random.PRNGKey(1), args=fit.args)
    return {k: np.asarray(v) for k, v in det.items()}


def rank_of(truth_val, draws, tie_rng):
    """Rank in {0..L}: draws strictly below truth, plus a uniform tie-break
    over exact float ties (Talts et al.; ties are measure-zero for continuous
    functionals but float equality is handled correctly, not ignored)."""
    draws = np.asarray(draws, dtype=float)
    below = int(np.sum(draws < truth_val))
    ties = int(np.sum(draws == truth_val))
    if ties:
        below += int(tie_rng.integers(0, ties + 1))
    return below, ties


def _ess_1d(values):
    """Single-chain autocorrelation ESS using a positive paired sequence.

    This intentionally caps antithetic estimates at the nominal draw count:
    SBC needs evidence that dependence is negligible, not credit for negative
    autocorrelation.
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    n = len(x)
    if n < 4:
        return float(n)
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if not np.isfinite(var) or var <= 0:
        return 1.0
    acov = np.correlate(x, x, mode="full")[n - 1:] / n
    rho = acov / acov[0]
    positive_pairs = []
    for lag in range(1, n - 1, 2):
        pair = float(rho[lag] + rho[lag + 1])
        if not np.isfinite(pair) or pair <= 0:
            break
        if positive_pairs:
            pair = min(pair, positive_pairs[-1])
        positive_pairs.append(pair)
    tau = max(1.0, 1.0 + 2.0 * sum(positive_pairs))
    return float(min(n, n / tau))


def min_quantile_ess(draws, n_quantiles=19):
    """Minimum ESS of empirical-CDF indicators across interior quantiles."""
    x = np.asarray(draws, dtype=float).reshape(-1)
    if len(x) < 4:
        return float(len(x))
    probs = np.linspace(0.05, 0.95, n_quantiles)
    cuts = np.quantile(x, probs)
    return float(min(_ess_1d(x <= cut) for cut in cuts))


def uniformly_thin(draws, L):
    """Uniformly thin a raw chain to exactly L states, truncating leftovers."""
    x = np.asarray(draws)
    if L < 1 or len(x) < L:
        raise ValueError(f"cannot thin {len(x)} raw draws to L={L}")
    stride = len(x) // L
    indices = np.arange(L) * stride
    return x[indices], indices, stride


def run_replicate(r, gen, priors, cfg):
    seeds = replicate_seeds(cfg.master_seed, r)
    truth = draw_truth(seeds["prior_key"], priors)
    events = simulate_replicate(gen, truth, seeds)
    n = len(events)
    if n < 2:
        raise SystemExit(
            f"FATAL: replicate {r} simulated {n} events. The prior was supposed to "
            "bound the count away from degeneracy (runbook: no post-hoc rejection; "
            "silently skipping would distort the ranks). Stop and fix the prior, "
            "then restart with a FRESH out_dir.")

    t0 = time.time()
    fit = build_model(events, priors)
    # The fit path prints a per-fit summary and progress bar; capture stdout so
    # an overnight log stays readable (progress bars go to stderr and are
    # suppressed via the fit path's own NUMPYRO_SPHINXBUILD switch in main()).
    with contextlib.redirect_stdout(io.StringIO()):
        fit.run_mcmc(num_warmup=cfg.num_warmup, num_samples=cfg.num_samples,
                     num_chains=1, thinning=1, rng_key=seeds["mcmc_key"])
    elapsed = time.time() - t0

    raw_L = int(np.asarray(fit.samples["alpha"]).shape[0])
    if raw_L != cfg.num_samples:
        raise SystemExit(
            f"FATAL: replicate {r} returned {raw_L} raw draws, expected "
            f"num_samples={cfg.num_samples}; rank-draw contract is ambiguous")
    for name in LATENT:
        vals = np.asarray(fit.samples[name])
        if not np.all(np.isfinite(vals)):
            raise SystemExit(f"FATAL: non-finite posterior for {name} in replicate {r}")

    det_true = truth_deterministics(fit, truth)
    det_post = posterior_deterministics(fit)
    bg_true = det_true["Itot_txy"] - det_true["Itot_excite"]
    bg_post = det_post["Itot_txy"] - det_post["Itot_excite"]
    if bg_true <= 0 or not np.all(np.asarray(bg_post) > 0):
        raise SystemExit(f"FATAL: non-positive background mass in replicate {r}")
    logbg_true = float(np.log(bg_true))
    logbg_post = np.log(bg_post)
    share_true = det_true["Itot_excite"] / det_true["Itot_txy"]
    share_post = np.asarray(det_post["Itot_excite"]) / np.asarray(det_post["Itot_txy"])

    diagnostic_draws = {
        "alpha": np.asarray(fit.samples["alpha"]),
        "beta": np.asarray(fit.samples["beta"]),
        "sigmax_2": np.asarray(fit.samples["sigmax_2"]),
        "log_background": np.asarray(logbg_post),
        "exc_share": np.asarray(share_post),
    }
    quantile_ess = {name: min_quantile_ess(vals)
                    for name, vals in diagnostic_draws.items()}
    ess_threshold = cfg.min_ess_ratio * cfg.rank_draws
    weak = {name: ess for name, ess in quantile_ess.items()
            if name in PRIMARY and ess < ess_threshold}
    if weak:
        details = ", ".join(f"{name}={ess:.1f}" for name, ess in weak.items())
        raise SystemExit(
            f"FATAL: replicate {r} has insufficient minimum quantile ESS for "
            f"L={cfg.rank_draws} (required >= {ess_threshold:.1f}: {details}). "
            "Do not skip it; increase num_samples and restart with a fresh out_dir.")

    thinned = {}
    thin_indices = None
    thin_stride = None
    for name, vals in {**diagnostic_draws, "a_0": np.asarray(fit.samples["a_0"])}.items():
        thinned[name], indices, stride = uniformly_thin(vals, cfg.rank_draws)
        if thin_indices is None:
            thin_indices, thin_stride = indices, stride
        else:
            assert np.array_equal(indices, thin_indices) and stride == thin_stride

    L = cfg.rank_draws
    ranks, ties = {}, {}
    for name in ["alpha", "beta", "sigmax_2"]:
        ranks[name], ties[name] = rank_of(truth[name], thinned[name], seeds["tie_rng"])
    ranks["a_0"], ties["a_0"] = rank_of(
        truth["a_0"], thinned["a_0"], seeds["tie_rng"])
    ranks["log_background"], ties["log_background"] = rank_of(
        logbg_true, thinned["log_background"], seeds["tie_rng"])
    ranks["exc_share"], ties["exc_share"] = rank_of(
        share_true, thinned["exc_share"], seeds["tie_rng"])
    if ranks["a_0"] != ranks["log_background"]:
        raise AssertionError(
            "stage-1 identity violated: a_0 and log_background ranks differ")

    diverging = int(np.asarray(fit.mcmc.get_extra_fields()["diverging"]).sum())
    return {
        "r": r, "n_events": n, "L": L, "raw_draws": raw_L,
        "thin_stride": int(thin_stride),
        "truth": truth,
        "truth_log_background": logbg_true,
        "truth_exc_share": float(share_true),
        "ranks": ranks, "ties": ties,
        "min_quantile_ess": {k: round(v, 3) for k, v in quantile_ess.items()},
        "diverging": diverging,
        "elapsed_s": round(elapsed, 2),
    }


def _paths(out_dir):
    return (os.path.join(out_dir, "config.json"),
            os.path.join(out_dir, "replicates.jsonl"))


def _load_records(jsonl_path):
    records = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def run_sbc(cfg, priors=None):
    """Resumable replicate loop. Returns (records, n_new_fits)."""
    priors = stage1_priors() if priors is None else priors
    cfg_path, jsonl_path = _paths(cfg.out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)

    this_hash = cfg.config_hash(priors)
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            stored = json.load(f)
        if stored["config_hash"] != this_hash:
            raise RuntimeError(
                f"config mismatch in {cfg.out_dir}: stored hash {stored['config_hash'][:12]} != "
                f"current {this_hash[:12]}. Ranks from different configs must not be pooled; "
                "use a fresh out_dir (raising `replicates` alone is allowed and does not "
                "change the hash).")
    else:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"config_hash": this_hash,
                       "identity": cfg.identity(priors),
                       "unit_annotation": UNIT_ANNOTATION,
                       "provenance": provenance()}, f, indent=2)

    records = _load_records(jsonl_path)
    record_ids = [rec["r"] for rec in records]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError(f"duplicate replicate ids in {jsonl_path}; refusing to pool ranks")
    done = {rec["r"] for rec in records}
    gen = build_model(make_placeholder(), priors)

    n_new = 0
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for r in range(cfg.replicates):
            if r in done:
                continue
            rec = run_replicate(r, gen, priors, cfg)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            records.append(rec)
            n_new += 1
            min_primary_ess = min(rec["min_quantile_ess"][name] for name in PRIMARY)
            print(f"[sbc] r={rec['r']:4d}  n={rec['n_events']:4d}  L={rec['L']}  "
                  f"minESS={min_primary_ess:.1f}  div={rec['diverging']}  "
                  f"{rec['elapsed_s']:.1f}s", flush=True)
    return records, n_new


def ecdf_sup_stat(ranks, L):
    """sup_k |ECDF(k) - (k+1)/(L+1)| over k in {0..L} for integer ranks."""
    ranks = np.asarray(ranks, dtype=int)
    counts = np.bincount(ranks, minlength=L + 1)
    ecdf = np.cumsum(counts) / len(ranks)
    uniform = (np.arange(L + 1) + 1) / (L + 1)
    return float(np.max(np.abs(ecdf - uniform)))


def mc_pvalue(stat_obs, R, L, mc_draws=10000, seed=2026):
    """Monte Carlo p-value of the sup-ECDF statistic under exact discrete
    uniformity on {0..L} (envelope by simulation: no asymptotic band to trust,
    exact up to MC error; +1/+1 correction keeps p valid)."""
    rng = np.random.default_rng(seed)
    sims = rng.integers(0, L + 1, size=(mc_draws, R))
    flat = sims + (L + 1) * np.arange(mc_draws)[:, None]
    counts = np.bincount(flat.ravel(), minlength=mc_draws * (L + 1))
    counts = counts.reshape(mc_draws, L + 1)
    ecdf = np.cumsum(counts, axis=1) / R
    uniform = (np.arange(L + 1) + 1) / (L + 1)
    stats = np.max(np.abs(ecdf - uniform[None, :]), axis=1)
    p = (1 + int(np.sum(stats >= stat_obs))) / (mc_draws + 1)
    return float(p)


def mc_sup_critical(R, L, probability=0.95, mc_draws=10000, seed=2026):
    """Simulated simultaneous ECDF-difference half-width under uniformity."""
    rng = np.random.default_rng(seed)
    sims = rng.integers(0, L + 1, size=(mc_draws, R))
    flat = sims + (L + 1) * np.arange(mc_draws)[:, None]
    counts = np.bincount(flat.ravel(), minlength=mc_draws * (L + 1))
    counts = counts.reshape(mc_draws, L + 1)
    ecdf = np.cumsum(counts, axis=1) / R
    uniform = (np.arange(L + 1) + 1) / (L + 1)
    stats = np.max(np.abs(ecdf - uniform[None, :]), axis=1)
    return float(np.quantile(stats, probability))


def report(out_dir, mc_draws=10000, mc_seed=2026, hist_bins=8, plot=False):
    """Read replicates.jsonl -> per-functional ECDF test + rank histograms.
    Writes report.json beside the replicates; returns the report dict."""
    _, jsonl_path = _paths(out_dir)
    records = _load_records(jsonl_path)
    if not records:
        raise RuntimeError(f"no replicates found in {jsonl_path}")
    record_ids = [rec["r"] for rec in records]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError(f"duplicate replicate ids in {jsonl_path}; refusing to pool ranks")
    Ls = {rec["L"] for rec in records}
    if len(Ls) != 1:
        raise RuntimeError(f"mixed L across replicates ({sorted(Ls)}): ranks cannot be pooled")
    L = Ls.pop()
    R = len(records)

    out = {"R": R, "L": L, "functionals": {}, "supplementary": {},
           "divergences": {
               "total_divergent_transitions": int(sum(rec["diverging"] for rec in records)),
               "replicates_with_divergences": int(sum(rec["diverging"] > 0 for rec in records)),
               "note": "divergent replicates are NOT dropped (pre-registered); "
                       "a material rate is a finding to investigate, not to filter",
           },
           "sampling_diagnostics": {
               "minimum_quantile_ess_by_functional": {
                   name: round(min(rec["min_quantile_ess"][name] for rec in records), 3)
                   for name in PRIMARY + ["exc_share"]
               },
               "note": "minimum over replicates; each primary replicate passed its "
                       "pre-registered ESS threshold before ranking",
           }}
    edges = np.linspace(0, L + 1, hist_bins + 1)
    for name in PRIMARY + ["exc_share"]:
        ranks = np.array([rec["ranks"][name] for rec in records], dtype=int)
        stat = ecdf_sup_stat(ranks, L)
        p = mc_pvalue(stat, R, L, mc_draws=mc_draws, seed=mc_seed)
        entry = {
            "ecdf_sup_stat": round(stat, 6),
            "mc_p_value": round(p, 6),
            "n_ties": int(sum(rec["ties"][name] for rec in records)),
            "rank_hist_bins": np.histogram(ranks, bins=edges)[0].tolist(),
        }
        (out["functionals"] if name in PRIMARY else out["supplementary"])[name] = entry

    out["decision"] = {
        "rule": "PASS iff every PRIMARY functional has mc_p_value >= 0.01 "
                "(pre-registered; family-wise false alarm ~4% over 4 functionals)",
        "primary_pass": all(out["functionals"][n]["mc_p_value"] >= 0.01 for n in PRIMARY),
        "interpretation": "PASS means no calibration deviation was detected at this "
                          "resolution; it is not proof that the implementation is correct",
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if plot:
        _plot_report(out_dir, records, L, hist_bins, mc_draws, mc_seed)
    return out


def _plot_report(out_dir, records, L, hist_bins, mc_draws, mc_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = PRIMARY + ["exc_share"]
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.0))
    uniform = (np.arange(L + 1) + 1) / (L + 1)
    R = len(records)
    simultaneous_band = mc_sup_critical(
        R, L, probability=0.95, mc_draws=mc_draws, seed=mc_seed)
    for j, name in enumerate(names):
        ranks = np.array([rec["ranks"][name] for rec in records], dtype=int)
        axes[0, j].hist(ranks, bins=np.linspace(0, L + 1, hist_bins + 1),
                        edgecolor="black")
        axes[0, j].axhline(R / hist_bins, ls="--", lw=1)
        axes[0, j].set_title(name + ("" if name in PRIMARY else " (suppl.)"))
        counts = np.bincount(ranks, minlength=L + 1)
        ecdf = np.cumsum(counts) / R
        axes[1, j].step(np.arange(L + 1), ecdf - uniform, where="post")
        axes[1, j].axhline(0.0, lw=1)
        # Simultaneous 95% band from the same discrete-uniform Monte Carlo
        # reference family as the formal sup-statistic test.
        axes[1, j].axhspan(-simultaneous_band, simultaneous_band, alpha=0.2)
        axes[1, j].set_xlabel("rank")
    axes[0, 0].set_ylabel("count")
    axes[1, 0].set_ylabel("ECDF - uniform")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "report.png"), dpi=150)
    plt.close(fig)


def prior_check(priors=None, draws=200, master_seed=PRIOR_CHECK_MASTER_SEED,
                lo=50, hi=500):
    """Prior-predictive event-count check (no fits): simulate `draws` prior
    replicates and report the count distribution. This is how the 'a_0 in a
    range giving ~50-500 events' claim is VERIFIED rather than trusted --
    run it before any overnight job. Its default master seed is deliberately
    distinct from `run`: inspecting and gating on the exact real-run simulations
    would select the SBC ensemble by its event counts."""
    priors = stage1_priors() if priors is None else priors
    gen = build_model(make_placeholder(), priors)
    counts = []
    for d in range(draws):
        seeds = replicate_seeds(master_seed, d)
        truth = draw_truth(seeds["prior_key"], priors)
        counts.append(len(simulate_replicate(gen, truth, seeds)))
    counts = np.array(counts)
    q = np.quantile(counts, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).astype(int)
    summary = {
        "draws": draws,
        "master_seed": master_seed,
        "quantiles_0_5_25_50_75_95_100": q.tolist(),
        "frac_below_lo": float(np.mean(counts < lo)),
        "frac_above_hi": float(np.mean(counts > hi)),
        "n_zero_event": int(np.sum(counts == 0)),
    }
    # The 50-500 range is a soft two-sided computational-budget band, not a
    # truncation of the generative model: simulations are never rejected, and
    # only zero-event draws (which would abort a real replicate) are forbidden.
    summary["budget_gate_pass"] = (
        summary["n_zero_event"] == 0
        and summary["frac_below_lo"] <= 0.03
        and summary["frac_above_hi"] <= 0.03
    )
    print(json.dumps(summary, indent=2))
    if summary["n_zero_event"]:
        print("WARNING: zero-event prior draws observed -- tighten the prior "
              "before running (a degenerate replicate aborts the run).")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="resumable SBC run (stage 1)")
    p_run.add_argument("--stage", type=int, default=1)
    p_run.add_argument("--replicates", type=int, default=150)
    p_run.add_argument("--master-seed", type=int, default=0)
    p_run.add_argument("--num-warmup", type=int, default=300)
    p_run.add_argument("--num-samples", type=int, default=508)
    p_run.add_argument("--rank-draws", type=int, default=127)
    p_run.add_argument("--min-ess-ratio", type=float, default=0.95)
    p_run.add_argument("--out-dir", default=os.path.join("results", "sbc_stage1"))

    p_chk = sub.add_parser("check", help="prior-predictive event-count check (no fits)")
    p_chk.add_argument("--draws", type=int, default=200)
    p_chk.add_argument("--master-seed", type=int, default=PRIOR_CHECK_MASTER_SEED)

    p_rep = sub.add_parser("report", help="ranks -> ECDF uniformity report")
    p_rep.add_argument("--out-dir", default=os.path.join("results", "sbc_stage1"))
    p_rep.add_argument("--plot", action="store_true")

    args = ap.parse_args()
    if args.cmd == "check":
        summary = prior_check(draws=args.draws, master_seed=args.master_seed)
        if not summary["budget_gate_pass"]:
            raise SystemExit("prior-predictive budget gate failed; do not start SBC")
    elif args.cmd == "run":
        if args.stage != 1:
            raise NotImplementedError(
                f"stage {args.stage} is not implemented yet; see docs/sbc_runbook.md "
                "'Staging' -- stage 2 (LGCP) and stage 3 (cox-Hawkes) follow a green stage 1.")
        # Suppress per-fit progress bars for the overnight log via the fit
        # path's own existing switch (inference_functions.run_mcmc checks this
        # env var); print_summary is captured per-replicate instead. No fit
        # path change beyond the optional per-fit RNG key involved.
        os.environ["NUMPYRO_SPHINXBUILD"] = "1"
        cfg = SBCConfig(replicates=args.replicates, master_seed=args.master_seed,
                        num_warmup=args.num_warmup, num_samples=args.num_samples,
                        rank_draws=args.rank_draws,
                        min_ess_ratio=args.min_ess_ratio, out_dir=args.out_dir)
        records, n_new = run_sbc(cfg)
        print(f"[sbc] complete: {len(records)} replicates on disk ({n_new} new). "
              f"Run `python scripts/sbc.py report --out-dir {args.out_dir}` next.")
    elif args.cmd == "report":
        out = report(args.out_dir, plot=args.plot)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
