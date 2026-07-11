# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fork of [imanring/BSTPP](https://github.com/imanring/BSTPP.git) — Bayesian spatiotemporal point processes (LGCP, Hawkes, Cox-Hawkes, with optional spatial covariates) on numpyro/JAX — adapted for Philadelphia illegal-dumping analysis. Model math is in the README; API doc in `docs/bstpp_API_doc.pdf`. Work happens on branch `audit-fixes` with a one-atomic-commit-per-fix discipline: commit bodies explain the bug, the fix, and the verification (read `git log` for precedent before committing).

## Environment and commands

The system Python has none of the dependencies. Use the conda env `illegal-dumping`:

```bash
PY="C:/Users/Terhi/miniconda3/envs/illegal-dumping/python.exe"   # numpyro 0.15.0, jax/jaxlib 0.4.23 (CPU), geopandas 1.x, pytest
JAX_PLATFORM_NAME=cpu "$PY" -m pytest tests/ -q                  # full suite (~1 min)
JAX_PLATFORM_NAME=cpu "$PY" -m pytest tests/test_smoke.py::test_hawkes_traces -v   # single test
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg "$PY" scripts/recover_test.py                 # simulate-and-recover harness (SVI; --nuts for reference)
```

(The `bstpp` conda env also has the stack but lacks pytest.)

The dependency stack is deliberately pinned and fragile: `jax==0.4.23` forces `numpy<2`, which forces `scipy<1.13` and `rasterio<1.4` — the reasons are commented in `requirements.txt`. `geopandas>=1.0` is required (the `rng=` kwarg to `sample_points` in `_sim_cox`). Do not bump pins casually.

Gotchas: `run_svi(num_steps, lr, ...)` — `lr` is a required positional; pass `plot_loss=False` (or set `MPLBACKEND=Agg`) in non-interactive runs because it calls `plt.show()`. Cox/LGCP tests skip if the seasonal decoder artifact `bstpp/decoders/decoder_1d_T24_circ_small_l8` is missing.

## Architecture

- `bstpp/main.py` — `Point_Process_Model` base: the constructor rescales data to internal units, builds the temporal/seasonal/spatial grids, joins covariates, precomputes the excitation pairs and the `season_overlap` matrix, and assembles `self.args` (the single dict passed to the numpyro model). Subclasses `Hawkes_Model` / `LGCP_Model` pick the model function; also home to the simulators (`_sim_cox`, `_sim_hawkes_bg`, `_sim_offspring`, `simulate`) and metrics (`log_expected_likelihood`, `expected_AIC`).
- `bstpp/inference_functions.py` — the numpyro model functions: `spatiotemporal_hawkes_model` (branches on `args['model']` = `'hawkes'` vs `'cox_hawkes'`) and `spatiotemporal_LGCP_model`. Likelihood = event term − compensator, emitted as one `loglik_factor` factor site plus `loglik`/`Itot_*` deterministics.
- `bstpp/trigger.py` — pluggable excitation kernels (`Temporal_Exponential` samples `beta`; `Spatial_Symmetric_Gaussian` samples `sigmax_2`).
- `bstpp/utils.py` — `aligned_difference_pairs`: O(n log n + P) construction of excitation pairs; its docstring IS the contract (receiver `coords[:,0]`, source `coords[:,1]`, strict `0 < dt <= window`, ordering not guaranteed).
- `bstpp/vae_functions.py` + `bstpp/decoders/` — pretrained VAE decoders that stand in for GP priors on the background fields `f_t` (temporal, n_t=50), `f_a` (seasonal, n_s=24), `f_xy` (spatial, 25×25). The `.meta.txt` sidecar next to the seasonal decoder has honest UNKNOWN provenance fields — never fill them with invented values.
- `scripts/recover_test.py` — simulate-and-recover harness (plumbing check, not SBC). Pass/fail scores identified targets only: `log_background = log(Itot_txy − Itot_excite)` and the plug-in excitation share; the mean-log `a0+fbar` row is a diagnostic with no verdict.

## Invariants that span files (break one, break the model)

**Internal units.** Data are rescaled to the unit square and an internal time horizon `args['T'] = 50` (real horizon is `self.T` days); seasonal coordinate is internal `args['S'] = 24` over a real year `self.S = 365`. All posterior samples and `args` quantities are in internal units; `simulate()` converts times back to real days on return.

**Likelihood ↔ simulator coupling.** The background time integral is EXACT for the piecewise-constant fields: the likelihood contracts `exp(f_a)` against the precomputed `(n_t × n_s)` overlap matrix `args['season_overlap']`; `_sim_cox` integrates the same field on its breakpoint partition. `Ig == Itot_time` is a float-precision identity — any change to one side must change the other in the same commit (guarded by `tests/test_seasonal_integral.py`). Similarly, `_sim_hawkes_bg` and the hawkes covariate likelihood share per-covariate-cell semantics (`mu_xyt` per cell, contracted against `cov_area`; events index in via `cov_ind`).

**Sampled site names are the posterior interface.** `a_0`, `w`, `alpha`, `beta`, `sigmax_2`, `z_*` are consumed by plotting, `simulate()`, and downstream notebooks — never rename a `numpyro.sample` site. Deterministics named `Itot_*`/`rate_*`/`loglik` are similarly load-bearing for tests and the harness.

**Diagnostics vs likelihood.** `season_idx_of_t` (midpoint seasonal index), `rate_time`, `rate_t/Itot_t`, `rate_a/Itot_a` are marginal diagnostics only — the likelihood does not use them, `Itot_t·Itot_a·Itot_xy ≠ Itot_txy` by design, and `rate_xy` excludes the covariate term `b_0`.

**Silent-failure traps already fixed once — don't reintroduce:**
- `jax.ops.segment_sum` drops out-of-range indices silently: excitation pairs must always be built from the SAME event set the likelihood scores (`log_expected_likelihood` rebuilds pairs on held-out events and rejects test times outside `[0, T]`).
- numpyro `Predictive` drops missing site names silently: the `run_svi` sites list is model-aware (`Itot_excite` only for hawkes/cox_hawkes).
- Simulators must operate on `.copy()`s — never write columns into `self.A` / `self.spatial_cov`.
- The `comp_grid.sjoin(A)` used for spatial mass has border duplicates, so `Ih` is not `Itot_xy` in general — mirror the simulator's own computation when checking it.

## Testing conventions

Tests are regression pins: each test in `tests/` maps to a specific audit fix (see the docstring headers and matching commits). The standard pattern is to trace the model directly — `handlers.trace(handlers.seed(model.model, PRNGKey(0))).get_trace(model.args)`, with `handlers.substitute` for fixed-parameter comparisons — rather than running inference. When adding a fix + test, verify the test actually FAILS on the pre-fix code (temporarily revert, run, restore) before committing; state that in the commit body.
