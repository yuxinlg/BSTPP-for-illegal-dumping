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

## General testing conventions

Tests are mostly regression pins: they typically map to a specific audit fix (see the docstring headers and matching commits). The standard pattern is to trace the model directly — `handlers.trace(handlers.seed(model.model, PRNGKey(0))).get_trace(model.args)`, with `handlers.substitute` for fixed-parameter comparisons — rather than running inference. When adding a fix + test, verify the test actually FAILS on the pre-fix code (temporarily revert, run, restore) before committing; state that in the commit body.

## Refactoring principles for BSTPP
 
Distilled from *Better Code, Better Science* (bettercode-book.org). BSTPP
is scientific numerical code: reproducible-but-wrong is a real failure
mode, so **validation and testing rank above cleanliness.**
 
### Prime directive
- Refactoring changes structure, **never external behavior**. Every
  change must be behavior-preserving and provable by a passing test.
- **Do not refactor without a test harness in place first.** If tests are thin,
  first pin current behavior (capture intermediate outputs from the existing
  code and assert the refactor reproduces them within tolerance), *then* change
  structure.
- I am responsible for correctness — do not take your word that "all tests
  pass" or "this is equivalent." Show me the diff and the test output.
### Testing (non-negotiable, TDD)
- Use `pytest`, functions over classes, fixtures for expensive/shared objects
  (e.g. a sampled posterior, a simulated event set). Enforce RED → GREEN →
  Refactor: write the failing test first, commit it before implementation.
- FORBIDDEN: editing a test just to make it pass. Test changes must reflect a
  real requirement change or a genuine bug in the test — flag these explicitly.
- FORBIDDEN: simplifying the problem, mocking away the real implementation, or
  taking the "happy path" to get green. A test must fail for anything short of
  the full correct behavior.
- **Numerical tolerance:** never compare floats with `==`. Use
  `np.allclose` / `pytest.approx` with tolerances calibrated to the scale of the
  quantity (intensities, log-likelihoods, GP draws span many orders of
  magnitude — a fixed atol will silently pass or spuriously fail).
- Test the **interface, not internals** — treat model objects as black boxes;
  don't assert on private attributes that a refactor is allowed to rename.
- Tests must be **independent** and order-free. No shared mutable global state
  between tests (a session-scoped fixture that gets mutated breaks isolation —
  `.copy()` it).
- Bug-driven testing: any bug found while refactoring gets a regression test
  before it's fixed.
- Track **coverage** but don't worship it; prioritize the numerically critical
  paths (likelihood evaluation, trigger functions, VAE decode, kernel/precision
  construction). Use `-W error::FutureWarning` to catch deprecations in
  JAX/NumPyro/NumPy.
- Test edge cases and adversarial inputs.
### Scientific validation (this is where wrong-but-reproducible hides)
- Preserve and lean on **parameter recovery**: simulate from a model with known
  parameters, fit, confirm estimates track truth. A refactor that leaves tests
  green but degrades recovery is a real regression — keep a recovery check.
- Keep **positive/negative controls** working: no injected signal ⇒ result at
  chance/null; injected signal ⇒ detected. Watch for leakage if any feature
  selection or scaling touches the data.
- Watch for **optimizer/sampler pathologies** the refactor could mask:
  non-convergence, estimates pinned at bounds, divergences, label/identifiability
  trade-offs between parameters. Don't "fix" a failing fit by loosening a
  tolerance.
### Reproducibility & randomness (critical for Bayesian code)
- Prefer explicit generator objects (`np.random.default_rng(seed)`, or the
  passed JAX PRNGKey) over the global seed. Thread the key/rng through function
  arguments — do not reach for global RNG state inside functions.
- Results must be **robust to seed**, not cherry-picked from one lucky seed
  (avoid "seed-hacking"). If a refactor changes the RNG call order, expect
  numeric drift and re-baseline deliberately, documenting why.
### Managing complexity (the core goal)
- **Single Responsibility:** each function/class has one cohesive purpose at one
  level of abstraction. Break long functions into named steps.
- **Avoid the God object.** If a model class bundles config + data loading +
  fitting + plotting, split it: dataclass/Pydantic config, pure functions for
  transformations, a thin orchestrator for the workflow. Constructors should do
  cheap configuration only — no heavy file I/O or fitting in `__init__`.
- Default to **functions**; reach for classes only when they hide real
  complexity (persistent fitted state, the sklearn-style `.fit()/.predict()`
  contract, many independent instances). Keep the extensible **Trigger** module
  a clean, minimal interface so users can drop in their own trigger functions.
- **Kill duplication (DRY):** repeated near-identical blocks → a loop over a
  dict/config. No numbers baked into variable names.
- **No magic numbers:** name every constant (thresholds, default β/σ, grid
  sizes, MCMC tuning counts) and comment the justification/source.
- **No wildcard imports** (`from x import *`) — they hide provenance and clash.
- **Restrict scope:** no global variables to pass state in/out of functions
  (module-level `logger` is the one accepted exception). Pass in, return out.
- Prefer a **config object** (frozen Pydantic model / dataclass) over long
  parameter lists; it enforces types, validates ranges, and keeps calls
  consistent.
### Defensive coding
- **Fail loud, fail early.** Raise specific exceptions with useful messages;
  never return `None` on error or silently `except:` and continue. Bare/broad
  exception swallowing is forbidden — it's the most common AI-code smell.
- **Assert bounded quantities** where they're created or consumed: rates α,β > 0;
  σ > 0; probabilities ∈ [0,1]; counts ≥ 0 (integer); covariance/precision
  positive-definite (the `cholesky` trick). For anything user-facing, prefer
  Pydantic validation over bare `assert` (asserts vanish under `python -O`).
- **Code portably:** no hard-coded local paths, secrets, or machine-specific
  assumptions — pull those from config / env (`.env`, `python-dotenv`), and keep
  config files out of version control (`.gitignore`).
### Readability
- Names: consistent, understandable, specific, pronounceable, boring. Functions
  = verbs, variables/classes = nouns. Name by intent
  (`filter_timeseries_bandpass`) not implementation (`fft_timeseries`).
- **Type hints** on all function signatures — they document intent, help me and
  the IDE, and enable `mypy`.
- Comments explain **why**, not what. Don't comment obvious code; refactor
  unclear code instead of commenting over it. Use `# TODO:` for known gaps.
- Docstrings (Google style, consistent) on public functions/classes: purpose,
  params, returns, raises. Keep them describing the *interface*, not internals.
- Run `ruff` for formatting/linting.
### How to work with me (agentic workflow hygiene)
- **Reread this file and PLANNING.md / TASKS.md / SCRATCHPAD.md at the start of
  each session** and after any context clear/compact.
- Work in **small, focused steps** on a dedicated git branch. Keep commits
  granular so any step can be reverted.
- Use **commit → clear-context → reload** at each natural breakpoint. Show me
  `git diff` before committing so I can spot introduced smells.
- **Don't gold-plate / scope-creep.** Solve exactly the stated refactor — no new
  features, no premature abstraction, no speculative generality (YAGNI).
- If you loop, whack-a-mole, or can't fix something after a couple of tries:
  **stop and tell me** rather than simplifying the problem or claiming success.
  Going in circles is a signal the approach is wrong, not that it needs one more
  patch. I have the domain context to redirect.
- Flag, don't hide: outdated/hallucinated APIs, remnants of old code left behind
  after a rewrite, inappropriate pattern imitation (e.g. thread-safety code where
  nothing is threaded), and any place your change alters numeric output.
- Never run in `--dangerously-skip-permissions` mode against this repo.
