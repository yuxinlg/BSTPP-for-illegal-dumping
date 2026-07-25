# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fork of [imanring/BSTPP](https://github.com/imanring/BSTPP.git) — Bayesian spatiotemporal point processes (LGCP, Hawkes, Cox-Hawkes, with optional spatial covariates) on numpyro/JAX — adapted for Philadelphia illegal-dumping analysis. Model math is in the README; API doc in `docs/bstpp_API_doc.pdf`; the frozen Phase 3 contracts and acceptance matrix are in `docs/phase3_baseline_and_decisions.tex`. Work happens on branch `refactor` with small, atomic commits. Bug fixes follow the test-first sequence required below: commit the failing regression test before the implementation commit. Commit bodies explain the bug, the change, and the verification (read `git log` for precedent before committing).

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

- `bstpp/main.py` — `Point_Process_Model` base and workflow orchestrator: validates/prepares the supplied data through the Phase 3 seams, assembles `self.args` (the numpyro-facing compatibility view), and retains fitted state. Subclasses `Hawkes_Model` / `LGCP_Model` pick the model function; also home to the simulators (`_sim_cox`, `_sim_hawkes_bg`, `_sim_offspring`, `simulate`) and metrics (`log_expected_likelihood`, `expected_AIC`).
- `bstpp/data_contracts.py` — pre-construction validation/reporting for events, domain geometry, time horizon, covariates, CRS, and coverage. Reject mode must surface invalid inputs without silently dropping, repairing, or snapping rows.
- `bstpp/preparation.py` + `bstpp/spatial_grid_helpers.py` — the three Phase 3b data-bearing objects (`ModelData`, `PreparedDomain`, `PreparedPartitions`) and the pure preparation steps for grids, clipped support, covariate refinement, membership, areas, and `season_overlap`. `ModelDomain`, `ReportingRegions`, and `ComputationPartition` remain deferred contracts, not classes.
- `bstpp/decode_fields.py` + `bstpp/likelihood.py` — shared field decoding and likelihood atoms used by the numpyro models, simulation checks, and diagnostics.
- `bstpp/excitation_support.py` + `bstpp/polygon_mass.py` — explicit rectangle/polygon excitation-support policy, sigma bounds/prior truncation, and the polygon Gaussian Hermite mass-table backend; public `prepare_polygon_mass_table` (NumPy/SciPy float64) with hard-require install at polygon construction.
- `bstpp/cutoffs.py` — Phase 3e real-day temporal conversion, computational cutoff resolution, omitted-mass calculations, and cutoff provenance.
- `bstpp/inference_functions.py` — the numpyro model functions: `spatiotemporal_hawkes_model` (branches on `args['model']` = `'hawkes'` vs `'cox_hawkes'`) and `spatiotemporal_LGCP_model`. Likelihood = event term − compensator, emitted as one `loglik_factor` factor site plus `loglik`/`Itot_*` deterministics.
- `bstpp/trigger.py` — pluggable excitation kernels (`Temporal_Exponential` and `Temporal_Power_Law` sample `beta`; `Spatial_Symmetric_Gaussian` samples `sigmax_2`). A trigger is usable only on paths that can evaluate both its event term and its matching compensator.
- `bstpp/utils.py` — `aligned_difference_pairs`: O(n log n + P) construction of excitation pairs; its docstring IS the contract (receiver `coords[:,0]`, source `coords[:,1]`, strict `0 < dt <= window`, ordering not guaranteed).
- `bstpp/vae_functions.py` + `bstpp/decoders/` — pretrained VAE decoders that stand in for GP priors on the background fields `f_t` (temporal, n_t=50), `f_a` (seasonal, n_s=24), `f_xy` (spatial, 25×25). The `.meta.txt` sidecar next to the seasonal decoder has honest UNKNOWN provenance fields — never fill them with invented values.
- `scripts/recover_test.py` — simulate-and-recover harness (plumbing check, not SBC). Pass/fail scores identified targets only: `log_background = log(Itot_txy − Itot_excite)` and the plug-in excitation share; the mean-log `a0+fbar` row is a diagnostic with no verdict.

## Invariants that span files (break one, break the model)

**Internal units.** Data are rescaled to the unit square and an internal time horizon `args['T'] = 50` (real horizon is `self.T` days); seasonal coordinate is internal `args['S'] = 24` over a real year `self.S = 365`. All posterior samples and `args` quantities are in internal units; `simulate()` converts times back to real days on return.

**Likelihood ↔ simulator coupling.** The background time integral is EXACT for the piecewise-constant fields: the likelihood contracts `exp(f_a)` against the precomputed `(n_t × n_s)` overlap matrix `args['season_overlap']`; `_sim_cox` integrates the same field on its breakpoint partition. `Ig == Itot_time` is a float-precision identity — any change to one side must change the other in the same commit (guarded by `tests/test_seasonal_integral.py`). Similarly, `_sim_hawkes_bg` and the hawkes covariate likelihood share per-covariate-cell semantics (`mu_xyt` per cell, contracted against `cov_area`; events index in via `cov_ind`).

**Sampled site names are the posterior interface.** `a_0`, `w`, `alpha`, `beta`, `sigmax_2`, `z_*` are consumed by plotting, `simulate()`, and downstream notebooks — never rename a `numpyro.sample` site. Deterministics named `Itot_*`/`rate_*`/`loglik` are similarly load-bearing for tests and the harness.

**Diagnostics vs likelihood.** `season_idx_of_t` (midpoint seasonal index), `rate_time`, `rate_t/Itot_t`, `rate_a/Itot_a` are marginal diagnostics only — the likelihood does not use them, `Itot_t·Itot_a·Itot_xy ≠ Itot_txy` by design, and `rate_xy` excludes the covariate term `b_0`.

**Phase 3 audit status (2026-07-24 tip `8580364`).** The prepared-data, event-indexed-state, polygon-support, and cutoff contracts below are enforced at tip. Acceptance evidence lives in `refactor-patches/phase3{a,c,d,e}/rebaseline_record.md`, `refactor-patches/phase3_tip_verification_2026-07-24.md`, and `refactor-patches/confirmation_8580364.md`. Phase 3f and the Stage 3 R=200 exit rerun are not started. Add a regression test that fails for any newly identified defect before changing production code.

**Prepared-data contract.** `T_max` / `horizon_days` must be finite and positive. A GeoDataFrame domain must have polygonal, finite, positive-area support. Domain-row overlap uses the explicit union policy on `PreparedDomain.union_geometry` / `area_ratio` — never mix summed row areas with a different geometric support. If the domain declares a CRS, covariates must declare the same CRS: GeoDataFrame covariates are self-describing; CSV/plain-DataFrame covariates with a CRS-bearing domain require public `spatial_cov_crs` (`CRS.from_user_input`), assigned before `validate_covariates`, never inferred by copying the domain CRS. Invalid inputs are rejected before partition construction.

**Event-indexed state.** Every object indexed by an event or parent must be prepared from the SAME realization, in the SAME row order, that the likelihood scores. This includes excitation pairs, `cov_ind`, event coordinates/times, and polygon mass-table rows. `log_expected_likelihood(test_data)` treats held-out data as a separate realization, not forward forecasting conditional on training history: it must rebuild all event-indexed state from `test_data` without mutating or reusing training-event state.

**Polygon excitation support.** The current Hermite mass-table backend integrates `Spatial_Symmetric_Gaussian` only (exact-type gate). Polygon mode must reject any custom/non-Gaussian spatial trigger until that trigger supplies a matching polygon-mass backend. Tables are built only via public `prepare_polygon_mass_table(...)` (host NumPy/SciPy float64 throughout; never `jax.config.update` / process-global x64 mutation). Polygon construction hard-requires a compatible supplied table — never silent synchronous rebuild (`excitation_support="polygon"` without a table fails before base data/decoder init). Compatibility metadata is schema `hybrid_quad_hermite_numpy_v2`: backend id/schema, sigma parameterization, interpolation convention, slope method/`slope_fd_eps`, and `events_hash_algorithm` (`sha256_le_f64_xy_v1` over contiguous little-endian float64 coords+shapes). Descriptive `extra_provenance` is nested under `provenance['extra']` and never consulted for compatibility. Legacy v1 / decimal-`.9g` event hashes are intentionally incompatible. Equal row counts are not evidence of compatibility. `min_sigma` / `max_sigma` are spatial standard deviations in real units (`sqrt(sigmax_2)` for variance priors).

**Cutoff semantics and provenance.** The tolerance formulas in `cutoffs.py` are kernel-specific: temporal omitted mass and `mean_lag_days` apply to `Temporal_Exponential`; spatial omitted mass applies to `Spatial_Symmetric_Gaussian` with the per-axis square cutoff. `Temporal_Power_Law.beta` is a shape parameter, not a mean lag. Do not apply the tolerance/scale interface to unsupported triggers; explicit physical `window` / `spatial_window` remains the compatible path. Validate every supplied tolerance as finite and in `(0, 1)` even when a physical cutoff wins precedence. `set_window` updates cutoffs, pairs, support, and `cutoff_provenance` transactionally. Temporal-only polygon window changes reuse the installed mass table; spatial-window changes require a compatible `mass_table=` replacement validated against the prospective window before mutation.

**Silent-failure traps already fixed once — don't reintroduce:**
- `jax.ops.segment_sum` drops out-of-range indices silently: excitation pairs and every other event-indexed object must always be built from the SAME event set the likelihood scores (`log_expected_likelihood` rebuilds held-out state and rejects test times outside `[0, T]`).
- numpyro `Predictive` drops missing site names silently: the `run_svi` sites list is model-aware (`Itot_excite` only for hawkes/cox_hawkes).
- Simulators must operate on `.copy()`s — never write columns into `self.A` / `self.spatial_cov`.
- The `comp_grid.sjoin(A)` used for spatial mass has border duplicates, so `Ih` is not `Itot_xy` in general — mirror the simulator's own computation when checking it.

## General testing conventions

Tests are mostly regression pins: they typically map to a specific audit fix (see the docstring headers and matching commits). The standard pattern is to trace the model directly — `handlers.trace(handlers.seed(model.model, PRNGKey(0))).get_trace(model.args)`, with `handlers.substitute` for fixed-parameter comparisons — rather than running inference. When adding a fix + test, verify the test actually FAILS on the pre-fix code (temporarily revert, run, restore) before committing; state that in the commit body.

A green full suite alone is not a Phase 3 acceptance record. For each checkpoint or semantic correction, preserve the targeted RED/GREEN evidence, run the full suite, run all four configurations in the machine-local golden-pin harness `refactor-patches/pin_check_v2.py`, and run `ruff`; record the exact commands, outputs, and change classification in the checkpoint acceptance/rebaseline document. Phase 3d/3e acceptance records are backfilled post-hoc where contemporaneous records were missing — label that honesty explicitly and never claim a historical RED run that was not observed.

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
- Never mutate process-global JAX configuration (including
  `jax_enable_x64`) inside model construction or a reusable library call.
  Precision-sensitive polygon-table construction belongs in an explicit,
  controlled preparation step.
- Results must be **robust to seed**, not cherry-picked from one lucky seed
  (avoid "seed-hacking"). If a refactor changes the RNG call order, expect
  numeric drift and re-baseline deliberately, documenting why.
- Golden pins and fixed-seed simulations are machine-local artifacts — never
  compare across machines
- `GeoSeries.sample_points` is geopandas-version-sensitive, so the geometry
  stack (`geopandas`/`shapely`/`GEOS`) belongs alongside the `jax`/`numpy` pins in the environment notes if cross-session reproducibility of simulations ever matters
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
  a clean, minimal interface so users can drop in their own trigger functions,
  but never advertise a trigger on a support mode that cannot compute its
  matching compensator.
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

## Verification environment notes (added Phase 2c)

- Golden pins (`pin_check_v2.py`) and fixed-seed simulations are
  MACHINE-LOCAL artifacts: always baseline and compare on the same machine.
  (Empirically, JAX/XLA traced-path values have matched bit-for-bit across
  machines here, but that is an observation, not a guarantee.)
- `GeoSeries.sample_points` internals are geopandas-version-sensitive; the
  geometry stack (geopandas / shapely / GEOS) belongs alongside the jax /
  numpy pins whenever cross-session simulation reproducibility matters.
- Dev tooling: `ruff` (linter/formatter) is installed with
  `pip install --no-deps ruff`; it is not part of the runtime pins.
- Boundary / bounding-box / window semantics across model cases are
  inventoried in `docs/boundary_and_window_semantics.md` (Phase 3 inputs).

## Unit contract (added with the real-unit trigger change)

The SPATIAL trigger is a REAL-unit object: `sigmax_2` (squared real units of
the input X/Y columns; user MUST supply its prior -- there is no default) and
`spatial_window` (real length; real-space square). Internal/real conversion
happens at exactly three declared sites -- `real_spatial_trigger_values`
(event term, includes the sx*sy Jacobian), the compensator's limit stretch,
and the simulator's direct real-unit draw -- plus the shared window predicate
`within_real_box_window`. Everything else (background fields, a_0, beta,
window, cell areas) stays internal-unit. Never mix the two implicitly: any
new expression crossing the boundary must go through a declared conversion
(guide rule; this codebase's most productive historical source of defects).
Input coordinates must be METRIC (a projected CRS): the constructor warns on
degree-like domains because lon/lat makes the isotropic kernel anisotropic
on the ground.
