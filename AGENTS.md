# [AGENTS.md](http://CLAUDE.md)

This file provides guidance to coding agents when working with code in this repository.

## What this is

Fork of [imanring/BSTPP](https://github.com/imanring/BSTPP.git) — Bayesian spatiotemporal point processes (LGCP, Hawkes, Cox-Hawkes, with optional spatial covariates) on numpyro/JAX — adapted for Philadelphia illegal-dumping analysis. Original model math is in the README; API doc in `docs/bstpp_API_doc.pdf`; the frozen Phase 3 contracts and acceptance matrix are in `phase3_record.tex`. Work happens on branch `refactor` with small, atomic commits. Bug fixes are RED-first under **D-41**: demonstrate the new row failing against the pre-change state, **commit the capture as evidence alongside the fix** — a separate test-only commit is not required and is not what recent series do. Commit bodies explain the bug, the change, and the verification, and **carry a change class** (BP behaviour-preserving / CF contract fix / SC semantic change / IV verification infrastructure / DOC), which the register's amendments key off (read `git log` for precedent before committing).

## Environment and commands

The system Python has none of the dependencies. Use the conda env `illegal-dumping`:

```bash
PY="C:/Users/Terhi/miniconda3/envs/illegal-dumping/python.exe"   # numpyro 0.15.0, jax/jaxlib 0.4.23 (CPU), geopandas 1.1.3, numpy 1.26.4, scipy 1.11.4, shapely 2.1.2, pytest
JAX_PLATFORM_NAME=cpu "$PY" -m pytest tests/ -q -m "not slow"    # FAST LANE, per-commit gate (~3m45s, 567 tests)
JAX_PLATFORM_NAME=cpu "$PY" -m pytest tests/ -q                  # FULL suite, series boundaries (~28 min, 579 tests)
JAX_PLATFORM_NAME=cpu "$PY" -m pytest tests/ -q -m slow          # the 12 slow tests alone (~11m30s)
JAX_PLATFORM_NAME=cpu "$PY" -m pytest tests/test_smoke.py::test_hawkes_traces -v   # single test
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg "$PY" scripts/recover_test.py                 # simulate-and-recover harness (SVI; --nuts for reference)
```

**Two-lane suite (measured 2026-08-05, `29058dc`).** The suite is ~28 min because
it *contains* slow tests, not because the machine is degraded: ten tests are 65%
of the runtime and the other 569 average 1.05 s. Twelve tests carry
`@pytest.mark.slow` — the two `test_polygon_i11_conservation` parametrisations
(R=40 Monte Carlo, 544 s, 32% of the suite on their own), the wheel/venv
packaging test (181 s), five SBC/NUTS smoke tests, and three others above ~40 s.

- **Per-commit gate: `-m "not slow"`** — 3m43s, 567 tests.
- **Series boundaries, and any commit touching the slow tests' subject matter
  (polygon mass/conservation, packaging metadata, SBC, the pair builder, SVI/NUTS
  entry): the FULL suite.** State which lane ran in the commit body; a fast-lane
  figure reported as if it were the full suite is a false gate record.

`pytest tests/` with no `-m` still runs **everything**. The default is
deliberately the complete suite: a default that silently skipped tests would be
the same silent-failure class this codebase keeps fixing. The fast lane is
opt-in, and its deselection count (`12 deselected`) is visible in its own output.

**Running the two lanes separately is cheaper than one full run, by a lot.**
Measured: fast 223.76 s + slow 686.15 s = **909.91 s**, against **1694.40 s** for
the same 579 tests in one process — the single process costs **86% more**, a
13-minute excess and 4× the observed run-to-run spread. So a boundary check is
better done as two invocations than one. The excess is superlinear in a single
process (accumulated JAX compilation caches, fixtures and arrays pushing a
15.4 GB box into paging), which is the grain of truth in the memory-pressure
story — it is real for the heavy tail, it is just not why the suite is 28 minutes.

(The `bstpp` conda env also has the stack but lacks pytest.)

The dependency stack is deliberately pinned and fragile: `jax==0.4.23` forces `numpy<2`, which forces `scipy<1.13` and `rasterio<1.4` — the reasons are commented in `requirements.txt`. `geopandas>=1.0` is required (the `rng=` kwarg to `sample_points` in `_sim_cox`). Do not bump pins casually.

Gotchas: `run_svi(num_steps, lr, ...)` — `lr` is a required positional; pass `plot_loss=False` (or set `MPLBACKEND=Agg`) in non-interactive runs because it calls `plt.show()`. Cox/LGCP tests skip if the seasonal decoder artifact `bstpp/decoders/decoder_1d_T24_circ_small_l8` is missing.

## Architecture

- `bstpp/main.py` — `Point_Process_Model` base and workflow orchestrator: validates/prepares the supplied data through the Phase 3 seams, assembles `self.args` (the numpyro-facing compatibility view), and retains fitted state. Subclasses `Hawkes_Model` / `LGCP_Model` pick the model function; also home to the simulators (`_sim_cox`, `_sim_hawkes_bg`, `_sim_offspring`, `simulate`) and metrics (`log_expected_likelihood`, `expected_AIC`).
- `bstpp/data_contracts.py` — pre-construction validation/reporting for events, domain geometry, time horizon, covariates, CRS, and coverage. Reject mode must surface invalid inputs without silently dropping, repairing, or snapping rows.
- `bstpp/preparation.py` + `bstpp/spatial_grid_helpers.py` — the three Phase 3b data-bearing objects (`ModelData`, `PreparedDomain`, `PreparedPartitions`) and the pure preparation steps for grids, clipped support, covariate refinement, membership, areas, and `season_overlap`. `ModelDomain`, `ReportingRegions`, and `ComputationPartition` remain deferred contracts, not classes.
- `bstpp/decode_fields.py` + `bstpp/likelihood.py` — shared field decoding and likelihood atoms used by the numpyro models, simulation checks, and diagnostics.
- `bstpp/excitation_support.py` + `bstpp/polygon_mass.py` — explicit rectangle/polygon excitation-support policy, sigma bounds/prior truncation, and the polygon Gaussian Hermite mass-table backend; public `prepare_polygon_mass_table` (NumPy/SciPy float64) with hard-require install at polygon construction.
- `bstpp/config.py` — Phase 3f frozen configuration objects behind the `args` adapter. Holds `NumericalConfig` (mass-table builder settings, measured-budget policy, package cutoff tolerance defaults, resolved σ bounds) and `NumericalConfigError`. Also the **single source for canonical error clauses** under D-40: one `*_invariant_clause` / `raise_*_violation` pair per invariant, rendered byte-for-byte wherever the violation is detected. `polygon_mass` reaches them by deferred import (config imports polygon_mass for its constants). Frozen dataclasses, validation in `__post_init__`, one factory (`create`) per object, no Pydantic.
- `bstpp/cutoffs.py` — Phase 3e real-day temporal conversion, computational cutoff resolution, omitted-mass calculations, and cutoff provenance.
- `bstpp/inference_functions.py` — the numpyro model functions: `spatiotemporal_hawkes_model` (branches on `args['model']` = `'hawkes'` vs `'cox_hawkes'`) and `spatiotemporal_LGCP_model`. Likelihood = event term − compensator, emitted as one `loglik_factor` factor site plus `loglik`/`Itot_*` deterministics.
- `bstpp/trigger.py` — pluggable excitation kernels (`Temporal_Exponential` and `Temporal_Power_Law` sample `beta`; `Spatial_Symmetric_Gaussian` samples `sigmax_2`). A trigger is usable only on paths that can evaluate both its event term and its matching compensator.
- `bstpp/utils.py` — `aligned_difference_pairs`: O(n log n + P) construction of excitation pairs; its docstring IS the contract (receiver `coords[:,0]`, source `coords[:,1]`, strict `0 < dt <= window`, ordering not guaranteed).
- `bstpp/vae_functions.py` + `bstpp/decoders/` — pretrained VAE decoders that stand in for GP priors on the background fields `f_t` (temporal, n_t=50), `f_a` (seasonal, n_s=24), `f_xy` (spatial, 25×25). The `.meta.txt` sidecar next to the seasonal decoder has honest UNKNOWN provenance fields — never fill them with invented values.
- `scripts/recover_test.py` — simulate-and-recover harness (plumbing check, not SBC). Pass/fail scores identified targets only: `log_background = log(Itot_txy − Itot_excite)` and the plug-in excitation share; the mean-log `a0+fbar` row is a diagnostic with no verdict.



## Invariants that span files (break one, break the model)

**Internal units.** Data are rescaled to the unit square and an internal time horizon `args['T'] = 50` (real horizon is `self.T` days); seasonal coordinate is internal `args['S'] = 24` over a real year `self.S = 365`.

> ⚠ **Those four numbers are written twice and nothing reconciles them (A-41/A-42; OP-27 → WP10).** `preparation.py` names them — `T_INTERNAL=50`, `S_INTERNAL=24`, `S_DAYS=365`, and the decoder-pinned `N_T`/`N_S`/`N_XY` — and `main.py:338/339/365` write the same quantities as **bare literals**, in a module that already imports `T_INTERNAL` and uses it at three other sites. **The coupling is directional: change the constant and `cutoffs.py`'s real↔internal conversions follow it while `args['T']` does not**, so the invariant above breaks with nothing raised. In unit terms this is the real/internal boundary crossed without the conversion being named — `365` real days against `24` internal, with neither name at the site. **If you are doing unit work, fix the site or read `refactor-patches/phase3f/wp10_args_removal_entry.md` first.** That these literals are the same quantity as those constants took adjudication to establish and is *not recoverable from source*. All posterior samples and `args` quantities are in internal units; `simulate()` converts times back to real days on return.

**Likelihood ↔ simulator coupling.** The background time integral is EXACT for the piecewise-constant fields: the likelihood contracts `exp(f_a)` against the precomputed `(n_t × n_s)` overlap matrix `args['season_overlap']`; `_sim_cox` integrates the same field on its breakpoint partition. `Ig == Itot_time` is a float-precision identity — any change to one side must change the other in the same commit (guarded by `tests/test_seasonal_integral.py`). Similarly, `_sim_hawkes_bg` and the hawkes covariate likelihood share per-covariate-cell semantics (`mu_xyt` per cell, contracted against `cov_area`; events index in via `cov_ind`).

**Sampled site names are the posterior interface.** `a_0`, `w`, `alpha`, `beta`, `sigmax_2`, `z_`* are consumed by plotting, `simulate()`, and downstream notebooks — never rename a `numpyro.sample` site. Deterministics named `Itot_`*/`rate_*`/`loglik` are similarly load-bearing for tests and the harness.

**Diagnostics vs likelihood.** `season_idx_of_t` (midpoint seasonal index), `rate_time`, `rate_t/Itot_t`, `rate_a/Itot_a` are marginal diagnostics only — the likelihood does not use them, `Itot_t·Itot_a·Itot_xy ≠ Itot_txy` by design, and `rate_xy` excludes the covariate term `b_0`.

**Phase 3 audit status (pre-3f audited tip** `e0d7e437`**; living record `phase3_record.tex`).** The prepared-data, event-indexed-state, polygon-support, and cutoff contracts below are enforced at tip. A-21 records the freeze tip `938e32b`, the first follow-up series (`df06e55`--`04799d9`), the continuation on `04799d9` (one-sided TruncatedNormal intersection, rejection of truncated non-Normal bases, rectangle computational-grid simulate dedup, production slope-doc correction), and the local correctness/API series on `e0d7e437` (`af7c934`--`c501283`: kernel capability gates, untouched-axis provenance, LGCP `set_window` rejection, panel/`min_sigma` guard, grid/kernel/MCMC helpers, runtime packaging metadata, stale-doc retargets; deferred membership consolidation and extreme float32 TLN notes). Acceptance evidence lives in `refactor-patches/phase3{a,c,d,e}/rebaseline_record.md`, `refactor-patches/phase3_tip_verification_2026-07-24.md`, `refactor-patches/confirmation_8580364.md`, and `refactor-patches/pre3f_audit_e0d7e43.md`. The living Phase 3 contract/decision record is the repository-root `phase3_record.tex` (Part I frozen + Part II append-only amendments **through A-31 / D-41**). Add a regression test that fails for any newly identified defect before changing production code.

**Phase 3f status (current).** WP1 is landed: `NumericalConfig` exists behind the `args` adapter (A-22/A-23/D-35); the error-identity work is complete through A-31 — D-40 (one invariant, one identity, one canonical clause, owners assigned by which *quantity* the invariant attaches to), D-41 (the evidence-and-provenance standard) — and the argument-type series through A-37: D-42 (CI-7/CI-8), D-43 (the construction DAG; bind-time resolution with a sentinel). **WP1's construction sites are REOPENED under D-43 clause 2 while its object invariants stay closed.** **WP2 (`ModelConfig`, `PriorConfig`) is enumerated but not started**; a capped set of opening conditions is *proposed* at `refactor-patches/phase3f/wp2/wp2_opening_conditions_proposal.md` and is **not** yet a decision (`D-44` is free). Open items: OP-18 (I10 restatement); OP-19 + OP-21 + OP-26 → WP5; OP-23 reserved with NO TEXT — cited only here, and either its text lands at WP5's opening or this citation is withdrawn (A-36); OP-24 (a fifth polygon-mode pin) was the **WP5 entry precondition** and is **CLOSED at A-47**, forward only; OP-25 (the `warnings.warn` ASCII sibling) → WP2; **OP-27 is the two-owner class** (a quantity held in >1 place with no code reconciling them) — OP-19 and OP-26 are two of its five measured members, so **new instances become census rows, not new OP numbers** (A-40). Its denominator of 11 is the **config-anchored** population only; the config-external population is **known non-empty** (`T_INTERNAL`/`S_INTERNAL`/`S_DAYS` vs the bare literals at `main.py:338/339/365`) and **unmeasured** (A-41). **OP-28 is escalated and open**: `standardize_cov='domain_area'` is a *different estimator* from `main`'s `True`, not a weighted variant — which one the package intends is **not established** (A-41). OP-20 and OP-22 are CLOSED (OP-20's §11 row said so only from A-43 — it was closed in prose at A-33 and never marked in the table). **A-49 (DOC/process, `51c98f8`) added D-47–D-51 and `docs/wp_dependency_graph.md`: D-47 work-package order is a graph, not an integer (the graph governs, disagreements are recorded not renumbered away); **D-48 a gate must declare its coverage before it is read as evidence** — pin reachability `ρ_w = |S_w ∩ cov(P)| / |S_w|` is *measured* and recorded before a work package opens, `ρ_w = 0` makes the pin gate **vacuous for that package**; D-49 WP5's polygon pin is a parallel-track entry precondition (discharged by A-47); D-50 corrective depth capped at three per package; D-51 WP10 intake freezes at a named tip. **⚠ A-49 introduced 21 fill-anchors** across three decisions (D-48, D-50, D-51) and two files — 5 in the register, 16 in `docs/wp_dependency_graph.md` (A-51 measured this; the earlier figure "four" was wrong *and* never said what it counted). **D-48's coverage unit and D-51's freeze tip are now defined (A-51); D-50's remaining anchor defers a ledger entry, not a rule.** Census instrument: `results/_a51_anchor_census.py`. **⚠ `ρ_WP2` was never recorded — WP2 opened at A-45, before D-48 existed — so it is an outstanding declared debt against WP2, not a satisfied condition** (A-50). **A-52 amends D-48 in three clauses** (empty seam set: rho_w = 0/0 is undefined, not zero, and the rho_w = 0 vacuity clause answers a different question; rho_w comparable only within a fixed tree state, since statement counts move under extraction; **a declared seam set is itself an opening precondition**, in `docs/seam_sets.md`) and **establishes D-52: a decision binds PROSPECTIVELY unless it enumerates what it invalidates.** **S_WP2 is declared retroactively (the only one) and is EMPTY of extant code** -- `ModelConfig`/`PriorConfig` do not exist, so rho_WP2 is `N/A (empty seam set)` and WP2's D-48 debt is discharged by the declaration, not by a measurement; WP1 is **grandfathered with its debt declared**. The other nine declare at opening. **Content check 5** (anchors in landed decision rows + D-47-authoritative documents, against a declared baseline of 13) and **two apparatus checks** (`results/_a52_apparatus_checks.py`: every gate hash-pinned by an instrument that consumes none of them; every document instrument opening with explicit `encoding=`) are live and mutation-tested. **Use/mention convention**: an anchor on a line marked `census:mention` is reported, never counted, so prose can quote the token without inflating the count. **OP-29 now rests on TWO instances, not three** -- its D-43 instance is reclassified as D-44-class at A-52 (measured: D-43 appears in no tracked file before `65e2708`, the commit that landed it), putting the class below the OP-22 enumeration threshold; **whether a two-instance class stays enumerated is unsettled.** **The D-46 census series 550/564/589/601 is NOT homogeneous** -- A-48's reading was pre-staging, later ones post-staging; the measurement point is now part of D-46's definition (post-staging, pre-commit, `git ls-files` cross-check required). Next free: **A-53, D-53, CI-11, OP-32**. **WP2 IS OPEN**; its first commit is A-45 (CI-9 enforced at construction, SC/API) and its second is A-50 (CI-10, SC/API). **⚠ `Hawkes_Model`'s `cox_background` default changed TYPE at A-50, from the string `'cox'` to `True`** — the same model (both truthy, no pinned value moves), but `cox_background='cox'` is now rejected, so any snippet copied out of a pre-A-50 signature breaks loudly rather than quietly. Archived call sites under `refactor-patches/phase0/`, `phase1/`, `phase2b/` and the A-36/A-41 probes still pass the retired string and are **knowingly stale historical artifacts, not live instruments** — they are not rewritten, because a rewritten archive stops being evidence of what was measured. **The seven-round apparatus cap is reached and the finding is registered (A-44; `refactor-patches/phase3f/phase3f_finding_at_cap.md`): two of four WP2 opening conditions unmet, recommendation adopted, WP2 OPEN with two declared gaps — G-A three ungated document gates, G-B `standardize_cov`'s bind-time relocation refused until OP-28 is answered.** **WP5 CANNOT OPEN**: six items route to it and OP-24's polygon pin is unbuilt. **D-45: a gate that goes red has its capture preserved before the fix.** **D-46 (A-46): preserved captures live under one declared path, `refactor-patches/captures/`, and every document-census and citation-sweep instrument excludes it BY CONSTRUCTION** — one pathspec in `results/_a46_capture_population.py`, never a per-instrument filename filter; the *citation resolution set* deliberately still includes it so captures stay citable. **Consequence, stated exactly: the C2 GATED/UNGATED count is stationary by construction — this does NOT validate 5/2 retroactively, the count stays provisional, the declared gap stands, and the cap forbids a fifth reading.** The population *size* is not stationary (green captures keep landing in `results/`), and **OP-30** records that D-46's routing is a convention, not a mechanically enforced invariant. **OP-31 is CLOSED at A-48**: every `pin_compare.py` verdict line now carries `compared=n/m`, `candidate_only=[...]` and `baseline_only=[...]`, and `MATCH` is reserved for a comparison covering the whole union — the canonical run of the six-configuration candidate reads `PIN_DIFFS 0 PARTIAL compared=4/6` and names both polygon keys, so a gate line copied from a pre-A-47 runbook stops reporting success instead of quietly narrowing. A `PARTIAL` still exits 0, deliberately. **OP-30 remains a declared limit routed to no work package.** **OP-29: prose-and-table divergence is now check 4 of the content-checks gate.**

**Work packages: the outline and the entries (A-43).** The Phase 3f outline — ten packages, seam sets, class per package — is `refactor-patches/phase3f/phase3f_work_package_outline.md`; **it originated outside this repository** and says so. Entries: `wp3_config_objects_entry.md`, `wp4_prepared_data_entry.md`, **`wp5_excitation_support_entry.md` (substantive)**, `wp6_mutators_membership_entry.md`, `wp7_decoder_contract_entry.md`, `wp8_input_metadata_entry.md`, `wp9_results_io_entry.md`, `wp10_args_removal_entry.md`. WP1 and WP2 live in the register. **D-44: no open item routes to a work package that has no register entry** — items already routed stay; new ones wait for the entry. **⚠ `Wn` (the completed pre-3f W0–W10 handoff in `phase3e_closeout_and_3f_readiness.tex`) and `WPn` (Phase 3f) are different sequences and collide — `W5`/`W10` mean other things.** Same hazard as `I1`–`I12` vs `CI-1`–`CI-9`; read the prefix.

**WP5's entry precondition is DISCHARGED (A-47), and WP5 is still not open.** OP-24's polygon-mode pin was **scheduled** (not deferred) at A-36 and is **built**: `pin_check_v2.py` now emits **six** configurations, adding `hawkes_notched_4to1_polygon_mode` and `hawkes_notched_4to1_rectangle_mode` over one 4:1 **non-rectangular** domain with the same events and σ bounds, so **both excitation support modes are pinned and the mode switch is the only difference between the two records**. All four previously-absent tokens (`polygon`/`mass_table`/`excitation_support`/`min_sigma`) are reachable, and the mass table's SHA-256 is *inside* the pinned record so a differently-built table is a DRIFT (D-27/A-11). **Two limits travel with it. (1) It is FORWARD**: the baseline is `refactor-patches/baselines-2026-08-polygon/pins.json`, dated at its capture, certifying nothing before it — six configurations are not six configurations of retroactive coverage, and it cannot certify `b98e91d` or `0e78f7d`. **(2) The canonical 2026-07 baseline does NOT gate it**: the two new keys report `NEW IN CANDIDATE` there and `pin_compare`'s walker counts a new key as no diff, so **any polygon-regime gate line must pass `--baseline refactor-patches/baselines-2026-08-polygon/pins.json` explicitly** or it reports `PIN_DIFFS 0 MATCH` while staying as silent as before. **WP5 opening is a separate decision A-47 declines to take**: OP-23's text still does not exist, and A-44's two declared gaps stand. **A-50 states the remainder exhaustively so the next round can be a decision rather than a survey — three items, each a decision to take and not a measurement to make: (1)** OP-23's text lands or the citation is withdrawn (A-36 fixed those terms); **(2)** gap G-A, three ungated document gates; **(3)** gap G-B, `standardize_cov`'s bind-time relocation, held until OP-28 is answered.

**No figure computed under `standardize_cov=True` is comparable to one computed under `'domain_area'` (OP-28/A-41)** — not log-likelihoods, AIC, `w`, `b_0`, or the excitation share. Re-derive, do not re-label; this applies to figures already produced. All six golden pins carry no covariates, so no pin is sensitive to it. **And `'domain_area'` makes the design matrix a function of the domain**, so the same column is standardized differently per fitted site and `w` is not comparable across sites — the covariate-level analogue of why the spatial trigger must be isotropic in *real* units. **OP-28 therefore blocks district-wide and city-wide modes in a way it does not block park-level work** (A-42): a single-park fit is internally consistent either way; a multi-site fit inherits the incomparability silently.

**WP2 opening: two of four conditions unmet at the end of the seven-round cap (A-42).** C1 and C3 closed; **C2** not (GATED 4 / UNGATED 3 / RED 0 over the closed set of 7) and **C4** not (4 rows, 2 classified, 2 undetermined). Recommendation on the table is **open WP2 with two declared gaps** — G-A the three ungated document gates, G-B `standardize_cov`'s bind-time relocation held out of scope until OP-28 is answered — with CI-9's enforcement (SC/API) as the first commit. See `refactor-patches/phase3f/wp2/wp2_opening_conditions_proposal.md`. **Not yet decided.**

**Scope limit on every behaviour-preservation claim (A-40).** Pins, trace equivalence, the full suite, every `MATCH` and every green lane cover **only the modules in the repository under test**. `bstpp.cox_hawkes_shared.Cox_Hawkes_Shared` and any sibling modules vendored untracked into a downstream working checkout are outside all of it, and putting them under version control elsewhere does not bring them inside it. This is a statement about coverage, not a defect.

**Two `I`-numberings exist and they collide — read this before citing one.** `I1`–`I12` are the Phase-3 **model identities** (mass atoms, pair window, derived seasonal coordinate, conservation, …) and are cited in the guide, the phase0 archive and `test_polygon_i11_conservation.py`. The **config invariants** under D-40 were briefly numbered `I1`–`I6` and were renumbered `CI-1`…`CI-6` at A-30: `CI-1` rectangle both-or-neither, `CI-2` polygon requires `min_sigma`, `CI-3` `min_sigma` finite and positive, `CI-4` `min_sigma < max_sigma`, `CI-5` support-mode validity, `CI-6` builder requires `max_sigma`, `CI-7` config real-argument type (A-33), `CI-8` config integral-argument type (A-33), `CI-9` enumerated-value validation independent of whether the consuming leg runs (A-39; **enforced at A-45** — `standardize_cov` is validated in `Point_Process_Model.__init__` unconditionally, canonical clause in `config.py`, second site in `preparation.py` by deferred import), `CI-10` a config-owned **boolean** argument is a `bool`, never merely truthy (A-50; `cox_background` was consumed as `if cox_background:` with nothing checking it, so `'false'` selected the *cox* background — a value accepted and ACTED ON in the direction opposite to what it said, which is CI-9's defect with the sign flipped. **`np.bool_` is accepted** because `bool` cannot be subclassed in CPython, so it has no way to opt into D-42's `np.float64` treatment; `0`/`1` are rejected, which is D-42's bool-is-not-an-int boundary from the other side). **`CI-n` is one sequence across all five config objects, extended as each lands its own — never restarted per object.** Next free is `CI-11`.

**Prepared-data contract.** `T_max` / `horizon_days` must be finite and positive. A GeoDataFrame domain must have polygonal, finite, positive-area support. Domain-row overlap uses the explicit union policy on `PreparedDomain.union_geometry` / `area_ratio` — never mix summed row areas with a different geometric support. If the domain declares a CRS, covariates must declare the same CRS: GeoDataFrame covariates are self-describing; CSV/plain-DataFrame covariates with a CRS-bearing domain require public `spatial_cov_crs` (`CRS.from_user_input`), assigned before `validate_covariates`, never inferred by copying the domain CRS. Invalid inputs are rejected before partition construction.

**Event-indexed state.** Every object indexed by an event or parent must be prepared from the SAME realization, in the SAME row order, that the likelihood scores. This includes excitation pairs, `cov_ind`, event coordinates/times, and polygon mass-table rows. `log_expected_likelihood(test_data, mass_table=...)` treats held-out data as a separate realization, not forward forecasting conditional on training history: it must rebuild all event-indexed state from `test_data` without mutating or reusing training-event state. Polygon mode hard-requires an explicit held-out `mass_table=` prepared for those events — never silent rebuild.

**Polygon excitation support.** The current Hermite mass-table backend integrates `Spatial_Symmetric_Gaussian` only (exact-type gate). Polygon mode must reject any custom/non-Gaussian spatial trigger until that trigger supplies a matching polygon-mass backend. Tables are built only via public `prepare_polygon_mass_table(...)` (host NumPy/SciPy float64 throughout; never `jax.config.update` / process-global x64 mutation). Polygon construction hard-requires a compatible supplied table — never silent synchronous rebuild. A supplied `PolygonMassTable` is valid only for the exact domain union, event coordinates and row order, spatial window, sigma range/grid, and build settings recorded in its metadata; equal row counts are not evidence of compatibility. `min_sigma` / `max_sigma` are spatial standard deviations in real units (`sqrt(sigmax_2)` for variance priors).

**Cutoff semantics and provenance.** The tolerance formulas in `cutoffs.py` are kernel-specific: temporal omitted mass and `mean_lag_days` apply to `Temporal_Exponential`; spatial omitted mass applies to `Spatial_Symmetric_Gaussian` with the per-axis square cutoff. `Temporal_Power_Law.beta` is a shape parameter, not a mean lag. Do not apply the tolerance/scale interface to unsupported triggers; explicit physical `window` / `spatial_window` remains the compatible path. Validate every supplied tolerance as finite and in `(0, 1)` even when a physical cutoff wins precedence. `set_window` uses a private `_UNSET` sentinel so omission differs from explicit `None`; updates cutoffs, pairs, support, and `cutoff_provenance` transactionally; persists design scales for honest realized-omission recomputation. Temporal-only polygon window changes reuse the installed mass table; spatial-window changes require a compatible `mass_table=` replacement validated against the prospective window before mutation.

**Silent-failure traps already fixed once — don't reintroduce:**

- `jax.ops.segment_sum` drops out-of-range indices silently: excitation pairs and every other event-indexed object must always be built from the SAME event set the likelihood scores (`log_expected_likelihood` rebuilds held-out state and rejects test times outside `[0, T]`).
- numpyro `Predictive` drops missing site names silently: the `run_svi` sites list is model-aware (`Itot_excite` only for hawkes/cox_hawkes).
- Simulators must operate on `.copy()`s — never write columns into `self.A` / `self.spatial_cov`.
- The `comp_grid.sjoin(A)` used for spatial mass has border duplicates, so `Ih` is not `Itot_xy` in general — mirror the simulator's own computation when checking it.



## General testing conventions

Tests are mostly regression pins: they typically map to a specific audit fix (see the docstring headers and matching commits). The standard pattern is to trace the model directly — `handlers.trace(handlers.seed(model.model, PRNGKey(0))).get_trace(model.args)`, with `handlers.substitute` for fixed-parameter comparisons — rather than running inference.

**D-41 governs evidence and provenance. Read it in `phase3_record.tex` (A-31) before the first commit of any series; this is a pointer, not a restatement.** Its clauses:

- Every enforcement row is demonstrated RED against the pre-change state before it is claimed to enforce anything, and the capture is committed.
- **The revert is minimal** — revert only what the row is meant to detect. Reverting shared API alongside it yields an `ImportError` at collection, a red that proves nothing.
- Rows that pass on both sides are recorded as **non-discriminating by construction** and never offered as evidence.
- Every capture records **the checked process's own exit status**, not a pipeline's (`$?` after a `| tail` is `tail`'s). A capture showing success beside a failure is worse than no capture.
- **Post-commit verification:** after committing, `git status --porcelain` is empty *and* `git show --stat HEAD` is compared against the file list stated *before* committing. A commit whose contents were not compared to a pre-stated list is ungated regardless of what the gates said. Gates run on a worktree; a commit is a claim about a subset of it.
- **Never route stderr to `/dev/null` on a git command.** One bad pathspec aborts the whole `git add`; with stderr discarded it aborts silently, and ` M` in `git status` is *unstaged*, not staged.
- Relabeling in place is permitted where no assertion changes and the head note carries the full old→new mapping; anything that changes an assertion uses `\supsd` per occurrence.
- Claims of different strength are checked at their own strength (production AST-identical; test files string-blanked structural match).

**Validation that never runs is not validation, and an unreached guard is indistinguishable from an absent one.** Show every new guard firing on at least one public path; a guard that cannot is a finding, not a footnote. A-27 found all six of `NumericalConfig`'s σ/mode branches dead on every public path because the resolver ran first.

A green suite alone is not a Phase 3 acceptance record. For each checkpoint or semantic correction, preserve the targeted RED/GREEN evidence; run the suite (**state which lane** — see the two-lane note above); run all **six** configurations in the machine-local golden-pin harness `refactor-patches/pin_check_v2.py` and take the verdict from `refactor-patches/pin_compare.py` (the harness prints no verdict; it writes the candidate) — **twice, once against each baseline, and state the pin-comparison population in the gate block**: the canonical 2026-07 file gates the four legacy configurations and now says so in its own verdict (`PIN_DIFFS 0 PARTIAL compared=4/6`, A-48), and `--baseline refactor-patches/baselines-2026-08-polygon/pins.json` is the *only* verdict that gates the two polygon-domain ones (`compared=6/6 MATCH`). **A verdict reading `PARTIAL` is not a failure and a `MATCH` over a subset is no longer possible;** both populations belong in the commit body; run the corpus-identity gate `refactor-patches/pin_corpus_identity.py`, whose property **CHANGED ONCE at A-47** and must not be cited in its old form: A-38's retroactive rescue **expired** there and covers only commits before it, and the gate now asks that **every content group equal one of the two DECLARED baselines** (2026-07 canonical, 2026-08 polygon), with an *undeclared* group red. Adding a third declared era is the one edit that could be a relaxation — it needs a register amendment, and `tests/test_pin_corpus_identity.py` fails until the era count is updated deliberately; run `results/_a48_ruff_population.py <touched files>` rather than bare `ruff`, and **state the ruff population** — it names each file's state, its counts by rule code before and after against *its own* `HEAD` version under `HEAD`'s config, and the per-code delta, with new files reported against `baseline: NONE` instead of subtracted against a zero; record inherited findings by name rather than fixing them silently; and run the register gates — the ASCII sweep (`results/_a26_ascii_sweep.py`, which evaluates the clause functions and **states its own coverage as a fraction of raise sites**), the content/decision-monotonicity checks (`results/_a25_content_checks.py`), the `\hypertarget` structural check, and the unreachable-citation sweep. Record the exact commands, outputs, and change classification in the rebaseline document. Capture `git status --porcelain` with every run and `bstpp.__file__` with every ad-hoc probe — a stale `bstpp` in `site-packages` shadows the repo for scripts run from a subdirectory, and the capture then describes a different object.

Phase 3d/3e acceptance records are backfilled post-hoc where contemporaneous records were missing — label that honesty explicitly and never claim a historical RED run that was not observed.

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
Refactor: write the test first and observe it fail against the pre-change
state, with the **minimal** revert (D-41) — reverting shared API alongside the
defect yields an `ImportError` at collection, which is a red that proves
nothing. Commit the RED capture with the fix.
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
- **No wildcard imports** (`from x import` *) — they hide provenance and clash.
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

- **Reread this file at the start of each session** and after any context
clear/compact, then read Part II of `phase3_record.tex` from the most recent
amendment backwards far enough to cover the work in hand — that is the governing
record, and this file is a summary of it. (`PLANNING.md`, `TASKS.md` and
`SCRATCHPAD.md` were named here historically and **do not exist**; the register
plus `refactor-patches/pre-3f-stabilization/traceability_matrix.md` replaced
them.)
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
- Dev tooling: `ruff` (0.15.21, linter/formatter) is installed with
`pip install --no-deps ruff`; it is not part of the runtime pins. **It is not on
PATH — invoke it as `"$PY" -m ruff check <paths>`**; bare `ruff` is
`command not found`. Lint only what you touched, and record inherited findings
by name and count rather than fixing them silently (D-41): demonstrate them
against `git show HEAD:<file>` so "pre-existing" is measured, not asserted.
- Suite runtime is dominated by twelve `slow`-marked tests, not by machine
health — see the two-lane note above. Before attributing a slow gate to the
environment, run `--durations=10` and check *where* the time goes; a stale
"~1 min" figure in this file previously supported an unsupported
memory-pressure diagnosis for months.
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