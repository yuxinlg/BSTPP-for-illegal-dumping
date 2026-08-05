# Findings ledger

**Candidate tip:** `c5e48713ec1abd58034a9dfc32f0cb8577ba756f`  
**Iteration:** 3 (repairs + verification)  
**Audit date:** 2026-08-04  
**Environment:** jax==0.4.23, jaxlib==0.4.23, numpyro==0.15.0, numpy==1.26.4, scipy==1.11.4, geopandas==1.1.3, jax_enable_x64=False, platform=cpu  

| ID | Finding or gap | Contract IDs | Evidence (mark + command/artifact) | Production reachability | Pin or frozen surface? | Change class | Severity | Class-level remediation | Required gates | Owner | Review date | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Install validated mass tables against DEFAULT panel/gl | D-25/26, A-21 | **verified** RED `9e37dc3`; GREEN `13a8525`; superseded by Commit C measured residual | polygon install | freeze | CF | BLOCKER (resolved) | table settings authoritative + measured residual | B1 suite | pre-3f | 2026-08-04 | **closed** |
| B2 | `simulate(rng=None)` live-consumes `np.random` | Lane D RNG | **verified** RED `0d44ce5` / `commitA_red_at_143d219.md`; GREEN `6ba2194` require Generator | simulate | freeze | CF | BLOCKER (resolved) | ownership class: no global entropy | ownership suite | pre-3f | 2026-08-04 | **closed** |
| B3 | Caller domain `A` aliased into prepared/args | Lane D; D-30 | **verified** RED `0d44ce5`; GREEN `6ba2194` copy at prepare/ModelData | array/GDF domains | freeze | CF | BLOCKER (resolved) | copy at ownership boundary | ownership suite | pre-3f | 2026-08-04 | **closed** |
| B4 | Mass table installed by identity; mutable `.values` | D-26/27 | **verified** RED `0d44ce5`; GREEN `6ba2194` `PolygonMassTable.copy()` at install | polygon install | freeze | CF | BLOCKER (resolved) | copy at install | ownership suite | pre-3f | 2026-08-04 | **closed** |
| B5 | Panel-ratio surrogate claimed to protect `PRODUCTION_TAU_ABS` but invalid at `gl_order=8` | A-21 | **verified** Commit C probe: ratio=8 / gl=8 max_abs≈1.8e-5…9e-4 > 1e-5; GREEN `86ca179` measured residual | polygon install | freeze | CF | BLOCKER (resolved) | measure vs elevated-GL host quad | polygon mass suites | pre-3f | 2026-08-04 | **closed** |
| B6 | `build_excitation_support` silent no-op `panel_h_m`/`gl_order` kwargs | §5.2 | **verified** iter2 del-unused; GREEN `86ca179` removed from signature | polygon install call sites | freeze API | CF/API | BLOCKER (resolved) | remove or honor; removed | TypeError on kwargs | pre-3f | 2026-08-04 | **closed** |
| G1 | Domain membership duplicated | D-4, D-30 | **verified** iter1 | validate+sim | freeze API | DOC/3f | NONBLOCKING-3F | structural single-source | membership suites | 3f | 2026-08-04 | open |
| G2 | `save_rslts` omits cutoff/excitation provenance | A-21; D-24 | **verified** Lane B strict xfail | after fit | freeze I/O | API | NONBLOCKING-3F | A-21 save/load | round-trip | 3f | 2026-08-04 | open |
| G3 | No package-wide distribution property suite | D-28 | **verified** TLN only | truncation | not pin | test | NONBLOCKING-3G | parametrized dist props | dist module | 3g | 2026-08-04 | open |
| G4 | Lane B pairwise incomplete | §6/§9 | **verified** closed Commit D `5f4a54b`: 9-axis pairwise 1.000; `docs/config_matrix.md` | config | freeze | test | closed | covering array | Lane B gate | pre-3f | 2026-08-04 | **closed** |
| G5 | Polygon I11 no standing test | I11 | **verified** closed Commit E `c5e4871`: `test_polygon_i11_conservation` R=40 @ 3·se (defaults + small σ) | polygon | not pin | test | closed | standing I11 | conservation | pre-3f | 2026-08-04 | **closed** |
| G6 | Pin JSON lacks commit/env/hash | §5.3/§5.6 | **reported** | pins | pin gate | DOC | NONBLOCKING-3G | stamp identity | pin_check | 3g | 2026-08-04 | open |
| G7 | OP-8/12, C6 open by register | OP-8, OP-12, C6 | **reported** A-21 | 3f/docs | decisions | DOC | DEFERRED | per A-21 | scheduled | 3f/3g | 2026-08-04 | open |
| G8 | Part I OP-3/4 prose stale vs A-21 | OP-3/4 | **verified** | standardize | doc | DOC | NONBLOCKING-3G | Part II governs | doc review | 3g | 2026-08-04 | dispositioned |
| G9 | Packaging wheel flake | packaging | **verified** closed `705d040` | packaging | gate | test hygiene | closed | isolate build | suite | pre-3f | 2026-08-04 | **closed** |
| G10 | `run_svi` no `rng_key` | 3f RNG seam | **verified** closed `6ba2194`: `rng_key=` with None→PRNGKey(10) | SVI | freeze | API/CF | closed | same as MCMC | ownership suite | pre-3f | 2026-08-04 | **closed** |
| G11 | Lane B pairwise ~53% (6-axis) | §6 | **verified** closed by G4 / Commit D | config | claim | test | closed | 9-axis CA | matrix | pre-3f | 2026-08-04 | **closed** |

## Declared behavior change (Commit B)

`simulate` requires explicit `numpy.random.Generator` via `rng=`; `rng=None` raises `ValueError`. Recorded in `commitB_rebaseline.md`.

## Commit A RED evidence

At tip `143d219`, `tests/test_caller_state_ownership.py` → 7 failed / 6 passed / EXIT:1 (`commitA_red_at_143d219.md`). GREEN at `6ba2194`.

## WP1.4c — archive `refactor-patches/phase0/test_identities.py` (IV/DOC)

**Class set used:** import/signature drift · fixture/construction drift · numerical defect · **supersession** (requires named register decision; otherwise stays numerical).

**Disposition:** archive excluded from default collection via `pyproject.toml` `testpaths = ["tests"]` — not repaired, not deleted. The four failures are identical at Phase 3f open tip `fe44de4` (`results/_wp14c_fe44de4_four.txt`, worktree) and at tip (`results/_wp14c_pytest_archive_four.txt`): same four nodeids, same error classes/messages (I10 loglik values bit-identical across tips). Explicit path collection still runs the archive and still fails those four.

### Per-failure classification

| Archive test | Class | Superseding decision | Evidence |
|---|---|---|---|
| `test_covariate_refinement_invariance[False]` (I9) | Fixture/construction / signature drift | **A-21** settles **OP-3 / OP-4**: `standardize_cov` is `None` or `"domain_area"` only; booleans rejected | Construction raises before the identity runs (`standardize_cov` bool → reject). Live pin: `tests/test_identities.py::test_covariate_refinement_invariance[None/"domain_area"]`; API: `tests/test_standardization_api.py` |
| `test_covariate_refinement_invariance[True]` (I9) | same | same | same |
| `test_spatial_affine_unit_invariance` (I10) | **Supersession** | **D-15** (§5.9–5.10): σ, spatial cutoff, and excitation distances are lengths in the supplied coordinate system | Internal state invariant (archive lines 255–262 pass: `t_events`, `xy_events`, pair sets identical). Only loglik assert (line 268) fails. Three-transform table at fixed `sigmax_2=0.1` (verified WP1.4c): |
| | | | Transform \| loglik \| Δ |
| | | | identity \| −55.53381348 \| 0 |
| | | | pure translation \| −55.53381348 \| 0, **bit-identical** |
| | | | isotropic scale ×3 \| −58.14003372 \| −2.606 |
| | | | anisotropic 3.0 / 0.5 \| −54.16902161 \| +1.365 |
| | | | Translation invariance survives D-15; any spatial rescaling at fixed σ changes the model (isotropy incidental). Archive premise (“internal-unit log-likelihood exactly invariant under affine spatial map at fixed σ”) is the pre–D-15 contract. **OP-18** records the restated unit-change form (not yet derived as a test). Live restated coverage: `test_spatial_similarity_covariance`, `test_spatial_kernel_family_is_real_unit_not_internal` |
| `test_simulated_count_matches_compensator` (I11) | Fixture/construction drift | **D-23** (A-9): nonrectangular / GeoDataFrame domain requires explicit `excitation_support`; no silent default | Fails at construction: `Nonrectangular domain requires an explicit excitation_support`. Identity never scored. Live: `tests/test_identities.py::test_simulated_count_matches_compensator` (`excitation_support="rectangle"`); polygon standing: `tests/test_polygon_i11_conservation.py` (`c5e4871`) |

None of the four is a numerical defect under the register. Supersession applies only to I10 (D-15 named).

### I1–I11 live-coverage inventory (read from `tests/`; precondition for exclusion under D-39)

| Guide ID | Live equivalent under `tests/` | Notes |
|---|---|---|
| I1 | `tests/test_seasonal_integral.py::test_sim_likelihood_integral_identity` (+ related exact-integral rows) | |
| I2 | `tests/test_seasonal_integral.py::test_overlap_matrix_properties` | |
| I3 | `tests/test_pairs.py` (`test_equivalence_vs_dense`, `test_likelihood_invariance`, …) | |
| I4 | `tests/test_identities.py::test_offspring_thinning_matches_compensator_mass` (+ finite-window variant) | Archive marked (HERE); live present |
| I5 | `tests/test_smoke.py::test_single_factor_site` | |
| I6 | `tests/test_smoke.py::test_A_derivation` | |
| I7 | `tests/test_smoke.py::test_log_expected_likelihood_rebuilds_pairs` | |
| I8 | `tests/test_identities.py::test_alpha_zero_reduces_cox_hawkes_to_lgcp`, `test_window_at_horizon_recovers_untruncated_loglik` | |
| I9 | `tests/test_identities.py::test_covariate_refinement_invariance` (`None` / `"domain_area"`) | Archive bool params superseded by A-21 API |
| I10 | `tests/test_identities.py::test_spatial_similarity_covariance`, `test_spatial_kernel_family_is_real_unit_not_internal` | Archive affine-at-fixed-σ form superseded (D-15); **OP-18** for full unit-change derivation (σ + cutoff + Jacobian) |
| I11 | `tests/test_identities.py::test_simulated_count_matches_compensator`; `tests/test_polygon_i11_conservation.py` | Rectangle + polygon standing (`c5e4871`) |

No identity left without live coverage → no OP-19+ opened for unaudited identities. **OP-18** opened for the restated I10 derivation only.

### Suite invocation change

Figures below were **re-measured for A-25 from a clean detached worktree**; each capture
carries its own `git status --porcelain` (empty) as provenance. They replace the original
WP1.4c figures, which were wrong — see the correction note.

- Bare `pytest --collect-only -q` **before** `testpaths` (at `82e9c09^` = `dbccd3d`): **567** (`results/_wp14c_collect_before.txt`)
- Bare `pytest --collect-only -q` **after** (at `82e9c09`): **560** (`results/_wp14c_collect_after.txt`)
- Delta: **7**. The composition is **one archive file**, `refactor-patches/phase0/test_identities.py`, collecting seven tests: six test functions, one of them parametrized over two values (`results/_wp14c_collect_archive.txt`, `pytest refactor-patches/ --collect-only -q` at `dbccd3d` → 7 collected). It is the only file matching pytest's default `test_*.py` / `*_test.py` patterns anywhere outside `tests/` in the repository that defines any test function; `scripts/recover_test.py` matches `*_test.py` but defines none, so it contributes nothing to collection. **There are no SBC test files under `refactor-patches/` in the repository.**
- Other bare pytest invocations (CI / Makefile / scripts expecting bare `pytest`): **none** found (no `.github` workflows; docs/AGENTS already quote `pytest tests/`).

#### Correction note — why the original figure was wrong

The original WP1.4c entry recorded **582** before / 560 after / delta **22**, and attributed
the difference to "all of `refactor-patches/` under default discovery … SBC smokes account
for the rest." That was measured in a working tree containing **untracked test files that
are not in the repository** — `refactor-patches/sbc1/test_sbc_smoke.py`,
`refactor-patches/sbc2/test_sbc_smoke_v2.py`, and `refactor-patches/test_sbc_smoke_v3.py`,
three files contributing the fifteen extra collected tests (582 − 567 = 15; 7 + 15 = 22).

This is the finding, not a typo. A collection count is a statement about the repository's
own structure, and a figure taken in a contaminated tree made a **false statement about
which files the package contains** — it named a category of archived SBC tests that does
not exist under version control. That is why A-25's amendment to A-22's gate-profile
paragraph requires every captured gate run to record `git status --porcelain` alongside its
result: without it the contamination is invisible in the artifact, and the wrong figure is
indistinguishable from the right one.
