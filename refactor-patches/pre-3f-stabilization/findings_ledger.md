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
