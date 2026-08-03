# Traceability matrix + Lane B state machine

**Candidate tip:** `cd62288`  
**Marks:** verified / reported / inferred as noted.

## Lane B — axis level sets (from code)

| Axis | Level set (verified in code) | Primary sites |
|---|---|---|
| Model family | `{plain Hawkes: cox_background=False, Cox–Hawkes: cox_background in {True,'cox'}, LGCP: LGCP_Model}` | `bstpp/main.py` `Hawkes_Model.__init__`, `LGCP_Model` |
| Support | `{rectangle, polygon}`; array domains default `rectangle`; GeoDataFrame nonrectangular requires explicit mode (D-23) | `resolve_excitation_support_mode`, `build_excitation_support` |
| Trigger temporal | `{Temporal_Exponential, Temporal_Power_Law, custom class}` — exponential-only kwargs gated by exact type | `trigger.py`, capability gates in `Hawkes_Model` |
| Trigger spatial | `{Spatial_Symmetric_Gaussian, custom}`; polygon requires **exact** Gaussian type | same |
| Cutoff input | physical `window` / `temporal_cutoff_days` / `spatial_window`; tols `temporal_cutoff_tol` / `spatial_cutoff_tol` / `cutoff_tol`; design scales; omitted → untruncated defaults; `set_window` `_UNSET` vs explicit `None` | `cutoffs.py`, `Hawkes_Model.set_window` |
| Entry path | constructor; `Hawkes_Model.set_window`; `LGCP_Model.set_window` → always `NotImplementedError`; no other public mutators found | `main.py` |
| Builder numerics | `panel_h_m` / `gl_order` on `prepare_polygon_mass_table` / `build_excitation_support` defaults; **not** Hawkes constructor kwargs | `polygon_mass.py`, `excitation_support.py` |
| Standardization | `standardize_cov in {None, "domain_area"}`; default **`None`**; bool rejected | `main.py:236`, `tests/test_standardization_api.py` (**verified**) |
| σ bounds | polygon: `min_sigma` required; `max_sigma` default 5 km via CRS; rectangle: both or neither | `resolve_sigma_bounds` |
| Outcomes | success; named `ValueError`/`TypeError`/`NotImplementedError` at validation points; transactional rollback on failed `set_window` | Phase 3d/3e tests |

### `PRODUCTION_TAU_ABS` disposition

| Source | Value | Status |
|---|---|---|
| Protocol §6 derivation text (10% of 3σ cutoff) | `5.392e-4` | **stale vs A-21**; retained historically as `LEGACY_SHOOTOUT_TAU_ABS` |
| A-21 / code | `PRODUCTION_TAU_ABS = TAU_ABS = 1e-5` | **verified** `polygon_mass.py:62–63` |
| Derivative gate | `TAU_DERIV = 5.39e-4` provisional (OP-12 open) | **verified** |

## Pairwise / forced coverage sketch

| Family × support | Existing discriminating evidence | Gap |
|---|---|---|
| Hawkes × rectangle | pins + smoke + identities + cutoffs | Lane B full pairwise matrix suite **absent** as one gate |
| Hawkes × polygon | phase3d, heldout, mass prepare/compat; Lane B matrix + B1 install suite | B1 closed (`13a8525`) |
| Cox–Hawkes × rectangle | pin + smoke + Lane B | — |
| Cox–Hawkes × polygon | Lane B `test_lane_b_cox_hawkes_polygon_constructs` | deepen in iteration 2 |
| LGCP × rectangle | pin + smoke + sim union filter | n/a excitation |
| LGCP × polygon domain (background only) | SBC stage2p artifact (**reported**); `set_window` rejected | n/a |

**Forced reject rows with tests (verified by suite subset):** unsupported kernel kwargs (`test_kernel_capability_gates`); LGCP `set_window` (`test_lgcp_set_window`); polygon without mass_table; held-out without/wrong table; bool `standardize_cov`; sentinel/`None` semantics (`test_set_window_sentinel`, untouched provenance).

**Specify (do not add in this pass):**

1. `test_polygon_install_uses_table_or_model_panel_gl_settings` — CRS-less `min_sigma` small enough that default panel fails prepare; prepare with `panel_h_m <= 8*min_sigma`; construct with matching settings → success; construct without forwarding → current failure (RED on tip).  
2. `test_nondefault_gl_order_round_trips_install` — prepare `gl_order=8`, install with same → success; mismatch → named error.  
3. Shared state-machine parametrization over every public mutator (currently only `set_window`) for success/sentinel/rollback/provenance identity.  
4. Pairwise family×support×cutoff-entry forced rows as one marked Lane B module.

## Register traceability (active IDs)

Register inventory from `phase3_record.tex` at tip: **D-1…D-34** contiguous; **A-1…A-21**; **OP-2…OP-13** (OP-1 never allocated); **I1, I3–I6, I8/I8a/I8b, I9–I12** (I2, I7 absent).

| ID | Contract (short) | Register status | Production sites | Execution legs | Existing evidence | Evidence type | Gap | Treatment | Owner |
|---|---|---|---|---|---|---|---|---|---|
| D-1 | Hybrid phase ordering | active | process | n/a | A-21 / this audit | manual review | n/a | follow ordering | team |
| D-2 | A is scientific+observed domain | active | `data_contracts`, `prepare_domain` | validate/sim | `test_data_contracts`, clipped support | integration | none material | keep | 3g docs |
| D-3 | Events outside A rejected | active | `validate_events`, membership | ctor/held-out | tests outside-domain | direct | none | keep | — |
| D-4 | Boundary points inside | active | `shapely.covers` / `covers_xy` | validate/sim | boundary tests | direct | duplicate predicates | structural single-source in 3f | 3f |
| D-5/D-22 | Half-open unique membership | active | membership helpers | grid joins | `test_membership_d22` | direct | none | keep | — |
| D-6/D-7 | Clipped support areas | active | `preparation` refinement | bg/lik | `test_clipped_support`, likelihood atoms | integration | none | keep | — |
| D-8/D-9 | District vs citywide excitation | active | model domain choice | n/a | doc | inferred | executable citywide test absent | review check | post-Phase-3 |
| D-10–D-12 | Standardization principles | active | `standardize_cov` | prep | `test_standardization_api` | direct | OP-5 weights deferred | keep | — |
| D-13/D-14/D-21 | Infinite kernels; cutoffs; square | active | `cutoffs`, `utils`, lik, sim | all three legs | phase3e + identities | integration | none | keep | — |
| D-15/D-16 | Human units; β mean lag | active | triggers/cutoffs | ctor | phase3e, capability gates | direct | none | keep | — |
| D-17/D-18/D-23 | Two modes; one support object; no silent default | active | `ExcitationSupport` | parenting+compensator | phase3d | structural | none | keep | — |
| D-19 | City-scale out of scope | active | n/a | n/a | doc | inferred | n/a | defer | post-Phase-3 |
| D-20 | Stage3 SBC R=200 exit | active | SBC harness | exit gate | baselines-2026-07 stage3 (**reported**); tip exit **not** rerun | integration | tip exit outstanding | run at Phase3 tip | Phase3 exit |
| D-24–D-27 | Cutoff tol; Hermite table; hard-require; compat identity | active | cutoffs, polygon_mass, excitation_support | prepare/install/score | phase3d/e, compat, heldout | direct | **B1** install vs defaults | CF+API repair | pre-3f |
| D-28/D-29 | Prior truncation; σ bounds disclosure | active | `truncate_sigmax_2_prior` | polygon ctor | truncated_lognormal, onesided tests | direct | extreme float32 notes | property suite later | 3g/num |
| D-30 | Union area authoritative | active | `PreparedDomain`, sim filter, hawkes bg | area/sim/bg | domain_union, hawkes_bg_union, simulate_* | direct | `_plot_grid` multi-row (plot only) | defer plot | 3g |
| D-31 | data_contracts default reject | active | ctor | validate | data_contracts tests | direct | none | keep | — |
| D-32 | Held-out standalone realization | active | `log_expected_likelihood` | scoring | `test_heldout_polygon_mass` | direct | none on instance | keep | — |
| D-33 | Transactional set_window | active | `set_window` | mutator | sentinel + untouched + phase3e | direct | class-level mutator matrix incomplete (only one mutator) | extend matrix | 3f |
| D-34 | (see record) | active | per record | per record | A-21 text | reported | confirm in record row if needed | review | — |
| A-1…A-20 | Phase amendments | active/historical | various | various | rebaseline records | reported | re-established selectively this pass | see closeouts | — |
| A-21 | Pre-3f freeze + CF series | active | many | many | `pre3f_audit_e0d7e43.md` + this audit | mixed | panel plumbing deferred but **B1** bites now | repair B1 before READY | pre-3f |
| OP-2 | Support mode default | **resolved→D-23** | resolve mode | ctor | phase3d | direct | none | closed | — |
| OP-3/OP-4 | Standardization default/API | **settled A-21**; Part I historical text still says open | `standardize_cov=None` | prep | `test_standardization_api` **verified** | direct | Part I prose stale by design | treat A-21 as operative | — |
| OP-5 | User weights | open/deferred | n/a | n/a | register | inferred | deferred | YAGNI | post-Phase-3 |
| OP-6 | Cutoff interface | **resolved→D-24** | cutoffs | ctor/set_window | phase3e | direct | none | closed | — |
| OP-7 | Cutoff config placement | **subsumed by OP-13** | future NumericalConfig | 3f | A-20/A-21 | inferred | implement in 3f | 3f | 3f |
| OP-8 | Remove legacy `args` | **open** (3f design) | `self.args` everywhere | all | A-21 | inferred | removal sequencing | 3f design | 3f |
| OP-9 | Polygon backend | **resolved→D-25** | polygon_mass | prepare/eval | production gate + shootout | numerical | none | closed | — |
| OP-10 | max_sigma default | **settled A-21** keep 5 km disclosed | `resolve_sigma_bounds` | polygon | code meta `default_5km` | direct | none | closed | — |
| OP-11 | Custom polygon mass | open post-Phase-3 | exact-type gate | polygon ctor | capability/phase3d | direct | placeholder gate | post-Phase-3 | post-Phase-3 |
| OP-12 | Derivative gate | **open** | `TAU_DERIV` | table QA | production gate reports separately | numerical | policy unsettled | before polygon SBC | pre-polygon-SBC |
| OP-13 | Config object placement | **settled A-21** five Pydantic models | not implemented | 3f | A-21 binding text | inferred | objects not created | implement in 3f | 3f |
| I1 | Sim/lik mass atoms | active | lik + `_sim_cox` | bg | seasonal_integral, likelihood_atoms | direct | none | keep | — |
| I3/I4 | Pair window contract | active | utils, lik, sim | pairs | pairs + identities | direct | none | keep | — |
| I5 | Single factor site | active | inference_functions | model | `test_smoke` | direct | none | keep | — |
| I6 | Seasonal coordinate | active | `_scale_xyt` | prep | record cites tests | reported | n/a | keep | — |
| I8/I8a/I8b | Special-case reductions | active | identities | model | `test_identities` | direct | none | keep | — |
| I9 | Refinement invariance | active | clipped support | prep | clipped tests | direct | none | keep | — |
| I10 | (per suite) | active | identities header | — | test_identities | reported | confirm label mapping | review | — |
| I11 | Conservation E[n]≈Λ | active | simulate + compensator | rectangle Hawkes | `test_simulated_count_matches_compensator` (+ finite w_s variants) | integration | **polygon excitation conservation not standing** | add polygon regime test | pre-3f/3g |
| I12 | Real-unit nonsquare invariance | active | utils/trigger/sim | nonsquare | pin 4:1 + identity tests | direct | none | keep | — |
| C6 | Guide identity | open | docs | n/a | A-21: does not block 3f start | inferred | unresolved identity | 3g/guide | 3g |

**Duplicated / missing / contradictory notes (Step 0, rechecked):**

- OP-1 missing (never allocated) — not a contradiction.  
- I2/I7 missing from inventory — not referenced as active obligations here.  
- Part I / A-5 prose still saying OP-3/4 open + legacy count default **contradicts** A-21 + code; amendment rules make A-21 operative (**disposition: not §5.5 blocker** if Part II governs).  
- Untracked root `phase3_baseline_and_decisions.tex` must not be treated as governing.
