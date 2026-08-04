# Traceability matrix + Lane B state machine

**Candidate tip:** Phase 3f WP1 (opened at `fe44de4`; see A-22/A-23/D-35)  
**Iteration:** 3f-WP1  
**Marks:** verified / reported / inferred as noted.  
**Governing record:** repository-root `phase3_record.tex`. Untracked root `phase3_baseline_and_decisions.tex` is **not** governing.

## Lane B — axis level sets (from code)

| Axis | Level set (verified in code) | Primary sites |
|---|---|---|
| Model family | `{plain Hawkes, Cox–Hawkes, LGCP}` | `main.py` |
| Support | `{rectangle, polygon}`; GeoDataFrame nonrectangular requires explicit mode (D-23) | `resolve_excitation_support_mode` |
| Trigger temporal | `{Temporal_Exponential, Temporal_Power_Law, custom}` | `trigger.py`, capability gates |
| Trigger spatial | `{Spatial_Symmetric_Gaussian, custom}`; polygon exact Gaussian | same |
| Cutoff input | physical / tol / omitted / `set_window` `_UNSET` vs `None` | `cutoffs.py`, `set_window` |
| Entry path | constructor; `Hawkes_Model.set_window`; `LGCP_Model.set_window` → `NotImplementedError` | `main.py` |
| Builder numerics | on `prepare_polygon_mass_table` only; model holds `NumericalConfig` (`bstpp/config.py`) behind `args` | `config.py`, `polygon_mass.py` |
| Standardization | `{None, "domain_area"}`; bool rejected | `main.py`, `test_standardization_api` |
| σ bounds | polygon `min_sigma` required; rectangle both or neither | `resolve_sigma_bounds` |

### `PRODUCTION_TAU_ABS` disposition (corrected)

| Source | Value | Status |
|---|---|---|
| A-21 / code | `PRODUCTION_TAU_ABS = TAU_ABS = 1e-5` | **authority for mass-table budget** (`polygon_mass.py:62–63`) |
| Legacy shootout / old derivation text | `LEGACY_SHOOTOUT_TAU_ABS = 5.39e-4` | **not** the production mass budget |
| OP-12 | `TAU_DERIV = 5.39e-4` provisional | open; not mass-table accept/reject |

### Mass-table budget design (`13a8525` + Commit C `86ca179`)

| Question | Answer (verified) |
|---|---|
| Where do install sites read `h_panel`/`gl_order`? | From the **table** inside `validate_polygon_mass_table`. |
| Budget assertion? | **Measured** residual vs host float64 elevated-GL (`BUDGET_REFERENCE_GL_ORDER=32`) ≤ `PRODUCTION_TAU_ABS`; panel-ratio is a prefilter only (surrogate invalid at `gl_order=8`). Budget policy fields are owned by `NumericalConfig` (WP1) and read by the three validation sites when the model passes `numerical_config=`. |
| `build_excitation_support` kwargs? | **Removed** (`panel_h_m` / `gl_order` no longer in signature); accepts optional `numerical_config`. |
| Change class | **CF** (pre-3f); WP1 layering is **BP** |

### Pairwise / forced coverage

| Family × support | Evidence | Gap |
|---|---|---|
| Hawkes × rectangle | pins + Lane B | — |
| Hawkes × polygon | phase3d + B1 + Lane B + I11 | — |
| Cox–Hawkes × rectangle/polygon | smoke + Lane B | — |
| LGCP × rectangle | smoke + Lane B set_window reject | — |

**Executable gate:** `tests/test_lane_b_config_matrix.py`. Claim: **forced rows + nine-axis pairwise covering array (fraction 1.000)** — see `docs/config_matrix.md`.

## Register traceability

Register inventory: **D-1…D-35**; **A-1…A-23**; **OP-2…OP-13** (OP-1 never allocated); **I1, I3–I6, I8/I8a/I8b, I9–I12**.

| ID | Contract (short) | Register status | Production sites | Execution legs | Existing evidence | Evidence type | Gap | Treatment | Owner |
|---|---|---|---|---|---|---|---|---|---|
| D-1 | Hybrid phase ordering | active | process | n/a | A-21 | inferred | n/a | follow ordering | team |
| D-2 | A is scientific+observed domain | active | data_contracts, prepare_domain | validate/sim | tests | integration | none material | keep | 3g |
| D-3 | Events outside A rejected | active | validate_events | ctor | tests + hole probe | direct | none | keep | — |
| D-4 | Boundary points inside | active | covers / covers_xy | validate/sim | tests + hole covers_xy | direct | duplicate preds | 3f single-source | 3f |
| D-5/D-22 | Half-open membership | active | membership | grid | test_membership_d22 | direct | none | keep | — |
| D-6/D-7 | Clipped support areas | active | preparation | bg/lik | clipped tests | integration | none | keep | — |
| D-8/D-9 | District vs citywide | active | domain choice | n/a | doc | inferred | no executable citywide | post-Phase-3 | post-Phase-3 |
| D-10–D-12 | Standardization | active | standardize_cov | prep | test_standardization_api | direct | OP-5 deferred | keep | — |
| D-13/D-14/D-21 | Infinite kernels; cutoffs; square | active | cutoffs, utils, lik, sim | three legs | phase3e + identities | integration | none | keep | — |
| D-15/D-16 | Human units; β mean lag | active | triggers/cutoffs | ctor | phase3e, gates | direct | none | keep | — |
| D-17/D-18/D-23 | Two modes; one support; no silent default | active | ExcitationSupport | parenting+comp | phase3d + Lane B | structural | none | keep | — |
| D-19 | City-scale out of scope | active | n/a | n/a | doc | inferred | n/a | defer | post-Phase-3 |
| D-20 | Stage3 SBC R=200 exit | active | SBC | exit | baselines (**reported**); tip not rerun | integration | tip exit | Phase3 exit | Phase3 exit |
| D-24 | Cutoff tol + provenance | active | `cutoffs.py`; package default tols also frozen on `NumericalConfig.default_*_tol` (`bstpp/config.py`) | ctor/set_window | phase3e + WP1 config | direct | none | keep | — |
| D-25 | Hermite mass table | active | `polygon_mass.py`; budget policy on `NumericalConfig` (`production_tau_abs`, `budget_reference_gl_order`) | prepare/eval/install | production + B1 + `test_numerical_config` | numerical | none | keep | 3f |
| D-26 | Hard-require prepared table | active | `excitation_support.build_excitation_support`, `main.py`; install reads `numerical_config` | install | B1 + heldout | direct | B4 closed `6ba2194` | keep | — |
| D-27 | Compat identity ≠ equal counts | active | `validate_polygon_mass_table` (budget via `numerical_config=`) | install | compat suite | direct | none | keep | — |
| D-35 | Freeze ships with enforcement; config `__post_init__` + single factory; set_window mass_table= rebuilds config transactionally | active | `bstpp/config.py` `NumericalConfig.create` / `__post_init__`; `Hawkes_Model.set_window` | ctor + set_window | `tests/test_numerical_config.py`; `tests/test_set_window_sentinel.py` (rebuild + rollback rows) | direct | — | keep | 3f |
| D-28/D-29 | Prior truncation; σ disclosure | active | truncate_sigmax_2_prior | polygon | TLN tests | direct | G3 | 3g | 3g |
| D-30 | Union area authoritative | active | PreparedDomain, sim, bg | area/sim | domain_union + iter2 OV probe | direct | B3 alias of bounds | copy bounds | pre-3f |
| D-31 | data_contracts default reject | active | ctor | validate | data_contracts tests | direct | none | keep | — |
| D-32 | Held-out standalone | active | log_expected_likelihood | scoring | heldout suite | direct | none | keep | — |
| D-33 | Transactional set_window (windows/pairs/support/provenance/`numerical_config`) | active | `Hawkes_Model.set_window` atomic commit | mutator | sentinel + Lane B rollback + WP1.4b numerical_config rows | direct | only one mutator | 3f | 3f |
| D-34 | CRS never adopted/inferred silently; tabular cov requires `spatial_cov_crs` | active | data_contracts, preparation attach | ctor | A-16 commits + `tests/test_crs_set_crs_paths.py` (suite presence); record text | **verified** (record+sites); suite not re-run iter2 | none material | keep | — |
| A-1…A-5 | Early Phase 3a–3c amendments (contracts, standardization freeze path, area union) | historical/active | various | various | phase3a/c rebaseline records | **reported** | not re-executed tip | keep records | 3g |
| A-6 / A-9…A-16 | 3c–3e contract amendments establishing D-23…D-34 cluster | active | see D-rows | ctor/set_window | phase3d/e records + tip suites | **re-established** selectively via suites/probes this series | — | keep | — |
| A-7 / A-17…A-18 | SBC 2p / scope / tip verification hygiene | active/historical | SBC, docs | exit | baselines + tip verify md | **reported** | Stage3 tip exit outstanding | Phase3 exit | Phase3 exit |
| A-19 | CF class | active | process | n/a | used for B1 | verified process | n/a | keep | — |
| A-20 / A-21 | Config placement + pre-3f freeze | active | many | many | A-21 text + this audit | mixed | B2–B4 closed pre-3f | keep | — |
| A-22 | Phase 3f opened at `fe44de4` | active | process | n/a | register | verified | G1/G2 open | 3f WPs | 3f |
| A-23 | Frozen dataclasses not Pydantic | active | `bstpp/config.py` | ctor | WP1.0 reasoning in register | verified | — | keep | 3f |
| OP-2 | → D-23 | resolved | resolve mode | ctor | phase3d | direct | none | closed | — |
| OP-3/OP-4 | standardize default/API | settled A-21 | standardize_cov | prep | test_standardization_api | verified | Part I stale | A-21 operative | — |
| OP-5 | User weights | deferred | n/a | n/a | register | inferred | YAGNI | post-Phase-3 | post-Phase-3 |
| OP-6 | → D-24 | resolved | cutoffs | ctor/set_window | phase3e | direct | none | closed | — |
| OP-7 | subsumed OP-13 | subsumed | `NumericalConfig` cutoff tols | 3f | A-20/21/23 | direct (WP1) | remaining configs | 3f | 3f |
| OP-8 | Remove `args` | open | self.args (adapter last) | all | A-21/A-22 | inferred | final WP | 3f | 3f |
| OP-9 | → D-25 | resolved | polygon_mass | prepare | B1/production | numerical | none | closed | — |
| OP-10 | max_sigma default | settled A-21 | resolve_sigma_bounds | polygon | code | direct | none | closed | — |
| OP-11 | Custom polygon mass | open post-Phase-3 | exact-type gate | polygon | capability | direct | placeholder | post-Phase-3 | post-Phase-3 |
| OP-12 | Derivative gate | open | TAU_DERIV | table QA | production | numerical | unsettled | pre-polygon-SBC | pre-polygon-SBC |
| OP-13 | Config objects | settled A-21; A-23 dataclass deviation | `bstpp/config.py` `NumericalConfig` (WP1); four others pending | 3f | A-21/A-23 + WP1 tests | direct (partial) | Model/Prior/Partition/Inference configs | 3f | 3f |
| I1 | Sim/lik mass atoms | active | lik + _sim_cox | bg | seasonal_integral | direct | none | keep | — |
| I3/I4 | Pair window contract | active | utils, lik, sim | pairs | identities | direct | none | keep | — |
| I5 | Single factor site | active | inference_functions | model | test_smoke | direct | none | keep | — |
| I6 | Derived seasonal coordinate | active | `_scale_xyt` | prep | `tests/test_smoke.py::test_A_derivation` | **verified** EXIT:0 iter2 | none | keep | — |
| I8* | Special-case reductions | active | identities | model | test_identities | direct | none | keep | — |
| I9 | Refinement invariance | active | clipped support | prep | clipped tests | direct | none | keep | — |
| I10 | Unit covariance / real-unit spatial kernel (internal affine invariance; non-uniform rescaling changes loglik) | active | identities + real-unit boundary | lik/sim | `test_spatial_similarity_covariance`, `test_spatial_kernel_family_is_real_unit_not_internal` | **verified** EXIT:0 iter2 | none | keep | — |
| I11 | Conservation E[n]≈Λ | active | simulate + compensator | rect+polygon | rectangle standing test; polygon `test_polygon_i11_conservation` R=40 @ 3·se (`c5e4871`) | integration | — | standing test | pre-3f |
| I12 | Real-unit nonsquare | active | utils/trigger/sim | nonsquare | pins + identities | direct | none | keep | — |
| C6 | Guide identity | open | docs | n/a | A-21 | inferred | unresolved | 3g | 3g |

**Duplicated / missing / contradictory notes:**

- OP-1 missing (never allocated). I2/I7 absent.  
- Part I OP-3/4 prose vs A-21: Part II governs (G8).  
- Untracked `phase3_baseline_and_decisions.tex` not governing.
