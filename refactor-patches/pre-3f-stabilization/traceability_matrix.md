# Traceability matrix + Lane B state machine

**Candidate tip:** Phase 3f (opened at `fe44de4`; A-22/A-23/D-35; A-24 adopts D-36–D-39 / OP-14–OP-17; A-25 records WP1 execution and corrects the WP1.4c suite figures; A-26 establishes D-40 and closes OP-17)  
**Iteration:** 3f-WP1 + A-24 register adoption + A-25 execution record + A-26 error-identity unification (WP1.4e-1)  
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

Register inventory: **D-1…D-40**; **A-1…A-26**; **OP-2…OP-19** (OP-1 never allocated. No OP was opened for an unaudited *identity*: all guide I1–I11 have live `tests/` coverage after the WP1.4c inventory. OP-18 is the restated I10 derivation; **OP-19** is A-26's held-out budget-policy source, not an identity gap); **I1, I3–I6, I8/I8a/I8b, I9–I12**.

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
| A-24 | Closeout decisions D-36–D-39; Lane B named as §12 gate; OP-14–OP-17; amends A-22's gate-profile paragraph | active | process (no production code site) | n/a | A-24 + this matrix | verified DOC | coverage delta OP-14–OP-17 | keep | 3f |
| A-25 | Phase 3f WP1 execution record; WP1.4c suite figures corrected (582/Δ22 → 567/Δ7); A-22 gate profile: `pytest tests/` supersedes “Full suite”, and every captured gate run records `git status --porcelain` | active | process (no production code site); enforcement site is `pyproject.toml` `testpaths` | n/a | A-25; `results/_a25_collect_*.txt`; `results/_wp14c_collect_*.txt`; `findings_ledger.md` “Suite invocation change” | **verified** (re-measured, clean worktree) | D-40 / OP-17 and OP-18 carried forward | keep | 3f |
| A-26 | Error-identity unification (SC): D-40 established; OP-17 closed; OP-19 opened; A-24 P1 regraded to *asserts*; A-25 probe provenance extended with `bstpp.__file__` | active | `bstpp/config.py`, `bstpp/polygon_mass.py`, `bstpp/main.py`, `bstpp/preparation.py` | ctor/set_window/held-out/build | A-26; `refactor-patches/phase3f/rebaseline_record.md`; RED evidence `results/_a26_red_entry_path_row.txt`; pins `results/_a26_1_pins.txt` | **verified** (RED→GREEN; `PIN_DIFFS 0 MATCH`) | sigma/mode split open | WP1.4e-2 | 3f |
| A-27 | sigma/mode identity unification (SC): D-40 refined by quantity (argument vs resolved vs mode invariants); five invariants given one identity + one canonical clause; `excitation_support._validate_sigma_pair` deleted; `resolve_sigma_bounds` now validates `mode`; builder `float(None)` TypeError named; `NumericalConfig` field docstring corrected; OP-20 opened | active | `bstpp/config.py`, `bstpp/excitation_support.py`, `bstpp/polygon_mass.py` | ctor/set_window/held-out/build/direct-config | A-27; `refactor-patches/phase3f/rebaseline_record.md`; enumeration `refactor-patches/pre-3f-stabilization/probe_wp14e2_*.log`; RED `results/_a27_red_sigma_mode_rows.txt`; after-state `results/_a27_probe_after_*.txt` | **verified** (RED->GREEN by execution; line-traced reachability) | argument-type coercion asymmetry (OP-20) deliberately unchanged | WP2 for OP-20 | 3f |
| A-27 / enforcement | `test_lane_b_sigma_mode_error_identity_is_owner_invariant` (5 parametrized rows, one per invariant, comparing every owner to every other before pinning the canonical clause) | active | `tests/test_lane_b_config_matrix.py` | ctor/resolver/config/builder/`assert_polygon_mass_table_budget` | RED at `f70ac7d` for all five | **verified** (discriminating) | none | keep | 3f |
| A-27 / enforcement | `test_lane_b_polygon_default_max_sigma_without_crs_is_rejected` -- freezes the resolver-rejects / config-accepts asymmetry so making the config the front gate cannot silently turn it into an accept | active | `tests/test_lane_b_config_matrix.py` | resolver + direct config | this row | **verified** (freeze pin; passes both sides by design, non-discriminating) | none | keep | 3f |
| A-27 / enforcement | `test_lane_b_prepare_polygon_mass_table_rejects_none_min_sigma_by_name` and `test_lane_b_resolve_sigma_bounds_validates_mode` -- the two declared behavioural changes beyond identity | active | `bstpp/polygon_mass.py`, `bstpp/excitation_support.py` | builder / direct resolver call | RED at `f70ac7d` | **verified** (discriminating) | none | keep | 3f |
| A-27 / deletion | `excitation_support._validate_sigma_pair` removed; its two call sites and `polygon_mass.build_quad_table`'s hand-written copy route to `config.validate_sigma_pair` | active | `bstpp/excitation_support.py`, `bstpp/polygon_mass.py` | resolver + builder | line-traced: `config.validate_sigma_pair` REACHED on public paths after the change, NEVER EXECUTED before | **verified** | none | keep | 3f |
| A-27 / gate | Part I decision-row monotonicity check added to `results/_a25_content_checks.py` (check 3) | active | gate script | register build | demonstrated to discriminate: mutated copy with D-40 before D-39 exits 1 | **verified** | duplicates/gaps also covered | keep | 3f |
| D-36 | Phase blocker test (B1–B6; pinned-path corollary) | active | **process decision — no code site** | n/a | protocol §5; A-24; applied at `fe44de4` | process | none | keep | team |
| D-37 | Finite readiness rule (principle; instance in A-24) | active | **process decision — no code site** | n/a | protocol §9; readiness_report.md; A-24 | process | none | keep | team |
| D-38 | Lane B matrix standing gate (M on triggers; E with D-20) | active | `tests/test_lane_b_config_matrix.py`; `covering_array_rows.json` | ctor/set_window/trigger/cutoff/provenance | Lane B suite; §12 row | direct | OP-14–OP-17 gaps | keep | 3f |
| D-39 | Coverage recorded; absence of findings ≠ evidence; same-commit traceability update | active | `traceability_matrix.md`; `audit_coverage_map.md` | process | this file + coverage map | **no executable enforcement** (documented state) | enforcement is DOC/process until gated | keep | 3f |
| D-40 | One invariant, one error identity, one message, independent of entry path; a site may append remediation but not restate the invariant; raised messages are ASCII | active | **single source:** `bstpp/config.py` `panel_ratio_invariant_clause` / `raise_panel_ratio_violation`; delegating sites `polygon_mass.assert_polygon_mass_table_budget`, `polygon_mass.prepare_polygon_mass_table`; WP1.4b call-site guard in `Hawkes_Model.set_window` **removed** | ctor + set_window + held-out scoring + table build | `tests/test_lane_b_config_matrix.py::test_lane_b_panel_ratio_error_identity_is_entry_path_invariant` (entry-path invariance, RED-demonstrated); `tests/test_numerical_config.py::test_panel_ratio_clause_is_ascii_and_single_sourced`; clause equality in `tests/test_polygon_mass_table_install.py` and `tests/test_panel_min_sigma_guard.py`; ASCII sweep `results/_a26_ascii_sweep.py` | **direct** (measured both entry paths, `results/_a26_probe_entry_path_split.txt` before / `_a26_probe_after.txt` after) | sigma/mode invariants not yet delegated | **WP1.4e-2**, before WP2 | 3f |
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
| OP-14 | Closeout §6 P2 three-leg leg consistency at every Lane B covered point | open | `_exercise_covering_row` / `test_lane_b_covering_array_row_admissible`; forced-success identity checks in `tests/test_lane_b_config_matrix.py` | ctor/set_window | A-24 mapping (partial: support object identity only) | gap recorded | unaudited under D-39; **discharge:** standing Lane B completion (no D-40 dependency) | standing Lane B | 3f |
| OP-15 | Closeout §6 P3 provenance completeness (untouched-axis + save_rslts) | open | `test_lane_b_save_rslts_roundtrips_cutoff_provenance` (strict xfail G2); covering-array success provenance checks | mutator/I/O | A-24 mapping | gap recorded | unaudited under D-39; **discharge:** save_rslts half by G2; untouched-axis by Lane B hardening | 3f G2 + Lane B | 3f |
| OP-16 | Closeout §6 C5 `set_window` rejected as covering-array axis level | open | `test_lane_b_rejected_set_window_rolls_back_whole_state`; `test_lane_b_polygon_incompatible_spatial_change_rolls_back` (isolated, not CA-crossed) | set_window reject | A-24 mapping | gap recorded | unaudited under D-39; **discharge:** standing Lane B CA extension (no D-40 dependency) | standing Lane B | 3f |
| OP-17 | Closeout §6 P1 message-substring enforcement across matrix rejections | **resolved** (A-26) | bare `pytest.raises(ValueError)` (no `match=`) in `test_lane_b_rejected_set_window_rolls_back_whole_state` (parametrized `bad_call`) and stale-table install in `test_lane_b_polygon_incompatible_spatial_change_rolls_back` | set_window reject identity | A-24 regrade; enumerated from code | **closed** by A-26 / WP1.4e-1 | both sites carry `match=`; generalized so no test in the affected set uses a bare `pytest.raises` for these invariants | closed | — |
| OP-18 | Restated I10 under D-15: translation invariance holds exactly (bit-identical); invariance under a genuine change of units requires σ, spatial cutoff, and background normalization to transform together and carries a Jacobian from domain-area rescaling — that form is **not derived**; do not write it as a test until derived | open | process / future test design (no production code site yet) | n/a | WP1.4c three-transform table + archive I10 supersession (D-15 §5.9–5.10); live pins `test_spatial_similarity_covariance`, `test_spatial_kernel_family_is_real_unit_not_internal` | gap recorded | derivation open under D-39 | keep as OP | 3f |
| OP-19 | Held-out scoring validates a supplied mass table against **module default** budget policy, not the model's `NumericalConfig` | open | `Hawkes_Model._build_heldout_excitation_support` → `build_excitation_support(numerical_config=self.args[...])` is passed for the *training* path but held-out validation reads module constants for a directly-supplied table | held-out scoring | A-26 unified the error **identity** on this path; the **policy source** was deliberately left alone (changing it would change which `panel_h_m`/`gl_order` govern held-out validation for a non-default table — a separate observable change with its own pin question) | gap recorded | instance of the dual-source debt accepted through OP-8 | **WP5**, with the `ExcitationSupport` seam | 3f |
| OP-21 | `validate_polygon_mass_table(sigma_max=None)` still dies in `float(None)` with an unnamed `TypeError` (`polygon_mass.py:810`) | open | third site found by A-28's enumeration; measured at `426d60a`, `probe_wp14f_max_sigma_sites_BEFORE.log` | mass-table install validation | **not** a builder — it compares a *built* table's recorded range against the *model's resolved* bound, so σ-I6's clause would misdescribe it. Unreachable from every model path (`excitation_support.py:404` asserts both bounds non-`None`); reachable only by direct call | labelled, not assumed (A-27's treatment of `assert_polygon_mass_table_budget`) | deciding its clause means deciding whether a `NumericalConfig` holding `max_sigma=None` may reach install validation — **the frozen asymmetry A-27 declined to move**; unfreezing it inside a CF commit would be the sleight OP-20 refused | **WP5**, with OP-19 | 3f |
| Suite invocation (enforcement + declaration) | Default pytest collection = `tests/` via `pyproject.toml` `testpaths`; the register's gate profile names `pytest tests/` | **enforced** WP1.4c (`82e9c09`); **declared** A-25, which supersedes A-22's “Full suite” | `pyproject.toml` `[tool.pytest.ini_options]`; A-22 gate-profile paragraph | bare `pytest`; `pytest tests/` | collect-only **567→560** at `82e9c09^`→`82e9c09` (**Δ7** = `refactor-patches/phase0/test_identities.py` alone); re-measured from a clean worktree, `results/_wp14c_collect_{before,after,archive}.txt`, `results/_a25_collect_*.txt` | **verified** (clean tree; `git status --porcelain` empty in every capture) | closed — enforcement and declaration now agree | keep | — |
| Gate profile (IV/DOC) | Full profile when gate-read files change (`pyproject.toml` is gate-read); DOC-only only when no gate-read file is touched; **every captured gate run records `git status --porcelain` alongside its result** | active process | A-24 DOC-only amendment; A-25 invocation supersession + provenance requirement (both amend the A-22 gate-profile paragraph) | pins + suites + ruff; provenance capture on all of them | WP1.4c used the **full** profile; A-25 / C3 is the first commit under the DOC-only profile with the provenance requirement in force | process | — | keep | 3f |
| I1 | Sim/lik mass atoms | active | lik + _sim_cox | bg | seasonal_integral | direct | none | keep | — |
| I3/I4 | Pair window contract | active | utils, lik, sim | pairs | identities | direct | none | keep | — |
| I5 | Single factor site | active | inference_functions | model | test_smoke | direct | none | keep | — |
| I6 | Derived seasonal coordinate | active | `_scale_xyt` | prep | `tests/test_smoke.py::test_A_derivation` | **verified** EXIT:0 iter2 | none | keep | — |
| I8* | Special-case reductions | active | identities | model | test_identities | direct | none | keep | — |
| I9 | Refinement invariance | active | clipped support | prep | clipped tests | direct | none | keep | — |
| I10 | Unit covariance / real-unit spatial kernel (similarity with covariant σ; non-uniform rescaling at fixed σ changes loglik). Archive affine-at-fixed-σ loglik invariance **superseded by D-15** | active | identities + real-unit boundary | lik/sim | live: `test_spatial_similarity_covariance`, `test_spatial_kernel_family_is_real_unit_not_internal`; archive `test_spatial_affine_unit_invariance` historical only | **verified** live EXIT:0; archive deliberately red | OP-18 derivation | keep live; archive excluded | 3f |
| I11 | Conservation E[n]≈Λ | active | simulate + compensator | rect+polygon | rectangle standing test; polygon `test_polygon_i11_conservation` R=40 @ 3·se (`c5e4871`) | integration | — | standing test | pre-3f |
| I12 | Real-unit nonsquare | active | utils/trigger/sim | nonsquare | pins + identities | direct | none | keep | — |
| C6 | Guide identity | open | docs | n/a | A-21 | inferred | unresolved | 3g | 3g |

**Duplicated / missing / contradictory notes:**

- OP-1 missing (never allocated). Register inventory historically omitted I2/I7; WP1.4c live inventory finds both under `tests/` (seasonal_integral / smoke held-out pairs).  
- Part I OP-3/4 prose vs A-21: Part II governs (G8).  
- Untracked `phase3_baseline_and_decisions.tex` not governing.
- WP1.4c OP range opened in this series: **OP-18** only (I1–I11 all have live coverage).
- **Two `I`-numberings are in use and they collide.** The `I1`–`I12` rows in this
  matrix are the Phase-3 **model identities** (mass atoms, pair window, seasonal
  coordinate, …). A-27's `I1`–`I5` and A-28's `I6` are the **σ/mode invariants**
  under D-40, a separate and unrelated sequence — so matrix `I6` ("Derived
  seasonal coordinate") and σ-`I6` ("builder requires `max_sigma`") are different
  things with the same label, as are `I1`, `I3`, `I4`, `I5`. The collision was
  created at A-27 and is recorded at A-28 rather than deepened silently. The two
  sides are **not equally expensive to renumber**: the model identities are
  entrenched (guide, phase0 archive, `test_polygon_i11_conservation.py`, several
  amendments), while the σ labels are days old. Renumbering is a real option on
  the σ side only, deferred to its own commit rather than folded into a CF
  change — see A-29.
