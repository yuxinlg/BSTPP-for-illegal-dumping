# Findings ledger

**Candidate tip:** `cd62288e8313a1bad7f90aa76068fe1738d9f2b6`  
**Audit date:** 2026-08-03

| ID | Finding or gap | Contract IDs | Evidence (mark + command/artifact) | Production reachability | Pin or frozen surface? | Change class | Severity | Class-level remediation | Required gates | Owner | Review date | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | `build_excitation_support` validated supplied mass tables against `DEFAULT_PANEL_H_M`/`DEFAULT_GL_ORDER`. CRS-less / small-`min_sigma`: guided prepare succeeded; install rejected; coarse default-panel tables were accepted (no budget check). | D-25, D-26, A-21 | **verified** RED at `705d040`/`9e37dc3` (7 fail); GREEN after CF | polygon ctor / held-out / set_window | freeze surface; not pin path | CF | **BLOCKER** (resolved) | table `h_panel`/`gl_order` authoritative; `assert_polygon_mass_table_budget` vs `PRODUCTION_TAU_ABS` / panel-ratio gate; one `validate_polygon_mass_table` path | B1 suite + polygon/heldout/phase3d + full suite + pins MATCH | pre-3f | 2026-08-03 | **closed** — RED `9e37dc3`; GREEN `13a8525` |
| G1 | Domain membership expressed twice: vectorized `validate_events` (`shapely.covers(union, pts)`) vs scalar `PreparedDomain.covers_xy` / sim filter | D-4, D-30, A-21 deferred | **verified** probe agree on overlap/boundary/out; A-21 records consolidation for 3f | validate + simulate paths | freeze surface (PreparedDomain API) but results agree on probe | DOC / 3f | NONBLOCKING-3F | structural single-source `covers_xy(array_x, array_y)` used by validation and sim | membership + simulate union/rect suites | 3f | 2026-08-03 | open (owned) |
| G2 | `save_rslts` pickles only inference samples/results — not cutoff/numerical/decoder/mass-table provenance | A-21 binding 3f list; D-24 provenance | **verified** `main.py:544–559` source read | after fit | freeze surface (results I/O) | API | NONBLOCKING-3F | implement A-21 save/load contract with hard-fail incompat | new round-trip tests + load incompat | 3f | 2026-08-03 | open (owned) |
| G3 | Class-level distribution property suite absent; TruncatedLogNormal instance tests strong but not parametrized over every package-defined distribution | D-28; Lane C table | **verified** `test_truncated_lognormal.py` green in focused run; only TLN (+ truncate adapter) covered | polygon prior truncation path | not pin path | CF (if new defects) / test | NONBLOCKING-3G | parametrized properties: shape/dtype/support/log_prob/normalization/transform/JIT | dist property module | 3g | 2026-08-03 | open |
| G4 | Lane B compatibility matrix not implemented as an executable gate; pairwise gaps remain | protocol §6/§9 | **verified** closed by `tests/test_lane_b_config_matrix.py` (27 passed + G2 xfail); full suite 496 passed / 2 skipped / 1 xfailed / EXIT:0 | config state machine | freeze surface | test | NONBLOCKING-3F until matrix exists; **blocks READY** under §9 even after B1 | add specified matrix module | Lane B gate | pre-3f / 3f | 2026-08-03 | **closed** — Commit 3 (`test_lane_b_config_matrix.py`); residual G2 save path remains xfail |
| G5 | Polygon-regime excitation conservation (I11 analogue) has no standing test | I11; Lane D | **verified** rectangle tests exist in `test_identities.py`; no polygon counterpart found | polygon Hawkes simulate/compensator | not pin | test | NONBLOCKING-3G | add polygon conservation or structural reduction | identities + polygon smoke | 3g / pre-polygon-SBC | 2026-08-03 | open |
| G6 | Pin JSON lacks commit/env/hash; baselines README commit `a5b91d5` predates tip — values still MATCH | §5.3/§5.6 hygiene | **verified** candidate compare PIN_DIFFS 0 MATCH | rectangle gate | pin gate | DOC | NONBLOCKING-3G | stamp identity into pin meta or verify log; rebaseline only if values move | pin_check compare | 3g | 2026-08-03 | open |
| G7 | OP-8 (`args` removal), OP-12 (derivative gate), C6 (guide identity) remain open by register | OP-8, OP-12, C6; A-21 | **reported** A-21 + **verified** code still uses `args`; `TAU_DERIV` provisional | 3f / docs / polygon QA | decisions: OP-8/13 sequencing settled enough to start 3f per A-21; OP-12 not needed to start | DOC | DEFERRED (A-21: not start-blockers) | resolve per A-21 schedule | as scheduled | 3f / 3g / pre-polygon-SBC | 2026-08-03 | open (owned) |
| G8 | Part I / A-5 historical prose still states OP-3/4 open and legacy count-weighted default | OP-3, OP-4, A-21 | **verified** code default `None` + `test_standardization_api`; A-21 settles | standardize path | document vs code | DOC | NONBLOCKING-3G | readers must use Part II amendments; optional editorial pointer | doc review | 3g | 2026-08-03 | dispositioned |
| G9 | `test_wheel_requires_dist_contains_critical_runtime_pins` failed once in batched focused run, passed in isolation | packaging API | **verified** fail-then-pass 2026-08-03; cause: shared repo `build/bdist` Permission denied under concurrent/Box load. Fix: isolate build-base/egg-base to `%TEMP%` (`test_packaging_runtime_metadata.py`). Gate: three consecutive full suites with `--ignore=tests/test_polygon_mass_table_install.py` → 460 passed / EXIT:0 each (see `commit0_g9_gate.md`) | packaging | gate validity flake | test hygiene | NONBLOCKING-3G | isolate build dir (done) | packaging + full suite ×3 | pre-3f | 2026-08-03 | closed (Commit 0) |
| C-close | Prior §7 instances re-established | see coverage | focused suite: 106 passed / 1 skipped / 1 flake fail (G9); instance sites listed in coverage | various | mixed | n/a | n/a | class gaps G1–G5 | per row | — | 2026-08-03 | instances mostly fixed; classes open |

## §7 instance closeout summary

| Finding | Instance status | Class-level test | Pin predates correction? |
|---|---|---|---|
| TruncatedLogNormal.sample / `_log_z` | **verified fixed** (`excitation_support.py:213–220`, tests green) | absent (G3) | pins don’t cover; N/A |
| Union vs row-sum (D-30) | **verified fixed** (prepare_domain, sim filter, hawkes bg) | partial (domain/sim tests; not all geometry consumers) | N/A |
| Held-out mass rebuild | **verified fixed** (heldout suite) | path-specific only | N/A |
| set_window untouched provenance | **verified fixed** | mutator matrix incomplete | N/A |
| panel_h_m vs tau_abs | tau_abs **verified** `1e-5`; panel guard exists; **install trap B1** | missing round-trip | N/A |
| Membership duplication | correct but duplicated (G1) | structural single-source absent | N/A |
| Trigger capability gates | **verified fixed** (suite green) | Lane B matrix incomplete | N/A |
| `_UNSET` / explicit None | **verified fixed** | same | N/A |
| Non-default panel/gl rejection | **B1 open** | specify tests above | N/A |
| Design scales on instance | **verified** present for cutoffs | save path G2 | N/A |
| Provenance via save_rslts | **gap G2** | absent | N/A |
| Dependency metadata | **verified** pins in runtime requirements; suite flake G9 | packaging tests exist | N/A |
| Supported fit init failures | historical items fixed; **B1 remains** for guided CRS-less polygon | B1 tests | N/A |
