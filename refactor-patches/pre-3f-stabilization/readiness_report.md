# NOT READY FOR 3F

**Candidate tip:** `cd62288e8313a1bad7f90aa76068fe1738d9f2b6` (`refactor`, tracks `origin/refactor`)  
**Dirty tracked tree:** clean (untracked noise only)  
**Environment:** Python 3.12.13 / jax 0.4.23 / numpyro 0.15.0 / numpy 1.26.4 / scipy 1.11.4 / geopandas 1.1.3 / `jax_enable_x64=False` / cpu  

This verdict is against the preregistered §9 entry criterion and the coverage declared in `audit_coverage_map.md`. It does **not** claim the package is defect-free.

## §9 conditions

| Condition | Status |
|---|---|
| No open blocker under §5 | **MET** for iteration-1 blockers — **B1 closed** (`9e37dc3` RED → `13a8525` GREEN). No new blockers from Lane B matrix. |
| Every preceding finding has exact closeout record | **PARTIAL** — B1/G4/G9 closed; class-level gaps G1–G3/G5–G8 open with owners |
| Every blocker repair has focused RED→GREEN regression with correct class | **MET** for B1 |
| Lane B matrix passes | **MET** — `tests/test_lane_b_config_matrix.py` 27 passed + G2 xfail; full suite EXIT:0 |
| §8 gates pass on one frozen candidate tip | **UNMET** — iteration-2 / §8 battery not yet run on post-B1 tip |
| Decisions needed for 3f config/sequencing settled and noncontradictory | **MET** for start decisions (OP-3/4/10/13 settled in A-21+code; OP-7 subsumed; OP-8/12 deferred per A-21). Historical Part I prose disposed as non-operative under amendment rules (G8). |
| Every residual gap has owner, rationale, review date | **MET** — see `findings_ledger.md` |
| Coverage map names unaudited areas | **MET** — `audit_coverage_map.md` |

## Unmet conditions only — repair sequence

Smallest **class-separated** sequence (do not mix into the audit commit):

### Commit 1 — test (RED): CRS-less / non-default builder install trap
- **Class:** test (RED first)  
- **Add:** `tests/test_polygon_builder_settings_install.py` (names indicative)  
  - `test_crs_less_small_min_sigma_guided_panel_installs` — default prepare raises panel-ratio; prepare with `panel_h_m <= MAX_PANEL_TO_MIN_SIGMA_RATIO * min_sigma`; `Hawkes_Model(..., excitation_support="polygon", mass_table=table, matching builder kwargs or equivalent)` must construct (today: fails).  
  - `test_nondefault_gl_order_installs_when_declared` — `gl_order=8` prepare + matching install succeeds.  
  - `test_mismatched_gl_order_still_rejects` — prepare 8, validate/install expecting 16 → named `ValueError` substring.  
- **Gates:** targeted pytest only (expect RED on tip).

### Commit 2 — CF+API: single-source builder settings on install
- **Class:** CF + API  
- **Change:** Plumb `panel_h_m` / `gl_order` from public construction (constructor kwargs now; `NumericalConfig` in 3f) through every `build_excitation_support` / held-out / `set_window` mass-table validation call. Defaults remain package defaults. Do **not** silently rebuild tables. Error messages that recommend a smaller `panel_h_m` must be satisfiable by the same settings on install.  
- **Gates:** RED tests → GREEN; `tests/test_polygon_mass_*`, `test_heldout_polygon_mass`, `test_phase3d_excitation_support`, `test_panel_min_sigma_guard`; full suite; four-config pins expect `PIN_DIFFS 0 MATCH`; ruff on touch; confirm `jax_enable_x64` unchanged.

### Commit 3 — test: Lane B matrix module (minimal forced rows)
- **Class:** test  
- **Add:** executable pairwise/forced-row module covering family×support×cutoff entry×sentinel outcomes named in `traceability_matrix.md` (success + named rejects; provenance identity; rollback snapshot).  
- **Gates:** new module green; full suite.

### Commit 4 — DOC (optional, separable): pin identity hygiene + Part II pointer
- **Class:** DOC  
- Record tip SHA/env on pin verify logs; optional one-line Part I→A-21 reader note for OP-3/4.  
- **Gates:** none numeric.

**Not in this pre-READY sequence (owned elsewhere):** G1 membership consolidation (3f); G2 `save_rslts` provenance (3f); G3 distribution property suite (3g); G5 polygon conservation (3g / pre-polygon-SBC); G7 OP-8/12/C6 per A-21 schedule; Stage-3 tip SBC exit (Phase 3 exit, not 3f start).

After commits 1–3 land: re-audit **only** B1 surfaces + Lane B coverage invalidated by the repair (protocol iteration 2), then run §8 on the new candidate tip.
