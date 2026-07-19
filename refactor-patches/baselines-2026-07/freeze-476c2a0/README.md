# Final Phase 3 baseline freeze evidence

## Identities (do not conflate)

| Role | Value |
|---|---|
| **Behavioral baseline tested** | `476c2a044780ced66afae12760b53ac75e304fc6` |
| **Evidence-only commit** | *(this folder’s commit; docs freeze record only — see `git log -1` after commit)* |

The behavioral tip was **not** amended. This directory records machine-local freeze runs and §5 dispositions only.

## Execution environment

All suite and pin commands were executed in an **isolated detached git worktree**, not in the analysis working copy:

- Worktree path: `C:\Users\Terhi\Box\BSTPP_terhi\BSTPP-freeze-476c2a0-worktree`
- Worktree creation: `git worktree add --detach <path> 476c2a044780ced66afae12760b53ac75e304fc6`
- Worktree HEAD confirmed: `476c2a044780ced66afae12760b53ac75e304fc6`
- Worktree state at run: clean tracked tree, detached HEAD
- Reason: the analysis working copy has pre-existing **untracked** artifacts (including `refactor-patches/test_sbc_smoke_v3.py`) that must not affect pytest collection

### Analysis-machine working-copy state at freeze (not cleaned)

Repository: `C:\Users\Terhi\Box\BSTPP_terhi\BSTPP-for-illegal-dumping`  
Branch: `refactor`  
HEAD (behavioral tip): `476c2a044780ced66afae12760b53ac75e304fc6`  
Origin: `https://github.com/yuxinlg/BSTPP-for-illegal-dumping`

Tracked/staged checks before freeze work:

- `git diff --quiet` → exit 0
- `git diff --cached --quiet` → exit 0
- `git ls-files --modified` → empty

Working-copy classification: **dirty — untracked files only; tracked tree clean.**

Initial `git status --porcelain` (untracked preserved; not deleted, moved, stashed, edited, added, or committed by this freeze):

```
?? batch_park_fits.py
?? cbg-park-seasonal.ipynb
?? nuts_recover_log.txt
?? pins2d_2.json
?? pins2d_3.json
?? pins2d_4.json
?? pins2d_base.json
?? pins_1.json
?? pins_2.json
?? pins_3.json
?? pins_4.json
?? pins_5.json
?? pins_base.json
?? pins_final.json
?? pins_wt_2b.json
?? pins_wt_2c1.json
?? pins_wt_2c2.json
?? refactor-patches/commit_prompt_sbc_stage3.md
?? refactor-patches/sbc1/
?? refactor-patches/sbc2/
?? refactor-patches/sbc_runbook_stage3.md
?? refactor-patches/sbc_v3.py
?? refactor-patches/test_sbc_smoke_v3.py
?? results/_sbc_stage2_batch_driver.py
?? results/sbc_stage1_adaptive_chunk470.log
?? results/sbc_stage1_adaptive_chunk520.log
?? results/sbc_stage1_adaptive_chunk570.log
?? results/sbc_stage1_adaptive_chunk600.log
?? results/sbc_stage1_adaptive_probe420.log
?? results/sbc_stage1_adaptive_run.log
?? results/sbc_stage1_ess_pilot/
?? results/sbc_stage1_run.log
?? results/sbc_stage2_chunk100.log
?? results/sbc_stage2_chunk200.log
?? results/sbc_stage2_chunk300.log
?? results/sbc_stage2_chunk400.log
?? results/sbc_stage2_chunk500.log
?? results/sbc_stage2_chunk600.log
?? results/sbc_stage2_driver.log
?? results/sbc_stage3_run.log
```

(Also present later as untracked, not part of freeze commit: `docs/phase3_baseline_and_decisions.tex`.)

## A. Fresh full-suite run

| Field | Value |
|---|---|
| Exact command | `JAX_PLATFORM_NAME=cpu "C:/Users/Terhi/miniconda3/envs/illegal-dumping/python.exe" -m pytest tests/ -q` |
| CWD | detached worktree (above) |
| Conda env | `illegal-dumping` |
| Python | `C:\Users\Terhi\miniconda3\envs\illegal-dumping\python.exe` — 3.12.13 |
| HEAD | `476c2a044780ced66afae12760b53ac75e304fc6` |
| Start | 2026-07-19 19:44:15 -04:00 |
| End | 2026-07-19 19:49:50 -04:00 |
| Elapsed | ~325.68 s pytest / ~5 m 35 s wall |
| Exit code | 0 |
| Result | **68 passed**, 1 warning |
| Warning | `tests/test_ingestion_contract.py::test_geographic_crs_contract_warning_is_nonvacuous` — geopandas geographic CRS `area` UserWarning from `bstpp/main.py:196` |

Full terminal capture: `full-suite-output.txt`.

## B. Golden-pin re-verification

| Field | Value |
|---|---|
| Exact command | `JAX_PLATFORM_NAME=cpu "C:/Users/Terhi/miniconda3/envs/illegal-dumping/python.exe" refactor-patches/pin_check_v2.py .` |
| Compared to | committed `refactor-patches/baselines-2026-07/pins.json` (**not** regenerated or rewritten) |
| `pins.json` SHA-256 | `F2141FD558704057C87C8C20A3F2B9516C8815F232B6EAAFC8F71BAA0864E7EB` |
| HEAD | `476c2a044780ced66afae12760b53ac75e304fc6` |
| Timestamp (end) | 2026-07-19 19:50:58 -04:00 |
| Exit code | 0 |
| Result | **PASS — bit-identical** JSON values for all four configs (`cox_hawkes`, `hawkes`, `hawkes_nonsquare_4to1`, `lgcp`), including `loglik` and all `grad_*` entries |

Full capture (metadata + fresh stdout): `pin-verify-output.txt`.

## C. Pipeline repository snapshot

Target remote: `https://github.com/yuxinlg/Illegal-Dumping`.

**Result: no local git clone of `yuxinlg/Illegal-Dumping` was found on this analysis machine** (searched user home / Box / Documents / Desktop / Downloads / OneDrive / dssg for `.git/config` containing that origin; zero hits).

Closest analysis working copies used by notebooks (layout matches the remote’s `code/` + nested BSTPP pattern) are **not git repositories**:

| Path | Git? | Notes |
|---|---|---|
| `C:\Users\Terhi\Box\terhi-illegal-dumping-Jan15-2026` | **no `.git`** | Active analysis tree (`code/`, `data/`, `output/`, nested `BSTPP/`) |
| `C:\Users\Terhi\Box\illegal-dumping-Jan15-2026` | **no `.git`** | Sibling copy |

Nested BSTPP inside those trees is a **different** clone (`origin https://github.com/imanring/BSTPP.git`, branch `main`, dirty) and is **not** the pipeline repository.

Pipeline FILL disposition: record **absent local clone**; analysis trees are dirty/non-git working copies preserved as-is (not repaired).

## D. Analysis-machine environment

| Component | Value |
|---|---|
| Conda env | `illegal-dumping` |
| Python | 3.12.13 |
| JAX | 0.4.23 (backend `cpu`) |
| NumPyro | 0.15.0 |
| NumPy | 1.26.4 |
| geopandas | 1.1.3 |
| shapely | 2.1.2 |
| `jax_enable_x64` | **False** |

Committed exports:

- `conda-env-illegal-dumping.yml` — `conda env export -n illegal-dumping --no-builds` with only the machine-specific `prefix:` line removed
- `pip-freeze.txt` — `python -m pip freeze`
- `env_versions.txt` — runtime version probe

No packages were installed, removed, or upgraded.

## E. §5 current-behavior anchors

See `section5-evidence-table.md` for all 18 substantive placeholders (§5.1–§5.7, §5.14, §5.15).

No raw `[[FILL]]` remains without either tip evidence or an explicit non-blocking demotion (3a/3c).

**Doc inconsistency:** §1.2 item 5 / reviewer-blocker text omit §5.15 from the parenthetical list even though §5.15 carries a FILL; this freeze audited §5.15 anyway.

## Contents of this directory

- `README.md` — this freeze record
- `full-suite-output.txt`
- `pin-verify-output.txt`
- `conda-env-illegal-dumping.yml`
- `pip-freeze.txt`
- `env_versions.txt`
- `section5-evidence-table.md`

## Non-goals confirmed

- No Phase 3 implementation
- No package / test / pin / decoder / result-record changes
- Phase 3 LaTeX document not edited in this task (paste-ready FILL text returned in the freeze chat response)
- No push
