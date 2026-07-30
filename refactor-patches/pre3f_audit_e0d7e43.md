# Pre-3f local correctness/API pass — audit of tip `e0d7e437`

**Audited base:** `e0d7e43750ecfb0413dafb4603cc486448efc487`  
**Final implementation tip (pre-documentation):** `c501283`  
**Branch:** `refactor` (local only; nothing pushed)  
**Date:** 2026-07-30  
**Classification:** CF + API + IV/API + reproducibility/documentation (no new scientific semantics)

Phase 3f architectural restructuring was **not** started. Final SBC obligations were **not** run.

---

## Findings and disposition

| # | Finding | Disposition | Class |
|---|---------|-------------|-------|
| 1 | Kernel-specific cutoff/scale APIs applied to unsupported triggers | **Fixed now** | CF + API |
| 2 | `set_window` rewrote untouched-axis provenance as physical | **Fixed now** | CF + API |
| 3 | LGCP inherited `set_window` order-dependent / can KeyError | **Fixed now** | CF + API |
| 4 | Production polygon panel may not resolve `min_sigma` (CRS-less severe) | **Fixed now** | CF + IV/API |
| 5 | `get_grid_post_mean` plain Hawkes depended on plot-mutated `post_mean` | **Fixed now** | CF |
| 6 | `dist_euclid` 1-D `z` reshape from `x`; `exp_sq_kernel` rectangular noise | **Fixed now** | CF |
| 7 | Short MCMC fails in `print_summary` after successful sampling | **Fixed now** | CF + API |
| 8 | `setup.py` install_requires permitted incompatible stacks | **Fixed now** | API + repro/docs |
| 9 | Stale comments / `phase3_baseline_and_decisions.tex` live refs | **Fixed now** | DOC |
| 10a | Membership predicate duplication (`validate_events` vs sim filter) | **Recorded for 3f** | DOC / 3f |
| 10b | Extreme float32 TruncatedLogNormal / TruncatedNormal edge behavior | **Recorded numerical limitation** | DOC |

No item was classified as a new scientific semantics change.

---

## RED / GREEN evidence (per defect)

### 1. Kernel capability gates
- **RED commit:** `af7c934` — `tests/test_kernel_capability_gates.py`  
  Command: `pytest tests/test_kernel_capability_gates.py -v`  
  Result: **8 failed, 3 passed** (DID NOT RAISE / missing gates / mandatory `sigmax_2` for custom spatial).
- **GREEN commit:** `317ac62` — exact-type gates in `Hawkes_Model` + trigger docs  
  Result: capability + excitation + cutoff suites green.

### 2. Untouched-axis provenance
- **RED:** `12e1e78` — `tests/test_set_window_untouched_provenance.py`  
  Result: **3 failed, 5 passed** (tolerance / default_untruncated rewritten as physical).
- **GREEN:** `921b76d` — per-axis local records; `_UNSET` keeps verbatim  
  Result: untouched-provenance + sentinel + Phase 3e suites green (62 passed in combined run).

### 3. LGCP `set_window`
- **RED:** `0ee14e7` — `tests/test_lgcp_set_window.py`  
  Result: **3 failed** (DID NOT RAISE `NotImplementedError`).
- **GREEN:** `9e0328b` — `LGCP_Model.set_window` raises clear `NotImplementedError`.

### 4. Panel / `min_sigma` guard
- **RED:** `23a51b7` — `tests/test_panel_min_sigma_guard.py`  
  Result: default CRS-less panel **DID NOT RAISE**; valid small panel already met `PRODUCTION_TAU_ABS`.
- **GREEN:** `54eb20d` — `MAX_PANEL_TO_MIN_SIGMA_RATIO = 8.0` in `prepare_polygon_mass_table`  
  `PRODUCTION_TAU_ABS` unchanged at `1e-5`; `jax_enable_x64` unchanged.

### 5. `get_grid_post_mean` plain Hawkes
- **RED:** `2769877` — KeyError on missing `spatial_cov['post_mean']`.
- **GREEN:** `9fa85e1` — derive from `b_0` or `w @ X_s`; local area-weighted refinement; no mutation.

### 6. Kernel helpers
- **RED:** `29e480a` — wrong shape / rectangular broadcast failure / cross diagonal noise.
- **GREEN:** `9d35efc` — reshape `z` from `z`; diagonal noise only when `x is z`.

### 7. Short MCMC summary
- **RED:** `e18147a` — NumPyro diagnostics `AssertionError` after sampling (`num_samples=2`).
- **GREEN:** `cdc88b7` — skip summary with `UserWarning` when retained draws `< 4`.

### 8. Runtime dependency source
- **RED:** `298b588` — wheel metadata still had `geopandas>=0.14.0`, unpinned numpy/scipy, `numpyro>=0.10.0`.
- **GREEN:** `d887ad8` — `requirements-runtime.txt` + `setup.py` reader + `MANIFEST.in`  
  Disposable venv install/import of wheel: **PASSED** (full resolver path succeeded on this machine).

### 9. Stale docs / identity
- Commit `1d6beae` — comment/docstring retargets; live refs → root `phase3_record.tex`.  
  Historical `docs/phase3_baseline_and_decisions.tex` deletion staged with documentation tip.  
  Does **not** claim resolution of guide-identity conflict C6.

### 10. Deferred observations
Recorded in A-21 continuation and below; **not** implemented in this pass.

---

## Final verification (measured)

| Check | Result |
|-------|--------|
| New focused regressions | 34 passed |
| Trigger + excitation | 21 passed |
| Phase 3e + set_window sentinel | 54 passed |
| Polygon prepare/compat/production/shootout | 86 passed |
| Smoke / fit / LGCP sim | 14 passed |
| Full suite | **461 passed, 1 skipped, 2 warnings** |
| Golden pins vs `refactor-patches/baselines-2026-07/pins.json` | **`PIN_DIFFS 0 MATCH`** |
| `PRODUCTION_TAU_ABS` | `1e-5` (unchanged) |
| `jax_enable_x64` before/after | **False / False** |
| `compileall` on touched modules | clean |
| Ruff on touched Python files | clean (after F401 chore) |
| Repo-wide Ruff | **126 errors** (legacy baseline; not cleaned) |
| `git diff --check` | clean |
| Wheel build + Requires-Dist | critical pins present |
| Disposable venv `import bstpp` | passed |
| Phase 3f started? | **No** |
| Pushed? | **No** |

---

## API-visible behavior changes

- Unsupported triggers: loud early `TypeError` for exponential/Gaussian-only cutoff kwargs; custom rectangle spatial triggers no longer require `sigmax_2`.
- `set_window`: omitted axis provenance preserved verbatim.
- `LGCP_Model.set_window`: always `NotImplementedError`.
- `prepare_polygon_mass_table`: may reject coarse `panel_h_m` vs `min_sigma`.
- `get_grid_post_mean(include_cov=True)` on plain Hawkes works without prior plotting.
- Short `run_mcmc`: warning instead of post-sample crash when draws `< 4`.
- Packaging: tighter `install_requires` (notably `numpyro==0.15.0`, `geopandas>=1.0`, numpy/scipy upper bounds).

---

## Remaining recorded obligations

1. **Phase 3f:** consolidate domain membership into one vectorized `PreparedDomain.covers_xy(array_x, array_y)` used by validation and simulation.
2. **Later numerical:** extreme float32 TruncatedLogNormal / TruncatedNormal edge cases (see A-21) — no ad hoc clipping in this pass.
3. **OP-12** remains open (derivative gate).
4. Guide-identity conflict **C6** remains open (duplicate-doc removal does not resolve it).
5. Final SBC obligations remain outstanding (explicitly out of scope here).
6. `pdflatex phase3_record.tex` still blocked on this machine’s MiKTeX L3 toolchain.
