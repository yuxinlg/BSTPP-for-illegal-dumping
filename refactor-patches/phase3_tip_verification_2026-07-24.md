# Tip verification — Phase 3 audit follow-up through A/B (2026-07-24)

Tip at verification: `8580364` on branch `refactor`
(`85803644f4244eac53593ebc3d9a2f988f4eb6da`).

Env: conda `illegal-dumping`, `JAX_PLATFORM_NAME=cpu`, `MPLBACKEND=Agg`.

- python 3.12.13; jax 0.4.23 (`jax_enable_x64=False`); numpyro 0.15.0
- numpy 1.26.4; scipy 1.11.4; geopandas 1.1.3

This note supersedes the earlier tip snapshot at `05da465` for **current**
acceptance status. Historical repair evidence at `05da465` remains valid for
that tip; it is not re-described as evidence of the A/B contracts below.

## Ownership summary (what belongs where)

| Phase | Ownership recorded here |
|---|---|
| **3a** | Explicit `spatial_cov_crs` for tabular covariates; missing/mismatched CRS rejection; unchanged GeoDataFrame and array-domain paths. Earlier leftovers: finite horizon; polygonal domain; GeoDataFrame missing CRS. |
| **3c** | Canonical `PreparedDomain.union_geometry` plumbing; union-consistent area semantics; equals/WKB/hash characterization; **BP** for disjoint/current valid regimes; **SC** for positively overlapping domain rows. |
| **3d** | Exact built-in Gaussian restriction; `PolygonMassTable` compatibility validation; explicit `prepare_polygon_mass_table`; **API** hard-require prepared table at polygon construction (not BP); NumPy/SciPy float64 build without `jax.config.update`; held-out correction; transactional replacement-table path; shootout/wide-range/conservation/both-mode evidence. |
| **3e** | Validate every supplied tolerance before precedence; atomic cutoff/provenance on `set_window`; transactional failure behavior; temporal-only polygon window reuse of installed table; spatial-window changes require compatible `mass_table=`. |

Phase 3f and Stage 3 R=200 exit rerun: **not started**.

## Commit series since `origin/refactor` (`0648659`)

Honesty: distinguish original phase features (already on `origin/refactor`),
later audit discoveries, genuine RED→GREEN repairs, API changes, and this
documentation-only commit.

### Cutoff / transactional `set_window` (Phase 3e repairs)

| Commit | Class | Content |
|---|---|---|
| `8a22fd6` | test (RED) | Reject invalid shared `cutoff_tol` even when axis-specific override wins |
| `2a7c558` | fix (IV) | Validate every raw non-`None` tolerance before precedence |
| `33037f3` | test (RED) | `set_window` must leave observable state unchanged on validation/rebuild failure |
| `798f551` | fix (SC) | Validate → prepare pairs/support/provenance locally → assign once |

### Canonical domain plumbing (Phase 3c)

| Commit | Class | Content |
|---|---|---|
| `2c7264a` | test | Characterize `PreparedDomain.union_geometry` equals/WKB/hash identity |
| `2a67962` | refactor (BP/SC) | Partitions and excitation consume `union_geometry` (no downstream recompute) |

### CRS tabular covariates (Phase 3a)

| Commit | Class | Content |
|---|---|---|
| `5cd9355` | test (RED) | Tabular covariates require `spatial_cov_crs` when domain has CRS |
| `adea3d3` | fix (IV/API) | Parse `spatial_cov_crs` via `CRS.from_user_input`; never infer domain CRS |

### Polygon hard-require preparation (Phase 3d; **API change**)

| Commit | Class | Content |
|---|---|---|
| `9342a19` | test (RED) | Polygon ctor requires prepared table; build must not toggle `jax_enable_x64` |
| `ae27947` | fix (**API**, not BP) | `prepare_polygon_mass_table` (NumPy/SciPy float64); ctor hard-requires table |
| `a8ec985` | test (RED) | `set_window` requires `mass_table=` on spatial-window change; transactional failure |
| `8580364` | fix (API/SC) | Transactional `set_window(..., mass_table=)`; no silent rebuild |

### Pre-`8580364` intermediate failure (not a tip failure)

An intermediate polygon-suite run after the first `set_window` GREEN attempt
failed with `AttributeError: 'Hawkes_Model' object has no attribute
'excitation_support'` (21 failed / 58 passed). That failure is **evidence of
the constructor/`set_window` defect before `8580364`**, not a final-tip
failure. Tip `8580364` re-verified green (below).

## Durable tip gates at `8580364`

| Gate | Result | Exact command / artifact |
|---|---|---|
| Targeted polygon + 3d + 3e + shootout + wide-range | **96 passed** (~85s) | `JAX_PLATFORM_NAME=cpu python -m pytest tests/test_polygon_mass_prepare_api.py tests/test_polygon_mass_table_validation.py tests/test_heldout_polygon_mass.py tests/test_polygon_mass_backend_shootout.py tests/test_polygon_mass_wide_range.py tests/test_phase3d_excitation_support.py tests/test_phase3e_cutoffs.py -q --tb=line` |
| Full suite | **277 passed**, 3 warnings (~623s) | `JAX_PLATFORM_NAME=cpu python -m pytest tests/ -q --tb=line` → `results/_full_suite_ab.txt` |
| Collect-only | **277 tests** | `JAX_PLATFORM_NAME=cpu python -m pytest tests/ --collect-only -q` |
| Four-config pins | **PIN_DIFFS 0 / MATCH** vs `pins_wt_2c2.json` | `refactor-patches/pin_check_v2.py` (candidate also saved as `results/_pins_ab_candidate.json`) |
| `ruff check bstpp` | clean | — |
| `ruff check bstpp tests` | **133** findings: **127 E402**, **3 E702**, **3 F401** — all inherited | — |
| §12 smoke/confirmation | all selectors green | `refactor-patches/confirmation_8580364.md` |

### Warnings at full suite (non-blocking)

1. Two CRS tests emit pandas `DeprecationWarning` about overriding CRS via
   attribute assignment (`test_tabular_cov_matching_spatial_cov_crs_accepted`,
   `test_geodataframe_cov_path_unchanged_without_spatial_cov_crs`).
2. Existing geographic-CRS area warning in
   `tests/test_ingestion_contract.py::test_geographic_crs_contract_warning_is_nonvacuous`.

### `jax_enable_x64`

Confirmed unchanged across preparation and construction by
`tests/test_polygon_mass_prepare_api.py::test_quad_table_build_does_not_toggle_jax_enable_x64`
and
`::test_prepare_polygon_mass_table_and_ctor_install_without_x64_toggle`
(**2 passed**; see confirmation record).

## Conditional SBC

Unchanged-regime pins match; confirmation commands show no anomaly → **no**
Stage 1/2 conditional SBC rerun for this tip. Stage 3 R=200 exit remains a
Phase 3-tip obligation (not an A/B obligation). Phase 3f not started.

## Phase records

- `refactor-patches/phase3a/rebaseline_record.md`
- `refactor-patches/phase3c/rebaseline_record.md`
- `refactor-patches/phase3d/rebaseline_record.md`
- `refactor-patches/phase3e/rebaseline_record.md`
- `refactor-patches/confirmation_8580364.md`
