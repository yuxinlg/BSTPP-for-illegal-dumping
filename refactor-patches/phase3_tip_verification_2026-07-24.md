# Tip verification — Phase 3 audit follow-up through reaudit strengthenings

**Current production-code tip for gates:** `2cf326d` on branch `refactor`.

Env: conda `illegal-dumping`, `JAX_PLATFORM_NAME=cpu`, `MPLBACKEND=Agg`.

- python 3.12.13; jax 0.4.23 (`jax_enable_x64=False`); numpyro 0.15.0
- numpy 1.26.4; scipy 1.11.4; geopandas 1.1.3

This note supersedes earlier tip snapshots (`05da465`, `8580364` / docs
`a7c0bd7`, `0eeaebd`) for **current** acceptance status. Prior tips remain
historical evidence for their SHAs. Full durable commands for the current
gates: `refactor-patches/reaudit_verification_2cf326d.md`.

## Ownership summary

| Phase | Ownership |
|---|---|
| **3a** | `spatial_cov_crs` for tabular covariates; missing/mismatched CRS rejection; unchanged GDF/array paths; `set_crs` (no deprecated CRS attribute writes); loud invariant if mismatch reaches attach |
| **3c** | Canonical `PreparedDomain.union_geometry`; union-consistent area; BP for disjoint regimes; SC for overlapping rows |
| **3d** | Exact Gaussian restriction; `PolygonMassTable` validation; NPZ/sidecar self-consistency on load; `prepare_polygon_mass_table`; hard-require table (**API**, not BP); NumPy/SciPy float64 build; schema **v2** compat metadata; nested `extra_provenance`; exact le-f64 event identity + `xy_events_real` snapshot; early missing-table preflight; held-out + transactional replacement; shootout/wide-range |
| **3e** | Every tol validated; transactional `set_window`; temporal-only polygon table reuse; spatial change requires `mass_table=` |

Phase 3f and Stage 3 R=200 exit: **not started**.

## Commit series since `e706107` (remaining reaudit)

| Commit | Class | Content |
|---|---|---|
| `34f6db6` | test (RED) | NPZ/sidecar tamper tests |
| `90d45bb` | fix | NPZ/sidecar consistency on load |
| `513cb1a` | test (post-hoc) | Malformed compat strings + float64 snapshot regression |
| `f6269a5` | test (post-hoc) | Focused CRS coverage for `e706107` |
| `b7d12f8` | test (RED) | Attach must not silently override CRS mismatch |
| `2cf326d` | fix | Loud CRS-mismatch invariant (no `allow_override`) |
| `dfc342a` | chore | `git diff --check` whitespace |
| *(docs tip)* | docs | This update + `reaudit_verification_2cf326d.md` |

Earlier series through `0eeaebd` / `e706107` remains on branch history.
`e706107` changed CRS handling **after** the recorded `0eeaebd` 299-test
gate and is covered by the current record.

## Durable tip gates at `2cf326d`

Exact commands and outputs:
`refactor-patches/reaudit_verification_2cf326d.md`.

| Gate | Result |
|---|---|
| Sidecar/compat + float64 snapshot | **53 passed** |
| Focused CRS (`e706107`) | **7 passed** |
| Polygon + 3d/3e + shootout + wide-range | **150 passed** |
| Confirmation selectors | **9 passed** |
| Full suite | **338 passed, 1 warning** |
| Collect-only | **338 tests** |
| Pins vs machine-local `pins_wt_2c2.json` | **PIN_DIFFS 0 MATCH** |
| `ruff check bstpp` | clean |
| `ruff check bstpp tests` | **133** (127 E402, 3 E702, 3 F401) |
| `jax_enable_x64` | unchanged (prepare-api selectors) |

## Conditional SBC

Pins MATCH; no confirmation anomaly → no Stage 1/2 conditional SBC.
Phase 3f not started. Not pushed pending approval.
