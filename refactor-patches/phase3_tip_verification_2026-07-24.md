# Tip verification — Phase 3 audit follow-up through reaudit strengthenings

Tip at verification: `0eeaebd` on branch `refactor`.

Env: conda `illegal-dumping`, `JAX_PLATFORM_NAME=cpu`, `MPLBACKEND=Agg`.

- python 3.12.13; jax 0.4.23 (`jax_enable_x64=False`); numpyro 0.15.0
- numpy 1.26.4; scipy 1.11.4; geopandas 1.1.3

This note supersedes earlier tip snapshots (`05da465`, `8580364` / docs
`a7c0bd7`) for **current** acceptance status. Prior tips remain historical
evidence for their SHAs.

## Ownership summary

| Phase | Ownership |
|---|---|
| **3a** | `spatial_cov_crs` for tabular covariates; missing/mismatched CRS rejection; unchanged GDF/array paths |
| **3c** | Canonical `PreparedDomain.union_geometry`; union-consistent area; BP for disjoint regimes; SC for overlapping rows |
| **3d** | Exact Gaussian restriction; `PolygonMassTable` validation; `prepare_polygon_mass_table`; hard-require table (**API**, not BP); NumPy/SciPy float64 build; schema **v2** compat metadata; nested `extra_provenance`; exact le-f64 event identity; early missing-table preflight; held-out + transactional replacement; shootout/wide-range |
| **3e** | Every tol validated; transactional `set_window`; temporal-only polygon table reuse; spatial change requires `mass_table=` |

Phase 3f and Stage 3 R=200 exit: **not started**.

## Commit series since `a7c0bd7` (reaudit)

| Commit | Class | Content |
|---|---|---|
| `1c3c39b` | test (RED) | Compat metadata / nested extra / exact event identity |
| `ce5508f` | fix | Schema v2 + nested extra + binary event hash |
| `12e57c9` | test (RED) | Missing table before base init |
| `238fbd3` | fix | Constructor preflight |
| `0eeaebd` | fix | `xy_events_real` + non-Gaussian test migration |
| *(docs tip)* | docs | API + acceptance-record corrections (this update) |

Prior A/B series `8a22fd6`…`8580364` + docs `a7c0bd7` remains on the branch
history. Pre-`8580364` missing-`excitation_support` failure is superseded
historical evidence only.

## Durable tip gates at `0eeaebd`

Exact commands and outputs:
`refactor-patches/reaudit_verification_0eeaebd.md`.

| Gate | Result |
|---|---|
| Focused compat + early-preflight | **22 passed** |
| Polygon + 3d/3e + shootout + wide-range | **118 passed** |
| Confirmation selectors | **9 passed** |
| Full suite | **299 passed**, 3 warnings |
| Collect-only | **299 tests** |
| Pins vs `pins_wt_2c2.json` | **PIN_DIFFS 0 MATCH** |
| `ruff check bstpp` | clean |
| `ruff check bstpp tests` | **133** (127 E402, 3 E702, 3 F401) |
| `jax_enable_x64` | unchanged (prepare-api selectors) |

## Conditional SBC

Pins MATCH; no confirmation anomaly → no Stage 1/2 conditional SBC.
Phase 3f not started.
