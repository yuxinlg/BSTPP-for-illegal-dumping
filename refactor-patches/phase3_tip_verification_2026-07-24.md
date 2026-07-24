# Tip verification — Phase 3d/3e trail repair session (2026-07-24)

Tip at verification: `05da465` on branch `refactor`.
Env: `illegal-dumping`, `JAX_PLATFORM_NAME=cpu`, `MPLBACKEND=Agg`.

This note closes the repair session that (1) characterized missing 3d/3e
acceptance records honestly, (2) fixed confirmed tip defects with RED→GREEN
commits, and (3) landed Phase 3a/3c leftover contract repairs that were in
scope for the same audit but are **not** Phase 3d/3e features.

## Repair commits since feature tips

See:

- `refactor-patches/phase3d/rebaseline_record.md`
- `refactor-patches/phase3e/rebaseline_record.md`

### Phase 3a leftovers (post-hoc; not a rewrite of the original 3a record)

| Commit | Content |
|---|---|
| `5cea6c9` / `6e63efc` | Finite positive horizon → `horizon_invalid` |
| `dde9b80` / `511ca5b` | Polygonal finite positive-area domain |
| `5d357ed` / `c22aa51` | Missing covariate CRS when domain declares CRS |

Clean 3a leftover group gate (ignore uncommitted 3c RED at the time):
targeted 37; full suite **236 passed**; pins bit-identical; `ruff check bstpp`
clean (agent shell `923302`).

### Phase 3c leftover (union-consistent domain area)

| Commit | Content |
|---|---|
| `26a4c3c` / `05da465` | `PreparedDomain.area_ratio` / `union_geometry` from set-union of domain rows (SC); disjoint/single-row unchanged |

3c leftover group gate: targeted 16; full suite **240 passed**; pins
bit-identical; `ruff check bstpp` clean (agent shell `923303`).

## Tip gates

| Gate | Result | Evidence |
|---|---|---|
| Collect-only | **240 tests collected** | tip `05da465` |
| Full suite | **240 passed** (~12m07s) | 3c group gate `923303` (tip unchanged since) |
| Four pins | **bit-identical** | final tip pass `923304` |
| All targeted repair tests | **109 passed** (~1m20s) | final tip pass `923304` |
| `ruff check bstpp` | clean | group gates + final tip pass |
| `ruff check bstpp tests` | **133 findings, all inherited E402** (env-before-imports test pattern) except one pre-existing unused-import note in `tests/test_phase3d_excitation_support.py` (F401 `jax.numpy`); **zero findings** in new repair-session files `test_domain_union_area.py`, `test_polygon_mass_table_validation.py`, `test_heldout_polygon_mass.py`, `test_phase3e_cutoffs.py`, `test_data_contracts.py` | final tip pass `923304` |

## Deferred (session decisions; not started)

- General trigger-capability redesign for non-Gaussian polygon mass backends.
- Power-law tolerance / `mean_lag_days`.
- Phase 3f / SBC escalation without separate approval.
- Do not amend/rebase/rewrite `edfce53`, `2ce665c`, or `8c4a702`.
