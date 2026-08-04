# Commit B — rebaseline / declared behavior change

**Class:** CF (caller-state ownership boundary)  
**Tip before:** `0d44ce5` (Commit A RED)  
**Env:** jax==0.4.23, numpyro==0.15.0, numpy 1.26.4, scipy 1.11.4, geopandas 1.1.3, `jax_enable_x64=False`

## Declared behavior change

`Hawkes_Model.simulate` / `LGCP_Model.simulate` previously accepted `rng=None` and drew from the process-global `np.random` stream. That path is **removed**. Callers must pass an explicit `numpy.random.Generator` via `rng=`. Omission raises `ValueError` naming the requirement (pre-3f B2).

This is intentional: a caller who believed they controlled the seed via process state (or who omitted `rng` while mutating global seed elsewhere) could not actually isolate simulation entropy. A default-seeded convenience path is not acceptable for the same reason.

## Unchanged defaults (documented)

- `run_mcmc(rng_key=None)` → `PRNGKey(10)` split chain (deterministic, reproducible).
- `run_svi(rng_key=None)` → same `PRNGKey(10)` convention as `run_mcmc` (new parameter; default path unchanged).

## Gate evidence (pre-commit)

| Gate | Result |
|---|---|
| `tests/test_caller_state_ownership.py` | 13 passed, EXIT:0 |
| Full suite `tests/` | 509 passed, 2 skipped, 1 xfailed, EXIT:0 |
| Four-config pins vs `baselines-2026-07/pins.json` | `PIN_DIFFS 0 MATCH`, EXIT:0 |
| `jax_enable_x64` | False before and after |

## Requirement-change test edits (flagged)

Identity asserts that encoded the pre-ownership alias (`is` caller object) were updated to content-equality / non-identity in:

- `tests/test_preparation_seam.py`
- mass-table install/API/Lane B tests (content equality after ownership copy)
- simulate call sites now pass `rng=` (`test_smoke`, simulate membership/union, `scripts/recover_test.py`)

These are requirement changes matching the ownership contract, not edits to silence a failing wrong assertion about the old alias.
