# Phase 3e rebaseline record — computational cutoffs (OP-6)

Governing document: `docs/phase3_baseline_and_decisions.tex` (§10.e / OP-6,
acceptance matrix §12). Frozen Phase 3 baseline tip: `476c2a0`. Branch:
`refactor`. Pins are MACHINE-LOCAL
(`refactor-patches/baselines-2026-07/pins.json`).

## Honesty note (verification trail)

No contemporaneous `refactor-patches/phase3e/` acceptance record was written
when the feature landed. This file is a **post-hoc** reconstruction
(2026-07-24) from commit messages, RED→GREEN repair commits, and group-gate
runs after those repairs. It does **not** claim a historical RED/GREEN suite
run at `2ce665c` that was not observed in this repair session.

## Commits

### Feature (original Phase 3e)

| Commit | Class | Content |
|---|---|---|
| `2ce665c` | feat | Human-unit temporal interface (`mean_lag_days`) + computational cutoffs (OP-6): tolerance formulas, physical overrides, `cutoff_provenance` |

### Post-hoc contract repairs (RED→GREEN; this repair session)

| Commit | Class | Content |
|---|---|---|
| `fec8f78` | test | RED — reject invalid cutoff tolerances even when a physical window wins precedence |
| `5f5bb8d` | fix | `_validate_cutoff_tol` + early validation in `resolve_computational_cutoffs` (`bstpp/cutoffs.py`) |
| `360ea57` | test | RED — `set_window` must update `cutoff_provenance` atomically |
| `a46c960` | fix | `set_window` re-resolves provenance as physical; constructor restores OP-6 provenance after install so tolerance selection is preserved (`bstpp/main.py`) |

Tolerance formulas remain kernel-specific: temporal omitted mass /
`mean_lag_days` for `Temporal_Exponential`; spatial omitted mass for
`Spatial_Symmetric_Gaussian` with the per-axis square cutoff. Every supplied
tolerance must be finite and in `(0, 1)` even when a physical cutoff wins.

## Deferred (explicitly out of scope here)

- Tolerance / `mean_lag_days` interface for `Temporal_Power_Law` (`beta` is a
  shape parameter, not a mean lag). Explicit physical `window` /
  `spatial_window` remains the compatible path for unsupported triggers.

## Gate evidence (post-hoc repair group, after `a46c960`)

| Gate | Result |
|---|---|
| Targeted | `tests/test_phase3e_cutoffs.py` — **29 passed** |
| Full suite | **228 passed**, 3 warnings (~8m00s); `JAX_PLATFORM_NAME=cpu`, `MPLBACKEND=Agg` |
| Four pins | **bit-identical** vs `baselines-2026-07/pins.json` (`pin_check_v2.py`) |
| `ruff check bstpp` | clean |

Terminal evidence: agent shell `297125` (2026-07-24).

## Related but not Phase 3e

`8c4a702` (held-out polygon mass rebuild) sits between `2ce665c` and the
post-hoc 3e repairs chronologically; it is a Phase 3d / event-indexed-state
fix and is recorded in `refactor-patches/phase3d/rebaseline_record.md`.

## Classification

- Original `2ce665c`: feature (cutoff resolution + provenance + human-unit
  temporal scale interface).
- Invalid-tolerance validation: **IV** — malformed tol fails loud regardless
  of physical precedence.
- `set_window` provenance atomicity: **SC** for the public window API —
  realized cutoffs and `cutoff_provenance` update together. Rectangle golden
  pins stayed bit-identical through the repair group.
