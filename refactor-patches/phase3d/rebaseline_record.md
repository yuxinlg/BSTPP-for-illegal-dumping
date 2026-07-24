# Phase 3d rebaseline record — polygon excitation support

Governing document: `docs/phase3_baseline_and_decisions.tex` (§10.d / D-17,
acceptance matrix §12). Frozen Phase 3 baseline tip: `476c2a0`. Branch:
`refactor`. Pins are MACHINE-LOCAL
(`refactor-patches/baselines-2026-07/pins.json`).

## Honesty note (verification trail)

No contemporaneous `refactor-patches/phase3d/` acceptance record was written
when the feature landed. This file is a **post-hoc** reconstruction
(2026-07-24) from commit messages, RED→GREEN repair commits, and group-gate
runs after those repairs. It does **not** claim a historical RED/GREEN suite
run at `edfce53` that was not observed in this repair session.

## Commits

### Feature (original Phase 3d)

| Commit | Class | Content |
|---|---|---|
| `906b30b` | prep | §13 polygon-mass diagnostic on Philadelphia geometries |
| `62bde1f` | prep | OP-9 backend shootout — quad-built Hermite tables win |
| `edfce53` | feat | Polygon excitation support modes + hybrid Hermite mass tables |

### Post-feature tip fix (not part of the original 3d feature commit)

| Commit | Class | Content |
|---|---|---|
| `8c4a702` | fix | Rebuild polygon mass table for held-out scoring (`log_expected_likelihood`); accompanied by `tests/test_heldout_polygon_mass.py` (GREEN at tip; not re-characterized as RED in this session) |

### Post-hoc contract repairs (RED→GREEN; this repair session)

| Commit | Class | Content |
|---|---|---|
| `8b95324` | test | RED — polygon mode must reject non-exact Gaussian spatial trigger |
| `850a4b9` | fix | Exact-type gate: `spatial_trig is Spatial_Symmetric_Gaussian` (not `isinstance`) in `Hawkes_Model.__init__` |
| `47cf260` | test | RED — validate supplied `PolygonMassTable` identities before reuse |
| `1adeb00` | fix | `validate_polygon_mass_table` in `bstpp/polygon_mass.py`; wired from `bstpp/excitation_support.py` |

Approved repair decisions for this session: polygon custom/non-Gaussian
spatial triggers are rejected by **exact type** match to
`Spatial_Symmetric_Gaussian`; a supplied mass table is valid only for the
recorded domain union, event coordinates/row order, spatial window, sigma
range/grid, and build settings (equal row counts are not enough).

## Deferred (explicitly out of scope here)

- General trigger-capability redesign (matching polygon-mass backend per
  custom spatial kernel).
- Reinterpreting covariate / reporting layers as model-domain union.

## Gate evidence (post-hoc repair group, after `1adeb00`)

| Gate | Result |
|---|---|
| Targeted | `tests/test_phase3d_excitation_support.py` + `tests/test_polygon_mass_table_validation.py` + `tests/test_heldout_polygon_mass.py` — **27 passed** |
| Full suite | **215 passed**, 3 warnings (~13m46s); `JAX_PLATFORM_NAME=cpu`, `MPLBACKEND=Agg` |
| Four pins | **bit-identical** vs `baselines-2026-07/pins.json` (`pin_check_v2.py`) |
| `ruff check bstpp` | clean |

Terminal evidence: agent shell `297124` (2026-07-24).

## Tip context after later leftover repairs

Subsequent Phase 3e / 3a / 3c leftover commits increased the collected suite
size; see tip verification at `05da465` (240 collected/passed). Those
commits are **not** Phase 3d and are recorded in their own phase folders /
commit messages.

## Classification

- Original `edfce53`: feature (polygon excitation + Hermite tables).
- `8c4a702`: semantic correctness fix on the held-out path (event-indexed
  state must match the scored realization).
- Exact-type gate + mass-table validation: **IV / contract enforcement** —
  invalid likelihood configurations fail loud; valid rectangle custom
  triggers and matching supplied tables remain accepted. Rectangle golden
  pins stayed bit-identical through the repair group.
