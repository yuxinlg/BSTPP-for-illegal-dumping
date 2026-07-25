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

Subsequent Phase 3e / 3a / 3c leftover commits and the A/B preparation /
`set_window` API work increased the collected suite; tip verification at
`8580364` records **277** collected/passed with pins MATCH. See that tip
note for the full ownership table.

## Classification

- Original `edfce53`: feature (polygon excitation + Hermite tables).
- `8c4a702`: semantic correctness fix on the held-out path (event-indexed
  state must match the scored realization).
- Exact-type gate + mass-table validation: **IV / contract enforcement** —
  invalid likelihood configurations fail loud; valid rectangle custom
  triggers and matching supplied tables remain accepted. Rectangle golden
  pins stayed bit-identical through the repair group.

## Explicit preparation API and hard-require table (A/B; tip `8580364`)

**Not behavior-preserving.** Polygon construction previously could build a
mass table synchronously (including process-global JAX precision mutation in
older paths). The approved hard-require design is an **API / construction
contract change**:

| Commit | Class | Content |
|---|---|---|
| `9342a19` | test (RED) | Ctor without table fails clearly; `build_quad_table` must not toggle `jax_enable_x64` |
| `ae27947` | fix (**API**) | Public `prepare_polygon_mass_table(...)`; host NumPy/SciPy float64 throughout quadrature/nodes/values/slopes; **no** `jax.config.update`; ctor validates/installs supplied table only — never rebuilds on incompatibility; error names `prepare_polygon_mass_table` |
| `a8ec985` | test (RED) | Temporal-only `set_window` reuses table; spatial change requires `mass_table=`; failed replacement transactional |
| `8580364` | fix (API/SC) | `Hawkes_Model.set_window(window, spatial_window=None, *, mass_table=None)`; validate candidate against prospective spatial window before mutation; commit only after local prepare succeeds |

**Sigma semantics:** `min_sigma` / `max_sigma` are spatial standard deviations
in real coordinate units; compatibility with a `sigmax_2` prior uses
`sqrt(sigmax_2)`. Table interpolation convention and metadata preserved.

**Compatibility identity** (validation covers at least): canonical
`PreparedDomain.union_geometry` hash; event coordinates and row order; event
count; realized spatial window; sigma range/knot grid/interpolation
convention; panel height after unit conversion; quadrature order and
backend/schema version; array shapes and finite values.
`extra_provenance` remains descriptive unless it affects numerical
construction.

**Held-out:** rebuilds event-indexed state and prepares a table for the
held-out realization via `prepare_polygon_mass_table` (does not reuse
training table rows).

**Rectangle mode:** unchanged; no `mass_table` required.

### Numerical / both-mode evidence (tip `8580364`)

Combined command (also recorded in tip verification):

```bash
JAX_PLATFORM_NAME=cpu python -m pytest \
  tests/test_polygon_mass_prepare_api.py \
  tests/test_polygon_mass_table_validation.py \
  tests/test_heldout_polygon_mass.py \
  tests/test_polygon_mass_backend_shootout.py \
  tests/test_polygon_mass_wide_range.py \
  tests/test_phase3d_excitation_support.py \
  tests/test_phase3e_cutoffs.py -q --tb=line
```

Result: **96 passed**. Includes shootout, wide-range, conservation /
rectangle-degeneracy, and both-mode property tests.

§12 smoke/confirmation selectors (rectangle + polygon + family traces +
fixed-cutoff/`set_window` + `jax_enable_x64` + plain-Hawkes NUTS smoke):
`refactor-patches/confirmation_8580364.md` — all green.

Full suite **277 passed**; pins **PIN_DIFFS 0 MATCH**; no conditional SBC
rerun; Phase 3f not started.

### Intermediate RED before `8580364`

A mid-implementation suite failure
(`AttributeError: ... no attribute 'excitation_support'`) is pre-`8580364`
evidence of the ctor/`set_window` install-order defect, **not** a tip
failure.

## Reaudit strengthening (after tip `a7c0bd7`; tip `0eeaebd`)

**Reaudit finding:** acceptance text claimed compatibility checks that
`validate_polygon_mass_table` did not enforce; `extra_provenance` could
overwrite reserved fields; event identity used lossy ``.9g`` formatting;
missing-table errors arrived after expensive base construction; API docs
still said tables were built at construction.

| Commit | Class | Content |
|---|---|---|
| `1c3c39b` | test (RED) | Missing/malformed/wrong compat metadata; nested extra protection; ``.9g`` collision; legacy schema |
| `ce5508f` | fix | Schema **v2** constants; nested `extra`; binary le-f64 event hash; validate+load reject legacy/falsified metadata |
| `12e57c9` | test (RED) | Explicit polygon + missing table must not enter `Point_Process_Model.__init__` |
| `238fbd3` | fix | Earliest preflight for `excitation_support="polygon"` |
| `0eeaebd` | fix | `xy_events_real` float64 ingestion snapshot so exact hashes match prepare inputs; migrate non-Gaussian trigger test to supply a table |

**Intentional incompatibilities (not BP):**

- Legacy `hybrid_quad_hermite_numpy_v1` sidecars / missing required metadata.
- Decimal-``.9g`` event hashes (including distinct float64 events that collided).
- Falsified reserved fields via `extra_provenance` (ignored; nested under `extra` only).

Durable commands/results: `refactor-patches/reaudit_verification_0eeaebd.md`
(full suite **299 passed**; pins **PIN_DIFFS 0 MATCH**; polygon group **118 passed**).
