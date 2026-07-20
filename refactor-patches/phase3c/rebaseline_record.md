# Phase 3c rebaseline record — background/covariate polygon support corrections

Scope: `phase3_baseline_and_decisions.tex` §10.c, decisions D-6/D-7/D-10/D-11,
classification rows in §7.3. Commits (branch `refactor`):

| Commit | Class | Change |
|---|---|---|
| `985c771` (3c-1) | SC | No-covariate background support is the clipped `|C_c ∩ A|` areas; one `support_cells` object feeds the likelihood integration arrays and `_sim_cox`'s location draw. Touch-only/sliver cells (≤1e-10 normalized) leave the support. |
| `3120268` (3c-2) | SC | Event-to-field-cell membership joins the same clipped support (constructor + held-out path). Out-of-domain events fail loudly at membership (D-3); one 3a-era report-mode test updated accordingly (flagged in the commit body). |
| `6d89b6e` (3c-3) | SC | Covariate support is the common refinement `C_c ∩ A_m ∩ A` (overlay on the clipped support cells); the covariate-sjoin override of the in-domain cell set is removed (A authoritative); plain-Hawkes `cov_area` and `_sim_hawkes_bg` sampling use the clipped `cov_support` layer. |
| `1465e79` (3c-4) | IV | Coverage contract: gaps and positive-area overlaps of the covariate layer inside A are violations under `data_contracts='reject'` (warnings under `'report'`); findings export the actual gap/overlap/sliver geometries on `ContractCheck.geometry`. |
| `62ea1c1` (3c-5) | API | Standardization always reported (`model.standardization`, invertible mean/scale); explicit `standardize_cov="domain_area"` convenience weighted by `|C_c ∩ A|`; unknown strings rejected. Boolean semantics and the True default unchanged (OP-3/OP-4 remain open). |

Interleaved independent fixes (separate sessions, spawned from this one):
`f372e93` (3a grid-line diagnostic bounds transpose), `65b586e` (`run_svi`
ignored `plot_loss=False`).

## What changed numerically (polygon regime only)

- `Itot_xy` / background compensators DECREASE by the previously charged
  outside-A mass (full boundary cells → exact clipped intersections; covariate
  refinement pieces clipped to A; plain-Hawkes covariate areas clipped).
- Background simulation (`_sim_cox`, `_sim_hawkes_bg` covariate leg) is
  supported on A: points outside A are never drawn, and the Poisson count mean
  now equals the (clipped) compensator exactly instead of being thinned down by
  `simulate()`'s A-filter after the draw (verified by a seeded 200-draw Monte
  Carlo identity, RED z=16.7 → GREEN |z|<5).
- Boundary events on grid-line-aligned stretches of ∂A whose left-closed D-22
  cell has zero support map to the supported adjacent cell (measure-zero set;
  the one D-22 refinement 3c introduces).
- Events outside A (report/off contract modes; held-out scoring) fail loudly at
  membership instead of being scored on bounding-rectangle cells.
- Gapped/overlapping covariate layers are rejected by default (previously
  silent zeros / double-charging).

The excitation compensator still integrates over the bounding rectangle — 3d
scope (D-17), deliberately untouched.

## Unchanged-regime gates (§12 "M", rectangle regime)

- All four golden pin configs (`pin_check_v2.py`: hawkes, cox_hawkes, lgcp,
  hawkes_nonsquare_4to1 — all array-rectangle domains) bit-identical to
  `refactor-patches/baselines-2026-07/pins.json` after EVERY 3c commit
  (machine-local comparison).
- Array-rectangle support is constructed as the exact constant `1/n_xy²`
  (never geometrically), reproducing the legacy `np.full` float32 array
  bit-identically; interior-event membership pinned bit-unchanged by a
  200-event property test; rectangle-polygon degeneracy acceptance (§10.d)
  passes within float32 tolerance.
- Full suite at the 3c tip: 153/153 (`MPLBACKEND=Agg`, quiescent tree — see
  the 3c-3 commit body for the two reproduced-and-explained environment/process
  flakes that are NOT code defects).

## Semantic/property tests (§12 "M", semantic)

`tests/test_clipped_support.py` (16), `tests/test_covariate_coverage.py` (11),
`tests/test_standardization_api.py` (6) — every SC/IV/API claim above has a
RED-verified test; RED states are recorded per commit body.

## Conditional SBC escalation (§12 "C") — OPEN, requires a decision

3c intentionally changed the LGCP/Cox background path (clipped `Itot_xy`,
covariate refinement, sampler support), which is exactly the §12 example
trigger: "intentional change to a model-specific path (e.g., 3c touches the
LGCP background path → the stage covering LGCP)". The covering stage is
**SBC stage 2 (LGCP, R=600, PASS at `23cdf94`)**. Note the trigger's letter is
C (conditional), and the changed semantics live in the polygon regime while
the archived stage-2 config is the unit-rectangle domain — on that config the
integrands are bit-identical (pins above), so a rerun would re-verify an
unchanged regime. A polygon-domain SBC stage has never been run and is not
part of the archived program. Decision needed (not taken unilaterally):
rerun stage 2 as-is, extend the SBC program with a polygon-domain
configuration, or record the rectangle-regime bit-identity as discharging the
trigger. Runtime for a stage-2 rerun is substantial (see
`results/sbc_stage2/` logs).
