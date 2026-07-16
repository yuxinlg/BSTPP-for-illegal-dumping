# Commit prompt: Phase 2c REVISED (5 patches: cascade clipping, REAL-UNIT spatial trigger, REAL-UNIT spatial_window, RNG unification, docs + geographic warning)

## STOP: sign-off gates (supersede the withdrawn original 2c prompt)

- This series REPLACES the original 2c patches 2-4. Patch 1 is byte-identical
  to the original (already approved in principle: rectangle clipping).
- Patches 2 and 3 are MODEL SEMANTICS CHANGES gated on the decision memo
  (decision_memo_yuxinlg.md): (patch 2) the spatial trigger becomes a
  REAL-unit object -- offspring displacements N(0, sigmax_2 * I) in the
  units of the input X/Y columns, sigmax_2 and its (user-supplied,
  no-default) prior in squared real units; (patch 3) spatial_window becomes
  a REAL length with per-axis box semantics symmetric across event term,
  compensator, and offspring thinning. Terhi's decisions D1-D4 are recorded
  (D1 contract yes + highlight; D2 conversion at the likelihood boundary,
  A2; D3 no default prior; D4 ingestion warning); yuxinlg's sign-off must be
  recorded in your instructions before applying patches 2-5. Patches 2 and 3
  edit tests (I10 restated as covariance + negative control; test_pairs
  dense oracle) -- covered by the same sign-off, and flagged as such in the
  commit bodies.
- IMPORTANT ORDERING: patch 3 assumes patch 2 (the ws clip is a SCALAR only
  because the compensator limits are already real-unit). Do not reorder.

## Context

Repo: `yuxinlg/BSTPP-for-illegal-dumping`, refactor branch, on top of landed
Phase 2b (tip 17bb7b3, suite 38 passing). CLAUDE.md rules; `jax==0.4.23`
(verify); `pin_check_v2.py`; `verify_sim.py` (acceptance).

DRAFTING PROVENANCE: unlike prior series, these patches were drafted AND
executed against a live clone of the refactor branch on the drafting machine
(jax 0.4.23 / numpyro 0.15.0 / numpy 1.26.4 / scipy 1.12.0 / geopandas
1.1.4): every suite count, RED verification, and pin claim below was
OBSERVED there, not predicted. Your job is to reproduce on Terhi's machine
(machine-local pins) -- treat any deviation as a finding, not noise.

Pin contract: unit-box pinned configs bit-identical throughout (verified on
the drafting machine); the drafting machine ALSO pinned a non-square 4:1 box
config, which moved ONLY at patch 2 (that movement is the fix's signature --
if `pin_check_v2.py` has no non-square config, note that in your report; do
not add one silently). RNG streams change in patch 4 by design.

## The series

1. `0001-fix-sim-offspring-outside-the-bounding-rectangle*.patch`
   UNCHANGED from the original 2c. Cascade discards out-of-rectangle
   offspring BEFORE they parent (Prop 1.1(ii)). Regression verified RED
   pre-fix: 1130/1491 cascade events outside the rectangle. Honest severity
   record retained: the conservation bias is second-order. Suite: 39.

2. `0002-fix-model-sim-MODEL-SEMANTICS-CHANGE*.patch`  [GATED]
   REAL-UNIT SPATIAL TRIGGER. All three legs move in one commit (symmetry
   cannot survive a split): event term via new atom
   `real_spatial_trigger_values` (per-axis stretch + Jacobian sx*sy
   converting the real-area density to the internal measure -- the
   review-caught unit point); compensator limits stretched to real units
   (axis_scales REQUIRED keyword-only); simulator draws real-unit directly
   (the box-span rescale deleted). I10 restated: similarity COVARIANCE
   (sigmax_2 -> c^2 sigmax_2) + negative control asserting the old axis-wise
   invariance now FAILS. RED observed: isotropy ratio 16.97 on a 4:1 box
   (predicted 16). Square-box cancellation pinned as an algebraic special
   case. Suite: 43. Unit-box pins bit-identical (exact *1.0f).

3. `0003-fix-model-sim-MODEL-SEMANTICS-CHANGE*.patch`  [GATED]
   REAL-UNIT spatial_window, three-leg symmetric. Single-sourced predicate
   `utils.within_real_box_window` (pair mask + offspring thinning); the
   compensator clip is a SCALAR jnp.minimum on the real limits -- exact
   mirror of min(T - t, w). Box semantics retained (disc has no closed
   form). Identities: (I4-ws) m = alpha*F_beta(w)*BoxMass(ws) through the
   production atom; (I11-ws) conservation at ws=0.2; NEW (I12, box
   invariance): one dataset under two differently-shaped rectangles agrees
   on pair set, real displacements/densities, clipped compensator, and
   byte-identical seeded cascades. RED observed at (I12) clause (a): 33 vs
   68 pairs across a 4:1 and 3:1 box. Suite: 46. Pins bit-identical
   (ws=None in all pinned configs).

4. `0004-refactor-sim-RNG-unification*.patch`
   The original 2c patch 3, REBASED across patches 2-3: `_sim_offspring`'s
   Generator draw is the real-unit displacement, thinning through the shared
   predicate; conflicts were exactly the two semantics commits' regions,
   resolved by taking both changes. Everything else unchanged from the
   original (rng=None legacy fallback; old-signature user triggers via
   TypeError fallback; Generator-subclass spy in capture tests; parametrized
   byte-identity test plain + cox). Suite: 48. Pins bit-identical.

5. `0005-docs-contract-ingestion*.patch`
   Inventory doc REWORKED (row 8 kernel geometry RESOLVED; row 9 real-unit
   ws; NEW row 10: background PRIOR geometry -- PriorVAE decoder trained on
   an isotropic internal-square SE kernel, so prior real-space correlation
   lengths stretch per axis; prior-side only, the background LIKELIHOOD is
   exactly affine-invariant). Phase 3 candidates updated (spatial half of
   the units candidate DONE; optional (sigma_x, sigma_y) extension listed as
   a deliberate decision). D4 geographic-coordinate ingestion warning +
   two-sided test (fires on a degree-like box; unit box constructs
   warning-free). CLAUDE.md: 2c verification-environment notes + Unit
   Contract section. Suite: 49.
   If CLAUDE.md was already edited locally after the 3199/3200 episode,
   resolve the `git am` conflict by MERGING the content, not duplicating it,
   and say so in your report.

## Procedure

1. `python -c "import jax; print(jax.__version__)"` -> 0.4.23.
2. `python pin_check_v2.py . > pins_base.json`.
3. Apply patches in order with `git am`. After EACH: full suite -- run
   `python -m pytest tests/ -q; echo EXIT:$?` and read the EXIT status, do
   not rely on piped output (expected counts 39 / 43 / 46 / 48 / 49) -- and
   `diff pins_base.json pins_N.json` -> MUST be empty for all five (see the
   pin-contract note above re: any non-square config).
4. After patch 4: `python verify_sim.py .` -> VERIFY_SIM: ALL PASS. NOTE:
   verify_sim exercises the simulator with fixed parameters -- if any of its
   configs sets sigmax_2 or spatial_window as INTERNAL-unit values on a
   non-unit-box domain, those values now mean something different (real
   units); expected drift is in kernel-scale-sensitive checks only. Report
   any failure verbatim rather than adjusting verify_sim silently.
5. Reproducibility spot check: run the parametrized
   test_simulate_fully_reproducible_with_generator twice; both runs green.
6. `ruff check bstpp/` -> clean. Reset authorship
   (`git rebase --exec 'git commit --amend --reset-author --no-edit' HEAD~5`),
   rerun the suite once. No pushes to `main`.

## Report back

Per-patch pytest tails WITH exit statuses, pin diffs, verify_sim output,
ruff line, `git log --oneline -6`, how the CLAUDE.md merge (if any) was
resolved, and whether pin_check_v2.py contains a non-square-box config.
