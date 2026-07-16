# Commit prompt: Phase 2d REVISED (4 patches: non-square pin, LGCP simulation path, KIND-AWARE trigger rng routing, ingestion-contract pins) -- base ce648df

## Supersession and scope

This prompt SUPERSEDES the previous Phase 2d prompt and bundle, which were
drafted against a reproduction of the HISTORICAL 2c patch files. The 2c
series actually landed on `refactor` is a REVISED series (tip `ce648df`,
subject `chore(patches): archive the revised Phase 2c series`), which
already contains CRS-authoritative geographic detection and a new-style
(`rng=None`) `Temporal_Power_Law`. Consequences, verified against the
landed tree:

- Old patches 1-2 carry over (re-hashed onto the new base).
- Old patch 3 is REWRITTEN: the detector is now KIND-aware (a review-caught
  defect in the withdrawn draft: name-only detection classified a
  POSITIONAL_ONLY `rng` as new-style, and the `rng=` call it then makes
  raises TypeError -- turning a WORKING legacy trigger into a crash;
  observed), and all "Temporal_Power_Law is old-style" claims are corrected
  (it gained `rng=None` in the landed RNG-unification commit; the tests now
  pin BOTH in-repo trigger classes as new-style).
- Old patch 4 is WITHDRAWN (its detection logic landed with 2c) and replaced
  by a residual test+docs patch: a NON-VACUOUS pin for the geographic-CRS
  contract warning -- the landed positive-CRS assertion matches
  'geographic', which geopandas' own area-on-geographic-CRS warning
  satisfies even with the contract warning deleted (MUTATION-VERIFIED:
  disabling the CRS-branch warn leaves the landed test green and the new
  pin red) -- plus the untested CRS-less GeoDataFrame fallback and the
  shared-linear-unit contract sentence. No detection logic changes; the
  historical test is untouched.

Out of scope, deferred by decision: the guide rewrite (disc -> square
window, Q_box formula, I11 restated as E[N] = E[Lambda_T], I12 rename,
prior-claim softening) happens after Phase 2 closes; the square window is
the deliberate model choice -- the windows are computational devices, not
part of the original model, and the box is the only shape whose truncation
the compensator can charge in closed form.

## Context

Repo: `yuxinlg/BSTPP-for-illegal-dumping`, branch `refactor`, tip `ce648df`
(suite 49). CLAUDE.md rules apply: show diffs before committing, never
`--dangerously-skip-permissions`, never edit a test to make it pass. Pinned
stack (verify BEFORE concluding anything from a test run): `jax==0.4.23`,
`numpyro==0.15.0`, `numpy<2`, `scipy<1.13`, `geopandas>=1.0`.

DRAFTING PROVENANCE: drafted AND executed against a fresh clone of the
LANDED `refactor` branch at `ce648df` (jax 0.4.23 / numpyro 0.15.0 /
numpy 1.26.4 / scipy 1.12.0 / geopandas 1.1.4). Every suite count, RED
verification, mutation check, pin claim, and byte-identity below was
OBSERVED on that clone, not predicted; `git am` replay of the four patches
from pristine `ce648df` confirmed clean with 59/59. A second static review
independently replayed patches 1-3 (ruff, compilation, `git diff --check`)
but could not run the dynamic checks (its CPU cannot load jaxlib 0.4.23);
the dynamic confirmation below is therefore REQUIRED on Terhi's machine.
Treat any deviation as a finding, not noise. If the base tip is not
`ce648df` -- or anything else seems missing -- STOP and report rather than
working an incorrect base.

## Manifest

Base: `ce648df`. Apply order = filename order. Expected pin movement: NONE.

| # | file | sha256 | suite after |
|---|------|--------|-------------|
| 1 | 0001-test-infra-pins-add-the-non-square-4-1-golden-pin-co.patch | 23efd745bc24ab4706548ec9ff3c1aba15236aeef9523af82894fd4dc70808e0 | 49 |
| 2 | 0002-fix-sim-restore-the-LGCP-simulation-path-_sample_cel.patch | 9dfc59b2edc015ba9a7920fb08b75438d998f068dd563f36802c00149d71db23 | 52 |
| 3 | 0003-fix-sim-custom-trigger-rng-routing-by-KIND-AWARE-sig.patch | b36a170c244cef0708a68312de6c47c614725764973193aa7a1ccc063e50eaa7 | 57 |
| 4 | 0004-test-docs-contract-non-vacuous-pin-for-the-geographi.patch | 63de11a7c73eeda28d15ecc81480b318aeeaad356ce84d9542d3fcf64e06991f | 59 |

Verify with `sha256sum` before applying; the previous Phase 2d bundle is
superseded and must not be mixed in.

## The series

1. `0001-...non-square-4-1-golden-pin...` -- UNCHANGED content, re-hashed.
   Adds config `hawkes_nonsquare_4to1` to `refactor-patches/pin_check_v2.py`
   (A_ = [[0,4],[0,1]], the 60-point cloud stretched 4x in X, plain hawkes,
   value + full gradient pins): the discriminating pin the audit found
   missing -- the unit-box configs cannot distinguish the internal-isotropic
   kernel from the real-isotropic one. Retro movement demonstration,
   OBSERVED against the LANDED commits via worktrees (the script pins any
   checkout by path): `17bb7b3` (2b tip) -> `ea42816` (2c patch 1) all four
   configs bit-identical; `ea42816` -> `6d89280` (2c patch 2, the contract
   commit) unit-box configs bit-identical, `hawkes_nonsquare_4to1` moved in
   all 5 entries (worst rel: grad_beta 0.706, grad_sigmax_2 0.574,
   grad_alpha 0.247, grad_a_0 0.056, loglik 0.0023); `6d89280` -> `ce648df`
   all bit-identical. Bracketed to exactly the contract commit. Suite: 49
   (script-only). The four-config output at `ce648df` is the Phase 3 pin
   baseline.

2. `0002-...restore-the-LGCP-simulation-path...` -- UNCHANGED content,
   re-hashed; applies cleanly on the landed tree (`_sample_cells` is still a
   `Hawkes_Model` method at the landed tip while its caller `_sim_cox` is on
   `Point_Process_Model`). Two stacked defects, RED-verified: (a)
   `LGCP_Model.simulate()` -> `AttributeError: 'LGCP_Model' object has no
   attribute '_sample_cells'`; (b) the z->f decode lived only in
   `Hawkes_Model.simulate()` -> `KeyError('f_t')` on any z-only parameter
   dict (the shape of `mcmc.get_samples()` output; predates 2b and masks (a)
   in RED runs). Fix: byte-identical relocation of `_sample_cells` to the
   base class; decode extracted as shared
   `Point_Process_Model._decode_field_parameters`; `LGCP_Model.simulate()`
   gains the final `points.sjoin(self.A)` clip that `_sim_cox`'s docstring
   promises -- DECLARED BEHAVIOR CHANGE on the LGCP path (out-of-polygon
   points were previously returned for non-rectangular domains; nothing
   covered the path). New `tests/test_lgcp_sim.py` (3). Suite: 52.
   Fixed-seed cox_hawkes Hawkes simulate byte-identical across the commit
   (3162 events, exact array equality); pins bit-identical.

3. `0003-...KIND-AWARE-signature...` -- REWRITTEN. Replaces the per-draw
   broad `except TypeError` fallback (still present at the landed tip,
   `_sim_offspring`) with signature inspection once per call
   (`utils.accepts_rng_kwarg`). Two RED-verified defects motivate the exact
   form: (a) the MASK -- a new-style trigger raising TypeError on its
   Generator path was swallowed, re-executed without rng, and
   `_sim_offspring` completed with 9 rows drawn off the Generator; (b) the
   withdrawn draft's NAME-ONLY detection -- 'rng' anywhere in the signature
   -- classified a POSITIONAL_ONLY `rng` (`def f(pars, rng, /)`) as
   new-style, and the `rng=` call then raises TypeError('got some
   positional-only arguments passed as keyword'), crashing a working legacy
   trigger (observed; review-caught). Detection is therefore KIND-aware:
   'rng' counts only as POSITIONAL_OR_KEYWORD or KEYWORD_ONLY, plus
   **kwargs; POSITIONAL_ONLY `rng` and VAR_POSITIONAL `*rng` classify
   old-style; uninspectable C callables fall back to the legacy form.
   CORRECTED CLAIM throughout code comments and tests: `Temporal_Power_Law`
   is NEW-style at the landed tip (`simulate_trigger(self, pars, rng=None)`
   via scipy's `random_state=rng`); the old-signature shape is a legacy
   THIRD-PARTY concern, represented in tests by `_OldStyleSpatial`. The
   classification test pins BOTH in-repo trigger classes as new-style -- a
   detector that dropped `Temporal_Power_Law` from the Generator stream
   would silently break reproducibility. RNG streams unchanged for all
   in-repo triggers. New `tests/test_trigger_compat.py` (5): classification
   incl. kind edge cases; the mask regression; legacy-shape support; the
   positional-only trigger running live through `_sim_offspring`; **kwargs
   triggers receiving the Generator. Suite: 57. Fixed-seed simulate
   byte-identical; pins bit-identical.

4. `0004-...non-vacuous-pin...` -- NEW, replaces the withdrawn CRS patch.
   Test+docs only; no detection logic changes; the historical
   `test_geographic_coordinate_warning` is untouched. (a) MUTATION-VERIFIED
   vacuity finding: disabling the landed CRS-branch contract warning leaves
   the landed positive-CRS assertion GREEN (geopandas' 'Geometry is in a
   geographic CRS' area warning satisfies `match="geographic"`); the new pin
   matches the contract body ('anisotropic on the ground') and goes RED
   under the same mutation -- observed both ways. (b) CRS-less GeoDataFrame
   fallback pinned in both directions (degree box warns with the contract
   message; metric unit box constructs warning-free); previously only array
   domains exercised the heuristic. (c) The contract comment gains the
   shared-linear-unit requirement: one projected unit on both axes, no
   X-in-meters / Y-in-feet mixtures -- a projected CRS guarantees it, raw
   arrays and CRS-less GeoDataFrames are on the user's honor. New
   `tests/test_ingestion_contract.py` (2). Suite: 59. Pins bit-identical.

## Procedure

1. `python -c "import jax; print(jax.__version__)"` -> 0.4.23. Confirm
   `git log --oneline -1` shows `ce648df`; if not, STOP and report.
   Baseline: `python -m pytest tests/ -q; echo EXIT:$?` -> 49, EXIT:0.
2. `sha256sum` the four patch files against the manifest.
3. Apply patch 1 with `git am`. Capture the FOUR-CONFIG pin baseline:
   `python refactor-patches/pin_check_v2.py . > pins_base.json`. Suite -> 49
   with EXIT status.
4. OPTIONAL retro movement reproduction (already observed; numbers in patch
   1's commit body): worktrees at `17bb7b3`, `ea42816`, `6d89280`; run
   `python refactor-patches/pin_check_v2.py <worktree>` against each and
   diff. Expected: identical through `ea42816`; ONLY `hawkes_nonsquare_4to1`
   moves at `6d89280`; identical after. Different pattern -> STOP and report.
5. Apply patches 2-4 in order with `git am`. After EACH: full suite with
   EXIT status (expected 52 / 57 / 59) and
   `python refactor-patches/pin_check_v2.py . > pins_N.json;
   diff pins_base.json pins_N.json` -> MUST be empty all three times.
6. Byte-identity across patches 2 and 3: before applying patch 2, capture a
   fixed-seed cox_hawkes simulation (unit-box GeoDataFrame domain, latents
   from a `PRNGKey(3)` trace, alpha=0.25, beta=2.0, sigmax_2=0.02,
   `rng=np.random.default_rng(17)` -- the test_identities reproducibility
   configuration), save the [X,Y,T] array; re-capture after patches 2 and 3
   and assert exact array equality both times. Expected: identical (3162
   events on the drafting clone; a differing count is a finding, not noise).
7. RED / mutation reproductions (scratch worktrees at the stated PRE-fix
   tips; never modify the mainline tree; revert every mutation):
   a. Pre-patch-2 tip: copy `tests/test_lgcp_sim.py` in and run -- all 3
      FAIL with `KeyError: 'f_t'`; then simulate an LGCP model with a
      PRE-DECODED f-dict -> `AttributeError: ... '_sample_cells'`.
   b. Pre-patch-3 tip: a trigger subclass whose
      `simulate_trigger(self, pars, rng=None)` raises TypeError iff rng is
      not None must COMPLETE through `_sim_offspring(...,
      rng=np.random.default_rng(0))` (the mask). Additionally reproduce the
      withdrawn-draft defect: a name-only detector (any parameter named
      'rng' or **kwargs) classifies `def simulate_trigger(self, pars,
      rng=None, /)` as new-style, and calling it with `rng=` raises
      TypeError -- the kind-aware rule in `utils.accepts_rng_kwarg` is what
      prevents this.
   c. At the SERIES TIP: mutate the CRS branch to
      `if False and _crs.is_geographic:` and run
      `tests/test_ingestion_contract.py::test_geographic_crs_contract_warning_is_nonvacuous`
      together with
      `tests/test_identities.py::test_geographic_coordinate_warning`.
      Expected: the new pin FAILS, the historical test PASSES (the vacuity
      finding, reproduced). Revert the mutation and rerun both -> green.
   If any RED/mutation does not reproduce: STOP and report.
8. Reproducibility, twice, both green both times:
   `python -m pytest tests/test_identities.py::test_simulate_fully_reproducible_with_generator
   tests/test_lgcp_sim.py::test_lgcp_simulate_runs_and_is_reproducible
   tests/test_trigger_compat.py -q` (8 tests per run).
9. `ruff check bstpp/` -> clean. (New test files carry the same E402 layout
   as every existing test file; the ruff gate is `bstpp/`, matching the 2c
   procedure. The full suite reports 1 warning: geopandas' geographic-CRS
   area warning inside the deliberately-geographic ingestion test --
   expected, and the very emission the vacuity finding is about.)
10. Archive: `mkdir -p refactor-patches/phase2d`, copy the four patch files
    and this prompt in, commit as
    `chore(patches): archive the phase 2d series in refactor-patches/`.
    Suite once more -> 59. Reset authorship
    (`git rebase --exec 'git commit --amend --reset-author --no-edit' HEAD~5`)
    and rerun the suite. No pushes to `main`; landing `refactor` into `main`
    remains a separate, explicitly approved step.

## Report back

- Baseline confirmation (tip `ce648df`, 49, EXIT:0) and manifest hash check.
- Per-patch pytest tails WITH exit statuses (49 / 52 / 57 / 59).
- All three pin diffs (must be empty) and, if step 4 was run, the movement
  table.
- Byte-identity results for step 6 (both comparisons), with the event count.
- Verbatim tails of 7a, 7b (both parts), and 7c (mutated AND reverted runs).
- The double run of step 8.
- `ruff check bstpp/` output and `git log --oneline -6`.
- Confirmation that no historical file (landed 2c content, archived 2c
  patches, decision memo, guide sources, `test_geographic_coordinate_warning`)
  was modified.
