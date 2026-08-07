# `refactor-patches/captures/` — the declared capture root (D-46)

**Every capture preserved under D-45 lives here, and nowhere else, from A-46
forward.** D-45 says a gate that goes red has its capture preserved before the
fix. It says nothing about *where*, and that omission has a measurable
consequence.

## The consequence D-46 exists to remove

A preserved capture is a text file. Two classes of instrument in this
repository search text files in the tree:

- **document-census instruments** — today
  `refactor-patches/phase3f/wp2/probe_a42_gate_capability_census.py`, which
  counts, for each gate, the tracked non-source files carrying that gate's own
  failure signature;
- **citation-sweep instruments** — today `results/_a25_citation_sweep.py`,
  which extracts path-like citations from a closed set of documents and
  reports any that git does not track.

A-44's fourth C2 reading moved from GATED 4 / UNGATED 3 to GATED 5 / UNGATED 2
because `results/_a44_content_check4_discrimination.log` — a preserved red
capture — was staged into the tree the census searches. **A-44 attributed that
move to the tree rather than to the instrument, and that attribution was
correct and is not revised here.** What A-44 did not state is that the
attribution generalises: under D-45 a capture is preserved at *every* red, so
the census population grows at every red, and a census whose population grows
is an instrument that gives a different answer each time it is asked the same
question.

## The rule

1. **One path.** Captures preserved under D-45 are written under
   `refactor-patches/captures/`.
2. **Excluded by construction, not by pattern.** Every document-census and
   citation-sweep instrument derives its population from
   `results/_a46_capture_population.py`, which is the single definition of
   what those populations are. The exclusion is applied by the `git ls-files`
   pathspec that *produces* the listing, so a file under this root is never
   enumerated; no instrument filters names afterwards, and no instrument has
   an accessor that returns the unexcluded listing except the discrimination
   probe, which is named for that purpose and used by nothing else.
3. **Citations to captures still resolve.** The exclusion applies to the
   *population an instrument searches*, never to the *set against which a
   citation is resolved*. `citation_resolution_set()` deliberately includes
   this directory: an amendment that cites a preserved capture must be able to
   reach it, and a rule that made captures uncitable would defeat D-45.

## What this does and does not do

**Does:** make the C2 census count stationary by construction. It cannot move
again as reds accumulate.

**Does not:** retroactively validate A-44's GATED 5 / UNGATED 2, or convert
that provisional count into a settled one. **The declared gap from the finding
at the cap stands unchanged** — the finding's state is 4/3, the fourth reading
is a provisional 5/2, and the seven-round apparatus cap forbids a fifth
reading. **Stationarity by construction is what replaces the fifth reading**,
not a fifth reading by another name.

## Existing captures are not relocated

The preserved captures already in the tree — `a37`/`a38`/`a40`/`a41`/`a45` red
captures under `refactor-patches/phase3f/wp2/`, and the discrimination logs
under `results/` — **stay where they are**. Relocating them would break the
register's citations to them, which the citation sweep would then report as
unreachable: a rule adopted to stop an instrument moving would have moved two.
This root is the destination for captures preserved from A-46 forward.

## Inhabitants

| file | what it is |
| --- | --- |
| `a45_red_capture_copy.txt` | A byte-identical copy of `refactor-patches/phase3f/wp2/a45_red_capture.txt`, placed here as the exclusion's test fixture. The original is not moved (see above). `results/_a46_exclusion_discrimination.py` asserts the copy is byte-identical to its source, absent from the census population, and present in the citation resolution set. |
| `a46_exclusion_vacuous_red.log` | The discrimination capture required by D-41 clause 1 and preserved under D-45: `results/_a46_exclusion_discrimination.py` run against the tree *before* any file existed under this root, where it reports `EXCLUSION_VACUOUS` and exits 1. An exclusion that excludes nothing passes its own test trivially, so the test is only evidence if it has been seen to fail when the exclusion is empty. |
| `a46_citation_sweep_red.log` | `results/_a25_citation_sweep.py` red at A-46 on citations to files the commit had not yet staged. Preserved before staging them, per D-45. |
| `a47_corpus_identity_expiry.log` | **The scheduled expiry.** `refactor-patches/pin_corpus_identity.py` at A-47, reporting `DISTINCT_CANONICAL_HASHES 2`, `IDENTITY_HOLDS False`, exit 1 — A-38's retroactive rescue lapsing at the commit A-40 predicted would end it. |
| `a47_corpus_identity_test_red.log` | The same event reaching `tests/`: `tests/test_pin_corpus_identity.py`, `1 failed, 4 passed`, `PYTEST_EXIT:1`, captured before the baseline was re-declared. |
| `a47_fastlane_red_with_broken_exit_wrapper.log` | The fast lane carrying that failure — **and a capture defect of its own**, preserved as found. Its trailing `PYTEST_FAST_EXIT:0` is wrong: pytest's own summary two lines above reads `1 failed, 610 passed`. The shell wrapper, not the run, produced the false status, which is the D-41 clause-4 failure mode (a capture showing success beside a failure is worse than no capture). Kept unedited because a corrected copy would destroy the only evidence that the wrapper could do this. |
| `a47_citation_sweep_red.log` | `results/_a25_citation_sweep.py` red at A-47. Six of its seven unreachable citations were files the commit had not yet staged; the seventh, `baselines-2026-08-polygon/pins.json`, was a genuine defect — a shorthand path written without its `refactor-patches/` prefix — and was fixed in the record rather than allowlisted. |
| `a48_citation_sweep_red.log` | `results/_a25_citation_sweep.py` red at A-48, all four unreachable citations that commit's own not-yet-staged evidence. |
| `a50_ci10_red.log` | **CI-10's D-41 clause-1 red.** `tests/test_ci10_boolean_argument.py` against `f40591e`: `14 failed, 4 passed`, `PYTEST_EXIT:1`. Every behavioural row fails as `DID NOT RAISE ValueError` — the pre-change accept set was every Python object — and the two clause-identity rows fail on their own `ImportError`, separately, because they import `bstpp.config` INSIDE the test rather than at module level. A module-level import of API that does not exist yet takes the whole file down at collection, which is the red that proves nothing (D-41's minimal-revert clause). |
| `a50_ascii_sweep_red.log` | `results/_a26_ascii_sweep.py` red at A-50 *before* its `cox_background` sample was added: `10 evaluated, 1 unevaluated`, `SWEEP_EXIT:1`. Preserved because the fix is a one-line addition to `CLAUSE_SAMPLES`, and a one-line fix committed without its red is indistinguishable from a clause that was never unevaluated. |
| `a50_citation_sweep_red.log` | `results/_a25_citation_sweep.py` red at A-50, all eight unreachable citations that commit's own not-yet-staged evidence — no genuine defect among them, unlike A-47's seventh. |
| `a50_fastlane_red_missed_call_site.log` | The fast lane red at A-50 on a call site the change missed: `tests/test_identities.py::test_simulate_fully_reproducible_with_generator[cox]` parametrised over `[False, "cox"]`, so the `"cox"` leg passed the retired string into the new rejection. `1 failed, 634 passed`, `PYTEST_FAST_EXIT:1`. **This is the capture that justifies running the gate rather than reasoning about the change**: the argument's call sites had already been swept once and this one was invisible to the sweep, because the literal is in a `parametrize` list and never appears next to `cox_background=`. |
