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
