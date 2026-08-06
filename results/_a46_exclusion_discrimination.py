"""A-46 / D-46: the capture-root exclusion, mutation-tested.

WHY A MUTATION TEST AND NOT AN ASSERTION. "The census population is unchanged
when a capture is added under the capture root" is satisfied by an exclusion
that works AND by an exclusion that excludes nothing at all -- if no tracked
file lives under the root, adding nothing to the population is what an absent
exclusion does too. So the confirmation is only evidence when it is paired with
the measurement that the exclusion is load-bearing: how many tracked files it
actually removes, and what the population would be without it. This is A-27's
finding (an unreached guard is indistinguishable from an absent one) applied to
an exclusion instead of a raise.

WHAT IT CHECKS.
  1. The exclusion is NOT vacuous: at least one tracked file lives under the
     capture root, so the two populations genuinely differ. Exit 1 if not --
     and that red is the discrimination capture, preserved at
     ``refactor-patches/captures/a46_exclusion_vacuous_red.log``.
  2. Adding the A-45 preserved capture under the root leaves the census
     population unchanged, and would have grown it by one without the
     exclusion. Measured as a difference between the two accessors on one
     tree, not by staging and unstaging.
  3. The fixture is a byte-identical copy of the capture it claims to copy.
  4. Captures remain CITABLE: the copy is absent from the census population and
     present in the citation resolution set.
  5. Every named document-census and citation-sweep instrument derives its
     population from ``_a46_capture_population``. An exclusion defined in one
     module and not imported by the instruments is a comment.

Exit status is this script's own (D-41): 0 when the exclusion is real and
complete, 1 otherwise.

Usage:  python results/_a46_exclusion_discrimination.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _a46_capture_population import (  # noqa: E402
    CAPTURE_ROOT,
    CITATION_SOURCE_PATHS,
    REPO,
    capture_root_members,
    census_population,
    citation_resolution_set,
    unfiltered_population_for_discrimination_only,
)

CAPTURE_SUFFIXES = (".txt", ".log", ".json")

#: The fixture and the capture it copies. Named as a pair so the copy cannot
#: quietly stop being a copy.
FIXTURE = f"{CAPTURE_ROOT}/a45_red_capture_copy.txt"
FIXTURE_SOURCE = "refactor-patches/phase3f/wp2/a45_red_capture.txt"

#: Instruments that must consume the shared definition. The census and the
#: sweep are the two classes D-46 names.
CONSUMERS = (
    "refactor-patches/phase3f/wp2/probe_a42_gate_capability_census.py",
    "results/_a25_citation_sweep.py",
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, text=True,
                          capture_output=True).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    failures: list[str] = []

    print("A46_EXCLUSION_DISCRIMINATION")
    print(f"  repo          : {REPO}")
    print(f"  git_rev       : {_git('rev-parse', '--short', 'HEAD')}")
    dirty = [ln for ln in _git("status", "--porcelain").splitlines()
             if not ln.startswith("??")]
    print(f"  tracked_dirty : {len(dirty)}")
    for ln in dirty:
        print(f"    {ln}")
    print()

    print("DEFINITIONS -- stated because a count without one is not a number")
    print("  CENSUS POPULATION: the tracked files a document-census instrument")
    print("    may search. Produced by `git ls-files -- . ':(exclude)"
          f"{CAPTURE_ROOT}/'`, so the capture root is removed AT THE")
    print("    ENUMERATION and no capture-root path is ever in the list.")
    print("  ELIGIBLE CAPTURE FILES: the census population restricted to")
    print(f"    {CAPTURE_SUFFIXES} -- A-42's rule that a capture is a file")
    print("    PRODUCED BY RUNNING an instrument, never a document written")
    print("    about one. This is the denominator C2 was read against.")
    print("  CITATION RESOLUTION SET: what a citation resolves against. The")
    print("    capture root is INCLUDED here on purpose; see check 4.")
    print()

    excluded = census_population()
    unfiltered = unfiltered_population_for_discrimination_only()
    members = capture_root_members()
    resolution = citation_resolution_set()
    eligible = [p for p in excluded if p.endswith(CAPTURE_SUFFIXES)]
    eligible_unfiltered = [p for p in unfiltered
                           if p.endswith(CAPTURE_SUFFIXES)]

    print("MEASURED")
    print(f"  census population (capture root excluded) : {len(excluded)}")
    print(f"  unfiltered population (exclusion removed) : {len(unfiltered)}")
    print(f"  tracked files under {CAPTURE_ROOT}/ : {len(members)}")
    for p in members:
        print(f"    {p}")
    print(f"  ELIGIBLE CAPTURE FILES in census population   : {len(eligible)}")
    print(f"  eligible capture files without the exclusion  : "
          f"{len(eligible_unfiltered)}")
    print()

    # -- 1. the exclusion is load-bearing ---------------------------------
    print("CHECK 1 -- the exclusion is not vacuous")
    delta = len(unfiltered) - len(excluded)
    print(f"  files removed by the exclusion : {delta}")
    if delta == 0:
        print("  EXCLUSION_VACUOUS -- no tracked file lives under "
              f"{CAPTURE_ROOT}/, so the two populations are identical and")
        print("  every check below would pass against an exclusion that does")
        print("  nothing. This is the red the mutation test exists to produce.")
        failures.append("exclusion is vacuous: nothing is excluded")
    else:
        print(f"  OK -- the two populations differ by {delta}")
    print()

    # -- 2. the A-45 capture, added under the root, moves nothing ---------
    print("CHECK 2 -- the A-45 preserved capture is inert in the census")
    if FIXTURE in members:
        # The counterfactual is exact, not approximate: the fixture is in the
        # unfiltered listing and not in the excluded one, so the population
        # WITHOUT the exclusion is larger by precisely this file.
        in_excluded = FIXTURE in excluded
        in_unfiltered = FIXTURE in unfiltered
        print(f"  {FIXTURE}")
        print(f"    in census population   : {in_excluded}  (must be False)")
        print(f"    in unfiltered listing  : {in_unfiltered}  (must be True)")
        print("    eligible-capture population with the exclusion    : "
              f"{len(eligible)}")
        print("    eligible-capture population without the exclusion : "
              f"{len(eligible_unfiltered)}  "
              f"(+{len(eligible_unfiltered) - len(eligible)})")
        if in_excluded:
            failures.append(f"{FIXTURE} reached the census population")
        if not in_unfiltered:
            failures.append(f"{FIXTURE} is not tracked at all")
        if len(eligible_unfiltered) <= len(eligible):
            failures.append(
                "the fixture does not grow the unexcluded eligible "
                "population, so its inertness demonstrates nothing")
    else:
        print(f"  ABSENT -- {FIXTURE} is not tracked")
        failures.append(f"fixture not tracked: {FIXTURE}")
    print()

    # -- 3. the copy is a copy --------------------------------------------
    print("CHECK 3 -- the fixture is byte-identical to the capture it copies")
    src, dst = REPO / FIXTURE_SOURCE, REPO / FIXTURE
    if src.is_file() and dst.is_file():
        h_src, h_dst = _sha256(src), _sha256(dst)
        print(f"  {FIXTURE_SOURCE}  sha256 {h_src[:16]}")
        print(f"  {FIXTURE}  sha256 {h_dst[:16]}")
        if h_src != h_dst:
            failures.append("fixture is not byte-identical to its source")
        else:
            print("  OK -- identical")
    else:
        missing = [p for p, q in ((FIXTURE_SOURCE, src), (FIXTURE, dst))
                   if not q.is_file()]
        print(f"  MISSING {missing}")
        failures.append(f"cannot compare fixture to source: {missing}")
    print()

    # -- 4. captures stay citable -----------------------------------------
    print("CHECK 4 -- exclusion applies to what is SEARCHED, not to what a")
    print("           citation RESOLVES against")
    citable = FIXTURE in resolution
    print(f"  {FIXTURE} in citation resolution set : {citable}  "
          "(must be True)")
    print(f"  citation source documents ({len(CITATION_SOURCE_PATHS)}), none "
          "under the capture root:")
    for p in CITATION_SOURCE_PATHS:
        under = p.startswith(CAPTURE_ROOT + "/")
        print(f"    {p}   under_capture_root={under}")
        if under:
            failures.append(f"citation source under the capture root: {p}")
    if not citable:
        failures.append(
            "a preserved capture is not citable; D-45 evidence would be "
            "unreachable")
    print()

    # -- 5. the instruments actually consume the shared definition --------
    print("CHECK 5 -- both instrument classes derive their population here")
    for rel in CONSUMERS:
        path = REPO / rel
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        ok = "_a46_capture_population" in text
        print(f"  {'OK  ' if ok else 'FAIL'} {rel}")
        if not ok:
            failures.append(
                f"{rel} does not import the shared population definition, so "
                "its exclusion is by pattern or absent")
    print()

    if failures:
        print(f"FAIL {len(failures)}")
        for f in failures:
            print(f"  {f}")
        print("A46_EXCLUSION_EXIT:1")
        return 1
    print("PASS -- the capture root is excluded by construction, the "
          "exclusion is load-bearing,")
    print("       and preserved captures remain citable.")
    print(f"CENSUS_POPULATION {len(excluded)}")
    print(f"ELIGIBLE_CAPTURE_FILES {len(eligible)}")
    print("A46_EXCLUSION_EXIT:0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
