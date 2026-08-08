"""A-46 / D-46: the declared capture root, and the populations that exclude it.

MEASUREMENT POINT (A-52) -- part of the DEFINITION, not of the procedure.

The population is read from the INDEX, so its value depends on where in the
staging sequence the reading is taken. That cannot be left to the operator: at
A-51 this instrument reported 599 because it ran before the last two evidence
files were staged, and the correct figure was 601. The error was caught by a
cross-check, not by inspection.

  * DECLARED POINT: post-staging, pre-commit. Every file the commit will
    contain is in the index, and nothing else is.
  * THE CROSS-CHECK IS PART OF TAKING THE READING, not an optional extra:
        git ls-files -- . ':(exclude)refactor-patches/captures/' | wc -l
    must equal CENSUS_POPULATION. A reading taken without it is not a reading.

THE SERIES IS NOT HOMOGENEOUS, and it is LABELLED rather than retro-corrected.
A-48's 550 was taken PRE-staging; A-50's 589 and A-51's 601 POST-staging. So
550 -> 564 -> 589 -> 601 mixes two measurement points and successive
differences across that boundary are not comparable. The earlier readings are
correct at their own point and are NOT restated: retro-correcting a published
count to make a series look smooth is how a series stops being evidence.

WHY THIS MODULE EXISTS. D-45 preserves a capture at every red. A preserved
capture is a text file, and two classes of instrument here search text files in
the tree: the document census
(``refactor-patches/phase3f/wp2/probe_a42_gate_capability_census.py``) and the
citation sweep (``results/_a25_citation_sweep.py``). A-44's fourth C2 reading
moved because one such capture was staged; A-44 attributed the move to the TREE
rather than to the instrument, correctly. D-46 is that attribution taken to its
conclusion: if the tree gains a capture at every red, the census answers the
same question differently every time it is asked, so the population must stop
depending on how many reds have happened.

BY CONSTRUCTION, NOT BY PATTERN -- and the difference is the whole point. A
pattern-based exclusion is a filter each instrument applies to a listing it has
already obtained: every instrument carries its own copy, the copies drift, and
a capture whose name the pattern does not anticipate is counted anyway. Here
the exclusion is a ``git ls-files`` PATHSPEC, so the listing is never produced
in the first place, and there is exactly one function that produces it. An
instrument importing this module has no accessor that returns the unexcluded
listing; the only one is named
``unfiltered_population_for_discrimination_only`` and is called by
``results/_a46_exclusion_discrimination.py`` and by nothing else.

THE ASYMMETRY IS DELIBERATE. ``census_population`` excludes the capture root.
``citation_resolution_set`` does NOT. They are different sets doing different
jobs: the first is what an instrument SEARCHES, the second is what a citation is
RESOLVED AGAINST. An amendment that cites a preserved capture must be able to
reach it, so excluding the root from the resolution set would make every
D-45 capture an unreachable citation -- a rule adopted to stop one instrument
moving would have broken another.

Usage (as a library):
    from _a46_capture_population import CAPTURE_ROOT, census_population
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The single declared path for captures preserved under D-45. Anything under
#: it is outside every document-census population by construction.
CAPTURE_ROOT = "refactor-patches/captures"

#: Pathspec magic that removes the capture root AT THE ENUMERATION. Written as
#: a directory prefix rather than a glob so that a capture in a future
#: subdirectory is excluded too, without anybody remembering to widen a pattern.
_EXCLUDE_PATHSPEC = f":(exclude){CAPTURE_ROOT}/"

#: The closed set of documents the citation sweep reads. A closed enumeration
#: cannot acquire a capture-root member by accident, which is the same
#: guarantee the pathspec gives the census, obtained a different way.
CITATION_SOURCE_PATHS = (
    "refactor-patches/pre-3f-stabilization/findings_ledger.md",
    "refactor-patches/pre-3f-stabilization/traceability_matrix.md",
    "phase3_record.tex",
)


class CapturePopulationError(RuntimeError):
    """Raised when a population cannot be produced, never returned as empty."""


def _ls_files(*pathspecs: str, repo: Path | None = None) -> list[str]:
    root = REPO if repo is None else Path(repo)
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", *pathspecs],
            cwd=root, text=True, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CapturePopulationError(
            f"git ls-files failed in {root}: {exc}") from exc
    return [line for line in out.splitlines() if line.strip()]


def census_population(repo: Path | None = None) -> list[str]:
    """Tracked files a document-census instrument may search.

    The capture root is removed by the pathspec that produces the listing, so
    the returned list has never contained a capture-root path at any point.
    """
    return _ls_files(".", _EXCLUDE_PATHSPEC, repo=repo)


def citation_resolution_set(repo: Path | None = None) -> set[str]:
    """Tracked files a citation may resolve to -- capture root INCLUDED.

    See the module docstring: excluding it here would make every preserved
    capture an unreachable citation.
    """
    return set(_ls_files(repo=repo))


def capture_root_members(repo: Path | None = None) -> list[str]:
    """Tracked files under the capture root. The exclusion's own subject."""
    return _ls_files(f"{CAPTURE_ROOT}/", repo=repo)


def unfiltered_population_for_discrimination_only(
        repo: Path | None = None) -> list[str]:
    """The listing WITHOUT the exclusion. For the mutation test alone.

    Named at length on purpose. An instrument that reached for this would be
    reintroducing exactly the coupling D-46 removes, and the name is the only
    thing standing between a reader and that mistake.
    """
    return _ls_files(".", repo=repo)


def citation_sources(repo: Path | None = None) -> dict[str, str]:
    """Read the closed citation-source set, asserting it avoids the root."""
    root = REPO if repo is None else Path(repo)
    inside = [p for p in CITATION_SOURCE_PATHS
              if p == CAPTURE_ROOT or p.startswith(CAPTURE_ROOT + "/")]
    if inside:
        raise CapturePopulationError(
            "citation sources must not live under the capture root: "
            + ", ".join(inside))
    return {p: (root / p).read_text(encoding="utf-8")
            for p in CITATION_SOURCE_PATHS}


def _main() -> int:
    """Print the capture population and its mandatory cross-check.

    WHY THIS EXISTS. This file is listed in `GATE_INSTRUMENTS` and hash-pinned
    in `_a52_gate_manifest.json`, but until now it had no `__main__`. Invoked
    directly -- which is how every other declared gate is run -- it printed
    nothing and exited 0, so it reported success having evaluated nothing. That
    is the masked-status hazard the gate rules forbid, arriving from the one
    direction no rule covered: not a wrapper hiding a real status, but a
    declared gate with no status to hide. Adding a reading here does not create
    a new instrument; it makes the declared one answerable.

    The cross-check is computed by a SECOND, INDEPENDENT `git ls-files`
    invocation rather than by reusing `_ls_files`. A cross-check that shares the
    code path it is checking is not a cross-check, and A-52 made this reading
    part of the DEFINITION of taking the measurement, not an optional extra.
    """
    import subprocess as _sp

    print("A-46 / D-46 CAPTURE POPULATION")
    print(f"  capture root                 : {CAPTURE_ROOT}")
    print(f"  declared exclusion pathspec  : {_EXCLUDE_PATHSPEC}")
    print("  measurement point            : post-staging, pre-commit (A-52)")
    print()

    census = census_population()
    citation = citation_resolution_set()
    root_members = capture_root_members()

    if not census:
        raise CapturePopulationError(
            "census population is empty; a population is never returned empty")

    try:
        raw = _sp.run(
            ["git", "ls-files", "--", ".", _EXCLUDE_PATHSPEC],
            cwd=REPO, text=True, capture_output=True, check=True).stdout
    except (OSError, _sp.CalledProcessError) as exc:
        raise CapturePopulationError(
            f"cross-check git ls-files failed in {REPO}: {exc}") from exc
    crosscheck = len([line for line in raw.splitlines() if line.strip()])

    print(f"  CENSUS_POPULATION            : {len(census)}")
    print(f"  git ls-files cross-check     : {crosscheck}")
    print(f"  citation resolution set      : {len(citation)}"
          "  (capture root INCLUDED, so captures stay citable)")
    print(f"  capture root members         : {len(root_members)}")
    print()

    agree = len(census) == crosscheck
    print(f"  CROSS_CHECK_HOLDS            : {agree}")
    if not agree:
        print("  READING REJECTED: population and cross-check disagree, so no")
        print("  number here is admissible as evidence.")
    print()
    print("  The exclusion is BY CONSTRUCTION: the capture root is removed by")
    print("  the pathspec that produces the listing, so no capture-root path")
    print("  was ever a member of the census population.")
    print(f"CAPTURE_POPULATION_EXIT:{0 if agree else 1}")
    return 0 if agree else 1


if __name__ == "__main__":
    import sys as _sys

    try:
        _sys.exit(_main())
    except CapturePopulationError as exc:
        print(f"CAPTURE_POPULATION_ERROR: {exc}")
        print("CAPTURE_POPULATION_EXIT:1")
        _sys.exit(1)
