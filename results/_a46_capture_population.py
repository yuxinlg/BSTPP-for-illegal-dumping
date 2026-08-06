"""A-46 / D-46: the declared capture root, and the populations that exclude it.

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
