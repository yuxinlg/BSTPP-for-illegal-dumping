"""Path aliases for instruments that moved, so frozen citations keep resolving.

WHY AN ALIAS TABLE AND NOT A REWRITE. The register is APPEND-ONLY. Closed
amendments cite instruments BY PATH, and those citations are inside text that
is not rewritten. Editing a closed amendment to keep a later refactor tidy
would make the record a description of the present rather than a record.

WHY NOT THIN ENTRYPOINTS AT THE OLD PATHS. A stub at each old path that
imports the new one resolves citations equally well and costs one real file
per alias, each of which is a second owner of the instrument's location
(OP-27) and each of which can drift from its target silently. This table is
one owner, and a citation sweep that consults it either resolves or does not.

WHAT THIS IS NOT. An alias does not make the old path exist.
`python results/_a26_ascii_sweep.py` fails after the move, correctly -- the
file is not there. The alias makes the CITATION resolvable; it does not make
a stale COMMAND work.

Usage:
    from _path_aliases import resolve_alias, repo_root
"""
from __future__ import annotations

from pathlib import Path

#: Old path -> new path, for instruments relocated by A-58.
#:
#: The pin battery (``refactor-patches/pin_check_v2.py``, ``pin_compare.py``,
#: ``pin_corpus_identity.py``) DID NOT MOVE and has no alias. It stays beside
#: the two baselines it reads, and ``pin_check_v2.py``'s path is additionally
#: keyed in ``pyproject.toml``'s per-file ruff ignore.
MOVED: dict[str, str] = {
    f"results/{name}": f"tools/gates/{name}"
    for name in (
        "_c1_hypertarget_check.py",
        "_a25_content_checks.py",
        "_a25_citation_sweep.py",
        "_a26_ascii_sweep.py",
        "_a30_label_check.py",
        "_a46_capture_population.py",
        "_a46_exclusion_discrimination.py",
        "_a48_ruff_population.py",
        "_a51_anchor_census.py",
        "_a52_apparatus_checks.py",
        "_a52_apparatus_discrimination.py",
        "_a52_check5_discrimination.py",
        "_a52_gate_manifest.json",
    )
}


def resolve_alias(path: str) -> str | None:
    """Return the current path for a cited one, or ``None`` if not aliased."""
    return MOVED.get(path)


def repo_root() -> Path:
    """Walk up to the directory that contains ``.git``.

    ``Path(__file__).parents[1]`` is correct in ``results/`` and silently
    wrong one directory deeper: REPO would become ``tools/`` with nothing
    raised, and the gates would measure an empty tree rather than fail.
    Counting parents is the defect; looking for the marker cannot be.
    """
    here = Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / ".git").exists():
            return cand
    raise SystemExit(f"could not find .git above {here}")
