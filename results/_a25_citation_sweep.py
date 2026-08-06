"""A-25 / C3 unreachable-citation sweep.

A coverage record citing a file that is not in the repository is structurally the same
defect as A-22 citing an uncommitted section: the reader cannot reach the evidence. This
sweep extracts every path-like citation from the findings ledger, the traceability matrix,
and Part II of the register, and reports which are not tracked by git.

Exit 1 if any untracked citation is not on the ALLOWED list below.

D-46 (A-46) AND THE TWO SETS THIS SWEEP USES. The SOURCE set -- the documents
searched for citations -- comes from ``_a46_capture_population`` as a closed
enumeration that cannot acquire a capture-root member. The RESOLUTION set --
what a citation is checked against -- is the full ``git ls-files`` and
DELIBERATELY INCLUDES the capture root, because an amendment that cites a
capture preserved under D-45 must be able to reach it. Excluding the root from
the resolution set would turn every preserved capture into an unreachable
citation: a rule adopted to stop one instrument moving would have broken this
one. The asymmetry is the point, so it is stated rather than left to be
inferred from two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _a46_capture_population import (  # noqa: E402
    CAPTURE_ROOT,
    CITATION_SOURCE_PATHS,
    citation_resolution_set,
    citation_sources,
)

LEDGER, MATRIX, RECORD = CITATION_SOURCE_PATHS

# Citations that are deliberately to objects NOT in the repository. Each is legitimate only
# because the surrounding text says so; the reason is recorded here so the exemption is not
# silent.
ALLOWED = {
    "docs/phase3_baseline_and_decisions.tex":
        "A-21: historical snapshot removed from the tree; git history preserves it. Labelled in place.",
    "tests/test_config_matrix.py":
        "A-24: the closeout's proposed deliverable, explicitly recorded as NOT implemented.",
    "baselines-2026-07/pins.json":
        "A-24 shorthand inside a sentence that names the refactor-patches/ prefix; the full path is cited too.",
    "refactor-patches/sbc1/test_sbc_smoke.py":
        "A-25 / ledger: named precisely BECAUSE it is untracked — it is the contamination that produced the wrong 582.",
    "refactor-patches/sbc2/test_sbc_smoke_v2.py":
        "as above.",
    "refactor-patches/test_sbc_smoke_v3.py":
        "as above.",
    "bstpp/cox_hawkes_shared.py":
        "A-39/A-41: named precisely BECAUSE it is absent. It looks like a path in THIS "
        "repo and is not one: it is untracked in a downstream working checkout on "
        "another machine, and the scope limit A-40 records is exactly that no claim "
        "here covers it. A reader who could reach it would be reading a different file.",
    "replication/cox_hawkes_offset.py":
        "A-41: a path in yuxinlg/Illegal-Dumping at f5f1382, read from a read-only "
        "clone that was discarded. Its measured dependency surface is committed as "
        "refactor-patches/phase3f/wp2/a41_vendored_dependency_extract.json, which IS "
        "tracked, so the citation's evidence is reachable even though the file is not.",
    "Illegal-Dumping/replication/README.md":
        "A-40: a path in yuxinlg/Illegal-Dumping, a DIFFERENT repository. Cited to "
        "flag a defect that is theirs to fix (the README installs from this fork's "
        "URL while 03_analysis.py requires a module that URL does not supply). It "
        "can never be tracked here, and naming the file is the point of the flag.",
}

PATH_RE = re.compile(
    r"[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+"
    r"\.(?:py|json|md|tex|txt|log|yml|yaml|csv|jsonl|toml|pdf|png)"
)


def de_latex(text: str) -> str:
    text = text.replace("\\_", "_")
    text = text.replace("\\allowbreak{}", "")
    text = text.replace("\\allowbreak", "")
    return text


def main() -> int:
    # Resolution set: the FULL listing, capture root included. See the module
    # docstring -- a preserved capture must stay citable.
    tracked = citation_resolution_set()

    raw = citation_sources()
    record = de_latex(raw[RECORD])
    part_ii = record[record.index("\\hypertarget{a-1}{%"):]

    # Source set: a closed enumeration, so it cannot acquire a capture-root
    # member as captures accumulate.
    sources = {
        LEDGER: de_latex(raw[LEDGER]),
        MATRIX: de_latex(raw[MATRIX]),
        f"{RECORD} (Part II)": part_ii,
    }
    print(f"D-46 populations: {len(sources)} closed source document(s); "
          f"resolution set {len(tracked)} tracked file(s), "
          f"{CAPTURE_ROOT}/ INCLUDED so captures stay citable.")
    print()

    unresolved: list[tuple[str, str]] = []
    for name, text in sources.items():
        cited = sorted(set(PATH_RE.findall(text)))
        missing = [p for p in cited if p not in tracked]
        print(f"--- {name}")
        print(f"    {len(cited)} path citations, {len(cited) - len(missing)} tracked, "
              f"{len(missing)} not tracked")
        for p in missing:
            if p in ALLOWED:
                print(f"    ALLOWED   {p}")
                print(f"              reason: {ALLOWED[p]}")
            else:
                exists = Path(p).exists()
                print(f"    UNTRACKED {p}   ({'exists on disk' if exists else 'DOES NOT EXIST'})")
                unresolved.append((name, p))
        print()

    if unresolved:
        print(f"FAIL {len(unresolved)} unreachable citation(s):")
        for name, p in unresolved:
            print(f"  {p}   (cited in {name})")
        return 1
    print("PASS — every path citation is either tracked or an explicitly recorded exemption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
