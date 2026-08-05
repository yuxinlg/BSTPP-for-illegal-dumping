"""A-25 / C3 unreachable-citation sweep.

A coverage record citing a file that is not in the repository is structurally the same
defect as A-22 citing an uncommitted section: the reader cannot reach the evidence. This
sweep extracts every path-like citation from the findings ledger, the traceability matrix,
and Part II of the register, and reports which are not tracked by git.

Exit 1 if any untracked citation is not on the ALLOWED list below.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

LEDGER = "refactor-patches/pre-3f-stabilization/findings_ledger.md"
MATRIX = "refactor-patches/pre-3f-stabilization/traceability_matrix.md"
RECORD = "phase3_record.tex"

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
    tracked = {
        line.strip()
        for line in subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        if line.strip()
    }

    record = de_latex(Path(RECORD).read_text(encoding="utf-8"))
    part_ii = record[record.index("\\hypertarget{a-1}{%"):]

    sources = {
        LEDGER: de_latex(Path(LEDGER).read_text(encoding="utf-8")),
        MATRIX: de_latex(Path(MATRIX).read_text(encoding="utf-8")),
        f"{RECORD} (Part II)": part_ii,
    }

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
