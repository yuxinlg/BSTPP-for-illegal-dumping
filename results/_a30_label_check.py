"""A-30: the config-invariant renumbering is complete and did not touch the
model identities.

Two claims, both mechanical:
  1. No `I1`-`I6` reference to the CONFIG sequence survives.
  2. The MODEL identity labels are exactly where they were.

The second matters more than the first. A global substitution would have
silently rewritten `test_identities.py`, `likelihood.py` and the matrix's own
identity rows, and nothing else in the gate set would have noticed.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABEL = re.compile(r'(?<![A-Za-z-])I[1-6]\b')

# Files whose every I1-I6 was the CONFIG sequence: none may survive.
CONFIG_ONLY = [
    "bstpp/config.py",
    "bstpp/polygon_mass.py",
    "tests/test_lane_b_config_matrix.py",
    "refactor-patches/phase3f/rebaseline_record_a27.md",
    "refactor-patches/phase3f/rebaseline_record_a28.md",
]

# Files that legitimately keep MODEL-identity labels; the count is pinned so a
# later careless sed is caught.
MODEL_BASELINE = "426d60a"
MODEL_FILES = [
    "bstpp/likelihood.py",
    "tests/test_identities.py",
    "tests/test_likelihood_atoms.py",
    "tests/test_clipped_support.py",
]


def at_baseline(path: str) -> str:
    return subprocess.run(["git", "show", f"{MODEL_BASELINE}:{path}"],
                          capture_output=True, text=True, encoding="utf-8",
                          cwd=REPO, check=True).stdout


def main() -> int:
    fails = 0
    print("1. CONFIG sequence fully renumbered (no surviving I1-I6)")
    print("   Exemption: the supersession note in each rebaseline record quotes")
    print("   the OLD labels on purpose ('was written as I1-I6'). Those lines")
    print("   are blockquoted and skipped; every other line must be clean.")
    for rel in CONFIG_ONLY:
        lines = (REPO / rel).read_text(encoding="utf-8").split("\n")
        text = "\n".join(ln for ln in lines
                         if not ln.lstrip().startswith(">"))
        hits = LABEL.findall(text)
        status = "OK" if not hits else f"FAIL {len(hits)} left: {sorted(set(hits))}"
        print(f"   {status:<40} {rel}")
        if hits:
            fails += 1

    print()
    print(f"2. MODEL identities untouched vs {MODEL_BASELINE}")
    for rel in MODEL_FILES:
        now = LABEL.findall((REPO / rel).read_text(encoding="utf-8"))
        was = LABEL.findall(at_baseline(rel))
        ok = now == was
        print(f"   {'OK' if ok else 'FAIL':<40} {rel}  ({len(was)} label(s))")
        if not ok:
            print(f"      was {was}\n      now {now}")
            fails += 1

    print()
    print("3. Matrix model-identity rows still present")
    matrix = (REPO / "refactor-patches/pre-3f-stabilization/"
              "traceability_matrix.md").read_text(encoding="utf-8")
    for row in ["| I1 |", "| I3/I4 |", "| I5 |", "| I6 |", "| I9 |",
                "| I10 |", "| I11 |", "| I12 |"]:
        ok = row in matrix
        print(f"   {'OK' if ok else 'FAIL':<40} row {row.strip('| ')}")
        if not ok:
            fails += 1

    print()
    print("4. CI-1..CI-6 all defined and used")
    for n in range(1, 7):
        found = sum(1 for rel in CONFIG_ONLY + ["phase3_record.tex"]
                    if f"CI-{n}" in (REPO / rel).read_text(encoding="utf-8"))
        ok = found > 0
        print(f"   {'OK' if ok else 'FAIL':<40} CI-{n} in {found} file(s)")
        if not ok:
            fails += 1

    print()
    if fails:
        print(f"FAIL {fails} label check(s)")
        return 1
    print("PASS - config sequence renumbered CI-1..CI-6; model identities intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
