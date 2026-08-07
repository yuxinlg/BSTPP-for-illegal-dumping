"""A-52: does content check 5 actually discriminate? Mutation test.

WHY THIS EXISTS. A-44 established that a gate is counted UNGATED until it has
been SEEN TO FAIL for the thing it claims to detect. Check 5 is adopted against
a declared baseline, so on the tree as committed it passes -- and a check that
has only ever passed is indistinguishable from a check that cannot fail. Check 4
was mutation-tested for exactly this reason (`_a44_content_check4_discrimination
.py`); this is the same treatment for check 5.

THE MUTATIONS, each targeting one clause of the check:

  1. OPERATIVE ANCHOR -- an anchor with an unclassified text placed inside a
     landed decision row. Must go RED regardless of the baseline: the baseline
     licenses anchors that defer a RECORD, never one that breaks a RULE.
  2. UNDECLARED ANCHOR -- one more `[[FILL: measure]]` in the authoritative
     document than the baseline declares. Must go RED. This is the clause that
     stops the anchor population growing quietly.
  3. AUTHORITATIVE-DOCUMENT SCOPE -- an anchor added to
     `docs/wp_dependency_graph.md`, which carries no LaTeX decision markers at
     all. Must go RED via the D-47 authoritative-document scope. Without that
     scope the check would exempt the file holding most of the anchors, so
     this row is what proves the scope is load-bearing rather than decorative.
  4. MENTION -- the same anchor text on a line carrying the mention marker.
     Must stay GREEN. This is the NEGATIVE control: a check that reddens on a
     mention would make the register unable to discuss its own subject matter,
     which is the defect A-50 hit and A-52 fixed.

Row 4 is the one that makes the other three meaningful. A check that fails on
everything discriminates nothing.

Usage:
    python results/_a52_check5_discrimination.py

Exit 0 if every mutation produced its expected verdict, 1 otherwise -- so the
capture records the probe's own status (D-41 clause 4).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

_RUN = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")

#: A landed decision row to mutate. Chosen because it is a real row that check 5
#: already scans; using a synthetic row would test the mutation, not the check.
ANCHOR_ROW_MARK = "\\amdnew{A-49} \\textbf{D-50} &"

MUTATIONS = [
    ("operative anchor in a landed decision row", "red",
     "phase3_record.tex", ANCHOR_ROW_MARK,
     ANCHOR_ROW_MARK + " [[FILL: an undeclared operative blank]]"),
    ("undeclared extra anchor in the authoritative document", "red",
     "docs/wp_dependency_graph.md", "| 1 | — (reopened under **D-43**) |",
     "| 1 | — (reopened under **D-43**) [[FILL: measure]] |"),
    ("anchor in the authoritative doc with no decision marker", "red",
     "docs/wp_dependency_graph.md", "## Exit criteria outside the graph",
     "## Exit criteria outside the graph [[FILL: unscoped and undeclared]]"),
    ("the SAME anchor text, marked as a mention", "green",
     "docs/wp_dependency_graph.md", "## Exit criteria outside the graph",
     "## Exit criteria outside the graph [[FILL: unscoped and undeclared]] "
     "<!-- census:mention -->"),
]


def run_check(tree):
    """Run the content checks inside `tree`; return (exit_code, stdout)."""
    out = subprocess.run([PY, os.path.join("results", "_a25_content_checks.py")],
                         cwd=tree, **_RUN)
    return out.returncode, (out.stdout or "") + (out.stderr or "")


def main() -> int:
    print("CHECK 5 DISCRIMINATION -- mutation test")
    print(f"  repo   : {REPO}")
    print()

    # Baseline: the tree as committed must be GREEN, or the mutations below
    # prove nothing -- a red baseline makes every mutation "red" for free.
    with tempfile.TemporaryDirectory() as tmp:
        tree = os.path.join(tmp, "base")
        _copy(REPO, tree)
        code, out = run_check(tree)
        print(f"  BASELINE (unmutated)                    exit {code}  "
              f"{'GREEN' if code == 0 else 'RED'}")
        if code != 0:
            print("  BASELINE IS RED -- every mutation below would be red for")
            print("  free and this probe would prove nothing. Aborting.")
            for line in out.splitlines():
                if line.startswith("FAIL") or line.startswith("  "):
                    print(f"      {line.rstrip()}")
            print("CHECK5_DISCRIMINATION_EXIT:1")
            return 1
        for line in out.splitlines():
            if line.strip().startswith("5 "):
                print(f"      {line.strip()}")
    print()

    failures = 0
    for name, expect, path, needle, replacement in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            tree = os.path.join(tmp, "t")
            _copy(REPO, tree)
            target = os.path.join(tree, path)
            with open(target, encoding="utf-8") as fh:
                text = fh.read()
            if needle not in text:
                print(f"  [SKIP ] {name}")
                print(f"          anchor text not found in {path} -- the probe "
                      "could not be applied, which is a probe defect")
                failures += 1
                continue
            with open(target, "w", encoding="utf-8", newline="") as fh:
                fh.write(text.replace(needle, replacement, 1))
            code, out = run_check(tree)
            got = "green" if code == 0 else "red"
            ok = got == expect
            failures += 0 if ok else 1
            print(f"  [{'OK   ' if ok else 'WRONG'}] {name}")
            print(f"          expected {expect.upper():<5} got {got.upper():<5}"
                  f" (exit {code})")
            for line in out.splitlines():
                if "5 operative" in line or "5 undeclared" in line:
                    print(f"            {line.strip()}")
    print()
    if failures:
        print(f"FAIL {failures} mutation(s) did not produce the expected verdict")
        print("CHECK5_DISCRIMINATION_EXIT:1")
        return 1
    print("PASS -- check 5 goes red for an operative anchor, for an undeclared")
    print("       anchor, and for an anchor in the authoritative document; and")
    print("       it stays green for a marked mention.")
    print("CHECK5_DISCRIMINATION_EXIT:0")
    return 0


def _copy(src, dst):
    """Copy just what the check reads. Copying the whole tree drags build/
    and the sbc logs along and makes each mutation cost seconds."""
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(os.path.join(src, "phase3_record.tex"), dst)
    os.makedirs(os.path.join(dst, "docs"), exist_ok=True)
    shutil.copy2(os.path.join(src, "docs", "wp_dependency_graph.md"),
                 os.path.join(dst, "docs"))
    shutil.copy2(os.path.join(src, "AGENTS.md"), dst)
    os.makedirs(os.path.join(dst, "results"), exist_ok=True)
    for f in ("_a25_content_checks.py", "_a51_anchor_census.py"):
        shutil.copy2(os.path.join(src, "results", f),
                     os.path.join(dst, "results"))
    # The census derives its population from `git ls-files`, so the mutated
    # tree needs to be a repository or the population comes back empty and
    # every mutation passes vacuously.
    # Check 4d (D-44) reads the WP entry FILES to decide which destinations
    # resolve. Without them the baseline goes red on 4d and every mutation
    # below would be red for free -- which the probe's baseline guard caught
    # on its first run, and which is the reason that guard exists.
    src_wp = os.path.join(src, "refactor-patches", "phase3f")
    dst_wp = os.path.join(dst, "refactor-patches", "phase3f")
    if os.path.isdir(src_wp):
        os.makedirs(dst_wp, exist_ok=True)
        for f in os.listdir(src_wp):
            if f.startswith("wp") and f.endswith("entry.md"):
                shutil.copy2(os.path.join(src_wp, f), dst_wp)
    # The census derives its population from `git ls-files`, so the mutated
    # tree must be a repository or the population comes back empty and every
    # mutation passes vacuously.
    subprocess.run(["git", "init", "-q"], cwd=dst, **_RUN)
    subprocess.run(["git", "add", "-A"], cwd=dst, **_RUN)


if __name__ == "__main__":
    sys.exit(main())
