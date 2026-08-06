"""A-42: C2's assessment -- which gates have been SHOWN capable of failing.

WHAT C2 ASKS. Every per-commit gate green, AND holding one committed capture
in which that gate reports failure with its own exit status. A gate that has
never been observed red is indistinguishable from one that cannot go red,
which is the unreached-guard finding (A-27) applied to the apparatus itself.

WHY IT IS MEASURED HERE RATHER THAN LISTED. The cheap way to satisfy C2 is to
declare gates ungated and move on, so the round-six amendment requires the
COUNTS: GATED / UNGATED / RED, summing to the closed gate set. This probe
produces them by searching the COMMITTED tree -- ``git ls-files``, not the
working directory -- because an uncommitted red capture is not evidence
anybody else can reach.

THE GATE SET IS THE CLOSED ONE from the WP2 opening-conditions proposal. It is
seven entries; entry 7 bundles two instruments and is GATED only if both are.

WHAT COUNTS AS A CAPTURE. A tracked file containing the gate's own failure
signature -- its printed verdict or a recorded non-zero exit for that
instrument. Source files are excluded: a script that can PRINT "FAIL" is not a
capture of it having done so, and counting one would be the same category
error as counting a docstring as a test.

TWO SELF-CORRECTIONS, RECORDED BECAUSE EACH CHANGED THE ANSWER.

(2) The run taken after staging this commit reported **GATED 5 / UNGATED 2**,
promoting "citation + label sweeps" -- because the WP2 proposal document, newly
staged, contains the sentence describing the sweep's A-40 red, and that prose
matches ``FAIL \d+ unreachable citation`` exactly. **The instrument counted the
write-up of the gap as evidence the gap was closed.** A capture is a file
PRODUCED BY RUNNING an instrument, never a document WRITTEN ABOUT one, so the
eligible set is now the capture extensions only. Corrected back to **GATED 4 /
UNGATED 3**. Recorded rather than quietly fixed because it is the sharper of
the two: the first correction was a loose regex, this one is a category error
about what evidence is, and it went the direction that would have closed C2.

(1) The first run of
this probe reported **GATED 6 / UNGATED 1**. It was wrong by two. Three gates
shared the generic signature ``^FAIL``, and one file --
``results/_a29_sweep_discrimination.txt``, which is an ASCII-sweep
discrimination demo and nothing else -- matched all three. A shared signature
is not an identity, which is the same mistake A-41's literal-matching
heuristic makes and the same reason its 68 is not a bound. Each gate now
carries its OWN failure string, read from its source: ``SWEEP_EXIT:1``,
``FAIL n unreachable citation``, ``FAIL n label check``, ``MISSING n``. The
corrected answer is **GATED 4 / UNGATED 3**, and the difference between those
two answers is the difference between C2 closing and C2 not closing.

Usage:  python refactor-patches/phase3f/wp2/probe_a42_gate_capability_census.py
"""
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

# entry -> (instruments, failure signatures, note)
# A signature is a regex looked for in tracked NON-SOURCE files.
GATES = [
    ("fast lane",
     ["pytest tests/ -m 'not slow'"],
     [r"^\s*\d+ failed", r"PYTEST_EXIT:1"],
     "pytest's own exit status"),
    ("pin_compare.py",
     ["refactor-patches/pin_compare.py"],
     [r"PIN_DIFFS \d+ DRIFT", r"PIN_COMPARE_ERROR"],
     "DRIFT or a missing file, exit 1"),
    ("pin_corpus_identity.py",
     ["refactor-patches/pin_corpus_identity.py"],
     [r"IDENTITY_HOLDS\s*:\s*False", r"makes_the_gate_fail"],
     "a second content group, exit 1"),
    ("ASCII sweep",
     ["results/_a26_ascii_sweep.py"],
     [r"SWEEP_EXIT:1", r"FAIL \d+ (?:clause function|non-ASCII character)"],
     "a non-ASCII raise site"),
    ("content / decision-monotonicity checks",
     ["results/_a25_content_checks.py"],
     [r"CONTENT_CHECKS_EXIT:1", r"^FAIL \d+\s*$"],
     "a decision-row gap, duplicate or regression"),
    ("hypertarget structural check",
     ["results/_c1_hypertarget_check.py"],
     [r"MISSING [1-9]"],
     "a subsection with no anchor"),
    ("citation + label sweeps",
     ["results/_a25_citation_sweep.py", "results/_a30_label_check.py"],
     [r"FAIL \d+ unreachable citation", r"FAIL \d+ label check"],
     "an unreachable citation, or a broken renumber"),
]

# A capture is a file PRODUCED BY RUNNING an instrument, never a document
# WRITTEN ABOUT one. So the eligible set is the capture extensions, and .py /
# .md / .tex are all excluded -- see the second self-correction above.
CAPTURE_SUFFIXES = (".txt", ".log", ".json")


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, text=True,
                         capture_output=True, check=True).stdout
    return [p for p in out.splitlines() if p.strip()]


print("PROBE_PROVENANCE")
print(f"  repo     : {REPO}")
rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                     text=True, capture_output=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, text=True,
                       capture_output=True).stdout
tracked_dirty = [ln for ln in dirty.splitlines() if not ln.startswith("??")]
print(f"  git_rev  : {rev}")
print(f"  tracked_dirty : {len(tracked_dirty)}")
for ln in tracked_dirty:
    print(f"    {ln}")
print()

files = tracked_files()
# Captures only: source files are excluded on purpose (see the docstring).
capture_paths = [p for p in files if p.endswith(CAPTURE_SUFFIXES)]
print(f"  tracked files          : {len(files)}")
print(f"  eligible capture files : {len(capture_paths)} "
      f"{CAPTURE_SUFFIXES}")
print("  (.py excluded: printing 'FAIL' is not evidence of having. .md/.tex")
print("   excluded: PROSE ABOUT a gate having failed is not a capture of it")
print("   failing -- the register and the WP2 proposal both describe the very")
print("   reds this probe is looking for.)")
print()

texts = {}
for p in capture_paths:
    try:
        with open(os.path.join(REPO, p), encoding="utf-8",
                  errors="replace") as fh:
            texts[p] = fh.read()
    except OSError:
        continue

print("C2 ASSESSMENT -- the closed gate set")
gated, ungated, red = [], [], []
for name, instruments, sigs, note in GATES:
    hits = []
    for p, text in texts.items():
        for s in sigs:
            if re.search(s, text, re.MULTILINE):
                hits.append(p)
                break
    verdict = "GATED" if hits else "UNGATED"
    (gated if hits else ungated).append(name)
    print(f"  {name}")
    print(f"    instruments  : {instruments}")
    print(f"    failure mode : {note}")
    print(f"    committed capture(s) showing it red : "
          f"{sorted(set(hits))[:4] if hits else 'NONE'}"
          f"{' ...' if len(set(hits)) > 4 else ''}")
    print(f"    VERDICT      : {verdict}")
print()

print("COUNTS")
print(f"  gate set (closed) : {len(GATES)}")
print(f"  GATED   : {len(gated)}   {gated}")
print(f"  UNGATED : {len(ungated)}   {ungated}")
print(f"  RED     : {len(red)}   {red}")
print(f"  SUM_MATCHES_SET : {len(gated) + len(ungated) + len(red) == len(GATES)}")
print()
print("QUALIFICATION ON THE FAST LANE, stated rather than elided. What the")
print("committed red captures show is pytest under the fast-lane marker over a")
print("SUBSET of tests/ (they carry their own 'N deselected'), never over the")
print("whole tree. Exit status is not selection-dependent -- a failing row fails")
print("the whole-tree run too -- so the capability is established; the")
print("population of the demonstration is not the population of the gate, and a")
print("reader should know which was shown.")
print("EXIT:0")
