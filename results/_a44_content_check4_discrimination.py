"""A-44: check 4 must be able to fail, once per sub-check.

WHY THIS EXISTS. Check 4 is the standing enumeration that replaces "somebody
looked". A tripwire never shown to trip is indistinguishable from an absent
one, and the content-checks gate was one of the three C2 counted as UNGATED --
green on every run in the series and never once observed red. This is both the
D-41 clause 1 demonstration for the new rows AND the gate's first capability
capture, and under the capture-then-fix practice adopted at A-44 it is
preserved rather than overwritten by the passing re-run.

FOUR MUTATIONS, ONE PER SUB-CHECK, each reproducing one enumerated instance of
the OP-29 class:

  M1 -> 4a   an item OPENED in prose whose §11 row is missing   (OP-21/OP-22)
  M2 -> 4b   an item CLOSED in prose whose row is not marked    (OP-20)
  M3 -> 4c   a decision cited with no §8 row                    (D-43)
  M4 -> 4d   a destination naming a work package with no entry  (D-44)

NOTHING IS MUTATED IN PLACE. Each run happens in a scratch tree holding a
mutated copy of the register plus a copy of the work-package entries, so the
repository is never in a broken state and the capture cannot leave residue --
which is the failure mode the A-40/A-41 revert captures had to guard by
restoring afterwards.

Usage:  python results/_a44_content_check4_discrimination.py
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECK = REPO / "results" / "_a25_content_checks.py"
REGISTER = REPO / "phase3_record.tex"
ENTRIES = REPO / "refactor-patches" / "phase3f"


def stage(mutate) -> tuple[int, str]:
    """Run the check over a scratch copy with `mutate` applied to the register."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "results").mkdir()
    shutil.copy(CHECK, tmp / "results" / CHECK.name)
    dst = tmp / "refactor-patches" / "phase3f"
    dst.mkdir(parents=True)
    for p in ENTRIES.glob("wp*_*entry.md"):
        shutil.copy(p, dst / p.name)
    text = REGISTER.read_text(encoding="utf-8")
    (tmp / "phase3_record.tex").write_text(mutate(text), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(tmp / "results" / CHECK.name)],
                          capture_output=True, text=True, cwd=str(tmp))
    return proc.returncode, proc.stdout + proc.stderr


def drop_row(item):
    def f(t):
        out = [ln for ln in t.splitlines()
               if not re.match(r"\s*(\\amdnew\{[^}]*\}\s*)*" + item + r"\s*&", ln)]
        return "\n".join(out)
    return f


def unmark_closed(item):
    def f(t):
        out = []
        for ln in t.splitlines():
            if re.match(r"\s*(\\amdnew\{[^}]*\}\s*)*" + item + r"\s*&", ln):
                ln = ln.replace("CLOSED", "closed-ish")
            out.append(ln)
        return "\n".join(out)
    return f


def cite_undefined(t):
    return t.replace(r"\end{document}",
                     "Text citing D-99 as authority.\n" + r"\end{document}")


def reroute(item, wp):
    def f(t):
        out = []
        for ln in t.splitlines():
            if re.match(r"\s*(\\amdnew\{[^}]*\}\s*)*" + item + r"\s*&", ln):
                head, _, tail = ln.rpartition("&")
                ln = head + "& " + wp + r" is where this goes \\"
            out.append(ln)
        return "\n".join(out)
    return f


CASES = [
    ("baseline (unmutated)", lambda t: t, 0, None),
    ("M1  4a: OP-27's §11 row deleted (it is declared Opened by A-40)",
     drop_row("OP-27"), 1, "4a"),
    ("M2  4b: OP-22's row no longer says CLOSED",
     unmark_closed("OP-22"), 1, "4b"),
    ("M3  4c: D-99 cited with no §8 row", cite_undefined, 1, "4c"),
    ("M4  4d: OP-19 re-pointed at WP11, which has no entry",
     reroute("OP-19", "WP11"), 1, "4d"),
]

print("A-44 -- check 4 discrimination. Exit codes are the CHECK's own,")
print("emitted as CONTENT_CHECKS_EXIT:n, which is the signature the C2 gate")
print("census searches for -- a capture nobody can find is not evidence.")
print("captured before any pipe. Nothing is mutated in place.")
print()
ok = True
for label, mut, want, sub in CASES:
    code, out = stage(mut)
    verdict = [ln.strip() for ln in out.splitlines()
               if ln.startswith("FAIL ") or "ALL CONTENT CHECKS PASS" in ln]
    detail = [ln.strip() for ln in out.splitlines()
              if sub and ln.strip().startswith(sub)]
    print(f"--- {label}")
    # The CHECK's own exit, named with the string the C2 gate census looks for
    # so this capture is findable as evidence rather than only readable (D-41).
    print(f"    CONTENT_CHECKS_EXIT:{code}   expected {want}")
    for v in verdict:
        print(f"    {v}")
    for d in detail:
        print(f"    {d}")
    if sub:
        fired = any(sub in ln for ln in out.splitlines() if ln.startswith("  " + sub)
                    or ln.strip().startswith(sub))
        print(f"    {sub}_FIRED:{fired}")
        ok = ok and fired
    ok = ok and (code == want)
    print()

print(f"ALL_SUBCHECKS_DISCRIMINATE:{ok}")
print("EXIT:0" if ok else "EXIT:1")
sys.exit(0 if ok else 1)
