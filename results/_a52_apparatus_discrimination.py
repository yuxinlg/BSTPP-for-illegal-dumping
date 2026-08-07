"""A-52: do the two apparatus checks discriminate? Mutation test.

Both checks in `_a52_apparatus_checks.py` PASS on the tree as committed, and a
check that has only ever passed is indistinguishable from one that cannot fail
(A-44). Check B in particular passed on its FIRST RUN, having found nothing --
which is exactly the shape of the silent pass it was written to prevent, so it
needs its red demonstrated more than check A does.

MUTATIONS:
  A1 EDITED GATE     -- change a byte in a pinned gate without touching the
                        manifest. Must go RED. This is the whole of check A.
  A2 UNDECLARED GATE -- add a gate file the manifest does not list. Must go
                        RED, so the population cannot grow silently.
  B1 BARE open()     -- introduce `open(path)` with no encoding in a document
                        instrument. Must go RED.
  B2 BINARY open()   -- introduce `open(path, "rb")`, which correctly carries
                        no encoding. Must stay GREEN. NEGATIVE CONTROL: a check
                        that reddened here would force a meaningless
                        `encoding=` onto every binary read, and the next author
                        would delete the check rather than the noise.

Usage:
    python results/_a52_apparatus_discrimination.py
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

CHECKS = os.path.join("results", "_a52_apparatus_checks.py")


def run(tree):
    out = subprocess.run([PY, CHECKS], cwd=tree, **_RUN)
    return out.returncode, (out.stdout or "") + (out.stderr or "")


def _fixture(dst):
    """A minimal tree: the checks, the manifest, and the pinned gates."""
    os.makedirs(os.path.join(dst, "results"), exist_ok=True)
    os.makedirs(os.path.join(dst, "refactor-patches"), exist_ok=True)
    sys.path.insert(0, os.path.join(REPO, "results"))
    import importlib
    mod = importlib.import_module("_a52_apparatus_checks")
    for rel in mod.GATE_INSTRUMENTS + [CHECKS.replace("\\", "/")]:
        src = os.path.join(REPO, rel)
        if os.path.isfile(src):
            tgt = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            shutil.copy2(src, tgt)
    subprocess.run([PY, CHECKS, "--update"], cwd=dst, **_RUN)
    return mod


def main() -> int:
    print("APPARATUS CHECKS DISCRIMINATION -- mutation test")
    print()
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "base")
        _fixture(base)
        code, out = run(base)
        print(f"  BASELINE (fixture, manifest freshly written)  exit {code}  "
              f"{'GREEN' if code == 0 else 'RED'}")
        if code != 0:
            print("  BASELINE IS RED -- mutations would be red for free.")
            for ln in out.splitlines():
                if ln.startswith("  A ") or ln.startswith("  B ") \
                        or ln.startswith("FAIL"):
                    print(f"      {ln.rstrip()}")
            print("APPARATUS_DISCRIMINATION_EXIT:1")
            return 1
    print()

    failures = 0
    rows = [
        ("A1 edited gate, manifest untouched", "red", "edit_gate"),
        ("A2 gate present but undeclared", "red", "undeclared_gate"),
        ("B1 open() with no encoding", "red", "bare_open"),
        ("B2 open(path, 'rb') -- binary, correctly no encoding", "green",
         "binary_open"),
    ]
    for name, expect, kind in rows:
        with tempfile.TemporaryDirectory() as tmp:
            tree = os.path.join(tmp, "t")
            _fixture(tree)
            target = os.path.join(tree, "results", "_a26_ascii_sweep.py")

            if kind == "edit_gate":
                with open(target, "a", encoding="utf-8") as fh:
                    fh.write("\n# mutation: one byte the manifest never saw\n")
            elif kind == "undeclared_gate":
                # Add a file to the DECLARED population without re-pinning, by
                # extending the list inside the copied checks module.
                cf = os.path.join(tree, CHECKS)
                with open(cf, encoding="utf-8") as fh:
                    src = fh.read()
                extra = os.path.join(tree, "results", "_zz_new_gate.py")
                with open(extra, "w", encoding="utf-8") as fh:
                    fh.write("# a new gate nobody pinned\n")
                src = src.replace('    "results/_a25_citation_sweep.py",',
                                  '    "results/_zz_new_gate.py",\n'
                                  '    "results/_a25_citation_sweep.py",', 1)
                with open(cf, "w", encoding="utf-8", newline="") as fh:
                    fh.write(src)
            elif kind == "bare_open":
                with open(target, "a", encoding="utf-8") as fh:
                    fh.write("\ndef _mutation():\n"
                             "    return open('x.txt').read()\n")
            elif kind == "binary_open":
                with open(target, "a", encoding="utf-8") as fh:
                    fh.write("\ndef _mutation():\n"
                             "    return open('x.bin', 'rb').read()\n")

            if kind in ("bare_open", "binary_open", "edit_gate"):
                # These edit a pinned gate, which check A would also flag. Re-pin
                # so the row tests ONLY its own clause -- otherwise B1 and B2
                # would both be red via check A and prove nothing about B.
                if kind != "edit_gate":
                    subprocess.run([PY, CHECKS, "--update"], cwd=tree, **_RUN)

            code, out = run(tree)
            got = "green" if code == 0 else "red"
            ok = got == expect
            failures += 0 if ok else 1
            print(f"  [{'OK   ' if ok else 'WRONG'}] {name}")
            print(f"          expected {expect.upper():<5} got {got.upper():<5}"
                  f" (exit {code})")
            for ln in out.splitlines():
                s = ln.strip()
                if s.startswith("A ") or s.startswith("B "):
                    print(f"            {s}")
    print()
    if failures:
        print(f"FAIL {failures} mutation(s) gave the wrong verdict")
        print("APPARATUS_DISCRIMINATION_EXIT:1")
        return 1
    print("PASS -- check A reddens on an edited gate and on an undeclared one;")
    print("       check B reddens on a bare open() and stays green on binary.")
    print("APPARATUS_DISCRIMINATION_EXIT:0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
