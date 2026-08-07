"""A-52: two mechanical checks against the self-referential apparatus class.

THE CLASS, ENUMERATED AT SEVEN INSTANCES. An instrument fails in a way its own
design makes it unable to observe:

  1. the C2 census accepted a DOCUMENT DESCRIBING A RED as evidence of a gate's
     capability, so writing about a red counted the same as having one (A-44);
  2. the A-40/A-41 citation-sweep captures were overwritten by the passing
     re-run, so a gate KNOWN to be capable of failing could not demonstrate it
     (D-45 exists because of this);
  3. `pin_compare`'s walker counted a key present on one side only as NO DIFF,
     so a six-configuration candidate compared four and printed MATCH (OP-31);
  4. `NON_ASCII_PROBE` was mojibake-double-encoded and INVISIBLE TO ITS OWN
     SWEEP, because the sweep only asks whether the probe is non-ASCII and both
     spellings are (A-50);
  5. content check 4c resolved a citation by finding its row and could not see
     the `[[FILL]]` inside the cell it had just resolved (A-51/A-52);
  6. the anchor census died with UnicodeDecodeError reading the register it
     audits, under this machine's cp1252 default (A-51);
  7. A-50's prose complaining about placeholders SPELLED ONE and thereby added
     two to the count it was reporting (A-51/A-52).

THE COMMON SHAPE: the instrument's own substrate -- its fixture, its probe, its
encoding, its notation -- is not itself under any instrument. The remedy
generalises `mass_table_sha256`, which is the one place this repository already
got it right: the polygon mass table's hash is INSIDE the pinned record, so a
differently-built table is a DRIFT rather than a silent difference.

TWO CHECKS, AND THEN STOP. Instances 3, 5 and 7 are each closed by their own
targeted fix already landed. What remains general is:

  CHECK A -- SUBSTRATE PINNING. Every declared gate instrument is hash-pinned
  in a manifest computed by THIS module, which is not a gate and consumes none
  of them. A gate edited without its manifest entry being updated is red. This
  is what instances 1, 2 and 4 have in common: nothing outside the gate was
  watching the thing the gate depended on.

  CHECK B -- EXPLICIT ENCODING. Every document instrument opens files with an
  explicit `encoding=`. Instance 6 is this defect; instance 4 is its cousin one
  layer up. A platform-default decode makes an instrument give a different
  answer to a different reader, which is indistinguishable from the tree having
  changed.

Usage:
    python results/_a52_apparatus_checks.py            # verify
    python results/_a52_apparatus_checks.py --update   # rewrite the manifest

Exit 0 if both checks pass, 1 otherwise.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "results", "_a52_gate_manifest.json")

#: The declared gate population. Named explicitly rather than globbed: a glob
#: would silently adopt every new file in `results/`, and a population that
#: grows without a decision is the defect this file exists to stop.
GATE_INSTRUMENTS = [
    "results/_a25_citation_sweep.py",
    "results/_a25_content_checks.py",
    "results/_a26_ascii_sweep.py",
    "results/_a30_label_check.py",
    "results/_a46_capture_population.py",
    "results/_a46_exclusion_discrimination.py",
    "results/_a48_ruff_population.py",
    "results/_a51_anchor_census.py",
    "results/_c1_hypertarget_check.py",
    "refactor-patches/pin_compare.py",
    "refactor-patches/pin_corpus_identity.py",
    "refactor-patches/pin_check_v2.py",
]

#: Document instruments that must open files with an explicit encoding. Same
#: population as above minus the pin harness, which reads no documents.
ENCODING_SCOPE = [p for p in GATE_INSTRUMENTS if p.startswith("results/")]


def sha256(path):
    h = hashlib.sha256()
    with open(os.path.join(REPO, path), "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_a(update=False):
    """Substrate pinning: every gate hashed by an instrument that is not it."""
    print("CHECK A -- gate substrate pinning")
    print("  Each gate's bytes are hashed HERE. This module is not a gate and")
    print("  consumes none of them, so no instrument certifies itself --")
    print("  which is the `mass_table_sha256` property, generalised.")
    observed = {p: sha256(p) for p in GATE_INSTRUMENTS
                if os.path.isfile(os.path.join(REPO, p))}
    missing = [p for p in GATE_INSTRUMENTS
               if not os.path.isfile(os.path.join(REPO, p))]

    if update:
        with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
            json.dump({"gates": observed}, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"  MANIFEST WRITTEN: {len(observed)} gate(s)")
        return []

    if not os.path.isfile(MANIFEST):
        return ["A manifest absent -- run with --update to declare it"]
    with open(MANIFEST, encoding="utf-8") as fh:
        declared = json.load(fh).get("gates", {})

    problems = []
    for p in sorted(set(observed) | set(declared)):
        d, o = declared.get(p), observed.get(p)
        if d is None:
            problems.append(f"A undeclared gate: {p} ({o[:12]})")
        elif o is None:
            problems.append(f"A declared gate absent from the tree: {p}")
        elif d != o:
            problems.append(f"A gate CHANGED without its pin: {p} "
                            f"{d[:12]} -> {o[:12]}")
    for p in missing:
        print(f"    MISSING (declared in GATE_INSTRUMENTS): {p}")
    print(f"  gates pinned : {len(declared)}")
    print(f"  verified     : {len(observed)}")
    print(f"  problems     : {problems or 'none'}")
    return problems


class _OpenVisitor(ast.NodeVisitor):
    """Find `open(...)` calls with no `encoding=` keyword."""

    def __init__(self):
        self.bad = []

    def visit_Call(self, node):
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name == "open":
            kwargs = {k.arg for k in node.keywords if k.arg}
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            # Binary reads carry no encoding by definition and are correct.
            if "b" not in mode and "encoding" not in kwargs:
                self.bad.append(node.lineno)
        self.generic_visit(node)


def check_b():
    """Explicit encoding: no platform-default decode in a document tool."""
    print()
    print("CHECK B -- explicit encoding in document instruments")
    print("  A platform-default decode makes an instrument answer differently")
    print("  for a different reader, which is indistinguishable from the tree")
    print("  having changed. Binary modes are exempt by definition.")
    problems = []
    for path in ENCODING_SCOPE:
        full = os.path.join(REPO, path)
        if not os.path.isfile(full):
            continue
        with open(full, encoding="utf-8") as fh:
            try:
                tree = ast.parse(fh.read(), filename=path)
            except SyntaxError as exc:
                problems.append(f"B unparseable: {path}: {exc}")
                continue
        v = _OpenVisitor()
        v.visit(tree)
        if v.bad:
            problems.append(f"B open() without encoding= : {path}"
                            f" line(s) {', '.join(map(str, v.bad))}")
    print(f"  files scanned: {len(ENCODING_SCOPE)}")
    print(f"  problems     : {problems or 'none'}")
    return problems


def main() -> int:
    update = "--update" in sys.argv
    problems = check_a(update=update) + ([] if update else check_b())
    print()
    if problems:
        print(f"FAIL {len(problems)}")
        for p in problems:
            print("  " + p)
        print("APPARATUS_CHECKS_EXIT:1")
        return 1
    print("PASS -- gate substrate pinned by an instrument that consumes none")
    print("       of it; every document instrument declares its encoding.")
    print("APPARATUS_CHECKS_EXIT:0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
