"""Canonical golden-pin comparison. ONE implementation, not one per amendment.

WHY THIS FILE EXISTS. A-37's comparison read A-34's candidate path, reported
``PIN_DIFFS 0 MATCH``, and described a different tree. It was self-caught, but
nothing prevented it: the practice was a fresh ``sed`` copy of the previous
amendment's comparator with the candidate path edited by hand, so a missed
edit produces a verdict that is TEXTUALLY IDENTICAL to a correct one. A MATCH
against the wrong file and a MATCH against the right file were the same six
characters. That is the failure this file removes.

THE FIX IS THAT A VERDICT DESCRIBES ITSELF. Every run prints, in the verdict
block, both sides of the comparison by PATH and by SHA-256 of the bytes read,
and states whether the baseline was the canonical one or was overridden. The
hash is of the file that was actually opened, so a capture is interpretable
from the artifact alone, without knowing what the operator typed.

THE OBSERVED DEFECT WAS ON THE CANDIDATE SIDE. Across the whole PIN_DIFFS
series there has only ever been one baseline path
(``refactor-patches/baselines-2026-07/pins.json``); the candidate side is the
one that accumulates a new file per amendment. Both are therefore reported,
and the candidate is a REQUIRED argument with no default -- there is no
canonical candidate, and inventing one would recreate the guessing this file
exists to stop.

NON-CANONICAL BASELINES REQUIRE --baseline, DELIBERATELY. A re-baseline is a
declared event, so it should cost a flag and announce itself in the output
rather than being reachable by editing a string. Runs that omit the flag get
the canonical baseline and say so.

A VERDICT ALSO DESCRIBES ITS POPULATION (A-48, closing OP-31). The same defect
one axis over: the verdict said which files it read and not how many
CONFIGURATIONS it compared. The walker treats a key present on one side only
as no diff, so once pin 5 added two configurations the canonical baseline does
not carry, the routine run compared four of six, found nothing, and printed
the six characters a complete comparison prints -- silent about the polygon
regime, which is the one pin 5 was built to cover. `NEW IN CANDIDATE` was
printed above the verdict, where a runbook grepping the verdict line does not
look. Every verdict line therefore now carries `compared=n/m`,
`candidate_only=[...]` and `baseline_only=[...]`, on clean lines too, and the
word `MATCH` is reserved for a comparison that covered the whole union.
Anything short of that reads `PARTIAL`, so a gate line written when the corpus
had four configurations stops reporting success rather than quietly narrowing.

Usage:
    python refactor-patches/pin_compare.py results/_aNN_pins_candidate.json
    python refactor-patches/pin_compare.py <candidate> --baseline <other>

Exit status is 0 when no compared value differs and 1 on DRIFT or a missing
file, so a capture records the comparison's own verdict rather than a shell's
(D-41). A PARTIAL exits 0 ON PURPOSE: adding configurations is a legitimate
declared event, and making it nonzero would leave a per-commit gate red on
every commit after a re-baseline -- the failure mode A-47 had to undo one
gate over. The incompleteness is carried by the verdict word, which is read
by the same eye and does not decay into background noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANONICAL_BASELINE = REPO / "refactor-patches" / "baselines-2026-07" / "pins.json"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    # utf-8-sig: some captures were written by a shell that emitted a BOM.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare(candidate: dict, baseline: dict) -> tuple[list, list, dict]:
    """Return (per-config lines, diffs, population). Walker unchanged from A-28.

    The third return value is A-48's addition and the walker's own admission:
    a configuration present on one side only is **not compared**, and counting
    it as no diff is what let a four-of-six comparison print the word a
    complete one prints. The walker is not changed to treat it as a diff --
    a re-baseline legitimately adds keys -- so the count is reported instead.
    """
    diffs: list = []

    def walk(x, y, path=""):
        if type(x) is not type(y):
            diffs.append((path, type(x).__name__, type(y).__name__))
            return
        if isinstance(x, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x:
                    diffs.append((path + "." + k, "MISSING", y[k]))
                elif k not in y:
                    diffs.append((path + "." + k, x[k], "MISSING"))
                else:
                    walk(x[k], y[k], path + "." + k)
        elif isinstance(x, list):
            if len(x) != len(y):
                diffs.append((path, f"len{len(x)}", f"len{len(y)}"))
            else:
                for i, (u, v) in enumerate(zip(x, y)):
                    walk(u, v, f"{path}[{i}]")
        else:
            if x != y:
                diffs.append((path, x, y))

    lines = []
    union = sorted(set(candidate) | set(baseline))
    population = {"union": len(union), "compared": 0,
                  "candidate_only": [], "baseline_only": []}
    for cfg in union:
        if cfg not in candidate:
            lines.append(f"{cfg}: MISSING FROM CANDIDATE")
            population["baseline_only"].append(cfg)
        elif cfg not in baseline:
            lines.append(f"{cfg}: NEW IN CANDIDATE")
            population["candidate_only"].append(cfg)
        else:
            before = len(diffs)
            walk(candidate[cfg], baseline[cfg], cfg)
            lines.append(f"{cfg}: {'MATCH' if len(diffs) == before else 'DRIFT'}")
            population["compared"] += 1
    return lines, diffs, population


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidate",
                    help="this tree's pin capture (no default, by design)")
    ap.add_argument("--baseline", default=None,
                    help="non-canonical baseline; announced in the output")
    args = ap.parse_args(argv)

    cand_path = Path(args.candidate).resolve()
    if args.baseline is None:
        base_path, canonical = CANONICAL_BASELINE.resolve(), True
    else:
        base_path, canonical = Path(args.baseline).resolve(), False

    for label, p in (("candidate", cand_path), ("baseline", base_path)):
        if not p.is_file():
            print(f"PIN_COMPARE_ERROR {label} not found: {p}")
            return 1

    lines, diffs, pop = compare(load(cand_path), load(base_path))
    for line in lines:
        print(line)

    # The provenance block. Printed on EVERY run, MATCH or DRIFT -- a block
    # that appeared only on failure would make "not printed" and "not run"
    # the same observable, which is the A-35 lesson about the hazard section.
    print()
    print("PIN_PROVENANCE")
    print(f"  candidate : {cand_path}")
    print(f"  candidate_sha256 : {sha256_of(cand_path)}")
    print(f"  baseline  : {base_path}")
    print(f"  baseline_sha256  : {sha256_of(base_path)}")
    print(f"  baseline_source  : "
          f"{'CANONICAL' if canonical else 'NON-CANONICAL (--baseline given)'}")
    if not canonical:
        print(f"  canonical_would_be : {CANONICAL_BASELINE}")
    print()

    # THE VERDICT WORD IS ABOUT THE WHOLE POPULATION, NOT THE COMPARED SUBSET.
    # `MATCH` is reserved for a comparison that covered every configuration on
    # both sides; anything else is PARTIAL, and the word is what a runbook
    # grepping for `PIN_DIFFS 0 MATCH` will fail to find. That failure is the
    # feature: a gate line written when the corpus had four configurations
    # must stop reporting success once it covers four of six.
    one_sided = pop["candidate_only"] + pop["baseline_only"]
    if diffs:
        verdict = "DRIFT"
    elif one_sided:
        verdict = "PARTIAL"
    else:
        verdict = "MATCH"
    print(f"PIN_DIFFS {len(diffs)} {verdict} "
          f"compared={pop['compared']}/{pop['union']} "
          f"candidate_only=[{','.join(pop['candidate_only'])}] "
          f"baseline_only=[{','.join(pop['baseline_only'])}] "
          f"candidate={cand_path.name}:{sha256_of(cand_path)[:12]} "
          f"baseline={base_path.name}:{sha256_of(base_path)[:12]} "
          f"{'canonical' if canonical else 'NON-CANONICAL'}")
    for d in diffs[:20]:
        print(d)
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
