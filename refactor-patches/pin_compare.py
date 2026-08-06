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

Usage:
    python refactor-patches/pin_compare.py results/_aNN_pins_candidate.json
    python refactor-patches/pin_compare.py <candidate> --baseline <other>

Exit status is 0 on MATCH and 1 on DRIFT or on a missing file, so a capture
records the comparison's own verdict rather than a shell's (D-41).
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


def compare(candidate: dict, baseline: dict) -> tuple[list, list]:
    """Return (per-config verdict lines, diffs). Walker unchanged from A-28."""
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
    for cfg in sorted(set(candidate) | set(baseline)):
        if cfg not in candidate:
            lines.append(f"{cfg}: MISSING FROM CANDIDATE")
        elif cfg not in baseline:
            lines.append(f"{cfg}: NEW IN CANDIDATE")
        else:
            before = len(diffs)
            walk(candidate[cfg], baseline[cfg], cfg)
            lines.append(f"{cfg}: {'MATCH' if len(diffs) == before else 'DRIFT'}")
    return lines, diffs


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

    lines, diffs = compare(load(cand_path), load(base_path))
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

    verdict = "MATCH" if not diffs else "DRIFT"
    print(f"PIN_DIFFS {len(diffs)} {verdict} "
          f"candidate={cand_path.name}:{sha256_of(cand_path)[:12]} "
          f"baseline={base_path.name}:{sha256_of(base_path)[:12]} "
          f"{'canonical' if canonical else 'NON-CANONICAL'}")
    for d in diffs[:20]:
        print(d)
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
