"""A-40: a standing check on the property A-38's retroactive rescue rests on.

WHAT IS BEING GUARDED. A-38 measured that every pin candidate in the tree
normalises to ONE canonical-JSON hash, equal to the canonical baseline's.
That single fact is what rescues twenty-four historical ``PIN_DIFFS 0``
claims whose artifacts do not record which files were read: with one content
group, the candidate-side ambiguity could not have changed a verdict. The
rescue is therefore not a fact about the discipline -- it is a fact about the
data, and it LAPSES the moment a second content group appears.

WHY IT IS A STANDING CHECK AND NOT A PARAGRAPH. A property that holds today
and silently stops holding is the failure mode this repository keeps
correcting. Written down once, its lapse is discovered by a later audit, if
at all. Run every commit, its lapse is discovered AT THE COMMIT THAT CAUSES
IT, which is the only moment at which the record can say so honestly.

THE EXPECTED TRIGGER IS ALREADY SCHEDULED. OP-24 requires a fifth pinned
configuration in POLYGON mode as a WP5 entry precondition. A polygon-mode pin
adds a configuration key the baseline does not have, so the first candidate
carrying it will not normalise to the baseline's hash -- this check goes red,
and it is SUPPOSED to. A re-baseline is a declared event; the correct response
is to re-baseline and record that A-38's rescue has expired as of that commit,
NOT to relax the check.

WHAT A FAILURE MEANS, PRECISELY. It means the corpus has more than one
content group, so from that commit forward the A-38 argument covers only the
commits before it. It does NOT mean a historical MATCH was wrong, and it does
not mean the pins have drifted -- ``pin_compare.py`` is what says that.

THE POPULATION IS NAMED, BECAUSE TWO RUNS WILL DISAGREE ON THE NUMBER (A-41).
This gate reads the working tree, so a fresh clone sees only the tracked
subset. The identity property holds on both populations, but a CI line
reporting 19 beside a local line reporting 31 is indistinguishable, at a
glance, from a real disagreement. Every run therefore prints
``VERDICT_POPULATION`` and both counts. Same discipline the ASCII sweep applies
to its raise-site denominator (A-39): a ratio whose population is unstated is
the defect, not the ratio.

Usage:
    python refactor-patches/pin_corpus_identity.py

Exit status is 0 while the identity holds and 1 once it has lapsed, so a
capture records this check's own verdict rather than a shell's (D-41).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANONICAL_BASELINE = REPO / "refactor-patches" / "baselines-2026-07" / "pins.json"
PATTERN = "*pins*candidate*.json"
RESULTS = REPO / "results"


def canonical_hash(path: Path) -> str:
    """Hash of the JSON CONTENT, not the bytes.

    Four distinct byte hashes across the corpus are BOM and line-ending
    differences, which no comparison has ever been sensitive to. The property
    being guarded is about values.
    """
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def tracked_candidates() -> list[Path] | None:
    """Candidates git knows about, or None if git cannot answer."""
    try:
        out = subprocess.run(
            ["git", "ls-files", f"results/{PATTERN}"],
            cwd=REPO, text=True, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [REPO / line for line in out.split("\n") if line.strip()]


def _group(paths) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for p in paths:
        groups.setdefault(canonical_hash(p), []).append(p.name)
    return groups


def measure() -> dict:
    """The census, over BOTH populations.

    A-41: the gate reads the working tree, so a fresh clone sees only the
    tracked subset and reports a different denominator for the same property.
    That is not a defect -- the identity holds on both -- but a CI line and a
    local line that disagree on a number are indistinguishable from a real
    disagreement. So both populations are measured and NAMED, in the same
    discipline the ASCII sweep applies to its raise-site denominator (A-39).
    The VERDICT is taken on ON_DISK, which is the superset and therefore the
    stronger claim.
    """
    on_disk = sorted(RESULTS.glob(PATTERN))
    tracked = tracked_candidates()
    base = canonical_hash(CANONICAL_BASELINE)
    groups = _group(on_disk)
    tracked_groups = None if tracked is None else _group(
        p for p in tracked if p.is_file())
    disk_names = {p.name for p in on_disk}
    missing = ([] if tracked is None
               else sorted(p.name for p in tracked if p.name not in disk_names))
    return {
        "on_disk": [p.name for p in on_disk],
        "tracked": None if tracked is None else [p.name for p in tracked],
        "tracked_not_on_disk": missing,
        "groups": groups,
        "distinct": len(groups),
        "tracked_distinct": None if tracked_groups is None else len(
            tracked_groups),
        "tracked_holds": None if tracked_groups is None else (
            len(tracked_groups) == 1 and base in tracked_groups),
        "baseline_canonical": base,
        "baseline_in_groups": base in groups,
        "holds": len(groups) == 1 and base in groups,
        "verdict_population": "ON_DISK",
    }


def report(m: dict) -> None:
    n_tracked = "unavailable" if m["tracked"] is None else len(m["tracked"])
    print("PIN_CORPUS_IDENTITY")
    print(f"  pattern                        : results/{PATTERN}")
    # POPULATION, stated. Two runs of this gate on the same property will give
    # different denominators (a clone has only the tracked files); naming which
    # set produced each number is what keeps that from reading as a conflict.
    print(f"  VERDICT_POPULATION             : {m['verdict_population']}")
    print(f"  population ON_DISK             : {len(m['on_disk'])} "
          "(working tree; includes uncommitted captures)")
    print(f"  population TRACKED             : {n_tracked} "
          "(git ls-files; what a fresh clone sees)")
    # The superset claim is MEASURED, not asserted: a tracked file absent from
    # disk would mean the population is not a superset and the stronger-claim
    # argument below does not hold.
    print(f"  tracked but not on disk        : "
          f"{m['tracked_not_on_disk'] or 'none (ON_DISK is a superset)'}")
    print("  (the verdict is taken on ON_DISK because it is the superset and")
    print("   therefore the stronger claim; TRACKED is reported beside it so a")
    print("   clone and a working tree do not appear to disagree)")
    print(f"  baseline                       : {CANONICAL_BASELINE.name}")
    print(f"  baseline canonical hash        : {m['baseline_canonical'][:16]}")
    print(f"  DISTINCT_CANONICAL_HASHES      : {m['distinct']}   [ON_DISK]")
    print(f"  DISTINCT_CANONICAL_HASHES_TRACKED : "
          f"{m['tracked_distinct'] if m['tracked_distinct'] is not None else 'unavailable'}"
          "   [TRACKED]")
    for h, names in sorted(m["groups"].items(), key=lambda kv: -len(kv[1])):
        mark = "== BASELINE" if h == m["baseline_canonical"] else "!= baseline"
        print(f"    {h[:16]}  {mark}  n={len(names)}")
        if h != m["baseline_canonical"]:
            for name in names:
                print(f"      {name}")
    print(f"  IDENTITY_HOLDS                 : {m['holds']}   [ON_DISK]")
    print(f"  IDENTITY_HOLDS_TRACKED         : "
          f"{m['tracked_holds'] if m['tracked_holds'] is not None else 'unavailable'}"
          "   [TRACKED]")
    if m["holds"]:
        print("  A-38's retroactive rescue of the 24 historical PIN_DIFFS 0")
        print("  claims still stands: one content group, so the candidate-side")
        print("  ambiguity could not have changed a verdict.")
    else:
        print("  LAPSED. From this commit forward A-38's retroactive rescue")
        print("  covers only the commits BEFORE this one. Re-baseline and")
        print("  record the expiry; do not relax this check. If the cause is")
        print("  OP-24's polygon-mode pin, that is the expected trigger.")


def main() -> int:
    m = measure()
    report(m)
    return 0 if m["holds"] else 1


if __name__ == "__main__":
    sys.exit(main())
