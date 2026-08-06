"""A-40, re-declared at A-47: every pin candidate belongs to a declared era.

THE PROPERTY HAS CHANGED ONCE, ON SCHEDULE. READ THIS BEFORE CITING IT.

*What it was, A-40 to A-46.* A-38 measured that every pin candidate in the
tree normalised to ONE canonical-JSON hash, equal to the canonical baseline's.
That single fact rescued twenty-four historical ``PIN_DIFFS 0`` claims whose
artifacts do not record which files were read: with one content group, the
candidate-side ambiguity could not have changed a verdict. The rescue was
never a fact about the discipline -- it was a fact about the data, and it
LAPSED the moment a second content group appeared.

*The lapse, dated.* It appeared at A-47, from the cause A-40 named in advance:
OP-24's polygon-mode pin adds two configuration keys the 2026-07 baseline does
not carry, so no candidate carrying them can normalise to that baseline's
hash. **A-38's retroactive rescue therefore covers commits BEFORE A-47 and
nothing after.** That is recorded, not repaired; no later measurement can give
it back, and this module does not pretend otherwise.

*What it is from A-47 on.* Not a rescue -- a forward drift check with the same
strength as the original, over a population that now has two eras:

    every content group in the corpus equals one of the DECLARED baselines.

One declared baseline was the A-40 form of this sentence; two is the A-47 form.
The check that matters is unchanged in kind: an UNDECLARED group is red. What
would be a relaxation is declaring a third baseline to accommodate a group
rather than explaining it, so ``DECLARED_BASELINES`` is short, ordered, and
each entry says which commit introduced it and what it covers.

WHY IT IS A STANDING CHECK AND NOT A PARAGRAPH. A property that holds today
and silently stops holding is the failure mode this repository keeps
correcting. Written down once, its lapse is discovered by a later audit, if
at all. Run every commit, its lapse is discovered AT THE COMMIT THAT CAUSES
IT, which is the only moment at which the record can say so honestly. That is
what happened at A-47, and the capture is preserved at
``refactor-patches/captures/a47_corpus_identity_expiry.log``.

WHAT A FAILURE MEANS, PRECISELY. It means the corpus holds a content group
that no declared baseline explains, so some capture in ``results/`` was
produced by a harness or a tree this baseline set does not describe. It does
NOT mean a historical MATCH was wrong, and it does not mean the pins have
drifted -- ``pin_compare.py`` is what says that.

WHAT THIS CHECK DOES NOT DO, SINCE THE ERAS BECAME TWO (OP-31). It says every
candidate matches SOME declared baseline. It cannot say that a given
``PIN_DIFFS 0 MATCH`` artifact was taken against the RIGHT one, because the
artifact is the only place that is recorded. Under one group that distinction
did not exist; under two it does, and nothing here enforces it.

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
FORWARD_BASELINE = (REPO / "refactor-patches" / "baselines-2026-08-polygon"
                    / "pins.json")

#: The declared eras, oldest first. A group matching none of these is red.
#: Adding an entry is a re-declaration and belongs in the register with the
#: commit that causes it -- it is the one edit here that CAN be a relaxation,
#: so it is deliberately the most visible line in the file.
DECLARED_BASELINES = (
    ("2026-07 canonical", CANONICAL_BASELINE,
     "four configurations; the era A-38's rescue was measured over"),
    ("2026-08 polygon (A-47)", FORWARD_BASELINE,
     "six configurations; adds the two notched-domain polygon/rectangle keys"),
)

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


def declared_hashes() -> dict[str, str]:
    """Canonical hash -> era label, for every declared baseline present."""
    found = {}
    for label, path, _covers in DECLARED_BASELINES:
        if path.is_file():
            found[canonical_hash(path)] = label
    return found


def _undeclared(groups: dict, declared: dict[str, str]) -> list[str]:
    return sorted(h for h in groups if h not in declared)


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
    declared = declared_hashes()
    groups = _group(on_disk)
    tracked_groups = None if tracked is None else _group(
        p for p in tracked if p.is_file())
    disk_names = {p.name for p in on_disk}
    missing = ([] if tracked is None
               else sorted(p.name for p in tracked if p.name not in disk_names))
    undeclared = _undeclared(groups, declared)
    return {
        "on_disk": [p.name for p in on_disk],
        "tracked": None if tracked is None else [p.name for p in tracked],
        "tracked_not_on_disk": missing,
        "groups": groups,
        "distinct": len(groups),
        "declared": declared,
        "undeclared": undeclared,
        "tracked_distinct": None if tracked_groups is None else len(
            tracked_groups),
        "tracked_holds": None if tracked_groups is None else not _undeclared(
            tracked_groups, declared),
        "baseline_canonical": base,
        "baseline_in_groups": base in groups,
        "holds": not undeclared,
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
    # The declared eras are printed in full every run. A re-declaration is the
    # only edit to this gate that could be a relaxation, so it is never
    # inferable from a silent pass -- it is on the face of every capture.
    print(f"  DECLARED_BASELINES             : {len(DECLARED_BASELINES)}")
    for label, path, covers in DECLARED_BASELINES:
        state = (canonical_hash(path)[:16] if path.is_file()
                 else "MISSING -- every group it explains is now undeclared")
        print(f"    {label:<24} {state}")
        print(f"      {path.relative_to(REPO).as_posix()}  ({covers})")
    print(f"  DISTINCT_CANONICAL_HASHES      : {m['distinct']}   [ON_DISK]")
    print(f"  UNDECLARED_GROUPS              : {len(m['undeclared'])}   "
          "[ON_DISK]")
    print(f"  DISTINCT_CANONICAL_HASHES_TRACKED : "
          f"{m['tracked_distinct'] if m['tracked_distinct'] is not None else 'unavailable'}"
          "   [TRACKED]")
    for h, names in sorted(m["groups"].items(), key=lambda kv: -len(kv[1])):
        mark = m["declared"].get(h, "UNDECLARED")
        print(f"    {h[:16]}  {mark}  n={len(names)}")
        if h not in m["declared"]:
            for name in names:
                print(f"      {name}")
    print(f"  IDENTITY_HOLDS                 : {m['holds']}   [ON_DISK]")
    print(f"  IDENTITY_HOLDS_TRACKED         : "
          f"{m['tracked_holds'] if m['tracked_holds'] is not None else 'unavailable'}"
          "   [TRACKED]")
    # A-38's rescue is expired at A-47 and no run of this gate can restore it.
    # Printing that on every capture, pass or fail, is the point: a green line
    # here must never be read as the one-content-group claim it used to be.
    print("  A-38's retroactive rescue EXPIRED at A-47 and is not restored by")
    print("  a pass here. It covers commits before A-47; this gate now checks")
    print("  only that every group belongs to a declared era.")
    if not m["holds"]:
        print("  UNDECLARED GROUP. Some capture in results/ was produced by a")
        print("  harness or tree no declared baseline describes. Explain it,")
        print("  or re-declare deliberately -- do not relax this check. See")
        print("  pin_compare.py for whether the pins themselves drifted.")


def main() -> int:
    m = measure()
    report(m)
    return 0 if m["holds"] else 1


if __name__ == "__main__":
    sys.exit(main())
