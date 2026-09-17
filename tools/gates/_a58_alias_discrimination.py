"""The moved-path alias table must resolve real moves and nothing else.

WHY THIS EXISTS. An alias table is a mechanism for making an unreachable
citation reachable. That is exactly one edit away from being a mechanism for
making an unreachable citation LOOK reachable, and the two are
indistinguishable from a green citation sweep. So the table is checked for
what it must NOT do as well as what it must.

Nothing here mutates a file. The guard is exercised in memory against
fabricated tables, so the check cannot leave the working tree changed.

    python tools/gates/_a58_alias_discrimination.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _path_aliases import MOVED, resolve_alias  # noqa: E402

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())


def tracked_files() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, text=True,
                         capture_output=True, check=True)
    return {ln for ln in out.stdout.splitlines() if ln.strip()}


def honoured(cited: str, table: dict[str, str], tracked: set[str]) -> bool:
    alias = table.get(cited)
    return alias is not None and alias in tracked


def main() -> int:
    tracked = tracked_files()
    failures: list[str] = []

    print(f"ALIAS TABLE {len(MOVED)} entries; {len(tracked)} tracked files")
    print()

    print("CHECK 1 -- every alias target is tracked")
    for old, new in sorted(MOVED.items()):
        if new in tracked:
            print(f"  OK       {old} -> {new}")
        else:
            failures.append(f"1 alias target not tracked: {old} -> {new}")
            print(f"  MISSING  {old} -> {new}")
    print()

    print("CHECK 2 -- an alias pointing at an absent path is NOT honoured")
    fabricated = {"results/_gone.py": "tools/gates/DOES_NOT_EXIST.py"}
    if honoured("results/_gone.py", fabricated, tracked):
        failures.append("2 alias to an absent path was honoured")
        print("  HONOURED  -- the table can launder an unreachable citation")
    else:
        print("  REFUSED   -- absent target is not a resolution")
    print()

    print("CHECK 3 -- an unaliased, untracked citation stays unresolved")
    if honoured("results/_never_existed.py", MOVED, tracked):
        failures.append("3 unaliased untracked citation was honoured")
        print("  HONOURED  -- the pre-existing property was weakened")
    else:
        print("  REFUSED   -- unchanged from before the table existed")
    print()

    print("CHECK 4 -- no aliased old path is still tracked")
    for old in sorted(MOVED):
        if old in tracked:
            failures.append(f"4 aliased path still tracked: {old}")
            print(f"  STILL THERE  {old}")
    print(f"  {sum(1 for o in MOVED if o not in tracked)}/{len(MOVED)} old paths absent")
    print()

    print("CHECK 5 -- resolve_alias agrees with the table it exports")
    for old, new in sorted(MOVED.items()):
        got = resolve_alias(old)
        if got == new:
            print(f"  OK       {old}")
        else:
            failures.append(f"5 resolve_alias({old!r}) -> {got!r}, table has {new!r}")
            print(f"  DRIFT    {old}  function={got}  table={new}")
    if resolve_alias("results/_never_existed.py") is not None:
        failures.append("5 resolve_alias honoured an unlisted path")
        print("  DRIFT    unlisted path resolved")
    else:
        print("  OK       unlisted path returns None")
    print()

    if failures:
        print(f"FAIL {len(failures)}")
        for f in failures:
            print("  " + f)
        return 1
    print("PASS - the table resolves real moves, and refuses everything else.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
