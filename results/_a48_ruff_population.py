"""A-48: run ruff over a DECLARED population, against a MEASURED baseline.

WHY THIS EXISTS. "ruff on touched files" is not a population. At A-47 six
files were linted against a baseline built from three, because three of the
six were new and have no HEAD version -- so "13 against a baseline of 10"
compared unlike sets and the difference was arithmetic rather than a finding.
Nobody was misled, but nothing in the artifact said which files were on each
side, and a count whose population is unstated is the defect this repository
keeps naming (A-39's raise-site denominator, A-41's ON_DISK/TRACKED split).

WHAT IT REPORTS. For every path given:

  * its state -- MODIFIED (exists at HEAD, differs), NEW (no HEAD version),
    or UNCHANGED;
  * its finding counts by rule code, after and (where a HEAD version exists)
    before;
  * the delta by rule code, so INTRODUCED and INHERITED are separated
    mechanically instead of by recollection.

NEW files are reported with `baseline: NONE (new file)` and are excluded from
the delta arithmetic, not folded into it with a zero. A new file's findings
are all introduced by definition, and hiding that inside a subtraction is how
the A-47 count came out looking like a comparison.

THE BASELINE IS HEAD'S CONTENT UNDER HEAD'S CONFIG, and both halves matter.
Per-file ignores are keyed on the path, so the HEAD copies are written to a
temp tree at their ORIGINAL relative paths -- a copy linted under a scratch
name silently loses its ignore and reports a baseline the file never had. And
`HEAD:pyproject.toml` is copied in beside them, so a commit that adds an
ignore shows up as a NEGATIVE delta on the file it exempts, with the exemption
visible as a change rather than absorbed into the measuring instrument. A
baseline that used the new config would make an ignore self-justifying.

Usage:
    python results/_a48_ruff_population.py <path> [<path> ...]

Exit status is 0 if every path was measurable and 1 if any HEAD read or ruff
invocation failed, so a capture records this script's own verdict (D-41). It
is deliberately NOT 1 on findings: ruff is not a gate here (the WP2 opening
conditions say so explicitly), it is a measurement whose population must be
legible.
"""
from __future__ import annotations

import collections
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

#: ruff colours its output even into a pipe, so the counts arrive wrapped in
#: SGR escapes. Stripping them is not cosmetic: a parser that missed them
#: returns zero findings and reports a clean tree, which is a silent pass.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: `ruff check --statistics` emits "<count>\t<CODE>\t[*] <description>".
_STAT = re.compile(r"^\s*(\d+)\s+([A-Z]+\d+)\s")


def _ruff_statistics(target: Path, cwd: Path) -> dict[str, int]:
    proc = subprocess.run(
        [PY, "-m", "ruff", "check", "--statistics", "--no-cache", str(target)],
        cwd=cwd, text=True, capture_output=True)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"ruff failed on {target}:\n{proc.stderr}")
    counts: dict[str, int] = {}
    for line in _ANSI.sub("", proc.stdout).splitlines():
        m = _STAT.match(line)
        if m:
            counts[m.group(2)] = int(m.group(1))
    # ruff exits 1 iff it found something, so the two channels must agree.
    # Without this, a parser that silently stopped matching would report a
    # clean file -- the exact silent pass this script exists to prevent, one
    # level down. `--statistics` prints NOTHING on a clean file, so an empty
    # dict is only trustworthy beside exit 0.
    if bool(counts) != (proc.returncode == 1):
        raise RuntimeError(
            f"ruff exit {proc.returncode} disagrees with {len(counts)} parsed "
            f"statistic line(s) for {target}:\n{proc.stdout}")
    return counts


def _head_bytes(rel: str) -> bytes | None:
    proc = subprocess.run(["git", "show", f"HEAD:{rel}"],
                          cwd=REPO, capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def measure(rel_paths: list[str]) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        base_root = Path(tmp)
        head_cfg = _head_bytes("pyproject.toml")
        if head_cfg is None:
            raise RuntimeError("HEAD has no pyproject.toml; baseline config "
                               "cannot be reconstructed")
        (base_root / "pyproject.toml").write_bytes(head_cfg)
        for rel in rel_paths:
            live = REPO / rel
            if not live.is_file():
                raise RuntimeError(f"not a file in the working tree: {rel}")
            after = _ruff_statistics(live, cwd=REPO)
            head = _head_bytes(rel)
            if head is None:
                rows.append({"path": rel, "state": "NEW", "after": after,
                             "before": None})
                continue
            dest = base_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(head)
            before = _ruff_statistics(dest, cwd=base_root)
            state = "UNCHANGED" if head == live.read_bytes() else "MODIFIED"
            rows.append({"path": rel, "state": state, "after": after,
                         "before": before})
    return rows


def report(rows: list[dict]) -> None:
    print("RUFF_POPULATION")
    print(f"  declared population : {len(rows)} file(s), named below")
    print("  baseline            : each file's own HEAD version, linted with "
          "HEAD's pyproject.toml")
    print("                        (so a newly added per-file ignore shows as "
          "a negative delta,")
    print("                         not as an absence)")
    print()
    introduced: dict[str, int] = collections.defaultdict(int)
    inherited: dict[str, int] = collections.defaultdict(int)
    new_file: dict[str, int] = collections.defaultdict(int)
    for row in rows:
        after_total = sum(row["after"].values())
        print(f"  {row['path']}")
        print(f"    state    : {row['state']}")
        if row["before"] is None:
            print("    baseline : NONE (new file; every finding is introduced "
                  "by definition and is excluded from the delta)")
            print(f"    after    : {after_total} "
                  f"{dict(sorted(row['after'].items())) or '{}'}")
            for code, n in row["after"].items():
                new_file[code] += n
            print()
            continue
        before_total = sum(row["before"].values())
        print(f"    before   : {before_total} "
              f"{dict(sorted(row['before'].items())) or '{}'}")
        print(f"    after    : {after_total} "
              f"{dict(sorted(row['after'].items())) or '{}'}")
        deltas = {c: row["after"].get(c, 0) - row["before"].get(c, 0)
                  for c in set(row["after"]) | set(row["before"])}
        moved = {c: d for c, d in sorted(deltas.items()) if d}
        print(f"    delta    : {moved or 'none'}")
        for code, d in deltas.items():
            if d > 0:
                introduced[code] += d
            kept = min(row["after"].get(code, 0), row["before"].get(code, 0))
            if kept:
                inherited[code] += kept
        print()
    print("  TOTALS over the declared population")
    print(f"    INTRODUCED (modified files) : {sum(introduced.values())} "
          f"{dict(sorted(introduced.items())) or '{}'}")
    print(f"    INHERITED  (modified files) : {sum(inherited.values())} "
          f"{dict(sorted(inherited.items())) or '{}'}")
    print(f"    IN NEW FILES (no baseline)  : {sum(new_file.values())} "
          f"{dict(sorted(new_file.items())) or '{}'}")
    print("  A new file's findings are listed separately because subtracting "
          "them against")
    print("  an absent baseline is what made A-47's '13 against 10' look "
          "like a comparison.")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python results/_a48_ruff_population.py <path> [...]")
        return 1
    try:
        rows = measure(argv)
    except RuntimeError as exc:
        print(f"RUFF_POPULATION_ERROR {exc}")
        print("RUFF_POPULATION_EXIT:1")
        return 1
    report(rows)
    print("RUFF_POPULATION_EXIT:0")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
