"""A-25 / C3 content checks (documentation-only gate profile).

Two checks the LaTeX build cannot be run to perform on this machine, and that the
structural \\hypertarget check cannot see:

1. §12.2 column-1 phase-name check. The matrix's row axis is Phase; anything else in
   column 1 is the A-24 transposed-row category error. Unchanged by this commit, but
   cheap, and it guards a defect class no other gate sees.
2. longtable field-count check. Every longtable body row must supply exactly
   (number of p{} column specs) fields, i.e. (ncols - 1) unescaped '&'. A row with the
   wrong count is the failure mode that produced the A-24 transposed row: it can still
   compile, or it can break the build on a machine where the build is not runnable here.
"""
from __future__ import annotations

import re
from pathlib import Path

text = Path("phase3_record.tex").read_text(encoding="utf-8")
failures: list[str] = []
known: list[str] = []

# Pre-existing short rows in FROZEN Part I. Recorded, not suppressed: each is reported as
# KNOWN-PREEXISTING so a regression elsewhere still fails, while a defect this check found
# on its first run in a section no commit may edit in place does not block the gate.
# §4.3, table "Item / Value": the artifact clause was left as its own one-field row instead
# of continuing the preceding Value cell. LaTeX fills the missing cell, so it compiles and
# renders as a row with an empty right column. Part I is frozen (A-1): any correction is a
# Part II statement or 3g editorial work, not an in-place edit.
KNOWN_SHORT_ROWS = (
    "same suite command/result as",
)

# ---------------------------------------------------------------- check 1
sec = text.find(r"\hypertarget{matrix}")
if sec < 0:
    raise SystemExit("could not locate §12.2 matrix")
lt_start = text.find(r"\begin{longtable}", sec)
lt_end = text.find(r"\end{longtable}", lt_start)
body = text[lt_start:lt_end]
after_head = body.split(r"\endhead", 1)[-1]

col1: list[str] = []
for raw in after_head.splitlines():
    line = raw.strip()
    if not line or line.startswith("%") or line.startswith("\\bottomrule"):
        continue
    if "&" not in line:
        continue
    first = line.split("&", 1)[0]
    first = re.sub(r"\\(begin|end)\{minipage\}(\[[^\]]*\])?(\{[^}]*\})*", "", first)
    first = re.sub(r"\\raggedright|\\strut|\\arraybackslash|\\tightlist", "", first)
    first = re.sub(r"\\(texttt|textbf|emph)\{([^}]*)\}", r"\2", first)
    first = first.replace("\\", "").strip()
    if first:
        col1.append(first)

ALLOWED = {"Prep (freeze)", "3a", "3b", "3c", "3d", "3e", "3f", "3g / exit", "Phase"}
print(f"COLUMN1_COUNT {len(col1)}")
for v in col1:
    ok = v in ALLOWED
    print(f"COL1: {v}" + ("" if ok else "   <-- NOT A PHASE NAME"))
    if not ok:
        failures.append(f"§12.2 column 1 holds a non-phase value: {v!r}")

# ---------------------------------------------------------------- check 2
print()
print("LONGTABLE FIELD COUNTS")
for m in re.finditer(r"\\begin\{longtable\}(.*?)\\end\{longtable\}", text, re.S):
    block = m.group(1)
    ncols = len(re.findall(r">\{\\raggedright\\arraybackslash\}p\{", block))
    if ncols == 0:
        continue
    line_no = text[: m.start()].count("\n") + 1
    # Rows may span several source lines (pandoc emits minipage cells that way), so
    # accumulate until a line ends the row with '\\', then count separators once.
    bad = 0
    rows = 0
    acc: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not acc and (
            not line
            or line.startswith(
                (r"\toprule", r"\midrule", r"\bottomrule", r"\endhead", r"\endlastfoot",
                 r"\addlinespace", r"\noalign", "%")
            )
        ):
            continue
        acc.append(line)
        if not line.endswith(r"\\"):
            continue
        row = " ".join(acc)
        acc = []
        if r"\multicolumn" in row:
            continue
        rows += 1
        n = len(re.findall(r"(?<!\\)&", row))
        if n != ncols - 1:
            msg = (
                f"longtable at line {line_no}: row has {n + 1} fields, expected {ncols}: {row[:70]!r}"
            )
            if row.startswith(KNOWN_SHORT_ROWS):
                known.append(msg)
            else:
                bad += 1
                failures.append(msg)
    status = "OK" if bad == 0 else f"BAD {bad}"
    print(f"  line {line_no:>5}: {ncols} cols, {rows} body rows -> {status}")

print()
if known:
    print(f"KNOWN-PREEXISTING {len(known)}  (frozen Part I; reported, not suppressed)")
    for k in known:
        print("  " + k)
    print()
if failures:
    print(f"FAIL {len(failures)}")
    for f in failures:
        print("  " + f)
    raise SystemExit(1)
print("ALL CONTENT CHECKS PASS")
raise SystemExit(0)
