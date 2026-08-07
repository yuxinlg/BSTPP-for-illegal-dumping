"""A-25 / C3 content checks (documentation-only gate profile).

Three checks the LaTeX build cannot be run to perform on this machine, and that the
structural \\hypertarget check cannot see:

1. §12.2 column-1 phase-name check. The matrix's row axis is Phase; anything else in
   column 1 is the A-24 transposed-row category error. Unchanged by this commit, but
   cheap, and it guards a defect class no other gate sees.
2. longtable field-count check. Every longtable body row must supply exactly
   (number of p{} column specs) fields, i.e. (ncols - 1) unescaped '&'. A row with the
   wrong count is the failure mode that produced the A-24 transposed row: it can still
   compile, or it can break the build on a machine where the build is not runnable here.
3. §8 Part I decision-row monotonicity (added A-27). D-40 landed out of numeric order
   and was caught by reading. A misordered row is structurally perfect, so neither
   check above can see it; nor can either see a duplicate or a gap. Demonstrated to
   discriminate by mutating a copy (D-40 moved before D-39) and confirming exit 1.
4. PROSE-AND-TABLE DIVERGENCE (added A-44; OP-29). Three instances in one series:
   D-43 was cited as authority before it existed; OP-21 and OP-22 were opened in
   prose and never entered the §11 table; OP-20 was CLOSED in prose at A-33 and its
   §11 row never marked, so a reader enumerating open items from the table -- which
   is what the table is for -- counted it open, and C1's "exhaustive" WP2 list
   omitted it for exactly that reason. Three is the OP-22 precedent's threshold for
   enumerating a class rather than accumulating instances, so the enumeration stops
   being something done when somebody happens to look and becomes this check.
   Four sub-checks, each catching one instance of the class:
     4a  every OP-n an amendment says it OPENS has a §11 row
     4b  every OP-n an amendment says it CLOSES has CLOSED in that row
     4c  every D-n cited anywhere in the register has a §8 row
     4d  D-44: every work package named in a §11 destination cell has an entry
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

# ---------------------------------------------------------------- check 3
# A-27: Part I decision-row label monotonicity. D-40 landed out of numeric order
# and was caught by reading; neither the structural \hypertarget check nor the
# field-count check above can see ordering, because a misordered row is
# structurally perfect. A few lines close the class instead of leaving it to
# attention. Duplicates and gaps are reported too: the same reading pass that
# would catch a misorder is the only thing that would catch those.
print()
print("PART I DECISION-ROW MONOTONICITY")
dsec = text.find(r"\hypertarget{decision-register-settled}")
if dsec < 0:
    failures.append("could not locate the section 8 decision register")
else:
    dlt = text.find(r"\begin{longtable}", dsec)
    dend = text.find(r"\end{longtable}", dlt)
    dbody = text[dlt:dend].split(r"\endhead", 1)[-1]
    labels: list[int] = []
    for raw in dbody.splitlines():
        line = raw.strip()
        if not line or "&" not in line:
            continue
        m = re.search(r"D-(\d+)", line.split("&", 1)[0])
        if m:
            labels.append(int(m.group(1)))
    print(f"  {len(labels)} decision rows: D-{labels[0]} .. D-{labels[-1]}"
          if labels else "  no decision rows found")
    if not labels:
        failures.append("section 8 decision register has no D-* rows")
    prev = None
    for n in labels:
        if prev is not None and n <= prev:
            failures.append(
                f"section 8 decision rows are not strictly increasing: D-{n} follows "
                f"D-{prev}")
        prev = n
    missing = sorted(set(range(1, labels[-1] + 1)) - set(labels)) if labels else []
    if missing:
        failures.append(
            "section 8 decision numbers have gaps: "
            + ", ".join(f"D-{n}" for n in missing))
    if labels and not missing and labels == sorted(set(labels)):
        print("  strictly increasing, no duplicates, no gaps -> OK")

# ---------------------------------------------------------------- check 4
# Prose-and-table divergence (OP-29). The register states things twice -- once in an
# amendment's prose and once in a table row -- and nothing has ever compared the two.
print()
print("CHECK 4 -- prose vs table (OP-29)")

# §11 rows: "OP-n & ... & <destination>". Collect the row text per item.
op_rows: dict[str, str] = {}
for line in text.splitlines():
    m = re.match(r"\s*(?:\\amdnew\{[^}]*\}\s*)*(OP-\d+)\s*&", line)
    if m:
        op_rows.setdefault(m.group(1), line)

# Amendment headers: "\textbf{Opens} OP-19 and OP-26." / "\textbf{Closes} OP-20."
def _declared(verb: str) -> set[str]:
    out: set[str] = set()
    for frag in re.findall(r"\\textbf\{" + verb + r"\}([^.]*)\.", text):
        # A quoted mention inside prose is not a header declaration; headers are
        # short and list bare identifiers.
        if len(frag) > 60:
            continue
        out.update(re.findall(r"OP-\d+", frag))
    return out


opened, closed = _declared("Opens"), _declared("Closes")
print(f"  amendments declare OPENS : {len(opened)}  {sorted(opened)}")
print(f"  amendments declare CLOSES: {len(closed)}  {sorted(closed)}")
print(f"  items with a §11 row     : {len(op_rows)}")

missing_row = sorted(opened - set(op_rows))
if missing_row:
    failures.append(
        "4a opened in prose with no §11 row: " + ", ".join(missing_row))
print(f"  4a opened-with-no-row    : {missing_row or 'none'}")

unmarked = sorted(i for i in closed
                  if i in op_rows and "CLOSED" not in op_rows[i])
if unmarked:
    failures.append(
        "4b closed in prose, §11 row not marked CLOSED: " + ", ".join(unmarked))
print(f"  4b closed-but-row-unmarked: {unmarked or 'none'}")

# 4c: every D-n mentioned anywhere must have a §8 row (D-43 was cited before it
# existed). `labels` is check 3's list of §8 decision numbers.
cited_d = {int(n) for n in re.findall(r"\bD-(\d+)\b", text)}
undefined_d = sorted(cited_d - set(labels))
if undefined_d:
    failures.append(
        "4c decision cited with no §8 row: "
        + ", ".join(f"D-{n}" for n in undefined_d))
print(f"  4c cited-with-no-row      : "
      f"{[f'D-{n}' for n in undefined_d] or 'none'}")

# 4d (D-44): a destination naming a work package must name one that has an entry.
wp_dir = Path("refactor-patches/phase3f")
entries = set()
if wp_dir.is_dir():
    for p in wp_dir.glob("wp*_*entry.md"):
        m = re.match(r"wp(\d+)_", p.name)
        if m:
            entries.add(f"WP{int(m.group(1))}")
entries |= {"WP1", "WP2"}          # defined in the register itself, not as files
routed: dict[str, set[str]] = {}
for item, row in op_rows.items():
    dest = row.rsplit("&", 1)[-1]
    for wp in re.findall(r"\bWP(\d+)\b", dest):
        routed.setdefault(f"WP{int(wp)}", set()).add(item)
unresolved = sorted(w for w in routed if w not in entries)
if unresolved:
    failures.append(
        "4d D-44: destination names a work package with no entry: "
        + ", ".join(f"{w} ({', '.join(sorted(routed[w]))})" for w in unresolved))
print(f"  work packages with entries: {len(entries)}  {sorted(entries, key=lambda w: int(w[2:]))}")
print(f"  4d routed-to-no-entry     : {unresolved or 'none'}")

# ---------------------------------------------------------------- check 5 --
# A-52. A LANDED decision must not carry a fill-anchor in an OPERATIVE clause.
#
# WHY 4c COULD NOT COVER THIS. Check 4c resolves a citation by finding the row
# it names; it is satisfied the moment the row exists and CANNOT SEE INSIDE THE
# CELL. D-48 and D-51 landed with their operative clauses carrying anchors, so
# 4c passed on rows stating rules that could not be executed -- a gate blind to
# a defect inside a cell it had just resolved (the fifth instance in the
# self-referential apparatus class, A-52).
#
# SCOPE, and it is deliberately wider than the register. D-47 makes
# `docs/wp_dependency_graph.md` OPERATIVE -- it governs execution order -- so a
# check confined to the .tex would exempt the document that carries most of the
# anchors. Scope = landed decision rows + every document a decision declares
# authoritative.
#
# BASELINE, in the A-47 two-era pattern. Adopting this bare would go red on
# arrival: anchors legitimately remain, and a gate that is red from its first
# commit trains its reader to ignore it. So the REMAINING anchors are DECLARED,
# and an UNDECLARED anchor is red. Declared is not discharged: these are open
# items with a number attached, and the number is checked every commit.
try:
    import importlib
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    _census = importlib.import_module("_a51_anchor_census")
except Exception as exc:                                   # pragma: no cover
    failures.append(f"5 anchor census unavailable: {type(exc).__name__}: {exc}")
    _census = None

#: The declared anchor population, by (path, anchor text) -> count. Measured at
#: A-52 and held as a baseline. Raising a count needs a register amendment;
#: LOWERING one is always safe and is reported, never failed.
DECLARED_ANCHORS = {
    ("phase3_record.tex", "[[FILL: state in the ledger]]"): 1,
    ("docs/wp_dependency_graph.md", "[[FILL: measure]]"): 8,
    ("docs/wp_dependency_graph.md", "[[FILL: enumerate]]"): 4,
}

if _census is not None:
    AUTHORITATIVE_DOCS = {"docs/wp_dependency_graph.md"}   # D-47
    observed: dict[tuple, int] = {}
    for _p in _census._tracked(_census.POPULATION_GLOBS):
        if not Path(_p).is_file():
            continue
        in_scope_doc = _p in AUTHORITATIVE_DOCS
        for _n, _t, _d in _census._anchor_scopes(_p):
            # In scope if inside a landed decision clause, or anywhere in a
            # document a decision declares authoritative.
            if _d or in_scope_doc:
                observed[(_p, _t)] = observed.get((_p, _t), 0) + 1

    undeclared = []
    for key, count in sorted(observed.items()):
        allowed = DECLARED_ANCHORS.get(key, 0)
        if count > allowed:
            undeclared.append(f"{key[0]}: {key[1]} x{count} (declared {allowed})")
    shrunk = [f"{k[0]}: {k[1]} {DECLARED_ANCHORS[k]}->{observed.get(k, 0)}"
              for k in DECLARED_ANCHORS if observed.get(k, 0) < DECLARED_ANCHORS[k]]

    # An OPERATIVE anchor is red regardless of the baseline: the baseline
    # licenses anchors that defer a RECORD, never one that breaks a RULE.
    ops = _census.operative_anchors()
    if ops:
        failures.append(
            "5 operative anchor in a landed decision: "
            + "; ".join(f"{p}:{n} [{d}] {t}" for p, n, t, d in ops))
    if undeclared:
        failures.append("5 undeclared anchor: " + "; ".join(undeclared))

    print(f"  5 in-scope anchors        : {sum(observed.values())} "
          f"over {len(observed)} (path, text) key(s)")
    print(f"  5 operative               : {len(ops) or 'none'}")
    print(f"  5 undeclared              : {undeclared or 'none'}")
    if shrunk:
        print(f"  5 declared-but-fewer      : {shrunk}  (safe; reported)")

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
