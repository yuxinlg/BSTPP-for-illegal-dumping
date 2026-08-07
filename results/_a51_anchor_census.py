"""A-51: count the `[[FILL]]` anchors over a DECLARED population.

WHY THIS IS AN INSTRUMENT AND NOT A GREP. Three counts of the same quantity
have now been offered and no two agree:

  * A-50's prose said **four** unresolved anchors in A-49's text;
  * the brief commissioning this round said **three**, from a clone at
    `51c98f8`;
  * neither is the number of anchor OCCURRENCES, and neither looked at
    `docs/wp_dependency_graph.md`, which is A-49's own deliverable.

Every one of those readings is defensible and they are answers to DIFFERENT
QUESTIONS -- distinct decisions carrying an anchor, versus anchor occurrences,
versus occurrences in one file rather than in the population A-49 actually
wrote. **A count whose population is unstated is the defect this register keeps
naming** (A-39's raise-site denominator, A-41's ON_DISK/TRACKED split, OP-31's
compared=n/m, A-48's ruff sets). It has now happened to the instrument that was
supposed to be watching for it, which is why the fix is an artifact rather than
a corrected sentence.

WHAT IS REPORTED, ALL OF IT, so no reader has to guess which number they hold:

  * the FILE population -- named, and derived from `git ls-files` so an
    untracked scratch copy cannot inflate it;
  * per file, anchor OCCURRENCES (every `[[FILL`), not lines containing one;
  * per file, the DISTINCT DECISION IDs whose text carries an anchor;
  * PROVENANCE -- which commit introduced each anchor, so "A-49's anchors" is
    a measured set and not a recollection;
  * OPERATIVE versus DEFERRING, the distinction item 4 of the brief turns on:
    an anchor inside a clause that STATES THE RULE leaves the rule
    unexecutable, while an anchor deferring a LEDGER ENTRY leaves a rule that
    works and a record that is incomplete. Those are not the same defect and
    collapsing them is how D-50 would get "closed" by filling a blank that
    was never blocking anything.

Usage:
    python results/_a51_anchor_census.py [--json]

Exit 0 always: this is a CENSUS, not a gate. The gate that fails on an
operative anchor is content check 5, landing in the next commit, which
consumes this module's `operative_anchors()`. Keeping the census exit-0 means
the gate's verdict and the census's population cannot drift apart into two
opinions.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The anchor token. Deliberately matched on the OPENING delimiter alone:
#: `[[FILL]]`, `[[FILL: x]]` and a malformed `[[FILL: x]` must all count, or a
#: typo in a closing bracket would hide an anchor from the census built to
#: find anchors -- the self-referential failure this round is declaring.
ANCHOR = re.compile(r"\[\[FILL")

#: Anchor WITH its text, where the text is well formed. Used for grouping only.
ANCHOR_FULL = re.compile(r"\[\[FILL[^\]]*\]\]")

#: Where a decision's text BEGINS. Two forms, and both are needed: a §8
#: longtable row opens `\amdnew{A-nn} \textbf{D-nn} &`, and a Part II clause
#: opens `\paragraph{D-nn ---`.
#:
#: ATTRIBUTION IS BY ENCLOSING CLAUSE, NOT BY LINE CO-OCCURRENCE, and the
#: difference is not pedantic: co-occurrence reported D-49 as anchored because
#: its ID is *mentioned* on an anchored line of another decision's row, and it
#: missed D-50 entirely because D-50's anchor sits in a body line that does not
#: repeat its own ID. That is a census naming the wrong decisions in both
#: directions -- exactly what this instrument exists to stop.
DECISION_ROW = re.compile(r"\\textbf\{(D-\d+)\}\s*&")
DECISION_PARA = re.compile(r"\\paragraph\{(D-\d+)\b")

#: Where a decision's scope ENDS. Without this the forward walk never lets go:
#: the three pre-existing bare anchors in ordinary prose were attributed to
#: D-51 simply because its row was the last decision seen hundreds of lines
#: earlier. A scope that never closes makes every later anchor look like a
#: broken rule, which would have this census reporting three unexecutable
#: decisions that do not exist.
SCOPE_END = re.compile(
    r"\\subsection\{|\\subsubsection\{|\\hypertarget\{|\\end\{longtable\}|"
    r"\\paragraph\{(?!D-\d)")

#: A table row ends at its own line: `... & process & team; A-49 \\`. Rows are
#: single-line in this document, so a row-opened scope closes at end of line.
ROW_SCOPE_IS_LINE = True

#: Files that may carry an anchor and are part of the declared population.
#: Restricted to the LIVE record and its deliverables. Baseline snapshots under
#: `refactor-patches/baselines-*/` are frozen artifacts -- an anchor there is a
#: historical fact, not an open item, and folding them in would make the count
#: grow every time a baseline is archived.
POPULATION_GLOBS = ("phase3_record.tex", "docs/*.md", "AGENTS.md")

#: Anchors that are NOT operative: they defer a record, not a rule. Keyed by
#: the exact anchor text, because "which anchors are harmless" is a judgement
#: and a judgement belongs in one reviewable place rather than in a reader's
#: head. Each carries the reason it is not blocking.
DEFERRING_ANCHORS = {
    "[[FILL: state in the ledger]]": (
        "D-50. The RULE is complete and executable -- corrective depth is "
        "capped at three, a fourth closes the package. What is deferred is a "
        "LEDGER STATEMENT of WP1's history against that cap, which is a "
        "record about the past and cannot make the rule unexecutable."),
    "[[FILL: enumerate]]": (
        "wp_dependency_graph routed-items column. The graph's AUTHORITY is "
        "its edges and preconditions; an unenumerated routed-items cell "
        "leaves D-44's destination rule intact because D-44 binds the ITEM to "
        "name a resolvable WP, not the WP to list its items."),
    "[[FILL: measure]]": (
        "wp_dependency_graph rho_w column. Blocking by design: D-48 makes "
        "this a PRECONDITION on opening, so an unfilled cell is the rule "
        "working, not the rule broken. It is listed as deferring so it is not "
        "counted as an unexecutable clause, and it is NOT thereby discharged."),
}


#: Every subprocess read is pinned to UTF-8 with `errors="replace"`. On this
#: machine Python's default is cp1252, which raised UnicodeDecodeError on the
#: register's own bytes and left `stdout` as None -- an instrument that dies
#: reading the file it audits. Same class as A-48's ruff-colour defect: the
#: tool's OUTPUT ENCODING is part of its contract, and assuming the platform
#: default is how a reader silently gets a different answer than the author.
_RUN = dict(capture_output=True, text=True, encoding="utf-8",
            errors="replace")


def _tracked(patterns):
    out = subprocess.run(["git", "ls-files", "--", *patterns],
                         cwd=REPO, check=True, **_RUN)
    return sorted(p for p in (out.stdout or "").splitlines() if p.strip())


def _blame_commits(path, needle="[[FILL"):
    """Which commit introduced each anchor line. Provenance, not recollection."""
    try:
        out = subprocess.run(["git", "blame", "--line-porcelain", "--", path],
                             cwd=REPO, check=True, **_RUN)
    except subprocess.CalledProcessError:
        return {}
    if not out.stdout:
        return {}
    commits, sha, summary = {}, None, None
    for line in out.stdout.splitlines():
        if re.match(r"^[0-9a-f]{40} ", line):
            sha = line.split()[0][:7]
        elif line.startswith("summary "):
            summary = line[len("summary "):]
        elif line.startswith("\t") and needle in line:
            commits.setdefault(sha, {"summary": summary, "count": 0})
            commits[sha]["count"] += line.count(needle)
    return commits


def _anchor_scopes(path):
    """Yield (lineno, anchor_text, decision_or_None) for every anchor.

    Attribution is to the ENCLOSING decision clause and the scope is closed at
    the first structural boundary, so an anchor in ordinary prose comes back
    with `None` rather than with whichever decision was last seen.
    """
    current = None
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            row = DECISION_ROW.search(line)
            para = DECISION_PARA.search(line)
            if SCOPE_END.search(line) and not (row or para):
                current = None
            if row or para:
                current = (row or para).group(1)
            if ANCHOR.search(line):
                found = ANCHOR_FULL.findall(line) or \
                    ["[[FILL (MALFORMED -- no ]] found)"]
                for text in found:
                    yield n, text, current
            if row and ROW_SCOPE_IS_LINE:
                current = None


def _decisions_with_anchors(path):
    """Distinct decision IDs whose OWN clause text carries an anchor."""
    ids = {d for _, _, d in _anchor_scopes(path) if d}
    return sorted(ids, key=lambda d: int(d.split("-")[1]))


def operative_anchors():
    """Anchors that leave a RULE unexecutable, as (path, lineno, text, decision).

    This is the function a later anchor gate consumes. Two conditions, and BOTH are
    required:

      1. the anchor sits INSIDE a decision clause -- an anchor in narrative
         prose leaves an incomplete sentence, not an unexecutable rule, and
         three such anchors predate A-49 by many commits;
      2. its text is not classified in DEFERRING_ANCHORS.

    Condition 2 is default-blocking: an anchor nobody has classified fails
    loudly rather than being assumed benign. Condition 1 is what keeps the
    gate from failing forever on prose it was never meant to police -- without
    it the check could only be made to pass by editing Part I.
    """
    hits = []
    for path in _tracked(POPULATION_GLOBS):
        if not os.path.isfile(os.path.join(REPO, path)):
            continue
        for n, text, decision in _anchor_scopes(path):
            if decision and text not in DEFERRING_ANCHORS:
                hits.append((path, n, text, decision))
    return hits


def narrative_anchors():
    """Anchors outside any decision clause. Reported, never gated on."""
    out = []
    for path in _tracked(POPULATION_GLOBS):
        if not os.path.isfile(os.path.join(REPO, path)):
            continue
        for n, text, decision in _anchor_scopes(path):
            if not decision:
                out.append((path, n, text))
    return out


def main() -> int:
    paths = _tracked(POPULATION_GLOBS)
    report = {"population": paths, "files": {}}

    print("ANCHOR CENSUS -- `[[FILL]]` over a declared population")
    print()
    print("FILE POPULATION (git ls-files; globs "
          f"{', '.join(POPULATION_GLOBS)})")
    for p in paths:
        print(f"    {p}")
    print("  EXCLUDED, and named rather than dropped: baseline snapshots under")
    print("  refactor-patches/baselines-*/ (frozen artifacts -- an anchor")
    print("  there is a historical fact, not an open item), and commit-message")
    print("  captures under results/ (they QUOTE anchors, they do not carry")
    print("  them).")
    print()

    total_occ = 0
    for path in paths:
        full = os.path.join(REPO, path)
        if not os.path.isfile(full):
            continue
        text = open(full, encoding="utf-8").read()
        occ = len(ANCHOR.findall(text))
        if not occ:
            continue
        total_occ += occ
        decisions = _decisions_with_anchors(path)
        prov = _blame_commits(path)
        report["files"][path] = {"occurrences": occ, "decisions": decisions,
                                 "provenance": prov}
        print(f"  {path}")
        print(f"    anchor OCCURRENCES            : {occ}")
        print(f"    lines containing an anchor    : "
              f"{sum(1 for ln in text.splitlines() if ANCHOR.search(ln))}"
              "   <- the two differ; `grep -c` reports this one")
        print(f"    DISTINCT decisions anchored   : {len(decisions)}"
              f"  {decisions if decisions else ''}")
        print("    by anchor text:")
        in_clause = {(n, t) for n, t, d in _anchor_scopes(path) if d}
        for t, c in sorted(
                {t: text.count(t) for t in set(ANCHOR_FULL.findall(text))}
                .items()):
            if t in DEFERRING_ANCHORS:
                kind = "deferring"
            elif any(tt == t for _, tt in in_clause):
                kind = "OPERATIVE"
            else:
                kind = "narrative"
            print(f"      {c:>3}x  [{kind}]  {t}")
        print("    introduced by:")
        for sha, info in sorted(prov.items(),
                                key=lambda kv: -kv[1]["count"]):
            print(f"      {sha}  {info['count']:>3} anchor(s)  "
                  f"{(info['summary'] or '')[:58]}")
        print()

    ops = operative_anchors()
    narr = narrative_anchors()
    print("TOTALS -- three classes, because collapsing them is how a harmless")
    print("anchor and an unexecutable rule come to look like the same number.")
    print(f"  anchor occurrences, whole population : {total_occ}")
    print("  OPERATIVE  (in a decision clause,")
    print(f"              unclassified -> blocks a RULE) : {len(ops)}")
    print("  deferring  (in a decision clause,")
    print(f"              classified -> blocks a RECORD) : "
          f"{total_occ - len(ops) - len(narr)}")
    print(f"  narrative  (outside any decision clause)   : {len(narr)}")
    print()
    print("COVERAGE OF THIS CLASSIFICATION (D-39: a gate states its own)")
    print("  Decision-clause detection is LATEX-SPECIFIC -- it keys on")
    print("  `\\textbf{D-nn} &` rows and `\\paragraph{D-nn` headings. MARKDOWN")
    print("  FILES HAVE NEITHER, so every anchor in docs/*.md and AGENTS.md")
    print("  falls to `narrative` BY CONSTRUCTION, not by judgement. The")
    print("  live instance is docs/wp_dependency_graph.md's rho_w and")
    print("  routed-items cells: those ARE governed by D-48/D-44 and are")
    print("  legitimately open (an unfilled rho_w cell is D-48's precondition")
    print("  working, not broken), but this instrument is not what")
    print("  establishes that -- DEFERRING_ANCHORS is, and it is consulted")
    print("  only for anchors already inside a decision clause.")
    print("  SO: `OPERATIVE 0` is a claim about the LATEX REGISTER only.")
    print("  It is not a claim that every Markdown anchor is harmless.")
    print()
    if narr:
        print("NARRATIVE ANCHORS -- reported, never gated on. In the register")
        print("these leave an incomplete sentence, not an unexecutable rule,")
        print("and predate A-49; a gate failing on them could only be made to")
        print("pass by editing Part I. In Markdown, see the coverage note.")
        for path, n, t in narr:
            print(f"    {path}:{n}  {t}")
        print()
    if ops:
        print("OPERATIVE ANCHORS, each one a clause that cannot be executed:")
        for path, n, t, d in ops:
            print(f"    {path}:{n}  [{d}]  {t}")
    else:
        print("No operative anchor: every remaining anchor defers a record,")
        print("not a rule, and each is classified in DEFERRING_ANCHORS with")
        print("its reason.")
    print()
    print("ANCHOR_CENSUS_EXIT:0")

    if "--json" in sys.argv:
        with open(os.path.join(REPO, "results",
                               "_a51_anchor_census.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
