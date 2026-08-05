"""Structural check: every Part II \\subsection{A-nn} is immediately preceded by \\hypertarget{a-nn}{%."""
from __future__ import annotations

import re
from pathlib import Path

text = Path("phase3_record.tex").read_text(encoding="utf-8")
# Part II amendment subsections begin at a-1 hypertarget
start = text.find(r"\hypertarget{a-1}{%")
if start < 0:
    raise SystemExit("could not locate Part II start (a-1 hypertarget)")
body = text[start:]

subs = list(re.finditer(r"\\subsection\{A-(\d+)\b", body))
ok: list[str] = []
missing: list[tuple[str, str]] = []
for m in subs:
    n = m.group(1)
    lines_before = body[: m.start()].rstrip().splitlines()
    prev = lines_before[-1].strip() if lines_before else ""
    expected = rf"\hypertarget{{a-{n}}}{{%"
    if prev == expected:
        ok.append(n)
    else:
        missing.append((n, prev[:100]))

print(f"SUBSECTIONS {len(subs)} OK {len(ok)} MISSING {len(missing)}")
for n in ok:
    print(f"OK a-{n}")
for n, prev in missing:
    print(f"MISSING a-{n} prev={prev!r}")
raise SystemExit(0 if not missing else 1)
