"""A-26 / D-40: raised messages must be ASCII.

A message containing a non-ASCII character can raise UnicodeEncodeError while a
traceback carrying it is being printed to a cp1252 console -- an error path that
fails while failing. This sweep parses every module under bstpp/ and reports any
non-ASCII character in a string that is an argument to `raise`.

Docstrings, comments and register text are NOT constrained and are not scanned.
"""
from __future__ import annotations

import ast
import sys
import unicodedata
from pathlib import Path


def _string_parts(node: ast.AST):
    """Yield (lineno, value) for every string literal inside an expression."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.lineno, sub.value


def scan(path: Path) -> list[tuple[int, str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        for lineno, text in _string_parts(node.exc):
            bad = sorted({c for c in text if ord(c) > 127})
            if bad:
                names = ", ".join(
                    f"U+{ord(c):04X} {unicodedata.name(c, '?')}" for c in bad)
                hits.append((lineno, names, text.strip()[:70]))
    return hits


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "bstpp"
    files = sorted(root.rglob("*.py"))
    total = 0
    print(f"scanned {len(files)} modules under bstpp/")
    for f in files:
        for lineno, names, snippet in scan(f):
            total += 1
            rel = f.relative_to(root.parent)
            print(f"  {rel}:{lineno}  {names}")
            # The snippet is escaped rather than printed raw: on a cp1252
            # console, printing it raw raises UnicodeEncodeError -- which is
            # the defect this sweep exists to find, and it crashed this
            # reporter on its first run.
            print(f"      {ascii(snippet)}")
    print()
    if total:
        print(f"FAIL {total} non-ASCII character(s) in raised message strings")
        return 1
    print("PASS - every raised message string under bstpp/ is ASCII")
    return 0


if __name__ == "__main__":
    sys.exit(main())
