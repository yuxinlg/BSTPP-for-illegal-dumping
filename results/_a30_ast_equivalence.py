"""A-30: prove the production diff is comments and docstrings only.

The register claims this renumbering cannot change behaviour. That claim is
checkable rather than assertable: parse each touched production module before
and after, strip every docstring, and compare the ASTs. Comments never reach
the AST at all, so an identical dump means the executable content is identical.

Compared against A-29's tip (the commit before this one), not against an
arbitrary base, so the comparison isolates exactly this change.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# PINNED to the A-29 commit, not "HEAD". Run before A-30 was committed, HEAD
# was A-29 and the comparison was the real one; run afterwards, HEAD becomes
# A-30 and the check silently compares the commit to itself -- production still
# reports IDENTICAL and the test file reports "strings differ: False", which
# looks like a pass and measures nothing. That happened once here, on a re-run
# after the register was edited. A pinned baseline makes the capture mean the
# same thing whenever it is run, which is the property an artifact needs.
#
# The "after" side is the WORKING TREE. A-30's own hash is not pinned here
# because this check is run while that commit is still being amended, and a
# pinned hash would go stale on every amend.
BEFORE = "1dd0d7a"  # A-29 tip, immediately before the renumbering

# Production: the labels live only in comments and docstrings, so the AST must
# be IDENTICAL. This is the gate on the "cannot change behaviour" claim.
PRODUCTION = ["bstpp/config.py", "bstpp/polygon_mass.py"]

# Tests: the Lane B parametrize labels and assertion messages ARE string
# constants, so they legitimately reach the AST. The claim there is weaker and
# more precise -- only STRING LITERALS changed, no assertion, call, argument or
# control-flow node did. Checked by normalising every str constant away and
# requiring the remaining structure to be identical.
TESTS = ["tests/test_lane_b_config_matrix.py"]


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
    return tree


def blank_strings(tree: ast.AST) -> ast.AST:
    """Replace every str constant with a fixed placeholder."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = "<str>"
    return tree


def dump(src: str, *, blank: bool = False) -> str:
    tree = strip_docstrings(ast.parse(src))
    if blank:
        tree = blank_strings(tree)
    return ast.dump(tree, annotate_fields=True)


def at_before(rel: str) -> str:
    return subprocess.run(["git", "show", f"{BEFORE}:{rel}"],
                          capture_output=True, text=True, encoding="utf-8",
                          cwd=REPO, check=True).stdout


def main() -> int:
    fails = 0

    print(f"1. PRODUCTION vs {BEFORE}: AST must be IDENTICAL")
    print("   (labels live in comments and docstrings only; comments never")
    print("    reach the AST, docstrings are stripped)")
    for rel in PRODUCTION:
        same = dump(at_before(rel)) == dump((REPO / rel).read_text("utf-8"))
        print(f"   {'IDENTICAL' if same else 'DIFFERS':<12} {rel}")
        if not same:
            fails += 1

    print()
    print(f"2. TESTS vs {BEFORE}: only STRING LITERALS may differ")
    print("   Parametrize labels and assertion messages are real constants, so")
    print("   they do reach the AST. With every str blanked, the remaining")
    print("   structure must be identical -- that is 'renames only, no")
    print("   assertion touched', checked rather than asserted.")
    for rel in TESTS:
        old, new = at_before(rel), (REPO / rel).read_text("utf-8")
        raw_same = dump(old) == dump(new)
        blank_same = dump(old, blank=True) == dump(new, blank=True)
        print(f"   {'IDENTICAL' if blank_same else 'DIFFERS':<12} {rel}"
              f"   (structure; strings differ: {not raw_same})")
        if not blank_same:
            fails += 1

    print()
    if fails:
        print(f"FAIL {fails} module(s) changed executable structure -- "
              f"this is NOT a labels-only change")
        return 1
    print("PASS - production AST identical; test changes are string literals "
          "only, no assertion structure touched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
