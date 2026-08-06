"""A-26 / D-40: raised messages must be ASCII. Extended by A-29 (IV).

A message containing a non-ASCII character can raise UnicodeEncodeError while a
traceback carrying it is being printed to a cp1252 console -- an error path that
fails while failing.

Docstrings, comments and register text are NOT constrained and are not scanned.

------------------------------------------------------------------ A-29 -----
The A-26 sweep scanned string literals inside `raise` statements only. Under
D-40 that is the wrong population, and it got worse as single-sourcing spread:

  * The canonical clause text lives in `*_invariant_clause` functions and
    reaches a `raise` as a VARIABLE (`raise NumericalConfigError(msg)`),
    presenting no literal to find. Every one of those raisers is invisible.
  * The per-site remediation text lives at `raise_*_violation(...)` CALL sites,
    which are `ast.Call` nodes, not `ast.Raise` nodes -- so they were never
    scanned either, by any definition.

So the sweep passed over exactly the text D-40 governs. Three populations are
now covered, and the sweep REPORTS ITS OWN COVERAGE as a fraction of raise
sites rather than printing a bare PASS (D-39 applied to a gate).

A note on a figure that does not reconcile. The A-29 brief cited "69 raise
sites with a literal first argument, 114 with a non-literal, 38% inspected".
That split is `first argument is a bare str Constant`, which is NOT what the
A-26 sweep did: `_string_parts` walks the WHOLE exc expression, so it also saw
the constant chunks of f-strings -- 176 of 183 sites at 426d60a, 96%. Both
figures are reproduced by `--census`. The gap was never 114 unseen literals;
it was that clause and remediation text are evaluated NOWHERE, which is what
this extension fixes.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _string_parts(node: ast.AST):
    """Yield (lineno, value) for every string literal inside an expression."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.lineno, sub.value


def _non_ascii(text: str) -> str | None:
    bad = sorted({c for c in text if ord(c) > 127})
    if not bad:
        return None
    return ", ".join(
        f"U+{ord(c):04X} {unicodedata.name(c, '?')}" for c in bad)


# ------------------------------------------------------ population 1 and 2 --
def scan_module(path: Path):
    """Return (hits, n_raise, n_raise_seen, blind, n_delegated).

    Population 1: string literals inside `raise` statements.
    Population 2: string literals at `raise_*_violation(...)` call sites --
    the remediation clauses, which are Call nodes and were never scanned.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[tuple[int, str, str, str] ] = []
    n_raise = n_seen = n_deleg = 0
    blind: list[tuple[int, str]] = []

    # Which functions are the D-40 raisers? A blind `raise X(msg)` inside one
    # of these is not uncovered -- its text is evaluated in population 3.
    raiser_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("raise_"):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Raise):
                    raiser_lines.add(sub.lineno)

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            n_raise += 1
            parts = list(_string_parts(node.exc))
            if parts:
                n_seen += 1
            else:
                kind = ("delegated-raiser" if node.lineno in raiser_lines
                        else "UNCOVERED")
                blind.append((node.lineno, kind))
            for lineno, text in parts:
                names = _non_ascii(text)
                if names:
                    hits.append((lineno, names, text.strip()[:70], "raise"))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id.startswith("raise_")):
            n_deleg += 1
            for lineno, text in _string_parts(node):
                names = _non_ascii(text)
                if names:
                    hits.append((lineno, names, text.strip()[:70], "remediation"))

    return hits, n_raise, n_seen, blind, n_deleg


# ------------------------------------------------------------ population 3 --
# Representative arguments per PARAMETER NAME. Values must exercise every
# interpolated field -- a clause called with nothing in its slots proves
# nothing about the slots. There is deliberately no fallback: a parameter with
# no sample here makes the clause UNEVALUATED, which fails the gate rather
# than being skipped quietly (a skipped clause is an uncovered clause).
CLAUSE_SAMPLES: dict[str, object] = {
    "panel_h_m": 20.0,
    "min_sigma": 5.0,
    "max_sigma": 40.0,
    "ratio_ceil": 8.0,
    "tau_abs": 1e-5,
    "support_mode": "triangle",
    # A-33 (CI-7 / CI-8): the argument-type clauses take the offending
    # argument's NAME and VALUE. Both reach the message, so both are filled.
    "name": "min_sigma",
    "value": "5",
}

# Second pass: can a caller's own non-ASCII value reach the message through
# interpolation? Only slots rendered without numeric coercion can. This is a
# HAZARD CENSUS, not a gate -- user input is not the clause's to control.
NON_ASCII_PROBE = "triangléé"


def evaluate_clauses():
    """Call every *_invariant_clause in bstpp.config and check the result.

    Returns (results, unevaluated, propagating). Discovered by introspection,
    not a hard-coded list, so a clause added for a future config is covered on
    the day it lands rather than the day someone remembers this file.
    """
    from bstpp import config as cfg

    names = sorted(n for n in dir(cfg) if n.endswith("_invariant_clause"))
    results: list[tuple[str, str | None, str]] = []
    unevaluated: list[tuple[str, str]] = []
    propagating: list[str] = []

    for name in names:
        fn = getattr(cfg, name)
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        kwargs, missing = {}, []
        for pname in sig.parameters:
            if pname in CLAUSE_SAMPLES:
                kwargs[pname] = CLAUSE_SAMPLES[pname]
            else:
                missing.append(pname)
        if missing:
            unevaluated.append((name, f"no sample for {', '.join(missing)}"))
            continue
        try:
            text = fn(**kwargs)
        except Exception as e:  # noqa: BLE001 - an unevaluable clause is the finding
            unevaluated.append((name, f"{type(e).__name__}: {e}"))
            continue
        results.append((name, _non_ascii(text), text))

        # Hazard probe, one slot at a time. Injecting only where the SAMPLE is
        # already a str would have missed `rectangle_bounds_invariant_clause`,
        # whose bounds are typed `object` and rendered with !r -- its samples
        # are floats, so a type-based probe never reaches the !r. Probing every
        # slot is also the honest test of coercion: a slot that raises on a
        # non-ASCII string is protected BY the float() in the clause, which is
        # exactly the property being claimed.
        for slot in kwargs:
            probe_kwargs = dict(kwargs)
            probe_kwargs[slot] = NON_ASCII_PROBE
            try:
                rendered = fn(**probe_kwargs)
            except Exception:  # noqa: BLE001 - coercion refused it; that is the protection
                continue
            if _non_ascii(rendered):
                propagating.append(f"{name}({slot}=)")

    return results, unevaluated, propagating


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", action="store_true",
                    help="also print the two raise-site count definitions")
    args = ap.parse_args()

    root = REPO / "bstpp"
    files = sorted(root.rglob("*.py"))
    all_hits: list[tuple[Path, int, str, str, str]] = []
    tot_raise = tot_seen = tot_deleg = 0
    all_blind: list[tuple[Path, int, str]] = []

    for f in files:
        hits, n_raise, n_seen, blind, n_deleg = scan_module(f)
        tot_raise += n_raise
        tot_seen += n_seen
        tot_deleg += n_deleg
        for h in hits:
            all_hits.append((f.relative_to(REPO), *h))
        for lineno, kind in blind:
            all_blind.append((f.relative_to(REPO), lineno, kind))

    clauses, unevaluated, propagating = evaluate_clauses()

    print(f"scanned {len(files)} modules under bstpp/")
    print()
    print("COVERAGE (D-39: a gate states its own coverage)")
    pct = 100.0 * tot_seen / tot_raise if tot_raise else 0.0
    print(f"  direct raise sites                    : {tot_raise}")
    print(f"    literal text scanned                : {tot_seen}  ({pct:.0f}%)")
    uncovered = [b for b in all_blind if b[2] == "UNCOVERED"]
    delegated_blind = [b for b in all_blind if b[2] == "delegated-raiser"]
    print(f"    no literal, text from a clause fn   : {len(delegated_blind)}"
          f"  (covered below by evaluation)")
    print(f"    no literal, text built at runtime   : {len(uncovered)}"
          f"  (NOT covered -- listed below)")
    print(f"  delegated raise_*() call sites        : {tot_deleg}"
          f"  (remediation literals scanned)")
    print(f"  *_invariant_clause functions          : {len(clauses)} evaluated,"
          f" {len(unevaluated)} unevaluated")
    covered = tot_seen + len(delegated_blind)
    print(f"  => raise sites whose text is checked  : {covered}/{tot_raise}"
          f"  ({100.0 * covered / tot_raise:.0f}%)")
    print()

    if uncovered:
        print("RESIDUAL BLIND SPOTS (text assembled at runtime; stated, not hidden)")
        for rel, lineno, _ in uncovered:
            print(f"  {rel}:{lineno}")
        print()

    print("CLAUSE EVALUATION (population 3)")
    for name, bad, text in clauses:
        status = "ASCII" if bad is None else f"NON-ASCII {bad}"
        print(f"  {status:<10} {name}")
        if bad is not None:
            print(f"      {ascii(text[:70])}")
    print()

    if propagating:
        print("INTERPOLATION HAZARD (informational, not a failure)")
        print("  These clauses render a caller-supplied value without numeric")
        print("  coercion, so a non-ASCII argument reaches the message. Not the")
        print("  clause's fault and not gated here; see OP-22.")
        for name in propagating:
            print(f"    {name}")
        print()

    if args.census:
        print("CENSUS (reconciles the two definitions in circulation)")
        fc = fo = 0
        for f in files:
            for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                if isinstance(n, ast.Raise) and n.exc is not None:
                    a = n.exc.args if isinstance(n.exc, ast.Call) else []
                    a0 = a[0] if a else None
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        fc += 1
                    else:
                        fo += 1
        print(f"  'first arg is a bare str Constant' : {fc} / {fc + fo}"
              f"  ({100.0 * fc / (fc + fo):.0f}%)   <- the brief's figure")
        print(f"  'any str Constant inside exc'      : {tot_seen} / {tot_raise}"
              f"  ({pct:.0f}%)   <- what the sweep actually walked")
        print()

    # ------------------------------------------------------------- verdict --
    fails = 0
    if all_hits:
        fails += len(all_hits)
        print(f"FAIL {len(all_hits)} non-ASCII character(s) in message literals")
        for rel, lineno, names, snippet, kind in all_hits:
            print(f"  {rel}:{lineno}  [{kind}]  {names}")
            # Escaped rather than printed raw: on a cp1252 console, printing it
            # raw raises UnicodeEncodeError -- the very defect this sweep exists
            # to find, and it crashed this reporter on its first run.
            print(f"      {ascii(snippet)}")
    bad_clauses = [(n, b) for n, b, _ in clauses if b is not None]
    if bad_clauses:
        fails += len(bad_clauses)
        print(f"FAIL {len(bad_clauses)} clause function(s) return non-ASCII")
    if unevaluated:
        fails += len(unevaluated)
        print(f"FAIL {len(unevaluated)} clause function(s) could not be "
              f"evaluated -- an unevaluated clause is an UNCOVERED clause")
        for name, why in unevaluated:
            print(f"  {name}: {why}")
            print(f"      add a sample for it to CLAUSE_SAMPLES in {Path(__file__).name}")

    if fails:
        return 1
    print(f"PASS - {covered}/{tot_raise} raise sites' text checked "
          f"({100.0 * covered / tot_raise:.0f}%); "
          f"{len(clauses)} clause functions evaluated, all ASCII")
    return 0


if __name__ == "__main__":
    sys.exit(main())
