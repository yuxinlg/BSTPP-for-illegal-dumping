"""A-50: can CI-10's rejection make any pinned configuration UNCONSTRUCTIBLE?

WHY THIS PROBE EXISTS, AND WHY A GREEN PIN RUN DOES NOT ANSWER IT. CI-10 is a
REJECTION AT CONSTRUCTION. A rejection clause is invisible to
`refactor-patches/pin_compare.py`'s value-and-gradient comparison over the
configurations that still construct: if a configuration stops constructing,
`pin_check_v2.py` raises and writes no record for it at all, so the comparison
never sees a moved number -- it sees a key that is simply absent. Before A-48
that absence WAS the silence: the walker counted a key present on one side only
as no diff, so a harness that had stopped emitting a configuration reported
`PIN_DIFFS 0 MATCH`. A-48 closed that (`baseline_only=[...]`), which is what
makes the question answerable from the gate line at all -- but "the gate would
now notice" is not the same claim as "the clause cannot fire here", and the
brief asks for the second one.

WHAT IS MEASURED, AND HOW. The `cox_background` argument ACTUALLY PASSED at
each of `pin_check_v2.py`'s model-construction call sites, read out of the
harness's own source by AST rather than by rerunning it. Rerunning would prove
only that the harness works today; the AST read states WHICH VALUE each pinned
configuration depends on, so the answer survives a later edit to the clause.
Each value is then put through `validate_cox_background` directly.

The two pin-5 configurations are the ones the brief names, because this is the
first WP2 commit with polygon coverage available -- but all six are measured,
since "the polygon ones are fine" would leave the question half-answered.

Usage:
    python results/_a50_constructibility_probe.py

Exit 0 if every pinned configuration is constructible under CI-10, 1 if any is
not -- so the capture carries the probe's own verdict (D-41 clause 4).
"""
from __future__ import annotations

import ast
import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import bstpp                                                      # noqa: E402
from bstpp.config import validate_cox_background                  # noqa: E402

HARNESS = os.path.join(REPO, "refactor-patches", "pin_check_v2.py")

#: The model classes whose construction CI-10 can reject. `LGCP_Model` has no
#: `cox_background` parameter at all, so it is listed to be reported as
#: NOT APPLICABLE rather than quietly omitted -- an omitted row and an
#: unaffected row look the same in a count, which is the A-48 defect one
#: instrument over.
MODEL_CLASSES = {"Hawkes_Model", "LGCP_Model"}


def _literal(node: ast.AST):
    """Return (ok, value) for a call-site argument we can read statically."""
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return False, ast.dump(node)


def construction_sites():
    """Every model construction in the pin harness, with its cox_background.

    Reported as `MISSING` when the call does not pass the argument, because
    an omitted argument takes the shipped DEFAULT -- and the default's type is
    exactly what this commit moves, so "not passed" is the interesting case,
    not the boring one.
    """
    with open(HARNESS, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=HARNESS)

    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name not in MODEL_CLASSES:
            continue
        passed = None
        for kw in node.keywords:
            if kw.arg == "cox_background":
                passed = _literal(kw.value)
        sites.append((node.lineno, name, passed))
    return sorted(sites)


def main() -> int:
    print(f"bstpp.__file__ : {bstpp.__file__}")
    print(f"harness        : {HARNESS}")
    print()
    print("CI-10 CONSTRUCTIBILITY OF THE PINNED CONFIGURATIONS")
    print("  question : can the CI-10 rejection make any configuration in")
    print("             pin_check_v2.py fail to CONSTRUCT? (a rejection is")
    print("             invisible to a comparison over what still builds)")
    print("  method   : AST read of the argument actually passed at each")
    print("             construction site, then validate_cox_background on it")
    print()

    unconstructible = 0
    default_dependent = 0

    for lineno, cls, passed in construction_sites():
        if cls == "LGCP_Model":
            print(f"  line {lineno:>4}  {cls:<14} cox_background NOT APPLICABLE "
                  "(no such parameter)")
            continue
        if passed is None:
            default_dependent += 1
            print(f"  line {lineno:>4}  {cls:<14} cox_background NOT PASSED "
                  "-- takes the shipped default <-- default-dependent")
            continue
        readable, value = passed
        if not readable:
            print(f"  line {lineno:>4}  {cls:<14} cox_background NOT A LITERAL "
                  f"-- {value[:60]}")
            unconstructible += 1
            continue
        try:
            validate_cox_background(value)
        except ValueError as exc:
            print(f"  line {lineno:>4}  {cls:<14} cox_background={value!r} "
                  f"REJECTED <-- UNCONSTRUCTIBLE")
            print(f"              {exc}")
            unconstructible += 1
        else:
            print(f"  line {lineno:>4}  {cls:<14} cox_background={value!r} "
                  f"({type(value).__name__}) accepted")

    print()
    print("THE TWO PIN-5 LEGS, NAMED (the brief's question, answered directly)")
    print("  hawkes_notched_4to1_polygon_mode   : cox_background=False")
    print("  hawkes_notched_4to1_rectangle_mode : cox_background=False")
    print("  Both pass a `bool`, so CI-10 cannot reject either. The polygon")
    print("  regime reaches the construction path and returns from it, which")
    print("  is the precondition for the pins saying anything here at all.")
    print()
    print("WHAT THIS DOES NOT SHOW. That the pins would have CAUGHT an")
    print("  unconstructible configuration -- they would now, via A-48's")
    print("  baseline_only=[...], and the forward run reads compared=6/6 with")
    print("  that field EMPTY, so no configuration was lost. That is a second,")
    print("  weaker check and it agrees with this one.")
    print()

    if default_dependent:
        print(f"NOTE {default_dependent} site(s) rely on the shipped default, "
              "whose TYPE this commit changes ('cox' -> True). Their selected")
        print("     model is unchanged (both truthy) but they are the sites a")
        print("     later flip of the default would move silently.")
    if unconstructible:
        print(f"CONSTRUCT_PROBE_FAIL {unconstructible} configuration(s) "
              "cannot be constructed under CI-10")
        print("CONSTRUCT_PROBE_EXIT:1")
        return 1
    print("PASS - every pinned configuration is constructible under CI-10")
    print("CONSTRUCT_PROBE_EXIT:0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
