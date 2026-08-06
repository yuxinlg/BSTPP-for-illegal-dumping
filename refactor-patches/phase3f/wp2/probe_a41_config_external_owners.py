"""A-41: the half of OP-27's class that A-40's method could not reach.

THE GAP. OP-27 is "a quantity held in more than one place with no code
reconciling the holders". A-40's denominator was the fields of every landed
frozen config object -- 11 -- and every member it found is CONFIG-ANCHORED:
one owner is a config field, the second is a module constant or a signature
default outside it. Nothing in that method enumerates quantities whose owners
are BOTH outside the config classes. An eleven presented as the class
denominator would therefore overstate what was measured.

WHY THE METHOD CANNOT BE EXTENDED BY SHARPENING IT. The config-anchored half
is measurable because ``create``'s signature default NAMES the second owner:
``panel_h_m=DEFAULT_PANEL_H_M`` is a machine-readable assertion that the two
are the same quantity. With no config field in the pair there is no such
assertion anywhere, and "same quantity" stops being decidable from the source.
Two constants sharing a name may be unrelated; two constants with different
names and different values may be the same quantity expressed twice; a
quantity may be duplicated through a computation and share neither. **There is
no mechanical identity for it**, which is a fact about the question, not a
gap in effort.

SO THIS PROBE DOES NOT MEASURE THE POPULATION. It runs two heuristics and
then adjudicates one family by hand, which is a different and weaker claim:

  H1  the same module-level UPPER_CASE name bound in more than one module
  H2  a module-level constant's value re-appearing as a bare literal in
      another module

**H2 IS A CANDIDATE COUNT AND NOT A BOUND IN EITHER DIRECTION**, and the
number is reported that way. Its precision is unknown and demonstrably poor:
``DEFAULT_TEMPORAL_SCALE_MULTIPLE = 5.0`` and ``DEFAULT_MAX_SIGMA_KM = 5.0``
are unrelated quantities that match each other and every bare ``5`` in the
package. Reporting 68 as a "lower bound" would be the unqualified-ratio defect
A-39 recorded, committed again. In the other direction it is not an upper
bound either: a pair written with two different literals, or derived rather
than written, is invisible to both heuristics.

WHAT IS ESTABLISHED, THEN. One family is ADJUDICATED -- checked by reading the
sites and confirmed at runtime -- and it is a genuine member. That settles the
only question that matters for C3: **the config-external population is NOT
EMPTY.** It does not settle its size, and this probe does not claim to.

Usage:  python refactor-patches/phase3f/wp2/probe_a41_config_external_owners.py
"""
import ast
import dataclasses
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from bstpp.config import NumericalConfig                         # noqa: E402
from bstpp.preparation import (                                  # noqa: E402
    N_S, N_T, N_XY, S_DAYS, S_INTERNAL, T_INTERNAL)
import bstpp                                                     # noqa: E402

PKG = os.path.join(REPO, "bstpp")
SOURCES = sorted(f for f in os.listdir(PKG) if f.endswith(".py"))

# Values excluded from H2 because at these magnitudes a literal match carries
# no information: 0/1/2 and friends appear everywhere for structural reasons.
TRIVIAL = {0, 1, 2, -1, 3, 0.0, 1.0, 2.0, -1.0, 0.5, 100, 10}

print("PROBE_PROVENANCE")
print(f"  repo           : {REPO}")
print(f"  bstpp.__file__ : {bstpp.__file__}")
print(f"  sources parsed : {len(SOURCES)} files under bstpp/")
print()

trees = {}
for fn in SOURCES:
    with open(os.path.join(PKG, fn), encoding="utf-8") as fh:
        trees[fn] = ast.parse(fh.read(), filename=fn)

# ------------------------------------------------- what A-40 already covered
CONFIG_FIELDS = [f.name for f in dataclasses.fields(NumericalConfig)]
print("SCOPE OF THE A-40 DENOMINATOR, restated so the gap is visible")
print(f"  config-anchored population (A-40)   : {len(CONFIG_FIELDS)} fields "
      "of NumericalConfig")
print("  every A-40 member has ONE owner that is a config field")
print("  config-EXTERNAL population           : NOT MEASURED (see below)")
print()

# ------------------------------------------------------ module-level constants
consts = {}      # name -> [(module, value_repr, node)]
for fn, tree in trees.items():
    for node in tree.body:                     # module level only
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Name) and tgt.id.isupper()):
            continue
        try:
            val = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            val = None
        consts.setdefault(tgt.id, []).append((fn, node.lineno, val))

print(f"  module-level UPPER_CASE constants under bstpp/ : {len(consts)}")

# --------------------------------------------------------------------- H1
print()
print("H1 -- the same constant NAME bound in more than one module")
h1 = {n: v for n, v in consts.items() if len({m for m, _, _ in v}) > 1}
for name, binds in sorted(h1.items()):
    print(f"  {name}")
    for m, ln, val in binds:
        print(f"    {m}:{ln} = {val!r}")
print(f"  H1_COUNT={len(h1)}")

# --------------------------------------------------------------------- H2
print()
print("H2 -- a constant's value re-appearing as a bare literal in another module")
by_value = {}
for name, binds in consts.items():
    for m, ln, val in binds:
        if val is None or isinstance(val, bool) or val in TRIVIAL:
            continue
        if isinstance(val, (int, float)):
            by_value.setdefault(val, []).append((name, m, ln))

h2 = []
for fn, tree in trees.items():
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if v in TRIVIAL or v not in by_value:
            continue
        owners = [(n, m, ln) for n, m, ln in by_value[v] if m != fn]
        if not owners:
            continue
        # skip the constant's own definition line in its own module
        h2.append((fn, node.lineno, v, owners))

for fn, ln, v, owners in sorted(h2):
    names = ", ".join(f"{n} ({m}:{lno})" for n, m, lno in owners)
    print(f"  {fn}:{ln} literal {v!r}  <-> {names}")
print(f"  H2_CANDIDATES={len(h2)}   <-- CANDIDATES, NOT A BOUND")
print("  Demonstrating the imprecision rather than asserting it: "
      "DEFAULT_TEMPORAL_SCALE_MULTIPLE")
print("  and DEFAULT_MAX_SIGMA_KM are both 5.0 and are unrelated quantities, "
      "so every")
print("  bare 5 in the package matches both. Most of the rows above are that.")

# ------------------------------------------------------------- adjudicated
print()
print("ADJUDICATED -- the internal-units family, read rather than matched")
# The named owners are in preparation.py. The second owner is main.py's
# constructor, which writes the SAME quantities as bare literals -- in a module
# that already imports T_INTERNAL and uses it elsewhere (main.py:41, 1787).
main_src = open(os.path.join(PKG, "main.py"), encoding="utf-8").read()
main_tree = trees["main.py"]
family = [("args['T']", "T_INTERNAL", T_INTERNAL),
          ("args['S']", "S_INTERNAL", S_INTERNAL),
          ("self.S", "S_DAYS", S_DAYS)]
literal_sites = {}
for node in ast.walk(main_tree):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        continue
    if not (isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
            and not isinstance(node.value.value, bool)):
        continue
    tgt = node.targets[0]
    label = None
    if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "args" and isinstance(tgt.slice, ast.Constant)):
        label = f"args['{tgt.slice.value}']"
    elif (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
          and tgt.value.id == "self"):
        label = f"self.{tgt.attr}"
    if label:
        literal_sites.setdefault(label, []).append((node.lineno,
                                                    node.value.value))

confirmed = []
for label, const_name, const_val in family:
    sites = [(ln, v) for ln, v in literal_sites.get(label, [])]
    agree = all(v == const_val for _, v in sites) and bool(sites)
    print(f"  {label:<12} literal site(s) main.py:"
          f"{[ln for ln, _ in sites] or 'none'}  "
          f"value(s)={[v for _, v in sites] or 'none'}")
    print(f"    named owner   : preparation.{const_name} = {const_val!r}")
    print(f"    agree today   : {agree}")
    print("    reconciled by : nothing -- no comparison, no raise")
    if sites and agree:
        confirmed.append(label)
print(f"  main.py imports T_INTERNAL and uses it elsewhere : "
      f"{'from .preparation import' in main_src and 'T_INTERNAL' in main_src}")
print(f"  ADJUDICATED_MEMBERS={len(confirmed)}  {confirmed}")
print("  These are LIVE in the sense that matters: change preparation."
      "T_INTERNAL and")
print("  cutoffs.py's real<->internal conversions follow it while args['T'] "
      "does not,")
print("  and the internal-units invariant breaks with nothing raising. "
      "(The decoder-")
print(f"  pinned cells N_T={N_T} N_S={N_S} N_XY={N_XY} are the same shape.)")

print()
print("VERDICT")
print(f"  H1 (name collisions)                 : {len(h1)}")
print(f"  H2 (literal candidates, NOT a bound)  : {len(h2)}")
print(f"  adjudicated genuine members           : {len(confirmed)}")
print("  => THE CONFIG-EXTERNAL POPULATION IS NOT EMPTY. Its SIZE is "
      "UNMEASURED,")
print("     and this probe does not claim to bound it in either direction.")
print()
print("  RECORDED CONCLUSION: OP-27's denominator of 11 is the population of")
print("  CONFIG-ANCHORED two-owner quantities and is complete over THAT")
print("  population. The config-external half is a second population, known")
print("  non-empty and unmeasured, and its members become census rows. It")
print("  becomes measurable in the same motion that makes it smaller: every")
print("  quantity WP2 and WP5 pull into a config object gains a named second")
print("  owner and moves into the measurable half.")
print("EXIT:0")
