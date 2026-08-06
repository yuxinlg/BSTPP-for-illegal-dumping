"""A-40: the two-owner class, enumerated once instead of accumulated one at a time.

WHAT THIS COUNTS. OP-19 and OP-26 are not two findings; they are two members
of one class -- **a quantity held in more than one place with no code
reconciling them**. Per the OP-22 precedent (enumerate rather than open a
third), this measures the whole class before WP5 opens.

SCOPED BY QUANTITY, NOT BY CALL SITE. A quantity with three readers is one
member, not three. The unit of the census is the config field.

THE DENOMINATOR, WITH ITS METHOD. The population is every field of every
landed frozen config object. Today that is exactly one object --
``NumericalConfig`` -- so the denominator is ``len(dataclasses.fields(
NumericalConfig))``, read from the class, not typed here. ``ModelConfig`` and
``PriorConfig`` (WP2) and the WP5 objects are enumerated but not landed and
contribute nothing; the denominator therefore GROWS as WP2/WP5 land, and that
is the point of sizing it now.

THE THREE TESTS, each mechanical.

  SECOND OWNER. ``NumericalConfig.create``'s signature default for the field
  is read from the AST. If it is a bare module constant, that constant is a
  second binding of the same quantity, and every OTHER place the constant is
  bound as a default (function signature, dataclass field) or read in
  executable position is a further one. A field whose default is a literal or
  ``None`` has no second owner by this test.

  RECONCILED. Does any code compare the config's field against the constant
  and raise on disagreement? Detected as an ``If`` whose test is a ``Compare``
  mentioning the constant and whose body raises. Three fields are reconciled
  this way in ``__post_init__``; those are NOT members of the class -- the
  code does the reconciling, which is exactly what the class is defined by
  the absence of.

  LIVE vs LATENT. Every ``NumericalConfig.create(...)`` call site in
  ``bstpp/`` is read from the AST and its keyword arguments collected, each
  classified CONST (the module constant), SELF (carried forward off another
  ``NumericalConfig``'s same field -- not an independent source, since the
  value came from the same object), or OTHER. A field with no OTHER site
  always holds the constant, so the two owners CANNOT currently disagree:
  LATENT. A field with at least one OTHER site can hold something else while
  a reader still consults the constant: LIVE. The SELF category is not
  decoration: without it this probe reported the two ``CutoffProvenance``
  tolerances as LIVE, contradicting A-39's measured "latent, not live", and
  the disagreement was the probe's.

WHAT THIS IS NOT. It is a report. Nothing here is fixed, and nothing here
opens a new OP number -- the class gets one (OP-27) and the members are its
rows.

Usage:  python refactor-patches/phase3f/wp2/probe_a40_two_owner_census.py
"""
import ast
import dataclasses
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from bstpp.config import NumericalConfig                         # noqa: E402
import bstpp                                                     # noqa: E402

PKG = os.path.join(REPO, "bstpp")
SOURCES = sorted(f for f in os.listdir(PKG) if f.endswith(".py"))

print("PROBE_PROVENANCE")
print(f"  repo           : {REPO}")
print(f"  bstpp.__file__ : {bstpp.__file__}")
print(f"  sources parsed : {len(SOURCES)} files under bstpp/")
print()

trees = {}
for fn in SOURCES:
    with open(os.path.join(PKG, fn), encoding="utf-8") as fh:
        trees[fn] = ast.parse(fh.read(), filename=fn)


# ---------------------------------------------------------------- population
FIELDS = [f.name for f in dataclasses.fields(NumericalConfig)]
print("POPULATION (the denominator)")
print("  landed frozen config objects : 1  (NumericalConfig)")
print(f"  fields, read from the class  : {len(FIELDS)}")
print(f"  {FIELDS}")
print("  not landed, contributing 0   : ModelConfig, PriorConfig (WP2); "
      "WP5 objects")
print()


# ------------------------------------------------- create() signature defaults
def _create_defaults():
    """field -> the NAME of the module constant it defaults from, or None."""
    for node in ast.walk(trees["config.py"]):
        if isinstance(node, ast.FunctionDef) and node.name == "create":
            args = node.args.kwonlyargs
            out = {}
            for a, d in zip(args, node.args.kw_defaults):
                out[a.arg] = d.id if isinstance(d, ast.Name) else None
            return out
    raise AssertionError("NumericalConfig.create not found")


DEFAULTS = _create_defaults()


# ---------------------------------------------------- other bindings of a name
def _other_bindings(const: str):
    """Every place the constant is bound as a default or read, outside the

    one ``create`` signature default that defines the pairing.
    """
    hits = []
    for fn, tree in trees.items():
        for node in ast.walk(tree):
            # function-signature defaults
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                pairs = list(zip(a.args[len(a.args) - len(a.defaults):],
                                 a.defaults))
                pairs += list(zip(a.kwonlyargs, a.kw_defaults))
                for arg, d in pairs:
                    if isinstance(d, ast.Name) and d.id == const:
                        if fn == "config.py" and node.name == "create":
                            continue          # the pairing itself
                        hits.append(f"{fn}:{node.lineno} "
                                    f"{node.name}({arg.arg}=)")
            # dataclass field defaults / module assignments
            if isinstance(node, ast.AnnAssign) and isinstance(
                    node.value, ast.Name) and node.value.id == const:
                tgt = getattr(node.target, "id", "?")
                hits.append(f"{fn}:{node.lineno} field {tgt}=")
            if isinstance(node, ast.Assign) and isinstance(
                    node.value, ast.Name) and node.value.id == const:
                for t in node.targets:
                    hits.append(f"{fn}:{node.lineno} alias "
                                f"{getattr(t, 'id', '?')}=")
    return hits


def _executable_reads(const: str):
    """Reads of the constant inside a function body -- a reader consulting the

    module rather than the config it was handed.
    """
    hits = []
    for fn, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == const and isinstance(
                        sub.ctx, ast.Load):
                    if fn == "config.py" and node.name in (
                            "create", "__post_init__"):
                        continue      # the pairing, and the reconciliation
                    hits.append(f"{fn}:{sub.lineno} in {node.name}()")
    return sorted(set(hits))


# -------------------------------------------------------- reconciliation test
def _reconciled(const: str) -> str:
    """An ``if <compare mentioning const>: raise ...`` anywhere in bstpp/."""
    for fn, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if not isinstance(node.test, ast.Compare):
                continue
            names = {n.id for n in ast.walk(node.test)
                     if isinstance(n, ast.Name)}
            if const not in names:
                continue
            if any(isinstance(s, ast.Raise) for s in ast.walk(node)):
                return f"{fn}:{node.lineno}"
    return ""


# --------------------------------------------------- live vs latent at create
def _config_typed_names(tree):
    """Names in this module that hold a NumericalConfig.

    Needed because a carry-forward (``default_temporal_tol=
    prev_cfg.default_temporal_tol``) is NOT an independent source: the value
    came from the same object, so the two owners still cannot disagree.
    Without this, the mechanical test over-reports LIVE -- and it did, on the
    first run of this probe, which is why the resolution is here rather than
    in the prose.
    """
    names = {"self.numerical_config"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        v = node.value
        from_create = (isinstance(v, ast.Call)
                       and isinstance(v.func, ast.Attribute)
                       and v.func.attr == "create"
                       and isinstance(v.func.value, ast.Name)
                       and v.func.value.id == "NumericalConfig")
        from_args = (isinstance(v, ast.Subscript)
                     and isinstance(v.slice, ast.Constant)
                     and v.slice.value == "numerical_config")
        if from_create or from_args:
            names.add(tgt.id)
    return names


def _create_call_kwargs():
    """field -> [(site, kind)] over every NumericalConfig.create call.

    kind is CONST (the module constant itself), SELF (carried forward off
    another NumericalConfig's same field), or OTHER (an independent source).
    """
    passed = {}
    for fn, tree in trees.items():
        cfg_names = _config_typed_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Attribute) and f.attr == "create"
                    and isinstance(f.value, ast.Name)
                    and f.value.id == "NumericalConfig"):
                continue
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                const = DEFAULTS.get(kw.arg)
                v = kw.value
                if isinstance(v, ast.Name) and const and v.id == const:
                    kind = "CONST"
                elif (isinstance(v, ast.Attribute) and v.attr == kw.arg
                      and isinstance(v.value, ast.Name)
                      and v.value.id in cfg_names):
                    kind = "SELF"
                else:
                    kind = "OTHER"
                passed.setdefault(kw.arg, []).append(
                    (f"{fn}:{node.lineno}", kind))
    return passed


PASSED = _create_call_kwargs()

print("CENSUS")
print(f"  {'field':<32} {'2nd owner':<9} {'reconciled':<11} {'verdict':<8} "
      f"other bindings / reads")
members, latent, live, reconciled_n, single = [], [], [], 0, []
for name in FIELDS:
    const = DEFAULTS.get(name)
    if const is None:
        single.append(name)
        print(f"  {name:<32} {'no':<9} {'-':<11} {'SINGLE':<8} "
              f"(default is a literal or None)")
        continue
    rec = _reconciled(const)
    binds = _other_bindings(const)
    reads = _executable_reads(const)
    if rec:
        reconciled_n += 1
        verdict = "RECONC."
    else:
        sites = PASSED.get(name, [])
        can_differ = any(kind == "OTHER" for _, kind in sites)
        verdict = "LIVE" if can_differ else "LATENT"
        members.append(name)
        (live if can_differ else latent).append(name)
    detail = "; ".join(binds + reads) or "(none)"
    print(f"  {name:<32} {const:<9.9s} {rec or '-':<11.11s} {verdict:<8} "
          f"{detail}")

print()
print("  per-field detail for the members")
for name in members:
    const = DEFAULTS[name]
    sites = PASSED.get(name, [])
    print(f"    {name}")
    print(f"      second owner        : {const}")
    print(f"      other bindings      : {_other_bindings(const) or ['(none)']}")
    print(f"      executable reads    : {_executable_reads(const) or ['(none)']}")
    print(f"      create() sites      : {sites or '(never passed)'}")
print()

print("TOTALS")
print(f"  denominator (fields of landed config objects) : {len(FIELDS)}")
print(f"  SINGLE   (no second owner)                    : {len(single)}")
print(f"  RECONCILED (code compares and raises)         : {reconciled_n}")
print(f"  TWO-OWNER CLASS MEMBERS                       : {len(members)}")
print(f"    of which LIVE   (owners can disagree today) : {len(live)}  {live}")
print(f"    of which LATENT (cannot yet disagree)       : {len(latent)}  "
      f"{latent}")
print()
print("  cross-check against the register's named instances:")
print("    OP-19 (held-out budget policy) covers  : panel_h_m, gl_order")
print("    OP-26 (CutoffProvenance tolerances)    : default_temporal_tol, "
      "default_spatial_tol")
covered = {"panel_h_m", "gl_order",
           "default_temporal_tol", "default_spatial_tol"}
print(f"    named instances inside the census      : "
      f"{covered.issubset(set(members))}")
print(f"    census members NOT already named       : "
      f"{sorted(set(members) - covered) or '(none)'}")
print("EXIT:0")
