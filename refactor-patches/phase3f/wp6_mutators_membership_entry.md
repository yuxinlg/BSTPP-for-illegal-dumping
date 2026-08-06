# WP6 — public mutators + G1 membership single-source: entry

**Class: BP** (`phase3f_work_package_outline.md`).
**Status: PLACEHOLDER.** Seams named, scope not yet specified.

## Seams

Public mutators · **G1** membership single-source.

**G1 is already a declared 3f change**, from A-22's closeout ledger:
*"membership predicate consolidated to a single `covers_xy` used by validation
and simulation. Structural; no numerical change expected."* The public mutator
in the tree today is `set_window` (plus `LGCP_Model.set_window`'s rejection
path).

## Open items routed here

**None.** No §11 row names WP6.

## Constraints and known limits already in force

- **`set_window`'s contract is already frozen and is a floor, not a starting
  point**: a private `_UNSET` sentinel so omission differs from explicit
  `None`; cutoffs, pairs, support and `cutoff_provenance` updated
  *transactionally*; design scales persisted for honest realized-omission
  recomputation; temporal-only polygon window changes reuse the installed mass
  table while spatial-window changes require a compatible `mass_table=`
  validated **before** mutation.
- **G1 is declared "no numerical change expected"** — expected, not
  established. A-42's own probe found the membership predicate's double
  spelling among the pre-3f items ("first determine whether the two spellings
  can disagree"), and whether that determination was ever made is not recorded
  here.

### This is where OP-27's stated closure path lands

A-42 recorded the general result behind OP-27's config-external half: **a
literal's owner set is undecidable precisely because no machine-readable
assertion binds it to a constant** — the same fact that makes the
config-anchored population decidable, seen from the other side. So the remedy
is not a better search but **a named binding**, and a quantity acquires a
decidable owner set the moment one exists.

**That is this package's work.** It is why OP-27's class has a terminus rather
than an unbounded audit, and it is recorded here so the terminus is reachable
from the package that provides it, not only from the census that needs it.

**But the routing is not changed by saying so**: OP-27 routes to **WP5 for the
fixes** and **WP2 for census rows**. Whether the config-external half is
discharged here, or per-package as each work package lands its own named
bindings, is question 2 below.

## Adjacencies — subject matter, NOT routings

- **OP-14** (closeout §6 P2: three-leg consistency at every Lane B covered
  point) and **OP-16** (`set_window` rejected as a covering-array axis level)
  are both about public mutators. Both are routed to **"standing Lane B
  completion"**, not to WP6.
- The completed pre-3f programme's `W3(b)` was "invariant test over *every*
  public setter: P3 + P4". Different sequence, completed programme — see the
  `Wn`/`WPn` collision warning in the outline.

## Questions this entry leaves open

1. **Which mutators are in scope.** "Public mutators" names a class; the tree
   has `set_window` and its LGCP rejection path, and whether WP6 covers only
   those is not stated.
2. **Whether OP-27's config-external half is discharged here in one pass, or
   per-package as each lands its named bindings.** The closure path is stated;
   its owner is not.
3. **Whether G1's "no numerical change expected" has been established.**
   Expected is not measured, and the determination the pre-3f brief asked for
   (can the two spellings disagree?) is not recorded as having been made.
4. **Scope and sequencing.** Not specified, not invented.
