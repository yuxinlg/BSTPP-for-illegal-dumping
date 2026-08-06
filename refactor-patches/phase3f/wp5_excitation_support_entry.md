# WP5 — `ExcitationSupport`, `PolygonMassTable`, cutoff provenance: entry

**Class: BP** (`phase3f_work_package_outline.md`).
**Status: SUBSTANTIVE.** This is the one work package with items already
pointing at it, and the entry exists because six of them were routing to a
name with no text.

## Seams

`ExcitationSupport` · `PolygonMassTable` · cutoff provenance.

In the tree today those are `bstpp/excitation_support.py`,
`bstpp/polygon_mass.py`, and `bstpp/cutoffs.py`'s `CutoffProvenance`, together
with the polygon branch of `Point_Process_Model.__init__` and `set_window`.

## The six open items routed here

**Enumerated from `phase3_record.tex` §11 by reading each row's destination
cell, not from memory.** Six rows name WP5, and that is the complete set.

| item | what it is | how it routes |
| --- | --- | --- |
| **OP-19** | held-out scoring validates a supplied mass table against *module default* budget policy, not the model's `NumericalConfig` | "WP5, with the `ExcitationSupport` seam" |
| **OP-21** | `validate_polygon_mass_table(sigma_max=None)` raises an unnamed `TypeError`; deciding what it *should* raise means deciding whether a `NumericalConfig` holding `max_sigma=None` may reach install validation at all | "**WP5**, with OP-19" |
| **OP-23** | *reserved; content unrecovered* — see below | "**WP5** opening: either the text lands or the citation is withdrawn" |
| **OP-24** | the golden pins certify nothing about the polygon regime | "**WP5** entry precondition" — **not** an item inside WP5 |
| **OP-26** | `CutoffProvenance` reports the *module* default tolerances, never the model's `NumericalConfig` ones | "**WP5**, with OP-19 and OP-23" |
| **OP-27** | the two-owner class | "**WP5** for the fixes; **WP2** adds rows to the census as its objects land" |

**Count: 6.** Two of them are not ordinary work items and the entry says so
rather than letting them be read as a to-do list: **OP-24 is an entry
precondition** (below) and **OP-27 routes its *fixes* here while its *census*
keeps growing elsewhere.**

**OP-19 and OP-26 are the same defect twice** — both are OP-27's two-owner
class, OP-19 live and OP-26 latent — so WP5 inherits the class's fixes, not
three unrelated repairs.

### OP-23 — reserved, content unrecovered

**Its text does not exist and is not reconstructed here.** `AGENTS.md` cites
"OP-23 (the ownerless cutoff cluster) → WP5"; A-36 established by grep that
**that citation is the only occurrence of OP-23 anywhere in the repository** —
no amendment opened it and no row defined it. The §11 row records its content
as *absent* rather than guessed.

Writing a plausible ownerless-cutoff cluster to fill the name would be the
defect this entry exists to close, committed a second time. **At WP5's opening
either the text lands or the `AGENTS.md` citation is withdrawn.** Until then
the number is reserved so it cannot be re-issued.

## Coverage limits — both already established, carried here

**1. No pinned configuration builds a polygon mass table, so `PIN_DIFFS` is
silent on every surface WP5 touches.**

Measured, not asserted: `refactor-patches/pin_check_v2.py` emits **four**
configurations — `hawkes`, `cox_hawkes`, `lgcp`, `hawkes_nonsquare_4to1`
(confirmed against the top-level keys of `baselines-2026-07/pins.json`) — and
the harness contains **zero occurrences** of `polygon`, `mass_table`,
`excitation_support` or `min_sigma`. The polygon surfaces are not merely
unexercised; they are unreachable from the harness as written.

So `PIN_DIFFS 0 MATCH` is **not evidence** about `ExcitationSupport`,
`PolygonMassTable` or cutoff provenance, which is precisely the set WP5
changes. A polygon-regime regression reports MATCH on every configuration the
harness runs.

**2. OP-24's polygon-mode pin is a WP5 ENTRY PRECONDITION, not a task inside
WP5.** A-36's reason: *an instrument built by the package it is meant to gate
is not a gate.* The alternative — opening WP5 declaring itself uncovered — was
considered and rejected there for a specific reason: WP5's surfaces are exactly
the ones the existing pins are blind to, so "proceeding without coverage" and
"proceeding with the coverage we have" would produce identical gate lines.

### Where the scheduled instrument stands

**Polygon-regime excitation validation was SCHEDULED, not deferred** — A-36
records the decision in those words ("A-36 decides: schedule it") and
deliberately recorded it then "so it is not carried into WP5's opening as
scheduling TBD".

**As of A-43 it has not been built.** The harness still emits four
configurations, none polygon. So the scheduling decision stands and the
instrument does not exist, and the two facts together are the entry
precondition's whole current status: **WP5 cannot open until the fifth pin
lands.**

One consequence, recorded because it is easy to trip over: the fifth pin adds
a configuration key the baseline does not carry, so it will make
`pin_corpus_identity.py` go red and expire A-38's retroactive rescue. That is
the *expected* trigger, named in the gate's own failure message; the response
is to re-baseline and record the expiry, not to relax the check.

## Questions this entry leaves open rather than settling

1. **OP-23's content** — absent, and not reconstructed. Either it lands at
   WP5's opening or the `AGENTS.md` citation is withdrawn.
2. **OP-21's underlying question** — whether a `NumericalConfig` holding
   `max_sigma=None` may reach install validation at all. A-27 explicitly
   declined to move this asymmetry; nothing since has moved it.
3. **Whether OP-19's fix changes held-out validation for existing callers.**
   Passing `numerical_config` there changes which `panel_h_m`/`gl_order` govern
   validation for anyone holding a non-default table — a separate observable
   change with its own pin question, and the pin question is unanswerable while
   coverage limit 1 holds.
4. **Sequencing within WP5** — not specified by the outline and not invented
   here.
