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

## Coverage limits — both established below; limit 1 is DISCHARGED at A-47

**1. ~~No pinned configuration builds a polygon mass table, so `PIN_DIFFS` is
silent on every surface WP5 touches.~~ DISCHARGED at A-47 — forward only.**

*The limit as it stood, kept because the discharge is only interpretable
against it.* Measured, not asserted: `refactor-patches/pin_check_v2.py` emitted
**four** configurations — `hawkes`, `cox_hawkes`, `lgcp`,
`hawkes_nonsquare_4to1` (confirmed against the top-level keys of
`baselines-2026-07/pins.json`) — and the harness contained **zero occurrences**
of `polygon`, `mass_table`, `excitation_support` or `min_sigma`. The polygon
surfaces were not merely unexercised; they were unreachable from the harness as
written. So `PIN_DIFFS 0 MATCH` was **not evidence** about `ExcitationSupport`,
`PolygonMassTable` or cutoff provenance, which is precisely the set WP5 changes.

**What A-47 landed.** Two configurations —
`hawkes_notched_4to1_polygon_mode` and `hawkes_notched_4to1_rectangle_mode` —
over one 4:1 **non-rectangular** domain with the same events and σ bounds, so
**both excitation support modes are pinned and the mode switch is the only
difference between the two records.** All four previously-absent tokens are
reachable. The mass table's SHA-256 is inside the pinned record, so a
differently-built table is a DRIFT rather than a silent shift.

**Read the discharge with its two limits attached.**

- **It is FORWARD.** The baseline is `baselines-2026-08-polygon/pins.json`,
  dated at its capture. It certifies commits after that capture and **nothing
  before it**; six configurations are not six configurations of retroactive
  coverage.
- **The canonical comparison still does not gate it.** Against
  `baselines-2026-07/pins.json` the two new keys report `NEW IN CANDIDATE`, and
  `pin_compare`'s walker counts a new key as no diff. **Any WP5 gate line must
  cite the forward baseline explicitly** (`pin_compare.py --baseline`), or it
  reports `PIN_DIFFS 0 MATCH` while remaining as silent about the polygon
  regime as it was before A-47.

**2. OP-24's polygon-mode pin is a WP5 ENTRY PRECONDITION, not a task inside
WP5.** A-36's reason: *an instrument built by the package it is meant to gate
is not a gate.* The alternative — opening WP5 declaring itself uncovered — was
considered and rejected there for a specific reason: WP5's surfaces are exactly
the ones the existing pins are blind to, so "proceeding without coverage" and
"proceeding with the coverage we have" would produce identical gate lines.

### Where the scheduled instrument stands — BUILT at A-47

**Polygon-regime excitation validation was SCHEDULED, not deferred** — A-36
records the decision in those words ("A-36 decides: schedule it") and
deliberately recorded it then "so it is not carried into WP5's opening as
scheduling TBD".

**It was unbuilt as of A-43 and is built as of A-47.** The harness emits six
configurations, two of them polygon-domain, and **the entry precondition is
discharged.**

**Discharged is not the same as open.** WP5's opening remains a separate
decision that A-47 explicitly declines to take: **OP-23's text still does not
exist**, which is its own condition at WP5's opening, and A-44's two declared
gaps stand.

The consequence recorded here in advance did occur, exactly as predicted: the
fifth pin added a configuration key the 2026-07 baseline does not carry, so
`pin_corpus_identity.py` went red and **A-38's retroactive rescue expired at
A-47**. The check was not relaxed, the red is preserved at
`refactor-patches/captures/a47_corpus_identity_expiry.log`, and A-47 opens
**OP-31** for what the expiry costs that gate (its exit status is saturated, so
a *third* content group is now indistinguishable from the second).

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
