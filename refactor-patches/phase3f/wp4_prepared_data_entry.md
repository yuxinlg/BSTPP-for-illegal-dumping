# WP4 — prepared domain and partitions: entry

> **Superseded for current execution by A-54.** Current Phase 3f status, scope,
> dependencies, gates, corrective-cycle counts, and completion are governed only
> by `docs/phase3f_completion_manifest.yaml`. This file remains historical or
> supporting evidence and must not be updated as a parallel task tracker.

**Class: BP** (`phase3f_work_package_outline.md`).
**Status: PLACEHOLDER.** Seams named, scope not yet specified.

## Seams

`ModelData` · `PreparedDomain` · `PreparedPartitions` — in
`bstpp/preparation.py` and `bstpp/spatial_grid_helpers.py`.

**These three objects already exist.** They landed at Phase 3b (A-4, "the
three-object seam"), so WP4 is not creating them; what WP4 does to them is not
specified.

## Open items routed here

**None.** No §11 row names WP4.

## Constraints and known limits already in force

- **The adapter is an *identity* adapter** (A-4): each `args` entry the seam
  owns is assigned the seam object's field *itself*, not a copy. A copying
  adapter would let the two drift. Any WP4 change has to preserve that or
  declare it changed.
- **A temporary asymmetry is already recorded here** (A-4): *"Covariate event
  membership deliberately stayed event-side in the constructor rather than
  moving into the seam, purely to preserve the legacy operation order exactly.
  That is a temporary asymmetry, recorded so it is not mistaken for a design
  intention."* It is still event-side.
- **Three named contracts are deferred, not classes** (`AGENTS.md`):
  `ModelDomain`, `ReportingRegions`, `ComputationPartition`. Whether WP4 is
  where they become classes is not stated.
- **D-19, city-scale**: architecture must merely avoid *precluding* later
  city-scale work — no hard-coded single-domain assumptions in the seam objects
  — without building unused generality. A live constraint on this seam, and see
  the adjacency below.

## Adjacencies — subject matter, NOT routings

Recorded so they are visible, and labelled so they are not read as
reassignments. **Nothing below is routed to WP4.**

- **OP-27's three config-external census rows** (`T_INTERNAL`/`args['T']`,
  `S_INTERNAL`/`args['S']`, `S_DAYS`/`self.S`) are *partition* quantities —
  `preparation.py` names them — but they are **routed to WP10**, because the
  literal writes are writes into `args`.
- **OP-28** blocks district-wide and city-wide modes, which is the same
  city-scale direction D-19 constrains this seam against precluding. OP-28 is
  **escalated**, not routed here.

## Questions this entry leaves open

1. **What WP4 does to objects that already exist.** Tighten them, split them,
   move membership into the seam, promote the three deferred contracts — none
   of this is stated.
2. **Whether the event-side membership asymmetry is WP4's to close or WP6's.**
   WP6's seam includes "G1 membership single-source"; WP4's includes the seam
   the asymmetry was kept out of. The boundary is not drawn.
3. **Scope and sequencing.** Not specified, not invented.
