# WP9 — results I/O (G2): entry

**Class: API — declared**, and this is the one package the outline marks as
not-BP (`phase3f_work_package_outline.md`).
**Status: PLACEHOLDER with one substantive element** — G2's deliverable and
its gate are already declared; everything else is unspecified.

## Seams

Results I/O — **G2**, i.e. `save_rslts` and the load path, plus the provenance
the record carries across a round trip.

## Open items routed here

**None by name.** No §11 row names WP9.

**But one item routes to WP9's deliverable, from before the `WPn` numbering
existed.** **OP-15** (closeout §6 P3, provenance completeness in Lane B) is
routed to *"3f G2 + Lane B hardening"*, and its discharge is split in the row
itself: *"`save_rslts` half by G2; untouched-axis half by standing Lane B
hardening."* **The `save_rslts` half is WP9's.** This is recorded as a
by-deliverable routing rather than silently converted into a by-package one —
the row still says G2, and changing what a row says is not this entry's job.

## Constraints and known limits already in force

- **G2 is a declared API change with its gate already specified.** A-22's
  closeout ledger: *"`save_rslts` implements the A-21 save/load contract with
  provenance and hard-fail incompatibility. Declared **API** change; Lane B
  strict xfail flips to a passing assertion in the same commit."* The outline's
  class column agrees independently.
- **The gate exists today and is red-by-design.**
  `tests/test_lane_b_config_matrix.py:1319`,
  `test_lane_b_save_rslts_roundtrips_cutoff_provenance`, is under
  `@pytest.mark.xfail(strict=...)`. It is the `1 xfailed` in every fast-lane
  figure in this series. **The flip is the exit check**, not an afterthought:
  strict xfail means it will fail loudly if the round trip starts working
  without the declaration landing.
- **Provenance completeness is only half-covered.** Per OP-15: Lane B's
  covering-array success path checks provenance *presence*, not untouched-axis
  *bit-identity*. WP9 inherits the presence check and not the identity one.

## Questions this entry leaves open

1. **What the save/load format is**, and whether it is a format WP10's `args`
   removal will invalidate. `save_rslts` today serialises a record shaped by
   `args`; WP10 dissolves `args`. Which package owns that ordering is not
   stated.
2. **What "hard-fail incompatibility" fails on.** Version, schema, environment,
   pin identity — the phrase is declared, its predicate is not.
3. **Whether OP-15's `save_rslts` half discharges here formally**, i.e. whether
   the §11 row should be re-pointed at WP9 now that WP9 has an entry. Under the
   standing rule adopted at A-43 it *may* now be routed here; re-pointing an
   existing row is a register edit and is not done unasked.
4. **Scope and sequencing.** Not specified, not invented.
