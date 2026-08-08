# Phase 3f work-package dependency graph

> **Superseded for current execution by A-54.** Current Phase 3f status, scope,
> dependencies, gates, corrective-cycle counts, and completion are governed only
> by `docs/phase3f_completion_manifest.yaml`. This file remains historical or
> supporting evidence and must not be updated as a parallel task tracker.

**Authority:** D-47 (A-49). The work-package index is an identifier; this
file governs execution order. Where this graph and the index disagree, the
graph wins and the disagreement is recorded here rather than resolved by
renumbering.

**Register tip at authorship:** `f40591e` (A-48 tip; A-49 lands on top).
Pins and ρ_w measurements are MACHINE-LOCAL.

**Evidence marks:** *verified-here* / *reported* / *inferred* as in
`phase3_record.tex` Part II. Cells marked **undetermined** have not been
examined; that is distinct from “no precondition”.

**D-43 (construction DAG)** is *verified-here* in the register (A-36;
ratified A-39): `ModelConfig → NumericalConfig` and
`ModelConfig → PriorConfig`, with `ModelData` outside the DAG. The 1↔2
edge below is annotated with that decision.

**OP-24 status:** *verified-here* **CLOSED by A-47** (both excitation
modes pinned). D-49’s process rule stands; the WP5 entry precondition is
discharged.

**Coverage unit for ρ_w:** **statements** — declared at **A-51** under D-48 and
held fixed. Statements rather than branches because ρ_w is a *reachability*
ratio over a declared seam set: branch coverage would make ρ_w depend on how a
seam's internal control flow happens to be written, so refactoring a seam
would move ρ_w without changing anything about what the pins reach.

**Empty seam sets (A-51).** Where a package's deliverable is to *create* the
objects in its seam set, |S_w| = 0 at open time and ρ_w = 0/0 is **undefined,
not zero** — D-48's "ρ_w = 0 ⇒ pin gate vacuous" clause does not apply, and the
cell reads `N/A (empty seam set)`. **WP2 is in this state**: `ModelConfig` and
`PriorConfig` do not exist at `c7b9ef0`.

| WP | Depends on (graph edges) | Entry preconditions | ρ_w | Open items routed here |
|---|---|---|---|---|
| 1 | — (reopened under **D-43**) | closed | [[FILL: measure]] | — |
| 2 | 1; **D-43** on the 1↔2 edge | OP-20 series closed (*verified-here*, A-33) | **N/A (empty seam set)** — *verified-here* A-51 | `sp_var_mu` CI-7 (next commit); D-43 cl.2 `panel_h_m` sentinel; D-43 cl.2 `standardize_cov` (blocked, G-B); OP-25 (blocked) |
| 3 | 2 | **undetermined** | [[FILL: measure]] | [[FILL: enumerate]] |
| 4 | 3 | **undetermined** | [[FILL: measure]] | [[FILL: enumerate]] |
| 5 | 4 | **OP-24 polygon-mode pin** — **discharged (A-47)**; polygon-regime excitation validation status stated | was zero before OP-24 (*reported*); remeasure after A-47 | OP-19; OP-21; OP-23 (content absent); OP-26 (*verified-here* rows) |
| 6 | 2 | note: F1 `set_window` staleness already discharged in WP1.4 (*reported*) | [[FILL: measure]] | OP-27 closure path if still open (*reported*) |
| 7 | 4 | **undetermined** | [[FILL: measure]] | custom-decoder support; provenance metadata (*reported*) |
| 8 | 4 | **undetermined** | [[FILL: measure]] | [[FILL: enumerate]] |
| 9 | — (independent of 2–8) | G2 xfail disposition (*reported*) | [[FILL: measure]] | [[FILL: enumerate]] |
| 10 | all | intake frozen (**D-51**) at **`c7b9ef0`** (named A-51) | [[FILL: measure]] | `T_INTERNAL` vs. `args['T']` (*reported*) |

## Graph properties this table must keep

1. An edge that contradicts the index is annotated with the decision that
   created it — here **D-43** on the 1↔2 edge.
2. A work package with no recorded entry precondition is marked
   **undetermined**, never left blank.
3. Where ρ_w = 0, the pin gate is vacuous for that package (D-48) and the
   separately constructed entry gate is named in the precondition column.
4. Destinations must resolve (D-44): every routed open item names a WP
   that has a register entry.

## Related decisions (A-49)

| ID | Rule |
|---|---|
| D-47 | Order is this graph, not the WP integer |
| D-48 | Gate coverage (ρ_w) must be declared before the gate is read as evidence |
| D-49 | WP5 polygon-mode pin is an entry precondition on a parallel track (OP-24 discharged A-47) |
| D-50 | Corrective depth capped at three per package (`WP n.1`–`WP n.3`) |
| D-51 | WP10 intake freezes at a named tip |

## Exit criteria outside the graph

- Stage 3 SBC at R = 200 at the Phase 3 tip (D-20 / §12.1) — parallel
  track; harness hash recorded; not a work-package gate.
- Phase 3g — unaffected.
