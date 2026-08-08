# WP2 opening conditions --- a PROPOSAL, not a decision

> **Superseded for current execution by A-54.** Current Phase 3f status, scope,
> dependencies, gates, corrective-cycle counts, and completion are governed only
> by `docs/phase3f_completion_manifest.yaml`. This file remains historical or
> supporting evidence and must not be updated as a parallel task tracker.

**Status: proposed at A-40, awaiting review. Establishes nothing.** Under the
standing rule adopted at A-39, a decision held for review does not enter
register text in the same pass that raises it, so this lives here and not in
`phase3_record.tex`. `D-44` is deliberately still free. If accepted, this
becomes D-44 in a later pass; if rejected or amended, nothing has to be
unwound.

## The problem this is answering

Five rounds of apparatus repair have now run, and each has found real defects
concentrated at one seam: **the gap between a declared invariant and an
executable gate**. That is not evidence the apparatus is converging. It is
evidence that the seam is productive, and a productive seam does not empty
just because it is worked. "Repair until clean" therefore fails as a stopping
rule for exactly the reason "audit until clean" failed: the predicate is not
one anybody can evaluate, so the loop terminates on fatigue rather than on a
criterion.

The conditions below are chosen to be **finite, enumerable today, and
mechanically checkable**. Each is a thing that is either present in the tree
or is not. None of them is a quality bar.

## The conditions --- four, and the list is closed

**C1. Every item already routed to WP2 has either a landed decision or an
explicit, register-recorded deferral.** The list, exhaustive as of A-40:

| item | what it is |
| --- | --- |
| OP-25 | `data_contracts.enforce`'s `warnings.warn` renders the same `summary()` the raise does, unescaped --- D-40's corollary names *raised* messages, so its scope is the decision |
| CI-9 | an enumerated-value argument validated at construction independently of whether its consuming leg runs (`standardize_cov='nonsense'` with no covariates) |
| D-43 cl.2 | WP1's construction sites, REOPENED under bind-time resolution; the object invariants stay closed |

**The list is closed at this commit.** Items discovered later are routed to
WP5 or to a numbered open item; they do not extend C1. Without that clause C1
is "repair until clean" wearing a table.

**C2. Every per-commit gate is green *and* has one committed capture showing
it capable of failing.** The gate set, exhaustive: fast lane; `pin_compare.py`;
`pin_corpus_identity.py`; the ASCII sweep; the content/decision-monotonicity
checks; the `\hypertarget` structural check; the citation and label sweeps.
A gate with no such capture is **listed by name at WP2's opening as ungated**
--- it is not repaired first. Naming it is the whole discharge; that is what
keeps C2 finite.

> **Amended at round six.** The "listed as ungated" clause is doing real work
> and is the obvious way to satisfy C2 cheaply --- declare everything ungated
> and the condition is met by writing a list. It is therefore **bounded**:
>
> - C2's assessment **states the count in each category**: `GATED` (green with
>   a committed capability-to-fail capture), `UNGATED` (green, no such
>   capture), `RED`. The three must sum to the enumerated gate set, and the
>   set is the one above, closed.
> - **If `UNGATED` is non-empty at round seven, that goes in the finding**,
>   named gate by gate, and is not carried into WP2 as a silently accepted
>   list. An ungated gate entering WP2 unremarked is the same defect as an
>   invariant with no executable check --- the one this whole series exists to
>   close.
> - The counts are reported every time C2 is assessed, not only at the end, so
>   the set cannot drift between assessments without the numbers moving.

**C3. The two-owner census (OP-27) is committed with its denominator and its
method.** Satisfied at A-40. Its standing effect on WP2 is negative: WP2 adds
*rows* to the census as `ModelConfig`/`PriorConfig` land, and does not open a
new OP number per instance.

**C4. WP2's change classification is written before its first commit.**
Specifically: for each field moving into `ModelConfig`/`PriorConfig`, whether
it is BP or SC. Bind-time resolution is observable at some of these sites, so
a package that declares itself behaviour-preserving as a whole would be making
a claim it has not checked. One table, written once, before the work --- not a
per-commit re-derivation.

## What is explicitly NOT required

Stated as flatly as the conditions, because an unstated exclusion is how a
capped list grows.

- **`cox_hawkes_shared` and any sibling modules being reviewed, integrated,
  audited, or made to import.** They are outside the repo under test. Putting
  them under version control (the housekeeping item) does not put them inside
  it, and is not meant to.

  > **Round six: one entry split off, because it was in the wrong list.**
  > Review and integration are correctly *not required*. But "no WP2-scope
  > change alters a signature or default the vendored copy depends on" is a
  > different claim, and it is **not confirmable**, so it moves out of
  > not-required and into its own category. Measured
  > (`probe_a41_vendored_dependency_surface.py`, upstream `f5f1382`):
  > `replication/cox_hawkes_offset.py` reads **44 `args` keys, 33 of them
  > ours**, and **17 are of the kind WP2 is named for** --- including
  > `args['priors']`, which is exactly `PriorConfig`'s quantity, plus `T`,
  > `S`, `sp_var_mu`, the trigger handles and all seven VAE dimensions. So the
  > copy has a live dependency on WP2's surface, read straight out of `args`
  > by a file that does not follow this repository.
  >
  > It cannot be confirmed for two independent reasons, both recorded rather
  > than worked around: (1) `bstpp/cox_hawkes_shared.py` --- the base class and
  > the function that file copies --- is in **no reachable tree**; (2) **WP2's
  > change set does not exist yet**, so the confirmation is a claim about an
  > unenumerated set of changes and is not yet a statement at all. C4 is what
  > makes it one.
  >
  > **Entry: UNMEASURABLE at round six, not NOT-REQUIRED.** It is not a
  > condition on WP2's opening; it is a declared blind spot that C4 must state
  > it does not cover.
- **OP-24's polygon-mode pin.** It is a WP5 entry precondition and stays one.
  WP2 does not touch the polygon surfaces.

  > **Round six: confirmed, not assumed.** All three citations still read
  > *entry precondition* --- `phase3_record.tex` §11 row ("**WP5** entry
  > precondition"), A-36's prose ("a WP5 *entry precondition*, recorded as
  > OP-24, **not a task inside WP5**"), and `AGENTS.md` ("a **WP5 entry
  > precondition**"). Nothing has drifted to "scheduled". A-36's phrase
  > "**A-36 decides: schedule it**" is the decision *to make it a
  > precondition* and is the one place a skim could misread; it is followed in
  > the same sentence by the precondition wording and the reason (an
  > instrument built by the package it is meant to gate is not a gate).
- **OP-19, OP-21, OP-23, OP-26 resolved.** All WP5.
- **OP-18** (the I10 restatement under D-15).
- **A clean `ruff`.** Inherited findings are recorded by name and count; they
  are not a gate.
- **Any further RED-strengthening of already-committed rows.** A-40 strengthens
  the pin-comparator capture because it was flagged; that discharges the
  request, and re-auditing older captures for coarseness is precisely the
  unbounded loop this document exists to stop.
- **A re-baselined pin corpus, or backward compatibility / a shim for
  `standardize_cov`.** Both explicitly declined upstream.

## The cap

**Two further apparatus rounds, i.e. rounds six and seven.** A-40 is round
five.

If C1--C4 are not all met at the end of round seven, **that is a finding about
the plan, not a reason for round eight.** The finding is recorded in the
register in those words, together with which conditions are unmet and why, and
the choice that follows --- open WP2 with the unmet conditions declared as
gaps, or restructure the plan --- is escalated, not absorbed into another
round. The cap has no extension clause. A cap with an extension clause is not
a cap.

## C1 — the list, CLOSED at round seven

Every item has a landed decision or an explicit deferral. **Three items, and
the list does not grow.**

| # | item | decision state |
| --- | --- | --- |
| 1 | **CI-9** | **LANDED (A-39).** The invariant is established: *a config-owned argument drawn from a fixed set of permitted values is validated against that set at construction, independently of whether the leg that consumes it executes.* What is outstanding is its **enforcement**, which is WP2 work, not a decision. |
| 2 | **D-43 cl.2 — `panel_h_m` sentinel leg** | **LANDED (A-36 established, A-39 reviewed and accepted).** `DEFAULT_PANEL_H_M = 20.0` is metres and needs the domain CRS, which is `ModelData`; it is stored as an explicit sentinel and resolved at bind. |
| 3 | **D-43 cl.2 — `standardize_cov` bind-time leg** | **LANDED as a *location* decision (A-39).** But see C4: its **classification** is undetermined pending OP-28, and it is left, not classified provisionally. |
| 4 | **OP-25** | **DEFERRED, explicitly.** Deciding whether D-40's corollary governs text that reaches a console *without being raised* is a decision about the corollary's governance, and A-39's standing rule bars settling it in a bookkeeping pass. The deferral carries content rather than being a shrug: the two candidate answers are (a) widen D-40's corollary to all message-bearing surfaces, which makes the ASCII sweep's population wrong and requires re-deriving 184/184 over a larger denominator, or (b) escape `warnings.warn` at the site and record the asymmetry as deliberate, which is cheaper and leaves the general question open a second time. **Criterion for choosing:** whether any *other* non-raised surface renders a caller-supplied value — if `enforce`'s warning is the only one, (b) is a point fix; if there are others, (a) is the only answer that closes them. That count has not been taken. |

Item 3 is listed separately from item 2 because a single decision asserted
over both legs is the shape C4's constraint 1 exists to forbid.

## C4 — BP/SC/API classification, per item

**Four rows against the closed C1 list; three classified, one undetermined.**
No row is classified on inference: where a classification would require
knowing something not in the repo under test, it says so.

| # | item | class | why |
| --- | --- | --- | --- |
| 1 | CI-9 enforcement | **SC / API** | The accepted configuration space narrows. `standardize_cov='nonsense'` with no covariates **constructs silently today** and would raise; `True` likewise. Observable behaviour changes (raise where there was none) *and* the accept set shrinks, so it is both. Not BP, and calling it BP because "nobody passes nonsense" would be classification on inference. |
| 2 | D-43 cl.2, `panel_h_m` sentinel | **SC** | Resolving metres→CRS units at bind changes the stored `panel_h_m` for any domain whose CRS is not metric — today `create` stores 20.0 unconditionally, so a CRS-less unit-scale domain holds a panel twenty times the domain. Fixing that changes the quadrature panel and therefore the mass table, so **values move**. **And no pin will see it**: all four pinned configurations resolve to rectangle mode, and this quantity only reaches polygon mode (OP-24). A value-moving change with no pin coverage is stated here, not discovered at the commit. |
| 3 | D-43 cl.2, `standardize_cov` bind-time | **UNDETERMINED — pending OP-28** | If `'domain_area'` is the intended estimator, moving its resolution to bind time is a relocation: same computation, different place, values unchanged. If OP-28 resolves the other way, the remedy is likely to move the standardization **away from the clipped-support site** — and the reason the population is restricted at all is that the site's weights were to hand (A-41). That is *the same edit*, and it is then part of a semantic correction with values moving, not a relocation. **The class depends on an answer that is not in the repo under test.** Left. |
| 4 | OP-25 | **UNDETERMINED — decision deferred** | Its class follows its decision: answer (a) is **CF** across a widened surface, answer (b) is **CF** at one site plus a **DOC** declaration of the asymmetry. Classifying before the decision would be classifying a change that has not been chosen. |

**Count: 4 rows. Classified 2. Undetermined 2** — one blocked on OP-28, one
blocked on a deferred decision that C1 records as deferred rather than absent.

### What C4 does not cover, stated where a reader of C4 will see it

**This classification covers the repository under test and nothing else.** In
particular it makes **no claim about the vendored downstream copy.** Measured
at A-41 against upstream `f5f1382`:
`Illegal-Dumping/replication/cox_hawkes_offset.py` reads **44 `args` keys, 33
of them ours, and 17 of the kind WP2 is named for — including
`args['priors']`, which is exactly `PriorConfig`'s quantity** (plus `T`, `S`,
`sp_var_mu`, both trigger handles, the three field-cell counts and all seven
VAE dimensions). It reads them straight out of `args`, in a file that does not
follow this repository.

**`bstpp/cox_hawkes_shared.py` — the base class that file subclasses and the
function it copies — is in no reachable tree.** Every row above is therefore
a classification of the effect on *this* repository. A row marked BP or SC
here is **not** a statement that the vendored copy still works. That surface is
**UNMEASURABLE**, not not-required, and C4 does not cover it.

## C2 — assessed, with counts

Measured over the **committed** tree by
`refactor-patches/phase3f/wp2/probe_a42_gate_capability_census.py` (not the
working directory: an uncommitted red capture is not evidence anyone can
reach). The probe was **wrong twice before it was right, both times in the
direction that would have closed C2** — first on a shared `^FAIL` signature,
then, after this document was staged, because *this section's own description
of the citation sweep's A-40 red matched the signature it was searching for*.
A capture is a file produced by running an instrument, never a document
written about one. Both corrections are recorded in the probe's docstring.

| | count | gates |
| --- | --- | --- |
| **GATED** | **4** | fast lane; `pin_compare.py`; `pin_corpus_identity.py`; ASCII sweep |
| **UNGATED** | **3** | content / decision-monotonicity checks; `\hypertarget` structural check; citation + label sweeps |
| **RED** | **0** | — |
| **sum** | **7** | = the closed gate set |

**The UNGATED three, gate by gate, per the round-six amendment:**

- **content / decision-monotonicity checks** — no committed capture shows it
  reporting a decision-row gap, duplicate or regression. It has printed
  `KNOWN-PREEXISTING 1` on every run in the series and has never been observed
  to fail.
- **`\hypertarget` structural check** — no committed capture shows
  `MISSING n` for `n > 0`. Every recorded run is `MISSING 0`.
- **citation + label sweeps** — this one is the sharpest, because the sweep
  **did** go red twice in this series, at A-40 (`FAIL 5 unreachable
  citation(s)`) and A-41 (`FAIL 2`), and **both captures were overwritten by
  the passing re-run before the commit.** The gate demonstrated its own
  capability and the evidence was destroyed by the capture-then-fix workflow.
  That is worth more than the other two entries put together: it is not that
  the gate cannot fail, it is that the process throws away the proof.

Discharging all three is small, known work — one mutate-run-restore capture
each, the shape A-40's State B and A-41's revert already use. It was **not
done at round seven**, because round seven's declared scope is C1's list
closure and C4's classification and nothing else. Naming it here is the
condition's own prescribed treatment, not a substitute for it.

## Status at the end of round six

| condition | state |
| --- | --- |
| C1 | **moved, not closed.** The three-item list is written and declared closed; what remains is landing a decision or an explicit deferral for each. |
| C2 | not yet assessed. Its category counts are due at round seven, per the amendment above. |
| C3 | **CLOSED.** The census is committed with its denominator and method, and round six supplied the basis statement the denominator was missing: eleven is the *config-anchored* population, complete over that population; the config-external population is **known non-empty** (three adjudicated members) and **unmeasured**. |
| C4 | not started. |

**Round seven's scope is C1's list closure and C4's classification, and
nothing else.** If either needs work beyond that, that is the finding.

## Final states at the end of round seven — and the finding

| condition | state |
| --- | --- |
| **C1** | **CLOSED.** Three items, each with a landed decision or an explicit deferral; the list did not grow. |
| **C2** | **NOT CLOSED.** GATED 4 / UNGATED 3 / RED 0, summing to the closed set of 7. |
| **C3** | **CLOSED** at round six. |
| **C4** | **NOT CLOSED.** The table is written and structurally complete — 4 rows, per item, with the count and the UNMEASURABLE statement in the text. 2 rows classified, 2 undetermined. |

**THE FINDING, in the words the cap requires.** *Two of four opening
conditions are unmet at the end of round seven. That is a finding about the
plan, not a reason for round eight.*

**What remains, exactly:**

- **C2** — three red captures, one per ungated gate. Small, known, mechanical.
  Not blocked on anything.
- **C4** — two classifications. One (row 3) is blocked on **OP-28**, which is
  escalated and is a question about modelling intent, **not repo work**: no
  number of further rounds resolves it, because the answer is not in the
  repository. One (row 4) is blocked on OP-25's deferred decision, and C1
  records that deferral with its two candidate answers and its choosing
  criterion.

**RECOMMENDATION: open WP2 with two declared gaps, not restructure.** The
reasoning, stated so it can be rejected:

1. **The unmet conditions are bounded and named, not diffuse.** C2's three
   gaps are all **document** gates — none of them guards `bstpp/`. An ungated
   `\hypertarget` check cannot let a numerical defect through; it can let a
   broken anchor through. The risk they carry is register hygiene, which is
   real and is not WP2's failure mode.
2. **C4's blocking item is excludable.** The undetermined row is exactly one
   leg — `standardize_cov`'s relocation. WP2 can proceed on the other items
   and **not touch that leg** until OP-28 is answered. A gap you can draw a
   boundary around is a gap; a gap you cannot is a reason to restructure.
3. **Restructuring would not address either cause.** C2's cause is a workflow
   habit (capture, then fix, then overwrite the capture). C4's cause is an open
   question about modelling intent. Neither is a defect in the plan's shape.

**Declared gaps, to be carried in WP2's opening entry rather than assumed:**

- **G-A.** Three per-commit gates are UNGATED — content checks, `\hypertarget`,
  citation + label. Named, counted, and to be discharged as a by-product of
  any WP2 commit touching the register, not as a prerequisite.
- **G-B.** `standardize_cov`'s bind-time relocation is **out of WP2's scope
  until OP-28 is answered.** Any WP2 commit touching
  `attach_covariate_partitions`' standardization block violates this gap and
  should be refused.

**WP2's first commit, if opened: CI-9's enforcement.** Class **SC / API**.
It is the smallest item on the list, it is independent of OP-28's semantics
(the *set* of permitted values is unchanged whichever estimator
`'domain_area'` denotes), it lands an enforcement rather than a relocation,
and it is RED-demonstrable today — `standardize_cov='nonsense'` with no
covariates constructs silently at tip.
