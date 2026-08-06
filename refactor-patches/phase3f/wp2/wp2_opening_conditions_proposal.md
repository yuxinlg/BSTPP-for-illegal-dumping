# WP2 opening conditions --- a PROPOSAL, not a decision

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
- **OP-24's polygon-mode pin.** It is a WP5 entry precondition and stays one.
  WP2 does not touch the polygon surfaces.
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
