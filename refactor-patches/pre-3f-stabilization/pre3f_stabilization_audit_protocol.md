# Pre-3f stabilization audit protocol

**Status:** proposed entry gate for Phase 3f.
**Claim it supports:** that the semantics 3f will preserve and freeze are sound enough to restructure. Not that the package is defect-free.

This document is normative. The Cursor prompt is its operational wrapper and does not restate it.

## 1. Decision

Do not open 3f yet; do not continue open-ended whole-package auditing. Run one bounded stabilization audit against a preregistered scope, blocker rule, and stopping rule.

"A further general audit found nothing" is not part of the criterion. Absence of findings measures the instrument, not the artifact — which is why §4 requires declared coverage and §9 caps iteration.

## 2. Why this boundary

The preceding findings are not a random stream. Each is a declared contract that lacked executable enforcement: a distribution sampled outside its own support; row-sum where union area was required; hidden rebuild in a scoring path; a mutator corrupting untouched-axis provenance; a numerical default violating its declared error budget; one membership rule with two implementations; supported fit paths that could not initialize under a green 427-test suite.

Two consequences:

1. The missing artifact is traceability between register and gate set, plus coverage of option *combinations*. Re-reading saturated modules has low yield; first coverage of unaudited modules and untested combinations has high yield.
2. 3f is behavior-preserving except for declared API/validation change, gated by trace equivalence against recorded pins. **A defect on a pinned path is therefore a blocker regardless of severity: the gate would certify its preservation and report success.** Severity is not the criterion; position relative to the pins and the freeze surface is.

## 3. Governing sources

Active repository versions of: the Phase 3 baseline/decision/record documents; the full `D-*`, `A-*`, `OP-*`, `I*` registers; the change-classification and acceptance matrices; the current guide and any superseding addenda; pin, trace, smoke, simulator, polygon-numerical, conservation, and SBC records.

Do not assume the supplied historical baseline carries the current highest identifier. Inventory the active register from the repository and report missing, duplicate, superseded, or contradictory entries.

Change classes are `BP`, `IV`, `MR`, `SC`, `CA`, `API`, `DOC`, `DEF`, **and `CF`** (correctness repair implementing an already-declared target, added by A-19). Most repairs arising from this audit are `CF`, not `SC`. Never label a behavior-changing correction `BP`.

## 4. Evidence standard

Every assertion in every deliverable carries one mark:

- **verified** — established here by execution or source reading at the candidate tip, with the command or artifact named.
- **reported** — asserted by a prior pass or by a handoff document, not re-established. A reported item must be reproduced from code before it is treated as real, as fixed, or as unreachable.
- **inferred** — follows from documents, not code. Weakest; documents may be stale.

No finding is closed, and no path is declared unreachable, on the basis of a document's description of it. Prose summaries locate the question; code answers it.

## 5. Blocker rule

A finding blocks 3f if any of:

1. **Pin-path defect** — reachable in a deterministic pin/trace, a required smoke/confirmation path, or the reference generator, so equivalence could fossilize it.
2. **Frozen-surface defect** — it changes or leaves ambiguous configuration, compatibility, defaults, public state transitions, sentinel semantics, rollback, or provenance that 3f will encode in frozen objects.
3. **Gate-validity defect** — it can make a required gate pass for the wrong reason, or fail to exercise its claimed path. Includes environment drift: pins and traces are meaningful only in the declared environment on the declared machine.
4. **Preserved supported-path failure** — a documented supported combination cannot initialize or execute the likelihood, compensator, simulator, or scoring, and 3f intends to preserve that support.
5. **Unresolved normative conflict** — two active sources give incompatible instruction for a 3f design choice.
6. **Stale pin** — a pin, trace, or baseline artifact was recorded before a landed correction to the path it covers. Rebaselining is required before it can gate 3f.

Nonblocking requires a demonstration, not an absence: outside the pin and frozen-configuration surfaces, survivable through 3f without constraining the architecture wrongly, does not invalidate a gate, and entered in the ledger with an owner. "Minor" or "not currently tested" is not a disposition.

Historical or editorial cleanup is 3g or later. A stale document becomes a blocker when it is normative input to 3f.

## 6. Lanes

**Lane A — deterministic reference and pin paths.** For every pin/trace configuration, trace actual production execution from public construction through: data and domain preparation; configuration and default resolution; prior sampling and initialization; decoder preparation and gain; pair construction and membership; event term; background and excitation compensators; the simulation path the baseline represents; deterministics and gradients; recorded provenance. Cover all model families and every discriminating geometry in the baseline, including the nonsquare real-unit discriminator.

Record what pins do **not** reach — expected: polygon support, held-out scoring, setters, custom triggers. Audit the generator and comparison logic itself: configuration identity, environment identity, artifact hash and commit, assertion strength, and whether unused or malformed state could still yield a pass. Name explicitly every defect a trace-equivalence gate would preserve.

**Lane B — configuration, compatibility, mutation, provenance.** Audit constructor and public mutators as one state machine. Axes: model family {plain Hawkes, LGCP, Cox–Hawkes}; support {rectangle, polygon}; trigger {every built-in, custom}; cutoff input {tolerance, physical override, omitted, explicit `None`}; entry path {constructor, each public setter}; outcome {success, each validation/preparation failure point}. Enumerate level sets from code, not from this document.

Pairwise coverage of the feasible space, plus every explicitly supported and every explicitly rejected combination as forced rows. Full Cartesian product is not required.

Assert at every covered point:

- *Admissibility* — constructs, or raises a **named** error identifying the offending combination. A bare `KeyError`, a downstream `AttributeError`, or silent acceptance all fail. Assert error type and message substring, never bare `raises(Exception)`.
- *Leg consistency* — event term, compensator, simulator, and scoring consume the same support and cutoff atoms. Assert by **object identity** where the quantity is single-sourced; numerical agreement elsewhere, which itself records that the leg is not yet single-sourced.
- *Provenance* — touched axis updated, untouched axis preserved bit-identically, whole record round-trips through the results-save path.
- *Transactionality* — after a rejected transition, all public fields, pair set, support object, and provenance equal a pre-call snapshot. Whole-state comparison, not spot checks.
- *Sentinel stability* — omitted vs. explicit `None` vs. explicit value have stable, intentional, documented meanings.
- *Constructor/setter equivalence* — the same final state by either route.
- *Numerical budget* — polygon mode **at shipped defaults** meets `PRODUCTION_TAU_ABS`. **A-21 is authoritative: `PRODUCTION_TAU_ABS = TAU_ABS = 1e-5`** (`polygon_mass.py`). Do not derive this value. The older derivation — omitted mass `eps_s = 1 - erf(w_s / (sqrt(2) sigma))^2`, which at `w_s = 3 sigma` gives `5.3923032e-3`, ten per cent of which is `5.3923032e-4` — governs `TAU_DERIV` (OP-12, open) and survives as `LEGACY_SHOOTOUT_TAU_ABS`. It is **not** the mass-table budget; using it would relax the gate by roughly fifty times.

  The assertion must be a **measured residual against a higher-accuracy reference**, not a resolution surrogate. A panel-to-sigma ratio ceiling is a proxy calibrated at one quadrature order; at a different `gl_order` it admits tables outside budget while the error message names a tolerance nothing measured. Name the reference method and its own error bound in the test.

**Lane C — close classes, not instances.** For each class the preceding passes exposed, search package-wide siblings and specify one reusable guard:

| Class | Required closure |
|---|---|
| Distribution violates its own transformation/support | Parametrized properties over **every** package-defined distribution: sample shape/dtype, declared support, finite in-support `log_prob`, normalization, transformation consistency, JIT/vmap where claimed |
| Geometry total depends on overlapping input rows | Adversarial overlap/duplicate/hole/multipolygon tests proving every "domain area" quantity uses authoritative union semantics |
| Hidden preparation in a scoring path | Preparation contract explicit; test-realization state correct; caller state not silently mutated; training state not reused |
| Public mutator corrupts state or provenance | Shared state-machine test over **every** public mutator: success, sentinels, untouched state, rollback, provenance identity |
| Numerical default violates declared budget | Default-parameter regression against the higher-accuracy reference and `tau_abs`: values, slopes/gradients where used, table bounds, off-knot points, supported precision |
| One semantic predicate, several implementations | Structural single-source check that pins use of the canonical predicate/object — in the manner of the pinned `aligned_difference_pairs` signature — not two hand-written formulas agreeing on examples |
| Capability combinations fail late | The Lane B matrix: every supported combination initializes, every prohibited one fails early with the intended message |

An instance fix without class-level closure is a registered gap even when it is not itself a blocker. New classes discovered are added to this table.

**Lane D — first coverage of unaudited code.** Priority: the current equivalents of `_sim_cox`, `preparation.py`, `data_contracts.py`, and their direct callers. Names may have changed; locate the symbols. Audit: shared mass/integration atoms between likelihood and sampling; clipping and union semantics under holes, duplicates, multipolygons; event rejection and cell membership at domain boundaries and internal grid lines; covariate gaps, overlaps, slivers, NaNs, dtypes, CRS, empty geometries; each family-specific simulation branch and the background/excitation/support state it uses; explicit RNG threading without key reuse; prepared-object immutability and aliasing; agreement between prepared fields and recorded provenance; any silent fallback or rebuild that changes the event configuration or numerical method.

Also close the excitation-leg conservation check (`E[n]` = compensator), which has no standing test.

**Lane E — 3f seams.** Enumerate the seam set explicitly — the readiness rule quantifies over it, so it must be written down — then inspect what must move behind: the frozen configuration objects; prepared data/domain/partition objects; the legacy `args` adapter and its removal sequence; explicit RNG ownership; decoder contract, identity, gain, provenance; plotting delegation and compatibility wrappers.

Report implicit state, duplicated sources of truth, mutation or aliasing, cycles, hidden default resolution, undocumented compatibility behavior — anything that would force 3f to either change semantics or preserve ambiguity. Verify that the decisions needed for the configuration freeze (`OP-3`, `OP-4`, `OP-7`, `OP-8`, `OP-13`, or their active successors) are settled and consistent between code and normative documents. **Do not assume any of these is resolved**; several are recorded open, and an unresolved one needed by 3f is a blocker under §5.5.

This is a seam audit, not a request to design or implement 3f.

## 7. Known-finding closeout

Locate exact implementation sites, fix commits, and regression evidence — not prose summaries — for: `TruncatedLogNormal.sample` transformation; row-sum vs. authoritative union area; held-out scoring and mass-table preparation/rebuild; `set_window` untouched-axis provenance; `panel_h_m` default vs. `tau_abs`; duplicated membership-predicate implementation; trigger capability gates and trigger-specific argument handling (`sigmax_2`, Gaussian/exponential arguments, custom triggers); omitted vs. explicit-`None` transition semantics; non-default `panel_h_m`/`gl_order` rejection; design scales stored on the instance; provenance persistence through the results-save path; dependency metadata drift; any supported fit path that could not initialize.

Per finding record: instance status; class-level test present or absent; siblings searched; correct change class and gate profile; and whether any pin or baseline predates the correction (§5.6).

## 8. Verification after approved repairs

The audit is read-only over production code and tests. On blockers, stop with a repair plan; never mix unapproved fixes into an audit commit.

After repairs land as class-separated commits:

1. focused regression tests — each **demonstrated RED at the pre-fix commit** before being made GREEN; a test written only after the fix does not discharge this, since it has not been shown to discriminate the defect;
2. Lane B compatibility and state-transition matrix;
3. full suite;
4. lint and static checks;
5. dependency/environment reproducibility;
6. deterministic traces and rectangular pins on the designated analysis machine;
7. rectangle degeneracy, polygon numerical-reference and default-accuracy, conservation, and global JAX-state gates;
8. all-family smoke/confirmation paths touching changed code;
9. the existing conditional SBC escalation policy where a model-specific path changed, equivalence failed, or a confirmation anomaly appeared.

SBC and code audit have near-disjoint sensitivity: none of the preceding findings would have been caught by a rectangular unit-gain Stage 3 run, and no amount of code review establishes calibration. Neither is evidence about the other's domain. An SBC checkpoint may run in parallel with the read-only audit; it does not substitute for the mandatory Phase 3 exit run at the tip.

## 9. Stopping rule, iteration cap, verdict

One audit pass ends when all five lanes have completed coverage entries and every active register entry has a traceability disposition. A nonblocking finding does not restart it.

**Iteration.** If a pass finds blockers: repair under §8, then re-audit only the surfaces the repairs touched plus any lane whose coverage the repairs invalidated. Each iteration records its candidate tip by SHA.

**Cap: three iterations.** Iteration *k+1* must differ from iteration *k* in auditor or in method — correlated blind spots mean a repeated search is not evidence. If the third iteration still finds a blocker, stop and hold a scope review instead of a fourth pass: the evidence then supports "the semantic surface is not yet specified," not "one more fix is needed," and the correct response may be to move work into 3f as *declared* semantic change rather than to keep asserting behavior preservation.

Declare **READY FOR 3F** only when:

- no open blocker under §5;
- every preceding finding has an exact closeout record;
- every blocker repair has a focused regression test, shown RED first, with the correct change class;
- the Lane B matrix passes;
- §8 gates pass on one frozen candidate tip;
- decisions needed for 3f configuration and sequencing are settled and noncontradictory;
- every residual gap has an owner (3f, 3g, post-Phase-3), rationale, and review date;
- the coverage map names every area not audited, so silence is not reported as coverage.

Otherwise **NOT READY FOR 3F**: list only the unmet conditions and the smallest class-separated repair sequence.

After a READY verdict, new discoveries are triaged under the same blocker rule. They do not reopen a package-wide loop.

## 10. Deliverable schemas

Five versioned records. No empty cells; `n/a` with a reason is acceptable, blank is not.

**`traceability_matrix.md`** — one row per active `D-*`, `A-*`, `OP-*`, `I*`.

| ID | Contract | Register status | Production sites | Execution legs | Existing evidence | Evidence type | Gap | Treatment | Owner |

*Register status*: active, superseded, open, contradictory. *Evidence type*: direct, property/reduction, structural, integration, manual review, none. *Treatment*: named executable check, or documented reason it cannot be executable plus a named review check, or registered gap with rationale. Name the test and state which assertion proves the contract; "covered by suite" is not an entry. Distinguish a test that *executes* a path from one that *discriminates* the defect.

**`pin_path_map.md`** — one row per pin/trace configuration.

| Pin/config | Public entry | Resolved configuration | Production symbols reached | Branches not reached | Assertions and discriminators | Provenance | Artifact predates a correction? |

**`audit_coverage_map.md`** — one row per module or area.

| Area/symbol | First audited here? | Methods | Configurations/legs | Result | Residual gap |

*Methods*: call-path tracing, adversarial example, property review, numerical reference comparison, state-transition review, provenance round trip, structural single-source check. "Read code" is not a method.

**`findings_ledger.md`**

| ID | Finding or gap | Contract IDs | Evidence (mark + command/artifact) | Production reachability | Pin or frozen surface? | Change class | Severity | Class-level remediation | Required gates | Owner | Review date | Status |

*Severity*: `BLOCKER`; `NONBLOCKING-3F` (resolvable during 3f without changing its semantic reference); `NONBLOCKING-3G` (verification/documentation closeout); `DEFERRED` (outside Phase 3, with rationale). No finding carries "minor" instead of a disposition.

**`readiness_report.md`** — exactly one headline, `READY FOR 3F` or `NOT READY FOR 3F`, the conditions of §9 each marked met/unmet, and on NOT READY the repair sequence.
