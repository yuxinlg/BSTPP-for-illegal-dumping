# Phase 3f execution protocol

Status: authoritative when adopted by A-54
Manifest: `docs/phase3f_completion_manifest.yaml`
Baseline for this protocol: `refactor` at `c4e069f` (2026-08-07)

## 1. Purpose

Phase 3f completes the architecture rewrite specified in `phase3_record.tex`
section 10.f. Its unit of progress is a completed, verified deliverable — not
an audit round, amendment count, commit count, or `WPn.m` label.

The completion manifest is the sole source of current execution truth. The
Phase 3 record remains the append-only scientific and historical authority.
Older dependency graphs, seam-set files, package entries, findings, and gate
captures remain evidence of what was known when written; they are not current
task trackers.

**The goal this protocol serves is resuming the analysis.** That happens at the
S3 boundary (D-62), not at Phase 3f exit. S4 is internal hygiene behind an
unchanged public surface.

## 2. Non-negotiable constraints

The reset changes project management, not the model. The following remain in
force:

- The 3a–3g phase boundary is frozen. Phase 3f is architecture completion;
  Phase 3g owns final statistical verification and guide/documentation work.
- Behavior-preserving work preserves public sampled-site names,
  deterministic-site names, likelihood values, and supported public calls.
- A behavior-changing correction is classified honestly as CF, IV, SC, CA, or
  API. It may not be described as BP.
- A new invariant ships with its enforcement. When an enforcement or bug fix is
  claimed, preserve a minimal RED capture before the fix and a GREEN capture
  after it.
- Pure structural extraction does not require a manufactured RED. Characterize
  the old behavior before the edit and demonstrate equivalence afterward.
- Prepared data and geometry remain outside frozen configuration.
- Configuration objects do not read `ModelData`; data-dependent defaults use an
  explicit sentinel and resolve at bind time.
- Missing decoder-training provenance remains `UNKNOWN`. It is never inferred
  from filenames or reconstructed from plausibility.
- The `cbg-park-seasonal` analysis is the downstream focus. The absent
  historical `cox_hawkes_shared` feature is unsupported and creates no Phase 3
  compatibility obligation.

## 3. How work is organized

### 3.1 Vertical slices

| Slice | Work packages | Outcome |
|---|---|---|
| S1 — configuration spine | WP2, remaining WP1, WP3 | Five frozen configs, one owner per configuration quantity, legacy adapters retained |
| S2 — runtime ownership | WP4, WP5, WP6 | Prepared/runtime objects own state; shared atoms and transactional mutation |
| S3 — artifacts and interfaces | WP7, WP8, WP9 | Decoder contract, typed metadata ownership, versioned results. **Analysis may resume here.** |
| S4 — architectural cutover | WP10 | Plot delegation, thin wrappers, direct typed-state consumption, `args` removed |

The manifest records the order within each slice and every package's exit
conditions. A work-package number is an identifier, not a schedule.

**Indicative budget: 12–18 production commits across S1–S4, and at most one
register amendment per slice.** These are planning figures, not gates. A slice
consuming twice its share is a signal to re-read the slice brief, not a
violation.

### 3.2 Commit boundaries

Use small commits, but choose boundaries by an executable contract rather than
by administrative round:

1. one invariant or public compatibility rule;
2. one typed owner plus its adapter;
3. one consumer migration with an equivalence gate;
4. one explicitly classified correctness/API change;
5. one slice-boundary rebaseline record.

Do not mix unrelated BP extraction and behavior-changing repair. Do not create
an amendment merely to record that another audit pass occurred.

### 3.3 Corrective-depth rule

A work package may absorb at most **three corrective cycles**. A corrective
cycle is a landed production implementation that must be repaired because its
own declared exit gate failed. Planned commits, documentation updates,
RED-to-GREEN completion of the same declared change, and `WPn.m` label
increments are **not** cycles.

The count lives in the manifest's `corrective_cycles_used` field, incremented in
the repairing commit. At the third cycle, record the failed assumption, split or
narrow the work package, and obtain a decision. Do not respond with a fourth
general audit.

## 4. Before a slice opens

Update the manifest — do not add a planning file. The slice brief consists of:

- each package's `scope` field, declared in the opening commit;
- exact deliverables and completion owner;
- production files expected to change;
- public behavior expected to remain stable;
- any declared behavior/API change;
- tests that characterize the pre-change behavior;
- relevant pin configurations and whether they reach the changed surface;
- targeted negative tests for new validation;
- planned commit boundaries, listed in advance;
- slice-boundary gates, exclusions, and non-goals.

Pin reachability is recorded qualitatively against the changed surface. A
statement-count ratio may be recorded as a diagnostic; it is not an entry
precondition and does not substitute for a targeted test.

Raise all decisions the slice needs **together, before production work**. If a
newly discovered question would materially change the model, API, or result
schema, stop that part of the slice and ask; continue independent parts.

## 5. Engineering practice

These are house rules for the code this phase produces, not process ceremony.

### 5.1 Configuration objects

- `@dataclass(frozen=True)`, one public factory per object, validation in
  `__post_init__`.
- No config object reads `ModelData`. A field whose correct value depends on a
  data fact holds an explicit sentinel and resolves at bind time. `panel_h_m` is
  the live instance: the module default `DEFAULT_PANEL_H_M = 20.0` is silently
  wrong by orders of magnitude for a CRS-less domain, so it must not be the
  unresolved value.
- Every object exposes a deterministic `to_record()` returning only versioned
  mappings, scalars, strings, booleans, `None`, and arrays with explicit
  dtype/shape. Results schema v1 consumes these; designing them late is how the
  schema ends up serializing internals.
- **Reuse the `CI-n` sequence and the existing `require_config_*` helpers.** One
  invariant is one number, one identity, one clause rendered byte-for-byte at
  every site that detects it. A second unenforcing site of an existing invariant
  takes that invariant's number. Do not fork an exception class per config
  object.
- Relocating landed enforcement into `__post_init__` is BP only if the clause
  text and exception identity are unchanged. Changing either is SC/API and is
  declared.

### 5.2 Adapters

- Legacy constructor and `run_svi`/`run_mcmc` keywords stay accepted throughout
  Phase 3f. Keep the mapping in **one registry** — a single declared
  keyword-to-owner table — rather than shims scattered across call sites. The
  registry is what WP10 deletes.
- An adapter translates. It does not validate differently, coerce differently,
  or accept a value the typed owner rejects.

### 5.3 Tests

- One test module per contract; negative tests colocated with the contract they
  bound.
- A new invariant lands with a test that matches on the **canonical clause
  text**, not on an exception type. An incidental `TypeError` is not
  enforcement.
- When an existing test must change to detect the new behavior, that is a
  planned test edit: name it in the slice brief, get sign-off, and preserve the
  pre-edit state as the RED capture. The G2 xfail is the known instance.

### 5.4 Restraint

- No abstraction for city-scale, multiresolution, reporting regions, or
  computation partitions until a decision asks for one. Keep evolution possible;
  do not build for it.
- Do not rename a public symbol during a BP commit.
- Do not repair a historical artifact to make it look current.

## 6. Verification profiles

### 6.1 Per production commit

Run only gates that cover the changed surface:

1. targeted tests for the changed contract;
2. the opt-in fast lane:

   ```bash
   JAX_PLATFORM_NAME=cpu "$PY" -m pytest tests/ -q -m "not slow"
   ```

3. the repository's touched-file ruff population instrument, naming the files
   and inherited findings;
4. document structure/citation gates **only if a file in their declared
   population changed** — those populations are `phase3_record.tex`,
   `docs/*.md`, and `AGENTS.md`. The manifest is YAML and is outside them; a
   manifest-only edit runs the manifest validator alone;
5. targeted pins only if the changed code is reached by those pins.

Every capture records the checked process's own exit status. Preserve a RED
capture under `refactor-patches/captures/` before applying a fix. Write GREEN
outputs outside the populations scanned by document-census tools.

**Gate outputs use the current amendment's prefix.** `_a53_*` belongs to
`c4e069f` and is never written to; the prefix *is* the amendment number.

### 6.2 Definition of done, per commit

A commit is done when all of the following hold and are stated in its message:

- its declared contract has a targeted test that fails without it;
- its class (BP, IV, SC, CF, CA, API) is stated and defensible;
- the fast lane passes with its exit status recorded;
- every applicable escalation in 6.3 has run;
- the manifest reflects any status, scope, dependency, blocker, or cycle-count
  change, in this same commit;
- the file list matches what was pre-stated.

### 6.3 Change-triggered escalations

| Changed surface | Additional required gate |
|---|---|
| Polygon mass, support, conservation, or `set_window` spatial leg | Relevant slow tests and the 2026-08 six-configuration forward pins |
| Pair builder or simulator/likelihood excitation coupling | Pair tests, conservation/distributional checks, smoke simulation |
| Decoder loader, contract, shapes, gain, or partition dimensions | Decoder-contract negative tests and LGCP/Cox-Hawkes smoke, with artifacts present |
| SVI/NUTS entry or RNG routing | Relevant slow smoke tests and deterministic explicit-key comparison |
| Packaging/runtime metadata | Packaging slow test |
| Saved-results schema or compatibility | G2 round trip, negative compatibility matrix, trusted-pickle warning tests |
| Sampled/deterministic site registration | Deterministic trace comparison and all affected family pins |
| Public constructor accept set | Both pin baselines and both lanes — a narrowed accept set reaches every model build |

### 6.4 Slice boundary

At each vertical-slice boundary:

- run fast and slow lanes as separate processes in the pinned environment;
- run the canonical 2026-07 pin comparison and state its four-of-six shared
  population rather than calling the partial comparison a whole-corpus match;
- run the 2026-08 forward comparison and require six-of-six `MATCH`;
- run affected end-to-end family smoke tests across the three configurations;
- create one rebaseline record for the slice;
- update the completion manifest in the same commit that closes deliverables.

The boundary record contains commands, exit codes, environment identity,
candidate and baseline hashes, comparison populations, and any intentional
drift. It does not copy the history of earlier amendments.

### 6.5 Analysis resumption (S3 boundary)

The S3 boundary record additionally states that the `milestones.analysis_
resumption_point.required` list in the manifest is satisfied at one named
commit, and names that commit. Downstream analysis may resume against it. S4
proceeds afterward and must not change any result computed there.

### 6.6 Phase 3f exit

The Phase 3f exit gate is the `phase3f_exit.required` list in the manifest. All
items must be satisfied at one named commit. Phase 3f does not close merely
because the suite is green or another audit found nothing.

### 6.7 Phase 3 exit

Phase 3g, not an individual work package, owns the new full Cox-Hawkes Stage 3
SBC run at `R=200` at the Phase 3 tip, final equation-to-code and implementation
documentation, and the final `cbg-park-seasonal` confirmation. The SBC run is a
parallel planned exit track, not a reason to rerun SBC after every BP commit.

## 7. Status and documentation discipline

### 7.1 One current-state edit

For any production commit, update at most these current-state surfaces:

1. the completion manifest;
2. a slice rebaseline record when a slice closes;
3. the Phase 3 register only when a decision, classification, correction, or
   historical fact must enter the append-only record.

`AGENTS.md` contains stable engineering instructions and a pointer to the
manifest, not a duplicated narrative of every open item.

### 7.2 Contracts carry no lineage

`docs/decoder_contract.md` and `docs/results_format.md` state rules without
register IDs. Their provenance lives in the adopting amendment and the
manifest's `settled_decisions` block. This protocol and `AGENTS.md` may name
IDs; for them the governing decision is the content.

### 7.3 Historical records

Do not rewrite old captures, archived call sites, or old package entries to make
them current. Add a short banner pointing to the manifest if a reader might
mistake them for current instructions. Do not fill their placeholders and do not
repair their old measurements.

### 7.4 Findings

Classify a finding immediately:

- **blocker** — violates a settled model/API contract or prevents an exit gate;
- **in-slice defect** — belongs to the active deliverable and receives a
  targeted regression test;
- **later-package item** — add it to the named package in the manifest;
- **deferred/non-goal** — state the owner or exclusion and stop investigating;
- **tooling issue** — repair only if it invalidates evidence needed for the
  active change.

Do not open a new OP number for another instance of an already named class. Do
not invent text for reserved OP-23.

## 8. Stopping and escalation

### Stop a work package when

- every deliverable and exit gate in its manifest entry is complete;
- all blockers assigned to it are closed or explicitly deferred;
- no known failing applicable gate remains.

### Stop a slice when

- all packages in the slice meet their exit conditions;
- the slice-boundary gate is green at a named commit;
- the manifest is updated and the rebaseline record is committed.

### Escalate when

- a choice changes the scientific target, accepted input space, serialized
  schema, sampled-site interface, or public signature;
- the third corrective cycle reveals the package is not decomposable as scoped;
- required evidence cannot be produced in the pinned environment;
- a protected permission or external dependency blocks the requested work.

### Never use as a stopping rule

- audit until clean;
- no findings on pass N;
- all files have been reread;
- the amendment count stopped increasing;
- the same green gate was recaptured after an unrelated documentation edit.

## 9. Immediate execution sequence

After the operational-reset documents land:

1. Record the adoption commit in the manifest's separate adoption field.
2. Open S1 by declaring the WP2, WP1, and WP3 `scope` fields in one commit.
3. Land `ModelConfig`, then `PriorConfig`, each as a frozen dataclass with one
   factory and its adapter, relocating the already-enforced CI-7/CI-9/CI-10
   clauses without altering their text or exception identity.
4. Resolve the `panel_h_m` sentinel through the config bind API and route the
   cutoff-tolerance construction sites through the bound instance.
5. Land `PartitionDecoderConfig` and `InferenceRunConfig` with deterministic
   `to_record()` snapshots, then close S1 with a boundary record.
6. Continue through S2–S4, each opening only after the preceding boundary
   record exists.

The first implementation prompt after this document is for **S1 as a whole**,
with its commit boundaries listed in advance. It is not another general
Phase 3f audit.
