# The finding at the cap — Phase 3f, WP2 opening conditions

**Registered at A-44, 2026-08-06.** Seven apparatus rounds ran under a cap of
seven. **Two of four opening conditions are unmet. That is a finding about the
plan, not a reason for round eight.** No further apparatus rounds.

This document is written to be read by someone reconstructing the plan later,
so it repeats what it needs rather than only citing it.

---

## ⚠ Before you resolve any reference in this document: `Wn` ≠ `WPn`

**Two work-package sequences exist and they collide.** The completed pre-3f
stabilization programme numbers its packages `W0`–`W10`
(`refactor-patches/pre-3f-stabilization/phase3e_closeout_and_3f_readiness.tex`).
Phase 3f numbers its packages `WP1`–`WP10`
(`refactor-patches/phase3f/phase3f_work_package_outline.md`). They mean
different things at the same numbers:

| n | `Wn` — pre-3f, **complete** | `WPn` — Phase 3f |
| --- | --- | --- |
| 5 | `docs/register_test_traceability.md` | `ExcitationSupport`, `PolygonMassTable`, cutoff provenance — **blocked, cannot open** |
| 6 | `docs/audit_coverage_map.md` + audit the unaudited seam modules | public mutators + G1 membership single-source |
| 10 | final bounded audit; 3f opens | `args` removal (OP-8) |

Every reference below is `WPn`. Same hazard `AGENTS.md` flags for `I1`–`I12`
(model identities) against `CI-1`–`CI-9` (config invariants): **read the
prefix.**

---

## 1. The finding

| condition | state at cap |
| --- | --- |
| **C1** — every item routed to WP2 has a landed decision or explicit deferral | **CLOSED** (see §3 — the method had a known blind spot) |
| **C2** — every per-commit gate green *and* holding a committed capability-to-fail capture | **NOT CLOSED** — GATED 4 / UNGATED 3 / RED 0, sum 7 |
| **C3** — the two-owner census committed with denominator and method | **CLOSED** |
| **C4** — per-item BP/SC/API classification written before WP2's first commit | **NOT CLOSED** — 4 rows, 2 classified, 2 undetermined |

**Recommendation adopted: open WP2 with two declared gaps (§8). Not
restructure.** The unmet conditions are bounded and named; C4's blocking item
is excludable; and restructuring addresses neither cause — C2's is a workflow
habit and C4's is an open question about modelling intent, and neither is a
defect in the plan's shape.

---

## 2. C2's count is provisional, and convergence is not demonstrated

The gate-capability census was read three times. **The reading sequence is
6/1 → 5/2 → 4/3.**

| reading | result | why it was wrong |
| --- | --- | --- |
| first | GATED 6 / UNGATED 1 | three gates shared the generic signature `^FAIL`, and one file — `results/_a29_sweep_discrimination.txt`, an ASCII-sweep demo and nothing else — matched all three. A shared signature is not an identity. |
| second | GATED 5 / UNGATED 2 | with per-gate signatures installed, the run after staging read **this series' own prose describing the citation sweep's A-40 red** as evidence of that red. |
| third | GATED 4 / UNGATED 3 | prose path closed: eligible captures restricted to `.txt`/`.log`/`.json`. |

**Every correction moved in the closing direction.** Each error made C2 look
closer to satisfied than it was. **Three readings trending one way is not
convergence** — it is three errors of the same sign, and nothing establishes
that the third is the last one. The count is therefore recorded as
**provisional**, and re-read once with the prose path closed before WP2's
first commit; the result is in §9.

### The second failure, declared in its own right

The second reading is not a milder version of the first. **An instrument that
accepts a document describing a red as evidence of a red has no defined
population.** It was not searching captures; it was searching text, and the
register is text. The distinction it lacked is the one that makes the whole
census meaningful:

> **A capture is a file produced by *running* an instrument. A document
> *written about* one is not a capture, however accurately it describes the
> run.**

Recorded separately because the first failure was a loose regex — a defect in
a pattern — and this one was a category error about what evidence *is*, which
no amount of pattern-sharpening would have caught.

---

## 3. C1 closed by a method with a known blind spot

**C1 still closes. OP-20 is genuinely closed** — A-33 declares "Closes OP-20"
and it is closed on the merits.

But: **OP-20 was closed in prose at A-33 and its §11 row was never marked**,
while OP-17's and OP-22's rows are. So a reader enumerating open items from the
table — **which is what the table is for** — would have counted OP-20 as open
and routed to WP2, and would have built a different C1 list. **C1's
"exhaustive" three-item list omitted it for exactly that reason.**

**The closure was reached before the table correction.** It is not invalidated
by it, and the sequence is recorded rather than smoothed: a list declared
exhaustive was assembled by a method that could not see one of its candidates.

---

## 4. Prose-and-table divergence — enumerated, not accumulated (OP-29)

**Three instances in one series:**

1. **D-43 was cited as authority before it existed.** It occurred nowhere in
   the repository — not the register, not `AGENTS.md`, not the commit messages
   — while WP2's ordering and the WP1 reopening rested on it (A-36).
2. **OP-21 and OP-22 were opened in prose and never entered the §11 table.** A
   reader enumerating open items from the table would have found neither
   (A-36).
3. **OP-20 was closed in prose and its row never marked** (A-43/A-44, §3
   above).

**Three is the OP-22 precedent's threshold for enumerating a class rather than
accumulating a third instance.** The class is **OP-29**, and its remedy is
mechanical rather than attentional: the destination-cell enumeration that found
instance 3 is now **check 4 of `results/_a25_content_checks.py`**, a standing
per-commit gate, not something done when somebody happens to look.

Four sub-checks, one per failure mode, each demonstrated capable of firing
(`results/_a44_content_check4_discrimination.log`, `CONTENT_CHECKS_EXIT:1` on
all four):

- **4a** every item an amendment says it OPENS has a §11 row
- **4b** every item an amendment says it CLOSES has `CLOSED` in that row
- **4c** every `D-n` cited anywhere has a §8 row
- **4d** (D-44) every work package named in a destination cell has an entry

---

## 5. The work-package definition gap — partly closed

**Was:** WP3, WP4 and WP6–WP9 had no register text, and WP5 was the declared
destination of six open items with no entry of its own. Work packages were
being routed to by number faster than they were defined — the OP-23 defect at
plan scale.

**Now (A-43):** the Phase 3f outline is recorded (**it originated outside the
repository and says so**); WP5's entry is **substantive**; WP9's is a
**placeholder with one substantive element** (G2's deliverable and gate are
declared); WP3, WP4, WP6, WP7, WP8 are **placeholders**; **D-44** forbids
routing to a package with no entry.

**What remains:** **scope and sequencing for all six placeholders**, plus the
open-question list A-43 records per entry — where `PartitionDecoderConfig` and
`InferenceRunConfig` sit in D-43's construction DAG; what WP4 does to objects
that already exist; whether the event-side membership asymmetry is WP4's or
WP6's; whether OP-27's config-external half is discharged in WP6 or
per-package; what "identity" means for a decoder; whether WP8 is BP or API
(the outline itself leaves this open); what "hard-fail incompatibility" fails
on; and whether WP10's `args` removal invalidates WP9's save format.

**Restructuring is still not warranted** — for the reasons in §1, **plus one
specific to this gap: it is closable by writing entries.** A gap whose remedy
is "write the document" is not evidence that the plan is the wrong shape.

---

## 6. WP5 cannot open — a hard blocker, with its discharge condition

**Six open items — OP-19, OP-21, OP-23, OP-24, OP-26, OP-27 — route to a work
package whose entry precondition is unmet.**

**OP-24's polygon-mode pin was SCHEDULED, not deferred.** A-36 decided it in
those words, and recorded the decision then specifically so it would not arrive
at WP5's opening as scheduling TBD. **It has not been built.**

**Measured, and stronger than the coverage limit previously carried:**
`refactor-patches/pin_check_v2.py` emits **four** configurations — matching the
four top-level keys of `baselines-2026-07/pins.json` — and contains **zero
occurrences of `polygon`, `mass_table`, `excitation_support` or `min_sigma`**.

> **The polygon surfaces are not merely unexercised. They are unreachable from
> the harness as written.** `PIN_DIFFS 0 MATCH` is not weak evidence about
> `ExcitationSupport`, `PolygonMassTable` and cutoff provenance — it is no
> evidence at all, and those are exactly the surfaces WP5 changes.

**Discharge condition: land the fifth pinned configuration in polygon mode.**
Not inside WP5 — A-36's reason is that an instrument built by the package it is
meant to gate is not a gate.

**One consequence to expect rather than discover.** The fifth pin adds a
configuration key the baseline does not carry, so the first candidate holding
it cannot normalise to the baseline's canonical hash.
**`pin_corpus_identity.py` will go red and A-38's retroactive rescue of the 24
historical `PIN_DIFFS 0` claims will expire.** That is the *expected* trigger,
named in the gate's own failure message. The response is to re-baseline and
record the expiry at that commit — **not** to relax the check.

---

## 7. Capture-then-fix — standing practice, effective now (D-45)

**A gate that goes red has its capture preserved before the fix is applied.**

The evidence for needing this is in the series' own artifacts. **The citation
sweep went red twice — A-40 (`FAIL 5 unreachable citation(s)`) and A-41
(`FAIL 2`) — and both captures were overwritten by the passing re-run before
the commit.** The gate demonstrated its own capability, twice, and the proof
was destroyed both times. It is why an instrument *known* to be capable of
failing counts as UNGATED.

**This practice was implicitly in force all series**, in the sense that nobody
decided against it — the workflow simply ran the gate, saw red, fixed the
cause, re-ran, and committed the passing output over the failing one. **It is a
plausible partial cause of how hard demonstrated sensitivity was to establish**
across A-38 through A-42: the reds were happening and were not being kept, so
each demonstration had to be constructed from scratch rather than found.

Stated as a cause and not proven as one — the overwritten captures cannot now
be counted.

---

## 8. The two declared gaps

**G-A.** Three per-commit gates are UNGATED — content/decision-monotonicity
checks, `\hypertarget`, citation + label. **None of them guards `bstpp/`**; an
ungated `\hypertarget` check can let a broken anchor through, not a numerical
defect. Discharged as a **by-product of any WP2 commit touching the register**,
not as a prerequisite. *(A-44 itself discharges one — see §9.)*

**G-B.** **`standardize_cov`'s bind-time relocation is out of WP2's scope until
OP-28 is answered.** A WP2 commit touching the standardization block in
`attach_covariate_partitions` violates this gap and is **refused**. The reason
is not caution: if OP-28 resolves as "not deliberate", the remedy is likely to
move the standardization away from the clipped-support site, and that is *the
same edit* — a relocation and a semantic correction are indistinguishable in
the diff and are not the same commit.

---

## 9. State at registration

**C2 re-read, prose path closed, A-43's documents committed
(`results/_a44_c2_reread.txt`): GATED 5 / UNGATED 2 / RED 0, sum 7.**

**4/3 held**, and then one entry moved. **This fourth reading also closes
further, and that is not a fourth instrument error.** The discriminator is
where the cause lives:

| reading | moved because | cause lives in |
| --- | --- | --- |
| 6/1 → 5/2 | signatures sharpened per gate | the **instrument** |
| 5/2 → 4/3 | prose excluded from the capture population | the **instrument** |
| 4/3 → 5/2 | a `.log` produced by *running* the gate to red was added | the **tree** |

The promoted entry is matched by
`results/_a44_content_check4_discrimination.log`, and the census names the file
it matched, so the distinction is checkable rather than asserted.

**One gate moved during this commit, with a recorded cause.** Check 4 is a new
enforcement, and D-41 clause 1 forbids landing an enforcement without a
demonstrated red — so `results/_a44_content_check4_discrimination.log` is the
content-checks gate's first capability capture, and it is *preserved* under
§7's practice rather than overwritten. **Content checks: UNGATED → GATED.**
The finding's 4/3 is the state **at cap**; the two still ungated are
`\hypertarget` and citation + label.

---

## Not in scope, and staying that way

- **OP-28's resolution** — a question about modelling intent, not repo work.
  **Escalated.** No number of rounds resolves it, because the answer is not in
  the repository.
- **The `main.py:338` comment** naming `T_INTERNAL` — WP10's first-commit
  one-liner.
- **The `cox_hawkes_shared` blind spot** — declared UNMEASURABLE, not
  not-required. C4 does not cover that surface: the vendored copy reads 44
  `args` keys, 33 of them ours and 17 of the kind WP2 is named for, including
  `args['priors']`, and its base module is in no reachable tree.
