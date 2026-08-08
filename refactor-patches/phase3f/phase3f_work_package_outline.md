# Phase 3f work-package outline — the source the entries derive from

> **Superseded for current execution by A-54.** Current Phase 3f status, scope,
> dependencies, gates, corrective-cycle counts, and completion are governed only
> by `docs/phase3f_completion_manifest.yaml`. This file remains historical or
> supporting evidence and must not be updated as a parallel task tracker.

**Origin: OUTSIDE THIS REPOSITORY.** This table was supplied by the project
owner in the round brief of 2026-08-06 (A-43). It had never been in the
repository, in the register, in `AGENTS.md`, or in any commit message. It is
recorded here verbatim so that the entries beside it cite something reachable
instead of something remembered — the OP-23 defect (a citation whose target has
no text) at plan scale.

**Every work package is class BP unless the table marks it otherwise.**

| WP | Seam | Class |
| --- | --- | --- |
| 1 | `NumericalConfig` | BP |
| 2 | `ModelConfig`, `PriorConfig` | BP |
| 3 | `PartitionDecoderConfig`, `InferenceRunConfig` | BP |
| 4 | Prepared domain and partitions (`ModelData`, `PreparedDomain`, `PreparedPartitions`) | BP |
| 5 | `ExcitationSupport`, `PolygonMassTable`, cutoff provenance | BP |
| 6 | Public mutators + G1 membership single-source | BP |
| 7 | Decoder contract, identity, gain, provenance | BP |
| 8 | Input metadata (`spatial_cov_crs`, `data_contracts` mode) | BP or API |
| 9 | Results I/O — G2 API, declared | API |
| 10 | `args` removal (OP-8) | BP |

## What this table does and does not settle

**It settles the seam sets**, which is what the entries were blocked on, and
it settles a class per package except at WP8, where "BP or API" is itself an
open question and is left named rather than resolved.

**It does not settle scope, sequencing, or any decision.** No entry beside it
invents those. Where an entry would have to settle a question to be written,
the question is named and left open, and every such question is listed in the
A-43 amendment.

**Two corroborations worth recording**, because they are the only points where
this table meets text that was already in the repository:

- **WP9's class.** The table says API; A-22's declared-changes list already
  said of **G2** — `save_rslts` implementing the A-21 save/load contract with
  provenance and hard-fail incompatibility — "Declared **API** change; Lane B
  strict xfail flips to a passing assertion in the same commit." Independent
  agreement.
- **WP3's seam.** The table says `PartitionDecoderConfig`, `InferenceRunConfig`;
  the register already said (D-41's relabeling corollary) "Recorded here so it
  is not re-litigated at **WP3**, when `PartitionDecoderConfig` adds its own
  invariants." Independent agreement on half the seam.

And one correction the table makes to an inference that was in flight:
**WP4 is not `InferenceRunConfig`.** Reading A-23's five config objects against
WP1/WP2/WP3, `InferenceRunConfig` looked like WP4's natural seam. It is WP3's.
WP4 is the prepared-data seam. That guess was never committed; it is recorded
because it is exactly what an entry written without this table would have said.

## ⚠ Numbering collision — read before citing a work package

**`Wn` and `WPn` are two different sequences and they collide.**

`refactor-patches/pre-3f-stabilization/phase3e_closeout_and_3f_readiness.tex`
§"Implementation handoff — work packages" carries a **`W0`–`W10`** table. That
is the **pre-3f stabilization** programme, which is **complete**, and its
numbers mean entirely different things:

| n | pre-3f `Wn` (completed) | Phase 3f `WPn` (this table) |
| --- | --- | --- |
| 3 | class-level tests (distributions, setters, budget, membership) | `PartitionDecoderConfig`, `InferenceRunConfig` |
| 4 | compatibility and state-transition matrix | prepared domain and partitions |
| 5 | `docs/register_test_traceability.md` | `ExcitationSupport`, `PolygonMassTable`, cutoff provenance |
| 6 | `docs/audit_coverage_map.md` + audit the unaudited seam modules | public mutators + G1 membership single-source |
| 9 | ledger of non-blocking findings + register amendment | results I/O, G2 |
| 10 | final bounded audit + full R4 gate run; 3f opens | `args` removal (OP-8) |

This is the same hazard `AGENTS.md` already flags at length for `I1`–`I12`
(model identities) against `CI-1`–`CI-9` (config invariants). **Cite `WP5`, not
`W5`**, and read the prefix before resolving a reference.
