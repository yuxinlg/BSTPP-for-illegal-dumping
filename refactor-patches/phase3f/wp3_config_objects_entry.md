# WP3 — `PartitionDecoderConfig`, `InferenceRunConfig`: entry

> **Superseded for current execution by A-54.** Current Phase 3f status, scope,
> dependencies, gates, corrective-cycle counts, and completion are governed only
> by `docs/phase3f_completion_manifest.yaml`. This file remains historical or
> supporting evidence and must not be updated as a parallel task tracker.

**Class: BP** (`phase3f_work_package_outline.md`).
**Status: PLACEHOLDER.** Seams named, scope not yet specified.

## Seams

`PartitionDecoderConfig` · `InferenceRunConfig` — the fourth and fifth of
A-23's five frozen config objects (`ModelConfig`, `PriorConfig`,
`PartitionDecoderConfig`, `NumericalConfig`, `InferenceRunConfig`). Neither
exists in the tree.

## Open items routed here

**None.** No §11 row names WP3, measured by reading every destination cell.

## Constraints already in force

These are not scope. They are rules that will apply whenever WP3 is written,
recorded so they are not re-derived or re-litigated.

- **D-41's relabeling corollary was recorded pre-emptively for this package.**
  The register says, in D-41's discussion: *"Recorded here so it is not
  re-litigated at WP3, when `PartitionDecoderConfig` adds its own invariants."*
  Relabeling in place is permitted where no assertion changes and the head note
  carries the complete old→new mapping; anything changing an assertion uses
  `\supsd` per occurrence.
- **The `CI-n` sequence is one sequence across all five config objects,
  extended as each lands its own, never restarted per object** (A-30). WP3's
  invariants continue from whatever is free when it opens. **Next free is
  `CI-10`.**
- **D-35**: every freeze ships with its enforcement in the same commit. A
  config object that does not validate its own invariants at construction is
  not done.
- **A-23**: frozen dataclasses with validation in `__post_init__` and a single
  factory per object. No Pydantic.
- **D-42 / CI-7 / CI-8**: the argument-type policy is one policy across all
  five factories, so WP3 inherits it rather than choosing.

## Questions this entry leaves open

1. **Where `PartitionDecoderConfig` and `InferenceRunConfig` sit in D-43's
   construction DAG.** D-43 states `ModelConfig` → `NumericalConfig` and
   `ModelConfig` → `PriorConfig`, and places `ModelData` outside the DAG. It
   places neither of WP3's objects. Whether either has a `ModelData` dependency
   — and therefore a bind-time resolving field under D-43 clause 1 — is not
   decided.
2. **What `PartitionDecoderConfig` owns**, given that WP4 owns the prepared
   partitions and WP7 owns the decoder contract. The name spans two other
   packages' seams and the boundary is not stated.
3. **Scope, sequencing, and every invariant.** Not specified, not invented.
