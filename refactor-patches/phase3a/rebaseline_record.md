# Phase 3a rebaseline record — data contracts and unique membership

Governing document: `docs/phase3_baseline_and_decisions.tex` (§7.3, §10.a,
§12.2 row 3a; decisions D-3, D-4, D-5, D-22). Baseline: frozen tip `476c2a0`.
Machine: analysis machine of §4.2 (pins are MACHINE-LOCAL).

## Commits

| Commit | Class | Content |
|---|---|---|
| `3e69836` | IV | `bstpp/data_contracts.py` validation layer; constructor arg `data_contracts='report'|'reject'` (default **report**); held-out `dropna` made loud; 26 adversarial tests |
| `48f7d10` | MR | D-22 deterministic unique membership: per-point **max-id** dedup of the field-grid and covariate sjoins (training and held-out paths); 10 membership tests |

## IV changes (invalid inputs; valid inputs bit-unchanged)

Contracts enforced under `data_contracts='reject'`, surfaced as one
`UserWarning` + `model.data_contract_report` under the current default
`'report'`:

- non-numeric / NaN / inf event `X`/`Y`/`T` (constructor and held-out path);
- event time outside `[0, T]` days;
- event outside the model domain A — **polygon boundary is INSIDE (D-4)**,
  implemented as shapely `covers`;
- invalid / empty domain or covariate geometry; malformed rectangle domains;
- nonfinite covariate values in used `cov_names`;
- CRS mismatch between a CRS-bearing domain and CRS-bearing covariates;
- events covered by no covariate polygon.

Diagnostics (never rejected): events exactly on the boundary of A (valid,
D-4); events exactly on internal 25×25 grid lines per axis (exact-float
against the grid's own edge arithmetic); events covered by >1 covariate
polygon; held-out rows the legacy path drops. These are the §14 dry-run
inventory fields.

**Report-only staging (reviewer decision, 2026-07-19):** the default stays
`'report'` — every legacy numerical path and legacy failure mode is
bit-preserved — until the §14 report-only dry run against the actual
Philadelphia package-boundary inputs is committed and reviewed. The flip of
the default to `'reject'` is a separate, gated commit. Any contract
adjustment motivated by the dry run will be a recorded decision.

## MR change (otherwise-valid boundary events; intentional rebaseline)

**Old:** a grid-line event double-joined and crashed with the misleading
"Computational grid does not encompass all data points!"; a covariate-edge
event crashed at training ("Spatial covariates are not defined for all data
points!") and at held-out scoring silently emitted two `cov_ind` rows with
no length check (misaligning all later events' covariates).

**New (D-22):** left-closed/right-open per axis, `[e_k, e_{k+1})`, outermost
right/top edges closed; temporal axis was already compliant
(`searchsorted(side='right')−1`, `t = T̃` closed into the last cell) and the
seasonal circle seamless — now pinned as contract by tests. Spatial and
covariate ties resolve by per-point **max id** on the unchanged sjoin:
row-major grid ids make max-id ≡ left-closed on x-edges (+1), y-edges (+25),
and corners; for arbitrary covariate polygon layers max-`cov_ind` is a
documented deterministic convention, not a geometric statement.

**No previously-working input changes numerically:** every affected event
crashed (or, held-out covariate ties, silently corrupted the covariate
alignment) before this commit; unique-join (interior) events pass through
bit-identically because the sjoin itself is unchanged and only tied joins
are deduplicated.

## Baseline-document corrections

- §5.1/§6.1 anchor **"NaN rows silently dropped at ingestion
  (`main.py:603`)"** is mis-anchored: line 603's `dropna` is in
  `log_expected_likelihood` (held-out scoring). The **constructor** never
  dropped NaN rows — a NaN coordinate produced an empty geometry that missed
  the membership sjoin and crashed with the misleading encompass message.
  Both entry points are now covered (warn-and-drop → reject for the held-out
  path; contract violation for the constructor path).
- §5.1 "grid-line events double-join … crash" — retired by the MR commit.
- §5.1 "events outside polygon A silently accepted" — surfaced in report
  mode; rejected under `'reject'`; full retirement lands with the default
  flip.

## Gate evidence (§12.2 row 3a)

| Gate | Result |
|---|---|
| Det. trace equiv. + rect. pins (interior events) | all four `pins.json` configs **bit-identical** after each commit (pin_check_v2 vs `refactor-patches/baselines-2026-07/pins.json`) |
| Unit/property tests (adversaries) | 26 (IV) + 10 (MR) new tests; RED verified against pre-change code and stated in each commit body |
| Full suite | 94/94 after IV; **104/104** after MR |
| Boundary/grid membership (polygon-geometry gate) | x-edge, y-edge, corner, outermost-edge closure, min-edge, hole/multipolygon containment, D-4 boundary-inside, covariate shared-edge, held-out tie |
| SBC escalation (C) | not triggered: deterministic equivalence held on the unchanged regime; no model-specific path changed intentionally |
| Provenance review | `data_contracts` mode stored on the model (`_data_contracts_mode`, `data_contract_report`); full §16 provenance object is 3f scope |
| Rebaseline record | this file |

## Outstanding for 3a completion

1. §14 report-only dry run on the real package-boundary inputs (events,
   domain polygons, covariates, X/Y/T-mapping script) — committed report,
   then reviewer sign-off.
2. Default flip `report → reject` (separate commit, gated on 1).
