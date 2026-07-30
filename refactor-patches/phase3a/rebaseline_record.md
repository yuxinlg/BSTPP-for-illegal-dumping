# Phase 3a rebaseline record — data contracts and unique membership

Governing document: `phase3_record.tex` (living; supersedes historical
`docs/phase3_baseline_and_decisions.tex`; §7.3, §10.a,
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

**Report-only staging (reviewer decision, 2026-07-19) — completed:** the
default stayed `'report'` until the §14 report-only dry run against the
actual Philadelphia package-boundary inputs was committed (`ff1aef5`) and
reviewed. **Reviewer sign-off received 2026-07-20 and the default flipped to
`'reject'`** (commit recorded below). The dry run's sole finding — five
events on 2024-12-31 with T ∈ (1460.30, 1460.64], caused by the pipeline's
`total_days = 365*4` ignoring the 2024 leap day — is an upstream data/config
defect to fix in the pipeline (`total_days = 1461`), not a contract
adjustment. `data_contracts='report'` remains available as the dry-run
instrument.

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

## 3a completion status

1. §14 report-only dry run — **done** (`ff1aef5`,
   `refactor-patches/phase3a/dry_run_report.md`); reviewer sign-off
   2026-07-20.
2. Default flip `report → reject` — **done** (commit following this record's
   update). 3a is complete pending the upstream pipeline fix of
   `total_days` (1460 → 1461), which lives in the analysis tree, not this
   package.

## Post-hoc leftovers and tabular CRS (audit follow-up, 2026-07-24)

These commits are **later audit discoveries / repairs**, not a rewrite of the
original 3a feature commits above. They close contracts that the freeze tip
audit (§5.2) correctly recorded as missing at `476c2a0`.

### Domain / horizon leftovers (pre-A/B tip `05da465`)

| Commit | Class | Content |
|---|---|---|
| `5cea6c9` / `6e63efc` | IV | Finite positive `T_max` / horizon → `horizon_invalid` |
| `dde9b80` / `511ca5b` | IV | Polygonal, finite, positive-area domain geometry |
| `5d357ed` / `c22aa51` | IV | Missing covariate CRS when domain declares CRS (GeoDataFrame path) |

### Explicit `spatial_cov_crs` for tabular covariates (A; tip `adea3d3`)

| Commit | Class | Content |
|---|---|---|
| `5cd9355` | test (RED) | Missing / matching / mismatched tabular CRS; GDF + array characterization |
| `adea3d3` | fix (IV/API) | Public arg `spatial_cov_crs`; `CRS.from_user_input`; assign to constructed GeoDataFrame **before** `validate_covariates`; never infer by copying domain CRS; reject declared CRS ≠ domain CRS |

**Contract (enforced):**

- CSV / plain-`DataFrame` covariates with a CRS-bearing domain **require**
  `spatial_cov_crs`.
- GeoDataFrame covariates remain self-describing (no `spatial_cov_crs`
  required); behavior unchanged when CRS already matches.
- Array-domain tabular covariates remain numerically unchanged and do not
  require the argument.

**Not BP:** requiring `spatial_cov_crs` for the CRS-bearing-domain tabular path
is an intentional API / invalid-input contract; valid GeoDataFrame and
array-domain paths stay characterization-pinned.

Gate evidence for the `spatial_cov_crs` pair is recorded at tip `8580364` in
`refactor-patches/phase3_tip_verification_2026-07-24.md` (suite 277;
pins MATCH).
