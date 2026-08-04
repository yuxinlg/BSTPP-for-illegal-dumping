# Lane B configuration matrix

**Superseded for generation details by repository-root `docs/config_matrix.md` (Commit D).**  
**Executable module:** `tests/test_lane_b_config_matrix.py`  
**Covering array:** `covering_array_rows.json` (17 rows, 9-axis pairwise 1.000)

## Iteration-2 baseline (retained)

**Frozen tip then:** `143d2190c2fe79d651ecc1d09718cb776876bd96`  
**Generation method then:** hand-curated — **not** a covering-array generator.

## Axis level sets (from code)

| Axis | Levels |
|---|---|
| model_family | hawkes, cox_hawkes, lgcp |
| support | rectangle, polygon |
| temporal_trigger | Temporal_Exponential, Temporal_Power_Law, custom |
| spatial_trigger | Spatial_Symmetric_Gaussian, custom |
| cutoff_input | tolerance, physical, omitted, explicit_None_via_set_window |
| entry_path | constructor, set_window |

(Additional non-matrix axes exercised elsewhere: `standardize_cov`, σ bounds, builder numerics.)

## Achieved pairwise coverage

Computed by `probe_iter2_lane_b_coverage.py` over the 27 assigned points vs the Cartesian product of the six axes above:

**Overall pairwise coverage fraction: 0.533 (53.3%).**

Notable weak pairs: `temporal_trigger × spatial_trigger` 1/6; `model_family × temporal_trigger` 3/9; `temporal_trigger × cutoff_input` 4/12.

**Coverage claim (corrected):** forced rows plus **partial** pairwise — not “pairwise matrix.”

## Forced-row rationale

- Explicit rejects that encode D-23 / capability gates / LGCP no-cutoff / standardize bool / polygon hard-require table.
- Explicit success for family×support including Cox–Hawkes×polygon.
- Constructor/setter physical-window equivalence; sentinel omit vs `None`; whole-state rollback; shipped-default panel budget.
- G2 `save_rslts` provenance asserted under strict xfail.

## Attack residuals (iteration 2)

See `probe_iter2_lane_b_coverage.log`. No combination constructed when it should reject, or failed with an unnamed bare `Exception` on supported paths. LGCP unknown kwargs surface as `Exception`/`ValueError` via the prior-kwargs path (`Unknown argument …`). Tolerance-ctor vs physical-setter yields equal windows/coords but different provenance `selection` (expected semantic difference).
