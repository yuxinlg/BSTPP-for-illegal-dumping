# WP8 — input metadata: entry

**Class: BP *or* API — the outline does not decide, and neither does this
entry.** (`phase3f_work_package_outline.md`.)
**Status: PLACEHOLDER.** Seams named, scope not yet specified.

## Seams

`spatial_cov_crs` · `data_contracts` mode — in `bstpp/data_contracts.py` and
the corresponding `Point_Process_Model.__init__` arguments.

## Open items routed here

**None.** No §11 row names WP8.

## Constraints and known limits already in force

- **`spatial_cov_crs` is a public argument with a deliberate non-inference
  rule.** GeoDataFrame covariates are self-describing; CSV/plain-DataFrame
  covariates paired with a CRS-bearing domain **require** public
  `spatial_cov_crs` (parsed by `CRS.from_user_input`), assigned before
  `validate_covariates`, and **never inferred by copying the domain CRS**. The
  non-inference is the contract, not an implementation detail.
- **`data_contracts` mode already flipped once, on sign-off.** The default went
  `report` → `reject`; `report` warns and leaves legacy behaviour bit-unchanged
  and is described in the register as the section-14 dry-run instrument.
- **`report` mode is the migration surface**, which is why it carries the
  weaker guarantee: the OP-25 row records that *"`report` mode is the migration
  instrument, so the surface most likely to meet unfamiliar data is the one
  still able to fail while reporting."* Measured in passing at A-40: on the real
  Philadelphia layer, `reject` refuses the fit over a 3.44% covariate gap and
  five in-gap events, so `report` is not a corner case — it is the mode real
  data reaches first.

## Adjacencies — subject matter, NOT routings

- **OP-25** — `data_contracts.enforce`'s `warnings.warn` renders the same
  `summary()` the raise does, unescaped — is squarely on this seam. It is
  **routed to WP2**, "where D-40's scope is next opened", and it stays there:
  the item is about the *scope of D-40's ASCII corollary*, not about input
  metadata. Recorded because a reader of WP8 will otherwise find the defect and
  assume it is unrouted.

## Questions this entry leaves open

1. **Whether this package is BP or API.** The outline says "BP or API" and
   leaves it. It matters: moving `spatial_cov_crs` or the `data_contracts` mode
   into a config object changes the constructor's accepted-input surface, which
   A-27 and A-33 both treat as an API-class change rather than BP. **Named, not
   settled here.**
2. **Whether `data_contracts` mode is model configuration or run
   configuration.** It selects an enforcement policy, not a model; whether it
   belongs with WP2's `ModelConfig`, WP3's `InferenceRunConfig`, or stays a
   constructor argument is not stated anywhere.
3. **Scope and sequencing.** Not specified, not invented.
