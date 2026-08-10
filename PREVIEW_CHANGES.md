# BSTPP pre-S3 analysis preview

This branch is an early-testing snapshot for the Philadelphia
`cbg-park-seasonal` analysis. It starts from BSTPP commit `c04c989` (package
and test code identical to the authoritative `c4e069f` evidence baseline).
It is **not** the official S3 analysis-resumption release; results must be
regenerated when S3 is reached.

## What the refactor adds

- Explicit validation for event times, domain geometry, CRS, covariate
  coverage, excitation support, trigger capabilities, and cutoff inputs.
- Prepared data/domain/partition objects and shared likelihood/simulator
  atoms behind the existing public model interface.
- Real-unit spatial-trigger parameters and computational cutoffs.
- Explicit rectangle versus polygon excitation support.
- Transactional cutoff mutation and recorded cutoff provenance.
- Reproducible simulation through an explicit NumPy generator.

## Preview analysis decisions

- Preserve the historical covariate estimator: unweighted population moments
  over all 1,338 supplied CBG rows, calculated before park clipping. The
  transformed covariates are passed with `standardize_cov=None`.
- Use one fixed calendar interval: 2021-01-01 inclusive through 2025-01-01
  exclusive. This is 1,461 days because 2024 is a leap year.
- Use `cox_background=True` and explicit bounding-rectangle excitation support,
  matching the historical analysis.
- Use a 1,000 m per-axis square computational spatial cutoff for every park.
- Treat the real-unit spatial prior as sensitivity analysis. Candidate prior
  scales are 50, 100, and 250 m; the provisional default is 250 m.
- Run input contracts in `report` mode for the preview. Tacony currently has a
  3.4402% covariate-coverage gap; the warning and exported geometry are retained
  while the old zero-valued uncovered-region behavior continues.

## Known limitations

- Saved-result schema v1 and decoder identity contracts do not land until S3.
- `model.args` remains compatibility state and is scheduled for removal.
- The 1,000 m cutoff produces about 1.69 million Tacony excitation pairs,
  roughly five times the old notebook count, so first JAX compilation is slower.
- Real-data Tacony construction was verified for all three provisional prior
  scales (50, 100, and 250 m); a five-step SVI plumbing run at 250 m produced
  finite losses. This is an execution check, not an inferential result.
- The copied notebook prioritizes model construction and fitting. Full posterior
  point-process simulation is intentionally the final, optional migration step.

All migration edits in `cbg-park-seasonal-refactor-preview.ipynb` are marked
`PRE-S3 MIGRATION EDIT`; the historical notebook remains unchanged.
