# BSTPP pre-S3 preview

This branch is an early-testing snapshot of the **BSTPP package** (not the
Philadelphia analysis). It starts from BSTPP commit `c04c989` (package and
test code identical to the authoritative `c4e069f` evidence baseline). It
is **not** the official S3 analysis-resumption release.

Park-analysis files (`batch_park_fits.py`, the cbg-park-seasonal notebook,
and the analysis adapter tests) live in
[yuxinlg/Illegal-Dumping](https://github.com/yuxinlg/Illegal-Dumping)
on branch **`preview/parks`**.

## What the refactor adds

- Explicit validation for event times, domain geometry, CRS, covariate coverage, excitation support, trigger capabilities, and cutoff inputs.
- Prepared data/domain/partition objects and shared likelihood/simulator atoms behind the existing public model interface.
- Increased interpretability through real-unit spatial-trigger parameters and computational cutoffs.
- Improved accuracy and transparency through explicit rectangle versus polygon excitation support.
- Transactional cutoff mutation and recorded cutoff provenance.
- Reproducible simulation through an explicit NumPy generator.

## Known package limitations

- Saved-result schema v1 and decoder identity contracts do not land until S3.
- `model.args` remains compatibility state and is scheduled for removal.
