# Running `cbg-park-seasonal` with the BSTPP pre-S3 preview

## Environment and checkout

Use the repository's pinned `illegal-dumping` environment. From the preview
checkout, install the local package. Configure the notebook frontend to use
that environment as its kernel:

```powershell
conda activate illegal-dumping
python -m pip install -e .
```

Before launching the notebook, set the separate code and data roots if needed:

```powershell
$env:BSTPP_PREVIEW_ROOT = 'C:\path\to\BSTPP-cbg-preview'
$env:BSTPP_ANALYSIS_ROOT = 'C:\path\to\analysis-checkout'
```

The first notebook cell changes its working directory to the analysis root so
historical relative data paths continue to work, then prints `bstpp.__file__`.
Stop if that path does not point into the preview checkout.

## Required analysis files

The notebook expects these analysis-owned files relative to its working
directory:

- `output/illegal_dumping_full.geojson`
- `output/all_parks_gdf.geojson`
- `output/all_boxes_gdf.geojson`
- `output/cov_cbg.geojson`
- the mapping inputs referenced by later plotting cells under `data/` and
  `output/`

The package repository does not publish those analysis data files. Point
`BSTPP_ANALYSIS_ROOT` at the coworker's local analysis checkout; do not commit
confidential or bulky inputs to the package branch.

## Required input changes

### Events and time horizon

- Parse `start_time` as datetime.
- Keep events from 2021-01-01 inclusive through 2025-01-01 exclusive.
- Use one shared origin, 2021-01-01, for every park.
- Set `T=1461`, not `365 * 4`; the interval contains the 2024 leap day.
- Event `X`, `Y`, and `T` values must be finite and `T` must lie in `[0, T]`.

### CRS and geometry

- Reproject events, park domains, and CBG covariates to EPSG:26918.
- Spatial trigger distances are metres and `sigmax_2` is square metres.
- The preview explicitly uses `excitation_support="rectangle"`, preserving the
  old bounding-rectangle compensator after the park polygons are reprojected.

### Legacy covariate standardization

The old package standardized every supplied CBG row before geographic
clipping. To reproduce that estimator, the preview calculates, for each of the
23 named columns:

```python
values = cov_gdf[column_names].to_numpy(dtype=float)
mean = values.mean(axis=0)
scale = values.var(axis=0) ** 0.5  # population scale, ddof=0
cov_gdf_model[column_names] = (values - mean) / scale
```

Pass `cov_gdf_model` with `standardize_cov=None`. Do not use
`standardize_cov="domain_area"` for these cross-park preview fits: it would use
different park-specific reference distributions and change coefficient meaning.
Batch `run_config.json` records the row count, column order, means, and scales.

### Spatial cutoff and prior sensitivity

Use `spatial_window=1000.0`. This is a per-axis square cutoff: a pair is kept
when both `abs(dx)` and `abs(dy)` are at most 1,000 m; diagonal separation may
reach about 1,414 m.

The historical Tacony fit attributed only about 0.24% of total intensity to
self-excitation, so it does not identify a reliable physical spatial scale.
Run the analysis with spatial prior scale choices 50, 100, and 250 m before
interpreting `sigmax_2`; 250 m is only the provisional default. The code places
the prior on the variance:

```python
sigmax_2 = dist.HalfNormal(SPATIAL_SIGMA_PRIOR_SCALE_M ** 2)
```

Record the chosen value with every result.

### Covariate coverage warning

The current Tacony inputs leave 3.4402% of the model domain uncovered by the
CBG covariate layer. `data_contracts="report"` warns and exports the gap while
continuing the historical zero-valued uncovered-region behavior. This is a
provisional migration choice, not a statement that the gap is scientifically
acceptable.

## Quick verification

Before a long fit, construct the real-data model:

```powershell
python scripts\preview_analysis_smoke.py --analysis-root C:\path\to\analysis --park tacony_box
```

Add `--svi-steps 5 --num-samples 10` for a compilation/plumbing check. A
successful construction reports the preview `bstpp_file`, 1,338 standardization
rows, the selected spatial settings, and the number of eligible event pairs.

Then open `cbg-park-seasonal-refactor-preview.ipynb`. Run data preparation,
one short single-park fit, and diagnostic cells before starting the full batch.

## Reporting problems

Include the preview commit/tag, `bstpp.__file__`, park name, random seed,
environment package versions, standardization row count, spatial cutoff and
prior scale, event/pair counts, and the complete exception or contract warning.
