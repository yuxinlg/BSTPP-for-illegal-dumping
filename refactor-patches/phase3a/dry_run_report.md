# Section-14 dry run — 3a data contracts on the project data (report-only)

Run: 2026-07-20T05:08:39+00:00 — validators only; no fits, no data modification, no enforcement.

## Inputs (read-only, analysis tree)

- events: `C:\Users\Terhi\Box\terhi-illegal-dumping-Jan15-2026\output\illegal_dumping_full.geojson` — 118,152,474 bytes, mtime 2025-08-28, sha256 `02450e1e267f22c1…`
- park boxes (domains): `C:\Users\Terhi\Box\terhi-illegal-dumping-Jan15-2026\output\all_boxes_gdf.geojson` — 1,557 bytes, mtime 2025-07-23, sha256 `991a807097fba3a1…`
- covariates: `C:\Users\Terhi\Box\terhi-illegal-dumping-Jan15-2026\output\cov_cbg.geojson` — 4,400,088 bytes, mtime 2025-11-11, sha256 `45b0ec1aad33d289…`
- events CRS as stored: `EPSG:26918`; boxes/covariates reprojected to EPSG:26918 by the pipeline (metric, meters — satisfies the real-unit contract).

## Pipeline funnel (mirrored verbatim)

- raw events in file: 127,598
- after `start_time <= 2025-01-01`: 109,815
- after `within` park boxes: 22,719
- after `within` covariate layer: 22,713 (lost to covariate join: 6)

Note: the pipeline's own `within` prefilter excludes exact box-boundary events BEFORE BSTPP sees them, so the validator's out-of-domain counts below measure only what survives that filter; `within` vs the contract's boundary-inclusive `covers` (D-4) can differ only for boundary points, which `within` removes.

## Findings per scope

| Scope | n events | unparseable start_time | Violations | Diagnostics |
|---|---|---|---|---|
| tacony_box | 3,888 | 0 | event_time_out_of_range: 2 | none |
| cobbscreek_box | 8,012 | 0 | none | none |
| mifflin_box | 1,319 | 0 | event_time_out_of_range: 1 | none |
| fairmount_box | 9,494 | 0 | event_time_out_of_range: 2 | none |
| pooled(4 boxes) | 22,713 | 0 | event_time_out_of_range: 5 | none |

Row-level detail: `dry_run_findings.csv` (scope, check, kind, positional row in the scope's locs_s frame, message).

## Analysis of the findings (appended after the scripted run)

**The only violation class is `event_time_out_of_range`, and all five hits
are the 2024 leap day.** The pipeline sets `total_days = 365*4 = 1460`, but
the data span Jan 1 2021 → Dec 31 2024 contains the 2024-02-29 leap day, so
the last calendar day of data runs from T = 1460.0 to 1461.0. The five
offending events are all on 2024-12-31 (T between 1460.30 and 1460.64):

| Scope | start_time | T (days) |
|---|---|---|
| fairmount_box | 2024-12-31 07:17 | 1460.303 |
| fairmount_box | 2024-12-31 09:18 | 1460.388 |
| tacony_box | 2024-12-31 10:49 | 1460.451 |
| tacony_box | 2024-12-31 15:22 | 1460.640 |
| mifflin_box | 2024-12-31 10:13 | 1460.426 |

Current silent behavior for these events: `searchsorted` clamps them into
the last temporal cell, and the excitation compensator's
`min(T̃ − t̃, w)` goes **negative** for them — the exact silent-integral-
corruption mode the held-out path already guards against
(`bstpp/main.py` horizon check), unguarded in the constructor until now.

**Recommended upstream fix (not a contract adjustment):** set
`total_days = 1461` in the pipeline (or compute it from the calendar span).
The contract is correct to reject these; nothing suggests loosening it.

Everything else is clean: zero unparseable `start_time`, zero NaN/nonfinite
coordinates, zero out-of-domain events (after the pipeline's own `within`
prefilter), zero grid-line or covariate-membership ties, zero nonfinite
values in the 23 covariate columns, no CRS mismatches, all geometries valid.

## Reviewer decision requested

Per the staged 3a plan, the flip of the `data_contracts` default from `'report'` to `'reject'` is gated on review of this report. Any contract adjustment motivated by it must be a recorded decision, not a silent relaxation.