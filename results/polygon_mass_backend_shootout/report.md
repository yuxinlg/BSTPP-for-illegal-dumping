# OP-9 polygon-mass backend shootout — report

Experiment and decision record only. No production likelihood, simulator, API,
or support-mode code was modified. Nothing is committed.

Run: 2026-07-21, branch `refactor`, HEAD `1a9dd22306e83e709ef6dbb0ab61dbb680331897`
(working tree contains only pre-existing untracked user files plus this
experiment's new files; no tracked file modified).
Environment: Python 3.12.13, jax 0.4.23 (CPU, `CpuDevice(id=0)`),
numpy 1.26.4, shapely 2.1.2, scipy 1.11.4, geopandas 1.1.3.
Accuracy phase ran with `jax_enable_x64=True` (default is False —
see the float32 finding below). Inputs (SHA-256, first 16):
`output/illegal_dumping_full.geojson` 02450e1e267f22c1,
`data/planning_districts.geojson` fce0942aaa2df6fc,
`output/all_parks_gdf.geojson` 9507e210d7c48eb5.

## 1. Target mass and cutoff semantics (code-inspection answers)

The four questions the spec required answering before benchmarking:

1. **Is `spatial_window` fixed during a fit?** Yes. It is a user-supplied
   REAL-unit length set at construction (`main.py:1651`) or via
   `set_window` (`main.py:1530`), constant while `sigmax_2` is sampled.
   C_j (the event-centered square of half-width w_s) is therefore
   sigma-independent and known at preprocessing time.
2. **What does the §13 diagnostic compute?** The UNCUT mass
   ∫_A N(s−s_j; σ²I) ds — no cutoff square.
3. **Uncut, 8σ-cut, or production-cut?** Effectively uncut: its 8σ clip is a
   numerical evaluation device with neglected mass
   < 4·½·erfc(8/√2) ≈ 2.5e-15, far below every tolerance here. It is NOT
   the production cutoff.
4. **Must the backend integrate A or A ∩ C_j?** Both, matching the current
   contract: `spatial_window` is OPTIONAL, default None → uncut A;
   finite → A ∩ C_j. The PRIMARY benchmark target is uncut A (current
   default, and exactly what the oracle computes). The A ∩ C_j machinery is
   implemented and validated separately with a REPRESENTATIVE w_s = 400 m
   (explicitly not a recommended production value — tolerance-derived
   cutoffs belong to 3e; the fixed-cutoff semantics were not replaced with
   w_s = 3σ). The §13 oracle was preserved unchanged as the uncut
   reference; the smallest wrapper (`oracle_mass(..., ws=...)`: exact
   shapely intersection with the C_j square, then the same integral)
   provides the A ∩ C_j oracle.

**Unresolved semantic issue found in passing:** `batch_park_fits.py` still
passes `spatial_window=0.1` while fitting in EPSG:26918 metres. Under the
current real-unit contract that is a 10 cm box window — clearly a stale
value predating the real-unit semantics change (under the old
internal-unit semantics 0.1 meant 0.1 × box span). Not fixed here; needs
its own decision.

## 2. What was compared

Target quantity, events, polygons, sigma values, precision (float64), and
aggregation weights identical for both candidates.

* **Backend A — fixed-node JAX boundary quadrature.** Same
  divergence-theorem formulation as the oracle; ring orientations
  normalized (CCW exteriors, CW holes); Polygon/MultiPolygon/concavity/
  holes supported; boundary split into fixed panels (h metres) at
  preprocessing; fixed Gauss–Legendre nodes per panel; all geometry and
  Python control flow outside the jitted path; padded panels + masks with
  all-zero (finite, NaN-safe) padding; in cutoff mode per-event exact
  A ∩ C_j panels plus an analytic erf² fast path when C_j ⊂ A;
  dM/dlog σ via forward-mode AD. Configurations: (h, GL order) =
  (10, 8), (20, 8), (20, 16), (40, 16). Node counts/topology/shapes never
  change with σ.
* **Backend B — per-event σ lookup tables.** Oracle values at K ∈
  {24, 40, 64} log-uniform knots on [10, 500] m; C1 cubic Hermite in
  log σ, PCHIP slopes prepared offline; rows with interior extrema
  (PCHIP flattens there) detected and switched to centered-difference
  Hermite slopes — counts recorded per region (0–110 of ≤128 rows; the
  uncut M(σ) of near-boundary events is genuinely non-monotone, so this
  affected most district rows). Jitted, differentiable; extrapolation
  prohibited (range check raises before the call; the kernel returns NaN
  outside the knot span — verified on both sides of the end knots).
* **Addendum — quad-built tables** (motivated by the main-run failures,
  §5): same Hermite evaluator, but knot values from backend A (which had
  already passed every oracle gate independently — the disclosed
  dependency the spec anticipated) and knot slopes from backend A's AD.
  Gates still scored against the shapely oracle, never against quad.

Oracle: `scripts/polygon_mass_diagnostic.gaussian_polygon_mass`, float64,
convergence settings untouched (GL-20, step ≤ 0.5σ, 8σ clip); its
rectangle self-test passes unchanged. Derivative oracle: centered FD in
log σ at h = 1e-3 repeated at h/2; max Richardson uncertainty across all
regions/σ: **1.3e-7** — three orders below the derivative target, so the
derivative-oracle floor never masks backend error (reported separately as
`deriv_floor_max` per row).

## 3. Geometries, events, σ grid

Regions (resolved exactly as named in the files): Lower South,
North Delaware, Central (planning districts); East Fairmount Park,
Mifflin Square (parks; East Fairmount required `make_valid` repair, noted
at run time). Per region: up to 64 boundary-nearest events (always
including the most boundary-adjacent) + up to 64 seeded interior events
(seed 0), all events when fewer — 128/128/128/110/47 selected; coordinate
SHA-256 hashes in `run_config.json`.

σ: registered {10, 20, 50, 100, 200, 500} m; geometric midpoints; 20
seeded log-uniform values (seed 1234); knot-adjacent points (knots ×
(1 ± 1e-3)) for every K — 49 distinct values, so interpolation was never
tested only at its own knots. Derivatives compared at the 11
registered+midpoint values.

Synthetic correctness cases (rectangle-analytic, triangle, concave,
polygon-with-hole, multipolygon, rectangle degeneracy against the
production `Spatial_Symmetric_Gaussian.compute_integral`, cutoff
square + fast path, NaN-safe padding, table C1/knot-side behavior, bounds,
no-extrapolation, gradient-vs-FD) live in
`tests/test_polygon_mass_backend_shootout.py`: **13/13 pass**.

## 4. Accuracy results (full tables: summary.csv, summary_addendum.csv)

Tolerances: value τ = 5.39e-4 = 0.1·ε_s(3σ) (D-21 square tail); same
provisional number for |d(M)/d log σ| error, reported separately. For the
w_s = 400 m runs the spec's τ(σ) = min(5.39e-4, 0.1·ε_s(σ)) was applied
with an explicit floor of 1e-7: at small σ a fixed 400 m cutoff retains
essentially all mass, ε_s(σ) → 0, and the literal formula demands an
unachievable ~1e-300; the floor (float32 scale) is the documented
deviation, not a silent relaxation. This never mattered — the quad error
(~1e-11) is below even the floored targets.

| Candidate | max abs value err (worst region) | max deriv err | value pass | deriv pass |
|---|---|---|---|---|
| quad, all four (h, q) configs | 2.1e-11 | 1.3e-7 (= FD floor) | all 5 regions | all 5 regions |
| quad (20, 16), w_s = 400 m | 1.8e-11 | — (values only) | 2/2 regions | — |
| table K=24 (oracle-built) | 1.29e-3 | 4.9e-2 | **1/5** | **0/5** |
| table K=40 (oracle-built) | 2.9e-4 | 2.9e-2 | 5/5 | **0/5** |
| table K=64 (oracle-built) | 1.5e-5 | 1.8e-2 | 5/5 | **0/5** |
| table K=40 (quad-built, AD slopes) | 3.9e-6 | 7.0e-5 | 5/5 | 5/5 |
| table K=64 (quad-built, AD slopes) | 5.7e-7 | 1.6e-5 | 5/5 | 5/5 |

Notes.

* Quad is oracle-exact at every tested resolution — even the cheapest
  (h=40 m, q=16; h=20, q=8) — with zero NaNs; the accuracy sweep cannot
  distinguish panel configurations, so the cheapest tested config already
  meets the target and cost, not accuracy, discriminates. Relative errors
  (oracle ≥ 1e-3): ≤ 4e-11. Signed aggregate and summed-mass relative
  errors ≤ 2e-10 / 2e-12.
* Oracle-built tables fail derivatives at EVERY K: the PCHIP /
  centered-difference slopes are only O(h²–h³) accurate in the knot
  spacing, and near-boundary events have strongly curved M(σ). K=24 also
  fails values outright. Bound violations (M > 1 + 1e-12) occur
  (387 → 19 entries from K=24 → K=64; magnitudes bounded by the value
  errors above). This failure is structural to slope estimation from
  values alone, not to the Hermite evaluator.
* Quad-built tables (true AD slopes) pass both gates everywhere with
  ≥ 8× (deriv, K=40) to ~35× (K=64) margin; K=64 has ZERO bound
  violations; no NaNs; no extrapolation attempts (range checks verified
  loud in tests).
* Float32 spot check (production default precision): quad (20, 16) on
  Lower South fails badly in f32 — max error 1.6e-2 (catastrophic
  cancellation in the signed boundary sum over ~1.4k panels). **The quad
  evaluation requires float64** (or compensated summation). Table values
  built in f64 and cast to f32 lose only ~1e-7 — tables are f32-safe at
  NUTS time.

## 5. Performance (perf.csv; CPU, x64, block_until_ready, median of 5–15 reps)

Largest actual single fit resolved from the data: the current batch
workflow's largest fit is `fairmount_box` (10,967 events, a RECTANGLE
domain, so it never needs this backend); the largest realistic
polygon-domain fit is the largest planning district, **Lower North,
16,479 events** — used for scaling. Batches drawn from its real events.

| n events | quad value+grad (h=20, q=16) | table value+grad (K=40) | rect-baseline value+grad |
|---|---|---|---|
| 128 | 0.52 s | 0.10 ms | 0.10 ms |
| 512 | 0.66 s | 0.09 ms | 0.14 ms |
| 2,048 | 3.68 s | 0.17 ms | 0.81 ms |
| 8,192 | 13.8 s | 0.56 ms | 3.8 ms |
| 16,479 (largest fit) | **28.5 s** | **1.3 ms** | 7.1 ms |

* Quad: geometric preprocessing negligible (< 0.02 s; 69 KB shared panel
  arrays, 1,728 panels at h=20); JIT compile 0.6–1.4 s per shape; events
  chunked at 256 (reported batch size) to bound the (B×P×q) node tensor.
  Warm cost ~1.7 ms/event/gradient — ~4,000× the rectangle baseline at
  full-fit size, and ~28 s per leapfrog-step gradient contribution at
  n=16,479 is not production-viable for NUTS.
* Tables: evaluation is 5× FASTER than the existing rectangle
  compensator's erf product at n=16,479; memory 10.5 MB (K=40, f64,
  16,479 events; ~640 B/event); serialized accuracy-sample tables total
  410 KB (sizes+SHA-256 per file recorded; kept under
  `results/polygon_mass_backend_shootout/tables/`, EXCLUDED from any
  proposed commit).
* Table construction, oracle-built: measured 0.14–0.81 s/event →
  **projected 1.4–2.2 h** for the largest fit → fails the quick-viability
  test (projection, not measured; the expensive job was not run, per
  spec). Quad-built: measured 0.057 (K=40) / 0.083 (K=64) s/event on the
  accuracy samples → **projected ~16 / ~23 min** single-threaded for
  16,479 events — under the 30-min bar, embarrassingly parallel, once per
  fit, and reusable across chains. Full-fit numbers are PROJECTED from
  linear per-event scaling; all other numbers are measured.
* No claim of "% overhead to NUTS" is made from these microbenchmarks; the
  §10.g pilot profile (967 s at n=1,642, machine-local) is the only
  full-model reference and was not re-run.

## 6. Cutoff (A ∩ C_j) machinery

Validated on Lower South and Mifflin Square at w_s = 400 m, all registered
σ: quad matches the exact-intersection oracle to 1.8e-11; the analytic
fast path (C_j ⊂ A) equals erf(w_s/√2σ)² to 1e-12 (test); per-event
clipped panel sets are σ-independent because w_s is fit-constant, so the
same fixed-shape machinery serves the finite-cutoff mode. For tables, a
finite w_s enters only through the builder (same quad evaluation with the
ws-aware kernel); the evaluator is unchanged.

## 7. Verification

* New focused tests: 13 passed (includes the §13 oracle self-test).
* Existing full suite (all 155 tests incl. likelihood/geometry):
  **155 passed** (10:46 wall — slowed by CPU contention with the
  background shootout run, not by any code change; no tracked file was
  modified).
* §13 oracle behavior preserved: reused unchanged, convergence settings
  untouched, self-test passing.
