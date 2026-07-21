# OP-9 decision record — polygon-mass backend

Status: RECOMMENDATION for reviewer sign-off. Experiment only; no
production code touched; nothing committed. Full evidence in `report.md`,
`summary.csv`, `summary_addendum.csv`, `perf.csv`, `run_config.json`.

## Target mass and cutoff semantics

M_j(σ) = ∫_{A ∩ C_j} N(s − s_j; σ²I) ds, where C_j is the event-centered
real-unit square of half-width `spatial_window` (w_s). w_s is FIXED during
a fit (user-supplied; `main.py:1530/1651`) and OPTIONAL: None (the current
default) means the uncut ∫_A. σ is sampled; C_j is σ-independent. The §13
diagnostic oracle computes the uncut mass (its 8σ clip is numerical,
< 2.5e-15) and was preserved unchanged; A ∩ C_j gets a minimal oracle
wrapper (exact intersection, then the same integral). Fixed-cutoff
semantics were NOT replaced by w_s = 3σ; the finite-cutoff runs used a
representative w_s = 400 m, labeled as such. Cutoff selection is 3e.

## Configurations tested

* Quadrature (fixed-node, jitted, padded+masked, AD gradient):
  (h, GL) = (10,8), (20,8), (20,16), (40,16); uncut on 5 regions; w_s=400
  on 2 regions; float32 spot check.
* Oracle-built Hermite tables: K = 24, 40, 64 (PCHIP slopes; CD slopes on
  detected non-monotone rows).
* Addendum: quad-built Hermite tables, K = 40, 64 (quad values + AD
  slopes; gated against the shapely oracle).

Regions: Lower South, North Delaware, Central, East Fairmount Park,
Mifflin Square. σ: 6 registered + 5 midpoints + 20 seeded log-uniform +
knot-adjacent pairs (49 values); derivatives at 11.

## Pass/fail by geometry and backend

| Backend | Value gate (τ=5.39e-4) | Derivative gate (5.39e-4, provisional) |
|---|---|---|
| Quad, every config | PASS all 5 regions (≤2.1e-11) | PASS all 5 (≤1.3e-7 = FD floor) |
| Table K=24 (oracle-built) | FAIL 4/5 (≤1.29e-3) | FAIL 5/5 (≤4.9e-2) |
| Table K=40 (oracle-built) | pass 5/5 (≤2.9e-4) | FAIL 5/5 (≤2.9e-2) |
| Table K=64 (oracle-built) | pass 5/5 (≤1.5e-5) | FAIL 5/5 (≤1.8e-2) |
| Table K=40 (quad-built) | PASS 5/5 (≤3.9e-6) | PASS 5/5 (≤7.0e-5) |
| Table K=64 (quad-built) | PASS 5/5 (≤5.7e-7) | PASS 5/5 (≤1.6e-5); 0 bound violations |

No geometry favored a different method than any other.

## Cost and memory (largest realistic polygon fit: Lower North, n=16,479)

* Quad warm value+gradient: 28.5 s (~1.7 ms/event) — ~4,000× the
  rectangle-compensator baseline (7.1 ms); prep negligible; 69 KB arrays;
  event-chunked at 256. NOT viable inside NUTS.
* Table warm value+gradient: 1.3 ms — 5× FASTER than the rectangle
  baseline; ~640 B/event memory; f32-safe.
* Table build: oracle-built PROJECTED 1.4–2.2 h (fails quick viability;
  job not run). Quad-built MEASURED 0.057–0.083 s/event on accuracy
  samples → PROJECTED ~16 min (K=40) / ~23 min (K=64) single-threaded,
  once per fit, parallelizable. All full-fit numbers are projections from
  linear per-event scaling; everything else is measured.

## Recommended OP-9 resolution

**Per-event lookup tables in log σ (C1 cubic Hermite), with knot values
AND knot slopes produced by the fixed-node quadrature backend in float64
(values ~1e-11 vs oracle; slopes by forward-mode AD).** Recommended
configuration: **K = 64 log-uniform knots** over the σ range the prior
supports (K=40 also passes; K=64 buys ~35× margin and zero bound
violations for ~7 min more build). Quadrature panel config for building:
h = 20 m, GL-16 (already indistinguishable from the oracle at h=40/GL-16;
h=20 kept for margin at negligible cost).

This is ONE production path with an offline builder stage, not automatic
backend selection and not two evaluation paths: the quadrature code is the
table builder (and the rectangle-degeneracy/acceptance-test tool); the
Hermite table is the only thing the likelihood evaluates. Neither original
candidate passes alone: pure quadrature fails production cost by ~3
orders of magnitude on the worst promised geometry; pure oracle-built
tables fail the derivative tolerance at every tested K and the build-time
viability test. The combination passes every gate with margin. (Had the
combination also failed, the recorded next-smallest experiment was
σ-banded static panel pruning for quad; it is unnecessary.)

The provisional derivative tolerance 5.39e-4 did not need relaxing — the
recommended configuration beats it 34×, so no sensitivity analysis is
required for the decision.

## Contract obligations the 3d implementation must carry

* Build tables in float64 (quad in f32 fails: 1.6e-2 error, catastrophic
  cancellation); cast to f32 for evaluation is safe (~1e-7).
* No-extrapolation contract: knot range must cover the σ prior's support
  with margin, derived at fit setup and recorded in provenance
  (range check raises before evaluation; kernel yields NaN → loud, per
  invariant 11.9).
* Finite w_s enters through the builder (ws-aware quad kernel + analytic
  C_j ⊂ A fast path); D-18 single-support-object discipline: the same
  prepared support object must feed simulator thinning and table builder.
* Table provenance: geometry hash, event-set hash, knot range/count,
  builder config (h, GL order), builder git commit — following the
  `implementation_identity()` precedent (§16).
* Rectangle degeneracy acceptance test (10.d): table on a rectangle A vs
  the closed-form erf product (tested here at the evaluator level;
  13/13 focused tests pass, suite 155/155).

## Remaining risks before 3d production integration

1. Tables are per-event: simulation-time offspring PROPOSALS at new
   locations cannot use event tables. The simulator's acceptance step is
   pointwise (within_real_box_window + final A clip) and does not need
   M_j; but any future use of per-location mass inside simulation would
   need the quad kernel directly (f64, slow path) — flag at design time.
2. Build cost scales with events × knots; a city-scale fit (127k events)
   projects to ~3 h single-threaded — parallelize or coarsen before D-19
   work relies on it.
3. σ-range misdeclaration surfaces as a loud error mid-NUTS if the
   sampler proposes log σ outside the table; the 3e cutoff/prior layer
   should validate the range against the prior BEFORE sampling starts.
4. Bound violations at K=40 (≤16 entries, ≤4e-6 above 1) are harmless at
   these tolerances but argue for K=64 where they are zero.
5. The stale `spatial_window=0.1` in `batch_park_fits.py` (real-unit
   semantics change) needs a decision before any park fit is rerun.
6. Machine-locality: all timings are this machine (CPU, x64); the 30-min
   build viability margin should be rechecked if fits move hardware.

## Files a later 3d implementation would change

* `bstpp/likelihood.py` — polygon-mass evaluation atom (Hermite lookup)
  alongside `rectangular_excitation_compensator`; mode dispatch per D-17.
* `bstpp/main.py` — support-object construction (panelization + table
  build at fit setup), mode selection/provenance, simulator leg reuse of
  the single support object (D-18).
* NEW `bstpp/polygon_mass.py` (or similar) — panelization, quad builder
  kernel, Hermite evaluator (promoted from
  `scripts/polygon_mass_backend_shootout.py`).
* `bstpp/utils.py` — only if the support-object seam lands there.
* `tests/` — promote the shootout's synthetic cases to production tests;
  add D-18 consistency and provenance tests.
* Docs: §15 guide addendum hooks; OP-9 row moved to the decision register.
