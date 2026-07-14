# Boundary, bounding-box, and window semantics: an inventory

Status: verified against the code at the revised Phase 2c tip (real-unit
trigger contract). Purpose: Terhi's request to have the between-case
differences in bounding-box handling "clearly listed" ahead of Phase 3. Each
row states what the code does NOW; the final section lists the Phase 3
candidates. Nothing here is hidden in the likelihood: every approximation
below is either exact, or declared.

## Conventions

Internal coordinates: the domain's bounding rectangle A_ is affinely mapped
to the unit square (per-axis; the real spans are `args['axis_scales']`); the
computational grid (25 x 25) tiles the RECTANGLE, not the domain polygon.
Background quantities (fields, cell areas, a_0, window) live in internal
units. THE SPATIAL TRIGGER IS THE DELIBERATE EXCEPTION: sigmax_2 and
spatial_window are REAL-unit quantities (units of the input X/Y columns) --
the kernel is isotropic in real space, offspring displacements are
N(0, sigmax_2 * I) in real coordinates, and the truncation window is a
real-space square. The internal/real conversion happens at exactly three
declared sites: the event-term atom `real_spatial_trigger_values`
(displacements stretched per axis, density converted to the internal measure
by the Jacobian sx*sy), the compensator's limit stretch (mass is
dimensionless, no Jacobian), and the simulator's direct real-unit draw. The
temporal trigger (beta, window) remains internal-unit: a declared asymmetry,
Phase 3 conversion-layer work, and a pure relabel rather than a family
change.

## Inventory: how each integral treats the domain boundary

| # | Quantity | Support used | Boundary treatment | Status |
|---|---|---|---|---|
| 1 | Constant background compensator (plain Hawkes, gdf domain) | exact polygon A | `A_area` = polygon area / rectangle area | EXACT |
| 2 | Cox background spatial integral, no covariates (`Itot_xy`) | grid cells intersecting A | every intersecting cell charged FULL internal area 1/n_xy^2 | OVERCHARGE on boundary cells of non-rectangular gdf domains; exact for rectangles/arrays |
| 3 | Cox background spatial integral, with covariates | common refinement of grid x covariate polygons | EXACT intersection areas from `int_df` | exact w.r.t. the COVARIATE support -- which is not necessarily A (see #4) |
| 4 | Covariate support vs domain | covariate polygons as given | no intersection with A is taken; `cov_area` = covariate polygon areas | if cov support != A: under-/over-coverage (the known multi-support issue) |
| 5 | Excitation compensator (eq. 27) | bounding RECTANGLE A_, all domain types | per-axis erf mass at REAL-unit limits; scalar clip at the real-unit spatial_window | OVERCHARGE for non-rectangular domains (declared approximation); exact for rectangles |
| 6 | Simulator: background locations | grid cells (Cox) / domain rows (plain) | full-cell sampling, then final sjoin clips to the true polygon | realizes the TRUE polygon; mismatches #2/#5's charges for non-rect domains -- consistent with their declared status |
| 7 | Simulator: offspring cascade | bounding rectangle, in-loop (Prop 1.1(ii)) + final polygon sjoin | discarded before parenting w.r.t. the rectangle -- matching #5's charge exactly | EXACT vs the compensator on rectangles |
| 8 | Kernel geometry | -- | ISOTROPIC IN REAL COORDINATES (real-unit contract): the kernel family no longer depends on the bounding rectangle's shape; sigmax_2 is a real-unit parameter, directly interpretable and comparable across domains | RESOLVED (was: internal-isotropic => real-space anisotropy ratio = box aspect ratio); pinned by identity I12 (box invariance) and the I10 negative control |
| 9 | spatial_window | -- | REAL length; per-axis box max(dx,dy) <= ws in REAL units, identical in pair set (within_real_box_window), compensator (real limits clipped at scalar ws), and offspring thinning | EXACT and symmetric on rectangles; semantics changed from internal-unit Euclidean disc (no closed-form disc-rectangle mass exists; an internal-unit box would have made the truncation region box-shape-dependent in real space) |
| 10 | Background PRIOR geometry (Cox) | -- | the PriorVAE spatial decoder was trained against an ISOTROPIC SE kernel on the INTERNAL unit square, so the prior's real-space correlation lengths are l*sx vs l*sy and cell resolution is sx/25 x sy/25 per axis | DECLARED, prior-side only: the background LIKELIHOOD is exactly affine-invariant (piecewise-constant fields, cell lookups, area-weighted sums -- no metric object), so this is a soft, data-dominated regularization asymmetry, distinct from and compounding the documented PriorVAE oversmoothing; not fixable without retraining decoders |

## Consequences worth remembering

- For RECTANGULAR domains (array or box gdf), every likelihood/simulator row
  above is exact: simulator, compensator, and event term agree by
  construction. This is the SBC-supported regime -- spatial_window now
  allowed, AND the trigger legs are invariant to the choice of bounding
  rectangle (I12), so per-domain sigmax_2 posteriors are directly comparable
  in real units.
- The trigger's unit contract is REAL; the background's is INTERNAL. That is
  a deliberate split: the trigger is a metric object in the likelihood (a
  family choice the data cannot undo), the background is not (row #10 is a
  prior-side echo only). Any new metric object added to the likelihood must
  declare its units the same way.
- For non-rectangular gdf domains, #2 and #5 overcharge in DIFFERENT ways
  (full boundary cells vs full rectangle) while the simulator realizes the
  true polygon: simulate-and-recover on such domains conflates these three
  conventions and is not currently a clean identity regime.
- #3 vs #2 is an internal inconsistency: the covariate path already owns the
  exact-intersection machinery that #2 lacks.
- Geographic (lon/lat) input coordinates would reintroduce ground-truth
  anisotropy through the front door (1 deg lon = cos(lat) * 111 km); the
  constructor warns heuristically. Use a metric CRS.

## Phase 3 candidates (in suggested order)

1. Unify #2 with #3's exact treatment: intersect the grid with A itself
   (same `int_df` machinery), giving exact boundary-cell areas for the
   no-covariate Cox background. Declared-rebaseline commit (Itot_xy moves
   for non-rect gdf fits).
2. Support intersection for covariates (#4): integrate over cov-support
   INTERSECT A, with the multi-support geometric intersection evaluated in
   the pre-refactor analysis (exact under piecewise-constant covariates).
3. Excitation beyond the rectangle (#5) for non-rect domains: either keep
   the declared approximation (cheap, current), or charge polygon-clipped
   mass numerically. Decide only if a non-rectangular domain becomes a
   research target.
4. Conversion layer for the remaining internal-unit interface: beta and
   window (temporal) accepted/reported in days; report axis_scales-derived
   quantities alongside posteriors. (The spatial half of the old candidate
   is DONE: sigmax_2 / spatial_window are real-unit by contract.)
5. Optional trigger extension: two sampled real-unit scales (sigma_x,
   sigma_y) if corridor-aligned spreading becomes a modeling question --
   the compensator stays closed-form since everything factorizes per axis.
   A deliberate decision, not a default.
