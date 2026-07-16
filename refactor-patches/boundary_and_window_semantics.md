# Boundary, bounding-box, and window semantics: an inventory

Status: verified against the code at the Phase 2c tip. Purpose: Terhi's
request to have the between-case differences in bounding-box handling
"clearly listed" ahead of Phase 3. Each row states what the code does NOW;
the final section lists the Phase 3 candidates. Nothing here is hidden in
the likelihood: every approximation below is either exact, or declared.

## Conventions

Internal coordinates: the domain's bounding rectangle A_ is affinely mapped
to the unit square; all likelihood quantities (sigmax_2, window,
spatial_window, cell areas) live in internal units. The computational grid
(25 x 25) tiles the RECTANGLE, not the domain polygon.

## Inventory: how each integral treats the domain boundary

| # | Quantity | Support used | Boundary treatment | Status |
|---|---|---|---|---|
| 1 | Constant background compensator (plain Hawkes, gdf domain) | exact polygon A | `A_area` = polygon area / rectangle area | EXACT |
| 2 | Cox background spatial integral, no covariates (`Itot_xy`) | grid cells intersecting A | every intersecting cell charged FULL internal area 1/n_xy^2 | OVERCHARGE on boundary cells of non-rectangular gdf domains; exact for rectangles/arrays |
| 3 | Cox background spatial integral, with covariates | common refinement of grid x covariate polygons | EXACT intersection areas from `int_df` | exact w.r.t. the COVARIATE support -- which is not necessarily A (see #4) |
| 4 | Covariate support vs domain | covariate polygons as given | no intersection with A is taken; `cov_area` = covariate polygon areas | if cov support != A: under-/over-coverage (the known multi-support issue) |
| 5 | Excitation compensator (eq. 27) | bounding RECTANGLE A_, all domain types | per-axis erf mass; since Phase 2c, per-axis clip at spatial_window | OVERCHARGE for non-rectangular domains (declared approximation); exact for rectangles |
| 6 | Simulator: background locations | grid cells (Cox) / domain rows (plain) | full-cell sampling, then final sjoin clips to the true polygon | realizes the TRUE polygon; mismatches #2/#5's charges for non-rect domains -- consistent with their declared status |
| 7 | Simulator: offspring cascade | bounding rectangle, in-loop (Phase 2c) + final polygon sjoin | discarded before parenting (Prop 1.1(ii)) w.r.t. the rectangle -- matching #5's charge exactly | EXACT vs the compensator on rectangles |
| 8 | Kernel geometry | -- | internal isotropic Gaussian is ANISOTROPIC in real coordinates for non-square rectangles (per-axis scaling); sigmax_2 is an internal-unit parameter | by construction; document when reporting real-space kernel scales |
| 9 | spatial_window (Phase 2c) | -- | PER-AXIS box in internal units, identical in pair set, compensator (clipped limits), and offspring thinning | EXACT and symmetric on rectangles; semantics changed from Euclidean disc (no closed-form disc-rectangle mass exists) |

## Consequences worth remembering

- For RECTANGULAR domains (array or box gdf), every row above is exact:
  simulator, compensator, and event term agree by construction. This is the
  SBC-supported regime (spatial_window now allowed, post-Phase 2c).
- For non-rectangular gdf domains, #2 and #5 overcharge in DIFFERENT ways
  (full boundary cells vs full rectangle) while the simulator realizes the
  true polygon: simulate-and-recover on such domains conflates these three
  conventions and is not currently a clean identity regime.
- #3 vs #2 is an internal inconsistency: the covariate path already owns the
  exact-intersection machinery that #2 lacks.

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
4. Units: window / spatial_window are internal-unit parameters accepted at a
   real-unit-looking interface -- route through the Phase 3 conversion layer.
