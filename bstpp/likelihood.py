"""Pure likelihood atoms for the BSTPP discrete model (guide, Section 2).

Design contract (Phase 2):
- Pure JAX only: no numpyro, no reads/writes of the model `args` dict, no
  pandas/GeoPandas. All inputs are explicit arrays, scalars, trigger-parameter
  dicts, or trigger objects.
- Trace-site registration (numpyro.deterministic / numpyro.factor) stays in
  inference_functions.py: sampled and deterministic site names are the
  posterior interface and are never created here.
- This module MAY depend on the trigger protocol (compute_trigger /
  compute_integral, both pure) -- that protocol is the package's documented
  extension point. Atoms that do not need kernel knowledge (e.g. pair
  aggregation) deliberately do not receive trigger objects.

Equation concordance (guide numbering):
- aggregate_pair_trigger_values ......... inner sum of eq. (23), given
  per-pair kernel values (kernel evaluation stays at the model layer)
- real_spatial_trigger_values ........... spatial kernel values at the
  excitation pairs under the REAL-unit trigger contract: internal pair
  displacements stretched to real units per axis, the real-area density
  evaluated there, then converted to the internal measure by the Jacobian
  axis_scales[0] * axis_scales[1] of the affine ingestion map. This atom is
  the SINGLE declared unit boundary of the event term.
- seasonal_time_integral ................ exact background time integral on
  the seasonal diagonal a = sigma(t), eq. (26): the (n_t x n_s) overlap
  matrix W (eq. 25, carrying the internal cell measure) contracts exp(f_a),
  then exp(a_0 + f_t) weights the per-cell seasonal mass. Returns
  (seasonal_mass, total) -- the per-cell mass also feeds the rate_time
  diagnostic at the model layer, and (Phase 2b) the _sim_cox normalizer,
  which makes identity (I1) structural.
- spatial_refinement_masses ............. per-cell masses of the eq. (24)
  integrand, exp(f_xy[c] (+ b_0[m])) * |C_c intersect A_m|: the BASE
  computation, consumed by the simulator's conditional cell draw.
- spatial_refinement_integral ........... eq. (24) integral, literally
  sum(spatial_refinement_masses(...)): one source of truth for the
  integrand. The no-covariate grid is the special case of uniform areas
  1/n_xy^2 and no b_0. Refinement invariance (I9) is a property of this
  expression.
- background_masses ..................... per-cell masses of the plain-Hawkes
  background compensator, mu * |cell| * T (mu scalar for the constant
  background, vector over the covariate partition): the simulator's per-cell
  Poisson rates. Unit tests pin sum(background_masses) against the two
  integral atoms below.
- constant_background_integral .......... plain-Hawkes background compensator
  mu * T * |X|.
- covariate_background_integral ......... plain-Hawkes covariate background
  compensator (mu_cells @ cell_areas) * T over the covariate partition.
- rectangular_excitation_compensator .... truncated excitation compensator,
  eq. (14) specialized as implemented: exponential temporal mass F_beta on
  min(T - t_j, w) (temporal truncation matched to the pair set) and Gaussian
  rectangle mass (eq. 27) over axis-aligned bounds, evaluated at REAL-unit
  limits (internal edge distances stretched per axis; the mass itself is a
  probability, so no Jacobian appears here, unlike the event term). HONEST
  SCOPE: this is NOT the general-domain compensator -- for non-rectangular
  domains X the rectangle mass overcharges (declared approximation, guide
  Sec. 2.5). A finite spatial_window (REAL length) is charged EXACTLY: the
  real-unit limits are clipped at the scalar ws -- the same per-axis box the
  pair set and the offspring thinning use (within_real_box_window), so all
  three legs agree on rectangles and the historical event-side-only
  discrepancy is closed.
- polygon_excitation_compensator ........ Phase 3d polygon-mode compensator:
  same temporal factor as the rectangle atom; spatial mass is the hybrid
  Hermite lookup M_j(sigma) from an offline float64 quad-built table (no
  polygon quadrature inside the likelihood).

Unit contract (real-unit spatial trigger). The spatial trigger is a density
per REAL area, isotropic in the units of the input X/Y columns; sigmax_2 and
its prior are REAL-unit quantities. The likelihood's intensity remains per
INTERNAL volume, so exactly two conversions exist, both in this module:
real_spatial_trigger_values (displacements out, Jacobian back) and the
compensator's limit stretch (mass is dimensionless). The temporal trigger
remains internal-unit (declared asymmetry; the temporal conversion is a pure
relabel deferred to the Phase 3 conversion layer). On a square box of side L,
sigma_real = sigma_internal * L reproduces the historical internal-unit
kernel exactly (Jacobian L^2 cancels the density normalization).
"""
import jax
import jax.numpy as jnp


def aggregate_pair_trigger_values(coords, temporal_values, spatial_values, n_events):
    """Per-event excitation sums from aligned per-pair kernel values.

    Inner sum of the event term, eq. (23): for each event i, sums
    temporal_values[p] * spatial_values[p] over all admissible pairs
    p = (i, j) with coords[p] = (i, j). The three arrays are aligned by the
    pair axis; ordering within the pair axis is irrelevant (I3).

    Parameters: coords (P, 2) int pair indices from aligned_difference_pairs;
    temporal_values, spatial_values (P,) kernel evaluations at the pair lags /
    displacements; n_events static int (segment_sum needs a static size).
    Returns (n_events,) -- the per-event sums; the caller multiplies by alpha.
    """
    pair_values = temporal_values * spatial_values
    return jax.ops.segment_sum(pair_values, coords[:, 0], n_events)


def real_spatial_trigger_values(spatial_trigger, spatial_parameters, coords,
                                x_vals, y_vals, axis_scales):
    """Per-pair spatial kernel values under the REAL-unit trigger contract,
    expressed in the internal measure the likelihood integrates over.

    Contract: the spatial trigger is a probability density per REAL area
    (isotropic in the units of the input X/Y columns). The likelihood's
    intensity is per INTERNAL (unit-square) area, so two conversions happen
    HERE and only here -- the single declared unit boundary of the event term:

      1. the internal pair displacements (x_vals, y_vals) are stretched to
         real units per axis: dx_real = dx * axis_scales[0], etc.;
      2. the real-area density converts to the internal measure by the
         Jacobian of the per-axis affine ingestion map:
         phi_internal = axis_scales[0] * axis_scales[1] * phi_real.

    On a square box of side L with sigma_real = sigma_internal * L the two
    factors cancel exactly and this reproduces the historical internal-unit
    kernel (pinned in tests/test_likelihood_atoms.py).
    """
    _, values = spatial_trigger.compute_trigger(
        spatial_parameters,
        (coords, x_vals * axis_scales[0], y_vals * axis_scales[1]))
    return values * (axis_scales[0] * axis_scales[1])


def polygon_excitation_compensator(alpha, t_events, horizon, temporal_window,
                                   temporal_parameters, spatial_parameters,
                                   temporal_trigger, mass_table):
    """Total excitation compensator over [0, horizon] x polygon support.

    Same temporal factor as :func:`rectangular_excitation_compensator`; the
    spatial mass at each parent is the Phase 3d Hermite lookup
    ``M_j(sigma)`` for ``sigma = sqrt(sigmax_2)`` (table built offline over
    ``[min_sigma, max_sigma]``; no polygon quadrature in this path).
    """
    from .polygon_mass import _TABLE_MASSES

    temp_part = alpha * temporal_trigger.compute_integral(
        temporal_parameters, jnp.minimum(horizon - t_events, temporal_window))
    sigma = jnp.sqrt(spatial_parameters['sigmax_2'])
    log_sigma = jnp.log(sigma)
    # Online dtype follows the model (typically float32); tables were built
    # float64 and cast here. Extrapolation yields NaN by construction.
    dt = jnp.result_type(log_sigma, temp_part)
    sp_part = _TABLE_MASSES(
        log_sigma,
        jnp.asarray(mass_table.log_knots, dtype=dt),
        jnp.asarray(mass_table.values, dtype=dt),
        jnp.asarray(mass_table.slopes, dtype=dt),
    )
    return jnp.sum(temp_part * sp_part)


def rectangular_excitation_compensator(alpha, t_events, xy_events, horizon,
                                       temporal_window, rectangular_bounds,
                                       temporal_parameters, spatial_parameters,
                                       temporal_trigger, spatial_trigger, *,
                                       axis_scales, spatial_window=None):
    """Total excitation compensator over [0, horizon] x rectangle.

    Sum over parents j of
        alpha * F_temporal(min(horizon - t_j, temporal_window))
              * (Gaussian rectangle mass at s_j, eq. 27),
    computed through the SAME trigger protocol the likelihood samples from
    (temporal_trigger.compute_integral / spatial_trigger.compute_integral),
    so the simulator identity (I4)/(I11) is tied to one expression.

    rectangular_bounds: (x_min, x_max, y_min, y_max) in internal units.
    axis_scales: (2,) per-axis real lengths of the bounding rectangle
    (REQUIRED, keyword-only: unit interfaces are declared, never implicit).
    The spatial integral limits handed to compute_integral are the REAL-unit
    distances from each parent to the rectangle edges -- internal distances
    stretched per axis -- matching the real-unit kernel the event term
    evaluates. The spatial mass is a probability (dimensionless), so unlike
    the event term no Jacobian factor appears here.

    spatial_window: optional REAL length. When given, each real-unit limit is
    clipped at the SCALAR spatial_window -- charging the Gaussian mass of
    (real-space square of half-width ws at s_j) INTERSECT rectangle, the
    exact integral of the event-side pair predicate (within_real_box_window)
    and the offspring thinning, and a perfect per-axis mirror of
    min(horizon - t, temporal_window) on the time axis. The clip lands AFTER
    the real-unit stretch, which is what makes it a scalar; per-axis box
    semantics are the only choice with a closed-form mass (disc-intersect-
    rectangle has none).
    See module docstring for the honest scope of this specialization.
    Returns a scalar.
    """
    x_min, x_max, y_min, y_max = rectangular_bounds
    temp_part = alpha * temporal_trigger.compute_integral(
        temporal_parameters, jnp.minimum(horizon - t_events, temporal_window))
    sp_limits = jnp.stack((x_max - xy_events[0], xy_events[0] - x_min,
                           y_max - xy_events[1], xy_events[1] - y_min)
                          ).reshape(2, 2, -1)
    # real-unit trigger contract: stretch internal edge distances per axis
    sp_limits = sp_limits * axis_scales[:, None, None]
    if spatial_window is not None:
        # real-unit box window: all three legs share these semantics exactly
        sp_limits = jnp.minimum(sp_limits, spatial_window)
    sp_part = spatial_trigger.compute_integral(spatial_parameters, sp_limits)
    return jnp.sum(temp_part * sp_part)


def seasonal_time_integral(a_0, f_t, f_a, season_overlap):
    """Exact time integral of the Cox background on the seasonal diagonal.

    Eq. (26): Lambda_time = sum_c exp(a_0 + f_t[c]) * (W @ exp(f_a))[c],
    where W = season_overlap (eq. 25) already carries the internal cell
    measure (no extra T/n_t factor -- see the seasonal-diagonal fix).
    Returns (seasonal_mass, total): seasonal_mass = W @ exp(f_a), shape
    (n_t,); total is the scalar Itot_time.
    """
    seasonal_mass = season_overlap @ jnp.exp(f_a)
    total = jnp.sum(jnp.exp(a_0 + f_t) * seasonal_mass)
    return seasonal_mass, total


def spatial_refinement_integral(f_xy, field_indices, areas, b_0=None,
                                covariate_indices=None):
    """Spatial integral of the background over a refinement partition.

    Eq. (24): sum_k exp(f_xy[field_indices[k]] + b_0[covariate_indices[k]])
    * areas[k], with the b_0 term omitted when b_0 is None. field_indices map
    refinement cells to spatial-field cells; areas are the refinement-cell
    areas in internal (unit-square) measure. The uniform no-covariate grid is
    the special case field_indices = in-domain cells, areas = 1/n_xy^2.

    Rebaseline history: (1) pre-extraction the no-covariate code computed
    sum(exp(f_xy)[cells]) / n_xy^2; (2) the first extraction computed
    exp(log_rate) @ areas; (3) this version computes sum(exp(log_rate) *
    areas) = sum(spatial_refinement_masses(...)) so the integrand has one
    source of truth. All three are algebraically equal with different
    floating-point reduction orders; each transition was a DECLARED
    expression change, tolerance-verified against the value+gradient pins.
    """
    return jnp.sum(spatial_refinement_masses(f_xy, field_indices, areas,
                                              b_0, covariate_indices))


def spatial_refinement_masses(f_xy, field_indices, areas, b_0=None,
                              covariate_indices=None):
    """Per-cell masses of the eq. (24) integrand: exp(log_rate) * areas.

    This is the BASE computation: spatial_refinement_integral is literally
    sum(masses), so the integrand exists in exactly one place and the
    simulator's conditional cell draw and the likelihood's integral cannot
    drift apart. The simulator self-normalizes the conditional (masses /
    masses.sum()) and takes the count rate from the same vector's sum.
    """
    log_rate = f_xy[field_indices]
    if b_0 is not None:
        log_rate = log_rate + b_0[covariate_indices]
    return jnp.exp(log_rate) * areas


def constant_background_integral(mu, horizon, domain_area):
    """Compensator of a constant background: mu * T * |X| (internal units)."""
    return mu * horizon * domain_area


def covariate_background_integral(mu_cells, cell_areas, horizon):
    """Compensator of a piecewise-constant covariate background.

    (mu_cells @ cell_areas) * horizon over the covariate partition {A_m};
    cell_areas are |A_m| in internal measure (args['cov_area']).
    """
    return mu_cells @ cell_areas * horizon


def background_masses(mu, cell_areas, horizon):
    """Per-cell masses of the plain-Hawkes background compensator.

    mu * cell_areas * horizon, broadcasting: mu is a scalar for the constant
    background (cells = domain rows) or a vector over the covariate partition.
    The simulator superposes per-cell Poisson draws on these rates, which is
    distributionally identical to Poisson(total) + multinomial; unit tests
    (not runtime asserts) pin sum(background_masses) against
    constant_background_integral / covariate_background_integral.
    """
    return mu * cell_areas * horizon
