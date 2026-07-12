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
- seasonal_time_integral ................ exact background time integral on
  the seasonal diagonal a = sigma(t), eq. (26): the (n_t x n_s) overlap
  matrix W (eq. 25, carrying the internal cell measure) contracts exp(f_a),
  then exp(a_0 + f_t) weights the per-cell seasonal mass. Returns
  (seasonal_mass, total) -- the per-cell mass also feeds the rate_time
  diagnostic at the model layer, and (Phase 2b) the _sim_cox normalizer,
  which makes identity (I1) structural.
- spatial_refinement_integral ........... spatial background integral over
  the common refinement, eq. (24): exp(f_xy[c] (+ b_0[m])) contracted against
  the refinement-cell areas |C_c intersect A_m|. The no-covariate grid
  integral is the same atom with uniform areas 1/n_xy^2 and no b_0.
  Refinement invariance (I9) is a property of this expression.
- constant_background_integral .......... plain-Hawkes background compensator
  mu * T * |X| (reused by _sim_hawkes_bg in Phase 2b, making the background
  half of (I11) structural).
- covariate_background_integral ......... plain-Hawkes covariate background
  compensator (mu_cells @ cell_areas) * T over the covariate partition.
- rectangular_excitation_compensator .... truncated excitation compensator,
  eq. (14) specialized as implemented: exponential temporal mass F_beta on
  min(T - t_j, w) (temporal truncation matched to the pair set) and Gaussian
  rectangle mass (eq. 27) over axis-aligned bounds. HONEST SCOPE: this is NOT
  the general-domain compensator -- for non-rectangular domains X the
  rectangle mass overcharges (declared approximation, guide Sec. 2.5), and a
  finite spatial_window is applied on the EVENT side only (pair construction),
  not here (known, documented discrepancy).
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


def rectangular_excitation_compensator(alpha, t_events, xy_events, horizon,
                                       temporal_window, rectangular_bounds,
                                       temporal_parameters, spatial_parameters,
                                       temporal_trigger, spatial_trigger):
    """Total excitation compensator over [0, horizon] x rectangle.

    Sum over parents j of
        alpha * F_temporal(min(horizon - t_j, temporal_window))
              * (Gaussian rectangle mass at s_j, eq. 27),
    computed through the SAME trigger protocol the likelihood samples from
    (temporal_trigger.compute_integral / spatial_trigger.compute_integral),
    so the simulator identity (I4)/(I11) is tied to one expression.

    rectangular_bounds: (x_min, x_max, y_min, y_max) in internal units.
    See module docstring for the honest scope of this specialization.
    Returns a scalar.
    """
    x_min, x_max, y_min, y_max = rectangular_bounds
    temp_part = alpha * temporal_trigger.compute_integral(
        temporal_parameters, jnp.minimum(horizon - t_events, temporal_window))
    sp_limits = jnp.stack((x_max - xy_events[0], xy_events[0] - x_min,
                           y_max - xy_events[1], xy_events[1] - y_min)
                          ).reshape(2, 2, -1)
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

    Note (rebaseline history): this atom computes exp(log_rate) @ areas; the
    pre-extraction no-covariate code computed sum(exp(f_xy)[cells]) / n_xy^2,
    an algebraically equal expression with a different floating-point
    rounding sequence (declared expression change, tolerance-verified).
    """
    log_rate = f_xy[field_indices]
    if b_0 is not None:
        log_rate = log_rate + b_0[covariate_indices]
    return jnp.exp(log_rate) @ areas


def constant_background_integral(mu, horizon, domain_area):
    """Compensator of a constant background: mu * T * |X| (internal units)."""
    return mu * horizon * domain_area


def covariate_background_integral(mu_cells, cell_areas, horizon):
    """Compensator of a piecewise-constant covariate background.

    (mu_cells @ cell_areas) * horizon over the covariate partition {A_m};
    cell_areas are |A_m| in internal measure (args['cov_area']).
    """
    return mu_cells @ cell_areas * horizon
