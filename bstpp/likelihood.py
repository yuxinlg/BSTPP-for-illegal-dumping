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
