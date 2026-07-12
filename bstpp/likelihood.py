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
"""
import jax


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
