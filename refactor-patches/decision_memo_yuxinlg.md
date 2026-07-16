# Decision memo: real-unit spatial trigger contract (supersedes the 2c patch-2 gate)

**From:** Terhi
**Needs:** yuxinlg sign-off before patches 2-3 of the revised series land.
**Status of prior gate:** The original Phase 2c patch 2 (per-axis box `spatial_window`
in internal units) was drafted but NOT landed. This memo re-scopes that gate; the
original patch 2 is withdrawn and replaced.
**Status of this series:** Drafted AND executed against a live clone of the refactor
branch (pinned environment reproduced); all suite counts, RED verifications, and pin
claims in the commit bodies were observed, not predicted. Terhi's design decisions are
recorded: D1 contract adopted with commit-message highlight; D2 conversion at the
likelihood boundary (A2); D3 no default prior (user must supply, now in real units);
D4 geographic-coordinate ingestion warning included.

## What is being decided

Two coupled model-semantics changes, adopted together as one contract:

1. **Spatial trigger kernel.** Offspring displacements are distributed
   N(0, sigma^2 * I_2) **in real coordinate units** (the units of the input X/Y
   columns). Previously the kernel was isotropic in *internal* (unit-square)
   coordinates, which under the per-axis affine ingestion map made the real-space
   kernel an axis-aligned anisotropic Gaussian whose anisotropy ratio equals the
   bounding rectangle's aspect ratio (consolidation doc, Prop. "aniso"). The old
   behavior tied the fitted kernel *family* to a preprocessing choice (the bounding
   box); the new contract makes the trigger invariant to it.

2. **Spatial window.** `spatial_window` = w_s is a **real length**: pairs kept iff
   max(|dx|, |dy|) <= w_s in real units (a real-space square). The compensator
   charges this truncation exactly (per-axis erf limits, converted to real units,
   clipped at the scalar w_s -- the exact mirror of min(T - t, w) temporally), and
   offspring thinning applies the identical predicate. This retains the box (not
   disc) semantics of the withdrawn patch -- box is the only shape with a closed-form
   compensator -- but the box is now square in real space rather than in internal
   coordinates.

## What changes for users

- `sigmax_2` (the sampled trigger variance) is now in **squared real units**; its
  prior must be specified accordingly. There is no default prior; existing scripts
  that pass a prior calibrated to the unit square will fit a *different model* until
  the prior is restated in real units. This is deliberate and documented.
- `spatial_window` is now a real length, not an internal fraction.
- Posterior summaries of the trigger become directly interpretable (e.g., meters)
  and **comparable across parks** -- previously each park's bounding-box aspect
  ratio selected a different real-space kernel family, making cross-park sigma
  comparisons incommensurable.
- Custom user triggers (the `Trigger` subclass API) now receive real-unit
  difference matrices and real-unit integration limits. `spatial_double_exp`-style
  user code keeps working mechanically but its parameter units change meaning.

## What does not change

- The likelihood measure stays internal; the background (all three fields,
  covariates, compensator) is untouched -- it is exactly affine-invariant
  (piecewise-constant fields; no metric object in its likelihood legs).
- alpha keeps its meaning (expected offspring per parent, up to truncation mass).
- Temporal trigger and temporal window remain internal-unit (declared asymmetry;
  their conversion is a pure relabel deferred to the Phase 3 conversion layer).
- On a square bounding box the new kernel with sigma = sigma_int * L reproduces the
  old one exactly (pinned by test).

## Why now

The anisotropy is not cosmetic for this project: the creek-corridor parks have
strongly elongated bounding boxes, so the imposed real-space anisotropy is large,
non-estimable, and silently different per park. Fixing it before SBC and before
Phase 3 means the SBC-supported regime and all downstream posteriors are stated in
the intended model from the start.

## The alternative (rejected, recorded)

Keep the internal-isotropic kernel and document the per-axis conversion of
posteriors. Rejected because the defect is in the model *family* (fixed, wrong
anisotropy ratio), not in the reporting: no conversion of a fitted isotropic-internal
posterior recovers an isotropic-real fit on a non-square box.

## Sign-off requested

- [ ] Real-unit isotropic kernel contract (item 1)
- [ ] Real-unit square-box `spatial_window` semantics (item 2)
- [ ] The test edits entailed (dense pair reference; I4/I11 finite-ws variants;
      new box-invariance identity I12) as part of the same signed-off change
