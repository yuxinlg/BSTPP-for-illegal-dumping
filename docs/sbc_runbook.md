# SBC runbook: design, prior units, and pre-registered caveats

Status: living SBC design record. Stage 1 is complete; the Stage 2 design
below was frozen from prior-predictive measurements before the first Stage 2
rank run. This document is the single reference the harness author works
from: it records the accepted SBC design, restates the prior units under the
real-unit trigger contract, and pre-registers known effects so they cannot be
"discovered" as anomalies mid-run.

## Why SBC, and why now

Everything since Phase 0 built the preconditions: the simulator and
likelihood share one set of expressions (a calibration failure cannot be
blamed on simulator/likelihood drift -- that class of explanation is
structurally eliminated), the identity suite pins the pieces individually,
and the golden pins guard the inference path. What no test checks is
end-to-end Bayesian self-consistency -- that prior -> simulate -> fit ->
posterior actually inverts. That is the layer where the double-factor bug
lived, and SBC is the instrument that would have caught it (overconfident
posteriors -> U-shaped rank histograms). Running SBC before Phase 3 means
any finding is chased against a stable, freshly verified codebase; the
Phase 3 config dataclass should then be shaped by what the harness actually
needed to parameterize.

## Staging

Mirrors the atom decomposition, so a failure localizes to the stage that
introduced it (same logic as the commit discipline):

1. **Stage 1: plain Hawkes** (`cox_background=False`) -- four scalar
   parameters, fastest NUTS, exercises pairs, excitation compensator, and
   backgrounds in isolation.
2. **Stage 2: LGCP** -- fields and decoders, no self-excitation. (Would
   have crashed on its first replicate before 965f683: the prior-predictive
   loop feeds z-dicts to `simulate()`, the exact path fixed and pinned by
   `tests/test_lgcp_sim.py`.)
3. **Stage 3: full cox-Hawkes.**

## Ranked functionals (pre-registered)

- Scalars `alpha`, `beta`, `sigmax_2`: rank directly.
- Background: rank `log_background = log(Itot_txy - Itot_excite)` -- or,
  where a pointwise view is wanted, log-intensity at a FIXED set of grid
  points -- **not `a_0`**. `a_0` alone is only approximately identified:
  integral-preserving tilts between fields leave it free to drift (the
  recover_test harness finding; see the `intercept_combination` docstring
  in `scripts/recover_test.py`). Ranking `a_0` would produce SBC "failures"
  that are identifiability facts, not implementation bugs. This distinction
  is pre-registered here precisely so it is not re-litigated mid-run.
- Optional supplementary diagnostics: a few `z` components and derived
  totals (`Itot_txy`).

## Stage 2 pre-registration: LGCP

Stage 2 adds the temporal, seasonal, and spatial PriorVAE fields but no
self-excitation. Its latent truth is `a_0` plus the model's three standard-
normal decoder vectors (`z_temporal`, `z_seasonal`, and `z_spatial`), drawn at
the dimensions declared by the constructed model. The same `a_0` distribution
object is sampled for truth and passed to the fitting model; the `z` priors
are hardcoded standard normals on both sides.

### Fixed spatial gain and scope of the result

The spatial field is

```
f_xy = exp(sp_var_mu) * decode_spatial(z_spatial).
```

Thus `sp_var_mu = 0.0` means **unit gain**, not a disabled spatial field: the
raw decoder output remains random and enters the log-intensity unchanged.
The package default `sp_var_mu = 2.0` multiplies the spatial log-field by
`exp(2)`, making the induced prior on event counts computationally unusable
for unrejected SBC. In 60 measured prior draws, counts ranged from 114 to
approximately 6.7 million; changing the center of an independent `a_0` prior
shifts this distribution but does not narrow its roughly seven-order spread.

Stage 2 is therefore fixed at `sp_var_mu = 0.0`, recorded in the config
identity. This validates end-to-end LGCP composition and NUTS inversion in a
computationally feasible unit-gain regime. It does **not** validate posterior
geometry, numerical stability, or calibration under the production-gain
prior at `sp_var_mu = 2.0`. Nonzero-gain algebra and simulator wiring are
covered separately by `tests/test_decode_fields.py` and
`tests/test_lgcp_sim.py`; the gain itself lives in the shared
`decode_fields.decode_spatial_field` atom used by simulation and likelihood.
The production-gain diffuseness is a modeling finding and motivates the
planned sampled-amplitude/recalibration follow-up.

### Priors and computational budget

- `a_0 ~ Normal(0.4, 0.3)`.
- All three `z` vectors use the model's `Normal(0, I)` prior.
- Prior-predictive budget band: 20--2000 events, with at most 5% below and at
  most 5% above; zero-event draws are forbidden.
- At the distinct check seed, 150 measured draws had median 124,
  5th/95th percentiles 23/687, maximum 1926, 3.3% below 20, 0% above 2000,
  and no zero-event draws.

The real run is never conditioned on this count check: it uses a distinct
master seed and never rejects or skips a replicate based on event count.

### Ranked targets and decision rule

Nine targets are **PRIMARY** and therefore enter the per-replicate ESS gate
and the final acceptance decision:

1. `log_background = log(Itot_txy)`;
2. pointwise log-intensity at five fixed spacetime grid cells,
   `a_0 + f_t[i] + f_a[season_idx_of_t[i]] + f_xy[j]`;
3. the first component of each decoder vector: `z_temporal_0`,
   `z_seasonal_0`, and `z_spatial_0`.

`a_0` is **SUPPLEMENTARY**: its rank and quantile ESS are recorded and
reported, but it does not drive ESS retries or the pass/fail decision. Exact
SBC ranks remain uniform even for weakly identified parameters; the exclusion
is operational, because the soft intercept/field direction can mix slowly.

Stage 2 passes if every primary Monte Carlo ECDF p-value is at least 0.005.
The Bonferroni family-wise false-alarm bound is at most 4.5% across nine
targets; because the targets are correlated, no exact independence-based
family-wise probability is claimed.

## Inference: NUTS as the acceptance instrument

- **NUTS-SBC tests whether the implementation is self-consistent.** It is
  the acceptance instrument.
- **SVI-SBC tests whether the variational approximation is calibrated** --
  a genuinely interesting second question given the PriorVAE oversmoothing
  history (both known effects push toward overconfidence; SVI-SBC would
  quantify the combined shrinkage). Run it only if stage 3 passes under
  NUTS, and read it as a research result about the actual pipeline, not a
  bug hunt.

### Posterior RNG and dependence correction (stage-1 pre-run amendment)

Each SBC fit must receive its own reproducible MCMC key. Reusing the package's
historical fixed `PRNGKey(10)` across every replicate would condition the rank
ensemble on one posterior-sampling stream, whereas the discrete-uniform SBC
reference integrates over independent posterior draws.

Likewise, `L=127` is not obtained by assuming that every fourth NUTS state is
independent. Each replicate starts with an unthinned raw chain (default 508
draws), estimates the minimum ESS across empirical-quantile indicator functions
for every primary rank target (Talts et al., Algorithm 2), and requires
minimum ESS >= 0.95 L. If a replicate fails that gate, the same `(theta, y)` is
refit with `n_next = n_current * ceil(L / E)` (capped at a pre-registered
`max_num_samples`, default 4064), using an attempt-folded MCMC key; only the
final passing chain is uniformly thinned to exactly L states and ranked.
Mixing raw chain lengths across replicates is valid: every replicate still
contributes exactly L near-independent posterior draws under the same ESS
criterion. Exhausting the cap without passing aborts the run; the replicate is
never skipped.

## Priors: tight, matched, no rejection -- and their UNITS

SBC's validity requires simulating from exactly the priors the fit uses.

- **Tight**: `a_0 ~ N(0, 5)` would occasionally hand you exp(10)-rate
  simulations. Use SBC-specific tight priors (e.g. `a_0` in a range giving
  ~50-500 events) -- identically on both sides.
- **No post-hoc rejection**: rejecting oversized simulations after the fact
  distorts the ranks and is not an option. Bound the event count through
  the prior itself.

### Unit annotation block (REQUIRED in the harness config)

The harness config knowingly mixes two unit systems; it must say so inline.
Post-realification (see `CLAUDE.md` "Unit contract" and
`docs/boundary_and_window_semantics.md`):

| Quantity | Units | Notes |
|---|---|---|
| `sigmax_2` | SQUARED REAL units of the input X/Y columns | real-unit trigger contract; prior must be restated in these units |
| `spatial_window` (ws) | REAL length (real-space square, per-axis box) | same contract |
| `beta` | INTERNAL rescaled-time units (data time -> [0, 50]) | temporal conversion layer is Phase 3 |
| `window` | INTERNAL rescaled-time units | rule of thumb `window >= 5*beta` |
| `a_0`, background fields | internal-unit objects | background contract unchanged |

On a unit-square domain the real and internal spatial units coincide, so a
unit-box stage config is numerically unchanged -- but the annotation must
still be present, because the same config transplanted to a real domain is
NOT unit-neutral.

### Finite-ws rule across the prior support

If a finite-`spatial_window` stage is included, choose the `sigmax_2` prior
so that `ws >= 4 * sqrt(sigmax_2)` (both in REAL units; the documented rule
of thumb in the `Hawkes_Model` docstring) holds across the prior's
support -- e.g. for an upper prior quantile q of `sigmax_2`, require
`ws >= 4 * sqrt(q)`. This dovetails with the small-kernel requirement in
the caveats below: the same concentrated `sigmax_2` prior serves both.

## Scope and pre-registered caveats

Scope: **rectangular domain, `spatial_window=None`** -- the regime where
Phase 2b made conservation structural (every likelihood/simulator row in
`docs/boundary_and_window_semantics.md` is exact there).

Two known small effects go on record BEFORE the first run:

1. **Out-of-domain-parenting second-order gap**: kept negligible by a
   `sigmax_2` prior concentrated on small kernels.
2. **Unseeded background draw**: irrelevant for SBC -- SBC needs no
   per-replicate reproducibility, only correct distributions.

## Mechanics

- R ~ 100-200 replicates; L = 127 posterior draws after ESS-qualified uniform
  thinning (Talts et al.); uniformity judged by ECDF envelope, not eyeballed
  histograms. **Stage 1 complete** at harness tip `43556b4` (R=600 adaptive
  run, `primary_pass`; results in `refactor-patches/baselines-2026-07/sbc_stage1_adaptive/`).
- The prior-predictive budget check uses a distinct master seed from the real
  run. Inspecting and gating on the exact real-run simulations would select
  the SBC ensemble by event count.
- Stage-1 fits run a few minutes each on CPU -- an overnight job. The
  harness MUST be resumable: per-replicate results written incrementally,
  restartable mid-run.
- Lives as `scripts/sbc.py` in the repo, with a tiny smoke test (R=2, short
  chains) in the suite so it cannot rot. Real runs happen on the analysis
  machine, not in drafting sessions.
- The fit path itself is exercised by `tests/test_fit_smoke.py` (plain
  Hawkes NUTS end-to-end, marked `slow`), landed together with this
  runbook after a green `scripts/recover_test.py --nuts` reference run.

## Reference baselines and the verify_sim caveat

- Fresh machine-local baselines at the SBC tip live in
  `refactor-patches/baselines-2026-07/` (four-config golden pins including
  the non-square 4:1 discriminator, plus verify_sim output). Store SBC
  results beside them; they are the reference state for any mid-run
  investigation. Reminder: pins and fixed-seed simulations are
  MACHINE-LOCAL -- never compare across machines.
- **verify_sim's configs are all unit-box, so post-realification it cannot
  see a real-unit regression.** Acceptable because the 4:1 non-square pin
  guards the inference side and the (I12) byte-identical cascade test
  guards the simulator side -- but keep this line in mind when reading its
  ALL PASS.
