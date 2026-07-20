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
3. **Stage 3: full cox-Hawkes** -- the composition; pre-registered below
   from measurement, after green stages 1 and 2.

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

## Stage 3 pre-registration: full Cox-Hawkes

Stage 3 is the composition test. Stages 1 and 2 each validated a leg in
isolation at R=600; stage 3 fits `Hawkes_Model(cox_background=True)` --
the LGCP background plus self-excitation in one likelihood -- so a failure
localizes to the composition terms: the additivity of the background and
excitation compensators, the branching simulator over inhomogeneous
parents, and the joint posterior geometry where background flexibility and
excitation compete for clustered mass.

### Priors and budget

The stage-3 prior is **stage 2's background prior times stage 1's trigger
priors**: `a_0 ~ N(0.4, 0.3)` (stage 2) with `alpha ~ Beta(2, 6)`,
`beta ~ LogNormal(0, 0.5)`, `sigmax_2 ~ LogNormal(log 0.005, 0.5)`
(stage 1), composed structurally from the two stage prior functions in the
harness. This is deliberately not described as the product of both complete
stage priors: stage 1's `a_0` center is unused, and the `z` priors remain
the model's hardcoded standard normals on both truth and fit sides.

Budget band: unchanged from stage 2 -- 20-2000 events, at most 5% in
either tail, zero-event draws forbidden. The excitation cascade lifts the
stage-2 count distribution by roughly its expected multiplier
`E[1/(1-alpha)] = 1.4`, which the band absorbs without moving `a_0`.
Measured at a drafting-side seed (150 draws): count quantiles
(0/5/25/50/75/95/100) = 7/35/88/170/307/931/1642, 2.7% below 20, 0%
above 2000, no zero-event draws. As always, the pre-registered
`check --stage 3` at the distinct check seed is the record, and the real
run is never conditioned on it.

### Unit gain, inherited

Stage 3 runs at the same unit spatial gain as stage 2 (`sp_var_mu = 0.0`,
recorded in the config identity). The amplitude decision is a property of
the fields that the cascade only multiplies; the production-gain
seven-order count spread is unchanged by excitation. The scope statement
carries over verbatim: stage 3 validates end-to-end **unit-gain
composition** and NUTS inversion; it does not validate posterior geometry,
numerical stability, or calibration under the production-gain prior at
`sp_var_mu = 2.0`. In the harness the shared decision is now named by one
constant (`UNIT_GAIN_SP_VAR_MU`), replacing the stage-2-scoped name; the
identity key and value are unchanged, so archived stage-2 results report
identically.

### Ranked targets and decision rule

Twelve targets are **PRIMARY** -- the union of the stage-1 and stage-2
primaries:

1. `alpha`, `beta`, `sigmax_2`, ranked directly;
2. `log_background = log(Itot_txy - Itot_excite)` -- the stage-1 formula,
   with both terms field-dependent and live for the first time;
3. pointwise log **background** intensity
   `a_0 + f_t[i] + f_a[season_idx_of_t[i]] + f_xy[j]` at the **same five
   pre-registered spacetime grid cells as stage 2** (a functional of the
   latents alone; identical cells make the pointwise histograms directly
   comparable across stages);
4. the first component of each decoder vector.

**SUPPLEMENTARY** (ranked and ESS-reported, never gating): `a_0` -- the
integral-preserving tilt direction against the fields, hypothetical in
stage 1, actually exists here -- and `exc_share = Itot_excite/Itot_txy`,
the decomposition's summary diagnostic, kept supplementary per the stage-1
precedent.

Design note on pointwise **total** intensity: a functional adding the
excitation term at a cell would be a valid data-dependent rank target and
would exercise the composition pointwise. It is omitted as a design
choice -- its information overlaps substantially with the ranked scalars,
the background points, and `exc_share` -- not because it would be
information-free.

Stage 3 passes if every primary Monte Carlo ECDF p-value is at least
**0.004**. The Bonferroni family-wise false-alarm bound is at most 4.8%
across twelve targets, continuing the bound invariant across stages (4%,
4.5%, 4.8%); the per-test threshold has no independent meaning, and the
correlated targets again do not justify an independence calculation.

### Composition identity (tested, with a float caveat)

The likelihood defines `Itot_txy = Itot_excite + Itot_txy_back` at the
trace level. The fast test asserts `Itot_txy - Itot_excite ==
Itot_txy_back` to one-ulp float32 tolerance (rtol 1e-6): `Itot_txy` is
stored as the *rounded* float32 sum, so bitwise equality of the
subtraction fails in roughly a quarter of prior draws at ~1e-7 relative
error. That is a float fact, an order of magnitude below any structural
composition defect the assertion is there to catch.

### Computational profile and an operational amendment

The pairs term makes per-fit cost roughly quadratic in the event count, so
unlike stages 1-2 the **tail replicates dominate wall clock**
(drafting-machine pilots: 21 s at n=288, 226 s at n=850, 967 s at n=1642;
machine-local, indicative only). Amendment to the watch-the-first-replicates
advice: early replicates do not bound cost, because cost correlates with
`n` -- watch the first **tail** replicate (n above ~900) specifically
before trusting an overnight extrapolation.

### Pilot evidence

Drafting-environment pilots at production chain settings and the real
stage-3 priors: four warmup-300 fits spanning n in {7, 288, 850, 1642} --
zero divergences, minimum primary quantile ESS 174-263 against the 120.65
gate, no retries. A warmup-500 comparison at n=288 was indistinguishable
(minESS 225 vs 222, zero divergences). Default warmup 300 stands for
stage 3. The excitation/field confounding concern does not materialize at
unit gain: `alpha`'s ESS is as healthy as the scalars were in stage 1.

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
`ws >= 4 * sqrt(q)`. The concentrated `sigmax_2` prior originally also
served the out-of-domain-parenting caveat below; that caveat is now
retired for the rectangular scope, and the prior stands on this rule and
on NUTS geometry (bounded away from zero).

## Scope and pre-registered caveats

Scope: **rectangular domain, `spatial_window=None`** -- the regime where
Phase 2b made conservation structural (every likelihood/simulator row in
`docs/boundary_and_window_semantics.md` is exact there).

Effects on record:

1. **RETIRED (stage-3 pre-registration): out-of-domain-parenting
   second-order gap.** Registered before stage 1 as "kept negligible by a
   `sigmax_2` prior concentrated on small kernels"; retired for the
   rectangular `spatial_window=None` scope because the gap does not exist
   there at all. Per `docs/boundary_and_window_semantics.md` rows 5 and 7,
   offspring are discarded w.r.t. the bounding rectangle **before
   parenting** (Prop 1.1(ii)), exactly matching the excitation
   compensator's per-axis erf charge on rectangles -- no out-of-domain
   parent is ever created in this regime. The stage-1 registration was
   conservative; post-Phase-2c the exactness is structural. The
   registration stays in this list as the record of its own retirement.
2. **Unseeded background draw**: irrelevant for SBC -- SBC needs no
   per-replicate reproducibility, only correct distributions.

## Stage 2p pre-registration: polygon-domain LGCP background (post-3c)

Registered 2026-07-20, BEFORE any harness implementation or run, per the
team decision discharging Phase 3c's conditional SBC trigger (doc §12;
`refactor-patches/phase3c/rebaseline_record.md`): 3c changed the LGCP
background path on polygon domains. The decision taken:

- **No rectangular stage-2 rerun.** The archived rectangular stage 2
  (R=600 PASS at `30734f1`/`23cdf94`) is RETAINED on the bit-identity
  evidence: after every 3c commit the four golden pin configs are
  bit-identical and the array-rectangle support is constructed as the
  exact legacy constants, so a rerun would re-verify an unchanged regime.
- **Stage 2p is added**: the same stage-2 design run on a polygon domain,
  where the 3c semantics are live.

**Scope statement.** Stage 2p validates the polygon **background** path
only: the clipped support `|C_c ∩ A|` in the compensator (`Itot_txy`),
background sampling on the clipped geometries, and event-to-field-cell
membership through the same support object -- at unit gain
(`sp_var_mu = 0.0`), inherited from stage 2 with its scope caveats. It
does NOT claim calibration of the polygon excitation support mode
introduced in 3d (LGCP has no excitation term), nor of the
production-gain prior, nor of covariate layers (none are supplied).

### Domain (pre-registered)

A single convex octagon: the unit square with four generic
(non-grid-aligned, asymmetric) corner cuts, vertices counterclockwise

    (0.22, 0), (0.79, 0), (1, 0.17), (1, 0.77),
    (0.86, 1), (0.24, 1), (0, 0.81), (0, 0.13)

supplied as a GeoDataFrame; bounding rectangle = the unit square, so the
internal-unit geometry matches stage 2. Exact area
|A| = 1 − (0.0143 + 0.01785 + 0.0161 + 0.0228) = **0.928950**.
Rationale: the four diagonal edges cut ~30+ boundary cells of the 25×25
grid into generic partial intersections and leave several corner cells
fully outside (dropped from the support), so every 3c mechanism is live;
all five pre-registered stage-2 pointwise grid cells remain strictly
interior (verified in the smoke test), so the pointwise functionals stay
directly comparable with the archived stages. Placeholder/generator
events are rejection-sampled inside the polygon (the generator's grids
are data-independent, as before).

### Everything else: stage 2 verbatim

Priors (`a_0 ~ N(0.4, 0.3)`, z hardcoded N(0, I)), unit gain, R = 600,
warmup 300, initial 508 raw draws with the adaptive ESS ladder capped at
4064, L = 127, min ESS ratio 0.95, the nine PRIMARY targets and
supplementary `a_0`, and the decision rule: **stage 2p passes iff every
primary Monte Carlo ECDF p-value is at least 0.005** (same Bonferroni
family-wise bound ≤ 4.5% across nine correlated targets, no independence
claim). Budget band unchanged (20-2000, ≤5% per tail, zero-event draws
fatal); the count distribution scales by |A| ≈ 0.93, which the band
absorbs; the pre-registered `check --stage 2p` at the distinct check
seed (`PRIOR_CHECK_MASTER_SEED`) is the record, and the real run
(master seed 0) is never conditioned on it. Config identity records the
polygon vertices in place of the rectangle, under stage id `"2p"` and
out_dir `results/sbc_stage2p`.

## Mechanics

- R ~ 100-200 replicates; L = 127 posterior draws after ESS-qualified uniform
  thinning (Talts et al.); uniformity judged by ECDF envelope, not eyeballed
  histograms. **Stage 1 complete** at harness tip `43556b4` (R=600 adaptive
  run, `primary_pass`; results in `refactor-patches/baselines-2026-07/sbc_stage1_adaptive/`).
  **Stage 2 complete** at harness tip `30734f1` (R=600 unit-gain LGCP,
  `primary_pass`; results in `refactor-patches/baselines-2026-07/sbc_stage2/`).
  **Stage 3 COMPLETE** — R=200 PASS (min primary p = 0.092, zero
  divergences, zero ESS retries); results archived at
  `refactor-patches/baselines-2026-07/sbc_stage3/`.
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
