# Pin / gate path map (Lane A)

**Candidate tip:** `c5e48713ec1abd58034a9dfc32f0cb8577ba756f`  
**Evidence marks:** verified = executed/read here; reported = prior artifact; inferred = document-only.

**Pin generator:** `refactor-patches/pin_check_v2.py`  
**Baseline artifact inspected (not regenerated):** `refactor-patches/baselines-2026-07/pins.json` (README provenance commit `a5b91d5`, 2026-07-16)  
**Existing tip candidates inspected:** all `results/_pins_*_candidate.json` → **PIN_DIFFS 0 MATCH** vs baseline (utf-8-sig tolerant compare; verified 2026-08-03).

## Per-config map

| Pin/config | Public entry | Resolved configuration | Production symbols reached | Branches not reached | Assertions and discriminators | Provenance | Artifact predates a correction? |
|---|---|---|---|---|---|---|---|
| `hawkes` | `Hawkes_Model(DATA, A_unit, T_DAYS, cox_background=False, **PRIORS)` | plain Hawkes; axis-aligned array domain → rectangle support default; `Temporal_Exponential` + `Spatial_Symmetric_Gaussian`; unit box `[[0,1],[0,1]]`; 60 events; fixed float32 latents | constructor → `prepare_domain` / partitions → `args` assembly → pairs → `spatiotemporal_hawkes_model` → event term + rectangle excitation compensator + background → `loglik` + `jax.value_and_grad` over `a_0,alpha,beta,sigmax_2` | polygon support; held-out scoring; `set_window`; custom triggers; covariates / CRS / `standardize_cov`; mass tables; power-law temporal; LGCP fields; simulate; MCMC/SVI | `repr(float(loglik))` and `repr(grad.tolist())` exact string equality on compare | Baseline README commit `a5b91d5`; pin JSON itself has **no** commit/env/hash fields (**verified**) | Predates many CF/SC commits, but all inspected candidates still **MATCH** bit-identical on these four keys (**verified**). No value-changing correction on this path observed → §5.6 rebaseline **not** required for gating, but provenance metadata is stale (**NONBLOCKING-3G**). |
| `cox_hawkes` | `Hawkes_Model(..., cox_background="cox", **PRIORS)` | Cox–Hawkes; same unit rectangle; latents include `z_temporal/z_seasonal/z_spatial` from seeded trace | as above + VAE decode fields `f_t/f_a/f_xy` + Cox background integral | same as hawkes, plus plain-Hawkes covariate `w` path | same string pins + z-grads | same | same |
| `lgcp` | `LGCP_Model(DATA, A_unit, T_DAYS, a_0=...)` | LGCP; no excitation | LGCP model function; temporal/seasonal/spatial fields; no `Itot_excite` | all Hawkes excitation; polygon; cutoffs/set_window; covariates | loglik + z and `a_0` grads | same | same |
| `hawkes_nonsquare_4to1` | `Hawkes_Model(DATA_NS, A_NS=[[0,4],[0,1]], ...)` | plain Hawkes; **4:1** real-unit discriminator | same as `hawkes` plus nonsquare `axis_scales` / real-unit spatial trigger Jacobian | same as hawkes | **only** pin that can see internal-vs-real spatial trigger regressions (I12 companion) | same | same |

## Gate / generator audit (verified)

| Check | Result |
|---|---|
| Configuration identity in artifact | **Absent** — JSON is values only; config is implicit in generator source |
| Environment identity in artifact | **Absent** — stack recorded only in baselines README |
| Assertion strength | Exact `repr` string match of float32 loglik/grads — strong for bit-identity; silent if a site is dropped from the pin dict (compare only intersects keys present in both when using naive equality of nested dicts; current compare scripts iterate union of keys) |
| Unused/malformed state → pass? | Generator does not assert site-name completeness against a schema; a refactor that stops emitting a grad key would look like a DIFF (good) only if the baseline still has it. Polygon/setter bugs cannot fail this gate. |
| Smoke / confirmation | `tests/test_smoke.py` family traces; confirmation logs under `results/_confirmation_8580364.txt` (**reported**). Not re-run as full ritual here. |

## A. Production branches **no pin reaches**

1. Polygon `excitation_support` + Hermite mass-table install / compensator  
2. Held-out `log_expected_likelihood` (rectangle or polygon `mass_table=`)  
3. `set_window` / `_UNSET` sentinel / transactional rollback / untouched-axis provenance  
4. Custom temporal/spatial triggers; `Temporal_Power_Law`  
5. GeoDataFrame domains, CRS contract, `spatial_cov_crs`, covariate gaps/coverage  
6. `standardize_cov` (`None` / `"domain_area"`)  
7. `prepare_polygon_mass_table` builder settings (`panel_h_m`, `gl_order`)  
8. Public `simulate` / offspring cascade / union filter  
9. SVI / MCMC / `save_rslts` / plotting helpers  
10. LGCP `set_window` rejection path  
11. Prior truncation `truncate_sigmax_2_prior` / `TruncatedLogNormal`  
12. Data-contract reject vs report modes beyond constructor defaults used by pins  

## B. Defects a trace-equivalence gate **would preserve**

Any defect confined to the branches in (A), including:

- Hidden-default `panel_h_m`/`gl_order` validation vs prepared tables (**B1**, this pass — verified)  
- Held-out silent rebuild / training-table reuse (historically fixed; pins would not catch regressions)  
- `set_window` provenance corruption  
- Union vs row-sum geometry in simulate / background sampling  
- Membership-predicate drift between `validate_events` and `covers_xy`  
- Packaging / dependency metadata drift  
- `save_rslts` omitting cutoff/config provenance  
- Non-Gaussian polygon trigger acceptance  

Pins **would** catch: rectangle-path loglik/grad changes, including the nonsquare real-unit contract on `hawkes_nonsquare_4to1`.
