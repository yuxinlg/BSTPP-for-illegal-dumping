# Commit prompt: land tests/test_identities.py on yuxinlg/audit-fixes

## Context

Repo: `yuxinlg/BSTPP-for-illegal-dumping`, branch off `audit-fixes` (HEAD `4e6d1c8`).
Read `CLAUDE.md` in the repo root first and follow it; in particular: show the
diff before committing, never run with `--dangerously-skip-permissions`, and do
not edit a test to make it pass.

This commit adds `tests/test_identities.py`: pytest conversions of the
machine-checkable identities (I4), (I8)–(I11) from the BSTPP guide §2.6. The
identities (I1)–(I3), (I5)–(I7) are already covered by the existing 19 tests
(the file's docstring carries the full coverage map). These tests are behavior
pins for the upcoming architecture rewrite: they test the mathematical contract,
not internals, so they must survive restructuring unchanged.

The file has already been run and mutation-checked externally on the pinned
stack (26/26 pass); your job is to land it and reproduce that verification in
this environment.

## Environment

Per CLAUDE.md:14,22 the stack is deliberately pinned: `jax==0.4.23`,
`jaxlib==0.4.23`, `numpyro==0.15.0`, `numpy<2`, `scipy<1.13`, `geopandas>=1.0`.
Known failure mode if the pins are violated: current JAX fails to unpickle the
decoder artifacts with `TypeError: ShapedArray.__new__() got an unexpected
keyword argument 'named_shape'` — this breaks pre-existing cox/lgcp tests too
and is an environment defect, not a code defect. Verify
`python -c "import jax; print(jax.__version__)"` prints `0.4.23` before
concluding anything from a test run.

## Anchors the new tests depend on (verify before running)

Verify each anchor exists at (or near) the stated location; if any has moved,
locate it and note the drift in your report — do not silently adapt the tests.

- `bstpp/inference_functions.py:199-214` — excitation compensator:
  `temp_part = alpha * t_trig.compute_integral(t_pars, min(T - t_events, win))`,
  `sp_limits` stacked from the `A_` bounds, `Itot_excite`, `Itot_txy`, and the
  single `numpyro.factor("loglik_factor", loglik)`. Test (I11)'s `_compensator`
  helper mirrors exactly these lines.
- `bstpp/inference_functions.py:35-51` — covariate background:
  `b_0 = spatial_cov @ w`, `Itot_txy_back = mu_xyt @ args['cov_area'] * args['T']`,
  events indexed via `args['cov_ind']`. Test (I9) exercises both sides.
- `bstpp/main.py:1785-1800` — `_sim_offspring`, window-consistent thinning
  (`if t_dif[0] > self.args['window']: continue`). Test (I4).
- `bstpp/main.py:~1801-1848` — `simulate()`: internal-to-real rescale and final
  `points.sjoin(self.A[['geometry']])` spatial censoring. Test (I11).
- `bstpp/main.py` — `set_window` (recomputes pairs and sets `args['window']`).
- `bstpp/trigger.py:232-233` — `Temporal_Exponential.compute_integral =
  1 - exp(-dif/beta)`; the (I4) expectation `m = alpha * F_beta(w)` is computed
  through this method so simulator and compensator are tied to one expression.
- `tests/test_smoke.py` — fixture conventions (`DATA`, `A_RECT`, `A_GDF`,
  `PRIORS`, `COV_DATA`, `COV_GDF`, `_loglik_at`) that the new file mirrors.

## Steps

1. Create a working branch off `audit-fixes` (e.g. `identity-tests`).
2. Create `tests/test_identities.py` with EXACTLY the content of the
   accompanying file — byte-for-byte, no reformatting, no "improvements".
3. Run the full suite: `python -m pytest tests/ -q`.
   Expected: **26 passed** (19 existing + 7 new; the two `@needs_decoder` /
   parametrized cases count within the 7). Runtime ~30 s on CPU.
   - If ANY test fails on the clean tree: STOP. Do not modify the test, the
     tolerance, or the code. Report the failure verbatim — a failing identity
     test is a finding about the code or the spec, and I have the domain
     context to adjudicate it.
4. Reproduce the two mutation checks (each: mutate, run, confirm the expected
   FAILURE, then `git checkout bstpp/main.py` to revert; paste both outputs):
   a. In `bstpp/main.py::_sim_offspring`, change the thinning condition to
      `t_dif[0] > 2*self.args['window']`. Then
      `python -m pytest tests/test_identities.py::test_offspring_thinning_matches_compensator_mass -q`
      must FAIL (the hard parent-lag rule fires).
   b. In `bstpp/main.py::_sim_offspring`, change
      `np.random.poisson(lam=par['alpha'])` to
      `np.random.poisson(lam=par['alpha']*1.4)`. Then
      `python -m pytest tests/test_identities.py::test_simulated_count_matches_compensator -q`
      must FAIL (E[n - Lambda] > 0 beyond 5 SE).
   If a mutation does NOT produce the expected failure: STOP and report — the
   pin is not detecting defects and must not be committed as if it were.
5. Confirm the working tree is clean except for the new test file
   (`git status`, `git diff` — show me both), then run the full suite once
   more and confirm 26 passed.
6. Commit with exactly this message:

   test: machine-checkable identities (I4, I8-I11) from guide sec. 2.6 -- offspring thinning tied to the compensator mass through t_trig.compute_integral (m = alpha*F_beta(w), per-parent cascade expectation m/(1-m) + hard parent-lag rule); alpha=0 reduces cox_hawkes loglik to LGCP at shared latents; window >= T recovers the untruncated loglik (event term + compensator jointly, extending test_window_default); covariate-partition refinement invariance under a uniform 2-way cell split, both standardize_cov settings (event-side cov_ind and compensator-side cov_area); spatial affine unit invariance of the internal representation and loglik; Monte Carlo conservation E[n] = E[Lambda] with per-replicate compensator pairing and a traced Itot_txy cross-check. Verified: 26/26 on the pinned stack (jax 0.4.23 / numpyro 0.15.0); mutation checks confirm detectability (thinning window x2 -> I4 fails; offspring rate x1.4 -> I11 fails). Scoping documented in the module docstring: I8's f_a=0 reduction needs the upstream likelihood (not in-repo); I9 restricted to the covariate partition (field partitions decoder-pinned); I10 restricted to spatial units (time-in-days is the data contract); I11 requires a rectangular domain (guide eq. 27) and notes the out-of-domain-parenting second-order gap in _sim_offspring.

7. Do NOT push or merge to `main` in this session. Landing `audit-fixes` into
   `main` is a separate, explicitly approved step.

## Report back

- pytest output for steps 3, 4a, 4b, and 5 (verbatim tails).
- The verified anchor list, with any line-number drift noted.
- `git log --oneline -1` after the commit.
