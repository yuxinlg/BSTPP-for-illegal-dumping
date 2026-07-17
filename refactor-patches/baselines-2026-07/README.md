# Machine-local baselines at the pre-SBC tip (2026-07)

Reference state for any mid-run SBC investigation (see
`docs/sbc_runbook.md`). Captured on Terhi's machine; golden pins and
fixed-seed simulations are MACHINE-LOCAL artifacts -- never compare these
values against runs from another machine.

## Provenance

- Commit: `a5b91d5adfd69af7b148bbadd2129f9c0be7a170` (branch `refactor`,
  post Phase 2d)
- Date captured: 2026-07-16
- Stack: jax 0.4.23 (CPU, `JAX_PLATFORM_NAME=cpu`), numpy 1.26.4,
  numpyro 0.15.0, geopandas 1.1.3, shapely 2.1.2
- Python: conda env `illegal-dumping`

## Contents

- `pins.json` -- four-config golden pins from
  `refactor-patches/pin_check_v2.py` (configs: `cox_hawkes`, `hawkes`,
  `hawkes_nonsquare_4to1`, `lgcp`). The non-square 4:1 config is the
  real-unit trigger-contract discriminator on the inference side.
- `verify_sim_output.txt` -- output of
  `refactor-patches/phase2b/verify_sim.py` at this tip (ALL PASS).
  Caveat (pre-registered in the runbook): verify_sim's configs are all
  unit-box, so it cannot detect a real-unit regression; the 4:1 pin and
  the (I12) byte-identical cascade test cover that axis.

## Reproduce

```
JAX_PLATFORM_NAME=cpu python refactor-patches/pin_check_v2.py . > pins.json
JAX_PLATFORM_NAME=cpu python refactor-patches/phase2b/verify_sim.py . > verify_sim_output.txt
```
