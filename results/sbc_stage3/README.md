**SBC Stage 3 (full Cox-Hawkes) — completion record**

*Provenance.* Branch `refactor`, harness tip `5da5784`, run completed
2026-07-18 (provenance date `2026-07-18 00:40:17`), analysis machine
(jax 0.4.23 / numpyro 0.15.0). Harness: `scripts/sbc.py`, config identity
`9d8fbbb9774916dfefecadb81dd4abc032b92bef430082f70d9e6942c6ed7e95`.
Pre-registration: `docs/sbc_runbook.md`, Stage 3 section.

*Design.* R = 200 replicates — exactly the top of the pre-registered
100–200 range; **no deviation note applies** (unlike the two R=600
runs). L = 127 ranked draws; 1 chain, warmup 300, raw-draw cap 508,
thin stride 4, Talts-style adaptive lengthening armed. Twelve PRIMARY
functionals (`alpha`, `beta`, `sigmax_2`, `log_background`,
`log_intensity_p0–p4`, `z_temporal_0`, `z_seasonal_0`, `z_spatial_0`);
SUPPLEMENTARY `a_0` and `exc_share` (ranked and reported, never gating).
Decision rule: PASS iff every primary mc_p_value ≥ 0.004 (Bonferroni
family-wise false-alarm ≤ 4.8% over 12).

*Outcome.* **PASS.** Minimum primary p = 0.092 (`z_temporal_0`,
sup-stat 0.084) — unremarkable as the smallest of twelve correlated
p-values; its rank histogram shows no U- or dome-shape. All other
primaries p ≥ 0.175. Zero ties on every functional.

*Sampling.* Zero divergent transitions across all 200 full Cox-Hawkes
NUTS fits (divergent replicates are NOT dropped — pre-registered; a
material rate is a finding to investigate, not to filter). Zero
ESS retries; `max_raw_draws` = 508, i.e. no replicate needed
lengthening — the pilot-based chain-length prediction held at scale.
Minimum quantile ESS over replicates: tightest primary `alpha` at
141.15, tightest overall `exc_share` (supplementary) at 134.21, vs the
120.65 gate — the visible trace of background/excitation competition,
present as expected and operationally immaterial.

*Scope.* Unit gain (`UNIT_GAIN_SP_VAR_MU = 0.0`), rectangular domain,
`spatial_window=None`, NUTS as the instrument. PASS means no calibration
deviation was detected at this resolution; it is not proof the
implementation is correct, and it is not evidence of polygon-mode
calibration.

*Program status.* This closes the staged SBC program: stages 1–3 all
PASS under NUTS at R = {600, 600, 200}. Together with the identity suite
and the golden pins, coverage now exists at every level: atoms
individually, inference path bitwise, joint distributionally.
