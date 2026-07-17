# Quarantined: fixed-length stage-1 SBC abort (do not pool)

Incomplete run under fixed `num_samples=508` (pre-adaptive harness): aborted at
replicate 44 when minimum primary quantile ESS fell below 0.95×L
(`alpha` ESS ≈ 91.7). Replicates r=0…43 are on disk; **do not resume, extend,
or pool ranks** with `results/sbc_stage1_adaptive/` or any other SBC directory.

Kept as evidence that motivated the Talts-style adaptive chain-length change
(commit `43556b4`). MACHINE-LOCAL.
