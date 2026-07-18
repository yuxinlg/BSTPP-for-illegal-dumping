# Stage-2 SBC results (LGCP, unit gain)

Harness tip: `30734f1`. MACHINE-LOCAL — never compare ranks across machines.

**R = 600** — deliberate upward deviation from the runbook's pre-registered
R ≈ 100–200 (and from the harness prompt's R=200 default), decided **before**
the run for power / resolution only (validity unchanged). Same rationale as
Stage 1's archived R=600 adaptive run.

Config identity: stage 2, sp_var_mu=0.0 (unit gain), defaults warmup 300 /
num_samples 508 / L=127 with Talts-style adaptive ESS retries.

`primary_pass: true` (R=600, L=127); zero divergences; 0/600 ESS retries
(max raw draws 508). See `report.json` / `report.png`.
