# Stage-3 SBC results (full cox-Hawkes, unit gain)

Harness tip: `5da5784`. MACHINE-LOCAL — never compare ranks across machines.

**R = 200** — top of the runbook's pre-registered R ≈ 100–200 range (no
deviation note needed).

Config identity: stage 3, sp_var_mu=0.0 (unit gain), priors = stage-2
background × stage-1 triggers, defaults warmup 300 / num_samples 508 /
L=127 with Talts-style adaptive ESS retries. Decision rule: every primary
p >= 0.004.

`primary_pass: true` (R=200, L=127); zero divergences; 0/200 ESS retries
(max raw draws 508). See `report.json` / `report.png`.

Cost note: per-fit wall clock is ~quadratic in event count; watch the first
TAIL replicate (n ≳ 900), not just the first ten.
