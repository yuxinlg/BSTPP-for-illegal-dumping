# `baselines-2026-08-polygon/pins.json` — pin 5's forward baseline (OP-24 / A-47)

**This is a FORWARD baseline. It certifies commits taken after its capture and
says nothing about any commit before it.** It cannot certify `b98e91d` or
`0e78f7d`, and the six-configuration count must not be read as six
configurations' worth of retroactive coverage: four of the six have been pinned
since the 2026-07 baseline, and two were first captured here.

## Why a second baseline exists rather than an extended first one

`refactor-patches/baselines-2026-07/pins.json` is **not extended, and that is a
decision.** Its SHA-256 —
`f2141fd558704057c87c8c20a3f2b9516c8815f232b6eaafc8f71baa0864e7eb` — is quoted
in the provenance block of every historical pin capture, and A-38's retroactive
rescue of twenty-four `PIN_DIFFS 0` claims is an argument about that file's
content. Rewriting it in place would make every one of those captures describe a
file that no longer exists, which is a worse defect than the one a single
canonical baseline avoids.

## What is in it

Six configurations, byte-identical to `results/_a47_pins_candidate.json`
(`sha256 e8ade72f7947ec350abe96c7f0d09c3fa9641940708964beff5c47cda1509e81`):

| key | mode | what it covers |
| --- | --- | --- |
| `hawkes` | rectangle | since 2026-07 |
| `cox_hawkes` | rectangle | since 2026-07 |
| `lgcp` | rectangle | since 2026-07 |
| `hawkes_nonsquare_4to1` | rectangle | since 2026-07; the real-unit spatial contract |
| `hawkes_notched_4to1_polygon_mode` | **polygon** | new at A-47 |
| `hawkes_notched_4to1_rectangle_mode` | rectangle on a **polygon domain** | new at A-47 |

**A baseline always begins as a capture, and this one says so rather than
implying independent derivation.** The 2026-07 canonical baseline began the same
way.

## Pin 5's configuration, as provenance and never as a package default

D-29 forbids a σ-bound default and is not relaxed to make a pin convenient.
Every value below is stated explicitly in `pin_check_v2.py` and echoed to stderr
in the `PIN5_POLYGON_PROVENANCE` block of each capture:

| choice | value | why this value |
| --- | --- | --- |
| domain | notched octagon, bounds `(0, 0, 4, 1)`, area `3.56` | **aspect 4:1** because on a unit box the internal-isotropic and real-isotropic spatial kernels coincide algebraically, so a unit-box polygon pin would reintroduce in the polygon regime exactly the blind spot `hawkes_nonsquare_4to1` exists to close. **Non-rectangular** because against a plain 4:1 box the polygon Hermite mass and the rectangle's analytic box mass agree by construction, and the mode-switch pin would then be two copies of one number. |
| events | 60, `RandomState(0)`, rejection-sampled inside the polygon | the existing 60-point clouds lie partly in the notch, so they cannot be reused |
| `min_sigma` | `0.05` real units | explicit; `sqrt(sigmax_2) = 0.316` at the substituted latent lies inside `[0.05, 0.5]`, so the Hermite interpolant is evaluated inside its knot span rather than at a NaN edge |
| `max_sigma` | `0.5` real units | explicit; truncates the `sigmax_2` prior to `[0.0025, 0.25]` |
| `panel_h_m` | `0.4` | explicit. The shipped default is `20.0` **metres**, about twenty times this domain, and is rejected by the panel/`min_sigma` ratio guard — `0.4` sits exactly at the `MAX_PANEL_TO_MIN_SIGMA_RATIO = 8` ceiling |
| `gl_order` | `16` | the shipped default, taken deliberately so the pin exercises the shipped quadrature |
| excitation modes | **both** `polygon` and `rectangle` | over the same domain, events and σ bounds, so the only difference between the two records is the mode switch — the surface all six items routed to WP5 touch |

## The mass table is part of the pin's identity

`MASS_TABLE_SHA256 =
09b055c20e5b750f6271bda8ca5d12d6362d9908fd805e5bd184416137d11fbf` (SHA-256 over
`log_knots`, `values`, `slopes` as contiguous float64). It is recorded **inside**
the polygon-mode record as `mass_table_sha256`, not only in the stderr block, so
a table rebuilt with different settings reports **DRIFT** rather than moving the
loglik silently (D-27 / A-11 integrity clause). A pin that depends on a
regenerable artifact without pinning it is not reproducible.

## Reproducibility, measured

Two independent process invocations produced **byte-identical** captures —
`sha256 e8ade72f…` both times — and byte-identical mass tables
(`results/_a47_determinism.txt`). **No tolerance was introduced or loosened
anywhere to reach this**; the comparison is the same exact-equality walker
`pin_compare.py` has always used. Golden pins remain machine-local artifacts:
baseline and compare on the same machine.
