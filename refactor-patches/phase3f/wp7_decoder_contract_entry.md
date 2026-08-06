# WP7 — decoder contract, identity, gain, provenance: entry

**Class: BP** (`phase3f_work_package_outline.md`).
**Status: PLACEHOLDER.** Seams named, scope not yet specified.

## Seams

Decoder **contract** · **identity** · **gain** · **provenance** — in
`bstpp/vae_functions.py`, `bstpp/decode_fields.py`, and the artifacts under
`bstpp/decoders/`.

The pretrained decoders stand in for GP priors on the three background fields:
`f_t` (temporal, `n_t=50`), `f_a` (seasonal, `n_s=24`), `f_xy` (spatial,
25×25).

## Open items routed here

**None.** No §11 row names WP7.

## Constraints and known limits already in force

- **Provenance is honestly UNKNOWN and must stay that way until it is known.**
  The `.meta.txt` sidecar beside the seasonal decoder carries UNKNOWN
  provenance fields, and `AGENTS.md` states the rule directly: *never fill them
  with invented values.* WP7 is where they are either established or left
  UNKNOWN with the reason — not where they are made plausible.
- **Gain has a named, already-recorded gap.** `sp_var_mu` is a *fixed*
  log-amplitude multiplier on the spatial decoder output, and `main.py:290`
  records the intended follow-up in the docstring: *"A sampled amplitude (and a
  matching knob for the seasonal field, which currently has none) is planned
  follow-up work."* So the spatial field has a fixed knob and **the seasonal
  field has none at all**.
- **Coverage limit, environment-dependent.** Cox/LGCP tests **skip** if the
  seasonal decoder artifact `bstpp/decoders/decoder_1d_T24_circ_small_l8` is
  missing. Decoder-path coverage is therefore conditional on the artifact being
  present, and a green suite on a machine without it says less than a green
  suite on one with it.
- **Custom-decoder support and provenance metadata are expected here** (round
  brief, A-43). Neither exists; the expectation is recorded as an expectation.

## Adjacencies — subject matter, NOT routings

- **`N_T=50`, `N_S=24`, `N_XY=25` are decoder-pinned constants** in
  `preparation.py`, and `main.py` also writes the VAE dimensions
  (`hidden_dim*`, `z_dim*`) as bare literals. The literal-vs-constant half of
  that is **routed to WP10**; whether the decoder dimensions should be
  *configuration* rather than pinned constants is question 2 below.

## Questions this entry leaves open

1. **What "identity" means for a decoder.** The seam names it; nothing in the
   repository defines it. Candidate readings — a hash of the artifact, a
   declared architecture identity, a trained-distribution identity — are not
   enumerated here because choosing among them is a decision.
2. **Whether the decoder dimensions become configuration or stay
   decoder-pinned constants.** They are currently constants with a comment
   saying they "become real configuration only with the 3f decoder contract" —
   which names WP7 without deciding it.
3. **Whether the seasonal field gets an amplitude knob**, and whether adding
   one is BP. The outline says WP7 is BP; a sampled amplitude adds a sample
   site, which is not obviously behaviour-preserving. Named, not resolved.
4. **Scope and sequencing.** Not specified, not invented.
