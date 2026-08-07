# Declared seam sets (D-48, amended A-52)

**Authority:** D-48 as amended at A-52, clause 3 — *a declared seam set is an
opening precondition for a work package*. This file is where a declaration
lives. A package whose seam set is not declared here has not met the
precondition, whatever else is true of it.

**Coverage unit:** statements (D-48, declared A-51, held fixed).

**Comparability:** ρ_w is comparable only within a fixed tree state (D-48
clause 2, A-52). Statement counts move under extraction, inlining and
reformatting, so a ρ_w from one tree may not be differenced against a ρ_w from
another. Every recorded ρ_w names the tip it was measured at.

---

## S_WP2 — `ModelConfig`, `PriorConfig`

**Declared:** A-52. **Retroactive**, under D-52's enumerated-retroactivity
rule: WP2 opened at A-45, before D-48 existed, and this declaration is what
discharges the debt A-51 recorded. **This is the only seam set declared
retroactively.** The other nine declare at opening.

**Status at `96c01ba`: the set is EMPTY of extant code.**

```
git grep -n "class ModelConfig\|class PriorConfig" -- bstpp/     ->  (no output)
```

Neither class exists. WP2's deliverable is to *create* them, so |S_WP2| = 0 and
**ρ_WP2 = 0/0 is undefined, not zero** (D-48 clause 1, A-52). The `ρ_w = 0`
vacuity clause answers a different question — a seam set that exists and is
unreached — and does not apply.

### Intended membership, so the set is a declaration and not a blank

When the objects land, S_WP2 is the statements of:

| member | quantity it owns | state at `96c01ba` |
|---|---|---|
| `ModelConfig` | `cox_background`, `sp_var_mu`, `standardize_cov`, `model` selection | **does not exist**; quantities currently live as bare `Point_Process_Model.__init__` parameters |
| `PriorConfig` | the sampled-site priors: `a_0`, `alpha`, `beta`, `sigmax_2` | **does not exist**; currently `**priors` kwargs |

**Not in S_WP2**, with the owner named rather than dropped: `NumericalConfig`
fields (WP1); `ModelData` quantities — events, domain, horizon, CRS, seasonal
offset (WP4); `excitation_support` / `min_sigma` / `max_sigma` / `mass_table`
(WP5); `data_contracts` and `spatial_cov_crs` (WP8).

### What this declaration does and does not buy

**Does:** discharge WP2's D-48 opening precondition, retroactively and by name,
so the debt A-51 recorded is closed rather than carried.

**Does not:** make ρ_WP2 measurable. That needs members. **The enforcement
commits landing against WP2 (CI-9 at A-45, CI-10 at A-50, CI-7 next) act on the
quantities S_WP2 will own, at their current pre-object sites** — which is why
they are WP2 work despite the objects being absent, and why no pin gate over
them is being read as evidence about `ModelConfig`.

---

## S_WP1, S_WP3 … S_WP10 — not declared

**Declared at opening**, per D-48 clause 3. WP1 is landed and its seam set was
never declared; that is a pre-D-48 fact recorded under D-52's grandfathering
clause, not a debt this file discharges.
