# Audit coverage map

**Candidate tip:** `c5e48713ec1abd58034a9dfc32f0cb8577ba756f`  
**Iteration:** 3  

| Area/symbol | First audited here? | Methods | Configurations/legs | Result | Residual gap |
|---|---|---|---|---|---|
| Lane A pins (`pin_check_v2`) | iter1 | call-path; artifact compare | 4 rectangle pins | MATCH at Commits B/C tips | pin JSON lacks commit/env (G6) |
| Lane B config matrix | iter1–3 | state-transition; covering array | 9 axes; forced rows | pairwise **1.000**; forced rows kept | G2 save_rslts xfail |
| Lane C TruncatedLogNormal | prior | property suite | TLN | green | package-wide dist suite (G3) |
| Lane D ownership / RNG | iter2–3 | adversarial alias/RNG; RED→GREEN | simulate, prepare, mass table, SVI | B2–B4 closed `6ba2194` | membership single-source (G1) |
| Lane D polygon I11 | iter2–3 | Monte Carlo conservation | defaults + small σ, R=40, 3·se | PASS; standing test `c5e4871` | — |
| Mass-table budget | iter1–3 | numerical reference | install residual vs GL=32 host quad | measured gate `86ca179` | OP-12 derivatives |
| Excitation support kwargs | iter2–3 | signature / TypeError | `build_excitation_support` | silent kwargs removed | — |
| 3f seams (E) | iter1 | inventory | RNG, provenance I/O, membership | RNG seam closed; G1/G2 remain | 3f owners |

**Not claimed as coverage:** Stage 3 SBC; full Cartesian config product; cross-machine pin identity.
