# READY FOR 3F

**Candidate tip:** `c5e48713ec1abd58034a9dfc32f0cb8577ba756f` (`refactor`)  
**Iteration:** 3 of 3  
**Environment:** jax==0.4.23, numpyro==0.15.0, numpy==1.26.4, scipy==1.11.4, geopandas==1.1.3, `jax_enable_x64=False`, cpu  

This verdict is against the preregistered §9 entry criterion and the coverage declared in `audit_coverage_map.md`. It does **not** claim the package is defect-free. Stage 3 SBC at the Phase 3 tip remains a separate mandatory exit criterion — this closeout is not evidence toward it.

## §9 conditions

| Condition | Status |
|---|---|
| No open blocker under §5 | **MET** — B1–B6 closed |
| Every preceding finding has exact closeout record | **MET** — ledger rows with SHAs / RED evidence |
| Every blocker repair has focused RED→GREEN regression with correct class | **MET** — Commit A RED at `143d219`; B/C CF; E test |
| Lane B matrix passes | **MET** — 9-axis pairwise 1.000 + forced rows |
| §8 gates pass on one frozen candidate tip | **MET** — battery below |
| Decisions needed for 3f configuration/sequencing settled | **MET** — ownership/RNG/budget settled; G1/G2 owned by 3f |
| Every residual gap has owner/rationale/review date | **MET** — G1–G3, G6–G8 |
| Coverage map names unaudited areas | **MET** — `audit_coverage_map.md` |

## Residual gaps (nonblocking)

| ID | Owner | Rationale |
|---|---|---|
| G1 membership dual predicate | 3f | seam consolidation |
| G2 save_rslts provenance | 3f | A-21 I/O contract |
| G3 package-wide dist properties | 3g | verification |
| G6 pin identity stamps | 3g | gate hygiene |
| G7 OP-8/12, C6 | 3f/3g | register open items |
| G8 Part I OP prose | 3g | Part II governs |

## §8 battery (tip `c5e4871`)

| Gate | Result |
|---|---|
| Focused ownership / Lane B / polygon mass / smoke | 82 passed, 1 xfailed, EXIT:0 |
| Full suite `tests/` | 530 passed, 2 skipped, 1 xfailed, EXIT:0 |
| Ruff on touched production/tests | All checks passed |
| Pins vs `baselines-2026-07/pins.json` | `PIN_DIFFS 0 MATCH`, EXIT:0 |
| `jax_enable_x64` | False before/after |
| Polygon I11 standing | 2 passed (defaults + small σ), EXIT:0 |
| Conditional SBC escalation | **n/a** — no model-specific likelihood path change requiring Stage 1/2 rerun; Stage 3 remains separate mandatory exit |

Env on every result: jax==0.4.23, numpyro==0.15.0, numpy==1.26.4 (<2), scipy==1.11.4 (<1.13), geopandas==1.1.3 (≥1.0), jax_enable_x64=False.

## Declared coverage

Gates establish: ownership boundary (B2–B4), measured mass-table budget (B5), removal of silent install kwargs (B6), Lane B pairwise completeness, polygon I11 at defaults and small σ under 3·se / R=40, rectangular pins MATCH, full suite green at tip. They do **not** establish Stage 3 SBC calibration or absence of unaudited defects outside the declared map.
