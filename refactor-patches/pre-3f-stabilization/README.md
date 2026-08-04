# Pre-3f stabilization audit

**Status:** iteration 3 complete — verdict **READY FOR 3F** (`readiness_report.md`)  
**Governing spec:** repository-root `pre3f_stabilization_audit_protocol.md`  
**Claim supported:** whether semantics Phase 3f will preserve are sound enough to restructure — not that the package is defect-free. Stage 3 SBC remains a separate mandatory exit criterion.

## Candidate tip (iteration 3)

| Field | Value |
|---|---|
| Branch | `refactor` |
| `HEAD` | `c5e48713ec1abd58034a9dfc32f0cb8577ba756f` |
| Subject | `test(pre-3f): standing polygon I11 conservation at defaults and small sigma` |

## Environment

| Field | Value |
|---|---|
| Audit date | 2026-08-04 |
| Python | 3.12.13 (conda `illegal-dumping`) |
| jax / jaxlib | 0.4.23 / 0.4.23 |
| numpyro | 0.15.0 |
| numpy | 1.26.4 (`<2`) |
| scipy | 1.11.4 (`<1.13`) |
| geopandas | 1.1.3 (`>=1.0`) |
| `jax_enable_x64` | `False` |
| platform | `cpu` |

## Deliverables (§10)

| File | Role |
|---|---|
| `traceability_matrix.md` | Register × evidence |
| `findings_ledger.md` | B1–B6 closed; G* residuals |
| `audit_coverage_map.md` | Lane coverage + unaudited silence list |
| `readiness_report.md` | Single headline + §9 table + §8 battery |
| `pin_path_map.md` | Four-config pin reachability |

Also: `docs/config_matrix.md` (Lane B generation); `commitB_rebaseline.md` (simulate rng require); `commitC_resolution.md` (measured budget).
