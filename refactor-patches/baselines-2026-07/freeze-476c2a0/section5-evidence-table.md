# §5 evidence / disposition table (freeze at 476c2a0)

Behavioral tip audited: `476c2a044780ced66afae12760b53ac75e304fc6`.
Execution environment: detached worktree
`C:\Users\Terhi\Box\BSTPP_terhi\BSTPP-freeze-476c2a0-worktree` (HEAD = tip).
Evidence from code/tests only (not the guide or consolidation document).

| # | § | Claim | Status | Evidence / disposition |
|---|---|---|---|---|
| 1 | 5.1 | Affine time/space rescaling to T̃=50 and unit square via bounding rectangle | **verified (tip)** | `bstpp/main.py:180–209` (args T/S, A_, axis_scales); `_scale_xyt` `bstpp/main.py:555–593` @ 476c2a0 |
| 2 | 5.1 | Seasonal coordinate derived; supplied A validated (I6) | **verified (tip)** | Derivation/validation in `_scale_xyt` `bstpp/main.py:561–570` @ 476c2a0; test `tests/test_smoke.py::test_A_derivation` |
| 3 | 5.1 | NaN covariate values unvalidated | **verified (tip)** — no adversarial test; code gap | Covariate path `bstpp/main.py:384–391` applies mean/var with no `isnan`/`fillna` check @ 476c2a0. **No** named adversarial ingest test exists → non-blocking for freeze; schedule adversarial coverage in **3a** |
| 4 | 5.1 | Events outside polygon A but inside bounding rectangle A_□ silently accepted | **verified (tip)** | Membership is `points.sjoin(comp_grid)` on the rectangle-tiled grid (`bstpp/main.py:582–590`); no containment check against polygon `self.A` @ 476c2a0. Compensator support uses cells ∩ A (`bstpp/main.py:269–272`) while events may index any rectangle cell |
| 5 | 5.2 | No CRS-compatibility enforcement beyond geographic warning | **verified (tip)** | Geographic warning only: `bstpp/main.py:210–240` @ 476c2a0; tests `tests/test_ingestion_contract.py::test_geographic_crs_contract_warning_is_nonvacuous`, `::test_crsless_gdf_falls_back_to_heuristic`. No CRS-equality check between domain and covariates (crs assigned at `bstpp/main.py:371`, `375`) |
| 6 | 5.2 | Malformed geometries unvalidated at ingestion | **evidence gap (non-blocking) → 3a** | Constructor path has no `is_valid` / `make_valid` gate on A or covariates @ 476c2a0. **No** adversarial test. Demote: known untested gap; 3a contract work |
| 7 | 5.3 | Array domain = rectangle; GeoDataFrame = polygon A with envelope A_□ | **verified (tip)** | `bstpp/main.py:192–200`, `269–276` @ 476c2a0 |
| 8 | 5.3 | Compensator limits in `args` are internal unit-square bounds | **verified (tip)** | `args['t_min']/x_min/y_min=0`, `x_max/y_max=1`, `T=50` at `bstpp/main.py:180–187` @ 476c2a0 |
| 9 | 5.5 | Partitions 50 / 24 / 25×25; spatial grid tiles rectangle in real coords; covariate cells data-given | **verified (tip)** | `n_t=50`, `n_s=24`, `n_xy=25` at `bstpp/main.py:245–267`; covariate geometry from data at `bstpp/main.py:350–372` @ 476c2a0 |
| 10 | 5.5 | Partition sizes pinned by pretrained decoder load sites | **verified (tip)** | Loads at `bstpp/main.py:339–348` (`decoder_1d_T50_fixed_ls`, `decoder_1d_T24_circ_small_l8`, `2d_decoder_15_5_large.pkl`); dims in `bstpp/decode_fields.py:9–12` @ 476c2a0 |
| 11 | 5.6 | Common refinement with exact intersection areas | **verified (tip)** | `gpd.overlay(..., how='intersection')` + area normalization `bstpp/main.py:397–400`; integration arrays `bstpp/main.py:411–421` @ 476c2a0 |
| 12 | 5.6 | Covariate gaps silently become zero-valued regions | **inferred → demote non-blocking → 3c** | Support from covariate sjoin (`bstpp/main.py:402`); no gap/coverage validation. **No** gapped-layer construction test at tip. Schedule explicit gap rejection/report in **3c** |
| 13 | 5.6 | Standardization count-weighted (cell rows), not area-weighted | **verified (tip)** | `(X_s-X_s.mean(axis=0))/(X_s.var(axis=0)**0.5)` over covariate rows `bstpp/main.py:388–389` @ 476c2a0 (no area weights) |
| 14 | 5.7 | Single `numpyro.factor` per model (I5) | **verified (tip)** | `tests/test_smoke.py::test_single_factor_site` (asserts factor sites `== ["loglik_factor"]`) |
| 15 | 5.7 | Sampler draws consume same mass atoms as likelihood (I1 / distributional) | **verified (tip)** | I1 seasonal: `tests/test_seasonal_integral.py::test_sim_likelihood_integral_identity`; mass-atom coupling: `tests/test_likelihood_atoms.py::test_spatial_refinement_masses_and_integral_share_one_integrand`, `::test_background_masses_sum_to_the_integral_atoms`; conservation: `tests/test_identities.py::test_simulated_count_matches_compensator` |
| 16 | 5.14 | Decoder loading, shapes, single-sourced gain | **verified (tip)** | Load: `bstpp/main.py:38–52`, `339–348`; gain single-sourced in `bstpp/decode_fields.py:36–50`; tests `tests/test_decode_fields.py::test_decode_fields_match_direct_decoder_application`, `::test_spatial_calibration_gain_contract` |
| 17 | 5.14 | Seasonal decoder artifact identity / provenance | **known provenance gap (UNKNOWN retained)** | Artifact `bstpp/decoders/decoder_1d_T24_circ_small_l8`; SHA-256 `C3AEE483BC02481FBD3D2E211358795C678C00B5D6DABD5E8E44B6A513A9655E`; sidecar `decoder_1d_T24_circ_small_l8.meta.txt` documents UNKNOWN GP-prior/training fields. History search (`git log` on artifact + “seasonal decoder”) found shipping/skip-provenance commits (`168a5c3`, `2e50187`) — **no training record** in repo |
| 18 | 5.15 | I8 / model-family reduction tests | **verified (tip)** | `tests/test_identities.py::test_alpha_zero_reduces_cox_hawkes_to_lgcp` (I8a); `::test_window_at_horizon_recovers_untruncated_loglik` (I8b); family smoke: `tests/test_smoke.py::test_hawkes_traces`, `::test_cox_hawkes_traces`, `::test_lgcp_traces` |

## Document inconsistency flag

§1.2 item 5 and the Reviewer-decisions blocker text name discharge of §5.1–§5.7 and §5.14 only; they omit §5.15, which still carries a `[[FILL]]` for identity-test names. §5.15 is included in this freeze audit (row 18).
