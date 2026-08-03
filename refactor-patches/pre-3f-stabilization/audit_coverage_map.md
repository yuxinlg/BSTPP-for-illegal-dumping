# Audit coverage map

**Candidate tip:** `cd62288`  
**Iteration:** 1 (stabilization protocol)

| Area/symbol | First audited here? | Methods | Configurations/legs | Result | Residual gap |
|---|---|---|---|---|---|
| Lane A: `pin_check_v2` + `baselines-2026-07/pins.json` + `results/_pins_*` | yes (this schema) | call-path tracing; artifact comparison | 4 rectangle pins | MATCH all candidates; branches-not-reached listed | pin JSON lacks commit/env identity |
| Lane B: constructors / `set_window` / cutoffs / standardize / σ bounds / panel-gl | yes as matrix | state-transition review; adversarial install probe; **executable matrix** `tests/test_lane_b_config_matrix.py` | pairwise + forced rows (family×support×cutoff×entry); Cox–Hawkes×polygon; ctor/setter equivalence; rollback; panel budget; G2 xfail | B1 repaired; matrix **green** (27 pass + 1 xfail G2) | iteration-2 adversarial deepen; save_rslts G2 owned 3f |
| Lane C: TruncatedLogNormal | re-established | property review + suite run | sample/log_z/truncation tests | instance fixed (**verified** suite green) | no package-wide distribution property suite |
| Lane C: union area D-30 | re-established | call-path + suite | domain_union, hawkes_bg, simulate filters | instance fixed | plot overlay multi-row only |
| Lane C: held-out mass D-26/32 | re-established | suite | heldout polygon/rect/nonfinite | instance fixed | class “hidden prep” guard not generalized beyond held-out |
| Lane C: set_window provenance / `_UNSET` | re-established | suite | sentinel + untouched | instance fixed | only one public mutator |
| Lane C: panel vs tau_abs | re-established | code read + production constant | `PRODUCTION_TAU_ABS=1e-5`; panel ratio guard | constants match A-21; **install ignores prepared panel** | B1 |
| Lane C: membership duplication | re-established | adversarial probe `probe_membership_predicates.py` | overlap union boundary/out | predicates **agree** on probe points | still two implementations (3f consolidate) |
| Lane C: capability gates | re-established | suite | kernel kwargs / custom | green | class combination matrix incomplete |
| Lane C: design scales / save_rslts | yes for save path | source read | `save_rslts` pickles samples only | provenance **not** persisted | binding 3f requirement |
| Lane C: packaging metadata | re-established | suite | Requires-Dist pins | passed alone; **1 flake fail** in batched run | flake hygiene |
| Lane D: `preparation.py` / `data_contracts.py` / `_sim_cox` | first coverage emphasis | call-path; RNG review; union | prepare_domain, validate_*, sim cox rng threading | no new pin-path defect; RNG optional with np.random fallback | aliasing/immutability of prepared objects not exhaustively probed; holes/multipolygon adversarial thin beyond existing clipped tests |
| Lane D: excitation conservation I11 | re-check | test read | rectangle Hawkes standing tests exist | protocol claim “no standing test” **refuted** for rectangle | polygon excitation conservation absent |
| Lane D: saturated modules (`polygon_mass`, `excitation_support`, `cutoffs`, `likelihood`, `utils`, audited `main` paths) | not re-saturated | targeted only where A/B demanded | panel validation path; save_rslts; covers_xy | B1 from excitation_support defaults | deeper re-read deferred |
| Lane E: seam set | yes enumerated | structural review | see below | OP-3/4/10/13 settled in record+code; OP-8/12 open by design; configs not implemented | 3f implementation |
| Ephemeral probes | yes | `probe_panel_gl_install.py`, `probe_membership_predicates.py` | CRS-less panel; membership agree | recorded under this directory | leave in audit dir only |

## Explicit 3f seam set (Lane E)

1. **Frozen configs (not yet classes):** `ModelConfig`, `PriorConfig`, `PartitionDecoderConfig`, `NumericalConfig`, `InferenceRunConfig` (A-21).  
2. **Prepared/runtime:** `ModelData`, `PreparedDomain`, `PreparedPartitions`, `ExcitationSupport`, `PolygonMassTable`, cutoff provenance records.  
3. **Legacy `args` adapter** and OP-8 removal sequence.  
4. **RNG ownership:** fit `rng_key`; simulate `rng`; internal `PRNGKey(10)` defaults; `np.random` fallbacks.  
5. **Decoder contract / gain / provenance** (`sp_var_mu`, decoder artifacts, UNKNOWN sidecars).  
6. **Plotting delegation / compatibility wrappers** (`plot_*`, `get_grid_post_mean`).  
7. **Public mutators:** currently `set_window` only (LGCP rejects).  
8. **Input metadata:** `spatial_cov_crs`, `data_contracts` mode (migration).  
9. **Results I/O:** `save_rslts` / `load_rslts` provenance+compat (unimplemented relative to A-21 binding list).

## Areas explicitly not audited (silence ≠ coverage)

- Full SBC Stage 1–3 tip exit rerun  
- Guide §15 amendments / C6 identity proof  
- Exhaustive multipolygon/hole adversarial geometry beyond existing clipped-support tests  
- NUTS divergence diagnostics / recover_test harness  
- Notebook / `batch_park_fits.py` production scripts  
- Repository-wide ruff debt (126 legacy errors per prior report)
