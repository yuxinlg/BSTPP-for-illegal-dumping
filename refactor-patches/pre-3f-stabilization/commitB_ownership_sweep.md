# Commit B — ownership sweep inventory (pre-fix)

**Tip before CF:** `0d44ce5` (Commit A RED)  
**Rule:** a prepared object copies or freezes every caller-supplied array it retains; no path draws entropy the caller did not supply.

| Site | Retention | Disposition |
|---|---|---|
| `prepare_domain` array branch `A_ = A` | `PreparedDomain.bounds` / `args['A_']` alias caller ndarray | **copy** `np.array(A, dtype=float, copy=True)` |
| `PreparedDomain.domain` | stores caller `A` (array or GeoDataFrame) | **copy** array; GeoDataFrame `.copy()` |
| `ModelData.domain` / `events` | constructor stores caller refs | **copy** at `ModelData` construction in `main.py` |
| `self.data = data` | alias event frame | **copy** |
| `self.A = A` (polygon) | alias domain GDF | **copy** (use `prepared_domain.domain` after prepare copies) |
| `self.spatial_cov = spatial_cov` | alias cov GDF | **copy** |
| `PreparedPartitions.cov_gdf = spatial_cov` | alias | **copy** at attach |
| `args['spatial_cov']` / `cov_values` | already new array from `.values` (+ standardize) | keep (already isolated) |
| event `t_events` / `xy_events` | built in `_scale_xyt` from data | keep (already isolated) |
| `PolygonMassTable` install | same object / same `.values` buffer | **copy** table arrays at install (`PolygonMassTable.copy()`) |
| decoder params | loaded from package artifacts | n/a (not caller-supplied) |
| `simulate(rng=None)` | `np.random` fallback in `_sim_*` / triggers | **require** explicit Generator; named `ValueError` |
| trigger `simulate_trigger(rng=None)` | `np.random` if None | only reached with threaded `rng` from simulate; keep defensive require when called from package sim path |
| `run_svi` | always `PRNGKey(10)` | add `rng_key=` like `run_mcmc`; `None` → `PRNGKey(10)` documented |
| `run_mcmc(rng_key=None)` | `PRNGKey(10)` | leave; document |

Freeze (`setflags(write=False)`) not used — tables are moderate size; prefer copy.
