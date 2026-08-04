# Commit A — RED at `143d219`

**Class:** test  
**Module:** `tests/test_caller_state_ownership.py`  
**Environment:** jax==0.4.23, numpyro==0.15.0, numpy==1.26.4, scipy==1.11.4, geopandas==1.1.3, jax_enable_x64=False, cpu  

## Command

```text
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg python -m pytest tests/test_caller_state_ownership.py -v --tb=line
```

## Result (pre-strengthen spatial_cov frame check)

**6 failed, 7 passed, EXIT:1**

| Failed case | Defect |
|---|---|
| `domain_A_array` | B3 — `args['A_']` aliases caller `A` |
| `polygon_mass_table_values` | B4 — installed table shares `.values` |
| `model_data_domain_array` | same alias via `ModelData.domain` |
| mass-table provenance after mutate | B4 — installed arrays move with caller |
| `simulate` without rng | B2 — 76 `np.random` hits |
| `run_svi` rng_key | no parameter |

Passed (already isolated or Generator path clean): inventory meta; event DF; model_data events; simulate with Generator (bit-id + no leak); MCMC with key no leak.

## Spatial_cov strengthen (same tip, before Commit A land)

After asserting `self.spatial_cov` isolation, `spatial_cov_gdf` also **FAILED** (frame aliased). Final RED inventory includes that member.

```text
pytest tests/test_caller_state_ownership.py::test_mutating_caller_object_does_not_change_model_state -v
→ 4 failed, 2 passed, EXIT:1
```
