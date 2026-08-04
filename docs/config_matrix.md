# Lane B configuration matrix

**Executable gate:** `tests/test_lane_b_config_matrix.py`  
**Generator probe:** `refactor-patches/pre-3f-stabilization/probe_iter3_covering_array.py`  
**Frozen rows:** `refactor-patches/pre-3f-stabilization/covering_array_rows.json` (seed=1 greedy CA, 17 rows)

## Axis level sets (from code)

| Axis | Levels | Primary sites |
|---|---|---|
| `model_family` | hawkes, cox_hawkes, lgcp | `Hawkes_Model` / `LGCP_Model` |
| `support` | rectangle, polygon | `resolve_excitation_support_mode` |
| `temporal_trigger` | Temporal_Exponential, Temporal_Power_Law, custom | `trigger.py`, capability gates |
| `spatial_trigger` | Spatial_Symmetric_Gaussian, custom | same |
| `cutoff_input` | tolerance, physical, omitted, explicit_None_via_set_window | `cutoffs.py`, `set_window` |
| `entry_path` | constructor, set_window | `main.py` |
| `builder_numerics` | default_panel_gl, guided_small_panel, nondefault_gl_order | `prepare_polygon_mass_table` |
| `standardization` | none, domain_area, bool_rejected | `standardize_cov` |
| `sigma_bounds` | both, neither_rect, polygon_min_required, custom_spatial_rejects | `resolve_sigma_bounds` |

## Iteration-2 drop (why six axes)

Pairwise at iteration 2 used only the first six axes. The three dropped from that pairwise claim were:

1. **builder_numerics** — treated as a prepare-path / B1 concern, not as a matrix axis in the executable 27-point set  
2. **standardization** — covered by a single forced reject (`bool_standardize_cov`) and `test_standardization_api`  
3. **sigma_bounds** — covered by forced rejects (`custom_spatial_rejects`, polygon `min_sigma`) rather than crossed with every other axis  

**Coverage before (iteration 2 hand points):**  
- 6-axis pairwise: **0.533** (56/105)  
- 9-axis pairwise: **0.275** (82/298)

## Generation method

Greedy pairwise covering array: while any level-pair among the nine axes is uncovered, sample candidate full assignments and keep the one covering the most remaining pairs (see probe; seed search; freeze seed=1 → 17 rows). Regenerate with:

```text
python refactor-patches/pre-3f-stabilization/probe_iter3_covering_array.py
```

## Achieved coverage

Covering array alone: **1.000** (298/298) over all nine axes.  
Forced supported/rejected rows are kept in addition (not subtracted from the array).

## Forced-row rationale

Forced rows encode explicit register gates (D-23 mode requirement, polygon hard-require table, non-Gaussian spatial×polygon, power-law×mean_lag, bool standardize, LGCP `set_window`, custom spatial×σ-bounds) and required family×support successes, independent of whether the covering array already labels those pairs. Sentinel / rollback / constructor–setter equivalence / shipped-default measured budget remain separate assertion-family tests.
