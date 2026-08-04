# Commit C — budget resolution (CF)

**Resolution chosen:** **(i) Measure** — install validates a measured residual against `PRODUCTION_TAU_ABS`.

## Why not (ii)

At `panel/min_sigma = MAX_PANEL_TO_MIN_SIGMA_RATIO = 8`, host tables with `gl_order=8` fail `PRODUCTION_TAU_ABS=1e-5` against both the shapely §13 oracle and the elevated-GL host reference (unit-octagon and unit-square CRS-less fixtures; max abs err ≈ 1.8e-5 … 9e-4). The constant ratio ceiling is therefore **not** valid across the supported `gl_order` range. Protocol: if (ii) fails, (i) is required.

## Measured gate

| Item | Value |
|---|---|
| Reference method | Host NumPy/SciPy float64 `_quad_masses_numpy` at `BUDGET_REFERENCE_GL_ORDER=32` on `prepare_quadrature(..., h=table.h_panel)` |
| Reference own bound | `BUDGET_REFERENCE_ORACLE_BOUND=1e-6` vs shapely §13 oracle (calibration: ≤6.3e-7 on unit-octagon at ratio≤8) |
| Production tolerance | `PRODUCTION_TAU_ABS=1e-5` |
| Ratio role | Prepare/install **prefilter** only (`MAX_PANEL_TO_MIN_SIGMA_RATIO`); no longer claimed as sufficient for tau |

Sites: `validate_polygon_mass_table` → `assert_polygon_mass_table_accuracy` on construct / held-out / `set_window`.

## Silent kwargs removed

`build_excitation_support(..., panel_h_m=, gl_order=)` deleted from the signature (were `del`'d no-ops). Build settings are chosen only at `prepare_polygon_mass_table`.

## Env

jax==0.4.23, numpyro==0.15.0, numpy 1.26.4, scipy 1.11.4, geopandas 1.1.3, `jax_enable_x64=False`
