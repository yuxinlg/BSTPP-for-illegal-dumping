# Confirmation / smoke evidence at tip `8580364`

Env: conda env `illegal-dumping`; `JAX_PLATFORM_NAME=cpu`; `MPLBACKEND=Agg`.

- python 3.12.13
- jax 0.4.23 (`jax_enable_x64=False`)
- numpyro 0.15.0
- numpy 1.26.4
- scipy 1.11.4
- geopandas 1.1.3

Tip SHA: `85803644f4244eac53593ebc3d9a2f988f4eb6da` (2026-07-24).

These commands were re-run at tip after A/B code settled, to close the §12
smoke/confirmation evidence gap for Phase 3d/3e acceptance. Results are
recorded here with exact selectors (not ephemeral shell IDs).

## Family smoke traces (plain Hawkes / Cox–Hawkes / LGCP)

```bash
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg python -m pytest \
  tests/test_smoke.py::test_hawkes_traces \
  tests/test_smoke.py::test_cox_hawkes_traces \
  tests/test_smoke.py::test_lgcp_traces -q --tb=line
```

Result: **3 passed** in 17.41s.

## Rectangle excitation-support confirmation

```bash
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg python -m pytest \
  tests/test_phase3d_excitation_support.py::test_rectangle_modes_agree_on_array_domain \
  tests/test_phase3d_excitation_support.py::test_rectangle_mode_trace_unchanged_without_bounds \
  tests/test_heldout_polygon_mass.py::test_heldout_rectangle_unequal_counts_unchanged \
  tests/test_polygon_mass_backend_shootout.py::test_rectangle_analytic \
  tests/test_polygon_mass_backend_shootout.py::test_rectangle_degeneracy_vs_production_compensator \
  -q --tb=line
```

Result: **5 passed** in 21.03s.

## Polygon excitation-support confirmation

```bash
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg python -m pytest \
  tests/test_phase3d_excitation_support.py::test_polygon_mode_accepts_exact_builtin_gaussian \
  tests/test_phase3d_excitation_support.py::test_polygon_parenting_discards_outside_A \
  tests/test_phase3d_excitation_support.py::test_polygon_table_export_reload_roundtrip \
  tests/test_heldout_polygon_mass.py::test_heldout_polygon_unequal_counts_scores \
  tests/test_polygon_mass_backend_shootout.py::test_finite_cutoff_hybrid_table_confirmation \
  tests/test_polygon_mass_wide_range.py::test_wide_range_hybrid_table_confirmation \
  -q --tb=line
```

Result: **7 passed** in 24.36s.

## Fixed-cutoff / `set_window` confirmation

```bash
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg python -m pytest \
  tests/test_phase3e_cutoffs.py::test_legacy_fixed_cutoffs_unchanged_and_report_provenance \
  tests/test_phase3e_cutoffs.py::test_set_window_updates_cutoff_provenance_atomically \
  tests/test_phase3e_cutoffs.py::test_set_window_success_updates_pairs_cutoffs_support_consistently \
  tests/test_phase3e_cutoffs.py::test_set_window_polygon_success_and_failed_rebuild_leave_or_update_consistently \
  tests/test_polygon_mass_prepare_api.py::test_set_window_temporal_only_reuses_polygon_mass_table \
  tests/test_polygon_mass_prepare_api.py::test_set_window_spatial_change_with_compatible_table_commits \
  tests/test_polygon_mass_prepare_api.py::test_set_window_incompatible_replacement_leaves_state_unchanged \
  -q --tb=line
```

Result: **7 passed** in 13.03s.

## `jax_enable_x64` unchanged across prepare + construct

```bash
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg python -m pytest \
  tests/test_polygon_mass_prepare_api.py::test_quad_table_build_does_not_toggle_jax_enable_x64 \
  tests/test_polygon_mass_prepare_api.py::test_prepare_polygon_mass_table_and_ctor_install_without_x64_toggle \
  -q --tb=line
```

Result: **2 passed** in 10.18s.

## Plain-Hawkes NUTS fit-path smoke

```bash
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg python -m pytest \
  tests/test_fit_smoke.py::test_nuts_fit_smoke_plain_hawkes -q --tb=line
```

Result: **1 passed** in 25.83s.

## §12 family-coverage note

- Trace-level family smoke for all three families: selectors above (also a
  subset of the full suite at tip: **277 passed**).
- Fit-path NUTS smoke exists for plain Hawkes only
  (`tests/test_fit_smoke.py`); LGCP / Cox–Hawkes fit-path coverage remains
  the archived staged SBC program and was not re-run here.
- The combined polygon/3d/3e/shootout/wide-range gate at tip (**96 passed**)
  already contained the rectangle/polygon/`set_window` selectors above plus
  the shootout and wide-range property suites.

## Conditional SBC determination (A/B tip)

- Unchanged-regime four-config pins: `PIN_DIFFS 0 MATCH` vs `pins_wt_2c2.json`.
- No confirmation anomaly in the commands above.
- Therefore **no** Stage 1/2 conditional SBC rerun is triggered by this tip.
- Stage 3 R=200 exit rerun remains a **Phase 3-tip** obligation, not an A/B
  obligation; not started; Phase 3f not started.
