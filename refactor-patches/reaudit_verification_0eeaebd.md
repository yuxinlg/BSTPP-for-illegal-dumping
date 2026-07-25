# Reaudit verification — PolygonMassTable contract strengthening

Tip: `0eeaebd` (`0eeaebd` short; full SHA from `git rev-parse HEAD` at docs commit).

Env: conda `illegal-dumping`; `JAX_PLATFORM_NAME=cpu`; `MPLBACKEND=Agg`.

- python 3.12.13; jax 0.4.23 (`jax_enable_x64=False`); numpyro 0.15.0
- numpy 1.26.4; scipy 1.11.4; geopandas 1.1.3

## Commits in this follow-up (after `a7c0bd7`)

| Commit | Class | Content |
|---|---|---|
| `1c3c39b` | test (RED) | Schema/interpolation/provenance/exact event-identity contract tests |
| `ce5508f` | fix | Schema v2 metadata, nested `extra`, le-f64 event hash, load/validate |
| `12e57c9` | test (RED) | Explicit polygon missing table must fail before base init |
| `238fbd3` | fix | Constructor preflight for `excitation_support="polygon"` |
| `0eeaebd` | fix | `xy_events_real` float64 snapshot; non-Gaussian trigger test supplies table |

## Exact commands and results

### Focused new RED→GREEN repairs

```bash
JAX_PLATFORM_NAME=cpu python -m pytest \
  tests/test_polygon_mass_compat_contract.py \
  tests/test_polygon_mass_prepare_api.py::test_explicit_polygon_missing_table_fails_before_base_constructor \
  -q --tb=line
```

Result: **22 passed** in 12.37s.

### Polygon / Phase 3d / 3e / shootout / wide-range group

```bash
JAX_PLATFORM_NAME=cpu python -m pytest \
  tests/test_polygon_mass_prepare_api.py \
  tests/test_polygon_mass_table_validation.py \
  tests/test_polygon_mass_compat_contract.py \
  tests/test_heldout_polygon_mass.py \
  tests/test_polygon_mass_backend_shootout.py \
  tests/test_polygon_mass_wide_range.py \
  tests/test_phase3d_excitation_support.py \
  tests/test_phase3e_cutoffs.py -q --tb=line
```

Result: **118 passed** in 111.36s.

### Phase 3d/3e confirmation selectors (+ family smoke + x64)

```bash
JAX_PLATFORM_NAME=cpu python -m pytest \
  tests/test_smoke.py::test_hawkes_traces \
  tests/test_smoke.py::test_cox_hawkes_traces \
  tests/test_smoke.py::test_lgcp_traces \
  tests/test_phase3d_excitation_support.py::test_rectangle_modes_agree_on_array_domain \
  tests/test_phase3d_excitation_support.py::test_polygon_mode_accepts_exact_builtin_gaussian \
  tests/test_phase3e_cutoffs.py::test_legacy_fixed_cutoffs_unchanged_and_report_provenance \
  tests/test_phase3e_cutoffs.py::test_set_window_updates_cutoff_provenance_atomically \
  tests/test_polygon_mass_prepare_api.py::test_quad_table_build_does_not_toggle_jax_enable_x64 \
  tests/test_polygon_mass_prepare_api.py::test_prepare_polygon_mass_table_and_ctor_install_without_x64_toggle \
  -q --tb=line
```

Result: **9 passed** in 37.01s.

### Collect-only

```bash
JAX_PLATFORM_NAME=cpu python -m pytest tests/ --collect-only -q
```

Result: **299 tests collected**.

### Full suite

```bash
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg python -m pytest tests/ -q --tb=line
```

Result: **299 passed**, 3 warnings in 645.98s.

Warnings (unchanged class): two CRS-test pandas deprecation warnings; one geographic-CRS area warning in ingestion contract test.

### Four-config pins

```bash
JAX_PLATFORM_NAME=cpu python -c "..."  # refactor-patches/pin_check_v2.py vs pins_wt_2c2.json
```

Result: **PIN_DIFFS 0 MATCH**.

### Ruff

```bash
python -m ruff check bstpp
```

Result: clean.

```bash
python -m ruff check bstpp tests --statistics
```

Result: **133** findings — **127 E402**, **3 E702**, **3 F401** (all inherited).

### `jax_enable_x64`

Confirmed unchanged across preparation and construction by the two prepare-api selectors above (included in the 9-pass confirmation command).

### `git diff --check`

Run at docs commit time on the documentation tip.

## Conditional SBC

Unchanged-regime pins MATCH; no confirmation anomaly → **no** Stage 1/2 conditional SBC rerun. Phase 3f not started. Stage 3 R=200 exit remains a Phase 3-tip obligation.

## Historical note

The pre-`8580364` missing-`excitation_support` AttributeError remains superseded historical evidence of the earlier ctor/`set_window` install-order defect, not a final-tip failure.
