# Reaudit verification — NPZ/sidecar consistency, float64 snapshot, e706107 CRS

**Production-code tip for gates:** `2cf326d`
(`2cf326d83123e44847da118d6c702670d73dd292`)

**Documentation tip (this commit):** recorded after the gates below; does not
change production behavior. A one-line whitespace cleanup `dfc342a` sits
between the production tip and this docs commit (`git diff --check` only).

Env: conda `illegal-dumping`; `JAX_PLATFORM_NAME=cpu`; `MPLBACKEND=Agg`.

- python 3.12.13; jax 0.4.23 (`jax_enable_x64=False`); numpyro 0.15.0
- numpy 1.26.4; scipy 1.11.4; geopandas 1.1.3

`$PY` below means
`C:/Users/Terhi/miniconda3/envs/illegal-dumping/python.exe`.

## Historical context (do not conflate)

| Tip / note | Status |
|---|---|
| `0eeaebd` + `refactor-patches/reaudit_verification_0eeaebd.md` | Earlier reaudit evidence (schema v2 / event hash / preflight / `xy_events_real`). Full suite then: **299 passed, 3 warnings**. |
| `e706107` | Production CRS `set_crs` change **after** that recorded 299-test gate; previously **unverified** by full-suite/pin gates in the `0eeaebd` record. |
| This record (`2cf326d` gates) | Completes remaining reaudit findings + covers `e706107` paths. |

## Commits in this follow-up (after `e706107`)

| Commit | Class | Content |
|---|---|---|
| `34f6db6` | test (RED) | NPZ/sidecar tamper tests (duplicated compat fields) |
| `90d45bb` | fix (GREEN) | Enforce NPZ/sidecar field consistency on `load_npz` |
| `513cb1a` | test (post-hoc) | Malformed string-field matrix + float64 `xy_events_real` regression for `0eeaebd` |
| `f6269a5` | test (post-hoc) | Focused CRS `set_crs` coverage for `e706107` |
| `b7d12f8` | test (RED) | `attach_covariate_partitions` must not silently override CRS mismatch |
| `2cf326d` | fix (GREEN) | Replace `allow_override=True` with loud `RuntimeError` invariant |
| `dfc342a` | chore | Trailing blank line for `git diff --check` |
| *(this docs tip)* | docs | Verification record + tip note + real pin command |

`table_id` remains **descriptive** in `load_npz` / provenance; it is **not**
recomputed from the full artifact as an integrity check (`_assert_sidecar_matches_npz`
documents this explicitly).

## Exact commands and results

### Sidecar / compatibility / float64 snapshot

```bash
JAX_PLATFORM_NAME=cpu "$PY" -m pytest \
  tests/test_polygon_mass_compat_contract.py \
  tests/test_phase3d_excitation_support.py::test_xy_events_real_preserves_float64_against_float32_axis_scales \
  -q --tb=line
```

Result: **53 passed** in 48.70s.

Parent demonstration for the float64 regression (non-destructive temporary
checkout of `bstpp/main.py` from `ce5508f`, then restored): the new test
failed with `events_sha256 does not match the model event coordinates` —
the intended reason. Classified as **post-hoc** coverage for `0eeaebd`.

### Focused CRS tests for `e706107`

```bash
JAX_PLATFORM_NAME=cpu "$PY" -m pytest tests/test_crs_set_crs_paths.py -q --tb=line
```

Result: **7 passed** in 10.10s (after GREEN `2cf326d`).

CRS-less polygon `prepare_partitions` succeeds on this geopandas stack with
`set_crs(None)`; no conditional-`set_crs` repair was required for that path.

### Polygon / Phase 3d / 3e / shootout / wide-range group

```bash
JAX_PLATFORM_NAME=cpu "$PY" -m pytest \
  tests/test_polygon_mass_prepare_api.py \
  tests/test_polygon_mass_table_validation.py \
  tests/test_polygon_mass_compat_contract.py \
  tests/test_heldout_polygon_mass.py \
  tests/test_polygon_mass_backend_shootout.py \
  tests/test_polygon_mass_wide_range.py \
  tests/test_phase3d_excitation_support.py \
  tests/test_phase3e_cutoffs.py -q --tb=line
```

Result: **150 passed** in 113.62s.

### Family smoke + confirmation + x64 selectors

```bash
JAX_PLATFORM_NAME=cpu "$PY" -m pytest \
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

Result: **9 passed** in 23.20s.

### Collect-only

```bash
JAX_PLATFORM_NAME=cpu "$PY" -m pytest tests/ --collect-only -q
```

Result: **338 tests collected**.

### Full suite

```bash
JAX_PLATFORM_NAME=cpu MPLBACKEND=Agg "$PY" -m pytest tests/ -q --tb=line
```

Result: **338 passed, 1 warning** in 658.13s.

Warning (only): geographic-CRS `area` UserWarning in
`tests/test_ingestion_contract.py::test_geographic_crs_contract_warning_is_nonvacuous`
(intentional contract probe). The two deprecated `GeoDataFrame.crs` attribute
assignment DeprecationWarnings targeted by `e706107` are **absent**.

### Four-config pins (real command)

Generate candidate (tracked runner; stdout only; does not modify the baseline):

```bash
JAX_PLATFORM_NAME=cpu "$PY" refactor-patches/pin_check_v2.py . > results/_pins_reaudit_candidate.json
```

- repository-path argument: `.` (current repo root)
- candidate-output location: `results/_pins_reaudit_candidate.json`
- baseline: machine-local untracked `pins_wt_2c2.json` (UTF-16 on this machine
  from historical PowerShell redirection; **not** staged or rebaselined)

Compare (encoding-tolerant; counts field-level disagreements):

```bash
JAX_PLATFORM_NAME=cpu "$PY" -c "
import json
from pathlib import Path

def load_json(path):
    raw = Path(path).read_bytes()
    for enc in ('utf-8-sig', 'utf-16', 'utf-16-le'):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue
    raise RuntimeError(path)

cand = load_json('results/_pins_reaudit_candidate.json')
base = load_json('pins_wt_2c2.json')
diffs = []
for cfg in sorted(set(cand) | set(base)):
    if cfg not in cand or cfg not in base:
        diffs.append((cfg, '<missing config>'))
        continue
    for k in sorted(set(cand[cfg]) | set(base[cfg])):
        if cand[cfg].get(k) != base[cfg].get(k):
            diffs.append((cfg, k))
print('PIN_DIFFS', len(diffs), 'MATCH' if not diffs else 'DRIFT')
for d in diffs:
    print(' ', d)
print('configs', sorted(cand.keys()))
"
```

Result: **`PIN_DIFFS 0 MATCH`**
configs `['cox_hawkes', 'hawkes', 'hawkes_nonsquare_4to1', 'lgcp']`.

Fresh generation for this record also executed the tracked script via
`runpy` with `sys.argv = [pin_check_v2.py, <repo>]`, writing UTF-8 candidate
bytes (2324) and the same **PIN_DIFFS 0 MATCH** outcome.

### Ruff

```bash
"$PY" -m ruff check bstpp
```

Result: clean (`All checks passed!`).

```bash
"$PY" -m ruff check bstpp tests --statistics
```

Result: **133** findings — **127 E402**, **3 E702**, **3 F401** (all inherited).

### `git diff --check`

Clean on `e706107..HEAD` after `dfc342a`.

### `jax_enable_x64`

Unchanged across preparation and construction (two prepare-api selectors in
the 9-pass confirmation command above).

## Conditional SBC

Unchanged-regime pins **MATCH**; no confirmation anomaly; no pin / shootout /
oracle drift; no JAX global-state change → **no** Stage 1/2 conditional SBC
rerun. Phase 3f **not started**. Stage 3 R=200 exit remains a Phase 3-tip
obligation. **Not pushed** pending approval.
