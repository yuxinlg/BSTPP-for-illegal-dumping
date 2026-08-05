# A-27 / WP1.4e-2 — the five σ/mode invariants

**Class SC.** Declared change to observable behaviour. Not behaviour-preserving,
and not to be described as such.

Base tip `f70ac7d`. Analysis machine; pins are machine-local (A-1).

---

## 1. The brief's premise, corrected

The work was scoped on the reading that `NumericalConfig.min_sigma` / `max_sigma`
hold the **user-supplied** bounds while `resolve_sigma_bounds` holds the
**resolved** ones. That is false in production.

`main.py:2069` and `main.py:2307` pass the resolver's output. Recorded by a shim
on every `NumericalConfig.create` call across every accepting public path
(`refactor-patches/pre-3f-stabilization/probe_wp14e2_two_quantities.log`):

```
--- E2 polygon ctor, max_sigma OMITTED, projected CRS
    user passed: min_sigma=1000.0  max_sigma=None
    resolve_sigma_bounds(mode='polygon', min=1000.0, max=None, crs=projected) -> (1000.0, 5000.0)
    NumericalConfig.create(support_mode='polygon', min_sigma=1000.0, max_sigma=5000.0)
    => config received the USER-SUPPLIED pair: False
```

The config was already the resolved-bound validator. Only its docstring said
otherwise, and that docstring is corrected in this commit. Recorded because a
false docstring on a frozen config object produced a design question that cost a
round trip before any code was touched.

## 2. Enumeration (Step 1, at `f70ac7d`)

Artifacts: `refactor-patches/pre-3f-stabilization/probe_wp14e2_sigma_crossproduct.{py,log}`,
`probe_wp14e2_entry_paths.{py,log}`, `probe_wp14e2_two_quantities.{py,log}`.
Every probe asserts `bstpp.__file__` is the working tree (A-25 extension) — the
first run of the cross-product probe loaded the stale `site-packages` copy and
the assert caught it.

### Six raise sites, not two

| # | Site | Type | I3 text | I4 text |
|---|---|---|---|---|
| 1 | `excitation_support._validate_sigma_pair` | `ValueError` | `min_sigma must be finite and positive; got {lo}` | `require min_sigma < max_sigma; got …` |
| 2 | `config.NumericalConfig._validate_sigma_pair` | `NumericalConfigError` | *byte-identical to 1* | `sigma-bound coherence requires min_sigma < max_sigma; got …` |
| 3 | `config.__post_init__` polygon `max=None` branch | `NumericalConfigError` | same text, `!r` | — |
| 4 | `polygon_mass.assert_polygon_mass_table_budget` | `ValueError` | `model min_sigma must be finite and > 0 for the mass-table accuracy budget; …` | — |
| 5 | `polygon_mass.prepare_polygon_mass_table` | `ValueError` | `min_sigma must be finite and > 0; got …` | — |
| 6 | `polygon_mass.build_quad_table` | `ValueError` | `sigma_min must be finite and positive; got …` | `require finite sigma_max > sigma_min; got …` |

Two types, five distinct I3 messages, three distinct I4 messages. Site 6 is a
verbatim copy of site 1 with the parameters renamed.

I5 had three implementations (`resolve_excitation_support_mode`,
`build_excitation_support`, config) **plus a fourth behaviour**:
`resolve_sigma_bounds` did not validate `mode` at all — its `else` branch *was*
the polygon branch, so `mode='triangle'` returned `(0.05, 0.5)`.

### Reachability: all five config branches dead

Line tracer over 28 public entry-path calls at `f70ac7d`:

```
config.py:133  I5 support_mode validity                            NEVER EXECUTED
config.py:196  I1 rectangle both-or-neither                        NEVER EXECUTED
config.py:207  I2 polygon requires min_sigma                       NEVER EXECUTED
config.py:213  I3 min_sigma finite/positive (polygon max=None)     NEVER EXECUTED
config.py:230  I3 min_sigma finite/positive (_validate_sigma_pair) NEVER EXECUTED
config.py:233  I4 min_sigma < max_sigma                            NEVER EXECUTED
```

After the change, same tracer and battery: `config.validate_sigma_pair` is
**REACHED** on public paths (`results/_a27_probe_after_entry_paths.txt`). The
config's *argument* branches remain unreachable by design — they are the guard
for direct `NumericalConfig.create` callers.

### Which site wins, per entry path (pre-change)

E3/E4/E5 re-resolve `self._min_sigma_arg` / `_max_sigma_arg`, frozen at
construction, so they cannot carry a value the constructor rejected — verified,
not assumed: probe C records each calling `resolve_sigma_bounds` with exactly
the constructor's arguments.

| Invariant | E1 rect ctor | E2 poly ctor | E3/E4/E5 | E6 LGCP | E7 builder |
|---|---|---|---|---|---|
| I1 | `exsup:119` | n/a | frozen | no σ params | n/a |
| I2 | n/a | `exsup:131` | frozen | n/a | `polygon_mass:1043` **bare TypeError** |
| I3 | `exsup:151` | `exsup:151` | frozen | n/a | `polygon_mass:1045` |
| I4 | `exsup:154` | `exsup:154` | frozen | n/a | `polygon_mass:928` |
| I5 | `exsup:82` | `exsup:82` | `exsup:82` | n/a (literal) | n/a |

## 3. Design adopted — three families, owners by quantity

Signed off before implementation. `\supsd` refinement of D-40, no new decision
number.

- **I1, I2 are argument invariants** — they test which argument was omitted, and
  defaulting erases that. Only `resolve_sigma_bounds` can express them.
- **I3, I4 are resolved-bound invariants** — I4 is only meaningful after
  defaulting. `NumericalConfig` already validated the right quantity.
- **I5 is a mode invariant**, upstream of both.

Delegating everything to the config after resolution would have silently dropped
I1 and I2 — a **reject→accept** change on two of five.

Single-source functions in `bstpp/config.py`:
`rectangle_bounds_invariant_clause`, `polygon_min_sigma_invariant_clause`,
`min_sigma_positive_invariant_clause`, `sigma_order_invariant_clause`,
`support_mode_invariant_clause`, matching raisers, and one shared predicate
`validate_sigma_pair`. `polygon_mass` reaches them by deferred import (A-26's
arrangement; `config` imports `polygon_mass` for its constants).

## 4. Declared behavioural changes

1. **Error type on all five**: bare `ValueError` → `NumericalConfigError`
   (subclass; `ValueError` catchers unaffected). Message text changes for I1,
   I2, I4, I5. I3's two implementations were already byte-identical; the
   canonical clause keeps that text.
2. **`prepare_polygon_mass_table(min_sigma=None)`**: `TypeError` →
   `NumericalConfigError`. **Not a subclass relation** — a caller catching
   `TypeError` is affected.
3. **`resolve_sigma_bounds(mode=<invalid>)`**: silent acceptance →
   `NumericalConfigError`. Unreachable through any model path; affects direct
   callers.

**No accept→reject change reaches any model constructor.**

### Excluded: argument-type coercion (OP-20)

`float()` vs `_require_real`. Nine input classes accept today and would reject
under a type-delegating unification — `str`, `bool`, `np.float32`, `np.int64`,
0-d `ndarray`, `Decimal`, `__float__` objects; `np.float64` passes both. Live on
both public constructors (`min_sigma='0.05'` → ACCEPT, `min_sigma=True` →
ACCEPT). This is A-23's argument-type invariant, not one of the five. Routed to
**WP2**.

Cross-product divergence count: **21 before → 18 after**. The three that closed
are the mode cases. The 18 remaining are 17 coercion cases (OP-20) and the one
frozen asymmetry below.

### Frozen, not fixed

Polygon + `max_sigma=None` + `crs=None`: resolver rejects, config accepts.
Pinned from both sides by
`test_lane_b_polygon_default_max_sigma_without_crs_is_rejected`, because making
the config the front gate would turn this into a silent accept that passes every
other gate.

### Site 4 reachability

`assert_polygon_mass_table_budget`'s `min_sigma` check is unreachable through
every public model path (`resolve_sigma_bounds` rejects first, and
`validate_polygon_mass_table` requires `table.sigma_min == sigma_min`, which
`build_quad_table` now refuses to build). Reachable only by direct call, which is
what the Lane B I3 row does. **Labelled, not assumed.**

## 5. Test edits (signed off at Step 1, before any test was touched)

| # | File | Edit |
|---|---|---|
| 1 | `tests/test_numerical_config.py` | five σ/mode alternation rows removed from the `match=` table |
| 2 | `tests/test_numerical_config.py` | **new** `test_sigma_mode_invariants_render_the_canonical_clause`, 10 rows, equality against the canonical clause + ASCII |
| 3 | `tests/test_phase3d_excitation_support.py` | `raises(ValueError, match="min_sigma")` → exact type + clause equality + ASCII |
| 4 | `tests/test_lane_b_config_matrix.py` | **new** `test_lane_b_sigma_mode_error_identity_is_owner_invariant` (5 parametrized rows) |
| 5 | `tests/test_lane_b_config_matrix.py` | **new** `test_lane_b_polygon_default_max_sigma_without_crs_is_rejected` |
| 6 | `tests/test_lane_b_config_matrix.py` | **new** `test_lane_b_prepare_polygon_mass_table_rejects_none_min_sigma_by_name` |
| 7 | `tests/test_lane_b_config_matrix.py` | **new** `test_lane_b_resolve_sigma_bounds_validates_mode` |

No assertion weakened. Nothing edited outside this list.

## 6. RED before GREEN

`results/_a27_red_sigma_mode_rows.txt` — **12 failed, 7 passed** with
`excitation_support.py` and `polygon_mass.py` reverted to `f70ac7d` and
`config.py` keeping only the new clause functions so the tests could import them.
Restoration verified: post-restore `git diff --stat` byte-identical to pre-RED.

The 7 that passed are **recorded as non-discriminating by construction**, not
claimed as evidence: the frozen-asymmetry pin (must pass both sides — that is
what freezing means) and the I3/I4 equality rows whose canonical clause
deliberately kept the text the config already used.

## 7. Gates

| Gate | Result | Artifact |
|---|---|---|
| `pytest tests/` | **572 passed, 2 skipped, 1 xfailed** `EXIT:0` (1582.26s) | `results/_a27_suite.txt` |
| Collection delta | 562 -> 575 (+13, exactly the tests added) | as above |
| Four-config pins | **`PIN_DIFFS 0 MATCH`** | `results/_a27_pins.txt`, `results/_a27_pins_candidate.json` |
| Lane B incl. new rows | green (8 new rows) | in suite |
| `ruff` on touch | `EXIT:1` - one **pre-existing** `F401` | `results/_a27_ruff.txt` |
| ASCII sweep | **PASS** `EXIT:0` | `results/_a27_ascii_sweep.txt` |
| Content checks (incl. new monotonicity) | **PASS** `EXIT:0` | `results/_a27_content_checks.txt` |
| hypertarget | **27/27** `EXIT:0` | `results/_a27_hypertarget.txt` |
| Citation sweep | **PASS** `EXIT:0` | `results/_a27_citation_sweep.txt` |
| After-state probes | five invariants, one identity, canonical clause on every entry path | `results/_a27_probe_after_*.txt` |

**Pins are the SC rebaseline record.** The Step 2 fence permitted changing who
raises and with what identity, and forbade changing any value
`resolve_sigma_bounds` returns - the resolved `lo`/`hi` feed
`truncate_sigmax_2_prior` at `main.py:2057`, so a moved bound moves the prior,
the posterior and the pins. A match was expected; expectation is not
measurement. A diff would have meant the fence was breached.

**Suite runtime.** 26m22s against the usual ~1 min. The machine was under heavy
memory pressure for this whole session - 1.05 GB free of 15.4 GB, one hard
`ImportError: DLL load failed ... paging file is too small` in an ad-hoc probe,
and a sevenfold spread across identical Lane B selections (21s vs 180s).
Environmental; recorded so the timing is not later read as a regression.

**`ruff` EXIT:1 is reported, not waved through.** The single finding is
`F401 jax.numpy imported but unused` in `tests/test_phase3d_excitation_support.py`,
demonstrated pre-existing by running `ruff` against `git show HEAD:` of that
file. Removing it would be a test edit outside the Step 1 sign-off.

## 8. Carried forward

- **OP-20** — coercion asymmetry; WP2.
- **ASCII sweep blind spot.** `results/_a26_ascii_sweep.py` scans string literals
  that are arguments to `raise`. A clause returned from `*_invariant_clause` and
  raised via `raise NumericalConfigError(msg)` presents no literal, so the sweep
  passes over exactly the text D-40 cares most about. True since A-26 created the
  pattern; stated now because it is the dominant case. The runtime
  `.encode("ascii")` assertions in the Lane B rows are what enforce the corollary
  for these five. Extending the sweep is small and is **not** done here.
- **`prepare_polygon_mass_table(max_sigma=None)`** still dies in `float(None)`
  with a bare `TypeError` — sibling of the `min_sigma` defect fixed here, outside
  the signed-off scope, left rather than folded in silently.
- **Pre-existing `ruff` F401** in `tests/test_phase3d_excitation_support.py`
  (`jax.numpy` unused). Verified pre-existing at `f70ac7d` by running `ruff`
  against `git show HEAD:` of that file. Not fixed — an unapproved test edit.
