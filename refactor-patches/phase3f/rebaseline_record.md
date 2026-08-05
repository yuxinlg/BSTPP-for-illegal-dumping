# Phase 3f rebaseline record — WP1.4e-1: panel/`min_sigma` error identity (SC)

Governing document: `phase3_record.tex` (living; Part II A-26 establishes
D-40). Branch: `refactor`. Entry tip: `e9c605a`. Pins are MACHINE-LOCAL
(`refactor-patches/baselines-2026-07/pins.json`), analysis machine of §4.2.

**This is a declared change to observable behaviour (SC). It is not
behaviour-preserving and must not be described as such.**

## What changed, in one sentence

The same violation — a `PolygonMassTable` whose panel is too coarse for the
model's `min_sigma` — now raises one error type with one canonical message
clause from every entry path, instead of two types and three messages
depending on which check happened to run first.

## Observable before / after

Measured, not reasoned about: `results/_a26_probe_entry_path_split.txt` was run
at `e9c605a` (before) and re-run after the change. Same violation throughout:
`panel_h_m=200.0`, `min_sigma=5.0`, ratio `40.0`.

| Entry path | Before: type | After: type |
|---|---|---|
| constructor | `NumericalConfigError` | `NumericalConfigError` |
| `set_window(mass_table=)` | `ValueError` | `NumericalConfigError` |
| `log_expected_likelihood(mass_table=)` | `ValueError` | `NumericalConfigError` |
| `prepare_polygon_mass_table` | `ValueError` | `NumericalConfigError` |

Distinct types: **2 → 1**. Distinct messages: **3 → 3**, but all three now
begin with the *byte-identical* canonical clause; they differ only in the
site-specific remediation appended after it. Constructor and `set_window` —
the pair the WP1.4b defect split — are now byte-identical end to end.

`NumericalConfigError` subclasses `ValueError`, so every caller catching
`ValueError` is unaffected. No test in the repository pins an exact exception
class (checked: no `type(e) is ValueError` anywhere), so the type change
cannot silently alter a `raises()` outcome.

### Also changed: raised message text is ASCII (D-40)

Three raised messages contained non-ASCII characters and now do not:

| Site | Was | Now |
|---|---|---|
| `bstpp/config.py` sigma-bound coherence | `σ-bound coherence …` | `sigma-bound coherence …` |
| `bstpp/preparation.py` `standardize_cov` bool rejection | `\|C_c ∩ A\|` | `\|C_c intersect A\|` |
| `bstpp/preparation.py` `standardize_cov` unknown method | `\|C_c ∩ A\|` | `\|C_c intersect A\|` |

Not cosmetic: printing such a message to a cp1252 console raises
`UnicodeEncodeError` *while the traceback carrying it is being rendered* — an
error path that fails while failing. Observed twice during this work, once in
the Step 1 probe and once in the ASCII sweep's own reporter.

## Production changes

| File | Change |
|---|---|
| `bstpp/config.py` | New `panel_ratio_invariant_clause()` (renders the one canonical clause) and `raise_panel_ratio_violation()` (raises the one identity, optional site remediation appended). `__post_init__` calls the raiser instead of restating the message. |
| `bstpp/polygon_mass.py` | `assert_polygon_mass_table_budget` (install/held-out) and `prepare_polygon_mass_table` (builder) delegate to the raiser, each supplying its own remediation clause. Deferred import — `config` imports this module for its constants. |
| `bstpp/main.py` | **Removed** the WP1.4b call-site guard in `set_window` and the now-unused `assert_polygon_mass_table_budget` import. |
| `bstpp/preparation.py` | Two `standardize_cov` messages made ASCII. |

### The WP1.4b guard, and why removing it is safe

The guard called `assert_polygon_mass_table_budget` directly before rebuilding
`NumericalConfig`, and its own comment recorded that its purpose was to make
the legacy message win over `NumericalConfigError`. With one identity and one
canonical clause it has no remaining purpose, and removing it restores D-35's
claim that the config is the single source. Checked for other load-bearing
roles: it performed no mutation, its return value was discarded, and the
ratio it checked is re-checked by `NumericalConfig.create` two lines later
with the same `panel_h_m` and `min_sigma` (both paths read `table_arg.h_panel`
and the resolved `lo_cfg`). Transactionality is unaffected — the rejection
still happens in local state before the atomic commit block, and the Lane B
rollback tests confirm whole-state equality after the rejection.

## Test changes (SC sign-off recorded)

Sign-off for this list was given by Terhi before any test was edited, on the
enumeration reported at Step 1. **No assertion was weakened**; every edit
either pins an identity that was previously unpinned or strengthens a loose
substring alternation to the canonical clause.

| # | File | Edit |
|---|---|---|
| 1 | `tests/test_numerical_config.py` | sigma-bound `match` alternation no longer spells the clause with U+03C3 |
| 2 | `tests/test_numerical_config.py` | **new** `test_panel_ratio_clause_is_ascii_and_single_sourced` |
| 3 | `tests/test_numerical_config.py` | `test_immutable_after_construction`: bare `raises(Exception)` → `FrozenInstanceError` + `match` |
| 4 | `tests/test_panel_min_sigma_guard.py` | builder rejection pinned to `NumericalConfigError` + canonical clause prefix + its own remediation; ASCII assertion |
| 5 | `tests/test_polygon_mass_table_install.py` | builder-guard site pinned to identity + clause |
| 6 | `tests/test_polygon_mass_table_install.py` | constructor site: exact clause equality, identity pinned |
| 7 | `tests/test_polygon_mass_table_install.py` | held-out site: identity + clause prefix; OP-19 noted in place |
| 8 | `tests/test_polygon_mass_table_install.py` | `test_rejected_table_does_not_construct_or_rebuild`: two bare `raises(ValueError)` → identity + clause |
| 9 | `tests/test_lane_b_config_matrix.py` | OP-17 site 1: `match=` per parametrized `bad_call` |
| 10 | `tests/test_lane_b_config_matrix.py` | OP-17 site 2: `match=` on the stale-table `spatial_window` mismatch |
| 11 | `tests/test_lane_b_config_matrix.py` | **new** `test_lane_b_panel_ratio_error_identity_is_entry_path_invariant` |

Items 3 and 8 are the approved generalization: after this commit no test in
the affected set uses a bare `pytest.raises` for any of these invariants.

### RED demonstrated before GREEN

`results/_a26_red_entry_path_row.txt`. The new Lane B row was run with
`bstpp/main.py` and `bstpp/polygon_mass.py` reverted to `e9c605a` (the WP1.4b
guard restored, R2/R3 restating their own text) and `bstpp/config.py` kept only
so the clause helper could be imported. It **FAILED**, on the
`ValueError` raised by the restored guard rather than the expected
`NumericalConfigError` — i.e. it discriminates precisely the defect it is
written to catch, not merely the absence of the fix.

The five tests that were already green before the production change stayed
green after it, because their substring alternations were too loose to
distinguish the two identities. That is the defect restated, and it is why the
edits above replace those alternations with clause equality.

## Gates

All runs on the analysis machine, `JAX_PLATFORM_NAME=cpu`, with
`git status --porcelain` captured alongside each result (A-25).

| Gate | Result | Capture |
|---|---|---|
| Four-config pins | `PIN_DIFFS 0 MATCH` | `results/_a26_1_pins.txt`, `results/_a26_1_pins_candidate.json` |
| `pytest tests/` | **559 passed, 2 skipped, 1 xfailed**, `EXIT:0` | `results/_a26_1_suite.txt` |
| Lane B incl. new row | pass | in the suite capture |
| ruff on touch | `All checks passed!` EXIT:0 | `results/_a26_1_ruff.txt` |
| ASCII sweep over raised messages | PASS | `results/_a26_ascii_sweep.py` |
| `jax_enable_x64` | unchanged (`False`) | asserted in-suite |

Collection moves 560 → **562**: the two tests added here
(`test_panel_ratio_clause_is_ascii_and_single_sourced` and the Lane B
entry-path row). Passes move 557 → 559 accordingly; skips and the G2 strict
xfail are unchanged.

**Pins as the SC rebaseline record.** Error paths are not traced by
`pin_check_v2.py`, so a match was expected — but expectation is not
measurement, and the run is the record. `PIN_DIFFS 0 MATCH` on all four
configurations establishes that this SC changed no traced numerical result. A
diff here would have meant the change reached a traced path and would have
stopped the phase.

## Carried forward

- **OP-19** — held-out scoring validates a supplied table against module
  default budget policy, not the model's `NumericalConfig`. Its error identity
  is unified here; its *policy source* is deliberately not changed, because
  passing `numerical_config` on that path would change which `panel_h_m` and
  `gl_order` govern held-out validation for anyone holding a non-default
  table. That is a separate observable change with its own pin question. An
  instance of the dual-source debt already accepted through OP-8; routed to
  WP5 with the `ExcitationSupport` seam.
- **WP1.4e-2** — the five sigma/mode invariants have the same split, with the
  config's branches unreachable through any public entry path. Separate SC
  commit, before WP2.
