# A-28 / WP1.4f — closing the σ family: `max_sigma` at the mass-table builder

**Class CF.** Declared change to observable behaviour on error paths only. Not
behaviour-preserving, and not to be described as such.

Base tip `426d60a`. Analysis machine; pins are machine-local (A-1).

---

## 1. The family question, and the answer

A-27 fixed `prepare_polygon_mass_table(min_sigma=None)` and left its sibling
one line below, dying in `float(None)`:

```
prepare_polygon_mass_table(max_sigma=None)
  -> TypeError: float() argument must be a string or a real number, not 'NoneType'
     bstpp/polygon_mass.py:1092
```

The brief left the family to be decided from the code. **It is a distinct
invariant — I6 — not a member of I2.** A-27's own test decides it: owners of one
invariant validate the same *quantity*.

| | I2 (polygon requires `min_sigma`) | I6 (builder requires `max_sigma`) |
|---|---|---|
| Quantity | `min_sigma` | `max_sigma` |
| Owners | constructor, `resolve_sigma_bounds`, `NumericalConfig.create`, public builder | the two builders only |
| True at the model boundary? | **Yes** — nothing anywhere defaults `min_sigma` | **No** — the resolver defaults it |

The third row is decisive. `resolve_sigma_bounds` defaults an omitted polygon
`max_sigma` to `DEFAULT_MAX_SIGMA_KM` through the projected CRS
(`excitation_support.py:153-157`), and `NumericalConfig` accepts `max_sigma=None`
outright — the asymmetry A-27 froze and pinned. I2's clause reads *"min_sigma is
required and has no default"*. Rendering it for a `max_sigma` violation would
**raise a message asserting a package-wide requirement that does not exist**. A
borrowed clause here is not untidy; it is untrue.

Why the requirement is the builder's alone: `max_sigma` there is not a prior
bound but the **top knot** of the table's log-sigma grid
(`log_knots(sigma_min, sigma_max)`, `polygon_mass.py:939`), past which
`validate_sigma_in_range` prohibits evaluation. The builder also cannot borrow
the resolver's default — that default needs a projected CRS and `crs` is
*optional* at the builder, so defaulting would be a reject→accept widening
rather than the requested fix.

New single source in `bstpp/config.py`:
`builder_max_sigma_invariant_clause` / `raise_builder_max_sigma_violation`,
reached from `polygon_mass` by the deferred import A-26 established.

## 2. Enumeration (Step 1, at `426d60a`)

Artifacts: `refactor-patches/pre-3f-stabilization/probe_wp14f_max_sigma_sites.py`
and `..._BEFORE.log`. The probe asserts `bstpp.__file__` is the working tree
(A-25 extension) — see §6.

**Three `TypeError` sites, not one.** Measured by execution, not inferred from
call order:

| # | Site | Pre-change | Disposition |
|---|---|---|---|
| 1 | `prepare_polygon_mass_table` `:1092` | `TypeError` from `float(max_sigma)` | **fixed** → I6 |
| 2 | `build_quad_table` `:935` | `TypeError` from `float(sigma_max)` | **fixed** → I6 |
| 2b | `build_quad_table` `:935` | `TypeError` from `float(sigma_min)` | **fixed** → I2 |
| 3 | `validate_polygon_mass_table` `:810` | `TypeError` from `float(sigma_max)` | **left**, OP-21 |

Site 2 exists because `validate_sigma_pair` deliberately does not coerce
(OP-20), so `validate_sigma_pair(float(sigma_min), float(sigma_max))` coerces
*both* arguments in front of it and either `None` dies unnamed. Fixing only
`max_sigma` there would have left `build_quad_table(None, 40.0)` raising an
unnamed `TypeError` one line from this commit's own guard — recreating the exact
situation the commit exists to close. Both are fixed; site 2b is declared.

Non-owners, confirmed accepting (this is the contrast I6 rests on):

```
resolve_sigma_bounds(polygon, max_sigma=None, crs=EPSG:32618) -> ACCEPTED (1000.0, 5000.0)
NumericalConfig.create(polygon, max_sigma=None)               -> ACCEPTED  max=None
```

## 3. Declared behavioural changes

1. `prepare_polygon_mass_table(max_sigma=None)`: `TypeError` → `NumericalConfigError`.
   **Not a subclass relation — a caller catching `TypeError` IS affected.**
2. `build_quad_table(sigma_max=None)`: same change, same clause.
3. `build_quad_table(sigma_min=None)`: `TypeError` → `NumericalConfigError` (I2).

**No accept→reject change reaches any model constructor**, and none of the three
is reachable from one: every model path resolves bounds before building, and
`build_quad_table` is not called from `main.py` at all.

## 4. Precedence preserved deliberately

At `426d60a`, `max_sigma=None` **lost to every other check** in
`prepare_polygon_mass_table` (measured, `..._BEFORE.log`):

```
(min=None, max=None)        -> I2  polygon requires min_sigma
(min=0.0,  max=None)        -> I3  min_sigma finite and positive
(min=5.0,  max=None, h=1e6) -> panel ratio
(min=5.0,  max=None)        -> TypeError            <- the only case reaching float()
```

The I6 guard is therefore placed **last**, immediately before the delegation.
Guarding earlier would have changed which error a doubly-invalid call reports —
an *undeclared* behaviour change riding along with the declared ones.

## 5. RED before GREEN

`results/_a28_red_builder_max_sigma_rows.txt` — **3 failed, 1 passed**, with
`polygon_mass.py` alone reverted to `426d60a`.

`config.py` was held at candidate for the RED run on purpose: the new clause
functions are the single-source API, not the defect, and reverting them too
would have produced an `ImportError` at collection — a red that proves nothing.
The three failures are on the defect itself, `TypeError: float() argument …` at
`polygon_mass.py:1092` and `:935`.

The one pass is `test_lane_b_builder_max_sigma_guard_preserves_error_precedence`.
It is **non-discriminating by construction and is not claimed as evidence**: it
pins the three pre-change messages, which the change is designed to preserve, so
it must pass on both sides.

## 6. Provenance

The enumeration probe's **first run loaded a stale `bstpp` from `site-packages`**
and died on `bstpp.config` — the same failure class A-26 recorded. A script under
`refactor-patches/` does not put the repo root on `sys.path`. The probe now
inserts the repo root and **asserts `bstpp.__file__` lies under it**, so the
capture cannot silently describe a different object.

## 7. OP-21 — opened, deliberately not taken

`validate_polygon_mass_table(sigma_max=None)` (`polygon_mass.py:810`) is
**labelled, not assumed**, in A-27's manner with `assert_polygon_mass_table_budget`.

It is not a builder: it compares a *built* table's recorded range against the
*model's resolved* bound, so its quantity is the model-side bound and the I6
clause would misdescribe it. It is unreachable from every model path —
`excitation_support.py:404` asserts `lo is not None and hi is not None` before
the call — and reachable only by direct call.

Deciding what it should raise means deciding whether a `NumericalConfig` holding
`max_sigma=None` may reach install validation at all. That is **the frozen
asymmetry A-27 explicitly declined to move**, pinned from both sides by
`test_lane_b_polygon_default_max_sigma_without_crs_is_rejected`. Unfreezing it
inside a CF commit would be exactly the sleight A-27 refused with OP-20. Routed
to **WP5** alongside OP-19.

## 8. Gates

| Gate | Result | Artifact |
|---|---|---|
| `pytest tests/` | **576 passed, 2 skipped, 1 xfailed**, EXIT:0 | `results/_a28_suite.txt` |
| Four-config pins | **`PIN_DIFFS 0 MATCH`** | `results/_a28_pins.txt` |
| `ruff` on touch | **EXIT:0** | `results/_a28_ruff.txt` |
| ASCII sweep | **PASS** (17 modules) | `results/_a28_ascii_sweep.txt` |
| `\hypertarget` | **28/28** | `results/_a28_hypertarget.txt` |
| Content checks (incl. D-row monotonicity) | **PASS**, 40 rows D-1..D-40 | `results/_a28_content_checks.txt` |
| Citation sweep | **PASS** | `results/_a28_citation_sweep.txt` |

Pins are the CF rebaseline record. The change touches only error paths — two
guards that raise before any quadrature runs — so a match was *expected*;
expectation is not measurement, and this run is the measurement. A diff would
have meant a guard was placed where it can fire on an accepting path.

Collection **575 → 579**, the four rows added here; 572 → 576 passing.

The suite took **27m05s** against the usual ~1 min, matching the 26m22s A-27
recorded on this machine in the same session. Environmental (memory pressure),
not attributable to this change, and recorded so the timing is not later read as
a regression.

`git status --porcelain` is captured with every run; `bstpp.__file__` with every
probe.
