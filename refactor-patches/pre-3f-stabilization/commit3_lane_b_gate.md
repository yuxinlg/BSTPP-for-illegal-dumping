# Commit 3 — Lane B matrix gate

**Class:** test (coverage completion, not repair)  
**Module:** `tests/test_lane_b_config_matrix.py`

## What landed

Executable protocol §6 Lane B gate: axis inventory from code; forced reject/success rows; Cox–Hawkes × rectangle/polygon; LGCP `set_window` reject; support object identity; constructor/setter physical-window equivalence; sentinel no-op / explicit `None`; whole-state rollback; polygon shipped-defaults panel budget vs `PRODUCTION_TAU_ABS`; `save_rslts` provenance asserted under **strict xfail** (ledger G2 / 3f).

## Results

| Gate | Result |
|---|---|
| `pytest tests/test_lane_b_config_matrix.py -v` | **27 passed, 1 xfailed** |
| `ruff check tests/test_lane_b_config_matrix.py` | clean |
| `pytest tests/ -q` | **496 passed, 2 skipped, 1 xfailed, EXIT:0** (~11 min) |

## Further defects

No **new** unexpected production defects in this matrix run. Known G2 (`save_rslts` omits provenance) remains documented as strict xfail inside the matrix — not closed.

## READY status

Still **NOT READY FOR 3F**: iteration-2 re-audit of B1 surfaces + Lane B, then §8 battery on a frozen tip, remain outstanding (see `readiness_report.md`).
