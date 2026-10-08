# Box copy salvage record (2026-10-08)

Status: provenance and triage record, not a register amendment. Box amendments
are cited by **commit hash only**: the Box line reused A-50 onward and D-52
onward for different content, so a Box "A-58" is not local A-58 and must never
be cited as if it were.

## What happened

The working tree moved off Box Drive to `C:\Users\Terhi\dev\BSTPP_terhi` on
2026-08-05, but the Box copy (`C:\Users\Terhi\Box\BSTPP_terhi\BSTPP-refactor`)
stayed usable and was developed in again from another machine:

| Date | Box copy | This copy |
|---|---|---|
| 08-06 | stops at A-49 (`51c98f8`) | — |
| 08-06..08-10 | — | A-50..A-54, `a72f5f4`, preview branch, S1.0 `abe0f8b`, S1.1 `f74fd9c` (unpushed) |
| 09-15 | writes its own A-50..A-61 on top of A-49 (`5dc1110`..`1736799`) | — |
| 09-16 | resets to `origin/refactor` (`a72f5f4`), re-lands part of that line as `0327b41`..`b44795b` (A-55..A-60), **pushes** | — |
| 09-17 | `e763867`, a second ModelConfig implementation (unpushed) | — |

Consequences: local `refactor` is 2 ahead / 7 behind `origin/refactor`; the
preview branch (`origin/preview`, base `c04c989`) is **unaffected** — its
package and test code equal `c04c989` and no Box-only commit is in its ancestry.

## Where the Box-only material now lives

- Local branches `box/laptop-docs-a61` (`1736799`), `box/modelconfig-e763867`,
  `box/a28-wrong-numbers` (`ecdfa51`, the A-49 commit under its old wrong
  numbers — superseded, nothing substantive).
- `C:\Users\Terhi\dev\BSTPP_terhi\_archive\` (machine-local, outside the repo):
  `Box_BSTPP_terhi_2026-10-08.zip` (whole Box folder, 3516 files, verified
  member-by-member against the source by size and CRC-32) and
  `Box_BSTPP-refactor_only-branches_2026-10-08.bundle` (the three branches).
- Everything else in the Box tree (`data/`, `output/`, 113 archived results,
  31 untracked files, the freeze-worktree copy) was byte-identical to this copy.

## Adopted in this patch

1. **D-41 post-commit form** (from Box `5c9ac91`, register text and capture
   `refactor-patches/captures/a58_red_postcommit_filelist_mismatch.txt` on
   `box/laptop-docs-a61`). `git show --stat HEAD` renders a rename as
   `{a => b}/x`, matching neither path of a pre-stated list; measured there as
   12 "missing", 10 false. The Box commit also records the failure it was
   checking: one bad pathspec aborted the whole `git add`, and the commit then
   **exited 0** on the index an earlier add had left, silently omitting the
   amendment it described. `AGENTS.md` now prescribes
   `git show HEAD --name-status -M --format=''`. **`origin/refactor` does not
   carry this correction** (its register, `AGENTS.md` and `CONTRIBUTING.md` still
   say `--stat`). The register's D-41 text is amended at reconciliation.
2. **Supersession label** on `refactor-patches/phase2c/boundary_and_window_semantics.md`
   (from Box `507418a`). The only diverged archive/live pair with no pointer
   forward; the config-matrix pair was already labelled.
3. **Number-range warning and single-working-copy rule** in `AGENTS.md`.

## Deferred — needs a register amendment, so waits for reconciliation

Any amendment written on local `refactor` now would collide with `origin`'s
A-55..A-60 / D-63 / OP-32..OP-34 a second time.

1. **Content-check row population is short by four rows.** The §11 row regex in
   `_a25_content_checks.py` (`results/` here, `tools/gates/` on origin) is
   `\s*(?:\\amdnew\{[^}]*\}\s*)*(OP-\d+)\s*&`. It cannot see the
   `\supsd{A-9}{OP-2}`, `\supsd{A-13}{OP-6}`, `\supsd{A-9}{OP-9}` rows or
   `OP-7\amdref{A-20}` — verified against this register, lines ~1931–1938 — so
   checks 4a/4b/4d run on a population four short and 4b silently skips settled
   rows. A gate whose population depends on the document conforming to it is not
   measuring the document. Fix and new check 4e (every cited OP has a row) are
   in Box `503a748` (`_ROW_PREFIX`, `cited_op` hunks).
2. **OP-18 has no §11 row.** Cited three times (this register ~2907, ~2997, and
   the A-25 prose) but never given a row. Check 4e goes red on it; the row and
   its rationale are Box `503a748` (register ~1786, ~3912–3933, red captures
   `refactor-patches/captures/a52_red_content_check4e_op18{,_after_rowfix}.txt`).
   Land 1 and 2 together, RED-first.
3. **D-41 register text** — see Adopted 1.

## Deferred — needs a decision from Terhi

- **Declared `args` compatibility surface** (Box `5dc1110`, its D-55). It rests
  on two facts not recorded on this line — a preview fork is consumed downstream
  today, and publication is intended — and conflicts with local D-58/D-62, which
  treat `args` removal at S4 as internal. Policy, not a merge.

## Findings against local S1.1 (`f74fd9c`), exposed by the Box implementation

Box `e763867` implements the same ModelConfig slice. Local is stronger overall
(62 tests vs 17; the sp_var_mu ordering change is declared and pinned, where
Box made the same change undeclared). Pins are identical (candidate hash
`e8ade72f7947`, polygon baseline `compared=6/6 MATCH`). But it exposes:

1. **`_cox_background` leaks through `LGCP_Model(**kwargs)`.** Before S1.1,
   `LGCP_Model(..., _cox_background=True)` raised "Unknown argument"; at S1.1 it
   builds `ModelConfig(model='lgcp', cox_background=True)`. An accept set moved,
   against S1.1's BP claim. `Hawkes_Model(..., _cox_background=...)` changes
   exception type (`Exception` → `TypeError`).
2. **No model/cox_background consistency.** `ModelConfig.create(model='lgcp',
   cox_background=True)` and `('hawkes', True)` construct; a direct
   `Point_Process_Model('hawkes', ...)` stores `cox_background=None` where the
   model name determines it.
3. **`np.bool_` stored unnormalised** in `ModelConfig.cox_background` (only
   `to_record` converts). Box normalises in `__post_init__`.

Each needs a RED row before its fix (D-41). Box tests worth porting as rows:
`test_model_and_cox_background_must_agree`,
`test_create_requires_model_or_cox_background` (`e763867:tests/test_model_config.py`).

## Merge notes for reconciling with `origin/refactor`

- Origin moved the gate scripts to `tools/gates/` but its `AGENTS.md` still
  gives `results/` commands; its own `_path_aliases.py` says an alias makes a
  citation resolve but not a stale command work. Repoint the commands on merge.
- Origin's CI runs neither `_a58_alias_discrimination.py` nor
  `_a46_exclusion_discrimination.py`; Box's workflow ran both
  (`box/laptop-docs-a61:.github/workflows/gates.yml`, document-gates job).
- Origin dropped rationale comments the Box versions carried
  (`_path_aliases.py`, `_a58_alias_discrimination.py`, the commit-msg hook's
  measured class frequencies). Comments only.

## Evidence-file facts not committed anywhere

- Box `refactor-patches/captures/a60_red_fast_lane_polygon_compat.txt` is **not
  a code RED**: all 30 errors are `PermissionError` on the pytest temp root
  (`pytest-of-Terhi`). It belongs to Box `f300f30`, not origin `b44795b`. Do
  not cite it as evidence of a defect.
- Box `f300f30`'s "FULL SUITE 628 passed, 2 skipped, 1 xfailed" equals the
  *sum* of its fast (617/1/12 deselected/1 xfailed, 183 s) and slow (11 passed,
  1 skipped, 1220 s) lane captures; the quoted 1233.68 s matches neither run nor
  their sum. Treat it as two lanes, not one full run. One slow test skipped,
  unidentified.
- The long draft of origin `0327b41` (Box `results/_a55_commit_msg.txt`) adds:
  every prior measurement in this project was taken on **Windows**; CI does not
  sweep historical placeholders; the autocrlf apparatus failure was pinned to
  `results/_a52_gate_manifest.json` with `git status --porcelain` empty.

## Reconciled (same day)

Local S1.0, S1.1 and this record were rebased onto `origin/refactor`
(`b44795b`); S1.0/S1.1 replay patch-identical, and the pre-rebase tip is kept
at `backup/refactor-pre-rebase-2026-10-08`. The renumbering commit that follows
moves S1's closing amendment from A-55 to A-61 (manifest
`s1_closes_under_a61`, which also registers the four remediation commits that
no published amendment registered and records the `_a55_` prefix exception),
and repoints `AGENTS.md`'s gate commands to `tools/gates/` — the first merge
note above. The other merge notes, and every item under "Deferred", are still
open, and none can be registered before A-61. S1.1 needs no `CHANGELOG.md`
entry: it adds an `args` key and removes, renames or redefines none, which is
the `CONTRIBUTING.md` trigger.

## Rejected, with reason

- Guidance/record split, AGENTS.md → CONTRIBUTING.md + `docs/reference/*`,
  generated `docs/status.md`, placeholder census, pointer ADRs `D-001..D-055`
  (Box `5dc1110`, `c6232a7`, `aba107e`, `507418a`, `edfcdf0`, `7ef2128`):
  restate this `AGENTS.md`, or contradict D-53 (the manifest is the sole
  execution truth), D-54 (OP-28 closed — Box still shows it escalated), D-59 and
  D-60, and are keyed to Box's colliding decision numbers. One lesson survives
  from `edfcdf0`: a generated projection must key its staleness check on its
  source's content, never on `HEAD`, or the check embeds the value it compares
  and can never pass.
- `archive/` and `record/` moves (Box `1736799`): naming only; origin declined
  them for stated reasons.
- Commits already re-landed on origin (Box `c07f280`, `87c4c39`, `e0c851d`,
  `f300f30`): origin's versions are the corrected ones.
