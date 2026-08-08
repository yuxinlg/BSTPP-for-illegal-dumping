# A-54 rebaseline record — Phase 3f operational reset (documentation/process only)

**Class DOC/process**, with the model/API decisions declared in D-54–D-58.
**Amendment A-54; decisions D-53–D-62.** **Evidence baseline `c4e069f`.**
No file under `bstpp/` or `tests/` is touched and no production implementation
begins here.

---

## 1. Pre-state

```
branch            refactor
HEAD              c4e069fca3018a7471c824d8bb26e25724943a29
HEAD subject      WP2.8 [SC/API]: CI-7 enforced at the sp_var_mu site -- the same
                  invariant at a second unenforcing site
HEAD date         2026-08-07
tracked worktree  CLEAN  (git status --porcelain showed no tracked modification;
                  139 untracked entries, none of them a target of this install)
git diff --name-only   (empty)
```

Register tip before this install: **A-53 / D-52**, `\hypertarget{a-53}` present,
maximum decision row D-52. `AGENTS.md` stated **"Next free: A-54, D-53, CI-11,
OP-32"**. Both matched the expected pre-state exactly, so no drift stop was
triggered.

**Prefix occupancy, checked before writing any output:** 20 `results/_a53_*`
files and 2 `refactor-patches/captures/a53_*` captures exist and belong to
`c4e069f`. `ls results/_a54_* refactor-patches/captures/a54_*` returned nothing —
**no collision**. Nothing under the `_a53_` / `a53_` prefix was read for writing,
modified, or deleted at any point.

## 2. Source of the installed documents

The supplied bundle was not present in the repository. It was located at
`C:\Users\Terhi\Downloads\files.zip`, whose payload is
`phase3f-operational-reset-v2` — the **v2** revision, already renumbered from the
superseded A-53 draft. Loose top-level copies in the same archive were verified
byte-identical to the nested `docs/` and `refactor-patches/` copies before use.
The bundle's `REVISION_NOTES.md` and `cursor_install_operational_reset.md` are
**not installed**: the first is the supplier's changelog, the second is an agent
instruction file, and neither is a normative artifact of this repository.

**SHA-256 of the five supplied files, as received:**

| file | sha256 |
|---|---|
| `docs/phase3f_completion_manifest.yaml` | `6744539d5779f4e865d1f495f0991f586f9d1a0d7115cf1e8ec00f2026697d03` |
| `docs/phase3f_execution_protocol.md` | `9329698669245ea4287bc7dbcfe30a16a6078b83610884c82495a2936d41f14e` |
| `docs/decoder_contract.md` | `dd67c521b1c3ce53d358b9f4045571d2a1bdc23411629dd99c76283423af558c` |
| `docs/results_format.md` | `f8baaa32a67c081efbd61a1dc4e1ad5db52e7473a3c75ffe28e36c990d99fd70` |
| `refactor-patches/phase3f/phase3f_operational_reset_amendment.tex` | `2c3169d564607490f8440527d09807ef8d2e85d2e85148a199a245215d91d490` |

The four `docs/` files were installed **unchanged**. The amendment source was
edited only by the two gate-forced wording repairs recorded in §5, so that the
integration source and the integrated register text continue to agree.

### Staleness scan (the declared stop condition)

`grep -E 'A-53|a-53|_a53_|db55ee1'` over the five supplied files returned six
hits. **Each was read in context and none is a leftover:**

| hit | verdict |
|---|---|
| manifest `register_tip: A-53 / D-52`; `evidence_basis: ... through A-53` | correct — at `c4e069f` the tip *is* A-53/D-52 |
| protocol §6.1 "`_a53_*` belongs to `c4e069f` and is never written to" | a prohibition, matching this install's own rule |
| amendment header + §"The number, first" (`db55ee1`, A-53) | the deliberate renumbering provenance; states "This entry is A-54" |
| amendment "A-53 existed because A-52's seam-set declaration…" | correct historical reference to the real A-53 |

Self-identification confirms the renumber landed: twelve `\amdnew{A-54}` marks,
`\hypertarget{a-54}{%` immediately preceding the Part II `\subsection`, rows
D-53–D-62 present, and `next_free_ids.amendment: A-55`. The condition's purpose —
catching a bundle that still claims A-53 — is satisfied. The owner confirmed
proceeding before any file was written.

## 3. Manifest validation

PyYAML is **absent from both the project conda env** (`illegal-dumping`) **and
the system Python**. This is a **machine-local validation-dependency gap**; it
was *not* remedied by adding a dependency. `ruamel.yaml 0.18.16`, already present
in the miniconda base interpreter, was used as the parser. `requirements.txt`,
`requirements-runtime.txt`, `pyproject.toml` and `setup.py` are untouched.

```
MANIFEST_OK 10 17
vertical_slices: S0_operational_reset, S1_configuration_spine,
                 S2_runtime_ownership, S3_artifacts_and_interfaces,
                 S4_architectural_cutover
corrective_cycles_used: WP1..WP10 all present, all 0
adopted_at_commit = PENDING_INTEGRATION_COMMIT
exit status 0
```

Asserted and passing: `schema_version == 1`; `as_of.commit_short == c4e069f`;
`as_of.next_free_ids.amendment == A-55`; work packages are exactly WP1–WP10;
`responsibility_coverage.items` non-empty (**17 items**) with every value naming
a declared package; five vertical slices; `corrective_cycles_used` present on
every package.

**Every original §10.f responsibility maps to exactly one completion owner** —
17 items over 10 packages, no item unmapped and none owned twice.

### The adoption hash, and why two commits

The amendment requires its integration commit to record the adoption hash. A
commit cannot contain its own hash. So, deliberately:

- **this commit** keeps `as_of.commit` at `c4e069f` — the **evidence baseline**,
  the tip every claim in the manifest was verified against — and adds a separate
  `as_of.adopted_at_commit: PENDING_INTEGRATION_COMMIT`;
- **one immediate DOC-only follow-up** replaces *only* that pending field with
  the real A-54 commit hash and records why the two-commit form was necessary.

The evidence-baseline hash is never rewritten. The sentinel is a
**one-commit-lived YAML field**: it is not a fill-anchor of the class the census
counts, it is in no anchor-census population (the census globs
`phase3_record.tex`, `docs/*.md`, `AGENTS.md` — the manifest is YAML), and it
appears in no decision row.

## 4. File lists

### Intended (pre-stated before editing)

**New (5)** — `docs/phase3f_completion_manifest.yaml`,
`docs/phase3f_execution_protocol.md`, `docs/decoder_contract.md`,
`docs/results_format.md`,
`refactor-patches/phase3f/phase3f_operational_reset_amendment.tex`.

**Edited (14)** — `phase3_record.tex`, `AGENTS.md`,
`docs/wp_dependency_graph.md`, `docs/seam_sets.md`,
`refactor-patches/phase3f/phase3f_work_package_outline.md`,
`refactor-patches/phase3f/wp2/wp2_opening_conditions_proposal.md`, and the WP3–WP10
entry records (`wp3_config_objects_entry.md`, `wp4_prepared_data_entry.md`,
`wp5_excitation_support_entry.md`, `wp6_mutators_membership_entry.md`,
`wp7_decoder_contract_entry.md`, `wp8_input_metadata_entry.md`,
`wp9_results_io_entry.md`, `wp10_args_removal_entry.md`).

**Evidence, listed separately** (under the existing conventions, not part of the
19 above) — this record; nine `results/_a54_*` gate outputs; two
`refactor-patches/captures/a54_*_red.log` preserved RED captures.

### Actual, post-staging

19 document files: **5 `A`, 14 `M`** — identical to the intended list, with no
additions and no omissions. No standing checker and no new task-tracking
document was added.

## 5. Two gates went RED, were captured, and were repaired

Both RED captures were preserved under `refactor-patches/captures/` with the
`a54_` prefix **before** any fix was applied (D-45):
`a54_content_checks_red.log`, `a54_citation_sweep_red.log`.

**Neither was a defect in the supplied documents.** Both were parse artifacts of
live instruments meeting new text:

1. **`_a25_content_checks` — "section 8 decision rows are not strictly
   increasing: D-54 follows D-60".** Check 3 reads `D-(\d+)` from the text
   *before the first `&`* on every line containing one. The supplied D-60 row
   wrapped so that the prose sentence "G-B is closed by D-54." shared a line with
   the row's closing cells, so a prose reference parsed as an out-of-order row
   label. **Repair: a line break.** The LaTeX and the rendered text are
   unchanged.

2. **`_a25_content_checks` — "4c decision cited with no §8 row: D-63"**, and the
   same token in the header sentence. Check 4c treats **every** `D-n` token in
   the register as a citation requiring a section 8 row, so naming the *next
   free* decision number red-lights the gate by construction. **A-53 already
   established the convention** — it wrote "the register stays at 52 rows, D-1 to
   D-52" and "Next free remains CI-11", naming no unissued D-number. **Repair:
   the register and the amendment source now state "next free A-55, CI-11,
   OP-32" plus "the decision register stands at 62 rows, D-1 to D-62, so the next
   decision number follows D-62".** The fact is recorded and the unissued token
   is not spelled. **`AGENTS.md` keeps the literal "Next free: A-55, D-63,
   CI-11, OP-32"**, since it is outside check 4c's population (the check reads
   `phase3_record.tex` only).

   *This is the one place the install departs from the letter of its brief*,
   which asked that the register record "A-55, D-63, CI-11, OP-32". Spelling
   `D-63` in `phase3_record.tex` is not compatible with a live gate. No gate was
   modified, no allowlist was added, and no audit apparatus was created — the
   register's own existing convention was followed instead.

3. **`_a25_citation_sweep` — 4 unreachable citations** (`decoder_contract.md`,
   `phase3f_completion_manifest.yaml`, `phase3f_execution_protocol.md`,
   `results_format.md`). The sweep resolves citations against `git ls-files`, and
   A-54 cites four documents that were still untracked. **This is a staging-order
   artifact of the same family as D-46's declared measurement point, not a
   defect.** It cleared on re-run once the documents were staged: `175 path
   citations, 166 tracked`.

**The citation sweep reports no register citation inside the two technical
contracts.** `docs/decoder_contract.md` and `docs/results_format.md` contain
**zero** D-, A-, CI- or OP-numbers, verified by direct grep before install and
confirmed by the sweep. D-61 holds as written; the supplied files carry no defect
of that kind.

## 6. Gate results

Each gate was invoked directly on the interpreter — **no wrapper**, so no exit
status is masked. stdout and stderr were captured together and the checked
process's own status appended as `EXIT_STATUS:n`.

`PY="C:/Users/Terhi/miniconda3/envs/illegal-dumping/python.exe"`

| # | command | output | exit |
|---|---|---|---|
| 1 | `"$PY" results/_c1_hypertarget_check.py` | `results/_a54_hypertarget_check.txt` | **0** |
| 2 | `"$PY" results/_a25_content_checks.py` | `results/_a54_content_checks.txt` | **0** (RED 1 → repaired) |
| 3 | `"$PY" results/_a26_ascii_sweep.py` | `results/_a54_ascii_sweep.txt` | **0** |
| 4 | `"$PY" results/_a30_label_check.py` | `results/_a54_label_check.txt` | **0** |
| 5 | `"$PY" results/_a25_citation_sweep.py` | `results/_a54_citation_sweep.txt` | **0** (RED 1 → repaired) |
| 6 | `"$PY" results/_a51_anchor_census.py` | `results/_a54_anchor_census.txt` | **0** |
| 7 | `"$PY" results/_a52_apparatus_checks.py` | `results/_a54_apparatus_checks.txt` | **0** |
| 8 | `results/_a46_capture_population.py` via its declared API — see below | `results/_a54_capture_population.txt` | **0** |

**Gate 8 needed its usage checked, and the check mattered.**
`results/_a46_capture_population.py` is a **library module**: it declares
`census_population()`, `citation_resolution_set()`, `capture_root_members()` and
`citation_sources()`, and has **no `__main__` guard and no CLI**. Executing it as
a script terminates successfully having evaluated nothing — an exit 0 that
certifies nothing, which is exactly the masked-status hazard this install was
told to avoid. It was first run that way, the empty output was noticed, and the
reading was retaken through the module's declared API with the `git ls-files`
cross-check that A-52 made part of the *definition* of taking the reading. The
driver was run from a scratch path and is **deliberately not committed**: a
one-off reading is not a standing checker, and adding one would be the apparatus
growth D-60 and D-61 exist to stop.

**Readings.**

- **`\hypertarget` 54/54** — every Part II `\subsection{A-nn}` is immediately
  preceded by its `\hypertarget{a-nn}{%`, `OK a-54` included.
- **Content checks** — **62 decision rows, D-1 to D-62, strictly increasing, no
  duplicates, no gaps.** Sub-checks 4a/4b/4c/4d all `none`. Check 5: 13 in-scope
  anchors, **0 operative**, 0 undeclared. The one `KNOWN-PREEXISTING` longtable
  field-count report at frozen Part I line 414 is reported, not suppressed, and
  is untouched by this commit.
- **ASCII sweep** — **PASS 183/183** raise sites, 11 clause functions, all ASCII.
  Unchanged by this commit: its population is production raise sites.
- **Label check** — **PASS**, within its declared A-30-renumber scope.
- **Citation sweep** — **PASS**, 175 path citations in Part II, 166 tracked, the
  9 remainder all pre-existing `ALLOWED` entries with recorded reasons.
- **Anchor census** — **16 anchor USES, 1 marked MENTION, 0 OPERATIVE**, over the
  declared population. **Identical to A-53's reading**, which is the intended
  result: this install added no anchor and filled none.
- **Apparatus checks A and B** — **PASS**.
- **Capture population** — taken at the declared point, **post-staging,
  pre-commit**. **`CENSUS_POPULATION 657`**, and the mandatory cross-check
  `git ls-files -- . ':(exclude)refactor-patches/captures/' | wc -l` **= 657**, so
  **`CROSS_CHECK_HOLDS True`** and the reading stands. Citation resolution set
  **678** (capture root included, by design, so preserved captures stay citable).
  Capture root members **21**, of which the **2** new `a54_` RED captures are
  this commit's and the **2** `a53_` captures are `c4e069f`'s and are untouched.
  The reading is **not** comparable to A-51's 601 or A-48's 550 by subtraction:
  the series is non-homogeneous and the population has grown by ordinary tracked
  files since.

**`results/_a52_gate_manifest.json` was NOT updated.** No declared gate
instrument changed and no standing gate was added; these documents add none.

**Ruff touched-file population: EMPTY — not applicable.** This installation
creates and modifies **zero** Python files, confirmed against the staged list.
Bare `ruff` over unrelated code was deliberately **not** run: that would report a
population this commit does not own.

## 7. Why the production gates do not apply

**No file under `bstpp/` or `tests/` changed** — verified against the staged
name list, which contains only `docs/`, `refactor-patches/`, `phase3_record.tex`
and `AGENTS.md`.

- **pytest (fast or slow lane)** — the suite gates behaviour of `bstpp/`. No
  importable module, no test, and no fixture changed, so neither lane can
  discriminate. Running it would produce a figure attributable to an unrelated
  tree state and record it as this commit's evidence.
- **Golden pins (both baselines)** — pins compare fitted numerical records
  produced by `bstpp/`. Nothing this commit touches is reachable from any pinned
  construction path.
- **Slow lane / three end-to-end family smokes** — same reasoning, at greater
  cost; they gate model behaviour, which is unchanged by construction.
- **SBC** — a statistical calibration procedure over sampler output. It is
  Phase 3g's, it is unaffected by documentation, and nothing here alters a prior,
  a likelihood, or an RNG path.
- **Pin regeneration** — explicitly out of scope. No pin baseline, SBC artifact,
  or historical capture was created, moved, or modified.

The documents adopted here **declare** future gate obligations (schema-v1 and
decoder-contract negative tests, slice-boundary lanes). Declaring an obligation
is not discharging one, and none of those tests exists yet.

## 8. Post-staging status

```
19 document files staged:  5 A, 14 M
0 files under bstpp/ or tests/
0 Python files
0 modifications to any results/_a53_* or refactor-patches/captures/a53_* artifact
0 literal fill-anchor tokens added anywhere
```

The pre-existing fill anchors in `docs/wp_dependency_graph.md` were **left
exactly as they are**: not filled, not rewritten as completed, not repaired. That
file is now banner-marked historical, and the point of D-59 is to stop
synchronizing it — not to finish it.

All twelve superseded planning files carry the **same** banner naming the **same**
manifest. `docs/seam_sets.md` carries the one additional sentence recording that
the declared seam set is retired as an opening precondition for WP1 and WP3–WP10
under D-59, with S_WP2's retroactive declaration standing as historical record.

## 9. Post-commit file-list comparison

Recorded in the immediate DOC-only follow-up commit that fills
`adopted_at_commit`, because the comparison cannot exist inside the commit it
describes — the same constraint that forces the two-commit adoption-hash form
in §3.
