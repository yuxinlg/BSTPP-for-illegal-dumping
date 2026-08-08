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
edited during integration by the two gate-forced wording repairs recorded in §5,
and has since been **restored to the byte-exact supplied form** so that it
remains evidence of what was supplied rather than a copy of what landed. The
source and the register therefore **differ, deliberately**; every difference is
enumerated verbatim in §12, and both states are addressable by digest there.

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

**Installation commit: `6072a194d67f1b45a4fbb4bd122b2badc1e8ca99` (`6072a19`).**

`git diff HEAD^ --name-only`, sorted, was compared against the list pre-stated in
§4 before any editing began. **The two are IDENTICAL — 30 files, no addition and
no omission.** `git show --stat --oneline HEAD` reports
**30 files changed, 3570 insertions(+), 11 deletions(-)**, of which 16 are new.
`git status --porcelain` after the commit shows **no tracked file left modified**.

Re-verified against the committed tree, not merely the staged one: **zero** paths
matching `bstpp/`, `tests/`, a pin baseline, an SBC artifact, or the `_a53_` /
`a53_` prefixes appear anywhere in the commit.

**`git diff --check`** reports one item, deliberately left alone: trailing
whitespace on a line of `results/_a54_anchor_census.txt`. That file is verbatim
instrument output. Editing a gate capture to satisfy a whitespace linter would
falsify evidence, which is a worse defect than the whitespace.

## 10. The adoption-hash follow-up

`adopted_at_commit` was filled in the immediate DOC-only follow-up, replacing
`PENDING_INTEGRATION_COMMIT` with `6072a194d67f1b45a4fbb4bd122b2badc1e8ca99`
and adding `adopted_at_commit_short: 6072a19`.

**Why two commits.** A commit's own hash is not knowable while that commit is
being written, so a single commit cannot both adopt these documents and record
the hash at which they were adopted. The alternatives were rejected: amending
would rewrite a published gate record, and leaving the field empty would leave
the manifest unable to say when it took effect. The two-commit form keeps the
**evidence baseline** (`as_of.commit = c4e069f`, the tip every claim in the
manifest was verified against) and the **adoption point** as separate, separately
true facts. The evidence-baseline hash is never rewritten.

**Gates re-run for the follow-up: the manifest validator alone.** This is not a
shortcut but D-53's stated rule — the manifest is YAML and sits outside the
declared populations of the document-prose gates (`phase3_record.tex`,
`docs/*.md`, `AGENTS.md`), so a manifest-only edit runs the validator and nothing
else. The follow-up also appends this section to the present record, which lives
under `refactor-patches/phase3f/` and is likewise outside those populations. It
adds no file, so the capture population is unchanged at 657 and the citation
resolution set is unchanged; no document-prose gate has an input that moved.

## 11. Environment action — the manifest validator's parser

Recorded here because it is an **environment change, not a repository change**,
and the two are separately reversible. The commit that declares the dependency
carries no evidence that it was installed; this section is that evidence.

**The gap.** A-54 validated the manifest on `C:\Users\Terhi\miniconda3\python.exe`
(3.13.13) using `ruamel.yaml 0.18.16`, because **neither PyYAML nor ruamel.yaml
was importable from the project interpreter**. That made a standing gate — the
manifest validator runs on every commit changing a Phase 3f status, scope,
dependency, blocker, cycle count or exit gate — unreproducible on the documented
environment. Declaring the dependency without installing it would have left the
gate exactly as unreproducible, so both were done, separately.

**Interpreter.** `$PY` = `C:\Users\Terhi\miniconda3\envs\illegal-dumping\python.exe`,
Python **3.12.13**, conda env `illegal-dumping`.

**Install.** `"$PY" -m pip install --no-deps PyYAML`, exit **0**.
`--no-deps` deliberately: the pinned stack is fragile (`jax==0.4.23` forces
`numpy<2`, which forces `scipy<1.13` and `rasterio<1.4`), and a resolver allowed
to walk that graph is the one thing this install must not do.

**Before/after, `pip list --format=freeze`, 93 → 94 packages. The entire diff:**

```
72a73
> PyYAML==6.0.3
```

**One line moved.** The pinned stack was checked entry by entry and every one is
byte-identical across the install:

| package | before | after |
|---|---|---|
| `numpy` | `1.26.4` | `1.26.4` |
| `jax` | `0.4.23` | `0.4.23` |
| `jaxlib` | `0.4.23` | `0.4.23` |
| `numpyro` | `0.15.0` | `0.15.0` |
| `scipy` | `1.11.4` | `1.11.4` |
| `geopandas` | `1.1.3` | `1.1.3` |
| `shapely` | `2.1.2` | `2.1.2` |

The rollback condition — anything other than PyYAML moving — was not met, so no
rollback was performed.

**Footprint.** `pip show PyYAML` reports `Requires:` **empty** and `Required-by:`
**empty**. It pulls in nothing and cannot interact with the `numpy<2` /
`jax 0.4.23` pins, which is the same argument `requirements.txt` already makes
for `ruff` in the section this pin was added to.

**Validation on the declared parser.**

```
interpreter : C:\Users\Terhi\miniconda3\envs\illegal-dumping\python.exe
parser      : PyYAML 6.0.3
MANIFEST_OK 10 17
EXIT_STATUS:0
```

Output: `results/_a55_manifest_validate_declared_parser.txt`. The validator now
runs on `$PY` with the parser the repository declares, so the A-54 reading is
reproducible on the documented environment rather than on a second interpreter
that happened to have a different parser.

**Where the pin went, and where it did not.** `requirements.txt` only, under its
existing **Dev tooling** section beside `ruff`, as an exact `==` pin.
**Not** `requirements-runtime.txt`: that file's own header states it is the
"Canonical runtime install_requires for BSTPP (consumed by setup.py)" and that
"Notebook / plotting / dev tools belong in requirements.txt only", and
`setup.py:9-11,29` reads it and nothing else. PyYAML is a gate dependency, not a
package dependency, and publishing it in wheel metadata would misdescribe what
BSTPP needs to run.

**Ruff population: EMPTY / not applicable.** The commit changes no Python file
(`git diff --name-only` over its file list matches no `*.py`). The instrument was
**not** invoked: with no arguments it prints its usage and returns **1**, so a
no-argument run would record an error, not an empty reading.

**Tests.** Not run, per the owner's instruction for this commit. For the record,
the packaging tests could not have been affected: `tests/test_packaging_runtime_metadata.py`
builds a wheel and compares its `Requires-Dist` against a `CRITICAL` set derived
from `requirements-runtime.txt`, and neither of its two tests reads
`requirements.txt`.

## 12. Integration deviations — what was supplied, what landed, and why

Each deviation is listed individually with the supplied text and the landed text
verbatim. They are **not** summarized as a count, because a count cannot be
checked against either artifact.

**The two artifacts, each addressable by hash.** Digests are SHA-256 of the
**LF content** — the git blob — because `core.autocrlf=true` in this clone
checks the file out with CRLF, so a `sha256sum` run against the working tree on
Windows yields a third, platform-dependent value that identifies nothing.

| state | SHA-256 (LF content) |
|---|---|
| as supplied in the bundle, and as restored by this commit | `2c3169d564607490f8440527d09807ef8d2e85d2e85148a199a245215d91d490` |
| as committed at `6072a19` (edited during integration) | `846163c0dcde062169ce6e1e23795509c0d9e8a3b45b58db6f37037495bfc4e4` |

### Deviation 1 — the next-free line in the Part II entry

**Supplied** (`phase3f_operational_reset_amendment.tex:212`):

```
\textbf{A-55, D-63, CI-11, OP-32}.
```

**Landed** (`phase3_record.tex:4202`):

```
\textbf{A-55, CI-11, OP-32}; the decision register stands at \textbf{62 rows, D-1 to D-62}, so the next decision number follows D-62.
```

**Why.** Check 4c of `results/_a25_content_checks.py` collects every `D-n` token
in the register and requires each to have a section 8 decision row. A next-free
identifier has no row by definition, so **naming `D-63` in the register
red-lights the gate by construction**. The gate is right; the supplied text was
written without it in view.

**The convention followed is the register's own, not an invention.** A-51 wrote
"51 rows, D-1 to D-51" and A-53 wrote "Next free remains CI-11" — both state the
next free identifier without spelling an unissued decision number. `AGENTS.md:96`
carries the literal **"Next free: A-55, D-63, CI-11, OP-32"** and is outside
check 4c's population, which reads `phase3_record.tex` only. So the number is on
the record in full; it is simply not spelled in the one file a gate sweeps for it.

No gate was modified, no allowlist added, and no audit apparatus created.

### Deviation 2 — the header/date next-free clause

**Supplied.** The amendment carries no header sentence; the `\date{}` line is the
register's own.

**Landed** (`phase3_record.tex:121`), the clause added by the integration:

```
\textbf{Next free: A-55, CI-11, OP-32}, the decision register standing at 62 rows, D-1 to D-62.
```

**Why.** Identical to deviation 1 — same gate, same token, same convention. It is
listed separately because it is a second site, in a different part of the file,
and a reader checking only the Part II entry would not find it.

### Deviation 3 — the D-60 decision row's line break

**Supplied** (`phase3f_operational_reset_amendment.tex:148`), one line:

```
revise the old census. G-B is closed by D-54. & DOC / process & team; bounded
```

**Landed** (`phase3_record.tex`), the same text across two lines:

```
revise the old census. G-B is closed by D-54.
& DOC / process & team; bounded
```

**Why.** Check 3 of `_a25_content_checks.py` reads a row's decision label from the
text **before the first ampersand** on each line that has one. With the supplied
wrapping, the prose sentence "G-B is closed by D-54." sat on the same line as the
row's closing cells, so the parser read `D-54` as a row label appearing after
`D-60` and reported **"section 8 decision rows are not strictly increasing:
D-54 follows D-60"**. The rendered LaTeX is identical; only the line break moved.
RED capture: `refactor-patches/captures/a54_content_checks_red.log`.

### Deviation 4 — the amendment source was edited, and has now been reverted

**What happened.** During the A-54 installation, deviations 1 and 3 were applied
to the amendment source as well as to the register, and the source was committed
at `6072a19` in that edited form (`846163c0…`). **This commit restores it to the
byte-exact supplied form (`2c3169d5…`).**

**The conflict that caused it, named.** Two instructions were in force and they
are not compatible:

- the installation prompt's section 8 required that "the amendment source and the
  integrated register text agree";
- the remediation prompt's commit 3 required that the source **not** be edited to
  match the register, because it is evidence of what was supplied, and editing it
  destroys the only record that the two ever differed.

**Section 8 is the one that was wrong.** An integration source and an integrated
text that are required to agree cannot record an integration that changed
anything, and this integration changed three things. The source's value is
precisely that it differs. Agreement was the wrong invariant; **faithfulness to
what was supplied** is the right one, and the register is where the landed form
lives.

**Population check, before and after.** The file is **outside every declared gate
population** in both states, so restoring text containing `D-63` cannot red-light
anything. Verified against each population by name rather than assumed:
`CITATION_SOURCE_PATHS` in `_a46_capture_population.py` is a closed three-tuple
(`findings_ledger.md`, `traceability_matrix.md`, `phase3_record.tex`) — **out**;
`POPULATION_GLOBS` in `_a51_anchor_census.py` is `("phase3_record.tex",
"docs/*.md", "AGENTS.md")` — **out**; `_a25_content_checks.py` reads
`phase3_record.tex`, the `wp*_*entry.md` glob (8 files), and reuses
`POPULATION_GLOBS` — **out** of all three; `_a26_ascii_sweep.py` globs `*.py` —
**out**, this being a `.tex` file. It is reachable only as a citation *target*,
which requires it to be tracked — it is — and imposes no constraint on its
content. Confirmed by execution after the revert: content checks, citation sweep,
anchor census and ASCII sweep all exit 0.

### Deviation 5 — two bundle files were not installed

`REVISION_NOTES.md` is the supplier's changelog for the v2 revision: it describes
how the bundle was corrected, not what the repository should do, and its claims
are already reflected in the documents themselves.
`cursor_install_operational_reset.md` is an agent instruction file. Neither is a
normative artifact of this repository, and installing an instruction file as
documentation would make instructions look like adopted policy. Both remain
available in the source bundle.

### Deviation 6 — the manifest validator's YAML parser

A-54 validated the manifest on a second interpreter
(`C:\Users\Terhi\miniconda3\python.exe`, `ruamel.yaml 0.18.16`) because no YAML
parser was importable from `$PY`. **Closed** by the follow-up commit that added
`PyYAML==6.0.3` to `requirements.txt` under Dev tooling and installed it into the
`illegal-dumping` env with `--no-deps`; full before/after in section 11. The
validator now runs on `$PY` with the declared parser: `MANIFEST_OK 10 17`, exit 0.

### Deviation 7 — the capture-population gate could not report a status

`results/_a46_capture_population.py` is listed in `GATE_INSTRUMENTS` and
hash-pinned in `_a52_gate_manifest.json`, but had no `__main__`. A-54 caught this
when the bare invocation produced empty output, and took the reading through the
module's declared API instead. **Closed** by the follow-up commit that added a
`__main__` block calling that same API with an independent `git ls-files`
cross-check. RED capture of the original behaviour, 16 bytes and exit 0:
`refactor-patches/captures/a55_a46_bare_run_red.log`.

**No landed amendment was ever affected.** Every capture-population reading in the
register series — 550 (A-48), 589 (A-50), 601 (A-51), 622 (A-52), 643 (A-53),
657 (A-54) — was produced by an instrument that *imports* this module, and
import-time use is unaffected by a missing `__main__`. Measured, not assumed.

## 13. Which A-54 commit holds which files

Recorded here so the split is on the record rather than reconstructed later.

**`6072a19` — the installation commit, 30 files:**

| group | count |
|---|---|
| documents (5 new + 14 edited) | 19 |
| this rebaseline record | 1 |
| `results/_a54_*` gate outputs | 8 |
| `refactor-patches/captures/a54_*` RED captures | 2 |
| **total** | **30** |

**`29efe7c` — the adoption-hash follow-up, 3 files:**
`docs/phase3f_completion_manifest.yaml`,
`refactor-patches/phase3f/rebaseline_record_a54.md`,
`results/_a54_manifest_validate_followup.txt`.

**Note on the arithmetic.** The split is *not* 19 + 9 + 2. There are **eight**
`_a54_` gate outputs in the installation commit, and the rebaseline record is a
separate file rather than one of them. The ninth `_a54_` output,
`_a54_manifest_validate_followup.txt`, belongs to the follow-up commit.

**This record landed in `6072a19` and was completed afterwards.** Section 9 was a
forward reference when it landed — the post-commit file-list comparison cannot
exist inside the commit it describes — and was filled in `29efe7c` along with
section 10. Sections 11 to 13 were added later still, by the remediation commits.
The record therefore describes both A-54 commits and its own later completion,
and says so here rather than leaving a reader to infer it from timestamps.
