# Contributing

BSTPP is scientific numerical code. **Reproducible-but-wrong is a real failure
mode, so validation and testing rank above cleanliness.** Distilled from
*Better Code, Better Science* (bettercode-book.org) and from what this
repository has actually been bitten by.

Work happens on branch `refactor`, in small atomic commits.

## Before you start

1. Read `AGENTS.md` -- the entrypoint, one page.
2. Read Part II of `phase3_record.tex` from the most recent amendment backwards
   far enough to cover the work in hand. **That is the governing record**; the
   guidance files summarise it and can lag it.
3. If you are touching units, the likelihood, the simulators, or excitation
   support, read the unit-contract and likelihood-simulator coupling paragraphs
   in `AGENTS.md` first, then the matching Part II amendments.
4. Current Phase 3f *execution* state -- scope, status, dependencies, gates --
   is in `docs/phase3f_completion_manifest.yaml` (**D-53**), not in this file.

## The prime directive

**Refactoring changes structure, never external behaviour.** Every change must
be behaviour-preserving and provable by a passing test.

**Do not refactor without a test harness in place first.** If tests are thin,
pin the current behaviour before changing structure: capture intermediate
outputs from the existing code and assert the refactor reproduces them within
tolerance.

Correctness is the maintainer's to verify. "All tests pass" and "this is
equivalent" are not evidence -- **show the diff and the test output.**

## Change classes

Every commit body carries one. The register's amendments key off them.

| Class | Meaning |
|---|---|
| **BP** | behaviour-preserving |
| **CF** | contract fix |
| **SC** | semantic change |
| **API** | accepted-input-surface change |
| **IV** | verification infrastructure |
| **DOC** | documentation |

There are **six** classes, not five. `API` is in the list because it is in
use: A-45 is classified `SC/API`, and a vocabulary that omitted it would have
rejected a commit the register already accepted. Compound classes are allowed
in either order (`IV/DOC` and `DOC/IV` both occur). Read `git log` for
precedent before committing. Commit bodies explain the bug, the change, and
the verification.

## Commit trailers

```
Change-Class: BP | CF | SC | API | IV | DOC
Gate-Lane: fast | full | none (documentation-only profile)
Pins: canonical <verdict>; polygon <verdict>   # omit when not run, with the reason in the body
```

A `commit-msg` hook **rejects** a missing or misspelled `Change-Class`.
`Gate-Lane` and `Pins` are warned about but not rejected: their correct value
depends on what the commit touched, and a hook that guessed would only teach
people to pass `--no-verify`, which would disable the part that works.

Install once per clone -- git config is local and hooks are not cloned:

```bash
git config core.hooksPath tools/hooks
```

The hooks live in a tracked directory rather than in `.git/hooks`, so they are
version-controlled and a stale one shows up in a diff instead of hiding in a
clone. Check the hook still discriminates in both directions after editing it:

```bash
tools/hooks/test_commit_msg.sh
```

Merge, revert, `fixup!` and `squash!` messages are exempt, recognised by the
shape of the generated message rather than by a flag, so the exemption cannot
be borrowed by a hand-written commit that would rather not declare a class.

## Testing (non-negotiable)

`pytest`, functions over classes, fixtures for expensive or shared objects such
as a sampled posterior or a simulated event set.

**RED -> GREEN -> refactor.** Write the test first and observe it fail against
the pre-change state. Bug fixes are RED-first: demonstrate the new row failing
against the pre-change state and **commit the capture as evidence alongside the
fix**. A separate test-only commit is not required and is not what recent
series do.

**The revert must be minimal.** Revert only what the row is meant to detect.
Reverting shared API alongside the defect yields an `ImportError` at
collection, which is a red that proves nothing.

A row that passes on both sides is **non-discriminating by construction** and
is recorded as such, never offered as evidence.

### Forbidden

- **Editing a test just to make it pass.** Test changes must reflect a real
  requirement change or a genuine bug in the test -- flag these explicitly.
- **Simplifying the problem**, mocking away the real implementation, or taking
  the happy path to get green. A test must fail for anything short of the full
  correct behaviour.
- **Loosening a tolerance to fix a failing fit.**

### Numerical tolerance

**Never compare floats with `==`.** Use `np.allclose` or `pytest.approx` with
tolerances calibrated to the scale of the quantity. Intensities,
log-likelihoods and GP draws span many orders of magnitude, so a fixed `atol`
will silently pass or spuriously fail.

Commands, the two-lane suite, and the pin baselines are in `AGENTS.md`.

## Evidence and provenance

**D-41 governs this and lives in `phase3_record.tex` (A-31). Read it before the
first commit of any series; what follows is a pointer, not a restatement.**

Two that are cheap to get wrong:

- **`git status --porcelain` shows ` M` for *unstaged*.** One bad pathspec
  aborts the whole `git add`, and with stderr discarded it aborts silently.
- **After committing, `git status --porcelain` must be empty and
  `git show --stat HEAD` must be compared against the file list stated before
  committing.** A commit whose contents were not compared to a pre-stated list
  is ungated regardless of what the gates said. Gates run on a worktree; a
  commit is a claim about a subset of it.

**A green suite alone is not an acceptance record.** The two-lane suite,
the pin baselines, and the gate-profile rules live in `AGENTS.md`.

## Documentation

- **Guidance** -- `AGENTS.md`, this file, and `docs/` -- states what is true
  *now* and is **edited in place**.
- **The record** -- `phase3_record.tex` and the archived deliberation -- states
  what *happened* and is **only appended**. When they disagree, the register is
  right and the guidance is stale.
- **Execution state** -- `docs/phase3f_completion_manifest.yaml` is the sole
  source of current Phase 3f execution truth (**D-53**). Do not restate WP
  status in a second page.

A change that removes, renames or alters the meaning of a consumed `args` key
is recorded in `CHANGELOG.md` **when it lands**, not reconstructed at release.

## Working with an agent on this repository

- Work in small focused steps. Keep commits granular so any step can be
  reverted. Use **commit -> clear context -> reload** at each natural
  breakpoint.
- **Do not gold-plate or scope-creep.** Solve exactly the stated refactor: no
  new features, no premature abstraction, no speculative generality.
- **If you loop, whack-a-mole, or cannot fix something after a couple of tries,
  stop and say so** rather than simplifying the problem or claiming success.
  Going in circles means the approach is wrong, not that it needs one more
  patch.
- **Flag, do not hide**: outdated or hallucinated APIs, remnants of old code
  left after a rewrite, inappropriate pattern imitation, and **any place a
  change alters numeric output**.
- Never run in `--dangerously-skip-permissions` mode against this repository.
