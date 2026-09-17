#!/bin/sh
#
# Self-test for the commit-msg hook: BOTH DIRECTIONS.
#
# A hook demonstrated only on the message it rejects has not been shown to
# accept anything, and one demonstrated only on the message it accepts has not
# been shown to enforce anything (D-41 clauses 1 and 3). Each case below
# records the hook's OWN exit status.
#
#     tools/hooks/test_commit_msg.sh
#
set -u

HOOK="$(dirname "$0")/commit-msg"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0

# check <expected-exit> <name> <message>
check() {
  want="$1"; name="$2"; msg="$3"
  printf '%s\n' "$msg" > "$TMP/msg"
  out="$(sh "$HOOK" "$TMP/msg" 2>&1)"
  got=$?
  if [ "$got" = "$want" ]; then
    printf 'PASS  exit=%s  %s\n' "$got" "$name"
    pass=$((pass + 1))
  else
    printf 'FAIL  exit=%s want=%s  %s\n' "$got" "$want" "$name"
    printf '%s\n' "$out" | sed 's/^/        /'
    fail=$((fail + 1))
  fi
}

echo "commit-msg hook discrimination"
echo

echo "-- must REJECT (exit 1) --"
check 1 "no trailer at all" \
  "fix(x): something

A body with no trailer."

check 1 "class not in the vocabulary" \
  "fix(x): something

Change-Class: REFACTOR"

check 1 "lowercase trailer name (git trailers are case-sensitive)" \
  "fix(x): something

change-class: BP"

check 1 "class only in the subject, the old convention" \
  "fix(x): something (IV/DOC)

A body with no trailer line."

check 1 "class present only inside a comment line" \
  "fix(x): something

# Change-Class: BP"

check 1 "compound class with one invalid half" \
  "fix(x): something

Change-Class: IV/NOPE"

echo
echo "-- must ACCEPT (exit 0) --"
check 0 "single class" \
  "fix(x): something

Change-Class: BP
Gate-Lane: fast
Pins: 4/6 canonical PARTIAL + 6/6 polygon MATCH"

check 0 "compound class" \
  "docs(x): something

Change-Class: IV/DOC
Gate-Lane: none
Pins: not run (no gate-read path in diff)"

check 0 "each of the six classes: CF" \
  "fix(x): y

Change-Class: CF"

check 0 "each of the six classes: SC" \
  "feat(x): y

Change-Class: SC"

check 0 "each of the six classes: DOC" \
  "docs(x): y

Change-Class: DOC"

# Pinned as its own case. The hook was first written against the five classes
# the agent entrypoint summarised, and this is the class that was missing: it
# would have rejected A-45, which the register classifies SC/API. The case
# exists so the vocabulary cannot be narrowed back without a red.
check 0 "API, the class the entrypoint summary omitted (A-45 is SC/API)" \
  "feat(x): y

Change-Class: API"

check 0 "compound with API, as A-45 is classified" \
  "feat(x): y

Change-Class: SC/API"

check 0 "DOC/IV, the reversed order the register also uses" \
  "docs(x): y

Change-Class: DOC/IV"

check 0 "missing Gate-Lane and Pins are advisory, not fatal" \
  "fix(x): something

Change-Class: BP"

check 0 "merge commit is exempt by message shape" \
  "Merge branch 'refactor' into main"

check 0 "revert is exempt by message shape" \
  "Revert \"fix(x): something\""

check 0 "fixup is exempt by message shape" \
  "fixup! fix(x): something"

echo
echo "TOTAL pass=$pass fail=$fail"
[ "$fail" = 0 ] || exit 1
echo "ALL CASES BEHAVE -- the hook rejects what it must and accepts what it must."
