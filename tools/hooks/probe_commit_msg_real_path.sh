#!/bin/sh
# A-56 -- demonstrate the commit-msg hook on the REAL commit path.
#
# The self-test invokes the hook directly. That shows the logic is right and
# says nothing about whether git actually calls it. A guard shown only inside
# its own harness has not been shown to be reachable, which is A-27's finding.
# This drives it through `git commit` itself.
set -u
cd "$(dirname "$0")/../.." || exit 1

echo "A-56 -- commit-msg hook on the real commit path"
echo
echo "\$ git config core.hooksPath"
git config core.hooksPath
echo

printf 'chore: demonstrate the hook on a real commit\n\nThis message deliberately carries no Change-Class trailer.\n' > results/_a56_badmsg.txt
echo "scratch" > results/_a56_probe_scratch.txt
git add -- results/_a56_probe_scratch.txt

head_before="$(git rev-parse --short HEAD)"

echo "--- REJECT: a real 'git commit' with no Change-Class trailer ---"
git commit -F results/_a56_badmsg.txt > results/_a56_out.txt 2>&1
rc=$?
sed 's/^/    /' results/_a56_out.txt
echo "    GIT COMMIT EXIT:$rc      (the git process's own status)"
echo

head_after="$(git rev-parse --short HEAD)"
echo "HEAD before : $head_before"
echo "HEAD after  : $head_after"
if [ "$head_before" = "$head_after" ]; then
  echo "UNCHANGED -- the commit did not land."
else
  echo "MOVED -- the hook did NOT stop the commit. This is a failure."
fi
echo

git reset -q HEAD -- results/_a56_probe_scratch.txt
rm -f results/_a56_probe_scratch.txt results/_a56_badmsg.txt results/_a56_out.txt

# ACCEPT, in a throwaway repository so a real commit can land.
#
# NOT demonstrated with `git commit --dry-run`: --dry-run does not invoke the
# commit-msg hook at all, so its exit 0 would show the hook was never called
# and be read as the hook approving. A capture that cannot distinguish "passed"
# from "never ran" is the non-discriminating-by-construction case D-41 clause 3
# forbids offering as evidence.
echo "--- ACCEPT: a real commit, in a scratch repository, hook installed ---"
hookdir="$(pwd)/tools/hooks"
scratch="$(mktemp -d)"
(
  cd "$scratch" || exit 1
  git init -q .
  git config user.email probe@example.invalid
  git config user.name probe
  git config core.hooksPath "$hookdir"
  echo x > f.txt
  git add f.txt
  printf 'chore: a message that declares its class\n\nChange-Class: IV\nGate-Lane: none\nPins: not run (probe)\n' > m.txt
  git commit -q -F m.txt > out.txt 2>&1
  rc=$?
  echo "    git commit EXIT:$rc      (the git process's own status)"
  sed 's/^/    /' out.txt
  n=$(git rev-list --count HEAD 2>/dev/null || echo 0)
  echo "    commits in scratch repo: $n   (1 = the commit landed)"

  echo
  echo "    and the SAME repo rejects the same change without the trailer:"
  echo y > g.txt
  git add g.txt
  printf 'chore: a message that does not declare its class\n' > m2.txt
  git commit -q -F m2.txt > out2.txt 2>&1
  rc2=$?
  echo "    git commit EXIT:$rc2      (nonzero = rejected)"
  n2=$(git rev-list --count HEAD 2>/dev/null || echo 0)
  echo "    commits in scratch repo: $n2   (still 1 = the second did not land)"
)
rm -rf "$scratch"
echo
echo "scratch removed; tracked working tree:"
git status --porcelain | grep -v '^??' | sed 's/^/    /'
