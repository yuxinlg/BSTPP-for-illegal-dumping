"""A-38: a MATCH must be distinguishable from a MATCH against the wrong file.

The defect these rows enforce against is A-37's: a comparison read the
previous amendment's candidate path, reported ``PIN_DIFFS 0 MATCH``, and
described a different tree. The verdict text was IDENTICAL to a correct one,
so no reader of the capture could have told. The requirement is therefore not
"do not read the wrong file" -- that is unenforceable -- but "if you do, the
output says so".

RED at 8cf19fb: refactor-patches/pin_compare.py does not exist; the
comparators of that era (results/_a37_pin_compare.py and its predecessors)
hard-code both paths, print no provenance, and always exit 0.
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPARE = os.path.join(REPO, "refactor-patches", "pin_compare.py")
CANONICAL = os.path.join(
    REPO, "refactor-patches", "baselines-2026-07", "pins.json")


def _run(*args):
    proc = subprocess.run([sys.executable, COMPARE, *args],
                          capture_output=True, text=True, cwd=REPO)
    return proc.returncode, proc.stdout


def _verdict_line(out):
    for line in out.splitlines():
        if line.startswith("PIN_DIFFS "):
            return line
    raise AssertionError(f"no PIN_DIFFS line in output:\n{out}")


@pytest.fixture
def twin_baselines(tmp_path):
    """Two byte-identical copies of the canonical baseline in different files.

    Comparing the baseline against ITSELF is a MATCH by construction, so both
    runs below produce PIN_DIFFS 0 -- which is exactly the situation A-37 was
    in. The rows assert the two MATCHes are still tellable apart.
    """
    with open(CANONICAL, encoding="utf-8-sig") as fh:
        payload = json.load(fh)
    decoy = tmp_path / "decoy_pins.json"
    decoy.write_text(json.dumps(payload, indent=0, sort_keys=True),
                     encoding="utf-8")
    return decoy


def test_match_against_canonical_names_the_files_and_their_hashes():
    code, out = _run(CANONICAL)
    assert code == 0
    line = _verdict_line(out)
    assert line.startswith("PIN_DIFFS 0 MATCH")
    assert "canonical" in line
    # Both sides identified by name AND by content hash, in the verdict line.
    digest = hashlib.sha256(open(CANONICAL, "rb").read()).hexdigest()
    assert digest[:12] in line
    assert "candidate=" in line and "baseline=" in line
    assert "PIN_PROVENANCE" in out
    assert "baseline_source  : CANONICAL" in out


def test_match_against_a_wrong_baseline_is_distinguishable(twin_baselines):
    """THE FIRING ROW. Same verdict, different file -- and the output says so.

    Without the provenance block these two runs are the same six characters.
    """
    code_ok, out_ok = _run(CANONICAL)
    code_bad, out_bad = _run(CANONICAL, "--baseline", str(twin_baselines))

    # Both are MATCH: the decoy is byte-equivalent content in another file.
    assert code_ok == 0 and code_bad == 0
    assert _verdict_line(out_ok).startswith("PIN_DIFFS 0 MATCH")
    assert _verdict_line(out_bad).startswith("PIN_DIFFS 0 MATCH")

    # ... and yet the two verdict lines differ. That is the whole requirement.
    assert _verdict_line(out_ok) != _verdict_line(out_bad), (
        "a MATCH against a non-canonical baseline is indistinguishable from a "
        "MATCH against the canonical one -- this is the A-37 defect")
    assert "NON-CANONICAL" in _verdict_line(out_bad)
    assert "NON-CANONICAL" not in _verdict_line(out_ok)
    assert "canonical_would_be" in out_bad


def test_candidate_side_is_identified_not_only_the_baseline(twin_baselines):
    """A-37's stale read was of a CANDIDATE path, so that side must speak too.

    Two runs sharing one baseline and differing only in candidate must have
    different verdict lines.
    """
    _, out_a = _run(CANONICAL)
    _, out_b = _run(str(twin_baselines))
    assert _verdict_line(out_a) != _verdict_line(out_b)
    assert "decoy_pins.json" in _verdict_line(out_b)


def test_drift_exits_nonzero_and_still_prints_provenance(tmp_path):
    """A capture records the comparison's own status (D-41), and a DRIFT is
    as much in need of provenance as a MATCH."""
    with open(CANONICAL, encoding="utf-8-sig") as fh:
        payload = json.load(fh)
    cfg = sorted(payload)[0]
    payload[cfg]["loglik"] = "0.0"  # pins store repr() strings
    moved = tmp_path / "moved_pins.json"
    moved.write_text(json.dumps(payload, indent=0, sort_keys=True),
                     encoding="utf-8")

    code, out = _run(str(moved))
    assert code == 1, "DRIFT must not exit 0"
    assert _verdict_line(out).startswith("PIN_DIFFS 1 DRIFT")
    assert "PIN_PROVENANCE" in out
    assert "moved_pins.json" in _verdict_line(out)


def test_stale_candidate_beside_a_real_drift_is_readable_from_the_capture(
        tmp_path):
    """A-40: THE FINE ROW -- it fails for a reason other than "no instrument".

    The five rows above all go red at 8cf19fb for one coarse reason, the file
    not existing, which A-38 recorded as weak evidence. This row is red under
    a MINIMAL revert instead: the instrument PRESENT, the canonical baseline
    PRESENT, a stale candidate path, and values that GENUINELY DIFFER -- the
    exact A-37 situation, with the one ingredient the real corpus has never
    had. Nothing about it needs the file to be absent, so a capture holding
    both states discriminates wrong-file from no-file.

    What it requires is precisely the self-describing verdict: reading the
    stale file yields a legitimate ``PIN_DIFFS 0 MATCH`` (that candidate
    really does equal the baseline), while the file that should have been
    read reports DRIFT. No instrument can stop the operator typing the wrong
    path. The capture must let a LATER READER see which was typed.
    """
    with open(CANONICAL, encoding="utf-8-sig") as fh:
        payload = json.load(fh)
    cfg = sorted(payload)[0]
    drifted = json.loads(json.dumps(payload))
    drifted[cfg]["loglik"] = "-1.0"                  # pins store repr() strings

    right = tmp_path / "_a40_right_candidate.json"   # what should be compared
    stale = tmp_path / "_a40_stale_candidate.json"   # last amendment's path
    right.write_text(json.dumps(drifted, indent=0, sort_keys=True),
                     encoding="utf-8")
    stale.write_text(json.dumps(payload, indent=0, sort_keys=True),
                     encoding="utf-8")
    assert hashlib.sha256(right.read_bytes()).digest() != \
        hashlib.sha256(stale.read_bytes()).digest(), "the fixture proves nothing"

    code_stale, out_stale = _run(str(stale))
    code_right, out_right = _run(str(right))

    # The regression is real and the wrong file conceals it.
    assert (code_stale, code_right) == (0, 1)
    assert _verdict_line(out_stale).startswith("PIN_DIFFS 0 MATCH")
    assert _verdict_line(out_right).startswith("PIN_DIFFS 1 DRIFT")

    # THE REQUIREMENT: the concealing capture identifies the file it read, by
    # name AND by content hash, so the mistake is visible after the fact.
    stale_line = _verdict_line(out_stale)
    assert "_a40_stale_candidate.json" in stale_line
    assert hashlib.sha256(stale.read_bytes()).hexdigest()[:12] in stale_line
    assert "_a40_right_candidate.json" not in stale_line
    assert str(stale) in out_stale, "the provenance block must give the path"


def test_missing_file_is_an_error_not_a_silent_match(tmp_path):
    code, out = _run(str(tmp_path / "does_not_exist.json"))
    assert code == 1
    assert "PIN_COMPARE_ERROR" in out
    assert "PIN_DIFFS 0 MATCH" not in out


# --------------------------------------------------------------------------
# A-48 / OP-31: the verdict line must carry its own POPULATION.
#
# A-37's defect was a verdict that did not say which FILES it read. This is
# the same defect one axis over: a verdict that does not say how many
# CONFIGURATIONS it compared. The walker treats a key present on one side only
# as no diff, so a six-configuration candidate read against the four-
# configuration canonical baseline compared four, found nothing, and printed
# the same `PIN_DIFFS 0 MATCH` a complete comparison prints. `NEW IN
# CANDIDATE` appeared above the verdict, where a runbook grepping the verdict
# line never sees it.
# --------------------------------------------------------------------------


def _variant(tmp_path, name, mutate):
    """A candidate derived from the canonical baseline by one key-set edit."""
    with open(CANONICAL, encoding="utf-8-sig") as fh:
        payload = json.load(fh)
    mutate(payload)
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=0, sort_keys=True),
                    encoding="utf-8")
    return path


def test_a_complete_comparison_states_its_population_on_the_verdict_line():
    """The population is unconditional: it is on clean lines too.

    A field printed only in the interesting case makes 'not printed' and 'not
    run' the same observable, which is the A-35 lesson this file already
    applies to the provenance block.
    """
    with open(CANONICAL, encoding="utf-8-sig") as fh:
        n = len(json.load(fh))
    _, out = _run(CANONICAL)
    line = _verdict_line(out)
    assert f"compared={n}/{n}" in line
    assert "candidate_only=[]" in line
    assert "baseline_only=[]" in line


def test_a_candidate_key_the_baseline_lacks_is_not_an_unqualified_match(
        tmp_path):
    """THE FIRING ROW for OP-31. This is the A-47 situation exactly.

    Pin 5 added two configurations the canonical baseline does not carry. Run
    against that baseline the comparison covers four of six and is silent on
    the other two -- which is the polygon regime, i.e. the whole reason pin 5
    was built. The line must not be readable as covering six.
    """
    with open(CANONICAL, encoding="utf-8-sig") as fh:
        n = len(json.load(fh))
    cand = _variant(tmp_path, "wider_pins.json",
                    lambda p: p.update({"zz_new_config": {"loglik": "1.0"}}))
    code, out = _run(str(cand))
    line = _verdict_line(out)

    assert "MATCH" not in line, (
        "a comparison that did not compare every configuration must not print "
        "the word a complete comparison prints; this is OP-31")
    assert "PARTIAL" in line
    assert f"compared={n}/{n + 1}" in line
    assert "candidate_only=[zz_new_config]" in line
    # Surfaced IN the verdict, not only above it (the per-config lines stay).
    assert "zz_new_config: NEW IN CANDIDATE" in out
    # Exit status is deliberately unchanged: an expected re-baseline must not
    # make this gate permanently red. The verdict WORD carries the fact.
    assert code == 0


def test_a_baseline_key_the_candidate_lacks_is_named_in_the_verdict(tmp_path):
    """The mirror hole, which is worse and was equally silent.

    A harness that stopped emitting a configuration loses that coverage
    entirely, and the walker counted that as no diff too.
    """
    with open(CANONICAL, encoding="utf-8-sig") as fh:
        dropped = sorted(json.load(fh))[0]
    cand = _variant(tmp_path, "narrower_pins.json",
                    lambda p: p.pop(dropped))
    _, out = _run(str(cand))
    line = _verdict_line(out)
    assert "MATCH" not in line
    assert f"baseline_only=[{dropped}]" in line
    assert "compared=3/4" in line


def test_the_real_six_config_candidate_is_partial_against_canonical():
    """Not a fixture -- the committed A-47 capture, against the real baseline.

    A synthetic key proves the walker's arithmetic. This proves the thing the
    runbook actually does, and it is the line AGENTS.md now requires be read
    twice, once per baseline.
    """
    cand = os.path.join(REPO, "results", "_a47_pins_candidate.json")
    if not os.path.isfile(cand):
        pytest.skip("A-47 capture absent; machine-local artifact")
    _, out = _run(cand)
    line = _verdict_line(out)
    assert "compared=4/6" in line
    assert "MATCH" not in line
    for key in ("hawkes_notched_4to1_polygon_mode",
                "hawkes_notched_4to1_rectangle_mode"):
        assert key in line

    forward = os.path.join(REPO, "refactor-patches",
                           "baselines-2026-08-polygon", "pins.json")
    _, out_fwd = _run(cand, "--baseline", forward)
    fwd_line = _verdict_line(out_fwd)
    assert "compared=6/6" in fwd_line
    assert "PIN_DIFFS 0 MATCH" in fwd_line
