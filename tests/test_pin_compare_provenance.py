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


def test_missing_file_is_an_error_not_a_silent_match(tmp_path):
    code, out = _run(str(tmp_path / "does_not_exist.json"))
    assert code == 1
    assert "PIN_COMPARE_ERROR" in out
    assert "PIN_DIFFS 0 MATCH" not in out
