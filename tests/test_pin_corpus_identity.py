"""A-40: the corpus-identity property, checked every commit instead of once.

A-38's retroactive rescue of twenty-four historical ``PIN_DIFFS 0`` claims
rests entirely on one measurement: every pin candidate in the tree normalises
to a single canonical-JSON hash, equal to the baseline's. With one content
group the candidate-side ambiguity could not have changed a verdict. That is a
fact about the data, and it expires at the first re-baseline.

These rows exist so the expiry is observed AT the commit that causes it. The
expected trigger is already scheduled: OP-24's fifth pinned configuration in
polygon mode adds a configuration key the baseline does not carry, so the
first candidate holding it will not normalise to the baseline's hash. When
that lands, this file goes red, the record says the rescue has expired as of
that commit, and the baseline is re-declared. Relaxing the assertion instead
would be the whole point missed.

RED at 3a9eafb: refactor-patches/pin_corpus_identity.py does not exist.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(REPO, "refactor-patches", "pin_corpus_identity.py")


def _run():
    proc = subprocess.run([sys.executable, GATE],
                          capture_output=True, text=True, cwd=REPO)
    return proc.returncode, proc.stdout


def _field(out, key):
    """The raw right-hand side, population label and all."""
    for line in out.splitlines():
        if line.strip().startswith(key):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no {key} line in output:\n{out}")


def _value(out, key):
    """The value with its trailing [POPULATION] label removed (A-41)."""
    return _field(out, key).split("[")[0].strip()


def test_corpus_is_one_content_group_equal_to_the_baseline():
    """THE EXPIRY GATE. Red here means the property has lapsed, not that the

    pins have drifted -- pin_compare.py is what says that.
    """
    code, out = _run()
    distinct = int(_value(out, "DISTINCT_CANONICAL_HASHES"))
    assert distinct == 1, (
        f"the pin candidate corpus now holds {distinct} distinct content "
        "groups. A-38's retroactive rescue of the 24 historical PIN_DIFFS 0 "
        "claims covered only a corpus with ONE group, so from this commit "
        "forward it applies only to commits before this one. Re-baseline and "
        "record the expiry. If this is OP-24's polygon-mode pin, it is the "
        "expected trigger and the correct response is still to record it.")
    assert _value(out, "IDENTITY_HOLDS") == "True"
    assert code == 0


def test_the_gate_reports_the_distinct_count_whether_or_not_it_passes():
    """A gate that printed only on failure would make 'not printed' and 'not

    run' the same observable (the A-35 lesson). The count is the artifact.
    """
    _, out = _run()
    assert "DISTINCT_CANONICAL_HASHES" in out
    assert "IDENTITY_HOLDS" in out
    assert int(_value(out, "population ON_DISK").split()[0]) > 0
    # The population must be stated, not implied: an identity over an
    # unstated set of files is not a measurement.
    assert "pattern" in out and "baseline canonical hash" in out


def test_population_is_a_superset_of_the_tracked_set():
    """The stronger-claim argument depends on this, so it is measured."""
    _, out = _run()
    assert _field(out, "tracked but not on disk").startswith("none")


def test_the_gate_names_which_population_it_measured():
    """A-41. The gate reads the working tree, so a clone measures the tracked

    subset and reports a smaller denominator for the SAME property. Both
    numbers are right; a run that printed one unlabelled would make a CI line
    and a local line look like a disagreement. Same discipline as the ASCII
    sweep's raise-site denominator (A-39).
    """
    _, out = _run()
    assert _field(out, "VERDICT_POPULATION") == "ON_DISK"
    on_disk = int(_value(out, "population ON_DISK").split()[0])
    tracked = int(_value(out, "population TRACKED").split()[0])
    assert on_disk >= tracked > 0
    # The property, not just the count, is reported for both.
    assert _value(out, "DISTINCT_CANONICAL_HASHES_TRACKED").isdigit()
    assert _value(out, "IDENTITY_HOLDS_TRACKED") == "True"
    # Every count carries its population label, so no bare number can be
    # read against the wrong denominator.
    for key in ("DISTINCT_CANONICAL_HASHES", "IDENTITY_HOLDS"):
        assert "[ON_DISK]" in _field(out, key + " ")


def test_a_second_content_group_makes_the_gate_fail(tmp_path, monkeypatch):
    """The gate must be capable of failing. A tripwire never shown to trip is

    indistinguishable from an absent one (AGENTS.md: an unreached guard).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_pin_corpus_identity", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import json
    import shutil
    fake = tmp_path / "results"
    fake.mkdir()
    real = os.path.join(REPO, "results")
    names = [n for n in os.listdir(real)
             if n.endswith(".json") and "pins" in n and "candidate" in n]
    assert names, "no candidates to copy; the fixture would prove nothing"
    shutil.copy(os.path.join(real, names[0]), fake / names[0])
    # ... plus one that genuinely differs, which the real corpus has none of.
    payload = json.loads((fake / names[0]).read_text(encoding="utf-8-sig"))
    cfg = sorted(payload)[0]
    payload[cfg]["loglik"] = "0.0"          # pins store repr() strings
    (fake / "_drifted_pins_candidate.json").write_text(
        json.dumps(payload, indent=0, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr(mod, "RESULTS", fake)
    m = mod.measure()
    assert m["distinct"] == 2
    assert m["holds"] is False
    # ... and the exit status follows, so a capture records the lapse.
    assert mod.main() == 1
