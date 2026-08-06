"""A-40: the corpus-identity property, checked every commit instead of once.

A-38's retroactive rescue of twenty-four historical ``PIN_DIFFS 0`` claims
rested entirely on one measurement: every pin candidate in the tree normalises
to a single canonical-JSON hash, equal to the baseline's. With one content
group the candidate-side ambiguity could not have changed a verdict. That is a
fact about the data, and it expires at the first re-baseline.

These rows existed so the expiry would be observed AT the commit that causes
it. **It was, and this is that file after the event.** The trigger was the one
A-40 scheduled -- OP-24's polygon-mode pin adds configuration keys the 2026-07
baseline does not carry -- and the sequence went as designed: this file went
red, the red was captured
(``refactor-patches/captures/a47_corpus_identity_test_red.log``, exit 1), the
register recorded at A-47 that the rescue covers commits before A-47 and
nothing after, and **the baseline was then re-declared, which is what A-40
said to do and is not the relaxation A-40 warned against.**

The distinction, because it is the whole point. A-40 forbade weakening the
assertion to accommodate the new group -- ``distinct <= 2``, or deleting the
row. What replaced it is the same sentence over a two-era population: every
content group must equal one of the DECLARED baselines, an undeclared group is
red, and the number of declared eras is itself asserted below so that a third
baseline cannot be slipped in to make a group go away. What is NOT recovered
is the rescue: two content groups mean a ``PIN_DIFFS 0`` artifact that does not
name its baseline is ambiguous, and no assertion here can fix an artifact
written earlier (OP-31).

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


def test_every_content_group_belongs_to_a_declared_baseline():
    """THE DRIFT GATE, in its post-A-47 form. Red here means some capture in

    results/ was produced by a harness or a tree no declared baseline
    describes -- not that the pins have drifted, which pin_compare.py says.
    """
    code, out = _run()
    undeclared = int(_value(out, "UNDECLARED_GROUPS"))
    assert undeclared == 0, (
        f"{undeclared} content group(s) in the pin candidate corpus match no "
        "declared baseline. The gate names the files under the UNDECLARED "
        "marker. Explain the group or re-declare deliberately, recording the "
        "re-declaration in the register with the commit that causes it; do "
        "not add a baseline entry merely to make this row green.")
    assert _value(out, "IDENTITY_HOLDS") == "True"
    assert code == 0


def test_the_declared_era_count_is_itself_pinned():
    """The one edit to this gate that CAN be a relaxation is declaring a new

    baseline to absorb a group instead of explaining it. That edit is
    therefore not silent: it fails here until the number is updated
    deliberately, alongside the register entry that justifies it.
    """
    _, out = _run()
    assert _value(out, "DECLARED_BASELINES") == "2", (
        "the declared-era count moved. Two eras are declared: the 2026-07 "
        "canonical baseline, and the 2026-08 polygon forward baseline added "
        "at A-47 when OP-24's fifth pin landed. A third is a re-declaration "
        "and needs a register amendment, not just a passing test.")
    assert "2026-07 canonical" in out
    assert "2026-08 polygon (A-47)" in out


def test_the_gate_states_the_expiry_on_every_capture_including_green_ones():
    """A green line here used to mean 'one content group', which is what

    A-38's rescue rested on. It no longer does, and a capture that did not say
    so would let a later reader take the old meaning from the new pass -- the
    silent-shift class this gate was built to prevent.
    """
    code, out = _run()
    assert code == 0
    assert "A-38's retroactive rescue EXPIRED at A-47" in out
    assert "is not restored by" in out


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


def test_an_undeclared_content_group_makes_the_gate_fail(tmp_path, monkeypatch):
    """The gate must be capable of failing. A tripwire never shown to trip is

    indistinguishable from an absent one (AGENTS.md: an unreached guard). This
    row is the reason the A-47 re-declaration is not a relaxation: the fixture
    below is a candidate that matches NEITHER declared baseline, and it is
    still red.
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
    assert len(m["undeclared"]) == 1
    assert m["undeclared"][0] not in m["declared"]
    assert m["holds"] is False
    # ... and the exit status follows, so a capture records the lapse.
    assert mod.main() == 1
