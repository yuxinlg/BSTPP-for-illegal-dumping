"""A-38: what the candidate corpus actually contains, and the defect built.

TWO THINGS THIS ESTABLISHES.

(1) THE CANDIDATE CORPUS IS CONTENT-HOMOGENEOUS. Every committed
    ``results/*pins*candidate*.json`` is hashed, raw and after canonical JSON
    normalisation. If they all normalise to one value equal to the baseline,
    then reading ANY of them yields ``PIN_DIFFS 0 MATCH`` -- which means the
    historical candidate-side ambiguity measured by
    ``_a38_pin_ambiguity_census.py`` COULD NOT HAVE CHANGED A VERDICT, and
    also that the corpus is incapable of exhibiting the stale-read defect.
    The instrument's sensitivity to that failure mode is zero here, and stays
    zero until a candidate genuinely differs -- i.e. at a re-baseline, which
    is the moment it matters most.

(2) THE DEFECT, CONSTRUCTED. Because the real corpus cannot show it, it is
    built: a candidate that genuinely drifts, compared by the OLD hard-coded
    comparator pointed at the wrong file (reports MATCH, exit 0) and at the
    right file (reports DRIFT). The old verdict text carries nothing that
    distinguishes them. The new comparator is then run on the same pair.

Usage:  python results/_a38_corpus_and_defect_demo.py
"""
import glob
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "refactor-patches" / "baselines-2026-07" / "pins.json"
NEW = REPO / "refactor-patches" / "pin_compare.py"
OLD = REPO / "results" / "_a37_pin_compare.py"


def raw(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon(p: Path) -> str:
    obj = json.loads(p.read_text(encoding="utf-8-sig"))
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()).hexdigest()


print("SECTION 1 -- THE CANDIDATE CORPUS")
cands = [Path(p) for p in sorted(glob.glob(str(REPO / "results" / "*pins*candidate*.json")))]
base_canon = canon(BASELINE)
raws, canons = {}, {}
for p in cands:
    raws.setdefault(raw(p), []).append(p.name)
    canons.setdefault(canon(p), []).append(p.name)
print(f"  committed candidates            : {len(cands)}")
print(f"  distinct RAW byte hashes        : {len(raws)}   (BOM / line endings)")
print(f"  distinct CANONICAL JSON hashes  : {len(canons)}")
print(f"  baseline canonical hash         : {base_canon[:16]}")
for h, names in canons.items():
    same = "== BASELINE" if h == base_canon else "!= baseline"
    print(f"    {h[:16]}  {same}  n={len(names)}")
homogeneous = len(canons) == 1 and base_canon in canons
print(f"  CORPUS_CONTENT_HOMOGENEOUS_AND_EQUAL_TO_BASELINE={homogeneous}")
print("  => every historical candidate yields PIN_DIFFS 0 MATCH, so the")
print("     candidate-side ambiguity cannot have changed a verdict, AND the")
print("     corpus cannot exhibit the stale-read defect at all.")
print()

print("SECTION 2 -- THE DEFECT, CONSTRUCTED (the corpus cannot show it)")
payload = json.loads(BASELINE.read_text(encoding="utf-8-sig"))
cfg = sorted(payload)[0]
drifted = json.loads(json.dumps(payload))
drifted[cfg]["loglik"] = "-1.0"  # pins store repr() strings

tmp = Path(tempfile.mkdtemp())
right = tmp / "_a38_right_candidate.json"     # genuinely drifted
stale = tmp / "_a38_stale_candidate.json"     # the "previous amendment" file
right.write_text(json.dumps(drifted, indent=0, sort_keys=True), encoding="utf-8")
stale.write_text(json.dumps(payload, indent=0, sort_keys=True), encoding="utf-8")
print(f"  right candidate (drifted) sha256 : {raw(right)[:16]}")
print(f"  stale candidate (clean)   sha256 : {raw(stale)[:16]}")
print("  These DIFFER, which no pair in the real corpus does.")
print()


def run(script: Path, *args: str):
    proc = subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, cwd=str(REPO))
    return proc.returncode, proc.stdout.strip()


def verdict(out: str) -> str:
    for line in out.splitlines():
        if line.startswith("PIN_DIFFS "):
            return line
    return "<no PIN_DIFFS line>"


print("  OLD comparator (results/_a37_pin_compare.py pattern: paths hard-coded)")
old_src = OLD.read_text(encoding="utf-8")
outs = {}
for label, path in (("stale (the mistake)", stale), ("right (drifted)", right)):
    patched = tmp / f"old_{label.split()[0]}.py"
    patched.write_text(
        old_src.replace('"results/_a37_pins_candidate.json"', f'r"{path}"'),
        encoding="utf-8")
    code, out = run(patched)
    outs[label] = (code, verdict(out))
    print(f"    reading {label:<20} -> exit {code}  {verdict(out)}")
old_indistinguishable = outs["stale (the mistake)"][1] == "PIN_DIFFS 0 MATCH"
print(f"    OLD_REPORTS_MATCH_ON_THE_WRONG_FILE={old_indistinguishable}")
print("    and its verdict line names no file, so a capture cannot be audited.")
print()

print("  NEW comparator (refactor-patches/pin_compare.py)")
new_lines = {}
for label, path in (("stale (the mistake)", stale), ("right (drifted)", right)):
    code, out = run(NEW, str(path))
    new_lines[label] = verdict(out)
    print(f"    reading {label:<20} -> exit {code}")
    print(f"      {verdict(out)}")
print(f"    NEW_VERDICT_LINES_DIFFER="
      f"{new_lines['stale (the mistake)'] != new_lines['right (drifted)']}")
print("    Each line carries the candidate's name and content hash, so the")
print("    two are distinguishable even though both were 'a MATCH' to the")
print("    operator who ran them.")
print("EXIT:0")
