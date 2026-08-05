"""A-28 four-config pin comparison against the 2026-07 baseline.

Same walker as results/_a27_pin_compare.py; only the candidate path differs.
"""
import json
from pathlib import Path


def load(p: str):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


a = load("results/_a28_pins_candidate.json")
b = load("refactor-patches/baselines-2026-07/pins.json")
diffs = []


def walk(x, y, path=""):
    if type(x) != type(y):
        diffs.append((path, type(x).__name__, type(y).__name__))
        return
    if isinstance(x, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x:
                diffs.append((path + "." + k, "MISSING", y[k]))
            elif k not in y:
                diffs.append((path + "." + k, x[k], "MISSING"))
            else:
                walk(x[k], y[k], path + "." + k)
    elif isinstance(x, list):
        if len(x) != len(y):
            diffs.append((path, f"len{len(x)}", f"len{len(y)}"))
        else:
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
    else:
        if x != y:
            diffs.append((path, x, y))


for cfg in sorted(set(a) | set(b)):
    if cfg not in a:
        print(f"{cfg}: MISSING FROM CANDIDATE")
    elif cfg not in b:
        print(f"{cfg}: NEW IN CANDIDATE")
    else:
        before = len(diffs)
        walk(a[cfg], b[cfg], cfg)
        print(f"{cfg}: {'MATCH' if len(diffs) == before else 'DRIFT'}")

print()
print("PIN_DIFFS", len(diffs), "MATCH" if not diffs else "DRIFT")
for d in diffs[:20]:
    print(d)
