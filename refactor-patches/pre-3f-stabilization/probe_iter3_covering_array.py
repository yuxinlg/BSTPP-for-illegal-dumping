"""Generate a pairwise covering array over the nine Lane B axes; report coverage."""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

AXES9 = {
    "model_family": ("hawkes", "cox_hawkes", "lgcp"),
    "support": ("rectangle", "polygon"),
    "temporal_trigger": (
        "Temporal_Exponential", "Temporal_Power_Law", "custom",
    ),
    "spatial_trigger": ("Spatial_Symmetric_Gaussian", "custom"),
    "cutoff_input": (
        "tolerance", "physical", "omitted", "explicit_None_via_set_window",
    ),
    "entry_path": ("constructor", "set_window"),
    "builder_numerics": (
        "default_panel_gl", "guided_small_panel", "nondefault_gl_order",
    ),
    "standardization": ("none", "domain_area", "bool_rejected"),
    "sigma_bounds": (
        "both", "neither_rect", "polygon_min_required", "custom_spatial_rejects",
    ),
}
AXES6 = {k: AXES9[k] for k in (
    "model_family", "support", "temporal_trigger", "spatial_trigger",
    "cutoff_input", "entry_path",
)}

# Approximate iteration-2 hand points (partial assignments).
OLD_POINTS = [
    dict(model_family="hawkes", support="polygon", entry_path="constructor",
         spatial_trigger="Spatial_Symmetric_Gaussian"),
    dict(model_family="hawkes", support="polygon", entry_path="constructor"),
    dict(model_family="hawkes", support="polygon", entry_path="constructor",
         spatial_trigger="custom"),
    dict(model_family="hawkes", support="rectangle", entry_path="constructor",
         temporal_trigger="Temporal_Power_Law", cutoff_input="physical"),
    dict(model_family="hawkes", support="rectangle", entry_path="constructor",
         standardization="bool_rejected"),
    dict(model_family="lgcp", entry_path="set_window"),
    dict(model_family="hawkes", support="rectangle", entry_path="constructor",
         spatial_trigger="custom", sigma_bounds="custom_spatial_rejects"),
    dict(model_family="hawkes", support="rectangle", entry_path="constructor",
         temporal_trigger="Temporal_Exponential",
         spatial_trigger="Spatial_Symmetric_Gaussian", cutoff_input="physical"),
    dict(model_family="hawkes", support="rectangle", entry_path="constructor",
         temporal_trigger="Temporal_Exponential",
         spatial_trigger="Spatial_Symmetric_Gaussian", cutoff_input="tolerance"),
    dict(model_family="hawkes", support="rectangle", entry_path="constructor",
         temporal_trigger="Temporal_Exponential",
         spatial_trigger="Spatial_Symmetric_Gaussian", cutoff_input="omitted"),
    dict(model_family="hawkes", support="polygon", entry_path="constructor",
         temporal_trigger="Temporal_Exponential",
         spatial_trigger="Spatial_Symmetric_Gaussian", cutoff_input="omitted",
         builder_numerics="default_panel_gl",
         sigma_bounds="polygon_min_required"),
    dict(model_family="hawkes", support="polygon", entry_path="constructor",
         temporal_trigger="Temporal_Exponential",
         spatial_trigger="Spatial_Symmetric_Gaussian", cutoff_input="omitted",
         builder_numerics="guided_small_panel"),
    dict(model_family="cox_hawkes", support="rectangle", entry_path="constructor",
         temporal_trigger="Temporal_Exponential",
         spatial_trigger="Spatial_Symmetric_Gaussian", cutoff_input="omitted"),
    dict(model_family="cox_hawkes", support="polygon", entry_path="constructor",
         temporal_trigger="Temporal_Exponential",
         spatial_trigger="Spatial_Symmetric_Gaussian", cutoff_input="omitted"),
    dict(model_family="lgcp", support="rectangle", entry_path="constructor"),
    dict(model_family="hawkes", support="rectangle", entry_path="set_window",
         cutoff_input="physical"),
    dict(model_family="hawkes", support="polygon", entry_path="set_window",
         cutoff_input="physical"),
    dict(model_family="hawkes", support="rectangle", entry_path="set_window",
         cutoff_input="omitted"),
    dict(model_family="hawkes", support="rectangle", entry_path="set_window",
         cutoff_input="explicit_None_via_set_window"),
    dict(model_family="hawkes", support="polygon", entry_path="constructor",
         builder_numerics="default_panel_gl"),
]


def coverage(axes: dict, points: list[dict]) -> tuple[float, int, int]:
    names = list(axes)
    total = covered = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            need = set(itertools.product(axes[a], axes[b]))
            have = set()
            for p in points:
                if a in p and b in p:
                    have.add((p[a], p[b]))
            total += len(need)
            covered += len(have & need)
    return covered / total, covered, total


def greedy_ca(axes: dict, seed: int = 0, trials: int = 5000) -> list[dict]:
    rng = random.Random(seed)
    names = list(axes)
    uncovered: set[tuple] = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for la, lb in itertools.product(axes[a], axes[b]):
                uncovered.add((a, la, b, lb))
    rows: list[dict] = []
    while uncovered:
        best = None
        best_cov: set[tuple] = set()
        for _ in range(trials):
            row = {a: rng.choice(axes[a]) for a in names}
            cov = {
                (a, row[a], b, row[b])
                for i, a in enumerate(names)
                for b in names[i + 1:]
                if (a, row[a], b, row[b]) in uncovered
            }
            if len(cov) > len(best_cov):
                best, best_cov = row, cov
        if not best_cov:
            a, la, b, lb = next(iter(uncovered))
            row = {n: rng.choice(axes[n]) for n in names}
            row[a] = la
            row[b] = lb
            best = row
            best_cov = {
                (x, best[x], y, best[y])
                for i, x in enumerate(names)
                for y in names[i + 1:]
                if (x, best[x], y, best[y]) in uncovered
            }
        rows.append(dict(best))
        uncovered -= best_cov
    return rows


def main() -> int:
    c6, n6, t6 = coverage(AXES6, OLD_POINTS)
    c9, n9, t9 = coverage(AXES9, OLD_POINTS)
    print(f"OLD 6-axis pairwise: {c6:.4f} ({n6}/{t6})")
    print(f"OLD 9-axis pairwise: {c9:.4f} ({n9}/{t9})")
    print("Dropped from 6-axis pairwise (present in traceability nine):")
    print("  builder_numerics, standardization, sigma_bounds")

    best = None
    for seed in range(40):
        rows = greedy_ca(AXES9, seed=seed, trials=6000)
        if best is None or len(rows) < len(best):
            best = rows
            cc, _, _ = coverage(AXES9, rows)
            print(f"seed={seed} nrows={len(rows)} cov={cc}")

    assert best is not None
    cc, n, t = coverage(AXES9, best)
    assert cc == 1.0, cc
    out = Path(__file__).with_name("covering_array_rows.json")
    out.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"WROTE {out} nrows={len(best)} cov={cc} ({n}/{t})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
