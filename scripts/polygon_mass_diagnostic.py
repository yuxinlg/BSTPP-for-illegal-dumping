"""Phase 3 §13 polygon diagnostic: rectangle-vs-polygon excitation spatial mass.

Quantifies, on real Philadelphia geometries, the discrepancy between

  M_rect(s_j; sigma) -- the per-axis erf product over the region's bounding
      rectangle (the §5.12 excitation-compensator mass, unclipped), and
  M_poly(s_j; sigma) -- a high-accuracy reference of the Gaussian mass over
      the region polygon A itself,

reported as a function of sigma (not at a posterior estimate), per region.
Consumers: OP-9 accuracy target, 3d test tolerances, rectangle-mode guidance.

The polygon reference uses the divergence theorem: with
F = (0, phi_sigma(x - sx) * Phi_sigma(y - sy)),  div F = the Gaussian density,
so  M_poly = -oint_{dA, CCW} phi_sigma(x - sx) * Phi_sigma(y - sy) dx,
evaluated with Gauss-Legendre quadrature on edges subdivided to <= sigma/2.
The polygon is first clipped to the [s +/- CLIP_RADIUS_SIGMA * sigma] box
(neglected exterior mass < 4*erfc(CLIP_RADIUS_SIGMA/sqrt(2)) ~ 2.5e-15), so
cost is local and independent of region size. CCW exteriors add mass, CW
holes subtract with the same formula. Convergence is checked by re-running a
subsample at half the step and higher order.

Events farther than FAR_SIGMA * sigma from dA (interior distance) satisfy
M_rect - M_poly < 2*erfc(FAR_SIGMA/sqrt(2)) ~ 1e-14 (A contained in the box
implies dist(s, dA) <= dist(s, d-box)), so the reference is only evaluated on
a boundary-stratified subsample of near-boundary events; far events
contribute M_poly := M_rect to the (weighted) aggregate sums.

Usage (from repo root, illegal-dumping env):
  python scripts/polygon_mass_diagnostic.py --regions districts
  python scripts/polygon_mass_diagnostic.py --regions parks
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.special import erf, erfc
from shapely.geometry import box as shapely_box
from shapely.geometry.polygon import orient

REPO = Path(__file__).resolve().parents[1]

METRIC_CRS = "EPSG:26918"  # UTM 18N (metres); the events file's native CRS
EVENTS_PATH = REPO / "output" / "illegal_dumping_full.geojson"
DISTRICTS_PATH = REPO / "data" / "planning_districts.geojson"
PARKS_PATH = REPO / "output" / "all_parks_gdf.geojson"

SIGMA_GRID_M = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]  # §13 log-spaced grid

# Reference-integral accuracy knobs (justifications in module docstring).
CLIP_RADIUS_SIGMA = 8.0   # clip box half-width, in sigmas
FAR_SIGMA = 8.0           # beyond this boundary distance, M_poly := M_rect
GL_ORDER = 20             # Gauss-Legendre nodes per edge piece
MAX_STEP_SIGMA = 0.5      # max edge-piece extent, in sigmas
CONV_GL_ORDER = 32        # convergence-check re-run: higher order ...
CONV_MAX_STEP_SIGMA = 0.25  # ... and half the step
CONV_SAMPLE = 20          # events per (region, sigma) re-run for convergence

# Boundary-distance strata (in sigmas) for the reference subsample; the last
# stratum boundary is FAR_SIGMA -- events beyond it are handled analytically.
STRATA_SIGMA = [0.0, 1.0, 2.0, 3.0, FAR_SIGMA]
DEFAULT_STRATUM_CAP = 300

# D-21 square-cutoff tail mass eps_s = 1 - erf(w_s/(sqrt(2) sigma))^2,
# evaluated at these w_s/sigma ratios for the tolerance recommendation.
CUTOFF_RATIOS = (2.0, 3.0, 4.0)
# Production polygon-mass error should be dominated by the declared cutoff:
TOL_FRACTION_OF_EPS = 0.1


def gauss_legendre_01(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights mapped from [-1, 1] to [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(order)
    return (x + 1.0) / 2.0, w / 2.0


def ring_mass(coords: np.ndarray, sx: float, sy: float, sigma: float,
              max_step: float, glx: np.ndarray, glw: np.ndarray) -> float:
    """Line integral -oint phi(x-sx) Phi(y-sy) dx over one closed ring.

    Signed: CCW rings return (+) enclosed Gaussian mass, CW rings (-).
    Edges are subdivided so each piece spans <= max_step per axis (the
    integrand varies on scale sigma in both x and y).
    """
    x1, y1 = coords[:-1, 0], coords[:-1, 1]
    x2, y2 = coords[1:, 0], coords[1:, 1]
    dx, dy = x2 - x1, y2 - y1
    reps = np.maximum(
        np.ceil(np.maximum(np.abs(dx), np.abs(dy)) / max_step), 1
    ).astype(int)
    seg = np.repeat(np.arange(len(dx)), reps)
    within = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
    t0 = within / reps[seg]
    t1 = (within + 1) / reps[seg]
    xa, xb = x1[seg] + t0 * dx[seg], x1[seg] + t1 * dx[seg]
    ya, yb = y1[seg] + t0 * dy[seg], y1[seg] + t1 * dy[seg]
    xn = xa[:, None] + glx[None, :] * (xb - xa)[:, None]
    yn = ya[:, None] + glx[None, :] * (yb - ya)[:, None]
    phi = np.exp(-((xn - sx) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    cdf = 0.5 * (1.0 + erf((yn - sy) / (sigma * np.sqrt(2.0))))
    return -float(np.sum((phi * cdf) @ glw * (xb - xa)))


def gaussian_polygon_mass(poly, sx: float, sy: float, sigma: float,
                          max_step_sigma: float, gl_order: int) -> float:
    """High-accuracy Gaussian mass over polygon A, clipped to +/- 8 sigma."""
    r = CLIP_RADIUS_SIGMA * sigma
    clipped = poly.intersection(shapely_box(sx - r, sy - r, sx + r, sy + r))
    if clipped.is_empty:
        return 0.0
    glx, glw = gauss_legendre_01(gl_order)
    max_step = max_step_sigma * sigma
    total = 0.0
    parts = getattr(clipped, "geoms", [clipped])
    for part in parts:
        if part.geom_type != "Polygon" or part.is_empty:
            continue  # slivers can degenerate to lines/points: zero mass
        part = orient(part, 1.0)  # CCW exterior, CW holes
        total += ring_mass(np.asarray(part.exterior.coords), sx, sy, sigma,
                           max_step, glx, glw)
        for hole in part.interiors:
            total += ring_mass(np.asarray(hole.coords), sx, sy, sigma,
                               max_step, glx, glw)
    return total


def rect_mass(x: np.ndarray, y: np.ndarray, bounds: tuple, sigma: float) -> np.ndarray:
    """§5.12 per-axis erf product over the region bounding rectangle (unclipped)."""
    x_min, y_min, x_max, y_max = bounds
    s2 = sigma * np.sqrt(2.0)
    mx = 0.5 * (erf((x_max - x) / s2) + erf((x - x_min) / s2))
    my = 0.5 * (erf((y_max - y) / s2) + erf((y - y_min) / s2))
    return mx * my


def self_test() -> None:
    """On a rectangle A the polygon reference must equal the erf product."""
    rect = shapely_box(0.0, 0.0, 300.0, 150.0)
    rng = np.random.default_rng(0)
    for sigma in (10.0, 100.0):
        xs = rng.uniform(0, 300, 5)
        ys = rng.uniform(0, 150, 5)
        want = rect_mass(xs, ys, (0.0, 0.0, 300.0, 150.0), sigma)
        got = np.array([
            gaussian_polygon_mass(rect, sx, sy, sigma, MAX_STEP_SIGMA, GL_ORDER)
            for sx, sy in zip(xs, ys)
        ])
        if not np.allclose(got, want, rtol=0, atol=1e-12):
            raise AssertionError(
                f"self-test failed at sigma={sigma}: max err "
                f"{np.max(np.abs(got - want)):.3e}")


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, 0.5 * cw[-1])])


def stratified_sample(dist: np.ndarray, sigma: float, cap: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Indices and inverse-inclusion weights of a boundary-stratified sample.

    Strata are boundary-distance bands in units of sigma (STRATA_SIGMA);
    events beyond FAR_SIGMA*sigma are excluded (handled analytically).
    """
    edges = np.array(STRATA_SIGMA) * sigma
    idx_all, w_all = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        members = np.flatnonzero((dist >= lo) & (dist < hi))
        if len(members) == 0:
            continue
        take = members if len(members) <= cap else rng.choice(
            members, size=cap, replace=False)
        idx_all.append(take)
        w_all.append(np.full(len(take), len(members) / len(take)))
    if not idx_all:
        return np.array([], dtype=int), np.array([])
    return np.concatenate(idx_all), np.concatenate(w_all)


def load_regions(kind: str) -> gpd.GeoDataFrame:
    if kind == "districts":
        gdf = gpd.read_file(DISTRICTS_PATH).to_crs(METRIC_CRS)
        gdf["region"] = gdf["dist_name"]
    elif kind == "parks":
        gdf = gpd.read_file(PARKS_PATH).to_crs(METRIC_CRS)
        gdf["region"] = gdf["PUBLIC_NAME"]
    else:
        raise ValueError(f"unknown region kind: {kind!r}")
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        print(f"note: repairing {int(invalid.sum())} invalid region "
              f"geometries via make_valid: {gdf.loc[invalid, 'region'].tolist()}")
        repaired = gdf.loc[invalid, "geometry"].make_valid()
        # make_valid can return GeometryCollections: keep polygonal parts only
        gdf.loc[invalid, "geometry"] = repaired.apply(
            lambda g: shapely.unary_union([p for p in getattr(g, "geoms", [g])
                                           if p.geom_type in
                                           ("Polygon", "MultiPolygon")]))
        if not gdf.geometry.is_valid.all():
            raise ValueError("region geometry still invalid after make_valid")
    return gdf[["region", "geometry"]]


def load_events() -> gpd.GeoDataFrame:
    ev = gpd.read_file(EVENTS_PATH).to_crs(METRIC_CRS)
    n_missing = int(ev.geometry.is_empty.sum() + ev.geometry.isna().sum())
    if n_missing:
        print(f"note: dropping {n_missing} events with missing/empty geometry")
        ev = ev[~(ev.geometry.is_empty | ev.geometry.isna())]
    # some rows may be non-point (matching batch_park_fits.prepare_park_locs)
    ev = ev.set_geometry(ev.geometry.centroid)
    return ev[["geometry"]]


def run(kind: str, sigmas: list[float], cap: int, seed: int,
        out_root: Path) -> Path:
    self_test()
    regions = load_regions(kind)
    events = load_events()
    joined = gpd.sjoin(events, regions, how="inner", predicate="within")
    out_dir = out_root / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows, conv_rows = [], []
    tracemalloc.start()
    for _, reg in regions.iterrows():
        name, poly = reg["region"], reg["geometry"]
        pts = joined[joined["region"] == name].geometry
        n = len(pts)
        if n == 0:
            print(f"{name}: no events, skipped")
            continue
        x, y = pts.x.to_numpy(), pts.y.to_numpy()
        boundary = poly.boundary
        dist = shapely.distance(pts.values, boundary)
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        bbox_fill = poly.area / ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))
        n_vertices = shapely.get_num_coordinates(poly)

        for sigma in sigmas:
            m_rect = rect_mass(x, y, bounds, sigma)
            idx, w = stratified_sample(dist, sigma, cap, rng)
            t0 = time.perf_counter()
            m_poly = np.array([
                gaussian_polygon_mass(poly, x[i], y[i], sigma,
                                      MAX_STEP_SIGMA, GL_ORDER) for i in idx
            ])
            wall = time.perf_counter() - t0
            diff = m_rect[idx] - m_poly
            if len(diff) and diff.min() < -1e-9:
                raise AssertionError(
                    f"{name} sigma={sigma}: M_rect < M_poly ({diff.min():.3e})"
                    " -- A is not contained in its bounding box?")
            far = dist >= FAR_SIGMA * sigma  # diff < ~1e-14 there
            sum_diff = float((w * diff).sum())
            sum_poly = float((w * m_poly).sum() + m_rect[far].sum())
            # convergence: finer step + higher order on a subsample
            conv_idx = idx[rng.permutation(len(idx))[:CONV_SAMPLE]]
            conv = max(
                (abs(gaussian_polygon_mass(poly, x[i], y[i], sigma,
                                           CONV_MAX_STEP_SIGMA, CONV_GL_ORDER)
                     - m_poly[np.flatnonzero(idx == i)[0]])
                 for i in conv_idx),
                default=np.nan)
            eps = {r: float(1.0 - erf(r / np.sqrt(2.0)) ** 2)
                   for r in CUTOFF_RATIOS}
            rows.append({
                "region": name, "sigma_m": sigma, "n_events": n,
                "n_sampled": len(idx),
                "frac_within_1sigma": float((dist <= sigma).mean()),
                "frac_within_2sigma": float((dist <= 2 * sigma).mean()),
                "frac_within_3sigma": float((dist <= 3 * sigma).mean()),
                "median_diff": (weighted_median(diff, w) if len(diff) else 0.0),
                "max_diff": (float(diff.max()) if len(diff) else 0.0),
                "rel_overcharge": sum_diff / sum_poly if sum_poly else np.nan,
                "sum_M_poly_est": sum_poly,
                "ref_wall_s": wall,
                "ref_ms_per_event": 1e3 * wall / max(len(idx), 1),
                "conv_max_abs_diff": conv,
                "bbox_fill_ratio": bbox_fill, "n_vertices": int(n_vertices),
                "eps_ws2sigma": eps[2.0], "eps_ws3sigma": eps[3.0],
                "eps_ws4sigma": eps[4.0],
                "tol_target": TOL_FRACTION_OF_EPS * eps[3.0],
            })
            conv_rows.append({"region": name, "sigma_m": sigma,
                              "conv_max_abs_diff": conv,
                              "clip_neglect_bound":
                                  float(4 * 0.5 * erfc(CLIP_RADIUS_SIGMA / np.sqrt(2)))})
            print(f"{name} | sigma={sigma:g} m | n={n} sampled={len(idx)} "
                  f"| rel_overcharge={rows[-1]['rel_overcharge']:.3e} "
                  f"| max_diff={rows[-1]['max_diff']:.3e} | {wall:.1f}s")
    peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(conv_rows).to_csv(out_dir / "convergence.csv", index=False)
    meta = {
        "kind": kind, "sigma_grid_m": sigmas, "stratum_cap": cap, "seed": seed,
        "events_path": str(EVENTS_PATH), "n_events_total": int(len(events)),
        "n_events_in_regions": int(len(joined)),
        "regions_path": str(DISTRICTS_PATH if kind == "districts" else PARKS_PATH),
        "metric_crs": METRIC_CRS, "gl_order": GL_ORDER,
        "max_step_sigma": MAX_STEP_SIGMA, "clip_radius_sigma": CLIP_RADIUS_SIGMA,
        "far_sigma": FAR_SIGMA, "tracemalloc_peak_mb": peak_mb,
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2))
    write_report(out_dir, kind, summary, meta)
    return out_dir


def df_to_md(df: pd.DataFrame) -> str:
    """Minimal markdown table (the pinned env has no tabulate)."""
    cells = [[str(c) for c in df.columns]] + df.astype(str).values.tolist()
    widths = [max(len(r[i]) for r in cells) for i in range(len(cells[0]))]
    def fmt_row(r): return "| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([fmt_row(cells[0]), sep] + [fmt_row(r) for r in cells[1:]])


def write_report(out_dir: Path, kind: str, s: pd.DataFrame, meta: dict) -> None:
    def pivot(col: str, fmt: str) -> str:
        p = s.pivot(index="region", columns="sigma_m", values=col)
        p.columns = [f"σ={c:g} m" for c in p.columns]
        return df_to_md(p.map(lambda v: format(v, fmt)).reset_index())

    lines = [
        f"# §13 polygon-mass diagnostic — {kind}",
        "",
        f"Run {meta['timestamp']}; events: `{meta['events_path']}` "
        f"({meta['n_events_in_regions']} of {meta['n_events_total']} inside regions); "
        f"regions: `{meta['regions_path']}`; CRS {meta['metric_crs']}.",
        "",
        "M_rect is the §5.12 per-axis erf product over the region bounding box "
        "(unclipped); M_poly is the divergence-theorem Gauss–Legendre reference "
        f"(order {meta['gl_order']}, step ≤ {meta['max_step_sigma']}σ, clip "
        f"{meta['clip_radius_sigma']}σ). Diffs from a boundary-stratified "
        f"subsample (cap {meta['stratum_cap']}/stratum); aggregates are "
        "inclusion-weighted; events ≥ 8σ from ∂A contribute M_poly := M_rect "
        "(analytic bound < 1e-14).",
        "",
        "## Relative excitation-compensator overcharge  Σ(M_rect − M_poly)/ΣM_poly",
        "", pivot("rel_overcharge", ".2e"), "",
        "## Median spatial-mass difference  M_rect − M_poly",
        "", pivot("median_diff", ".2e"), "",
        "## Maximum spatial-mass difference (over sample)",
        "", pivot("max_diff", ".2e"), "",
        "## Fraction of events within 1σ of ∂A",
        "", pivot("frac_within_1sigma", ".3f"), "",
        "## Bounding-box fill ratio (area(A)/area(A_□))",
        "",
        df_to_md(s[["region", "bbox_fill_ratio", "n_vertices"]]
                 .drop_duplicates("region").round(3)),
        "",
        "## Reference cost",
        "",
        f"Peak traced memory {meta['tracemalloc_peak_mb']:.0f} MB. "
        "Per-event wall time (ms):",
        "", pivot("ref_ms_per_event", ".1f"), "",
        "## Convergence of the reference",
        "",
        f"Max |ΔM_poly| under step/2 + order-{CONV_GL_ORDER} re-run "
        f"(subsample of {CONV_SAMPLE}): "
        f"{s['conv_max_abs_diff'].max():.2e}.",
        "",
        "## Accuracy target for a production polygon-mass method",
        "",
        "Per D-21 the declared square-cutoff tail is "
        "ε_s = 1 − erf(w_s/(√2σ))²: "
        f"ε_s(2σ) = {s['eps_ws2sigma'].iloc[0]:.2e}, "
        f"ε_s(3σ) = {s['eps_ws3sigma'].iloc[0]:.2e}, "
        f"ε_s(4σ) = {s['eps_ws4sigma'].iloc[0]:.2e}. "
        "For the induced compensator error to be dominated by the declared "
        f"cutoff, a per-event absolute mass tolerance of "
        f"{TOL_FRACTION_OF_EPS} × ε_s(3σ) = "
        f"{TOL_FRACTION_OF_EPS * s['eps_ws3sigma'].iloc[0]:.1e} suffices for "
        "w_s = 3σ; tighten toward the float32 scale (~1e-7) only if w_s ≥ 4σ "
        "is declared.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", choices=["districts", "parks"], required=True)
    ap.add_argument("--sigma", type=float, nargs="+", default=SIGMA_GRID_M)
    ap.add_argument("--cap", type=int, default=DEFAULT_STRATUM_CAP,
                    help="max sampled events per boundary stratum")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=REPO / "output" / "polygon_diagnostic")
    args = ap.parse_args()
    out = run(args.regions, args.sigma, args.cap, args.seed, args.out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
