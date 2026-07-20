"""Section-14 report-only dry run of the 3a data contracts on the project data.

Mirrors the production pipeline's preparation EXACTLY (read-only):
  code/cbg-park-seasonal.ipynb cells 18/37/39/47 + batch_park_fits.py
  prepare_park_locs / build_model in the analysis tree -- events filtered to
  start_time <= 2025-01-01, sjoin 'within' the park boxes, sjoin 'within' the
  covariate layer, per-park PARKNAME filter, X/Y = centroid in EPSG:26918,
  T = days since Jan 1 of the scope's earliest event year, total_days = 1460.

Then runs bstpp.data_contracts validators (report mode semantics: nothing is
rejected, nothing is modified, no model is constructed) on each per-park
event frame against its park-box domain, and on the pooled in-park frame
against the union of the four boxes (the pooled scope is an INVENTORY
aggregate; production fits are per park).

Outputs (written next to this script):
  dry_run_report.md    -- human-readable summary for reviewer sign-off
  dry_run_findings.csv -- long format: scope, check, kind, row, message

Usage:
  python refactor-patches/phase3a/dry_run_report.py
"""
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bstpp.data_contracts import (  # noqa: E402
    DataContractReport, validate_covariates, validate_events)

TREE = Path("C:/Users/Terhi/Box/terhi-illegal-dumping-Jan15-2026")
EVENTS_FILE = TREE / "output/illegal_dumping_full.geojson"
BOXES_FILE = TREE / "output/all_boxes_gdf.geojson"
COV_FILE = TREE / "output/cov_cbg.geojson"
TOTAL_DAYS = 365 * 4
OFFSET_SEASONAL = 0
PARK_NAMES = ["tacony_box", "cobbscreek_box", "mifflin_box", "fairmount_box"]
# COLUMN_NAMES from code/batch_park_fits.py in the analysis tree
COLUMN_NAMES = [
    "pop_density", "med_inc_avg", "lep_density", "edu_hs_avg",
    "ndvi_mean_4yr", "RLD", "RMD", "RHD", "use_com", "use_civ", "I", "T",
    "GW", "vac_area", "landcare_area", "alloc_avg_d_cnt", "alloc_avg_s_cnt",
    "unique_device_ratio_aw", "reporting_rate", "betweenness_avg_w",
    "pagerank_avg_w", "betweenness_avg_d", "pagerank_avg_d",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_line(path: Path) -> str:
    st = path.stat()
    return (f"`{path}` — {st.st_size:,} bytes, mtime "
            f"{datetime.fromtimestamp(st.st_mtime).date()}, sha256 "
            f"`{_sha256(path)[:16]}…`")


def prepare_locs(points: gpd.GeoDataFrame) -> tuple[pd.DataFrame, int]:
    """batch_park_fits.prepare_park_locs, verbatim semantics, plus a count of
    rows whose start_time fails to parse (they would surface as NaN T)."""
    points = points.copy()
    start = pd.to_datetime(points["start_time"], errors="coerce")
    n_bad_time = int(start.isna().sum())
    min_year = start.min()
    min_time = pd.Timestamp(f"{min_year.year}-01-01")
    time = (start - min_time).dt.total_seconds() / (24 * 60 * 60)
    seasonal = (time + OFFSET_SEASONAL) % 365
    locs_s = pd.DataFrame({
        "X": points.geometry.centroid.x.astype(float).values,
        "Y": points.geometry.centroid.y.astype(float).values,
        "T": time.astype(float).values,
        "A": seasonal.astype(float).values,
    })
    return locs_s, n_bad_time


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    print("loading events (large file)...", flush=True)
    illegal_dumping = gpd.read_file(EVENTS_FILE)
    events_crs = str(illegal_dumping.crs)
    n_raw = len(illegal_dumping)
    illegal_dumping = illegal_dumping[
        illegal_dumping["start_time"] <= "2025-01-01"]
    n_cut = len(illegal_dumping)

    all_boxes = gpd.read_file(BOXES_FILE).to_crs(epsg=26918)
    cov_gdf = gpd.read_file(COV_FILE).to_crs(epsg=26918)

    # pipeline joins, verbatim
    in_park = illegal_dumping.sjoin(all_boxes, predicate="within")
    n_after_box = len(in_park)
    in_park = in_park.drop(columns=["index_right"])
    in_park = in_park.sjoin(cov_gdf, predicate="within", how="inner")
    n_after_cov = len(in_park)

    scopes = [(p, in_park[in_park["PARKNAME"] == p],
               all_boxes[all_boxes["PARKNAME"] == p]) for p in PARK_NAMES]
    scopes.append(("pooled(4 boxes)", in_park, all_boxes))

    findings = []
    scope_rows = []
    for name, points, domain in scopes:
        locs_s, n_bad_time = prepare_locs(points)
        checks = validate_events(locs_s, domain, TOTAL_DAYS)
        checks += validate_covariates(
            cov_gdf, COLUMN_NAMES, domain,
            points_xy=locs_s[["X", "Y"]].to_numpy(dtype=float))
        report = DataContractReport(checks, len(locs_s), "report")
        v = {c.name: len(c.indices) for c in report.violations}
        d = {c.name: len(c.indices) for c in report.diagnostics}
        scope_rows.append((name, len(locs_s), n_bad_time, v, d))
        frame = report.to_frame()
        frame.insert(0, "scope", name)
        findings.append(frame)
        print(f"{name}: n={len(locs_s)} violations={v} diagnostics={d}",
              flush=True)

    pd.concat(findings, ignore_index=True).to_csv(
        out_dir / "dry_run_findings.csv", index=False)

    # ---- markdown report -------------------------------------------------
    lines = [
        "# Section-14 dry run — 3a data contracts on the project data "
        "(report-only)",
        "",
        f"Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} — "
        "validators only; no fits, no data modification, no enforcement.",
        "",
        "## Inputs (read-only, analysis tree)",
        "",
        f"- events: {_file_line(EVENTS_FILE)}",
        f"- park boxes (domains): {_file_line(BOXES_FILE)}",
        f"- covariates: {_file_line(COV_FILE)}",
        f"- events CRS as stored: `{events_crs}`; boxes/covariates reprojected "
        "to EPSG:26918 by the pipeline (metric, meters — satisfies the "
        "real-unit contract).",
        "",
        "## Pipeline funnel (mirrored verbatim)",
        "",
        f"- raw events in file: {n_raw:,}",
        f"- after `start_time <= 2025-01-01`: {n_cut:,}",
        f"- after `within` park boxes: {n_after_box:,}",
        f"- after `within` covariate layer: {n_after_cov:,} "
        f"(lost to covariate join: {n_after_box - n_after_cov:,})",
        "",
        "Note: the pipeline's own `within` prefilter excludes exact "
        "box-boundary events BEFORE BSTPP sees them, so the validator's "
        "out-of-domain counts below measure only what survives that filter; "
        "`within` vs the contract's boundary-inclusive `covers` (D-4) can "
        "differ only for boundary points, which `within` removes.",
        "",
        "## Findings per scope",
        "",
        "| Scope | n events | unparseable start_time | Violations | "
        "Diagnostics |",
        "|---|---|---|---|---|",
    ]
    for name, n, n_bad_time, v, d in scope_rows:
        vtxt = "; ".join(f"{k}: {c}" for k, c in v.items()) or "none"
        dtxt = "; ".join(f"{k}: {c}" for k, c in d.items()) or "none"
        lines.append(f"| {name} | {n:,} | {n_bad_time} | {vtxt} | {dtxt} |")
    lines += [
        "",
        "Row-level detail: `dry_run_findings.csv` (scope, check, kind, "
        "positional row in the scope's locs_s frame, message).",
        "",
        "## Reviewer decision requested",
        "",
        "Per the staged 3a plan, the flip of the `data_contracts` default "
        "from `'report'` to `'reject'` is gated on review of this report. "
        "Any contract adjustment motivated by it must be a recorded "
        "decision, not a silent relaxation.",
    ]
    (out_dir / "dry_run_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("wrote", out_dir / "dry_run_report.md")


if __name__ == "__main__":
    main()
