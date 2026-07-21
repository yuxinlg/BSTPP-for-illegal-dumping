"""OP-9 backend shootout: fixed-node JAX boundary quadrature vs sigma lookup tables.

Experiment and decision record ONLY -- nothing here touches the production
likelihood, simulator, or API. Compares two candidate production backends for
the Phase 3d polygon excitation mass

    M_j(sigma) = int_{A ∩ C_j} N(s - s_j; 0, sigma^2 I) ds

where A is the domain polygon and C_j the event-centered real-unit cutoff
square of half-width spatial_window (w_s). Semantics established by code
inspection (recorded in the report):

  * spatial_window is FIXED during a fit (user-supplied real length,
    main.py:1530/1651); sigma is sampled while w_s stays constant, so C_j is
    sigma-independent and known at preprocessing time.
  * spatial_window is OPTIONAL and defaults to None -> the target is the
    uncut mass over A. The PRIMARY benchmark target is therefore uncut A
    (which is also exactly what the §13 oracle computes); the finite-cutoff
    machinery (A ∩ C_j) is implemented and validated separately with a
    representative w_s (not a recommended production value -- cutoff
    selection is 3e's problem).
  * The §13 oracle (scripts/polygon_mass_diagnostic.gaussian_polygon_mass)
    is the uncut mass with a purely NUMERICAL 8-sigma evaluation clip
    (neglected mass < 4*0.5*erfc(8/sqrt(2)) ~ 2.5e-15); it is NOT the
    production cutoff. For A ∩ C_j the smallest oracle wrapper is used:
    exact shapely intersection with the C_j square, then the same integral.

Backend A -- fixed-node JAX boundary quadrature: divergence-theorem line
integral (same formulation as the oracle) on panels fixed at preprocessing
time; static shapes, padded panels + masks, no Python in the jitted path;
per-event clipped-polygon panels in cutoff mode with an analytic erf^2 fast
path when C_j lies entirely inside A; dM/dlog(sigma) by jax.grad.

Backend B -- per-event lookup tables over log-spaced sigma knots, values
from the oracle; C1 cubic Hermite in log sigma with PCHIP slopes prepared
offline (centered-difference slopes for rows where PCHIP's monotone
flattening would distort derivatives at interior extrema); jit-compatible,
differentiable, extrapolation prohibited.

Outputs under results/polygon_mass_backend_shootout/.
Run:  JAX_ENABLE_X64=1 python scripts/polygon_mass_backend_shootout.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import jax

import geopandas as gpd
import jax.numpy as jnp
import numpy as np
import pandas as pd
import shapely
from scipy.interpolate import PchipInterpolator
from scipy.special import erf
from shapely.geometry import box as shapely_box
from shapely.geometry.polygon import orient

# ---------------------------------------------------------------- oracle ----
_spec = importlib.util.spec_from_file_location(
    "polygon_mass_diagnostic", REPO / "scripts" / "polygon_mass_diagnostic.py")
_diag = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _diag
_spec.loader.exec_module(_diag)

ORACLE_MAX_STEP_SIGMA = _diag.MAX_STEP_SIGMA   # do not weaken (0.5)
ORACLE_GL_ORDER = _diag.GL_ORDER               # do not weaken (20)

OUT_DIR = REPO / "results" / "polygon_mass_backend_shootout"

SIGMA_REGISTERED = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
SIGMA_RANGE = (10.0, 500.0)
TAU_ABS = 5.39e-4            # 0.1 * eps_s(3 sigma), D-21 square tail
TAU_DERIV = 5.39e-4          # provisional; reported separately (see report)
TAU_FLOOR = 1e-7             # explained deviation: 0.1*eps_s(sigma) -> 0 when the
                             # fixed cutoff retains ~all mass; floored at the
                             # float32 scale instead of an unachievable 1e-300.
REL_DENOM_MIN = 1e-3         # "safely nonzero" oracle mass for relative errors
REPRESENTATIVE_WS = 400.0    # metres; validates A∩C_j machinery only (3e picks w_s)

KNOT_COUNTS = [24, 40, 64]
PANEL_CONFIGS = [(10.0, 8), (20.0, 8), (20.0, 16), (40.0, 16)]  # (h metres, GL order)

N_BOUNDARY = 64
N_INTERIOR = 64
SELECT_SEED = 0
OFFGRID_SEED = 1234
N_OFFGRID = 20
FD_H = 1e-3                  # centered FD step in log sigma (halved for check)

EVAL_BATCH = 256             # backend-A event batch inside performance runs
SMOKE_BATCH_CAP = None       # set by --smoke to cap performance batches

REGION_SPECS = [
    # (label, source, name-in-file)
    ("Lower South", "districts", "Lower South"),
    ("North Delaware", "districts", "North Delaware"),
    ("Central", "districts", "Central"),
    ("East Fairmount Park", "parks", "East Fairmount Park"),
    ("Mifflin Square", "parks", "Mifflin Square"),
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_line(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def oracle_mass(poly, sx: float, sy: float, sigma: float,
                ws: float | None = None) -> float:
    """§13 oracle; with ws, the smallest wrapper for A ∩ C_j (exact clip first)."""
    if ws is not None:
        poly = poly.intersection(shapely_box(sx - ws, sy - ws, sx + ws, sy + ws))
        if poly.is_empty:
            return 0.0
    return _diag.gaussian_polygon_mass(poly, sx, sy, sigma,
                                       ORACLE_MAX_STEP_SIGMA, ORACLE_GL_ORDER)


def oracle_dmass_dlogsigma(poly, sx, sy, sigma, ws=None) -> tuple[float, float]:
    """Centered FD in log sigma; returns (derivative, uncertainty).

    Repeated at half step; Richardson-style uncertainty |D_h - D_h2| / 3
    (both are O(h^2), so the difference bounds the leading error of D_h2).
    """
    def dm(h):
        return (oracle_mass(poly, sx, sy, sigma * np.exp(h), ws)
                - oracle_mass(poly, sx, sy, sigma * np.exp(-h), ws)) / (2 * h)
    d1, d2 = dm(FD_H), dm(FD_H / 2)
    return d2, abs(d1 - d2) / 3.0


# ------------------------------------------------------- geometry loading ----
def load_case_polygons() -> dict[str, shapely.Geometry]:
    districts = gpd.read_file(_diag.DISTRICTS_PATH).to_crs(_diag.METRIC_CRS)
    parks = gpd.read_file(_diag.PARKS_PATH).to_crs(_diag.METRIC_CRS)
    repaired = []
    out = {}
    for label, source, name in REGION_SPECS:
        gdf, col = ((districts, "dist_name") if source == "districts"
                    else (parks, "PUBLIC_NAME"))
        match = gdf[gdf[col] == name]
        if len(match) != 1:
            raise ValueError(f"region {name!r} not uniquely found in {source}")
        geom = match.geometry.iloc[0]
        if not geom.is_valid:
            geom = shapely.unary_union([p for p in
                                        getattr(shapely.make_valid(geom), "geoms",
                                                [shapely.make_valid(geom)])
                                        if p.geom_type in ("Polygon", "MultiPolygon")])
            repaired.append(label)
        out[label] = geom
    if repaired:
        print(f"note: repaired invalid geometries via make_valid: {repaired}")
    return out


def load_events() -> gpd.GeoDataFrame:
    ev = gpd.read_file(_diag.EVENTS_PATH).to_crs(_diag.METRIC_CRS)
    ev = ev[~(ev.geometry.is_empty | ev.geometry.isna())]
    return ev.set_geometry(ev.geometry.centroid)[["geometry"]]


def select_events(poly, events: gpd.GeoDataFrame,
                  rng: np.random.Generator) -> dict:
    """Deterministic per-region selection: <=64 boundary-near (the nearest,
    always including the most boundary-adjacent), <=64 seeded interior."""
    inside = events[events.within(poly)]
    x = inside.geometry.x.to_numpy()
    y = inside.geometry.y.to_numpy()
    dist = shapely.distance(inside.geometry.values, poly.boundary)
    n = len(inside)
    if n <= N_BOUNDARY + N_INTERIOR:
        idx = np.arange(n)
    else:
        order = np.argsort(dist, kind="stable")
        near = order[:N_BOUNDARY]
        rest = order[N_BOUNDARY:]
        interior = rng.choice(rest, size=min(N_INTERIOR, len(rest)), replace=False)
        idx = np.concatenate([near, np.sort(interior)])
    coord_hash = hashlib.sha256(
        b"".join(f"{x[i]:.3f},{y[i]:.3f};".encode() for i in idx)).hexdigest()
    return {"x": x[idx], "y": y[idx], "dist": dist[idx], "n_region": n,
            "coord_hash": coord_hash, "n_selected": len(idx)}


# --------------------------------------------- backend A: fixed-node quad ----
def panelize(geom, h: float) -> np.ndarray:
    """(P, 4) panel endpoints (xa, ya, xb, yb), max per-axis extent <= h.

    Exterior rings CCW, holes CW, so the signed line integral adds exterior
    mass and subtracts hole mass with one formula.
    """
    panels = []
    parts = getattr(geom, "geoms", [geom])
    for part in parts:
        if part.geom_type != "Polygon" or part.is_empty:
            continue
        part = orient(part, 1.0)
        for ring in [part.exterior, *part.interiors]:
            c = np.asarray(ring.coords)
            x1, y1, x2, y2 = c[:-1, 0], c[:-1, 1], c[1:, 0], c[1:, 1]
            reps = np.maximum(np.ceil(np.maximum(np.abs(x2 - x1), np.abs(y2 - y1))
                                      / h), 1).astype(int)
            seg = np.repeat(np.arange(len(x1)), reps)
            within = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
            t0 = within / reps[seg]
            t1 = (within + 1) / reps[seg]
            panels.append(np.column_stack([
                x1[seg] + t0 * (x2 - x1)[seg], y1[seg] + t0 * (y2 - y1)[seg],
                x1[seg] + t1 * (x2 - x1)[seg], y1[seg] + t1 * (y2 - y1)[seg]]))
    if not panels:
        return np.zeros((0, 4))
    return np.concatenate(panels, axis=0)


@dataclass
class QuadPrep:
    """Static-shape prepared arrays for the jitted quadrature evaluation."""
    ev_xy: np.ndarray            # (B, 2)
    panels: np.ndarray           # (B, P, 4) padded; masked entries all-zero
    mask: np.ndarray             # (B, P) 1.0 valid / 0.0 padding
    inside_flag: np.ndarray      # (B,) 1.0 when C_j entirely inside A
    ws: float | None
    h_panel: float = 0.0
    gl_order: int = 0
    prep_seconds: float = 0.0
    nbytes: int = 0
    panel_count_max: int = 0
    panel_count_mean: float = 0.0


def prepare_quadrature(poly, x: np.ndarray, y: np.ndarray, h: float,
                       ws: float | None) -> QuadPrep:
    """Geometry preprocessing (Python/shapely; outside the jitted path).

    Uncut: one shared panel set, broadcast per event (stored (1, P, 4) and
    broadcast in eval -- the per-event 'relevant panel mapping' is the full
    set, since sigma spans 10-500 m and at sigma=500 the 8-sigma horizon
    covers these districts entirely; the mask is all-ones).
    Cutoff: per-event exact clip A ∩ C_j, padded to the max panel count;
    events whose C_j lies entirely inside A take the analytic fast path
    (mass = erf(w_s / sqrt(2) sigma)^2) and carry zero panels.
    """
    t0 = time.perf_counter()
    B = len(x)
    if ws is None:
        p = panelize(poly, h)
        panels = p[None, :, :]
        mask = np.ones((1, len(p)))
        inside = np.zeros(B)
        pcounts = np.array([len(p)])
    else:
        per_event, inside_l = [], []
        shapely.prepare(poly)
        for sx, sy in zip(x, y):
            square = shapely_box(sx - ws, sy - ws, sx + ws, sy + ws)
            if shapely.contains_properly(poly, square):
                inside_l.append(1.0)
                per_event.append(np.zeros((0, 4)))
            else:
                inside_l.append(0.0)
                per_event.append(panelize(poly.intersection(square), h))
        P = max((len(p) for p in per_event), default=1) or 1
        panels = np.zeros((B, P, 4))
        mask = np.zeros((B, P))
        for i, p in enumerate(per_event):
            panels[i, :len(p)] = p
            mask[i, :len(p)] = 1.0
        inside = np.array(inside_l)
        pcounts = np.array([len(p) for p in per_event])
    prep = QuadPrep(
        ev_xy=np.column_stack([x, y]), panels=panels, mask=mask,
        inside_flag=inside, ws=ws, h_panel=h,
        prep_seconds=time.perf_counter() - t0,
        nbytes=panels.nbytes + mask.nbytes,
        panel_count_max=int(pcounts.max()), panel_count_mean=float(pcounts.mean()))
    return prep


def make_quad_eval(gl_order: int, ws: float | None):
    """Jitted per-event mass evaluation; fixed shapes, no Python control flow.

    Masked padding panels are all-zero endpoints: the integrand is evaluated
    there too but is finite (Gaussian of finite args), then zeroed by the
    mask -- no NaN can leak into the reduction.
    """
    glx64, glw64 = np.polynomial.legendre.leggauss(gl_order)
    glx01_np = (glx64 + 1.0) / 2.0
    glw01_np = glw64 / 2.0

    def per_event_mass(log_sigma, ev_xy, panels, mask, inside_flag):
        # GL constants follow the input dtype so a float32 spot-check stays f32
        glx01 = jnp.asarray(glx01_np, dtype=panels.dtype)
        glw01 = jnp.asarray(glw01_np, dtype=panels.dtype)
        sigma = jnp.exp(log_sigma)
        xa, ya, xb, yb = (panels[..., 0], panels[..., 1],
                          panels[..., 2], panels[..., 3])
        # nodes: (B, P, q)
        xn = xa[..., None] + glx01 * (xb - xa)[..., None]
        yn = ya[..., None] + glx01 * (yb - ya)[..., None]
        sx = ev_xy[:, 0][:, None, None]
        sy = ev_xy[:, 1][:, None, None]
        phi = jnp.exp(-((xn - sx) ** 2) / (2 * sigma**2)) / (
            sigma * jnp.sqrt(2 * jnp.pi))
        cdf = 0.5 * (1.0 + jax.scipy.special.erf((yn - sy) / (sigma * jnp.sqrt(2.0))))
        quad = -jnp.sum(((phi * cdf) @ glw01) * (xb - xa) * mask, axis=-1)
        if ws is None:
            return quad
        analytic = jax.scipy.special.erf(ws / (jnp.sqrt(2.0) * sigma)) ** 2
        return inside_flag * analytic + quad

    masses = jax.jit(per_event_mass)
    # per-event dM/dlog(sigma): (B,) forward-mode Jacobian wrt the scalar
    d_masses = jax.jit(jax.jacfwd(per_event_mass, argnums=0))

    def mass_sum(log_sigma, ev_xy, panels, mask, inside_flag, weights):
        return jnp.sum(weights * per_event_mass(log_sigma, ev_xy, panels, mask,
                                                inside_flag))

    return masses, d_masses, jax.jit(jax.value_and_grad(mass_sum, argnums=0))


# -------------------------------------------- backend B: lookup tables ------
@dataclass
class TablePrep:
    log_knots: np.ndarray        # (K,) uniform in log sigma
    values: np.ndarray           # (B, K) oracle masses
    slopes: np.ndarray           # (B, K) d value / d log sigma at knots
    n_nonmonotone: int
    build_seconds: float
    nbytes: int
    npz_path: Path | None = None
    npz_sha256: str = ""
    npz_bytes: int = 0


def build_table(poly, x, y, K: int, ws: float | None,
                tag: str) -> TablePrep:
    """Oracle-valued lookup table over K log-spaced knots spanning SIGMA_RANGE.

    Slopes: PCHIP (shape-preserving C1). PCHIP is defined for non-monotone
    data but forces slope 0 at interior extrema, which biases dM/dlog sigma
    there; rows with an interior extremum are detected and switched to
    centered-difference (Catmull-Rom style) Hermite slopes -- smooth,
    non-monotone-safe, still C1. Counts reported.
    """
    t0 = time.perf_counter()
    log_knots = np.linspace(np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1]), K)
    sig = np.exp(log_knots)
    vals = np.array([[oracle_mass(poly, sx, sy, s, ws) for s in sig]
                     for sx, sy in zip(x, y)])
    pch = PchipInterpolator(log_knots, vals, axis=1, extrapolate=False)
    slopes = np.asarray(pch(log_knots, 1))  # (B, K): d value / d log sigma
    dm = np.diff(vals, axis=1)
    nonmono = (np.sign(dm[:, :-1]) * np.sign(dm[:, 1:]) < 0).any(axis=1)
    if nonmono.any():
        h = log_knots[1] - log_knots[0]
        cd = np.empty_like(vals)
        cd[:, 1:-1] = (vals[:, 2:] - vals[:, :-2]) / (2 * h)
        cd[:, 0] = dm[:, 0] / h
        cd[:, -1] = dm[:, -1] / h
        slopes[nonmono] = cd[nonmono]
    build = time.perf_counter() - t0
    prep = TablePrep(log_knots=log_knots, values=vals, slopes=slopes,
                     n_nonmonotone=int(nonmono.sum()), build_seconds=build,
                     nbytes=vals.nbytes + slopes.nbytes + log_knots.nbytes)
    npz = OUT_DIR / "tables" / f"table_{tag}_K{K}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, log_knots=log_knots, values=vals, slopes=slopes)
    prep.npz_path = npz
    prep.npz_sha256 = sha256_file(npz)
    prep.npz_bytes = npz.stat().st_size
    return prep


def make_table_eval():
    """Jitted C1 cubic-Hermite interpolation along log sigma (uniform knots).

    Extrapolation is prohibited: callers must range-check sigma BEFORE the
    jitted call (validate_sigma_in_range); the kernel additionally returns
    NaN outside the knot span so silent extrapolation is structurally loud.
    """
    def per_event_mass(log_sigma, log_knots, values, slopes):
        K = log_knots.shape[0]
        h = log_knots[1] - log_knots[0]
        i = jnp.clip(jnp.floor((log_sigma - log_knots[0]) / h).astype(int), 0, K - 2)
        t = (log_sigma - log_knots[i]) / h
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t**2 * (3 - 2 * t)
        h11 = t**2 * (t - 1)
        m = (h00 * values[:, i] + h10 * h * slopes[:, i]
             + h01 * values[:, i + 1] + h11 * h * slopes[:, i + 1])
        out_of_range = (log_sigma < log_knots[0]) | (log_sigma > log_knots[-1])
        return jnp.where(out_of_range, jnp.nan, m)

    masses = jax.jit(per_event_mass)
    d_masses = jax.jit(jax.jacfwd(per_event_mass, argnums=0))

    def mass_sum(log_sigma, log_knots, values, slopes, weights):
        return jnp.sum(weights * per_event_mass(log_sigma, log_knots, values, slopes))

    return masses, d_masses, jax.jit(jax.value_and_grad(mass_sum, argnums=0))


def validate_sigma_in_range(sigma: float, log_knots: np.ndarray) -> None:
    if not (np.log(sigma) >= log_knots[0] - 1e-12
            and np.log(sigma) <= log_knots[-1] + 1e-12):
        raise ValueError(
            f"sigma={sigma} outside table-supported range "
            f"[{np.exp(log_knots[0]):.6g}, {np.exp(log_knots[-1]):.6g}] m; "
            "extrapolation is prohibited (rebuild the table or widen knots)")


# ----------------------------------------------------------- sigma grids ----
def build_test_sigmas() -> dict[str, np.ndarray]:
    reg = np.array(SIGMA_REGISTERED)
    mid = np.sqrt(reg[:-1] * reg[1:])
    rng = np.random.default_rng(OFFGRID_SEED)
    rand = np.exp(rng.uniform(np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1]),
                              N_OFFGRID))
    knot_adjacent = {}
    for K in KNOT_COUNTS:
        lk = np.linspace(np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1]), K)
        picks = np.exp(lk[[1, K // 2, K - 2]])
        knot_adjacent[K] = np.concatenate([picks * (1 - 1e-3), picks * (1 + 1e-3)])
    return {"registered": reg, "midpoints": mid, "offgrid": rand,
            **{f"knot_adjacent_K{K}": v for K, v in knot_adjacent.items()}}


# ------------------------------------------------------------- benchmark ----
def timed(fn, *args, reps: int = 30) -> dict:
    """Median/p95 wall time of fn(*args) with block_until_ready."""
    out = fn(*args)
    jax.block_until_ready(out)  # warm (compile excluded by caller timing sep.)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return {"median_s": float(np.median(ts)), "p95_s": float(np.quantile(ts, 0.95))}


def eps_s(ws_over_sigma: np.ndarray | float) -> np.ndarray | float:
    return 1.0 - erf(np.asarray(ws_over_sigma) / np.sqrt(2.0)) ** 2


def tau_for(sigma: float, ws: float | None) -> float:
    if ws is None:
        return TAU_ABS
    return float(min(TAU_ABS, max(0.1 * eps_s(ws / sigma), TAU_FLOOR)))


def run_quad_built_tables() -> None:
    """Addendum experiment: Hermite tables built by the QUAD backend.

    Motivated by the main run: shapely-oracle-built tables fail the
    derivative gate at every K (PCHIP / centered-difference slopes are only
    O(h^2-h^3) accurate) and their build is oracle-bound (~2 h projected for
    the largest fit). Here both knot values AND knot slopes come from the
    fixed-node quadrature backend (values ~1e-11 from the oracle, slopes by
    forward-mode AD) -- an explicit, disclosed dependency permitted because
    quad independently passed every oracle gate in the main run. Gates are
    still scored against the SHAPELY oracle, never against quad itself.
    """
    jax.config.update("jax_enable_x64", True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _diag.self_test()
    polys = load_case_polygons()
    events = load_events()
    rng = np.random.default_rng(SELECT_SEED)
    cases = {}
    for label in polys:
        cases[label] = select_events(polys[label], events, rng)
        cases[label]["poly"] = polys[label]
    sig_sets = build_test_sigmas()
    all_sigmas = np.unique(np.concatenate(list(sig_sets.values())))
    deriv_sigmas = np.unique(np.concatenate([sig_sets["registered"],
                                             sig_sets["midpoints"]]))
    h, q = 20.0, 16
    masses_fn, d_masses_fn, _ = make_quad_eval(q, None)
    table_masses, table_d, _ = make_table_eval()
    rows = []
    cache = {}
    for label, c in cases.items():
        poly, x, y = c["poly"], c["x"], c["y"]
        mv = np.array([[oracle_mass(poly, sx, sy, s) for s in all_sigmas]
                       for sx, sy in zip(x, y)])
        dv = np.zeros((len(x), len(deriv_sigmas)))
        du = np.zeros_like(dv)
        for j, s in enumerate(deriv_sigmas):
            for i, (sx, sy) in enumerate(zip(x, y)):
                dv[i, j], du[i, j] = oracle_dmass_dlogsigma(poly, sx, sy, s)
        cache[label] = (mv, dv, du)
        print(f"oracle (addendum) done: {label}")
        prep = prepare_quadrature(poly, x, y, h, None)
        ej, pj, mj, fj = (jnp.asarray(prep.ev_xy), jnp.asarray(prep.panels),
                          jnp.asarray(prep.mask), jnp.asarray(prep.inside_flag))
        for K in [40, 64]:
            log_knots = np.linspace(np.log(SIGMA_RANGE[0]),
                                    np.log(SIGMA_RANGE[1]), K)
            t0 = time.perf_counter()
            vals = np.array([masses_fn(lk, ej, pj, mj, fj)
                             for lk in log_knots]).T
            slopes = np.array([d_masses_fn(lk, ej, pj, mj, fj)
                               for lk in log_knots]).T
            build_s = time.perf_counter() - t0
            lk, vj, sj = (jnp.asarray(log_knots), jnp.asarray(vals),
                          jnp.asarray(slopes))
            got = np.array([table_masses(np.log(s), lk, vj, sj)
                            for s in all_sigmas]).T
            dgot = np.array([table_d(np.log(s), lk, vj, sj)
                             for s in deriv_sigmas]).T
            viol = int(((got < -1e-12) | (got > 1 + 1e-12)).sum())
            rows.append(accuracy_row(
                backend=f"table_quadbuilt_K{K}", region=label, ws=None,
                got=got, oracle=mv, sigmas=all_sigmas,
                deriv_got=dgot, deriv_oracle=dv, deriv_floor=du,
                prep_seconds=build_s,
                nbytes=vals.nbytes + slopes.nbytes + log_knots.nbytes,
                extra={"bound_violations": viol,
                       "build_s_per_event": build_s / len(x),
                       "slope_source": "quad jacfwd (AD)"}))
            print(f"  table_quadbuilt K={K} {label}: build {build_s:.1f}s "
                  f"max_abs_err {rows[-1]['max_abs_err']:.2e} "
                  f"deriv_err {rows[-1]['max_deriv_abs_err']:.2e} "
                  f"pass={rows[-1]['value_pass']}/{rows[-1]['deriv_pass']}")
    np.savez_compressed(OUT_DIR / "tables" / "oracle_cache_addendum.npz",
                        all_sigmas=all_sigmas, deriv_sigmas=deriv_sigmas,
                        **{f"{k}_{n}": v for k, (mv, dv, du) in cache.items()
                           for n, v in (("vals", mv), ("derivs", dv),
                                        ("derivunc", du))})
    pd.DataFrame(rows).to_csv(OUT_DIR / "summary_addendum.csv", index=False)
    print(f"wrote {OUT_DIR / 'summary_addendum.csv'}")


def main() -> None:
    # accuracy study in f64 (f32 spot-check casts explicitly); set here, not at
    # import, so importing this module (e.g. from tests) does not flip the
    # global JAX precision for unrelated code
    jax.config.update("jax_enable_x64", True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end pipeline check; NOT decision data")
    ap.add_argument("--quad-built-tables", action="store_true",
                    help="addendum: Hermite tables with quad-backend values "
                         "and AD slopes, gated against the shapely oracle")
    args = ap.parse_args()
    if args.quad_built_tables:
        run_quad_built_tables()
        return
    if args.smoke:
        globals().update(
            REGION_SPECS=[r for r in REGION_SPECS
                          if r[0] in ("Lower South", "Mifflin Square")],
            N_BOUNDARY=8, N_INTERIOR=8, N_OFFGRID=4,
            KNOT_COUNTS=[24], PANEL_CONFIGS=[(20.0, 16)],
            SMOKE_BATCH_CAP=512)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _diag.self_test()  # oracle self-test unchanged and passing

    run_config = {
        "timestamp": datetime.now().isoformat(),
        "git_branch": git_line("branch", "--show-current"),
        "git_head": git_line("rev-parse", "HEAD"),
        "git_status_short": git_line("status", "--short").splitlines(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__, "numpy": np.__version__,
        "shapely": shapely.__version__, "geopandas": gpd.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "inputs": {str(p): sha256_file(p) for p in
                   [_diag.EVENTS_PATH, _diag.DISTRICTS_PATH, _diag.PARKS_PATH]},
        "oracle": {"max_step_sigma": ORACLE_MAX_STEP_SIGMA,
                   "gl_order": ORACLE_GL_ORDER,
                   "module": "scripts/polygon_mass_diagnostic.py (unchanged)"},
        "select_seed": SELECT_SEED, "offgrid_seed": OFFGRID_SEED,
        "fd_h": FD_H, "tau_abs": TAU_ABS, "tau_deriv": TAU_DERIV,
        "tau_floor": TAU_FLOOR, "representative_ws_m": REPRESENTATIVE_WS,
        "panel_configs": PANEL_CONFIGS, "knot_counts": KNOT_COUNTS,
        "eval_batch": EVAL_BATCH,
        "primary_target": "uncut mass over A (spatial_window=None, current default)",
        "cutoff_target": "A ∩ C_j with representative w_s=400 m (machinery check)",
    }

    polys = load_case_polygons()
    events = load_events()
    rng = np.random.default_rng(SELECT_SEED)
    cases = {}
    for label in polys:
        cases[label] = select_events(polys[label], events, rng)
        cases[label]["poly"] = polys[label]
        print(f"{label}: {cases[label]['n_selected']} of "
              f"{cases[label]['n_region']} events selected "
              f"(hash {cases[label]['coord_hash'][:12]})")
    run_config["cases"] = {
        label: {k: (v if not isinstance(v, np.ndarray) else None)
                for k, v in c.items() if k in
                ("n_region", "n_selected", "coord_hash")}
        for label, c in cases.items()}

    sig_sets = build_test_sigmas()
    all_sigmas = np.unique(np.concatenate(list(sig_sets.values())))
    deriv_sigmas = np.unique(np.concatenate([sig_sets["registered"],
                                             sig_sets["midpoints"]]))

    rows = []
    perf_rows = []

    # ---------------- oracle values and derivatives (primary, uncut) --------
    oracle_vals = {}      # (label) -> (B, S) masses
    oracle_derivs = {}    # (label) -> (B, D), (B, D) uncertainty
    t_oracle0 = time.perf_counter()
    for label, c in cases.items():
        poly, x, y = c["poly"], c["x"], c["y"]
        mv = np.array([[oracle_mass(poly, sx, sy, s) for s in all_sigmas]
                       for sx, sy in zip(x, y)])
        dv = np.zeros((len(x), len(deriv_sigmas)))
        du = np.zeros_like(dv)
        for j, s in enumerate(deriv_sigmas):
            for i, (sx, sy) in enumerate(zip(x, y)):
                dv[i, j], du[i, j] = oracle_dmass_dlogsigma(poly, sx, sy, s)
        oracle_vals[label] = mv
        oracle_derivs[label] = (dv, du)
        print(f"oracle done: {label} ({time.perf_counter() - t_oracle0:.0f}s cum)")
    run_config["oracle_wall_s"] = time.perf_counter() - t_oracle0
    run_config["deriv_uncertainty_max"] = float(
        max(du.max() for _, du in oracle_derivs.values()))

    # ---------------- backend A accuracy (primary, uncut) -------------------
    quad_evals = {(q, None): make_quad_eval(q, None) for _, q in PANEL_CONFIGS}
    for h, q in PANEL_CONFIGS:
        for label, c in cases.items():
            prep = prepare_quadrature(c["poly"], c["x"], c["y"], h, None)
            masses_fn, d_masses_fn, _ = quad_evals[(q, None)]
            pj = jnp.asarray(prep.panels)
            mj = jnp.asarray(prep.mask)
            ej = jnp.asarray(prep.ev_xy)
            fj = jnp.asarray(prep.inside_flag)
            got = np.array([masses_fn(np.log(s), ej, pj, mj, fj)
                            for s in all_sigmas]).T          # (B, S)
            dgot = np.array([d_masses_fn(np.log(s), ej, pj, mj, fj)
                             for s in deriv_sigmas]).T       # (B, D)
            dv, du = oracle_derivs[label]
            rows.append(accuracy_row(
                backend=f"quad_h{h:g}_q{q}", region=label, ws=None,
                got=got, oracle=oracle_vals[label], sigmas=all_sigmas,
                deriv_got=dgot, deriv_oracle=dv, deriv_floor=du,
                prep_seconds=prep.prep_seconds, nbytes=prep.nbytes,
                extra={"panel_count": prep.panel_count_max}))
    # ---------------- backend B accuracy (primary, uncut) -------------------
    table_masses, table_d, table_vg = make_table_eval()
    for K in KNOT_COUNTS:
        for label, c in cases.items():
            prep = build_table(c["poly"], c["x"], c["y"], K, None,
                               tag=label.replace(" ", "_"))
            lk = jnp.asarray(prep.log_knots)
            vj = jnp.asarray(prep.values)
            sj = jnp.asarray(prep.slopes)
            for s in all_sigmas:
                validate_sigma_in_range(s, prep.log_knots)
            got = np.array([table_masses(np.log(s), lk, vj, sj)
                            for s in all_sigmas]).T
            dgot = np.array([table_d(np.log(s), lk, vj, sj)
                             for s in deriv_sigmas]).T
            dv, du = oracle_derivs[label]
            # bound check: 0 <= M <= 1 (uncut) at every test sigma
            viol = int(((got < -1e-12) | (got > 1 + 1e-12)).sum())
            rows.append(accuracy_row(
                backend=f"table_K{K}", region=label, ws=None,
                got=got, oracle=oracle_vals[label], sigmas=all_sigmas,
                deriv_got=dgot, deriv_oracle=dv, deriv_floor=du,
                prep_seconds=prep.build_seconds, nbytes=prep.nbytes,
                extra={"n_nonmonotone_rows": prep.n_nonmonotone,
                       "bound_violations": viol,
                       "npz_bytes": prep.npz_bytes,
                       "npz_sha256": prep.npz_sha256[:16]}))
    # -------------- cutoff machinery check (A ∩ C_j, backend A only) --------
    quad_ws_evals = {}
    for h, q in [(20.0, 16)]:
        quad_ws_evals[(q, REPRESENTATIVE_WS)] = make_quad_eval(q, REPRESENTATIVE_WS)
        for label in ["Lower South", "Mifflin Square"]:
            c = cases[label]
            prep = prepare_quadrature(c["poly"], c["x"], c["y"], h,
                                      REPRESENTATIVE_WS)
            masses_fn, _, _ = quad_ws_evals[(q, REPRESENTATIVE_WS)]
            sigmas = np.array(SIGMA_REGISTERED)
            ora = np.array([[oracle_mass(c["poly"], sx, sy, s, REPRESENTATIVE_WS)
                             for s in sigmas]
                            for sx, sy in zip(c["x"], c["y"])])
            got = np.array([masses_fn(np.log(s), jnp.asarray(prep.ev_xy),
                                      jnp.asarray(prep.panels),
                                      jnp.asarray(prep.mask),
                                      jnp.asarray(prep.inside_flag))
                            for s in sigmas]).T
            rows.append(accuracy_row(
                backend=f"quad_h{h:g}_q{q}", region=label,
                ws=REPRESENTATIVE_WS, got=got, oracle=ora, sigmas=sigmas,
                deriv_got=None, deriv_oracle=None, deriv_floor=None,
                prep_seconds=prep.prep_seconds, nbytes=prep.nbytes,
                extra={"panel_count": prep.panel_count_max,
                       "fast_path_events": int(prep.inside_flag.sum())}))

    # -------- float32 spot check (production precision) ---------------------
    h, q = 20.0, 16
    label = "Lower South"
    c = cases[label]
    prep = prepare_quadrature(c["poly"], c["x"], c["y"], h, None)
    masses_fn, _, _ = quad_evals[(q, None)]
    got32 = np.array([
        masses_fn(np.float32(np.log(s)),
                  jnp.asarray(prep.ev_xy, jnp.float32),
                  jnp.asarray(prep.panels, jnp.float32),
                  jnp.asarray(prep.mask, jnp.float32),
                  jnp.asarray(prep.inside_flag, jnp.float32))
        for s in SIGMA_REGISTERED]).T
    idx32 = np.searchsorted(all_sigmas, SIGMA_REGISTERED)
    f32_err = float(np.abs(got32 - oracle_vals[label][:, idx32]).max())
    run_config["float32_spot_check"] = {
        "backend": f"quad_h{h:g}_q{q}", "region": label,
        "max_abs_err": f32_err, "passes_tau": bool(f32_err < TAU_ABS)}

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    # ---------------- performance ------------------------------------------
    if not args.skip_perf:
        perf_rows = run_performance(cases, events, quad_evals,
                                    (table_masses, table_vg))
        pd.DataFrame(perf_rows).to_csv(OUT_DIR / "perf.csv", index=False)
    run_config["synthetic_tests"] = (
        "tests/test_polygon_mass_backend_shootout.py (rectangle/triangle/"
        "concave/hole/multipolygon/rectangle-degeneracy)")
    (OUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2,
                                                        default=str))
    print(f"\nwrote {OUT_DIR}")


def accuracy_row(*, backend, region, ws, got, oracle, sigmas,
                 deriv_got, deriv_oracle, deriv_floor,
                 prep_seconds, nbytes, extra=None) -> dict:
    err = got - oracle
    abs_err = np.abs(err)
    denom_ok = oracle >= REL_DENOM_MIN
    rel = np.where(denom_ok, abs_err / np.maximum(oracle, REL_DENOM_MIN), 0.0)
    taus = np.array([tau_for(s, ws) for s in sigmas])
    per_sigma_max = abs_err.max(axis=0)
    signed_by_sigma = err.sum(axis=0)
    worst_signed = signed_by_sigma[int(np.abs(signed_by_sigma).argmax())]
    row = {
        "backend": backend, "region": region,
        "ws": ws if ws is not None else "none",
        "max_abs_err": float(abs_err.max()),
        "median_abs_err": float(np.median(abs_err)),
        "max_rel_err_safe": float(rel.max()),
        "signed_aggregate_err": float(worst_signed),
        "aggregate_abs_err_max_sigma": float(abs_err.sum(axis=0).max()),
        "sum_mass_rel_err_max_sigma": float(
            np.max(np.abs(signed_by_sigma) / np.maximum(oracle.sum(axis=0),
                                                        1e-300))),
        "value_pass": bool((per_sigma_max <= taus).all()),
        "worst_sigma": float(sigmas[int(abs_err.max(axis=0).argmax())]),
        "n_nan": int(np.isnan(got).sum()),
        "prep_seconds": prep_seconds,
        "prep_nbytes": nbytes,
    }
    if deriv_got is not None:
        # per-event derivative errors; the pass rule allows the documented
        # floor: backend error must be <= max(TAU_DERIV, 10x FD uncertainty)
        dabs = np.abs(deriv_got - deriv_oracle)
        row.update({
            "max_deriv_abs_err": float(dabs.max()),
            "deriv_floor_max": float(deriv_floor.max()),
            "deriv_pass": bool((dabs <= np.maximum(TAU_DERIV,
                                                   10 * deriv_floor)).all()),
        })
    if extra:
        row.update(extra)
    return row


def run_performance(cases, events, quad_evals, table_fns) -> list[dict]:
    """Warm value / value+grad timings; preprocessing separated; batches."""
    perf = []
    table_masses, table_vg = table_fns
    # scaling batches drawn from the largest-district event set (actual data)
    districts = gpd.read_file(_diag.DISTRICTS_PATH).to_crs(_diag.METRIC_CRS)
    counts = gpd.sjoin(events, districts[["dist_name", "geometry"]],
                       predicate="within")["dist_name"].value_counts()
    biggest = counts.index[0]
    poly_big = districts.set_index("dist_name").geometry[biggest]
    if not poly_big.is_valid:
        poly_big = shapely.make_valid(poly_big)
    ev_big = events[events.within(poly_big)]
    xb = ev_big.geometry.x.to_numpy()
    yb = ev_big.geometry.y.to_numpy()
    n_big = len(xb)
    batches = [b for b in (128, 512, 2048, 8192) if b <= n_big] + [n_big]
    if SMOKE_BATCH_CAP is not None:
        batches = [b for b in batches if b <= SMOKE_BATCH_CAP]
    rng = np.random.default_rng(7)
    h, q = 20.0, 16  # timed config = cheapest passing (asserted in report)
    masses_fn, _, vg_fn = quad_evals[(q, None)]
    sigma0 = 100.0

    def chunked_vg(log_sigma, ej, pj, mj, fj, w):
        """Sum value_and_grad over fixed-shape event chunks (one compile).

        Arrays are pre-padded to a multiple of EVAL_BATCH with zero-weight
        dummy events, so every chunk has identical static shapes.
        """
        v = 0.0
        g = 0.0
        for s0 in range(0, ej.shape[0], EVAL_BATCH):
            vv, gg = vg_fn(log_sigma, ej[s0:s0 + EVAL_BATCH], pj, mj,
                           fj[s0:s0 + EVAL_BATCH], w[s0:s0 + EVAL_BATCH])
            v = v + vv
            g = g + gg
        return v, g

    for B in batches:
        idx = rng.choice(n_big, size=B, replace=False)
        x, y = xb[idx], yb[idx]
        prep = prepare_quadrature(poly_big, x, y, h, None)
        pad = (-B) % EVAL_BATCH
        ev_pad = np.vstack([prep.ev_xy,
                            np.tile(prep.ev_xy[:1], (pad, 1))])
        fl_pad = np.concatenate([prep.inside_flag, np.zeros(pad)])
        ej, pj, mj, fj = (jnp.asarray(ev_pad), jnp.asarray(prep.panels),
                          jnp.asarray(prep.mask), jnp.asarray(fl_pad))
        w = jnp.asarray(np.concatenate([rng.uniform(0.5, 1.5, B), np.zeros(pad)]))
        w1 = jnp.asarray(np.concatenate([np.ones(B), np.zeros(pad)]))
        chunked = B + pad > EVAL_BATCH
        t0 = time.perf_counter()
        jax.block_until_ready(chunked_vg(np.log(sigma0), ej, pj, mj, fj, w1))
        compile_s = time.perf_counter() - t0
        reps = 5 if B > 2048 else 15
        if chunked:
            tv = None
            tg = timed(chunked_vg, np.log(sigma0), ej, pj, mj, fj, w1, reps=reps)
            tgw = timed(chunked_vg, np.log(sigma0), ej, pj, mj, fj, w, reps=reps)
        else:
            tv = timed(masses_fn, np.log(sigma0), ej, pj, mj, fj, reps=reps)
            tg = timed(vg_fn, np.log(sigma0), ej, pj, mj, fj, w1, reps=reps)
            tgw = timed(vg_fn, np.log(sigma0), ej, pj, mj, fj, w, reps=reps)
        perf.append({
            "backend": f"quad_h{h:g}_q{q}", "batch": B, "region": biggest,
            "chunked_batch": EVAL_BATCH if chunked else B,
            "prep_seconds": prep.prep_seconds, "prep_nbytes": prep.nbytes,
            "panel_count": prep.panel_count_max, "compile_s": compile_s,
            "value_median_s": tv["median_s"] if tv else np.nan,
            "value_p95_s": tv["p95_s"] if tv else np.nan,
            "valgrad_median_s": tg["median_s"], "valgrad_p95_s": tg["p95_s"],
            "valgrad_weighted_median_s": tgw["median_s"],
        })
        print(f"perf quad B={B}: prep {prep.prep_seconds:.1f}s "
              f"grad {tg['median_s']*1e3:.2f}ms (chunked={chunked})")
    # table backend: build cost scales linearly in B via oracle calls; timing
    # the JAX evaluation with synthetic (already-built) tables is fair for the
    # warm path. Build-time is PROJECTED from the accuracy-sample rate.
    K = 40
    per_event_build_s = None
    for B in batches:
        idx = rng.choice(n_big, size=B, replace=False)
        lk = np.linspace(np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1]), K)
        vals = rng.uniform(0.0, 1.0, (B, K)).cumsum(axis=1)  # placeholder arrays
        vals /= vals[:, -1:]
        slopes = np.gradient(vals, lk, axis=1)
        lkj, vj, sj = jnp.asarray(lk), jnp.asarray(vals), jnp.asarray(slopes)
        w1 = jnp.ones(B)
        t0 = time.perf_counter()
        jax.block_until_ready(table_vg(np.log(sigma0), lkj, vj, sj, w1))
        compile_s = time.perf_counter() - t0
        tv = timed(table_masses, np.log(sigma0), lkj, vj, sj)
        tg = timed(table_vg, np.log(sigma0), lkj, vj, sj, w1)
        perf.append({
            "backend": f"table_K{K}", "batch": B, "region": biggest,
            "prep_seconds": np.nan,  # projected separately below
            "prep_nbytes": vals.nbytes + slopes.nbytes + lk.nbytes,
            "compile_s": compile_s,
            "value_median_s": tv["median_s"], "value_p95_s": tv["p95_s"],
            "valgrad_median_s": tg["median_s"], "valgrad_p95_s": tg["p95_s"],
        })
        print(f"perf table B={B}: value {tv['median_s']*1e3:.3f}ms "
              f"grad {tg['median_s']*1e3:.3f}ms")
    # baseline: the existing rectangle compensator's spatial mass (per-axis
    # erf product through the production trigger), jitted, same events
    from bstpp.trigger import Spatial_Symmetric_Gaussian
    trig = Spatial_Symmetric_Gaussian({})
    bounds = poly_big.bounds

    def rect_mass_sum(log_sigma, x, y, weights):
        s2 = jnp.exp(log_sigma) ** 2
        lim = jnp.stack([jnp.stack([bounds[2] - x, x - bounds[0]]),
                         jnp.stack([bounds[3] - y, y - bounds[1]])])
        return jnp.sum(weights * trig.compute_integral({"sigmax_2": s2}, lim))

    rect_vg = jax.jit(jax.value_and_grad(rect_mass_sum, argnums=0))
    for B in batches:
        idx = rng.choice(n_big, size=B, replace=False)
        xj, yj = jnp.asarray(xb[idx]), jnp.asarray(yb[idx])
        w1 = jnp.ones(B)
        jax.block_until_ready(rect_vg(np.log(sigma0), xj, yj, w1))
        tg = timed(rect_vg, np.log(sigma0), xj, yj, w1)
        perf.append({"backend": "rect_baseline", "batch": B, "region": biggest,
                     "valgrad_median_s": tg["median_s"],
                     "valgrad_p95_s": tg["p95_s"]})
    return perf


if __name__ == "__main__":
    main()
