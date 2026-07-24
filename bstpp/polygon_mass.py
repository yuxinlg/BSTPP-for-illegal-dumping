"""Polygon excitation mass: offline float64 quadrature + online Hermite tables.

Phase 3d / OP-9 hybrid backend. The likelihood evaluates ONLY the C1 cubic
Hermite lookup; fixed-node boundary quadrature builds knot values and
forward-mode AD slopes offline in float64 and must never run inside NUTS/SVI.

Target mass (real-unit sigma, optional fixed spatial_window w_s)::

    M_j(sigma) = int_{A ∩ C_j} N(s - s_j; 0, sigma^2 I) ds

with C_j the event-centered real-space square of half-width w_s when
``spatial_window`` is set, else uncut A.

Validated shootout spacing: K=64 log-uniform knots on [10, 500] (metres).
Wider configured ranges keep at least that log-sigma spacing by increasing K.
Extrapolation is prohibited.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from shapely.geometry import box as shapely_box
from shapely.geometry.polygon import orient

# Shootout-validated knot spacing (metres); K autoscaling preserves this
# log-step regardless of the CRS unit the production range is expressed in.
VALIDATED_SIGMA_MIN_M = 10.0
VALIDATED_SIGMA_MAX_M = 500.0
VALIDATED_K = 64
VALIDATED_DLOG = (
    (np.log(VALIDATED_SIGMA_MAX_M) - np.log(VALIDATED_SIGMA_MIN_M))
    / (VALIDATED_K - 1)
)

DEFAULT_PANEL_H_M = 20.0
DEFAULT_GL_ORDER = 16

# Compensator error gate from the shootout (0.1 * eps_s(3 sigma), D-21).
TAU_ABS = 5.39e-4
TAU_DERIV = 5.39e-4


def knot_count(sigma_min: float, sigma_max: float) -> int:
    """Knot count covering [sigma_min, sigma_max] at validated log spacing."""
    if not (np.isfinite(sigma_min) and np.isfinite(sigma_max)):
        raise ValueError("sigma_min and sigma_max must be finite")
    if not (sigma_min > 0.0 and sigma_max > sigma_min):
        raise ValueError(
            f"require 0 < sigma_min < sigma_max; got sigma_min={sigma_min}, "
            f"sigma_max={sigma_max}")
    span = np.log(sigma_max) - np.log(sigma_min)
    return max(2, 1 + int(np.ceil(span / VALIDATED_DLOG - 1e-15)))


def log_knots(sigma_min: float, sigma_max: float) -> np.ndarray:
    """Log-spaced knots on exactly [sigma_min, sigma_max]."""
    k = knot_count(sigma_min, sigma_max)
    return np.linspace(np.log(sigma_min), np.log(sigma_max), k)


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
            reps = np.maximum(
                np.ceil(np.maximum(np.abs(x2 - x1), np.abs(y2 - y1)) / h),
                1).astype(int)
            seg = np.repeat(np.arange(len(x1)), reps)
            within = (np.arange(reps.sum())
                      - np.repeat(np.cumsum(reps) - reps, reps))
            t0 = within / reps[seg]
            t1 = (within + 1) / reps[seg]
            panels.append(np.column_stack([
                x1[seg] + t0 * (x2 - x1)[seg],
                y1[seg] + t0 * (y2 - y1)[seg],
                x1[seg] + t1 * (x2 - x1)[seg],
                y1[seg] + t1 * (y2 - y1)[seg]]))
    if not panels:
        return np.zeros((0, 4))
    return np.concatenate(panels, axis=0)


@dataclass
class QuadPrep:
    """Static-shape prepared arrays for the jitted quadrature evaluation."""
    ev_xy: np.ndarray
    panels: np.ndarray
    mask: np.ndarray
    inside_flag: np.ndarray
    ws: float | None
    h_panel: float = 0.0
    prep_seconds: float = 0.0


def prepare_quadrature(poly, x: np.ndarray, y: np.ndarray, h: float,
                       ws: float | None) -> QuadPrep:
    """Geometry preprocessing (Python/shapely; outside the jitted path)."""
    import time

    import shapely

    t0 = time.perf_counter()
    B = len(x)
    if ws is None:
        p = panelize(poly, h)
        panels = p[None, :, :]
        mask = np.ones((1, len(p)))
        inside = np.zeros(B)
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
    return QuadPrep(
        ev_xy=np.column_stack([x, y]), panels=panels, mask=mask,
        inside_flag=inside, ws=ws, h_panel=h,
        prep_seconds=time.perf_counter() - t0)


def make_quad_eval(gl_order: int, ws: float | None):
    """Jitted per-event mass; fixed shapes, no Python control flow in the body."""
    glx64, glw64 = np.polynomial.legendre.leggauss(gl_order)
    glx01_np = (glx64 + 1.0) / 2.0
    glw01_np = glw64 / 2.0

    def per_event_mass(log_sigma, ev_xy, panels, mask, inside_flag):
        glx01 = jnp.asarray(glx01_np, dtype=panels.dtype)
        glw01 = jnp.asarray(glw01_np, dtype=panels.dtype)
        sigma = jnp.exp(log_sigma)
        xa, ya, xb, yb = (panels[..., 0], panels[..., 1],
                          panels[..., 2], panels[..., 3])
        xn = xa[..., None] + glx01 * (xb - xa)[..., None]
        yn = ya[..., None] + glx01 * (yb - ya)[..., None]
        sx = ev_xy[:, 0][:, None, None]
        sy = ev_xy[:, 1][:, None, None]
        phi = jnp.exp(-((xn - sx) ** 2) / (2 * sigma**2)) / (
            sigma * jnp.sqrt(2 * jnp.pi))
        cdf = 0.5 * (1.0 + jax.scipy.special.erf(
            (yn - sy) / (sigma * jnp.sqrt(2.0))))
        quad = -jnp.sum(((phi * cdf) @ glw01) * (xb - xa) * mask, axis=-1)
        if ws is None:
            return quad
        analytic = jax.scipy.special.erf(ws / (jnp.sqrt(2.0) * sigma)) ** 2
        return inside_flag * analytic + quad

    masses = jax.jit(per_event_mass)
    d_masses = jax.jit(jax.jacfwd(per_event_mass, argnums=0))

    def mass_sum(log_sigma, ev_xy, panels, mask, inside_flag, weights):
        return jnp.sum(weights * per_event_mass(
            log_sigma, ev_xy, panels, mask, inside_flag))

    return masses, d_masses, jax.jit(jax.value_and_grad(mass_sum, argnums=0))


@dataclass
class PolygonMassTable:
    """Per-event Hermite table over log sigma (production online artifact)."""
    log_knots: np.ndarray       # (K,) float64
    values: np.ndarray          # (n_events, K) float64
    slopes: np.ndarray          # (n_events, K) float64  dM / d log sigma
    sigma_min: float
    sigma_max: float
    spatial_window: float | None
    h_panel: float
    gl_order: int
    geometry_sha256: str
    events_sha256: str
    build_seconds: float
    provenance: dict

    @property
    def n_knots(self) -> int:
        return int(self.log_knots.shape[0])

    @property
    def n_events(self) -> int:
        return int(self.values.shape[0])

    def export_npz(self, path: str | Path) -> Path:
        """Write table arrays + provenance sidecars for later refit reload."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: (None if v is None else v) for k, v in self.provenance.items()}
        np.savez_compressed(
            path,
            log_knots=self.log_knots,
            values=self.values,
            slopes=self.slopes,
            sigma_min=np.array(self.sigma_min),
            sigma_max=np.array(self.sigma_max),
            spatial_window=np.array(
                np.nan if self.spatial_window is None else self.spatial_window),
            h_panel=np.array(self.h_panel),
            gl_order=np.array(self.gl_order),
            geometry_sha256=np.array(self.geometry_sha256),
            events_sha256=np.array(self.events_sha256),
        )
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        import json
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        return path

    @staticmethod
    def load_npz(path: str | Path) -> "PolygonMassTable":
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        ws = float(data["spatial_window"])
        import json
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        prov = {}
        if meta_path.exists():
            prov = json.loads(meta_path.read_text(encoding="utf-8"))
        return PolygonMassTable(
            log_knots=np.asarray(data["log_knots"], dtype=np.float64),
            values=np.asarray(data["values"], dtype=np.float64),
            slopes=np.asarray(data["slopes"], dtype=np.float64),
            sigma_min=float(data["sigma_min"]),
            sigma_max=float(data["sigma_max"]),
            spatial_window=None if np.isnan(ws) else ws,
            h_panel=float(data["h_panel"]),
            gl_order=int(data["gl_order"]),
            geometry_sha256=str(data["geometry_sha256"]),
            events_sha256=str(data["events_sha256"]),
            build_seconds=float(prov.get("build_seconds", np.nan)),
            provenance=prov,
        )


def _geometry_sha256(poly) -> str:
    wkb = poly.wkb if hasattr(poly, "wkb") else bytes(str(poly), "utf-8")
    return hashlib.sha256(wkb).hexdigest()


def _events_sha256(x: np.ndarray, y: np.ndarray) -> str:
    payload = b"".join(
        f"{float(xi):.9g},{float(yi):.9g};".encode() for xi, yi in zip(x, y))
    return hashlib.sha256(payload).hexdigest()


def validate_polygon_mass_table(
    table: PolygonMassTable,
    *,
    domain_geom,
    event_x_real: np.ndarray,
    event_y_real: np.ndarray,
    spatial_window: float | None,
    sigma_min: float,
    sigma_max: float,
    h_panel: float,
    gl_order: int,
) -> None:
    """Reject a supplied Hermite table that is not identity-compatible.

    Equal event counts are not evidence of compatibility. Validates domain
    geometry hash, event coordinates and row order, event count, spatial
    window, sigma range and knot grid, build settings (h_panel, gl_order),
    array shapes, and finite array values.
    """
    if not isinstance(table, PolygonMassTable):
        raise TypeError(
            f"mass_table must be a PolygonMassTable; got {type(table).__name__}")

    x = np.asarray(event_x_real, dtype=float)
    y = np.asarray(event_y_real, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(
            f"event_x_real/event_y_real must be 1-d and same shape; "
            f"got {x.shape} and {y.shape}")
    n = int(x.shape[0])

    if int(table.n_events) != n:
        raise ValueError(
            f"supplied mass table n_events={table.n_events} does not match "
            f"model event count {n}")

    expected_events = _events_sha256(x, y)
    if table.events_sha256 != expected_events:
        raise ValueError(
            "supplied mass table events_sha256 does not match the model event "
            "coordinates and row order (equal counts are not sufficient)")

    expected_geom = _geometry_sha256(domain_geom)
    if table.geometry_sha256 != expected_geom:
        raise ValueError(
            "supplied mass table geometry_sha256 does not match the model "
            "domain union geometry")

    if spatial_window is None:
        if table.spatial_window is not None:
            raise ValueError(
                f"supplied mass table spatial_window={table.spatial_window} "
                "but the model has spatial_window=None")
    else:
        if table.spatial_window is None or not np.isfinite(table.spatial_window):
            raise ValueError(
                "supplied mass table spatial_window is missing/nonfinite but "
                f"the model has spatial_window={spatial_window}")
        if float(table.spatial_window) != float(spatial_window):
            raise ValueError(
                f"supplied mass table spatial_window={table.spatial_window} "
                f"does not match model spatial_window={spatial_window}")

    if not (np.isfinite(table.sigma_min) and np.isfinite(table.sigma_max)):
        raise ValueError("supplied mass table sigma_min/sigma_max must be finite")
    if float(table.sigma_min) != float(sigma_min) or float(table.sigma_max) != float(sigma_max):
        raise ValueError(
            f"supplied mass table sigma range [{table.sigma_min}, {table.sigma_max}] "
            f"does not match model [{sigma_min}, {sigma_max}]")

    expected_knots = log_knots(float(sigma_min), float(sigma_max))
    if table.log_knots.shape != expected_knots.shape or not np.allclose(
            table.log_knots, expected_knots, rtol=0.0, atol=0.0):
        raise ValueError(
            "supplied mass table log_knots do not match the sigma range knot grid")

    if float(table.h_panel) != float(h_panel):
        raise ValueError(
            f"supplied mass table h_panel={table.h_panel} does not match "
            f"build setting h_panel={h_panel}")
    if int(table.gl_order) != int(gl_order):
        raise ValueError(
            f"supplied mass table gl_order={table.gl_order} does not match "
            f"build setting gl_order={gl_order}")

    k = int(expected_knots.shape[0])
    if table.values.shape != (n, k) or table.slopes.shape != (n, k):
        raise ValueError(
            f"supplied mass table array shapes must be values/slopes=({n}, {k}); "
            f"got values={table.values.shape}, slopes={table.slopes.shape}")
    if table.log_knots.shape != (k,):
        raise ValueError(
            f"supplied mass table log_knots shape must be ({k},); "
            f"got {table.log_knots.shape}")

    for name, arr in (
        ("log_knots", table.log_knots),
        ("values", table.values),
        ("slopes", table.slopes),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"supplied mass table {name} contains nonfinite values")


BACKEND_SCHEMA_VERSION = "hybrid_quad_hermite_numpy_v1"
# Central finite-difference step on log-sigma for float64 slope tables.
_LOG_SIGMA_FD_EPS = 1e-6


def _legendre_01(gl_order: int) -> tuple[np.ndarray, np.ndarray]:
    glx, glw = np.polynomial.legendre.leggauss(int(gl_order))
    return (glx + 1.0) / 2.0, glw / 2.0


def _quad_masses_numpy(
    log_sigma: float,
    ev_xy: np.ndarray,
    panels: np.ndarray,
    mask: np.ndarray,
    inside_flag: np.ndarray,
    glx01: np.ndarray,
    glw01: np.ndarray,
    ws: float | None,
) -> np.ndarray:
    """Host float64 per-event Gaussian polygon mass at one log-sigma knot."""
    from scipy.special import erf

    sigma = float(np.exp(log_sigma))
    xa, ya, xb, yb = (panels[..., 0], panels[..., 1],
                      panels[..., 2], panels[..., 3])
    xn = xa[..., None] + glx01 * (xb - xa)[..., None]
    yn = ya[..., None] + glx01 * (yb - ya)[..., None]
    sx = ev_xy[:, 0][:, None, None]
    sy = ev_xy[:, 1][:, None, None]
    phi = np.exp(-((xn - sx) ** 2) / (2.0 * sigma ** 2)) / (
        sigma * np.sqrt(2.0 * np.pi))
    cdf = 0.5 * (1.0 + erf((yn - sy) / (sigma * np.sqrt(2.0))))
    quad = -np.sum(((phi * cdf) @ glw01) * (xb - xa) * mask, axis=-1)
    if ws is None:
        return np.asarray(quad, dtype=np.float64)
    analytic = float(erf(ws / (np.sqrt(2.0) * sigma)) ** 2)
    return np.asarray(inside_flag * analytic + quad, dtype=np.float64)


def _quad_slopes_numpy(
    log_sigma: float,
    ev_xy: np.ndarray,
    panels: np.ndarray,
    mask: np.ndarray,
    inside_flag: np.ndarray,
    glx01: np.ndarray,
    glw01: np.ndarray,
    ws: float | None,
    eps: float = _LOG_SIGMA_FD_EPS,
) -> np.ndarray:
    """dM / d log(sigma) via central differences in float64."""
    up = _quad_masses_numpy(
        log_sigma + eps, ev_xy, panels, mask, inside_flag, glx01, glw01, ws)
    dn = _quad_masses_numpy(
        log_sigma - eps, ev_xy, panels, mask, inside_flag, glx01, glw01, ws)
    return (up - dn) / (2.0 * eps)


def build_quad_table(
    poly,
    x: np.ndarray,
    y: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    ws: float | None = None,
    h_panel: float = DEFAULT_PANEL_H_M,
    gl_order: int = DEFAULT_GL_ORDER,
    extra_provenance: Optional[dict] = None,
) -> PolygonMassTable:
    """Build Hermite table with host NumPy/SciPy float64 quadrature + slopes.

    ``sigma_min`` / ``sigma_max`` are spatial standard deviations in real
    coordinate units (compatible with ``sqrt(sigmax_2)``). Does not mutate
    process-global JAX config.
    """
    import time

    if not (np.isfinite(sigma_min) and sigma_min > 0):
        raise ValueError(f"sigma_min must be finite and positive; got {sigma_min}")
    if not (np.isfinite(sigma_max) and sigma_max > sigma_min):
        raise ValueError(
            f"require finite sigma_max > sigma_min; got "
            f"sigma_min={sigma_min}, sigma_max={sigma_max}")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    knots = np.asarray(log_knots(sigma_min, sigma_max), dtype=np.float64)
    glx01, glw01 = _legendre_01(gl_order)
    glx01 = np.asarray(glx01, dtype=np.float64)
    glw01 = np.asarray(glw01, dtype=np.float64)

    t0 = time.perf_counter()
    prep = prepare_quadrature(poly, x, y, float(h_panel), ws)
    ev = np.asarray(prep.ev_xy, dtype=np.float64)
    panels = np.asarray(prep.panels, dtype=np.float64)
    mask = np.asarray(prep.mask, dtype=np.float64)
    inside = np.asarray(prep.inside_flag, dtype=np.float64)
    vals = np.array([
        _quad_masses_numpy(float(lk), ev, panels, mask, inside, glx01, glw01, ws)
        for lk in knots
    ], dtype=np.float64).T
    slopes = np.array([
        _quad_slopes_numpy(float(lk), ev, panels, mask, inside, glx01, glw01, ws)
        for lk in knots
    ], dtype=np.float64).T
    build_s = time.perf_counter() - t0

    geom_hash = _geometry_sha256(poly)
    ev_hash = _events_sha256(x, y)
    table_id = hashlib.sha256(
        (geom_hash + ev_hash + f"{sigma_min}:{sigma_max}:{ws}:{h_panel}:{gl_order}"
         + vals.tobytes().hex()[:64]).encode()
    ).hexdigest()
    prov: dict[str, Any] = {
        "backend": "hybrid_quad_hermite",
        "backend_schema": BACKEND_SCHEMA_VERSION,
        "sigma_parameterization": "standard_deviation",
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "n_knots": int(knots.shape[0]),
        "validated_dlog": float(VALIDATED_DLOG),
        "spatial_window": None if ws is None else float(ws),
        "h_panel": float(h_panel),
        "gl_order": int(gl_order),
        "geometry_sha256": geom_hash,
        "events_sha256": ev_hash,
        "table_id": table_id,
        "build_seconds": build_s,
        "dtype_build": "float64",
        "n_events": int(len(x)),
        "slope_method": "central_fd_log_sigma",
        "slope_fd_eps": float(_LOG_SIGMA_FD_EPS),
    }
    if extra_provenance:
        # Descriptive only unless it already participates in construction.
        prov.update(extra_provenance)
    return PolygonMassTable(
        log_knots=knots,
        values=vals,
        slopes=slopes,
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        spatial_window=None if ws is None else float(ws),
        h_panel=float(h_panel),
        gl_order=int(gl_order),
        geometry_sha256=geom_hash,
        events_sha256=ev_hash,
        build_seconds=build_s,
        provenance=prov,
    )


def prepare_polygon_mass_table(
    domain_geom,
    event_x_real,
    event_y_real,
    *,
    min_sigma: float,
    max_sigma: float,
    spatial_window: float | None = None,
    crs=None,
    panel_h_m: float = DEFAULT_PANEL_H_M,
    gl_order: int = DEFAULT_GL_ORDER,
    extra_provenance: Optional[dict] = None,
) -> PolygonMassTable:
    """Explicit public preparation API for polygon Hermite mass tables.

    ``min_sigma`` and ``max_sigma`` are spatial standard deviations in real
    coordinate units. When the model prior is expressed as ``sigmax_2``
    (variance), the compatible interval is ``[min_sigma**2, max_sigma**2]``.

    Builds entirely in host NumPy/SciPy float64. Does not mutate process-global
    JAX precision. Online fitting may later consume the table at the process
    default JAX precision.
    """
    from .excitation_support import metres_to_crs_units

    if crs is not None and not getattr(crs, "is_geographic", False):
        h_panel = float(metres_to_crs_units(float(panel_h_m), crs))
    else:
        h_panel = float(panel_h_m)
    return build_quad_table(
        domain_geom,
        np.asarray(event_x_real, dtype=np.float64),
        np.asarray(event_y_real, dtype=np.float64),
        float(min_sigma),
        float(max_sigma),
        ws=None if spatial_window is None else float(spatial_window),
        h_panel=h_panel,
        gl_order=int(gl_order),
        extra_provenance=extra_provenance,
    )


def _hermite_per_event_mass(log_sigma, log_knots, values, slopes):
    """C1 cubic Hermite along log sigma (uniform knots); NaN outside span."""
    K = log_knots.shape[0]
    h = log_knots[1] - log_knots[0]
    i = jnp.clip(
        jnp.floor((log_sigma - log_knots[0]) / h).astype(int), 0, K - 2)
    t = (log_sigma - log_knots[i]) / h
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)
    m = (h00 * values[:, i] + h10 * h * slopes[:, i]
         + h01 * values[:, i + 1] + h11 * h * slopes[:, i + 1])
    out_of_range = (log_sigma < log_knots[0]) | (log_sigma > log_knots[-1])
    return jnp.where(out_of_range, jnp.nan, m)


def _hermite_mass_sum(log_sigma, log_knots, values, slopes, weights):
    return jnp.sum(weights * _hermite_per_event_mass(
        log_sigma, log_knots, values, slopes))


# Module-level jitted evaluators so the likelihood does not re-jit per call.
_TABLE_MASSES = jax.jit(_hermite_per_event_mass)
_TABLE_D_MASSES = jax.jit(jax.jacfwd(_hermite_per_event_mass, argnums=0))
_TABLE_VG = jax.jit(jax.value_and_grad(_hermite_mass_sum, argnums=0))


def make_table_eval():
    """Jitted C1 cubic Hermite along log sigma (uniform knots).

    Returns NaN outside the knot span (no silent extrapolation).
    """
    return _TABLE_MASSES, _TABLE_D_MASSES, _TABLE_VG


def validate_sigma_in_range(sigma: float, log_knots_arr: np.ndarray) -> None:
    lo, hi = float(np.exp(log_knots_arr[0])), float(np.exp(log_knots_arr[-1]))
    if not (np.log(sigma) >= log_knots_arr[0] - 1e-12
            and np.log(sigma) <= log_knots_arr[-1] + 1e-12):
        raise ValueError(
            f"sigma={sigma} outside table-supported range [{lo:.6g}, {hi:.6g}]; "
            "extrapolation is prohibited (rebuild the table or widen "
            "min_sigma/max_sigma)")


def hermite_polygon_masses(log_sigma, table: PolygonMassTable,
                           dtype=None):
    """Per-event polygon masses at log_sigma from a prepared table."""
    masses_fn, _, _ = make_table_eval()
    dt = dtype or jnp.result_type(log_sigma)
    return masses_fn(
        jnp.asarray(log_sigma, dtype=dt),
        jnp.asarray(table.log_knots, dtype=dt),
        jnp.asarray(table.values, dtype=dt),
        jnp.asarray(table.slopes, dtype=dt),
    )


def warn_if_sigma_near_bound(sigmax_2_samples, max_sigma_real: float,
                             frac: float = 0.9) -> None:
    """Warn when posterior sigma approaches the configured upper bound."""
    s = np.sqrt(np.asarray(sigmax_2_samples, dtype=float)).ravel()
    if s.size == 0:
        return
    q95 = float(np.quantile(s, 0.95))
    if q95 >= frac * float(max_sigma_real):
        warnings.warn(
            f"Posterior sigma 95th percentile ({q95:.6g}) is within "
            f"{frac:.0%} of max_sigma ({max_sigma_real:.6g}). Sigma may be "
            "weakly identified; consider increasing max_sigma and rebuilding "
            "the polygon-mass table.",
            UserWarning,
            stacklevel=2,
        )
