"""Polygon excitation mass: offline float64 quadrature + online Hermite tables.

Phase 3d / OP-9 hybrid backend. The likelihood evaluates ONLY the C1 cubic
Hermite lookup. Production tables are built offline by
``prepare_polygon_mass_table`` / ``build_quad_table`` using host NumPy/SciPy
float64 boundary quadrature and central finite-difference knot slopes
(``SLOPE_METHOD = central_fd_log_sigma``); that preparation must never run
inside NUTS/SVI. Historical shootout/experimental builders may use JAX
forward-mode AD slopes; those are not the production path.

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
import math
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
# Cheap prepare-time resolution prefilter: after converting panel_h_m into
# effective domain-coordinate units, effective_panel_h / min_sigma must not
# exceed this ratio. Necessary but not sufficient for PRODUCTION_TAU_ABS —
# install validates a measured residual (see measure_polygon_mass_table_residual).
# Default panel_h_m=20 on CRS-less unit-scale domains is otherwise silently
# too coarse for small min_sigma.
MAX_PANEL_TO_MIN_SIGMA_RATIO = 8.0

# Independent polygon-table oracle accuracy gate (pre-3f / A-21).
# Numerical approximation error is a separate budget from spatial-cutoff
# omission; do not describe this as universally equal to 10% of eps_s.
# Historical shootout preregistered value gate (retrospective only):
LEGACY_SHOOTOUT_TAU_ABS = 5.39e-4
# Current production gate for the adopted quad-built K=64 Hermite path:
PRODUCTION_TAU_ABS = 1e-5
TAU_ABS = PRODUCTION_TAU_ABS
# Derivative gate remains provisional (OP-12) and is NOT tied to tau_abs.
TAU_DERIV = 5.39e-4

# Install-time residual reference: host float64 panel quadrature at an
# elevated Gauss–Legendre order on the table's recorded h_panel tiling.
# Empirically, vs the independent shapely §13 oracle
# (scripts/polygon_mass_backend_shootout.oracle_mass), this reference's
# max abs discrepancy on the unit-octagon calibration case at
# panel/min_sigma <= 8 was <= 6.3e-7; claim a conservative 1e-6 floor.
BUDGET_REFERENCE_GL_ORDER = 32
BUDGET_REFERENCE_ORACLE_BOUND = 1e-6

# Compatibility contract constants (builder + validator share these names).
# Schema v2: nested extra_provenance, exact le-f64 event identity, required
# metadata fields. Legacy v1 / decimal-.9g tables are intentionally incompatible.
BACKEND_ID = "hybrid_quad_hermite"
BACKEND_SCHEMA_VERSION = "hybrid_quad_hermite_numpy_v2"
SIGMA_PARAMETERIZATION = "standard_deviation"
INTERPOLATION_CONVENTION = "c1_cubic_hermite_uniform_log_sigma"
SLOPE_METHOD = "central_fd_log_sigma"
SLOPE_FD_EPS = 1e-6
EVENTS_HASH_ALGORITHM = "sha256_le_f64_xy_v1"

_REQUIRED_COMPAT_PROVENANCE_KEYS = (
    "backend",
    "backend_schema",
    "sigma_parameterization",
    "interpolation_convention",
    "slope_method",
    "slope_fd_eps",
    "events_hash_algorithm",
)


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
    # True after an ownership copy at install; not part of the on-disk schema.
    _owned: bool = False

    @property
    def n_knots(self) -> int:
        return int(self.log_knots.shape[0])

    @property
    def n_events(self) -> int:
        return int(self.values.shape[0])

    def copy(self) -> "PolygonMassTable":
        """Deep-copy arrays so install does not alias the caller's buffers."""
        return PolygonMassTable(
            log_knots=np.array(self.log_knots, dtype=np.float64, copy=True),
            values=np.array(self.values, dtype=np.float64, copy=True),
            slopes=np.array(self.slopes, dtype=np.float64, copy=True),
            sigma_min=float(self.sigma_min),
            sigma_max=float(self.sigma_max),
            spatial_window=(
                None if self.spatial_window is None
                else float(self.spatial_window)),
            h_panel=float(self.h_panel),
            gl_order=int(self.gl_order),
            geometry_sha256=str(self.geometry_sha256),
            events_sha256=str(self.events_sha256),
            build_seconds=float(self.build_seconds),
            provenance=dict(self.provenance),
            _owned=True,
        )

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
        import json
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not meta_path.exists():
            raise ValueError(
                "PolygonMassTable sidecar metadata is missing "
                f"({meta_path.name}); legacy tables without "
                f"{BACKEND_SCHEMA_VERSION} provenance are incompatible. "
                "Rebuild with bstpp.polygon_mass.prepare_polygon_mass_table(...).")
        prov = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(prov, dict):
            raise ValueError(
                "PolygonMassTable sidecar provenance must be a JSON object")
        _validate_compat_provenance(prov)

        log_knots = np.asarray(data["log_knots"], dtype=np.float64)
        values = np.asarray(data["values"], dtype=np.float64)
        slopes = np.asarray(data["slopes"], dtype=np.float64)
        sigma_min = float(data["sigma_min"])
        sigma_max = float(data["sigma_max"])
        ws_npz = float(data["spatial_window"])
        spatial_window = None if np.isnan(ws_npz) else ws_npz
        h_panel = float(data["h_panel"])
        gl_order = int(data["gl_order"])
        geometry_sha256 = str(data["geometry_sha256"])
        events_sha256 = str(data["events_sha256"])
        _assert_sidecar_matches_npz(
            prov,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            spatial_window=spatial_window,
            h_panel=h_panel,
            gl_order=gl_order,
            geometry_sha256=geometry_sha256,
            events_sha256=events_sha256,
            n_knots=int(log_knots.shape[0]),
            n_events=int(values.shape[0]),
        )
        return PolygonMassTable(
            log_knots=log_knots,
            values=values,
            slopes=slopes,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            spatial_window=spatial_window,
            h_panel=h_panel,
            gl_order=gl_order,
            geometry_sha256=geometry_sha256,
            events_sha256=events_sha256,
            build_seconds=float(prov.get("build_seconds", np.nan)),
            provenance=prov,
        )


def _sidecar_exact_float(prov: dict, key: str) -> float:
    if key not in prov:
        raise ValueError(
            f"sidecar provenance missing {key} required for NPZ consistency")
    raw = prov[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"sidecar {key}={raw!r} is malformed; must be a JSON number "
            "matching the NPZ artifact exactly")
    val = float(raw)
    if not np.isfinite(val):
        raise ValueError(
            f"sidecar {key}={raw!r} is nonfinite; must match the NPZ artifact")
    return val


def _sidecar_exact_int(prov: dict, key: str) -> int:
    if key not in prov:
        raise ValueError(
            f"sidecar provenance missing {key} required for NPZ consistency")
    raw = prov[key]
    # Reject bool (subclass of int) and non-integral floats / strings.
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            f"sidecar {key}={raw!r} is malformed; must be a JSON integer "
            "matching the NPZ artifact exactly")
    return int(raw)


def _sidecar_exact_str(prov: dict, key: str) -> str:
    if key not in prov:
        raise ValueError(
            f"sidecar provenance missing {key} required for NPZ consistency")
    raw = prov[key]
    if not isinstance(raw, str) or raw.strip() == "":
        raise ValueError(
            f"sidecar {key}={raw!r} is malformed; must be a non-empty string "
            "matching the NPZ artifact exactly")
    return raw


def _assert_sidecar_matches_npz(
    prov: dict,
    *,
    sigma_min: float,
    sigma_max: float,
    spatial_window: float | None,
    h_panel: float,
    gl_order: int,
    geometry_sha256: str,
    events_sha256: str,
    n_knots: int,
    n_events: int,
) -> None:
    """Reject contradictions between JSON sidecar and NPZ arrays/scalars.

    Does not silently prefer either source. Nested ``extra`` is ignored.
    ``table_id`` is descriptive only and is not recomputed here.
    """
    for key, expected in (
        ("sigma_min", float(sigma_min)),
        ("sigma_max", float(sigma_max)),
        ("h_panel", float(h_panel)),
    ):
        got = _sidecar_exact_float(prov, key)
        if got != expected:
            raise ValueError(
                f"sidecar {key}={got!r} contradicts NPZ {key}={expected!r}")

    if "spatial_window" not in prov:
        raise ValueError(
            "sidecar provenance missing spatial_window required for NPZ "
            "consistency")
    raw_ws = prov["spatial_window"]
    if spatial_window is None:
        if raw_ws is not None:
            raise ValueError(
                f"sidecar spatial_window={raw_ws!r} contradicts NPZ "
                "spatial_window=None (NaN sentinel)")
    else:
        if isinstance(raw_ws, bool) or not isinstance(raw_ws, (int, float)):
            raise ValueError(
                f"sidecar spatial_window={raw_ws!r} is malformed; must be a "
                "JSON number matching the NPZ artifact exactly")
        got_ws = float(raw_ws)
        if not np.isfinite(got_ws) or got_ws != float(spatial_window):
            raise ValueError(
                f"sidecar spatial_window={raw_ws!r} contradicts NPZ "
                f"spatial_window={spatial_window!r}")

    for key, expected in (
        ("gl_order", int(gl_order)),
        ("n_knots", int(n_knots)),
        ("n_events", int(n_events)),
    ):
        got = _sidecar_exact_int(prov, key)
        if got != expected:
            raise ValueError(
                f"sidecar {key}={got!r} contradicts NPZ {key}={expected!r}")

    for key, expected in (
        ("geometry_sha256", str(geometry_sha256)),
        ("events_sha256", str(events_sha256)),
    ):
        got = _sidecar_exact_str(prov, key)
        if got != expected:
            raise ValueError(
                f"sidecar {key}={got!r} contradicts NPZ {key}={expected!r}")


def _geometry_sha256(poly) -> str:
    wkb = poly.wkb if hasattr(poly, "wkb") else bytes(str(poly), "utf-8")
    return hashlib.sha256(wkb).hexdigest()


def _events_sha256(x: np.ndarray, y: np.ndarray) -> str:
    """Exact event-identity hash over little-endian float64 coordinates.

    Includes both arrays, their shapes, row order, and the algorithm tag.
    Never uses rounded decimal formatting (legacy ``.9g`` hashes are
    incompatible with ``EVENTS_HASH_ALGORITHM``).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape or x_arr.ndim != 1:
        raise ValueError(
            f"event x/y must be 1-d and same shape; got {x_arr.shape} and "
            f"{y_arr.shape}")
    x_le = np.ascontiguousarray(x_arr, dtype="<f8")
    y_le = np.ascontiguousarray(y_arr, dtype="<f8")
    shape_x = np.asarray(x_le.shape, dtype="<i8")
    shape_y = np.asarray(y_le.shape, dtype="<i8")
    payload = (
        EVENTS_HASH_ALGORITHM.encode("ascii")
        + b"\0"
        + shape_x.tobytes()
        + shape_y.tobytes()
        + x_le.tobytes()
        + y_le.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_compat_provenance(prov: dict) -> None:
    """Reject missing/malformed/legacy compatibility metadata.

    ``prov['extra']`` is descriptive only and is never consulted here.
    """
    missing = [k for k in _REQUIRED_COMPAT_PROVENANCE_KEYS if k not in prov]
    if missing:
        raise ValueError(
            "supplied mass table provenance is missing required compatibility "
            f"field(s) {missing}; legacy or incomplete tables are incompatible "
            f"with {BACKEND_SCHEMA_VERSION}. Rebuild with "
            "bstpp.polygon_mass.prepare_polygon_mass_table(...).")

    checks = (
        ("backend", BACKEND_ID, str),
        ("backend_schema", BACKEND_SCHEMA_VERSION, str),
        ("sigma_parameterization", SIGMA_PARAMETERIZATION, str),
        ("interpolation_convention", INTERPOLATION_CONVENTION, str),
        ("slope_method", SLOPE_METHOD, str),
        ("events_hash_algorithm", EVENTS_HASH_ALGORITHM, str),
    )
    for key, expected, caster in checks:
        raw = prov[key]
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            raise ValueError(
                f"supplied mass table provenance field {key!r} is missing or "
                f"malformed; incompatible with {BACKEND_SCHEMA_VERSION}")
        try:
            got = caster(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"supplied mass table provenance field {key!r} is malformed "
                f"({raw!r})") from exc
        if got != expected:
            raise ValueError(
                f"supplied mass table {key}={got!r} is incompatible with the "
                f"current builder/evaluator contract {key}={expected!r}")

    raw_eps = prov["slope_fd_eps"]
    try:
        eps = float(raw_eps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"supplied mass table slope_fd_eps is malformed ({raw_eps!r})"
        ) from exc
    if not np.isfinite(eps) or float(eps) != float(SLOPE_FD_EPS):
        raise ValueError(
            f"supplied mass table slope_fd_eps={raw_eps!r} is incompatible "
            f"with slope_fd_eps={SLOPE_FD_EPS}")


def assert_polygon_mass_table_budget(
    table: PolygonMassTable,
    *,
    sigma_min: float,
) -> float:
    """Reject a table whose recorded panel fails the prepare-time ratio prefilter.

    The table's own ``h_panel`` / ``gl_order`` are authoritative. This check is
    necessary but not sufficient for ``PRODUCTION_TAU_ABS``; install also runs
    ``measure_polygon_mass_table_residual``. Prefilter::

        table.h_panel / sigma_min <= MAX_PANEL_TO_MIN_SIGMA_RATIO

    Returns the realized ``panel / min_sigma`` ratio. Never rebuilds a table.
    """
    if not isinstance(table, PolygonMassTable):
        raise TypeError(
            f"mass_table must be a PolygonMassTable; got {type(table).__name__}")
    h = float(table.h_panel)
    s = float(sigma_min)
    if not (math.isfinite(h) and h > 0.0):
        raise ValueError(
            f"supplied mass table h_panel must be finite and > 0; got {h!r}")
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError(
            f"model min_sigma must be finite and > 0 for the mass-table "
            f"accuracy budget; got {sigma_min!r}")
    try:
        gl = int(table.gl_order)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"supplied mass table gl_order must be an integer; "
            f"got {table.gl_order!r}") from exc
    if gl < 1:
        raise ValueError(
            f"supplied mass table gl_order must be >= 1; got {gl}")
    ratio = h / s
    if ratio > MAX_PANEL_TO_MIN_SIGMA_RATIO:
        raise ValueError(
            "supplied mass table is too coarse for model min_sigma under the "
            f"panel-resolution prefilter (PRODUCTION_TAU_ABS={PRODUCTION_TAU_ABS} "
            "is enforced by measured residual at install): "
            f"table h_panel={h}, gl_order={gl}, min_sigma={s}, "
            f"panel/min_sigma ratio={ratio} exceeds "
            f"MAX_PANEL_TO_MIN_SIGMA_RATIO={MAX_PANEL_TO_MIN_SIGMA_RATIO}. "
            "Rebuild with prepare_polygon_mass_table(..., panel_h_m=...) so "
            "effective_panel_h / min_sigma <= "
            f"{MAX_PANEL_TO_MIN_SIGMA_RATIO}."
        )
    return float(ratio)


def measure_polygon_mass_table_residual(
    table: PolygonMassTable,
    *,
    domain_geom,
    event_x_real: np.ndarray,
    event_y_real: np.ndarray,
    spatial_window: float | None,
) -> float:
    """Max abs Hermite-vs-reference residual at the table's ``h_panel`` / knots.

    Reference method: host NumPy/SciPy float64 ``_quad_masses_numpy`` at
    ``BUDGET_REFERENCE_GL_ORDER`` on ``prepare_quadrature(..., h=table.h_panel)``.
    Reference's own error bound vs the independent shapely §13 oracle is
    ``BUDGET_REFERENCE_ORACLE_BOUND`` (see module constants). Probes knots and
    mid-interval sigmas (endpoints + mid + mid-interval thirds).
    """
    if not isinstance(table, PolygonMassTable):
        raise TypeError(
            f"mass_table must be a PolygonMassTable; got {type(table).__name__}")
    x = np.asarray(event_x_real, dtype=np.float64)
    y = np.asarray(event_y_real, dtype=np.float64)
    h = float(table.h_panel)
    ws = None if spatial_window is None else float(spatial_window)
    prep = prepare_quadrature(domain_geom, x, y, h, ws)
    glx01, glw01 = _legendre_01(BUDGET_REFERENCE_GL_ORDER)
    glx01 = np.asarray(glx01, dtype=np.float64)
    glw01 = np.asarray(glw01, dtype=np.float64)
    ev = np.asarray(prep.ev_xy, dtype=np.float64)
    panels = np.asarray(prep.panels, dtype=np.float64)
    mask = np.asarray(prep.mask, dtype=np.float64)
    inside = np.asarray(prep.inside_flag, dtype=np.float64)

    log_k = np.asarray(table.log_knots, dtype=np.float64)
    mid = 0.5 * (log_k[:-1] + log_k[1:])
    probe_logs = np.unique(np.concatenate([
        log_k[[0, len(log_k) // 2, -1]],
        mid[[0, len(mid) // 2, -1]],
    ]))

    masses_fn, _, _ = make_table_eval()
    lk = jnp.asarray(table.log_knots)
    vj = jnp.asarray(table.values)
    sj = jnp.asarray(table.slopes)
    max_abs = 0.0
    for ls in probe_logs:
        got = np.asarray(masses_fn(float(ls), lk, vj, sj), dtype=np.float64)
        want = _quad_masses_numpy(
            float(ls), ev, panels, mask, inside, glx01, glw01, ws)
        max_abs = max(max_abs, float(np.max(np.abs(got - want))))
    return float(max_abs)


def assert_polygon_mass_table_accuracy(
    table: PolygonMassTable,
    *,
    domain_geom,
    event_x_real: np.ndarray,
    event_y_real: np.ndarray,
    spatial_window: float | None,
) -> float:
    """Reject a table whose measured residual exceeds ``PRODUCTION_TAU_ABS``."""
    max_abs = measure_polygon_mass_table_residual(
        table,
        domain_geom=domain_geom,
        event_x_real=event_x_real,
        event_y_real=event_y_real,
        spatial_window=spatial_window,
    )
    if max_abs > PRODUCTION_TAU_ABS:
        raise ValueError(
            "supplied mass table fails the production accuracy budget "
            f"PRODUCTION_TAU_ABS={PRODUCTION_TAU_ABS}: measured max abs "
            f"residual={max_abs} against host float64 panel quadrature at "
            f"BUDGET_REFERENCE_GL_ORDER={BUDGET_REFERENCE_GL_ORDER} on the "
            f"table's h_panel={float(table.h_panel)} tiling "
            f"(gl_order={int(table.gl_order)}; reference oracle bound "
            f"BUDGET_REFERENCE_ORACLE_BOUND={BUDGET_REFERENCE_ORACLE_BOUND}). "
            "Rebuild with prepare_polygon_mass_table at higher gl_order "
            "and/or smaller panel_h_m."
        )
    return float(max_abs)


def validate_polygon_mass_table(
    table: PolygonMassTable,
    *,
    domain_geom,
    event_x_real: np.ndarray,
    event_y_real: np.ndarray,
    spatial_window: float | None,
    sigma_min: float,
    sigma_max: float,
) -> tuple[float, float]:
    """Reject a supplied Hermite table that is not identity-compatible.

    Equal event counts are not evidence of compatibility. Validates required
    compatibility provenance (backend, schema, sigma parameterization,
    interpolation convention, slope method/settings, event-hash algorithm),
    domain geometry hash, exact float64 event identity and row order, event
    count, spatial window, sigma range and knot grid, array shapes, and
    finite array values. Build settings ``h_panel`` / ``gl_order`` are read
    from the table itself (never compared to caller-declared defaults).
    Acceptance for ``PRODUCTION_TAU_ABS`` is a measured residual against the
    elevated-GL host quadrature reference (plus the panel-ratio prefilter).

    Returns ``(panel/min_sigma ratio, measured max abs residual)``.
    Descriptive ``provenance['extra']`` is ignored for compatibility.
    """
    if not isinstance(table, PolygonMassTable):
        raise TypeError(
            f"mass_table must be a PolygonMassTable; got {type(table).__name__}")

    if not isinstance(table.provenance, dict):
        raise ValueError(
            "supplied mass table provenance must be a dict; legacy tables "
            f"without {BACKEND_SCHEMA_VERSION} metadata are incompatible")
    _validate_compat_provenance(table.provenance)

    x = np.asarray(event_x_real, dtype=np.float64)
    y = np.asarray(event_y_real, dtype=np.float64)
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

    # Table-recorded build settings are authoritative; ratio prefilter then
    # measured residual vs PRODUCTION_TAU_ABS.
    ratio = assert_polygon_mass_table_budget(table, sigma_min=float(sigma_min))

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

    residual = assert_polygon_mass_table_accuracy(
        table,
        domain_geom=domain_geom,
        event_x_real=x,
        event_y_real=y,
        spatial_window=spatial_window,
    )
    return float(ratio), float(residual)


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
    eps: float = SLOPE_FD_EPS,
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
    process-global JAX config. Descriptive ``extra_provenance`` is stored
    under nested ``provenance['extra']`` and never overwrites builder-owned
    compatibility fields.
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
        "backend": BACKEND_ID,
        "backend_schema": BACKEND_SCHEMA_VERSION,
        "sigma_parameterization": SIGMA_PARAMETERIZATION,
        "interpolation_convention": INTERPOLATION_CONVENTION,
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "n_knots": int(knots.shape[0]),
        "validated_dlog": float(VALIDATED_DLOG),
        "spatial_window": None if ws is None else float(ws),
        "h_panel": float(h_panel),
        "gl_order": int(gl_order),
        "geometry_sha256": geom_hash,
        "events_sha256": ev_hash,
        "events_hash_algorithm": EVENTS_HASH_ALGORITHM,
        "table_id": table_id,
        "build_seconds": build_s,
        "dtype_build": "float64",
        "n_events": int(len(x)),
        "slope_method": SLOPE_METHOD,
        "slope_fd_eps": float(SLOPE_FD_EPS),
    }
    if extra_provenance is not None:
        if not isinstance(extra_provenance, dict):
            raise TypeError(
                "extra_provenance must be a dict of descriptive metadata; "
                f"got {type(extra_provenance).__name__}")
        # Copy; nest under extra so callers cannot overwrite reserved fields.
        prov["extra"] = dict(extra_provenance)
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

    Descriptive ``extra_provenance`` is copied under nested
    ``provenance['extra']`` and cannot overwrite builder-owned compatibility
    fields. Tables are schema
    ``hybrid_quad_hermite_numpy_v2`` with exact little-endian float64 event
    identity; legacy v1 / decimal-``.9g`` hashes are intentionally
    incompatible and must be rebuilt.
    """
    from .excitation_support import metres_to_crs_units

    if crs is not None and not getattr(crs, "is_geographic", False):
        h_panel = float(metres_to_crs_units(float(panel_h_m), crs))
    else:
        h_panel = float(panel_h_m)

    min_s = float(min_sigma)
    if not (math.isfinite(min_s) and min_s > 0):
        raise ValueError(
            f"min_sigma must be finite and > 0; got {min_sigma!r}")
    if not (math.isfinite(h_panel) and h_panel > 0):
        raise ValueError(
            f"effective panel height must be finite and > 0; got {h_panel!r} "
            f"(from panel_h_m={panel_h_m!r})")
    ratio = h_panel / min_s
    if ratio > MAX_PANEL_TO_MIN_SIGMA_RATIO:
        raise ValueError(
            "Polygon mass panel is too coarse relative to min_sigma: "
            f"effective_panel_h={h_panel} (domain-coordinate units), "
            f"min_sigma={min_s}, ratio={ratio} exceeds allowed "
            f"MAX_PANEL_TO_MIN_SIGMA_RATIO={MAX_PANEL_TO_MIN_SIGMA_RATIO}. "
            "Pass a smaller explicit panel_h_m so "
            "effective_panel_h / min_sigma <= "
            f"{MAX_PANEL_TO_MIN_SIGMA_RATIO}.")

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
                             min_sigma_real: float | None = None,
                             frac: float = 0.9) -> None:
    """Warn when posterior sigma approaches a declared lower or upper bound."""
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
    if min_sigma_real is not None:
        q05 = float(np.quantile(s, 0.05))
        lo = float(min_sigma_real)
        # Mirror the upper check: within (1-frac) of the bound from above.
        if q05 <= lo / frac:
            warnings.warn(
                f"Posterior sigma 5th percentile ({q05:.6g}) is within "
                f"{(1.0/frac - 1.0):.0%} of min_sigma ({lo:.6g}). Sigma may "
                "be weakly identified against the lower truncation; consider "
                "lowering min_sigma and rebuilding the polygon-mass table.",
                UserWarning,
                stacklevel=2,
            )
