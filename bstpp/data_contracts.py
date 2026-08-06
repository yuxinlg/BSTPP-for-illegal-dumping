"""Phase 3a data-contract validation layer (class IV).

Boundary validation constructed BEFORE model construction, per
phase3_record.tex (Part I §10.a / amendments): the constructor's job is
to fit the model the user asked for on exactly the events the user supplied;
anything that would silently alter the event set (NaN dropping, out-of-domain
acceptance, ambiguous membership) is either surfaced (report mode) or refused
(reject mode). Nothing here mutates, fills, snaps, or drops data.

Two modes (constructor argument ``data_contracts``):

- ``"reject"`` (default): any violation raises :class:`DataContractError`
  listing every offending row. The default was flipped from ``"report"`` on
  reviewer sign-off of the committed section-14 report-only dry run against
  the project data (2026-07-20; sole finding: five leap-day horizon
  violations, an upstream ``total_days`` defect -- see
  ``refactor-patches/phase3a/``).
- ``"report"``: violations and diagnostics are collected into a
  :class:`DataContractReport` (stored as ``model.data_contract_report``) and
  violations are emitted as a single ``UserWarning``. Construction still
  proceeds for migration dry-runs, but held-out scoring
  (``log_expected_likelihood``) now rejects all nonfinite inputs under every
  mode -- including ``report`` -- and does not silently drop NaN/Inf rows.

Violations (would-reject) vs diagnostics (informational, never reject):

- violations: missing/non-numeric/nonfinite event coordinates; event time
  outside [0, T]; event outside the model domain A (polygon boundary is
  INSIDE per D-4); invalid/empty domain or covariate geometry; nonfinite
  covariate values; CRS mismatch between domain and covariates; events not
  covered by the covariate layer.
- diagnostics: events exactly on the polygon boundary of A (valid, D-4);
  events exactly on internal computational-grid lines (valid; deterministic
  membership is the D-22 micro-rebaseline); events covered by more than one
  covariate polygon (tie).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import warnings

try:  # geopandas is a hard dependency of bstpp; guard only for doc tooling
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

EVENT_COLUMNS = ("X", "Y", "T")


class DataContractError(ValueError):
    """A data-contract violation under data_contracts='reject'."""


@dataclass
class ContractCheck:
    """One finding: a named check, the rows it implicates, and its severity.

    ``kind`` is "violation" (rejected under reject mode) or "diagnostic"
    (informational only). ``indices`` are positional row indices into the
    validated frame (events) or the covariate frame, as stated per check.
    ``geometry`` (3c coverage contract, doc section 10.c) carries the ACTUAL
    offending region as a shapely geometry -- gap/overlap/sliver exports --
    when the finding is geometric rather than row-based; None otherwise.
    """

    name: str
    kind: str
    message: str
    indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    geometry: object = None

    def __post_init__(self):
        self.indices = np.asarray(self.indices, dtype=int)
        if self.kind not in ("violation", "diagnostic"):
            raise ValueError(f"unknown check kind: {self.kind!r}")


@dataclass
class DataContractReport:
    """All findings from one validation pass, plus enough context to act."""

    checks: list
    n_events: int
    mode: str

    @property
    def violations(self):
        return [c for c in self.checks if c.kind == "violation"]

    @property
    def diagnostics(self):
        return [c for c in self.checks if c.kind == "diagnostic"]

    @property
    def ok(self):
        return not self.violations

    def summary(self) -> str:
        lines = [
            f"BSTPP data-contract report: {len(self.violations)} violation "
            f"check(s), {len(self.diagnostics)} diagnostic(s) over "
            f"{self.n_events} events (mode={self.mode})."
        ]
        for c in self.checks:
            head = "VIOLATION" if c.kind == "violation" else "diagnostic"
            shown = ", ".join(map(str, c.indices[:10]))
            more = "" if len(c.indices) <= 10 else f", ... ({len(c.indices)} total)"
            where = f" rows [{shown}{more}]" if len(c.indices) else ""
            lines.append(f"  {head} {c.name}: {c.message}{where}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Long-format export for the committed section-14 dry-run report."""
        rows = []
        for c in self.checks:
            if len(c.indices):
                for i in c.indices:
                    rows.append((c.name, c.kind, int(i), c.message))
            else:
                rows.append((c.name, c.kind, -1, c.message))
        return pd.DataFrame(rows, columns=["check", "kind", "row", "message"])


def _domain_union(A):
    """Union geometry of a GeoDataFrame domain, or None for array domains."""
    if gpd is not None and isinstance(A, gpd.GeoDataFrame):
        return A.geometry.union_all() if hasattr(A.geometry, "union_all") \
            else A.geometry.unary_union
    return None


def validate_domain(A) -> list:
    """Contract checks on the model domain A (GeoDataFrame or 2x2 array)."""
    checks = []
    if gpd is not None and isinstance(A, gpd.GeoDataFrame):
        if len(A) == 0 or A.geometry.is_empty.all():
            checks.append(ContractCheck(
                "domain_geometry_empty", "violation",
                "domain GeoDataFrame has no non-empty geometry"))
            return checks
        bad = np.flatnonzero(A.geometry.isna() | A.geometry.is_empty
                             | ~A.geometry.is_valid)
        if len(bad):
            checks.append(ContractCheck(
                "domain_geometry_invalid", "violation",
                "domain rows with missing/empty/invalid geometry "
                "(self-intersection etc.); repair upstream (e.g. make_valid), "
                "BSTPP will not repair silently", bad))
            return checks
        # Require polygonal support (Polygon / MultiPolygon). Do not dissolve
        # or repair nonpolygonal geometry into a domain.
        types = A.geometry.geom_type.to_numpy()
        nonpoly = np.flatnonzero(
            ~np.isin(types, ("Polygon", "MultiPolygon")))
        if len(nonpoly):
            checks.append(ContractCheck(
                "domain_not_polygonal", "violation",
                "domain GeoDataFrame rows must be Polygon or MultiPolygon "
                f"(got {sorted(set(types[nonpoly]))}); BSTPP will not "
                "reinterpret lines/points as areal support", nonpoly))
            return checks
        union = _domain_union(A)
        area = float(getattr(union, "area", float("nan")))
        if not (np.isfinite(area) and area > 0.0):
            checks.append(ContractCheck(
                "domain_nonpositive_area", "violation",
                "domain union must have finite positive area; "
                f"got area={area}"))
    else:
        A_np = np.asarray(A, dtype=float)
        if A_np.shape != (2, 2) or not np.all(np.isfinite(A_np)) \
                or A_np[0, 1] <= A_np[0, 0] or A_np[1, 1] <= A_np[1, 0]:
            checks.append(ContractCheck(
                "domain_rectangle_invalid", "violation",
                "array domain must be a finite 2x2 [[x0,x1],[y0,y1]] with "
                "x1>x0 and y1>y0"))
    return checks


def validate_events(data, A, T_max, n_xy=25) -> list:
    """Contract checks on the event frame against domain A and horizon T_max.

    Indices reported are positional rows of ``data``. The polygon boundary is
    inside the domain (D-4); internal grid-line coincidences are diagnostics
    only (deterministic membership per D-22 handles them).
    """
    checks = []
    try:
        T_val = float(T_max)
    except (TypeError, ValueError):
        T_val = float("nan")
    if not (np.isfinite(T_val) and T_val > 0.0):
        checks.append(ContractCheck(
            "horizon_invalid", "violation",
            f"model horizon T_max / horizon_days must be finite and positive; "
            f"got {T_max!r}"))
        return checks

    missing = [c for c in EVENT_COLUMNS if c not in data.columns]
    if missing:
        checks.append(ContractCheck(
            "event_columns_missing", "violation",
            f"required event columns absent: {missing}"))
        return checks

    vals = {}
    nonnum = np.zeros(len(data), dtype=bool)
    for col in EVENT_COLUMNS:
        v = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        nonnum |= ~np.isfinite(v)
        vals[col] = v
    bad = np.flatnonzero(nonnum)
    if len(bad):
        checks.append(ContractCheck(
            "event_coordinates_nonfinite", "violation",
            "rows with non-numeric or nonfinite X/Y/T (held-out scoring now "
            "rejects these under every data_contracts mode; constructor "
            "reject mode refuses them before fit)", bad))
    finite = ~nonnum

    t = vals["T"]
    bad = np.flatnonzero(finite & ((t < 0) | (t > float(T_max))))
    if len(bad):
        checks.append(ContractCheck(
            "event_time_out_of_range", "violation",
            f"event times outside the model horizon [0, {T_max}] days", bad))

    # Spatial containment. Domain-invalid inputs skip geometry checks (the
    # domain violation is already recorded by validate_domain).
    domain_checks = validate_domain(A)
    union = _domain_union(A)
    if union is not None and not any(c.kind == "violation" for c in domain_checks):
        import shapely
        pts = shapely.points(vals["X"], vals["Y"])
        covered = shapely.covers(union, pts)  # covers: boundary INSIDE (D-4)
        bad = np.flatnonzero(finite & ~covered)
        if len(bad):
            checks.append(ContractCheck(
                "event_outside_domain", "violation",
                "events outside the model domain polygon A (legacy behavior: "
                "silently accepted when inside the bounding rectangle -- "
                "likelihood credit without compensator debit)", bad))
        on_boundary = covered & ~shapely.contains(union, pts)
        idx = np.flatnonzero(finite & on_boundary)
        if len(idx):
            checks.append(ContractCheck(
                "event_on_domain_boundary", "diagnostic",
                "events exactly on the boundary of A; INSIDE per D-4", idx))
        bounds = np.asarray(
            np.stack((A.bounds.min(axis=0)[["minx", "miny"]],
                      A.bounds.max(axis=0)[["maxx", "maxy"]])), dtype=float).T
    elif union is None and not any(c.kind == "violation" for c in domain_checks):
        bounds = np.asarray(A, dtype=float)
        x, y = vals["X"], vals["Y"]
        bad = np.flatnonzero(finite & (
            (x < bounds[0, 0]) | (x > bounds[0, 1])
            | (y < bounds[1, 0]) | (y > bounds[1, 1])))
        if len(bad):
            checks.append(ContractCheck(
                "event_outside_domain", "violation",
                "events outside the rectangular domain A (legacy behavior: "
                "misleading 'Computational grid does not encompass all data "
                "points!' crash)", bad))
    else:
        bounds = None

    # Internal computational-grid-line coincidences (diagnostic). Edges are
    # generated with the same arithmetic the grid construction uses
    # (k/n_xy scaled to the bounding rectangle), so equality is exact-float
    # against the geometry the legacy sjoin double-joins on.
    if bounds is not None:
        for axis, col in ((0, "X"), (1, "Y")):
            lo, hi = bounds[axis]
            interior_edges = (np.arange(1, n_xy) / n_xy) * (hi - lo) + lo
            idx = np.flatnonzero(finite & np.isin(vals[col], interior_edges))
            if len(idx):
                checks.append(ContractCheck(
                    f"event_on_grid_line_{col.lower()}", "diagnostic",
                    f"events exactly on an internal {col}-axis grid line of "
                    "the 25x25 computational grid (legacy behavior: "
                    "double-join crash; D-22 assigns the left-closed cell)",
                    idx))
    return domain_checks + checks


def validate_covariates(spatial_cov, cov_names, A, points_xy=None) -> list:
    """Contract checks on a covariate GeoDataFrame (post-normalization).

    ``points_xy`` (optional) is an (n, 2) array of event coordinates used for
    coverage (violation) and multi-cover tie (diagnostic) checks. Indices are
    covariate rows for geometry/value checks and event rows for
    coverage/ties.
    """
    checks = []
    if gpd is None or not isinstance(spatial_cov, gpd.GeoDataFrame):
        return checks

    bad = np.flatnonzero(spatial_cov.geometry.isna()
                         | spatial_cov.geometry.is_empty
                         | ~spatial_cov.geometry.is_valid)
    if len(bad):
        checks.append(ContractCheck(
            "covariate_geometry_invalid", "violation",
            "covariate rows with missing/empty/invalid geometry", bad))

    for name in (cov_names or []):
        if name not in spatial_cov.columns:
            checks.append(ContractCheck(
                "covariate_column_missing", "violation",
                f"covariate column {name!r} absent from spatial_cov"))
            continue
        v = pd.to_numeric(spatial_cov[name], errors="coerce").to_numpy(dtype=float)
        idx = np.flatnonzero(~np.isfinite(v))
        if len(idx):
            checks.append(ContractCheck(
                "covariate_values_nonfinite", "violation",
                f"non-numeric or nonfinite values in covariate {name!r} "
                "(legacy behavior: NaN propagates into the standardized "
                "design matrix and kills the likelihood)", idx))

    A_crs = A.crs if (gpd is not None and isinstance(A, gpd.GeoDataFrame)) else None
    if A_crs is not None and spatial_cov.crs is None:
        checks.append(ContractCheck(
            "crs_missing", "violation",
            f"domain declares CRS ({A_crs}) but covariates have no CRS; "
            "assign/reproject covariates to the domain CRS upstream "
            "(legacy behavior: silently assigns the domain CRS)"))
    elif A_crs is not None and spatial_cov.crs is not None \
            and A_crs != spatial_cov.crs:
        checks.append(ContractCheck(
            "crs_mismatch", "violation",
            f"domain CRS ({A_crs}) != covariate CRS ({spatial_cov.crs}); "
            "reproject upstream (legacy behavior: silently overrides the "
            "event CRS with the covariate CRS)"))

    if points_xy is not None and not len(bad):
        import shapely
        pts = shapely.points(points_xy[:, 0], points_xy[:, 1])
        tree = shapely.STRtree(spatial_cov.geometry.values)
        q = tree.query(pts, predicate="intersects")
        counts = np.bincount(q[0], minlength=len(pts))
        idx = np.flatnonzero(counts == 0)
        if len(idx):
            checks.append(ContractCheck(
                "event_missing_covariate", "violation",
                "events covered by no covariate polygon (legacy behavior: "
                "'Spatial covariates are not defined for all data points!' "
                "crash)", idx))
        idx = np.flatnonzero(counts > 1)
        if len(idx):
            checks.append(ContractCheck(
                "covariate_membership_tie", "diagnostic",
                "events covered by more than one covariate polygon (legacy "
                "behavior: duplicate-join crash; deterministic membership is "
                "the D-22 micro-rebaseline)", idx))
    return checks


# Normalized-area tolerance below which a coverage defect is a "sliver"
# diagnostic rather than a violation: the same 1e-10 internal-area threshold
# the common refinement uses to drop numerical slivers
# (bstpp.preparation.SLIVER_AREA_INTERNAL; duplicated here so this module
# stays importable without the jax stack).
COVERAGE_SLIVER_TOL = 1e-10


def _polygonal(geom):
    """Polygonal component of a geometry (drop lowerdim pieces)."""
    import shapely
    if geom.geom_type == "GeometryCollection":
        return shapely.unary_union(
            [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")])
    return geom


def validate_covariate_coverage(spatial_cov, A) -> list:
    """3c coverage contract (IV; D-7, doc section 10.c): the covariate layer
    must cover the model domain A exactly once.

    - ``covariate_gap`` (violation): region of A covered by NO covariate
      polygon. Legacy behavior: silent zero-valued covariate region; since
      the 3c-3 refinement, a silently uncharged region of the compensator.
    - ``covariate_overlap`` (violation): positive-area pairwise overlap
      between covariate polygons, clipped to A -- the refinement would
      double-charge it. ``indices`` are the covariate rows involved.
    - ``covariate_sliver`` (diagnostic): coverage defects with normalized
      area <= COVERAGE_SLIVER_TOL, the pieces the refinement drops anyway.

    Every finding EXPORTS the actual offending geometry on
    ``ContractCheck.geometry`` (not merely failing ids). Areas are
    normalized by the bounding-rectangle area, matching the internal
    measure of the refinement. A is authoritative (D-7): coverage is of A
    itself, polygon or rectangle.
    """
    checks = []
    if gpd is None or not isinstance(spatial_cov, gpd.GeoDataFrame) \
            or len(spatial_cov) == 0:
        return checks
    # Invalid covariate or domain geometry: skip coverage analysis -- GEOS
    # set operations on invalid inputs raise TopologyException. The
    # invalid-geometry VIOLATION is validate_covariates'/validate_domain's
    # job and is already recorded; same skip pattern as validate_events.
    if (spatial_cov.geometry.isna() | spatial_cov.geometry.is_empty
            | ~spatial_cov.geometry.is_valid).any():
        return checks
    if isinstance(A, gpd.GeoDataFrame) and \
            (A.geometry.isna() | A.geometry.is_empty
             | ~A.geometry.is_valid).any():
        return checks
    import shapely

    union = _domain_union(A)
    if union is None:
        A_np = np.asarray(A, dtype=float)
        union = shapely.box(A_np[0, 0], A_np[1, 0], A_np[0, 1], A_np[1, 1])
        bounds = A_np
    else:
        # .T so rows are per-axis (x0, x1) / (y0, y1), matching
        # preparation.prepare_domain's A_ construction.
        bounds = np.asarray(
            np.stack((A.bounds.min(axis=0)[["minx", "miny"]],
                      A.bounds.max(axis=0)[["maxx", "maxy"]])), dtype=float).T
    rect_area = float((bounds[0, 1] - bounds[0, 0])
                      * (bounds[1, 1] - bounds[1, 0]))

    def _classify(name, geom, message, indices=()):
        area = geom.area / rect_area
        if area <= 0.0:
            return
        if area <= COVERAGE_SLIVER_TOL:
            checks.append(ContractCheck(
                "covariate_sliver", "diagnostic",
                f"sub-tolerance {name.split('_', 1)[1]} of normalized area "
                f"{area:.3e} (<= {COVERAGE_SLIVER_TOL:.0e}; the refinement "
                "drops such pieces); geometry exported", indices,
                geometry=geom))
        else:
            checks.append(ContractCheck(
                name, "violation", message + "; geometry exported",
                indices, geometry=geom))

    # gap: A minus the covariate union
    cov_union = spatial_cov.geometry.union_all() \
        if hasattr(spatial_cov.geometry, "union_all") \
        else spatial_cov.geometry.unary_union
    gap = _polygonal(union.difference(cov_union))
    _classify(
        "covariate_gap", gap,
        f"the covariate layer leaves {gap.area / rect_area:.4%} of the "
        "model domain A uncovered (legacy behavior: silent zero-valued "
        "covariate region)")

    # overlaps: positive-area pairwise intersections, clipped to A
    geoms = spatial_cov.geometry.values
    tree = shapely.STRtree(geoms)
    pairs = tree.query(geoms, predicate="intersects")
    pieces, rows = [], set()
    for i, j in zip(*pairs):
        if i >= j:
            continue
        piece = _polygonal(shapely.intersection(geoms[i], geoms[j]))
        piece = _polygonal(shapely.intersection(piece, union))
        if piece.area > 0.0:
            pieces.append(piece)
            rows.update((int(i), int(j)))
    if pieces:
        _classify(
            "covariate_overlap", shapely.unary_union(pieces),
            "covariate polygons overlap with positive area inside A (the "
            "common refinement would charge the region more than once)",
            sorted(rows))
    return checks


def enforce(checks, n_events, mode) -> DataContractReport:
    """Assemble a report and apply the configured enforcement mode.

    reject: raise DataContractError on any violation.
    report: emit one UserWarning listing violations; never raise.
    """
    # Deferred: bstpp.config reaches jax through polygon_mass, and this module
    # is importable standalone today (measured: importing bstpp.data_contracts
    # loads neither jax nor bstpp.config). A top-level import would spend that
    # property to reach one function.
    from .config import ascii_safe

    if mode not in ("report", "reject"):
        raise ValueError(
            f"data_contracts must be 'report' or 'reject', got {mode!r}")
    report = DataContractReport(list(checks), int(n_events), mode)
    if not report.ok:
        if mode == "reject":
            # D-40 encoding corollary at the one raise site the ASCII sweep
            # cannot scan. summary() has no literal to scan BECAUSE it is
            # assembled at runtime -- and what it assembles is caller state:
            # covariate column labels ({name!r}) and CRS strings ({A_crs},
            # {spatial_cov.crs}). Not ASCII-safe by construction, so the
            # guarantee is applied here instead of asserted upstream.
            raise DataContractError(ascii_safe(report.summary()))
        warnings.warn(
            "BSTPP data-contract violations detected (report mode; these "
            "are rejected under the default data_contracts='reject'):\n"
            + report.summary(),
            UserWarning, stacklevel=3)
    return report
