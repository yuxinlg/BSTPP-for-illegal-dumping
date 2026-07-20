"""Phase 3b domain/partition seam: the three data-bearing objects.

Behavior-preserving extraction (phase3_baseline_and_decisions section 10.b)
of the model-construction pipeline into exactly three objects:

- :class:`ModelData` -- the user's inputs as supplied: events, domain input,
  horizon, covariate sources. No derived quantities.
- :class:`PreparedDomain` -- validated model geometry: bounding rectangle,
  area ratio, CRS/unit information, the real/internal axis scales.
- :class:`PreparedPartitions` -- field cells, clipped/intersection
  geometries, common refinements, areas, index maps, seasonal overlap W
  (extracted in the follow-up 3b commit).

Deferred vocabulary (contracts, NOT classes -- do not add them here):
*ModelDomain* (the scientific/observed-domain contract statement),
*ReportingRegions*, *ComputationPartition* (10.b, deferred until consumers
exist). The legacy ``args`` dict remains the numpyro-facing view, populated
from these objects by the constructor; its removal is OP-8 (3f).

Binding invariant (10.b.7): excitation pairs are a function of the global
event set and cutoff/window only -- nothing in this module may feed a
partition or tile into pair construction.

Every numerical expression here is moved VERBATIM from the constructor at
the pre-3b tip; bit-identity is gated by the four-config golden pins.
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import jax.numpy as jnp
from shapely.geometry import Polygon

# Internal-coordinate geometry, previously inline magic numbers in the
# constructor. Values are pinned by the pretrained decoder output shapes
# (decoder_1d_T50_fixed_ls, decoder_1d_T24_circ_small_l8,
# 2d_decoder_15_5_large; see bstpp/decode_fields.py) -- they become real
# configuration only with the 3f decoder contract.
T_INTERNAL = 50   # internal time horizon (args['T'])
S_INTERNAL = 24   # internal seasonal coordinate length (args['S'])
S_DAYS = 365      # real seasonal period in days (self.S)
N_T = 50          # temporal field cells (decoder-pinned)
N_S = 24          # seasonal field cells (decoder-pinned)
N_XY = 25         # spatial field cells per axis (decoder-pinned)


@dataclass(frozen=True)
class ModelData:
    """The user's inputs, exactly as supplied (post file-loading of events).

    ``covariate_source`` is kept raw (path, DataFrame, or GeoDataFrame as
    passed); covariate normalization happens downstream so that load-error
    ordering is unchanged from the legacy constructor.
    """

    events: pd.DataFrame
    domain: Union[np.ndarray, gpd.GeoDataFrame]
    horizon_days: float
    offset_seasonal: float
    covariate_source: Optional[Any] = None
    cov_names: Optional[list] = None
    cov_grid_size: Optional[Any] = None


@dataclass(frozen=True)
class PreparedDomain:
    """Validated model geometry and the declared real/internal unit scales.

    ``bounds`` is the real-unit bounding rectangle A_ ([[x0,x1],[y0,y1]]);
    ``area_ratio`` is |A| / |A_rect| (1 for rectangle domains);
    ``axis_scales`` are the per-axis REAL lengths of the bounding rectangle,
    the declared conversion constants at the internal/real unit boundary
    (consumed by the event-term atom and the excitation compensator only).
    ``domain`` is the model domain as supplied (polygon GeoDataFrame or the
    rectangle array); ``is_polygon`` distinguishes the two constructor
    regimes; ``crs`` is the domain CRS or None.
    """

    domain: Union[np.ndarray, gpd.GeoDataFrame]
    bounds: np.ndarray
    area_ratio: float
    axis_scales: jnp.ndarray
    crs: Optional[Any]
    is_polygon: bool


def prepare_domain(A: Union[np.ndarray, gpd.GeoDataFrame]) -> PreparedDomain:
    """Build the PreparedDomain from the supplied domain input.

    Verbatim extraction of the legacy constructor's domain block, including
    the geographic-coordinate contract warning (a declared CRS is
    authoritative; the bounds heuristic covers array domains and CRS-less
    GeoDataFrames). Warning semantics, message, and trigger conditions are
    unchanged; stacklevel is raised by one to point at the same frame as
    before the extraction.
    """
    if type(A) is gpd.GeoDataFrame:
        A_ = np.stack((A.bounds.min(axis=0)[['minx', 'miny']],
                       A.bounds.max(axis=0)[['maxx', 'maxy']])).T
        # proportion of area of rectangle A_ covered by A. Used for Hawkes integral.
        area_ratio = A.area.sum() / ((A_[0, 1] - A_[0, 0]) * (A_[1, 1] - A_[1, 0]))
    else:  # A is rectangle specified by np.array
        area_ratio = 1
        A_ = A
    # Per-axis REAL lengths of the bounding rectangle: the affine ingestion
    # map is x_int = (x - x_min) / axis_scales[0] (and likewise in y).
    # The spatial-trigger contract is REAL-unit -- the kernel is isotropic
    # in the units of the input X/Y columns -- so these scales are the
    # declared conversion constants at the internal/real unit boundary
    # (consumed by the event-term atom, the excitation compensator, and
    # nothing else; the background never needs them).
    A_np = np.asarray(A_, dtype=float)
    axis_scales = jnp.asarray(A_np[:, 1] - A_np[:, 0])
    # Data-contract warning (warns, never blocks): the spatial trigger is
    # isotropic in the units of the input X/Y columns, so GEOGRAPHIC
    # coordinates (lon/lat degrees) make the "isotropic" kernel
    # anisotropic ON THE GROUND -- one degree of longitude shrinks by
    # cos(latitude) -- silently reintroducing the aspect-ratio defect the
    # real-unit contract removed. Project to a metric CRS (e.g. a state
    # plane or UTM zone) before ingestion. A declared CRS on a
    # GeoDataFrame domain is AUTHORITATIVE (crs.is_geographic decides both
    # ways); the bounds heuristic is the fallback for array domains and
    # CRS-less GeoDataFrames only.
    _geo_warning = (
        "Spatial domain %s geographic coordinates (lon/lat degrees). "
        "The spatial trigger is isotropic in COORDINATE units, so in "
        "degrees it is anisotropic on the ground by cos(latitude), and "
        "sigmax_2 / spatial_window are in squared degrees / degrees. "
        "Project X/Y to a metric CRS before ingestion.")
    _crs = A.crs if type(A) is gpd.GeoDataFrame else None
    if _crs is not None:
        if _crs.is_geographic:
            warnings.warn(_geo_warning % "has a geographic CRS, i.e. uses",
                          UserWarning, stacklevel=3)
    else:
        _x0, _x1 = float(A_np[0, 0]), float(A_np[0, 1])
        _y0, _y1 = float(A_np[1, 0]), float(A_np[1, 1])
        if (-180.0 <= _x0 <= 180.0 and -180.0 <= _x1 <= 180.0
                and -90.0 <= _y0 <= 90.0 and -90.0 <= _y1 <= 90.0
                and (_x1 - _x0) < 2.0 and (_y1 - _y0) < 2.0
                and max(abs(_x0), abs(_x1)) > 5.0
                and max(abs(_y0), abs(_y1)) > 5.0):
            warnings.warn(_geo_warning % "looks like",
                          UserWarning, stacklevel=3)
    return PreparedDomain(
        domain=A,
        bounds=A_,
        area_ratio=area_ratio,
        axis_scales=axis_scales,
        crs=_crs,
        is_polygon=type(A) is gpd.GeoDataFrame,
    )


@dataclass
class PreparedPartitions:
    """Field cells, clipped/intersection geometries, refinements, areas,
    index maps, and the seasonal overlap matrix W.

    Not frozen: the covariate leg attaches its products after event
    membership has been established (matching the legacy constructor's
    operation order exactly). ``comp_grid`` is the 25x25 spatial field grid
    in REAL coordinates; ``spatial_grid_cells`` the in-domain cell ids;
    ``season_overlap`` the exact (n_t x n_s) matrix W of eq. (26).

    Covariate fields (None without covariates): ``cov_gdf`` the normalized
    covariate GeoDataFrame; ``cov_values`` the (standardized) design matrix;
    ``int_df`` the comp-grid x covariate common refinement with exact
    intersection areas (Cox/LGCP); ``cov_area`` the covariate-cell areas
    (plain Hawkes); ``integration_*`` the pure-NumPy arrays consumed by the
    spatial_refinement_integral atom.
    """

    n_t: int
    x_t: jnp.ndarray
    n_s: int
    x_a: jnp.ndarray
    n_xy: int
    comp_grid: gpd.GeoDataFrame
    spatial_grid_cells: np.ndarray
    season_idx_of_t: np.ndarray
    season_overlap: jnp.ndarray
    cov_gdf: Optional[gpd.GeoDataFrame] = None
    cov_values: Optional[np.ndarray] = None
    int_df: Optional[gpd.GeoDataFrame] = None
    cov_area: Optional[np.ndarray] = None
    integration_field_indices: Optional[np.ndarray] = None
    integration_cov_indices: Optional[np.ndarray] = None
    integration_areas: Optional[np.ndarray] = None


def prepare_partitions(domain: PreparedDomain, horizon_days: float,
                       offset_seasonal: float) -> PreparedPartitions:
    """Build the temporal/seasonal/spatial partitions on a PreparedDomain.

    Verbatim extraction of the legacy constructor's grid block: internal
    grids x_t/x_a, the real-coordinate 25x25 comp_grid tiled over the
    bounding rectangle, in-domain cell selection (sjoin with the polygon;
    all cells for rectangle domains), the diagnostic season_idx_of_t, and
    the exact seasonal overlap matrix W.
    """
    A_ = domain.bounds
    T = horizon_days

    # time grid
    n_t = N_T
    x_t = jnp.arange(0, T_INTERNAL, T_INTERNAL / n_t)

    # seasonal grid
    n_s = N_S
    x_a = jnp.arange(0, S_INTERNAL, S_INTERNAL / n_s)

    # spatial grid
    n_xy = N_XY
    cols = np.arange(0, 1, 1 / n_xy)
    polygons = []
    for y in cols:
        for x in cols:
            polygons.append(Polygon([(x, y), (x + 1 / n_xy, y),
                                     (x + 1 / n_xy, y + 1 / n_xy),
                                     (x, y + 1 / n_xy)]))
    comp_grid = gpd.GeoDataFrame({'geometry': polygons,
                                  'comp_grid_id': np.arange(n_xy ** 2)})
    comp_grid.geometry = comp_grid.geometry.scale(
        xfact=A_[0, 1] - A_[0, 0], yfact=A_[1, 1] - A_[1, 0],
        origin=(0, 0)).translate(A_[0, 0], A_[1, 0])

    if domain.is_polygon:
        # find grid cells overlapping with A
        comp_grid.crs = domain.domain.crs
        spatial_grid_cells = np.unique(
            comp_grid.sjoin(domain.domain, how='inner')['comp_grid_id'])
    else:
        spatial_grid_cells = np.arange(N_XY ** 2)

    # Seasonal index of each temporal grid cell midpoint (diagonal a = sigma(t)):
    # internal cell i covers real days [i, i+1) * (T / n_t); map its midpoint through
    # day-of-year (mod S_DAYS) onto the internal seasonal grid [0, S_INTERNAL).
    # NOTE: the likelihood no longer integrates via this midpoint index (it uses the
    # exact overlap matrix season_overlap below); season_idx_of_t is retained for
    # diagnostics (rate_time) and scripts/recover_test.py's intercept combination.
    t_mid_days = (np.arange(n_t) + 0.5) * (T / n_t)
    a_mid = ((t_mid_days + offset_seasonal) % S_DAYS) / S_DAYS * S_INTERNAL
    season_idx_of_t = np.searchsorted(np.asarray(x_a), a_mid, side='right') - 1

    # Exact seasonal overlap matrix W (n_t x n_s), in INTERNAL time units:
    #   W[i, k] = measure{ t in temporal cell i : sigma(t) in seasonal cell k }.
    # f_t (n_t cells) and f_a (n_s cells) are piecewise constant but the cell widths
    # differ (T/n_t days vs S_DAYS/n_s days), so temporal cells straddle seasonal
    # boundaries and the time integral of exp(a_0 + f_t + f_a[sigma(t)]) has a closed
    # form: contract exp(f_a) against W. Temporal cell i covers real days
    # [i, i+1) * (T / n_t); sigma(d) = (d + offset_seasonal) mod S_DAYS; seasonal cell
    # k = floor(sigma / S_DAYS * n_s). In the shifted coordinate s = d + offset_seasonal
    # every split point we need -- seasonal-cell edges AND the year wrap -- is an integer
    # multiple of h_day = S_DAYS / n_s (since S_DAYS = n_s * h_day), and on the interval
    # [p*h_day, (p+1)*h_day) the seasonal cell is p mod n_s. Day-lengths convert to
    # internal units via * (T_INTERNAL / T).
    h_day = S_DAYS / n_s
    dt_day = T / n_t
    W = np.zeros((n_t, n_s))
    for i in range(n_t):
        s = i * dt_day + offset_seasonal
        s_end = (i + 1) * dt_day + offset_seasonal
        while s < s_end - 1e-9:
            p = int(np.floor(s / h_day + 1e-9))
            seg_end = min((p + 1) * h_day, s_end)
            W[i, p % n_s] += seg_end - s
            s = seg_end
    W *= T_INTERNAL / T
    assert np.allclose(W.sum(axis=1), T_INTERNAL / n_t, rtol=1e-6), \
        "season_overlap rows must sum to the internal temporal cell width T_INTERNAL/n_t"

    return PreparedPartitions(
        n_t=n_t, x_t=x_t, n_s=n_s, x_a=x_a, n_xy=n_xy,
        comp_grid=comp_grid, spatial_grid_cells=spatial_grid_cells,
        season_idx_of_t=season_idx_of_t, season_overlap=jnp.asarray(W))


def attach_covariate_partitions(partitions: PreparedPartitions,
                                domain: PreparedDomain,
                                spatial_cov: gpd.GeoDataFrame,
                                cov_names: list,
                                standardize_cov: bool,
                                model: str) -> None:
    """Attach the covariate leg's partition products (geometry side only).

    Verbatim extraction: the (standardized) design matrix, and for Cox/LGCP
    the comp-grid x covariate common refinement with exact intersection
    areas plus the covariate-based in-domain cell override; for plain
    Hawkes the covariate-cell areas. Event membership (cov_ind) is NOT
    built here -- it is event-side and stays with the constructor's
    membership step.
    """
    A_ = domain.bounds
    partitions.cov_gdf = spatial_cov

    X_s = spatial_cov[cov_names].values
    if X_s.ndim == 1:
        X_s = X_s[:, None]
    # standardize covariates
    if standardize_cov:
        partitions.cov_values = (X_s - X_s.mean(axis=0)) / (X_s.var(axis=0) ** 0.5)
    else:
        partitions.cov_values = X_s

    if model in ['lgcp', 'cox_hawkes']:
        partitions.comp_grid.crs = spatial_cov.crs
        # find covariate cell intersection with computational grid cells area
        intersect = gpd.overlay(partitions.comp_grid, spatial_cov,
                                how='intersection', keep_geom_type=True)
        intersect['area'] = intersect.area / ((A_[0, 1] - A_[0, 0]) * (A_[1, 1] - A_[1, 0]))
        intersect = intersect[intersect['area'] > 1e-10]
        partitions.int_df = intersect
        # find cells on the computational grid that are in the domain
        partitions.spatial_grid_cells = np.unique(
            partitions.comp_grid.sjoin(spatial_cov, how='inner')['comp_grid_id'])
    else:
        partitions.cov_area = (spatial_cov.area / ((A_[0, 1] - A_[0, 0]) * (A_[1, 1] - A_[1, 0]))).values


def finalize_integration_arrays(partitions: PreparedPartitions,
                                model: str) -> None:
    """Derive the pure-NumPy integration arrays for the
    spatial_refinement_integral atom (eq. 24) -- once, here, so no pandas /
    GeoPandas object is ever read inside traced likelihood code. The
    no-covariate grid is the special case of the common refinement with
    uniform cell areas 1/n_xy^2. Verbatim extraction.
    """
    if model in ['lgcp', 'cox_hawkes']:
        if partitions.int_df is not None:
            partitions.integration_field_indices = partitions.int_df['comp_grid_id'].values
            partitions.integration_cov_indices = partitions.int_df['cov_ind'].values
            partitions.integration_areas = partitions.int_df['area'].values
        else:
            partitions.integration_field_indices = partitions.spatial_grid_cells
            partitions.integration_cov_indices = None
            partitions.integration_areas = np.full(
                len(partitions.spatial_grid_cells), 1.0 / partitions.n_xy ** 2,
                dtype=np.float32)
