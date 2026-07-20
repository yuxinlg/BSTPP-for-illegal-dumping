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
