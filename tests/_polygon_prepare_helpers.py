"""Helpers shared by polygon prepare-then-construct migrations."""

from __future__ import annotations

from shapely.geometry import box as shapely_box

from bstpp.polygon_mass import prepare_polygon_mass_table
from bstpp.preparation import prepare_domain


def prepare_table_for_model(
    data,
    A,
    *,
    min_sigma: float,
    max_sigma: float,
    spatial_window=None,
):
    """Build a PolygonMassTable matching how Hawkes_Model will validate it."""
    dom = prepare_domain(A)
    if dom.is_polygon:
        geom = dom.union_geometry
    else:
        A_ = dom.bounds
        geom = shapely_box(A_[0, 0], A_[1, 0], A_[0, 1], A_[1, 1])
    return prepare_polygon_mass_table(
        geom,
        data["X"].to_numpy(dtype=float),
        data["Y"].to_numpy(dtype=float),
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
        spatial_window=spatial_window,
        crs=dom.crs,
    )
