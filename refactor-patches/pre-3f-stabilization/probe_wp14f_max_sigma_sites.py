"""WP1.4f Step 1: enumerate by EXECUTION every site that can receive
max_sigma=None, and record what each does with it.

The A-26/A-27 precedent: the brief names one site; enumeration decides the
work. Run against the PRE-CHANGE tip.
"""
import os
import sys
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import numpy as np
from shapely.geometry import box

import bstpp

# A-25/A-26 provenance: a stale bstpp in site-packages shadows the repo for
# scripts run from a subdirectory. The first run of this probe loaded it and
# died on `bstpp.config`. Assert the object under measurement is the repo copy.
assert os.path.abspath(bstpp.__file__).startswith(_REPO), (
    f"probe loaded the WRONG bstpp: {bstpp.__file__} (repo={_REPO})")
from bstpp.config import NumericalConfig
from bstpp.excitation_support import resolve_sigma_bounds
from bstpp.polygon_mass import (
    DEFAULT_GL_ORDER,
    build_quad_table,
    prepare_polygon_mass_table,
    validate_polygon_mass_table,
)

POLY = box(0.0, 0.0, 200.0, 200.0)
EX = np.array([10.0, 20.0])
EY = np.array([10.0, 20.0])


def probe(name, fn):
    print(f"--- {name}")
    try:
        out = fn()
    except BaseException as e:  # noqa: BLE001 - identity is the measurement
        mro = [c.__name__ for c in type(e).__mro__]
        print(f"    RAISED {type(e).__name__}  MRO={mro}")
        print(f"    MSG   {e}")
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"    SITE  {tb.filename.split('bstpp')[-1]}:{tb.lineno}  {tb.line}")
    else:
        print(f"    ACCEPTED -> {type(out).__name__}")
        if name.startswith("resolve_sigma_bounds"):
            print(f"    value    {out[0]!r}, {out[1]!r}")
        if name.startswith("NumericalConfig"):
            print(f"    stored   min={out.min_sigma!r} max={out.max_sigma!r}")


print("bstpp.__file__ =", bstpp.__file__)

probe("resolve_sigma_bounds(polygon, max_sigma=None, crs=None)",
      lambda: resolve_sigma_bounds(
          mode="polygon", min_sigma=5.0, max_sigma=None, crs=None))

import geopandas as gpd  # noqa: E402
CRS_M = gpd.GeoSeries([box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:32618").crs

probe("resolve_sigma_bounds(polygon, max_sigma=None, crs=EPSG:32618)",
      lambda: resolve_sigma_bounds(
          mode="polygon", min_sigma=1000.0, max_sigma=None, crs=CRS_M))

probe("resolve_sigma_bounds(rectangle, max_sigma=None)",
      lambda: resolve_sigma_bounds(
          mode="rectangle", min_sigma=5.0, max_sigma=None, crs=None))

probe("NumericalConfig.create(polygon, max_sigma=None)",
      lambda: NumericalConfig.create(
          support_mode="polygon", min_sigma=5.0, max_sigma=None, panel_h_m=1.0))

probe("prepare_polygon_mass_table(max_sigma=None)",
      lambda: prepare_polygon_mass_table(
          POLY, EX, EY, min_sigma=5.0, max_sigma=None, panel_h_m=1.0))

probe("prepare_polygon_mass_table(min_sigma=None, max_sigma=None)",
      lambda: prepare_polygon_mass_table(
          POLY, EX, EY, min_sigma=None, max_sigma=None, panel_h_m=1.0))

probe("prepare_polygon_mass_table(min_sigma=0.0, max_sigma=None) [precedence]",
      lambda: prepare_polygon_mass_table(
          POLY, EX, EY, min_sigma=0.0, max_sigma=None, panel_h_m=1.0))

probe("prepare_polygon_mass_table(min_sigma=5.0, max_sigma=None, bad panel) [precedence]",
      lambda: prepare_polygon_mass_table(
          POLY, EX, EY, min_sigma=5.0, max_sigma=None, panel_h_m=1e6))

probe("build_quad_table(sigma_max=None)",
      lambda: build_quad_table(
          POLY, EX, EY, 5.0, None, ws=None, h_panel=1.0,
          gl_order=int(DEFAULT_GL_ORDER)))

_good = build_quad_table(POLY, EX, EY, 5.0, 40.0, ws=None, h_panel=1.0,
                         gl_order=int(DEFAULT_GL_ORDER))

probe("validate_polygon_mass_table(sigma_max=None)",
      lambda: validate_polygon_mass_table(
          _good, domain_geom=POLY, event_x_real=EX, event_y_real=EY,
          spatial_window=None, sigma_min=5.0, sigma_max=None))
