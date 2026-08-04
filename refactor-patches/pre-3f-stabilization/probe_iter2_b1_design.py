"""Iteration-2 adversarial probe: B1 install design (table-authoritative + budget).

Read-only over production. Writes nothing outside this process.
Env: jax 0.4.23 / numpyro 0.15.0 / numpy<2 / scipy<1.13 / geopandas>=1.0 / x64=False.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# Prefer the repo checkout over any stale site/build install.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import geopandas as gpd
import jax
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from shapely.geometry import box

from bstpp.main import Hawkes_Model
from bstpp.polygon_mass import (
    DEFAULT_GL_ORDER,
    DEFAULT_PANEL_H_M,
    MAX_PANEL_TO_MIN_SIGMA_RATIO,
    PRODUCTION_TAU_ABS,
    build_quad_table,
    prepare_polygon_mass_table,
    validate_polygon_mass_table,
)
import inspect
from bstpp.main import Hawkes_Model as HM

T_DAYS = 30.0
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(0.5, 0.5),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(0.25),
)
MIN_S, MAX_S = 0.05, 0.5


def _env():
    import numpyro, scipy, geopandas as gpd0
    print("ENV jax", jax.__version__, "x64", jax.config.jax_enable_x64,
          "numpyro", numpyro.__version__,
          "numpy", np.__version__, "scipy", scipy.__version__,
          "geopandas", gpd0.__version__)


def main():
    _env()
    print("PRODUCTION_TAU_ABS", PRODUCTION_TAU_ABS)
    print("MAX_PANEL_TO_MIN_SIGMA_RATIO", MAX_PANEL_TO_MIN_SIGMA_RATIO)

    sig = inspect.signature(HM.__init__)
    print("Hawkes_Model has panel_h_m kwarg:", "panel_h_m" in sig.parameters)
    print("Hawkes_Model has gl_order kwarg:", "gl_order" in sig.parameters)

    gdf = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)])
    rng = np.random.RandomState(0)
    data = pd.DataFrame({
        "X": rng.uniform(0.1, 0.9, 6),
        "Y": rng.uniform(0.1, 0.9, 6),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 6)),
    })
    geom = gdf.geometry.union_all()
    xy = (data["X"].to_numpy(dtype=np.float64),
          data["Y"].to_numpy(dtype=np.float64))

    # --- Probe A: coarse table (budget violate) via build_quad_table ---
    coarse = build_quad_table(
        geom, xy[0], xy[1], MIN_S, MAX_S,
        h_panel=DEFAULT_PANEL_H_M, gl_order=DEFAULT_GL_ORDER,
    )
    ratio = float(coarse.h_panel) / MIN_S
    print("PROBE_A coarse h_panel", coarse.h_panel, "ratio", ratio,
          "violates", ratio > MAX_PANEL_TO_MIN_SIGMA_RATIO)
    try:
        Hawkes_Model(
            data, gdf, T_DAYS, cox_background=False,
            excitation_support="polygon",
            min_sigma=MIN_S, max_sigma=MAX_S,
            mass_table=coarse, **PRIORS)
        print("PROBE_A RESULT: CONSTRUCTED (UNEXPECTED)")
    except Exception as e:
        print("PROBE_A RESULT:", type(e).__name__, str(e)[:400])
        print("PROBE_A mentions PRODUCTION_TAU_ABS:",
              "PRODUCTION_TAU_ABS" in str(e))
        print("PROBE_A mentions ratio/ceiling:",
              "MAX_PANEL_TO_MIN_SIGMA_RATIO" in str(e) or "ratio" in str(e))

    # --- Probe B: guided fine table must install ---
    fine = prepare_polygon_mass_table(
        geom, xy[0], xy[1],
        min_sigma=MIN_S, max_sigma=MAX_S,
        panel_h_m=MAX_PANEL_TO_MIN_SIGMA_RATIO * MIN_S,
        crs=None,
    )
    print("PROBE_B fine h_panel", fine.h_panel, "gl", fine.gl_order)
    try:
        m = Hawkes_Model(
            data, gdf, T_DAYS, cox_background=False,
            excitation_support="polygon",
            min_sigma=MIN_S, max_sigma=MAX_S,
            mass_table=fine, **PRIORS)
        print("PROBE_B RESULT: OK mode", m.excitation_support.mode,
              "table_h", m.excitation_provenance.get("table_h_panel"),
              "ratio", m.excitation_provenance.get("panel_min_sigma_ratio"),
              "tau", m.excitation_provenance.get("PRODUCTION_TAU_ABS"))
    except Exception as e:
        print("PROBE_B RESULT: FAIL", type(e).__name__, e)

    # --- Probe C: validate_polygon_mass_table signature has no h_panel/gl_order ---
    sig_v = inspect.signature(validate_polygon_mass_table)
    print("PROBE_C validate params:", list(sig_v.parameters))
    print("PROBE_C no caller h_panel:", "h_panel" not in sig_v.parameters)

    # --- Probe D: panel_h_m / gl_order removed from build_excitation_support
    #     (Commit C); passing them must TypeError, not silently ignore.
    from bstpp.excitation_support import build_excitation_support
    from bstpp.preparation import prepare_domain
    dom = prepare_domain(gdf)
    sig_b = inspect.signature(build_excitation_support)
    print("PROBE_D no panel_h_m param:", "panel_h_m" not in sig_b.parameters)
    print("PROBE_D no gl_order param:", "gl_order" not in sig_b.parameters)
    try:
        build_excitation_support(
            mode="polygon",
            bounds=dom.bounds,
            domain_gdf=gdf,
            is_polygon_domain=True,
            crs=None,
            spatial_window=None,
            min_sigma=MIN_S,
            max_sigma=MAX_S,
            event_x_real=xy[0],
            event_y_real=xy[1],
            mass_table=fine,
            union_geometry=dom.union_geometry,
            panel_h_m=DEFAULT_PANEL_H_M,
            gl_order=8,
        )
        print("PROBE_D RESULT: UNEXPECTED accept of removed kwargs")
    except TypeError as e:
        print("PROBE_D RESULT: TypeError on removed kwargs OK", str(e)[:120])
    except Exception as e:
        print("PROBE_D RESULT: FAIL", type(e).__name__, e)

    # --- Probe E: coarse table must still reject (no silent-wrong) ---
    try:
        build_excitation_support(
            mode="polygon",
            bounds=dom.bounds,
            domain_gdf=gdf,
            is_polygon_domain=True,
            crs=None,
            spatial_window=None,
            min_sigma=MIN_S,
            max_sigma=MAX_S,
            event_x_real=xy[0],
            event_y_real=xy[1],
            mass_table=coarse,
            union_geometry=dom.union_geometry,
        )
        print("PROBE_E RESULT: CONSTRUCTED (UNEXPECTED — silent-wrong)")
    except Exception as e:
        print("PROBE_E RESULT: rejected", type(e).__name__)
        print("PROBE_E msg snippet:", str(e)[:280])

    print("EXIT:0")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("EXIT:1")
        sys.exit(1)
