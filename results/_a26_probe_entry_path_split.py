"""A-26 / WP1.4e Step 1 probe: measure the entry-path error split.

Read-only. Triggers each invariant from the constructor and from
``set_window(mass_table=...)`` and prints the concrete exception type and
message from each, so the split is measured rather than inferred from call
order. Nothing here edits production code.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# This file lives in results/, so sys.path[0] is results/ and `import bstpp`
# would otherwise resolve to the STALE COPY INSTALLED IN site-packages (which
# predates polygon_mass.py). Put the repository root first, as the tests do.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402

from bstpp.main import Hawkes_Model  # noqa: E402
from bstpp.polygon_mass import (  # noqa: E402
    DEFAULT_GL_ORDER,
    build_quad_table,
    prepare_polygon_mass_table,
)
from bstpp.preparation import prepare_domain  # noqa: E402

import geopandas as gpd  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

T_DAYS = 2.5 * 365.0
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))

MIN_S, MAX_S = 5.0, 40.0
COARSE_PANEL = 200.0          # 200 / 5 = 40 > 8
SPATIAL_WINDOW = 50.0


def _domain():
    """Metric square domain as a GeoDataFrame (polygon support is required)."""
    return gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])],
        crs="EPSG:32618")


def _data(seed=3, n=40):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "X": rng.uniform(50, 950, n),
        "Y": rng.uniform(50, 950, n),
        "T": np.sort(rng.uniform(0, T_DAYS, n)),
    })


def _table(data, A, *, h_panel, min_sigma=MIN_S, max_sigma=MAX_S,
           spatial_window=None):
    """Build directly via build_quad_table so prepare's ratio guard is bypassed."""
    dom = prepare_domain(A)
    return build_quad_table(
        dom.union_geometry,
        data["X"].to_numpy(dtype=np.float64),
        data["Y"].to_numpy(dtype=np.float64),
        float(min_sigma), float(max_sigma),
        ws=None if spatial_window is None else float(spatial_window),
        h_panel=float(h_panel),
        gl_order=int(DEFAULT_GL_ORDER),
    )


def _safe(text: str) -> str:
    """stdout here is cp1252; NumericalConfig's message contains a literal U+03C3."""
    enc = sys.stdout.encoding or "ascii"
    return text.encode(enc, errors="backslashreplace").decode(enc)


def show(label, fn):
    try:
        fn()
    except BaseException as exc:            # noqa: BLE001 - probe
        print(f"{label}")
        print(f"    type:  {type(exc).__module__}.{type(exc).__name__}")
        print(f"    mro:   {[c.__name__ for c in type(exc).__mro__[:4]]}")
        print(f"    msg:   {_safe(str(exc))}")
        return type(exc).__name__, str(exc)
    print(f"{label}\n    NO ERROR RAISED")
    return None, None


def main() -> None:
    # A-26 extension to A-25's provenance clause: an ad-hoc probe records
    # which bstpp it actually imported. A stale copy in site-packages shadows
    # the repository for any script whose directory is not the repo root, and
    # the capture then describes a different object while looking correct.
    import bstpp
    print(f"bstpp.__file__: {bstpp.__file__}")
    print()

    A = _domain()
    data = _data()
    print("=" * 78)
    print("INVARIANT 1 -- panel_h_m / min_sigma <= max_panel_to_min_sigma_ratio")
    print("=" * 78)

    coarse = _table(data, A, h_panel=COARSE_PANEL)
    t1, m1 = show("[constructor]  Hawkes_Model(..., mass_table=coarse)",
                  lambda: Hawkes_Model(
                      data, A, T_DAYS, cox_background=False,
                      excitation_support="polygon",
                      min_sigma=MIN_S, max_sigma=MAX_S,
                      mass_table=coarse, **PRIORS))

    good = _table(data, A, h_panel=20.0, spatial_window=SPATIAL_WINDOW)
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=MIN_S, max_sigma=MAX_S, spatial_window=SPATIAL_WINDOW,
        mass_table=good, **PRIORS)
    coarse_sw = _table(data, A, h_panel=COARSE_PANEL,
                       spatial_window=SPATIAL_WINDOW)
    t2, m2 = show("[set_window]   m.set_window(mass_table=coarse)",
                  lambda: m.set_window(spatial_window=SPATIAL_WINDOW,
                                       mass_table=coarse_sw))

    # Third entry path: held-out scoring goes straight to
    # build_excitation_support with no NumericalConfig rebuild.
    heldout = _data(seed=11, n=35)
    coarse_ho = _table(heldout, A, h_panel=COARSE_PANEL,
                       spatial_window=SPATIAL_WINDOW)
    t3, m3 = show("[log_expected_likelihood]  m.log_expected_likelihood("
                  "test, mass_table=coarse)",
                  lambda: m.log_expected_likelihood(heldout,
                                                    mass_table=coarse_ho))

    # Fourth entry path: the sanctioned builder's own pre-build guard.
    dom = prepare_domain(A)
    t4, m4 = show("[prepare_polygon_mass_table]  panel_h_m=200, min_sigma=5",
                  lambda: prepare_polygon_mass_table(
                      dom.union_geometry,
                      data["X"].to_numpy(dtype=float),
                      data["Y"].to_numpy(dtype=float),
                      min_sigma=MIN_S, max_sigma=MAX_S,
                      panel_h_m=COARSE_PANEL, crs=dom.crs))

    print()
    print("    entry path                    type                     message id")
    for name, t, msg in (("constructor", t1, m1), ("set_window", t2, m2),
                         ("log_expected_likelihood", t3, m3),
                         ("prepare_polygon_mass_table", t4, m4)):
        head = (msg or "")[:34].replace("\n", " ")
        print(f"    {name:<29} {str(t):<24} {head!r}")
    types = {t1, t2, t3, t4}
    msgs = {m1, m2, m3, m4}
    print()
    print(f"    DISTINCT TYPES:    {len(types)}  {sorted(str(x) for x in types)}")
    print(f"    DISTINCT MESSAGES: {len(msgs)}")
    print(f"    SAME TYPE ctor vs set_window?    {t1 == t2}")
    print(f"    SAME MESSAGE ctor vs set_window? {m1 == m2}")
    print()

    print("=" * 78)
    print("INVARIANT 2 -- sigma-bound coherence (min_sigma < max_sigma)")
    print("=" * 78)
    show("[constructor]  min_sigma=40, max_sigma=5",
         lambda: Hawkes_Model(
             data, A, T_DAYS, cox_background=False,
             excitation_support="polygon",
             min_sigma=40.0, max_sigma=5.0,
             mass_table=good, **PRIORS))
    from bstpp.config import NumericalConfig
    show("[NumericalConfig.create] same violation, config's own branch",
         lambda: NumericalConfig.create(
             support_mode="polygon", min_sigma=40.0, max_sigma=5.0,
             panel_h_m=20.0))

    print()
    print("=" * 78)
    print("INVARIANT 3 -- support-mode validity")
    print("=" * 78)
    from bstpp.excitation_support import build_excitation_support
    show("[build_excitation_support] mode='banana'",
         lambda: build_excitation_support(
             mode="banana", is_polygon_domain=True,
             bounds=np.array([[0., 1000.], [0., 1000.]]),
             domain_gdf=A, crs=None,
             spatial_window=None, min_sigma=MIN_S, max_sigma=MAX_S,
             event_x_real=np.array([1.0]), event_y_real=np.array([1.0])))
    show("[NumericalConfig.create] support_mode='banana'",
         lambda: NumericalConfig.create(support_mode="banana"))

    print()
    print("=" * 78)
    print("INVARIANT 4 -- polygon requires min_sigma")
    print("=" * 78)
    show("[constructor]  polygon, min_sigma=None",
         lambda: Hawkes_Model(
             data, A, T_DAYS, cox_background=False,
             excitation_support="polygon",
             min_sigma=None, max_sigma=MAX_S,
             mass_table=good, **PRIORS))
    show("[NumericalConfig.create] polygon, min_sigma=None",
         lambda: NumericalConfig.create(support_mode="polygon",
                                        min_sigma=None, max_sigma=MAX_S))

    print()
    print("=" * 78)
    print("INVARIANT 5 -- rectangle: both bounds or neither")
    print("=" * 78)
    show("[constructor]  rectangle, min_sigma only",
         lambda: Hawkes_Model(
             data, np.array([[0., 1000.], [0., 1000.]]), T_DAYS,
             cox_background=False, min_sigma=MIN_S, max_sigma=None, **PRIORS))
    show("[NumericalConfig.create] rectangle, min_sigma only",
         lambda: NumericalConfig.create(support_mode="rectangle",
                                        min_sigma=MIN_S, max_sigma=None))


if __name__ == "__main__":
    main()
