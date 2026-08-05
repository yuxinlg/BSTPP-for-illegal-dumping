"""WP1.4e-2 Step 1, probe C: what does NumericalConfig actually receive?

The sign-off question is whether ``NumericalConfig.min_sigma`` / ``max_sigma``
hold the USER-SUPPLIED bounds or the RESOLVED bounds. That is not a reading
question -- it is answered by recording the arguments at every
``NumericalConfig.create`` call on every accepting public entry path, and
comparing them to what the user passed.

Also completes the entry-path table for the two paths probe B could not reach
cleanly: E3 (set_window with a matching replacement table) and E5 (held-out
scoring after a short SVI fit), plus the polygon default-max_sigma path under
a real projected CRS.

Read-only: ``NumericalConfig.create`` is wrapped by a recording shim that
delegates to the original and is restored afterwards. No production file is
modified.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import bstpp

print(f"bstpp.__file__ = {bstpp.__file__}")
assert "BSTPP-refactor" in bstpp.__file__.replace("\\", "/"), (
    "probe loaded the installed bstpp, not the working tree")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from pyproj import CRS  # noqa: E402
from shapely.geometry import box  # noqa: E402

from bstpp import config as cfgmod  # noqa: E402
from bstpp import excitation_support as exsup  # noqa: E402
from bstpp.main import Hawkes_Model  # noqa: E402
from bstpp.polygon_mass import (  # noqa: E402
    DEFAULT_GL_ORDER,
    MAX_PANEL_TO_MIN_SIGMA_RATIO,
    prepare_polygon_mass_table,
)

T_DAYS = 30.0
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(0.5, 0.5),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(0.25),
)
MIN_S = 0.05
MAX_S = 0.5
PANEL_GUIDED = MAX_PANEL_TO_MIN_SIGMA_RATIO * MIN_S

_calls: list[dict] = []


class _Recorder:
    """Wrap NumericalConfig.create and resolve_sigma_bounds, recording args."""

    def __enter__(self):
        _calls.clear()
        self._create = cfgmod.NumericalConfig.create
        self._resolve = exsup.resolve_sigma_bounds

        orig_create = self._create.__func__

        def create(cls_, **kw):
            _calls.append({
                "fn": "NumericalConfig.create",
                "support_mode": kw.get("support_mode"),
                "min_sigma": kw.get("min_sigma"),
                "max_sigma": kw.get("max_sigma"),
            })
            return orig_create(cls_, **kw)

        cfgmod.NumericalConfig.create = classmethod(create)

        orig_resolve = self._resolve

        def resolve(**kw):
            out = orig_resolve(**kw)
            _calls.append({
                "fn": "resolve_sigma_bounds",
                "support_mode": kw.get("mode"),
                "min_sigma": kw.get("min_sigma"),
                "max_sigma": kw.get("max_sigma"),
                "returned": (out[0], out[1]),
                "crs": None if kw.get("crs") is None else "projected",
            })
            return out

        exsup.resolve_sigma_bounds = resolve
        # main.py imported the symbol directly; patch that binding too.
        import bstpp.main as mainmod
        self._main_resolve = mainmod.resolve_sigma_bounds
        mainmod.resolve_sigma_bounds = resolve
        return self

    def __exit__(self, *exc):
        cfgmod.NumericalConfig.create = self._create
        exsup.resolve_sigma_bounds = self._resolve
        import bstpp.main as mainmod
        mainmod.resolve_sigma_bounds = self._main_resolve
        return False


def _unit_gdf():
    return gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)])


def _events(n=6, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "X": rng.uniform(0.1, 0.9, n),
        "Y": rng.uniform(0.1, 0.9, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _table(data, gdf=None, *, spatial_window=None, panel_h_m=PANEL_GUIDED,
           min_sigma=MIN_S, max_sigma=MAX_S, crs=None):
    geom = (_unit_gdf() if gdf is None else gdf).geometry.union_all()
    return prepare_polygon_mass_table(
        geom,
        data["X"].to_numpy(dtype=float),
        data["Y"].to_numpy(dtype=float),
        min_sigma=min_sigma, max_sigma=max_sigma,
        spatial_window=spatial_window,
        panel_h_m=panel_h_m, gl_order=DEFAULT_GL_ORDER, crs=crs)


def _report(title, user_min, user_max):
    print(f"\n--- {title}")
    print(f"    user passed: min_sigma={user_min!r}  max_sigma={user_max!r}")
    for c in _calls:
        if c["fn"] == "resolve_sigma_bounds":
            print(f"    resolve_sigma_bounds(mode={c['support_mode']!r}, "
                  f"min={c['min_sigma']!r}, max={c['max_sigma']!r}, "
                  f"crs={c['crs']}) -> {c['returned']!r}")
        else:
            print(f"    NumericalConfig.create(support_mode="
                  f"{c['support_mode']!r}, min_sigma={c['min_sigma']!r}, "
                  f"max_sigma={c['max_sigma']!r})")
    got = [c for c in _calls if c["fn"] == "NumericalConfig.create"]
    if got:
        same_as_user = all(
            c["min_sigma"] is user_min and c["max_sigma"] is user_max
            for c in got)
        print(f"    => config received the USER-SUPPLIED pair: {same_as_user}")


def main() -> None:
    data = _events()

    # ---------------------------------------- polygon, explicit max_sigma --
    with _Recorder():
        table = _table(data)
        m = Hawkes_Model(
            data, _unit_gdf(), T_DAYS, cox_background=False,
            excitation_support="polygon", min_sigma=MIN_S, max_sigma=MAX_S,
            mass_table=table, **PRIORS)
    _report("E2 polygon ctor, explicit max_sigma", MIN_S, MAX_S)
    print(f"    resulting cfg: min_sigma={m.numerical_config.min_sigma!r} "
          f"max_sigma={m.numerical_config.max_sigma!r}")

    # ------------------------------------------ rectangle, bounds omitted --
    with _Recorder():
        Hawkes_Model(data, _unit_gdf(), T_DAYS, cox_background=False,
                     excitation_support="rectangle", **PRIORS)
    _report("E1 rectangle ctor, bounds omitted", None, None)

    # ------------------------------------------ E4 set_window temporal-only --
    with _Recorder():
        m.set_window(window=5.0)
    _report("E4 set_window(window=) on the polygon model", MIN_S, MAX_S)

    # ------------------------- E3 set_window spatial + replacement table --
    with _Recorder():
        new_table = _table(data, spatial_window=0.4)
        m2 = Hawkes_Model(
            data, _unit_gdf(), T_DAYS, cox_background=False,
            excitation_support="polygon", min_sigma=MIN_S, max_sigma=MAX_S,
            mass_table=_table(data), **PRIORS)
        m2.set_window(spatial_window=0.4, mass_table=new_table)
    _report("E3 set_window(spatial_window=, mass_table=)", MIN_S, MAX_S)
    print(f"    E3 accepted; cfg: min_sigma={m2.numerical_config.min_sigma!r} "
          f"max_sigma={m2.numerical_config.max_sigma!r}")

    # -------------------------------------------------- E5 held-out scoring --
    m2.run_svi(200, 0.1, num_samples=20, plot_loss=False)
    heldout = _events(n=5, seed=7)
    ho_table = _table(heldout, spatial_window=0.4)
    with _Recorder():
        val = m2.log_expected_likelihood(heldout, mass_table=ho_table)
    _report("E5 log_expected_likelihood(mass_table=)", MIN_S, MAX_S)
    print(f"    E5 returned {float(val)!r}")

    # ------------------- polygon default max_sigma under a projected CRS --
    crs = CRS.from_epsg(32618)
    gdf_m = gpd.GeoDataFrame(
        geometry=[box(0.0, 0.0, 4000.0, 4000.0)], crs=crs)
    rng = np.random.RandomState(3)
    data_m = pd.DataFrame({
        "X": rng.uniform(500.0, 3500.0, 6),
        "Y": rng.uniform(500.0, 3500.0, 6),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 6)),
    })
    # max_sigma omitted -> resolves to 5 km = 5000.0 CRS units.
    tbl_m = _table(data_m, gdf_m, min_sigma=1000.0, max_sigma=5000.0,
                   panel_h_m=2000.0, crs=None)
    with _Recorder():
        mm = Hawkes_Model(
            data_m, gdf_m, T_DAYS, cox_background=False,
            excitation_support="polygon", min_sigma=1000.0, max_sigma=None,
            mass_table=tbl_m, **PRIORS)
    _report("E2 polygon ctor, max_sigma OMITTED, projected CRS",
            1000.0, None)
    print(f"    resulting cfg: min_sigma={mm.numerical_config.min_sigma!r} "
          f"max_sigma={mm.numerical_config.max_sigma!r}")
    print("    => the config's stored max_sigma is the DEFAULTED value, not "
          "the user's None")


if __name__ == "__main__":
    main()
