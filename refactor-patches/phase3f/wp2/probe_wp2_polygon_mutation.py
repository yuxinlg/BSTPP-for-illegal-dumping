"""WP2 Step 1c, discriminating case: polygon set_window that INSTALLS a table.

This is the only public call that re-resolves sigma bounds and rebuilds
NumericalConfig. The question for WP2: if PriorConfig freezes the sigmax_2
prior, does anything on this path invalidate it?
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.geometry import box  # noqa: E402

import bstpp  # noqa: E402

assert os.path.abspath(bstpp.__file__).startswith(_REPO), bstpp.__file__

from bstpp.main import Hawkes_Model  # noqa: E402
from bstpp.polygon_mass import prepare_polygon_mass_table  # noqa: E402

T_DAYS = 30.0
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
rng = np.random.default_rng(3)
n = 6
DATA = pd.DataFrame({"X": rng.uniform(20, 180, n), "Y": rng.uniform(20, 180, n),
                     "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n))})
GDF = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 200.0, 200.0)])


def table(ws):
    return prepare_polygon_mass_table(
        box(0.0, 0.0, 200.0, 200.0),
        DATA["X"].to_numpy(float), DATA["Y"].to_numpy(float),
        min_sigma=5.0, max_sigma=40.0, spatial_window=ws, panel_h_m=1.0)


print("bstpp.__file__ =", bstpp.__file__)
m = Hawkes_Model(DATA, GDF, T_DAYS, cox_background=False,
                 excitation_support="polygon", min_sigma=5.0, max_sigma=40.0,
                 spatial_window=60.0, mass_table=table(60.0), **PRIORS)


def snap(mm):
    return {
        "numerical_config id": id(mm.args["numerical_config"]),
        "nc min/max sigma": (mm.args["numerical_config"].min_sigma,
                             mm.args["numerical_config"].max_sigma),
        "nc panel_h_m": mm.args["numerical_config"].panel_h_m,
        "sigmax_2 support": repr(getattr(
            mm.args["priors"]["sigmax_2"], "support", None)),
        "sigmax_2 obj id": id(mm.args["priors"]["sigmax_2"]),
        "spatial_window": mm.args.get("spatial_window"),
    }


b = snap(m)
m.set_window(spatial_window=30.0, mass_table=table(30.0))
a = snap(m)

print()
print("polygon set_window(spatial_window 60 -> 30, mass_table=<new>)")
print("=" * 78)
for k in b:
    same = b[k] == a[k]
    print(f"  {k:<22}{'same' if same else 'CHANGED':<10}"
          f"{str(b[k])[:22]} -> {str(a[k])[:22]}")
print()
print("READING: NumericalConfig is REBUILT (new id) on this path, as WP1.4b")
print("requires. The sigmax_2 prior object is NOT rebuilt and does not need to")
print("be: set_window re-resolves bounds from the ORIGINAL constructor args")
print("(_min_sigma_arg/_max_sigma_arg), which no public call can change, so the")
print("resolved pair is invariant and the truncation stays correct.")
print()
print("=> The staleness coupling is LATENT, not live: PriorConfig's truncated")
print("   sigmax_2 depends on NumericalConfig's resolved bounds, and today")
print("   nothing can move those after construction. WP2 must not ADD a public")
print("   way to move them without re-truncating, which is exactly the")
print("   half-applied state D-35's rebuild-on-mutation rule exists to prevent.")
