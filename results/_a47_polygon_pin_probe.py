"""A-47 feasibility probe for OP-24's fifth pin. Not the pin; the probe.

Answers, before pin_check_v2.py is touched:
  1. Does a polygon-mode Hawkes model build on a NON-UNIT-ASPECT (4:1),
     NON-RECTANGULAR domain, with an explicitly prepared mass table?
  2. Is the RECTANGLE mode buildable on the SAME domain and events, so the
     mode switch itself can be pinned rather than one side of it?
  3. Is loglik differentiable through the Hermite table (grad w.r.t. sigmax_2)?
  4. Is the mass table BIT-REPRODUCIBLE across two builds in one process, and
     across processes? Its hash is part of the pin's identity (D-27/A-11).
"""
import hashlib
import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro import handlers  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

import bstpp  # noqa: E402
from bstpp.main import Hawkes_Model  # noqa: E402
from bstpp.polygon_mass import prepare_polygon_mass_table  # noqa: E402

print("PROBE_PROVENANCE")
print(f"  bstpp.__file__ : {bstpp.__file__}")

# --- the domain: 4:1 bounding box, NON-RECTANGULAR (notch in the top edge).
# Non-unit aspect ratio for the same reason hawkes_nonsquare_4to1 exists: on a
# unit box the real-unit and internal-unit spatial kernels coincide
# algebraically. Non-rectangular so that polygon mass is NOT the rectangle's
# analytic box mass -- otherwise the two modes would agree by construction and
# the mode switch would be pinned by a pair of near-identical numbers.
POLY = Polygon([
    (0.0, 0.0), (4.0, 0.0), (4.0, 1.0),
    (2.4, 1.0), (2.4, 0.45), (1.6, 0.45), (1.6, 1.0),
    (0.0, 1.0),
])
A_POLY = gpd.GeoDataFrame(geometry=[POLY])
print(f"  domain bounds  : {POLY.bounds}   area={POLY.area}")
print(f"  aspect ratio   : "
      f"{(POLY.bounds[2] - POLY.bounds[0]) / (POLY.bounds[3] - POLY.bounds[1])}")

T_DAYS = 2.5 * 365.0
MIN_SIGMA, MAX_SIGMA = 0.05, 0.5
PANEL_H = 0.4          # ratio 8 == MAX_PANEL_TO_MIN_SIGMA_RATIO ceiling
N = 60


def events():
    """Deterministic draw inside the polygon; rejection on a fixed stream."""
    rng = np.random.RandomState(0)
    xs, ys = [], []
    while len(xs) < N:
        x = rng.uniform(0.05, 3.95)
        y = rng.uniform(0.05, 0.95)
        if POLY.contains(__import__("shapely").geometry.Point(x, y)):
            xs.append(x)
            ys.append(y)
    t = np.sort(rng.uniform(0, T_DAYS, N))
    return pd.DataFrame({"X": np.array(xs), "Y": np.array(ys), "T": t})


DATA = events()
print(f"  events         : {len(DATA)}  X[{DATA.X.min():.3f},{DATA.X.max():.3f}]"
      f" Y[{DATA.Y.min():.3f},{DATA.Y.max():.3f}]")

PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))


def table_hash(tbl):
    h = hashlib.sha256()
    for arr in (np.asarray(tbl.log_knots), np.asarray(tbl.values),
                np.asarray(tbl.slopes)):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def build_table():
    return prepare_polygon_mass_table(
        POLY,
        DATA["X"].to_numpy(dtype=float),
        DATA["Y"].to_numpy(dtype=float),
        min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA,
        panel_h_m=PANEL_H, crs=None,
    )


print()
print("1. mass table build")
t1 = build_table()
h1 = table_hash(t1)
print(f"   knots={t1.n_knots} events={t1.n_events} h_panel={float(t1.h_panel)} "
      f"gl_order={int(t1.gl_order)}")
print(f"   TABLE_SHA256 {h1}")

print()
print("2. same-process rebuild -> bit-identical?")
t2 = build_table()
h2 = table_hash(t2)
print(f"   TABLE_SHA256 {h2}")
print(f"   SAME_PROCESS_BIT_IDENTICAL {h1 == h2}")

print()
print("3. polygon-mode model + traced loglik and gradients")
mp = Hawkes_Model(DATA, A_POLY, T_DAYS, cox_background=False,
                  excitation_support="polygon",
                  min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA,
                  mass_table=t1, **PRIORS)
print(f"   mode={mp.excitation_support.mode}")

p = {k: jnp.float32(v) for k, v in
     dict(a_0=0.5, alpha=0.3, beta=2.0, sigmax_2=0.1).items()}


def loglik_fn(model):
    def f(params):
        s = handlers.substitute(handlers.seed(model.model,
                                              jax.random.PRNGKey(0)), params)
        return handlers.trace(s).get_trace(model.args)["loglik"]["value"]
    return f


val, grads = jax.value_and_grad(loglik_fn(mp))(p)
print(f"   loglik={float(val)!r}")
for k in sorted(grads):
    g = np.asarray(grads[k])
    print(f"   grad_{k} finite={bool(np.all(np.isfinite(g)))} {g.tolist()!r}")

print()
print("4. RECTANGLE mode on the SAME domain and events (the mode switch)")
try:
    mr = Hawkes_Model(DATA, A_POLY, T_DAYS, cox_background=False,
                      excitation_support="rectangle",
                      min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA, **PRIORS)
    print(f"   mode={mr.excitation_support.mode}  BUILDABLE True")
    vr, gr = jax.value_and_grad(loglik_fn(mr))(p)
    print(f"   loglik={float(vr)!r}")
    print(f"   MODES_DIFFER {float(vr) != float(val)}  "
          f"delta={float(vr) - float(val)!r}")
except Exception as exc:  # noqa: BLE001 - probe reports, never hides
    print(f"   BUILDABLE False -- {type(exc).__name__}: {exc}")

print()
print("PROBE_EXIT:0")
