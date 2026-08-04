"""Commit E probe: polygon I11 at shipped defaults and small sigma.

Criterion: |mean(n - Itot_txy)| < 3 * se, with se = s/sqrt(R).
Reports minimum detectable bias ≈ 3*se under that gate.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from numpyro import handlers
from shapely.geometry import box

from bstpp.main import Hawkes_Model
from bstpp.polygon_mass import prepare_polygon_mass_table
from bstpp.utils import aligned_difference_pairs
from tests._polygon_prepare_helpers import prepare_table_for_model


def _run_regime(*, label, A, min_s, max_s, sigmax_2, panel_h_m, R, seed):
    T = 40.0
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "X": rng.uniform(float(A[0, 0]) + 20, float(A[0, 1]) - 20, 12),
        "Y": rng.uniform(float(A[1, 0]) + 20, float(A[1, 1]) - 20, 12),
        "T": np.sort(rng.uniform(0.5, T - 0.5, 12)),
    })
    if panel_h_m is None:
        table = prepare_table_for_model(
            data, A, min_sigma=min_s, max_sigma=max_s)
    else:
        geom = box(float(A[0, 0]), float(A[1, 0]),
                   float(A[0, 1]), float(A[1, 1]))
        table = prepare_polygon_mass_table(
            geom,
            data["X"].to_numpy(dtype=float),
            data["Y"].to_numpy(dtype=float),
            min_sigma=min_s, max_sigma=max_s,
            panel_h_m=panel_h_m, crs=None,
        )
    m = Hawkes_Model(
        data, A, T, cox_background=False,
        excitation_support="polygon",
        min_sigma=min_s, max_sigma=max_s, mass_table=table,
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(float(sigmax_2)),
    )
    truth = dict(
        a_0=0.2, alpha=0.25, beta=2.0,
        sigmax_2=np.float32(sigmax_2),
    )
    params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in truth.items()}
    geom = box(float(A[0, 0]), float(A[1, 0]),
               float(A[0, 1]), float(A[1, 1]))
    diffs = []
    counts = []
    gen = np.random.default_rng(seed + 21)
    for _ in range(R):
        sim = m.simulate(parameters=dict(truth), rng=gen)
        n = len(sim)
        counts.append(n)
        ta, _ = m._scale_xyt(
            pd.DataFrame(sim[["X", "Y", "T"]]), m.args.copy(), m.comp_grid)
        mt_kw = dict(
            min_sigma=min_s, max_sigma=max_s,
            spatial_window=m.args.get("spatial_window"),
            crs=None,
        )
        if panel_h_m is not None:
            mt_kw["panel_h_m"] = panel_h_m
        mt = prepare_polygon_mass_table(
            geom,
            sim["X"].to_numpy(dtype=float),
            sim["Y"].to_numpy(dtype=float),
            **mt_kw,
        )
        support = m._excitation_support_for_events(
            sim["X"].to_numpy(dtype=float),
            sim["Y"].to_numpy(dtype=float),
            mass_table=mt,
        )
        t_events = np.asarray(ta["t_events"])
        xy = np.asarray(ta["xy_events"])
        coords, t_vals, x_vals, y_vals = aligned_difference_pairs(
            t_events, xy[0], xy[1],
            window=float(m.args["window"]),
            spatial_window=m.args.get("spatial_window"),
            axis_scales=m.args["axis_scales"],
        )
        ta["excitation_support"] = support
        ta["coords"] = coords
        ta["t_vals"] = t_vals
        ta["x_vals"] = x_vals
        ta["y_vals"] = y_vals
        tr = handlers.trace(handlers.seed(
            handlers.substitute(m.model, data=params),
            jax.random.PRNGKey(0))).get_trace(ta)
        lam = float(np.asarray(tr["Itot_txy"]["value"]))
        diffs.append(n - lam)

    diffs = np.asarray(diffs, dtype=float)
    counts = np.asarray(counts, dtype=float)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(R))
    mdb = 3.0 * se
    passed = abs(mean) < 3.0 * se + 1e-9
    print(
        f"{label}: R={R} mean_diff={mean:.6g} se={se:.6g} "
        f"MDB_3se={mdb:.6g} mean_n={float(counts.mean()):.4g} "
        f"pass={passed}"
    )
    return passed, mean, se, mdb


def main() -> int:
    import numpyro, scipy, geopandas as gpd
    print(
        "ENV jax", jax.__version__, "x64", jax.config.jax_enable_x64,
        "numpyro", numpyro.__version__, "numpy", np.__version__,
        "scipy", scipy.__version__, "geopandas", gpd.__version__,
    )
    A = np.array([[0.0, 200.0], [0.0, 200.0]])
    R = 40  # target ~3·se power vs iter2 R=15 / 5·se
    ok1, *_ = _run_regime(
        label="DEFAULTS", A=A, min_s=5.0, max_s=40.0,
        sigmax_2=100.0, panel_h_m=None, R=R, seed=0,
    )
    # Small σ: min_sigma=0.5 on 200-unit domain; guided panel at ratio<=8.
    ok2, *_ = _run_regime(
        label="SMALL_SIGMA", A=A, min_s=0.5, max_s=5.0,
        sigmax_2=1.0, panel_h_m=4.0, R=R, seed=1,
    )
    print("OVERALL_PASS", ok1 and ok2)
    print("EXIT:0" if (ok1 and ok2) else "EXIT:1")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        print("EXIT:1")
        raise SystemExit(1)
