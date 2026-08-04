"""Polygon I11 conservation: E[n - Itot_txy(sim)] ≈ 0 (pre-3f Commit E / G5).

Each replicate rebuilds the polygon mass table and excitation pairs from the
simulated events (same realization the likelihood scores). Criterion is the
~3·se gate used in ``probe_iter3_poly_i11`` (R=40), which is higher power than
the iteration-2 R=15 / 5·se empiric check.

Minimum detectable bias under this gate is ≈ 3·se (reported in the assertion
message). Env: jax==0.4.23, numpyro==0.15.0, numpy<2, scipy<1.13,
geopandas>=1.0, jax_enable_x64=False.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro import handlers
from shapely.geometry import box

from bstpp.main import Hawkes_Model
from bstpp.polygon_mass import prepare_polygon_mass_table
from bstpp.utils import aligned_difference_pairs
from tests._polygon_prepare_helpers import prepare_table_for_model

T_DAYS = 40.0
A = np.array([[0.0, 200.0], [0.0, 200.0]])
R = 40


def _conservation_diffs(
    *,
    min_sigma: float,
    max_sigma: float,
    sigmax_2: float,
    panel_h_m: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "X": rng.uniform(20.0, 180.0, 12),
        "Y": rng.uniform(20.0, 180.0, 12),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 12)),
    })
    geom = box(0.0, 0.0, 200.0, 200.0)
    if panel_h_m is None:
        table = prepare_table_for_model(
            data, A, min_sigma=min_sigma, max_sigma=max_sigma)
    else:
        table = prepare_polygon_mass_table(
            geom,
            data["X"].to_numpy(dtype=float),
            data["Y"].to_numpy(dtype=float),
            min_sigma=min_sigma, max_sigma=max_sigma,
            panel_h_m=panel_h_m, crs=None,
        )
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=min_sigma, max_sigma=max_sigma, mass_table=table,
        a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
        beta=dist.HalfNormal(1.0),
        sigmax_2=dist.HalfNormal(float(sigmax_2)),
    )
    truth = dict(
        a_0=0.2, alpha=0.25, beta=2.0,
        sigmax_2=np.float32(sigmax_2),
    )
    params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in truth.items()}
    gen = np.random.default_rng(seed + 21)
    diffs = np.empty(R)
    counts = np.empty(R)
    for r in range(R):
        sim = m.simulate(parameters=dict(truth), rng=gen)
        counts[r] = len(sim)
        ta, _ = m._scale_xyt(
            pd.DataFrame(sim[["X", "Y", "T"]]), m.args.copy(), m.comp_grid)
        mt_kw = dict(
            min_sigma=min_sigma, max_sigma=max_sigma,
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
        diffs[r] = counts[r] - lam
    return diffs, counts


@pytest.mark.parametrize(
    "label,min_sigma,max_sigma,sigmax_2,panel_h_m,seed",
    [
        ("shipped_defaults", 5.0, 40.0, 100.0, None, 0),
        ("small_sigma", 0.5, 5.0, 1.0, 4.0, 1),
    ],
)
def test_polygon_i11_conservation(
        label, min_sigma, max_sigma, sigmax_2, panel_h_m, seed):
    x64_before = bool(jax.config.read("jax_enable_x64"))
    diffs, counts = _conservation_diffs(
        min_sigma=min_sigma, max_sigma=max_sigma,
        sigmax_2=sigmax_2, panel_h_m=panel_h_m, seed=seed,
    )
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(R))
    mdb = 3.0 * se
    assert counts.mean() > 20, (
        f"{label}: degenerate simulation size (mean n={counts.mean():.1f})")
    assert abs(mean) < 3.0 * se + 1e-9, (
        f"{label}: E[n - Itot_txy] = {mean:.4g} ± {se:.4g} over R={R} "
        f"(MDB≈{mdb:.4g} at 3·se; mean n={counts.mean():.1f}) — "
        "polygon conservation violated")
    assert bool(jax.config.read("jax_enable_x64")) is x64_before
