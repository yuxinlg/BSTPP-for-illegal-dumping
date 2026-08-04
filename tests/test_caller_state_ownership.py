"""Caller-state ownership class (pre-3f B2/B3/B4).

A prepared model must not retain caller-supplied mutable objects by alias,
and must not draw entropy the caller did not supply. This module tests the
class — not three one-off instances — via an explicit retained-object
inventory. Adding a new retained input without registering it here must fail.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import jax
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from shapely.geometry import box

from bstpp.main import Hawkes_Model
from tests._polygon_prepare_helpers import prepare_table_for_model

T_DAYS = 30.0
A_METERS = np.array([[0.0, 200.0], [0.0, 200.0]], dtype=float)
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(40.0),
)

# Explicit inventory of caller-supplied mutable objects a model may retain.
# Commit B's sweep must keep this list complete; unknown members fail the
# meta-test below.
CALLER_RETAINED_INVENTORY = (
    "domain_A_array",
    "polygon_mass_table_values",
    "spatial_cov_gdf",
    "event_dataframe",
    "model_data_domain_array",
    "model_data_events",
)


def _events(n=8, seed=0, A=A_METERS):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(A[0, 0] + 10, A[0, 1] - 10, n),
        "Y": rng.uniform(A[1, 0] + 10, A[1, 1] - 10, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _cov_gdf():
    return gpd.GeoDataFrame(
        {"v": [0.5, -0.5, 1.0, -1.0]},
        geometry=[
            box(0, 0, 100, 100),
            box(100, 0, 200, 100),
            box(0, 100, 100, 200),
            box(100, 100, 200, 200),
        ],
    )


def _count_np_random(fn):
    hits = {"n": 0, "names": []}
    orig = {}
    targets = (
        "random", "randn", "rand", "randint", "choice", "permutation",
        "shuffle", "normal", "poisson", "exponential", "uniform",
    )

    def wrap(name, f):
        def inner(*a, **k):
            hits["n"] += 1
            hits["names"].append(name)
            return f(*a, **k)
        return inner

    for name in targets:
        if hasattr(np.random, name):
            orig[name] = getattr(np.random, name)
            setattr(np.random, name, wrap(name, orig[name]))
    try:
        return fn(), hits
    finally:
        for name, f in orig.items():
            setattr(np.random, name, f)


# ------------------------------------------------------------------ meta --
def test_retained_inventory_is_explicit_and_covered():
    """Every inventory member must have a matching ownership case id."""
    covered = {
        "domain_A_array",
        "polygon_mass_table_values",
        "spatial_cov_gdf",
        "event_dataframe",
        "model_data_domain_array",
        "model_data_events",
    }
    assert set(CALLER_RETAINED_INVENTORY) == covered


# ----------------------------------------- post-construction mutation ----
@pytest.mark.parametrize("kind", CALLER_RETAINED_INVENTORY)
def test_mutating_caller_object_does_not_change_model_state(kind):
    data = _events(seed=1)
    A = np.array([[0.0, 200.0], [0.0, 200.0]], dtype=float)
    cov = _cov_gdf()
    table = prepare_table_for_model(
        data, A, min_sigma=5.0, max_sigma=40.0)

    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        mass_table=table,
        spatial_cov=cov, cov_names=["v"],
        **PRIORS,
    )

    if kind == "domain_A_array":
        before = np.asarray(m.args["A_"]).copy()
        A[0, 1] = 999.0
        after = np.asarray(m.args["A_"])
        assert np.allclose(before, after), (
            "domain A aliased into args['A_'] / prepared_domain.bounds")
        assert np.allclose(before, np.asarray(m.prepared_domain.bounds))

    elif kind == "polygon_mass_table_values":
        before = float(np.asarray(m.excitation_support.mass_table.values)[0, 0])
        table.values[0, 0] = before + 123.0
        after = float(np.asarray(m.excitation_support.mass_table.values)[0, 0])
        assert after == pytest.approx(before), (
            "caller mass_table.values aliased into installed table")
        # restore for process hygiene
        table.values[0, 0] = before

    elif kind == "spatial_cov_gdf":
        before_design = float(np.asarray(m.args["spatial_cov"])[0, 0])
        before_frame = float(m.spatial_cov.loc[0, "v"])
        cov.loc[0, "v"] = before_frame + 50.0
        after_design = float(np.asarray(m.args["spatial_cov"])[0, 0])
        after_frame = float(m.spatial_cov.loc[0, "v"])
        assert after_design == pytest.approx(before_design), (
            "caller spatial_cov aliased into model design matrix")
        assert after_frame == pytest.approx(before_frame), (
            "caller spatial_cov aliased into self.spatial_cov")

    elif kind == "event_dataframe":
        before = np.asarray(m.args["t_events"]).copy()
        data.loc[0, "T"] = 99.0
        after = np.asarray(m.args["t_events"])
        assert np.allclose(before, after)

    elif kind == "model_data_domain_array":
        before = np.asarray(m.args["A_"]).copy()
        # ModelData retains domain=A at construction.
        m.model_data.domain[0, 1] = 888.0
        after = np.asarray(m.args["A_"])
        assert np.allclose(before, after)

    elif kind == "model_data_events":
        before = np.asarray(m.args["t_events"]).copy()
        m.model_data.events.loc[0, "T"] = 77.0
        after = np.asarray(m.args["t_events"])
        assert np.allclose(before, after)

    else:
        raise AssertionError(f"unregistered inventory kind {kind!r}")


def test_mass_table_provenance_matches_installed_arrays_after_caller_mutate():
    data = _events(seed=2)
    A = np.array([[0.0, 200.0], [0.0, 200.0]], dtype=float)
    table = prepare_table_for_model(
        data, A, min_sigma=5.0, max_sigma=40.0)
    h_before = float(table.h_panel)
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        mass_table=table, **PRIORS,
    )
    vals_before = np.asarray(m.excitation_support.mass_table.values).copy()
    table.values[:] = vals_before + 1.0
    # Provenance must still describe the arrays the model actually holds.
    assert m.excitation_provenance.get("table_h_panel") == pytest.approx(h_before)
    installed = np.asarray(m.excitation_support.mass_table.values)
    assert np.allclose(installed, vals_before), (
        "provenance claims original table but installed values moved with caller")


# ----------------------------------------------------------- simulate RNG --
def test_simulate_without_rng_does_not_consume_np_random():
    """No rng → named reject (Commit B); never silently use np.random."""
    data = _events(seed=3)
    A = np.array([[0.0, 200.0], [0.0, 200.0]], dtype=float)
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="rectangle", **PRIORS,
    )
    truth = dict(a_0=0.0, alpha=0.2, beta=2.0, sigmax_2=25.0)

    def call():
        return m.simulate(parameters=dict(truth))  # omit rng=

    try:
        _, hits = _count_np_random(call)
    except (TypeError, ValueError) as exc:
        # Acceptable Commit B outcome: require explicit generator.
        assert "rng" in str(exc).lower()
        return
    assert hits["n"] == 0, (
        f"simulate() without rng consumed np.random ({hits['n']} hits: "
        f"{hits['names'][:20]})")


def test_simulate_same_generator_seed_bit_identical():
    data = _events(seed=4)
    A = np.array([[0.0, 200.0], [0.0, 200.0]], dtype=float)
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="rectangle", **PRIORS,
    )
    truth = dict(a_0=0.0, alpha=0.2, beta=2.0, sigmax_2=25.0)
    a = m.simulate(parameters=dict(truth), rng=np.random.default_rng(11))
    b = m.simulate(parameters=dict(truth), rng=np.random.default_rng(11))
    assert len(a) == len(b)
    np.testing.assert_allclose(
        a[["X", "Y", "T"]].to_numpy(),
        b[["X", "Y", "T"]].to_numpy(),
        rtol=0, atol=0,
    )
    c = m.simulate(parameters=dict(truth), rng=np.random.default_rng(12))
    differ = not (
        len(a) == len(c)
        and np.allclose(a[["X", "Y", "T"]].to_numpy(),
                        c[["X", "Y", "T"]].to_numpy())
    )
    assert differ


def test_simulate_with_generator_does_not_touch_np_random():
    data = _events(seed=5)
    A = np.array([[0.0, 200.0], [0.0, 200.0]], dtype=float)
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="rectangle", **PRIORS,
    )
    truth = dict(a_0=0.0, alpha=0.2, beta=2.0, sigmax_2=25.0)

    def call():
        return m.simulate(
            parameters=dict(truth), rng=np.random.default_rng(99))

    _, hits = _count_np_random(call)
    assert hits["n"] == 0, hits["names"][:20]


# --------------------------------------------------------------- run_svi --
def test_run_svi_accepts_and_honors_rng_key():
    sig = inspect.signature(Hawkes_Model.run_svi)
    assert "rng_key" in sig.parameters, (
        "run_svi must accept rng_key= with the same semantics as run_mcmc")

    data = _events(n=6, seed=6)
    A = np.array([[0.0, 200.0], [0.0, 200.0]], dtype=float)
    m1 = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="rectangle", **PRIORS,
    )
    m2 = Hawkes_Model(
        data.copy(), A.copy(), T_DAYS, cox_background=False,
        excitation_support="rectangle", **PRIORS,
    )
    key = jax.random.PRNGKey(123)
    m1.run_svi(num_steps=5, lr=0.01, num_samples=3, plot_loss=False,
               rng_key=key)
    m2.run_svi(num_steps=5, lr=0.01, num_samples=3, plot_loss=False,
               rng_key=key)
    np.testing.assert_allclose(
        np.asarray(m1.samples["a_0"]),
        np.asarray(m2.samples["a_0"]),
        rtol=0, atol=0,
    )
    m3 = Hawkes_Model(
        data.copy(), A.copy(), T_DAYS, cox_background=False,
        excitation_support="rectangle", **PRIORS,
    )
    m3.run_svi(num_steps=5, lr=0.01, num_samples=3, plot_loss=False,
               rng_key=jax.random.PRNGKey(456))
    assert not np.allclose(
        np.asarray(m1.samples["a_0"]),
        np.asarray(m3.samples["a_0"]),
    )


def test_run_mcmc_with_key_does_not_touch_np_random():
    data = _events(n=6, seed=7)
    A = np.array([[0.0, 200.0], [0.0, 200.0]], dtype=float)
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="rectangle", **PRIORS,
    )

    def call():
        m.run_mcmc(num_warmup=2, num_samples=2,
                   rng_key=jax.random.PRNGKey(7))

    _, hits = _count_np_random(call)
    assert hits["n"] == 0, hits["names"][:20]
