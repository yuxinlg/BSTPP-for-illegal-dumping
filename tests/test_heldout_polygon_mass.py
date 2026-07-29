"""Held-out polygon scoring requires an explicit held-out mass table (pre-3f).

Scientific reading (D-32): held-out data are a complete standalone realization;
event-indexed state (pairs, mass-table rows) is rebuilt from that realization.
Acquisition contract (D-26, superseding silent rebuild): polygon mode
hard-requires ``mass_table=`` prepared for the held-out events — never built
implicitly; a training-event table must not be accepted for a different
realization. Rectangle mode is unaffected.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import geopandas as gpd
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro.infer import log_likelihood
from shapely.geometry import box

from bstpp.data_contracts import DataContractError
from bstpp.main import Hawkes_Model
from tests._polygon_prepare_helpers import prepare_table_for_model

T_DAYS = 30.0
A = np.array([[0.0, 200.0], [0.0, 200.0]])
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(40.0),
)
PARAMS = dict(
    a_0=np.float32(0.0),
    alpha=np.float32(0.3),
    beta=np.float32(2.0),
    sigmax_2=np.float32(20.0 ** 2),
)


def _events(n, seed, *, x_lo=20.0, x_hi=180.0, y_lo=20.0, y_hi=180.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(x_lo, x_hi, n),
        "Y": rng.uniform(y_lo, y_hi, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _polygon_model(data):
    table = prepare_table_for_model(
        data, A, min_sigma=5.0, max_sigma=40.0)
    return Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=5.0, max_sigma=40.0,
        mass_table=table,
        **PRIORS,
    )


def _rectangle_model(data):
    return Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="rectangle",
        min_sigma=5.0, max_sigma=40.0,
        **PRIORS,
    )


def _inject_samples(model, params=PARAMS, n_draws=2):
    model.samples = {
        k: jnp.full((n_draws,), jnp.asarray(v, dtype=jnp.float32))
        for k, v in params.items()
    }


def _support_fingerprint(support):
    table = support.mass_table
    return {
        "mode": support.mode,
        "min_sigma": support.min_sigma,
        "max_sigma_real": support.max_sigma_real,
        "spatial_window": support.spatial_window,
        "table_id": None if table is None else table.provenance.get("table_id"),
        "events_sha256": None if table is None else table.events_sha256,
        "n_events": None if table is None else table.n_events,
        "values": None if table is None else np.array(table.values, copy=True),
        "slopes": None if table is None else np.array(table.slopes, copy=True),
    }


def _heldout_table(test_data):
    return prepare_table_for_model(
        test_data, A, min_sigma=5.0, max_sigma=40.0)


# -------------------------------------------------------------------- RED ----

def test_heldout_polygon_without_mass_table_raises():
    train = _polygon_model(_events(4, seed=1))
    test = _events(7, seed=2)
    _inject_samples(train)
    with pytest.raises(ValueError, match="mass_table"):
        train.log_expected_likelihood(test)


def test_heldout_polygon_training_table_rejected():
    """A table built on training events must not score a different realization."""
    train_data = _events(5, seed=10)
    test_data = _events(5, seed=11)
    train = _polygon_model(train_data)
    _inject_samples(train)
    with pytest.raises(ValueError, match="mass table|events|sha256|compat"):
        train.log_expected_likelihood(
            test_data, mass_table=train.excitation_support.mass_table)


def test_heldout_polygon_unequal_counts_scores_with_explicit_table():
    train = _polygon_model(_events(4, seed=1))
    test = _events(7, seed=2)
    _inject_samples(train)
    got = train.log_expected_likelihood(test, mass_table=_heldout_table(test))
    assert np.isfinite(got)


def test_heldout_polygon_equal_counts_uses_test_location_masses():
    """Equal n but different locations must use TEST event masses."""
    train_data = _events(5, seed=10, x_lo=20.0, x_hi=80.0, y_lo=20.0, y_hi=80.0)
    test_data = _events(5, seed=11, x_lo=120.0, x_hi=180.0, y_lo=120.0, y_hi=180.0)
    assert not np.allclose(train_data[["X", "Y"]].to_numpy(),
                           test_data[["X", "Y"]].to_numpy())

    train = _polygon_model(train_data)
    oracle = _polygon_model(test_data)
    _inject_samples(train)
    _inject_samples(oracle)

    from jax.scipy.special import logsumexp
    from bstpp.utils import aligned_difference_pairs

    wrong_args, _ = train._scale_xyt(
        test_data, train.args.copy(), train.prepared_partitions.support_cells)
    coords, t_vals, x_vals, y_vals = aligned_difference_pairs(
        wrong_args["t_events"], wrong_args["xy_events"][0],
        wrong_args["xy_events"][1], train.args["window"],
        spatial_window=train.args.get("spatial_window"),
        axis_scales=np.asarray(train.args["axis_scales"]),
    )
    wrong_args.update(coords=coords, t_vals=t_vals, x_vals=x_vals, y_vals=y_vals)
    for k in ("batch_size", "num_samples", "num_warmup", "num_chains", "thinning"):
        wrong_args.pop(k, None)
    wrong_ll = log_likelihood(train.model, train.samples, wrong_args)["loglik_factor"]
    wrong_score = float(
        (logsumexp(wrong_ll, axis=0) - jnp.log(wrong_ll.shape[0])).sum())

    oracle_ll = log_likelihood(
        oracle.model, oracle.samples, oracle.args)["loglik_factor"]
    oracle_score = float(
        (logsumexp(oracle_ll, axis=0) - jnp.log(oracle_ll.shape[0])).sum())
    assert abs(wrong_score - oracle_score) > 1e-3, (
        "fixture too weak: train-mass scoring already matches oracle")

    got = train.log_expected_likelihood(
        test_data, mass_table=_heldout_table(test_data))
    assert got == pytest.approx(oracle_score, abs=1e-4, rel=0)


def test_heldout_polygon_scoring_does_not_mutate_training_support():
    train = _polygon_model(_events(4, seed=20))
    test = _events(6, seed=21)
    _inject_samples(train)

    before_args_id = id(train.args)
    before_support = train.excitation_support
    before_support_id = id(before_support)
    before_table_id = id(before_support.mass_table)
    before_fp = _support_fingerprint(before_support)
    before_keys = set(train.args.keys())

    train.log_expected_likelihood(test, mass_table=_heldout_table(test))
    train.log_expected_likelihood(test, mass_table=_heldout_table(test))

    assert id(train.args) == before_args_id
    assert set(train.args.keys()) == before_keys
    assert id(train.excitation_support) == before_support_id
    assert train.args.get("excitation_support") is before_support
    assert id(train.excitation_support.mass_table) == before_table_id
    after_fp = _support_fingerprint(train.excitation_support)
    assert after_fp["events_sha256"] == before_fp["events_sha256"]
    assert after_fp["table_id"] == before_fp["table_id"]
    assert after_fp["n_events"] == before_fp["n_events"]
    np.testing.assert_array_equal(after_fp["values"], before_fp["values"])
    np.testing.assert_array_equal(after_fp["slopes"], before_fp["slopes"])


def test_heldout_rectangle_unequal_counts_unchanged():
    """Rectangle mode must keep working and not require a mass table."""
    train_data = _events(4, seed=30)
    test_data = _events(9, seed=31)
    train = _rectangle_model(train_data)
    oracle = _rectangle_model(test_data)
    _inject_samples(train)
    _inject_samples(oracle)

    assert train.excitation_support.mode == "rectangle"
    assert train.excitation_support.mass_table is None
    support_id = id(train.excitation_support)

    from jax.scipy.special import logsumexp
    oracle_ll = log_likelihood(
        oracle.model, oracle.samples, oracle.args)["loglik_factor"]
    oracle_score = float(
        (logsumexp(oracle_ll, axis=0) - jnp.log(oracle_ll.shape[0])).sum())

    got = train.log_expected_likelihood(test_data)
    assert np.isfinite(got)
    assert got == pytest.approx(oracle_score, abs=1e-4, rel=0)
    assert train.excitation_support.mass_table is None
    assert id(train.excitation_support) == support_id


def test_heldout_nonfinite_rejected():
    train = _rectangle_model(_events(4, seed=40))
    _inject_samples(train)
    bad = _events(3, seed=41)
    bad.loc[1, "X"] = np.nan
    with pytest.raises(DataContractError, match="nonfinite|non-numeric"):
        train.log_expected_likelihood(bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["X", "Y", "T"])
def test_heldout_each_nonfinite_type_rejected(bad, field):
    train = _rectangle_model(_events(4, seed=40))
    _inject_samples(train)
    held = _events(3, seed=42)
    held.loc[0, field] = bad
    with pytest.raises(DataContractError, match="nonfinite|field counts"):
        train.log_expected_likelihood(held)


def test_heldout_inf_rejected_under_report_mode():
    domain = gpd.GeoDataFrame(geometry=[box(0, 0, 200, 200)])
    train_data = _events(4, seed=50)
    train = Hawkes_Model(
        train_data, domain, T_DAYS, cox_background=False,
        excitation_support="rectangle",
        data_contracts="report",
        **PRIORS,
    )
    _inject_samples(train)
    held = _events(3, seed=51)
    held.loc[1, "Y"] = np.inf
    with pytest.raises(DataContractError, match="nonfinite|field counts"):
        train.log_expected_likelihood(held)


def test_heldout_cov_ind_length_validated():
    """cov_ind must match the held-out event count; partial coverage fails loud."""
    domain = gpd.GeoDataFrame(geometry=[box(0, 0, 200, 200)])
    cov = gpd.GeoDataFrame(
        {"v": [1.0]},
        geometry=[box(0, 0, 100, 200)],  # covers only half the domain
    )
    train_data = _events(4, seed=50, x_lo=10, x_hi=90, y_lo=10, y_hi=190)
    train = Hawkes_Model(
        train_data, domain, T_DAYS, cox_background=False,
        excitation_support="rectangle",
        spatial_cov=cov, cov_names=["v"],
        data_contracts="report",  # allow domain gap so held-out can miss cov
        **PRIORS,
    )
    _inject_samples(
        train,
        params=dict(PARAMS, w=np.float32(0.0), b_0=np.float32(0.0)),
    )
    # Held-out point outside the covariate footprint → incomplete cov_ind.
    test = pd.DataFrame({
        "X": [50.0, 150.0],
        "Y": [100.0, 100.0],
        "T": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="cov_ind"):
        train.log_expected_likelihood(test)
