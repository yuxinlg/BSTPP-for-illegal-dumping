"""Supplied PolygonMassTable must match the model realization exactly.

A table is valid only for the recorded domain union, event coordinates and
row order, event count, spatial window, sigma range/grid, and build settings.
Equal row counts alone are not evidence of compatibility; mismatched tables
must be rejected before reuse.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import dataclasses

import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest

from bstpp.main import Hawkes_Model

T_DAYS = 30.0
A = np.array([[0.0, 200.0], [0.0, 200.0]])
PRIORS = dict(
    a_0=dist.Normal(0, 5),
    alpha=dist.Beta(2, 2),
    beta=dist.HalfNormal(1.0),
    sigmax_2=dist.HalfNormal(40.0),
)


def _events(n, seed, *, x_lo=20.0, x_hi=180.0, y_lo=20.0, y_hi=180.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(x_lo, x_hi, n),
        "Y": rng.uniform(y_lo, y_hi, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _polygon_model(data, *, mass_table=None, spatial_window=None,
                   min_sigma=5.0, max_sigma=40.0):
    return Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=min_sigma, max_sigma=max_sigma,
        spatial_window=spatial_window,
        mass_table=mass_table,
        **PRIORS,
    )


def _compatible_table():
    m = _polygon_model(_events(4, seed=1))
    return m, m.excitation_support.mass_table


def test_supplied_table_accepted_when_identities_match():
    m0, table = _compatible_table()
    data = _events(4, seed=1)
    m = _polygon_model(data, mass_table=table)
    assert m.excitation_support.mass_table.events_sha256 == table.events_sha256
    assert m.excitation_support.mass_table.geometry_sha256 == table.geometry_sha256
    assert m.excitation_support.mass_table.n_events == 4


def test_supplied_table_rejects_unequal_event_count():
    _, table = _compatible_table()
    with pytest.raises(ValueError, match="event|n_events|mass table"):
        _polygon_model(_events(7, seed=2), mass_table=table)


def test_supplied_table_rejects_equal_count_different_locations():
    _, table = _compatible_table()
    other = _events(4, seed=3, x_lo=20.0, x_hi=80.0, y_lo=20.0, y_hi=80.0)
    assert not np.allclose(
        _events(4, seed=1)[["X", "Y"]].to_numpy(),
        other[["X", "Y"]].to_numpy(),
    )
    with pytest.raises(ValueError, match="events_sha256|event"):
        _polygon_model(other, mass_table=table)


def test_supplied_table_rejects_spatial_window_mismatch():
    _, table = _compatible_table()
    assert table.spatial_window is None
    with pytest.raises(ValueError, match="spatial_window"):
        _polygon_model(_events(4, seed=1), mass_table=table, spatial_window=40.0)


def test_supplied_table_rejects_sigma_range_mismatch():
    _, table = _compatible_table()
    with pytest.raises(ValueError, match="sigma"):
        _polygon_model(
            _events(4, seed=1), mass_table=table,
            min_sigma=5.0, max_sigma=80.0,
        )


def test_supplied_table_rejects_geometry_mismatch():
    _, table = _compatible_table()
    # Same event set, different domain rectangle → different geometry hash.
    A2 = np.array([[0.0, 300.0], [0.0, 200.0]])
    with pytest.raises(ValueError, match="geometry"):
        Hawkes_Model(
            _events(4, seed=1), A2, T_DAYS, cox_background=False,
            excitation_support="polygon",
            min_sigma=5.0, max_sigma=40.0,
            mass_table=table,
            **PRIORS,
        )


def test_supplied_table_rejects_nonfinite_values():
    _, table = _compatible_table()
    bad_values = np.array(table.values, copy=True)
    bad_values[0, 0] = np.nan
    bad = dataclasses.replace(table, values=bad_values)
    with pytest.raises(ValueError, match="finite|nonfinite|NaN"):
        _polygon_model(_events(4, seed=1), mass_table=bad)


def test_supplied_table_rejects_shape_mismatch():
    _, table = _compatible_table()
    bad_values = table.values[:, :-1]
    bad = dataclasses.replace(table, values=bad_values)
    with pytest.raises(ValueError, match="shape"):
        _polygon_model(_events(4, seed=1), mass_table=bad)
