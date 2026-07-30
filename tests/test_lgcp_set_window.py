"""LGCP_Model.set_window must fail clearly (pre-3f CF+API).

LGCP has no excitation pairs or computational cutoffs. The inherited base
set_window is order-dependent and can raise KeyError('window'). Public calls,
including zero-argument calls, must raise a clear NotImplementedError
independent of call history. Do not populate unused Hawkes state.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest

from bstpp.main import LGCP_Model

T_DAYS = 30.0
A = np.array([[0.0, 1.0], [0.0, 1.0]])


def _data(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(0.05, 0.95, n),
        "Y": rng.uniform(0.05, 0.95, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _lgcp():
    return LGCP_Model(
        _data(), A, T_DAYS,
        a_0=dist.Normal(0, 5),
    )


def test_lgcp_set_window_zero_arg_raises_clearly():
    m = _lgcp()
    assert "window" not in m.args
    with pytest.raises(NotImplementedError, match="excitation|Hawkes|cutoff"):
        m.set_window()
    assert "window" not in m.args
    assert "coords" not in m.args


def test_lgcp_set_window_with_args_raises_clearly():
    m = _lgcp()
    with pytest.raises(NotImplementedError, match="excitation|Hawkes|cutoff"):
        m.set_window(12.0, spatial_window=0.3)
    assert "window" not in m.args


def test_lgcp_set_window_independent_of_call_history():
    m = _lgcp()
    # First call must not leave partial Hawkes state that changes a later error.
    with pytest.raises(NotImplementedError, match="excitation|Hawkes|cutoff"):
        m.set_window(10.0)
    with pytest.raises(NotImplementedError, match="excitation|Hawkes|cutoff"):
        m.set_window()
    with pytest.raises(NotImplementedError, match="excitation|Hawkes|cutoff"):
        m.set_window(spatial_window=0.2)
    assert "window" not in m.args
    assert "spatial_window" not in m.args
    assert "coords" not in m.args
    assert not hasattr(m, "cutoff_provenance")
