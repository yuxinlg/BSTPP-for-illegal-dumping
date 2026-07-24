"""Polygon mass preparation API: hard-require table, no global JAX x64 mutation.

Construction in polygon mode must supply a compatible PolygonMassTable from
``prepare_polygon_mass_table``. The preparation path must not call
``jax.config.update`` (including temporary x64 toggles).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
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
MIN_SIGMA = 5.0
MAX_SIGMA = 40.0


def _events(n=4, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(20.0, 180.0, n),
        "Y": rng.uniform(20.0, 180.0, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def test_polygon_construction_without_mass_table_fails():
    """Polygon mode must not silently build a Hermite table in the constructor."""
    with pytest.raises(ValueError, match="prepare_polygon_mass_table"):
        Hawkes_Model(
            _events(), A, T_DAYS, cox_background=False,
            excitation_support="polygon",
            min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA,
            **PRIORS,
        )


def test_quad_table_build_does_not_toggle_jax_enable_x64(monkeypatch):
    """Table construction must not call jax.config.update for jax_enable_x64."""
    from shapely.geometry import box

    from bstpp.polygon_mass import build_quad_table

    calls = []
    real_update = jax.config.update

    def _spy(key, value):
        calls.append((key, value))
        return real_update(key, value)

    monkeypatch.setattr(jax.config, "update", _spy)
    prev = bool(jax.config.jax_enable_x64)

    build_quad_table(
        box(0.0, 0.0, 200.0, 200.0),
        np.array([40.0, 80.0]),
        np.array([40.0, 120.0]),
        MIN_SIGMA, MAX_SIGMA,
        ws=None,
    )
    x64_calls = [c for c in calls if c[0] == "jax_enable_x64"]
    assert x64_calls == [], f"jax_enable_x64 mutated via config.update: {x64_calls}"
    assert bool(jax.config.jax_enable_x64) is prev


def test_prepare_polygon_mass_table_and_ctor_install_without_x64_toggle(monkeypatch):
    from tests._polygon_prepare_helpers import prepare_table_for_model

    calls = []
    real_update = jax.config.update

    def _spy(key, value):
        calls.append((key, value))
        return real_update(key, value)

    monkeypatch.setattr(jax.config, "update", _spy)
    prev = bool(jax.config.jax_enable_x64)
    data = _events()
    table = prepare_table_for_model(
        data, A, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA)
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA,
        mass_table=table,
        **PRIORS,
    )
    assert m.excitation_support.mass_table is table
    x64_calls = [c for c in calls if c[0] == "jax_enable_x64"]
    assert x64_calls == []
    assert bool(jax.config.jax_enable_x64) is prev


def test_rectangle_mode_unchanged_without_mass_table():
    m = Hawkes_Model(
        _events(), A, T_DAYS, cox_background=False,
        excitation_support="rectangle",
        **PRIORS,
    )
    assert m.excitation_support.mode == "rectangle"
    assert m.excitation_support.mass_table is None


# ---------------------- set_window mass_table replacement (polygon) --------
def _polygon_model_with_table(*, spatial_window=50.0, window=20.0, seed=2):
    from tests._polygon_prepare_helpers import prepare_table_for_model

    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "X": rng.uniform(20.0, 180.0, 8),
        "Y": rng.uniform(20.0, 180.0, 8),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, 8)),
    })
    table = prepare_table_for_model(
        data, A, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA,
        spatial_window=spatial_window)
    m = Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA,
        window=window, spatial_window=spatial_window,
        mass_table=table,
        **PRIORS,
    )
    return m, data


def test_set_window_temporal_only_reuses_polygon_mass_table():
    """Temporal-only change must not invalidate / rebuild the spatial table."""
    m, _ = _polygon_model_with_table(spatial_window=50.0, window=20.0)
    before = m.excitation_support.mass_table
    m.set_window(12.0, spatial_window=50.0)
    assert m.args["window"] == pytest.approx(12.0)
    assert m.args["spatial_window"] == pytest.approx(50.0)
    assert m.excitation_support.mass_table is before


def test_set_window_spatial_change_requires_replacement_mass_table():
    m, _ = _polygon_model_with_table(spatial_window=50.0)
    with pytest.raises(ValueError, match="mass_table"):
        m.set_window(12.0, spatial_window=40.0)


def test_set_window_spatial_change_with_compatible_table_commits():
    from tests._polygon_prepare_helpers import prepare_table_for_model

    m, data = _polygon_model_with_table(spatial_window=50.0)
    before = m.excitation_support.mass_table
    new_table = prepare_table_for_model(
        data, A, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA,
        spatial_window=40.0)
    m.set_window(12.0, spatial_window=40.0, mass_table=new_table)
    assert m.args["spatial_window"] == pytest.approx(40.0)
    assert m.excitation_support.mass_table is new_table
    assert m.excitation_support.mass_table is not before
    assert m.cutoff_provenance.spatial.spatial_window == pytest.approx(40.0)


def test_set_window_incompatible_replacement_leaves_state_unchanged():
    from tests._polygon_prepare_helpers import prepare_table_for_model

    m, data = _polygon_model_with_table(spatial_window=50.0, window=20.0)
    before_window = float(m.args["window"])
    before_sw = float(m.args["spatial_window"])
    before_table = m.excitation_support.mass_table
    before_prov = m.cutoff_provenance.to_dict()
    # Table prepared for a different spatial window than the request.
    bad_table = prepare_table_for_model(
        data, A, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA,
        spatial_window=30.0)
    with pytest.raises(ValueError, match="spatial_window|mass table"):
        m.set_window(12.0, spatial_window=40.0, mass_table=bad_table)
    assert m.args["window"] == pytest.approx(before_window)
    assert m.args["spatial_window"] == pytest.approx(before_sw)
    assert m.excitation_support.mass_table is before_table
    assert m.cutoff_provenance.to_dict() == before_prov
