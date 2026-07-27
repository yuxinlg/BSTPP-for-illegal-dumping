"""Compatibility contract for PolygonMassTable metadata and event identity.

Required metadata (backend, schema, sigma parameterization, interpolation,
slope method/settings, event-hash algorithm) must be present, well-formed,
and match the builder/evaluator constants. ``extra_provenance`` must not
overwrite reserved fields. Event identity must be an exact float64 binary
hash — not a lossy ``.9g`` decimal encoding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import dataclasses

import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from shapely.geometry import box as shapely_box

import bstpp.polygon_mass as pm
from bstpp.main import Hawkes_Model
from bstpp.polygon_mass import (
    PolygonMassTable,
    prepare_polygon_mass_table,
    validate_polygon_mass_table,
)
from tests._polygon_prepare_helpers import prepare_table_for_model

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

# Contract values the package must expose as canonical module constants.
_EXPECTED = {
    "BACKEND_ID": "hybrid_quad_hermite",
    "BACKEND_SCHEMA_VERSION": "hybrid_quad_hermite_numpy_v2",
    "SIGMA_PARAMETERIZATION": "standard_deviation",
    "INTERPOLATION_CONVENTION": "c1_cubic_hermite_uniform_log_sigma",
    "SLOPE_METHOD": "central_fd_log_sigma",
    "SLOPE_FD_EPS": 1e-6,
    "EVENTS_HASH_ALGORITHM": "sha256_le_f64_xy_v1",
}

_COMPAT_KEYS = (
    "backend",
    "backend_schema",
    "sigma_parameterization",
    "interpolation_convention",
    "slope_method",
    "slope_fd_eps",
    "events_hash_algorithm",
)


def _const(name: str):
    assert hasattr(pm, name), f"package must expose canonical constant {name}"
    return getattr(pm, name)


def _events(n=4, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X": rng.uniform(20.0, 180.0, n),
        "Y": rng.uniform(20.0, 180.0, n),
        "T": np.sort(rng.uniform(0.5, T_DAYS - 0.5, n)),
    })


def _base_table():
    data = _events()
    table = prepare_table_for_model(
        data, A, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA)
    return data, table


def _validate(table, data):
    geom = shapely_box(0.0, 0.0, 200.0, 200.0)
    validate_polygon_mass_table(
        table,
        domain_geom=geom,
        event_x_real=data["X"].to_numpy(dtype=float),
        event_y_real=data["Y"].to_numpy(dtype=float),
        spatial_window=None,
        sigma_min=MIN_SIGMA,
        sigma_max=MAX_SIGMA,
        h_panel=float(table.h_panel),
        gl_order=int(table.gl_order),
    )


class _Missing:
    pass


_MISSING = _Missing()


def _with_prov(table, **updates):
    prov = dict(table.provenance)
    for key, value in updates.items():
        if value is _MISSING:
            prov.pop(key, None)
        else:
            prov[key] = value
    return dataclasses.replace(table, provenance=prov)


def test_package_exports_canonical_compat_constants():
    for name, expected in _EXPECTED.items():
        got = _const(name)
        if isinstance(expected, float):
            assert float(got) == float(expected)
        else:
            assert got == expected


@pytest.mark.parametrize(
    "key,bad,match",
    [
        # String-valued compatibility fields: missing / empty / wrong.
        ("backend", _MISSING, "backend"),
        ("backend", "", "backend"),
        ("backend", "forged_backend", "backend"),
        ("backend_schema", _MISSING, "backend_schema|schema"),
        ("backend_schema", "", "backend_schema|schema"),
        ("backend_schema", "hybrid_quad_hermite_numpy_v0_legacy",
         "backend_schema|schema"),
        ("sigma_parameterization", _MISSING, "sigma_parameterization"),
        ("sigma_parameterization", "", "sigma_parameterization"),
        ("sigma_parameterization", "variance", "sigma_parameterization"),
        ("interpolation_convention", _MISSING, "interpolation"),
        ("interpolation_convention", "", "interpolation"),
        ("interpolation_convention", "linear_log_sigma", "interpolation"),
        ("slope_method", _MISSING, "slope_method"),
        ("slope_method", "", "slope_method"),
        ("slope_method", "pchip", "slope_method"),
        ("events_hash_algorithm", _MISSING, "events_hash|hash"),
        ("events_hash_algorithm", "", "events_hash|hash"),
        ("events_hash_algorithm", "sha256_decimal_9g_legacy",
         "events_hash|hash"),
        # slope_fd_eps: missing / incompatible numeric (nonnumeric below).
        ("slope_fd_eps", _MISSING, "slope_fd_eps|slope"),
        ("slope_fd_eps", 1e-3, "slope_fd_eps|slope"),
    ],
)
def test_compat_metadata_missing_malformed_or_wrong_rejected(key, bad, match):
    data, table = _base_table()
    assert table.provenance.get("backend") == _const("BACKEND_ID")
    assert table.provenance.get("backend_schema") == _const(
        "BACKEND_SCHEMA_VERSION")
    assert table.provenance.get("sigma_parameterization") == _const(
        "SIGMA_PARAMETERIZATION")
    assert table.provenance.get("interpolation_convention") == _const(
        "INTERPOLATION_CONVENTION")
    assert table.provenance.get("slope_method") == _const("SLOPE_METHOD")
    assert float(table.provenance.get("slope_fd_eps")) == float(
        _const("SLOPE_FD_EPS"))
    assert table.provenance.get("events_hash_algorithm") == _const(
        "EVENTS_HASH_ALGORITHM")

    bad_table = _with_prov(table, **{key: bad})
    with pytest.raises(ValueError, match=match):
        _validate(bad_table, data)


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-number",
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        object(),
    ],
)
def test_slope_fd_eps_nonnumeric_or_nonfinite_rejected(bad):
    """slope_fd_eps is covered separately from the string-field matrix."""
    data, table = _base_table()
    bad_table = _with_prov(table, slope_fd_eps=bad)
    with pytest.raises(ValueError, match="slope_fd_eps|slope"):
        _validate(bad_table, data)


def test_extra_provenance_cannot_overwrite_reserved_compat_fields():
    data = _events()
    forged = {
        "backend": "forged_backend",
        "backend_schema": "forged_schema",
        "sigma_parameterization": "variance",
        "interpolation_convention": "linear_log_sigma",
        "slope_method": "pchip",
        "slope_fd_eps": 0.5,
        "events_hash_algorithm": "sha256_decimal_9g_legacy",
        "geometry_sha256": "0" * 64,
        "note": "descriptive only",
    }
    from bstpp.preparation import prepare_domain

    dom = prepare_domain(A)
    geom = shapely_box(dom.bounds[0, 0], dom.bounds[1, 0],
                       dom.bounds[0, 1], dom.bounds[1, 1])
    table = prepare_polygon_mass_table(
        geom,
        data["X"].to_numpy(dtype=float),
        data["Y"].to_numpy(dtype=float),
        min_sigma=MIN_SIGMA,
        max_sigma=MAX_SIGMA,
        extra_provenance=forged,
    )
    assert forged["backend"] == "forged_backend"
    assert table.provenance["backend"] == _const("BACKEND_ID")
    assert table.provenance["backend_schema"] == _const("BACKEND_SCHEMA_VERSION")
    assert table.provenance["sigma_parameterization"] == _const(
        "SIGMA_PARAMETERIZATION")
    assert table.provenance["interpolation_convention"] == _const(
        "INTERPOLATION_CONVENTION")
    assert table.provenance["slope_method"] == _const("SLOPE_METHOD")
    assert float(table.provenance["slope_fd_eps"]) == float(
        _const("SLOPE_FD_EPS"))
    assert table.provenance["events_hash_algorithm"] == _const(
        "EVENTS_HASH_ALGORITHM")
    assert table.provenance["geometry_sha256"] != "0" * 64
    assert "extra" in table.provenance
    assert table.provenance["extra"]["note"] == "descriptive only"
    # Nested forged reserved keys must not affect top-level compatibility.
    for key in _COMPAT_KEYS:
        assert table.provenance[key] != forged[key]
    _validate(table, data)
    Hawkes_Model(
        data, A, T_DAYS, cox_background=False,
        excitation_support="polygon",
        min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA,
        mass_table=table, **PRIORS,
    )


def test_export_load_preserves_nested_extra_and_rejects_incomplete_sidecar(
        tmp_path):
    data = _events()
    from bstpp.preparation import prepare_domain

    dom = prepare_domain(A)
    geom = shapely_box(dom.bounds[0, 0], dom.bounds[1, 0],
                       dom.bounds[0, 1], dom.bounds[1, 1])
    table = prepare_polygon_mass_table(
        geom,
        data["X"].to_numpy(dtype=float),
        data["Y"].to_numpy(dtype=float),
        min_sigma=MIN_SIGMA,
        max_sigma=MAX_SIGMA,
        extra_provenance={"run_id": "unit-test", "backend": "forged"},
    )
    path = tmp_path / "table.npz"
    table.export_npz(path)
    loaded = PolygonMassTable.load_npz(path)
    assert loaded.provenance["extra"]["run_id"] == "unit-test"
    assert loaded.provenance["backend"] == _const("BACKEND_ID")
    _validate(loaded, data)

    meta_path = Path(str(path) + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["backend_schema"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="backend_schema|schema|incompatible"):
        PolygonMassTable.load_npz(path)


def test_legacy_decimal_9g_event_hash_collision_rejected():
    """Distinct float64 coordinates that collide under the old ``.9g`` hash."""
    x_a = 1.0
    x_b = float(np.nextafter(1.0, 2.0))
    assert f"{x_a:.9g}" == f"{x_b:.9g}"
    assert x_a != x_b

    y = 40.0
    x1 = np.array([40.0, x_a], dtype=np.float64)
    x2 = np.array([40.0, x_b], dtype=np.float64)
    y12 = np.array([40.0, y], dtype=np.float64)
    geom = shapely_box(0.0, 0.0, 200.0, 200.0)

    table = prepare_polygon_mass_table(
        geom, x1, y12, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA)
    with pytest.raises(ValueError, match="events_sha256|event"):
        validate_polygon_mass_table(
            table,
            domain_geom=geom,
            event_x_real=x2,
            event_y_real=y12,
            spatial_window=None,
            sigma_min=MIN_SIGMA,
            sigma_max=MAX_SIGMA,
            h_panel=float(table.h_panel),
            gl_order=int(table.gl_order),
        )

    legacy = _with_prov(
        table, events_hash_algorithm="sha256_decimal_9g_legacy")
    data = pd.DataFrame({"X": x1, "Y": y12, "T": np.array([1.0, 2.0])})
    with pytest.raises(ValueError, match="events_hash|hash|incompatible"):
        _validate(legacy, data)


def test_legacy_v1_schema_rejected_as_incompatible():
    data, table = _base_table()
    legacy = _with_prov(
        table, backend_schema="hybrid_quad_hermite_numpy_v1")
    with pytest.raises(ValueError, match="backend_schema|schema|incompatible"):
        _validate(legacy, data)


# ---------------------- NPZ / sidecar self-consistency (tamper) ------------

def _export_table(tmp_path, *, spatial_window=None):
    data = _events(n=3, seed=7)
    from bstpp.preparation import prepare_domain

    dom = prepare_domain(A)
    geom = shapely_box(dom.bounds[0, 0], dom.bounds[1, 0],
                       dom.bounds[0, 1], dom.bounds[1, 1])
    table = prepare_polygon_mass_table(
        geom,
        data["X"].to_numpy(dtype=np.float64),
        data["Y"].to_numpy(dtype=np.float64),
        min_sigma=MIN_SIGMA,
        max_sigma=MAX_SIGMA,
        spatial_window=spatial_window,
        extra_provenance={"run_id": "sidecar-tamper"},
    )
    path = tmp_path / "table.npz"
    table.export_npz(path)
    meta_path = Path(str(path) + ".meta.json")
    return path, meta_path, table


def _tamper_meta(meta_path, key, value):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if value is _MISSING:
        meta.pop(key, None)
    else:
        meta[key] = value
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")


@pytest.mark.parametrize(
    "key,bad,match",
    [
        ("sigma_min", 9.0, "sigma_min"),
        ("sigma_min", "not-a-float", "sigma_min"),
        ("sigma_min", _MISSING, "sigma_min"),
        ("sigma_max", 99.0, "sigma_max"),
        ("sigma_max", None, "sigma_max"),
        ("spatial_window", 40.0, "spatial_window"),
        ("spatial_window", "null-as-string", "spatial_window"),
        ("h_panel", 1.0, "h_panel"),
        ("h_panel", "bad", "h_panel"),
        ("gl_order", 8, "gl_order"),
        ("gl_order", 16.5, "gl_order"),
        ("geometry_sha256", "0" * 64, "geometry_sha256"),
        ("geometry_sha256", "", "geometry_sha256"),
        ("events_sha256", "f" * 64, "events_sha256"),
        ("events_sha256", 12345, "events_sha256"),
        ("n_knots", 1, "n_knots"),
        ("n_knots", "35", "n_knots"),
        ("n_events", 99, "n_events"),
        ("n_events", None, "n_events"),
    ],
)
def test_load_rejects_sidecar_npz_field_mismatch(tmp_path, key, bad, match):
    """Sidecar must agree with NPZ for every duplicated identity/numeric field."""
    path, meta_path, table = _export_table(tmp_path)
    assert table.spatial_window is None
    assert table.provenance.get("spatial_window") is None
    assert table.provenance["extra"]["run_id"] == "sidecar-tamper"
    _tamper_meta(meta_path, key, bad)
    with pytest.raises(ValueError, match=match):
        PolygonMassTable.load_npz(path)


def test_load_rejects_spatial_window_none_vs_finite_mismatch(tmp_path):
    """NPZ stores NaN for None; sidecar null must not disagree with a finite NPZ."""
    path, meta_path, table = _export_table(tmp_path, spatial_window=25.0)
    assert table.spatial_window == 25.0
    _tamper_meta(meta_path, "spatial_window", None)
    with pytest.raises(ValueError, match="spatial_window"):
        PolygonMassTable.load_npz(path)


def test_load_accepts_consistent_spatial_window_none(tmp_path):
    path, meta_path, table = _export_table(tmp_path, spatial_window=None)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["spatial_window"] is None
    loaded = PolygonMassTable.load_npz(path)
    assert loaded.spatial_window is None
    assert loaded.provenance["extra"]["run_id"] == "sidecar-tamper"
