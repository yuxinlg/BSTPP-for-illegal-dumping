"""Packaging metadata must publish the validated runtime pins (pre-3f API).

setup.py install_requires must match requirements-runtime.txt critical
constraints so ``pip install`` cannot resolve known-incompatible stacks.

Wheel builds isolate ``build-base`` and ``egg-base`` under ``%TEMP%``.
Leaving them at the repo root races on ``build/bdist.win-amd64/wheel``
(Permission denied on decoder artifacts) when packaging tests overlap or
when Box sync locks in-tree build outputs — the G9 flake.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

CRITICAL = {
    "geopandas": "geopandas>=1.0",
    "jax": "jax==0.4.23",
    "jaxlib": "jaxlib==0.4.23",
    "numpy": "numpy>=1.24.1,<2",
    "numpyro": "numpyro==0.15.0",
    "scipy": "scipy>=1.9.0,<1.13",
}


def _req_key(req: str) -> tuple:
    """Normalize a Requires-Dist string for comparison (specifier order)."""
    from packaging.requirements import Requirement
    r = Requirement(req)
    specs = tuple(sorted((s.operator, s.version) for s in r.specifier))
    return (r.name.lower(), specs)


def _requires_dist_from_wheel(wheel_path: Path) -> list[str]:
    import email
    with zipfile.ZipFile(wheel_path) as zf:
        metas = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        assert len(metas) == 1, metas
        raw = zf.read(metas[0]).decode("utf-8")
    parsed = email.message_from_string(raw)
    return [v for k, v in parsed.items() if k.lower() == "requires-dist"]


def _build_wheel(dist_dir: Path) -> Path:
    """Build one wheel with build/egg trees outside the Box-synced repo.

    ``dist-dir`` alone is not enough: setuptools still writes
    ``build/bdist.win-amd64/wheel`` under the cwd unless ``--build-base``
    is redirected.
    """
    dist_dir = Path(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bstpp-wheel-build-") as build_root:
        root = Path(build_root)
        build_base = root / "build"
        egg_base = root / "egg"
        egg_base.mkdir()
        proc = subprocess.run(
            [
                sys.executable,
                "setup.py",
                "egg_info", f"--egg-base={egg_base}",
                "build", f"--build-base={build_base}",
                "bdist_wheel", f"--dist-dir={dist_dir}",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def test_wheel_requires_dist_contains_critical_runtime_pins():
    with tempfile.TemporaryDirectory() as td:
        wheel = _build_wheel(Path(td))
        reqs = _requires_dist_from_wheel(wheel)
        got = {_req_key(r)[0]: _req_key(r) for r in reqs}
        for name, pin in CRITICAL.items():
            want = _req_key(pin)
            assert got.get(name) == want, (
                f"missing or mismatched Requires-Dist for {name}: "
                f"want {pin!r}; got {reqs}")


def test_disposable_venv_can_import_bstpp_from_wheel():
    """Install the built wheel into a disposable venv; never touch conda env.

    If the resolver cannot fetch matching wheels (offline), record that by
    skipping after still building/inspecting metadata in the sibling test.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wheel = _build_wheel(td / "dist")

        venv_dir = td / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        if os.name == "nt":
            pip = venv_dir / "Scripts" / "pip.exe"
            py = venv_dir / "Scripts" / "python.exe"
        else:
            pip = venv_dir / "bin" / "pip"
            py = venv_dir / "bin" / "python"

        # Prefer offline install of the local wheel only (deps already may be
        # unresolved). Try --no-deps first for importability of bstpp package
        # code; then attempt a full resolve if network allows.
        no_deps = subprocess.run(
            [str(pip), "install", "--no-deps", str(wheel)],
            capture_output=True, text=True,
        )
        assert no_deps.returncode == 0, no_deps.stdout + no_deps.stderr
        # Package import may fail without deps; that is still useful evidence.
        # Attempt full install when possible.
        # Bound network resolve so a stalled index cannot hang the suite
        # (observed multi-minute idle during G9 diagnosis).
        try:
            full = subprocess.run(
                [str(pip), "install", str(wheel)],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            pytest.skip(
                "Resolver-based disposable-venv install timed out "
                f"after {exc.timeout}s"
            )
        if full.returncode != 0:
            pytest.skip(
                "Resolver-based disposable-venv install unavailable "
                f"(pip exit {full.returncode}): {full.stderr[-500:]}"
            )
        imp = subprocess.run(
            [str(py), "-c", "import bstpp; print(bstpp.__name__)"],
            capture_output=True, text=True,
        )
        assert imp.returncode == 0, imp.stdout + imp.stderr
        assert "bstpp" in imp.stdout
