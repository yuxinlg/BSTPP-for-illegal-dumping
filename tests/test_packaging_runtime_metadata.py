"""Packaging metadata must publish the validated runtime pins (pre-3f API).

setup.py install_requires must match requirements-runtime.txt critical
constraints so ``pip install`` cannot resolve known-incompatible stacks.
"""

from __future__ import annotations

import email.message
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


def _requires_dist_from_wheel(wheel_path: Path) -> list[str]:
    import email
    with zipfile.ZipFile(wheel_path) as zf:
        metas = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        assert len(metas) == 1, metas
        raw = zf.read(metas[0]).decode("utf-8")
    parsed = email.message_from_string(raw)
    return [v for k, v in parsed.items() if k.lower() == "requires-dist"]


def test_wheel_requires_dist_contains_critical_runtime_pins():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        # Build a wheel from the repo (isolated build env may need network for
        # build backends; use the working interpreter's setuptools).
        proc = subprocess.run(
            [sys.executable, "setup.py", "bdist_wheel", f"--dist-dir={out}"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
        wheels = list(out.glob("*.whl"))
        assert len(wheels) == 1, wheels
        reqs = _requires_dist_from_wheel(wheels[0])
        reqs_l = [r.replace(" ", "") for r in reqs]
        for name, pin in CRITICAL.items():
            pin_compact = pin.replace(" ", "")
            assert any(pin_compact == r or r.startswith(name) and pin_compact in r
                       for r in reqs_l), (
                f"missing critical Requires-Dist {pin!r}; got {reqs}")


def test_disposable_venv_can_import_bstpp_from_wheel():
    """Install the built wheel into a disposable venv; never touch conda env.

    If the resolver cannot fetch matching wheels (offline), record that by
    skipping after still building/inspecting metadata in the sibling test.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dist = td / "dist"
        dist.mkdir()
        proc = subprocess.run(
            [sys.executable, "setup.py", "bdist_wheel", f"--dist-dir={dist}"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
        wheel = next(dist.glob("*.whl"))

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
        full = subprocess.run(
            [str(pip), "install", str(wheel)],
            capture_output=True, text=True,
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
