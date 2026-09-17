"""Build shim.

All static metadata moved to pyproject.toml's [project] table at OP-32's
closure. Once that table exists setuptools treats it as authoritative, and
passing the same fields to setup() again is an error rather than a duplicate --
so this file is deliberately empty of metadata.

The runtime pins are still owned by requirements-runtime.txt; pyproject reaches
them through [tool.setuptools.dynamic], which is why `dynamic = ["dependencies"]`
is declared there. This file remains so that the legacy
`python setup.py bdist_wheel` invocation in
tests/test_packaging_runtime_metadata.py keeps working.
"""

from setuptools import setup

setup()
