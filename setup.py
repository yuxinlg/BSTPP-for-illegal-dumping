from pathlib import Path

from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as f:
    desc = f.read()


def _load_runtime_requirements() -> list[str]:
    """Read canonical runtime pins from requirements-runtime.txt."""
    path = Path(__file__).resolve().parent / "requirements-runtime.txt"
    reqs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


setup(
    name='BSTPP',
    version='0.1.3',

    url='https://github.com/imanring/BSTPP.git',
    author='Isaac Manring',
    author_email='isaacamanring@gmail.com',

    install_requires=_load_runtime_requirements(),
    packages=['bstpp'],
    package_data={'bstpp': ['decoders/*', 'data/*']},

    license='MIT',
    py_modules=['bstpp'],
    description="Bayesian Spatiotemporal Point Process",
    long_description=desc,
    long_description_content_type='text/markdown',
)
