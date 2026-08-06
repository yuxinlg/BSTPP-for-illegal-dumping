"""CI-9: an enumerated-value argument is validated whether or not its consumer runs.

THE DEFECT. ``standardize_cov`` is drawn from a fixed set --- ``None`` or
``'domain_area'``. The only validation lives in
``preparation.attach_covariate_partitions``, which runs ONLY inside the
``spatial_cov is not None`` branch. So with no covariates supplied, EVERY value
is accepted and ignored: ``'nonsense'``, ``True``, ``42``.

AND IT IS WORSE THAN SILENT. ``main.py`` sets
``self.standardization = {'method': 'none', ...}`` before the covariate branch,
so a model built with a value the package does not accept then REPORTS a
standardization record that looks exactly like a legitimate "off". D-10's
always-report clause --- which exists so the model can never be silent about
whether it standardized --- emits a well-formed record for an input that was
never valid. A user reading ``model.standardization`` sees nothing wrong.

WHY IT IS NOT CLOSED BY MOVING THE CHECK. D-43 clause 1 relocates a field whose
value depends on ``ModelData`` to bind time. Bind time still needs a
``ModelData`` to bind to, and the no-covariates path is precisely the one where
the consuming leg never runs. CI-9 is therefore a distinct invariant: validate
the enumerated value at CONSTRUCTION, independently of whether the leg that
consumes it executes.

CI-9 is a VALUE invariant; CI-7/CI-8 are TYPE invariants. Separate by D-40's
owners-by-quantity test: ``'nonsense'`` is a valid ``str`` and an invalid
``standardize_cov``, so the accept sets differ.

RED at b98e91d: all four rows below fail --- three because construction
succeeds where it must raise, one because the reported record is well-formed
for a rejected value.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import geopandas as gpd                                          # noqa: E402
import numpy as np                                               # noqa: E402
import numpyro.distributions as dist                             # noqa: E402
import pandas as pd                                              # noqa: E402
import pytest                                                    # noqa: E402
from shapely.geometry import box                                 # noqa: E402

from bstpp.main import Hawkes_Model                              # noqa: E402

T_DAYS = 200.0
A = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
COV = gpd.GeoDataFrame(
    {"v": [0.5, -1.0, 1.5, -0.5]},
    geometry=[box(0, 0, 0.5, 0.5), box(0.5, 0, 1, 0.5),
              box(0, 0.5, 0.5, 1), box(0.5, 0.5, 1, 1)])

# Values the package does not accept. `True` is the legacy migration case and
# is named separately in the clause; the other two are ordinary rejects.
BAD = ["nonsense", True, 42]


def _data(n=15, seed=4):
    r = np.random.RandomState(seed)
    return pd.DataFrame({"X": r.uniform(0.05, 0.95, n),
                         "Y": r.uniform(0.05, 0.95, n),
                         "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n))})


def _build(value, *, covariates: bool):
    kw = dict(spatial_cov=COV.copy(), cov_names=["v"]) if covariates else {}
    return Hawkes_Model(_data(), A, T_DAYS, cox_background=False,
                        standardize_cov=value, **kw, **PRIORS)


@pytest.mark.parametrize("value", BAD)
def test_rejected_without_covariates(value):
    """THE FIRING ROW. No covariates, so the consuming leg never runs -- and

    the value is still not one the package accepts.
    """
    with pytest.raises(ValueError, match="standardize_cov"):
        _build(value, covariates=False)


@pytest.mark.parametrize("value", BAD)
def test_rejected_with_covariates_too(value):
    """The pre-existing path keeps rejecting. CI-9 adds a site; it does not

    move one, so this must not become the only place the check lives.
    """
    with pytest.raises(ValueError, match="standardize_cov"):
        _build(value, covariates=True)


def test_legacy_boolean_is_still_named_as_such():
    """OP-3/OP-4 settled that legacy booleans are rejected EXPLICITLY, never

    silently reinterpreted. Consolidating the clause must not lose the word
    that tells a migrating user what happened to their argument.
    """
    for covariates in (False, True):
        with pytest.raises(ValueError, match="boolean"):
            _build(True, covariates=covariates)


@pytest.mark.parametrize("value", BAD)
def test_no_standardization_record_survives_a_rejected_value(value):
    """The reported record is the reason this is not merely untidy.

    Before CI-9 a rejected value yielded a model whose ``.standardization``
    read ``{'method': 'none', ...}`` -- well-formed, and indistinguishable from
    a legitimate ``standardize_cov=None``. Nothing downstream could tell.
    """
    try:
        model = _build(value, covariates=False)
    except ValueError:
        return                                   # rejected: nothing to report
    pytest.fail(
        f"standardize_cov={value!r} constructed, and the model now reports "
        f"standardization={model.standardization!r} -- a well-formed record "
        "for a value the package does not accept")


def test_accepted_values_still_construct():
    """The accept set is unchanged. CI-9 narrows it to what was already

    documented, and must not narrow it further.
    """
    for value in (None, "domain_area"):
        for covariates in (False, True):
            if value == "domain_area" and not covariates:
                continue     # no columns to standardize; not an accept-set claim
            _build(value, covariates=covariates)
