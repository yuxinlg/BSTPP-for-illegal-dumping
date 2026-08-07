"""CI-7 at the ``sp_var_mu`` site: a config-owned real is type-checked.

THIS IS CI-7, NOT A NEW INVARIANT. D-42's text settles it: *"A config-owned
numeric argument -- any value a config factory validates or stores -- is checked
at the point where the caller's argument still exists, not after a ``float()``
has erased what they passed. CI-7 (real): int or float only; bool rejected as an
int subclass, np.float64 accepted as a float subclass."* ``sp_var_mu`` is
config-owned (it belongs to ``ModelConfig`` per the declared S_WP2), it is a
real, and ``main.py`` stored it as ``float(sp_var_mu)`` -- literally the
``float()`` that erases what the caller passed. A new number would present one
invariant as two.

THE HARM IS NOT CI-9's, AND THIS FILE MUST NOT BORROW ITS JUSTIFICATION.
CI-9's defect was a value accepted and IGNORED, leaving a misleading provenance
record. CI-10's was a value accepted and ACTED ON in the direction opposite to
what it said. THIS one is different again, and worse in a quieter way:

``exp(sp_var_mu)`` is a PAIRED GAIN. ``decode_spatial_field``'s own docstring
says so -- it "restores the log-amplitude factored out of the spatial draws
during VAE training". The gain is paired with the decoder parameters. So:

    sp_var_mu=True  ->  float(True) == 1.0  ->  gain exp(1.0) ~ 2.72

against the calibrated default's ``exp(2.0) ~ 7.39``. The spatial log-intensity
field comes out at the wrong amplitude by a factor of ~2.7, the fit runs, every
diagnostic is well formed, and NOTHING ANYWHERE REPORTS A PROBLEM. That is a
DIFFERENT MODEL, not a mislabelled one.

``'2.0'`` IS THE SUBTLER ACCEPT, and it is the reason a type check is required
rather than a range check. ``float('2.0') == 2.0``, so the string yields the
CORRECT NUMBER BY COINCIDENCE. No inspection of results can ever catch it,
because there is nothing wrong with the results -- until someone passes
``'2,0'``, or a value from a CSV column, and the coincidence stops holding.

``bool`` IS THE ROW THAT NEEDS THE MOST CARE. ``isinstance(True, int)`` is
``True``, so a check written as ``isinstance(v, (int, float))`` ADMITS PRECISELY
THE VALUE MOST NEEDING REJECTION. ``require_config_real`` tests ``isinstance(
value, bool)`` FIRST and rejects, which is why this file reuses it rather than
writing a fresh predicate.

RED at db55ee1: every rejection row fails because construction succeeds
(``float()`` accepts bools and numeric strings), and the ``None`` row fails
because the incidental ``TypeError`` from ``float(None)`` is NOT the canonical
clause -- an accidental exception is not enforcement.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np                                                 # noqa: E402
import numpyro.distributions as dist                               # noqa: E402
import pandas as pd                                                # noqa: E402
import pytest                                                      # noqa: E402

from bstpp.main import Hawkes_Model                                 # noqa: E402

T_DAYS = 200.0
A = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))

#: Every one of these is stored without complaint at the pre-change tip.
#: `True`/`False` first because they are the ones an `(int, float)` check
#: would wave through; `'2.0'` next because it is the one that produces a
#: correct result and therefore can never be found by looking at output.
BAD = [True, False, "2.0", "nonsense", None, [], object()]


def _data(n=15, seed=5):
    r = np.random.RandomState(seed)
    return pd.DataFrame({"X": r.uniform(0.05, 0.95, n),
                         "Y": r.uniform(0.05, 0.95, n),
                         "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n))})


def _build(**over):
    kw = dict(PRIORS)
    kw.update(over)
    return Hawkes_Model(_data(), A, T_DAYS, cox_background=False, **kw)


@pytest.mark.parametrize("value", BAD)
def test_a_non_real_sp_var_mu_is_rejected_at_construction(value):
    """THE FIRING ROW. The constructor must refuse, with the CANONICAL clause.

    Matching on the clause text rather than on the exception type is
    deliberate: ``float(None)`` already raises ``TypeError`` at the pre-change
    tip, and a test satisfied by that would record an accident as enforcement.
    """
    with pytest.raises(ValueError, match="sp_var_mu must be a real number"):
        _build(sp_var_mu=value)


def test_the_bool_row_is_not_admitted_by_an_int_float_check():
    """``isinstance(True, int)`` is True -- state the boundary in a row.

    D-42 rejects ``bool`` as an ``int`` subclass; A-50's CI-10 states the same
    boundary from the other side. Pinning it here keeps the two policies from
    drifting into a mutual exception.
    """
    assert isinstance(True, int), "CPython fact this row depends on"
    for value in (True, False):
        with pytest.raises(ValueError, match="sp_var_mu must be a real number"):
            _build(sp_var_mu=value)


def test_the_string_that_coerces_correctly_is_still_rejected():
    """``float('2.0') == 2.0``: right answer, wrong type, invisible in output."""
    assert float("2.0") == 2.0, "the coincidence this row is about"
    with pytest.raises(ValueError) as excinfo:
        _build(sp_var_mu="2.0")
    assert "'2.0'" in str(excinfo.value), (
        "the rejected value must appear in the message")


@pytest.mark.parametrize("value", [2.0, 3, -1.5, 0, np.float64(2.5)])
def test_reals_are_accepted_and_stored_as_float(value):
    """The accept set narrows to exactly the documented one and no further.

    ``np.float64`` is accepted because it IS a ``float`` subclass (D-42), the
    asymmetry with ``np.bool_`` that A-23 reason 3 established.
    """
    model = _build(sp_var_mu=value)
    assert model.args["sp_var_mu"] == float(value)
    assert type(model.args["sp_var_mu"]) is float


def test_the_shipped_default_is_a_real_and_is_unchanged():
    """Behaviour preservation: the default's value and type both stay put.

    Pinning only the type would let a later edit move the default amplitude --
    a different spatial field -- under cover of this row.
    """
    import inspect
    from bstpp.main import Point_Process_Model
    default = inspect.signature(Point_Process_Model.__init__) \
        .parameters["sp_var_mu"].default
    assert default == 2.0
    assert type(default) is float
    assert _build().args["sp_var_mu"] == 2.0


def test_validation_runs_on_a_path_where_the_spatial_decoder_never_does():
    """CI-9's property, inherited: validated whether or not the consumer runs.

    This model is ``cox_background=False``, so ``decode_spatial_field`` -- the
    only consumer of ``sp_var_mu`` -- is never called. The rejection must still
    happen, or the guard is only as good as the path the user happened to take.
    """
    model = _build()
    assert model.args["model"] == "hawkes", "no spatial decoder on this path"
    with pytest.raises(ValueError, match="sp_var_mu must be a real number"):
        _build(sp_var_mu=True)


def test_one_clause_one_identity_reachable_without_a_constructor():
    """D-40: the same clause text, rendered from the canonical function.

    CI-7 already had an identity before this commit. Reusing it -- rather than
    adding a ``sp_var_mu``-specific clause -- is what makes this the SAME
    invariant at a second site instead of a second invariant wearing its name.
    """
    from bstpp.config import (config_real_invariant_clause,
                              require_config_real)
    with pytest.raises(ValueError) as excinfo:
        require_config_real("sp_var_mu", "nonsense")
    assert str(excinfo.value) == config_real_invariant_clause(
        name="sp_var_mu", value="nonsense")


def test_the_constructor_and_the_validator_render_byte_identical_text():
    """The second site must not paraphrase the first (D-40)."""
    from bstpp.config import config_real_invariant_clause
    with pytest.raises(ValueError) as excinfo:
        _build(sp_var_mu="nonsense")
    assert str(excinfo.value) == config_real_invariant_clause(
        name="sp_var_mu", value="nonsense")
