"""CI-10: a boolean config argument is a bool, not whatever is truthy.

THE DEFECT, AND IT IS THE OPPOSITE OF CI-9's. ``cox_background`` is documented
``bool`` and consumed as ``if cox_background:``. Nothing checks it, so the
accept set is "every Python object", and the branch is taken by TRUTHINESS.
Measured at the pre-change tip on the no-covariate path, all of these
construct silently:

    cox_background='false'     -> selects the COX background
    cox_background='nonsense'  -> selects the COX background
    cox_background=0           -> selects plain hawkes
    cox_background=[]          -> selects plain hawkes

THE FIRST ROW IS THE WHOLE POINT. CI-9's defect was a value accepted and
IGNORED; this one is a value accepted and ACTED ON, in the direction opposite
to what it says. A user who writes ``cox_background='false'`` gets a
Gaussian-process background, a different model with different parameters, and
nothing anywhere reports a problem -- ``model.args['model']`` reads
``'cox_hawkes'`` and is a perfectly well-formed record of a model they did not
ask for. Every figure downstream is then of the other model.

THE SHIPPED DEFAULT IS PART OF THE DEFECT, not an incidental detail. It is the
string ``'cox'``, so the package's own default is outside the type its
docstring declares, and any bool-only enforcement must move it. That is why
this is class SC/API and not a quiet tightening: the default's TYPE changes,
its behaviour does not (both ``'cox'`` and ``True`` are truthy, so the selected
model is identical), and ``cox_background='cox'`` stops being accepted.

WHY NOT ACCEPT ``'cox'`` AS A LEGACY ALIAS. Because keeping exactly one string
alive keeps the truthiness coercion alive with it, and the reader then has to
remember which string is the good one -- while ``'false'``, the string that
motivates the whole row, differs from it only in spelling. OP-3/OP-4 settled
the same question the other way round for ``standardize_cov``: legacy values
are rejected EXPLICITLY, never silently reinterpreted. The clause names the
replacement.

``np.bool_`` IS ACCEPTED, and the reason is a CPython fact rather than a
preference. D-42 accepts ``np.float64`` because it is a ``float`` subclass.
``bool`` CANNOT BE SUBCLASSED in CPython, so ``np.bool_`` has no way to opt
into the same treatment; rejecting it would punish numpy for a language
restriction rather than for being the wrong quantity. ``np.bool_(True)`` is a
boolean in every sense the invariant cares about.

RED at f40591e: the rejection rows fail because construction succeeds, and the
default-type row fails because the default is ``'cox'``.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import inspect                                                     # noqa: E402

import numpy as np                                                 # noqa: E402
import numpyro.distributions as dist                               # noqa: E402
import pandas as pd                                                # noqa: E402
import pytest                                                      # noqa: E402

from bstpp.main import Hawkes_Model                                 # noqa: E402

# The two clause-identity rows below import from ``bstpp.config`` INSIDE the
# test, not here. A module-level import of API that does not exist yet fails
# at collection and takes every behavioural row down with it -- the red that
# proves nothing, which D-41's minimal-revert clause names explicitly. This
# way the pre-change capture shows each row failing for its own reason.

T_DAYS = 200.0
A = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))

#: Every one of these constructs at the pre-change tip. `'false'` and `'cox'`
#: are listed first because they are the two that a reader would defend.
BAD = ["false", "cox", "nonsense", 0, 1, [], None, 1.0]


def _data(n=15, seed=4):
    r = np.random.RandomState(seed)
    return pd.DataFrame({"X": r.uniform(0.05, 0.95, n),
                         "Y": r.uniform(0.05, 0.95, n),
                         "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n))})


def _build(**over):
    kw = dict(PRIORS)
    kw.update(over)
    return Hawkes_Model(_data(), A, T_DAYS, **kw)


@pytest.mark.parametrize("value", BAD)
def test_a_non_bool_is_rejected_at_construction(value):
    """THE FIRING ROW. No covariates, no fit -- the constructor must refuse."""
    with pytest.raises(ValueError, match="cox_background"):
        _build(cox_background=value)


def test_the_string_that_reads_as_false_selects_cox_today():
    """The row that makes this worth an SC change rather than a docstring fix.

    Asserted through the PUBLIC record, not the branch: after the fix the
    value never gets far enough to select anything, and before it, the record
    a user would inspect says 'cox_hawkes'. Either way this row is about what
    ``cox_background='false'`` is allowed to produce.
    """
    with pytest.raises(ValueError) as excinfo:
        _build(cox_background="false")
    assert "'false'" in str(excinfo.value), (
        "the rejected value must appear in the message; a user who typed a "
        "string that reads as false needs to see their own argument back")


@pytest.mark.parametrize("value,expected", [(True, "cox_hawkes"),
                                            (False, "hawkes")])
def test_the_two_permitted_values_still_select_what_they_always_did(
        value, expected):
    """The accept set narrows to exactly the documented one and no further."""
    model = _build(cox_background=value)
    assert model.args["model"] == expected


@pytest.mark.parametrize("value,expected", [(np.bool_(True), "cox_hawkes"),
                                            (np.bool_(False), "hawkes")])
def test_numpy_booleans_are_accepted_because_bool_cannot_be_subclassed(
        value, expected):
    model = _build(cox_background=value)
    assert model.args["model"] == expected


def test_the_shipped_default_is_a_bool_and_still_means_cox():
    """The default's TYPE moves and its MEANING does not. Both are pinned.

    Pinning only the type would allow a later edit to flip the default to
    ``False``, which is a different model, under cover of this row.
    """
    default = inspect.signature(Hawkes_Model.__init__) \
        .parameters["cox_background"].default
    assert type(default) is bool, (
        f"the package default must satisfy its own invariant; got {default!r}")
    assert default is True
    assert _build().args["model"] == "cox_hawkes"


def test_the_clause_names_the_replacement_for_the_old_default():
    """A user upgrading hits ``cox_background='cox'`` first, since it was the
    default they may have copied out of the signature. The message has to be
    actionable for exactly that person."""
    with pytest.raises(ValueError) as excinfo:
        _build(cox_background="cox")
    text = str(excinfo.value)
    assert "True" in text and "'cox'" in text


def test_one_clause_one_identity_reachable_without_a_constructor():
    """D-40: the validator is callable on its own and renders the same text.

    A clause only reachable by building a model is a clause a second owner
    cannot delegate to, which is how the sigma family ended up split.
    """
    from bstpp.config import (cox_background_invariant_clause,
                              validate_cox_background)
    with pytest.raises(ValueError) as excinfo:
        validate_cox_background("nonsense")
    assert str(excinfo.value) == cox_background_invariant_clause(
        cox_background="nonsense")


def test_the_clause_is_ascii_even_for_a_non_ascii_value():
    """D-40's encoding corollary, at the interpolation slot."""
    from bstpp.config import cox_background_invariant_clause
    text = cox_background_invariant_clause(cox_background="\u00e9chec")
    assert text.encode("ascii")


def test_a_bool_is_not_reinterpreted_as_an_int_anywhere():
    """``0``/``1`` are rejected, so the reverse coercion cannot creep back.

    D-42 rejects ``bool`` as an ``int`` subclass; this is the same boundary
    from the other side, and stating it here keeps the two policies from
    drifting into a mutual exception.
    """
    for value in (0, 1):
        with pytest.raises(ValueError, match="cox_background"):
            _build(cox_background=value)
