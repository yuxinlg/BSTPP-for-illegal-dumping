"""ModelConfig: the typed owner of the model-level configuration quantities.

Phase 3f S1.1 (WP2). ``ModelConfig`` takes canonical ownership of four
quantities that lived as constructor locals and ``args`` entries: the resolved
model family, ``cox_background``, ``sp_var_mu`` and ``standardize_cov``. Three
already-landed enforcement clauses relocate into its ``__post_init__``:

  CI-10  cox_background is a bool          (A-50, main.py:2027)
  CI-9   standardize_cov enumerated value  (A-45, main.py:333)
  CI-7   sp_var_mu is a real               (A-53, main.py:564)

WHAT THIS MODULE ASSERTS, AND WHY IT IS SHAPED THIS WAY.

Relocation is behaviour-preserving only if the clause TEXT and the exception
IDENTITY are unchanged and the ORDER the three fire is unchanged. So every
rejection row builds its expected message by CALLING THE SHARED CLAUSE FUNCTION
and asserts string EQUALITY against it -- never a hand-written substring, and
never an alternation. A-26 is why: five pre-existing rows stayed green across a
real identity change because their alternations could not discriminate. The
exception type is asserted IN ADDITION, never instead (protocol section 5.3).

Equality rather than ``pytest.raises(match=...)``: ``match`` is a regex search,
and these clauses contain ``(``, ``)``, ``|`` and ``[``. A regex built from
clause text either needs escaping or silently matches less than it appears to.

THE ORDERING ROWS ARE NOT DISCRIMINATING BY CONSTRUCTION AND ARE NOT OFFERED AS
ENFORCEMENT EVIDENCE (D-41). They pass against both trees, because both trees
raise the same exception for the same input -- that is the point. They are the
equivalence evidence a BP relocation needs, which is a different claim from
enforcement. The pre-change readings they were written against are captured in
``refactor-patches/captures/a55_s1_1_pre_change_order.log``.
"""
import json

import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError

from bstpp.config import (
    CONFIG_KEYWORD_OWNERS,
    ModelConfig,
    NumericalConfig,
    NumericalConfigError,
    config_owner_of,
    config_real_invariant_clause,
    cox_background_invariant_clause,
    standardize_cov_invariant_clause,
)
from bstpp.main import Hawkes_Model, LGCP_Model, Point_Process_Model

T_DAYS = 200.0
A = np.array([[0.0, 1.0], [0.0, 1.0]])
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))

BAD_COX = "cox"      # the retired A-50 string
BAD_STD = True       # legacy boolean, rejected explicitly (OP-3/OP-4)
BAD_SPV = "2.0"      # coerces to the right number; invisible in output


def _data(n=15, seed=5):
    r = np.random.RandomState(seed)
    return pd.DataFrame({"X": r.uniform(0.05, 0.95, n),
                         "Y": r.uniform(0.05, 0.95, n),
                         "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n))})


def _hawkes(**over):
    kw = dict(PRIORS)
    kw.update(over)
    kw.setdefault("cox_background", False)
    return Hawkes_Model(_data(), A, T_DAYS, **kw)


def _lgcp(**over):
    return LGCP_Model(_data(), A, T_DAYS, a_0=dist.Normal(0, 5), **over)


def _valid(**over):
    kw = dict(model="hawkes", cox_background=False, sp_var_mu=2.0,
              standardize_cov=None)
    kw.update(over)
    return ModelConfig.create(**kw)


# ----------------------------------------------------- construction / shape --

@pytest.mark.parametrize("model,cox", [
    ("hawkes", False),
    ("cox_hawkes", True),
    ("lgcp", None),          # LGCP has no cox_background argument at all
])
def test_the_valid_space_constructs_per_family(model, cox):
    cfg = _valid(model=model, cox_background=cox)
    assert cfg.model == model
    assert cfg.cox_background is cox
    assert cfg.sp_var_mu == 2.0
    assert cfg.standardize_cov is None


def test_cox_background_none_skips_the_bool_check_rather_than_failing_it():
    """``None`` means 'this family has no background switch', not 'invalid'.

    Without the skip, LGCP could not build a ModelConfig at all: CI-10 rejects
    everything that is not a ``bool``, and ``None`` is not one.
    """
    assert _valid(model="lgcp", cox_background=None).cox_background is None


def test_the_object_is_frozen():
    cfg = _valid()
    for field, value in (("model", "lgcp"), ("cox_background", True),
                         ("sp_var_mu", 3.0), ("standardize_cov", "domain_area")):
        with pytest.raises(FrozenInstanceError):
            setattr(cfg, field, value)


def test_sp_var_mu_is_stored_coerced_as_the_args_entry_always_was():
    """``args['sp_var_mu']`` has been the COERCED float since A-53.

    ``require_config_real`` returns ``float(value)`` and main.py stored that
    return, so ``np.float64`` arrived as a plain ``float``. The owner must keep
    doing it or the adapter changes a stored type.
    """
    cfg = _valid(sp_var_mu=np.float64(2.5))
    assert cfg.sp_var_mu == 2.5
    assert type(cfg.sp_var_mu) is float
    assert type(_valid(sp_var_mu=3).sp_var_mu) is float


# ------------------------------------------- rejection, on the CANONICAL text --

def test_cox_background_rejects_with_the_canonical_ci10_clause():
    with pytest.raises(ValueError) as excinfo:
        _valid(cox_background=BAD_COX)
    assert str(excinfo.value) == cox_background_invariant_clause(
        cox_background=BAD_COX)
    assert type(excinfo.value) is ValueError, (
        "CI-10's identity is ValueError; a subclass here would be a new "
        "identity for an existing invariant (D-40)")


def test_standardize_cov_rejects_with_the_canonical_ci9_clause():
    with pytest.raises(ValueError) as excinfo:
        _valid(standardize_cov=BAD_STD)
    assert str(excinfo.value) == standardize_cov_invariant_clause(
        standardize_cov=BAD_STD)
    assert type(excinfo.value) is ValueError


def test_sp_var_mu_rejects_with_the_canonical_ci7_clause_and_identity():
    with pytest.raises(NumericalConfigError) as excinfo:
        _valid(sp_var_mu=BAD_SPV)
    assert str(excinfo.value) == config_real_invariant_clause(
        name="sp_var_mu", value=BAD_SPV)
    assert type(excinfo.value) is NumericalConfigError, (
        "CI-7 keeps ONE identity across both its sites (D-40). A "
        "ModelConfig-owned sibling exception would split it in two.")


@pytest.mark.parametrize("value", [True, False, "2.0", "nonsense", None, [],
                                   object()])
def test_the_full_sp_var_mu_reject_set_is_unchanged(value):
    with pytest.raises(NumericalConfigError) as excinfo:
        _valid(sp_var_mu=value)
    assert str(excinfo.value) == config_real_invariant_clause(
        name="sp_var_mu", value=value)


@pytest.mark.parametrize("value", ["cox", "true", 1, 0, None, np.float64(1.0)])
def test_the_cox_background_reject_set_is_unchanged(value):
    """``None`` is absent here on purpose -- it is tested as ACCEPTED above.

    The parametrisation deliberately includes ``1``/``0``: CI-10 rejects them
    because a bool argument is a bool, not whatever is truthy.
    """
    if value is None:
        pytest.skip("None means 'family has no background switch' (accepted)")
    with pytest.raises(ValueError) as excinfo:
        _valid(cox_background=value)
    assert str(excinfo.value) == cox_background_invariant_clause(
        cox_background=value)


def test_np_bool_is_accepted_because_bool_cannot_be_subclassed():
    """The CPython fact CI-10 turns on, restated at the new owner (A-50)."""
    assert _valid(cox_background=np.bool_(True)).cox_background is np.bool_(True)


@pytest.mark.parametrize("value", [None, "domain_area"])
def test_the_standardize_cov_accept_set_is_exactly_two_values(value):
    assert _valid(standardize_cov=value).standardize_cov == value


@pytest.mark.parametrize("value", [True, False, "count", "true", 1, ""])
def test_the_standardize_cov_reject_set_is_unchanged(value):
    with pytest.raises(ValueError) as excinfo:
        _valid(standardize_cov=value)
    assert str(excinfo.value) == standardize_cov_invariant_clause(
        standardize_cov=value)


def test_the_model_family_string_is_stored_and_NOT_validated():
    """Deliberate non-goal: adding a family check here NARROWS an accept set.

    ``Point_Process_Model`` accepts any ``model`` string today -- nothing
    validates it -- and S1 is declared behaviour-preserving. Narrowing it would
    be an SC change needing its own decision and its own CI number, and section
    8 of the slice brief makes 'a new CI number looks necessary' a stop-and-
    report. This row pins the absence so a later commit cannot add one silently
    while believing it is tidying up.
    """
    assert _valid(model="not_a_real_family").model == "not_a_real_family"


# ------------------------------------------------------------- ORDERING rows --
# Non-discriminating by construction (D-41): they pass on both sides of the
# change. Pre-change readings: captures/a55_s1_1_pre_change_order.log.

def test_cox_background_beats_standardize_cov():
    with pytest.raises(ValueError) as excinfo:
        _valid(cox_background=BAD_COX, standardize_cov=BAD_STD)
    assert str(excinfo.value) == cox_background_invariant_clause(
        cox_background=BAD_COX)


def test_standardize_cov_beats_sp_var_mu():
    with pytest.raises(ValueError) as excinfo:
        _valid(standardize_cov=BAD_STD, sp_var_mu=BAD_SPV)
    assert str(excinfo.value) == standardize_cov_invariant_clause(
        standardize_cov=BAD_STD)


def test_cox_background_beats_sp_var_mu():
    with pytest.raises(ValueError) as excinfo:
        _valid(cox_background=BAD_COX, sp_var_mu=BAD_SPV)
    assert str(excinfo.value) == cox_background_invariant_clause(
        cox_background=BAD_COX)


@pytest.mark.parametrize("build,expected", [
    (lambda: _hawkes(cox_background=BAD_COX, standardize_cov=BAD_STD),
     lambda: cox_background_invariant_clause(cox_background=BAD_COX)),
    (lambda: _hawkes(standardize_cov=BAD_STD, sp_var_mu=BAD_SPV),
     lambda: standardize_cov_invariant_clause(standardize_cov=BAD_STD)),
    (lambda: _hawkes(cox_background=BAD_COX, sp_var_mu=BAD_SPV),
     lambda: cox_background_invariant_clause(cox_background=BAD_COX)),
    (lambda: _lgcp(standardize_cov=BAD_STD, sp_var_mu=BAD_SPV),
     lambda: standardize_cov_invariant_clause(standardize_cov=BAD_STD)),
])
def test_the_same_order_holds_through_the_PUBLIC_constructor(build, expected):
    """The order that matters is the one a caller of the real API observes."""
    with pytest.raises(ValueError) as excinfo:
        build()
    assert str(excinfo.value) == expected()


def test_hawkes_still_rejects_cox_background_before_base_construction():
    """The D-40 second site at main.py:2027 must STAY.

    It fires before ``super().__init__`` does any data work, and Hawkes_Model
    derives its family name from ``cox_background`` at main.py:2029-2032 --
    before ``super()`` is reached at :2058. A bad value must therefore be
    refused without the expensive base constructor running at all. Passing
    events that would themselves fail the data contract proves the rejection
    happened first: if base construction ran, the DataContractError would win.
    """
    doomed_events = pd.DataFrame({"X": [0.5, np.nan], "Y": [0.5, 0.5],
                                  "T": [1.0, 2.0]})
    with pytest.raises(ValueError) as excinfo:
        Hawkes_Model(doomed_events, A, T_DAYS, cox_background=BAD_COX, **PRIORS)
    assert str(excinfo.value) == cox_background_invariant_clause(
        cox_background=BAD_COX)


def test_data_contracts_still_beat_every_config_clause():
    """Unchanged in both trees: ModelConfig is built AFTER the events contract.

    main.py:325 runs before :333, and ModelConfig takes :333's position, so a
    caller with bad events still learns about the events.
    """
    doomed_events = pd.DataFrame({"X": [0.5, np.nan], "Y": [0.5, 0.5],
                                  "T": [1.0, 2.0]})
    with pytest.raises(Exception) as excinfo:
        Hawkes_Model(doomed_events, A, T_DAYS, cox_background=False,
                     sp_var_mu=BAD_SPV, **PRIORS)
    assert type(excinfo.value).__name__ == "DataContractError"


def test_sp_var_mu_now_precedes_the_covariate_leg_A_DECLARED_ORDER_CHANGE():
    """DECLARED CHANGE, measured before it was made. Not an accept-set move.

    The covariate leg (main.py:440-520) runs BETWEEN standardize_cov at :333
    and sp_var_mu at :564. ModelConfig takes :333's position and owns all
    three, so sp_var_mu moves AHEAD of that leg. Pre-change, a caller with a
    bad sp_var_mu AND bad covariates saw DataContractError; now they see the
    sp_var_mu clause. Both inputs are invalid in both trees and both raise --
    no argument's accept set moves in either direction -- but WHICH error
    surfaces changes, so it is pinned here rather than left to be discovered.

    Pre-change reading: captures/a55_s1_1_pre_change_order.log, final block.
    """
    cov = pd.DataFrame({"X": [0.25, 0.75], "Y": [0.25, 0.75],
                        "value": [1.0, 2.0]})
    with pytest.raises(NumericalConfigError) as excinfo:
        _hawkes(spatial_cov=cov, cov_names=["no_such_column"],
                cov_grid_size=[0.5, 0.5], sp_var_mu=BAD_SPV)
    assert str(excinfo.value) == config_real_invariant_clause(
        name="sp_var_mu", value=BAD_SPV)


# ------------------------------------------------------------ the ACCEPT SET --

def test_nothing_previously_accepted_is_now_rejected():
    """Every construction the pre-change capture recorded as CONSTRUCTED."""
    assert _hawkes().model_config.model == "hawkes"
    assert _hawkes(cox_background=True).model_config.model == "cox_hawkes"
    assert _lgcp().model_config.model == "lgcp"
    assert _hawkes(standardize_cov="domain_area").model_config \
        .standardize_cov == "domain_area"
    assert _hawkes(sp_var_mu=np.float64(2.5)).model_config.sp_var_mu == 2.5
    # A Distribution-valued unknown kwarg is accepted as a prior at main.py:566
    # whatever its name; only NON-Distributions reach the :569 bare Exception
    # routed to WP9. Pinned so S1.3's verbatim move cannot quietly narrow it.
    assert _hawkes(nonsense_prior=dist.Normal(0, 1)) is not None


@pytest.mark.parametrize("bad,exc", [
    (dict(cox_background=BAD_COX), ValueError),
    (dict(standardize_cov=BAD_STD), ValueError),
    (dict(sp_var_mu=BAD_SPV), NumericalConfigError),
])
def test_nothing_previously_rejected_is_now_accepted(bad, exc):
    with pytest.raises(exc):
        _hawkes(**bad)


# ---------------------------------------------------------------- ADAPTER --

def test_the_adapter_writes_every_args_key_exactly_as_before():
    model = _hawkes(cox_background=True)
    assert model.args["model"] == "cox_hawkes"
    assert model.args["sp_var_mu"] == 2.0
    assert type(model.args["sp_var_mu"]) is float
    assert model.standardization == {'method': 'none', 'columns': [],
                                     'mean': None, 'scale': None}


def test_the_config_is_reachable_on_the_model_and_through_args():
    model = _hawkes()
    assert isinstance(model.model_config, ModelConfig)
    assert model.args["model_config"] is model.model_config


def test_the_adapter_does_not_accept_what_the_owner_rejects():
    """Protocol section 5.2: an adapter translates; it does not widen.

    The public constructor and the typed owner must have the SAME accept set,
    so the same value is offered to both and both must refuse.
    """
    with pytest.raises(NumericalConfigError):
        _valid(sp_var_mu=BAD_SPV)
    with pytest.raises(NumericalConfigError):
        _hawkes(sp_var_mu=BAD_SPV)


def test_the_public_constructor_signature_defaults_are_unchanged():
    import inspect
    params = inspect.signature(Point_Process_Model.__init__).parameters
    assert params["sp_var_mu"].default == 2.0
    assert params["standardize_cov"].default is None
    assert inspect.signature(Hawkes_Model.__init__) \
        .parameters["cox_background"].default is True


# --------------------------------------------------------------- to_record --

def _stable(record):
    return json.dumps(record, sort_keys=False, separators=(",", ":"))


def test_model_config_to_record_is_byte_stable_across_constructions():
    assert _stable(_valid().to_record()) == _stable(_valid().to_record())


def test_to_record_key_order_is_deterministic_not_merely_equal():
    """Equal dicts can serialize differently; the ORDER is the contract."""
    assert list(_valid().to_record()) == list(
        _valid(model="lgcp", cox_background=None).to_record())


def test_to_record_carries_a_schema_version_and_the_owning_type():
    rec = _valid().to_record()
    assert rec["schema_version"] == 1
    assert rec["config_type"] == "ModelConfig"


@pytest.mark.parametrize("over", [
    dict(),
    dict(model="lgcp", cox_background=None),
    dict(cox_background=True, standardize_cov="domain_area"),
    dict(sp_var_mu=np.float64(2.5)),
])
def test_to_record_emits_only_schema_v1_value_types(over):
    """Schema v1 consumes these. An np.float64 leaking through is not JSON."""
    for key, value in _valid(**over).to_record().items():
        assert isinstance(value, (str, bool, int, float, type(None))), (
            f"{key} is {type(value).__name__}, which schema v1 cannot store")
    json.dumps(_valid(**over).to_record())      # must not raise


def test_numerical_config_also_gained_a_to_record():
    """S1's exit gate requires to_record() on all FIVE objects.

    NumericalConfig had none at the slice baseline; this is the smallest place
    to add it, so it lands with the first object that needs the same contract.
    """
    cfg = NumericalConfig.create(support_mode="rectangle")
    rec = cfg.to_record()
    assert rec["schema_version"] == 1
    assert rec["config_type"] == "NumericalConfig"
    assert _stable(rec) == _stable(
        NumericalConfig.create(support_mode="rectangle").to_record())
    json.dumps(rec)


def test_numerical_config_to_record_reports_resolved_bounds_and_mode():
    cfg = NumericalConfig.create(support_mode="rectangle", min_sigma=0.05,
                                 max_sigma=0.5)
    rec = cfg.to_record()
    assert rec["support_mode"] == "rectangle"
    assert rec["min_sigma"] == 0.05
    assert rec["max_sigma"] == 0.5


# ------------------------------------------------------ the KEYWORD REGISTRY --

def test_the_registry_declares_model_configs_four_quantities():
    """Protocol section 5.2 wants ONE table, not shims at call sites.

    This table is what WP10 deletes, so it has to be findable and complete for
    the quantities that have landed. Later S1 commits add rows; they do not add
    shims elsewhere.
    """
    for keyword in ("model", "cox_background", "sp_var_mu", "standardize_cov"):
        assert CONFIG_KEYWORD_OWNERS[keyword].owner == "ModelConfig"
        assert config_owner_of(keyword) == "ModelConfig"


def test_every_registered_keyword_names_the_args_key_it_feeds():
    assert CONFIG_KEYWORD_OWNERS["sp_var_mu"].args_key == "sp_var_mu"
    assert CONFIG_KEYWORD_OWNERS["model"].args_key == "model"
    # standardize_cov feeds no args key: it is reported on self.standardization
    # and passed to attach_covariate_partitions. None records that honestly
    # rather than inventing a key that does not exist.
    assert CONFIG_KEYWORD_OWNERS["standardize_cov"].args_key is None
    assert CONFIG_KEYWORD_OWNERS["cox_background"].args_key is None


def test_an_unregistered_keyword_is_reported_not_guessed():
    assert config_owner_of("not_a_config_keyword") is None
