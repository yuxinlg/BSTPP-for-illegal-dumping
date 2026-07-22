"""Custom-trigger rng compatibility mechanism (Phase 2d).

The RNG-unification commit supported old-style triggers -- those whose
simulate_trigger accepts only (pars), a legacy THIRD-PARTY shape now that
both in-repo trigger classes (including Temporal_Power_Law, revised in the
same commit) declare rng -- through a per-draw broad ``except TypeError``
fallback. That catch misclassified ANY TypeError raised inside a new-style
trigger as an old signature and silently re-executed the trigger without
rng: a masked user bug plus quietly abandoned reproducibility. Generator/
RandomState API differences make this a realistic hazard, not a theoretical
one (Generator has no .randn, RandomState no .integers), and the double
execution also repeats side effects.

The mechanism is now signature inspection, once per _sim_offspring call
(utils.accepts_rng_kwarg). Classification is KIND-aware, not name-only:
'rng' counts only where rng=... is actually a valid call form
(POSITIONAL_OR_KEYWORD or KEYWORD_ONLY), plus **kwargs; a POSITIONAL_ONLY
'rng' (def f(pars, rng, /)) and a VAR_POSITIONAL *rng both make rng=... a
TypeError and classify old-style -- routing them as new-style would turn a
working legacy trigger into a crash.

- test_accepts_rng_kwarg_classification              -> the detector, incl.
  both in-repo triggers as new-style and the kind edge cases
- test_new_style_trigger_internal_typeerror_surfaces -> the mask, RED pre-fix
- test_old_signature_trigger_still_supported         -> the compatibility
  promise for the legacy third-party shape
- test_positional_only_rng_classified_old_style      -> the kind rule, live
  through _sim_offspring
- test_var_kwargs_trigger_receives_rng               -> the VAR_KEYWORD branch
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import numpyro.distributions as dist
import pytest

from bstpp.main import Hawkes_Model
from bstpp.trigger import (Spatial_Symmetric_Gaussian, Temporal_Exponential,
                           Temporal_Power_Law)
from bstpp.utils import accepts_rng_kwarg

T_DAYS = 2.5 * 365.0
_rng = np.random.RandomState(0)
_N = 60
DATA = pd.DataFrame({
    "X": _rng.uniform(0.05, 0.95, _N),
    "Y": _rng.uniform(0.05, 0.95, _N),
    "T": np.sort(_rng.uniform(0, T_DAYS, _N)),
})
A_GDF = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]})
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
# High alpha so the cascade reliably draws from the triggers.
PAR = {"a_0": 0.5, "alpha": 0.6, "beta": 2.0, "sigmax_2": 0.02}
BG = np.array([[0.5, 0.5, 5.0], [0.4, 0.6, 15.0], [0.6, 0.4, 25.0]])


class _BuggyNewStyleSpatial(Spatial_Symmetric_Gaussian):
    """New-style trigger (declares rng) with a bug that fires ONLY on the
    Generator code path -- the shape of a real Generator/RandomState API
    mismatch. Under the broad TypeError fallback this bug was swallowed, the
    trigger silently re-executed without rng, and simulation 'succeeded'
    while ignoring the caller's Generator."""

    def simulate_trigger(self, pars, rng=None):
        if rng is not None:
            raise TypeError("internal bug on the Generator path")
        return np.random.normal(scale=pars['sigmax_2'] ** 0.5, size=2)


class _OldStyleSpatial(Spatial_Symmetric_Gaussian):
    """Legacy third-party signature: simulate_trigger(pars) only -- the shape
    of user triggers written before the rng kwarg existed. (No in-repo
    trigger has this shape any more; Temporal_Power_Law gained rng in the
    RNG-unification commit.)"""

    def simulate_trigger(self, pars):  # noqa: ARG002 - legacy signature
        return np.random.normal(scale=pars['sigmax_2'] ** 0.5, size=2)


class _PositionalOnlyRngSpatial(Spatial_Symmetric_Gaussian):
    """'rng' in name only: POSITIONAL_ONLY, so rng=... is a TypeError. Must
    classify OLD-style -- a name-only detector would route it new-style and
    crash a working trigger."""

    def simulate_trigger(self, pars, rng=None, /):
        gen = rng if rng is not None else np.random
        return gen.normal(scale=pars['sigmax_2'] ** 0.5, size=2)


class _VarPositionalRngSpatial(Spatial_Symmetric_Gaussian):
    """'rng' in name only: VAR_POSITIONAL (*rng). Same kind rule."""

    def simulate_trigger(self, pars, *rng):  # noqa: ARG002
        return np.random.normal(scale=pars['sigmax_2'] ** 0.5, size=2)


class _VarKwargsSpatial(Spatial_Symmetric_Gaussian):
    """New-style via **kwargs: must be detected and receive the Generator."""

    def __init__(self, prior):
        super().__init__(prior)
        self.saw_generator = False

    def simulate_trigger(self, pars, **kwargs):
        rng = kwargs.get("rng")
        if isinstance(rng, np.random.Generator):
            self.saw_generator = True
        gen = rng if rng is not None else np.random
        return gen.normal(scale=pars['sigmax_2'] ** 0.5, size=2)


def _model(spatial_cls):
    return Hawkes_Model(DATA, A_GDF, T_DAYS, cox_background=False,
                        excitation_support="rectangle",
                        spatial_trig=spatial_cls, **PRIORS)


def test_accepts_rng_kwarg_classification():
    """The detector. New-style: keyword-passable rng or **kwargs -- including
    BOTH in-repo trigger classes (Temporal_Power_Law was revised to rng=None
    in the RNG-unification commit; a detector that classified it old-style
    would silently drop it from the Generator stream). Old-style: (pars)-only,
    POSITIONAL_ONLY rng, VAR_POSITIONAL *rng (rng=... is a TypeError for all
    three), and uninspectable C callables (legacy form fallback)."""
    assert accepts_rng_kwarg(Temporal_Exponential(PRIORS).simulate_trigger)
    assert accepts_rng_kwarg(Temporal_Power_Law(PRIORS).simulate_trigger)
    assert accepts_rng_kwarg(Spatial_Symmetric_Gaussian(PRIORS).simulate_trigger)
    assert accepts_rng_kwarg(_BuggyNewStyleSpatial(PRIORS).simulate_trigger)
    assert accepts_rng_kwarg(_VarKwargsSpatial(PRIORS).simulate_trigger)
    assert not accepts_rng_kwarg(_OldStyleSpatial(PRIORS).simulate_trigger)
    assert not accepts_rng_kwarg(
        _PositionalOnlyRngSpatial(PRIORS).simulate_trigger)
    assert not accepts_rng_kwarg(
        _VarPositionalRngSpatial(PRIORS).simulate_trigger)
    assert not accepts_rng_kwarg(np.random.default_rng(0).normal)


def test_new_style_trigger_internal_typeerror_surfaces():
    """REGRESSION (masked bug): a TypeError raised INSIDE a new-style
    trigger must surface to the caller. RED pre-fix: the broad fallback
    caught it, re-executed the trigger without rng, and _sim_offspring
    completed -- silently off the Generator."""
    m = _model(_BuggyNewStyleSpatial)
    with pytest.raises(TypeError, match="internal bug on the Generator path"):
        m._sim_offspring(BG.copy(), dict(PAR), rng=np.random.default_rng(0))


def test_old_signature_trigger_still_supported():
    """The compatibility promise: (pars)-only triggers keep working under a
    Generator-driven simulation, staying on np.random exactly as documented
    (the legacy third-party shape)."""
    m = _model(_OldStyleSpatial)
    np.random.seed(0)
    off = m._sim_offspring(BG.copy(), dict(PAR), rng=np.random.default_rng(0))
    assert off.ndim == 2 and off.shape[1] == 3


def test_positional_only_rng_classified_old_style():
    """The kind rule, live: a POSITIONAL_ONLY 'rng' trigger must run through
    the legacy call form -- routing it new-style would raise
    TypeError('positional-only ... passed as keyword') on the first draw."""
    m = _model(_PositionalOnlyRngSpatial)
    np.random.seed(0)
    off = m._sim_offspring(BG.copy(), dict(PAR), rng=np.random.default_rng(0))
    assert off.ndim == 2 and off.shape[1] == 3


def test_var_kwargs_trigger_receives_rng():
    """**kwargs triggers are new-style: they must receive the caller's
    Generator (VAR_KEYWORD branch of the detector)."""
    m = _model(_VarKwargsSpatial)
    trig = m.args['sp_trig']
    off = m._sim_offspring(BG.copy(), dict(PAR), rng=np.random.default_rng(0))
    assert off.ndim == 2 and off.shape[1] == 3
    assert trig.saw_generator, "**kwargs trigger never received the Generator"
