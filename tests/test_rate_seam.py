"""Tests for the nucleation-rate seam (work package 4).

The seam exists so that a three-dimensional rate -- which depends on the bounce
profile and the model's matching, neither of which is visible inside the
percolation integrands -- can be computed at the one place where both are
available. The acceptance condition for the refactor is that with the default
rate_mode nothing whatsoever changes, so most of what is tested here is that
the seam is *exactly* the historical expression.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models")
)

from transitionlistener import bubbledynamics as bd


def _potential():
    """The conformal U(1) with fermions at a benchmark point."""
    import TL_conformal_u1_fermions as mod
    return mod.specific_potential({"g": 0.9, "y": 0.3, "v_GeV": 1.0}, verbose=False)


# --------------------------------------------------- the default is unchanged
@pytest.mark.parametrize("T,S", [(0.2, 150.0), (1.0, 300.0), (12.5, 1e4),
                                 (0.05, 42.0), (3.3, 1.0)])
def test_seam_reproduces_the_historical_expression_exactly(T, S):
    pot = _potential()
    assert pot.logGammaNuc(T, S) == float(np.atleast_1d(bd.logGamma(T, S))[0])


def test_seam_matches_the_closed_form():
    """4 ln T + (3/2) ln(S/2 pi T) - S/T, written out independently."""
    pot = _potential()
    T, S = 0.37, 210.0
    expected = 4 * math.log(T) + 1.5 * math.log(S / (2 * math.pi * T)) - S / T
    assert pot.logGammaNuc(T, S) == pytest.approx(expected, rel=1e-12)


def test_seam_ignores_the_instanton_in_the_default_mode():
    """The profile is accepted but unused, so passing one changes nothing."""
    pot = _potential()
    T, S = 0.5, 120.0
    assert pot.logGammaNuc(T, S) == pot.logGammaNuc(T, S, instanton="anything")


def test_zero_action_gives_an_infinite_rate():
    pot = _potential()
    assert math.isinf(pot.logGammaNuc(1.0, 0.0))


def test_returns_a_python_float_not_an_array():
    """Downstream code stores this in a dict and splines it."""
    pot = _potential()
    assert isinstance(pot.logGammaNuc(1.0, 100.0), float)


# ------------------------------------------------------------- the switch
def test_model_mode_refuses_when_the_model_has_not_overridden_it():
    pot = _potential()
    pot.config.nucleationConf.rate_mode = "model"
    with pytest.raises(NotImplementedError, match="does not override"):
        pot.logGammaNuc(1.0, 100.0)


def test_unknown_rate_mode_raises():
    pot = _potential()
    pot.config.nucleationConf.rate_mode = "something_else"
    with pytest.raises(ValueError, match="unknown"):
        pot.logGammaNuc(1.0, 100.0)


def test_a_model_can_override_the_seam():
    """What the 3D mode will do: supply its own rate, using the profile."""
    pot = _potential()
    seen = {}

    def fake_rate(T, S, instanton=None):
        seen["instanton"] = instanton
        return -1.234

    pot.logGammaNuc = fake_rate
    assert pot.logGammaNuc(1.0, 100.0, instanton="profile") == -1.234
    assert seen["instanton"] == "profile"


# ------------------------------------------- the seam is actually populated
def test_the_seam_runs_inside_evalNucleationCriterion():
    """The seam must live on the code path that actually executes.

    Rather than re-implementing what evalNucleationCriterion does (which would
    test nothing), this calls the real method and checks that the rate was
    computed through the potential and stored. logGammaNuc is replaced by a
    recording stub so the test does not depend on the bounce solver succeeding.
    """
    from transitionlistener import transitions as tr

    pot = _potential()
    finder = tr.Transitions.__new__(tr.Transitions)   # skip __init__: needs phases
    finder.pot = pot
    finder.conversionFactor = pot.conversionFactor

    calls = []

    def recording_rate(T, S, instanton=None):
        calls.append((T, S, instanton))
        return -42.0

    pot.logGammaNuc = recording_rate

    # A cached entry makes evalNucleationCriterion return before it reaches the
    # seam, so seed an *empty* outdict and let it run the real path. The bounce
    # solver is bypassed by pre-populating the entry bounceAction would create.
    T, S = 0.5, 137.0

    class _Phase:
        key, Tmin, Tmax = "p", 0.0, 1e9
        def valAt(self, _T):
            return np.array([pot.v_stable])

    outdict = {}

    def fake_bounce(T_, V_, dV_, outdict_, tdict_, **kw):
        tdict_["action"] = S
        tdict_["instanton"] = "the-profile"
        outdict_[T_] = tdict_
        return outdict_

    tr.pathDeformation.bounceAction, saved = fake_bounce, tr.pathDeformation.bounceAction
    try:
        finder.evalNucleationCriterion(
            T, _Phase(), _Phase(),
            V=lambda x, t: np.array([0.0]),
            dV=lambda x, t: np.array([0.0]),
            phitol=1e-8, overlapAngle=45.0,
            nuclCriterion=lambda S_, T_, a, b: 0.0,
            fullTunneling_params={}, verbose=False, outdict=outdict,
        )
    finally:
        tr.pathDeformation.bounceAction = saved

    assert calls, "logGammaNuc was never called: the seam is not on the live path"
    assert calls[0][0] == T
    assert calls[0][1] == S
    assert calls[0][2] == "the-profile", "the bounce profile must reach the seam"
    assert outdict[T]["log_gamma"] == -42.0


# ------------------------------------------------ the array-level helpers
def test_rate_arrays_are_bit_for_bit_the_historical_functions_by_default():
    """The acceptance condition for the refactor: identical, not merely close."""
    pot = _potential()
    T = np.array([0.1, 0.3, 0.7, 1.5, 4.0])
    S = np.array([50.0, 120.0, 200.0, 340.0, 900.0])

    gamma, log_gamma = bd.rateArrays(pot, T, S)
    assert np.array_equal(gamma, np.asarray(bd.Gamma(T, S), dtype=float))
    assert np.array_equal(log_gamma, np.asarray(bd.logGamma(T, S), dtype=float))


def test_model_mode_needs_the_tunnelling_dictionary():
    """Because the rate then depends on more than (T, S), it must say so."""
    pot = _potential()
    pot.config.nucleationConf.rate_mode = "model"
    with pytest.raises(ValueError, match="more than"):
        bd.rateArrays(pot, np.array([1.0]), np.array([100.0]))


def test_model_mode_routes_through_the_model_and_exponentiates():
    pot = _potential()
    pot.config.nucleationConf.rate_mode = "model"
    pot.logGammaNuc = lambda T, S, instanton=None: -3.0

    outdict = {1.0: {"action": 100.0, "instanton": None},
               2.0: {"action": 200.0, "instanton": None}}
    gamma, log_gamma = bd.rateArrays(pot, np.array([1.0, 2.0]),
                                     np.array([100.0, 200.0]), outdict)
    assert np.allclose(log_gamma, -3.0)
    assert np.allclose(gamma, math.exp(-3.0))


def test_calcLogGamma_caches_into_the_dictionary():
    """The adaptive machinery queries the same temperature repeatedly."""
    pot = _potential()
    calls = []
    pot.logGammaNuc = lambda T, S, instanton=None: (calls.append(T), -1.0)[1]

    outdict = {0.5: {"action": 137.0, "instanton": "p"}}
    first = bd.calcLogGamma(pot, 0.5, outdict)
    second = bd.calcLogGamma(pot, 0.5, outdict)
    assert first == second == -1.0
    assert outdict[0.5]["log_gamma"] == -1.0
    assert len(calls) == 1, "the rate was recomputed instead of being cached"


def test_calcLogGamma_passes_the_profile_through():
    pot = _potential()
    seen = {}
    pot.logGammaNuc = lambda T, S, instanton=None: seen.setdefault("i", instanton) and 0.0 or 0.0

    outdict = {0.5: {"action": 137.0, "instanton": "the-profile"}}
    bd.calcLogGamma(pot, 0.5, outdict)
    assert seen["i"] == "the-profile"
