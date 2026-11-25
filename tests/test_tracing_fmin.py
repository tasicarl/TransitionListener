"""The compiled Nelder-Mead of the tracer (compiled.onefield.nelder_mead through phases.traceLocalMinimum) and the
kept answers of bubbledynamics._transition_action_outdict_temperatures: the same numbers as without them.
"""
import os
import sys
import types

import numpy as np
import pytest
from scipy import optimize

from transitionlistener import helper_functions as hf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
if MODELS not in sys.path:
    sys.path.insert(0, MODELS)


@pytest.fixture
def pot():
    return hf.load_potential(os.path.join(MODELS, "TL_conformal_u1_fermions_3d.py"), "specific_potential")(
        dict(g=0.6, y=0.4, v_GeV=1.0), verbose=False)


def _tracer_function(pot):
    def findPhases():
        def V_(X, T):
            return pot.Vtot(X, T, include_radiation=False, include_decoupled=False)
        return V_
    return findPhases()


@pytest.mark.parametrize("seed", range(4))
def test_nelder_mead_is_scipys_to_the_last_digit(pot, seed):
    assert pot.current_kernels() is not None
    V = _tracer_function(pot)
    rng = np.random.default_rng(seed)
    vev = float(pot.approxZeroTMin()[0][0])
    declined = 0
    for _ in range(150):
        x = np.array([0.0 if rng.random() < 0.1 else vev * 10 ** rng.uniform(-6, 0.2)])
        T = vev * 10 ** rng.uniform(-6, 0.3)
        xtol = 10 ** rng.uniform(-9, -2)
        fast = pot.current_kernels().minimum(x, T, xtol)
        slow = optimize.fmin(V, x, args=(T,), xtol=xtol, ftol=np.inf, disp=False)
        if fast is None:
            declined += 1
        else:
            assert fast.tobytes() == slow.tobytes()
    assert declined < 75


def test_search_from_the_origin_starts_where_scipy_does(pot):
    """From a field of exactly zero the second vertex is scipy's 0.00025, which shows where the origin is not the
    minimum: cold enough, the search walks away from it."""
    V = _tracer_function(pot)
    vev = float(pot.approxZeroTMin()[0][0])
    moved = 0
    for T in vev * np.geomspace(1e-8, 1e-2, 60):
        x = np.zeros(1)
        fast = pot.current_kernels().minimum(x, T, 1e-7)
        if fast is not None:
            assert fast.tobytes() == optimize.fmin(V, x, args=(T,), xtol=1e-7, ftol=np.inf, disp=False).tobytes()
            moved += fast[0] != 0.0
    assert moved > 0


def test_budget_of_scipy_is_left_to_scipy(pot):
    """A search that runs out of scipy's 200 evaluations is declined, not answered."""
    assert pot.current_kernels().minimum(np.array([1.0]), 1.0, 1e-300) is None


def test_the_tracer_hands_its_function_the_compiled_search(pot, monkeypatch):
    """findPhases gives the tracer a function that can answer in the kernels; a function without it goes to scipy."""
    from transitionlistener import phases
    seen = {}

    class Stop(Exception):
        pass

    def stop(f, *a, **k):
        seen["f"] = f
        raise Stop

    monkeypatch.setattr(phases, "traceMultiMin", stop)
    with pytest.raises(Stop):
        phases.Phases(pot, verbose=False).findPhases()
    V = seen["f"]
    kernels = pot.current_kernels()
    calls = []
    minimum = kernels.minimum
    monkeypatch.setattr(kernels, "minimum", lambda *a: calls.append(a) or minimum(*a))
    x, T = np.array([0.5]), 0.01
    expected = optimize.fmin(V, x, args=(T,), xtol=1e-5, ftol=np.inf, disp=False)
    assert phases.traceLocalMinimum(V, x, T, pot.gradV, pot.d2V, 1e-5).tobytes() == expected.tobytes()
    assert len(calls) == 1


def test_other_functions_go_to_scipy(pot, monkeypatch):
    from transitionlistener import phases
    kernels = pot.current_kernels()
    calls = []
    monkeypatch.setattr(kernels, "minimum", lambda *a: calls.append(a))
    f = lambda X, T: (X[0] - 2.0) ** 2
    x = phases.traceLocalMinimum(f, np.array([1.0]), 0.01, pot.gradV, pot.d2V, 1e-5)
    assert calls == [] and abs(x[0] - 2.0) < 1e-4
    other = _tracer_function(types.SimpleNamespace(Vtot=lambda X, T, **k: (X[0] - 3.0) ** 2))
    x = phases.traceLocalMinimum(other, np.array([1.0]), 0.01, pot.gradV, pot.d2V, 1e-5)
    assert calls == [] and abs(x[0] - 3.0) < 1e-4


class _Phase:
    def __init__(self, value):
        self.value, self.calls = value, 0

    def valAt(self, T):
        self.calls += 1
        return np.array([self.value])


def _entry(high, low, **k):
    return dict(action=100.0, high_vev=np.array([high]), low_vev=np.array([low]), **k)


def test_kept_answers_equal_fresh_ones():
    from transitionlistener.bubbledynamics import _transition_action_outdict_temperatures as scan
    sym, bro = _Phase(0.0), _Phase(5.0)
    outdict = {1.0: _entry(0.0, 5.0), 2.0: _entry(0.0, 50.0), 3.0: _entry(0.0, 5.0)}
    assert list(scan(outdict, sym, bro)) == [3.0, 1.0] or set(scan(outdict, sym, bro)) == {1.0, 3.0}
    solved = bro.calls
    assert set(scan(outdict, sym, bro)) == {1.0, 3.0}
    assert bro.calls == solved                        # neither a match nor a mismatch asks the phases again


def test_a_replaced_entry_is_compared_again():
    from transitionlistener.bubbledynamics import _transition_action_outdict_temperatures as scan
    sym, bro = _Phase(0.0), _Phase(5.0)
    outdict = {1.0: _entry(0.0, 50.0)}
    assert scan(outdict, sym, bro).size == 0
    outdict[1.0] = _entry(0.0, 5.0)
    assert set(scan(outdict, sym, bro)) == {1.0}


@pytest.mark.parametrize("key,start,now,kept", [("low_vev", 50.0, 5.0, True), ("low_vev", 5.0, 50.0, False),
                                                ("high_vev", 9.0, 0.0, True), ("high_vev", 0.0, 9.0, False)])
def test_one_field_of_an_entry_replaced(key, start, now, kept):
    from transitionlistener.bubbledynamics import _transition_action_outdict_temperatures as scan
    sym, bro = _Phase(0.0), _Phase(5.0)
    outdict = {1.0: _entry(0.0, 5.0)}
    outdict[1.0][key] = np.array([start])
    assert (scan(outdict, sym, bro).size == 1) is (not kept)
    outdict[1.0][key] = np.array([now])
    assert (scan(outdict, sym, bro).size == 1) is kept


def test_trace_of_an_image_is_compared_again():
    from transitionlistener.bubbledynamics import _transition_action_outdict_temperatures as scan
    sym, bro = _Phase(0.0), _Phase(5.0)
    outdict = {1.0: _entry(9.0, 5.0, high_vev_as_traced=np.array([9.0]))}
    assert scan(outdict, sym, bro).size == 0
    outdict[1.0]["high_vev_as_traced"] = np.array([0.0])
    assert scan(outdict, sym, bro).size == 1


def test_another_symmetric_phase_is_compared_again():
    from transitionlistener.bubbledynamics import _transition_action_outdict_temperatures as scan
    bro = _Phase(5.0)
    outdict = {1.0: _entry(0.0, 5.0)}
    assert set(scan(outdict, _Phase(0.0), bro)) == {1.0}
    assert scan(outdict, _Phase(9.0), bro).size == 0


def test_a_replaced_trace_is_compared_again():
    from transitionlistener.bubbledynamics import _transition_action_outdict_temperatures as scan
    from transitionlistener.phases import PhaseInfo
    T = np.array([0.5, 1.0, 2.0, 3.0])
    sym = PhaseInfo(0, np.zeros((4, 1)), T, np.zeros((4, 1)))
    bro = PhaseInfo(1, np.full((4, 1), 5.0), T, np.zeros((4, 1)))
    outdict = {1.0: _entry(0.0, 5.0)}
    assert set(scan(outdict, sym, bro)) == {1.0}
    bro.replaceTrace(np.full((4, 1), 50.0), np.zeros((4, 1)))
    assert scan(outdict, sym, bro).size == 0
    sym2 = PhaseInfo(0, np.zeros((4, 1)), T, np.zeros((4, 1)))
    bro2 = PhaseInfo(1, np.full((4, 1), 5.0), T, np.zeros((4, 1)))
    assert set(scan(outdict, sym2, bro2)) == {1.0}
    sym2.replaceTrace(np.full((4, 1), 9.0), np.zeros((4, 1)))
    assert scan(outdict, sym2, bro2).size == 0
