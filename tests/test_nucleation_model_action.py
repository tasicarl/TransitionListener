"""The nucleation search asks the model for its action (generic_potential.actionNuc).

The search solves the four-dimensional bounce. A model with its own action, such as a
three-dimensional rate, must be asked for it there too: the nucleation criterion has to be
evaluated on the model's action, and the actions cached by the search, which the percolation
reuses as support points, have to be the model's. Before this was routed through
Transitions, a three-dimensional run found a four-dimensional nucleation temperature and
percolated on a support that mixed four- and three-dimensional actions.
"""
import numpy as np

from transitionlistener.transitions import Transitions, _apply_model_action


class _ShiftedAction:
    """A stand-in model whose action is the solver's plus a fixed shift."""

    def __init__(self, shift):
        self.shift = shift
        self.calls = 0

    def actionNuc(self, T, action, instanton=None):
        self.calls += 1
        return action + self.shift


def _bare_transitions(pot):
    tr = Transitions.__new__(Transitions)   # the helpers only need the model and the cache
    tr.pot = pot
    tr._model_actions = {}
    return tr


def test_the_criterion_is_evaluated_on_the_model_action():
    tr = _bare_transitions(_ShiftedAction(10.0))
    criterion = tr._model_criterion(lambda S, T, *args: 140.0 - S / T)
    assert criterion(100.0, 1.0, "start", "end") == 140.0 - 110.0
    # an infinite solver action (no tunnelling) is passed through untouched
    assert criterion(np.inf, 1.0) == -np.inf


def test_cached_actions_become_the_model_action_and_keep_the_solver_value():
    pot = _ShiftedAction(10.0)
    tr = _bare_transitions(pot)
    tunn = {"Tnuc": 2.0, "action": 50.0,
            "full_tunneling_info": {1.0: {"action": 40.0, "log_gamma": -3.0},
                                    2.0: {"action": 50.0},
                                    3.0: {"action": np.inf}}}
    _apply_model_action(tr, tunn)
    info = tunn["full_tunneling_info"]
    assert info[1.0]["action"] == 50.0 and info[1.0]["action_4d"] == 40.0
    assert "log_gamma" not in info[1.0]          # a rate cached from the solver's action is dropped
    assert info[2.0]["action"] == 60.0 and tunn["action"] == 60.0
    assert info[3.0]["action"] == np.inf and "action_4d" not in info[3.0]
    # applying it twice changes nothing, and the model is asked once per (T, S)
    calls = pot.calls
    _apply_model_action(tr, tunn)
    assert info[1.0]["action"] == 50.0 and pot.calls == calls


def test_a_model_without_its_own_action_is_unchanged():
    class _Plain:
        def actionNuc(self, T, action, instanton=None):
            return action
    tr = _bare_transitions(_Plain())
    tunn = {"Tnuc": 1.0, "action": 40.0, "full_tunneling_info": {1.0: {"action": 40.0}}}
    _apply_model_action(tr, tunn)
    assert tunn["action"] == 40.0 and tunn["full_tunneling_info"][1.0]["action"] == 40.0


def test_a_search_driven_on_a_bare_namespace_keeps_the_solver_actions():
    """The transition searches are also driven on a namespace holding only what they read, and
    such an object has no model to ask. The solver's values stay, with nothing written beside them."""
    import types

    entry = {"action": 100.0, "log_gamma": -3.0}
    tunn = {"Tnuc": 2.0, "action": 100.0, "full_tunneling_info": {2.0: entry}}
    assert _apply_model_action(types.SimpleNamespace(), tunn) is tunn
    assert entry == {"action": 100.0, "log_gamma": -3.0}
