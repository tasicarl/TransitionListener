import unittest
from unittest import mock
from types import SimpleNamespace

from transitionlistener.bubbledynamics import calcAction


class _DummyPhase:
    def valAt(self, _T):
        return 0.0


class _DummyPotential:
    gradV = object()
    d2V = object()
    Vtot = object()
    conversionFactor = 1.0
    config = SimpleNamespace(tracingConf=SimpleNamespace(tunneling_params={}))


class TestCalcActionCache(unittest.TestCase):
    def test_cached_action_returns_before_minimum_search(self):
        pot = _DummyPotential()
        phase_hi = _DummyPhase()
        phase_lo = _DummyPhase()
        outdict = {1.23: {"action": 42.0}}

        # Both steps the cached path must skip, so that removing the short-circuit fails on
        # one of these assertions rather than on whatever the dummy potential does next.
        # `bounceActionOverImages` is the one `calcAction` calls; patching `bounceAction`
        # asserts nothing, because `calcAction` does not call it on any path.
        with mock.patch("transitionlistener.phases.findLocalMinimum") as find_minimum, \
             mock.patch(
                 "transitionlistener.bubbledynamics.bounceActionOverImages") as bounce_action:
            result = calcAction(pot, 1.23, phase_hi, phase_lo, outdict)

        find_minimum.assert_not_called()
        bounce_action.assert_not_called()
        self.assertEqual(result, 42.0)


if __name__ == "__main__":
    unittest.main()
