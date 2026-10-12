"""The cold walk must stop where the bounce stops being computable.

Two places walk the support grid toward T -> 0, and both size their stride against
``tmin``, the floor of the phase-tracing range, which has no relation to the transition:

* the step-1 scout bisects geometrically between its coldest point and ``tmin``
  (traced: 106.93 GeV -> 3.8e-06 GeV in one step, against a transition at 20-40 GeV),
* the step-2/3 range expansion uses ``rel_increment * log(T / tmin)``
  (traced: [61.642, 62.383] -> [2.2287, 62.383], reproduced to five figures).

Below the transition the symmetric phase is no longer metastable, so every action there
comes back non-finite. Measured on the 2HDM edge line, 38-57% of all support points in
every one of five dumps carried no computable action, with the coldest at T ~ 3.8e-06
GeV. Those evaluations cost a bounce solve each and tell the controller nothing, and
the scout's own loop perpetuates them: non-finite actions are excluded when the
approximate profile is rebuilt, so the window never becomes bracketed, so it extends
again.

The fix is not a gentler stride - that would take more steps and so *more* action
evaluations. It is to stop the walk once the bounce has stopped existing, while leaving
behaviour unchanged anywhere an action can still be computed.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from transitionlistener import percolation_adaptive_gridbuilders as gridbuilders
from transitionlistener import percolation_adaptivestepsize as pas


def settings(**overrides):
    values = dict(
        n_action_max=60, n_action_min=10, n_action_increment=5, maxit=20,
        f_start=1.0e-3, f_final=0.99, rel_increment=0.1, max_log10_rate_step=12.0,
        max_action_temperatures=100, time_temperature_mode="sound_speed",
        integral_method="ode", entropy_definition=None,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


class SettledActionRunTests(unittest.TestCase):
    """The shared rule: a run of non-finite actions at the cold end ends the walk."""

    def test_a_run_of_failures_is_recognised(self):
        actions = [250.0, 210.0, 170.0, np.nan, np.nan, np.nan]
        self.assertTrue(gridbuilders._cold_walk_has_left_the_bounce_region(actions))

    def test_a_single_failure_is_a_hiccup_not_an_edge(self):
        actions = [250.0, 210.0, 170.0, 150.0, 140.0, np.nan]
        self.assertFalse(
            gridbuilders._cold_walk_has_left_the_bounce_region(actions),
            "one non-convergent solve must not end the search: a transition just below "
            "it would be missed",
        )

    def test_nothing_computed_yet_is_not_an_edge(self):
        self.assertFalse(
            gridbuilders._cold_walk_has_left_the_bounce_region([np.nan] * 6),
            "a grid that has not started must not be mistaken for one that ran past the "
            "edge of the bounce region",
        )

    def test_failures_away_from_the_cold_end_do_not_count(self):
        """Only a *trailing* run matters; a hole in the middle is not the edge."""
        actions = [250.0, np.nan, np.nan, np.nan, 150.0, 140.0]
        self.assertFalse(gridbuilders._cold_walk_has_left_the_bounce_region(actions))

    def test_the_run_length_is_the_one_the_constant_names(self):
        """The boundary itself: two trailing failures are tolerated, three are the edge.

        ``_COLD_WALK_MAX_FAILED_ACTIONS`` is the whole criterion, so the step either side
        of it is checked rather than only a single failure against a long run.
        """
        base = [250.0, 210.0, 170.0, 150.0]
        for trailing in range(gridbuilders._COLD_WALK_MAX_FAILED_ACTIONS + 2):
            with self.subTest(trailing=trailing):
                self.assertEqual(
                    gridbuilders._cold_walk_has_left_the_bounce_region(
                        base + [np.nan] * trailing),
                    trailing >= gridbuilders._COLD_WALK_MAX_FAILED_ACTIONS,
                )
        self.assertEqual(gridbuilders._COLD_WALK_MAX_FAILED_ACTIONS, 3,
                         "the tolerated run length moved; a transition lying just below a "
                         "single bad solve has to stay reachable")


# hot -> cold, the order both walks work in
T_OVERSHOT = [122.0, 62.0, 31.6, 16.1, 8.2, 4.1]
P_OVERSHOT = [0.0, 0.1, 0.9, 0.9, 0.9, 0.9]
S_FINITE = [250.0, 210.0, 170.0, 150.0, 140.0, 135.0]


class Step23ColdExpansionTests(unittest.TestCase):
    """Site (b): ``_dynamiczoomwindow_range_targets`` must not push past the edge."""

    def targets(self, actions, probabilities=P_OVERSHOT):
        T = np.asarray(T_OVERSHOT, dtype=float)
        return pas._dynamiczoomwindow_range_targets(
            T,
            np.asarray(probabilities, dtype=float),
            np.asarray(actions, dtype=float),
            np.full(T.size, 1.0e-16),
            settings(),
            target_low=1.0e-13,
            target_high=615.0,
            iterator=1,
            hot_head_needs_support=False,
            maxit_margin=2,
            cold_tail_last_temperature=None,
            cold_tail_last_probability=None,
            cold_tail_last_integral=None,
            cold_tail_last_log_gamma_h4=None,
            cold_tail_post_peak_streak=0,
        )

    def test_the_cold_boundary_is_not_pushed_past_the_bounce_region(self):
        _, low, *_ = self.targets(S_FINITE[:3] + [np.nan, np.nan, np.nan])
        self.assertEqual(
            low, T_OVERSHOT[-1],
            f"the cold boundary was pushed to {low} GeV, below a temperature where the "
            "bounce already fails",
        )

    def test_a_single_failure_still_allows_one_more_step(self):
        _, low, *_ = self.targets(S_FINITE[:5] + [np.nan])
        self.assertLess(low, T_OVERSHOT[-1])

    def test_an_all_finite_tail_keeps_its_historical_behaviour(self):
        _, low, *_, cold_unresolved = self.targets(S_FINITE)
        self.assertLess(low, T_OVERSHOT[-1])
        self.assertTrue(cold_unresolved)

    def test_a_resolved_tail_is_still_left_alone(self):
        _, low, *_ = self.targets(S_FINITE, probabilities=[0.0, 0.1, 0.9, 0.995, 0.999, 0.9999])
        self.assertEqual(low, T_OVERSHOT[-1])


# The step-1 scout is deliberately NOT unit-tested here. A stubbed driver was written
# and then discarded: it passed before the guard existed, because the stub let the hot
# branch consume the extension budget and produced only one cold probe, so the test
# could not fail. A test that cannot fail is worse than no test. What the scout hands on
# to step 2 is covered instead, in tests/test_scout_free_bank_is_usable.py.


if __name__ == "__main__":
    unittest.main()
