"""Refinement of the interval that ends where the bounce stops solving.

The coldest end of a traced phase pair is a boundary, not a sample: below it the symmetric
phase is gone and the bounce has nothing to tunnel out of, so the action is not a number
there. The nucleation rate is still rising as that boundary is approached, which means the
support points that decide the percolation integral are the ones closest to it. No other part
of the controller places them: the step-size test compares two rates and an interval with a
rate at one end only never registers as unresolved.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import percolation_adaptive_gridbuilders as gridbuilders
from transitionlistener import percolation_adaptivestepsize as pas


# The measured interval of the dark U(1) point that failed on figure 13's grid:
# l = 1e-4, v_GeV = 1.2689610031679234e7, g_tilde = 2.69, benchmark precision. The traced
# phase pair ends at T = 5.553053 in internal units and the rate crosses one at 5.680.
WARM_END = 10.743091
BOUNDARY = 5.553053
RATE_CROSSING = 5.680


def settings(**overrides):
    values = dict(n_action_max=10, n_action_increment=5, maxit=10, f_start=1e-3, f_final=0.99,
                  max_log10_rate_step=12.0)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def boundary_grid(log10_rate, temperatures, *, max_new_points=8):
    """Run the candidate builder on a prescribed rate history."""

    log10_rate = np.asarray(log10_rate, dtype=float)
    temps = np.asarray(temperatures, dtype=float)
    with mock.patch.object(gridbuilders, "_log10_gamma_h4_array", lambda t, a, h: log10_rate):
        return gridbuilders.build_boundary_approach_candidates(
            temps, np.zeros(temps.size), np.ones(temps.size), settings(),
            max_new_points=max_new_points,
        )


class BoundaryApproachGridTests(unittest.TestCase):
    """The placement of the points between the coldest sample and the boundary."""

    def test_the_interval_losing_its_action_is_sampled(self):
        grid = boundary_grid([-2.0, np.nan], [WARM_END, BOUNDARY])
        self.assertIsNotNone(grid)
        self.assertEqual(grid.size, 8)
        self.assertTrue(np.all((grid < WARM_END) & (grid > BOUNDARY)),
                        f"points outside the interval: {grid}")

    def test_the_points_spread_over_the_interval_rather_than_crowding_the_boundary(self):
        grid = boundary_grid([-2.0, np.nan], [WARM_END, BOUNDARY])
        # A ladder that halves the remaining distance to the boundary puts every point
        # within a thousandth of it and samples none of the body of the interval, where
        # the rate crosses one. The geometric spread keeps the samples separated: the
        # widest logarithmic gap between neighbours is no larger than twice the narrowest.
        gaps = np.diff(np.log(grid[::-1]))
        self.assertLess(float(np.max(gaps)) / float(np.min(gaps)), 2.0,
                        f"uneven spacing: {gaps}")

    def test_two_rounds_reach_the_temperature_at_which_the_rate_crosses_one(self):
        # One round of eight points reaches 5.9755, still above the crossing at 5.680;
        # the allowance of two rounds is what carries the support across it.
        first = boundary_grid([-2.0, np.nan], [WARM_END, BOUNDARY])
        self.assertGreater(float(np.min(first)), RATE_CROSSING)
        second = boundary_grid([-2.0, -1.0, np.nan], [WARM_END, float(np.min(first)), BOUNDARY])
        self.assertLess(float(np.min(second)), RATE_CROSSING,
                        f"the second round stops at {np.min(second)}, above the crossing")

    def test_the_interval_with_the_highest_rate_is_served_first(self):
        temps = [10.0, 9.0, 8.0, 7.0]
        grid = boundary_grid([-5.0, np.nan, -2.0, np.nan], temps)
        self.assertTrue(np.all((grid < 8.0) & (grid > 7.0)),
                        f"the colder, livelier boundary was not chosen: {grid}")

    def test_nothing_while_every_sample_carries_an_action(self):
        self.assertIsNone(boundary_grid([-3.0, -2.0, -1.0], [10.0, 8.0, 6.0]))

    def test_nothing_where_the_rate_is_already_dead(self):
        # A cold tail whose rate sits a thousand decades below the peak can no longer
        # change the true vacuum fraction, so refining towards it buys nothing.
        self.assertIsNone(boundary_grid([0.0, -1500.0, np.nan], [10.0, 8.0, 6.0]))

    def test_no_points_without_a_budget(self):
        # Removing the budget guard is an equivalent mutation: a geometric spread asked for
        # two points and stripped of its endpoints is already empty, and an empty grid is
        # reported as no candidate. The test records the behaviour, not a live guard.
        self.assertIsNone(boundary_grid([-2.0, np.nan], [WARM_END, BOUNDARY], max_new_points=0))

    def test_no_points_across_a_vanishing_interval(self):
        self.assertIsNone(boundary_grid([-2.0, np.nan], [BOUNDARY, BOUNDARY * (1.0 - 1e-14)]))

    def test_no_points_below_a_boundary_at_zero_temperature(self):
        self.assertIsNone(boundary_grid([-2.0, np.nan], [WARM_END, 0.0]))


class RateStepHandsOverTheBoundaryIntervalTests(unittest.TestCase):
    """The step-size builder leaves the boundary interval to the builder above."""

    def test_an_interval_without_a_cold_rate_is_not_claimed_as_a_rate_step(self):
        temps = np.array([10.0, 9.0, 8.0])
        log10_rate = np.array([-200.0, -2.0, np.nan])
        with mock.patch.object(gridbuilders, "_log10_gamma_h4_array", lambda t, a, h: log10_rate):
            grid = gridbuilders._build_rate_step_grid(
                temps, np.zeros(3), np.ones(3), settings(), max_new_points=5)
        # 198 decades sit between 10 and 9, so that interval is the rate step; the
        # interval from 9 to 8 has no rate at its cold end and is not a candidate here.
        self.assertIsNotNone(grid)
        self.assertTrue(np.all((grid < 10.0) & (grid > 9.0)), f"wrong interval: {grid}")


class ControllerReachesTheBoundaryBuilderTests(unittest.TestCase):
    """The wiring: the controller asks for a boundary approach once the rate step is served."""

    def test_the_boundary_approach_is_tried_when_no_rate_step_is_left(self):
        T = np.linspace(10.0, 1.0, 6)
        # The true vacuum fraction does not reach f_final on this grid, so the controller
        # looks for something to refine.
        P = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.5])
        calls = []

        def apply_candidate_grid(grid, reason, **kwargs):
            calls.append(reason)
            return reason == "boundary_approach"

        common = dict(stage="test", iterator=1, previous_estimate=None, vw=1.0, hot_bound=20.0,
                      apply_candidate_grid=apply_candidate_grid,
                      promote_cached_action_points=lambda reason: False,
                      promote_hot_head_cached_points=lambda: False)
        with mock.patch.object(pas, "_hot_head_underresolved", return_value=(False, None, None)), \
                mock.patch.object(pas, "_build_rate_step_grid", return_value=None), \
                mock.patch.object(pas, "build_boundary_approach_candidates",
                                  return_value=np.linspace(6.0, 5.6, 8)):
            decision, *_ = pas._dynamiczoomwindow_post_sweep_controller(
                T, np.ones_like(T), P, np.full_like(T, 100.0), settings(), None,
                remaining_budget=0, **common)
        self.assertEqual(decision, "retry")
        self.assertIn("boundary_approach", calls)

    def test_the_controller_gives_up_when_the_boundary_builder_finds_nothing(self):
        T = np.linspace(10.0, 1.0, 6)
        P = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.5])
        common = dict(stage="test", iterator=1, previous_estimate=None, vw=1.0, hot_bound=20.0,
                      apply_candidate_grid=lambda grid, reason, **kwargs: False,
                      promote_cached_action_points=lambda reason: False,
                      promote_hot_head_cached_points=lambda: False)
        with mock.patch.object(pas, "_hot_head_underresolved", return_value=(False, None, None)), \
                mock.patch.object(pas, "_build_rate_step_grid", return_value=None), \
                mock.patch.object(pas, "build_boundary_approach_candidates", return_value=None):
            decision, *_ = pas._dynamiczoomwindow_post_sweep_controller(
                T, np.ones_like(T), P, np.full_like(T, 100.0), settings(), None,
                remaining_budget=0, **common)
        self.assertEqual(decision, "limit")


if __name__ == "__main__":
    unittest.main()
