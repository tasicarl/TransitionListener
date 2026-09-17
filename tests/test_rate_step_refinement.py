"""Bookkeeping of the support points that resolve the rise of the nucleation rate."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import percolation_adaptive_gridbuilders as gridbuilders
from transitionlistener import percolation_adaptivestepsize as pas


def settings(**overrides):
    values = dict(n_action_max=10, n_action_increment=5, maxit=10, f_start=1e-3, f_final=0.99,
                  max_log10_rate_step=12.0)
    values.update(overrides)
    return types.SimpleNamespace(**values)


class RateRepairAccountingTests(unittest.TestCase):
    def test_every_batch_of_new_points_is_counted(self):
        active = np.linspace(10.0, 1.0, 5)
        previous_batch = np.linspace(9.95, 1.05, 8)
        new_batch = np.linspace(9.9, 1.1, 6)
        bank_before = np.concatenate((active, previous_batch))
        bank_after = np.concatenate((bank_before, new_batch))
        # The update adds the actions of the previous batch to the free bank before it
        # admits the new one; measured against the free bank before the update, the
        # previous batch would cancel the new one.
        free_after = bank_before
        self.assertEqual(pas._new_controller_added_points(bank_before, bank_after, free_after), 6)
        self.assertEqual(
            pas._controller_added_support_points(bank_after, free_after)
            - pas._controller_added_support_points(bank_before, active),
            -2,
        )

    def test_points_with_a_cached_action_are_not_counted(self):
        active = np.linspace(10.0, 1.0, 5)
        cached = np.linspace(9.9, 1.1, 6)
        bank_after = np.concatenate((active, cached))
        free_after = np.concatenate((active, cached))
        self.assertEqual(pas._new_controller_added_points(active, bank_after, free_after), 0)


class RateStepGridTests(unittest.TestCase):
    """The builder that fills the intervals across which log10(Gamma/H^4) jumps."""

    @staticmethod
    def rate_step_grid(log10_rate, *, threshold=12.0, max_new_points=60, temperatures=None):
        log10_rate = np.asarray(log10_rate, dtype=float)
        temps = np.linspace(10.0, 1.0, log10_rate.size) if temperatures is None else np.asarray(temperatures, dtype=float)
        with mock.patch.object(gridbuilders, "_log10_gamma_h4_array", lambda t, a, h: log10_rate):
            return gridbuilders._build_rate_step_grid(
                temps, np.zeros(temps.size), np.ones(temps.size),
                settings(max_log10_rate_step=threshold), max_new_points=max_new_points,
            )

    def test_no_points_while_every_step_stays_below_the_threshold(self):
        self.assertIsNone(self.rate_step_grid([0.0, 6.0, 18.0, 30.0]))

    def test_an_offending_interval_gets_enough_points_to_resolve_it(self):
        # 40 decades across one interval: 4 steps of 10 decades need 3 new points.
        temps = np.array([10.0, 9.0, 8.0])
        grid = self.rate_step_grid([0.0, 5.0, 45.0], temperatures=temps)
        self.assertEqual(grid.size, 3)
        self.assertTrue(np.all((grid < 9.0) & (grid > 8.0)))

    def test_the_worst_interval_is_served_first_within_the_budget(self):
        temps = np.array([10.0, 9.0, 8.0, 7.0])
        # 120 decades between 9 and 8, 36 decades between 8 and 7, budget for 5 points.
        grid = self.rate_step_grid([0.0, 0.0, 120.0, 156.0], temperatures=temps, max_new_points=5)
        self.assertEqual(grid.size, 5)
        self.assertTrue(np.all((grid < 9.0) & (grid > 8.0)))

    def test_no_points_without_a_threshold(self):
        self.assertIsNone(self.rate_step_grid([0.0, 100.0], threshold=0.0))

    def test_no_points_without_two_finite_samples(self):
        self.assertIsNone(self.rate_step_grid([np.nan, 100.0]))
        self.assertIsNone(self.rate_step_grid([50.0]))


class ControllerTests(unittest.TestCase):
    def test_incomplete_profile_without_budget_still_gets_the_rate_repair(self):
        # The rate repair has its own allowance, so an exhausted ordinary budget must not stop it.
        T = np.linspace(10.0, 1.0, 6)
        P = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.5])  # the transition does not complete on this grid
        calls = []

        def apply_candidate_grid(grid, reason, **kwargs):
            calls.append(reason)
            return reason == "rate_step_refine"

        common = dict(stage="test", iterator=1, previous_estimate=None, vw=1.0, hot_bound=20.0,
                      apply_candidate_grid=apply_candidate_grid,
                      promote_cached_action_points=lambda reason: False,
                      promote_hot_head_cached_points=lambda: False)
        with mock.patch.object(pas, "_hot_head_underresolved", return_value=(False, None, None)), \
                mock.patch.object(pas, "_build_rate_step_grid", return_value=np.linspace(10.0, 1.0, 9)):
            decision, *_ = pas._dynamiczoomwindow_post_sweep_controller(
                T, np.ones_like(T), P, np.full_like(T, 100.0), settings(), None, remaining_budget=0, **common)
        self.assertEqual(decision, "retry")
        self.assertEqual(calls, ["rate_step_refine"])

        calls.clear()
        with mock.patch.object(pas, "_hot_head_underresolved", return_value=(False, None, None)), \
                mock.patch.object(pas, "_build_rate_step_grid", return_value=None):
            decision, *_ = pas._dynamiczoomwindow_post_sweep_controller(
                T, np.ones_like(T), P, np.full_like(T, 100.0), settings(), None, remaining_budget=0, **common)
        self.assertEqual(decision, "limit")


if __name__ == "__main__":
    unittest.main()
