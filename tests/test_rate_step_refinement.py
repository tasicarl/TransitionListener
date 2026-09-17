"""Bookkeeping of the support points that resolve the rise of the nucleation rate."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

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
