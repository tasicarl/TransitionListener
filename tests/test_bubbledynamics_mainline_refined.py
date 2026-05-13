import unittest
import inspect
from unittest import mock
from types import SimpleNamespace

import numpy as np

from transitionlistener import errors
from transitionlistener import bubbledynamics as bd
from transitionlistener import runtime_options
from transitionlistener.percolation_adaptivestepsize import (
    _apply_dynamiczoomwindow_support_update,
)
from transitionlistener.percolation_adaptive_gridbuilders import (
    _build_dynamiczoomwindow_jump_grid,
    _build_dynamiczoomwindow_range_grid,
)


def _mainline_settings() -> SimpleNamespace:
    return SimpleNamespace(
        f_perc=0.28957,
        f_start=1e-3,
        f_final=0.99,
        integral_method="ode",
        time_temperature_mode="bag",
        n_action_min=20,
        n_action_increment=5,
        n_action_max=50,
        max_action_temperatures=100,
        weight=1 / 1.5,
        maxit=10,
        rel_increment=0.1,
        max_boundary_n=10,
        large_delta_p_refine_threshold=0.1,
        large_delta_p_success_threshold=0.2,
        n_jitter_detect_GH4_oom=1.0,
        action_jitter_tunneltight_rescue=False,
        n_jitter_save=20,
        acc_tperc=1e-2,
        acc_tfinal=1e-2,
        acc_rh=1e-2,
    )


class TestMainlineRefinedPercolationHelpers(unittest.TestCase):
    def test_approximation_mode_is_not_configurable(self):
        self.assertNotIn("approx_criterion_mode", inspect.signature(bd.calcApproxPercolation).parameters)
        self.assertNotIn("percolation_approx_criterion_mode", runtime_options.PERCOLATION_OVERRIDE_KEYS)

    def test_approximation_failure_has_no_secondary_fallback(self):
        with mock.patch.object(bd, "_approx_percolation_criterion", return_value=1.0):
            with self.assertRaises(errors.PercolationApproximation1Error):
                bd.calcApproxPercolation(
                    {},
                    object(),
                    object(),
                    object(),
                    1.0,
                    verbose=False,
                    tmin=1.0,
                    tmax=10.0,
                )

    def test_jump_and_range_refinement_stay_incremental(self):
        jump_candidate = _build_dynamiczoomwindow_jump_grid(10.0, 6.0, max_new_points=5)
        self.assertEqual(len(jump_candidate), 5)
        self.assertTrue(np.all(jump_candidate < 10.0))
        self.assertTrue(np.all(jump_candidate > 6.0))

        temperatures = np.array([10.0, 8.0, 6.0, 4.0], dtype=float)
        range_candidate = _build_dynamiczoomwindow_range_grid(
            temperatures,
            new_high=12.0,
            new_low=2.0,
            max_new_points=5,
        )
        self.assertIsNotNone(range_candidate)
        self.assertLessEqual(len(range_candidate), 5)
        self.assertTrue(np.any(range_candidate > 10.0))
        self.assertTrue(np.any(range_candidate < 4.0))

    def test_mainline_support_update_keeps_full_admitted_bank_active(self):
        settings = _mainline_settings()
        settings.n_action_max = 2
        current = np.array([10.0, 8.0], dtype=float)
        free_bank = np.array([10.0, 8.0, 6.0, 4.0], dtype=float)
        candidate = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0], dtype=float)

        active, bank, dropped = _apply_dynamiczoomwindow_support_update(
            current,
            candidate,
            free_bank,
            free_bank,
            settings,
        )

        self.assertEqual(len(bank), 6)
        self.assertIn(9.0, bank)
        self.assertIn(7.0, bank)
        self.assertNotIn(5.0, bank)
        self.assertNotIn(3.0, bank)
        self.assertGreater(len(dropped), 0)
        np.testing.assert_allclose(active, np.array([10.0, 9.0, 8.0, 7.0]))


if __name__ == "__main__":
    unittest.main()
