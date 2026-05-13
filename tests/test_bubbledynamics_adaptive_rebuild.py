import unittest
from types import SimpleNamespace

import numpy as np

from transitionlistener.bubbledynamics import (
    PercolationDiagnostics,
    _temperature_grid,
    _transition_action_outdict_temperatures,
)
from transitionlistener.percolation_adaptivestepsize import (
    _apply_dynamiczoomwindow_support_update,
    _controller_added_support_points,
)


def _settings() -> SimpleNamespace:
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


class _Phase:
    def __init__(self, offset: float):
        self.offset = float(offset)

    def valAt(self, temperature: float) -> np.ndarray:
        return np.array([float(temperature) + self.offset], dtype=float)


class TestSupportBankDiagnostics(unittest.TestCase):
    def test_temperature_grid_filters_sorts_and_deduplicates(self):
        grid = _temperature_grid([1.0, np.nan, 3.0, 2.0, 3.0 + 1e-12, np.inf, 1.0])
        np.testing.assert_allclose(grid, np.array([3.0, 2.0, 1.0]))

    def test_controller_budget_counts_only_paid_points(self):
        free_bank = np.array([10.0, 8.0, 6.0, 4.0], dtype=float)
        support_bank = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 4.0], dtype=float)

        count = _controller_added_support_points(support_bank, free_bank)

        self.assertEqual(count, 2)

    def test_support_update_deduplicates_and_respects_budget(self):
        settings = _settings()
        active = np.array([10.0, 8.0], dtype=float)
        support_bank = np.array([10.0, 8.0], dtype=float)
        candidate_grid = np.array([9.0, 9.0, 7.0], dtype=float)

        new_active, new_bank, dropped = _apply_dynamiczoomwindow_support_update(
            active,
            candidate_grid,
            support_bank,
            None,
            settings,
            max_new_points=3,
        )

        np.testing.assert_allclose(new_active, np.array([10.0, 9.0, 8.0]))
        np.testing.assert_allclose(new_bank, np.array([10.0, 9.0, 8.0]))
        np.testing.assert_allclose(dropped, np.array([7.0]))

    def test_transition_action_outdict_temperatures_filters_other_transitions(self):
        phase_symmetric = _Phase(0.0)
        phase_broken = _Phase(10.0)
        outdict = {
            10.0: {"action": 120.0, "high_vev": [10.0], "low_vev": [20.0]},
            8.0: {"action": 110.0, "high_vev": [8.0], "low_vev": [18.0]},
            7.0: {"action": 105.0, "high_vev": [7.0], "low_vev": [999.0]},
            5.0: {"action": np.inf, "high_vev": [5.0], "low_vev": [15.0]},
        }

        temperatures = _transition_action_outdict_temperatures(
            outdict,
            phase_symmetric,
            phase_broken,
            tmin=6.0,
            tmax=10.0,
        )

        np.testing.assert_allclose(temperatures, np.array([10.0, 8.0]))

    def test_metadata_keeps_start_temperature(self):
        metadata = PercolationDiagnostics(start_temperature=10.0)
        self.assertEqual(metadata.start_temperature, 10.0)


if __name__ == "__main__":
    unittest.main()
