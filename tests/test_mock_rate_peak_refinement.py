"""Cheap regression tests for DZW rate-peak support placement.

These tests deliberately avoid real bounce evaluations.  They monkeypatch the
controller-level ``log10(Gamma/H^4)`` diagnostic with a tunable mock peak, then
exercise the support-placement helper that decides where the next expensive
action calls should be made.

The target failure mode is the 2HDM strongly-supercooled case where the active
grid had a huge interval on the hot-to-peak ramp, e.g. T~61 -> 31, while the
rate climbed by many decades.  The controller must bridge that interval before
it is allowed to spend more points in the cold tail.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transitionlistener import percolation_adaptive_rate as dzw_rate  # noqa: E402
from transitionlistener import percolation_adaptive_gridbuilders as dzw_gridbuilders  # noqa: E402
from transitionlistener.bubbledynamics import _temperature_grid as _dzw_temperature_grid  # noqa: E402


@dataclass
class MockSettings:
    large_delta_p_refine_threshold: float = 0.1
    large_delta_p_success_threshold: float = 0.2
    n_action_increment: int = 5
    f_perc: float = 0.71
    f_final: float = 0.99
    f_start: float = 1.0e-3
    refine_low: float = 1.0e-2


class BrokenPowerLawRate:
    """Asymmetric peak in log10(Gamma/H^4) as a function of log(T)."""

    def __init__(
        self,
        *,
        peak_temperature: float,
        peak_log10_rate: float,
        hot_width: float,
        cold_width: float,
    ) -> None:
        self.peak_temperature = float(peak_temperature)
        self.peak_log10_rate = float(peak_log10_rate)
        self.hot_width = float(hot_width)
        self.cold_width = float(cold_width)

    def __call__(self, temperature: float) -> float:
        x = math.log(float(temperature) / self.peak_temperature)
        width = self.hot_width if x >= 0.0 else self.cold_width
        return float(self.peak_log10_rate - (x / width) ** 2)


class RatePeakRefinementMockTests(unittest.TestCase):
    def _run_mock_controller(
        self,
        shape: BrokenPowerLawRate,
        initial_grid: list[float],
        *,
        iterations: int = 4,
        max_new_points: int = 5,
    ) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
        original = dzw_rate._log10_gamma_h4_array
        dzw_rate._log10_gamma_h4_array = lambda temperatures, actions, hubble: np.asarray(
            [shape(float(temperature)) for temperature in np.asarray(temperatures, dtype=float)],
            dtype=float,
        )
        try:
            settings = MockSettings()
            grid = _dzw_temperature_grid(np.asarray(initial_grid, dtype=float))
            batches: list[np.ndarray] = []
            reasons: list[str] = []
            for _ in range(iterations):
                actions = np.full_like(grid, 1.0, dtype=float)
                hubble = np.ones_like(grid, dtype=float)
                candidate, reason = dzw_rate._build_dynamiczoomwindow_rate_peak_grid(
                    grid,
                    actions,
                    hubble,
                    settings,
                    max_new_points=max_new_points,
                )
                if candidate is None or candidate.size == 0:
                    break
                batches.append(np.asarray(candidate, dtype=float))
                reasons.append("" if reason is None else str(reason))
                for point in candidate:
                    grid = _dzw_temperature_grid(np.append(grid, float(point)))
            return grid, batches, reasons
        finally:
            dzw_rate._log10_gamma_h4_array = original

    def test_steep_onramp_is_bridged_before_cold_tail(self) -> None:
        shape = BrokenPowerLawRate(
            peak_temperature=36.4,
            peak_log10_rate=2.3,
            hot_width=0.08,
            cold_width=0.04,
        )
        initial = [91.6, 76.3, 61.6, 31.7, 16.3, 8.4]
        grid, batches, reasons = self._run_mock_controller(shape, initial, iterations=1)

        self.assertTrue(batches, "the unresolved rate on-ramp should request support")
        self.assertIn("on-ramp", reasons[0])
        self.assertTrue(np.any((batches[0] > 31.7) & (batches[0] < 61.6)))
        self.assertFalse(np.any(batches[0] < 31.7), "do not spend the next batch in the cold tail")
        sampled_rates = np.asarray([shape(value) for value in grid], dtype=float)
        self.assertGreater(float(np.max(sampled_rates)), -5.0)

    def test_iterated_mock_finds_narrow_peak_band(self) -> None:
        shape = BrokenPowerLawRate(
            peak_temperature=33.8,
            peak_log10_rate=4.3,
            hot_width=0.08,
            cold_width=0.04,
        )
        initial = [91.6, 76.3, 62.0, 55.0, 31.6, 16.3, 8.4]
        grid, batches, reasons = self._run_mock_controller(shape, initial, iterations=4)

        self.assertGreaterEqual(len(batches), 1)
        self.assertTrue(any("on-ramp" in reason or "rate_peak_refine around" in reason for reason in reasons))
        sampled_rates = np.asarray([shape(value) for value in grid], dtype=float)
        best_index = int(np.argmax(sampled_rates))
        self.assertGreater(float(sampled_rates[best_index]), 0.0)
        self.assertLess(abs(math.log(float(grid[best_index]) / shape.peak_temperature)), 0.12)

    def test_cold_subcritical_tail_jump_is_not_refined_after_peak(self) -> None:
        shape = BrokenPowerLawRate(
            peak_temperature=36.4,
            peak_log10_rate=4.0,
            hot_width=0.08,
            cold_width=0.035,
        )
        original = dzw_rate._log10_gamma_h4_array
        dzw_rate._log10_gamma_h4_array = lambda temperatures, actions, hubble: np.asarray(
            [shape(float(temperature)) for temperature in np.asarray(temperatures, dtype=float)],
            dtype=float,
        )
        try:
            settings = MockSettings()
            temperatures = np.asarray([55.0, 44.0, 36.4, 28.0, 16.0, 8.0], dtype=float)
            probabilities = np.asarray([0.0, 0.01, 0.2, 0.4, 0.95, 0.995], dtype=float)
            actions = np.ones_like(temperatures)
            hubble = np.ones_like(temperatures)
            candidate = dzw_gridbuilders._build_dynamiczoomwindow_unresolved_jump_grid(
                temperatures,
                probabilities,
                settings,
                max_new_points=5,
                actions=actions,
                hubble=hubble,
            )
            self.assertIsNotNone(candidate)
            self.assertGreaterEqual(
                float(np.min(candidate)),
                float(shape.peak_temperature),
                "controller should refine the hot on-ramp, not the inert cold completion tail",
            )
        finally:
            dzw_rate._log10_gamma_h4_array = original

    def test_near_peak_probability_jump_is_still_refined(self) -> None:
        shape = BrokenPowerLawRate(
            peak_temperature=36.4,
            peak_log10_rate=4.0,
            hot_width=0.08,
            cold_width=0.045,
        )
        original = dzw_rate._log10_gamma_h4_array
        dzw_rate._log10_gamma_h4_array = lambda temperatures, actions, hubble: np.asarray(
            [shape(float(temperature)) for temperature in np.asarray(temperatures, dtype=float)],
            dtype=float,
        )
        try:
            settings = MockSettings()
            temperatures = np.asarray([55.0, 44.0, 36.4, 31.0, 16.0, 8.0], dtype=float)
            probabilities = np.asarray([0.0, 0.01, 0.05, 0.95, 0.98, 0.995], dtype=float)
            actions = np.ones_like(temperatures)
            hubble = np.ones_like(temperatures)
            candidate = dzw_gridbuilders._build_dynamiczoomwindow_unresolved_jump_grid(
                temperatures,
                probabilities,
                settings,
                max_new_points=5,
                actions=actions,
                hubble=hubble,
            )
            self.assertIsNotNone(candidate, "near-peak jump still carries physical information")
            self.assertTrue(np.any((candidate > 31.0) & (candidate < 36.4)))
        finally:
            dzw_rate._log10_gamma_h4_array = original

    def test_broad_peak_dense_grid_can_still_cut_inert_tail(self) -> None:
        shape = BrokenPowerLawRate(
            peak_temperature=36.4,
            peak_log10_rate=2.0,
            hot_width=0.35,
            cold_width=0.08,
        )
        original = dzw_rate._log10_gamma_h4_array
        dzw_rate._log10_gamma_h4_array = lambda temperatures, actions, hubble: np.asarray(
            [shape(float(temperature)) for temperature in np.asarray(temperatures, dtype=float)],
            dtype=float,
        )
        try:
            settings = MockSettings()
            temperatures = np.asarray([44.0, 40.0, 38.0, 36.4, 34.0, 28.0, 20.0, 12.0], dtype=float)
            probabilities = np.asarray([0.0, 0.03, 0.08, 0.2, 0.35, 0.5, 0.55, 0.995], dtype=float)
            actions = np.ones_like(temperatures)
            hubble = np.ones_like(temperatures)
            # The immediate neighbours of the broad peak differ by much less
            # than two decades.  The old prominence gate would fail here; the
            # monotone-tail criterion should still classify the cold tail as
            # inert once it is deeply subcritical.
            self.assertLess(shape(36.4) - max(shape(38.0), shape(34.0)), 2.0)
            candidate = dzw_gridbuilders._build_dynamiczoomwindow_unresolved_jump_grid(
                temperatures,
                probabilities,
                settings,
                max_new_points=5,
                actions=actions,
                hubble=hubble,
            )
            self.assertIsNotNone(candidate)
            self.assertGreaterEqual(
                float(np.min(candidate)),
                float(shape.peak_temperature),
                "controller should refine the unresolved peak/on-ramp before the inert cold tail",
            )
            contribution_limited_candidate = dzw_gridbuilders._build_dynamiczoomwindow_unresolved_jump_grid(
                temperatures,
                probabilities,
                settings,
                max_new_points=5,
                actions=actions,
                hubble=hubble,
                vw=1.0e30,
            )
            self.assertIsNotNone(
                contribution_limited_candidate,
                "the inert-tail gate must not fire when the interval dI bound is large",
            )
        finally:
            dzw_rate._log10_gamma_h4_array = original

    def test_unresolved_jump_refinement_targets_blocker_interval_first(self) -> None:
        settings = MockSettings()
        temperatures = np.asarray([45.0, 40.0, 31.7074, 29.8844, 25.0], dtype=float)
        probabilities = np.asarray([0.0, 0.02, 0.094, 0.172, 0.3], dtype=float)
        candidate = dzw_gridbuilders._build_dynamiczoomwindow_unresolved_jump_grid(
            temperatures,
            probabilities,
            settings,
            max_new_points=5,
        )
        self.assertIsNotNone(candidate)
        self.assertTrue(np.any((candidate > 25.0) & (candidate < 29.8844)))
        self.assertLessEqual(len(candidate), 5)


if __name__ == "__main__":
    unittest.main()
