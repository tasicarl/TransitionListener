"""Regression tests for the fitted action derivative behind (beta/H)_S3."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np
from scipy import interpolate

from transitionlistener import bubbledynamics as bd

T_PERC = 50.0


def _action(T):
    """S3(T) with S3/T = 140 - 3 (T - T_PERC) + 0.2 (T - T_PERC)**2."""
    x = np.asarray(T, dtype=float) - T_PERC
    return T * (140.0 - 3.0 * x + 0.2 * x * x)


# T d(S3/T)/dT at T_PERC for the action above.
BETA_H_EXACT = T_PERC * (-3.0)


def _setup(T_samples, S_samples=None):
    T_samples = np.sort(np.asarray(T_samples, dtype=float))
    if S_samples is None:
        S_samples = _action(T_samples)
    Sint = interpolate.interp1d(T_samples, S_samples, kind="linear")
    pot = types.SimpleNamespace(config=types.SimpleNamespace(percolationConf=types.SimpleNamespace()))
    phase = types.SimpleNamespace(Tmin=1.0, Tmax=200.0)
    return Sint, pot, phase


class FitActionSlopeTests(unittest.TestCase):
    def test_exact_for_a_quadratic(self):
        T = T_PERC * (1.0 + np.linspace(-0.004, 0.004, 11))
        self.assertAlmostEqual(bd._fit_action_slope(T, _action(T), T_PERC, 11), BETA_H_EXACT, places=8)

    def test_uses_only_the_nearest_samples(self):
        near = T_PERC * (1.0 + np.linspace(-0.004, 0.004, 11))
        far = np.array([10.0, 150.0])
        S = np.concatenate([_action(near), [1e6, -1e6]])  # far samples are wildly off
        self.assertAlmostEqual(
            bd._fit_action_slope(np.concatenate([near, far]), S, T_PERC, 11), BETA_H_EXACT, places=8
        )


class CalcBetaHS3Tests(unittest.TestCase):
    def test_dense_noisy_support_is_averaged_out(self):
        # Samples 1e-6 T_PERC apart with an irregular 1e-2 error in S3/T: the
        # slope of an interpolant through them would be off by orders of
        # magnitude, the fit over the nearest samples is not.
        rng = np.random.default_rng(1)
        T = T_PERC * (1.0 + np.concatenate([np.linspace(-0.004, 0.004, 21), [1e-6, -1e-6]]))
        S = _action(T) + T * 1e-2 * rng.standard_normal(T.size)
        Sint, pot, phase = _setup(T, S)
        diag = {}
        beta = bd.calc_betaH_S3(T_PERC, Sint, {}, pot, phase, phase, diagnostics=diag)
        self.assertLess(abs(beta / BETA_H_EXACT - 1.0), 0.05)
        self.assertFalse(diag["fallback"])

    def test_one_sided_support_falls_back_to_fresh_actions(self):
        T = T_PERC * (1.0 - np.linspace(0.001, 0.02, 15))  # all samples colder than T_PERC
        Sint, pot, phase = _setup(T)
        diag = {}
        with mock.patch.object(bd, "calcAction", side_effect=lambda pot, t, *args: float(_action(t))) as calc:
            beta = bd.calc_betaH_S3(T_PERC, Sint, {}, pot, phase, phase, diagnostics=diag)
        self.assertTrue(diag["fallback"])
        self.assertEqual(calc.call_count, 5)
        self.assertAlmostEqual(beta, BETA_H_EXACT, places=6)

    def test_structure_on_the_support_scale_is_flagged(self):
        # A bend in S3/T on one side, between the 7 and the 11 nearest samples,
        # makes the two fits disagree by more than the default tolerance of 3 %.
        T = T_PERC * (1.0 + np.linspace(-0.004, 0.004, 11))
        x = T - T_PERC
        S = T * (140.0 - 3.0 * x + 40.0 * np.maximum(x - 0.12, 0.0))
        Sint, pot, phase = _setup(T, S)
        diag = {}
        bd.calc_betaH_S3(T_PERC, Sint, {}, pot, phase, phase, diagnostics=diag)
        self.assertGreater(diag["check_rel_diff"], 0.03)
        self.assertTrue(diag["fit_unstable"])

    def test_smooth_action_is_not_flagged(self):
        T = T_PERC * (1.0 + np.linspace(-0.004, 0.004, 11))
        Sint, pot, phase = _setup(T)
        diag = {}
        beta = bd.calc_betaH_S3(T_PERC, Sint, {}, pot, phase, phase, diagnostics=diag)
        self.assertAlmostEqual(beta, BETA_H_EXACT, places=6)
        self.assertFalse(diag["fit_unstable"])

    def test_outside_the_phase_overlap_gives_nan(self):
        T = T_PERC * (1.0 + np.linspace(-0.004, 0.004, 11))
        Sint, pot, _ = _setup(T)
        phase = types.SimpleNamespace(Tmin=1.0, Tmax=40.0)
        self.assertTrue(np.isnan(bd.calc_betaH_S3(T_PERC, Sint, {}, pot, phase, phase)))


if __name__ == "__main__":
    unittest.main()
