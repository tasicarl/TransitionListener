"""Regression tests for the false-vacuum volume criterion (Lewicki criterion)."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener.transitionObservables import TransitionObservables
from transitionlistener.transitionObservables_fixedstep import (
    TransitionObservables as TransitionObservablesFixedStep,
)

# I = -ln(1 - f_perc) at the default percolation fraction f_perc = 0.29.
I_PERC = -np.log1p(-0.29)


def _history(power, amplitude, t_hot, t_cold, n=400):
    """Percolation history I(T) = amplitude * (t_hot / T)**power, ordered hot to cold.

    For this history T dI/dT = -power * I, so the criterion is exactly
    1 - cs_sq * power * I(T).
    """
    temperatures = np.linspace(t_hot, t_cold, n)
    integral = amplitude * (t_hot / temperatures) ** power
    fraction = -np.expm1(-integral)
    t_perc = t_hot * (amplitude / I_PERC) ** (1.0 / power)
    return temperatures, fraction, t_perc


class FalseVacuumVolumeGrowthRateTests(unittest.TestCase):
    def test_steep_history_is_shrinking(self):
        T, P, t_perc = _history(power=20, amplitude=1e-3, t_hot=1.0, t_cold=0.65)
        growth = bd.falseVacuumVolumeGrowthRate(T, P, t_perc)
        self.assertAlmostEqual(growth, 1.0 - 20 * I_PERC / 3.0, delta=1e-3)
        self.assertLess(growth, 0.0)

    def test_shallow_history_is_growing(self):
        T, P, t_perc = _history(power=2, amplitude=1e-2, t_hot=1.0, t_cold=0.1)
        growth = bd.falseVacuumVolumeGrowthRate(T, P, t_perc)
        self.assertAlmostEqual(growth, 1.0 - 2 * I_PERC / 3.0, delta=1e-3)
        self.assertGreater(growth, 0.0)

    def test_sound_speed_enters_linearly(self):
        T, P, t_perc = _history(power=20, amplitude=1e-3, t_hot=1.0, t_cold=0.65)
        growth = bd.falseVacuumVolumeGrowthRate(T, P, t_perc, cs_sq=0.2)
        self.assertAlmostEqual(growth, 1.0 - 0.2 * 20 * I_PERC, delta=1e-3)

    def test_temperature_order_does_not_matter(self):
        T, P, t_perc = _history(power=20, amplitude=1e-3, t_hot=1.0, t_cold=0.65)
        self.assertAlmostEqual(
            bd.falseVacuumVolumeGrowthRate(T, P, t_perc),
            bd.falseVacuumVolumeGrowthRate(T[::-1], P[::-1], t_perc),
            places=12,
        )

    def test_unusable_input_gives_nan(self):
        T, P, t_perc = _history(power=20, amplitude=1e-3, t_hot=1.0, t_cold=0.65)
        cases = {
            "two points": (T[:2], P[:2], float(T[1])),
            "above the history": (T, P, 1.5),
            "below the history": (T, P, 0.5),
            "length mismatch": (T, P[:-1], t_perc),
            "no true vacuum yet": (T, np.zeros_like(P), t_perc),
            "non-finite temperature": (T, P, np.nan),
        }
        for label, (temps, fractions, at) in cases.items():
            with self.subTest(label):
                self.assertTrue(np.isnan(bd.falseVacuumVolumeGrowthRate(temps, fractions, at)))


class PercolationSoundSpeedTests(unittest.TestCase):
    def test_bag_mode_uses_one_third(self):
        # A usable symmetric-phase sound speed is available, but the bag mode
        # integrates with dT/dt = -H T and must ignore it.
        phase = types.SimpleNamespace(valAt=lambda T: np.array([0.0]))
        with mock.patch.object(bd, "calcSoundSpeedSq", return_value=0.21):
            value = bd.percolation_sound_speed_sq(
                object(), phase, 0.5, time_temperature_mode="bag"
            )
        self.assertEqual(value, 1.0 / 3.0)

    def test_eff_potential_scheme_uses_the_potential(self):
        # The sound speed is the logarithmic derivative of the entropy that fixes a(T), taken
        # along the phase, for both schemes alike. So the potential route is identified by the
        # definition its entropy is asked for, not by a separate fixed-field routine:
        # s ~ T^4 must give c_s^2 = 1/4 whatever `calcSoundSpeedSq` would have said.
        phase = types.SimpleNamespace(valAt=lambda T: np.array([0.0]))
        seen = []

        def entropy(pot, ph, T, definition):
            seen.append(definition)
            return float(T) ** 4.0

        with mock.patch.object(bd, "calcSoundSpeedSq", return_value=0.21), \
                mock.patch.object(bd, "entropy_density", entropy):
            value = bd.percolation_sound_speed_sq(
                object(), phase, 0.5, time_temperature_mode="sound_speed",
                entropy_definition="eff_potential",
            )
        self.assertAlmostEqual(value, 0.25)
        self.assertEqual(set(seen), {"eff_potential"})

    def test_dof_table_scheme_uses_the_counted_entropy(self):
        # s ~ T^k gives c_s^2 = 1/k exactly. The potential's sound speed must not enter,
        # so it is mocked to a value the test would notice.
        phase = types.SimpleNamespace(valAt=lambda T: np.array([0.0]))
        for k in (3.0, 4.0, 5.0):
            with self.subTest(k=k):
                with mock.patch.object(bd, "entropy_density",
                                       lambda pot, ph, T, d, k=k: float(T) ** k), \
                     mock.patch.object(bd, "calcSoundSpeedSq", return_value=0.21):
                    value = bd.percolation_sound_speed_sq(
                        object(), phase, 0.5, time_temperature_mode="sound_speed",
                        entropy_definition="dof_table",
                    )
                self.assertAlmostEqual(value, 1.0 / k, places=6)

    def test_unusable_sound_speed_falls_back_to_one_third(self):
        # The potential route is the one that reads calcSoundSpeedSq, so the scheme has to
        # be named: under the default "dof_table" this would reach the bag fallback because
        # the stand-in potential makes the counted entropy raise, not because of the nan.
        # A non-positive entropy has no logarithm, so the stencil yields no sound speed. The
        # potential route reaches this where dV/dT changes sign.
        phase = types.SimpleNamespace(valAt=lambda T: np.array([0.0]))
        with mock.patch.object(bd, "entropy_density", lambda pot, ph, T, d: -1.0):
            value = bd.percolation_sound_speed_sq(
                object(), phase, 0.5, time_temperature_mode="sound_speed",
                entropy_definition="eff_potential",
            )
        self.assertEqual(value, 1.0 / 3.0)

    def test_unusable_counted_entropy_falls_back_to_one_third(self):
        # The counterpart on the "dof_table" route. The entropy has to fail only at T
        # itself: if it failed at the stencil points too, the sound speed would be nan and
        # the other half of the guard would already return 1/3, so the test would pass
        # without exercising the entropy half at all.
        phase = types.SimpleNamespace(valAt=lambda T: np.array([0.0]))

        def entropy(pot, ph, T, d, centre=0.5):
            return np.nan if float(T) == centre else float(T) ** 4

        with mock.patch.object(bd, "entropy_density", entropy):
            value = bd.percolation_sound_speed_sq(
                object(), phase, 0.5, time_temperature_mode="sound_speed",
                entropy_definition="dof_table",
            )
        self.assertEqual(value, 1.0 / 3.0)


class FalseVacuumWarningTests(unittest.TestCase):
    def _run(self, cls, power, amplitude, t_cold):
        T, P, t_perc = _history(power=power, amplitude=amplitude, t_hot=1.0, t_cold=t_cold)
        ctx = types.SimpleNamespace(
            derived_params={"WARNING:false_vacuum_not_shrinking": False},
            verbose=False,
            pot=None,
            phase_symmetric=None,
            PercolationConf=types.SimpleNamespace(
                time_temperature_mode="bag", integral_method="ode"
            ),
        )
        percolation = types.SimpleNamespace(TSYM=T, P=P, Tperc=t_perc)
        obs = cls.__new__(cls)
        cls._check_false_vacuum_shrinking(obs, ctx, percolation)
        return ctx.derived_params["WARNING:false_vacuum_not_shrinking"]

    def test_both_backends_flag_a_growing_false_vacuum(self):
        for cls in (TransitionObservables, TransitionObservablesFixedStep):
            with self.subTest(backend=cls.__module__):
                self.assertTrue(self._run(cls, power=2, amplitude=1e-2, t_cold=0.1))
                self.assertFalse(self._run(cls, power=20, amplitude=1e-3, t_cold=0.65))


if __name__ == "__main__":
    unittest.main()
