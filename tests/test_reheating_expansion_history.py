"""The reheated broken plasma is diluted with the scale factor of the percolation integral.

Between two support temperatures the false-vacuum plasma conserves its entropy, s_sym a^3 = const,
so a ~ 1/(T h_sym^(1/3)), not a ~ 1/T wherever its entropy degrees of freedom change. The step-3
profile and the reheating-temperature integration must both use that scale factor.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import percolation_adaptivestepsize as pas

DELTA_V = 1.0   # vacuum energy released by the transition
A_PT = 1.0      # radiation of the transitioning sector: e = A_PT T^4
H_DS = 5.0      # its entropy degrees of freedom
T_PLATEAU = 0.85  # below this the true-vacuum fraction no longer changes


def extra_dilution(T):
    """h_sym(T)/h_sym(T_PLATEAU) of a false-vacuum plasma that loses entropy degrees of freedom below T_PLATEAU."""
    T = np.asarray(T, dtype=float)
    return np.where(T < T_PLATEAU, 1.0 - 3.0 * (T_PLATEAU - T), 1.0)


def energy_density(sym):
    def energy(pot, phase, T, include_decoupled=True):
        return (DELTA_V if phase is sym else 0.0) + A_PT * T**4
    return energy


def fraction(TSYM):
    return np.clip((1.0 - TSYM) / (1.0 - T_PLATEAU), 0.0, 1.0) ** 2 * 0.95


class ComovingVolumeRatioTests(unittest.TestCase):
    def test_the_ratio_is_that_of_the_scale_factor_of_the_percolation_integral(self):
        T = np.array([1.0, 0.9, 0.8, 0.7])
        a = np.array([1.0, 1.2, 1.5, 2.0])
        with mock.patch.object(bd, "_time_temperature_factors", lambda *args, **kw: (np.full(4, 1 / 3), a)):
            ratios = bd.comoving_volume_ratios(None, None, T)
        np.testing.assert_allclose(ratios, [1.0, (1.0 / 1.2) ** 3, (1.2 / 1.5) ** 3, (1.5 / 2.0) ** 3], rtol=1e-14)

    def test_the_bag_mode_dilutes_as_t_cubed(self):
        T = np.array([1.0, 0.5, 0.25])
        ratios = bd.comoving_volume_ratios(None, None, T, time_temperature_mode="bag")
        np.testing.assert_allclose(ratios, [1.0, 0.125, 0.125], rtol=1e-14)


class Step3DilutionTests(unittest.TestCase):
    """Below T_PLATEAU nothing converts, so the broken plasma only redshifts: T_b a = const."""

    def profile(self):
        sym, bro = types.SimpleNamespace(name="sym"), types.SimpleNamespace(name="bro")
        TSYM = np.linspace(1.0, 0.8, 41)
        P = fraction(TSYM)
        a = (TSYM[0] / TSYM) / extra_dilution(TSYM) ** (1.0 / 3.0)
        n = TSYM.size
        state = bd.PercolationState(TSYM=TSYM, Sr=np.zeros(n), Hr=np.zeros(n), Pr=np.zeros(n), Tb=np.zeros(n))
        settings = types.SimpleNamespace(f_final=0.99, acc_tperc=1e-2, time_temperature_mode="sound_speed",
                                         integral_method="ode")
        with mock.patch.object(bd, "energyDensity", energy_density(sym)), \
                mock.patch.object(bd, "calcAction", lambda *args: 100.0), \
                mock.patch.object(bd, "h_eff_DS", lambda T, pot, phase: H_DS), \
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda T, pot: 0.0), \
                mock.patch.object(bd, "_time_temperature_factors", lambda *args, **kw: (np.full(n, 1 / 3), a)), \
                mock.patch.object(bd, "percIntegralODE_full_sweep", lambda TSYM, *a, **k: (None, P)):
            pas._compute_step3_profile(state, TSYM, P, settings, {}, types.SimpleNamespace(conversionFactor=1.0),
                                       sym, bro, 1.0, 1.0, TBROmin=0.5, TBROmax=5.0, verbose=False)
        return TSYM, a, state

    def test_the_broken_plasma_redshifts_with_the_scale_factor_of_the_percolation_integral(self):
        TSYM, a, state = self.profile()
        tail = TSYM <= T_PLATEAU
        self.assertGreater(np.count_nonzero(tail), 5)
        product = state.Tb[tail] * a[tail]
        np.testing.assert_allclose(product, product[0], rtol=1e-9)


class ReheatingTemperatureDilutionTests(unittest.TestCase):
    """integrate_broken_temperature dilutes with d ln s_sym/dT, not 3/T."""

    def treh(self, log_entropy):
        sym = types.SimpleNamespace(name="sym", Tmin=0.5, Tmax=5.0)
        bro = types.SimpleNamespace(name="bro", Tmin=0.5, Tmax=5.0)
        TSYM = np.linspace(1.0, 0.75, 51)
        with mock.patch.object(bd, "energyDensity", energy_density(sym)), \
                mock.patch.object(bd, "h_eff_DS", lambda T, pot, phase: H_DS), \
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda T, pot: 0.0):
            return bd.integrate_broken_temperature(None, sym, bro, TSYM, fraction(TSYM), 0.8,
                                                   log_entropy=log_entropy)

    def test_a_t_cubed_entropy_is_the_bag_relation(self):
        bag = self.treh(None)
        self.assertIsNotNone(bag)
        # Agreement to the tolerance of the integrator: the derivative of ln s_sym is a finite difference.
        self.assertAlmostEqual(self.treh(lambda T: 3.0 * np.log(T) + 7.0) / bag, 1.0, places=6)

    def test_degrees_of_freedom_leaving_the_false_vacuum_cool_the_broken_plasma(self):
        # Below T_PLATEAU the fraction no longer changes and S(T_b) = H_DS T_b^3 follows s_sym,
        # so the extra dilution lowers T_b by the cube root of the drop in h_sym.
        bag = self.treh(None)
        diluted = self.treh(lambda T: 3.0 * np.log(T) + np.log(extra_dilution(T)))
        expected = float(extra_dilution(0.8)) ** (1.0 / 3.0)
        self.assertLess(expected, 0.96)
        self.assertAlmostEqual(diluted / bag / expected, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
