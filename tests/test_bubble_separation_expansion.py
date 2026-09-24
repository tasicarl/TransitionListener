"""The mean bubble separation must use the expansion history of the percolation integral."""

from __future__ import annotations

import unittest
from pathlib import Path
import types
from unittest import mock

import numpy as np
from scipy import integrate

from transitionlistener import bubbledynamics as bd
from transitionlistener import errors
from transitionlistener.helper_functions import load_potential

REPO = Path(__file__).resolve().parents[1]


class FixedPhase:
    """Phase stand-in that returns the same field value at every temperature."""

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def valAt(self, T):
        return self.x


class ConventionTests(unittest.TestCase):
    def test_an_unknown_mode_is_rejected(self):
        # It used to reach _time_temperature_factors, which raises; the predicate must not
        # silently turn a mistyped mode into the bag limit.
        for mode in ("sound-speed", "Bag", "", "ode"):
            with self.subTest(mode=mode):
                with self.assertRaises(errors.PercolationError):
                    bd.percolation_uses_sound_speed(mode)

    def test_only_the_bag_mode_uses_the_bag_limit(self):
        self.assertTrue(bd.percolation_uses_sound_speed(None))
        self.assertTrue(bd.percolation_uses_sound_speed("sound_speed"))
        self.assertFalse(bd.percolation_uses_sound_speed("bag"))
        self.assertEqual(bd.expansion_interpolants(None, None, [1.0, 2.0], time_temperature_mode="bag"),
                         (None, None))

    def test_separation_against_direct_integral(self):
        # Constant 3 c_s^2 = k gives a ∝ T^(-1/k), dt = -dT/(k H T) and
        # n_B(T_p) = int dT' Gamma / (k H T') (T_p/T')^(3/k).
        k, H, Tp, Tmax = 0.8, 1e-3, 1.0, 1.5

        def S(T):
            return np.asarray(T, dtype=float) * (10.0 + 40.0 * (np.asarray(T, dtype=float) - 1.0))

        def P(T):
            return np.zeros_like(np.asarray(T, dtype=float))

        def Hint(T):
            return np.full_like(np.asarray(T, dtype=float), H)

        def integrand(T):
            return float(bd.Gamma(T, S(T))[0]) / (k * H * T) * (Tp / T) ** (3.0 / k)

        expected = integrate.quad(integrand, Tp, Tmax, epsrel=1e-10)[0] ** (-1.0 / 3.0)
        separation = bd.calcMeanBubbleSeparation(
            Tp, Tmax, S, P, Hint,
            entropyInt=lambda T: np.asarray(T, dtype=float) ** (3.0 / k),
            coolingInt=lambda T: np.full_like(np.asarray(T, dtype=float), k),
        )
        self.assertTrue(np.isclose(separation, expected, rtol=1e-5, atol=0))
        # The bag limit differs by much more than that, so the test tells the two apart.
        bag = bd.calcMeanBubbleSeparation(Tp, Tmax, S, P, Hint)
        self.assertGreater(abs(bag / expected - 1), 1e-2)


def _exponential_history(C, b=1400.0, n=20001):
    """Exponential nucleation rate with S/T = 70 + b (T - 1), H ∝ T^2, 3 c_s^2 = 1/C, a ∝ T^-C."""
    T = np.linspace(1.02, 0.97, n)
    S = T * (70.0 + b * (T - 1.0))
    H = 1e-10 * T**2
    return T, S, H, np.full_like(T, 1.0 / (3.0 * C)), (T / T[0]) ** (-C)


class PercolationHistoryTests(unittest.TestCase):
    def test_double_integral_with_bag_factors_is_the_bag_integral(self):
        T, S, H, _, _ = _exponential_history(1.0, n=801)
        self.assertTrue(np.isclose(
            bd.percIntegral(T, H, S),
            bd.percIntegral(T, H, S, entropy_density=T**3, cooling_factor=np.ones_like(T)),
            rtol=1e-12, atol=0))

    def test_double_integral_matches_the_ode(self):
        # Compared where I is O(1), which is where it fixes Tperc. Deeper the ODE
        # stops at I = 50, P = 1 - exp(-50), and the two no longer track each other.
        for C in (1.0, 1.08, 1.2):
            T, S, H, cs2, a = _exponential_history(C, n=2001)
            i_ode = bd.percIntegralODE(T, H, S, sound_speed_sq=cs2, scale_factor=a)
            for target in (0.05, 0.34, 2.0):
                k = int(np.argmax(i_ode >= target))
                with self.subTest(C=C, I=target):
                    i_double = bd.percIntegral(
                        T[: k + 1], H[: k + 1], S[: k + 1],
                        entropy_density=(a**-3)[: k + 1],
                        cooling_factor=(3.0 * cs2)[: k + 1],
                    )
                    self.assertTrue(np.isclose(i_double, i_ode[k], rtol=1e-3, atol=0))

    def test_beta_from_separation_for_an_exponential_rate(self):
        # beta/H from R_* reproduces -T dlnGamma/dT / C at Tperc up to the same O(1/beta)
        # residual for every C; the constant-g_s separation is off by C^(1/3).
        f = 0.29
        residuals, bag_ratios = {}, {}
        for C in (1.0, 1.08, 1.2):
            T, S, H, cs2, a = _exponential_history(C)
            P = 1.0 - np.exp(-bd.percIntegralODE(T, H, S, sound_speed_sq=cs2, scale_factor=a))
            j = int(np.argmax(P >= f))
            Tp = float(np.interp(f, [P[j - 1], P[j]], [T[j - 1], T[j]]))

            def Sint(x):
                return np.interp(x, T[::-1], S[::-1])

            def Pint(x):
                return np.interp(x, T[::-1], P[::-1])

            def Hint(x):
                return 1e-10 * np.asarray(x, dtype=float) ** 2

            separation = bd.calcMeanBubbleSeparation(
                Tp, T[0], Sint, Pint, Hint,
                entropyInt=lambda x: np.asarray(x, dtype=float) ** (3.0 * C),
                coolingInt=lambda x: np.full_like(np.asarray(x, dtype=float), 1.0 / C),
            )
            bag = bd.calcMeanBubbleSeparation(Tp, T[0], Sint, Pint, Hint)
            h = Tp * 1e-6
            ln_gamma = [float(np.log(bd.Gamma(np.array([t]), np.array([float(Sint(t))]))[0])) for t in (Tp + h, Tp - h)]
            beta = -(ln_gamma[0] - ln_gamma[1]) / (2 * h) * Tp / C
            residuals[C] = (8 * np.pi) ** (1 / 3) / (separation * Hint(Tp)) * f ** (-1 / 3) / beta - 1
            bag_ratios[C] = bag / separation
        for C in (1.08, 1.2):
            with self.subTest(C=C):
                self.assertLess(abs(residuals[C] - residuals[1.0]), 1e-3)
                self.assertTrue(np.isclose(bag_ratios[C], C ** (1 / 3), rtol=2e-3, atol=0))


class FullSweepTests(unittest.TestCase):
    """The double integral must receive the history through percIntegralODE_full_sweep."""

    def _sweep(self, mode, cs2, a):
        T, S, H, _, _ = _exponential_history(1.1, n=401)

        def factors(pot, phase, temps, m):
            return (None, None) if m == "bag" else (cs2, a)

        with mock.patch.object(bd, "_time_temperature_factors", side_effect=factors):
            I, P = bd.percIntegralODE_full_sweep(
                T, H, S, pot=object(), phase_symmetric=object(),
                time_temperature_mode=mode, integral_method="double_integral")
        return T, H, S, I, P

    def test_double_integral_sweep_uses_the_history(self):
        _, _, _, cs2, a = _exponential_history(1.1, n=401)
        T, H, S, I, P = self._sweep("sound_speed", cs2, a)
        expected = bd.percIntegral(T, H, S, entropy_density=a**-3, cooling_factor=3.0 * cs2)
        self.assertTrue(np.isclose(I[-1], expected, rtol=1e-12, atol=0))
        self.assertTrue(np.allclose(P, 1 - np.exp(-I), rtol=0, atol=1e-12))
        # It is a different answer from the bag limit, so the test can tell them apart.
        self.assertGreater(abs(I[-1] / bd.percIntegral(T, H, S) - 1), 1e-3)

    def test_bag_mode_sweep_ignores_the_history(self):
        _, _, _, cs2, a = _exponential_history(1.1, n=401)
        T, H, S, I, _ = self._sweep("bag", cs2, a)
        self.assertTrue(np.isclose(I[-1], bd.percIntegral(T, H, S), rtol=1e-12, atol=0))


class BetaTimeFactorTests(unittest.TestCase):
    """The 3 c_s^2 factor on (beta/H)_S3 must come from the percolation history."""

    def _factor(self, mode, gw_sound_speed):
        from transitionlistener.transitionObservables import TransitionObservables

        ctx = types.SimpleNamespace(
            derived_params={"c_s_sym": 1 / np.sqrt(3) if gw_sound_speed == "1/3" else 0.6},
            verbose=False,
            pot=object(),
            phase_symmetric=FixedPhase([0.0]),
            PercolationConf=types.SimpleNamespace(time_temperature_mode=mode),
        )
        obs = TransitionObservables.__new__(TransitionObservables)
        with mock.patch.object(bd, "calcSoundSpeedSq", return_value=0.21):
            return TransitionObservables._beta_time_temperature_factor(obs, ctx, 0.5)

    def test_gw_sound_speed_setting_does_not_enter(self):
        # GWconfig.sound_speed = "1/3" puts 1/sqrt(3) into derived["c_s_sym"]. Reading it
        # here would return 1.0 and silently drop the factor the history applied.
        for gw in ("compute", "1/3"):
            with self.subTest(gw_sound_speed=gw):
                self.assertAlmostEqual(self._factor("sound_speed", gw), 3.0 * 0.21)

    def test_bag_mode_gives_one(self):
        self.assertAlmostEqual(self._factor("bag", "compute"), 1.0)


class ModelThermodynamicsTests(unittest.TestCase):
    def test_interpolants_follow_the_model_across_the_qcd_crossover(self):
        # Conformal dark U(1) with v = 6 GeV: internal temperatures 15 to 50 span 90 to 300 MeV.
        pot = load_potential(str(REPO / "models/TL_conformal_dark_u1.py"), "specific_potential")(
            {"g": 0.692, "y": 0.01, "v_GeV": 6.0}, verbose=False)
        phase = FixedPhase([0.0])
        T = np.geomspace(50.0, 15.0, 200)
        entropyInt, coolingInt = bd.expansion_interpolants(pot, phase, T)
        for temp in (T[0], T[57], T[-1]):
            with self.subTest(T=temp):
                cs_sq = bd.calcSoundSpeedSq(pot, phase.valAt(temp), temp)
                self.assertTrue(np.isclose(coolingInt(temp), 3.0 * cs_sq, rtol=1e-10, atol=0))

        def entropy_density(temp):
            dT = temp * 1e-4
            return -float(np.squeeze(pot.dVdT(phase.valAt(temp), temp, dT=dT, include_decoupled=False)))

        model_ratio = entropy_density(T[0]) / entropy_density(T[-1])
        self.assertTrue(np.isclose(entropyInt(T[0]) / entropyInt(T[-1]), model_ratio, rtol=2e-3, atol=0))
        # Across the crossover the entropy density is far from T^3.
        self.assertGreater(abs(model_ratio / (T[0] / T[-1]) ** 3 - 1), 0.2)


if __name__ == "__main__":
    unittest.main()
