"""The mean bubble separation must use the expansion history of the percolation integral."""

from __future__ import annotations

import ast
import contextlib
import io
import unittest
from pathlib import Path
import types
from unittest import mock

import numpy as np
from scipy import integrate

from transitionlistener import bubbledynamics as bd
from transitionlistener import bubbledynamics_fixedstep as bdf
from transitionlistener import errors
from transitionlistener import runtime_options
from transitionlistener.config import PercolationConf
from transitionlistener import constants as cn
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


class DeepHistoryTests(unittest.TestCase):
    """A history of hundreds of e-folds must not collapse to an unexpanding universe."""

    def _separation(self, ln_a_span, **kw):
        Tp, Tmax = 1.0, 1.5
        Sint = lambda x: np.asarray(x, float) * (10.0 + 40.0 * (np.asarray(x, float) - 1.0))
        Pint = lambda x: np.zeros_like(np.asarray(x, float))
        Hint = lambda x: np.full_like(np.asarray(x, float), 1e-3)
        return bd.calcMeanBubbleSeparation(Tp, Tmax, Sint, Pint, Hint, **kw)

    def test_log_entropy_keeps_the_expansion(self):
        # ln a grows linearly from 0 at Tmax to ln_a_span at Tperc, so
        # ln s = -3 ln a and s(Tperc)/s(T') = exp(-3 (ln a(Tperc) - ln a(T'))).
        Tp, Tmax, k = 1.0, 1.5, 3.0
        for ln_a_span in (20.0, 300.0, 600.0):
            with self.subTest(ln_a_span=ln_a_span):
                def ln_a(x):
                    x = np.asarray(x, dtype=float)
                    return ln_a_span * (Tmax - x) / (Tmax - Tp)

                log_sep = self._separation(
                    ln_a_span, coolingInt=lambda x: np.full_like(np.asarray(x, float), k),
                    logEntropyInt=lambda x: -3.0 * ln_a(x))
                # No-expansion reference: a constant scale factor.
                flat_sep = self._separation(
                    ln_a_span, coolingInt=lambda x: np.full_like(np.asarray(x, float), k),
                    logEntropyInt=lambda x: np.zeros_like(np.asarray(x, float)))
                self.assertTrue(np.isfinite(log_sep))
                self.assertGreater(abs(log_sep / flat_sep - 1), 1e-6)

    def test_absolute_entropy_would_have_collapsed(self):
        # The same history through the entropy itself: a^-3 underflows and the ratio is
        # no longer recoverable, so it must not come back as the unexpanding answer.
        Tp, Tmax, k = 1.0, 1.5, 3.0

        def ln_a(x):
            x = np.asarray(x, dtype=float)
            return 600.0 * (Tmax - x) / (Tmax - Tp)

        flat = self._separation(
            600.0, coolingInt=lambda x: np.full_like(np.asarray(x, float), k),
            logEntropyInt=lambda x: np.zeros_like(np.asarray(x, float)))
        with np.errstate(divide="ignore", invalid="ignore"):
            underflowed = self._separation(
                600.0, coolingInt=lambda x: np.full_like(np.asarray(x, float), k),
                entropyInt=lambda x: np.exp(-3.0 * ln_a(x)))
        # It degenerates visibly instead of coming back as the unexpanding answer, which
        # is what clamping the underflowed entropy used to produce.
        self.assertTrue(np.isfinite(flat))
        self.assertTrue(np.isinf(underflowed))


class FullSweepTests(unittest.TestCase):
    """The double integral must receive the history through percIntegralODE_full_sweep."""

    def _sweep(self, mode, cs2, a):
        T, S, H, _, _ = _exponential_history(1.1, n=401)

        def factors(pot, phase, temps, m, definition=None):
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

    def _factor(self, mode, gw_sound_speed, definition="eff_potential", entropy=None):
        """The factor at one temperature, with a usable entropy unless one is given.

        A temperature contributes its sound speed and its scale factor together or not at all,
        so the entropy has to work for the sound speed to be read at all; the bare object below
        stands in for a potential and cannot provide one.
        """
        from transitionlistener.transitionObservables import TransitionObservables

        ctx = types.SimpleNamespace(
            derived_params={"c_s_sym": 1 / np.sqrt(3) if gw_sound_speed == "1/3" else 0.6},
            verbose=False,
            pot=object(),
            phase_symmetric=FixedPhase([0.0]),
            PercolationConf=types.SimpleNamespace(time_temperature_mode=mode,
                                                  entropy_definition=definition),
        )
        obs = TransitionObservables.__new__(TransitionObservables)
        entropy = entropy if entropy is not None else (lambda pot, ph, T, d: float(T) ** 3.0)
        with mock.patch.object(bd, "calcSoundSpeedSq", return_value=0.21), \
                mock.patch.object(bd, "entropy_density", entropy):
            return TransitionObservables._beta_time_temperature_factor(obs, ctx, 0.5)

    def test_gw_sound_speed_setting_does_not_enter(self):
        # GWconfig.sound_speed = "1/3" puts 1/sqrt(3) into derived["c_s_sym"]. Reading it
        # here would return 1.0 and silently drop the factor the history applied.
        # s ~ T^4 gives 3 c_s^2 = 3/4. A T^3 entropy would give exactly 1.0, which is also
        # what the bag fallback gives, so it could not tell the two apart.
        for gw in ("compute", "1/3"):
            with self.subTest(gw_sound_speed=gw):
                self.assertAlmostEqual(
                    self._factor("sound_speed", gw, definition="eff_potential",
                                 entropy=lambda pot, ph, T, d: float(T) ** 4.0),
                    0.75, places=5)

    def test_the_factor_follows_the_entropy_scheme(self):
        # The beta/H factor is 3 c_s^2 of the percolation history, so it must change with
        # the scheme: s ~ T^4 gives 3 c_s^2 = 3/4, whatever the potential says.
        self.assertAlmostEqual(
            self._factor("sound_speed", "compute", definition="dof_table",
                         entropy=lambda pot, ph, T, d: float(T) ** 4.0), 0.75, places=5)

    def test_bag_mode_gives_one(self):
        for definition in ("dof_table", "eff_potential"):
            with self.subTest(entropy_definition=definition):
                self.assertAlmostEqual(self._factor("bag", "compute", definition), 1.0)


class SchemeSeparationTests(unittest.TestCase):
    def test_the_two_schemes_disagree_where_the_choice_matters(self):
        """Guard against a vacuous parametrisation.

        The consistency tests above compare each scheme against its own entropy, so they
        would still pass if the setting were ignored and both branches returned the same
        thing. This fixes a point where the schemes are far apart, so that a change which
        stops reading ``entropy_definition`` fails somewhere.
        """
        with contextlib.redirect_stdout(io.StringIO()):  # the 2HDM prints its inputs
            pot = load_potential(str(REPO / "models/TL_2HDM.py"), "R2HDM")(dict(
                lambda1=0.006, lambda2=0.25, lambda3=8.27, lambda4=-2.55, lambda5=0.76,
                m12_sq_GeV2=14186.7, tan_beta=17.7, yukawa_type=1,
                v_GeV=246.21965079413735), verbose=False)
        phase = FixedPhase([0.0] * pot.Ndim)
        # 31.2 GeV is where that phase stops being traced, so the comparison stays inside
        # the range the solver can reach. Continuing to 20 GeV would widen the separation
        # to 24%, but by extrapolating the phase.
        T = np.geomspace(160.0, 31.2, 80)

        cooling, log_entropy = {}, {}
        for definition in ("dof_table", "eff_potential"):
            log_entropy[definition], cooling[definition] = bd.expansion_interpolants(
                pot, phase, T, entropy_definition=definition)

        counted = np.array([float(cooling["dof_table"](t)) for t in T])
        potential = np.array([float(cooling["eff_potential"](t)) for t in T])

        # Measured in this window: 3 c_s^2 stays within 0.977 to 0.997 for the counted
        # route against 0.955 to 1.037 from the potential, a 5.5% separation. Requiring 3%
        # leaves room for solver tuning while still failing outright if the two schemes
        # ever coincide.
        separation = np.abs(potential / counted - 1.0).max()
        self.assertGreater(separation, 0.03)

        # The scale factor must separate too, not only the sound speed. On the eff_potential
        # route the sound speed comes from `calcSoundSpeedSq` and never from
        # `entropy_density`, so without this the scheme could be ignored for a(T) alone.
        ratio_counted = float(log_entropy["dof_table"](T[0]) - log_entropy["dof_table"](T[-1]))
        ratio_potential = float(
            log_entropy["eff_potential"](T[0]) - log_entropy["eff_potential"](T[-1]))
        self.assertGreater(abs(ratio_potential - ratio_counted), 0.01)

    def test_the_potential_scale_factor_follows_the_potential(self):
        """The a(T) of "eff_potential" must be -dV/dT, on a point where that is visible.

        ``ModelThermodynamicsTests`` makes the same check on the conformal dark U(1), where
        the two schemes' entropy ratios differ by only 1.1e-3 and its tolerance cannot tell
        them apart. Here they differ by more than a percent.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            pot = load_potential(str(REPO / "models/TL_2HDM.py"), "R2HDM")(dict(
                lambda1=0.006, lambda2=0.25, lambda3=8.27, lambda4=-2.55, lambda5=0.76,
                m12_sq_GeV2=14186.7, tan_beta=17.7, yukawa_type=1,
                v_GeV=246.21965079413735), verbose=False)
        phase = FixedPhase([0.0] * pot.Ndim)
        T = np.geomspace(160.0, 31.2, 80)

        logEntropyInt, _ = bd.expansion_interpolants(
            pot, phase, T, entropy_definition="eff_potential")

        def minus_dVdT(temp):
            dT = temp * 1e-4
            return -float(np.squeeze(
                pot.dVdT(phase.valAt(temp), temp, dT=dT, include_decoupled=False)))

        ratio = float(np.exp(logEntropyInt(T[0]) - logEntropyInt(T[-1])))
        self.assertTrue(np.isclose(ratio, minus_dVdT(T[0]) / minus_dVdT(T[-1]),
                                   rtol=2e-3, atol=0))
        # and it must NOT be the counted entropy, which is what makes this test bite.
        counted_ratio = (bd.entropy_density(pot, phase, float(T[0]), "dof_table")
                         / bd.entropy_density(pot, phase, float(T[-1]), "dof_table"))
        self.assertGreater(abs(ratio / counted_ratio - 1.0), 0.01)


class ModelThermodynamicsTests(unittest.TestCase):
    def test_interpolants_follow_the_model_across_the_qcd_crossover(self):
        # Conformal dark U(1) with v = 6 GeV: internal temperatures 15 to 50 span 90 to 300 MeV.
        pot = load_potential(str(REPO / "models/TL_conformal_dark_u1.py"), "specific_potential")(
            {"g": 0.692, "y": 0.01, "v_GeV": 6.0}, verbose=False)
        phase = FixedPhase([0.0])
        T = np.geomspace(50.0, 15.0, 200)
        for definition in ("dof_table", "eff_potential"):
            logEntropyInt, coolingInt = bd.expansion_interpolants(
                pot, phase, T, entropy_definition=definition)
            for temp in (T[0], T[57], T[-1]):
                with self.subTest(entropy_definition=definition, T=temp):
                    # c_s^2 must be 1/(d ln s/d ln T) of the SAME entropy that fixes a(T):
                    # the two are one relation, so they may not come from different places.
                    h = temp * 1e-5
                    s_lo = bd.entropy_density(pot, phase, temp - h, definition)
                    s_hi = bd.entropy_density(pot, phase, temp + h, definition)
                    cs_sq = ((np.log(temp + h) - np.log(temp - h))
                             / (np.log(s_hi) - np.log(s_lo)))
                    self.assertTrue(np.isclose(coolingInt(temp), 3.0 * cs_sq, rtol=2e-3, atol=0))
                    # and a(T) must follow the same entropy: logEntropyInt is ln s up to a
                    # constant, so its differences reproduce the scheme's entropy ratios.
                    s_ref = bd.entropy_density(pot, phase, float(T[0]), definition)
                    s_here = bd.entropy_density(pot, phase, float(temp), definition)
                    self.assertTrue(np.isclose(
                        float(logEntropyInt(temp) - logEntropyInt(T[0])),
                        np.log(s_here / s_ref), rtol=0, atol=2e-3))

        # This block checks the potential's own entropy, so it must ask for that scheme
        # explicitly: under the default "dof_table" it would assert that the counted
        # degrees of freedom reproduce -dV/dT, which is the mixing this setting ends.
        logEntropyInt, coolingInt = bd.expansion_interpolants(
            pot, phase, T, entropy_definition="eff_potential")

        def entropy_density(temp):
            dT = temp * 1e-4
            return -float(np.squeeze(pot.dVdT(phase.valAt(temp), temp, dT=dT, include_decoupled=False)))

        model_ratio = entropy_density(T[0]) / entropy_density(T[-1])
        ratio = float(np.exp(logEntropyInt(T[0]) - logEntropyInt(T[-1])))
        self.assertTrue(np.isclose(ratio, model_ratio, rtol=2e-3, atol=0))
        # Across the crossover the entropy density is far from T^3.
        self.assertGreater(abs(model_ratio / (T[0] / T[-1]) ** 3 - 1), 0.2)



class EntropyDefinitionConsistencyTests(unittest.TestCase):
    """The two readings of one entropy, and the solver that does not implement the choice."""

    @staticmethod
    def _potential_with(mode, definition):
        """A stand-in whose settings are the real defaults with these two changed."""
        from transitionlistener.config import PercolationConf

        conf = PercolationConf()
        conf.algorithm_mode, conf.entropy_definition = mode, definition
        return types.SimpleNamespace(config=types.SimpleNamespace(percolationConf=conf))

    def test_the_fixed_step_solver_refuses_a_definition_it_does_not_implement(self):
        """It keeps its own time-temperature relation; ignoring the setting would be silent.

        The two solvers read their settings through separate functions of the same name, and the
        fixed step size path never reaches the adaptive one, so the refusal has to live in both.
        """
        from transitionlistener import bubbledynamics_fixedstep as bdf

        for module in (bd, bdf):
            with self.subTest(module=module.__name__):
                with self.assertRaises(errors.PercolationError) as caught:
                    module._build_percolation_settings(
                        self._potential_with("fixed_step_size", "eff_potential"), 20)
                self.assertIn("fixed_step_size", str(caught.exception))
                self.assertIn("eff_potential", str(caught.exception))

    def test_the_solver_that_runs_is_the_one_that_validates(self):
        """The fixed step size percolation reads its settings through its own module."""
        import inspect
        from transitionlistener import bubbledynamics_fixedstep as bdf

        source = inspect.getsource(bdf.calcPercAndEvolve)
        self.assertIn("_build_percolation_settings", source)

    def test_the_default_definition_passes_on_either_solver(self):
        for mode in ("adaptive_step_size", "fixed_step_size"):
            from transitionlistener import bubbledynamics_fixedstep as bdf

            with self.subTest(mode=mode):
                bd._build_percolation_settings(self._potential_with(mode, "dof_table"), 20)
                bdf._build_percolation_settings(self._potential_with(mode, "dof_table"), 20)

    def test_an_unknown_definition_is_refused(self):
        with self.assertRaises(errors.PercolationError):
            bd._build_percolation_settings(
                self._potential_with("adaptive_step_size", "something_else"), 20)

    def test_a_temperature_contributes_both_factors_or_neither(self):
        """Keeping one where the other is unusable mixes the schemes this setting exists to end."""

        class Phase:
            Tmin, Tmax = 1.0, 100.0
            def valAt(self, T):
                return np.array([1.0])

        temperatures = np.array([40.0, 50.0, 60.0])

        # Narrower than the derivative step of 1e-4 T, so the sound speed at the middle point is
        # perfectly well determined from its two neighbours while the entropy there is not. That
        # is the only configuration that tells the two behaviours apart: a window wide enough to
        # swallow the derivative steps leaves the sound speed undefined either way.
        def entropy_that_fails_at_one_point(pot, phase, T, definition="dof_table"):
            if abs(float(T) - 50.0) < 1.0e-3:
                raise RuntimeError("no entropy here")
            return float(T) ** 4               # 1/4, so a measured value differs from the fallback

        with mock.patch.object(bd, "entropy_density", entropy_that_fails_at_one_point):
            cs_sq, scale = bd._time_temperature_factors(object(), Phase(), temperatures,
                                                        definition="dof_table")
        cs_sq = np.asarray(cs_sq, dtype=float)
        # where the entropy works, the sound speed is measured from it
        self.assertAlmostEqual(cs_sq[0], 0.25, places=6)
        self.assertAlmostEqual(cs_sq[2], 0.25, places=6)
        # where it does not, the temperature contributes neither and keeps the bag fallback
        self.assertAlmostEqual(cs_sq[1], 1.0 / 3.0, places=12)

class TimeoutPropagationTests(unittest.TestCase):
    """A run's own timeout may not be absorbed by the entropy evaluations.

    ``errors.Timeout`` subclasses ``Exception`` and is raised from a signal handler, so it
    can fire inside any of the three guarded potential evaluations. Swallowing it turns a
    timed-out run into a silent bag fallback that looks like a result.
    """

    def _potential(self):
        return load_potential(str(REPO / "models/TL_conformal_dark_u1.py"),
                              "specific_potential")(
            {"g": 0.692, "y": 0.01, "v_GeV": 6.0}, verbose=False)

    def test_a_timeout_in_the_phase_evaluation_propagates(self):
        phase = FixedPhase([0.0])
        def boom(T, deriv=0):
            raise errors.Timeout()
        phase.valAt = boom
        with self.assertRaises(errors.Timeout):
            bd._time_temperature_factors(self._potential(), phase, np.array([20.0]),
                                         mode="sound_speed", definition="dof_table")

    def test_a_timeout_in_the_entropy_propagates(self):
        calls = {"n": 0}
        def boom(pot, phase, T, definition):
            calls["n"] += 1
            raise errors.Timeout()
        with mock.patch.object(bd, "entropy_density", boom):
            with self.assertRaises(errors.Timeout):
                bd._time_temperature_factors(self._potential(), FixedPhase([0.0]),
                                             np.array([20.0]), mode="sound_speed",
                                             definition="dof_table")
        self.assertGreater(calls["n"], 0)

    def test_a_timeout_in_the_stencil_propagates(self):
        # The entropy at T itself succeeds and only the derivative stencil times out, which is
        # the third handler rather than the second.
        centre = 20.0

        def entropy(pot, ph, T, definition):
            if float(T) == centre:
                return float(T) ** 4
            raise errors.Timeout()

        with mock.patch.object(bd, "entropy_density", entropy):
            with self.assertRaises(errors.Timeout):
                bd._time_temperature_factors(self._potential(), FixedPhase([0.0]),
                                             np.array([centre]), mode="sound_speed",
                                             definition="eff_potential")


class EntropyStencilClampTests(unittest.TestCase):
    """The entropy derivative may not step outside the range the phase was traced in."""

    class BoundedPhase(FixedPhase):
        Tmin = 40.0
        Tmax = 60.0

    def _cs_sq(self, phase, T):
        # s ~ T^4 gives c_s^2 = 1/4 exactly, whichever side the stencil falls on.
        with mock.patch.object(bd, "entropy_density", lambda pot, ph, t, d: float(t) ** 4):
            sound_speed_sq, _ = bd._time_temperature_factors(
                object(), phase, np.array([T]), mode="sound_speed", definition="dof_table")
        return float(sound_speed_sq[0])

    def test_the_stencil_is_one_sided_at_both_edges(self):
        phase = self.BoundedPhase([0.0])
        for T in (phase.Tmin, 50.0, phase.Tmax):
            with self.subTest(T=T):
                self.assertAlmostEqual(self._cs_sq(phase, T), 0.25, places=6)

    def test_a_phase_without_bounds_still_works(self):
        # Test doubles and older phase objects carry no Tmin/Tmax.
        self.assertAlmostEqual(self._cs_sq(FixedPhase([0.0]), 50.0), 0.25, places=6)

    def test_a_degenerate_range_falls_back_instead_of_dividing_by_zero(self):
        class Pinned(FixedPhase):
            Tmin = 50.0
            Tmax = 50.0
        self.assertAlmostEqual(self._cs_sq(Pinned([0.0]), 50.0), 1.0 / 3.0, places=12)


class EntropyDefinitionOverrideTests(unittest.TestCase):
    """The scheme is selectable per run, like the other percolation controls."""

    @staticmethod
    def percolation_conf(**overrides):
        # A real conf, because apply_percolation_overrides also touches the grid controls.
        conf = PercolationConf()
        runtime_options.apply_percolation_overrides(conf, overrides)
        return conf

    def test_the_definition_is_a_percolation_override(self):
        self.assertIn("percolation_entropy_definition",
                      runtime_options.PERCOLATION_OVERRIDE_KEYS)
        self.assertEqual(
            self.percolation_conf(
                percolation_entropy_definition="eff_potential").entropy_definition,
            "eff_potential")

    def test_the_default_is_left_alone_when_the_override_is_absent(self):
        self.assertEqual(self.percolation_conf().entropy_definition, "dof_table")

    def test_an_unknown_definition_is_refused_by_the_override(self):
        with self.assertRaises(ValueError):
            self.percolation_conf(percolation_entropy_definition="dof_tabel")

    def test_the_override_offers_exactly_the_documented_definitions(self):
        # One list, so the override cannot drift from what the solvers accept.
        for definition in cn.ENTROPY_DEFINITIONS:
            with self.subTest(definition=definition):
                self.assertEqual(
                    self.percolation_conf(
                        percolation_entropy_definition=definition).entropy_definition,
                    definition)


class SchemeForwardingTests(unittest.TestCase):
    """Every call that computes a time-temperature factor must be told the scheme.

    Deleting the forwarding leaves the suite green otherwise: `Tperc` would be computed
    with the default entropy while the mean bubble separation used the chosen one, which is
    the mixing this setting exists to end. This is a wiring invariant, so it is checked on
    the call graph rather than by running a scan.
    """

    CONSUMERS = ("percIntegralODE_full_sweep", "expansion_interpolants",
                 "_time_temperature_factors", "percolation_sound_speed_sq")

    MODULES = ("bubbledynamics.py", "percolation_adaptivestepsize.py",
               "percolation_adaptive_gridbuilders.py", "transitionObservables.py")

    @staticmethod
    def _called_name(node):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def test_every_consumer_call_forwards_the_entropy_definition(self):
        root = Path(bd.__file__).parent
        seen = 0
        for name in self.MODULES:
            tree = ast.parse((root / name).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = self._called_name(node)
                if called not in self.CONSUMERS:
                    continue
                keywords = {kw.arg for kw in node.keywords}
                seen += 1
                if called == "_time_temperature_factors":
                    # The private helper spells it `definition` and is called positionally,
                    # (pot, phase, T, mode, definition), so five positional arguments or the
                    # keyword both count as forwarding it.
                    if "definition" in keywords or len(node.args) >= 5:
                        continue
                with self.subTest(module=name, call=called, line=node.lineno):
                    self.assertIn("entropy_definition", keywords)
        # If the consumers are ever renamed this test would silently check nothing.
        self.assertGreater(seen, 5)


class CountedEntropyTimeoutTests(unittest.TestCase):
    """`g_eff_DS` and `h_eff_DS` catch BaseException, which includes the run's own timeout.

    They are the innermost potential evaluations on the counted route, below the handlers in
    `_time_temperature_factors`, so absorbing a timeout there would hand back the T = 0 vev
    and turn a timed-out point into a finite entropy. Both modules carry both functions.
    """

    def test_both_counters_let_the_timeout_through_in_both_modules(self):
        class TimingOutPhase:
            def valAt(self, T, deriv=0):
                raise errors.Timeout()

        pot = load_potential(str(REPO / "models/TL_conformal_dark_u1.py"),
                             "specific_potential")(
            {"g": 0.692, "y": 0.01, "v_GeV": 6.0}, verbose=False)
        for module in (bd, bdf):
            for name in ("g_eff_DS", "h_eff_DS"):
                with self.subTest(module=module.__name__, function=name):
                    with self.assertRaises(errors.Timeout):
                        getattr(module, name)(20.0, pot, TimingOutPhase())

    def test_an_ordinary_failure_still_falls_back(self):
        # The BaseException fallback must survive: only the timeout is singled out.
        class BrokenPhase:
            def valAt(self, T, deriv=0):
                raise ValueError("outside the traced range")

        pot = load_potential(str(REPO / "models/TL_conformal_dark_u1.py"),
                             "specific_potential")(
            {"g": 0.692, "y": 0.01, "v_GeV": 6.0}, verbose=False)
        for module in (bd, bdf):
            with self.subTest(module=module.__name__):
                with contextlib.redirect_stdout(io.StringIO()):   # it prints a warning
                    value = module.h_eff_DS(20.0, pot, BrokenPhase())
                self.assertTrue(np.isfinite(value))


class UnknownEntropyDefinitionTests(unittest.TestCase):
    def test_entropy_density_refuses_an_unknown_definition(self):
        # It is a public function and the tests call it directly, so it may not answer a
        # misspelled scheme with the counted entropy.
        pot = load_potential(str(REPO / "models/TL_conformal_dark_u1.py"),
                             "specific_potential")(
            {"g": 0.692, "y": 0.01, "v_GeV": 6.0}, verbose=False)
        for definition in ("dof_tabel", "", "DOF_TABLE", "potential"):
            with self.subTest(definition=definition):
                with self.assertRaises(errors.PercolationError):
                    bd.entropy_density(pot, FixedPhase([0.0]), 20.0, definition)

    def test_both_documented_definitions_are_accepted(self):
        pot = load_potential(str(REPO / "models/TL_conformal_dark_u1.py"),
                             "specific_potential")(
            {"g": 0.692, "y": 0.01, "v_GeV": 6.0}, verbose=False)
        for definition in cn.ENTROPY_DEFINITIONS:
            with self.subTest(definition=definition):
                self.assertTrue(np.isfinite(
                    bd.entropy_density(pot, FixedPhase([0.0]), 20.0, definition)))


class PhaseFollowingSoundSpeedTests(unittest.TestCase):
    """The sound speed must differentiate the entropy ALONG the phase, in both schemes.

    A fixed-field derivative drops d2V/dXdT * dX/dT. That term is exactly zero for a phase
    pinned at the origin, so a test on the symmetric phase cannot see it; it needs a phase
    that moves with temperature, as a projected false vacuum does.
    """

    class DriftPhase:
        Tmin, Tmax = 30.0, 200.0
        def __init__(self, slope, ndim):
            self.slope, self.ndim = slope, ndim
        def valAt(self, T, deriv=0):
            T = np.asarray(T, float)
            x = self.slope * (T - 100.0)
            cols = [x] + [np.zeros_like(x)] * (self.ndim - 1)
            return np.stack(np.broadcast_arrays(*cols), axis=-1)

    def _potential(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_potential(str(REPO / "models/TL_2HDM.py"), "R2HDM")(dict(
                lambda1=0.006, lambda2=0.25, lambda3=8.27, lambda4=-2.55, lambda5=0.76,
                m12_sq_GeV2=14186.7, tan_beta=17.7, yukawa_type=1,
                v_GeV=246.21965079413735), verbose=False)

    def _cs_sq(self, pot, phase, T):
        sound_speed_sq, _ = bd._time_temperature_factors(
            pot, phase, np.array([T]), mode="sound_speed", definition="eff_potential")
        return float(sound_speed_sq[0])

    def test_a_pinned_phase_agrees_with_the_fixed_field_derivative(self):
        pot = self._potential()
        phase = self.DriftPhase(0.0, pot.Ndim)
        fixed = bd.calcSoundSpeedSq(pot, phase.valAt(80.0), 80.0)
        self.assertAlmostEqual(self._cs_sq(pot, phase, 80.0) / fixed, 1.0, places=4)

    def test_a_drifting_phase_does_not(self):
        # Measured on this point: 0.086% at a drift of 0.5, and zero without drift. The
        # threshold is well above the 1e-6 agreement of the pinned case.
        pot = self._potential()
        phase = self.DriftPhase(0.5, pot.Ndim)
        fixed = bd.calcSoundSpeedSq(pot, phase.valAt(80.0), 80.0)
        self.assertGreater(abs(self._cs_sq(pot, phase, 80.0) / fixed - 1.0), 1e-4)



if __name__ == "__main__":
    unittest.main()
