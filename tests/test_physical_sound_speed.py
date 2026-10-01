"""The sound speed that feeds the bubble expansion speed, and the flags around it.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import contextlib
import io
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from transitionlistener import config
from transitionlistener import errors
from transitionlistener.helper_functions import load_potential
from transitionlistener.helper_functions import (temperatureDerivativeStep,
                                                 thermalDerivativeStep)
from transitionlistener.hydrodynamics import (Hydrodynamics,
                                              resolve_configured_sound_speed)
from transitionlistener.interface.samplers import get_empty_result

REPO = Path(__file__).resolve().parents[1]


def conformal(g=0.7, v_GeV=0.1, y=0.01):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_potential(str(REPO / "models/TL_conformal_dark_u1.py"),
                              "specific_potential")(
            {"g": g, "y": y, "v_GeV": v_GeV}, verbose=False)


class FlatPhase:
    Tmin, Tmax = 1.0e-9, 1.0e9

    def __init__(self, x):
        self.x = np.asarray(x, float)

    def valAt(self, T, deriv=0):
        T = np.asarray(T, float)
        return np.broadcast_to(self.x, T.shape + self.x.shape).copy()


class DaisyEvaluationTests(unittest.TestCase):
    """The daisy term, evaluated without throwing away significant digits."""

    def test_a_small_thermal_correction_matches_the_series(self):
        from transitionlistener.generic_potential import _daisy_mass_cubed_difference
        m20 = np.array([5.0e4])
        for Pi in (5.0e-5, 1.0e-9, 1.0e-14):
            with self.subTest(Pi=Pi):
                # Pi is handed over, not recovered from the sum: at 1e-14 it is below the
                # spacing of floats near 5e4, so `m20 + Pi` is exactly `m20` and the
                # information is gone before the function is even called. That is the whole
                # reason for the hook.
                got = complex(_daisy_mass_cubed_difference(
                    m20, m20 + Pi, Pi=np.array([Pi]))[0]).real
                u = Pi / m20[0]
                exact = m20[0] ** 1.5 * (1.5 * u + 0.375 * u * u)
                self.assertLess(abs(got / exact - 1.0), 1.0e-6)

    def test_the_sum_itself_loses_a_tiny_thermal_correction(self):
        # Stated as a measurement, because it is the premise of the hook.
        m20 = 5.0e4
        self.assertEqual(m20 + 1.0e-14, m20)
        self.assertNotEqual(m20 + 5.0e-5, m20)

    def test_the_direct_form_would_lose_those_digits(self):
        # The point of the rearrangement: the naive difference is wrong where it matters.
        m20 = 5.0e4
        Pi = 1.0e-9
        naive = (m20 + Pi) ** 1.5 - m20 ** 1.5
        exact = m20 ** 1.5 * 1.5 * (Pi / m20)
        self.assertGreater(abs(naive / exact - 1.0), 1.0e-3)

    def test_a_large_correction_is_unchanged(self):
        from transitionlistener.generic_potential import _daisy_mass_cubed_difference
        got = complex(_daisy_mass_cubed_difference(np.array([1.0]), np.array([100.0]))[0]).real
        self.assertAlmostEqual(got, 100.0 ** 1.5 - 1.0, places=9)

    def test_a_tachyonic_mode_still_goes_through_the_complex_branch(self):
        from transitionlistener.generic_potential import _daisy_mass_cubed_difference
        got = _daisy_mass_cubed_difference(np.array([-1.0e4]), np.array([-9.0e3]))[0]
        self.assertNotAlmostEqual(complex(got).imag, 0.0)

    def test_the_analytic_thermal_masses_match_the_subtraction_where_it_is_sound(self):
        pot = conformal()
        X = np.array([1000.0])
        # The tolerance follows the conditioning: the subtraction keeps thirteen digits at
        # the percolation temperature and about seven two decades below it.
        for T, rtol in ((22.12, 1.0e-12), (0.02, 1.0e-6)):
            with self.subTest(T=T):
                analytic = np.ravel(pot.debye_massSq(X, T))
                m20 = np.ravel(np.asarray(pot.boson_massSq(X, np.asarray([0.0]))[0], float))
                m2T = np.ravel(np.asarray(pot.boson_massSq(X, np.asarray([T]))[0], float))
                np.testing.assert_allclose(analytic, m2T - m20, rtol=rtol, atol=0)

    def test_the_subtraction_underflows_where_the_analytic_form_does_not(self):
        # This is why the hook exists: below about T = 1e-7 the thermal part of the masses
        # is lost entirely, which silently removes the daisy term.
        pot = conformal()
        X = np.array([1000.0])
        T = 1.0e-7
        m20 = np.ravel(np.asarray(pot.boson_massSq(X, np.asarray([0.0]))[0], float))
        m2T = np.ravel(np.asarray(pot.boson_massSq(X, np.asarray([T]))[0], float))
        self.assertTrue(np.all(m2T - m20 == 0.0))
        self.assertTrue(np.any(np.ravel(pot.debye_massSq(X, T)) > 0.0))

    def test_the_default_hook_declines(self):
        from transitionlistener import generic_potential as gp
        self.assertIsNone(gp.generic_potential.debye_massSq(conformal(), None, 1.0))


class DaisyFastPathTests(unittest.TestCase):
    """The all-small and none-small shortcuts must agree with the general masked path."""

    def _general(self, m20, m2T, Pi=None):
        """The masked two-branch form, written out, as the reference."""
        m20 = np.asarray(m20); m2T = np.asarray(m2T)
        shape = np.broadcast(m20, m2T).shape
        m20b, m2Tb = np.broadcast_arrays(m20, m2T)
        m20r = np.real(m20b)
        with np.errstate(divide="ignore", invalid="ignore"):
            positive = np.asarray(m20r > 0.0)
            u = np.zeros(shape, dtype=float)
            thermal = (np.real(np.broadcast_to(Pi, shape)) if Pi is not None
                       else np.real(m2Tb) - m20r)
            np.divide(thermal, m20r, out=u, where=positive)
            small = positive & (np.abs(u) < 1.0e-2)
        out = np.empty(shape, dtype=complex)
        hot = (m20r + np.real(np.broadcast_to(Pi, shape))) if Pi is not None else m2Tb
        out[small] = np.real(m20b)[small] ** 1.5 * np.expm1(1.5 * np.log1p(u[small]))
        rest = ~small
        out[rest] = pow(hot[rest] + 0j, 1.5) - pow(m20b[rest] + 0j, 1.5)
        return out

    def test_the_shortcuts_reproduce_the_general_path(self):
        from transitionlistener.generic_potential import _daisy_mass_cubed_difference as f
        cases = {
            "all small": (np.array([5.0e4, 1.0e9, 3.0]), np.array([1.0e-9, 1.0e-4, 1.0e-6])),
            "none small": (np.array([1.0, 2.0, 3.0]), np.array([5.0, 7.0, 11.0])),
            "mixed": (np.array([5.0e4, 1.0, 3.0]), np.array([1.0e-9, 9.0, 1.0e-6])),
            "with a tachyon": (np.array([-4.0, 5.0e4]), np.array([1.0, 1.0e-9])),
            "a zero mode": (np.array([0.0, 5.0e4]), np.array([1.0, 1.0e-9])),
        }
        for name, (m20, Pi) in cases.items():
            with self.subTest(case=name):
                m2T = m20 + Pi
                for label, kw in (("analytic Pi", {"Pi": Pi}), ("by subtraction", {})):
                    with self.subTest(thermal=label):
                        got = f(m20, m2T, **kw)
                        want = self._general(m20, m2T, **kw)
                        self.assertEqual(got.shape, want.shape)
                        self.assertEqual(got.dtype, want.dtype)
                        np.testing.assert_allclose(got, want, rtol=0.0, atol=0.0)

    def test_the_shortcuts_are_the_ones_being_taken(self):
        # Otherwise this would pass while measuring nothing: the two cases that the fast paths
        # exist for have to be the cases that occur.
        from transitionlistener.generic_potential import _daisy_mass_cubed_difference as f
        m20 = np.array([5.0e4, 1.0e9])
        self.assertTrue(np.all(np.isfinite(f(m20, m20 + 1.0e-9, Pi=np.full(2, 1.0e-9)))))
        hot = np.array([1.0, 2.0])
        self.assertTrue(np.all(np.isfinite(f(hot, hot * 7.0))))


class ThermalDerivativeTests(unittest.TestCase):
    """The sound speed is taken from the thermal part, not from the whole potential."""

    def _phases(self, pot):
        from transitionlistener.phases import Phases
        with contextlib.redirect_stdout(io.StringIO()):
            return Phases(pot, False).phases

    def _hydro(self, pot):
        phases = self._phases(pot)
        T = 22.12
        def norm(p):
            if not (p.Tmin <= T <= p.Tmax):
                return -1.0
            return float(np.linalg.norm(np.atleast_1d(np.squeeze(p.valAt(T)))))
        broken = max(phases.values(), key=norm)
        symmetric = min(phases.values(), key=lambda p: norm(p) if norm(p) >= 0 else 1e30)
        return Hydrodynamics(pot, symmetric, broken, False)

    def test_the_thermal_part_excludes_what_does_not_depend_on_temperature(self):
        pot = conformal()
        X = np.array([1000.0])
        T = 22.12
        thermal = float(np.squeeze(pot.V_thermal(X, T)))
        full = float(np.squeeze(pot.Vtot(X, T, include_decoupled=False)))
        constant = (float(np.squeeze(pot.V0(X))) + float(np.squeeze(pot.Vct(X)))
                    + float(np.squeeze(pot.V1_from_X(X))))
        self.assertAlmostEqual(thermal / (full - constant), 1.0, places=9)

    def test_the_sound_speed_survives_deep_supercooling(self):
        """Where the whole-potential derivative has no digits left.

        The released route returns nan below a temperature of about 1e-6 of the scale; this
        one is still finite at 1e-9 of it.
        """
        pot = conformal()
        hydro = self._hydro(pot)
        for ratio in (1.0e-5, 1.0e-7, 1.0e-9):
            with self.subTest(T_over_v=ratio):
                cs = hydro.calc_cs(ratio * pot.v_stable, sym=False)
                self.assertTrue(np.isfinite(cs), "no sound speed at this depth")
                self.assertGreater(cs, 0.0)
                self.assertLess(cs, 1.0)

    def test_it_agrees_with_the_whole_potential_where_that_still_works(self):
        pot = conformal()
        hydro = self._hydro(pot)
        T = 1.0e-1 * pot.v_stable
        X = np.atleast_1d(np.squeeze(hydro.low_phase.valAt(T)))
        h = T * 1.0e-3
        def V(t):
            return float(np.squeeze(pot.Vtot(X, t, include_decoupled=False)))
        d1 = (V(T + h) - V(T - h)) / (2 * h)
        d2 = (V(T + h) - 2 * V(T) + V(T - h)) / h ** 2
        self.assertAlmostEqual(hydro.calc_cs(T, sym=False) / np.sqrt(d1 / (T * d2)),
                               1.0, places=4)


class SuperluminalTests(unittest.TestCase):
    """A sound speed of one or more is refused, not replaced."""

    def _hydro_returning(self, cs_sq):
        class Pot:
            X0 = np.array([1.0])
            config = types.SimpleNamespace(
                gwConf=types.SimpleNamespace(coupled_hydrodynamics=True))

            def dV_thermal_dT(self, X, T, dT=None):
                return cs_sq * T

            def d2V_thermal_dT2(self, X, T, dT=None):
                return 1.0

        hydro = Hydrodynamics.__new__(Hydrodynamics)
        hydro.pot = Pot()
        hydro.high_phase = hydro.low_phase = FlatPhase([1.0])
        hydro.verbose = False
        return hydro

    def test_one_or_more_raises(self):
        for cs_sq in (1.0, 1.21, 2.9):
            with self.subTest(c_s=np.sqrt(cs_sq)):
                with self.assertRaises(errors.SuperluminalSoundSpeedError):
                    self._hydro_returning(cs_sq).calc_cs(100.0, sym=False)

    def test_the_error_carries_the_value_and_the_temperature(self):
        try:
            self._hydro_returning(2.9).calc_cs(100.0, sym=False)
        except errors.SuperluminalSoundSpeedError as err:
            self.assertAlmostEqual(err.c_s, np.sqrt(2.9), places=9)
            self.assertAlmostEqual(err.T, 100.0, places=9)
            self.assertIn("not a speed", str(err))
        else:
            self.fail("no error raised")

    def test_an_ordinary_value_passes(self):
        self.assertAlmostEqual(self._hydro_returning(1.0 / 3.0).calc_cs(100.0, sym=False),
                               1.0 / np.sqrt(3.0), places=9)

    def test_a_phase_without_a_plasma_is_still_nan_rather_than_an_error(self):
        # Zero over zero and a negative ratio are not impossibilities, they are a phase that
        # has frozen out; those stay as nan, as before.
        for cs_sq in (0.0, -1.0):
            with self.subTest(cs_sq=cs_sq):
                self.assertTrue(np.isnan(
                    self._hydro_returning(cs_sq).calc_cs(100.0, sym=False)))


class StepDependenceTests(unittest.TestCase):
    """A value can be round-off and still be a speed, so the value alone cannot be checked."""

    def _hydro(self, pot):
        return ThermalDerivativeTests._hydro(ThermalDerivativeTests(), pot)

    def test_a_healthy_point_is_not_flagged(self):
        pot = conformal()
        noisy, change = self._hydro(pot).sound_speed_is_step_dependent(22.12, sym=False)
        self.assertFalse(noisy)
        self.assertLess(change, 1.0e-2)

    def test_the_threshold_is_the_documented_one(self):
        from transitionlistener import hydrodynamics as hyd
        self.assertAlmostEqual(hyd.SOUND_SPEED_JUMP_TOLERANCE, 0.05)
        self.assertAlmostEqual(hyd.SOUND_SPEED_STEP_RATIO, 10.0)

    def test_a_value_that_moves_with_the_step_is_flagged(self):
        pot = conformal()
        hydro = self._hydro(pot)
        real = hydro._calc_cs_at_step

        def wobbling(T, sym, step_factor=1.0):
            cs, dT = real(T, sym, step_factor)
            return cs * (1.0 + 0.5 * (step_factor - 1.0)), dT

        with mock.patch.object(hydro, "_calc_cs_at_step", wobbling):
            noisy, change = hydro.sound_speed_is_step_dependent(22.12, sym=False)
        self.assertTrue(noisy)
        self.assertGreater(change, 0.05)


class DaisyValidityTests(unittest.TestCase):
    """The daisy resummation reports when it is being used where it does not apply."""

    def _X(self, pot, T):
        from transitionlistener.phases import Phases
        with contextlib.redirect_stdout(io.StringIO()):
            phases = Phases(pot, False).phases
        def norm(p):
            if not (p.Tmin <= T <= p.Tmax):
                return -1.0
            return float(np.linalg.norm(np.atleast_1d(np.squeeze(p.valAt(T)))))
        return np.atleast_1d(np.squeeze(max(phases.values(), key=norm).valAt(T)))

    def test_it_is_quiet_where_the_modes_are_light(self):
        pot = conformal()
        for ratio in (1.0e-1, 1.0e-2):
            T = ratio * pot.v_stable
            with self.subTest(T_over_v=ratio):
                outside, result = pot.daisy_outside_validity(self._X(pot, T), T)
                self.assertFalse(outside)
                self.assertLess(result, 1.0)

    def test_it_fires_where_the_daisy_term_has_overtaken_the_radiation(self):
        pot = conformal()
        for ratio in (1.0e-3, 1.0e-5, 1.0e-9):
            T = ratio * pot.v_stable
            with self.subTest(T_over_v=ratio):
                outside, result = pot.daisy_outside_validity(self._X(pot, T), T)
                self.assertTrue(outside)
                self.assertGreater(result, 1.0)

    def test_both_conditions_are_needed(self):
        """Neither the ratio nor the mass alone raises the flag.

        A large ratio on its own is not the signature: at high temperature the daisy term is
        legitimate and goes as `T^4` like the radiation. This is checked by driving each
        condition on its own rather than by picking a temperature where both happen to be off,
        which the quiet test above already covers and which would make this vacuous. Removing
        either half of the `and` makes one of the two subtests fail.
        """
        pot = conformal()
        T = 1.0e-1 * pot.v_stable
        X = self._X(pot, T)
        def daisy_at(factor):
            """A daisy term pinned to `factor` times the radiation at whatever T is asked."""
            def fake(self, b0, bT, t, Pi=None):
                return factor * np.squeeze(
                    self.constantTerms(np.asarray(t, dtype=float), include_decoupled=False))
            return fake

        # Heavy modes, small ratio: the resummation is outside its domain mode by mode, but it
        # is not distorting the thermodynamics, so there is nothing to report.
        # The field values are taken before the patch: `_X` retraces the phases, which would
        # run with the stand-in daisy term and give nothing usable.
        T_deep = 1.0e-9 * pot.v_stable
        X_deep = self._X(pot, T_deep)
        T_hot = 0.2 * pot.v_stable
        X_hot = self._X(pot, T_hot)
        with mock.patch.object(type(pot), "Vdaisy", daisy_at(0.1)):
            outside, ratio = pot.daisy_outside_validity(X_deep, T_deep)
        self.assertLess(ratio, 1.0, "the ratio was meant to be small here")
        self.assertFalse(outside, "a small ratio must not raise it")

        # Light modes, large ratio: a daisy term above the radiation at a temperature where
        # every mode still has m/T < 1 is the legitimate high-temperature behaviour. At
        # T/v = 0.2 the lightest mode of this model sits at m/T = 0.63, so the masses are the
        # real ones here and only the size of the term is imposed.
        with mock.patch.object(type(pot), "Vdaisy", daisy_at(50.0)):
            outside, ratio = pot.daisy_outside_validity(X_hot, T_hot)
        self.assertGreater(ratio, 1.0, "the ratio was meant to be large here")
        self.assertFalse(outside, "light modes must not raise it however large the ratio")


class DiagnosticTimeoutTests(unittest.TestCase):
    """A diagnostic wrapped in a broad `except` must not cancel an abort."""

    def test_the_daisy_check_lets_a_timeout_through(self):
        pot = conformal()
        with mock.patch.object(type(pot), "Vdaisy",
                               mock.Mock(side_effect=errors.Timeout())):
            with self.assertRaises(errors.Timeout):
                pot.daisy_outside_validity(np.array([1000.0]), 22.12)

    def test_it_still_swallows_an_ordinary_failure(self):
        # The broad `except` is deliberate: a diagnostic that cannot be computed must not bring
        # the run down over itself.
        pot = conformal()
        with mock.patch.object(type(pot), "Vdaisy",
                               mock.Mock(side_effect=ValueError("no"))):
            outside, ratio = pot.daisy_outside_validity(np.array([1000.0]), 22.12)
        self.assertFalse(outside)
        self.assertTrue(np.isnan(ratio))


class RegistrationTests(unittest.TestCase):
    KEYS = ("WARNING:noisy_c_s", "DIAG:c_s_step_change",
            "WARNING:daisy_outside_validity", "DIAG:daisy_over_radiation")

    def test_every_key_is_registered_where_registration_is_required(self):
        empty = get_empty_result()
        for key in self.KEYS:
            with self.subTest(key=key):
                self.assertIn(key, config.all_observables)
                self.assertIn(key, empty)
        self.assertEqual(set(config.all_observables) - set(empty), set())

    def test_both_solvers_default_them(self):
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof
        pot = conformal()
        phase = FlatPhase([0.0])
        tr = types.SimpleNamespace(Tnuc=30.0, full_tunneling_info={},
                                   high_phase="hot", low_phase="cold")
        for module in (to, tof):
            with self.subTest(module=module.__name__):
                cls = module.TransitionObservables
                obs = cls.__new__(cls)
                obs.pot = pot
                obs.phases = {"hot": phase, "cold": phase}
                obs.GWconfig = pot.config.gwConf
                obs.PercolationConf = pot.config.percolationConf
                obs.derived_param_names = []
                obs.verbose = False
                derived = cls._initialize_transition_context(obs, tr).derived_params
                self.assertIs(derived["WARNING:noisy_c_s"], False)
                self.assertIs(derived["WARNING:daisy_outside_validity"], False)
                self.assertTrue(np.isnan(derived["DIAG:c_s_step_change"]))
                self.assertTrue(np.isnan(derived["DIAG:daisy_over_radiation"]))


class ThermalStepRuleTests(unittest.TestCase):
    """The derivative step of the thermal part is its own rule, and lands on the plateau."""

    def test_it_does_not_inherit_the_whole_potential_rule(self):
        # The whole-potential rule saturates its 3e-2 ceiling under supercooling, where the
        # derived rule is two orders of magnitude finer. If the two ever coincide, the thermal
        # derivatives have silently gone back to being sized by the vacuum offset.
        pot = conformal()
        X = np.array([1000.0])
        T = 1.0e-5 * pot.v_stable
        self.assertAlmostEqual(temperatureDerivativeStep(pot, T, X) / T, 3.0e-2, places=6)
        self.assertAlmostEqual(thermalDerivativeStep(pot, T, X) / T, 9.275e-5, places=7)

    def test_it_is_the_derived_number_and_not_a_tuned_one(self):
        # (eps/3)^(1/4), the balance of the second difference's round-off against the first
        # difference's truncation for a radiation-like thermal part. Asserted against the
        # closed form rather than against a literal, so the docstring's derivation is the test.
        pot = conformal()
        expected = (np.finfo(float).eps / 3.0) ** 0.25
        for T in (1.0e-9, 1.0, 22.12, 1.0e4):
            with self.subTest(T=T):
                self.assertAlmostEqual(
                    thermalDerivativeStep(pot, T, np.array([1.0])) / T, expected, places=12)

    def test_the_step_does_not_depend_on_the_field_value(self):
        pot = conformal()
        a = thermalDerivativeStep(pot, 22.12, np.array([0.0]))
        b = thermalDerivativeStep(pot, 22.12, np.array([5000.0]))
        self.assertEqual(a, b)

    def test_the_sound_speed_is_on_the_converged_plateau(self):
        """The returned value must be the step-independent one, not merely a value.

        The plateau is measured here rather than quoted: `c_s` is recomputed over the band
        where neither round-off nor truncation dominates, and the shipped step has to agree
        with it. A step an order of magnitude too coarse fails this at 1e-5, which is what
        the earlier inherited rule was.
        """
        pot = conformal()
        hydro = ThermalDerivativeTests._hydro(ThermalDerivativeTests(), pot)
        for ratio in (1.0e-1, 1.0e-3, 1.0e-5, 1.0e-9):
            T = ratio * pot.v_stable
            X = np.atleast_1d(np.squeeze(hydro.low_phase.valAt(T)))
            with self.subTest(T_over_v=ratio):
                def cs_at(rel):
                    d1 = float(np.squeeze(pot.dV_thermal_dT(X, T, dT=T * rel)))
                    d2 = float(np.squeeze(pot.d2V_thermal_dT2(X, T, dT=T * rel)))
                    return float(np.sqrt(d1 / (T * d2)))
                plateau = float(np.median([cs_at(r) for r in (3.0e-5, 1.0e-4, 3.0e-4)]))
                self.assertAlmostEqual(hydro.calc_cs(T, sym=False) / plateau, 1.0, places=6)

    def test_deep_supercooling_reaches_the_daisy_limit(self):
        # A thermal part dominated by the daisy term goes as T^3, so c_s^2 -> 1/2. Asserting
        # the value, not just finiteness: this is what a step that is too coarse gets wrong,
        # and it is also what fails if the analytic Debye masses stop being passed through.
        pot = conformal()
        hydro = ThermalDerivativeTests._hydro(ThermalDerivativeTests(), pot)
        for ratio in (1.0e-7, 1.0e-9):
            with self.subTest(T_over_v=ratio):
                cs = hydro.calc_cs(ratio * pot.v_stable, sym=False)
                self.assertAlmostEqual(cs, 1.0 / np.sqrt(2.0), places=5)


class AnalyticDebyeMassTests(unittest.TestCase):
    """The analytic thermal masses have to reach the potential, not only exist."""

    def _broken(self, pot, T):
        from transitionlistener.phases import Phases
        with contextlib.redirect_stdout(io.StringIO()):
            phases = Phases(pot, False).phases
        def norm(p):
            if not (p.Tmin <= T <= p.Tmax):
                return -1.0
            return float(np.linalg.norm(np.atleast_1d(np.squeeze(p.valAt(T)))))
        return np.atleast_1d(np.squeeze(max(phases.values(), key=norm).valAt(T)))

    def test_the_potential_uses_them(self):
        """Mutating `debye_massSq` to decline must move the thermal part of the potential.

        Measured on the thermal part and not on `Vtot`: there the vacuum energy is larger by
        more than sixteen orders of magnitude and the shift is below the last digit of the sum,
        which is the whole reason the sound speed is taken from the thermal part. Measured
        deep, too: the subtraction the hook replaces keeps digits until about ``T/v = 1e-5``,
        so a shallow point cannot tell the two apart.
        """
        pot = conformal()
        for ratio, floor in ((1.0e-7, 1.0e-4), (1.0e-9, 1.0e3)):
            T = ratio * pot.v_stable
            X = self._broken(pot, T)
            with self.subTest(T_over_v=ratio):
                with_hook = float(np.squeeze(pot.V_thermal(X, T)))
                with mock.patch.object(type(pot), "debye_massSq", lambda self, X, T: None):
                    without = float(np.squeeze(pot.V_thermal(X, T)))
                self.assertGreater(abs(with_hook / without - 1.0), floor)
        # And it is invisible in the sum, which is the statement the branch rests on.
        T = 1.0e-7 * pot.v_stable
        X = self._broken(pot, T)
        full = float(np.squeeze(pot.Vtot(X, T, include_decoupled=False)))
        with mock.patch.object(type(pot), "debye_massSq", lambda self, X, T: None):
            full_hookless = float(np.squeeze(pot.Vtot(X, T, include_decoupled=False)))
        self.assertAlmostEqual(full / full_hookless, 1.0, places=14)

    def test_every_route_to_the_daisy_term_passes_them(self):
        """`Pi` has to be handed over wherever the daisy term is formed, not only in one place.

        Asserted structurally, by watching what `Vdaisy` is called with, because in `Vtot` the
        consequence is not observable: there the shift sits below the last digit of the vacuum
        energy, so comparing values cannot tell whether the masses were passed. A route that
        stops passing them is only noticed later, in a temperature derivative, which is exactly
        the failure this branch exists to remove. `Vdaisy_from_X` was missing it.
        """
        pot = conformal()
        T = 1.0e-7 * pot.v_stable
        X = self._broken(pot, T)
        routes = {
            "Vtot": lambda: pot.Vtot(X, T, include_decoupled=False),
            "V_thermal": lambda: pot.V_thermal(X, T),
            "Vdaisy_from_X": lambda: pot.Vdaisy_from_X(X, T),
            "daisy_outside_validity": lambda: pot.daisy_outside_validity(X, T),
        }
        real = type(pot).Vdaisy
        for name, call in routes.items():
            seen = []

            def spy(self, b0, bT, t, Pi=None, **kw):
                seen.append(Pi)
                return real(self, b0, bT, t, Pi=Pi, **kw)

            with self.subTest(route=name):
                with mock.patch.object(type(pot), "Vdaisy", spy):
                    call()
                self.assertTrue(seen, f"{name} never formed the daisy term")
                for Pi in seen:
                    self.assertIsNotNone(
                        Pi, f"{name} formed the daisy term without the analytic Debye masses")

    def test_the_standalone_daisy_helper_uses_them_too(self):
        # `Vdaisy_from_X` is a second route to the same term; if it does not pass `Pi` it
        # disagrees with the potential it is a piece of.
        pot = conformal()
        T = 1.0e-7 * pot.v_stable
        X = self._broken(pot, T)
        direct = float(np.squeeze(pot.Vdaisy_from_X(X, T)))
        Ta = np.asarray([T], dtype=float)
        expected = float(np.squeeze(pot.Vdaisy(
            pot.boson_massSq(X, Ta * 0.0), pot.boson_massSq(X, Ta), Ta,
            Pi=pot.debye_massSq(X, Ta))))
        self.assertAlmostEqual(direct / expected, 1.0, places=12)
        with mock.patch.object(type(pot), "debye_massSq", lambda self, X, T: None):
            hookless = float(np.squeeze(pot.Vdaisy_from_X(X, T)))
        self.assertNotAlmostEqual(direct / hookless, 1.0, places=9)


class PseudoTraceSoundSpeedTests(unittest.TestCase):
    """`calcSoundSpeedSq`, which normalises the pseudo-trace strengths, uses the thermal part.

    It is a separate function from `Hydrodynamics.calc_cs` and is reached by a different
    caller, so fixing one does not fix the other. Both backends have their own copy.
    """

    def _X(self, pot, T):
        from transitionlistener.phases import Phases
        with contextlib.redirect_stdout(io.StringIO()):
            phases = Phases(pot, False).phases
        def norm(p):
            if not (p.Tmin <= T <= p.Tmax):
                return -1.0
            return float(np.linalg.norm(np.atleast_1d(np.squeeze(p.valAt(T)))))
        return np.atleast_1d(np.squeeze(max(phases.values(), key=norm).valAt(T)))

    def _modules(self):
        from transitionlistener import bubbledynamics as bd
        from transitionlistener import bubbledynamics_fixedstep as bdf
        return (("adaptive", bd), ("fixed step", bdf))

    def test_it_survives_where_the_whole_potential_route_does_not(self):
        # The value, not just finiteness: a daisy-dominated thermal part goes as T^3, so
        # c_s^2 -> 1/2. The whole-potential route gives nothing usable this deep, so a copy
        # that still differences `Vtot` fails here rather than merely losing accuracy.
        pot = conformal()
        for ratio in (1.0e-7, 1.0e-9):
            T = ratio * pot.v_stable
            X = self._X(pot, T)
            for label, module in self._modules():
                with self.subTest(solver=label, T_over_v=ratio):
                    self.assertAlmostEqual(
                        float(np.squeeze(module.calcSoundSpeedSq(pot, X, T))), 0.5, places=5)

    def test_the_two_backends_agree(self):
        # They used to differ: the adaptive one took the whole-potential step rule and the
        # fixed-step one a hard-coded dT/T = 1e-3, on the same quantity.
        pot = conformal()
        from transitionlistener import bubbledynamics as bd
        from transitionlistener import bubbledynamics_fixedstep as bdf
        for ratio in (1.0e-1, 1.0e-3, 1.0e-7):
            T = ratio * pot.v_stable
            X = self._X(pot, T)
            with self.subTest(T_over_v=ratio):
                self.assertEqual(float(np.squeeze(bd.calcSoundSpeedSq(pot, X, T))),
                                 float(np.squeeze(bdf.calcSoundSpeedSq(pot, X, T))))

    def test_it_does_not_read_the_whole_potential(self):
        """Structural, because at shallow temperatures the two routes agree numerically.

        Mutating either copy back to `pot.dVdT`/`pot.d2VdT2` makes this fail wherever the
        values happen to agree, which is where a value-only test would not notice.
        """
        pot = conformal()
        T = 1.0e-1 * pot.v_stable
        X = self._X(pot, T)
        for label, module in self._modules():
            with self.subTest(solver=label):
                with mock.patch.object(
                        type(pot), "dVdT",
                        mock.Mock(side_effect=AssertionError(
                            "calcSoundSpeedSq differenced the whole potential"))):
                    with mock.patch.object(
                            type(pot), "d2VdT2",
                            mock.Mock(side_effect=AssertionError(
                                "calcSoundSpeedSq differenced the whole potential"))):
                        value = float(np.squeeze(module.calcSoundSpeedSq(pot, X, T)))
                self.assertTrue(np.isfinite(value))


class ConfiguredSoundSpeedTests(unittest.TestCase):
    """Both solvers accept the setting the superluminal refusal tells the user to use."""

    SETTINGS = {"1/3": 1.0 / np.sqrt(3.0), "0.4": 0.4, 0.55: 0.55}

    def test_the_resolver_reads_the_label_and_a_number(self):
        for setting, expected in self.SETTINGS.items():
            with self.subTest(setting=setting):
                self.assertAlmostEqual(resolve_configured_sound_speed(setting), expected)

    def test_it_refuses_what_is_not_a_speed(self):
        for setting in (1.0, 0.0, -0.2, 1.5):
            with self.subTest(setting=setting):
                with self.assertRaises(ValueError):
                    resolve_configured_sound_speed(setting)
        with self.assertRaises(NotImplementedError):
            resolve_configured_sound_speed("radiation")

    def test_the_message_the_refusal_advertises_is_accepted(self):
        # The error text recommends GWconfig.sound_speed = '1/3'. A recommendation one of the
        # two backends rejects is worse than none, so the text and the resolver are checked
        # against each other rather than separately.
        err = errors.SuperluminalSoundSpeedError(1.48, 1.0e-5)
        self.assertIn("'1/3'", str(err))
        self.assertAlmostEqual(resolve_configured_sound_speed("1/3"), 1.0 / np.sqrt(3.0))

    def test_both_solvers_populate_the_three_columns_from_the_setting(self):
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof
        for module in (to, tof):
            for setting, expected in self.SETTINGS.items():
                with self.subTest(module=module.__name__, setting=setting):
                    cls = module.TransitionObservables
                    obs = cls.__new__(cls)
                    ctx = types.SimpleNamespace(
                        derived_param_names=["c_s"], derived_params={},
                        GWconfig=types.SimpleNamespace(sound_speed=setting),
                        pot=None, phase_symmetric=None, phase_broken=None, verbose=False)
                    cls._ensure_sound_speed(obs, ctx, 30.0)
                    for key in ("c_s", "c_s_sym", "c_s_bro"):
                        self.assertAlmostEqual(ctx.derived_params[key], expected)


class SuperluminalReportingTests(unittest.TestCase):
    """The refusal says which phase, and why."""

    def _hydro(self, cs_sq):
        return SuperluminalTests._hydro_returning(SuperluminalTests(), cs_sq)

    def test_the_phase_is_named(self):
        for sym, name in ((True, "symmetric"), (False, "broken")):
            with self.subTest(sym=sym):
                try:
                    self._hydro(2.9).calc_cs(100.0, sym=sym)
                except errors.SuperluminalSoundSpeedError as err:
                    self.assertEqual(err.phase, name)
                    self.assertIn(f"The {name}-phase sound speed", str(err))
                else:
                    self.fail("no error raised")

    def test_a_superluminal_symmetric_phase_is_refused_too(self):
        with self.assertRaises(errors.SuperluminalSoundSpeedError):
            self._hydro(1.21).calc_cs(100.0, sym=True)

    def test_it_carries_an_error_code_the_interface_can_report(self):
        err = errors.SuperluminalSoundSpeedError(1.2, 1.0)
        self.assertEqual(err.errorcode, 18)
        codes = {}
        for name in dir(errors):
            obj = getattr(errors, name)
            if isinstance(obj, type) and issubclass(obj, Exception):
                code = getattr(obj(), "errorcode", None) if name !=                     "SuperluminalSoundSpeedError" else 18
                if code is not None:
                    codes.setdefault(code, []).append(name)
        self.assertEqual(codes[18], ["SuperluminalSoundSpeedError"],
                         "error code 18 is used twice")

    def test_the_diagnostics_reach_the_message(self):
        err = errors.SuperluminalSoundSpeedError(
            1.48, 1.0e-5, phase="broken", step_change=0.4, daisy_ratio=1.5e7)
        self.assertEqual(err.step_change, 0.4)
        self.assertEqual(err.daisy_ratio, 1.5e7)
        self.assertIn("losing", str(err))
        self.assertIn("outside its range", str(err))
        quiet = errors.SuperluminalSoundSpeedError(
            1.48, 1.0e-5, step_change=1.0e-5, daisy_ratio=0.2)
        self.assertIn("not a finite-difference artefact", str(quiet))
        self.assertIn("within the", str(quiet))

    def test_the_solvers_attach_them_before_refusing(self):
        # The diagnostics have to be taken before `calc_cs`, or the refusal cannot say which
        # mechanism produced the value. Mutating the order so they are taken afterwards makes
        # this fail, because the message would carry neither number.
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof
        for module in (to, tof):
            with self.subTest(module=module.__name__):
                cls = module.TransitionObservables
                obs = cls.__new__(cls)
                pot = types.SimpleNamespace(
                    daisy_outside_validity=lambda X, T: (True, 1.5e7))
                ctx = types.SimpleNamespace(
                    derived_param_names=["c_s"], derived_params={},
                    GWconfig=types.SimpleNamespace(sound_speed="compute"),
                    pot=pot, phase_symmetric=FlatPhase([0.0]),
                    phase_broken=FlatPhase([1.0]), verbose=False)
                fake = mock.Mock()
                fake.sound_speed_is_step_dependent.return_value = (False, 2.0e-5)
                fake.calc_cs.side_effect = errors.SuperluminalSoundSpeedError(
                    1.48, 30.0, phase="broken")
                with mock.patch.object(module, "Hydrodynamics", return_value=fake):
                    with self.assertRaises(errors.SuperluminalSoundSpeedError) as caught:
                        cls._ensure_sound_speed(obs, ctx, 30.0)
                self.assertAlmostEqual(caught.exception.step_change, 2.0e-5)
                self.assertAlmostEqual(caught.exception.daisy_ratio, 1.5e7)
                self.assertIn("not a finite-difference artefact", str(caught.exception))
                self.assertIn("outside its range", str(caught.exception))


class ThermalPartDaisySchemeTests(unittest.TestCase):
    """`V_thermal` dispatches on `self.daisy` a second time, beside `Vtot`.

    The two have to agree for every scheme, or the sound speed is taken from a potential the
    run is not using. Only the Arnold-Espinosa branch is the default, so the other two are
    checked here rather than left to a model that happens to set them.
    """

    def test_every_scheme_matches_the_temperature_dependence_of_Vtot(self):
        pot = conformal()
        X = np.array([900.0])
        # Differences rather than values: `Vtot(X, 0)` is not available, because
        # `constantTerms` takes the logarithm of the temperature.
        T1, T2 = 50.0, 49.0
        for daisy in ("ArnoldEspinosa", "Parwani", "off"):
            with self.subTest(daisy=daisy):
                pot.daisy = daisy
                thermal = float(np.squeeze(pot.V_thermal(X, T1) - pot.V_thermal(X, T2)))
                whole = float(np.squeeze(pot.Vtot(X, T1, include_decoupled=False)
                                         - pot.Vtot(X, T2, include_decoupled=False)))
                self.assertAlmostEqual(thermal / whole, 1.0, places=9)
        pot.daisy = "ArnoldEspinosa"

    def test_the_schemes_are_not_all_the_same_potential(self):
        # Without this the test above would pass for a `V_thermal` that ignored `self.daisy`.
        pot = conformal()
        X = np.array([900.0])
        values = []
        for daisy in ("ArnoldEspinosa", "Parwani", "off"):
            pot.daisy = daisy
            values.append(float(np.squeeze(pot.V_thermal(X, 50.0))))
        pot.daisy = "ArnoldEspinosa"
        self.assertEqual(len(set(values)), 3, f"the schemes did not differ: {values}")

    def test_a_scheme_it_cannot_build_is_refused(self):
        pot = conformal()
        pot.daisy = "not-a-scheme"
        try:
            with self.assertRaises(errors.PotentialError):
                pot.V_thermal(np.array([900.0]), 50.0)
        finally:
            pot.daisy = "ArnoldEspinosa"


class SuperluminalPhaseAttributionTests(unittest.TestCase):
    """The diagnostics in the message belong to the phase that was refused."""

    def _ctx_and_fake(self, module, failing_phase):
        pot = types.SimpleNamespace(
            daisy_outside_validity=lambda X, T: (
                (True, 7.0) if float(np.squeeze(X)) == 0.0 else (False, 0.25)))
        ctx = types.SimpleNamespace(
            derived_param_names=["c_s"], derived_params={},
            GWconfig=types.SimpleNamespace(sound_speed="compute"),
            pot=pot, phase_symmetric=FlatPhase([0.0]),
            phase_broken=FlatPhase([1.0]), verbose=False)
        fake = mock.Mock()
        fake.sound_speed_is_step_dependent.side_effect = (
            lambda T, sym: (True, 0.9) if sym else (False, 2.0e-5))
        fake.calc_cs.side_effect = (
            lambda T, sym: (_ for _ in ()).throw(
                errors.SuperluminalSoundSpeedError(1.48, T, phase="symmetric"))
            if sym else 0.5)
        return ctx, fake

    def test_a_symmetric_refusal_carries_the_symmetric_diagnostics(self):
        # Mutating the re-raise to pass the broken-phase `change` and `ratio` regardless of
        # the phase makes this fail: the message would then claim that a value it never
        # measured is "not a finite-difference artefact".
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof
        for module in (to, tof):
            with self.subTest(module=module.__name__):
                cls = module.TransitionObservables
                obs = cls.__new__(cls)
                ctx, fake = self._ctx_and_fake(module, "symmetric")
                with mock.patch.object(module, "Hydrodynamics", return_value=fake):
                    with self.assertRaises(errors.SuperluminalSoundSpeedError) as caught:
                        cls._ensure_sound_speed(obs, ctx, 30.0)
                err = caught.exception
                self.assertEqual(err.phase, "symmetric")
                self.assertAlmostEqual(err.step_change, 0.9)
                self.assertAlmostEqual(err.daisy_ratio, 7.0)
                self.assertIn("losing", str(err))
                self.assertIn("outside its range", str(err))
                # The columns stay the broken phase's: they are what the observable means.
                self.assertAlmostEqual(ctx.derived_params["DIAG:c_s_step_change"], 2.0e-5)
                self.assertAlmostEqual(ctx.derived_params["DIAG:daisy_over_radiation"], 0.25)


class StepDependenceRobustnessTests(unittest.TestCase):
    """The diagnostic must survive what it is diagnosing."""

    def test_it_does_not_propagate_the_refusal(self):
        # A step at which the value is not a speed is the strongest statement that the value
        # depends on the step. Letting the refusal out of here would make the diagnostic
        # unusable precisely where it matters, and would abort before it is recorded.
        hydro = SuperluminalTests._hydro_returning(SuperluminalTests(), 2.9)
        noisy, change = hydro.sound_speed_is_step_dependent(100.0, sym=False)
        self.assertTrue(noisy)
        self.assertTrue(np.isnan(change))

    def test_it_varies_the_step_in_both_directions(self):
        # Round-off grows as the step shrinks and truncation as it grows, so a check that only
        # coarsens misses the failure it is looking for. Mutating it to one side makes this
        # fail, because only the finer step moves here.
        pot = conformal()
        hydro = ThermalDerivativeTests._hydro(ThermalDerivativeTests(), pot)
        real = hydro._calc_cs_at_step
        seen = []

        def recording(T, sym, step_factor=1.0):
            seen.append(step_factor)
            cs, dT = real(T, sym, step_factor)
            return (cs * 1.5 if step_factor < 1.0 else cs), dT

        with mock.patch.object(hydro, "_calc_cs_at_step", recording):
            noisy, change = hydro.sound_speed_is_step_dependent(22.12, sym=False)
        self.assertIn(0.1, seen, "the finer step was never tried")
        self.assertIn(10.0, seen, "the coarser step was never tried")
        self.assertTrue(noisy)
        self.assertGreater(change, 0.05)

    def test_a_healthy_point_has_room_below_the_alarm(self):
        # With the derived step the measured change in a healthy regime is 1e-6 to 2e-3,
        # against a 0.05 alarm. The inherited whole-potential step left a factor 2.8, close
        # enough to false-alarm; this asserts the headroom, not merely the absence of a flag.
        pot = conformal()
        hydro = ThermalDerivativeTests._hydro(ThermalDerivativeTests(), pot)
        from transitionlistener import hydrodynamics as hyd
        for ratio in (1.0e-1, 1.0e-3, 1.0e-9):
            T = ratio * pot.v_stable
            with self.subTest(T_over_v=ratio):
                noisy, change = hydro.sound_speed_is_step_dependent(T, sym=False)
                self.assertFalse(noisy)
                self.assertLess(change, hyd.SOUND_SPEED_JUMP_TOLERANCE / 10.0)


if __name__ == "__main__":
    unittest.main()
