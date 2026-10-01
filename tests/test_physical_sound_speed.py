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
from transitionlistener.hydrodynamics import Hydrodynamics
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
        # A large ratio alone is not the signature: at high temperature the daisy term is
        # legitimate and goes as T^4 like the radiation.
        pot = conformal()
        T = 1.0e-1 * pot.v_stable
        X = self._X(pot, T)
        outside, _ = pot.daisy_outside_validity(X, T)
        self.assertFalse(outside)


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


if __name__ == "__main__":
    unittest.main()
