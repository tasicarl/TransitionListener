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
from transitionlistener import generic_potential
from transitionlistener import errors
from transitionlistener.helper_functions import load_potential
from transitionlistener.helper_functions import (temperatureDerivativeStep,
                                                 thermalDerivativeStep)
from transitionlistener.hydrodynamics import (Hydrodynamics,
                                              SOUND_SPEED_STEP_RATIO,
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

    def test_the_default_hook_measures_rather_than_declining(self):
        """The base class no longer returns `None` for everything.

        It reads the `T^2` coefficient off the model's own spectrum and verifies it, so a
        model that does not state its Debye masses still gets them. It declines only where the
        verification fails, which `DefaultDebyeMassTests` covers.
        """
        from transitionlistener import generic_potential as gp
        pot = conformal()
        X = np.array([900.0])
        value = gp.generic_potential.debye_massSq(pot, X, np.asarray([0.3 * pot.v_stable]))
        self.assertIsNotNone(value)
        self.assertTrue(np.any(np.ravel(np.asarray(value, dtype=float)) > 0.0))


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
        """Which branch runs, not merely that the answer is finite.

        Asserting finiteness pins nothing: routing both homogeneous cases through the masked
        general path gives the same values, so the test would pass with the shortcuts gone and
        the performance claim unmeasured. The general path is the only one that allocates its
        complex output with `np.empty`, so counting that allocation is what distinguishes them.
        """
        import numpy
        from transitionlistener.generic_potential import _daisy_mass_cubed_difference as f

        def allocations(call):
            with mock.patch.object(numpy, "empty", wraps=numpy.empty) as spy:
                out = call()
            return out, spy.call_count

        small = np.array([5.0e4, 1.0e9])
        out, n = allocations(lambda: f(small, small + 1.0e-9, Pi=np.full(2, 1.0e-9)))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(n, 0, "the all-small case took the masked general path")

        hot = np.array([1.0, 2.0])
        out, n = allocations(lambda: f(hot, hot * 7.0))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(n, 0, "the none-small case took the masked general path")

        # And the mixed case must still take it, or the shortcut is being applied where the
        # two forms disagree, which `test_the_shortcuts_reproduce_the_general_path` checks.
        mixed = np.array([5.0e4, 1.0])
        out, n = allocations(lambda: f(mixed, np.array([5.0e4 + 1.0e-9, 9.0])))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreaterEqual(n, 1, "the mixed case skipped the general path")


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


class WallVelocitySoundSpeedTests(unittest.TestCase):
    """`calcWallVelocityLTE` has its own copy of the sound speed, on the default path.

    `GWConf.wall_velocity` is `"LTE"` by default and `_ensure_wall_velocity` runs before
    `_ensure_sound_speed`, so this copy is evaluated first on every default run and the
    refusal in `calc_cs` cannot protect it.
    """

    def _hydro(self, pot, T):
        from transitionlistener.phases import Phases
        with contextlib.redirect_stdout(io.StringIO()):
            phases = Phases(pot, False).phases
        def norm(p):
            if not (p.Tmin <= T <= p.Tmax):
                return -1.0
            return float(np.linalg.norm(np.atleast_1d(np.squeeze(p.valAt(T)))))
        usable = [p for p in phases.values() if norm(p) >= 0.0]
        return Hydrodynamics(pot, min(usable, key=norm), max(usable, key=norm), False)

    def test_it_does_not_read_the_whole_potential(self):
        # Structural, because where the two routes agree a value test cannot tell them apart,
        # and this function is a fourth copy of the same quantity: reverting it is invisible
        # until a supercooled point, which is where it used to go superluminal.
        pot = conformal()
        T = 1.0e-1 * pot.v_stable
        hydro = self._hydro(pot, T)
        boom = mock.Mock(side_effect=AssertionError(
            "calcWallVelocityLTE differenced the whole potential"))
        with mock.patch.object(type(pot), "dVdT", boom):
            with mock.patch.object(type(pot), "d2VdT2", boom):
                vw = float(np.squeeze(hydro.calcWallVelocityLTE(T)))
        self.assertTrue(0.0 < vw <= 1.0)

    def test_it_uses_the_sector_its_caller_asks_for(self):
        # The enthalpy and the sound speeds must come from the same plasma as the pressures
        # they are combined with. `V_thermal` defaults to excluding the decoupled bath, so a
        # call that forgets to forward `coupled_hydrodynamics` would mix two sectors.
        pot = conformal()
        T = 1.0e-1 * pot.v_stable
        hydro = self._hydro(pot, T)
        seen = []
        real = type(pot).dV_thermal_dT

        def spy(self, X, t, dT=None, include_decoupled=False):
            seen.append(include_decoupled)
            return real(self, X, t, dT=dT, include_decoupled=include_decoupled)

        for coupled in (True, False):
            with self.subTest(coupled_hydrodynamics=coupled):
                seen.clear()
                pot.config.gwConf.coupled_hydrodynamics = coupled
                try:
                    with mock.patch.object(type(pot), "dV_thermal_dT", spy):
                        hydro.calcWallVelocityLTE(T)
                finally:
                    pot.config.gwConf.coupled_hydrodynamics = True
                self.assertTrue(seen, "the enthalpy was not taken from the thermal part")
                self.assertTrue(all(c is coupled for c in seen),
                                f"the sector was not forwarded: {seen}")

    def test_a_superluminal_value_is_refused_rather_than_used(self):
        # It used to go into `find_vw` unchecked: on the conformal line the whole-potential
        # route gave c_s = 1.26 at g = 0.650 and 1.11 at g = 0.750, both at T/v = 1e-5.
        pot = conformal()
        T = 1.0e-1 * pot.v_stable
        hydro = self._hydro(pot, T)
        for phase_is_broken in (True, False):
            with self.subTest(broken=phase_is_broken):
                target = (hydro.low_phase if phase_is_broken else hydro.high_phase).valAt(T)
                real_d2 = type(pot).d2V_thermal_dT2
                real_d1 = type(pot).dV_thermal_dT

                def superluminal(self, X, t, dT=None, include_decoupled=False):
                    # c_s^2 = (dV/dT)/(T d2V/dT2); halving the second derivative doubles it.
                    value = real_d2(self, X, t, dT=dT, include_decoupled=include_decoupled)
                    if np.allclose(np.atleast_1d(np.squeeze(X)),
                                   np.atleast_1d(np.squeeze(target))):
                        return value / 8.0
                    return value

                with mock.patch.object(type(pot), "d2V_thermal_dT2", superluminal):
                    with self.assertRaises(errors.SuperluminalSoundSpeedError) as caught:
                        hydro.calcWallVelocityLTE(T)
                self.assertGreaterEqual(caught.exception.c_s, 1.0)
                self.assertEqual(caught.exception.phase,
                                 "broken" if phase_is_broken else "symmetric")

    def test_a_phase_without_a_plasma_is_still_the_runaway_branch(self):
        # A missing plasma and a value that is not a speed are different statements; only the
        # second is refused. This is the behaviour the refusal must not swallow.
        pot = conformal()
        T = 1.0e-1 * pot.v_stable
        hydro = self._hydro(pot, T)
        with mock.patch.object(type(pot), "dV_thermal_dT",
                               lambda self, X, t, dT=None, include_decoupled=False: 0.0):
            self.assertEqual(hydro.calcWallVelocityLTE(T), 1)


class ReviewFollowUpTests(unittest.TestCase):
    """Three defects the first review of this branch found, each pinned by a test."""

    def test_the_message_takes_its_verdict_from_the_flag_not_the_ratio(self):
        """A large ratio with light modes is legitimate, and the message must say so.

        `daisy_outside_validity` needs both a ratio above one *and* a lightest mode heavier
        than the temperature. Deciding from the ratio alone, as the message used to, calls the
        legitimate high-temperature case a breakdown.
        """
        legit = errors.SuperluminalSoundSpeedError(
            1.2, 1.0, daisy_ratio=50.0, daisy_outside=False)
        self.assertNotIn("outside its range", str(legit))
        self.assertIn("light enough", str(legit))
        broken = errors.SuperluminalSoundSpeedError(
            1.2, 1.0, daisy_ratio=50.0, daisy_outside=True)
        self.assertIn("outside its range", str(broken))
        # Same ratio, opposite verdict: the ratio alone cannot be what decides.
        self.assertNotEqual(str(legit), str(broken))
        # Without a verdict the size is still reported, but nothing is concluded from it.
        silent = errors.SuperluminalSoundSpeedError(1.2, 1.0, daisy_ratio=50.0)
        self.assertIn("50 times the radiation", str(silent))
        self.assertNotIn("outside its range", str(silent))
        self.assertNotIn("light enough", str(silent))

    def test_the_message_and_the_flag_share_one_threshold(self):
        from transitionlistener import hydrodynamics as hyd
        from transitionlistener import constants as cn
        self.assertIs(hyd.SOUND_SPEED_JUMP_TOLERANCE, cn.SOUND_SPEED_JUMP_TOLERANCE)
        tol = cn.SOUND_SPEED_JUMP_TOLERANCE
        self.assertIn("losing", str(errors.SuperluminalSoundSpeedError(
            1.2, 1.0, step_change=tol * 1.1)))
        self.assertIn("stable", str(errors.SuperluminalSoundSpeedError(
            1.2, 1.0, step_change=tol * 0.9)))

    def test_the_daisy_check_is_silent_for_other_resummations(self):
        """With Parwani or no resummation there is no Arnold-Espinosa term to report on."""
        pot = conformal()
        X = DaisyValidityTests._X(DaisyValidityTests(), pot, 1.0e-9 * pot.v_stable)
        T = 1.0e-9 * pot.v_stable
        self.assertEqual(pot.daisy, "ArnoldEspinosa")
        outside, ratio = pot.daisy_outside_validity(X, T)
        self.assertTrue(outside, "the Arnold-Espinosa case must still fire")
        self.assertTrue(np.isfinite(ratio))
        for scheme in ("Parwani", "off"):
            with self.subTest(daisy=scheme):
                with mock.patch.object(pot, "daisy", scheme):
                    outside, ratio = pot.daisy_outside_validity(X, T)
                self.assertFalse(outside)
                self.assertTrue(np.isnan(ratio))

    def test_a_varied_step_that_is_not_a_speed_is_noisy(self):
        """One nan among the varied steps used to be discarded and the rest ranked quiet.

        That reported the stability of the step at which the sound speed still existed, which
        is the opposite of what happened.
        """
        pot = conformal()
        hydro = ThermalDerivativeTests._hydro(ThermalDerivativeTests(), pot)
        real = hydro._calc_cs_at_step
        for broken_factor in (10.0, 0.1):
            with self.subTest(step_factor=broken_factor):
                def partly_undefined(T, sym, step_factor=1.0):
                    cs, dT = real(T, sym, step_factor)
                    if np.isclose(step_factor, broken_factor):
                        return float("nan"), dT
                    return cs, dT
                with mock.patch.object(hydro, "_calc_cs_at_step", partly_undefined):
                    noisy, change = hydro.sound_speed_is_step_dependent(22.12, sym=False)
                self.assertTrue(noisy, "a varied step with no sound speed is step dependence")
                self.assertTrue(np.isnan(change))

    def test_a_non_positive_varied_step_counts_the_same(self):
        pot = conformal()
        hydro = ThermalDerivativeTests._hydro(ThermalDerivativeTests(), pot)
        real = hydro._calc_cs_at_step

        def non_positive(T, sym, step_factor=1.0):
            cs, dT = real(T, sym, step_factor)
            return (0.0 if step_factor > 1.0 else cs), dT

        with mock.patch.object(hydro, "_calc_cs_at_step", non_positive):
            noisy, change = hydro.sound_speed_is_step_dependent(22.12, sym=False)
        self.assertTrue(noisy)
        self.assertTrue(np.isnan(change))

    def test_both_solvers_hand_the_verdict_to_the_refusal(self):
        """The flag has to reach the exception, not just exist beside it."""
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof
        for module in (to, tof):
            with self.subTest(module=module.__name__):
                cls = module.TransitionObservables
                obs = cls.__new__(cls)
                pot = types.SimpleNamespace(
                    daisy_outside_validity=lambda X, T: (False, 50.0))
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
                self.assertIs(caught.exception.daisy_outside, False)
                self.assertNotIn("outside its range", str(caught.exception))


class InfiniteSoundSpeedTests(unittest.TestCase):
    """A second derivative that underflows to zero gives an infinite sound speed.

    That is a value above the limit, not the frozen-phase zero-over-zero, and the two used to
    be lumped together: `calc_cs` returned not-a-number for it and `calcWallVelocityLTE`
    returned the runaway branch, because its zero-denominator guard runs before the refusal.
    """

    def _hydro(self, d1, d2):
        class Pot:
            T_eps = 1.0e-3
            X0 = np.array([1.0])
            config = types.SimpleNamespace(
                gwConf=types.SimpleNamespace(coupled_hydrodynamics=True))

            @staticmethod
            def _broken(X):
                return bool(np.allclose(np.atleast_1d(X), 1.0))

            def dV_thermal_dT(self, X, T, dT=None, include_decoupled=False):
                return d1

            def d2V_thermal_dT2(self, X, T, dT=None, include_decoupled=False):
                return d2

            def dVdT(self, X, T, dT=None, include_decoupled=True, include_radiation=True):
                return d1

            def d2VdT2(self, X, T, dT=None, include_decoupled=True, include_radiation=True):
                return d2

            def Vtot(self, X, T, include_decoupled=True):
                return -0.25 if self._broken(X) else -1.0

        hydro = Hydrodynamics.__new__(Hydrodynamics)
        hydro.pot = Pot()
        hydro.high_phase, hydro.low_phase = FlatPhase([0.0]), FlatPhase([1.0])
        hydro.verbose = False
        return hydro

    # The entropy is positive, so dV/dT < 0; the sign the underflowed second derivative keeps
    # is what decides between an infinite and a negative ratio, and at underflow it is not
    # under anyone's control.
    INFINITE = (-1.0, -0.0)
    NO_PLASMA = (0.0, 0.0)
    NEGATIVE = (-1.0, 0.0)

    def test_calc_cs_refuses_it(self):
        with self.assertRaises(errors.SuperluminalSoundSpeedError) as caught:
            self._hydro(*self.INFINITE).calc_cs(100.0, sym=False)
        self.assertEqual(caught.exception.c_s, float("inf"))

    def test_the_wall_velocity_refuses_it(self):
        with self.assertRaises(errors.SuperluminalSoundSpeedError) as caught:
            self._hydro(*self.INFINITE).calcWallVelocityLTE(100.0)
        self.assertEqual(caught.exception.c_s, float("inf"))

    def test_it_is_told_apart_from_a_phase_with_no_plasma(self):
        """The cases that must keep their old answers, or the refusal has swallowed them."""
        for label, pair in (("no plasma", self.NO_PLASMA), ("negative ratio", self.NEGATIVE)):
            with self.subTest(case=label):
                hydro = self._hydro(*pair)
                self.assertTrue(np.isnan(hydro.calc_cs(100.0, sym=False)))
                self.assertEqual(hydro.calcWallVelocityLTE(100.0), 1)

    def test_the_symmetric_phase_is_refused_too(self):
        hydro = self._hydro(*self.INFINITE)
        with self.assertRaises(errors.SuperluminalSoundSpeedError) as caught:
            hydro.calc_cs(100.0, sym=True)
        self.assertEqual(caught.exception.phase, "symmetric")

    def test_the_step_check_reports_it_rather_than_raising(self):
        """It must not propagate the refusal, and must not call this noise either.

        This fixture's derivatives do not depend on the step, so the value is infinite at
        every one of them. That is a breakdown of the potential's thermodynamics, which the
        refusal itself reports, and not an artefact of the derivative step; calling it noisy
        would point at a finite-difference problem that is not there. The change cannot be put
        as a ratio of two infinities, so it comes back as not-a-number.
        """
        noisy, change = self._hydro(*self.INFINITE).sound_speed_is_step_dependent(
            100.0, sym=False)
        self.assertFalse(noisy)
        self.assertTrue(np.isnan(change))


class DefaultDebyeMassTests(unittest.TestCase):
    """The framework measures the Debye masses itself, and refuses where it cannot."""

    @staticmethod
    def _model(name):
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from test_potential_broadcasting import build
        with contextlib.redirect_stdout(io.StringIO()):
            return build(name)

    @staticmethod
    def _X(pot):
        return np.atleast_1d(np.asarray(pot.X0, dtype=float)).ravel()[:pot.Ndim]

    # Quadratic thermal part, so the coefficient can be read off and used at any temperature.
    QUADRATIC = ("dark_U1", "dark_U1_g", "template")
    # Masses that are eigenvalues of a mixing matrix, where no such coefficient exists.
    MIXING = ("2HDM", "flipflop")

    def test_it_matches_the_subtraction_where_the_subtraction_is_sound(self):
        for name in self.QUADRATIC:
            with self.subTest(model=name):
                pot = self._model(name)
                X = self._X(pot)
                T = np.asarray([0.1 * pot.v_stable])
                Pi = pot.debye_massSq(X, T)
                self.assertIsNotNone(Pi, "the default declined a quadratic thermal part")
                subtraction = (np.asarray(pot.boson_massSq(X, T)[0], dtype=float)
                               - np.asarray(pot.boson_massSq(X, T * 0.0)[0], dtype=float))
                a, b = np.ravel(np.asarray(Pi, dtype=float)), np.ravel(subtraction)
                nonzero = np.abs(b) > 0.0
                self.assertTrue(nonzero.any(), "no mode had a thermal mass to compare")
                np.testing.assert_allclose(a[nonzero], b[nonzero], rtol=1.0e-9)

    def test_it_declines_where_the_masses_mix(self):
        """Not a gap: for these the daisy term genuinely needs the two spectra.

        Where the bosonic masses are eigenvalues of a matrix whose entries go as `T^2`, the
        eigenvalues do not, so there is no `Pi` to hand over and the subtraction is the only
        route. Handing one over anyway would make the daisy term wrong rather than imprecise.
        """
        for name in self.MIXING:
            with self.subTest(model=name):
                pot = self._model(name)
                self.assertIsNone(pot.debye_massSq(self._X(pot), np.asarray([30.0])))

    def test_the_quadratic_law_is_checked_at_more_than_one_field_point(self):
        """`flipflop` is quadratic along X0 and not away from it, and must still be refused.

        Its second field vanishes at `X0`, where its mass matrix is diagonal and the law holds
        to `7e-18`; at a point with both fields on it is off by `5e-3`. A check at one field
        point would accept it and hand the daisy term a `Pi` wrong by half a per cent.
        """
        pot = self._model("flipflop")
        scale = pot.v_stable
        reference = self._X(pot)

        def coefficient(X, T):
            zero = np.asarray(pot.boson_massSq(X, np.asarray(0.0))[0], dtype=float)
            hot = np.asarray(pot.boson_massSq(X, np.asarray(T))[0], dtype=float)
            return (hot - zero) / T ** 2

        along = np.max(np.abs(coefficient(reference, scale)
                              - coefficient(reference, 0.5 * scale)))
        off_axis_point = 0.37 * reference + 0.11 * scale
        off_axis = np.max(np.abs(coefficient(off_axis_point, scale)
                                 - coefficient(off_axis_point, 0.5 * scale)))
        self.assertLess(along, 1.0e-15, "X0 was meant to be the deceptive direction")
        self.assertGreater(off_axis, 1.0e-4, "the off-axis point was meant to break the law")
        self.assertIsNone(pot.debye_massSq(reference, np.asarray([30.0])))

    def test_the_verdict_does_not_depend_on_the_field_it_first_saw(self):
        """`Vtot` is called with random field points while the model is built.

        A verdict read off whichever point arrived first would differ between runs of the same
        input, so it is taken at field points the model itself fixes.
        """
        import numpy
        verdicts = []
        for probe in (None, "far", "origin"):
            pot = self._model("dark_U1")
            scale = pot.v_stable
            if probe == "far":
                pot.debye_massSq(np.asarray([7.3 * scale]), np.asarray([0.5 * scale]))
            elif probe == "origin":
                pot.debye_massSq(np.asarray([0.0]), np.asarray([0.5 * scale]))
            value = pot.debye_massSq(self._X(pot), np.asarray([0.1 * scale]))
            verdicts.append(None if value is None
                            else np.ravel(np.asarray(value, dtype=float)).copy())
        self.assertTrue(all(v is not None for v in verdicts),
                        "the verdict changed with the field point seen first")
        for other in verdicts[1:]:
            np.testing.assert_allclose(other, verdicts[0], rtol=0.0, atol=0.0)

    def test_a_field_dependent_model_is_rechecked_where_it_is_asked(self):
        """The two fixed probes decide cacheability, not licence to extrapolate anywhere.

        A model can be quadratic where it was probed and not elsewhere. No model shipped here
        takes this path, because the four that pass the check all have a field-independent
        coefficient, so it is built here: a spectrum that is quadratic at the two reference
        fields and cubic in temperature away from them. The coefficient must be refused there
        and still provided at the reference fields.
        """
        pot = self._model("dark_U1")
        scale = pot.v_stable
        reference = self._X(pot)
        elsewhere = 0.37 * reference + 0.11 * scale
        probed = (float(reference[0]), float(elsewhere[0]))
        real = type(pot).boson_massSq

        def quadratic_only_where_probed(self, X, t):
            masses, dof, c, phys = real(self, X, t)
            here = float(np.ravel(np.asarray(X, dtype=float))[0])
            t2 = np.asarray(t, dtype=float) ** 2
            masses = np.array(masses, dtype=float, copy=True)
            if abs(here - probed[0]) < 1.0e-6 * scale:
                return masses, dof, c, phys
            if abs(here - probed[1]) < 1.0e-6 * scale:
                # Still quadratic, but with a different coefficient, so the two probes
                # disagree and the model is classified as field dependent rather than cached.
                masses[..., 0] = masses[..., 0] + 0.25 * t2
                return masses, dof, c, phys
            # Anywhere else the thermal part picks up a T^3 piece, so no single coefficient
            # reproduces it at both temperatures and it must be refused.
            masses[..., 0] = masses[..., 0] + 1.0e-6 * np.asarray(t, dtype=float) ** 3
            return masses, dof, c, phys

        # Building the model already called `Vtot`, and with it the classification, so the
        # verdict is cleared to let it be taken under the spectrum below.
        pot._debye_state = None
        with mock.patch.object(type(pot), "boson_massSq", quadratic_only_where_probed):
            at_reference = pot.debye_massSq(reference, np.asarray([0.1 * scale]))
            away = pot.debye_massSq(np.asarray([0.613 * scale]), np.asarray([0.1 * scale]))
        self.assertEqual(pot._debye_state, "field_dependent",
                         "the fixture was meant to be classified field dependent")
        self.assertIsNotNone(at_reference, "the probed field should still be served")
        self.assertIsNone(away, "a field where the law fails was handed a synthesised Pi")

    def test_a_small_coefficient_is_judged_against_itself(self):
        """A single tolerance scaled to the largest coefficient only ever tests that one.

        No model shipped here distinguishes the two rules: all eight get the same verdict
        either way and the worst per-mode deviation among those that pass is `8e-15`. The case
        has to be constructed, and it is worth guarding because a heavy mode can make a small
        Debye coefficient matter in the daisy term at low temperature.
        """
        from transitionlistener.generic_potential import _debye_agrees
        reference = np.array([1.0, 1.0e-12])
        magnitude = 1.0

        # The small mode is wrong by half of itself, and still sits well inside a tolerance of
        # 1e-9 times the largest coefficient, which is what made that rule blind to it.
        broken = np.array([1.0, 1.5e-12])
        self.assertLess(np.max(np.abs(broken - reference)), 1.0e-9 * magnitude,
                        "the fixture was meant to slip past a global tolerance")
        self.assertFalse(_debye_agrees(broken, reference, magnitude))

        agreeing = np.array([1.0, 1.0e-12 * (1.0 + 1.0e-12)])
        self.assertTrue(_debye_agrees(agreeing, reference, magnitude))

        # A coefficient of exactly zero, which the transverse gauge bosons have, has nothing
        # to be relative to and is covered by the round-off floor instead.
        with_zero = np.array([1.0, 0.0])
        self.assertTrue(_debye_agrees(np.array([1.0, 1.0e-18]), with_zero, magnitude))
        self.assertFalse(_debye_agrees(np.array([1.0, 1.0e-10]), with_zero, magnitude))

    def test_a_non_quadratic_small_mode_is_refused_end_to_end(self):
        # And the helper is actually what the classification uses, not merely present.
        pot = self._model("dark_U1")
        scale = pot.v_stable
        real = type(pot).boson_massSq

        def tiny_non_quadratic_mode(self, X, t):
            masses, dof, c, phys = real(self, X, t)
            masses = np.array(masses, dtype=float, copy=True)
            t = np.asarray(t, dtype=float)
            # Small against the other Debye masses, and cubic rather than quadratic.
            masses[..., 2] = masses[..., 2] + 1.0e-9 * scale * t ** 3
            return masses, dof, c, phys

        pot._debye_state = None
        with mock.patch.object(type(pot), "boson_massSq", tiny_non_quadratic_mode):
            value = pot.debye_massSq(self._X(pot), np.asarray([0.1 * scale]))
        self.assertIsNone(value, "a mode that is not quadratic was accepted as if it were")

    def test_a_cached_coefficient_survives_a_second_call(self):
        # The cached value is an array, and comparing an array against the string states is
        # elementwise; branching on that raises. It broke model construction, which calls
        # `Vtot` more than once.
        pot = self._model("dark_U1")
        X, T = self._X(pot), np.asarray([0.1 * pot.v_stable])
        first = pot.debye_massSq(X, T)
        self.assertIsNotNone(first)
        second = pot.debye_massSq(X, T)
        np.testing.assert_allclose(np.ravel(np.asarray(second, dtype=float)),
                                   np.ravel(np.asarray(first, dtype=float)),
                                   rtol=0.0, atol=0.0)

    def test_it_reproduces_a_model_that_states_its_own(self):
        """The conformal dark U(1) writes its Debye masses in closed form.

        That makes it the one case where the measured default can be checked against an
        analytic answer rather than against the subtraction it is meant to replace.
        """
        pot = conformal()
        X = np.array([900.0])
        T = np.asarray([0.3 * pot.v_stable])
        analytic = np.ravel(np.asarray(pot.debye_massSq(X, T), dtype=float))
        with mock.patch.object(type(pot), "debye_massSq",
                               generic_potential.generic_potential.debye_massSq):
            measured = pot.debye_massSq(X, T)
        self.assertIsNotNone(measured, "the default declined a model with a closed form")
        np.testing.assert_allclose(np.ravel(np.asarray(measured, dtype=float)),
                                   analytic, rtol=1.0e-12)


class CustomPotentialHookTests(unittest.TestCase):
    """A model that rewrites the effective potential states its own thermal part.

    `V_thermal` is the hook, and it is a separate one from `V1T_from_X` on purpose. The class
    documentation calls `V1T_from_X` the temperature-dependent part of `Vtot`, which would
    include the radiation bath, while the base implementation returns neither the bath nor the
    daisy term; nothing in the signature says which an override means. Adding the bath to one
    that has it double counts, and not adding it to one that does not drops it, so neither
    guess is safe and the model is asked instead.
    """

    def test_an_overridden_thermal_part_is_the_one_used(self):
        pot = conformal()
        calls = {"n": 0}

        def replacement(self, X, T, include_decoupled=False):
            calls["n"] += 1
            return -3.0 * np.asarray(T, dtype=float) ** 4

        with mock.patch.object(type(pot), "V_thermal", replacement):
            value = float(np.squeeze(pot.V_thermal(np.array([900.0]), 30.0)))
        self.assertGreater(calls["n"], 0)
        self.assertAlmostEqual(value, -3.0 * 30.0 ** 4, places=9)

    def test_the_sound_speed_follows_it(self):
        """A thermal part going as `T^4` has `c_s^2 = 1/3` exactly.

        Nothing is added to the override here, so unlike the model's own thermal part there is
        no radiation bath with counted degrees of freedom in it, and the value is the clean one.
        """
        pot = conformal()

        def replacement(self, X, T, include_decoupled=False):
            return -3.0 * np.asarray(T, dtype=float) ** 4

        hydro = ThermalDerivativeTests._hydro(ThermalDerivativeTests(), pot)
        untouched = hydro.calc_cs(22.12, sym=False)
        with mock.patch.object(type(pot), "V_thermal", replacement):
            cs = hydro.calc_cs(22.12, sym=False)
        self.assertAlmostEqual(cs, 1.0 / np.sqrt(3.0), places=6)
        self.assertNotAlmostEqual(cs, untouched, places=4)

    def test_rewriting_the_potential_without_it_is_refused(self):
        """Refused rather than reconstructed, which would not be the model's own thermal part.

        Both documented entry points are checked, because a model may have rewritten either.
        """
        for name in ("Vtot", "V1T_from_X"):
            with self.subTest(overrides=name):
                pot = conformal()
                with mock.patch.object(type(pot), name,
                                       lambda self, *a, **k: np.asarray(0.0)):
                    with self.assertRaises(errors.PotentialError) as caught:
                        pot.V_thermal(np.array([900.0]), 30.0)
                message = str(caught.exception)
                self.assertIn(name, message)
                self.assertIn("V_thermal", message, "the message must name the hook to use")

    def test_an_ordinary_model_is_not_refused(self):
        # The refusal must key on the override and not fire for every model.
        pot = conformal()
        self.assertTrue(np.isfinite(float(np.squeeze(
            pot.V_thermal(np.array([900.0]), 30.0)))))


class MasslessModeTests(unittest.TestCase):
    """The lightest mode is the lightest mode, including when it is massless."""

    def _phases(self, pot, T):
        from transitionlistener.phases import Phases
        with contextlib.redirect_stdout(io.StringIO()):
            phases = Phases(pot, False).phases
        usable = [p for p in phases.values() if p.Tmin <= T <= p.Tmax]
        key = lambda p: float(np.linalg.norm(np.atleast_1d(np.squeeze(p.valAt(T)))))
        return (np.atleast_1d(np.squeeze(min(usable, key=key).valAt(T))),
                np.atleast_1d(np.squeeze(max(usable, key=key).valAt(T))))

    def test_a_symmetric_phase_still_reports_its_ratio(self):
        """Every zero-temperature mass vanishes there, so a strict `m2 > 0` filter kept none.

        The diagnostic then returned `(False, nan)` and the ratio was lost from exactly the
        case a symmetric-phase refusal would want to explain.
        """
        pot = conformal()
        for ratio_T in (1.0e-1, 1.0e-3, 1.0e-5):
            T = ratio_T * pot.v_stable
            symmetric, _ = self._phases(pot, T)
            with self.subTest(T_over_v=ratio_T):
                m2 = np.ravel(np.asarray(pot.boson_massSq(symmetric, np.asarray([0.0]))[0],
                                         dtype=float))
                self.assertTrue(np.all(m2 == 0.0), "this phase was meant to be massless")
                outside, value = pot.daisy_outside_validity(symmetric, T)
                self.assertTrue(np.isfinite(value), "the ratio was discarded with the modes")
                self.assertGreater(value, 0.0)
                # Massless modes are the regime the resummation is for, so it must stay quiet.
                self.assertFalse(outside)

    def test_a_massless_mode_makes_the_spectrum_light(self):
        # With a zero mode present the lightest mass is zero whatever else is in the spectrum,
        # so the flag must not fire however heavy the rest is.
        pot = conformal()
        T = 1.0e-9 * pot.v_stable
        _, broken = self._phases(pot, T)
        outside_before, ratio_before = pot.daisy_outside_validity(broken, T)
        self.assertTrue(outside_before, "the heavy broken phase was meant to fire")

        real = type(pot).boson_massSq

        def with_a_zero_mode(self, X, t):
            m2, dof, c, phys = real(self, X, t)
            m2 = np.array(m2, dtype=float, copy=True)
            m2[..., 0] = 0.0
            return m2, dof, c, phys

        with mock.patch.object(type(pot), "boson_massSq", with_a_zero_mode):
            outside_after, ratio_after = pot.daisy_outside_validity(broken, T)
        self.assertFalse(outside_after, "a massless mode must keep the flag quiet")
        self.assertTrue(np.isfinite(ratio_after))

    def test_a_mode_with_no_debye_mass_does_not_count(self):
        """A transverse gauge boson is massless and thermally uncorrected, in every model here.

        It has degrees of freedom but adds exactly nothing to the daisy term, so counting it
        pinned the lightest mass at zero permanently and the flag could never fire. On
        `models/TL_2HDM.py` that is the transverse photon.
        """
        pot = conformal()
        T = 1.0e-9 * pot.v_stable
        _, broken = self._phases(pot, T)
        Tref = np.asarray([pot.v_stable])
        thermal = np.ravel(np.asarray(pot.debye_massSq(broken, Tref), dtype=float))
        m2 = np.ravel(np.asarray(pot.boson_massSq(broken, np.asarray([0.0]))[0], dtype=float))
        uncorrected = np.nonzero(thermal == 0.0)[0]
        self.assertTrue(uncorrected.size, "this model was meant to have a transverse mode")

        # Give that mode a vanishing mass, as a transverse gauge boson has in the symmetric
        # phase. It still has no Debye mass, so it must not be allowed to call the spectrum
        # light; every other mode here is far heavier than the temperature.
        real = type(pot).boson_massSq

        def massless_transverse(self, X, t):
            masses, dof, c, phys = real(self, X, t)
            masses = np.array(masses, dtype=float, copy=True)
            masses[..., uncorrected[0]] = 0.0
            return masses, dof, c, phys

        with mock.patch.object(type(pot), "boson_massSq", massless_transverse):
            outside, ratio = pot.daisy_outside_validity(broken, T)
        self.assertTrue(outside, "a mode with no Debye mass was allowed to count as light")
        self.assertTrue(np.isfinite(ratio))
        # And the same mode with a Debye mass would count, which is what separates the two.
        self.assertGreater(m2[thermal != 0.0].min(), 0.0)

    def test_modes_without_degrees_of_freedom_do_not_count(self):
        # The criterion is about modes that contribute to the daisy term; one with no degrees
        # of freedom contributes nothing and must not be able to make the spectrum look light.
        pot = conformal()
        T = 1.0e-9 * pot.v_stable
        _, broken = self._phases(pot, T)
        real = type(pot).boson_massSq

        def with_a_weightless_zero_mode(self, X, t):
            m2, dof, c, phys = real(self, X, t)
            m2 = np.array(m2, dtype=float, copy=True)
            dof = np.array(dof, dtype=float, copy=True)
            m2[..., 0] = 0.0
            dof[0] = 0.0
            return m2, dof, c, phys

        with mock.patch.object(type(pot), "boson_massSq", with_a_weightless_zero_mode):
            outside, _ = pot.daisy_outside_validity(broken, T)
        self.assertTrue(outside, "a mode with no degrees of freedom was allowed to count")


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
            1.48, 1.0e-5, phase="broken", step_change=0.4, daisy_ratio=1.5e7,
            daisy_outside=True)
        self.assertEqual(err.step_change, 0.4)
        self.assertEqual(err.daisy_ratio, 1.5e7)
        self.assertIs(err.daisy_outside, True)
        self.assertIn("losing", str(err))
        self.assertIn("outside its range", str(err))
        quiet = errors.SuperluminalSoundSpeedError(
            1.48, 1.0e-5, step_change=1.0e-5, daisy_ratio=0.2, daisy_outside=False)
        self.assertIn("not a finite-difference artefact", str(quiet))
        # Ratio below one: the verdict is false because the daisy term has not taken over, and
        # that says nothing about the masses, so the message must not claim light modes.
        self.assertIn("has not overtaken the radiation", str(quiet))
        self.assertNotIn("light enough", str(quiet))
        light = errors.SuperluminalSoundSpeedError(
            1.48, 1.0e-5, daisy_ratio=50.0, daisy_outside=False)
        self.assertIn("light enough", str(light))
        # The step check returns nan when one of the test steps is not a speed either, which
        # is the case where the step dependence is strongest. Saying nothing there would
        # leave that one case without a statement.
        unknown = errors.SuperluminalSoundSpeedError(
            1.48, 1.0e-5, step_change=float("nan"), daisy_ratio=0.2)
        self.assertIn("not a speed at one of the test steps", str(unknown))
        self.assertNotIn("stable under", str(unknown))

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
        """The refusal belongs to `calc_cs`, not to the check; and it is not itself noise.

        This fixture's derivatives do not depend on the step, so its sound speed is the same
        at every one of them. It is not a speed, but it is perfectly step independent, and
        reporting otherwise would send the user looking for a finite-difference problem that
        is not there. The check must come back quiet, and must not raise.
        """
        hydro = SuperluminalTests._hydro_returning(SuperluminalTests(), 2.9)
        noisy, change = hydro.sound_speed_is_step_dependent(100.0, sym=False)
        self.assertFalse(noisy)
        self.assertAlmostEqual(change, 0.0, places=12)

    def test_a_refusal_at_some_steps_but_not_others_is_noise(self):
        """That is the case the previous behaviour was reaching for, and it is a real one.

        A value that is a speed at one step and not at another has stopped being one somewhere
        inside the step range, which cannot be read off any single step.
        """
        hydro = SuperluminalTests._hydro_returning(SuperluminalTests(), 2.9)
        real = hydro._calc_cs_at_step

        def subluminal_when_coarse(T, sym, step_factor=1.0):
            if step_factor > 1.0:
                return 0.5, T * 1.0e-3
            return real(T, sym, step_factor)

        with mock.patch.object(hydro, "_calc_cs_at_step", subluminal_when_coarse):
            noisy, change = hydro.sound_speed_is_step_dependent(100.0, sym=False)
        self.assertTrue(noisy)
        self.assertTrue(np.isnan(change))

    def test_a_refused_value_that_moves_with_the_step_is_noise(self):
        # Refused at every step, but not the same value at every step: that is ordinary step
        # dependence and is reported as such, on the refused values.
        hydro = SuperluminalTests._hydro_returning(SuperluminalTests(), 2.9)
        real = hydro._calc_cs_at_step

        # Every step stays above one, so the refusal status is the same at all three and only
        # the refused value moves. A factor that dipped below one would be the other case,
        # which `test_a_refusal_at_some_steps_but_not_others_is_noise` covers.
        by_factor = {1.0: 2.9, SOUND_SPEED_STEP_RATIO: 2.9 * 9.0,
                     1.0 / SOUND_SPEED_STEP_RATIO: 2.9 * 4.0}

        def drifting(T, sym, step_factor=1.0):
            scaled = by_factor[step_factor]
            inner = SuperluminalTests._hydro_returning(SuperluminalTests(), scaled)
            return inner._calc_cs_at_step(T, sym, 1.0)

        with mock.patch.object(hydro, "_calc_cs_at_step", drifting):
            noisy, change = hydro.sound_speed_is_step_dependent(100.0, sym=False)
        self.assertTrue(noisy)
        self.assertGreater(change, 0.05)

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
