"""The guard on the sound speed that feeds the bubble expansion speed.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from transitionlistener import config
from transitionlistener.hydrodynamics import (
    RADIATION_SOUND_SPEED, physical_sound_speed)
from transitionlistener.interface.samplers import get_empty_result


class PhysicalSoundSpeedTests(unittest.TestCase):
    """Only values that are not speeds are replaced."""

    def test_an_ordinary_sound_speed_is_kept(self):
        for value in (0.1, 0.5, RADIATION_SOUND_SPEED, 0.9, 0.999999):
            with self.subTest(c_s=value):
                out, replaced = physical_sound_speed(value)
                self.assertAlmostEqual(out, value, places=12)
                self.assertFalse(replaced)

    def test_a_value_just_above_the_radiation_speed_is_kept(self):
        """The tighter 1/sqrt(3) bound is deliberately NOT applied here.

        It is the physical bound for a plasma whose particles have masses, but it fires on
        108 of 286 campaign runs where the excess is finite-difference accuracy, and on the
        released example point, whose sound speed is 0.58951, 2.1% above 1/sqrt(3). Clamping
        there would move the efficiency factors and the spectrum of points that have a
        perfectly ordinary thermal broken phase.
        """
        for value in (RADIATION_SOUND_SPEED * (1 + 1e-8), 0.5776, 0.58951, 0.7):
            with self.subTest(c_s=value):
                out, replaced = physical_sound_speed(value)
                self.assertAlmostEqual(out, value, places=12)
                self.assertFalse(replaced)

    def test_a_superluminal_value_is_replaced(self):
        # The seven points of the conformal scan came out between 1.01 and 1.71.
        for value in (1.0, 1.01, 1.025, 1.054, 1.332, 1.706, 5.0):
            with self.subTest(c_s=value):
                out, replaced = physical_sound_speed(value)
                self.assertAlmostEqual(out, RADIATION_SOUND_SPEED, places=12)
                self.assertTrue(replaced)

    def test_values_that_are_not_numbers_at_all_are_replaced(self):
        for value in (0.0, -0.3, float("nan"), float("inf"), -float("inf"), None, "x"):
            with self.subTest(c_s=value):
                out, replaced = physical_sound_speed(value)
                self.assertAlmostEqual(out, RADIATION_SOUND_SPEED, places=12)
                self.assertTrue(replaced)

    def test_the_replacement_is_the_massless_value(self):
        self.assertAlmostEqual(RADIATION_SOUND_SPEED, 1.0 / np.sqrt(3.0), places=15)

    def test_a_runaway_wall_keeps_its_expansion_speed(self):
        """What the guard is for: max(v_wall, c_s) must stay at the wall velocity.

        A superluminal sound speed replaces the wall velocity through that maximum and
        inflates (beta/H)_RH by the factor c_s.
        """
        v_wall = 1.0
        for raw in (1.01, 1.332, 1.706):
            with self.subTest(c_s=raw):
                self.assertGreater(max(v_wall, raw), v_wall)          # the defect
                usable, _ = physical_sound_speed(raw)
                self.assertEqual(max(v_wall, usable), v_wall)         # after the guard


class SolverUsesTheGuardTests(unittest.TestCase):
    """Both solvers must put the guarded value into `c_s`, not the computed one.

    `c_s` is what the bubble expansion speed, the efficiency factors and the spectrum are
    built from. A unit test of the guard cannot tell whether the solvers apply it: with the
    assignment reverted to the raw value, every other test in this file still passed.
    """

    RAW = 1.706          # the worst point of the conformal scan

    def _ensure(self, module, verbose=False):
        from pathlib import Path
        from unittest import mock
        from transitionlistener.helper_functions import load_potential
        from transitionlistener import hydrodynamics

        repo = Path(__file__).resolve().parents[1]
        pot = load_potential(str(repo / "models/TL_conformal_dark_u1.py"),
                             "specific_potential")(
            {"g": 0.692, "y": 0.01, "v_GeV": 6.0}, verbose=False)

        class Phase:
            Tmin, Tmax = 1.0e-6, 1.0e6

            def valAt(self, T, deriv=0):
                T = np.asarray(T, float)
                return np.zeros(T.shape + (1,))

        derived = {}
        ctx = types.SimpleNamespace(
            derived_param_names=["c_s"], derived_params=derived,
            GWconfig=types.SimpleNamespace(sound_speed="compute"),
            pot=pot, phase_symmetric=Phase(), phase_broken=Phase(), verbose=verbose)

        def fake_cs(self, T, sym):
            return 0.4 if sym else self_raw[0]

        self_raw = [self.RAW]
        with mock.patch.object(hydrodynamics.Hydrodynamics, "calc_cs", fake_cs):
            module.TransitionObservables._ensure_sound_speed(
                module.TransitionObservables.__new__(module.TransitionObservables), ctx, 30.0)
        return derived

    def test_both_solvers_replace_a_superluminal_value_in_c_s(self):
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof
        for module in (to, tof):
            with self.subTest(module=module.__name__):
                derived = self._ensure(module)
                self.assertAlmostEqual(derived["c_s"], RADIATION_SOUND_SPEED, places=12)
                self.assertTrue(derived["WARNING:unphysical_c_s"])

    def test_both_solvers_still_report_the_computed_value(self):
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof
        for module in (to, tof):
            with self.subTest(module=module.__name__):
                derived = self._ensure(module)
                self.assertAlmostEqual(derived["c_s_bro"], self.RAW, places=12)

    def test_both_solvers_say_so_when_they_replace(self):
        """A silent replacement is the thing a user would want told.

        Both solvers print it under `verbose`, and the message names the computed value and
        the one substituted for it.
        """
        import contextlib as _ctx
        import io as _io
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof
        for module in (to, tof):
            with self.subTest(module=module.__name__):
                buffer = _io.StringIO()
                with _ctx.redirect_stdout(buffer):
                    self._ensure(module, verbose=True)
                printed = buffer.getvalue()
                self.assertIn("not a speed", printed)
                self.assertIn("1.706", printed)
                self.assertIn("0.577350", printed)

    def test_neither_solver_says_anything_when_it_does_not_replace(self):
        import contextlib as _ctx
        import io as _io
        from unittest import mock
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof
        for module in (to, tof):
            with self.subTest(module=module.__name__):
                buffer = _io.StringIO()
                with mock.patch.object(type(self), "RAW", 0.58951):
                    with _ctx.redirect_stdout(buffer):
                        self._ensure(module, verbose=True)
                self.assertNotIn("not a speed", buffer.getvalue())

    def test_an_ordinary_value_reaches_c_s_unchanged(self):
        from unittest import mock
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof
        for module in (to, tof):
            with self.subTest(module=module.__name__):
                with mock.patch.object(type(self), "RAW", 0.58951):
                    derived = self._ensure(module)
                self.assertAlmostEqual(derived["c_s"], 0.58951, places=12)
                self.assertFalse(derived["WARNING:unphysical_c_s"])


class ConfiguredSoundSpeedTests(unittest.TestCase):
    """A configured value must satisfy the same notion of a speed as a computed one.

    The fixed step size solver accepts a number in `GWconfig.sound_speed`. It used to allow
    exactly one, which the guard on the computed value classifies as not a speed, and which
    the spectrum cannot use: it divides by `v_wall - c_s`, so one with a runaway wall is a
    division by zero.
    """

    def _apply(self, value):
        from pathlib import Path
        from transitionlistener.helper_functions import load_potential
        from transitionlistener import transitionObservables_fixedstep as tof

        repo = Path(__file__).resolve().parents[1]
        pot = load_potential(str(repo / "models/TL_conformal_dark_u1.py"),
                             "specific_potential")(
            {"g": 0.692, "y": 0.01, "v_GeV": 6.0}, verbose=False)

        class Phase:
            Tmin, Tmax = 1.0e-6, 1.0e6

            def valAt(self, T, deriv=0):
                T = np.asarray(T, float)
                return np.zeros(T.shape + (1,))

        derived = {}
        ctx = types.SimpleNamespace(
            derived_param_names=["c_s"], derived_params=derived,
            GWconfig=types.SimpleNamespace(sound_speed=str(value)),
            pot=pot, phase_symmetric=Phase(), phase_broken=Phase(), verbose=False)
        cls = tof.TransitionObservables
        cls._ensure_sound_speed(cls.__new__(cls), ctx, 30.0)
        return derived

    def test_a_configured_value_in_range_is_used(self):
        for value in (0.4, 0.5, 0.9):
            with self.subTest(c_s=value):
                self.assertAlmostEqual(self._apply(value)["c_s"], value, places=12)

    def test_exactly_one_is_refused(self):
        # The spectrum divides by (v_wall - c_s); one with a runaway wall is a zero divide.
        with self.assertRaises(ValueError):
            self._apply(1.0)

    def test_values_outside_the_range_are_refused(self):
        for value in (0.0, -0.2, 1.5):
            with self.subTest(c_s=value):
                with self.assertRaises(ValueError):
                    self._apply(value)


class RegistrationTests(unittest.TestCase):
    def test_the_warning_is_registered_where_registration_is_required(self):
        empty = get_empty_result()
        self.assertIn("WARNING:unphysical_c_s", config.all_observables)
        self.assertIn("WARNING:unphysical_c_s", empty)
        # Failed scan points must keep producing rows that line up with successful ones.
        self.assertEqual(set(config.all_observables) - set(empty), set())

    def test_both_solvers_default_the_warning_to_false(self):
        """The key is registered for every run, so every context must carry it.

        Left unset, the writer reports it as unimplemented and fills a boolean column with
        not-a-number. Read from the context each solver builds, not from its source: a
        source-text assertion passes even with the entry commented out, because the key name
        is still there in the comment.
        """
        from pathlib import Path
        from transitionlistener.helper_functions import load_potential
        from transitionlistener import transitionObservables as to
        from transitionlistener import transitionObservables_fixedstep as tof

        repo = Path(__file__).resolve().parents[1]
        pot = load_potential(str(repo / "models/TL_conformal_dark_u1.py"),
                             "specific_potential")(
            {"g": 0.692, "y": 0.01, "v_GeV": 6.0}, verbose=False)

        class Phase:
            Tmin, Tmax = 1.0e-6, 1.0e6

            def valAt(self, T, deriv=0):
                T = np.asarray(T, float)
                return np.zeros(T.shape + (1,))

        phase = Phase()
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
                ctx = cls._initialize_transition_context(obs, tr)
                self.assertIs(ctx.derived_params["WARNING:unphysical_c_s"], False)


if __name__ == "__main__":
    unittest.main()
