"""The two error codes left inside the bounce solver, and the high-temperature check.

A census of `errors.py` against the test suite left three raise sites that no test drove:

* code 11 `TunnelingError`, three sites in `tunneling1D.py`, inside `initialConditions` and
  `findProfile`. Each wraps a failure of another routine and re-raises it with a message that
  names the thin wall as the likely cause.
* code 13 `SplineError`, four sites in `pathDeformation.py`. Two are the cap on how many knots
  the path may be extended by, `_MAX_EXTENSION_KNOTS`; two re-raise a failure of the spline
  fit.
* code 14 `WrongHighTPhaseError` in `transitions.py`, where the first transition does not start
  from the high-temperature phase of the model. Its guard `do_high_T_phase_check` defaults to
  `True`, so that branch is live in every default run.

These are the paths a user meets when a model is hard rather than when it is wrong, so the
message is the whole of what they get. Each test drives the real routine; none asserts on a
message the code does not produce.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import errors
from transitionlistener import pathDeformation as pd
from transitionlistener import transitions
from transitionlistener import tunneling1D


class TheSplineFitTranslatesItsFailuresTests(unittest.TestCase):
    """Code 13 comes from `_fit_spline_path`, which translates what `SplinePath` raises.

    The two `nx > _MAX_EXTENSION_KNOTS` raises inside `SplinePath.__init__` cannot fire: the
    search that produces `nx` walks a ladder whose last point IS `_MAX_EXTENSION_KNOTS`, and a
    minimum is accepted only when a rise is seen after it, which past the last point is never
    sampled. Measured: with the minimum beyond the ladder, at its end, and with a potential
    that falls for ever, `_extension_minimum` returns 0, 0 and 0. The test below pins the
    invariant that keeps those raises unreachable, so that raising the ladder's end without
    raising the cap would be caught.
    """

    PTS = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    def _fit(self, failure):
        settings = types.SimpleNamespace(V_spline_samples=None)
        with mock.patch.object(pd, "SplinePath", side_effect=failure):
            return pd._fit_spline_path(self.PTS, settings, lambda x: 0.0, None, False)

    def test_a_value_error_from_the_spline_is_code_13(self):
        with self.assertRaises(errors.SplineError) as caught:
            self._fit(ValueError("the knots are not increasing"))
        self.assertEqual(caught.exception.errorcode, 13)
        self.assertIn("the knots are not increasing", str(caught.exception))

    def test_a_path_carrying_a_nan_is_an_infinite_action_rather_than_a_spline_failure(self):
        """A `nan` in the path is the bounce failing, not the spline.

        `InfiniteActionError` carries no `errorcode` at all, unlike `SplineError`: it is not
        one of the numbered outcomes a point can be reported with, it is how the bounce says it
        found no finite action. So this is not a different code, it is no code.
        """
        settings = types.SimpleNamespace(V_spline_samples=None)
        pts = np.array([[0.0, 0.0], [np.nan, 0.0], [2.0, 0.0]])
        with mock.patch.object(pd, "SplinePath", side_effect=ValueError("knots")):
            with self.assertRaises(errors.InfiniteActionError):
                pd._fit_spline_path(pts, settings, lambda x: 0.0, None, False)

    def test_an_oversized_spline_is_explained_rather_than_echoed(self):
        """scipy says only "Maximum allowed size exceeded"; the user needs the cause."""
        with self.assertRaises(errors.SplineError) as caught:
            self._fit(RuntimeError("Maximum allowed size exceeded"))
        message = " ".join(str(caught.exception).split())
        self.assertIn("phase.valAt(T)", message)
        self.assertNotEqual(message, "Maximum allowed size exceeded")

    def test_two_coincident_endpoints_are_no_barrier_rather_than_a_spline_failure(self):
        """The two minima having met is a statement about the potential, not the fit."""
        settings = types.SimpleNamespace(V_spline_samples=None)
        pts = np.array([[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]])
        with mock.patch.object(pd, "SplinePath", side_effect=RuntimeError("Invalid inputs.")):
            with self.assertRaises(errors.PotentialError) as caught:
                pd._fit_spline_path(pts, settings, lambda x: 0.0, None, False)
        self.assertEqual(caught.exception.error_type, "no barrier")
        self.assertIn("identical", str(caught.exception))

    def test_distinct_endpoints_with_the_same_message_stay_a_spline_failure(self):
        """"Invalid inputs." alone does not mean the barrier is gone."""
        with self.assertRaises(errors.SplineError) as caught:
            self._fit(RuntimeError("Invalid inputs."))
        self.assertIn("phitol", str(caught.exception))

    def test_the_run_deadline_passes_through(self):
        with self.assertRaises(errors.Timeout):
            self._fit(errors.Timeout("the run's deadline"))

    def test_the_path_extension_never_returns_more_knots_than_the_cap(self):
        """The invariant that makes the two `nx > _MAX_EXTENSION_KNOTS` raises unreachable."""
        p0, dp0 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        cap = float(pd._MAX_EXTENSION_KNOTS)
        for name, target in (("beyond the ladder", -3.0 * cap),
                             ("at the ladder's end", -cap),
                             ("inside the ladder", -0.5 * cap)):
            with self.subTest(minimum=name):
                def V(x, t=target):
                    return ((np.asarray(x, dtype=float)[..., 0] - t) ** 2) * 1e-12
                xmin = pd._extension_minimum(V, p0, dp0, -1.0)
                self.assertLessEqual(abs(xmin), cap)
                self.assertLessEqual(np.ceil(abs(xmin) - 0.5) + 1, cap,
                                     "the extension can now exceed the knot cap, so the "
                                     "raises in SplinePath.__init__ are reachable and need "
                                     "a test of their own")


class TheThinWallMessageOfTheShootingTests(unittest.TestCase):
    """Code 11 from `initialConditions`, where the starting radius cannot be bracketed.

    `initialConditions` brackets the radius at which the profile has moved `delta_phi_cutoff`
    away from the true vacuum, with `brentq`. The guard is around that bracket and nothing
    else: `exactSolution` is also called before the `try`, so a failure there escapes
    untranslated and patching it does not reach this path. Where the bracket fails the solver
    reports a wall too thin to integrate, which is the usual cause. The message is the only thing that reaches the user, because
    `pathDeformation._find_profile` turns this into an infinite action and `bounceAction`
    reports no tunnelling.
    """

    @staticmethod
    def _instanton():
        """A quartic with two minima, the shallow one metastable, as the solver expects."""
        def V(phi):
            phi = np.asarray(phi, dtype=float)
            return 0.25 * phi**4 - 0.49 * phi**3 + 0.235 * phi**2

        def dV(phi):
            phi = np.asarray(phi, dtype=float)
            return phi**3 - 1.47 * phi**2 + 0.47 * phi

        def d2V(phi):
            phi = np.asarray(phi, dtype=float)
            return 3.0 * phi**2 - 2.94 * phi + 0.47

        return tunneling1D.SingleFieldInstanton(1.0, 0.0, V, dV, d2V)

    def test_a_failure_of_the_exact_solution_is_code_11(self):
        inst = self._instanton()
        with mock.patch.object(tunneling1D.optimize, "brentq",
                               side_effect=ValueError("no solution here")):
            with self.assertRaises(errors.TunnelingError) as caught:
                inst.initialConditions(1e-3, 1e-4, 1e-2)
        self.assertEqual(caught.exception.errorcode, 11)

    def test_the_message_names_the_thin_wall_and_carries_the_original_failure(self):
        inst = self._instanton()
        with mock.patch.object(tunneling1D.optimize, "brentq",
                               side_effect=ValueError("no solution here")):
            with self.assertRaises(errors.TunnelingError) as caught:
                inst.initialConditions(1e-3, 1e-4, 1e-2)
        message = str(caught.exception)
        self.assertIn("thin-walled", message)
        self.assertIn("no solution here", message,
                      "the original failure is what says why the bracket could not be found")

    def test_the_run_deadline_is_not_read_as_a_thin_wall(self):
        """`Timeout` is raised from a signal handler and subclasses `Exception`.

        Absorbed here it would come back as a physics verdict on the wall, so a point that ran
        out of time would be indistinguishable from one the solver judged.
        """
        inst = self._instanton()
        with mock.patch.object(tunneling1D.optimize, "brentq",
                               side_effect=errors.Timeout("the run's deadline")):
            with self.assertRaises(errors.Timeout):
                inst.initialConditions(1e-3, 1e-4, 1e-2)

    def test_an_ordinary_potential_returns_initial_conditions(self):
        """The control: the message above must not be what this routine usually does."""
        inst = self._instanton()
        r0, phi, dphi = inst.initialConditions(1e-3, 1e-4, 1e-2)
        self.assertTrue(np.isfinite(r0))
        self.assertTrue(np.isfinite(phi))


class TheFirstTransitionMustStartFromTheHighTemperaturePhaseTests(unittest.TestCase):
    """Code 14 from `transitions.py`, the site its `phases.py` sibling does not cover.

    `do_high_T_phase_check` defaults to `True` in `config.py`, where its comment calls it a
    hotfix for the vev flip-flop model, so this branch runs in every default scan and no test
    drove it. The message asserts that reaching it "indicates a bug in the code"; that is a
    claim about the cause, and the test pins what the code does rather than endorsing it.
    """

    @staticmethod
    def _stub(check_on, first_high_phase):
        tr = types.SimpleNamespace(type=1, Tcrit=120.0, high_phase=first_high_phase,
                                   low_phase="0", Tnuc=90.0, action=152.6, instanton={},
                                   full_tunneling_info={}, high_vev=np.array([1.0, 0.5]),
                                   low_vev=np.array([0.0, 0.0]), high_vev_as_traced=None,
                                   step=0)
        phases = {"1": types.SimpleNamespace(key="1", T=np.array([10.0, 615.5])),
                  "0": types.SimpleNamespace(key="0", T=np.array([0.0, 200.0]))}
        pot = types.SimpleNamespace(
            Tmax=615.5,
            config=types.SimpleNamespace(
                percolationConf=types.SimpleNamespace(algorithm_mode="fixed_step_size"),
                tracingConf=types.SimpleNamespace(do_high_T_phase_check=check_on)))
        return tr, types.SimpleNamespace(
            pot=pot, phases=phases, verbose=False, tunnelFromPhase_args={},
            conversionFactor=1.0, V=lambda x, T: 0.0, dV=None, d2V=None,
            findTransitionPathFromTnuc=lambda verbose=False: [tr])

    def _run(self, check_on, first_high_phase):
        import contextlib
        import io

        tr, stub = self._stub(check_on, first_high_phase)
        result = dict(Tnuc=90.0, low_phase="0", high_phase=first_high_phase,
                      instanton={}, trantype=1, action=152.6, full_tunneling_info={})
        with mock.patch.object(transitions, "computeNucleationTemperature",
                               lambda *a, **k: result):
            with contextlib.redirect_stdout(io.StringIO()):
                return transitions.Transitions.findAllTransitions(stub)

    def test_a_first_transition_from_another_phase_is_code_14(self):
        with self.assertRaises(errors.WrongHighTPhaseError) as caught:
            self._run(check_on=True, first_high_phase="0")
        self.assertEqual(caught.exception.errorcode, 14)

    def test_the_message_says_what_the_check_found(self):
        with self.assertRaises(errors.WrongHighTPhaseError) as caught:
            self._run(check_on=True, first_high_phase="0")
        message = " ".join(str(caught.exception).split())
        self.assertIn("does not start from the high-T phase", message)

    def test_the_first_transition_from_the_high_temperature_phase_passes(self):
        found = self._run(check_on=True, first_high_phase="1")
        self.assertEqual(len(found), 1)

    def test_the_switch_turns_the_check_off(self):
        """`do_high_T_phase_check` is a hotfix toggle, so off has to mean off."""
        found = self._run(check_on=False, first_high_phase="0")
        self.assertEqual(len(found), 1)


if __name__ == "__main__":
    unittest.main()
