"""The error paths that no test drove, and the two messages that said nothing usable.

A census of `errors.py` against the test suite found nine of the twenty error codes raised by
code that no test mentions. Four of them, 4, 5, 12 and 17, are covered elsewhere. This file
covers the rest: 1 `TachyonError`, 6 `NoTransitionFound`, 9 `OnlySecondOrderTransitionsError`,
14 `WrongHighTPhaseError` and 999 `UnexpectedError`. For each, what the user is shown and the
state the code is in when the message is written had never been exercised.

Two of the paths were wrong where they were never run.

`tachyonTest` reported `m20 / v_stable`: a squared mass divided by the internal field scale,
which is neither a squared mass nor a mass, and the number it printed therefore changed with a
model's choice of internal scale for the same physical spectrum. It also called the vector of
eigenvalues a "Mass matrix" and printed every entry without saying which one was tachyonic.

`pathDeformation.bounceAction` translates a `PotentialError` into a tunnelling result for the
two tags the package raises and reports anything else as `UnexpectedError`, code 999. It said
only "Unexpected error message" and gave no tag, so the one thing the handler knows about the
failure it cannot handle was left out. `_execute_full_tunneling` carries a second copy of the
same translation, where `err.args[1]` raised `IndexError` instead, since
`PotentialError.__init__` passes only `message` to `Exception.__init__`; pull request #50 fixes
and tests that copy, and this branch leaves it alone. The branch is reached only from a model
that raises `PotentialError` with a third tag, which the public constructor allows.
"""

from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import errors
from transitionlistener import generic_potential as gp
from transitionlistener import pathDeformation
from transitionlistener import phases as phases_module
from transitionlistener import transitions


def _tachyon_pot(masses_sq, physical, v_stable=1.0):
    """A potential that answers `boson_massSq` and nothing else.

    `tachyonTest` reads `self.boson_massSq` and `self.v_stable`, so it runs unbound against a
    namespace and needs no model file. The third element of the tuple is the counterterm
    coefficient, which the test never reaches.
    """
    masses_sq = np.asarray(masses_sq, dtype=float)
    dofs = np.ones_like(masses_sq)
    return types.SimpleNamespace(
        v_stable=v_stable,
        boson_massSq=lambda X, T: (masses_sq, dofs, np.zeros_like(masses_sq),
                                   np.asarray(physical, dtype=bool)),
    )


class TheTachyonMessageNamesTheModeTests(unittest.TestCase):
    X0 = np.array([1.0, 0.0])

    def _raise(self, masses_sq, physical, v_stable=1.0):
        pot = _tachyon_pot(masses_sq, physical, v_stable)
        with self.assertRaises(errors.TachyonError) as caught:
            gp.generic_potential.tachyonTest(pot, self.X0)
        return str(caught.exception)

    def test_the_code_is_one(self):
        """The command-line interface reports this number; nothing else records it."""
        pot = _tachyon_pot([-1.0], [True])
        with self.assertRaises(errors.TachyonError) as caught:
            gp.generic_potential.tachyonTest(pot, self.X0)
        self.assertEqual(caught.exception.errorcode, 1)

    def test_the_message_names_which_boson_is_tachyonic(self):
        """With four modes and one of them tachyonic, the user needs to know which."""
        msg = self._raise([4.0, 9.0, -2.5, 16.0], [True, True, True, True])
        self.assertIn("boson 2", msg)
        self.assertIn("-2.5", msg)
        for other in ("boson 0", "boson 1", "boson 3"):
            self.assertNotIn(f"{other}: m^2", msg,
                             "only the tachyonic modes are listed by name")

    def test_every_tachyonic_mode_is_named_not_only_the_first(self):
        msg = self._raise([-2.5, 9.0, -7.5], [True, True, True])
        self.assertIn("boson 0", msg)
        self.assertIn("boson 2", msg)
        self.assertIn("-7.5", msg)

    def test_the_numbers_do_not_depend_on_the_internal_field_scale(self):
        """The same physical spectrum must read the same whatever the model's scale is.

        This is what `m20 / v_stable` got wrong: it is linear in the scale, so one model
        reported ten times another's number for the same squared mass over squared scale. The
        two cases below are the same physics expressed at scales 1 and 10.
        """
        at_one = self._raise([-2.5, 4.0], [True, True], v_stable=1.0)
        at_ten = self._raise([-250.0, 400.0], [True, True], v_stable=10.0)
        self.assertIn("m^2/v^2 = -2.5", at_one)
        self.assertIn("m^2/v^2 = -2.5", at_ten)

    def test_the_message_states_the_threshold_in_the_units_it_reports(self):
        """A reader comparing the listed number against the cut needs both in one unit.

        At `v_stable` = 2 the listed mass is -2.5/4 = -0.625 and the threshold is -1e-5/4 =
        -2.5e-06, so the assertion names the word before it: a bare "-2.5e-06" would also pass
        on a message that printed it as the mass.
        """
        msg = self._raise([-2.5], [True], v_stable=2.0)
        self.assertIn("below m^2/v^2 = -2.5e-06", msg)
        self.assertIn("boson 0: m^2/v^2 = -0.625", msg)

    def test_the_message_gives_the_minimum_it_tested(self):
        msg = self._raise([-2.5], [True])
        self.assertIn("1.0, 0.0", msg)

    def test_a_field_scale_whose_square_is_not_a_normal_double_still_gets_a_message(self):
        """The routine reports a refusal and must not fail while reporting it.

        Dividing by the square of the scale is what makes the numbers comparable, and a scale
        of zero would make that division raise `ZeroDivisionError` from inside the message,
        losing the error and its code. The squared masses are then reported as they are, and
        the message says which unit it is using.

        `1e-160` is the case a positivity test alone lets through: its square, `1e-320`, is
        subnormal rather than zero, so it passes a truth test and then turns every mass of
        order one into an infinity. `1e-100` is the control at the small end, where the square
        is still normal and the scaled numbers are finite.
        """
        for scale in (0.0, None, float("inf"), float("nan"), -2.0, 1.0e200, 1.0e-160,
                      5.0e-324):
            with self.subTest(v_stable=scale):
                msg = self._raise([-2.5, 4.0], [True, True], v_stable=scale)
                self.assertIn("boson 0: m^2 = -2.5", msg)
                self.assertIn("below m^2 = -1e-05", msg)
                self.assertNotIn("m^2/v^2", msg)

    def test_a_small_scale_whose_square_is_still_normal_is_used(self):
        """The control for the bound above: it must not refuse a scale that works.

        At `v_stable` = 1e-100 the square is 1e-200, a normal double, so the listed mass is
        -2.5e+200 and the threshold -1e+195. Without this the bound could be set anywhere and
        the test above would still pass.
        """
        msg = self._raise([-2.5], [True], v_stable=1.0e-100)
        self.assertIn("boson 0: m^2/v^2 = -2.5e+200", msg)
        self.assertIn("below m^2/v^2 = -1e+195", msg)

    def test_an_unphysical_tachyonic_mode_is_not_an_error(self):
        """Goldstone bosons are tachyonic off the minimum and are flagged unphysical."""
        pot = _tachyon_pot([-50.0, 4.0], [False, True])
        gp.generic_potential.tachyonTest(pot, self.X0)

    def test_the_threshold_is_on_the_squared_mass_and_lets_round_off_through(self):
        """-1e-5 in internal units, so -1e-6 passes and -1e-4 does not."""
        gp.generic_potential.tachyonTest(_tachyon_pot([-1.0e-6], [True]), self.X0)
        with self.assertRaises(errors.TachyonError):
            gp.generic_potential.tachyonTest(_tachyon_pot([-1.0e-4], [True]), self.X0)


class _UnhandledPotentialError(unittest.TestCase):
    """Both `UnexpectedError` sites, driven with a tag neither of them handles."""

    TAG = "the barrier moved"

    @staticmethod
    def _error(message="the potential did something new", nargs=1):
        err = errors.PotentialError(message, _UnhandledPotentialError.TAG)
        if nargs == 0:
            err.args = ()
        return err


class ThePotentialErrorDocstringDescribesItsOwnConstructorTests(unittest.TestCase):
    """It described the tuple convention the index `args[1]` was written for.

    The constructor takes the tag as its own argument and passes only the message up, so a
    reader following the docstring would write the handler that this change had to fix.
    """

    def test_the_message_is_one_argument_and_the_tag_is_another(self):
        err = errors.PotentialError("what happened", "no barrier")
        self.assertEqual(err.args, ("what happened",))
        self.assertEqual(err.error_type, "no barrier")

    def test_the_docstring_does_not_promise_a_tuple(self):
        doc = errors.PotentialError.__doc__
        self.assertIn("error_type", doc)
        self.assertNotIn("should be tuples", doc)


class TheUnhandledErrorTypeIsReportedFromBounceActionTests(_UnhandledPotentialError):
    """The same fallback, in the other routine that has one.

    `bounceAction` catches `PotentialError` from `_validate_tunneling_path` and carries its own
    copy of the two-tag translation, so a fix to one copy does not reach the other. Pull request
    #50 fixes the copy in `_execute_full_tunneling`, where `err.args[1]` raised `IndexError`
    instead of the `UnexpectedError` the line was written to raise, and tests it there. This
    copy reads `err.args[0]`, which works, but reported no tag.
    """

    def _bounce(self, err):
        tdict = {"low_vev": np.array([0.0]), "high_vev": np.array([1.0])}
        with mock.patch.object(pathDeformation, "_validate_tunneling_path", side_effect=err):
            with self.assertRaises(errors.UnexpectedError) as caught:
                pathDeformation.bounceAction(
                    1.0, lambda x, T: 0.0, lambda x, T: 0.0, {}, tdict)
        return caught.exception

    def test_the_tag_the_handler_could_not_match_is_in_the_message(self):
        msg = str(self._bounce(self._error()))
        self.assertIn(self.TAG, msg)
        self.assertIn("bounceAction", msg,
                      "the message names the routine, so the two copies are distinguishable")

    def test_the_message_carries_the_explanation_and_both_handled_tags(self):
        """The message promises three things beside the tag, so all three are asserted.

        The explanation is what the potential said; the two handled tags tell a model author
        which tag to use instead. A regression that dropped any of them would otherwise pass.
        """
        msg = str(self._bounce(self._error("the potential did something new")))
        self.assertIn("the potential did something new", msg)
        self.assertIn("no barrier", msg)
        self.assertIn("stable, not metastable", msg)

    def test_the_code_is_999(self):
        self.assertEqual(self._bounce(self._error()).errorcode, 999)

    def test_the_report_cannot_itself_fail_on_an_exception_without_a_message(self):
        self.assertIn(self.TAG, str(self._bounce(self._error(nargs=0))))

    def test_a_verbose_run_reaches_the_report_as_well(self):
        """The print above the raise reads the message too, so it must not fail first.

        With `verbose=True` and an exception carrying no message, the diagnostic print ran
        before the raise and would have raised `IndexError` there, so the guard on the raise
        would never have been reached.
        """
        tdict = {"low_vev": np.array([0.0]), "high_vev": np.array([1.0])}
        with mock.patch.object(pathDeformation, "_validate_tunneling_path",
                               side_effect=self._error(nargs=0)):
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                with self.assertRaises(errors.UnexpectedError) as caught:
                    pathDeformation.bounceAction(
                        1.0, lambda x, T: 0.0, lambda x, T: 0.0, {}, tdict, verbose=True)
        self.assertIn(self.TAG, str(caught.exception))
        self.assertIn(self.TAG, printed.getvalue(),
                      "with no message to print, the print falls back to the tag")

    def test_the_handled_tags_still_fill_the_dictionary_rather_than_raise(self):
        tdict = {"low_vev": np.array([0.0]), "high_vev": np.array([1.0])}
        for tag, action, trantype in (("no barrier", 0, 2),
                                      ("stable, not metastable", np.inf, 0)):
            with self.subTest(tag=tag):
                err = errors.PotentialError("known", tag)
                with mock.patch.object(pathDeformation, "_validate_tunneling_path",
                                       side_effect=err):
                    out = pathDeformation.bounceAction(
                        1.0, lambda x, T: 0.0, lambda x, T: 0.0, {}, dict(tdict))
                self.assertEqual(out[1.0]["action"], action)
                self.assertEqual(out[1.0]["trantype"], trantype)


class NoTransitionAndSecondOrderAreSeparateCodesTests(unittest.TestCase):
    """Codes 6 and 9 are raised side by side in `Transitions.__init__`.

    A run with no transition at all and a run whose every transition is second order are
    different outcomes for a user: the first found nothing, the second found something that
    radiates no gravitational waves. The two codes exist to tell them apart and neither was
    driven.
    """

    @staticmethod
    def _pot():
        tracing = types.SimpleNamespace(nucleation_Ttol=1e-3,
                                        approx_strength_threshold=0.0,
                                        tunneling_params={},
                                        do_high_T_phase_check=False)
        return types.SimpleNamespace(
            Vtot=lambda X, T: 0.0, gradV=lambda X, T: 0.0, d2V=lambda X, T: 0.0,
            nucleationCriterion=lambda S, T: 0.0, conversionFactor=1.0,
            config=types.SimpleNamespace(tracingConf=tracing))

    def _build(self, found):
        phases = {"1": object(), "2": object()}
        with mock.patch.object(transitions.Transitions, "findTransitionPathFromTnuc",
                               return_value=found):
            with contextlib.redirect_stdout(io.StringIO()):
                return transitions.Transitions(phases, self._pot(), verbose=False)

    def test_no_transition_at_all_is_code_6(self):
        with self.assertRaises(errors.NoTransitionFound) as caught:
            self._build([])
        self.assertEqual(caught.exception.errorcode, 6)

    def test_only_second_order_transitions_is_code_9(self):
        with self.assertRaises(errors.OnlySecondOrderTransitionsError) as caught:
            self._build([types.SimpleNamespace(type=2), types.SimpleNamespace(type=2)])
        self.assertEqual(caught.exception.errorcode, 9)

    def test_the_second_order_message_says_no_first_order_transition_was_found(self):
        with self.assertRaises(errors.OnlySecondOrderTransitionsError) as caught:
            self._build([types.SimpleNamespace(type=2)])
        self.assertIn("second-order", str(caught.exception))
        self.assertIn("first-order", str(caught.exception))

    def test_one_first_order_transition_among_second_order_ones_is_not_an_error(self):
        """The boundary between code 9 and a usable run is `all`, not `any`."""
        built = self._build([types.SimpleNamespace(type=2), types.SimpleNamespace(type=1)])
        self.assertEqual(len(built.transitions), 2)


class TwoPhasesCannotShareTheHighestTemperatureTests(unittest.TestCase):
    """Code 14 from `phases.py`: the high-temperature phase has to be one phase.

    `testTmax` is the only check that the phase the transition chain starts from is one
    phase. If two phases end at the same `Tmax` the choice between them is arbitrary, and the
    model's `Tmax_factor` is too low to separate them, which is what the message says.
    """

    @staticmethod
    def _stub(tmaxes, T_eps=1e-3):
        phases = {str(i): types.SimpleNamespace(key=str(i), Tmax=t)
                  for i, t in enumerate(tmaxes)}
        return types.SimpleNamespace(phases=phases, Tmax=max(tmaxes), T_eps=T_eps,
                                     conversionFactor=1.0)

    def _find(self, tmaxes, **kw):
        stub = self._stub(tmaxes, **kw)
        return phases_module.Phases.testTmax(stub)

    def test_two_phases_at_the_same_highest_temperature_is_code_14(self):
        with self.assertRaises(errors.WrongHighTPhaseError) as caught:
            self._find([615.5, 615.5, 300.0])
        self.assertEqual(caught.exception.errorcode, 14)

    def test_the_message_names_the_temperature_and_both_phases(self):
        with self.assertRaises(errors.WrongHighTPhaseError) as caught:
            self._find([615.5, 615.5, 300.0])
        msg = str(caught.exception)
        self.assertIn("615.5", msg)
        self.assertIn("'0'", msg)
        self.assertIn("'1'", msg)
        self.assertIn("Tmax_factor", msg,
                      "the message must say which setting separates them")

    def test_the_docstring_names_the_exception_the_method_raises(self):
        """It said `ValueError`, which this method cannot raise.

        A user reading the docstring to decide what to catch would have caught nothing.
        """
        doc = phases_module.Phases.testTmax.__doc__
        self.assertIn("WrongHighTPhaseError", doc)
        self.assertNotIn("ValueError", doc)

    def test_one_highest_phase_is_accepted(self):
        self._find([615.5, 300.0, 200.0])

    def test_the_comparison_is_made_at_the_tracing_resolution(self):
        """Two phases a hair apart are the same `Tmax`; `T_eps` is what decides."""
        self._find([615.5, 615.0, 300.0], T_eps=0.1)
        with self.assertRaises(errors.WrongHighTPhaseError):
            self._find([615.5, 615.0, 300.0], T_eps=1.0)


if __name__ == "__main__":
    unittest.main()
