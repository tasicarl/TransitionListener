"""A bounce whose bracketing trials disagree on the action is not resolved.

The shooting narrows `x` until an undershoot and an overshoot sit either side of the
bounce, so at convergence both must give the same action. Nothing checked that, and the
profile handed to `findAction` was simply whichever trial came last.

Measured on the 2HDM at lambda3 = 5.816407666 with `TL_2HDM_BSMPT_highacc`. At
T = 9.8546 GeV the bisection runs 296 integrations whose last six share one `x` while the
verdict alternates, and the two branches give +3743.07 at a bubble radius of 1.206 against
-501.35 at 3.214. A Euclidean bounce action cannot be negative, so one of them is certainly
wrong, and which one is returned changes between temperatures 2e-04 GeV apart. The branches
agree to 8.9e-06 at T = 60 GeV, 7.0e-04 at 40, 7.0e-03 at 25 and 3.2e-05 at 15, so the two
populations are three orders apart and `_ACTION_BRACKET_TOL` sits between them.

Two things decide where the test lives and what it raises:

* `fullTunneling` calls `findProfile` once per path-deformation iteration, and the early
  ones legitimately bracket poorly. Raising inside `findProfile` refused T = 40 GeV, whose
  final branches agree to 7.0e-04. So `findProfile` records the spread and the caller
  applies it to the profile it keeps.
* It raises `InfiniteActionError`, which callers already read as "no action at this
  temperature" and drop the sample for. `TunnelingError` leaves `fullTunneling` and ends
  the whole coupling as CLI code 11, which cost 6 of 11 couplings in a scan at 2.5e-04
  spacing before it was corrected.

Part of TransitionListener v2
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from transitionlistener import errors
from transitionlistener import pathDeformation
from transitionlistener import tunneling1D


_TILT = 0.6


def quartic():
    def V(phi):
        return 0.25*phi**4 - (1.0+_TILT)/3.0*phi**3 + 0.5*_TILT*phi**2

    def dV(phi):
        return phi**3 - (1.0+_TILT)*phi**2 + _TILT*phi

    def d2V(phi):
        return 3*phi**2 - 2*(1.0+_TILT)*phi + _TILT

    return V, dV, d2V


class TheSpreadIsRecordedNotActedOn(unittest.TestCase):
    """`findProfile` reports how far apart the branches were and raises nothing itself.

    The decision belongs to the caller, because only it knows which profile is being kept.
    """

    def test_a_solve_records_a_spread_and_still_returns_a_profile(self):
        inst = tunneling1D.SingleFieldInstanton(0.0, 1.0, *quartic())
        profile = inst.findProfile()
        self.assertIsNotNone(profile)
        self.assertTrue(hasattr(inst, "bracket_action_spread"),
                        "findProfile did not record a spread")
        self.assertIsInstance(inst.bracket_action_spread, float)
        self.assertGreaterEqual(inst.bracket_action_spread, 0.0)
        # On a well-behaved quartic the branches agree, so nothing is flagged.
        self.assertLess(
            inst.bracket_action_spread, tunneling1D._ACTION_BRACKET_TOL,
            f"the quartic's branches disagree by {inst.bracket_action_spread!r}, which "
            "would make every test below vacuous")


    def test_the_recorded_spread_is_computed_from_the_two_branches(self):
        """The arithmetic, not just the attribute.

        On a healthy quartic both branches agree, so a recorder that always writes zero is
        indistinguishable from a correct one. Making the action depend on the profile forces
        the two branches apart by a known amount, and the recorded spread must be that
        amount: the relative difference against the larger of the two.
        """
        inst = tunneling1D.SingleFieldInstanton(0.0, 1.0, *quartic())
        real_integrate = tunneling1D.SingleFieldInstanton.integrateProfile

        def never_converges(self_, *a, **k):
            # The quartic's bisection always reports `converged`, so the bracket branch is
            # never reached on it. At lambda3 = 5.816407666, T = 9.8546 GeV the 2HDM's
            # bisection reports only undershoot and overshoot across 296 trials and ends on
            # the floating-point stall; withholding `converged` reproduces that, and is the
            # only thing stubbed besides the action.
            rf, yf, ctype = real_integrate(self_, *a, **k)
            return rf, yf, ("undershoot" if ctype == "converged" else ctype)

        def by_radius(self_, profile):
            # a value that rises with how far the trial was integrated, so any difference
            # in the two branches' radii separates their actions
            return 1000.0 * (1.0 + float(profile.R[-1]))

        with mock.patch.object(tunneling1D.SingleFieldInstanton,
                               "integrateProfile", never_converges), \
             mock.patch.object(tunneling1D.SingleFieldInstanton, "findAction", by_radius):
            try:
                inst.findProfile()
            except errors.TunnelingError:
                pass
        branches = inst.bracket_branch_actions
        self.assertTrue(branches, "the bisection left no bracket, so nothing was compared")
        expected = abs(branches["undershoot"] - branches["overshoot"]) / max(
            abs(branches["undershoot"]), abs(branches["overshoot"]))
        self.assertGreater(expected, 0.0,
                           "the two branches were not separated, so this cannot fail")
        self.assertAlmostEqual(
            inst.bracket_action_spread, expected, places=12,
            msg=f"recorded {inst.bracket_action_spread!r} against {expected!r} from the "
                f"branch actions {branches!r}")


class AnUnresolvedBounceIsReportedAsNoAction(unittest.TestCase):
    """A spread past the tolerance must raise `InfiniteActionError`, not `TunnelingError`.

    `_tunnelFromPhaseAtT` catches `InfiniteActionError` and returns an infinite action for
    that temperature, which the controller drops. `TunnelingError` is not caught there and
    ends the coupling.
    """

    def _tunnel_with_spread(self, spread):
        """Run the real `fullTunneling` on the quartic, with the spread forced.

        The check under test lives in `fullTunneling`, so the test has to go through it.
        Patching `findProfile` to record a chosen spread leaves every other decision --
        the deformation, the profile, the action -- genuine, and is the only thing stubbed.
        """
        real = tunneling1D.SingleFieldInstanton.findProfile

        def patched(inst, *a, **k):
            profile = real(inst, *a, **k)
            inst.bracket_action_spread = float(spread)
            inst.bracket_branch_actions = {"undershoot": 4061.24, "overshoot": 3215.07}
            return profile

        V, dV, _ = quartic()
        with mock.patch.object(tunneling1D.SingleFieldInstanton, "findProfile", patched):
            return pathDeformation.fullTunneling(
                [np.array([0.0]), np.array([1.0])],
                lambda X: V(np.asarray(X)[..., 0]),
                lambda X: np.asarray(dV(np.asarray(X)[..., 0]))[..., None],
                verbose=False)

    def test_a_spread_past_the_tolerance_raises_infinite_action(self):
        with self.assertRaises(errors.InfiniteActionError):
            self._tunnel_with_spread(tunneling1D._ACTION_BRACKET_TOL * 10.0)

    def test_the_exception_is_one_the_caller_drops_the_sample_for(self):
        """`TunnelingError` would end the coupling; `InfiniteActionError` must not."""
        self.assertFalse(
            issubclass(errors.InfiniteActionError, errors.TunnelingError),
            "InfiniteActionError inherits from TunnelingError, so the handler that ends "
            "the coupling would catch it too")
        self.assertEqual(getattr(errors.TunnelingError(), "errorcode", None), 11)

    def test_a_spread_that_is_not_a_number_is_refused(self):
        """`nan > tol` is false, so the check has to be written the other way round."""
        with self.assertRaises(errors.InfiniteActionError):
            self._tunnel_with_spread(float("nan"))

    def test_an_infinite_spread_is_refused(self):
        with self.assertRaises(errors.InfiniteActionError):
            self._tunnel_with_spread(float("inf"))

    def test_a_spread_inside_the_tolerance_returns_an_action(self):
        result = self._tunnel_with_spread(tunneling1D._ACTION_BRACKET_TOL * 0.1)
        self.assertTrue(np.isfinite(result.action) and result.action > 0.0,
                        f"a resolved bounce returned {result.action!r}")


class ABisectionThatNeverBracketsIsUnresolved(unittest.TestCase):
    """One integrated side is not a comparison, and must not read as agreement.

    The spread used to start at zero, so a solve that ended without ever bracketing the
    bounce -- every trial an undershoot, or one side's trial never integrated -- left it
    at zero and the caller took the profile as resolved on the strength of a test that had
    not been run. It starts at infinity now for any exit that is not a convergence, and is
    overwritten only once both branch actions exist.
    """

    def _never_brackets(self):
        """The real solver, with every verdict forced to the same side."""
        real = tunneling1D.SingleFieldInstanton.integrateProfile

        def always_undershoot(inst, r0, y0, *a, **k):
            rf, yf, _ = real(inst, r0, y0, *a, **k)
            return rf, yf, "undershoot"

        V, dV, d2V = quartic()
        inst = tunneling1D.SingleFieldInstanton(0.0, 1.0, V, dV, d2V)
        with mock.patch.object(tunneling1D.SingleFieldInstanton, "integrateProfile",
                               always_undershoot):
            try:
                inst.findProfile()
            except Exception:
                pass
        return inst

    def test_no_pair_was_compared(self):
        self.assertEqual(dict(self._never_brackets().bracket_branch_actions), {})

    def test_the_spread_is_not_zero(self):
        inst = self._never_brackets()
        self.assertNotEqual(inst.bracket_action_spread, 0.0)
        self.assertGreater(inst.bracket_action_spread, tunneling1D._ACTION_BRACKET_TOL)

    def test_a_converged_solve_still_reports_no_spread(self):
        """The guard must not turn an ordinary converged bounce into a refusal."""
        V, dV, d2V = quartic()
        inst = tunneling1D.SingleFieldInstanton(0.0, 1.0, V, dV, d2V)
        inst.findProfile()
        self.assertEqual(inst.bracket_action_spread, 0.0)


class ABranchWithoutAFiniteActionIsUnresolved(unittest.TestCase):
    """A branch that returned `inf` or `nan` has not resolved the bounce either.

    The relative spread cannot report that on its own: `inf - inf` and `inf / inf` are both
    `nan`, and every comparison against `nan` is false, so a sample whose branches could not
    even be compared would be accepted. It is reported as an infinite spread instead.
    """

    def test_two_infinite_branches_do_not_cancel_into_a_passing_spread(self):
        self.assertEqual(tunneling1D._bracketSpread(float("inf"), float("inf")),
                         float("inf"))

    def test_one_branch_that_is_not_a_number_is_enough(self):
        for pair in ((float("nan"), 3215.07), (3215.07, float("nan")),
                     (float("inf"), 3215.07), (3215.07, float("-inf"))):
            with self.subTest(pair=pair):
                self.assertEqual(tunneling1D._bracketSpread(*pair), float("inf"))

    def test_finite_branches_still_give_the_relative_difference(self):
        a, b = 4061.24, 3215.07
        self.assertAlmostEqual(tunneling1D._bracketSpread(a, b), (a - b) / a, places=12)

    def test_two_branches_that_agree_give_zero(self):
        self.assertEqual(tunneling1D._bracketSpread(3215.07, 3215.07), 0.0)

    def test_two_zero_actions_do_not_divide_by_zero(self):
        self.assertEqual(tunneling1D._bracketSpread(0.0, 0.0), 0.0)


class TheToleranceSeparatesTheTwoMeasuredPopulations(unittest.TestCase):
    """The threshold has to sit between what agreement and disagreement actually look like.

    Both numbers are measured on the 2HDM, not chosen: the branches agree to at worst
    7.0e-03 where the solver is healthy, and disagree by 0.42 and 1.13 at the two
    temperatures where one branch returns a negative action.
    """

    WORST_AGREEMENT = 7.0e-3
    BEST_DISAGREEMENT = 0.42

    def test_the_threshold_lies_between_them(self):
        tol = tunneling1D._ACTION_BRACKET_TOL
        self.assertGreater(tol, self.WORST_AGREEMENT,
                           "the tolerance would reject temperatures the solver resolves")
        self.assertLess(tol, self.BEST_DISAGREEMENT,
                        "the tolerance would accept a bounce decided by which trial ran last")


if __name__ == "__main__":
    unittest.main()
