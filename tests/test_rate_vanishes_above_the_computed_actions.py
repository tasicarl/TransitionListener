"""Above the temperatures whose action was computed, the nucleation rate is zero.

The percolation result carries an action spline built from the support. Outside the support it
returns fixed values, and above it the value used to be the action at the hottest temperature
that had one. Holding the action constant while the temperature rises lets S3/T fall, and the
rate goes as exp(-S3/T), so the rate *grows* towards high temperature. That is backwards: the
barrier grows there and the rate has to vanish.

It only matters when something is evaluated above the computed range, which happens when an
action solve fails at the hot end of the support. Measured on the 2HDM benchmark at
lambda3 = 5.7727298627, where the solve failed at T = 57.18 and the hottest computed action was
at T = 49.28:

  * the action was held at 7441.88, so S3/T fell from 151.0 at T = 49.28 to 130.2 at T = 57.18,
    which is a factor exp(20.8) = 1.1e9 in the rate;
  * the bubble number density came out 7338 times too large, 2.3063e-36 against 3.1421e-40;
  * the mean bubble separation came out 19 times too small, RH = 0.002352 where the neighbouring
    couplings give 0.044533 and 0.046287;
  * `betaH_RH` was 1883 against 99.4 and 95.7 at those neighbours, and the LISA signal-to-noise
    ratio 7.8 against 23.9 and 26.5.

One coupling of the 333 resolved across the five validation scans was affected.

The estimator's own spline in `percolation_adaptivestepsize` already returns an infinite action
above its data, so this makes the two agree.
"""
import unittest

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener.transitionObservables import _BoundedPchipInterpolator


# the measured profile at the affected coupling, reduced to what the claim needs
HOTTEST_COMPUTED_T = 49.2843
HOTTEST_COMPUTED_S = 7441.8771
FAILED_SOLVE_T = 57.1813


def _action_spline(fill_high):
    """The percolation result's action spline, with the fill above the data chosen."""
    T = np.array([47.5670, 48.4000, HOTTEST_COMPUTED_T])
    S = np.array([6964.0, 7200.0, HOTTEST_COMPUTED_S])
    return _BoundedPchipInterpolator(T, S, fill_low=float(S[0]), fill_high=fill_high)


class TheRateDoesNotGrowAboveTheComputedActions(unittest.TestCase):
    def test_the_action_is_infinite_above_the_computed_range(self):
        spline = _action_spline(np.inf)
        self.assertTrue(np.isinf(float(spline(FAILED_SOLVE_T))),
                        "the action above the computed range is finite, so the rate is not zero")

    def test_the_rate_vanishes_there(self):
        """Gamma reports an infinite action as not-a-number, which every integrand maps to zero."""
        spline = _action_spline(np.inf)
        rate = bd.Gamma(np.array([FAILED_SOLVE_T]), np.array([float(spline(FAILED_SOLVE_T))]))
        self.assertFalse(np.isfinite(rate[0]),
                         "the rate above the computed range is a finite number")
        self.assertEqual(float(np.nan_to_num(rate)[0]), 0.0,
                         "the integrands take a non-finite rate as zero, and this one does not")

    def test_holding_the_action_constant_makes_the_rate_grow(self):
        """Mutation check: the rule this replaced raises the rate where it must fall."""
        spline = _action_spline(HOTTEST_COMPUTED_S)
        s_hot = float(spline(FAILED_SOLVE_T))
        self.assertAlmostEqual(s_hot, HOTTEST_COMPUTED_S, places=6,
                               msg="the mutation did not reinstate the constant action")
        cold = HOTTEST_COMPUTED_S / HOTTEST_COMPUTED_T
        hot = s_hot / FAILED_SOLVE_T
        self.assertLess(hot, cold,
                        f"S3/T does not fall with the constant action: {hot} against {cold}")
        self.assertGreater(
            bd.Gamma(np.array([FAILED_SOLVE_T]), np.array([s_hot]))[0],
            bd.Gamma(np.array([HOTTEST_COMPUTED_T]), np.array([HOTTEST_COMPUTED_S]))[0],
            "the rate does not grow under the replaced rule, so this test proves nothing")

    def test_the_spline_is_unchanged_inside_the_computed_range(self):
        """The fill is about the outside; the interpolation itself must not move."""
        inside = np.array([47.8, 48.4, 49.0])
        a = _action_spline(np.inf)(inside)
        b = _action_spline(HOTTEST_COMPUTED_S)(inside)
        self.assertTrue(np.allclose(a, b, rtol=0, atol=0),
                        "the fill above the range changed the values inside it")


class TheBubbleNumberIntegralStopsWhereTheActionIsKnown(unittest.TestCase):
    """The companion bound: the upper limit is a temperature whose integrand exists.

    `tmax_rh` used the hottest temperature in the support. A temperature whose solve failed
    carries no action, so taking it as the bound integrates over a stretch the spline has no
    data for. The two differ only where a solve failed at the hot end.
    """

    @staticmethod
    def _bound(support_T, support_S):
        """What the observables now use as the upper limit of the bubble-number integral."""
        t = np.asarray(support_T, dtype=float)
        s = np.asarray(support_S, dtype=float) if support_S is not None else None
        if s is not None and s.shape == t.shape:
            known = np.isfinite(t) & np.isfinite(s)
        else:
            known = np.isfinite(t)
        return float(np.nanmax(t[known])) if known.any() else float(np.nanmax(t))

    def test_a_failed_solve_at_the_hot_end_does_not_set_the_bound(self):
        T = np.array([FAILED_SOLVE_T, HOTTEST_COMPUTED_T, 47.5670])
        S = np.array([np.nan, HOTTEST_COMPUTED_S, 6964.0])
        self.assertAlmostEqual(self._bound(T, S), HOTTEST_COMPUTED_T, places=6)
        self.assertGreater(float(np.nanmax(T)), self._bound(T, S),
                           "the support's hottest temperature is not above the new bound, so "
                           "this coupling does not exercise the rule")

    def test_an_ordinary_support_is_unaffected(self):
        """Where every solve succeeded the bound is the one it always was."""
        T = np.array([61.291, 55.0, 47.567])
        S = np.array([9000.0, 8000.0, 6964.0])
        self.assertEqual(self._bound(T, S), float(np.nanmax(T)))

    def test_a_support_with_no_action_at_all_still_gives_a_bound(self):
        T = np.array([61.291, 55.0, 47.567])
        S = np.array([np.nan, np.nan, np.nan])
        self.assertEqual(self._bound(T, S), float(np.nanmax(T)))

    def test_it_falls_back_where_the_actions_are_not_usable(self):
        """The guard above this code establishes the temperatures, not the actions.

        `PercolationResult.S` is typed as optional and need not line up with `TSYM`, so a
        missing or mismatched action array has to give the bound this used before rather than
        raise, which is what the original single call to `nanmax` could never do.
        """
        T = np.array([61.291, 55.0, 47.567])
        for label, S in (("absent", None), ("a different length", np.array([1.0, 2.0]))):
            with self.subTest(actions=label):
                self.assertEqual(self._bound(T, S), float(np.nanmax(T)))


if __name__ == "__main__":
    unittest.main()
