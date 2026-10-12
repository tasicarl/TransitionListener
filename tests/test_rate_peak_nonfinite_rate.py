"""The rate-peak refinement must tolerate the NaNs its own peak search allows for.

``_refine_step1_rate_peak`` locates the peak of ``log(Gamma/H^4)`` with
``np.nanargmax``, so it is written for an array that may carry non-finite samples: a
support temperature whose bounce solve did not converge has no rate. Ten lines later it
hands that same array, NaNs included, to ``PchipInterpolator``, which rejects them.

The result is a bare ``ValueError: 'y' must contain only finite values`` with no typed
TL error and no diagnostic, reported as error -1. Captured on the 2HDM line scan at
lambda3 = 5.8191375291: ``y`` held 1 NaN of 79 samples, at T = 58.06 GeV, reached from
``_try_dynamiczoomwindow_ordered_support_refinement`` via
``_build_dynamiczoomwindow_rate_peak_grid``.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import unittest

import numpy as np

from transitionlistener import percolation_adaptive_rate as rate


def rate_profile(n: int = 41):
    """Descending temperatures with one prominent interior peak in log(Gamma/H^4)."""
    temps = np.linspace(80.0, 20.0, n)              # hot -> cold, as the solver orders it
    peak_at = 50.0
    log_rate = 6.0 - 0.02 * (temps - peak_at) ** 2  # prominence well above the 1.0 gate
    return temps, log_rate


class RatePeakNonFiniteTests(unittest.TestCase):

    def test_a_nan_far_from_the_peak_does_not_raise(self):
        """The bug: one unconverged action must not take the refinement down."""
        temps, log_rate = rate_profile()
        log_rate = log_rate.copy()
        log_rate[3] = np.nan                        # interior, far from the peak
        try:
            peak_T, peak_value, peak_index = rate._refine_step1_rate_peak(temps, log_rate)
        except ValueError as exc:  # pragma: no cover - this is the bug being fixed
            self.fail(
                "a non-finite rate sample reached the interpolant: "
                f"{exc}. The peak search uses np.nanargmax, so the interpolant has to "
                "drop the same samples."
            )
        self.assertIsNotNone(peak_T, "the peak was lost although 40 of 41 samples were fine")
        self.assertTrue(np.isfinite(float(peak_T)))

    def test_the_refined_peak_is_unchanged_by_a_distant_nan(self):
        """Dropping an unusable sample must not move the answer."""
        temps, log_rate = rate_profile()
        clean_T, _, _ = rate._refine_step1_rate_peak(temps, log_rate)
        holed = log_rate.copy()
        holed[3] = np.nan
        holed_T, _, _ = rate._refine_step1_rate_peak(temps, holed)
        self.assertIsNotNone(clean_T)
        self.assertIsNotNone(holed_T)
        self.assertAlmostEqual(
            float(clean_T), float(holed_T), places=6,
            msg="removing a sample 3 points from the hot edge moved the refined peak",
        )

    def test_an_ordinary_profile_still_refines(self):
        """Control: the all-finite case keeps working and lands on the true peak."""
        temps, log_rate = rate_profile()
        peak_T, peak_value, peak_index = rate._refine_step1_rate_peak(temps, log_rate)
        self.assertIsNotNone(peak_T)
        self.assertAlmostEqual(float(peak_T), 50.0, delta=1.0)

    def test_a_nan_at_the_peak_itself_is_declined_not_crashed(self):
        """Control: an unusable peak neighbourhood returns no peak, and does not raise."""
        temps, log_rate = rate_profile()
        holed = log_rate.copy()
        holed[int(np.argmax(log_rate)) + 1] = np.nan
        peak_T, peak_value, peak_index = rate._refine_step1_rate_peak(temps, holed)
        self.assertIsNone(peak_T)

    def test_too_few_usable_samples_returns_no_peak(self):
        """Control: an array that is almost entirely unusable must decline."""
        temps, log_rate = rate_profile(n=9)
        holed = np.full_like(log_rate, np.nan)
        holed[4] = 6.0
        peak_T, _, _ = rate._refine_step1_rate_peak(temps, holed)
        self.assertIsNone(peak_T)

    def test_an_infinite_rate_does_not_get_an_interpolant_built_around_it(self):
        """``logGamma`` is ``+inf`` where the action vanishes, which a "no barrier" solve
        stores, and ``+inf`` passes every comparison in the prominence test. The peak search
        can therefore land on a sample this function drops, which is why the number of
        remaining nodes is checked instead of assumed.
        """
        temps = np.linspace(80.0, 20.0, 5)
        log_rate = np.array([2.0, np.inf, 2.0, np.inf, np.nan])
        self.assertEqual(int(np.nanargmax(log_rate)), 1,
                         "the fixture's peak is not the infinite sample")
        peak_T, _peak_value, peak_index = rate._refine_step1_rate_peak(temps, log_rate)
        self.assertEqual(peak_index, 1)
        self.assertEqual(
            float(peak_T), float(temps[1]),
            "the refinement built an interpolant from the two remaining samples and moved "
            "the peak onto the edge of the bracket",
        )


if __name__ == "__main__":
    unittest.main()
