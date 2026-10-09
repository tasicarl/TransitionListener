"""Change 25: the action-rate jitter gate must not judge a dead rate.

Reproduces the measured situation at lambda3 = 5.82015 on the 2HDM edge line: the rate
falls through ~430 decades across the probability band while P climbs past f_perc, and a
~35-decade wobble sits deep in that tail, 432 decades below the rate peak and already past
percolation. The old gate called that an instability and rejected the whole point.
"""
import math
import unittest

import numpy as np


def _arrays(wobble_decades, wobble_at_index, n=40, span_decades=430.0, grading=3.0):
    """A strongly supercooled band: rate falling ~430 decades, P rising past f_perc.

    The depth below the rate peak grows super-linearly along the grid, because the adaptive
    controller refines where the rate rises and leaves the collapsed tail coarse. Measured
    on the 2HDM edge line, a 30-decade live window holds 8 to 99 cached support points
    (median 62 of ~150), so a uniform grid would misrepresent the geometry the gate sees:
    at ~11 decades per sample the live window holds 3 points and the gate cannot fit its
    quadratic at all, which is a state worth testing but is not the common one.
    """
    T = np.linspace(40.0, 3.0, n)
    H = np.full(n, 1.0e-16)
    depth = span_decades * (np.arange(n) / (n - 1.0)) ** grading
    s_over_t = 150.0 + depth * math.log(10.0)
    S = s_over_t * T
    P = np.linspace(0.02, 0.88, n)
    if wobble_decades:
        # perturb the action so the rate at one sample departs from the smooth trend
        S[wobble_at_index] += wobble_decades * math.log(10.0) * T[wobble_at_index]
    return T, H, S, P


class JitterDeadRateTests(unittest.TestCase):
    def setUp(self):
        from transitionlistener import bubbledynamics as bd
        self.bd = bd
        self.diag = bd._action_rate_jitter_diagnostic

    def test_deep_tail_wobble_is_not_flagged(self):
        """A 35-decade wobble 400+ decades below the peak must not register."""
        T, H, S, P = _arrays(35.0, wobble_at_index=36)
        out = self.diag(T, H, S, P)
        resid = out.get("max_residual_oom", math.nan)
        self.assertTrue(
            not np.isfinite(resid) or resid < 1.0,
            f"dead-tail wobble was judged: residual {resid}",
        )
        self.assertGreater(
            out.get("n_dead_rate_excluded", 0), 0,
            "no sample was excluded, so the exemption never engaged",
        )

    def test_live_region_wobble_is_still_flagged(self):
        """The same wobble near the rate peak must still be caught."""
        T, H, S, P = _arrays(35.0, wobble_at_index=1)
        out = self.diag(T, H, S, P)
        resid = out.get("max_residual_oom", math.nan)
        self.assertTrue(
            np.isfinite(resid) and resid > 1.0,
            f"live-region wobble was missed: residual {resid}",
        )

    def test_smooth_band_is_clean(self):
        T, H, S, P = _arrays(0.0, wobble_at_index=0)
        out = self.diag(T, H, S, P)
        resid = out.get("max_residual_oom", math.nan)
        self.assertTrue(
            not np.isfinite(resid) or resid < 1.0,
            f"a smooth band was flagged: residual {resid}",
        )

    def test_unassessable_band_is_reported_not_called_clean(self):
        """When the live window is too small to fit, the diagnostic must say so.

        One resolved profile of the 83 measured has only 2 support points within 30 decades
        of its rate peak. There the gate cannot judge anything, and the result must be
        distinguishable from a band that was judged and found smooth -- otherwise turning the
        gate off looks the same as passing it.
        """
        T, H, S, P = _arrays(35.0, wobble_at_index=36, grading=1.0)
        out = self.diag(T, H, S, P)
        self.assertLess(out["n_active"], 5, "this grid was meant to starve the fit")
        self.assertGreater(
            out.get("n_dead_rate_excluded", 0), 0,
            "an unassessable band is indistinguishable from a clean one",
        )

    def test_threshold_is_inside_the_derived_window(self):
        """Above ~20 decades it cannot mask a defect; below ~40 it catches the measured
        false positives. The constant must stay in that window."""
        self.assertGreaterEqual(self.bd._JITTER_DEAD_RATE_DECADES, 20.0)
        self.assertLessEqual(self.bd._JITTER_DEAD_RATE_DECADES, 40.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
