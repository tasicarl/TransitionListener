"""`RH` gains a convergence measure of its own, so `acc_rh` can be judged.

`Tperc` and `Tf` each fall back to the width of the bracket they were found in, so a single
accepted profile bounds them. `RH` had no such fallback: its uncertainty was a relative
difference against a previous iteration's `RH` or it was infinite, and the controller normally
reaches acceptance with nothing to difference against. Measured on the 2HDM benchmark, that was
164 of 164 resolved couplings reporting an infinite `RH` uncertainty, so `acc_rh` was never
once judged.

The second estimate recomputes `RH` from every second support point. What the difference
measures is how much the answer depends on the density of the grid it was integrated on, which
is what the target is asking. It solves no bounce: it reuses the samples already in hand.
"""

from __future__ import annotations

import math
import types
import unittest

import numpy as np

from transitionlistener import config
from transitionlistener import bubbledynamics as bd
from transitionlistener import percolation_adaptivestepsize as ad


def _profile(n=12):
    """A support that rises through `f_perc` and `f_final`, with a usable action and Hubble."""
    temps = np.linspace(100.0, 30.0, n)
    probabilities = np.linspace(0.0, 0.999, n)
    actions = np.linspace(150.0, 120.0, n)
    hubble = np.linspace(1e-16, 2e-16, n)
    return temps, probabilities, actions, hubble


def _pot(settings):
    return types.SimpleNamespace(config=types.SimpleNamespace(
        percolationConf=types.SimpleNamespace(f_final=settings.f_final)))


class HalvedSupportEstimateTests(unittest.TestCase):
    def test_it_returns_a_positive_value_on_a_usable_support(self):
        settings = config.PercolationConf()
        temps, p, a, h = _profile()
        value = ad._rh_on_halved_support(temps, p, a, h, 60.0, _pot(settings), settings)
        self.assertTrue(np.isfinite(value) and value > 0.0, value)

    def test_it_declines_a_support_too_short_to_halve(self):
        """Seven points leave three after halving, below what a cubic interpolant needs.

        Returning `nan` rather than a number keeps the caller's meaning intact: not measured,
        which is reported as an infinite uncertainty, rather than a measure of the
        interpolation instead of the profile.
        """
        settings = config.PercolationConf()
        temps, p, a, h = _profile(7)
        self.assertTrue(math.isnan(
            ad._rh_on_halved_support(temps, p, a, h, 60.0, _pot(settings), settings)))

    def test_eight_points_is_the_smallest_support_it_accepts(self):
        settings = config.PercolationConf()
        temps, p, a, h = _profile(8)
        self.assertTrue(np.isfinite(
            ad._rh_on_halved_support(temps, p, a, h, 60.0, _pot(settings), settings)))

    def test_it_declines_rather_than_raises_when_the_samples_are_unusable(self):
        settings = config.PercolationConf()
        temps, p, a, h = _profile()
        self.assertTrue(math.isnan(ad._rh_on_halved_support(
            temps, p, np.full(temps.size, np.nan), h, 60.0, _pot(settings), settings)))

    def test_the_halving_keeps_the_ends_of_the_support(self):
        """The integration range must not change, or the difference measures the range.

        Calls the production mask rather than restating it here. An earlier version of this
        test rebuilt the mask in the test body, so changing the real one could not fail it.

        Note on mutation testing this: forcing the hot end off is an EQUIVALENT mutant, because
        `keep[::2]` already includes index 0, and the production code no longer sets it twice.
        Forcing the cold end off and disabling the halving both turn this red.
        """
        for n in (8, 9, 12, 13, 41):
            with self.subTest(n=n):
                keep = ad._halved_support_mask(n)
                self.assertEqual(keep.size, n)
                self.assertTrue(keep[0], "the hot end sets the start of the integration")
                self.assertTrue(keep[-1], "the cold end sets its end")
                self.assertGreaterEqual(np.count_nonzero(keep), 4,
                                        "enough points left for a cubic interpolant")
                self.assertLessEqual(np.count_nonzero(keep), n // 2 + 2,
                                     "it has to actually halve, or it measures nothing")


class TheOptionalEstimateDoesNotSwallowTheDeadlineTests(unittest.TestCase):
    """The halved estimate is optional; the run deadline is not.

    The call to `_rh_on_halved_support` sits inside a broad `except Exception: pass`, which is
    right for the estimate -- a support it cannot halve should cost nothing -- and wrong for
    `errors.Timeout`, which subclasses `Exception` and is raised from a SIGALRM handler, so it
    can land inside the interpolations or `calcMeanBubbleSeparation` that the estimate calls.
    Swallowed there it becomes a missing second estimate and the run continues, reporting a point
    that never finished as one that did.

    A reviewer found this on PR #42. The package-wide sweep added in #41 did not, because it only
    flags broad handlers that *convert* the deadline into an error class, and this one drops it.
    That scope limit is why a guard of my own making got past it, and widening the sweep to
    swallowing handlers needs a waived baseline of the 85 that already exist, which belongs in its
    own change rather than here.
    """

    def test_a_deadline_inside_the_halved_estimate_reaches_the_caller(self):
        """Drive the real estimator, not a stub of the thing under test.

        The first version of this test patched `_rh_on_halved_support` and then called the patched
        helper, so it asserted that its own stub raises and exercised neither the estimator nor
        the helper's exception path. A reviewer caught it. It would have passed with the
        production bug fully present.

        Here `calcMeanBubbleSeparation` is patched so the **full-support** call succeeds and the
        **halved-support** one raises the deadline, which is the real sequence: the halved
        estimate is computed after the full one, from the same samples.
        """
        from unittest import mock
        from transitionlistener import errors

        settings = config.PercolationConf()
        temps, p, a, h = _profile()
        real = bd.calcMeanBubbleSeparation
        calls = {"n": 0}

        def first_then_deadline(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return real(*args, **kwargs)
            raise errors.Timeout()

        with mock.patch.object(bd, "calcMeanBubbleSeparation", first_then_deadline):
            with self.assertRaises(errors.Timeout) as caught:
                ad._estimate_dynamiczoomwindow_accuracy(
                    temps, h, p, a, settings, _pot(settings), previous=None)
        self.assertGreaterEqual(calls["n"], 2,
                                "the halved-support call has to be reached for this to mean "
                                f"anything; it was called {calls['n']} time(s)")
        self.assertEqual(caught.exception.errorcode, 16)

    def test_the_guard_precedes_the_broad_handler_at_that_call_site(self):
        """The behavioural test above cannot reach the real try/except, so pin it structurally."""
        import ast
        import inspect
        import textwrap

        src = inspect.getsource(ad)
        tree = ast.parse(src)
        sites = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            calls = [n for n in ast.walk(node)
                     if isinstance(n, ast.Call)
                     and getattr(n.func, "id", None) == "_rh_on_halved_support"]
            if not calls:
                continue
            guarded = False
            for h in node.handlers:
                t = h.type
                named = ((isinstance(t, ast.Attribute) and t.attr == "Timeout")
                         or (isinstance(t, ast.Name) and t.id == "Timeout"))
                if named and len(h.body) == 1 and isinstance(h.body[0], ast.Raise) \
                        and h.body[0].exc is None:
                    guarded = True
                if isinstance(t, ast.Name) and t.id in ("Exception", "BaseException"):
                    sites.append((node.lineno, guarded))
                    break
        self.assertTrue(sites, "no broad handler found around the halved-support call")
        for lineno, guarded in sites:
            self.assertTrue(
                guarded,
                f"the broad handler around _rh_on_halved_support (try at line {lineno} of "
                "percolation_adaptivestepsize.py) has no preceding `except errors.Timeout: "
                "raise`, so a run deadline there is dropped and the point is reported as finished")


class UncertaintyUsesBothMeasuresTests(unittest.TestCase):
    """The uncertainty takes the larger of the two, as Tperc and Tf already do."""

    @staticmethod
    def _estimate_with(previous_rh=None):
        settings = config.PercolationConf()
        temps, p, a, h = _profile()
        prev = None if previous_rh is None else {"Tperc": None, "Tf": None, "RH": previous_rh}
        return ad._estimate_dynamiczoomwindow_accuracy(
            temps, h, p, a, settings, _pot(settings), previous=prev), settings

    def test_a_first_estimate_is_now_bounded(self):
        est, _ = self._estimate_with()
        self.assertTrue(np.isfinite(est["RH_uncertainty"]),
                        "this is the case that used to be infinite on every point")

    def test_a_previous_estimate_that_disagrees_more_wins(self):
        """Drift across refinements is something the halving cannot see, so it is kept."""
        est, _ = self._estimate_with()
        half_only = est["RH_uncertainty"]
        far = est["RH"] * 2.0
        est2, _ = self._estimate_with(previous_rh=far)
        self.assertGreater(est2["RH_uncertainty"], half_only)
        self.assertAlmostEqual(est2["RH_uncertainty"],
                               abs(est2["RH"] - far) / abs(est2["RH"]), places=12)

    def test_a_previous_estimate_that_agrees_does_not_lower_it(self):
        est, _ = self._estimate_with()
        half_only = est["RH_uncertainty"]
        est2, _ = self._estimate_with(previous_rh=est["RH"])
        self.assertAlmostEqual(est2["RH_uncertainty"], half_only, places=12,
                               msg="the larger of the two, so a flattering comparison cannot "
                                   "hide what the halving found")


if __name__ == "__main__":
    unittest.main()
