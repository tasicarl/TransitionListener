"""A cached action is not automatically useful support.

Below a completed transition the bounce action does not fail - it *saturates* at its
zero-temperature value - so ``S_3/T`` grows without bound as ``T -> 0`` and the nucleation
rate underflows to exactly zero. Measured on the 2HDM edge line at
lambda3 = 5.8123128723, from the run's complete action cache: the action settles near 4357
and ``S_3/T`` reaches 1.15e+09 by T = 3.8e-06 (internal units), while nucleation needs
``S_3/T`` of order 140.

Such a point has a perfectly finite action, so it is promoted into the active grid as a
legitimate cached temperature. The grid's cold boundary lands there - traced as
``reason=cached_action_points, old=[2.2235, 62.279] -> new=[3.8032e-06, 68.772]`` - and
the rate-step refinement then fills six orders of magnitude of temperature with support
that cannot influence ``P``, which has already saturated, or the rate peak, which is far
hotter. 60 of the 106 support points in that run were of this kind.

Two earlier guards tested for a run of *non-finite* actions at the cold end and both
produced bit-identical runs, because the cold actions are all finite. The criterion has to
be the rate.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from transitionlistener import percolation_adaptive_gridbuilders as gridbuilders
from transitionlistener import percolation_adaptivestepsize as pas


class NucleationIsDeadTests(unittest.TestCase):
    """The criterion itself: S_3/T far past anything that can nucleate."""

    def test_the_measured_cold_points_are_dead(self):
        # the two extremes actually seen in the action cache
        self.assertTrue(gridbuilders._nucleation_is_dead(3.8032e-06, 4357.25))
        self.assertTrue(gridbuilders._nucleation_is_dead(8.7826e-04, 4357.17))

    def test_points_that_can_still_nucleate_are_not_dead(self):
        """Controls: the transition window, its cold edge, and a supercooled scale."""
        self.assertFalse(gridbuilders._nucleation_is_dead(31.6, 31.6 * 167.0))
        self.assertFalse(gridbuilders._nucleation_is_dead(24.44, 24.44 * 200.0))
        # conformal dark U(1) at g = 0.54 still nucleates at Tperc/v = 2.8e-04
        self.assertFalse(gridbuilders._nucleation_is_dead(3.92e-05, 3.92e-05 * 150.0))

    def test_an_unusable_action_is_unknown_rather_than_dead(self):
        """A non-finite action says nothing about the rate, so it must not be judged."""
        self.assertFalse(gridbuilders._nucleation_is_dead(10.0, float("nan")))
        self.assertFalse(gridbuilders._nucleation_is_dead(10.0, float("inf")))
        self.assertFalse(gridbuilders._nucleation_is_dead(0.0, 100.0))


class PromotionSkipsDeadPointsTests(unittest.TestCase):
    """The intent: the active grid's cold boundary must not be planted on a dead point."""

    @staticmethod
    def promote(outdict, active, bank):
        calls = []
        return pas._promote_dynamiczoomwindow_cached_points(
            np.asarray(active, dtype=float),
            np.asarray(bank, dtype=float),
            np.asarray([], dtype=float),
            types.SimpleNamespace(n_action_max=60, n_action_increment=5),
            outdict, None, None,
            tmin=1.0e-13, tmax=615.0,
            stage="step2", reason="cached_action_points",
            reset_arrays=lambda *a, **k: None,
            note_rebuild=lambda *a, **k: calls.append((a, k)),
        )

    def test_a_dead_cold_point_is_not_promoted(self):
        """The bug: the saturated-action point becomes the grid's cold boundary."""
        live = {float(T): {"action": float(T) * 170.0, "trantype": 1}
                for T in (76.4, 62.0, 41.1, 31.7, 25.4)}
        outdict = dict(live)
        outdict[3.8032e-06] = {"action": 4357.25, "trantype": 1}   # S_3/T = 1.1e9
        changed, new_active, new_bank, _ = self.promote(
            outdict, active=[62.0, 41.1], bank=[62.0, 41.1])
        grid = np.asarray(new_active, dtype=float)
        self.assertGreater(
            float(grid.min()), 1.0e-3,
            f"the active grid's cold boundary was planted at {grid.min():.4g}, where "
            "S_3/T is about 1e9 and the rate is identically zero",
        )

    def test_live_cached_points_are_still_promoted(self):
        """Control: ordinary cached support must keep being promoted."""
        outdict = {float(T): {"action": float(T) * 170.0, "trantype": 1}
                   for T in (76.4, 62.0, 41.1, 31.7, 25.4)}
        changed, new_active, _, _ = self.promote(
            outdict, active=[62.0, 41.1], bank=[62.0, 41.1])
        grid = np.asarray(new_active, dtype=float)
        self.assertTrue(changed, "no live cached point was promoted at all")
        self.assertIn(True, [bool(np.any(np.isclose(grid, T))) for T in (76.4, 25.4)],
                      "the promotion stopped admitting usable support")


class TheFilterStaysOnTheColdSideTests(unittest.TestCase):
    """``S_3/T`` is just as large on the hot shoulder, and there it must not disqualify.

    The action diverges as ``T`` approaches the critical temperature from below, and the hot
    head is only accepted once ``log(Gamma/H^4)`` has fallen below zero, which needs a large
    ``S_3/T`` by construction: across the 31 recorded edge-line profiles that carry one, the
    hottest retained support point sits at ``S_3/T`` between 478 and 509, within a factor two
    of the 1e3 threshold. If the test applied there too, this function would refuse exactly the
    support ``_raise_if_hot_head_underresolved`` rejects the profile for lacking.
    """

    @staticmethod
    def promote(outdict, active, bank, promotion_high, promotion_low):
        return pas._promote_dynamiczoomwindow_cached_points(
            np.asarray(active, dtype=float),
            np.asarray(bank, dtype=float),
            np.asarray([], dtype=float),
            types.SimpleNamespace(n_action_max=60, n_action_increment=5),
            outdict, None, None,
            tmin=1.0e-13, tmax=615.0,
            promotion_high=promotion_high, promotion_low=promotion_low,
            stage="step3", reason="hot_head_cached_points",
            reset_arrays=lambda *a, **k: None,
            note_rebuild=lambda *a, **k: None,
        )

    def test_a_hot_head_point_is_promoted_however_large_its_action(self):
        hot = 120.0
        outdict = {float(T): {"action": float(T) * 170.0, "trantype": 1}
                   for T in (41.1, 31.7)}
        outdict[hot] = {"action": hot * 1.0e4, "trantype": 1}       # S_3/T = 1e4
        self.assertTrue(
            gridbuilders._nucleation_is_dead(hot, outdict[hot]["action"]),
            "the fixture's hot point is not dead by the criterion, so it cannot show that "
            "the criterion is confined to the cold side",
        )
        _changed, new_active, _bank, _free = self.promote(
            outdict, active=[41.1, 31.7], bank=[41.1, 31.7],
            promotion_high=615.0, promotion_low=41.1)
        grid = np.asarray(new_active, dtype=float)
        self.assertTrue(
            bool(np.any(np.isclose(grid, hot))),
            f"the hot-head support point at {hot} GeV was refused promotion because its "
            "S_3/T is large, which is what a hot shoulder looks like",
        )

    def test_a_dead_point_colder_than_the_grid_is_still_refused(self):
        """Control: the cold side keeps the behaviour the measurement called for."""
        outdict = {float(T): {"action": float(T) * 170.0, "trantype": 1}
                   for T in (41.1, 31.7)}
        outdict[3.8032e-06] = {"action": 4357.25, "trantype": 1}
        _changed, new_active, _bank, _free = self.promote(
            outdict, active=[41.1, 31.7], bank=[41.1, 31.7],
            promotion_high=None, promotion_low=None)
        grid = np.asarray(new_active, dtype=float)
        self.assertGreater(float(grid.min()), 1.0e-3)


if __name__ == "__main__":
    unittest.main()
