"""The percolation estimate takes the hottest root, and declines when there is none.

`calcApproxPercolation` places the first adaptive grid. It handed the whole phase-overlap
interval to `optimize.brentq`, which assumes one root, and that interval carries several.
The criterion is positive below the percolation point and negative above it, but as `T`
falls the rate vanishes, so `crit = Gamma^(1/4) / (H betaH (I_perc / 8 pi v_w^3)^(1/4))`
tends to zero and `crit - 1` turns negative again at the cold end, while `T <= Tmin` returns
`+inf` by construction.

Nothing exposed that while the shooting aborted on a wrong-sign slope: the `isinf` guard
fired inside the band where the bounce would not solve and manufactured a sign change in
about the right place. Measured on the 2HDM at lambda3 = 5.815656566 with `TL_2HDM_BSMPT`,
the estimate returned 61.5057 GeV before the shooting was fixed and 1.96582e-12 GeV after,
and the whole run followed the second number to the cold end of the tracing range.

A scan of the criterion at that coupling, 40 points from 121 down to 3 GeV, is negative at
every temperature below the critical one, so the 61.5057 was an artefact too: the estimate
*should* decline there, and `_initial_temperature_grid` already falls back to
`hot_grid_start` when it does.

The classes here drive `calcApproxPercolation` with a synthetic criterion whose roots are
known, so the walk is tested rather than the 2HDM.

Part of TransitionListener v2
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import errors


TMIN, TMAX = 1.0e-13, 120.0
#: The synthetic criterion is positive on this band and negative on both sides of it, so it
#: has two roots: the physical one at the hot edge and an artefact at the cold edge.
HOT_ROOT, COLD_ROOT = 60.0, 10.0


def _make_criterion(action_of_T):
    """Negative above `HOT_ROOT`, positive between the roots, negative below `COLD_ROOT`.

    It caches an action in `outdict` under the temperature it was asked about, the way the
    real criterion does through `calcAction`, because the walk reads the action from there
    rather than asking for it again.
    """

    def criterion(T, outdict, *args, **kwargs):
        T = float(np.squeeze(T))
        if T < TMAX and T > TMIN:
            outdict[T] = {"action": float(action_of_T(T))}
        if T >= TMAX:
            return -np.inf
        if T <= TMIN:
            return np.inf
        if T > HOT_ROOT:
            return -1.0
        if T > COLD_ROOT:
            return +1.0
        return -1.0

    return criterion


class _Patched:
    """Replaces the criterion and the action for the duration of a test."""

    def __init__(self, action_of_T, criterion=None):
        self.action_of_T = action_of_T
        self.criterion = criterion
        self._crit = bd._approx_percolation_criterion
        self._act = bd.calcAction
        self.action_calls = []
        self.probes = []

    def __enter__(self):
        inner = self.criterion or _make_criterion(self.action_of_T)
        def criterion(T, outdict, *args, **kwargs):
            self.probes.append(float(np.squeeze(T)))
            return inner(T, outdict, *args, **kwargs)
        bd._approx_percolation_criterion = criterion
        def action(pot, T, a, b, outdict, verbose=False, *ar, **kw):
            self.action_calls.append(float(T))
            return float(self.action_of_T(float(T)))
        bd.calcAction = action
        return self

    def __exit__(self, *exc):
        bd._approx_percolation_criterion = self._crit
        bd.calcAction = self._act
        return False


def estimate():
    """`calcApproxPercolation` with everything it does not need stubbed out."""
    pot = types.SimpleNamespace(
        config=types.SimpleNamespace(percolationConf=types.SimpleNamespace(f_perc=0.34)))
    return bd.calcApproxPercolation({}, pot, object(), object(), 1.0, verbose=False,
                                    tmin=TMIN, tmax=TMAX)


class TheEstimateTakesTheHottestRoot(unittest.TestCase):
    """Two roots in the bracket; the hotter one is the physical one.

    Percolation is the first temperature the universe meets on the way down, so a root
    colder than another is an artefact of the bracket rather than a competing answer.
    """

    def test_the_colder_root_is_not_returned(self):
        # S_3/T stays well under the nucleation threshold across the band, so the walk is
        # not stopped by it and the root is what decides.
        with _Patched(lambda T: 100.0 * T) as p:
            Tperc = float(estimate())
        self.assertAlmostEqual(
            Tperc, HOT_ROOT, delta=HOT_ROOT * 1e-2,
            msg=f"the estimate returned {Tperc!r}, not the hottest root at {HOT_ROOT}")
        self.assertGreater(len(p.probes), 1, "the walk never ran")
        self.assertEqual(
            [T for T in p.probes if T < COLD_ROOT], [],
            "the walk probed below the colder root, which it should never reach")


class TheEstimateDeclinesWhenThereIsNoRoot(unittest.TestCase):
    """A criterion that never turns positive has no percolation point to report.

    This is the 2HDM case at lambda3 = 5.815656566. Returning a number there is worse than
    declining, because `_initial_temperature_grid` has a fallback and a wrong number has
    none.
    """

    def test_an_always_negative_criterion_raises(self):
        def negative(T, outdict, *a, **k):
            T = float(np.squeeze(T))
            if TMIN < T < TMAX:
                outdict[T] = {"action": 100.0 * T}
            return -np.inf if T >= TMAX else (np.inf if T <= TMIN else -1.0)
        with _Patched(lambda T: 100.0 * T, criterion=negative) as p:
            with self.assertRaises(errors.PercolationApproximation1Error):
                estimate()
        self.assertGreater(len(p.probes), 1, "the walk never ran")


class TheWalkStopsWhereNucleationDies(unittest.TestCase):
    """A criterion with no root must not be walked down to the floor of the tracing range.

    The phase-overlap interval reaches `Tmin`, which on the 2HDM is 2.35e-13 GeV, and each
    step of the walk costs two action evaluations. Below the temperature where `S_3/T`
    passes the threshold no bubble can nucleate at any Hubble rate, so there is nothing to
    find and the walk should end there rather than at `Tmin`.
    """

    #: Below this the synthetic action is past the threshold.
    DEAD_BELOW = 5.0

    def test_a_rootless_walk_ends_at_the_dead_edge_not_at_tmin(self):
        from transitionlistener.percolation_adaptive_gridbuilders import (
            _NUCLEATION_DEAD_S_OVER_T,
        )
        dead = float(_NUCLEATION_DEAD_S_OVER_T)

        def action(T):
            return dead * 10.0 * T if T < self.DEAD_BELOW else 100.0 * T

        def negative(T, outdict, *a, **k):
            T = float(np.squeeze(T))
            if TMIN < T < TMAX:
                outdict[T] = {"action": float(action(T))}
            return -np.inf if T >= TMAX else (np.inf if T <= TMIN else -1.0)

        # The premise: the band really is live above DEAD_BELOW and dead below it.
        self.assertLess(action(HOT_ROOT) / HOT_ROOT, dead)
        self.assertGreater(action(1.0) / 1.0, dead)

        with _Patched(action, criterion=negative) as p:
            with self.assertRaises(errors.PercolationApproximation1Error):
                estimate()
        self.assertGreater(len(p.probes), 1, "the walk never ran")
        coldest = min(p.probes)
        self.assertGreater(
            coldest, TMIN * 1e3,
            f"the walk reached {coldest!r}, essentially the floor of the range, instead of "
            f"stopping where nucleation dies near {self.DEAD_BELOW}")
        self.assertLess(
            coldest, self.DEAD_BELOW,
            "the walk stopped before reaching the dead edge, so it never tested it")


class TheHotEndIsNotMistakenForTheColdOne(unittest.TestCase):
    """`S_3/T` is large at both ends of the band, for opposite reasons.

    The action diverges as `T` approaches the critical temperature from below, so the hot
    end has a large `S_3/T` because the transition has not started, while the cold end has
    one because the rate has died. The walk stops on the second and must not stop on the
    first: testing the threshold before the walk has entered the live band ends it at the
    very first probe, which reads `Tmax` as too cold.
    """

    def test_a_large_s3_over_t_at_the_hot_end_does_not_end_the_walk(self):
        from transitionlistener.percolation_adaptive_gridbuilders import (
            _NUCLEATION_DEAD_S_OVER_T,
        )
        dead = float(_NUCLEATION_DEAD_S_OVER_T)

        def action(T):
            # above 90 GeV the action diverges towards T_crit; in the band it is modest
            return dead * 10.0 * T if T > 90.0 else 100.0 * T

        # The premise: the hot end really is past the threshold, so the test can fail.
        self.assertGreater(action(100.0) / 100.0, dead)
        self.assertLess(action(HOT_ROOT) / HOT_ROOT, dead)

        with _Patched(action):
            Tperc = float(estimate())
        self.assertAlmostEqual(
            Tperc, HOT_ROOT, delta=HOT_ROOT * 1e-2,
            msg=f"the estimate returned {Tperc!r}; a large S_3/T at the hot end ended the "
                "walk before it reached the root")


if __name__ == "__main__":
    unittest.main()
