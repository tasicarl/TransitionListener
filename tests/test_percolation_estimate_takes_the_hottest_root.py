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

Against the code this replaces, the only class that passes is
`TheHotEndIsNotMistakenForTheColdOne`: handing the whole interval to `brentq` happens to
converge to the hot root on this synthetic, which is why the 2HDM measurement above is what
the defect rests on and not this fixture. Each class was also measured against the current
code with one behaviour put back, and each mutation is caught. The shorthand below names a
class by its subject: the hottest root, the decline, the dead edge, the hot end, the first
step, no computable action, the interrupted stretch.

    behaviour put back                                classes that fail
    the whole interval handed to `brentq`             all but the hot end
    the walk keeps the coldest sign change            the hottest root, the first step,
                                                       the hot end,
                                                       the interrupted stretch
    the `-inf` at `Tmax` read as carrying no sign     the first step
    that stand-in given the wrong sign                the hottest root, the first step,
                                                       the hot end, the dead edge
    the nucleation-dead stop dropped                  the dead edge
    the stop tested before the live band is entered   the hot end, no computable action,
                                                       the interrupted stretch
    the finiteness test on the cold end dropped       the decline, no computable action,
                                                       the interrupted stretch
    the blind-step bound dropped                      no computable action
    blind steps counted cumulatively                  the dead edge,
                                                       the interrupted stretch
    the live test ignores `_nucleation_is_dead`       the dead edge,
                                                       the interrupted stretch
    `calcAction` called again instead of the cache    the dead edge

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


def _make_criterion(action_of_T, hot_root=HOT_ROOT):
    """Negative above `hot_root`, positive between the roots, negative below `COLD_ROOT`.

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
        if T > hot_root:
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


class AMinusInfinityMidWalkDoesNotHideTheRoot(unittest.TestCase):
    """A definite "too hot" one step above the root must still bracket it.

    The criterion answers `-inf` without measuring anything for `T >= Tmax` and for
    `T + dT >= Tmax`, and again when an action is exactly zero at either sample. Requiring
    both ends of a step to be finite throws those answers away, and the walk steps over
    any root that lies just below one of them. Carrying the structural `-inf` at `Tmax`
    alone is not enough, because the same answer can appear at any step.

    `+inf` is deliberately not treated this way: the criterion returns it at `T <= Tmin`
    by construction and wherever an action cannot be computed, so bracketing against it
    manufactures a root at the cold end, which is the defect the walk exists to remove.
    `TheEstimateDeclinesWhenThereIsNoRoot` is what holds that line.
    """

    #: One step above `HOT_ROOT` on the geometric walk from `TMAX`.
    BAD = 69.0

    def test_a_degenerate_step_above_the_root_still_brackets_it(self):
        def criterion(T, outdict, *a, **k):
            T = float(np.squeeze(T))
            if TMIN < T < TMAX:
                outdict[T] = {"action": 100.0 * T}
            if T >= TMAX:
                return -np.inf
            if T <= TMIN:
                return np.inf
            if abs(T - self.BAD) < 1.0:
                return -np.inf
            return -1.0 if T > HOT_ROOT else (+1.0 if T > COLD_ROOT else -1.0)

        # The premise: the degenerate patch really does sit on a step of the walk, one
        # step above the root, or the test cannot fail.
        probe, steps = TMAX, []
        for _ in range(8):
            probe /= bd._APPROX_PERCOLATION_SCAN_RATIO
            steps.append(probe)
        self.assertTrue(
            any(abs(t - self.BAD) < 1.0 for t in steps),
            f"the walk does not land in the patch at {self.BAD}; steps are {steps}")

        with _Patched(lambda T: 100.0 * T, criterion=criterion) as p:
            Tperc = float(estimate())
        self.assertGreater(len(p.probes), 1, "the walk never ran")
        self.assertAlmostEqual(
            Tperc, HOT_ROOT, delta=HOT_ROOT * 1e-2,
            msg=f"the estimate returned {Tperc!r}; a -inf one step above the root hid it, "
                f"and the artefact at {COLD_ROOT} was returned instead")


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
        # And it read `S_3/T` from the action the criterion had just cached, rather than
        # asking `calcAction` for one of its own: here the criterion is replaced, so it
        # computes no action, and any call recorded is the walk's.
        self.assertEqual(
            p.action_calls, [],
            f"the walk called calcAction itself at {p.action_calls!r} instead of reading "
            "the entry the criterion cached")


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


class ARootInsideTheFirstStepIsNotSteppedOver(unittest.TestCase):
    """The walk starts where the criterion is `-inf` by construction, and has to use it.

    `_approx_percolation_criterion` answers "too hot" for `T >= Tmax` before it computes
    an action, so the walk's first comparison is against `-inf` for every model. Read as
    "no usable sign" the whole first step is skipped, and a root inside it is stepped
    over: the walk carries on down, brackets the artefact at the cold end of the band and
    returns that instead, with no complaint. The sign is definite, so it is carried as a
    finite negative value and `brentq` is handed `Tmax` itself, which is what the
    whole-interval call did at every point TransitionListener has ever run.

    The first step spans a factor `_APPROX_PERCOLATION_SCAN_RATIO`, 13 per cent in
    temperature. Measured on the 2HDM at lambda3 = 5.5, the top of the phase-overlap
    interval is 124.135 GeV and the root sits at 89.5357, outside it by a factor 1.2.
    """

    #: Inside the first step of the walk, which the premise below asserts rather than
    #: assumes, so the test follows `_APPROX_PERCOLATION_SCAN_RATIO` if it changes.
    NEAR_ROOT = 110.0

    def test_a_root_within_one_step_of_tmax_is_the_one_returned(self):
        # The premises. First, that the criterion really is non-finite at `Tmax`,
        # measured on the real one: it returns before touching any of its arguments.
        self.assertEqual(
            bd._approx_percolation_criterion(TMAX, {}, None, None, None, TMAX, TMIN, 1.0),
            -np.inf,
            "the criterion no longer returns -inf at Tmax, so this test is about nothing")
        # Second, that the root is inside the first step, where the skipped comparison is.
        self.assertGreater(
            self.NEAR_ROOT, TMAX / bd._APPROX_PERCOLATION_SCAN_RATIO,
            f"a root at {self.NEAR_ROOT} is not inside the first step, which reaches "
            f"{TMAX / bd._APPROX_PERCOLATION_SCAN_RATIO}")

        def action_of_T(T):
            return 100.0 * T               # live across the whole band

        with _Patched(action_of_T,
                      criterion=_make_criterion(action_of_T,
                                                hot_root=self.NEAR_ROOT)) as p:
            Tperc = float(estimate())
        self.assertAlmostEqual(
            Tperc, self.NEAR_ROOT, delta=self.NEAR_ROOT * 1e-2,
            msg=f"the estimate returned {Tperc!r} for a root at {self.NEAR_ROOT}; the "
                f"colder artefact sits at {COLD_ROOT}")
        self.assertEqual(
            [T for T in p.probes if T < COLD_ROOT], [],
            "the walk probed below the colder root, so it stepped over the hot one")


class TheWalkStopsWhereNoActionCanBeComputed(unittest.TestCase):
    """A probe without a computable action can bracket nothing, and must not be paid for.

    The criterion returns `+inf` when `calcAction` gives it a non-finite action, so such a
    probe carries no sign; and the nucleation-dead stop reads `S_3/T` from the action the
    criterion cached, so it cannot see one either. Without a bound of its own the walk
    then pays for the whole interval: over the 2HDM's [2.35254e-13, 121.839] GeV, 14.7
    decades, that is 243 steps and 484 action evaluations, measured with every action
    non-finite. Each of those is a bounce solve, and a failing solve is not cheap.
    """

    def test_a_walk_with_no_computable_action_declines_within_the_bound(self):
        def criterion(T, outdict, *args, **kwargs):
            T = float(np.squeeze(T))
            if TMIN < T < TMAX:
                outdict[T] = {"action": np.inf}
            return -np.inf if T >= TMAX else np.inf

        with _Patched(lambda T: np.inf, criterion=criterion) as p:
            with self.assertRaises(errors.PercolationApproximation1Error):
                estimate()

        steps = [T for T in p.probes if T < TMAX]
        self.assertGreater(len(steps), 1, "the walk never ran")
        allowed = int(bd._APPROX_PERCOLATION_MAX_BLIND_STEPS)
        self.assertLessEqual(
            len(steps), allowed,
            f"the walk took {len(steps)} steps without a computable action, past the "
            f"{allowed} it is allowed, and reached {min(steps)!r}")
        # The premise: crossing this bracket would take far more steps than the bound, so
        # the bound is what stopped the walk rather than the bracket running out.
        crossing = np.log(TMAX / TMIN) / np.log(bd._APPROX_PERCOLATION_SCAN_RATIO)
        self.assertGreater(
            crossing, 10 * allowed,
            f"crossing the bracket takes {crossing} steps, not many more than the "
            f"{allowed} allowed, so this test would pass without the bound")


class AnInterruptedStretchOfFailedActionsIsWalkedThrough(unittest.TestCase):
    """The bound is on consecutive failures, so an isolated solve does not spend it.

    The failures this release fixed were *interleaved* rather than contiguous: at
    lambda3 = 5.8191375291 the action came back at 61.54, 59, 57, 55 and 50 GeV and was
    absent at 61.52, 61, 60, 58 and 56. A cumulative bound would be spent by that pattern
    and the walk would decline above the root; a bound on consecutive failures walks
    through it. The hot end supplies the isolated solve naturally, since its action is
    finite and its `S_3/T` large.
    """

    #: Steps below `Tmax`, counted in `_APPROX_PERCOLATION_SCAN_RATIO`. The two runs of
    #: failures are each shorter than the bound and together longer than it.
    SOLVED_STEP, ROOT_STEP = 13, 26

    def _step_of(self, T):
        return int(round(np.log(TMAX / T) / np.log(bd._APPROX_PERCOLATION_SCAN_RATIO)))

    def test_two_short_runs_of_failures_do_not_add_up_to_the_bound(self):
        from transitionlistener.percolation_adaptive_gridbuilders import (
            _NUCLEATION_DEAD_S_OVER_T,
        )
        dead = float(_NUCLEATION_DEAD_S_OVER_T)
        over_the_threshold = 10.0   # how far past it the isolated solve's S_3/T sits
        allowed = int(bd._APPROX_PERCOLATION_MAX_BLIND_STEPS)
        root = TMAX / bd._APPROX_PERCOLATION_SCAN_RATIO ** (self.ROOT_STEP + 0.5)

        # The premises: neither run of failures reaches the bound on its own, and the two
        # together pass it, so the test separates "consecutive" from "cumulative".
        first_run, second_run = self.SOLVED_STEP - 1, self.ROOT_STEP - self.SOLVED_STEP - 1
        self.assertLess(max(first_run, second_run), allowed,
                        "a single run of failures already reaches the bound")
        self.assertGreater(first_run + second_run, allowed,
                           "the two runs together stay under the bound, so a cumulative "
                           "count would pass this test as well")

        def action(T):
            step = self._step_of(T)
            if step == self.SOLVED_STEP:
                return dead * over_the_threshold * T   # finite, far too cold to nucleate
            if step >= self.ROOT_STEP:
                return 100.0 * T                   # live
            return np.inf                          # no computable bounce

        def criterion(T, outdict, *args, **kwargs):
            T = float(np.squeeze(T))
            if TMIN < T < TMAX:
                outdict[T] = {"action": float(action(T))}
            if T >= TMAX:
                return -np.inf
            if T <= TMIN:
                return np.inf
            if not np.isfinite(action(T)):
                return np.inf
            return -1.0 if T > root else +1.0

        with _Patched(action, criterion=criterion) as p:
            Tperc = float(estimate())

        self.assertAlmostEqual(
            Tperc, root, delta=root * 1e-2,
            msg=f"the estimate returned {Tperc!r} rather than the root at {root!r} below "
                "two runs of failed solves")
        # And the isolated solve really was the only one above the root, so it is what
        # kept the walk going.
        solved = [T for T in p.probes
                  if T < TMAX and np.isfinite(action(T)) and T > root]
        self.assertEqual(
            [self._step_of(T) for T in solved if self._step_of(T) < self.ROOT_STEP],
            [self.SOLVED_STEP],
            "more than one solve above the root, so the runs of failures are not what "
            "this test says they are")


if __name__ == "__main__":
    unittest.main()
