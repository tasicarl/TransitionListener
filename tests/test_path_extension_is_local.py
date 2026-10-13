"""The path extension must look for the nearest minimum, not wherever a fixed step lands.

`SplinePath(extend_to_minima=True)` extends the path past each end until the potential stops
falling. That search was `scipy.optimize.fmin(V_lin, 0.0, ...)`, whose initial simplex step
from `x0 = 0` is 2.5e-4 whatever the problem looks like -- the first four points it evaluates
are 0, 2.5e-4, -2.5e-4, 1.25e-4. The step is in units of the knot index, so in field units it
is 2.5e-4 times the spacing of the first two knots. On a path whose ends are far apart that is
larger than the distance to the barrier, the minimiser steps over it, and the extended path
connects two stable minima instead of a metastable one and a stable one. The solver then
reports "stable, not metastable" and the action comes back infinite. The same unbounded search
can also walk thousands of knots out and trip the 20000-knot cap with a `SplineError`.

Measured on the conformal U(1) at g = 0.54 with v_GeV = 1000: the path runs from the origin to
phi = 1000, so the step is 0.25 in field units, while the barrier top sits at 0.124 at
T_int = 0.1. The action was infinite there and finite at T_int = 0.149, which is the cutoff
reported from the dimred daisy work.

The cases below are the unit search itself and a two-field potential with the same geometry.
"""
import unittest

import numpy as np

from transitionlistener import errors, pathDeformation as pd


FMIN_FIRST_STEP = 2.5e-4   # scipy's initial simplex step from x0 = 0, measured


def _two_minima(near, barrier, far, depth=1.0):
    """A potential with a shallow minimum at `near`, a barrier at `barrier`, and a deep one at `far`.

    V' is proportional to (x - near)(x - barrier)(x - far), so the shape is exactly the one the
    extension has to resolve: the nearest minimum is the one the path should be extended to,
    and the deep one beyond the barrier is the wrong answer.
    """
    def V(p):
        x = np.asanyarray(p, dtype=float)
        x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
        poly = (0.25 * x**4
                - (near + barrier + far) / 3.0 * x**3
                + (near * barrier + near * far + barrier * far) / 2.0 * x**2
                - near * barrier * far * x)
        return depth * poly
    return V


class ExtensionSearchIsLocal(unittest.TestCase):
    def test_it_finds_the_nearest_minimum_and_not_the_one_past_the_barrier(self):
        """The whole defect in one assertion: the barrier must not be crossed."""
        near, barrier, far = 2.0e-5, 1.0e-4, 1.0
        self.assertLess(barrier, FMIN_FIRST_STEP,
                        "the construction is pointless unless the first fixed step clears "
                        "the barrier")
        V = _two_minima(near, barrier, far)
        found = pd._extension_minimum(V, np.array([0.0]), np.array([1.0]), +1.0)
        self.assertAlmostEqual(found, near, delta=0.1 * near,
                               msg=f"expected the nearest minimum at {near}, got {found}")
        self.assertLess(found, barrier,
                        f"the search crossed the barrier at {barrier} and returned {found}")

    def test_the_old_fixed_step_does_cross_that_barrier(self):
        """Mutation check for the test above: the search it replaced gets this wrong.

        Without this, the test above would pass just as well against the old code and would
        not be evidence of anything.
        """
        from scipy import optimize
        near, barrier, far = 2.0e-5, 1.0e-4, 1.0
        V = _two_minima(near, barrier, far)

        def V_lin(x, p0, dp0, Vf):
            return Vf(p0 + x * dp0)

        old = optimize.fmin(V_lin, 0.0, args=(np.array([0.0]), np.array([1.0]), V),
                            xtol=1e-6, disp=0)[0]
        self.assertGreater(old, barrier,
                           "the old search no longer crosses the barrier, so the fix and this "
                           "test are both about something that does not happen")
        self.assertAlmostEqual(old, far, delta=0.1 * far,
                               msg=f"the old search was expected at the far minimum {far}, "
                                   f"it returned {old}")

    def test_an_end_that_is_already_a_minimum_is_not_extended(self):
        """The common case: the ends handed in are the minima the phase tracing found."""
        V = _two_minima(2.0e-5, 1.0e-4, 1.0)
        at_near = pd._extension_minimum(V, np.array([2.0e-5]), np.array([1.0]), +1.0)
        self.assertEqual(at_near, 0.0,
                         f"an end sitting at a minimum was extended by {at_near}")

    def test_a_minimum_many_knots_away_is_still_reached(self):
        """What the old unbounded search could do, and must not be lost.

        A path whose ends are not minima can need an extension of many knots. The search walks
        a geometric ladder, so the reach has to be checked explicitly rather than assumed.
        """
        def V(p):
            x = np.asanyarray(p, dtype=float)
            x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
            return 0.5 * (x - 50.0) ** 2
        found = pd._extension_minimum(V, np.array([0.0]), np.array([1.0]), +1.0)
        self.assertAlmostEqual(found, 50.0, delta=0.05,
                               msg=f"a minimum 50 knots out was not reached: {found}")

    def test_the_front_end_searches_the_other_way(self):
        """`SplinePath` clamps the front to x <= 0, so a positive answer is silently dropped."""
        def V(p):
            x = np.asanyarray(p, dtype=float)
            x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
            return 0.5 * (x + 3.0) ** 2
        found = pd._extension_minimum(V, np.array([0.0]), np.array([1.0]), -1.0)
        self.assertLess(found, 0.0, f"the front search returned {found}, which is clamped away")
        self.assertAlmostEqual(found, -3.0, delta=0.05)


class TwoFieldPathWithASmallBarrier(unittest.TestCase):
    """The same geometry with two fields, where the extension is still on the code path."""

    @staticmethod
    def _potential(length, barrier):
        """A long path in the first field with a small barrier, and a tilted second field.

        The tilt is what makes this a two-field problem: the straight line between the minima
        is not the solution, so the deformation loop runs and the spline path is rebuilt.
        """
        tilt, curv = 0.02, 4.0

        def _along(x):
            return (0.25 * x**4
                    - (barrier + length) / 3.0 * x**3
                    + barrier * length / 2.0 * x**2) / length**2

        def V(p):
            p = np.asanyarray(p, dtype=float)
            a, b = p[..., 0], p[..., 1]
            return _along(a) + 0.5 * curv * (b - tilt * a * (1.0 - a / length)) ** 2

        def dV(p):
            p = np.asanyarray(p, dtype=float)
            a, b = p[..., 0], p[..., 1]
            off = b - tilt * a * (1.0 - a / length)
            da = ((a**3 - (barrier + length) * a**2 + barrier * length * a) / length**2
                  - curv * off * tilt * (1.0 - 2.0 * a / length))
            return np.stack([da, curv * off], axis=-1)

        return V, dV

    def test_a_long_path_with_a_small_barrier_now_returns_an_action(self):
        length, barrier = 1000.0, 0.12       # the conformal U(1) geometry at T_int = 0.1
        self.assertGreater(FMIN_FIRST_STEP * length, barrier,
                           "the construction is pointless unless the first fixed step clears "
                           "the barrier")
        V, dV = self._potential(length, barrier)
        ends = [np.array([length, 0.0]), np.array([0.0, 0.0])]
        result = pd.fullTunneling(ends, V, dV, maxiter=6, V_spline_samples=200000)
        self.assertTrue(np.isfinite(result.action),
                        f"the action is not finite: {result.action}")
        self.assertGreater(result.action, 0.0)

    def test_the_old_search_fails_on_that_same_potential(self):
        """Mutation check: put the fixed-step search back and the test above stops passing."""
        from scipy import optimize
        length, barrier = 1000.0, 0.12
        V, dV = self._potential(length, barrier)
        ends = [np.array([length, 0.0]), np.array([0.0, 0.0])]

        def fixed_step(Vf, p0, dp0, sign):
            def V_lin(x, a, b, c):
                return c(a + x * b)
            x = optimize.fmin(V_lin, 0.0, args=(p0, dp0, Vf), xtol=1e-6, disp=0)[0]
            return min(x, 0.0) if sign < 0 else max(x, 0.0)

        original = pd._extension_minimum
        pd._extension_minimum = fixed_step
        try:
            with self.assertRaises((errors.PotentialError, errors.SplineError,
                                    errors.InfiniteActionError)):
                pd.fullTunneling(ends, V, dV, maxiter=6, V_spline_samples=200000)
        finally:
            pd._extension_minimum = original




class WhenThereIsNoMinimumToExtendTo(unittest.TestCase):
    """What happens at the two ends of the range the search can cover."""

    def test_a_potential_that_keeps_falling_leaves_the_end_alone(self):
        """The tangent can point along a direction in which V simply never turns around.

        The old `fmin` returned zero there as well, so the end is left where it is rather than
        extended by an arbitrary amount. This is not the pathological case: it is a direction
        in which extending the path has no meaning.
        """
        def V(p):
            x = np.asanyarray(p, dtype=float)
            x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
            return -x
        found = pd._extension_minimum(V, np.array([0.0]), np.array([1.0]), +1.0)
        self.assertEqual(found, 0.0,
                         f"the end was extended by {found} towards no minimum at all")

    def test_a_minimum_beyond_the_splineable_range_leaves_the_end_alone(self):
        """A deliberate change from the old behaviour, so it is pinned here.

        The spline refuses a path longer than `_MAX_EXTENSION_KNOTS`, so a minimum past that
        cannot be extended to whatever the search does. The old unbounded `fmin` found it and
        the size guard then raised `SplineError`, which gave an infinite action. The search now
        reaches no further than the cap, finds nothing, and leaves the end where it is, so the
        action is computed on the path as supplied. That is the right answer whenever the ends
        are already the minima the phase tracing found, which is how `calcAction` calls it.
        """
        def V(p):
            x = np.asanyarray(p, dtype=float)
            x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
            return 0.5 * (x - 1.5 * pd._MAX_EXTENSION_KNOTS) ** 2
        found = pd._extension_minimum(V, np.array([0.0]), np.array([1.0]), +1.0)
        self.assertEqual(found, 0.0,
                         f"the end was extended by {found} towards a minimum that the spline "
                         f"cannot reach anyway")

    def test_the_search_reaches_as_far_as_the_spline_allows(self):
        """The cap and the ladder have to agree, or a reachable minimum is missed."""
        self.assertAlmostEqual(
            pd._extension_minimum(
                lambda p: 0.5 * (np.asanyarray(p, dtype=float)[..., 0]
                                 - 0.9 * pd._MAX_EXTENSION_KNOTS) ** 2,
                np.array([0.0]), np.array([1.0]), +1.0),
            0.9 * pd._MAX_EXTENSION_KNOTS, delta=1.0,
            msg="a minimum inside the splineable range was not reached")

    def test_a_potential_that_cannot_be_evaluated_leaves_the_end_alone(self):
        """A probe that raises is a wall, not a crash: the end stays where it is."""
        def V(p):
            x = np.asanyarray(p, dtype=float)
            x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
            if np.any(x > 1e-6):
                raise FloatingPointError("outside the region where V is defined")
            return -x
        found = pd._extension_minimum(V, np.array([0.0]), np.array([1.0]), +1.0)
        self.assertEqual(found, 0.0, f"the end was extended to {found} past a wall")


class ARiseMustClearTheRoundingOfThePotential(unittest.TestCase):
    """Flatness to machine precision near an end is not a minimum.

    The first version of the local search compared each probe with the one before it. Near a
    path end the potential can be flat to machine precision over many decades of the distance,
    so consecutive probes differ by rounding alone and the first apparent rise is noise. Measured
    on the 2HDM benchmark at T = 36.56: the potential at that end is -2.4e+07, so rounding is
    about 5e-09, and probes below 1e-4 knots are inside it. The search stopped at x = 4e-10 and
    lost the real minimum 11.01 knots out, which moved the action to 119.21 against a value
    converged at 143.69 over sample counts from 2000 to 200000.
    """

    MINIMUM_AT = 11.0
    BASE = -2.4e7
    DEPTH = 265.9

    @classmethod
    def _potential(cls):
        """A fall to a minimum, on a large offset, plus a wobble the size of its rounding."""
        def V(p):
            x = np.asanyarray(p, dtype=float)
            x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
            u = x / cls.MINIMUM_AT
            fall = cls.DEPTH * (2.0 * u - u * u)
            # the ladder's consecutive probes are about 12% apart, so a wobble that
            # oscillates in log x is non-monotonic from one probe to the next
            wobble = (4.0 * np.finfo(float).eps * abs(cls.BASE)
                      * np.sin(50.0 * np.log(np.maximum(x, 1e-300))))
            return cls.BASE - fall + wobble
        return V

    def test_the_construction_really_is_noise_dominated_near_the_end(self):
        """Otherwise the test below would not be about anything."""
        V = self._potential()
        rounding = np.finfo(float).eps * abs(self.BASE)
        probes = np.array([V(np.array([x])) for x in np.geomspace(1e-10, 1e-9, 12)])
        self.assertTrue(np.any(np.diff(probes) > 0.0),
                        "the probes near the end are monotonic, so there is no noise to resist")
        self.assertLess(float(np.max(np.abs(np.diff(probes)))), 100.0 * rounding,
                        "the variation near the end is larger than rounding, so it is real")

    def test_the_real_minimum_is_found_and_not_the_noise(self):
        found = pd._extension_minimum(self._potential(), np.array([0.0]), np.array([1.0]), +1.0)
        self.assertAlmostEqual(found, self.MINIMUM_AT, delta=0.05 * self.MINIMUM_AT,
                               msg=f"expected the minimum near {self.MINIMUM_AT}, got {found}")

    def test_comparing_consecutive_probes_instead_stops_at_the_noise(self):
        """Mutation check: the rule this replaced gets it wrong on the same potential."""
        V = self._potential()

        def without_the_noise_allowance(Vf, p0, dp0, sign):
            def value(x):
                return float(np.squeeze(Vf(p0 + (sign * x) * dp0)))
            ladder = np.geomspace(1e-10, 2e4, 300)
            previous = value(0.0)
            for index, x in enumerate(ladder):
                current = value(float(x))
                if current > previous:
                    return 0.0 if index == 0 else float(x)
                previous = current
            return 0.0

        stopped = without_the_noise_allowance(V, np.array([0.0]), np.array([1.0]), +1.0)
        self.assertLess(stopped, 0.1 * self.MINIMUM_AT,
                        f"the replaced rule reached {stopped}, so it did not stop at the noise "
                        f"and this test proves nothing")


class ExtendingMustNotDestroyTheMetastability(unittest.TestCase):
    """The metastable end has to stay the higher of the two, or there is nothing to tunnel.

    The conformal U(1) potential is even in the field, so walking outward from the symmetric
    minimum reaches the mirror of the broken minimum, which is exactly degenerate with it. The
    old search extended the two-knot path a full knot to it, which doubled the path to 2000
    units and left it connecting two equal minima; the solver compared the ends and reported
    "stable, not metastable". Measured at g = 0.54: the broken minimum and the point one knot
    past the symmetric end both sit at -2.0220e+08.

    The floor is the potential at the other end of the path. It has to separate that case from a
    real minimum a little way out, which must still be found: on the 2HDM at T = 36.56 the
    minimum 11.01 knots past the end is at -2.4348e+07 while the broken minimum is far below at
    -6.79e+07, and that extension is kept.
    """

    @staticmethod
    def _even_potential(depth, distance):
        """Minima of equal depth either side of the end, as an even potential has."""
        def V(p):
            x = np.asanyarray(p, dtype=float)
            x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
            u = np.abs(x) / distance
            return -depth * (2.0 * u - u * u)
        return V

    def test_a_degenerate_mirror_minimum_is_refused(self):
        depth, distance = 2.0220e8, 1.0
        V = self._even_potential(depth, distance)
        found = pd._extension_minimum(V, np.array([0.0]), np.array([1.0]), +1.0,
                                      floor=-depth)
        self.assertEqual(found, 0.0,
                         f"the end was extended by {found} to a minimum degenerate with the "
                         f"other end, which leaves nothing to tunnel through")

    def test_without_the_floor_it_walks_to_that_mirror(self):
        """Mutation check: the floor is what refuses it, not anything else in the search."""
        depth, distance = 2.0220e8, 1.0
        V = self._even_potential(depth, distance)
        found = pd._extension_minimum(V, np.array([0.0]), np.array([1.0]), +1.0)
        self.assertAlmostEqual(found, distance, delta=0.05 * distance,
                               msg=f"without the floor the search gave {found}, so the floor "
                                   f"is not what makes the difference")

    def test_a_minimum_well_above_the_other_end_is_still_kept(self):
        """The 2HDM case: shallow against the depth of the true minimum, so it is allowed."""
        V = self._even_potential(2.659e2, 11.0)
        found = pd._extension_minimum(V, np.array([0.0]), np.array([1.0]), +1.0,
                                      floor=-6.79e7)
        self.assertAlmostEqual(found, 11.0, delta=0.5,
                               msg=f"a minimum far above the other end was refused: {found}")


if __name__ == "__main__":
    unittest.main()
