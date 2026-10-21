"""A traced phase has to be a minimum at every temperature it claims.

The potential here is separable and its answer is known in closed form, so each assertion
compares against arithmetic rather than against another run of the code under test:

    V(x, y, t) = (c t^2 - m2) x^2 / 2 + l x^4 / 4 + k y^2 / 2

The origin is stationary for every `t`. Its Hessian is ``diag(c t^2 - m2, k)``, so it is a
minimum above ``t0 = sqrt(m2 / c)`` and a saddle below, where the minima sit at
``x = +-sqrt((m2 - c t^2) / l)``, ``y = 0``. That is the structure of a thermal mass term
tilting the origin over, which is what the 2HDM does between 39.0 GeV and 8.5 GeV at
lambda3 = 5.813.

Part of TransitionListener v2
"""
import unittest

import numpy as np

from transitionlistener import phases as ph

C, M2, LAM, K = 1.0, 4.0, 1.0, 3.0
T0 = np.sqrt(M2 / C)          # 2.0: above it the origin is a minimum, below it a saddle


def V(x, t):
    x = np.asarray(x, dtype=float)
    return (0.5 * (C * t ** 2 - M2) * x[..., 0] ** 2
            + 0.25 * LAM * x[..., 0] ** 4
            + 0.5 * K * x[..., 1] ** 2)


def dV(x, t):
    x = np.asarray(x, dtype=float)
    return np.array([(C * t ** 2 - M2) * x[..., 0] + LAM * x[..., 0] ** 3,
                     K * x[..., 1]])


def d2V(x, t):
    x = np.asarray(x, dtype=float)
    return np.array([[(C * t ** 2 - M2) + 3.0 * LAM * x[..., 0] ** 2, 0.0],
                     [0.0, K]])


def d2V_dxdt(x, t):
    x = np.asarray(x, dtype=float)
    return np.array([2.0 * C * t * x[..., 0], 0.0])


def displaced_minimum(t):
    """The closed-form minimum below `T0`, on the positive side."""
    return np.array([np.sqrt((M2 - C * t ** 2) / LAM), 0.0])


def smallest_eigenvalue(x, t):
    return float(np.linalg.eigvalsh(np.atleast_2d(np.squeeze(d2V(x, t))))[0])


class APhaseIsNotTracedWhereItIsNotAMinimum(unittest.TestCase):
    """The whole point of the saddle test in `traceMinimum`.

    It was written as `negeig is True` against a `numpy.bool_`, which is never the `True`
    singleton, so it never fired and the trace walked on through.
    """

    #: `traceMinimum` passes `xtol = DELTA_X_TARGET * 1e-2` to the minimiser, and
    #: `optimize.fmin` seeds a zero coordinate with a simplex of 0.00025, so above
    #: `xtol = 2.5e-04` the minimiser returns the origin untouched instead of rolling off
    #: it. The shipped value is 100 * `x_eps` = 0.1, which is well above that, and the
    #: check under test is then the only thing that can stop the trace. At 1e-3 the
    #: minimiser escapes on its own and this file would pass with the defect restored.
    DELTA_X_TARGET = 0.1

    def trace_down_from_above(self):
        return ph.traceMinimum(
            V, dV, d2V_dxdt, d2V, np.array([0.0, 0.0]), 3.0 * T0, 0.0,
            -0.05 * T0, self.DELTA_X_TARGET,
        )

    def test_no_traced_node_has_a_negative_curvature(self):
        X, T = self.trace_down_from_above()[0], self.trace_down_from_above()[1]
        bad = [(float(t), smallest_eigenvalue(x, t))
               for x, t in zip(np.atleast_2d(X), np.asarray(T, dtype=float))
               if smallest_eigenvalue(x, t) < 0.0]
        self.assertEqual(
            bad, [],
            msg=f"the trace kept {len(bad)} node(s) where the origin is a saddle: {bad[:4]}")

    def test_the_trace_stops_at_the_temperature_where_the_origin_tilts_over(self):
        T = np.asarray(self.trace_down_from_above()[1], dtype=float)
        self.assertGreater(
            T.min(), T0 - 0.25 * T0,
            msg=f"the trace ran down to {T.min()}, past t0 = {T0}")
        self.assertLess(
            T.min(), T0 + 0.25 * T0,
            msg=f"the trace stopped at {T.min()}, well above t0 = {T0}, so it is not the "
                "saddle that stopped it")

    def test_the_fixture_really_is_the_hard_case(self):
        """Without a working check the trace has somewhere wrong to go."""
        self.assertLess(smallest_eigenvalue(np.zeros(2), 0.5 * T0), 0.0)
        self.assertGreater(smallest_eigenvalue(np.zeros(2), 2.0 * T0), 0.0)
        self.assertAlmostEqual(
            float(np.linalg.norm(dV(np.zeros(2), 0.5 * T0))), 0.0, places=12,
            msg="the origin has to stay stationary below t0, or a minimiser would leave "
                "it on its own and the check under test would be unnecessary")

    def test_the_minimiser_does_not_leave_the_saddle_by_itself(self):
        """Otherwise the trace would stop for a reason other than the one under test."""
        from scipy import optimize
        stayed = optimize.fmin(V, np.zeros(2), args=(0.5 * T0,),
                               xtol=self.DELTA_X_TARGET * 1.0e-2, ftol=np.inf, disp=False)
        self.assertLess(
            float(np.linalg.norm(stayed)), 0.1 * float(displaced_minimum(0.5 * T0)[0]),
            msg=f"the minimiser walked to {stayed} on its own, so this file would pass "
                "with the saddle check removed")


class ASeedOnASaddleIsMovedToTheMinimumBesideIt(unittest.TestCase):
    """`descendFromSaddle`, which the gradient alone cannot do at a symmetric point."""

    def descend(self, t, max_displacement=10.0, start=None):
        start = np.zeros(2) if start is None else start
        return np.asarray(ph.descendFromSaddle(
            start, t, V, dV, d2V, 1.0e-5, 1.0e-3, max_displacement), dtype=float)

    def test_it_reaches_the_minimum_the_closed_form_gives(self):
        t = 0.5 * T0
        got = self.descend(t)
        self.assertAlmostEqual(
            abs(float(got[0])), float(displaced_minimum(t)[0]), places=4,
            msg=f"descended to {got}, expected +-{displaced_minimum(t)}")
        self.assertAlmostEqual(float(got[1]), 0.0, places=4)

    def test_a_point_that_is_already_a_minimum_is_returned_unchanged(self):
        t = 2.0 * T0
        got = self.descend(t)
        self.assertTrue(
            np.array_equal(got, np.zeros(2)),
            msg=f"the origin is a minimum at t = {t} and was moved to {got}")

    def test_a_minimum_further_than_the_bound_is_not_returned(self):
        """The bound is what keeps the descent from delivering a distant minimum."""
        t = 0.5 * T0
        reach = float(displaced_minimum(t)[0])
        got = self.descend(t, max_displacement=0.25 * reach)
        self.assertTrue(
            np.array_equal(got, np.zeros(2)),
            msg=f"the only minimum is at {reach}, the bound was {0.25 * reach}, "
                f"and {got} came back")

    def test_the_result_is_a_minimum_and_lower_than_the_saddle(self):
        t = 0.5 * T0
        got = self.descend(t)
        self.assertGreater(smallest_eigenvalue(got, t), 0.0)
        self.assertLess(float(V(got, t)), float(V(np.zeros(2), t)))


class TheTracedPhasesStayInTheCanonicalRegion(unittest.TestCase):
    """Which image of a displaced minimum is found decides the action downstream.

    The descent follows an eigenvector whose sign is arbitrary, so without mapping the
    result back the same model lands on either image from one coupling to the next. On the
    2HDM that is `S_3/T` of 135.8 against 151.7 at T = 37 GeV.
    """

    @staticmethod
    def canonical(x):
        return np.abs(np.asarray(x, dtype=float))

    def traced(self):
        # Seeded at the origin above `T0`, which is where the 2HDM's own starting points
        # lead. The trace stops as the origin tilts over, and the displaced branch below
        # is only reached by stepping off the saddle: a minimiser seeded on it returns it
        # unchanged at this `deltaX_target`.
        # The seed sits inside the range, not on `tHigh`: `traceMultiMin` discards a
        # candidate that sits on the upper bound with a step that would leave it.
        return ph.traceMultiMin(
            V, dV, d2V_dxdt, d2V,
            [[np.zeros(2), 2.5 * T0]],
            tLow=0.05 * T0, tHigh=3.0 * T0, deltaX_target=0.1,
            applySymmetries=self.canonical,
        )

    def test_the_branch_below_the_tilt_is_traced_at_all(self):
        """Without the descent there is no phase below `T0`, and the coupling is lost."""
        probe = 0.5 * T0
        expected = float(displaced_minimum(probe)[0])
        covering = [p for p in self.traced().values()
                    if float(p.Tmin) <= probe <= float(p.Tmax)]
        self.assertTrue(
            covering,
            msg=f"no phase covers t = {probe}, where the minimum sits at x = {expected}")
        best = min(covering,
                   key=lambda p: abs(abs(float(np.atleast_1d(p.valAt(probe))[0])) - expected))
        got = abs(float(np.atleast_1d(best.valAt(probe))[0]))
        self.assertAlmostEqual(
            got, expected, places=2,
            msg=f"the phase covering t = {probe} sits at x = {got}, not at the displaced "
                f"minimum {expected}; a phase left at the origin there is a saddle")

    def test_every_node_of_every_phase_is_canonical(self):
        # The tolerance is on the image, not on the minimiser. A node in the wrong image
        # sits a whole displacement away, about 1.7 here. Near the merge the displacement
        # itself goes to zero and the two images meet, so which side of it a node lands on
        # is not defined there; those nodes carry a few parts in a thousand.
        for key, phase in self.traced().items():
            X = np.atleast_2d(np.asarray(phase.X, dtype=float))
            offending = X[np.any(X < -0.05, axis=1)]
            self.assertEqual(
                offending.size, 0,
                msg=f"phase {key} has {len(offending)} node(s) outside the canonical "
                    f"region, e.g. {offending[:2]}")

    def test_the_descent_on_its_own_can_leave_the_canonical_region(self):
        """The fixture has to be able to fail, or the mapping back proves nothing."""
        t = 0.5 * T0
        got = np.asarray(ph.descendFromSaddle(np.zeros(2), t, V, dV, d2V,
                                              1.0e-5, 1.0e-3, 10.0), dtype=float)
        self.assertAlmostEqual(abs(float(got[0])), float(displaced_minimum(t)[0]), places=4)
        self.assertFalse(
            np.array_equal(got, self.canonical(got)) and float(got[0]) == 0.0,
            msg="the descent returned the origin, so this fixture tests nothing")


class ASeedThatIsStillASaddleStartsNoPhase(unittest.TestCase):
    """What happens when the descent cannot reach the minimum within its bound.

    The bound is a quarter of the field scale, and with no field scale among the starting
    points it falls back to the temperature range. Over a range narrow enough, the bound
    is shorter than the displacement just below the tilt, the descent returns the saddle
    it was given, and the candidate has to be dropped. Tracing from it anyway is the
    original defect by another route.
    """

    T_LOW, T_HIGH = 0.995 * T0, 1.005 * T0

    def traced(self):
        return ph.traceMultiMin(
            V, dV, d2V_dxdt, d2V,
            [[np.zeros(2), 1.0025 * T0]],
            tLow=self.T_LOW, tHigh=self.T_HIGH, deltaX_target=0.1,
        )

    def test_the_bound_really_is_too_short_here(self):
        bound = 0.25 * (self.T_HIGH - self.T_LOW)
        jump = 1.0e-3 * (self.T_HIGH - self.T_LOW)
        reach = float(displaced_minimum(T0 - jump)[0])
        self.assertLess(
            bound, reach,
            msg=f"the bound {bound} already reaches the minimum at {reach}, so the "
                "descent does not fail here and this fixture tests nothing")

    def test_no_trace_is_started_from_a_point_that_is_not_a_minimum(self):
        """`traceMinimum` keeps its seed whatever it is, so the caller has to vet it.

        Whether that seed survives into a phase afterwards depends on how many nodes the
        trace manages and on the rule that drops a single-point phase, which is not a
        property to lean on. The seed itself is the thing under control here.
        """
        seeds = []
        real = ph.traceMinimum

        def spy(f, df_dx, d2f_dxdt, d2f_dx2, x0, t0, *a, **k):
            seeds.append((np.asarray(x0, dtype=float).copy(), float(t0)))
            return real(f, df_dx, d2f_dxdt, d2f_dx2, x0, t0, *a, **k)

        ph.traceMinimum = spy
        try:
            self.traced()
        finally:
            ph.traceMinimum = real
        self.assertTrue(seeds, msg="no trace was started at all, so nothing was tested")
        bad = [(x, t) for x, t in seeds if smallest_eigenvalue(x, t) <= 0.0]
        self.assertEqual(
            bad, [],
            msg=f"{len(bad)} of {len(seeds)} traces were started from a point that is "
                f"not a minimum, e.g. x = {bad[0][0] if bad else None} at "
                f"t = {bad[0][1] if bad else None}")

    def test_no_phase_holds_a_node_that_is_not_a_minimum(self):
        for key, phase in self.traced().items():
            for x, t in zip(np.atleast_2d(np.asarray(phase.X, dtype=float)),
                            np.asarray(phase.T, dtype=float)):
                self.assertGreater(
                    smallest_eigenvalue(x, t), 0.0,
                    msg=f"phase {key} holds a node at t = {t}, x = {x} where the "
                        f"smallest eigenvalue is {smallest_eigenvalue(x, t)}")


class AForbiddenCandidateIsSkipped(unittest.TestCase):
    """`forbidCrit` is model-supplied and the shipped ones return a `numpy.bool_`.

    `TL_dark_U1.forbidPhaseCrit` is `(np.array([X])[..., 0] < -5.).any()`, which an
    identity test against `True` never matches, so the criterion that is there to stop a
    Z2-symmetric model double-counting its phases skipped nothing.
    """

    @staticmethod
    def forbid_negative_x(x):
        """A `numpy.bool_`, as every shipped implementation of this hook returns."""
        return (np.asarray(x, dtype=float)[..., 0] < 0.0).any()

    def test_the_hook_returns_a_numpy_bool(self):
        got = self.forbid_negative_x(np.array([-1.0, 0.0]))
        self.assertIsInstance(got, np.bool_)
        self.assertIsNot(got, True, msg="this fixture no longer reproduces the case")

    def test_no_traced_phase_starts_in_the_forbidden_region(self):
        seeds = []
        real = ph.traceMinimum

        def spy(f, df_dx, d2f_dxdt, d2f_dx2, x0, t0, *a, **k):
            seeds.append(np.asarray(x0, dtype=float).copy())
            return real(f, df_dx, d2f_dxdt, d2f_dx2, x0, t0, *a, **k)

        ph.traceMinimum = spy
        try:
            ph.traceMultiMin(
                V, dV, d2V_dxdt, d2V,
                [[-displaced_minimum(0.5 * T0), 0.5 * T0]],
                tLow=0.05 * T0, tHigh=3.0 * T0, deltaX_target=0.1,
                forbidCrit=self.forbid_negative_x,
            )
        finally:
            ph.traceMinimum = real
        forbidden = [x for x in seeds if bool(self.forbid_negative_x(x))]
        self.assertEqual(
            [list(np.round(x, 5)) for x in forbidden], [],
            msg=f"{len(forbidden)} of {len(seeds)} traces were started from a point the "
                "model forbids")


class TheDescentReturnsTheNearerMinimum(unittest.TestCase):
    """Which side of a saddle the descent lands on may not depend on an eigenvector's sign.

    `linalg.eigh` fixes the sign of its eigenvectors by a convention, not by the physics, so
    taking the first direction that finds a minimum decides the answer arbitrarily. Where
    the two directions lead to minima at different distances, the phase being traced is the
    nearer one; the farther one can be a different phase entirely.

    The potential is one-dimensional and asymmetric, with a saddle at the origin and two
    minima at known, unequal distances:

        V(x) = x^4 / 4 - x^2 / 2 + g x^3 / 3

    whose stationary points are the roots of `x (x^2 + g x - 1)`. Both signs of `g` are
    used, so that the nearer minimum lies on the positive side in one case and the negative
    side in the other. One sign alone cannot tell the rule from the direction that happens
    to be tried first.
    """

    @staticmethod
    def _potential(g):
        def V(x, t):
            x = np.asarray(x, dtype=float)
            return x[..., 0] ** 4 / 4.0 - x[..., 0] ** 2 / 2.0 + g * x[..., 0] ** 3 / 3.0

        def dV(x, t):
            x = np.asarray(x, dtype=float)
            return np.array([x[..., 0] ** 3 - x[..., 0] + g * x[..., 0] ** 2])

        def d2V(x, t):
            x = np.asarray(x, dtype=float)
            return np.array([[3.0 * x[..., 0] ** 2 - 1.0 + 2.0 * g * x[..., 0]]])
        return V, dV, d2V

    @staticmethod
    def _minima(g):
        """The two roots of `x^2 + g x - 1`, nearer first."""
        root = np.sqrt(g ** 2 + 4.0)
        pair = (0.5 * (-g + root), 0.5 * (-g - root))
        return min(pair, key=abs), max(pair, key=abs)

    def test_the_fixture_is_what_the_arithmetic_says(self):
        """Checked before it is used to judge anything, and that the two signs differ."""
        near_signs = set()
        for g in (0.6, -0.6):
            V, dV, d2V = self._potential(g)
            near, far = self._minima(g)
            for root in (near, far):
                self.assertAlmostEqual(float(dV(np.array([root]), 0.0)[0]), 0.0, places=12)
                self.assertGreater(float(np.squeeze(d2V(np.array([root]), 0.0))), 0.0)
            self.assertLess(abs(near), abs(far))
            near_signs.add(np.sign(near))
        self.assertEqual(near_signs, {1.0, -1.0},
                         msg="both orientations are needed, or the test cannot tell the "
                             "nearest-minimum rule from the order the directions are tried")

    def test_the_nearer_minimum_is_returned_whichever_side_it_is_on(self):
        for g in (0.6, -0.6):
            with self.subTest(g=g):
                V, dV, d2V = self._potential(g)
                near, _ = self._minima(g)
                got = float(np.atleast_1d(ph.descendFromSaddle(
                    np.array([0.0]), 0.0, V, dV, d2V,
                    xeps=1e-8, step=1e-3, max_displacement=4.0))[0])
                self.assertAlmostEqual(
                    got, near, places=4,
                    msg=f"g = {g}: landed on {got}, and the nearer minimum is {near}")

    def test_a_point_that_is_already_a_minimum_is_left_alone(self):
        V, dV, d2V = self._potential(0.6)
        start = np.array([self._minima(0.6)[0]])
        got = ph.descendFromSaddle(start, 0.0, V, dV, d2V,
                                   xeps=1e-8, step=1e-3, max_displacement=4.0)
        np.testing.assert_allclose(np.atleast_1d(got), start, atol=0.0)

    def test_a_minimum_beyond_the_bound_is_not_taken(self):
        for g in (0.6, -0.6):
            with self.subTest(g=g):
                V, dV, d2V = self._potential(g)
                near, far = self._minima(g)
                got = float(np.atleast_1d(ph.descendFromSaddle(
                    np.array([0.0]), 0.0, V, dV, d2V, xeps=1e-8, step=1e-3,
                    max_displacement=0.5 * (abs(near) + abs(far))))[0])
                self.assertAlmostEqual(got, near, places=4)


def _phase(key, T, X):
    T = np.asarray(T, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return ph.PhaseInfo(key, X, T, np.zeros_like(X))


class PhasesTracedInTwoPiecesArePutBackTogether(unittest.TestCase):
    """One minimum that the tracer stopped and restarted at a second-order merge.

    Left apart, the pair has no temperature at which both exist, so no critical temperature
    can be found between the lower piece and anything above it, and the coupling is lost to
    `NoViablePhasePairError` rather than merely being imprecise.
    """

    def test_two_pieces_that_meet_are_joined(self):
        # The gap is judged against the range the pieces span, so the fixture keeps the
        # ratio of the measured case: 4.5 MeV across 575.9 GeV on the 2HDM.
        phases = {0: _phase(0, [1.0, 2.0, 3.0], [[0.30, 0.0], [0.20, 0.0], [0.10, 0.0]]),
                  1: _phase(1, [3.001, 4.0, 5.0], [[0.11, 0.0], [0.05, 0.0], [0.0, 0.0]])}
        ph.mergePhasesMeetingAtAPoint(phases, diftol=1.0)
        self.assertEqual(len(phases), 1, msg=f"expected one phase, got {list(phases)}")
        joined = next(iter(phases.values()))
        self.assertAlmostEqual(float(joined.Tmin), 1.0)
        self.assertAlmostEqual(float(joined.Tmax), 5.0)
        self.assertEqual(len(np.asarray(joined.T)), 6)

    def test_pieces_that_disagree_in_field_space_are_left_alone(self):
        """A first-order transition also puts two phases end to end."""
        phases = {0: _phase(0, [1.0, 2.0, 3.0], [[0.30, 0.0], [0.20, 0.0], [0.10, 0.0]]),
                  1: _phase(1, [3.001, 4.0, 5.0], [[9.00, 0.0], [8.00, 0.0], [7.0, 0.0]])}
        ph.mergePhasesMeetingAtAPoint(phases, diftol=1.0)
        self.assertEqual(len(phases), 2, msg=f"the pair was joined: {list(phases)}")

    def test_pieces_far_apart_in_temperature_are_left_alone(self):
        phases = {0: _phase(0, [1.0, 1.001, 1.002], [[0.30, 0.0], [0.20, 0.0], [0.10, 0.0]]),
                  1: _phase(1, [4.0, 4.5, 5.0], [[0.11, 0.0], [0.05, 0.0], [0.0, 0.0]])}
        ph.mergePhasesMeetingAtAPoint(phases, diftol=1.0)
        self.assertEqual(len(phases), 2, msg=f"the pair was joined: {list(phases)}")

    def test_overlapping_phases_are_left_to_the_redundancy_pass(self):
        phases = {0: _phase(0, [1.0, 2.0, 3.0], [[0.30, 0.0], [0.20, 0.0], [0.10, 0.0]]),
                  1: _phase(1, [2.5, 4.0, 5.0], [[0.15, 0.0], [0.05, 0.0], [0.0, 0.0]])}
        ph.mergePhasesMeetingAtAPoint(phases, diftol=1.0)
        self.assertEqual(len(phases), 2, msg=f"an overlapping pair was joined: {list(phases)}")


if __name__ == "__main__":
    unittest.main()
