r"""The last point of a phase trace, and whether it is a minimum of the potential.

`traceMinimum` accepted its coldest point on the slope alone, so a trace could end at a point
that is not a minimum and the bounce would then be computed from a false vacuum that is not a
vacuum. Checking the curvature there was tried before and withdrawn, because the obvious test,
refusing a point whose smallest eigenvalue is negative, refuses the symmetric phase of a
conformal model: `d2V` is a finite difference of step `x_eps`, the potential has no quadratic
term at the origin of such a model, and the difference returns the step squared times the
quartic, which is negative.

What separates the two is how the value behaves when the step is halved. Measured on
`models/TL_conformal_dark_u1.py` at g = 0.7, v = 0.1 GeV, at the origin of the symmetric
phase, in the model's internal units, where `conversionFactor` is 1e-4:

    x_eps  4.0e-03  2.0e-03  1.0e-03  5.0e-04  2.5e-04
    value -6.77e-08 -1.69e-08 -4.23e-09 -1.06e-09 -2.65e-10

a factor 4.0000 at every halving, extrapolating to the zero curvature a conformal potential
has there. A curvature the potential really has does not move with the step: the broken
phase's endpoint of the same model holds +9160 across that whole range.

Part of TransitionListener v2
"""
import contextlib
import io
import unittest
from pathlib import Path

import numpy as np

from transitionlistener import helper_functions as hf
from transitionlistener import phases as PH
from transitionlistener.helper_functions import load_potential
from transitionlistener.errors import Timeout

REPO = Path(__file__).resolve().parents[1]
#: The step `d2V` uses by default in every model shipped with the code.
STEP = 1.0e-3


def hessian_scaling_as(power, size=1.0, sign=-1.0, ndim=1):
    """A pair of Hessian functions whose smallest eigenvalue goes as `step**power`.

    Returns `(at_the_step, at_half_the_step)`, which is what
    `smallestCurvatureIsTheSteps` is given. A power of 2 is what a finite difference
    returns where the potential has no quadratic term, a power of 1 is what the
    Arnold-Espinosa daisy's term in the modulus of the field cubed contributes, and a power
    of 0 is a curvature the potential has.
    """
    def at(step):
        def f(x, t, _step=step):
            value = sign * size * _step ** power
            return np.diag([value] + [abs(size)] * (ndim - 1))
        return f
    return at(STEP), at(0.5 * STEP)


class TheRatioSeparatesTheStepFromThePotential(unittest.TestCase):

    def test_a_value_that_goes_as_the_step_squared_is_the_steps(self):
        """The conformal case: a factor 4 per halving, so the direction is flat."""
        full, half = hessian_scaling_as(2)
        self.assertIs(PH.smallestCurvatureIsTheSteps(full, half, np.zeros(1), 1.0e-6), True)

    def test_a_value_that_does_not_move_with_the_step_is_the_potentials(self):
        """A real curvature, which is what makes the endpoint a saddle when it is negative."""
        full, half = hessian_scaling_as(0)
        self.assertIs(PH.smallestCurvatureIsTheSteps(full, half, np.zeros(1), 1.0e-6), False)

    def test_a_value_linear_in_the_step_counts_as_the_steps(self):
        """A ratio near 2, which no shipped model produces but the cut has to handle.

        A potential only once differentiable at the point would contribute a term linear in
        the step, which the Arnold-Espinosa daisy's term in the modulus of the field cubed
        would do at the origin. Switching the daisy off changes the ratio of none of the four
        shipped models at any temperature sampled, from 1e-6 to 1e2, so this is a possibility
        rather than a measurement. Either way it is not a curvature the potential has at the
        point, so the endpoint stands.
        """
        full, half = hessian_scaling_as(1)
        self.assertIs(PH.smallestCurvatureIsTheSteps(full, half, np.zeros(1), 1.0e-6), True)

    def test_the_cut_lies_between_the_two_measured_ratios(self):
        """The value of the cut, against the two behaviours measured on shipped models.

        Every other test here builds its ratio from the cut's own side of the question, so
        none of them says where the cut should sit. The measured ratios are 1.0000, where the
        value is the curvature, and 4.0000, where the potential has no quadratic term in that
        direction. The cut has to lie between them, and below 2 as well, so that a point only
        once differentiable would be accepted rather than refused.
        """
        self.assertGreater(PH._ENDPOINT_CURVATURE_RATIO_CUT, 1.0004,
                           msg="a curvature the potential has would be taken for the step's")
        self.assertLess(PH._ENDPOINT_CURVATURE_RATIO_CUT, 2.0,
                        msg="a value linear in the step would be taken for a curvature the "
                            "potential has, which refuses a phase that is perfectly good")
        self.assertLess(PH._ENDPOINT_CURVATURE_RATIO_CUT, 4.0)

    def test_a_ratio_either_side_of_the_cut(self):
        for name, ratio, expected in (("just below the cut", 0.9, False),
                                      ("exactly at the cut", 1.0, True),
                                      ("just above the cut", 1.1, True)):
            with self.subTest(name):
                target = PH._ENDPOINT_CURVATURE_RATIO_CUT * ratio

                def full(x, t):
                    return np.diag([-target])

                def half(x, t):
                    return np.diag([-1.0])

                self.assertIs(
                    PH.smallestCurvatureIsTheSteps(full, half, np.zeros(1), 1.0),
                    expected,
                    msg=f"{name}: a ratio of {target:.4f} against a cut of "
                        f"{PH._ENDPOINT_CURVATURE_RATIO_CUT}")

    def test_a_positive_fourth_derivative_never_reaches_the_ratio(self):
        r"""An origin whose first three derivatives vanish and whose fourth is positive.

        The fourth-order stencil `[-1, 16, -30, 16, -1] / (12 h^2)` annihilates a quartic
        term by term, so such an origin returns exactly zero and the question is declined. At
        the second order it returns `+2 c_4 h^2`, which is positive, and the question is never
        asked, because `endpointIsASaddlePoint` asks it only where an eigenvalue is negative.
        Either way the point is kept, which is the right answer: a quartic minimum is a
        minimum, and a positive fourth derivative cannot produce the negative value this
        check exists to judge.
        """
        c4 = 0.25

        def V(X, T=0.0, *args, **kwargs):
            return c4 * np.sum(np.asarray(X, dtype=float) ** 4, axis=-1)

        def hessian(step, order):
            f = hf.hessianFunction(V, step, 1, order)
            return lambda x, t: f(np.atleast_1d(np.asarray(x, dtype=float)),
                                  np.asarray(t)[..., np.newaxis], False)

        for order, expected_sign in ((4, 0.0), (2, +1.0)):
            with self.subTest(f"deriv_order {order}"):
                full, half = hessian(STEP, order), hessian(0.5 * STEP, order)
                value = float(np.min(np.linalg.eigvalsh(
                    np.atleast_2d(np.squeeze(full(np.zeros(1), 0.0))))))
                self.assertEqual(
                    np.sign(value), expected_sign,
                    msg=f"order {order}: the stencil returns {value!r} at a quartic origin, "
                        f"which is neither the exact zero of the fourth order nor the "
                        f"positive value of the second")
                self.assertFalse(
                    PH.endpointIsASaddlePoint(full, half, np.zeros(1), 0.0),
                    msg=f"order {order}: a quartic minimum was called a saddle point")

    def test_a_saddle_shallower_than_the_steps_own_term_is_kept(self):
        """The limit of the check, measured: the cut is reached at 1.25 times that term.

        The value a step returns is the true curvature plus the step's own term,
        `lambda + C h^2`, so the ratio is `(lambda + C h^2) / (lambda + C h^2 / 4)`, which
        equals the cut when `lambda = 1.25 C h^2`. Below that a genuine saddle is kept as a
        minimum. The check errs that way on purpose, because refusing the coldest point of a
        phase costs the transition that would be computed from it.
        """
        C = -1.0
        for multiple, expected in ((0.25, True), (1.0, True), (1.25, True),
                                   (1.5, False), (10.0, False)):
            with self.subTest(f"{multiple:g} times the step's own term"):
                lam = multiple * C * STEP ** 2

                def at(step, _lam=lam):
                    return lambda x, t: np.diag([_lam + C * step ** 2])

                self.assertIs(
                    PH.smallestCurvatureIsTheSteps(at(STEP), at(0.5 * STEP),
                                                   np.zeros(1), 0.0),
                    expected,
                    msg=f"a true curvature {multiple:g} times the step's own term was "
                        f"{'kept as the steps' if expected else 'called the potentials'}, "
                        f"which is the wrong side of the cut")

    def test_a_sign_that_flips_between_the_two_steps_is_not_the_potentials(self):
        """No single power of the step changes the sign, so the value is not a curvature."""
        def full(x, t):
            return np.diag([-1.0e-9])

        def half(x, t):
            return np.diag([+1.0e-9])

        self.assertIs(PH.smallestCurvatureIsTheSteps(full, half, np.zeros(1), 1.0), True)

    def test_a_question_that_cannot_be_answered_comes_back_as_none(self):
        """None, so that the caller leaves the endpoint where it is rather than guessing."""
        cases = {
            "a non-finite Hessian at the step":
                (lambda x, t: np.diag([np.nan]), lambda x, t: np.diag([-1.0])),
            "a non-finite Hessian at half the step":
                (lambda x, t: np.diag([-1.0]), lambda x, t: np.diag([np.inf])),
            "a zero eigenvalue at the step":
                (lambda x, t: np.diag([0.0]), lambda x, t: np.diag([-1.0])),
            "a zero eigenvalue at half the step":
                (lambda x, t: np.diag([-1.0]), lambda x, t: np.diag([0.0])),
            "a Hessian that raises":
                (lambda x, t: (_ for _ in ()).throw(ValueError("no")),
                 lambda x, t: np.diag([-1.0])),
        }
        for name, (full, half) in cases.items():
            with self.subTest(name):
                self.assertIsNone(
                    PH.smallestCurvatureIsTheSteps(full, half, np.zeros(1), 1.0),
                    msg=f"{name}: the question was answered rather than declined")

    def test_the_smallest_eigenvalue_is_the_one_judged(self):
        """A flat direction beside a stiff one is still a flat direction.

        The Hessian of a phase with several fields has one eigenvalue per direction and the
        question is about the smallest of them. The matrices below are built so that the two
        choices disagree: the smallest eigenvalue falls by four when the step is halved,
        which makes it the step's, while the largest does not move, so judging the largest
        would answer that the value is the potential's.
        """
        def at(step):
            def f(x, t, _step=step):
                return np.diag([-1.0 * _step ** 2, 1.0e6, 2.0e6])
            return f

        full, half = at(STEP), at(0.5 * STEP)
        smallest = [float(np.min(np.linalg.eigvalsh(g(np.zeros(3), 1.0))))
                    for g in (full, half)]
        largest = [float(np.max(np.linalg.eigvalsh(g(np.zeros(3), 1.0))))
                   for g in (full, half)]
        self.assertAlmostEqual(smallest[0] / smallest[1], 4.0, places=6,
                               msg="the fixture no longer scales the smallest eigenvalue")
        self.assertAlmostEqual(largest[0] / largest[1], 1.0, places=6,
                               msg="the fixture no longer holds the largest eigenvalue")
        self.assertIs(PH.smallestCurvatureIsTheSteps(full, half, np.zeros(3), 1.0), True)


class ATraceWhoseMinimumTurnsOverStopsOnItsOwn(unittest.TestCase):
    r"""Why no potential here reaches the refusal.

        V(x, T) = (T - T0) x^2 / 2 + x^4 / 4,   T0 = 1.0

    The origin is a minimum above `T0` and a saddle point below it, with a curvature of
    exactly `T - T0`. Traced downwards from T = 1.3 the trace stops at T = 1.002, just above
    `T0`, because the minimisation itself fails once the origin is no longer a minimum. So a
    potential cannot be used to drive the endpoint refusal: by the time the endpoint would be
    a saddle, the trace has already ended. This was measured, not expected.
    """

    T0 = 1.0
    STEP = 1.0e-3

    @classmethod
    def V(cls, x, T):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return float(0.5 * (T - cls.T0) * np.sum(x ** 2) + 0.25 * np.sum(x ** 4))

    @classmethod
    def dV(cls, x, T):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return (T - cls.T0) * x + x ** 3

    @staticmethod
    def d2V_dxdT(x, T):
        return np.atleast_1d(np.asarray(x, dtype=float)) * 0.0 + 1.0

    @classmethod
    def d2V(cls, x, T):
        return np.diag([(T - cls.T0) + 3.0 * float(np.atleast_1d(x)[0]) ** 2])

    def test_the_curvature_at_the_origin_is_the_temperature_offset(self):
        """The fixture, before anything is traced: the origin really does turn over."""
        for T, sign in ((1.2, +1.0), (1.0, 0.0), (0.8, -1.0)):
            with self.subTest(f"T = {T}"):
                value = float(self.d2V(np.zeros(1), T)[0, 0])
                self.assertAlmostEqual(value, T - self.T0, places=12)
                if sign:
                    self.assertEqual(np.sign(value), sign)

    def test_the_trace_stops_above_the_temperature_where_the_origin_turns_over(self):
        result = PH.traceMinimum(self.V, self.dV, self.d2V_dxdT, self.d2V,
                                 np.zeros(1), 1.3, 0.2, -0.02, 1.0e-2)
        self.assertGreaterEqual(
            float(np.min(result.T)), self.T0 - 1.0e-2,
            msg=f"the trace ran to T = {float(np.min(result.T)):.4f}, below the T0 = "
                f"{self.T0} where the origin stops being a minimum, so the endpoint refusal "
                f"could be driven from a potential after all and this class is out of date")


class ThePredicateAsksBothQuestions(unittest.TestCase):
    """`endpointIsASaddlePoint` is negative curvature AND curvature the potential has."""

    def test_a_point_whose_curvature_is_positive_is_not_a_saddle(self):
        full, half = hessian_scaling_as(0, sign=+1.0)
        self.assertFalse(PH.endpointIsASaddlePoint(full, half, np.zeros(1), 1.0))

    def test_a_negative_curvature_the_potential_has_is_a_saddle(self):
        full, half = hessian_scaling_as(0, sign=-1.0)
        self.assertTrue(PH.endpointIsASaddlePoint(full, half, np.zeros(1), 1.0))

    def test_a_negative_curvature_that_is_the_steps_is_not_a_saddle(self):
        """The conformal case, which the two withdrawn attempts refused."""
        full, half = hessian_scaling_as(2, sign=-1.0)
        self.assertFalse(PH.endpointIsASaddlePoint(full, half, np.zeros(1), 1.0))

    def test_one_negative_direction_among_several_is_enough(self):
        """A saddle point has at least one direction that falls away, not every direction.

        Every other test here works on a potential of one field, where asking whether any
        eigenvalue is negative and whether all of them are is the same question. A phase of
        two fields, stiff in one direction and falling away in the other, is a saddle point,
        and requiring every direction to fall away would accept it.
        """
        def at(step):
            def f(x, t, _step=step):
                return np.diag([-1.0, +1.0e6])
            return f

        full, half = at(STEP), at(0.5 * STEP)
        eigs = np.linalg.eigvalsh(full(np.zeros(2), 1.0))
        self.assertTrue((eigs < 0).any() and not (eigs < 0).all(),
                        msg="the fixture no longer mixes the signs, so it does not separate "
                            "the two questions")
        self.assertTrue(PH.endpointIsASaddlePoint(full, half, np.zeros(2), 1.0),
                        msg="a point that falls away in one direction of two was not called "
                            "a saddle point")

    def test_a_question_that_cannot_be_answered_is_not_a_saddle(self):
        """Dropping the coldest point of a phase on a guess costs the transition."""
        for name, half in (("not a number", lambda x, t: np.diag([np.nan])),
                           ("a zero eigenvalue", lambda x, t: np.diag([0.0]))):
            with self.subTest(name):
                self.assertFalse(
                    PH.endpointIsASaddlePoint(lambda x, t: np.diag([-1.0]), half,
                                              np.zeros(1), 1.0),
                    msg=f"{name}: a point was called a saddle on a question with no answer")

    def test_a_hessian_that_raises_is_not_a_saddle(self):
        self.assertFalse(PH.endpointIsASaddlePoint(
            lambda x, t: (_ for _ in ()).throw(ValueError("no")),
            lambda x, t: np.diag([-1.0]), np.zeros(1), 1.0))

    def test_the_second_hessian_is_not_called_where_the_curvature_is_positive(self):
        """It costs an evaluation of the potential at every offset of the stencil."""
        calls = []

        def half(x, t):
            calls.append((x, t))
            return np.diag([1.0])

        PH.endpointIsASaddlePoint(lambda x, t: np.diag([1.0]), half, np.zeros(1), 1.0)
        self.assertEqual(calls, [], msg="the second Hessian was evaluated although the "
                                        "curvature is positive and no question arises")


class TheTraceDropsAnEndpointThePredicateRefuses(unittest.TestCase):
    r"""`traceMinimum` driven with the decision supplied, on

        V(x, T) = T x^2 / 2 + x^4 / 4.

    The origin is a minimum at every positive temperature, so the trace reaches the
    temperature it is asked to stop at and its endpoint is a genuine minimum. The predicate
    is what each case below substitutes, which is the whole point of `traceMinimum` taking
    one: substituting the Hessian instead would change what the tracer solves with and break
    the trace before the decision is reached.

    Reaching this branch from a potential on `main` was not possible. Across 70 phase
    endpoints of the real 2HDM and of the two dark U(1) models shipped with the code, not one
    is a saddle point, and on a potential whose origin does turn into a saddle the
    minimisation fails and the trace ends on its own.

    It is reachable on the same model with the tracer as it stood before pull request #44,
    which tested its saddle flag with `negeig is True`, never true for a numpy boolean, so the
    test never ran. Built from the merge of pull request #43, the 2HDM at `lambda3 = 5.82`
    traces its high-temperature phase to T = 0.0000 GeV with the false vacuum at the origin,
    where the smallest eigenvalue is -69.75 GeV^2 at a ratio of 1.000, so the value is the
    potential's curvature and the endpoint is a saddle point. That is the one of 16 endpoints
    on that tree, against none of 24 on `main`. Both measurements are in
    `v96_trace_endpoint/RECORD.md`.
    """

    TSTOP = 0.4

    @staticmethod
    def V(x, T):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return float(0.5 * T * np.sum(x ** 2) + 0.25 * np.sum(x ** 4))

    @staticmethod
    def dV(x, T):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return T * x + x ** 3

    @staticmethod
    def d2V_dxdT(x, T):
        return np.atleast_1d(np.asarray(x, dtype=float)) * 0.0 + 1.0

    @staticmethod
    def d2V(x, T):
        return np.diag([T + 3.0 * float(np.atleast_1d(x)[0]) ** 2])

    def trace_with(self, predicate):
        return PH.traceMinimum(
            self.V, self.dV, self.d2V_dxdT, self.d2V,
            np.zeros(1), 1.0, self.TSTOP, -0.05, 1.0e-2,
            endpointIsASaddle=predicate)

    def test_the_trace_ends_exactly_at_the_stop_with_no_predicate(self):
        """The fixture, and the behaviour before this change.

        The stepping loop overshoots the temperature it was asked to stop at and the last
        node is then replaced by one taken exactly to it, so the trace ends on `TSTOP` to the
        last digit. That replacement is what the refusal withholds.
        """
        self.assertEqual(float(np.min(self.trace_with(None).T)), self.TSTOP,
                         msg="the trace does not end exactly on the stop even without a "
                             "predicate, so nothing below is testing the refusal")

    def test_a_predicate_that_refuses_withholds_the_final_step(self):
        """The endpoint at `TSTOP` is not taken, so the trace keeps the node before it.

        That node is the one the stepping loop overshot to, which is where the existing
        behaviour for an endpoint with no usable slope also leaves the trace, so the refusal
        does not introduce a new kind of ending.
        """
        result = self.trace_with(lambda x, t: True)
        self.assertNotEqual(
            float(np.min(result.T)), self.TSTOP,
            msg="the endpoint was kept although the predicate called it a saddle point")

    def test_a_predicate_that_accepts_changes_nothing(self):
        kept = float(np.min(self.trace_with(lambda x, t: False).T))
        plain = float(np.min(self.trace_with(None).T))
        self.assertEqual(kept, plain,
                         msg="a predicate that refuses nothing still moved the end of the "
                             "trace")
        self.assertEqual(kept, self.TSTOP)

    def test_the_predicate_is_asked_about_the_endpoint_and_not_every_node(self):
        """It costs two Hessians, so it must run once and at the end.

        A phase traced over tens of temperatures would otherwise pay for a second Hessian at
        each of them.
        """
        asked = []
        result = self.trace_with(lambda x, t: asked.append(float(t)) or False)
        self.assertGreaterEqual(len(result.T), 5,
                                msg="the trace is too short for this to mean anything")
        self.assertLessEqual(
            len(asked), 2,
            msg=f"the predicate was asked {len(asked)} times over a trace of "
                f"{len(result.T)} nodes")
        for t_asked in asked:
            self.assertLess(abs(t_asked - self.TSTOP), 0.05,
                            msg=f"the predicate was asked at T = {t_asked}, which is not the "
                                f"end of the trace at T = {self.TSTOP}")


class TheConformalModelIsStillAccepted(unittest.TestCase):
    """The case that withdrew the earlier attempt, on the real model.

    `models/TL_conformal_dark_u1.py` at g = 0.7, v_GeV = 0.1, y = 0.01 is the shipped
    example, and the symmetric phase of a conformal model sits at the origin all the way to
    zero temperature. A test that refuses a negative eigenvalue there stops the trace short
    and the run dies with `WrongT0MinimumError`.
    """

    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.pot = load_potential(
                str(REPO / "models/TL_conformal_dark_u1.py"), "specific_potential")(
                    dict(g=0.7, v_GeV=0.1, y=0.01), verbose=False)

    def test_the_step_scaling_at_the_origin_is_still_a_factor_four(self):
        """The measurement the whole change rests on, taken from the model itself."""
        origin, T = np.zeros(self.pot.Ndim), 1.0e-6
        values = []
        for step in (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4):
            hess = np.atleast_2d(np.squeeze(
                np.asarray(self.pot.d2V(origin, T, x_eps=step), dtype=float)))
            values.append(float(np.min(np.linalg.eigvalsh(hess))))
        for before, after in zip(values, values[1:]):
            self.assertAlmostEqual(
                before / after, 4.0, places=3,
                msg=f"halving the step no longer divides the value by four: {values}")
        self.assertTrue(all(v < 0.0 for v in values),
                        msg="the values are no longer negative, so the case this change "
                            "exists for is gone and the fixture needs rewriting")

    def test_the_factor_of_four_belongs_to_the_fourth_order_stencil(self):
        """The ratio is 4.0000 at `deriv_order` 4 and 3.79 to 3.83 at 2, and the cut covers both.

        Near a conformal origin the potential goes as the field to the fourth times the
        logarithm of the field. The fourth-order stencil annihilates the logarithm and leaves
        a term in the step squared, which halves to a quarter exactly; the second-order
        stencil does not, and its ratio drifts with the step. `tracing_derivative_order`
        defaults to 4, so 4.0000 is what the shipped models give, but the number belongs to
        the stencil as much as to the potential and the check has to work at either order.
        """
        origin = np.zeros(self.pot.Ndim)
        was = self.pot.deriv_order
        try:
            for order, lo, hi in ((4, 3.999, 4.001), (2, 3.75, 3.85)):
                with self.subTest(f"deriv_order {order}"):
                    self.pot.deriv_order = order
                    self.pot._d2V_by_step = {}
                    if hasattr(self.pot, "_d2V"):
                        del self.pot._d2V
                    values = []
                    for step in (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4):
                        self.pot._d2V_by_step = {}
                        if hasattr(self.pot, "_d2V"):
                            del self.pot._d2V
                        hess = np.atleast_2d(np.squeeze(np.asarray(
                            self.pot.d2V(origin, 1.0e-6, x_eps=step), dtype=float)))
                        values.append(float(np.min(np.linalg.eigvalsh(hess))))
                    ratios = [values[i] / values[i + 1] for i in range(len(values) - 1)]
                    for r in ratios:
                        self.assertGreater(r, lo, msg=f"order {order}: ratios {ratios}")
                        self.assertLess(r, hi, msg=f"order {order}: ratios {ratios}")
                        self.assertGreater(
                            r, PH._ENDPOINT_CURVATURE_RATIO_CUT,
                            msg=f"order {order}: a ratio of {r:.4f} falls below the cut, so "
                                f"the check would call this a curvature the potential has")
        finally:
            self.pot.deriv_order = was
            self.pot._d2V_by_step = {}
            if hasattr(self.pot, "_d2V"):
                del self.pot._d2V

    def test_the_origin_of_the_conformal_model_is_not_called_a_saddle(self):
        origin, T = np.zeros(self.pot.Ndim), 1.0e-6
        self.assertIs(
            PH.smallestCurvatureIsTheSteps(
                self.pot.d2V, lambda X, t: self.pot.d2V(X, t, x_eps=0.5 * self.pot.x_eps),
                origin, T),
            True,
            msg="the origin of a conformal model was judged to be a saddle point, which is "
                "what stopped the trace short and killed the shipped example before")

    def test_the_broken_phase_endpoint_is_the_potentials_own_curvature(self):
        """The control: a real minimum has to come back as the potential's.

        Without it the test above passes for a function that always answers True.
        """
        vev = np.full(self.pot.Ndim, float(self.pot.v_stable))
        self.assertIs(
            PH.smallestCurvatureIsTheSteps(
                self.pot.d2V, lambda X, t: self.pot.d2V(X, t, x_eps=0.5 * self.pot.x_eps),
                vev, 1.0e-6),
            False,
            msg="the broken phase's minimum was judged to belong to the step")


class TheOptionalStepIsHonoured(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.pot = load_potential(
                str(REPO / "models/TL_conformal_dark_u1.py"), "specific_potential")(
                    dict(g=0.7, v_GeV=0.1, y=0.01), verbose=False)

    def test_the_default_step_is_unchanged(self):
        origin, T = np.zeros(self.pot.Ndim), 1.0e-6
        np.testing.assert_allclose(
            np.asarray(self.pot.d2V(origin, T), dtype=float),
            np.asarray(self.pot.d2V(origin, T, x_eps=self.pot.x_eps), dtype=float),
            atol=0.0, err_msg="passing the model's own step changed the answer")

    def test_each_step_is_built_once(self):
        origin, T = np.zeros(self.pot.Ndim), 1.0e-6
        self.pot.d2V(origin, T, x_eps=5.0e-4)
        self.pot.d2V(origin, T, x_eps=2.5e-4)
        first = self.pot._d2V_by_step[5.0e-4]
        self.pot.d2V(origin, T, x_eps=5.0e-4)
        self.assertIs(self.pot._d2V_by_step[5.0e-4], first,
                      msg="the Hessian function for a step was rebuilt on a second call")
        self.assertEqual(sorted(self.pot._d2V_by_step), [2.5e-4, 5.0e-4])


if __name__ == "__main__":
    unittest.main()


class TheDeadlineIsNotASolverFailure(unittest.TestCase):
    """`errors.Timeout` comes from the run's alarm, so neither check may swallow it."""

    def _hessian(self):
        return np.array([[-1.0, 0.0], [0.0, 4.0]])

    def test_a_timeout_from_the_full_step_reaches_the_caller(self):
        def d2f(x, t):
            raise Timeout("the run's deadline")
        with self.assertRaises(Timeout):
            PH.endpointIsASaddlePoint(d2f, d2f, np.zeros(2), 1.0)

    def test_a_timeout_from_the_half_step_reaches_the_caller(self):
        """The one the second evaluation can raise after the first has succeeded."""
        def full(x, t):
            return self._hessian()

        def half(x, t):
            raise Timeout("the run's deadline")
        with self.assertRaises(Timeout):
            PH.smallestCurvatureIsTheSteps(full, half, np.zeros(2), 1.0)
        with self.assertRaises(Timeout):
            PH.endpointIsASaddlePoint(full, half, np.zeros(2), 1.0)

    def test_an_ordinary_solver_failure_is_still_swallowed(self):
        """Only the deadline is special; a failure to evaluate leaves the point accepted."""
        def d2f(x, t):
            raise ValueError("the minimisation did not converge")
        self.assertFalse(PH.endpointIsASaddlePoint(d2f, d2f, np.zeros(2), 1.0))
        self.assertIsNone(PH.smallestCurvatureIsTheSteps(d2f, d2f, np.zeros(2), 1.0))


class TheEndpointPaysForTwoHessiansAndNotThree(unittest.TestCase):
    """Each `d2V` call runs the whole finite-difference stencil, so the count is the cost."""

    def test_the_full_step_hessian_is_computed_once(self):
        calls = {"full": 0, "half": 0}

        def full(x, t):
            calls["full"] += 1
            return np.array([[-1.0e-9, 0.0], [0.0, 4.0]])

        def half(x, t):
            calls["half"] += 1
            return np.array([[-2.5e-10, 0.0], [0.0, 4.0]])

        PH.endpointIsASaddlePoint(full, half, np.zeros(2), 1.0)
        self.assertEqual(calls, {"full": 1, "half": 1},
                         msg=f"the endpoint cost {sum(calls.values())} Hessian evaluations")

    def test_the_handed_over_hessian_is_the_one_used(self):
        """Passing `full` must replace the evaluation, not sit beside it."""
        def full(x, t):
            raise AssertionError("d2f_dx2 was called although a Hessian was handed over")

        def half(x, t):
            return np.array([[-2.5e-10, 0.0], [0.0, 4.0]])
        verdict = PH.smallestCurvatureIsTheSteps(
            full, half, np.zeros(2), 1.0, full=np.array([[-1.0e-9, 0.0], [0.0, 4.0]]))
        self.assertTrue(verdict, msg="a value that falls by four with the step is the step's")
