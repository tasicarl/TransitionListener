r"""A traced phase has to lie on one image of the potential's symmetry.

Below the temperature at which the origin tilts over, the displaced minima are exactly
degenerate images of one another, and which one a trace walks into is settled by the sign
of a numerical step. Two seeds that reach the same physical phase on opposite images give
two phases that `removeRedundantPhases` cannot tell are one, because it compares field
values; the pair survives and the transition can then be built from a high phase on one
image to a low phase on the other. That path is not a path of the potential: on the 2HDM
at `lambda3 = 5.868` it gave a field separation of 264.54 GeV against 231.55 GeV at the
neighbouring couplings, `S_3/T = 336.6` against 152.6 and 153.3 at `T = 30.0` GeV, and no
percolation at all.

The potential here couples the two fields, so that the minimum has both components
resolved:

    V(x, y, t) = (c t^2 - m2) x^2 / 2 + l x^4 / 4 + k y^2 / 2 - g x y

The `- g x y` term is the one the 2HDM has in `m12^2 phi_1 phi_2`: it forbids either
reflection on its own and leaves the simultaneous flip `diag(-1, -1)`, which is the group
the model is found to have. Below the temperature at which the origin tilts over the
minima are known in closed form,

    y = g x / k,    x = +- sqrt((m2 + g^2 / k - c t^2) / l),

and the origin is a minimum while `c t^2 > m2 + g^2 / k`. A separable potential will not
do here: its minimum sits at `y = 0`, where the sign of the second component is numerical
noise of order 1e-5, and the measure below reads that sign.

Part of TransitionListener v2
"""
import unittest

import numpy as np

from transitionlistener import phases as ph

C, M2, LAM, K, G = 1.0, 4.0, 1.0, 3.0, 1.0
#: Above it the origin is a minimum, below it a saddle: 2.0817.
T0 = np.sqrt((M2 + G ** 2 / K) / C)

IDENTITY = np.diag([1.0, 1.0])
FLIP_X = np.diag([-1.0, 1.0])
FLIP_Y = np.diag([1.0, -1.0])
FLIP_BOTH = np.diag([-1.0, -1.0])


def V(x, t):
    x = np.asarray(x, dtype=float)
    return (0.5 * (C * t ** 2 - M2) * x[..., 0] ** 2
            + 0.25 * LAM * x[..., 0] ** 4
            + 0.5 * K * x[..., 1] ** 2
            - G * x[..., 0] * x[..., 1])


def dV(x, t):
    x = np.asarray(x, dtype=float)
    return np.array([(C * t ** 2 - M2) * x[..., 0] + LAM * x[..., 0] ** 3
                     - G * x[..., 1],
                     K * x[..., 1] - G * x[..., 0]])


def d2V(x, t):
    x = np.asarray(x, dtype=float)
    return np.array([[(C * t ** 2 - M2) + 3.0 * LAM * x[..., 0] ** 2, -G],
                     [-G, K]])


def d2V_dxdt(x, t):
    x = np.asarray(x, dtype=float)
    return np.array([2.0 * C * t * x[..., 0], 0.0])


def displaced_minimum(t):
    """The closed-form minimum below `T0`, on the image with both components positive."""
    x = np.sqrt((M2 + G ** 2 / K - C * t ** 2) / LAM)
    return np.array([x, G * x / K])


def displaced_slope(t):
    """`dX/dt` of that minimum, from `2 x dx/dt = -2 c t / l`."""
    x = float(displaced_minimum(t)[0])
    dx = -C * t / (LAM * x)
    return np.array([dx, G * dx / K])


def bsmpt_measure(x):
    r"""The measure `generic_potential.applySymmetries` maximises, `sum_i 2^i theta(x_i)`.

    `np.heaviside(0, 1)` counts a vanishing component as positive, as it does there.
    """
    x = np.asarray(x, dtype=float)
    return sum(2 ** i * np.heaviside(x[i], 1) for i in range(x.size))


def canonical_under(group):
    """An `applySymmetries` over `group`, built the way the potentials' own one is."""
    def applySymmetries(x):
        x = np.asarray(x, dtype=float)
        return max((x @ g for g in group), key=bsmpt_measure)
    return applySymmetries


class TheElementIsReadOffTheBestResolvedNode(unittest.TestCase):
    """`canonicalImageElement` picks one transformation for a whole trace."""

    GROUP = [IDENTITY, FLIP_X, FLIP_Y, FLIP_BOTH]

    def element(self, X, group=None):
        return ph.canonicalImageElement(
            np.asarray(X, dtype=float),
            canonical_under(self.GROUP if group is None else group),
            self.GROUP if group is None else group)

    def test_a_trace_on_the_canonical_image_is_left_where_it_is(self):
        X = np.array([[0.1, 0.2], [1.0, 0.5], [2.0, 1.0]])
        np.testing.assert_allclose(self.element(X), IDENTITY)

    def test_a_trace_on_the_opposite_image_is_mapped_onto_the_canonical_one(self):
        X = np.array([[-0.1, -0.2], [-1.0, -0.5], [-2.0, -1.0]])
        got = self.element(X)
        np.testing.assert_allclose(got, FLIP_BOTH)
        np.testing.assert_allclose(X @ got, -X)

    def test_the_element_comes_from_the_node_furthest_from_the_origin(self):
        """A node near the origin has no resolved sign, and must not decide the image.

        The two small nodes here sit in the canonical quadrant and the large one does not.
        Taken node by node the large node would be flipped and the small ones left, which
        puts a step of twice the field value into a trace that is smooth.
        """
        X = np.array([[1.0e-9, 1.0e-9], [2.0e-9, 1.0e-9], [-2.0, -1.0]])
        np.testing.assert_allclose(self.element(X), FLIP_BOTH)

    def test_the_whole_trace_moves_rigidly(self):
        """Every node is transformed by the same element, so no step is introduced."""
        X = np.array([[-1.0e-9, 1.0e-9], [-1.0, -0.5], [-2.0, -1.0], [-1.5, -0.75]])
        got = self.element(X)
        moved = X @ got
        for row, image in zip(X, moved):
            np.testing.assert_allclose(image, row @ got, atol=0.0)
        self.assertAlmostEqual(float(np.linalg.norm(moved[-1])),
                              float(np.linalg.norm(X[-1])), places=12)

    def test_the_group_of_the_two_higgs_doublet_model_keeps_the_second_field_positive(self):
        """With only the simultaneous flip, the measure is decided by the last component.

        That is the group the 2HDM has, `m12^2 phi_1 phi_2` forbidding either reflection on
        its own, and the canonical image is therefore the one with `phi_2 > 0`. The phase
        the transition at `lambda3 = 5.868` was built from sat at `(-27.42, -15.33)` GeV.
        """
        group = [IDENTITY, FLIP_BOTH]
        X = np.array([[-27.42, -15.33], [-20.0, -11.0]])
        got = self.element(X, group=group)
        np.testing.assert_allclose(got, FLIP_BOTH)
        self.assertGreater(float((X @ got)[0][1]), 0.0)


class ThereIsNothingToDoWithoutASymmetry(unittest.TestCase):
    """A potential with no symmetry has one image, and the trace is returned untouched."""

    def test_no_symmetry_map_means_no_element(self):
        X = np.array([[-2.0, -1.0], [-1.0, -0.5]])
        self.assertIsNone(ph.canonicalImageElement(X, None, [IDENTITY, FLIP_BOTH]))

    def test_an_empty_group_means_no_element(self):
        X = np.array([[-2.0, -1.0], [-1.0, -0.5]])
        self.assertIsNone(
            ph.canonicalImageElement(X, canonical_under([IDENTITY]), []))


class TheStoredPhaseIsOnTheChosenImage(unittest.TestCase):
    """`traceMultiMin` has to store the phase on the image `applySymmetries` chooses.

    The minimum below the tilt has `y = g x / k`, so both components carry the sign of `x`
    and the measure prefers the image with both positive. The assertions are against the
    closed form of that minimum and of its slope, not against another run of the code.
    """

    PROBE = 0.5 * T0

    def traced(self, group):
        return ph.traceMultiMin(
            V, dV, d2V_dxdt, d2V,
            [[np.zeros(2), 2.5 * T0]],
            tLow=0.05 * T0, tHigh=3.0 * T0, deltaX_target=0.1,
            applySymmetries=canonical_under(group),
            invGroupElements=group,
        )

    def phase_below_the_tilt(self, group):
        covering = [p for p in self.traced(group).values()
                    if float(p.Tmin) <= self.PROBE <= float(p.Tmax)]
        self.assertTrue(covering, msg=f"no phase covers t = {self.PROBE}")
        return covering[0]

    def nearest_node(self, phase):
        """The stored node nearest the probe, and its stored slope.

        Read from `X` and `dXdT` rather than through `valAt`, so that the two are compared
        as they are stored: an interpolant would hide a slope array that disagrees with the
        field values it belongs to.
        """
        T = np.asarray(phase.T, dtype=float)
        i = int(np.argmin(np.abs(T - self.PROBE)))
        return (np.atleast_2d(np.asarray(phase.X, dtype=float))[i],
                np.atleast_2d(np.asarray(phase.dXdT, dtype=float))[i],
                float(T[i]))

    def test_the_phase_below_the_tilt_is_on_the_image_with_both_fields_positive(self):
        """The group of the 2HDM, and the image its measure picks."""
        x, _, t = self.nearest_node(self.phase_below_the_tilt([IDENTITY, FLIP_BOTH]))
        self.assertGreater(float(x[0]), 0.0, msg=f"stored at x = {x} at t = {t}")
        self.assertGreater(float(x[1]), 0.0, msg=f"stored at x = {x} at t = {t}")
        np.testing.assert_allclose(x, displaced_minimum(t), rtol=2e-3)

    def test_the_element_applied_to_the_seed_is_applied_to_what_is_stored(self):
        """A one-element group flips twice, and the two flips have to cancel.

        `[FLIP_BOTH]` has no identity, so `applySymmetries` maps every point to its image:
        the seed is flipped before the trace starts, and the canonical image of the trace
        is the flip of the image it was laid down on. The stored phase therefore has to sit
        where the untransformed trace sits, which `[IDENTITY]` gives. With the seed flipped
        and the trace stored as traced, the two disagree by exactly the flip, which is the
        pairing that gave `S_3/T = 336.6` instead of 152.6 on the 2HDM.

        `dXdT` is read as well as `X`, because a phase whose slopes were left on the other
        image describes a minimum moving the wrong way.
        """
        x_plain, slope_plain, t_plain = self.nearest_node(
            self.phase_below_the_tilt([IDENTITY]))
        x_flipped, slope_flipped, t_flipped = self.nearest_node(
            self.phase_below_the_tilt([FLIP_BOTH]))
        self.assertAlmostEqual(t_plain, t_flipped, places=6)
        # Against arithmetic first, so that the comparison below is between two traces of
        # the minimum this potential has and not between two wrong answers.
        np.testing.assert_allclose(np.abs(x_plain), displaced_minimum(t_plain), rtol=2e-3)
        np.testing.assert_allclose(np.abs(slope_plain), np.abs(displaced_slope(t_plain)),
                                   rtol=5e-2)
        np.testing.assert_allclose(x_flipped, x_plain, rtol=1e-6)
        np.testing.assert_allclose(slope_flipped, slope_plain, rtol=1e-6)

    def test_the_stored_phase_is_its_own_canonical_image(self):
        """What canonical means, for the group the 2HDM has.

        A stored phase that `applySymmetries` would still move is on an image some other
        seed can also reach, and the pair is then carried as two phases.
        """
        group = [IDENTITY, FLIP_BOTH]
        phase = self.phase_below_the_tilt(group)
        X = np.atleast_2d(np.asarray(phase.X, dtype=float))
        reference = X[np.argmax(np.linalg.norm(X, axis=-1))]
        np.testing.assert_allclose(canonical_under(group)(reference), reference, atol=0.0)


class _StubPotential:
    """The toy potential behind the interface `findPhases` reads.

    Only the members `findPhases` and the passes after it touch are provided. The symmetry
    map and the group are set independently here, which is what makes it possible to see
    whether the group reaches the tracer at all: the map alone already acts on the seed.
    """

    Ndim = 2

    def __init__(self, applySymmetries, invGroupElements, zero_T_minimum):
        self.applySymmetries = applySymmetries
        self.invGroupElements = invGroupElements
        self.zero_T_minimum = np.asarray(zero_T_minimum, dtype=float)
        self.forbidPhaseCrit = None

    def approxZeroTMin(self):
        """The displaced minimum, as a model's own returns it.

        The origin will not do: `findPhases` seeds at the lowest temperature, where the
        origin is a saddle and the displaced minimum is further from it than the descent
        is allowed to step, so nothing is traced at all.
        """
        return [self.zero_T_minimum]

    def Vtot(self, X, T, **kwargs):
        return V(X, T)

    def gradV(self, X, T):
        return dV(X, T)

    def dgradV_dT(self, X, T):
        return d2V_dxdt(X, T)

    def d2V(self, X, T):
        return d2V(X, T)


class TheGroupReachesTheTracer(unittest.TestCase):
    """`findPhases` has to hand the tracer the transformations the potential is invariant under.

    Without them `canonicalImageElement` has nothing to choose from and returns the trace
    as it was laid down, which is the behaviour the mirror-image pairing came from. The map
    and the group are separated here: both runs below flip the seed, and only the first is
    given the group to store the phase on the canonical image, so the two phases differ by
    exactly the flip. A run that never receives the group gives the same phase twice.
    """

    PROBE = 0.5 * T0

    def node_below_the_tilt(self, invGroupElements):
        phases = ph.Phases.findPhases(self.stub(invGroupElements), genMirrorPhases=False)
        covering = [p for p in phases.values()
                    if float(p.Tmin) <= self.PROBE <= float(p.Tmax)]
        self.assertTrue(covering, msg=f"no phase covers t = {self.PROBE}")
        T = np.asarray(covering[0].T, dtype=float)
        i = int(np.argmin(np.abs(T - self.PROBE)))
        return np.atleast_2d(np.asarray(covering[0].X, dtype=float))[i], float(T[i])

    def stub(self, invGroupElements):
        phases = object.__new__(ph.Phases)
        phases.pot = _StubPotential(canonical_under([FLIP_BOTH]), invGroupElements,
                                    displaced_minimum(0.05 * T0))
        phases.verbose = False
        # `findPhases` computes `deltaX_target` as `100 * x_eps` and passes it itself;
        # repeating it here reaches `traceMinimum` twice.
        phases.tracingArgs = {}
        phases.Tmin = 0.05 * T0
        phases.Tmax = 3.0 * T0
        phases.conversionFactor = 1.0
        phases.diftol = 1.0e-2
        phases.x_eps = 1.0e-3
        return phases

    def test_the_phase_is_stored_on_the_image_the_group_makes_canonical(self):
        with_group, t_with = self.node_below_the_tilt([FLIP_BOTH])
        without_group, t_without = self.node_below_the_tilt([])
        self.assertAlmostEqual(t_with, t_without, places=6)
        np.testing.assert_allclose(np.abs(without_group), displaced_minimum(t_without),
                                   rtol=2e-3)
        np.testing.assert_allclose(with_group, -without_group, rtol=1e-6)


class AMapThatIsNotAGroupActionIsDeclined(unittest.TestCase):
    """A symmetry map no element of the group accounts for leaves the trace alone.

    `applySymmetries` is a method of the potential and a model may override it. Where what
    it returns is not the reference node carried by any one element, the nearest element is
    not the map it applied, and using it would move the phase to a point the potential does
    not relate to the one that was traced. Losing the deduplication is recoverable;
    `removeRedundantPhases` may still catch it. A phase moved to the wrong place is not.
    """

    GROUP = [IDENTITY, FLIP_BOTH]

    def test_an_image_outside_the_group_is_not_applied(self):
        X = np.array([[-1.0, -0.5], [-2.0, -1.0]])
        # a rotation by a quarter turn, which neither element of the group performs
        rotated = lambda x: np.asarray(x, dtype=float) @ np.array([[0.0, -1.0], [1.0, 0.0]])
        self.assertIsNone(ph.canonicalImageElement(X, rotated, self.GROUP))

    def test_a_scaled_image_is_not_applied(self):
        """A map that rescales is not a symmetry of the potential, whatever its direction."""
        X = np.array([[-1.0, -0.5], [-2.0, -1.0]])
        self.assertIsNone(
            ph.canonicalImageElement(X, lambda x: -2.0 * np.asarray(x, dtype=float),
                                     self.GROUP))

    def test_a_phase_at_the_origin_is_judged_on_an_absolute_floor(self):
        """The tolerance is relative to the field scale, and needs a floor at zero scale.

        A phase that sits at the origin has no field scale, so a purely relative tolerance
        is zero there and any map that does not return the origin exactly would be refused,
        including the identity on a point displaced by rounding. The symmetric phase of a
        conformal model is traced at the origin to within the tracing accuracy, so this is
        the case to get right rather than a contrived one.
        """
        X = np.zeros((3, 2))
        displaced = lambda x: np.full(2, 1.0e-9)
        got = ph.canonicalImageElement(X, displaced, self.GROUP)
        self.assertIsNotNone(
            got, msg="a phase at the origin was refused for a 1e-09 displacement")
        np.testing.assert_allclose(got, IDENTITY)

    def test_the_group_action_itself_is_still_applied(self):
        """The guard must not refuse the map the shipped `applySymmetries` performs."""
        X = np.array([[-1.0, -0.5], [-2.0, -1.0]])
        got = ph.canonicalImageElement(X, canonical_under(self.GROUP), self.GROUP)
        np.testing.assert_allclose(got, FLIP_BOTH)


if __name__ == "__main__":
    unittest.main()
