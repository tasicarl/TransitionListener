r"""The transformation that puts a traced phase on its canonical image, where it is decided.

`canonicalImageElement` reads one symmetry transformation off the node of a trace furthest
from the origin and returns it, so that the whole trace can be put on one image of the
potential's symmetry. Two things could decide that transformation by something other than the
potential.

The first is a tie. The transformation is identified by matching its action on the reference
node against the image `applySymmetries` chose, and `np.argmin` returns the first of several
equal distances. Any element of the group that fixes the reference node ties with the
identity, and at the origin every element of every group fixes it, so a trace at the origin
was given the first element of the group. For the real two Higgs doublet model that element
is the flip `(phi_1, phi_2) -> (-phi_1, -phi_2)`, so a phase that needed no transformation
was mirrored.

The second is the measure `sum_i 2^i theta(phi_i)` that `applySymmetries` maximises. The
component with the highest index carries the largest weight, so it decides the image on its
own, and the sign of a component far below the tracing resolution is settled by the tracing
noise. Measured on the 2HDM's group: a reference node at `(246.0, +1e-09)` GeV is kept as it
is, and one at `(246.0, -1e-09)` GeV is returned as `(-246.0, +1e-09)` GeV, so two nodes
2e-09 GeV apart are canonicalised onto images 492 GeV apart.

Part of TransitionListener v2
"""
import unittest

import numpy as np

from transitionlistener import phases as PH

IDENTITY = np.diag([1.0, 1.0])
FLIP_BOTH = np.diag([-1.0, -1.0])
#: The group `R2HDM` builds, in the order it builds it: the flip comes first, which is what
#: makes `np.argmin` return the flip on a tie.
TWO_HIGGS_DOUBLET = [FLIP_BOTH, IDENTITY]
#: The 2HDM's zero-temperature minimum, to the digits the benchmark reports.
TRUE_VACUUM = np.array([11.48, 246.33])


def apply_symmetries(X, group=TWO_HIGGS_DOUBLET, ndim=2):
    r"""`generic_potential.applySymmetries`, as it stands, so the measure is under test too.

    Copied rather than imported because the function is a method of a potential and needs a
    whole model behind it, while all it uses is the group and the number of fields. The copy
    is checked against the real method in
    `test_the_copied_measure_agrees_with_the_potentials_own`.
    """
    max_measure = 2 ** ndim - 1
    measure, chosen = -1, np.asarray(X, dtype=float).copy()
    for g in group:
        image = np.asarray(X, dtype=float) @ g
        new = sum(2 ** i * np.heaviside(image[i], 1) for i in range(ndim))
        if new > measure:
            measure, chosen = new, image.copy()
        if measure == max_measure:
            break
    return chosen


def element_for(trace, group=TWO_HIGGS_DOUBLET, measure=apply_symmetries):
    """The transformation `canonicalImageElement` returns for a trace, as a short name."""
    got = PH.canonicalImageElement(np.atleast_2d(np.asarray(trace, dtype=float)),
                                  measure, group)
    if got is None:
        return "none"
    if np.allclose(got, IDENTITY):
        return "identity"
    if np.allclose(got, FLIP_BOTH):
        return "flip"
    return "something else"


def moves_the_trace(trace, group=TWO_HIGGS_DOUBLET, measure=apply_symmetries):
    """Whether the caller would move the phase, which is what the claims are about.

    Two answers leave a phase where it is, and the tests are about the phase and not about
    which of the two came back: the identity, where the transformation is known and is the
    one that does nothing, and None, where the function declines a map it cannot account for
    and the caller leaves the trace untouched.
    """
    trace = np.atleast_2d(np.asarray(trace, dtype=float))
    got = PH.canonicalImageElement(trace, measure, group)
    if got is None:
        return False
    return not np.allclose(trace @ got, trace, atol=0.0, rtol=0.0)


class TheCopiedMeasureIsTheRealOne(unittest.TestCase):

    def test_the_copied_measure_agrees_with_the_potentials_own(self):
        """The copy above has to behave as the method, or the tests measure the copy.

        A bare object carrying only the two attributes the method reads is enough to call
        it, which is what shows that those two are all it reads.
        """
        from transitionlistener.generic_potential import generic_potential

        class _Minimal:
            Ndim = 2
            invGroupElements = TWO_HIGGS_DOUBLET
            applySymmetries = generic_potential.applySymmetries

        real = _Minimal()
        for point in ([11.48, 246.33], [-11.48, -246.33], [246.0, 1.0e-9],
                      [246.0, -1.0e-9], [0.0, 0.0], [5.0, 0.0], [0.0, -5.0]):
            with self.subTest(str(point)):
                np.testing.assert_allclose(
                    apply_symmetries(np.asarray(point, dtype=float)),
                    np.asarray(real.applySymmetries(np.asarray(point, dtype=float)),
                               dtype=float),
                    atol=0.0,
                    err_msg="the copied measure no longer matches the potential's")


class ATieDoesNotMirrorAPhase(unittest.TestCase):

    def test_a_trace_at_the_origin_is_left_alone(self):
        """Every element of every group fixes the origin, so all the distances tie."""
        self.assertFalse(
            moves_the_trace([[0.0, 0.0], [1.0e-14, -2.0e-14]]),
            msg="a trace at the origin was moved; with the flip first in the group that "
                "mirrors a phase that needed nothing done to it")
        self.assertEqual(element_for([[0.0, 0.0], [1.0e-14, -2.0e-14]]), "identity",
                         msg="the map is known here and is the identity, so it is returned "
                             "rather than declined")

    def test_a_trace_already_on_its_canonical_image_is_left_alone(self):
        self.assertFalse(moves_the_trace([TRUE_VACUUM, TRUE_VACUUM * 0.9]))
        self.assertEqual(element_for([TRUE_VACUUM, TRUE_VACUUM * 0.9]), "identity")

    def test_a_group_whose_first_element_fixes_the_reference_does_not_win_the_tie(self):
        """The tie is not special to the origin: any element that fixes the node ties.

        Here the group contains the reflection of the first field alone, which fixes a node
        whose first field vanishes, and it is listed first. The reference node is on its
        canonical image, so the answer has to be that there is nothing to do.
        """
        flip_first_field = np.diag([-1.0, 1.0])
        self.assertFalse(
            moves_the_trace([[0.0, 246.0]], group=[flip_first_field, IDENTITY]),
            msg="an element that fixes the reference node won the tie against the identity")

    def test_a_phase_genuinely_on_the_other_image_is_still_turned(self):
        """The case the function exists for has to keep working."""
        self.assertEqual(
            element_for([-TRUE_VACUUM, -TRUE_VACUUM * 0.9]), "flip",
            msg="a phase on the mirror image was not turned onto the canonical one")


class ASignTheTracerDoesNotResolveDecidesNothing(unittest.TestCase):

    def test_two_nodes_a_hair_apart_are_treated_the_same(self):
        """The measured case: `(246.0, +-1e-09)` GeV, images 492 GeV apart.

        Whatever the function does with these two, it has to do the same thing with both,
        because the potential cannot tell them apart. Leaving both alone is the answer that
        cannot move a phase on the strength of a digit the tracer does not resolve.
        """
        plus = moves_the_trace([[246.0, 1.0e-9]])
        minus = moves_the_trace([[246.0, -1.0e-9]])
        self.assertEqual(plus, minus,
                         msg="a change of 2e-09 GeV in the second field changed whether the "
                             "phase is moved")
        self.assertFalse(plus, msg="the phase was moved on a sign the tracer does not "
                                   "resolve")

    def test_the_sign_matters_once_the_component_is_resolved(self):
        """Above the resolution the sign is the potential's and must be read.

        The threshold is `_IMAGE_SIGN_RTOL` of the node's own length, so this walks the
        second field from far below it to far above it and asks where the answer changes.
        """
        for factor, expected in ((1.0e-3, False), (1.0e-1, False),
                                 (1.0e+1, True), (1.0e+3, True)):
            with self.subTest(f"{factor:g} times the threshold"):
                size = factor * PH._IMAGE_SIGN_RTOL * 246.0
                self.assertEqual(
                    moves_the_trace([[-246.0, -size]]), expected,
                    msg=f"a second field {size:.3e} GeV below zero, against a threshold of "
                        f"{PH._IMAGE_SIGN_RTOL * 246.0:.3e} GeV, "
                        f"{'moved' if not expected else 'did not move'} the phase")

    def test_the_threshold_admits_the_tracers_measured_residual(self):
        """The value itself, against the measurement that bounds it below.

        Every other test here reads the threshold out of the module, so none of them says
        what its value should be. On the 2HDM benchmark the phase tracer's residual is
        2.7e-06 of the field scale, and a component that small has a sign the tracer did not
        resolve, so the threshold has to be above it. It also has to stay far below 1, or a
        field component of the same order as the node itself would count as unresolved.
        """
        self.assertGreater(PH._IMAGE_SIGN_RTOL, 10.0 * 2.7e-06,
                           msg="the threshold no longer covers the tracer's measured "
                               "residual of 2.7e-06 of the field scale")
        self.assertLess(PH._IMAGE_SIGN_RTOL, 1.0e-02,
                        msg="the threshold now treats a field component a per cent of the "
                            "node's length as unresolved")

    def test_a_measure_that_returns_a_non_finite_point_for_the_reference_is_refused(self):
        """The case the test below says is "caught further down", which nothing asked.

        Where the reference's own image is not finite, every distance to a group element's
        action on it is `nan` or `inf`, `np.argmin` returns the first element, and for the
        2HDM that is the flip, so the phase would be mirrored on exactly the input saying the
        measure cannot be trusted. The refusal is the comparison in the element search, and
        removing it leaves every other test in this file passing.
        """
        reference = np.array([-246.0, -11.5])
        for name, bad in (("not a number", np.nan), ("positive infinity", np.inf),
                          ("negative infinity", -np.inf)):
            with self.subTest(name):
                def broken(X, _bad=bad, **k):
                    return np.full_like(np.asarray(X, dtype=float), _bad)

                self.assertEqual(
                    element_for([reference], measure=broken), "none",
                    msg="a measure that could not place the reference had its map applied")

    def test_a_measure_that_returns_a_non_finite_point_for_the_probe_is_refused(self):
        """A model may override `applySymmetries`, and the probe goes through it too.

        The measure below answers the reference node correctly and returns a non-finite
        point for the probe, which is the only arrangement under which this refusal decides
        anything. A measure that fails on the reference as well is caught further down, where
        the transformation is matched against the image, and a `nan` for the probe fails the
        comparison that follows anyway, since every comparison against a `nan` is false. An
        infinity does not: the distance between the probe and its image is then infinite,
        which reads as the probe having been moved onto a different image, and the
        transformation would be read off a point the measure could not place.
        """
        # A node with one resolved component and one below the threshold, so that a probe
        # is taken at all, and on the mirror image, so that the reference is moved and the
        # early return for a node already canonical does not fire first.
        reference = np.array([-246.0, -1.0e-9])
        for name, bad in (("not a number", np.nan), ("positive infinity", np.inf),
                          ("negative infinity", -np.inf)):
            with self.subTest(name):
                seen = []

                def broken(X, _bad=bad, _seen=seen, **k):
                    X = np.asarray(X, dtype=float)
                    _seen.append(X.copy())
                    # The probe is the one with the second field pushed above zero.
                    if X[1] > 0.0:
                        return np.full_like(X, _bad)
                    return apply_symmetries(X)

                got = element_for([reference], measure=broken)
                self.assertEqual(len(seen), 2, msg="the probe was not taken")
                self.assertTrue(np.isfinite(apply_symmetries(seen[0])).all(),
                                msg="the fixture has to answer the reference finitely, or "
                                    "the refusal under test is not the one that fires")
                self.assertEqual(
                    got, "none",
                    msg=f"{name}: a non-finite image of the probe was taken as the probe "
                        f"having moved, and the answer came back as {got}; a map that cannot "
                        f"be accounted for is declined, not answered with the identity")

    def test_a_node_with_no_unresolved_component_takes_the_short_path(self):
        """Nothing is probed where every component is resolved, and the answer is the map.

        The probe calls `applySymmetries` a second time, so it must not run on the common
        case. A measure that raises when called on anything but the reference shows that it
        does not.
        """
        calls = []

        def once(X, **k):
            X = np.asarray(X, dtype=float)
            calls.append(X.copy())
            if len(calls) > 1:
                raise AssertionError("applySymmetries was called a second time")
            return apply_symmetries(X)

        self.assertEqual(element_for([-TRUE_VACUUM], measure=once), "flip")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()


#: A group with independent flips, unlike the 2HDM's, whose only non-trivial element turns
#: both components at once. The first component alone, then both, then the second alone, then
#: the identity: a sign that the tracer does not resolve can select between these.
FLIP_FIRST = np.diag([-1.0, 1.0])
FLIP_SECOND = np.diag([1.0, -1.0])
INDEPENDENT_FLIPS = [FLIP_FIRST, FLIP_BOTH, FLIP_SECOND, IDENTITY]


def independent_measure(X, **k):
    """`apply_symmetries` over `INDEPENDENT_FLIPS`.

    The helpers above take the group for the element search and the measure separately, and
    the measure's own default group is the 2HDM's. Passing only the group would leave the
    measure choosing images with one group while the search matched elements of another,
    which answers neither question.
    """
    return apply_symmetries(X, group=INDEPENDENT_FLIPS)


class TheSignCutoffIsRelativeToTheNodeAndNotToOne(unittest.TestCase):
    """`scale` carries a floor of 1.0 for the match tolerance, which the cutoff must not use.

    With the floor, the cutoff is `_IMAGE_SIGN_RTOL` in absolute terms for every phase whose
    node is shorter than 1, which depends on the model's units: the conformal examples are
    quoted in GeV with `v_GeV = 0.1`. A component worth 5e-04 of its node is resolved, five
    times the relative cutoff, and must be treated as such however short the node is.
    """

    def test_a_resolved_component_on_a_short_node_decides_the_image(self):
        node = np.array([-0.1, -5.0e-05])
        length = float(np.linalg.norm(node))
        self.assertGreater(abs(node[1]) / length, PH._IMAGE_SIGN_RTOL,
                           msg="this component is below the relative cutoff, so the test "
                               "does not measure what it claims")
        self.assertLess(abs(node[1]), PH._IMAGE_SIGN_RTOL,
                        msg="this component is above the cutoff in absolute terms, so the "
                            "floor would not change the answer and the test is vacuous")
        self.assertEqual(element_for(node), "flip",
                         msg="a resolved component was treated as unresolved because the "
                             "cutoff was measured against 1 rather than the node")


class ADifferentElementOnTheProbeIsStillTheNoisesChoice(unittest.TestCase):
    """Both points moving is not the same as both moving the same way.

    With independent flips in the group, a reference whose second component is below the
    tracing resolution and a probe of the opposite unresolved sign are each moved, but by
    different elements, so the image still rests on the digit the tracer does not resolve.
    """

    def _element_the_measure_applies(self, point):
        """Which element of the group the measure's image of `point` corresponds to.

        Asked of the measure directly. Going through `canonicalImageElement` would run the
        logic under test, which declines this case, so it cannot establish the precondition.
        """
        image = independent_measure(np.asarray(point, dtype=float))
        for g, name in zip(INDEPENDENT_FLIPS,
                           ["flip_first", "flip_both", "flip_second", "identity"]):
            if np.allclose(np.asarray(point, dtype=float) @ g, image):
                return name
        return "none of the group"

    def test_the_two_points_are_carried_by_different_elements(self):
        """The precondition: without this the group does not separate them."""
        node = np.array([-2.0, -1.0e-05])
        length = float(np.linalg.norm(node))
        self.assertLess(abs(node[1]) / length, PH._IMAGE_SIGN_RTOL,
                        msg="the second component is resolved here, so there is nothing to "
                            "probe and the test is vacuous")
        probe = np.array([-2.0, +PH._IMAGE_SIGN_RTOL * length])
        self.assertEqual(self._element_the_measure_applies(node), "flip_both")
        self.assertEqual(self._element_the_measure_applies(probe), "flip_first")

    def test_a_probe_carried_by_another_element_declines_the_map(self):
        node = np.array([-2.0, -1.0e-05])
        self.assertEqual(
            element_for(node, group=INDEPENDENT_FLIPS, measure=independent_measure), "none",
            msg="the map was accepted although the probe is carried by another element, so "
                "the image rests on a sign the tracer does not resolve")

    def test_both_being_moved_is_not_enough(self):
        """The two are moved, which is what the earlier form of the test asked."""
        node = np.array([-2.0, -1.0e-05])
        length = float(np.linalg.norm(node))
        probe = np.array([-2.0, +PH._IMAGE_SIGN_RTOL * length])
        for point in (node, probe):
            with self.subTest(str(point)):
                moved = not np.allclose(
                    apply_symmetries(point, group=INDEPENDENT_FLIPS), point)
                self.assertTrue(moved, msg="this point is not moved, so 'both moved' is not "
                                           "the condition under test")
        self.assertFalse(moves_the_trace(node, group=INDEPENDENT_FLIPS, measure=independent_measure),
                         msg="the phase was moved although the probe chose a different "
                             "element, so the image rests on an unresolved sign")
