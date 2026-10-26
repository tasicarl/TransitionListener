r"""The bounce is computed on every image of the false vacuum and the smallest action kept.

Where the potential is invariant under a group, each minimum stands for a set of images and
all of them are the same physical state. Which image a phase is traced onto is settled by the
sign of a numerical step, so the two phases of a transition can be stored on different
images, and the bounce is then computed along a path between images rather than along the
path the field takes. On the real two Higgs doublet model at `lambda3 = 5.868` that gave a
field separation of 264.54 GeV against 231.55 GeV at the neighbouring couplings and
`S_3/T = 336.84` against 152.97, so the bubble nucleation rate never rose and the
coupling returned error 8, no percolation.

The bubble nucleates along whichever pairing carries the smallest action, so that is the
pairing to take. Every image is offered to the solver and the smallest action it returns is
the one the rate is built from. The images are ordered with the one nearest the true vacuum
first, which is not what decides the pairing but is measured to be the pairing that wins:
along the `lambda3` line scan of figure 15 of the code paper, 36 temperatures were sampled
across the three couplings, of which 22 have the false vacuum at the origin and one pairing
only. Of the remaining 14, both pairings resolved at 12, and at those 12 the nearest image
carried the smaller action every time, by a factor between 1.17 and 2.46.

Part of TransitionListener v2
"""
import ast
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import transitionlistener
from transitionlistener import bubbledynamics as bd
from transitionlistener import bubbledynamics_fixedstep as bdf
from transitionlistener import config, errors, nucleation, tunneling1D
from transitionlistener import pathDeformation as pd
from transitionlistener.pathDeformation import _IMAGE_SEPARATION_RTOL
from transitionlistener.pathDeformation import (bounceAction, bounceActionOverImages,
                                                imagesOfTheFalseVacuum, imagesToPairOver)

IDENTITY = np.diag([1.0, 1.0])
FLIP_BOTH = np.diag([-1.0, -1.0])
#: The group the 2HDM is found to have: `m12^2 phi_1 phi_2` forbids either reflection alone.
TWO_HIGGS_DOUBLET = [FLIP_BOTH, IDENTITY]

#: The two minima at T = 30.0 GeV on the 2HDM at lambda3 = 5.868, as they were traced.
FALSE_VACUUM = np.array([-27.416, -15.335])
TRUE_VACUUM = np.array([11.48, 246.33])


def nearest_image(x_false, x_true, group):
    """The first entry of the enumeration, which is the image nearest the true vacuum."""
    return imagesOfTheFalseVacuum(x_false, x_true, group)[0]


class TheImagesAreEnumeratedNearestFirst(unittest.TestCase):

    def test_the_image_on_the_other_side_comes_first(self):
        got = nearest_image(FALSE_VACUUM, TRUE_VACUUM, TWO_HIGGS_DOUBLET)
        np.testing.assert_allclose(got, -FALSE_VACUUM)

    def test_the_separation_it_gives_is_the_measured_one(self):
        """231.55 GeV, the value the neighbouring couplings give, not 264.54 GeV."""
        turned = nearest_image(FALSE_VACUUM, TRUE_VACUUM, TWO_HIGGS_DOUBLET)
        self.assertAlmostEqual(float(np.linalg.norm(TRUE_VACUUM - turned)), 231.55, places=1)
        self.assertAlmostEqual(
            float(np.linalg.norm(TRUE_VACUUM - FALSE_VACUUM)), 264.54, places=1,
            msg="the fixture no longer reproduces the separation that was measured")

    def test_both_images_are_offered(self):
        """Both are candidates, since the pairing is decided by comparing their actions."""
        got = imagesOfTheFalseVacuum(FALSE_VACUUM, TRUE_VACUUM, TWO_HIGGS_DOUBLET)
        self.assertEqual(len(got), 2, msg=f"got {len(got)} candidates, expected 2")
        np.testing.assert_allclose(got[0], -FALSE_VACUUM)
        np.testing.assert_allclose(got[1], FALSE_VACUUM)

    def test_a_pair_already_on_one_image_keeps_that_image_first(self):
        """The traced point has to come first on a tie, or a good pair is reordered."""
        got = nearest_image(-FALSE_VACUUM, TRUE_VACUUM, TWO_HIGGS_DOUBLET)
        np.testing.assert_allclose(got, -FALSE_VACUUM, atol=0.0)

    def test_an_image_equal_to_the_true_vacuum_is_ordered_first(self):
        """What lets `bounceActionOverImages` leave the no-barrier verdict alone.

        Where the false vacuum is itself an image of the true vacuum the two phases are one
        state and there is no transition. `bounceAction` says so by reporting `trantype` 2
        with action 0 for the pairing whose two points coincide, and action 0 is smaller
        than any bounce's, so it is kept. That is only right because the coinciding image
        is the one tried first, which it is because its distance to the true vacuum is
        zero.
        """
        mirror_of_the_true_vacuum = -TRUE_VACUUM
        got = imagesOfTheFalseVacuum(mirror_of_the_true_vacuum, TRUE_VACUUM,
                                     TWO_HIGGS_DOUBLET)
        self.assertEqual(len(got), 2, msg=f"got {len(got)} candidates, expected 2")
        np.testing.assert_allclose(got[0], TRUE_VACUUM, atol=0.0)
        self.assertEqual(float(np.linalg.norm(got[0] - TRUE_VACUUM)), 0.0)

    def test_a_false_vacuum_at_the_origin_costs_one_bounce(self):
        """Every image of the origin is the origin, so there is nothing to choose and
        nothing to compute twice."""
        got = imagesOfTheFalseVacuum(np.zeros(2), TRUE_VACUUM, TWO_HIGGS_DOUBLET)
        self.assertEqual(len(got), 1, msg=f"the origin gave {len(got)} candidates")
        np.testing.assert_allclose(got[0], np.zeros(2), atol=0.0)

    def test_the_same_point_is_recognised_in_any_units(self):
        """Two images equal to the precision the field values hold are one point.

        The separation has to be measured against a scale of the problem, which is the
        length of the path the bounce traverses, `|x_false - x_true|`, and not as an
        absolute distance: the 2HDM's fields are in GeV, while the conformal dark U(1)
        example shipped with the code works in internal units 1e4 times smaller, and an
        absolute threshold would mean something different in each. Measuring it against the
        false vacuum's own length would be worse still, since for a false vacuum at the
        origin that length is whatever the tracing left behind. Where the two minima
        coincide the path has no length, and the comparison falls back on the larger of the
        true vacuum's length and 1.
        """
        for name, true_vacuum in (("2HDM, fields in GeV", TRUE_VACUUM),
                                  ("internal units 1e4 smaller", TRUE_VACUUM * 1.0e-4),
                                  ("internal units 1e3 larger", TRUE_VACUUM * 1.0e3)):
            with self.subTest(name):
                point = true_vacuum * 1.0e-17
                got = imagesOfTheFalseVacuum(point, true_vacuum, TWO_HIGGS_DOUBLET)
                self.assertEqual(len(got), 1,
                                 msg=f"{name}: {len(got)} candidates for one point")
                far = true_vacuum * 0.1
                self.assertEqual(
                    len(imagesOfTheFalseVacuum(far, true_vacuum, TWO_HIGGS_DOUBLET)), 2,
                    msg=f"{name}: a genuine pair of images was collapsed into one")

    def test_every_image_the_tracer_resolves_is_offered_to_the_solver(self):
        """A false vacuum the group moves to a different point is a second bounce.

        These separations are far above the resolution the phase tracing works to, so the two
        images are two states and each one's action has to be computed. The pairing whose
        action differs by a factor two on the benchmark is of this kind.
        """
        true_vacuum = np.array([11.48, 246.33])
        for name, false_vacuum in (
                ("off the origin, separation 0.271 of the path",
                 np.array([27.416, 15.335])),
                ("off the origin, separation 0.283 of the path",
                 np.array([30.0, 13.7])),
                ("a hundredth of the field scale", true_vacuum * 1.0e-2),
                ("a thousandth of the field scale", true_vacuum * 1.0e-3)):
            with self.subTest(name):
                got = imagesOfTheFalseVacuum(false_vacuum, true_vacuum, TWO_HIGGS_DOUBLET)
                sep = float(np.linalg.norm(2 * false_vacuum)
                            / np.linalg.norm(false_vacuum - true_vacuum))
                self.assertGreater(sep, _IMAGE_SEPARATION_RTOL,
                                   msg=f"{name}: separation {sep:.3e} is below the tracing "
                                       "resolution, so the test is not about what it says")
                self.assertEqual(
                    len(got), 2,
                    msg=f"{name}: {len(got)} candidate(s) for two distinct images, "
                        f"separation {sep:.3e} of the path length")

    def test_a_pair_the_tracer_cannot_resolve_is_one_image(self):
        """A symmetric phase costs one bounce, the tracer's residual notwithstanding.

        The phase tracing leaves a false vacuum that belongs at a fixed point of the group a
        little off it rather than on it, and the group then maps that residual onto its
        reflection. Those two points are the same state, and computing a bounce on each is the
        same path solved twice: measured on commit 065ef81 with `models/TL_2HDM.py` class `R2HDM` at shipped settings, with the benchmark parameters of `examples/bp_lisa.yaml` at
        `lambda3 = 5.6518` the second bounce took 49 per cent of all the time spent on bounces
        and the two actions agreed to 3.0e-05, a factor 1600 inside the solver's own bracket
        tolerance. The separations below are the ones measured there.
        """
        true_vacuum = np.array([11.48, 246.33])
        for name, false_vacuum in (
                ("the tracer's residual, 3.6e-13 of the path",
                 np.array([-1.448e-11, 4.118e-11])),
                ("the tracer's residual, 1.8e-12 of the path",
                 np.array([-9.759e-11, -1.862e-10])),
                ("the tracer's residual, 2.3e-12 of the path",
                 np.array([-1.542e-10, 2.060e-10])),
                ("a millionth of the field scale", true_vacuum * 1.0e-6)):
            with self.subTest(name):
                got = imagesOfTheFalseVacuum(false_vacuum, true_vacuum, TWO_HIGGS_DOUBLET)
                sep = float(np.linalg.norm(2 * false_vacuum)
                            / np.linalg.norm(false_vacuum - true_vacuum))
                self.assertLess(sep, _IMAGE_SEPARATION_RTOL,
                                msg=f"{name}: separation {sep:.3e} is above the tracing "
                                    "resolution, so the test is not about what it says")
                self.assertEqual(
                    len(got), 1,
                    msg=f"{name}: {len(got)} candidates for one state, separation "
                        f"{sep:.3e} of the path length")

    def test_the_threshold_sits_between_the_two(self):
        """The constant is the tracing resolution and not whatever merges the cases above.

        A pair just inside it is one state and a pair just outside it is two, so moving the
        constant by an order of magnitude in either direction is visible here. Without this the
        two tests above pass for any threshold between 1e-5 and 1e-2 of the path length.
        """
        true_vacuum = np.array([11.48, 246.33])
        path = float(np.linalg.norm(true_vacuum))
        for factor, expected in ((0.3, 1), (3.0, 2)):
            with self.subTest(f"{factor} times the threshold"):
                # The separation is 2|x_false| against a path of about |x_true|.
                false_vacuum = np.array([1.0, 0.0]) * (
                    0.5 * factor * _IMAGE_SEPARATION_RTOL * path)
                got = imagesOfTheFalseVacuum(false_vacuum, true_vacuum, TWO_HIGGS_DOUBLET)
                sep = float(np.linalg.norm(2 * false_vacuum)
                            / np.linalg.norm(false_vacuum - true_vacuum))
                self.assertEqual(
                    len(got), expected,
                    msg=f"a pair {sep:.3e} of the path apart, {factor} times the threshold "
                        f"of {_IMAGE_SEPARATION_RTOL:.1e}, came back as {len(got)} image(s)")

    def test_a_false_vacuum_at_the_origin_still_costs_one_bounce(self):
        """The case the merge existed for, which an equality test still covers.

        A false vacuum exactly at a fixed point of the group has one image, so the group costs
        nothing there however large it is. That is true of the identity for every point too.
        """
        true_vacuum = np.array([11.48, 246.33])
        got = imagesOfTheFalseVacuum(np.zeros(2), true_vacuum, TWO_HIGGS_DOUBLET)
        self.assertEqual(len(got), 1,
                         msg=f"a false vacuum at the origin gave {len(got)} candidates")

    def test_a_potential_with_no_symmetry_costs_one_bounce(self):
        got = imagesOfTheFalseVacuum(FALSE_VACUUM, TRUE_VACUUM, [])
        self.assertEqual(len(got), 1)
        np.testing.assert_allclose(got[0], FALSE_VACUUM, atol=0.0)

    def test_a_repeated_group_element_is_not_computed_twice(self):
        """The group may name the same map more than once, and the bounce is the same."""
        got = imagesOfTheFalseVacuum(FALSE_VACUUM, TRUE_VACUUM,
                                     [FLIP_BOTH, IDENTITY, FLIP_BOTH, IDENTITY])
        self.assertEqual(len(got), 2, msg=f"got {len(got)} candidates, expected 2")

    def test_the_traced_point_comes_first_when_two_images_are_equidistant(self):
        """A tie must not reorder, whichever order the group is given in.

        The two images are equidistant from the true vacuum whenever the false vacuum is
        perpendicular to it, and they are then two different points, unlike at the origin
        where every image coincides and a tie cannot be seen. Sorting on the distance alone
        would hand the first place to whichever element the group happens to list first.
        """
        perpendicular, true_vacuum = np.array([1.0, 0.0]), np.array([0.0, 5.0])
        # The enumeration appends the traced point first and sorts on the distance alone,
        # and Python's sort is stable, so this is what keeps the traced point first.
        self.assertAlmostEqual(float(np.linalg.norm(true_vacuum - perpendicular)),
                               float(np.linalg.norm(true_vacuum + perpendicular)))
        for group in ([IDENTITY, FLIP_BOTH], [FLIP_BOTH, IDENTITY]):
            got = imagesOfTheFalseVacuum(perpendicular, true_vacuum, group)
            self.assertEqual(len(got), 2)
            np.testing.assert_allclose(
                got[0], perpendicular, atol=0.0,
                err_msg=f"a tie reordered for group order {[g[0, 0] for g in group]}")

    def test_a_non_finite_minimum_costs_no_arithmetic(self):
        """The input is refused before the loop, not inside it.

        Leaving it to the comparisons also returns the point unchanged, but only because
        every comparison against a `nan` is false, and it runs `x_false @ g` on the
        non-finite point on the way, which numpy warns about.
        """
        for name, false_vacuum, true_vacuum in (
                ("nan in the false vacuum", np.array([np.nan, -15.3]), TRUE_VACUUM),
                ("inf in the false vacuum", np.array([np.inf, -15.3]), TRUE_VACUUM),
                ("nan in the true vacuum", FALSE_VACUUM, np.array([np.nan, 246.33]))):
            with self.subTest(name):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    got = imagesOfTheFalseVacuum(false_vacuum, true_vacuum,
                                                 TWO_HIGGS_DOUBLET)
                self.assertEqual(len(got), 1, msg=f"{name}: {len(got)} candidates")
                np.testing.assert_array_equal(
                    np.asarray(got[0], float), np.asarray(false_vacuum, float),
                    err_msg=f"{name}: the point came back changed")

    def test_a_non_finite_image_is_dropped(self):
        """A group element that is itself not finite may not reach the solver.

        `imagesOfTheFalseVacuum` returns points that `bounceAction` is then asked to
        integrate between, and a `nan` among them would be refused only after the solver
        had been entered, where the failure reads as an unresolved bounce rather than as a
        symmetry that cannot be applied.
        """
        got = imagesOfTheFalseVacuum(FALSE_VACUUM, TRUE_VACUUM,
                                     [np.diag([np.nan, 1.0]), FLIP_BOTH])
        self.assertEqual(len(got), 2, msg=f"got {len(got)} candidates, expected 2")
        for image in got:
            self.assertTrue(np.isfinite(image).all(),
                            msg=f"a non-finite image reached the solver: {image}")

    def test_the_traced_point_survives_a_group_that_is_entirely_non_finite(self):
        """The enumeration is never empty, which is what lets the caller take its first
        entry without a test for it."""
        got = imagesOfTheFalseVacuum(FALSE_VACUUM, TRUE_VACUUM,
                                     [np.diag([np.nan, 1.0]), np.diag([1.0, np.inf])])
        self.assertEqual(len(got), 1, msg=f"got {len(got)} candidates, expected 1")
        np.testing.assert_allclose(got[0], FALSE_VACUUM, atol=0.0)

    def test_the_transformation_does_not_change_the_potential(self):
        r"""The images are a degeneracy, so the choice may not move any energy.

        `V(x, y) = (x^2 + y^2 - 1)^2 - g x y` is invariant under the simultaneous flip and
        under nothing else, which is the structure the 2HDM has.
        """
        def V(x):
            x = np.asarray(x, dtype=float)
            return (x[0] ** 2 + x[1] ** 2 - 1.0) ** 2 - 0.7 * x[0] * x[1]
        for point in (FALSE_VACUUM / 30.0, np.array([0.3, -0.9]), np.array([-1.2, 0.4])):
            for image in imagesOfTheFalseVacuum(point, TRUE_VACUUM, TWO_HIGGS_DOUBLET):
                self.assertAlmostEqual(V(image), V(point), places=12)


#: Amplitudes of the quartic terms of the toy potential below. They place two pairs of
#: minima, a deep pair and a shallow pair, each pair a mirror pair under the simultaneous
#: flip, and leave the origin a local maximum.
TOY_A2, TOY_B4, TOY_C4 = 0.20, 0.20, 0.15


def toy_V(x, T=0.0):
    r"""`(x^2+y^2-1)^2 - a (x^2-y^2) - b [(x^2-y^2)^2 - 4x^2y^2] - 4 c x y (x^2-y^2)`.

    Every term is invariant under `(x, y) -> (-x, -y)` and no term is invariant under
    either reflection alone or under exchanging the two fields, which is the symmetry
    structure of the real 2HDM. Without the last term the potential would also be
    invariant under reflection in the line `y = x`, and that reflection maps the two
    images of the false vacuum onto each other while fixing the true vacuum, so the two
    pairings would carry equal actions by construction and nothing could be measured.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    u, v = x[:, 0], x[:, 1]
    r2, d = u * u + v * v, u * u - v * v
    return ((r2 - 1.0) ** 2 - TOY_A2 * d - TOY_B4 * (d * d - 4.0 * u * u * v * v)
            - TOY_C4 * 4.0 * u * v * d)


def toy_dV(x, T=0.0):
    """The gradient of :func:`toy_V`, checked against a central difference in the tests."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    u, v = x[:, 0], x[:, 1]
    r2, d = u * u + v * v, u * u - v * v
    g = np.empty_like(x)
    g[:, 0] = (4.0 * (r2 - 1.0) * u - TOY_A2 * 2.0 * u
               - TOY_B4 * (4.0 * d * u - 8.0 * u * v * v)
               - TOY_C4 * 4.0 * (v * d + 2.0 * u * u * v))
    g[:, 1] = (4.0 * (r2 - 1.0) * v + TOY_A2 * 2.0 * v
               - TOY_B4 * (-4.0 * d * v - 8.0 * u * u * v)
               - TOY_C4 * 4.0 * (u * d - 2.0 * u * v * v))
    return g


def toy_V_point(x, T=0.0):
    """:func:`toy_V` for a single point, the shape the minimum search passes."""
    out = toy_V(x, T)
    return out[0] if np.ndim(x) == 1 else out


def toy_dV_point(x, T=0.0):
    """:func:`toy_dV` for a single point, returning shape ``(2,)`` rather than ``(1, 2)``."""
    out = toy_dV(x, T)
    return out[0] if np.ndim(x) == 1 else out


def toy_d2V_point(x, T=0.0):
    """A central difference of :func:`toy_dV_point`, which the minimum search needs."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    step = 1.0e-6
    hessian = np.empty((x.shape[-1], x.shape[-1]))
    for i in range(x.shape[-1]):
        shift = np.zeros(x.shape[-1])
        shift[i] = step
        hessian[:, i] = (toy_dV_point(x + shift, T)
                         - toy_dV_point(x - shift, T)) / (2.0 * step)
    return hessian


class _ConstantPhase:
    """The least a phase has to be for `nucleation.evalNucleationCriterion` to read it.

    The toy potential does not depend on the temperature, so the minimum does not either.
    """

    def __init__(self, key, x):
        self.key = key
        self.x = np.asarray(x, dtype=float)
        self.Tmin, self.Tmax = 0.5, 2.0

    def valAt(self, T):
        return self.x


def toy_minimum(guess):
    """The minimum of :func:`toy_V` nearest `guess`."""
    result = optimize.minimize(lambda z: float(toy_V(z[None, :])[0]), np.asarray(guess, float),
                               jac=lambda z: toy_dV(z[None, :])[0], tol=1e-14)
    return result.x


class TheSmallestActionDecidesThePairing(unittest.TestCase):
    """The real solver, on a potential whose two pairings carry different actions.

    The toy potential is built so that both pairings resolve as ordinary first-order
    bounces and their actions differ by a factor 1.59, which is the same mechanism as on
    the 2HDM: the path from the other image runs past the origin, where the potential is
    higher. What this class cannot test is a case where the image nearest the true vacuum
    is not the one with the smaller action, because none was found: at the 12 temperatures of
    the 2HDM scan and the 3 of 72 parameter sets of this toy family where both pairings
    resolve, the nearest image carried the smaller action every time. That case is covered
    with a stub in
    `OnePairingWithoutAnActionDoesNotCostTheTemperature`, where the two actions can be set
    directly.
    """

    def setUp(self):
        self.false_vacuum = toy_minimum([-0.2, 1.1])
        self.true_vacuum = toy_minimum([1.2, 0.17])

    def _tdict(self, **extra):
        return dict(low_vev=self.true_vacuum, high_vev=self.false_vacuum, **extra)

    def test_the_gradient_matches_a_central_difference(self):
        """The toy is only evidence if its gradient is the gradient of its potential."""
        step = 1.0e-6
        for point in ([0.3, 0.9], [1.0, 0.2], [-0.4, 0.6]):
            with self.subTest(str(point)):
                point = np.asarray(point, float)
                numerical = np.array(
                    [(float(toy_V((point + step * axis)[None, :])[0])
                      - float(toy_V((point - step * axis)[None, :])[0])) / (2.0 * step)
                     for axis in np.eye(2)])
                np.testing.assert_allclose(toy_dV(point[None, :])[0], numerical, rtol=1e-6)

    def test_the_toy_has_the_symmetry_of_the_2hdm_and_no_more(self):
        point = np.array([[0.37, 0.81]])
        self.assertAlmostEqual(float(toy_V(point)[0]), float(toy_V(-point)[0]), places=12,
                               msg="the toy is not invariant under the simultaneous flip")
        for name, other in (("one reflection", point * [[1.0, -1.0]]),
                            ("exchange of the fields", point[:, ::-1])):
            self.assertNotAlmostEqual(
                float(toy_V(point)[0]), float(toy_V(other)[0]), places=6,
                msg=f"the toy is also invariant under {name}, so the pairings cannot differ")

    def test_the_two_minima_are_not_images_of_each_other(self):
        """A false vacuum that is the mirror of the true one is one state, not a transition."""
        self.assertGreater(float(np.linalg.norm(self.false_vacuum + self.true_vacuum)), 0.5)
        self.assertLess(float(toy_V(self.true_vacuum[None, :])[0]),
                        float(toy_V(self.false_vacuum[None, :])[0]),
                        msg="the false vacuum is not the higher minimum")
        self.assertGreater(float(toy_V(np.zeros((1, 2)))[0]),
                           float(toy_V(self.false_vacuum[None, :])[0]),
                           msg="the origin is not a barrier, so the pairings cannot differ")

    def _action_of(self, start):
        outdict: dict = {}
        bounceAction(1.0, toy_V, toy_dV, outdict,
                     dict(low_vev=self.true_vacuum, high_vev=start), False, 1.0)
        entry = outdict[1.0]
        self.assertEqual(entry["trantype"], 1,
                         msg=f"the pairing from {start} is not an ordinary bounce")
        return float(np.squeeze(entry["action"]))

    def test_the_two_pairings_carry_different_actions(self):
        separate = (self._action_of(self.false_vacuum), self._action_of(-self.false_vacuum))
        self.assertGreater(max(separate) / min(separate), 1.2,
                           msg=f"the two pairings are too close to tell apart: {separate}")

    def test_the_action_returned_is_the_smaller_of_the_two(self):
        separate = (self._action_of(self.false_vacuum), self._action_of(-self.false_vacuum))
        outdict: dict = {}
        bounceActionOverImages(1.0, toy_V, toy_dV, outdict, self._tdict(),
                               TWO_HIGGS_DOUBLET, False, 1.0)
        got = float(np.squeeze(outdict[1.0]["action"]))
        self.assertAlmostEqual(got, min(separate), places=6)
        self.assertNotAlmostEqual(
            got, max(separate), places=3,
            msg="the larger of the two actions was returned")

    def test_it_says_how_many_pairings_it_tried_and_resolved(self):
        outdict: dict = {}
        bounceActionOverImages(1.0, toy_V, toy_dV, outdict, self._tdict(),
                               TWO_HIGGS_DOUBLET, False, 1.0)
        self.assertEqual(outdict[1.0]["pairings_tried"], 2)
        self.assertEqual(outdict[1.0]["pairings_resolved"], 2)

    def test_the_minimum_it_kept_is_the_one_whose_action_it_reports(self):
        """The stored `high_vev` has to be the image the action belongs to.

        Everything downstream reads the field values out of the same entry, so an entry
        carrying one pairing's action and another pairing's minima would describe a bounce
        that was never computed.
        """
        outdict: dict = {}
        bounceActionOverImages(1.0, toy_V, toy_dV, outdict, self._tdict(),
                               TWO_HIGGS_DOUBLET, False, 1.0)
        kept = np.asarray(outdict[1.0]["high_vev"], float)
        self.assertAlmostEqual(self._action_of(kept),
                               float(np.squeeze(outdict[1.0]["action"])), places=6)

    def test_the_caller_s_other_keys_are_carried_onto_the_entry(self):
        """The nucleation-temperature search puts `Tnuc` and the phase keys in the
        dictionary it passes and reads them back out of the stored entry, so a copy that
        kept only the two minima would lose them."""
        outdict: dict = {}
        bounceActionOverImages(1.0, toy_V, toy_dV, outdict,
                               self._tdict(Tnuc=1.0, low_phase="bro", high_phase="sym"),
                               TWO_HIGGS_DOUBLET, False, 1.0)
        entry = outdict[1.0]
        self.assertEqual(entry["Tnuc"], 1.0)
        self.assertEqual((entry["low_phase"], entry["high_phase"]), ("bro", "sym"))

    def test_the_caller_s_dictionary_is_not_the_stored_one(self):
        """A dictionary handed in may not come back mutated, or a caller that reuses it for
        a second temperature would carry the first temperature's action into it."""
        tdict = self._tdict()
        outdict: dict = {}
        bounceActionOverImages(1.0, toy_V, toy_dV, outdict, tdict,
                               TWO_HIGGS_DOUBLET, False, 1.0)
        self.assertEqual(sorted(tdict), ["high_vev", "low_vev"],
                         msg=f"the caller's dictionary was written into: {sorted(tdict)}")
        self.assertIsNot(outdict[1.0], tdict)

    def test_an_entry_already_in_the_dictionary_is_not_recomputed(self):
        outdict = {1.0: dict(action=1.0, trantype=1, low_vev=self.true_vacuum,
                             high_vev=self.false_vacuum)}
        bounceActionOverImages(1.0, toy_V, toy_dV, outdict, self._tdict(),
                               TWO_HIGGS_DOUBLET, False, 1.0)
        self.assertEqual(outdict[1.0]["action"], 1.0)
        self.assertNotIn("pairings_tried", outdict[1.0],
                         msg="the cached entry was replaced by a fresh computation")

    def test_a_potential_with_no_symmetry_computes_one_bounce(self):
        outdict: dict = {}
        bounceActionOverImages(1.0, toy_V, toy_dV, outdict, self._tdict(), [], False, 1.0)
        self.assertEqual(outdict[1.0]["pairings_tried"], 1)
        self.assertAlmostEqual(float(np.squeeze(outdict[1.0]["action"])),
                               self._action_of(self.false_vacuum), places=6)


#: Every error class raised anywhere in the two modules that compute a bounce,
#: `pathDeformation` and `tunneling1D`, against what `bounceAction` does with it. The census
#: below pins the keys, so a new class raised in either module fails the test rather than
#: quietly becoming a new way for one pairing to cost a temperature.
#:
#: "absorbed" means the class never leaves `bounceAction`: it comes back as a `trantype`,
#: which is what `bounceActionOverImages` compares. "escapes" means it reaches the caller,
#: which is what the run deadline and a fault in the code have to do.
WHAT_BOUNCEACTION_DOES_WITH_IT = {
    "DeformationError": "absorbed",     # caught in `_deform_path`, the path stays unconverged
    "InfiniteActionError": "absorbed",  # `trantype` 0 with an infinite action
    "PotentialError": "absorbed",       # `trantype` 2 or 0, or re-raised as UnexpectedError
    "SplineError": "absorbed",          # `trantype` 0 with an infinite action
    "TooSmalldVError": "absorbed",      # `_find_profile` makes it an InfiniteActionError
    "TunnelingError": "absorbed",       # `_find_profile` makes it an InfiniteActionError
    "UnexpectedError": "escapes",       # a `PotentialError` whose error type is unknown
    "ZeroActionError": "absorbed",      # caught, and raised nowhere: see the test below
}


class WhatTheSolverRaisesNeverLeavesBounceAction(unittest.TestCase):
    """Every failure the solver raises about a path is absorbed into a `trantype`.

    This is the measurement `bounceActionOverImages` rests on. It offers pairings to the
    solver that would never have been computed before, so a failure on one of them must not
    cost the temperature; it cannot, because `bounceAction` turns each of them into
    `trantype` 0 with an infinite action or `trantype` 2 with action 0 instead of letting it
    out. The branch's first form instead listed five error classes and passed a pairing over
    when it caught one, and all five were absorbed before that list ever saw them, because
    the measurement behind the list was taken on `fullTunneling`, one level below
    `bounceAction`.

    The scope of this class is the error classes, not the ways of producing them. The census
    above is pinned against the two modules' `raise` statements, so a new class fails the
    test; each class's entry is then measured by injecting it where it is really raised, with
    two exceptions, `ZeroActionError`, which has no raise site at all, and the second
    `PotentialError` error type, which no injection can reach separately because
    `_validate_tunneling_path` decides it from the potential. What this class does not cover
    is a failure that is not a `transitionlistener` error at all, a `ValueError` out of
    scipy, say: those reach the caller and end the temperature, which is the behaviour the
    single pairing already had and the price of offering a second one.

    The run deadline is the one failure that still comes out, and it has to.
    """

    low_vev = np.array([1.19615988, 0.17064297])
    high_vev = np.array([-0.20961488, 1.07826633])

    def _outcome_with(self, target, name, failure):
        """Run `bounceAction` with `target.name` replaced by something that raises."""
        original = getattr(target, name)

        def raises(*args, **kwargs):
            raise failure

        setattr(target, name, raises)
        try:
            outdict: dict = {}
            pd.bounceAction(1.0, toy_V, toy_dV, outdict,
                            dict(low_vev=self.low_vev, high_vev=self.high_vev), False, 1.0)
            return outdict[1.0]
        finally:
            setattr(target, name, original)

    def test_the_census_covers_every_class_the_two_modules_raise(self):
        """A class raised on the bounce path with no entry above is unclassified."""
        source = Path(transitionlistener.__file__).parent
        raised = set()
        for name in ("pathDeformation.py", "tunneling1D.py"):
            tree = ast.parse((source / name).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Raise) or node.exc is None:
                    continue
                target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if (isinstance(target, ast.Attribute)
                        and getattr(target.value, "id", "") == "errors"):
                    raised.add(target.attr)
        classified = set(WHAT_BOUNCEACTION_DOES_WITH_IT)
        self.assertEqual(raised - classified, set(),
                         msg=f"raised on the bounce path and unclassified: "
                             f"{sorted(raised - classified)}")
        # The other direction, so an entry cannot outlive its raise site unnoticed. The one
        # name allowed to have none is the one the test below measures to have none.
        self.assertEqual(classified - raised, {"ZeroActionError"},
                         msg=f"classified with no raise site: {sorted(classified - raised)}")

    def test_the_solver_s_failures_come_back_as_a_trantype(self):
        """Each case names the class it stands for, and the census has to agree it is
        absorbed, so the table's values are read here and not only its keys."""
        for name, classified, target, attribute, failure, trantype, action in (
                ("findProfile raises TooSmalldVError", "TooSmalldVError",
                 tunneling1D.SingleFieldInstanton, "findProfile",
                 errors.TooSmalldVError(), 0, np.inf),
                ("findProfile raises TunnelingError", "TunnelingError",
                 tunneling1D.SingleFieldInstanton, "findProfile",
                 errors.TunnelingError("the shooting failed"), 0, np.inf),
                ("the path spline refuses the points", "SplineError", pd, "SplinePath",
                 ValueError("no spline for these points"), 0, np.inf),
                ("the two points have no barrier between them", "PotentialError",
                 tunneling1D.SingleFieldInstanton, "findProfile",
                 errors.PotentialError("identical minima", "no barrier"), 2, 0.0),
                ("the high-temperature minimum is the stable one", "PotentialError",
                 tunneling1D.SingleFieldInstanton, "findProfile",
                 errors.PotentialError("stable", "stable, not metastable"), 0, np.inf)):
            with self.subTest(name):
                self.assertEqual(WHAT_BOUNCEACTION_DOES_WITH_IT[classified], "absorbed",
                                 msg=f"{name}: the census calls {classified} something else")
                entry = self._outcome_with(target, attribute, failure)
                self.assertEqual(entry["trantype"], trantype, msg=f"{name}: trantype")
                self.assertEqual(float(np.squeeze(entry["action"])), action,
                                 msg=f"{name}: action")

    def test_a_deformation_that_diverges_comes_back_as_a_bounce(self):
        """`_deform_path` catches `DeformationError` and keeps the unconverged path."""
        self.assertEqual(WHAT_BOUNCEACTION_DOES_WITH_IT["DeformationError"], "absorbed")
        entry = self._outcome_with(pd.Deformation_Spline, "deformPath",
                                   errors.DeformationError("the deformation diverged"))
        self.assertEqual(entry["trantype"], 1)
        self.assertTrue(np.isfinite(float(np.squeeze(entry["action"]))))

    def test_an_error_type_the_code_does_not_know_comes_out(self):
        """Every class the census marks as escaping, measured rather than read.

        There is one, and an `InfiniteActionError` is what the two error types the code
        does know come back as, so a class moved into "escapes" without a measurement here
        fails the first assertion rather than going unnoticed.
        """
        escaping = {name for name, verdict in WHAT_BOUNCEACTION_DOES_WITH_IT.items()
                    if verdict == "escapes"}
        self.assertEqual(escaping, {"UnexpectedError"},
                         msg=f"{sorted(escaping)} are marked as escaping, and only "
                             "UnexpectedError is measured below")
        with self.assertRaises(errors.UnexpectedError):
            self._outcome_with(tunneling1D.SingleFieldInstanton, "findProfile",
                               errors.PotentialError("something new", "an unknown type"))

    def test_the_run_deadline_comes_out(self):
        """Injected at all three stages, the path spline first.

        Fitting the spline is the first thing `fullTunneling` touches, so it is where the
        deadline is most likely to land, and `_fit_spline_path` has a handler of its own
        that turns a refusal into a `SplineError`. Relabelling the deadline there would
        have it absorbed into `trantype` 0 and read as a pairing that happened not to
        resolve, which is the one reading it may never get.
        """
        for name, target, attribute in (
                ("from the path spline", pd, "SplinePath"),
                ("from the shooting", tunneling1D.SingleFieldInstanton, "findProfile"),
                ("from the deformation", pd.Deformation_Spline, "deformPath")):
            with self.subTest(name):
                with self.assertRaises(errors.Timeout):
                    self._outcome_with(target, attribute, errors.Timeout("the run deadline"))

    def test_the_solver_raises_no_zero_action_error(self):
        """`errors.ZeroActionError` is caught in `_execute_full_tunneling` and raised
        nowhere in the package, so it is not one of the ways a pairing can fail."""
        source = Path(transitionlistener.__file__).parent
        raised = []
        for module in sorted(source.rglob("*.py")):
            tree = ast.parse(module.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Raise) or node.exc is None:
                    continue
                target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if isinstance(target, ast.Attribute) and target.attr == "ZeroActionError":
                    raised.append(f"{module.name}:{node.lineno}")
        self.assertEqual(raised, [], msg=f"ZeroActionError is raised at {raised}")


class OnePairingWithoutAnActionDoesNotCostTheTemperature(unittest.TestCase):
    """The outcomes a potential cannot be made to produce on demand, set by hand.

    These replace `bounceAction` with a stub, because what is under test is which pairing is
    kept when the pairings come back with different outcomes, and no potential can be asked
    to give its second pairing a chosen action while its first gives another. The stub
    speaks only the vocabulary the real `bounceAction` has, an `(action, trantype)` pair,
    which `WhatTheSolverRaisesNeverLeavesBounceAction` measures to be the whole of it. The
    same contract is driven through the real solver, with the real failure injected inside
    it, in `TheNearestPairingCanFailInsideTheSolver`.
    """

    def setUp(self):
        self.false_vacuum = np.array([-0.2, 1.1])
        self.true_vacuum = np.array([1.2, 0.17])
        #: The images in the order the enumeration produces them, nearest first.
        self.images = imagesOfTheFalseVacuum(self.false_vacuum, self.true_vacuum,
                                             TWO_HIGGS_DOUBLET)
        self.assertEqual(len(self.images), 2)

    def _run(self, outcomes, outdict=None):
        """Drive `bounceActionOverImages` with `outcomes` keyed by the image, nearest first.

        Each outcome is either an `(action, trantype)` pair to store or an exception to
        raise. Nothing in `bounceActionOverImages` catches, so an exception here is a test
        of that and not of a pairing that failed.
        """
        calls = []

        def fake_bounceAction(T, V, dV, out, tdict, verbose=False, conversionFactor=1.0,
                              **params):
            index = next(i for i, image in enumerate(self.images)
                         if np.allclose(image, tdict["high_vev"]))
            calls.append(index)
            outcome = outcomes[index]
            if isinstance(outcome, BaseException):
                raise outcome
            tdict["action"], tdict["trantype"] = outcome
            tdict["instanton"] = None
            out[T] = tdict
            return out

        original = pd.bounceAction
        pd.bounceAction = fake_bounceAction
        try:
            result = {} if outdict is None else outdict
            pd.bounceActionOverImages(
                1.0, toy_V, toy_dV, result,
                dict(low_vev=self.true_vacuum, high_vev=self.false_vacuum),
                TWO_HIGGS_DOUBLET, False, 1.0)
            return result, calls
        finally:
            pd.bounceAction = original

    def test_the_smaller_action_wins_when_it_is_not_the_first_tried(self):
        """The case the real solver was never seen to produce, set by hand.

        Returning the first pairing tried, or the last, agrees with the smallest action in
        every case measured, so only this test tells the three apart.
        """
        result, calls = self._run([(200.0, 1), (50.0, 1)])
        self.assertEqual(calls, [0, 1], msg="not every image was offered to the solver")
        self.assertEqual(result[1.0]["action"], 50.0)
        np.testing.assert_allclose(result[1.0]["high_vev"], self.images[1])

    def test_the_smaller_action_wins_when_it_is_the_first_tried(self):
        result, _ = self._run([(50.0, 1), (200.0, 1)])
        self.assertEqual(result[1.0]["action"], 50.0)
        np.testing.assert_allclose(result[1.0]["high_vev"], self.images[0])

    def test_a_pairing_without_an_action_does_not_win_against_one_with_an_action(self):
        """The 2HDM at `lambda3 = 5.8676` does this at its coldest temperature: one pairing
        has no action while the other gives `S_3/T = 399.97`.

        A pairing whose bounce the solver could not resolve comes back as `trantype` 0 with
        an infinite action, and it arrives in whichever order the geometry puts it. Keeping
        it because it was tried first would cost the temperature exactly where the branch
        says it does not, and the cold end is where percolation happens.
        """
        for name, outcomes, kept in (
                ("no action on the pairing tried second", [(400.0, 1), (np.inf, 0)], 0),
                ("no action on the pairing tried first", [(np.inf, 0), (400.0, 1)], 1)):
            with self.subTest(name):
                result, calls = self._run(outcomes)
                self.assertEqual(calls, [0, 1])
                self.assertEqual(result[1.0]["action"], 400.0, msg=name)
                self.assertEqual(result[1.0]["trantype"], 1, msg=name)
                np.testing.assert_allclose(result[1.0]["high_vev"], self.images[kept])
                self.assertEqual(result[1.0]["pairings_tried"], 2)
                self.assertEqual(result[1.0]["pairings_resolved"], 1,
                                 msg=f"{name}: a pairing with no action was counted as "
                                     "resolved")

    def test_the_first_entry_is_kept_when_no_pairing_has_an_action(self):
        """The caller then sees the infinite action it would have seen before the images
        were tried, and `pairings_resolved` says that none of them resolved."""
        result, _ = self._run([(np.inf, 0), (np.inf, 0)])
        self.assertEqual(result[1.0]["action"], np.inf)
        self.assertEqual(result[1.0]["trantype"], 0)
        self.assertEqual(result[1.0]["pairings_resolved"], 0)
        np.testing.assert_allclose(result[1.0]["high_vev"], self.images[0])

    def test_a_run_deadline_is_not_passed_over(self):
        """`errors.Timeout` is raised from the run deadline's alarm signal and says nothing
        about the pairing, so swallowing it would turn the end of the run's time budget into
        a pairing that happened not to resolve."""
        with self.assertRaises(errors.Timeout):
            self._run([(75.0, 1), errors.Timeout("the run deadline")])

    def test_a_fault_in_the_code_is_not_passed_over(self):
        """`errors.UnexpectedError` reports a fault in the code rather than in the pairing.

        `bounceAction` raises it where a `PotentialError` carries an error type it does not
        know, and a pairing is not the thing at fault.
        """
        with self.assertRaises(errors.UnexpectedError):
            self._run([(75.0, 1), errors.UnexpectedError("a fault")])

    def test_a_failed_pairing_leaves_no_entry_behind(self):
        """The caller's dictionary is written once, at the end, so a pairing that raises
        cannot leave a half-filled entry for the next reader to find."""
        outdict: dict = {}
        with self.assertRaises(errors.Timeout):
            self._run([(75.0, 1), errors.Timeout("the run deadline")], outdict=outdict)
        self.assertEqual(outdict, {}, msg=f"a partial entry was left behind: {outdict}")

    def test_two_points_that_coincide_are_not_overruled(self):
        """`bounceAction` reports two points it was handed that coincide as `trantype` 2
        with action 0, which says the false vacuum is an image of the true one and there is
        no transition to compute. Only the nearest image can produce it, since an image
        equal to the true vacuum is at distance zero and so is ordered first, and action 0
        is smaller than any bounce's, so it is kept.
        """
        result, _ = self._run([(0.0, 2), (140.0, 1)])
        self.assertEqual(result[1.0]["action"], 0.0)
        self.assertEqual(result[1.0]["trantype"], 2)
        self.assertEqual(result[1.0]["pairings_resolved"], 1,
                         msg="a pair of coinciding points was counted as a resolved bounce")

    def test_an_action_that_is_not_finite_does_not_win(self):
        """Neither a `nan` nor a negative infinity may displace a resolved action.

        A `nan` fails the comparison on its own, since every comparison against a `nan` is
        false. A negative infinity passes it, and is smaller than every action there is, so
        it would win every pairing it was offered in. The Euclidean action of a bounce is
        positive, and `bounceAction` refuses one that is not, so neither should arrive here;
        the test is what makes the second line of that refusal a claim rather than a hope.
        """
        for name, bad in (("not a number", float("nan")),
                          ("negative infinity", float("-inf"))):
            with self.subTest(name):
                result, _ = self._run([(85.0, 1), (bad, 1)])
                self.assertEqual(result[1.0]["action"], 85.0,
                                 msg=f"{name} displaced a resolved action")
                np.testing.assert_allclose(result[1.0]["high_vev"], self.images[0])

    def test_an_action_that_is_not_finite_is_displaced_when_it_comes_first(self):
        """A `nan` incumbent must not survive the comparison either.

        `action < best` is false against a `nan` best, so writing the comparison that way
        would keep the first pairing for ever once its action was not a number.
        """
        for name, bad in (("not a number", float("nan")),
                          ("negative infinity", float("-inf"))):
            with self.subTest(name):
                result, _ = self._run([(bad, 1), (85.0, 1)])
                self.assertEqual(result[1.0]["action"], 85.0,
                                 msg=f"{name} was kept against a resolved action")
                np.testing.assert_allclose(result[1.0]["high_vev"], self.images[1])


class TheNearestPairingCanFailInsideTheSolver(unittest.TestCase):
    """The same contract as above, through the real solver with a real failure inside it.

    The stub in `OnePairingWithoutAnActionDoesNotCostTheTemperature` sets the two outcomes
    by hand. Here the path spline is refused for the nearest image only, which is how a
    pairing really fails: `_fit_spline_path` turns the refusal into a `SplineError` and
    `_execute_full_tunneling` turns that into `trantype` 0 with an infinite action, so the
    failure arrives at the comparison by the route it takes in a run.
    """

    def setUp(self):
        self.false_vacuum = toy_minimum([-0.2, 1.1])
        self.true_vacuum = toy_minimum([1.2, 0.17])
        self.images = imagesOfTheFalseVacuum(self.false_vacuum, self.true_vacuum,
                                             TWO_HIGGS_DOUBLET)
        self.assertEqual(len(self.images), 2)
        # The nearest image is not the traced point here, so refusing it refuses the pairing
        # that is tried first.
        np.testing.assert_allclose(self.images[0], -self.false_vacuum)

    def test_the_other_pairing_answers_for_the_temperature(self):
        real_spline, refused = pd.SplinePath, self.images[0]

        def refuse_the_nearest_image(pts, *args, **kwargs):
            if np.allclose(np.asarray(pts, float)[-1], refused, rtol=0.0, atol=1e-6):
                raise ValueError("injected: no spline for this path")
            return real_spline(pts, *args, **kwargs)

        pd.SplinePath = refuse_the_nearest_image
        try:
            outdict: dict = {}
            pd.bounceActionOverImages(
                1.0, toy_V, toy_dV, outdict,
                dict(low_vev=self.true_vacuum, high_vev=self.false_vacuum),
                TWO_HIGGS_DOUBLET, False, 1.0)
        finally:
            pd.SplinePath = real_spline
        entry = outdict[1.0]
        self.assertEqual(entry["trantype"], 1,
                         msg="the refused pairing's verdict was kept")
        self.assertAlmostEqual(float(np.squeeze(entry["action"])), 112.934, places=2)
        np.testing.assert_allclose(np.asarray(entry["high_vev"], float),
                                   self.false_vacuum)
        self.assertEqual(entry["pairings_tried"], 2)
        self.assertEqual(entry["pairings_resolved"], 1)


class TheNucleationTemperatureSearchPairsOnOneImageToo(unittest.TestCase):
    """`nucleation.evalNucleationCriterion` reads the same two traced phases.

    It therefore inherits the same mismatched pairing, and it runs before percolation: on
    the fixed step size route a criterion that is never fulfilled raises `NucleationError`
    and the candidate transition is dropped, so the percolation fix would never be reached.
    The toy's two pairings give 112.934 and 71.152 at `T = 1`, and the criterion
    `140 - S/T` gives 27.07 on the first and 68.85 on the second.
    """

    def setUp(self):
        self.false_vacuum = toy_minimum([-0.2, 1.1])
        self.true_vacuum = toy_minimum([1.2, 0.17])

    def _criterion(self, group):
        outdict: dict = {}
        args = (_ConstantPhase("sym", self.false_vacuum),
                _ConstantPhase("bro", self.true_vacuum),
                toy_V_point, toy_dV_point, toy_d2V_point, 1.0e-8, 45.0,
                lambda S, T, *_: 140.0 - S / (T + 1.0e-100), {}, False, outdict, 1.0,
                group)
        return nucleation.evalNucleationCriterion(1.0, *args), outdict[1.0]

    def test_the_criterion_is_built_from_the_cheapest_pairing(self):
        criterion, entry = self._criterion(TWO_HIGGS_DOUBLET)
        self.assertAlmostEqual(float(np.squeeze(entry["action"])), 71.152, places=2)
        self.assertAlmostEqual(float(criterion), 68.85, places=2)
        np.testing.assert_allclose(np.asarray(entry["high_vev"], float),
                                   -self.false_vacuum)

    def test_the_traced_pairing_is_what_it_gives_without_a_group(self):
        """The same call with no symmetry, which is the number the search returned before:
        a criterion lower by 41.8, from an action larger by a factor 1.59."""
        criterion, entry = self._criterion([])
        self.assertAlmostEqual(float(np.squeeze(entry["action"])), 112.934, places=2)
        self.assertAlmostEqual(float(criterion), 27.07, places=2)

    def test_the_cache_it_leaves_behind_is_what_percolation_starts_from(self):
        """`transitionObservables` hands `tr.full_tunneling_info`, which is this cache, to
        the percolation walk as its `outdict` whenever the search found a `Tnuc`.

        `calcAction` returns a cached entry as it stands, so an entry the search wrote on
        the expensive pairing would be handed straight back and the percolation fix would
        never be reached at that temperature.
        """
        _, entry = self._criterion(TWO_HIGGS_DOUBLET)
        seeded = {1.0: {key: value for key, value in entry.items() if key != "instanton"}}
        pot = SimpleNamespace(
            Vtot=toy_V, gradV=toy_dV_point, d2V=toy_d2V_point, conversionFactor=1.0,
            invGroupElements=TWO_HIGGS_DOUBLET,
            config=SimpleNamespace(tracingConf=SimpleNamespace(tunneling_params={})))
        action = bd.calcAction(pot, 1.0, _ConstantPhase("sym", self.false_vacuum),
                               _ConstantPhase("bro", self.true_vacuum), seeded)
        self.assertAlmostEqual(float(np.squeeze(action)), 71.152, places=2)

    def test_the_phase_keys_and_nucleation_temperature_survive(self):
        """`computeNucleationTemperature` reads them off the stored entry."""
        _, entry = self._criterion(TWO_HIGGS_DOUBLET)
        self.assertEqual(entry["Tnuc"], 1.0)
        self.assertEqual((entry["low_phase"], entry["high_phase"]), ("bro", "sym"))

    def test_the_search_is_handed_the_potential_s_own_group(self):
        """`transitions.py` is the only caller, and it has to pass the group along or the
        keyword's default leaves the search on one pairing.

        What this pins is that the value comes from the potential, not how it is spelled. It
        was written against `getattr(self.pot, "invGroupElements", [])` and failed the moment
        that became `imagesToPairOver(self.pot)`, which reads the same attribute through the
        switch; the defect it exists for is a literal `[]` or a missing keyword, and both are
        still refused.
        """
        source = Path(transitionlistener.__file__).parent / "transitions.py"
        text = source.read_text()
        tree = ast.parse(text)
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "computeNucleationTemperature"]
        self.assertEqual(len(calls), 2, msg=f"{len(calls)} call sites, expected 2")
        for call in calls:
            passed = {keyword.arg: keyword.value for keyword in call.keywords}
            self.assertIn("invGroupElements", passed,
                          msg=f"the call at line {call.lineno} does not pass the group")
            value = passed["invGroupElements"]
            self.assertNotIsInstance(
                value, (ast.List, ast.Tuple, ast.Constant),
                msg=f"the call at line {call.lineno} passes a literal, which leaves the "
                    f"search on one pairing")
            expression = ast.get_source_segment(text, value) or ""
            self.assertIn(
                "pot", expression,
                msg=f"the call at line {call.lineno} takes the group from something other "
                    f"than the potential: {expression!r}")

    def test_the_switch_decides_whether_the_group_is_offered_at_all(self):
        """`imagesToPairOver` is the only thing the four call sites consult.

        Off, the solver is handed nothing and computes one bounce per temperature. On, it is
        handed the potential's own group. The setting is read from the potential's config, so a
        run can turn it on without touching the code.
        """
        group = [-np.identity(2), np.identity(2)]

        class Conf:
            pair_transition_over_images = False

        class Config:
            tracingConf = Conf()

        class Pot:
            invGroupElements = group
            config = Config()

        pot = Pot()
        self.assertEqual(
            imagesToPairOver(pot), [],
            msg="with the switch off the group was still offered, which costs a bounce per "
                "image at every temperature")
        Conf.pair_transition_over_images = True
        offered = imagesToPairOver(pot)
        self.assertEqual(len(offered), len(group),
                         msg=f"with the switch on {len(offered)} of {len(group)} elements "
                             "were offered")
        for given, got in zip(group, offered):
            np.testing.assert_array_equal(got, given)

    def test_the_shipped_default_is_off(self):
        """Which is a decision, not an accident, so it is pinned here.

        The bounce used to be computed between two images because the phase tracing could put
        the two phases on different ones, and that is now repaired in `phases.py` where it
        arises. Measured on commit 065ef81 with `models/TL_2HDM.py` class `R2HDM` at shipped settings, with the benchmark parameters of `examples/bp_lisa.yaml` at `lambda3 = 5.868`,
        the coupling the pairing was added for: at none of the 66 temperatures whose two images
        are genuinely apart did the image the tracing did not produce carry the smaller action,
        while the second bounce took 66 per cent of the time spent on bounces.

        If this is ever flipped back on, the thing to check first is whether
        `test_a_phase_stays_on_one_image.py` still holds, because that is what makes the
        pairing unnecessary.
        """
        self.assertFalse(
            config.TracingConf.pair_transition_over_images,
            msg="the pairing is on by default, which doubles the bounce count on every model "
                "with a symmetry group for no measured change in any result")

    def test_a_potential_without_a_config_is_not_paired_over(self):
        """A potential that carries no `tracingConf` gets the shipped default, not the opposite.

        The attribute is read with a fallback, and a fallback that disagreed with the shipped
        default would make the behaviour depend on how complete the config object is.
        """
        class Pot:
            invGroupElements = [-np.identity(2), np.identity(2)]

        self.assertEqual(
            imagesToPairOver(Pot()), [],
            msg="a potential with no config was paired over, so the fallback disagrees with "
                "the shipped default")

    def test_the_other_two_copies_of_the_search_are_reached_from_nowhere(self):
        """Why only `nucleation.py`'s copy was changed.

        There are three copies of the nucleation criterion. `nucleation.py`'s is the live
        one, imported by `transitions.py` for both backends, and it is the one that now
        enumerates the images. `bubbledynamics_fixedstep`'s `evalNucleationCriterion`,
        `computeNucleationTemperature` and `computeNucleationTemperature2` are a legacy copy,
        and `Transitions.evalNucleationCriterion` is a method with no caller.

        The scope of this sweep is the name: it collects every reference in `src/` to any of
        the three names and allows only the ones that are a copy referring to itself, or
        `computeNucleationTemperature` imported from `nucleation`. It therefore catches a
        call through any binding, including an instance attribute, and it does not catch a
        call assembled from a string through `getattr`, of which there are none in `src/`.
        It fails if either unchanged copy acquires a caller, which is the signal to
        enumerate the images there too rather than to waive it.
        """
        source = Path(transitionlistener.__file__).parent
        island = {"evalNucleationCriterion", "computeNucleationTemperature",
                  "computeNucleationTemperature2"}
        own = {"nucleation.py", "bubbledynamics_fixedstep.py"}
        reached, dynamic = [], []
        for module in sorted(source.rglob("*.py")):
            tree = ast.parse(module.read_text())
            for node in ast.walk(tree):
                where = f"{module.name}:{getattr(node, 'lineno', 0)}"
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "id", None) == "getattr"
                        and len(node.args) > 1
                        and isinstance(node.args[1], ast.Constant)
                        and node.args[1].value in island):
                    dynamic.append(where)
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name not in island:
                            continue
                        allowed = (node.module == "transitionlistener.nucleation"
                                   and alias.name == "computeNucleationTemperature")
                        if not allowed:
                            reached.append(f"{where}:import {alias.name}")
                    continue
                name = (node.attr if isinstance(node, ast.Attribute)
                        else node.id if isinstance(node, ast.Name) else None)
                if name not in island:
                    continue
                if isinstance(node, ast.Name) and module.name in own:
                    continue          # a copy calling itself
                if (isinstance(node, ast.Attribute)
                        and getattr(node.value, "id", None) == "nucleation"):
                    continue          # the live copy, reached through its module
                if (isinstance(node, ast.Name)
                        and name == "computeNucleationTemperature"):
                    continue          # the live copy, imported by `transitions.py`
                reached.append(f"{where}:{name}")
        self.assertEqual(reached, [],
                         msg=f"an unchanged copy of the search is reached at {reached}")
        self.assertEqual(dynamic, [],
                         msg=f"a copy is reached through getattr at {dynamic}, which the "
                             "sweep above cannot follow")


class BothPercolationRoutesReachTheEnumeration(unittest.TestCase):
    """`calcAction` is what a run calls, and both copies of it have to carry the group.

    A test that calls `bounceActionOverImages` directly passes while the entry point a run
    takes is unprotected. There are two copies of `calcAction`, one per percolation route,
    and the fixed step size module has its own `findLocalMinimum` as well, so the check is
    parametrised over both modules rather than written against one.
    """

    def setUp(self):
        self.false_vacuum = toy_minimum([-0.2, 1.1])
        self.true_vacuum = toy_minimum([1.2, 0.17])

    def _potential(self, group, paired=True):
        """The switch is set explicitly, because these tests are about the plumbing.

        What they check is that the group reaches the enumeration through the entry point a run
        takes, which is only visible with the pairing on; the shipped default is off and is
        pinned separately.
        """
        return SimpleNamespace(
            Vtot=toy_V, gradV=toy_dV_point, d2V=toy_d2V_point, conversionFactor=1.0,
            invGroupElements=group,
            config=SimpleNamespace(tracingConf=SimpleNamespace(
                tunneling_params={}, pair_transition_over_images=paired)))

    def test_both_copies_hand_the_potential_s_group_to_the_enumeration(self):
        for module in (bd, bdf):
            with self.subTest(module.__name__):
                seen = {}

                def spy(T, V, dV, outdict, tdict, invGroupElements=(), *args, **kwargs):
                    seen["group"] = invGroupElements
                    seen["tdict"] = dict(tdict)
                    outdict[T] = dict(tdict, action=1.0, trantype=1)
                    return outdict

                original = module.bounceActionOverImages
                module.bounceActionOverImages = spy
                try:
                    module.calcAction(self._potential(TWO_HIGGS_DOUBLET), 1.0,
                                      _ConstantPhase("sym", self.false_vacuum),
                                      _ConstantPhase("bro", self.true_vacuum), {})
                finally:
                    module.bounceActionOverImages = original
                self.assertEqual(len(seen["group"]), 2,
                                 msg=f"{module.__name__}: the group did not arrive")
                for given, got in zip(TWO_HIGGS_DOUBLET, seen["group"]):
                    np.testing.assert_allclose(got, given)
                self.assertEqual(sorted(seen["tdict"]), ["high_vev", "low_vev"])
                np.testing.assert_allclose(seen["tdict"]["high_vev"], self.false_vacuum,
                                           atol=1e-6)
                np.testing.assert_allclose(seen["tdict"]["low_vev"], self.true_vacuum,
                                           atol=1e-6)

    def test_both_copies_survive_a_potential_with_no_group_attribute(self):
        """Every model built on `generic_potential` has one, and a hand-written stub or an
        older pickled potential may not."""
        for module in (bd, bdf):
            with self.subTest(module.__name__):
                pot = self._potential(TWO_HIGGS_DOUBLET)
                del pot.invGroupElements
                action = module.calcAction(pot, 1.0,
                                           _ConstantPhase("sym", self.false_vacuum),
                                           _ConstantPhase("bro", self.true_vacuum), {})
                self.assertAlmostEqual(float(np.squeeze(action)), 112.934, places=2)

    def test_the_cheapest_pairing_is_what_the_entry_point_returns(self):
        """Through `calcAction`, not through `bounceActionOverImages`: 71.152 and not
        112.934, which is the action of the pairing the phases were traced onto."""
        for module in (bd, bdf):
            with self.subTest(module.__name__):
                action = module.calcAction(self._potential(TWO_HIGGS_DOUBLET), 1.0,
                                           _ConstantPhase("sym", self.false_vacuum),
                                           _ConstantPhase("bro", self.true_vacuum), {})
                self.assertAlmostEqual(float(np.squeeze(action)), 71.152, places=2)

    def test_the_switch_off_leaves_the_action_of_the_traced_pairing(self):
        """The end-to-end effect of the switch, through the entry point a run takes.

        Off, `calcAction` returns the action of the pairing the phases were traced onto, which
        on this toy potential is the more expensive one, 112.934 against 71.152. That is the
        behaviour from before the pairing existed, and it is what the shipped default now gives.
        A toy potential is enough to show the switch acts; which pairing is right on a real model
        is settled by the phase tracing, and `test_a_phase_stays_on_one_image.py` is what holds
        that.
        """
        for module in (bd, bdf):
            with self.subTest(module.__name__):
                action = module.calcAction(
                    self._potential(TWO_HIGGS_DOUBLET, paired=False), 1.0,
                    _ConstantPhase("sym", self.false_vacuum),
                    _ConstantPhase("bro", self.true_vacuum), {})
                self.assertAlmostEqual(
                    float(np.squeeze(action)), 112.934, places=2,
                    msg=f"{module.__name__}: with the switch off the action is not the traced "
                        "pairing's, so the switch does not reach this route")


class TheCacheEntryIsRecognisedAfterATransformedImageWins(unittest.TestCase):
    """A cached action is reused only if its entry can be tied to the two traced phases.

    `bounceAction` stores the `tdict` it is handed, and `bounceActionOverImages` hands it the
    winning image, so the entry's `high_vev` is that image and not the point the phase tracing
    produced. `bubbledynamics._transition_action_outdict_temperatures` compares the two with a
    tolerance of `max(1e-6, 2e-2 * scale)`, which an image far from the traced point cannot
    pass, and the temperature would then be dropped and its action computed again.
    """

    def setUp(self):
        self.false_vacuum = toy_minimum([-0.95, -0.12])
        self.true_vacuum = toy_minimum([1.2, 0.17])

    def test_the_entry_carries_the_false_vacuum_as_traced(self):
        outdict = {}
        bounceActionOverImages(
            1.0, toy_V, toy_dV, outdict,
            dict(low_vev=self.true_vacuum, high_vev=self.false_vacuum),
            [np.diag([-1.0, -1.0]), np.diag([1.0, 1.0])], False, 1.0)
        entry = outdict[1.0]
        self.assertIn("high_vev_as_traced", entry,
                      msg="the entry does not say which false vacuum it was asked about")
        self.assertTrue(np.allclose(entry["high_vev_as_traced"], self.false_vacuum),
                        msg="the as-traced false vacuum is not the one that was handed in")

    def _recognised(self, entry):
        symmetric = SimpleNamespace(valAt=lambda T, deriv=0: self.false_vacuum)
        broken = SimpleNamespace(valAt=lambda T, deriv=0: self.true_vacuum)
        got = bd._transition_action_outdict_temperatures(
            {1.0: entry}, symmetric, broken, tmin=0.5, tmax=1.5)
        return list(np.asarray(got, dtype=float))

    def test_a_transformed_entry_is_still_recognised(self):
        mirrored = -np.asarray(self.false_vacuum, dtype=float)
        separation = float(np.linalg.norm(mirrored - self.false_vacuum))
        self.assertGreater(
            separation,
            max(1.0e-6, 2.0e-2 * float(np.linalg.norm(self.false_vacuum))),
            msg="the mirror image is inside the matcher's tolerance of the traced point, so "
                "this test cannot see the defect")
        self.assertEqual(
            self._recognised({"action": 150.0, "low_vev": self.true_vacuum,
                              "high_vev": mirrored,
                              "high_vev_as_traced": self.false_vacuum}),
            [1.0],
            msg="the cached action was not recognised as this transition's, so it would be "
                "computed again")

    def test_an_entry_belonging_to_neither_phase_is_still_refused(self):
        """The relaxation may not make the matcher accept anything."""
        elsewhere = np.asarray(self.false_vacuum, dtype=float) + 9.0
        self.assertEqual(
            self._recognised({"action": 150.0, "low_vev": self.true_vacuum,
                              "high_vev": elsewhere,
                              "high_vev_as_traced": elsewhere}),
            [],
            msg="an entry whose false vacuum is neither the traced point nor an image of it "
                "was accepted")


class TheBounceMarkerFollowsTheImageTheBounceUsedTests(unittest.TestCase):
    """The profile plot draws the bounce beside markers for its two endpoints.

    `bounceActionOverImages` may compute the bounce on a transformed image of the false vacuum,
    and the marker then has to follow it. Evaluating the traced phase alone puts the marker a
    reflection away from the curve it labels, which on a model whose fields are hundreds of GeV
    is the whole width of the plot.

    This cannot be reached on the shipped benchmark: with the pairing on, the image the tracing
    produced carried the smaller action at all 66 of the temperatures whose images are
    genuinely apart, so no transformed image ever wins there. The test therefore builds the
    situation directly rather than running a model that cannot produce it.
    """

    @staticmethod
    def _plotter(group):
        from transitionlistener import plots

        plotter = plots.TLPlots.__new__(plots.TLPlots)
        plotter.pot = SimpleNamespace(invGroupElements=group, conversionFactor=1.0)
        plotter.phases = {"hi": SimpleNamespace(valAt=lambda T: np.array([246.0, 11.5]))}
        return plotter

    @staticmethod
    def _transition(high_vev, traced):
        return SimpleNamespace(high_phase="hi", high_vev=np.asarray(high_vev, float),
                               high_vev_as_traced=(None if traced is None
                                                   else np.asarray(traced, float)))

    def test_an_untransformed_transition_keeps_the_traced_point(self):
        """`None` means the bounce used the point the tracing produced, so nothing is mapped."""
        plotter = self._plotter(TWO_HIGGS_DOUBLET)
        got = plotter._bounce_false_vacuum_at_temperature(
            self._transition([246.0, 11.5], None), 30.0)
        np.testing.assert_allclose(got, [246.0, 11.5])

    def test_a_transformed_image_moves_the_marker_onto_it(self):
        """The marker is the phase point mapped by the element the bounce used."""
        plotter = self._plotter(TWO_HIGGS_DOUBLET)
        # The bounce was computed on the flipped image of a false vacuum at (1.0, 0.5).
        tr = self._transition(high_vev=[-1.0, -0.5], traced=[1.0, 0.5])
        got = plotter._bounce_false_vacuum_at_temperature(tr, 30.0)
        np.testing.assert_allclose(
            got, [-246.0, -11.5],
            err_msg="the marker stayed on the traced image while the bounce used its mirror, "
                    "so the curve starts a reflection away from the point labelling it")

    def test_the_marker_follows_the_image_even_when_the_phase_lookup_fails(self):
        """`valAt` fails where the plotting temperature is outside the traced interval.

        `_phase_point_at_temperature` then returns the fallback it was handed, untouched. Hand
        it the image the bounce used and the element below transforms it a second time, and
        since these elements are involutions the marker lands back on the traced image, which
        is the one place this helper exists to keep it off.
        """
        plotter = self._plotter(TWO_HIGGS_DOUBLET)
        plotter.phases = {}  # every lookup raises, so the fallback is what comes back
        tr = self._transition(high_vev=[-1.0, -0.5], traced=[1.0, 0.5])
        got = plotter._bounce_false_vacuum_at_temperature(tr, 30.0)
        np.testing.assert_allclose(
            got, [-1.0, -0.5],
            err_msg="the fallback was transformed twice and the marker came back on the "
                    "traced image")

    def test_an_element_that_only_matches_through_a_relative_tolerance_is_not_applied(self):
        """`np.allclose` adds its default rtol of 1e-5 of the point unless told otherwise.

        On a 246 GeV field that is 2.5e-3 GeV, a thousand times the absolute tolerance this
        match is supposed to use, so an element that merely lands near the right one passes.
        The winning image is `x_false @ g` for an element `g` of the group, computed and stored
        by `bounceActionOverImages`, so the only discrepancy a real match has to absorb is
        floating-point round-off and 1e-8 of the field scale is already generous.

        The group here holds one element, a flip whose first component is wrong by one part in
        a million. It reproduces the map to 2.5e-4 GeV, inside the relative tolerance and
        outside the absolute one.
        """
        near_flip = np.diag([-1.0 + 1.0e-6, -1.0])
        plotter = self._plotter([near_flip])
        tr = self._transition(high_vev=[-1.0, -0.5], traced=[1.0, 0.5])
        got = plotter._bounce_false_vacuum_at_temperature(tr, 30.0)
        np.testing.assert_allclose(
            got, [246.0, 11.5],
            err_msg="an element that only matches through the relative tolerance was applied, "
                    "and it is applied at the plotting temperature, where it need not agree "
                    "with the right one at all")

    def test_two_elements_that_both_match_leave_the_point_alone(self):
        """A tie is no more usable than no match: they need not agree away from this point.

        Both elements of the 2HDM group map a point at the origin to itself, so at a false
        vacuum that small either one reproduces the map. Applying the first found would move
        the marker by the whole width of the plot on the strength of a coin flip. The group is
        given flip-first so that taking the first match is distinguishable from refusing.
        """
        plotter = self._plotter([FLIP_BOTH, IDENTITY])
        tr = self._transition(high_vev=[-1.0e-12, -1.0e-12], traced=[1.0e-12, 1.0e-12])
        got = plotter._bounce_false_vacuum_at_temperature(tr, 30.0)
        np.testing.assert_allclose(
            got, [246.0, 11.5],
            err_msg="one of two elements that both reproduce the map was applied anyway")

    def test_a_map_the_group_does_not_contain_leaves_the_point_alone(self):
        """Better a marker that is wrong as before than a transformation invented here.

        The group here is the flip of both components, and the pair asks for a flip of the
        first alone, which the group does not contain. The group must therefore hold an element
        that would visibly move the point if it were applied without being matched: with only
        the identity in it this test passes whether the match is checked or not.
        """
        plotter = self._plotter([FLIP_BOTH])
        tr = self._transition(high_vev=[-1.0, 0.5], traced=[1.0, 0.5])
        got = plotter._bounce_false_vacuum_at_temperature(tr, 30.0)
        np.testing.assert_allclose(
            got, [246.0, 11.5],
            err_msg="an element that does not reproduce the map was applied anyway, so the "
                    "marker moved on the strength of a transformation nothing asked for")


if __name__ == "__main__":
    unittest.main()

