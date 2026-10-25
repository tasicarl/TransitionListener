"""The chain's last minimum against the potential's zero-temperature one.

`check_T0_global_min` asks whether the transition chain ended in the zero-temperature
minimum. It compared `phase.valAt(Tmin)` against `pot.X0` with `np.allclose`, which asks a
different question: `valAt` interpolates the tracer's nodes and carries its resolution, and
a per-component relative test gives a small field component a far tighter tolerance than a
large one while the tracer's error is absolute. On the 2HDM at lambda3 = 5.8672 the two
agree to 2.7e-06 in norm and `V` agrees to eight digits, and the test failed on the smaller
component alone by a factor 1.04.

The check now compares the depth of the potential at the two points, which is the physical
question, and keeps a field-space test measured against the field scale so that a pair of
degenerate minima is still reported. It does not move the endpoint before comparing: two
labels for the same minimum have the same depth whatever their coordinates differ by, so
there is nothing a minimisation would add.

Part of TransitionListener v2
"""
import types
import unittest

import numpy as np

from transitionlistener import transitionObservables as TO


SCALE = 246.0


#: How close to the second minimum a field point has to be for the stub to use its basin.
#: Smaller than the separations under test, or `X0` falls inside the other minimum's basin
#: and the two sides of the comparison get the same formula, which hides any depth
#: difference and makes the fixture pass for the wrong reason.
_BASIN = 0.05


def _potential(other_minimum=None, depth_offset=0.0):
    """A stub with a minimum at `X0` and optionally a second one elsewhere."""
    X0 = np.array([11.5, SCALE])

    def Vtot(X, T, *a, **k):
        X = np.atleast_1d(np.squeeze(np.asarray(X, dtype=float)))
        # quadratic about X0, plus an offset for anything near the other minimum
        v = 0.5 * float(np.sum((X - X0) ** 2)) - 4.5e7
        if other_minimum is not None and np.linalg.norm(X - other_minimum) < _BASIN:
            v = 0.5 * float(np.sum((X - other_minimum) ** 2)) - 4.5e7 + depth_offset
        return v

    def gradV(X, T, *a, **k):
        X = np.atleast_1d(np.squeeze(np.asarray(X, dtype=float)))
        target = X0
        if other_minimum is not None and np.linalg.norm(X - other_minimum) < _BASIN:
            target = other_minimum
        return X - target

    def d2V(X, T, *a, **k):
        return np.identity(2)

    return types.SimpleNamespace(Tmin=0.0, X0=X0, Vtot=Vtot, gradV=gradV, d2V=d2V,
                                 conversionFactor=1.0)


class _Phase:
    """The only thing `check_T0_global_min` asks a phase for."""

    def __init__(self, vev):
        self._vev = np.asarray(vev, dtype=float)

    def valAt(self, T, deriv=0):
        return self._vev


def _flagged(pot, last_vev):
    """Call the real `check_T0_global_min` and report whether it set the warning.

    Driving the method rather than repeating its arithmetic: a helper that recomputes the
    decision passes whatever the production code does, including nothing at all, and would
    not notice the comparison being dropped altogether.
    """
    transition = types.SimpleNamespace(type=1, low_phase="last", derived_params={})
    observables = {}
    checker = TO.TransitionObservables.__new__(TO.TransitionObservables)
    checker.pot = pot
    checker.phases = {"last": _Phase(last_vev)}
    checker.transitions = [transition]
    checker.transitionObservables = [observables]
    checker.GWconfig = types.SimpleNamespace(check_if_T0_global_min=False)
    checker.verbose = False
    checker.check_T0_global_min()
    return bool(transition.derived_params.get("WARNING:not_T0_global_min", False))


class BothRoutesAskTheSameQuestion(unittest.TestCase):
    """The fixed step size route has its own `check_T0_global_min`, and it had its own
    comparison: `np.allclose` on the coordinates, which is the defect this change is about.
    Both now call `theChainEndedElsewhere`, so the two cannot drift apart."""

    def test_the_fixed_step_route_calls_the_shared_function(self):
        from transitionlistener import transitionObservables_fixedstep as fixedstep

        self.assertIs(fixedstep.theChainEndedElsewhere, TO.theChainEndedElsewhere,
                      msg="the fixed step size route no longer uses the shared decision, so "
                          "it can answer a different question from the adaptive one")

    def test_neither_route_compares_the_coordinates_directly(self):
        """Reading the source, because driving the fixed step size route needs a whole run.

        `np.allclose(last_vev, ...)` is the comparison being replaced, and a copy of it
        reappearing in either file would reinstate the defect without failing anything else.
        """
        import inspect
        from transitionlistener import transitionObservables_fixedstep as fixedstep

        for module in (TO, fixedstep):
            with self.subTest(module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn(
                    "np.allclose(last_vev", source,
                    msg=f"{module.__name__} compares the chain's last field values against "
                        f"X0 component by component again")
                self.assertIn("theChainEndedElsewhere", source)


class TracingResidualIsNotADifferentMinimum(unittest.TestCase):
    """The case that made the old test fire on the 2HDM."""

    def test_an_endpoint_off_by_the_tracers_resolution_is_not_flagged(self):
        pot = _potential()
        # the measured offsets at lambda3 = 5.8672, in GeV
        off = np.array([1.2e-4, 6.5e-4])
        self.assertFalse(
            _flagged(pot, pot.X0 + off),
            msg="an endpoint within the tracer's resolution was reported as a different "
                "minimum")

    def test_the_old_comparison_did_fire_on_it(self):
        """Without this the test above would not be about anything."""
        pot = _potential()
        off = np.array([1.2e-4, 6.5e-4])
        self.assertFalse(
            np.allclose(pot.X0 + off, pot.X0),
            msg="np.allclose no longer fails on this offset, so the fixture is stale")


class AGenuinelyDifferentMinimumIsStillReported(unittest.TestCase):
    """The behaviour the check exists for."""

    def test_a_deeper_minimum_elsewhere_is_flagged(self):
        other = np.array([11.5, SCALE - 40.0])
        pot = _potential(other_minimum=other, depth_offset=-1.0e3)
        self.assertTrue(
            _flagged(pot, other),
            msg="the chain ended in a minimum 1000 GeV^4 deeper and 40 GeV away, and that "
                "was not reported")

    def test_a_deeper_minimum_too_close_for_the_field_test_is_still_flagged(self):
        """Depth is the question the check is really asking.

        `X0` is the model's analytic zero-temperature minimum, and the minimum of the full
        one-loop potential can sit slightly away from it. A displacement well inside the
        field tolerance can still carry a real difference in depth, and that difference is
        what says the chain did not end in the deepest minimum. Without a depth test this
        case is invisible.
        """
        other = np.array([11.5, SCALE - 0.15])
        pot = _potential(other_minimum=other, depth_offset=-1.0e3)
        separation = float(np.linalg.norm(other - pot.X0))
        self.assertLess(
            separation, TO._T0_MINIMUM_FIELD_RTOL * SCALE,
            msg="the fixture has to sit inside the field tolerance or it tests the other arm")
        self.assertTrue(
            _flagged(pot, other),
            msg=f"a minimum 1000 GeV^4 deeper, {separation:.3f} GeV away, was not reported")

    def test_a_degenerate_minimum_elsewhere_is_flagged_on_distance(self):
        """Equal depth, different place: the field test is what catches it."""
        other = np.array([11.5, SCALE - 40.0])
        pot = _potential(other_minimum=other, depth_offset=0.0)
        self.assertTrue(
            _flagged(pot, other),
            msg="a degenerate minimum 40 GeV away was not reported")

    def test_the_field_tolerance_lies_in_the_window_it_was_chosen_from(self):
        """The value itself, against the two measurements that bound it.

        Every other test here reads the tolerance out of the module, so none of them can say
        what its value should be: raise it by two decades and they all still pass, either
        side of the raised value. These two bounds are what fix it.

        The lower bound is the tracer's residual. On the 2HDM benchmark at
        `lambda3 = 5.8672` the traced endpoint of the chain sits 2.7e-06 of the field scale
        away from `X0` while being the same minimum, and the tolerance has to admit that with
        room to spare, since the residual is a property of the tracer's step and not a
        constant. The upper bound is the smallest separation that has to be reported: two
        minima a per cent of the field scale apart, 2.5 GeV on the 2HDM, are different
        vacua and the chain ending in the wrong one of them is what this check exists to
        catch.
        """
        self.assertGreater(
            TO._T0_MINIMUM_FIELD_RTOL, 10.0 * 2.7e-06,
            msg="the tolerance no longer admits the tracer's measured residual of 2.7e-06 "
                "of the field scale with a decade to spare, so the check will fire on a "
                "chain that ended in the right minimum")
        self.assertLess(
            TO._T0_MINIMUM_FIELD_RTOL, 1.0e-02,
            msg="the tolerance now admits two minima a per cent of the field scale apart as "
                "the same minimum, which is the case the check exists to report")

    def test_the_field_tolerance_is_where_it_says_it_is(self):
        """A degenerate pair either side of the tolerance, so its value is pinned.

        The two minima have equal depth, so the depth test says nothing and the field test
        decides on its own. A separation of 40 GeV, as the test above uses, is flagged
        whatever the tolerance is within four decades of its value, so it does not say where
        the tolerance sits. These two do.

        The displacement is spread equally over the two field components, which is what
        separates measuring it in norm from measuring the largest component: the norm is
        larger than the largest component by a factor of the square root of two, and the
        pair below is chosen to fall on opposite sides of the tolerance under the two
        measures. The old check compared component by component, and that is the defect
        being fixed.
        """
        X0 = np.array([11.5, SCALE])
        tolerance = TO._T0_MINIMUM_FIELD_RTOL * float(np.linalg.norm(X0))
        for name, factor, expected in (("just outside the tolerance", 1.22, True),
                                       ("just inside the tolerance", 0.49, False)):
            with self.subTest(name):
                step = factor * tolerance / np.sqrt(2.0)
                other = X0 + np.array([step, step])
                separation = float(np.linalg.norm(other - X0))
                self.assertGreater(
                    step, _BASIN,
                    msg="the displacement has to clear the stub's basin radius or the two "
                        "sides of the comparison get the same formula")
                self.assertAlmostEqual(separation / tolerance, factor, places=6)
                self.assertEqual(
                    _flagged(_potential(other_minimum=other, depth_offset=0.0), other),
                    expected,
                    msg=f"{name}: a degenerate minimum {separation:.4f} GeV away, against a "
                        f"tolerance of {tolerance:.4f} GeV, was "
                        f"{'not reported' if expected else 'reported'}")


class AQuestionThatCannotBeAnsweredIsReported(unittest.TestCase):
    """Neither comparison can be true of a `nan`, so one must not be left to reach them."""

    def test_a_non_finite_traced_endpoint_is_reported(self):
        pot = _potential()
        self.assertTrue(TO.theChainEndedElsewhere(pot, np.array([np.nan, SCALE])),
                        msg="an endpoint that is not a number was taken for the "
                            "zero-temperature minimum")

    def test_a_non_finite_potential_is_reported(self):
        pot = _potential()
        pot.Vtot = lambda X, T, *a, **k: float("nan")
        self.assertTrue(TO.theChainEndedElsewhere(pot, pot.X0),
                        msg="a potential that is not a number was taken for a match")

    def test_an_infinite_separation_is_reported(self):
        pot = _potential()
        self.assertTrue(TO.theChainEndedElsewhere(pot, np.array([np.inf, SCALE])))

    def test_a_finite_depth_does_not_excuse_a_separation_that_is_not_a_number(self):
        """The separation is checked in its own right, not only through the depths.

        A potential need not be non-finite where its argument is: one that clips or wraps its
        input returns an ordinary number at a field value that is not one, and then the depths
        agree while the distance between the two points is not a number. Both comparisons are
        false of that distance, so checking only the depths would report the pair as the
        zero-temperature minimum.
        """
        pot = _potential()
        pot.Vtot = lambda X, T, *a, **k: -4.5e7
        self.assertTrue(TO.theChainEndedElsewhere(pot, np.array([np.nan, SCALE])),
                        msg="a separation that is not a number passed because the depths "
                            "were finite")

    def test_this_is_what_the_comparison_it_replaced_did(self):
        """`np.allclose` returned False for a `nan`, so the old check reported too.

        The change may not turn a reported run into a silent one, and a `nan` is the input
        on which the two could differ.
        """
        self.assertFalse(np.allclose(np.array([np.nan, SCALE]), np.array([11.5, SCALE])))


if __name__ == "__main__":
    unittest.main()
