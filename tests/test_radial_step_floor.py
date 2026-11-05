"""The floor on the radial step is set by the radius, not by the step the integration began at.

`integrateProfile` integrates the bubble wall equation outward in the radius `r` with an
adaptive step. It needs a floor, so that an integration which cannot advance ends instead of
spinning. That floor used to be one per cent of the initial step, which says nothing about
whether progress is possible: a step smaller than `r` times the machine epsilon cannot advance
the radius at all, and the initial step is unrelated to that.

The consequence was measured on the dark U(1) model at a quartic coupling `l` of 1e-4 and a
vacuum expectation value of 1e10 GeV, where the barrier occupies about one part in a thousand of
the path between the minima. The smallest step a converging integration needs there wanders
between 1.6e-06 and 9.6e-06 from one temperature to the next, while the old floor sat at
6.2e-06, so whether a temperature could be solved at all alternated with the temperature:

    T_int    smallest step needed    old floor    old outcome
    5.600          1.63e-06          6.50e-06    refused
    5.620          4.06e-06          6.50e-06    refused
    5.630          3.84e-06          6.50e-06    refused
    5.635          2.66e-06          6.50e-06    refused
    5.640          9.62e-06          6.26e-06    solved, S_3/T = 58.261247
    5.645          2.48e-06          6.19e-06    refused
    5.650          7.65e-06          6.19e-06    solved, S_3/T = 64.748089

A refused integration is translated into an infinite action, which the percolation integral
reads as no nucleation, so the temperatures carrying the largest nucleation rate were dropped
and the point failed for want of a true vacuum fraction.

The floor is now `r` times `_RADIAL_STEP_FLOOR_FRACTION`, three orders above the machine
resolution of the radius and five orders below the smallest step a converging integration was
measured to need, and `_RADIAL_STEP_LIMIT` bounds the work the floor no longer bounds.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

import numpy as np

from transitionlistener import pathDeformation
from transitionlistener import tunneling1D
from transitionlistener.helper_functions import IntegrationError, load_potential
from transitionlistener.phases import Phases, findLocalMinimum
from transitionlistener.transitions import Transitions

REPO = Path(__file__).resolve().parents[1]


class TheFloorIsMeasuredAgainstTheRadius(unittest.TestCase):
    """The floor tracks the radius rather than the step the integration started from."""

    @staticmethod
    def instanton():
        """A quartic with minima at 0 and 1, the fixture of the other shooting tests."""

        def V(phi):
            return 0.25 * phi ** 4 - 0.49 * phi ** 3 + 0.235 * phi ** 2

        def dV(phi):
            return phi ** 3 - 1.47 * phi ** 2 + 0.47 * phi

        return tunneling1D.SingleFieldInstanton(1.0, 0.0, V, dV)

    def test_a_step_far_below_a_hundredth_of_the_initial_one_is_accepted(self):
        """A step the old floor refused is now taken, and the integration walks on it.

        The initial step is 1e-3, so the old floor stood at 1e-05 and a step of 1e-09 was
        refused outright. Here every step the controller offers is 1e-09, which is four
        orders below that floor and four orders above the radius times
        `_RADIAL_STEP_FLOOR_FRACTION` at a radius of order one. The integration therefore
        has to accept every one of them and walk the radius from 1.0 past `rmax`, which is
        the only exit left. Under the old floor the first step ends it with ``dr < drmin``.
        """
        reached_rmax = "r > rmax"
        step = 1e-9
        self.assertLess(step, 0.01 * 1e-3, "the step is not below the old floor")
        self.assertGreater(step, 1.0 * tunneling1D._RADIAL_STEP_FLOOR_FRACTION * 1e3,
                           "the step is not clear of the new floor")
        # `rmax` is a distance from the starting radius, so this is 1000 steps of walking.
        constant = (step for _ in range(4000))
        with self.assertRaises(IntegrationError) as caught:
            self.integrate_with_steps(constant, rmax=1000 * step)
        self.assertIn(reached_rmax, str(caught.exception),
                      "the integration did not walk on steps below the old floor")

    def test_the_saver_refuses_a_radius_that_cannot_advance(self):
        """`integrateAndSaveProfile` floors its own step and rejects a stalled radius.

        The saver has no `rmax` and no step count: its loop ends only when the radius
        passes the requested output points. A floor far below one unit in the last place of
        the radius would let it set ``r1 = r0`` for ever, so it carries the same
        radius-relative floor and refuses a step that leaves the radius where it was.
        """
        instanton = self.instanton()
        start = 1e8
        budget = 1000
        calls = {"n": 0}

        def controller(y, dydt, t, f, dt_try, epsfrac, epsabs, *args, **kwargs):
            # Far below one unit in the last place at a radius of 1e8, so `r0 + dr == r0`
            # unless the floor lifts the step.
            calls["n"] += 1
            if calls["n"] > budget:
                raise AssertionError("the saver did not advance the radius: %d steps at "
                                     "a step of 1e-30 and the requested radii not reached"
                                     % calls["n"])
            return np.zeros_like(np.asarray(y, dtype=float)), 1e-30, 1e-30

        original = tunneling1D.rkqs
        tunneling1D.rkqs = controller
        self.addCleanup(setattr, tunneling1D, "rkqs", original)
        # The floor at this radius is 1e8 times 1e-13, which is 1e-05, so eleven steps
        # carry the radius past the last requested point. Without the floor the step stays
        # at 1e-30, the radius never changes, and the controller runs out of budget.
        radii = start + np.array([0.0, 5.0e-5, 1.0e-4])
        profile = instanton.integrateAndSaveProfile(
            radii, np.array([0.5, 0.0]), 1e-3, 1e-10, np.array([1e-10, 1e-10]), 0.0)
        self.assertTrue(np.all(np.isfinite(np.asarray(profile.Phi, dtype=float))),
                        "the saved profile is not finite")
        self.assertLess(calls["n"], budget,
                        "the saver needed more steps than the floor allows")

    def test_the_saver_ends_when_the_floor_itself_cannot_advance_the_radius(self):
        """With the relative floor switched off, the guard ends the loop instead of spinning.

        The floor is what keeps the saver moving, so the guard behind it can only be
        reached by removing the floor. That is what this does: with the fraction at zero
        and no absolute floor, the step stays at 1e-30, `r0 + dr` returns `r0`, and the
        guard has to end the loop. Without the guard the controller runs out of budget.
        """
        instanton = self.instanton()
        budget = 1000
        calls = {"n": 0}

        def controller(y, dydt, t, f, dt_try, epsfrac, epsabs, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] > budget:
                raise AssertionError("the saver spun on a radius that does not advance: "
                                     "%d steps" % calls["n"])
            return np.zeros_like(np.asarray(y, dtype=float)), 1e-30, 1e-30

        original_rkqs = tunneling1D.rkqs
        original_fraction = tunneling1D._RADIAL_STEP_FLOOR_FRACTION
        tunneling1D.rkqs = controller
        tunneling1D._RADIAL_STEP_FLOOR_FRACTION = 0.0
        self.addCleanup(setattr, tunneling1D, "rkqs", original_rkqs)
        self.addCleanup(setattr, tunneling1D, "_RADIAL_STEP_FLOOR_FRACTION",
                        original_fraction)
        radii = 1e8 + np.array([0.0, 5.0e-5, 1.0e-4])
        with self.assertRaises(IntegrationError) as caught:
            instanton.integrateAndSaveProfile(
                radii, np.array([0.5, 0.0]), 1e-3, 1e-10, np.array([1e-10, 1e-10]), 0.0)
        self.assertIn("does not advance", str(caught.exception))

    def test_the_floor_is_three_orders_above_the_resolution_of_the_radius(self):
        """Below `r` times the machine epsilon a step cannot advance the radius at all."""
        fraction = tunneling1D._RADIAL_STEP_FLOOR_FRACTION
        self.assertGreater(fraction, np.finfo(float).eps,
                           "the floor is at or below the machine epsilon, where a step cannot "
                           "advance the radius and the integration would spin")
        self.assertLess(fraction, 1e-8,
                        "the floor is above the smallest step a converging integration was "
                        "measured to need, 1.6e-06 at a radius of 29.6")

    def integrate_with_steps(self, steps, rmax=1e30, drmin=0.0):
        """Run the loop against a step controller that returns a prescribed sequence.

        The step controller is what the floor and the step count act on, so prescribing its
        output is the only way to reach either guard without first producing a non-finite
        value, which some other check in the loop would catch instead.
        """
        instanton = self.instanton()
        sequence = iter(steps)

        def controller(y, dydt, t, f, dt_try, epsfrac, epsabs, *args, **kwargs):
            dt = next(sequence)
            return np.zeros_like(np.asarray(y, dtype=float)), dt, dt

        original = tunneling1D.rkqs
        tunneling1D.rkqs = controller
        self.addCleanup(setattr, tunneling1D, "rkqs", original)
        return instanton.integrateProfile(
            1.0, np.array([0.5, 0.0]), 1e-3, 1e-10,
            np.array([1e-10, 1e-10]), drmin, rmax)

    def test_an_integration_whose_step_collapses_still_ends(self):
        """A step that halves for ever leaves the radius bounded, so only the floor ends it."""
        # Halving from 1e-3, the radius converges on 1.001 and never reaches `rmax`, so the
        # loop has no other way out. 200 halvings take the step to 3e-63, far below the floor.
        collapsing = (1e-3 * 0.5 ** n for n in range(200))
        with self.assertRaises(IntegrationError) as caught:
            self.integrate_with_steps(collapsing)
        self.assertIn("drmin", str(caught.exception))

    def test_an_integration_that_never_arrives_is_ended_by_the_step_count(self):
        """A step above the floor that never reaches `rmax` is ended by the count.

        The limit is lowered for the test so that the loop ends in milliseconds whether the
        count is in force or not. Without that, removing the count does not fail this test,
        it hangs it for a million iterations, and a test that hangs under the mutation it
        exists to catch is a test a continuous-integration run cannot use.
        """
        limit = 50
        original = tunneling1D._RADIAL_STEP_LIMIT
        tunneling1D._RADIAL_STEP_LIMIT = limit
        self.addCleanup(setattr, tunneling1D, "_RADIAL_STEP_LIMIT", original)
        # A constant step of 1e-6 at a radius of order 1 is seven orders above the floor, so
        # the floor cannot fire, and `rmax` is out of reach within the permitted steps. The
        # sequence runs out shortly after the limit, so a missing count ends in a
        # `StopIteration` rather than in a hang.
        taken = {"n": 0}

        def counted():
            for _ in range(limit + 10):
                taken["n"] += 1
                yield 1e-6

        with self.assertRaises(IntegrationError) as caught:
            self.integrate_with_steps(counted(), rmax=1e30)
        self.assertIn("step count", str(caught.exception))
        self.assertEqual(taken["n"], limit,
                         "the integration took %d steps against the advertised limit of %d"
                         % (taken["n"], limit))


class TheBounceOfASupercooledPointIsSolved(unittest.TestCase):
    """The measured point of this module's docstring, through the solver a run uses."""

    #: The internal temperatures of the table above and the action per temperature that a
    #: converging integration gives. Measured with the floor removed altogether, and
    #: unchanged when it is lowered by a further four orders, so these are the values the
    #: integration converges to and not values a floor selected.
    #:
    #: All seven temperatures of the window, measured on an AMD EPYC 9454. The four that
    #: were recorded first read 33.300790, 51.859599, 61.494693 and 64.748089 on an AMD
    #: EPYC 7453, higher by 4.848e-04, 3.318e-04, 2.864e-04 and 2.738e-04 in relative
    #: terms, and bit-identical across six runs on each processor. So the value is
    #: deterministic per processor and differs between processors, and `VALUE_TOLERANCE`
    #: has to clear that spread. A first version of this test pinned the values to 1e-06
    #: and passed on one processor while failing on the other, which is the same defect
    #: that `tests/test_1d_tunneling.py` documents for the action of its own
    #: configuration.
    MEASURED = {5.600: 33.284647, 5.620: 45.538412, 5.630: 51.842391, 5.635: 55.031718,
                5.640: 58.243756, 5.645: 61.477080, 5.650: 64.730359}

    #: Ten times the measured spread between processors. It is loose on purpose: the gate
    #: this class rests on is that the action is finite at all seven temperatures, which five
    #: of the seven temperatures of the module docstring failed before the fix and which no
    #: processor difference can change. The value comparison is the weaker guard, against a
    #: floor that lets the integration through but truncates it, and a truncated integration
    #: is wrong by per cent and not by 1e-04.
    VALUE_TOLERANCE = 5.0e-3

    @classmethod
    def setUpClass(cls):
        params = {"l": 1.0e-4, "v_GeV": 1.0e10, "g_tilde": 2.69, "precision_mode": "benchmark"}
        with contextlib.redirect_stdout(io.StringIO()):
            potential = load_potential(
                str(REPO / "models/TL_dark_U1.py"), "specific_potential")(params, verbose=False)
            phases = Phases(potential, verbose=False)
            transitions = Transitions(phases, potential, False)
        first_order = [t for t in transitions if t is not None and t.type == 1]
        if not first_order:
            raise unittest.SkipTest("the model has no first-order transition at these parameters")
        cls.potential = potential
        cls.high = phases[first_order[0].high_phase]
        cls.low = phases[first_order[0].low_phase]
        cls.settings = dict(potential.config.tracingConf.tunneling_params)

    def action_over_temperature(self, temperature):
        with contextlib.redirect_stdout(io.StringIO()):
            high_vev = np.atleast_1d(np.squeeze(findLocalMinimum(
                self.high.valAt(temperature), temperature,
                self.potential.gradV, self.potential.d2V)))
            low_vev = np.atleast_1d(np.squeeze(findLocalMinimum(
                self.low.valAt(temperature), temperature,
                self.potential.gradV, self.potential.d2V)))
            result = pathDeformation.fullTunneling(
                [low_vev, high_vev],
                lambda X: self.potential.Vtot(X, temperature),
                lambda X: self.potential.gradV(X, temperature),
                callback_data=temperature, verbose=False, **self.settings)
        return float(result.action) / temperature

    def test_every_temperature_of_the_measured_window_gives_a_finite_action(self):
        """Five of these seven temperatures came back infinite under the old floor."""
        for temperature, expected in sorted(self.MEASURED.items()):
            with self.subTest(T_int=temperature):
                got = self.action_over_temperature(temperature)
                self.assertTrue(np.isfinite(got),
                                f"S_3/T at T_int = {temperature} is {got}")
                self.assertAlmostEqual(got / expected, 1.0, delta=self.VALUE_TOLERANCE,
                                       msg=f"S_3/T at T_int = {temperature} is {got:.6f} "
                                           f"against the converged {expected:.6f}")

    def test_the_action_falls_as_the_barrier_thins(self):
        """The sequence the infinite values were hiding: smooth and monotone in temperature."""
        temperatures = sorted(self.MEASURED)
        values = [self.action_over_temperature(T) for T in temperatures]
        self.assertEqual(values, sorted(values),
                         f"S_3/T is not increasing with temperature: {values}")


if __name__ == "__main__":
    unittest.main()
