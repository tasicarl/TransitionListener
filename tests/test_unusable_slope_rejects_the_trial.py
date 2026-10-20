"""A shooting trial without a usable slope is rejected; the solve carries on.

`findProfile` varies `x`, where a trial profile starts at
``phi(0) = phi_absMin + delta_phi0`` and ``delta_phi0 = exp(-x) * (phi_metaMin -
phi_absMin)``. Before integrating, it asks `dV_from_absMin` for the slope that drives the
field away from `phi_absMin`. Without one there is nothing for `initialConditions` to
start from, and the trial is useless.

It used to abort the whole solve, with a `TooSmalldVError` that `_find_profile` reports as
an infinite action. The test it aborted on was ``dV < 1e-200``, which compares a signed
quantity against a magnitude floor, so it fired on every negative `dV` as well as on an
underflowing one. Two things produce a negative `dV`:

* The path's own potential, near the true vacuum. `SplinePath.dV` differentiates a cubic
  spline through sampled `V` whenever `V_spline_samples` is set, and that spline ripples.
  Measured on the 2HDM at lambda3 = 5.8191375291 with `TL_2HDM_BSMPT_highacc`, the spline
  returns -23.5875 at the endpoint at T = 60 GeV, -196.002 at 58 GeV and +7.79409 at
  55 GeV, against an analytic ``grad V . dpath/dx`` of +0.0118, +0.0333 and -0.0110. The
  sign is set by where the samples fall. The solve failed at every temperature whose
  residual was negative and whose implied offset exceeded `phi_eps`, which made a band of
  temperatures with sharp non-monotonic edges: finite actions at 61.54 and 59 GeV, none
  between 61.52 and 58.
* A descending orientation, ``phi_absMin > phi_metaMin``, where `delta_phi0` is negative
  and so is every `dV` along it. No TransitionListener run takes that orientation,
  because `fullTunneling` measures the field coordinate as path length from the true
  vacuum, but `SingleFieldInstanton` is a general solver and the class refused it
  outright.

The seven classes here pin the parts of the replacement: that a wrong-sign slope costs the
trial and not the solve, that a slope missing everywhere is still reported as
`TooSmalldVError`, that a rejected trial bounds `x` from the side it lies on, that both
orientations of the minima now give the same bounce, that the profile handed back is built
from the trial it was integrated from, and that a shooting parameter that stops being a
number ends the bisection instead of repeating one trial for ever.

Against the code this replaces, six of the seven fail;
`NoUsableSlopeAnywhereIsStillReported` is the one that passes, because that code raised
`TooSmalldVError` for its case too. It is here so that a future change cannot drop the
report and return a bounce built from no trial at all. Each class was also measured
against the current solver with one behaviour put back, and each mutation is caught:

    behaviour put back                               classes that fail
    the guard reads abs(dV) rather than the slope    the wrong-sign window, both
      away from phi_absMin                            bracket sides, the kept profile
    a rejected trial always bounds x from above      the false-vacuum side
    the guard is dropped, every trial integrated     all but the orientations and the
                                                       non-finite x
    the report after the loop is dropped             no usable slope anywhere, the
                                                       non-finite x
    the interior points take the loop's delta_phi0   the kept profile
    the nan test on x is dropped                     the non-finite x

Measured in `scans/v211_validation/v57_bounce61/ROOT_CAUSE.md`.

Part of TransitionListener v2
"""
import unittest

import numpy as np

from transitionlistener import errors
from transitionlistener import tunneling1D


#: Tilt of the test quartic. Its stationary points are at 0, `_QUARTIC_TILT` and 1, so the
#: minima sit at 0 and 1 with the barrier between them. Above 0.5 the minimum at 0 is the
#: deeper one, which is the orientation a TransitionListener path takes.
_QUARTIC_TILT = 0.6

#: Slope injected in place of a healthy one. The magnitude is the value measured at the
#: 2HDM endpoint at T = 60 GeV, and it is deliberately far from zero: a guard that tests
#: only the magnitude of `dV` would let it through, and the profile built from it would not
#: match the one the unperturbed solver finds.
_WRONG_SIGN_SLOPE = -15.427


def quartic(tilt=_QUARTIC_TILT):
    """The test potential and its first two derivatives."""
    def V(phi):
        return 0.25*phi**4 - (1.0+tilt)/3.0*phi**3 + 0.5*tilt*phi**2

    def dV(phi):
        return phi**3 - (1.0+tilt)*phi**2 + tilt*phi

    def d2V(phi):
        return 3*phi**2 - 2*(1.0+tilt)*phi + tilt

    return V, dV, d2V


def ascending_instanton():
    """An instanton on the test quartic with the true vacuum at 0 and the false one at 1."""
    return tunneling1D.SingleFieldInstanton(0.0, 1.0, *quartic())


def action_of(instanton):
    return float(instanton.findAction(instanton.findProfile()))


class _SlopeSpy:
    """Replaces `dV_from_absMin` on one instance and records every trial it is asked about.

    The slope is read where `findProfile` reads it, so a trial that is rejected before any
    integration is recorded as well. `replace_inside` names a window of ``|delta_phi0|``
    in which the recorded value is handed back with the wrong sign instead of the true one,
    and `replace_exactly` names a single ``delta_phi0`` to do that to. Both are functions of
    ``delta_phi0`` alone, as a potential is, so every call site that asks about the same
    displacement is given the same answer.
    """

    def __init__(self, instanton, replace_inside=None, replace_everywhere=False,
                 replace_exactly=None):
        self.instanton = instanton
        self.replace_inside = replace_inside
        self.replace_everywhere = replace_everywhere
        self.replace_exactly = replace_exactly
        self.trials = []          # (delta_phi0, slope returned) in call order
        self.replaced = 0
        self._real = instanton.dV_from_absMin

    def __enter__(self):
        self.instanton.dV_from_absMin = self
        return self

    def __exit__(self, *exc):
        # `del`, not a rebind: `__enter__` shadows the class's method with an instance
        # attribute, so assigning the bound method back would leave the shadow in place and
        # the instance holding a bound method of itself. `_real` stays available for a test
        # that wants the true slope after the window closes.
        del self.instanton.dV_from_absMin
        return False

    def __call__(self, delta_phi0):
        value = self._real(delta_phi0)
        magnitude = abs(float(np.squeeze(delta_phi0)))
        if self.replace_everywhere:
            value, self.replaced = 0.0, self.replaced + 1
        elif self.replace_inside is not None and magnitude < self.replace_inside:
            value, self.replaced = _WRONG_SIGN_SLOPE, self.replaced + 1
        elif (self.replace_exactly is not None
              and float(np.squeeze(delta_phi0)) == self.replace_exactly):
            value, self.replaced = _WRONG_SIGN_SLOPE, self.replaced + 1
        self.trials.append((float(np.squeeze(delta_phi0)), float(np.squeeze(value))))
        return value

    def distinct_trial_displacements(self):
        """The `delta_phi0` of each trial, with the repeat each trial makes collapsed.

        `findProfile` asks for the slope once per trial and `initialConditions` asks again
        for the same `delta_phi0`, so the raw record holds each accepted trial twice.
        """
        out = []
        for delta_phi0, _ in self.trials:
            if not out or abs(out[-1] - delta_phi0) > 0.0:
                out.append(delta_phi0)
        return out


class AWrongSignSlopeCostsTheTrialNotTheSolve(unittest.TestCase):
    """A window of wrong-sign slopes must not change the bounce the solver returns.

    The window is chosen below the displacement the bisection converges to, which is
    where the 2HDM failure sits: there the window ran over ``delta_phi0`` from 3e-05 to
    1.5e-03 while the bounce converged at 5.67. On this quartic the bisection converges at
    ``delta_phi0 = 4.5661e-04`` and passes through 1.1802e-12 and 6.9692e-08 on the way,
    so a window at 1e-06 catches trials the solver takes and excludes the one it keeps.
    """

    WINDOW = 1e-06

    def test_the_action_is_the_one_the_unperturbed_solver_finds(self):
        reference = action_of(ascending_instanton())

        instanton = ascending_instanton()
        with _SlopeSpy(instanton, replace_inside=self.WINDOW) as spy:
            perturbed = action_of(instanton)

        # The premise, asserted first: without a rejected trial the comparison is empty.
        self.assertGreater(
            spy.replaced, 0,
            "no trial fell in the window, so this test cannot fail")
        # And the window must stop short of the trial the bisection keeps, or the bounce
        # itself is perturbed and the comparison is no longer about rejected trials.
        converged = spy.distinct_trial_displacements()[-1]
        self.assertGreater(
            abs(converged), self.WINDOW,
            f"the window reached the converged displacement {converged!r}, so what the "
            "comparison below measures is a perturbed bounce")
        self.assertNotIn(
            "dV_from_absMin", vars(instanton),
            "the spy left itself on the instance, so the comparison ran against it")
        self.assertAlmostEqual(
            perturbed, reference, delta=abs(reference)*1e-09,
            msg=f"the wrong-sign window moved the action from {reference!r} "
                f"to {perturbed!r}")


class NoUsableSlopeAnywhereIsStillReported(unittest.TestCase):
    """A solve in which no trial has a usable slope has no bounce, and must say so.

    This is the case the unconditional raise was reaching for. Rejecting trials one at a
    time has to keep it: the bisection runs out of bracket and leaves no integrated
    profile behind, and `findProfile` must report `TooSmalldVError` rather than return a
    profile built from nothing.
    """

    def test_a_solve_with_no_usable_trial_raises(self):
        instanton = ascending_instanton()
        with _SlopeSpy(instanton, replace_everywhere=True) as spy:
            with self.assertRaises(errors.TooSmalldVError):
                instanton.findProfile()
        self.assertGreater(
            spy.replaced, 0, "the spy never ran, so this test cannot fail")


class TheRejectedTrialBoundsXFromTheSideItLiesOn(unittest.TestCase):
    """A rejected trial near `phi_absMin` is an upper bound on `x`, so the next `x` is lower.

    `delta_phi0 = exp(-x) * delta_phi` falls as `x` rises, so a trial nearer `phi_absMin`
    than the potential's derivative is resolved bounds `x` from above and the bisection has
    to come back with a larger `delta_phi0`. Were it bounded from below instead, the
    bisection would step further into the region it cannot resolve.
    """

    WINDOW = 1e-06

    def test_the_trial_after_a_rejection_is_displaced_further(self):
        instanton = ascending_instanton()
        with _SlopeSpy(instanton, replace_inside=self.WINDOW) as spy:
            instanton.findProfile()
        displacements = spy.distinct_trial_displacements()
        rejected = [i for i, d in enumerate(displacements)
                    if abs(d) < self.WINDOW and i + 1 < len(displacements)]
        self.assertGreater(
            len(rejected), 0, "no rejected trial had a successor, so nothing is asserted")
        for i in rejected:
            self.assertGreater(
                abs(displacements[i+1]), abs(displacements[i]),
                f"trial {i} at delta_phi0 = {displacements[i]!r} was rejected and the "
                f"next one moved to {displacements[i+1]!r}, further towards phi_absMin")


class ATrialNearTheFalseVacuumBoundsXFromBelow(unittest.TestCase):
    """The other side of the bracket, which a small `xguess` reaches on the first trial.

    `dV_from_absMin` vanishes at `phi_metaMin` as well, because that is a minimum too, and
    just inside it the slope points back towards `phi_metaMin`. So a trial with
    ``delta_phi0`` near ``delta_phi`` has no usable slope either, and it has to bound `x`
    from *below*: the bisection must climb away from `phi_metaMin`. Bounding it from above
    there halves `x` towards zero instead, further into the region with no usable slope,
    and the solve ends with no integrated profile at all.

    ``x = 1e-12`` puts the first trial at ``delta_phi0 = 1 - 1e-12`` on this quartic,
    which is `phi_metaMin` to within 1e-12, and a model's
    `tunneling_findProfile_params` is free to pass such an `xguess`.
    """

    XGUESS = 1e-12

    def test_a_first_trial_at_the_false_vacuum_still_finds_the_bounce(self):
        reference = action_of(ascending_instanton())

        instanton = ascending_instanton()
        with _SlopeSpy(instanton) as spy:
            found = float(instanton.findAction(
                instanton.findProfile(xguess=self.XGUESS)))

        # The premise: the first trial really did start at the false vacuum, so the
        # undershoot branch is the one under test.
        first = spy.distinct_trial_displacements()[0]
        self.assertAlmostEqual(
            abs(first), 1.0, delta=1e-06,
            msg=f"the first trial was at delta_phi0 = {first!r}, not at phi_metaMin")
        self.assertAlmostEqual(
            found, reference, delta=abs(reference)*1e-06,
            msg=f"from an xguess at the false vacuum the solver returned {found!r} "
                f"against {reference!r}")


class BothOrientationsOfTheMinimaGiveTheSameBounce(unittest.TestCase):
    """Relabelling the field must not decide whether a bounce exists.

    The quartic with the true vacuum at 1 and the false one at 0 is the same physics as
    the same quartic mirrored through ``phi -> 1 - phi``, which puts the true vacuum at 0.
    Testing `dV` without its direction refused the first outright while solving the
    second.
    """

    def test_the_descending_orientation_agrees_with_its_mirror(self):
        V, dV, d2V = quartic(tilt=0.1)
        descending = tunneling1D.SingleFieldInstanton(1.0, 0.0, V, dV, d2V)
        mirrored = tunneling1D.SingleFieldInstanton(
            0.0, 1.0,
            lambda phi: V(1.0-phi), lambda phi: -dV(1.0-phi), lambda phi: d2V(1.0-phi))

        descending_action = action_of(descending)
        mirrored_action = action_of(mirrored)
        # `findProfile`'s signature defaults are used here, `xtol = phitol = 1e-4`, so the
        # two differ by the shooting tolerance and not by less: 1.7132091393850302 against
        # 1.7132017403828275, 4.3e-06 relative.
        self.assertAlmostEqual(
            descending_action, mirrored_action, delta=abs(mirrored_action)*1e-04,
            msg=f"the two orientations gave {descending_action!r} and "
                f"{mirrored_action!r}")


class TheKeptProfileIsBuiltFromTheTrialItWasIntegratedFrom(unittest.TestCase):
    """The interior points start from the `delta_phi0` the kept profile was integrated from.

    `findProfile` fills the radii inside ``R[0]`` from `exactSolution`, which needs the
    slope at ``phi(0)``. The loop can leave on an iteration whose trial was rejected, the
    stopping test at its foot running after the bracket has been narrowed by the rejection,
    and then the loop's `delta_phi0` belongs to an `x` that was never integrated. Taking it
    there builds the inside of an accepted profile from the slope that was judged unusable,
    and `dV_from_absMin` is a function of `delta_phi0`, so the same wrong-sign value comes
    back for it: on this quartic that returns an action of 4345443482289.82 against
    94.83569041562252, with no error raised. The solver carries the kept trial's
    `delta_phi0` alongside its `rf` instead.

    The trial to reject is the last one of an unperturbed solve, read off that solve rather
    than assumed, so this follows the solver if the bisection changes.
    """

    def test_a_rejected_last_trial_does_not_reach_the_interior_points(self):
        reference_instanton = ascending_instanton()
        with _SlopeSpy(reference_instanton) as record:
            reference = action_of(reference_instanton)
        trials = record.distinct_trial_displacements()
        last = trials[-1]
        self.assertEqual(
            trials.count(last), 1,
            f"delta_phi0 = {last!r} is reached by {trials.count(last)} trials, so "
            "rejecting it by value does not isolate the last one")

        instanton = ascending_instanton()
        integrated = []
        real_initial_conditions = instanton.initialConditions

        def initial_conditions(delta_phi0, *args):
            # Reached only by a trial whose slope passed the guard, so this records which
            # displacements were integrated.
            integrated.append(float(np.squeeze(delta_phi0)))
            return real_initial_conditions(delta_phi0, *args)

        instanton.initialConditions = initial_conditions
        with _SlopeSpy(instanton, replace_exactly=last) as spy:
            profile = instanton.findProfile()
        found = float(instanton.findAction(profile))

        # The premises: the rejected trial is the last one the bisection made, and it was
        # not integrated, so the kept profile belongs to an earlier one.
        self.assertGreaterEqual(
            spy.replaced, 1,
            "the targeted displacement was never asked about, so nothing was rejected")
        self.assertNotEqual(
            integrated[-1], last,
            "the rejected trial was integrated after all, so the guard did not reject it")
        # The record runs past the loop: the interior points ask for a slope too. What has
        # to hold is that no trial was integrated after the rejected one, so the loop left
        # on the rejection and the kept profile is the earlier trial's.
        record = spy.distinct_trial_displacements()
        self.assertGreater(
            record.index(last), record.index(integrated[-1]),
            f"the kept trial at {integrated[-1]!r} came after the rejected one at "
            f"{last!r}, so the loop did not leave on the rejection")

        # The contract: the profile at r = 0 is the kept trial's phi(0).
        self.assertEqual(profile.R[0], 0.0, "the interior points were not built")
        self.assertEqual(
            profile.Phi[0], instanton.phi_absMin + integrated[-1],
            f"the profile starts at {profile.Phi[0]!r}, the rejected trial's phi(0), "
            f"rather than at the kept trial's "
            f"{instanton.phi_absMin + integrated[-1]!r}")
        self.assertAlmostEqual(
            found, reference, delta=abs(reference)*1e-04,
            msg=f"rejecting the last trial moved the action from {reference!r} to "
                f"{found!r}")


class ANonFiniteSlopeNeverReachesTheIntegration(unittest.TestCase):
    """An infinite slope passes a floor test; only a finiteness test refuses it.

    `nan > _MIN_USABLE_DV_FROM_ABSMIN` is false, so a `nan` slope is rejected already, but
    by the accident that every comparison against a `nan` is false. `+inf` is greater than
    any floor, so it passed into `initialConditions`, and the `phi0` and `dphi0` it returns
    then reached `integrateProfile` as the starting values of an integration that cannot
    mean anything. `SplinePath.dV` differentiates a spline through sampled `V` and returns
    either where the sampling is degenerate.
    """

    #: Displacement whose slope is replaced, chosen to be the first trial's.
    WHERE = 0.5

    def _run(self, bad_value):
        instanton = ascending_instanton()
        real_slope = instanton.dV_from_absMin
        seen = {"replaced": 0}

        def slope(delta_phi0):
            value = real_slope(delta_phi0)
            if abs(abs(float(np.squeeze(delta_phi0))) - self.WHERE) < 1e-09:
                seen["replaced"] += 1
                return bad_value
            return value

        instanton.dV_from_absMin = slope
        starts = []
        real_integrate = instanton.integrateProfile

        def integrate(r0, y0, *a, **k):
            starts.append((float(np.squeeze(r0)),
                           [float(v) for v in np.atleast_1d(np.squeeze(y0))]))
            return real_integrate(r0, y0, *a, **k)

        instanton.integrateProfile = integrate
        try:
            action = float(instanton.findAction(
                instanton.findProfile(xguess=-np.log(self.WHERE))))
        finally:
            del instanton.dV_from_absMin
            del instanton.integrateProfile
        return action, starts, seen

    def test_the_fixture_replaces_a_slope_that_is_actually_asked_for(self):
        _, _, seen = self._run(np.inf)
        self.assertGreater(
            seen["replaced"], 0,
            msg=f"no trial was taken at |delta_phi0| = {self.WHERE}, so nothing was tested")

    def test_an_infinite_slope_starts_no_integration(self):
        _, starts, _ = self._run(np.inf)
        bad = [s for s in starts if not all(np.isfinite(v) for v in s[1])
               or not np.isfinite(s[0])]
        self.assertEqual(
            bad, [],
            msg=f"{len(bad)} integration(s) were started from non-finite values, e.g. {bad[:1]}")

    def test_a_nan_slope_starts_no_integration(self):
        _, starts, _ = self._run(np.nan)
        bad = [s for s in starts if not all(np.isfinite(v) for v in s[1])
               or not np.isfinite(s[0])]
        self.assertEqual(bad, [], msg=f"integration started from {bad[:1]}")

    def test_the_bounce_is_still_found(self):
        """Rejecting the trial must cost the trial, not the solve."""
        reference = action_of(ascending_instanton())
        action, _, _ = self._run(np.inf)
        self.assertAlmostEqual(
            action, reference, delta=1e-03 * abs(reference),
            msg=f"the action moved from {reference} to {action} when one trial's slope "
                "was replaced by an infinity")


class _TrialBudgetExhausted(Exception):
    """Raised by this module when the bisection will not stop, so a hang fails instead."""


class ANonFiniteShootingParameterEndsTheBisection(unittest.TestCase):
    """A `nan` `x` has to end the loop, because nothing at the foot of it can.

    The finiteness test beside `r0_` sits inside the branch a usable slope takes, so a trial
    rejected for want of a slope does not reach it. Neither stopping test at the foot of the
    loop can fire on a `nan` either: ``x == xmin`` is false and
    ``(xmax - xmin) < xtol * ...`` is false. So a `nan` is tested for at the top of the
    loop, before the slope is read; without that test the bisection repeats the same trial
    without end. `findProfile` takes `xguess` from a model's
    `tunneling_findProfile_params`, which is free to carry one.

    The budget turns a loop that does not terminate into a failing test rather than a
    hanging one.
    """

    TRIAL_BUDGET = 500

    def test_a_nan_xguess_stops_instead_of_repeating_the_same_trial(self):
        instanton = ascending_instanton()
        asked = []
        real = instanton.dV_from_absMin

        def counted(delta_phi0):
            asked.append(float(np.squeeze(delta_phi0)))
            if len(asked) > self.TRIAL_BUDGET:
                raise _TrialBudgetExhausted(
                    f"the bisection asked for the slope more than {self.TRIAL_BUDGET} "
                    "times on a non-finite x, so it does not terminate")
            return real(delta_phi0)

        instanton.dV_from_absMin = counted
        with self.assertRaises(Exception) as caught:
            instanton.findProfile(xguess=float("nan"))
        self.assertNotIsInstance(
            caught.exception, _TrialBudgetExhausted, str(caught.exception))
        self.assertIsInstance(
            caught.exception, errors.TooSmalldVError,
            f"a non-finite x came back as {type(caught.exception).__name__}; it leaves no "
            "integrated trial, which is what TooSmalldVError reports")


if __name__ == "__main__":
    unittest.main()
