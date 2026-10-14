"""`epsabs` must be a fraction of the bounce, not of the gap between the minima.

`findProfile` integrates the bounce from its centre out to the metastable vacuum and holds that
integration to an absolute error `epsabs`. That error was `phitol` times `delta_phi`, the whole
gap between the two minima, while the field range the integration actually covers is the
bounce's own excursion, `|phi(0) - phi_metaMin|`.

The two are the same thing on an ordinary transition, where the bounce starts somewhere between
the minima. They are not on a supercooled one: the escape point sits a few `T` from the false
vacuum while the minima stay `v` apart, so the excursion shrinks with the temperature and the
gap does not. Measured on the shipped conformal dark U(1) at g = 0.46, y = 0.1, where the gap is
1000 internal units at every temperature and the excursion is 4.4 to 6.2 T, `phitol |delta_phi|`
is 1.6e-07 of the integrated feature at T_int = 0.1 and 2.3e-02 of it at 1e-6.

`epsabs` is also the radius inside which `integrateProfile` calls a trial profile arrived, and
that is the use that decides: at T_int = 1e-7 the radius `3 phitol |delta_phi|` was 2.19 times
the whole excursion, the first trial was accepted after two integration steps with the profile
not having moved, and S3/T came back 4.7e-14 against 103.8. The classes below cover the scale of
`epsabs`, that it does not follow the distant minimum, that the acceptance radius never swallows
the bounce, that an iteration the loop discards does not set the tolerance the saved profile is
re-integrated with, and that the run's deadline is not read as a wall too thin to integrate.
"""
import unittest
from unittest import mock

import numpy as np

import models.TL_conformal_dark_u1 as conformal_u1
from transitionlistener import errors
from transitionlistener import pathDeformation
from transitionlistener import tunneling1D


def _two_minima(gap, barrier_at, height):
    """A false vacuum at 0, a barrier near the origin, and a deep minimum at `gap`.

    `V` is a quartic whose derivative vanishes at 0, `barrier_at` and `gap`, so the shape near
    the origin -- which is all the bounce sees -- is held fixed while `gap` moves the far
    minimum. Depth is set by `height` so the bounce stays thick-walled.
    """
    def V(p):
        x = np.asanyarray(p, dtype=float)
        x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
        return height * (0.25 * x**4
                         - (barrier_at + gap) / 3.0 * x**3
                         + barrier_at * gap / 2.0 * x**2)

    def dV(p):
        x = np.asanyarray(p, dtype=float)
        x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
        return height * x * (x - barrier_at) * (x - gap)
    return V, dV


class EpsabsIsAFractionOfTheBounce(unittest.TestCase):

    @staticmethod
    def _epsabs_and_excursion(gap, phitol=1e-10):
        """Run findProfile and report the `epsabs` it used and the excursion it covered."""
        V, dV = _two_minima(gap, barrier_at=1.0, height=1.0e-3)
        # reflected so the deep minimum is at coordinate 0, which is the orientation the
        # solver wants and the one the converged reference uses
        inst = tunneling1D.SingleFieldInstanton(
            0.0, gap, lambda x: V(gap - np.asarray(x)), lambda x: -dV(gap - np.asarray(x)))
        seen = {}
        real = tunneling1D.SingleFieldInstanton.integrateProfile

        def spy(self, r0, y0, dr0, epsfrac, epsabs, drmin, rmax, *a, **k):
            seen["epsabs"] = float(np.asarray(epsabs, dtype=float)[0])
            seen["phi0"] = float(y0[0])
            return real(self, r0, y0, dr0, epsfrac, epsabs, drmin, rmax, *a, **k)

        tunneling1D.SingleFieldInstanton.integrateProfile = spy
        try:
            inst.findProfile(xtol=1e-10, phitol=phitol)
        finally:
            tunneling1D.SingleFieldInstanton.integrateProfile = real
        return seen["epsabs"], abs(seen["phi0"] - inst.phi_metaMin)

    def test_it_is_phitol_times_the_excursion(self):
        phitol = 1e-10
        eps, excursion = self._epsabs_and_excursion(gap=50.0, phitol=phitol)
        self.assertAlmostEqual(
            eps / excursion, phitol, delta=0.25 * phitol,
            msg=f"epsabs is {eps:.6g} against an excursion of {excursion:.6g}, a ratio of "
                f"{eps/excursion:.3g} rather than phitol = {phitol:g}")

    def test_it_does_not_follow_the_distant_minimum(self):
        """Moving the far minimum away must not loosen the integration of the bounce.

        The bounce never reaches that minimum, so nothing about it may change `epsabs`. Under
        the rule this replaces, `epsabs` was proportional to the gap and grew with it.
        """
        ratios = {}
        for gap in (50.0, 500.0):
            eps, excursion = self._epsabs_and_excursion(gap=gap)
            ratios[gap] = eps / excursion
        self.assertAlmostEqual(
            ratios[500.0], ratios[50.0], delta=0.25 * ratios[50.0],
            msg=f"a tenfold more distant minimum changed epsabs relative to the bounce: "
                f"{ratios}")


class TheAcceptanceRadiusNeverSwallowsTheBounce(unittest.TestCase):
    """`3*epsabs` decides when a profile has reached the false vacuum, so it must be smaller.

    `integrateProfile` spends `epsabs` twice. Besides controlling the Runge-Kutta step it is
    the radius in ``abs(y - [phi_metaMin, 0]) < 3*epsabs`` inside which a trial profile counts
    as having arrived at the metastable minimum, and that verdict ends the shooting loop.
    Tied to the gap between the minima the radius was a fixed absolute distance while the
    bounce shrank with the temperature, so on the shipped conformal dark U(1) at g = 0.46,
    y = 0.1 the ratio ``3 phitol |delta_phi| / excursion`` grew from 5.8e-05 at an internal
    temperature of 1e-3 to 2.19 at 1e-7. At that last temperature the first trial was declared
    converged after two integration steps with the profile not having moved at all, and the
    action came back 4.7e-14 instead of 103.8.

    This goes through ``pathDeformation.bounceAction`` with the model's own configured
    tunnelling parameters, which is the entry a scan takes, rather than calling `findProfile`
    directly.
    """

    T_INT = 1e-7

    def _run(self):
        pot = conformal_u1.specific_potential(
            {"g": 0.46, "y": 0.1, "v_GeV": 1.0}, verbose=False)
        calls = []
        real = tunneling1D.SingleFieldInstanton.integrateProfile

        def spy(self, r0, y0, dr0, epsfrac, epsabs, drmin, rmax, *a, **k):
            out = real(self, r0, y0, dr0, epsfrac, epsabs, drmin, rmax, *a, **k)
            calls.append(dict(
                epsabs=float(np.asarray(epsabs, dtype=float)[0]),
                excursion=abs(float(y0[0]) - float(self.phi_metaMin)),
                verdict=out.convergence_type))
            return out

        tunneling1D.SingleFieldInstanton.integrateProfile = spy
        try:
            out = pathDeformation.bounceAction(
                self.T_INT, lambda X, T: pot.Vtot(X, T), lambda X, T: pot.gradV(X, T), {},
                dict(low_vev=np.array([float(pot.v_stable)]), high_vev=np.array([0.0]),
                     Tnuc=self.T_INT, low_phase=1, high_phase=0),
                verbose=False, conversionFactor=pot.conversionFactor,
                **pot.config.tracingConf.tunneling_params)
        finally:
            tunneling1D.SingleFieldInstanton.integrateProfile = real
        return calls, float(out[self.T_INT]["action"]) / self.T_INT

    def test_the_supercooled_bounce_is_not_accepted_before_it_has_moved(self):
        calls, action_over_T = self._run()
        self.assertTrue(calls, "integrateProfile was never reached")
        for i, call in enumerate(calls):
            self.assertLess(
                3 * call["epsabs"], 1e-6 * call["excursion"],
                msg=f"trial {i}: a profile counts as having reached the false vacuum within "
                    f"{3 * call['epsabs']:.4g} of it, against a field excursion of "
                    f"{call['excursion']:.4g}")
        self.assertEqual(
            [c["verdict"] for c in calls].count("converged"), 0,
            msg=f"a trial was accepted by the 3*epsabs test: {calls}")
        self.assertGreater(
            action_over_T, 50.0,
            msg=f"S3/T came out {action_over_T:.6g}; the bounce has collapsed")


class TheSavedProfileKeepsTheToleranceOfTheTrialItReIntegrates(unittest.TestCase):
    """`epsabs` is per iteration, so the exit that discards an iteration must not keep it.

    `findProfile` ends by re-integrating the accepted trial to save the profile along the way,
    and hands `integrateAndSaveProfile` an `epsabs`. One exit from the shooting loop discards
    the current trial: when `initialConditions` returns a non-finite radius, or `x` itself goes
    non-finite, the loop falls back on the last finite `r0`, `y0` and `rf`. `epsabs` is
    recomputed at the top of every iteration, so on that exit the current value belongs to the
    trial that was thrown away, while every other value handed to the second integration
    belongs to the one that was kept.

    Under the rule this replaces `epsabs` did not depend on the iteration at all, so the two
    could not disagree.
    """

    FAIL_AT = 3

    def test_the_discarded_iteration_does_not_set_the_tolerance(self):
        fail_at = self.FAIL_AT
        V, dV = _two_minima(50.0, barrier_at=1.0, height=1.0e-3)
        instanton = tunneling1D.SingleFieldInstanton(
            0.0, 50.0, lambda x: V(50.0 - np.asarray(x)),
            lambda x: -dV(50.0 - np.asarray(x)))

        cls = tunneling1D.SingleFieldInstanton
        real_ic = cls.initialConditions
        real_ip = cls.integrateProfile
        real_save = cls.integrateAndSaveProfile
        seen = {"initial_conditions": 0, "trials": [], "saved": []}

        def initial_conditions(self, delta_phi0, rmin, delta_phi_cutoff):
            seen["initial_conditions"] += 1
            out = real_ic(self, delta_phi0, rmin, delta_phi_cutoff)
            if seen["initial_conditions"] == fail_at:
                return (np.inf, out[1], out[2])
            return out

        def integrate_profile(self, r0, y0, dr0, epsfrac, epsabs, drmin, rmax, *a, **k):
            seen["trials"].append(float(np.asarray(epsabs, dtype=float)[0]))
            return real_ip(self, r0, y0, dr0, epsfrac, epsabs, drmin, rmax, *a, **k)

        def integrate_and_save(self, R, y0, dr, epsfrac, epsabs, drmin, *a, **k):
            seen["saved"].append(float(np.asarray(epsabs, dtype=float)[0]))
            return real_save(self, R, y0, dr, epsfrac, epsabs, drmin, *a, **k)

        cls.initialConditions = initial_conditions
        cls.integrateProfile = integrate_profile
        cls.integrateAndSaveProfile = integrate_and_save
        try:
            instanton.findProfile(xtol=1e-10, phitol=1e-10)
        finally:
            cls.initialConditions = real_ic
            cls.integrateProfile = real_ip
            cls.integrateAndSaveProfile = real_save

        # The fallback exit has to be the one that was taken, and the tolerance has to have
        # changed between the trials, or there is nothing for the assertion below to catch.
        self.assertEqual(seen["initial_conditions"], self.FAIL_AT,
                         f"the loop did not stop at the non-finite radius: {seen}")
        self.assertEqual(len(seen["trials"]), self.FAIL_AT - 1,
                         f"the discarded iteration still integrated a profile: {seen}")
        self.assertGreater(len(set(seen["trials"])), 1,
                           f"`epsabs` did not change between trials: {seen}")
        self.assertEqual(len(seen["saved"]), 1, f"{seen}")
        self.assertEqual(
            seen["saved"][0], seen["trials"][-1],
            msg="the saved profile was held to a tolerance no trial was integrated with: "
                f"{seen}")


class TheRunsDeadlineSurvivesTheInitialConditions(unittest.TestCase):
    """A timeout raised inside the root find must not come back as a thin wall.

    `errors.Timeout` is raised from a POSIX alarm handler, so it lands wherever the run happens
    to be, `scipy.optimize.brentq` included. It subclasses `Exception`, and the handler around
    that root find turns anything it catches into a `TunnelingError`, which
    `pathDeformation._find_profile` turns into an `InfiniteActionError` and `bounceAction`
    reports as no tunnelling: a timed-out point would come back as a finished one with a
    verdict.
    """

    def test_a_timeout_is_not_read_as_a_bubble_too_thin_to_integrate(self):
        V, dV = _two_minima(50.0, barrier_at=1.0, height=1.0e-3)
        instanton = tunneling1D.SingleFieldInstanton(
            0.0, 50.0, lambda x: V(50.0 - np.asarray(x)),
            lambda x: -dV(50.0 - np.asarray(x)))
        # These arguments reach the root find; the branches above it return at `rmin`. The
        # instanton is built before the patch, since its own set-up calls `brentq` too.
        delta_phi0, rmin, cutoff = 1e-4 * 50.0, 1e-4 * instanton.rscale, 0.01 * 50.0
        calls = []

        def brentq(*args, **kwargs):
            calls.append(args[1:3])
            raise errors.Timeout("deadline inside the initial-condition root find")

        with mock.patch.object(tunneling1D.optimize, "brentq", brentq):
            with self.assertRaises(errors.Timeout):
                instanton.initialConditions(delta_phi0, rmin, cutoff)
        self.assertEqual(len(calls), 1,
                         "the root find was never reached, so this proves nothing: "
                         f"{calls}")



if __name__ == "__main__":
    unittest.main()
