import unittest

import models.TL_conformal_dark_u1 as potential
from transitionlistener.phases import Phases
from transitionlistener import pathDeformation

class test1DPotential(unittest.TestCase):
    """Test the unittests."""

    def test_phase_tracing(self):
        pot = potential.specific_potential({"g": 0.7, "y": 0.24, "v_GeV": 1000}, verbose=True)
        phases = Phases(pot, verbose=True)

        # Should find 2 phases
        nphases = len(phases.keys())
        self.assertTrue(nphases == 2)

        # Finds correct minima? Near zero field the zero-temperature potential
        # rises only like phi^4 log(phi^2), so the tracer places the symmetric
        # minimum only to about 1e-5 of the vev (13 MeV here, since v2.0.0).
        self.assertAlmostEqual(phases[0].X[0][0], 1000, delta=1e-3)
        self.assertAlmostEqual(phases[1].X[0][0], 0, delta=5e-2)


    def test_action_calculation(self):
        pot = potential.specific_potential({"g": 0.7, "y": 0.24, "v_GeV": 1000}, verbose=True)
        T = 15

        def V(x, T):  return pot.Vtot(x,T)
        def dV(x, T):  return pot.gradV(x,T)

        def V_(x, T=T, V=V): return V(x,T)
        def dV_(x, T=T, dV=dV): return dV(x,T)
        x0 = [0.0]
        x1 = [1000]
        fullTunneling_params = {}
        tobj = pathDeformation.fullTunneling(
                [x1,x0], V_, dV_, callback_data=T,
                **fullTunneling_params)

        # 2388.130073975295 before the path extension searched for the nearest minimum
        # instead of starting from a fixed step, and 2388.130062021543 after it: the eighth
        # significant figure, the extension landing a little differently on a path whose ends
        # are already minima.
        #
        # 2388.030513420128 since `epsabs` became a fraction of the bounce's own field
        # excursion rather than of the gap between the minima. This call takes `findProfile`'s
        # signature defaults, `phitol = xtol = 1e-4`, which no TransitionListener run uses --
        # the configured value is 1e-10. At 1e-4 the bisection's own window is what decides, and
        # this is where making it relative to the `x` it holds moves the answer most, because at
        # that tolerance neither rule has converged: `x` at the accepted trial is 0.0857, so the
        # relative window is a factor 12 the tighter and the loop takes 16 bisection trials
        # against the absolute rule's 13. The action then moves by 0.024 per cent, against
        # 3.8e-08 at the configured 1e-10, where the counts are 36 against 33. The three values
        # at 1e-4 are 2388.1301 with an absolute window and a gap-scaled epsabs, 2388.0305 with
        # an absolute window and a bounce-scaled epsabs, and 2388.6013 here, against a converged
        # 2388.1790. Looser is not closer: at the model's own 10000 spline samples the distances
        # from the converged 2384.3235 are 0.4248, 0.0390 and 0.6498. At the configured 1e-10
        # this code is the closest of the three, 2388.1790354 against 2388.1789449 and
        # 2388.1788707, which is what the test below pins.
        self.assertAlmostEqual(tobj.action, 2388.601280038287, places=5)

    def test_the_action_converges_as_the_tolerance_is_tightened(self):
        """What the number above should have been pinned against.

        A single action at one tolerance says nothing about whether it is right. The bounce has
        a value the solver is approaching, and the test of the solver is that tightening its
        tolerances walks towards that value and stays there. This configuration settles on
        2388.1790, reached monotonically: 2388.17719 at phitol = xtol = 1e-8, 2388.17894 at
        1e-10, 2388.17903 at 1e-12 and 2388.17908 at 1e-14.
        """
        pot = potential.specific_potential({"g": 0.7, "y": 0.24, "v_GeV": 1000}, verbose=False)
        T = 15
        actions = {}
        for tol in (1e-10, 1e-12, 1e-14):
            tobj = pathDeformation.fullTunneling(
                [[1000], [0.0]], lambda x: pot.Vtot(x, T), lambda x: pot.gradV(x, T),
                callback_data=T,
                tunneling_findProfile_params={"phitol": tol, "xtol": tol})
            actions[tol] = float(tobj.action)
        for tol, value in actions.items():
            self.assertAlmostEqual(
                value, 2388.179, delta=2e-3,
                msg=f"at phitol = xtol = {tol:g} the action is {value}, which is not the "
                    f"value this configuration converges to")
        self.assertLess(
            abs(actions[1e-14] - actions[1e-12]), 1e-3,
            f"the action is still moving between 1e-12 and 1e-14: {actions}")
        # The three assertions above hold under either rule, since both converge to the same
        # bounce, so on their own they are not evidence for this change. At the configured
        # tolerance the two do differ, by 7.4e-05 absolute: 2388.1789449 when `epsabs` is a
        # fraction of the bounce's own excursion and 2388.1788707 when it is a fraction of the
        # gap between the minima. A window of 3e-05 holds the first and excludes the second.
        self.assertAlmostEqual(
            actions[1e-10], 2388.1790354, delta=3e-5,
            msg=f"at the configured phitol = xtol = 1e-10 the action is {actions[1e-10]}, "
                f"which is not what this code gives; an absolute bracket gives 2388.1789449 "
                f"and a gap-scaled epsabs with an absolute bracket 2388.1788707")


if __name__ == "__main__":
    unittest.main()
