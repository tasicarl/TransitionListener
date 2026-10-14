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
        # the configured value is 1e-10 -- and at 1e-4 the two differ by 4e-5 relative. The
        # test below says which of them is closer to the answer: both converge to 2388.179 as
        # the tolerance is tightened, and at the configured 1e-10 this code gives 2388.1789
        # against 2388.1729 before.
        self.assertAlmostEqual(tobj.action, 2388.030513420128, places=5)

    def test_the_action_converges_as_the_tolerance_is_tightened(self):
        """What the number above should have been pinned against.

        A single action at one tolerance says nothing about whether it is right. The bounce has
        a value the solver is approaching, and the test of the solver is that tightening its
        tolerances walks towards that value and stays there. 2388.179 is where this
        configuration settles, and it is reached from 1e-12 onwards.
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


if __name__ == "__main__":
    unittest.main()
