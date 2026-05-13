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

        # Finds correct minima?
        self.assertAlmostEqual(phases[0].X[0][0], 1000, delta=1e-3)
        self.assertAlmostEqual(phases[1].X[0][0], 0, delta=1e-2)


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

        self.assertAlmostEqual(tobj.action, 2388.1300739752196, places=5)


if __name__ == "__main__":
    unittest.main()
