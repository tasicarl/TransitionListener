import unittest
import numpy as np

from transitionlistener.thermodynamics import e_geffSM, p_geffSM, s_geffSM


class testThermodynamics(unittest.TestCase):
    """Test the unittests."""

    def test_geffSM(self):
        T = 10
        CF = 1
        e = np.pi**2/30*e_geffSM(T, CF) *T**4
        p = np.pi**2/90*p_geffSM(T, CF) * T**4
        s = 2*np.pi**2/45 * s_geffSM(T, CF) * T**3

        # Check thermodynamic relation for a heat bath
        self.assertAlmostEqual(s, (e + p)/T, delta=1e-5)

        T = 591.979845798
        CF = 3.0918
        e = np.pi**2/30*e_geffSM(T, CF) *T**4
        p = np.pi**2/90*p_geffSM(T, CF) * T**4
        s = 2*np.pi**2/45 * s_geffSM(T, CF) * T**3
        self.assertAlmostEqual(s, (e + p)/T, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
