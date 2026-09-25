"""Guards for a broken phase with no thermal pressure left to double precision.

At percolation temperatures far below the mass scale the thermal part of the potential
underflows in the broken phase. Its enthalpy and its sound speed then come back as zero or
as NaN, which two places used to carry into an integrator or a division.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener.hydrodynamics import Hydrodynamics


class WallVelocityGuardTests(unittest.TestCase):
    """find_vw must not integrate the plasma from a non-finite initial state."""

    def _find_vw(self, *args):
        h = Hydrodynamics.__new__(Hydrodynamics)
        h.verbose = False
        return Hydrodynamics.find_vw(h, *args)

    def test_non_finite_inputs_give_a_runaway(self):
        nan = float("nan")
        cases = {
            "all of them": (nan, nan, 0.3203, -0.0),
            "alpha only": (nan, 1 / 3, 1 / 3, 0.9),
            "broken sound speed": (0.1, nan, 1 / 3, 0.9),
            "enthalpy ratio": (0.1, 1 / 3, 1 / 3, nan),
        }
        for label, args in cases.items():
            with self.subTest(label):
                self.assertEqual(self._find_vw(*args), 1)

    def test_an_empty_broken_phase_gives_a_runaway(self):
        # psi_N = w_b/w_s = 0 means there is no broken-phase plasma for the wall to push
        # against; c_b^2 = 0 is the same statement through the sound speed.
        self.assertEqual(self._find_vw(0.1, 1 / 3, 1 / 3, 0.0), 1)
        self.assertEqual(self._find_vw(0.1, 0.0, 1 / 3, 0.9), 1)

    def test_usable_inputs_still_solve(self):
        vw = self._find_vw(0.1, 1 / 3, 1 / 3, 0.9)
        self.assertTrue(0.0 < vw < 1.0)
        # A very strong transition is still the runaway branch, as before.
        self.assertEqual(self._find_vw(1e17, 1 / 3, 1 / 3, 0.0), 1)


class PseudoTraceGuardTests(unittest.TestCase):
    """calcAlphas must not divide by a sound speed that does not exist."""

    def setUp(self):
        from transitionlistener.helper_functions import load_potential
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        self.pot = load_potential(
            str(repo / "models/TL_dark_U1_g_parameterization.py"), "specific_potential"
        )({"g": 1.0, "v_GeV": 0.1, "lambda": 0.03}, verbose=False)

        class Phase:
            def __init__(self, x):
                self.x = np.array([x])

            def valAt(self, T):
                return self.x

        self.high, self.low = Phase(0.0), Phase(0.5)
        self.T = 3.4e2
        self.names = ("alpha_p", "alpha_theta", "alpha_thetabar", "alpha_e",
                      "alpha_hyd", "alpha_inf", "alpha_eq")

    def _alphas(self, broken_cs_sq):
        real = bd.calcSoundSpeedSq

        def fake(pot, X, t):
            if np.allclose(np.atleast_1d(X), self.low.valAt(t)):
                return broken_cs_sq
            return real(pot, X, t)

        with mock.patch.object(bd, "calcSoundSpeedSq", side_effect=fake):
            values = bd.calcAlphas(self.T, self.pot, self.high, self.low, verbose=False)
        return dict(zip(self.names, (float(np.squeeze(v)) for v in values)))

    def test_a_vanishing_broken_sound_speed_does_not_raise(self):
        for cs_sq in (0.0, float("nan")):
            with self.subTest(broken_cs_sq=cs_sq):
                alphas = self._alphas(cs_sq)
                # The two definitions built on the pseudo-trace are unavailable ...
                self.assertTrue(np.isnan(alphas["alpha_thetabar"]))
                self.assertTrue(np.isnan(alphas["alpha_hyd"]))
                # ... and the ones that need no sound speed survive.
                for name in ("alpha_p", "alpha_theta", "alpha_e", "alpha_inf"):
                    self.assertTrue(np.isfinite(alphas[name]), name)

    def test_a_usable_sound_speed_still_gives_the_pseudo_trace(self):
        alphas = self._alphas(1.0 / 3.0)
        self.assertTrue(np.isfinite(alphas["alpha_thetabar"]))
        self.assertTrue(np.isfinite(alphas["alpha_hyd"]))


if __name__ == "__main__":
    unittest.main()
