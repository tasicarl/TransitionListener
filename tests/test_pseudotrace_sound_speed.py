"""The pseudo-trace uses the broken-phase sound speed in both phases.

arXiv:2004.06995 defines ``theta_bar = e - p / c_s^2`` with ``c_s`` the sound speed of the
broken phase, evaluated at the temperature in front of the wall, and
``alpha_theta_bar = D theta_bar / (3 w)``. arXiv:2010.09744 writes the same as
``e - p / c_{s,b}^2`` and keeps ``c_{s,s}`` as a separate quantity that does not enter the
strength. arXiv:2206.01130, section 2, spells the difference out as
``D theta_bar(T_n) = theta_s(T_n) - theta_b(T_n)``: both phases at one temperature, both
divided by the broken-phase sound speed. Dividing each phase by its own computes something
else, and the difference is not academic: whatever is common to both phases stops
cancelling.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import bubbledynamics_fixedstep as bdf
from transitionlistener.helper_functions import load_potential
from transitionlistener.thermodynamics import e_geffSM, p_geffSM

REPO = Path(__file__).resolve().parents[1]
CONFORMAL = (REPO / "models/TL_conformal_dark_u1.py", "specific_potential",
             {"g": 0.692, "y": 0.01, "v_GeV": 0.14})
# Internal units: the zero-temperature vacuum expectation value is 1000.
TEMPERATURES = (20.0, 60.0)


class FixedPhase:
    """Phase stand-in returning the same field value at every temperature."""

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def valAt(self, T):
        return self.x


def build(decoupled_sm_bath=False):
    """The conformal model; optionally with the Standard Model as the decoupled bath."""
    path, cls, params = CONFORMAL
    base = load_potential(str(path), cls)

    class Variant(base):
        def setConfigParameters(self):
            super().setConfigParameters()
            self.config.gwConf.coupled_hydrodynamics = True
            if decoupled_sm_bath:
                self.kin_coupled_e_geff = lambda T, cf: 0.0 * T
                self.kin_coupled_p_geff = lambda T, cf: 0.0 * T
                self.kin_decoupled_e_geff = e_geffSM
                self.kin_decoupled_p_geff = p_geffSM

    with contextlib.redirect_stdout(io.StringIO()):
        return Variant(dict(params), verbose=False)


class PseudoTraceConventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pot = build()

    def pieces(self, T):
        """Energy density, pressure and both sound speeds of the two phases at ``T``."""
        pot = self.pot
        high, low = np.array([0.0]), pot.findMinimum(np.array([1000.0]), T)
        dT = T * 1e-5
        V0_ref = float(np.squeeze(pot.V0(pot.X0) + pot.Vct(pot.X0) + pot.V1_from_X(pot.X0)))
        out = {"high": high, "low": low, "V0_ref": V0_ref,
               "csSq_sym": float(np.squeeze(bd.calcSoundSpeedSq(pot, high, T))),
               "csSq_bro": float(np.squeeze(bd.calcSoundSpeedSq(pot, low, T))),
               "csSq_bro_fixed": float(np.squeeze(bdf.calcSoundSpeedSq(pot, low, T)))}
        for name, phi in (("sym", high), ("bro", low)):
            V = float(np.squeeze(pot.Vtot(phi, T, include_decoupled=False))) - V0_ref
            dVdT = float(np.squeeze(pot.dVdT(phi, T, dT=dT, include_decoupled=False)))
            out[f"p_{name}"] = -V          # pressure, with the code's reference
            # Two energy densities: the derivative of the potential, which the adaptive
            # solver uses in theta, and energyDensity(), which it uses in the enthalpy and
            # which the fixed step size solver uses for both. They differ by the daisy
            # contribution, which is a separate issue and is not touched here.
            out[f"e_{name}"] = V - T * dVdT
            out[f"eD_{name}"] = float(np.squeeze(pot.energyDensity(phi, T, include_decoupled=False)))
        return out

    def alphas(self, T, module=bd):
        pot = self.pot
        low = pot.findMinimum(np.array([1000.0]), T)
        return module.calcAlphas(T, pot, FixedPhase([0.0]), FixedPhase(low))

    def test_the_difference_is_De_minus_Dp_over_the_broken_sound_speed(self):
        """alpha_hyd must equal (De - Dp / c_{s,b}^2) / (3 w) of the transitioning sector."""
        for T in TEMPERATURES:
            with self.subTest(T=T):
                q = self.pieces(T)
                De = q["e_sym"] - q["e_bro"]
                Dp = q["p_sym"] - q["p_bro"]
                Dtheta = De - Dp / q["csSq_bro"]
                w_sym = q["eD_sym"] + q["p_sym"]
                alpha_hyd = float(np.squeeze(self.alphas(T)[4]))
                self.assertTrue(np.isclose(alpha_hyd, Dtheta / (3 * w_sym), rtol=1e-10, atol=0),
                                f"alpha_hyd {alpha_hyd} vs reconstruction {Dtheta / (3 * w_sym)}")

    def test_it_does_not_depend_on_where_the_zero_of_the_potential_is_put(self):
        """A constant in the potential shifts e by +C and p by -C in both phases alike.

        Which vacuum is called zero is a convention; arXiv:2004.06995 notes that the
        hydrodynamics cannot depend on it. With one sound speed the constant cancels in
        De and Dp; with one sound speed per phase it survives as C (1/cs_s^2 - 1/cs_b^2).
        """
        for T in TEMPERATURES:
            q = self.pieces(T)
            De = q["e_sym"] - q["e_bro"]
            Dp = q["p_sym"] - q["p_bro"]
            reference = De - Dp / q["csSq_bro"]
            for factor in (1.0, -10.0):
                C = factor * abs(q["p_sym"] - q["p_bro"])
                with self.subTest(T=T, C=C):
                    shifted = ((q["e_sym"] + C) - (q["e_bro"] + C)) \
                        - ((q["p_sym"] - C) - (q["p_bro"] - C)) / q["csSq_bro"]
                    self.assertTrue(np.isclose(shifted, reference, rtol=1e-12, atol=0))
                    # the superseded convention does depend on it
                    two_speeds = ((q["e_sym"] + C) - (q["p_sym"] - C) / q["csSq_sym"]) \
                        - ((q["e_bro"] + C) - (q["p_bro"] - C) / q["csSq_bro"])
                    two_speeds_0 = (q["e_sym"] - q["p_sym"] / q["csSq_sym"]) \
                        - (q["e_bro"] - q["p_bro"] / q["csSq_bro"])
                    self.assertFalse(np.isclose(two_speeds, two_speeds_0, rtol=1e-6, atol=0))

    def test_the_change_is_exactly_the_symmetric_phase_pressure_term(self):
        """New minus old pseudo-trace difference = -p_sym (1/cs_b^2 - 1/cs_s^2)."""
        for T in TEMPERATURES:
            with self.subTest(T=T):
                q = self.pieces(T)
                new = (q["e_sym"] - q["p_sym"] / q["csSq_bro"]) \
                    - (q["e_bro"] - q["p_bro"] / q["csSq_bro"])
                old = (q["e_sym"] - q["p_sym"] / q["csSq_sym"]) \
                    - (q["e_bro"] - q["p_bro"] / q["csSq_bro"])
                expected = -q["p_sym"] * (1 / q["csSq_bro"] - 1 / q["csSq_sym"])
                self.assertTrue(np.isclose(new - old, expected, rtol=1e-10, atol=0))

    def test_the_bag_limit_is_the_trace_anomaly(self):
        """With both sound speeds at 1/3 the pseudo-trace is the trace anomaly e - 3p."""
        for T in TEMPERATURES:
            with self.subTest(T=T):
                q = self.pieces(T)
                bag = (q["e_sym"] - 3 * q["p_sym"]) - (q["e_bro"] - 3 * q["p_bro"])
                one_third = (q["e_sym"] - q["p_sym"] / (1 / 3)) \
                    - (q["e_bro"] - q["p_bro"] / (1 / 3))
                self.assertTrue(np.isclose(bag, one_third, rtol=1e-12, atol=0))

    def test_the_bath_enters_only_through_the_broken_phase_sound_speed(self):
        """Relabelling the Standard Model changes the pseudo-trace only through c_{s,b}.

        A radiation bath contributes the same energy and pressure to both phases, so it
        cancels in De and Dp whatever its label. What it does change is the sound speed of
        the plasma: taking the Standard Model out of the transitioning sector changes
        c_{s,b}^2 by about 12 percent at T = 20. The strength follows, and should.
        """
        for T in TEMPERATURES:
            with self.subTest(T=T):
                De, Dp, thetas = {}, {}, {}
                for label in ("coupled", "decoupled"):
                    pot = build(decoupled_sm_bath=(label == "decoupled"))
                    high, low = np.array([0.0]), pot.findMinimum(np.array([1000.0]), T)
                    dT = T * 1e-5
                    ref = float(np.squeeze(pot.V0(pot.X0) + pot.Vct(pot.X0)
                                           + pot.V1_from_X(pot.X0)))
                    e, p = {}, {}
                    for name, phi in (("sym", high), ("bro", low)):
                        V = float(np.squeeze(pot.Vtot(phi, T, include_decoupled=False))) - ref
                        dVdT = float(np.squeeze(pot.dVdT(phi, T, dT=dT,
                                                         include_decoupled=False)))
                        p[name], e[name] = -V, V - T * dVdT
                    De[label] = e["sym"] - e["bro"]
                    Dp[label] = p["sym"] - p["bro"]
                    csSq_bro = float(np.squeeze(bd.calcSoundSpeedSq(pot, low, T)))
                    thetas[label] = (De[label] - Dp[label] / csSq_bro, csSq_bro)
                # the bath itself cancels in both differences
                self.assertTrue(np.isclose(De["coupled"], De["decoupled"], rtol=1e-8, atol=0))
                self.assertTrue(np.isclose(Dp["coupled"], Dp["decoupled"], rtol=1e-8, atol=0))
                # and the whole change of the pseudo-trace is the sound speed
                predicted = De["coupled"] - Dp["coupled"] / thetas["decoupled"][1]
                self.assertTrue(np.isclose(predicted, thetas["decoupled"][0],
                                           rtol=1e-8, atol=0))

    def test_both_solvers_use_the_same_convention(self):
        """The fixed step size solver must divide by the broken-phase sound speed too."""
        for T in TEMPERATURES:
            with self.subTest(T=T):
                q = self.pieces(T)
                # the fixed-step solver builds theta from energyDensity() rather than from
                # -T dV/dT, so it is reconstructed with the same energy density here
                De = q["eD_sym"] - q["eD_bro"]
                Dp = q["p_sym"] - q["p_bro"]
                w_sym = q["eD_sym"] + q["p_sym"]
                # that solver has its own finite-difference step in calcSoundSpeedSq
                expected = (De - Dp / q["csSq_bro_fixed"]) / (3 * w_sym)
                alpha_hyd_fixed = float(np.squeeze(self.alphas(T, module=bdf)[3]))
                self.assertTrue(np.isclose(alpha_hyd_fixed, expected, rtol=1e-10, atol=0),
                                f"fixed-step alpha_hyd {alpha_hyd_fixed} vs {expected}")


if __name__ == "__main__":
    unittest.main()
