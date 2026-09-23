"""The pseudo-trace uses the broken-phase sound speed in both phases.

arXiv:2004.06995 eq. (2.13) with the definition below its eq. (2.11), arXiv:2010.09744, and
arXiv:2206.01130 sec. 2, which writes ``D theta_bar(T_n) = theta_s(T_n) - theta_b(T_n)``
with ``theta_bar = e - p / c_{s,b}^2``: both phases at one temperature, both divided by the
broken-phase sound speed. Dividing each phase by its own computes something else, in which
whatever the two phases share stops cancelling. In the bag limit, both sound speeds at 1/3,
the two forms coincide, so the test temperatures are checked to have sound speeds that
differ. Every test compares against the production ``calcAlphas``.
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


def pieces(pot, T):
    """Energy density, pressure and both sound speeds of the two phases at ``T``."""
    high, low = np.array([0.0]), pot.findMinimum(np.array([1000.0]), T)
    dT = T * 1e-5
    ref = float(np.squeeze(pot.V0(pot.X0) + pot.Vct(pot.X0) + pot.V1_from_X(pot.X0)))
    out = {"high": high, "low": low,
           "csSq_sym": float(np.squeeze(bd.calcSoundSpeedSq(pot, high, T))),
           "csSq_bro": float(np.squeeze(bd.calcSoundSpeedSq(pot, low, T))),
           "csSq_bro_fixed": float(np.squeeze(bdf.calcSoundSpeedSq(pot, low, T)))}
    for name, phi in (("sym", high), ("bro", low)):
        V = float(np.squeeze(pot.Vtot(phi, T, include_decoupled=False))) - ref
        dVdT = float(np.squeeze(pot.dVdT(phi, T, dT=dT, include_decoupled=False)))
        out[f"p_{name}"] = -V              # pressure, with the code's reference
        out[f"e_{name}"] = V - T * dVdT    # what the adaptive solver puts in theta
        # energyDensity() differs from that by the daisy term, a separate issue; the
        # adaptive solver uses it for the enthalpy, the fixed step size solver for both
        out[f"eD_{name}"] = float(np.squeeze(pot.energyDensity(phi, T,
                                                              include_decoupled=False)))
    return out


def alphas(pot, T, module=bd, **kwargs):
    low = pot.findMinimum(np.array([1000.0]), T)
    return module.calcAlphas(T, pot, FixedPhase([0.0]), FixedPhase(low), **kwargs)


def produced_Dtheta(pot, T, index=4):
    """The pseudo-trace difference behind the strength that calcAlphas returns."""
    q = pieces(pot, T)
    w_sym = q["eD_sym"] + q["p_sym"]
    return 3 * w_sym * float(np.squeeze(alphas(pot, T)[index])), q


class PseudoTraceConventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pot = build()

    def test_the_two_sound_speeds_differ_at_the_test_temperatures(self):
        """Otherwise every test below would pass under either convention."""
        for T in TEMPERATURES:
            with self.subTest(T=T):
                q = pieces(self.pot, T)
                self.assertGreater(abs(q["csSq_bro"] / q["csSq_sym"] - 1), 5e-3)

    def test_alpha_hyd_matches_the_definition(self):
        """alpha_hyd == (De - Dp / c_{s,b}^2) / (3 w)."""
        for T in TEMPERATURES:
            with self.subTest(T=T):
                q = pieces(self.pot, T)
                De, Dp = q["e_sym"] - q["e_bro"], q["p_sym"] - q["p_bro"]
                w_sym = q["eD_sym"] + q["p_sym"]
                expected = (De - Dp / q["csSq_bro"]) / (3 * w_sym)
                got = float(np.squeeze(alphas(self.pot, T)[4]))
                self.assertTrue(np.isclose(got, expected, rtol=1e-10, atol=0),
                                f"alpha_hyd {got} vs {expected}")

    def test_the_fixed_step_solver_uses_the_same_convention(self):
        for T in TEMPERATURES:
            with self.subTest(T=T):
                q = pieces(self.pot, T)
                De, Dp = q["eD_sym"] - q["eD_bro"], q["p_sym"] - q["p_bro"]
                w_sym = q["eD_sym"] + q["p_sym"]
                # that solver has its own finite-difference step for the sound speed
                expected = (De - Dp / q["csSq_bro_fixed"]) / (3 * w_sym)
                got = float(np.squeeze(alphas(self.pot, T, module=bdf)[3]))
                self.assertTrue(np.isclose(got, expected, rtol=1e-10, atol=0),
                                f"fixed-step alpha_hyd {got} vs {expected}")

    def test_the_difference_to_the_superseded_convention(self):
        """A sound speed per phase would shift it by -p_sym (1/cs_b^2 - 1/cs_s^2)."""
        for T in TEMPERATURES:
            with self.subTest(T=T):
                produced, q = produced_Dtheta(self.pot, T)
                superseded = (q["e_sym"] - q["p_sym"] / q["csSq_sym"]) \
                    - (q["e_bro"] - q["p_bro"] / q["csSq_bro"])
                expected = -q["p_sym"] * (1 / q["csSq_bro"] - 1 / q["csSq_sym"])
                self.assertTrue(np.isclose(produced - superseded, expected,
                                           rtol=1e-8, atol=0))

    def test_it_does_not_depend_on_the_zero_point_of_the_potential(self):
        """A constant in the potential raises e and lowers p equally in both phases.

        It cancels in De and Dp, so the produced strength is reproduced with a shifted
        reference too. With a sound speed per phase it would not be.
        """
        for T in TEMPERATURES:
            produced, q = produced_Dtheta(self.pot, T)
            superseded = (q["e_sym"] - q["p_sym"] / q["csSq_sym"]) \
                - (q["e_bro"] - q["p_bro"] / q["csSq_bro"])
            for factor in (1.0, -10.0):
                C = factor * abs(q["p_sym"] - q["p_bro"])
                with self.subTest(T=T, C=C):
                    shifted = ((q["e_sym"] + C) - (q["e_bro"] + C)) \
                        - ((q["p_sym"] - C) - (q["p_bro"] - C)) / q["csSq_bro"]
                    self.assertTrue(np.isclose(produced, shifted, rtol=1e-10, atol=0))
                    # the superseded form moves with the offset, by C (1/cs_s^2 - 1/cs_b^2)
                    superseded_shifted = ((q["e_sym"] + C) - (q["p_sym"] - C) / q["csSq_sym"]) \
                        - ((q["e_bro"] + C) - (q["p_bro"] - C) / q["csSq_bro"])
                    self.assertFalse(np.isclose(superseded_shifted, superseded, rtol=1e-6, atol=0))
                    self.assertTrue(np.isclose(
                        superseded_shifted - superseded,
                        C * (1 / q["csSq_sym"] - 1 / q["csSq_bro"]), rtol=1e-8, atol=0))

    def test_the_bath_enters_only_through_the_broken_phase_sound_speed(self):
        """Moving the Standard Model to the decoupled bath changes c_{s,b}, nothing else.

        A bath contributes the same energy and pressure to both phases, so it cancels in
        De and Dp. It is not reheated inside the bubbles, so it leaves the broken phase's
        equation of state and the sound speed changes; the strength follows.
        """
        for T in TEMPERATURES:
            with self.subTest(T=T):
                q, produced = {}, {}
                for label, decoupled in (("coupled", False), ("decoupled", True)):
                    pot = build(decoupled_sm_bath=decoupled)
                    q[label] = pieces(pot, T)
                    w = q[label]["eD_sym"] + q[label]["p_sym"]
                    produced[label] = 3 * w * float(np.squeeze(
                        alphas(pot, T, return_wall_strength=True)[7]))
                De = q["coupled"]["e_sym"] - q["coupled"]["e_bro"]
                Dp = q["coupled"]["p_sym"] - q["coupled"]["p_bro"]
                self.assertTrue(np.isclose(
                    De, q["decoupled"]["e_sym"] - q["decoupled"]["e_bro"], rtol=1e-8, atol=0))
                self.assertTrue(np.isclose(
                    Dp, q["decoupled"]["p_sym"] - q["decoupled"]["p_bro"], rtol=1e-8, atol=0))
                self.assertTrue(np.isclose(produced["decoupled"],
                                           De - Dp / q["decoupled"]["csSq_bro"],
                                           rtol=1e-8, atol=0))


if __name__ == "__main__":
    unittest.main()
