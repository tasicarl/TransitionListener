"""Normalisation of the transition strengths that drive the bubble wall and the sound waves."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from transitionlistener import bubbledynamics, bubbledynamics_fixedstep
from transitionlistener.config import GWConf
from transitionlistener.helper_functions import load_potential
from transitionlistener.hydrodynamics import calc_kappas
from transitionlistener.thermodynamics import e_geffSM, p_geffSM

REPO = Path(__file__).resolve().parents[1]
CONFORMAL = (REPO / "models/TL_conformal_dark_u1.py", "specific_potential",
             {"g": 0.692, "y": 0.01, "v_GeV": 0.14})
# Temperatures in internal units (vacuum expectation value 1000): strongly and mildly supercooled.
TEMPERATURES = (20.0, 60.0)


class FixedPhase:
    """Phase stand-in that returns the same field value at every temperature."""

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def valAt(self, T):
        return self.x


def build(decoupled_sm_bath=False, coupled_hydrodynamics=True):
    """Conformal model; optionally with the Standard Model radiation as a decoupled bath."""
    path, cls, params = CONFORMAL
    base = load_potential(str(path), cls)

    class Variant(base):
        def setConfigParameters(self):
            super().setConfigParameters()
            self.config.gwConf.coupled_hydrodynamics = coupled_hydrodynamics
            if decoupled_sm_bath:
                self.kin_coupled_e_geff = lambda T, cf: 0.0 * T
                self.kin_coupled_p_geff = lambda T, cf: 0.0 * T
                self.kin_decoupled_e_geff = e_geffSM
                self.kin_decoupled_p_geff = p_geffSM

    return Variant(dict(params), verbose=False)


def alphas(pot, T, module=bubbledynamics, **kwargs):
    broken = pot.findMinimum(np.array([1000.0]), T)
    return module.calcAlphas(T, pot, FixedPhase([0.0]), FixedPhase(broken), **kwargs)


class CalcAlphasTests(unittest.TestCase):
    def test_wall_strength_is_opt_in(self):
        pot = build()
        for T in TEMPERATURES:
            with self.subTest(T=T):
                default = alphas(pot, T)
                extended = alphas(pot, T, return_wall_strength=True)
                self.assertEqual(len(default), 7)
                self.assertEqual(len(extended), 8)
                self.assertEqual(extended[:7], default)
                # Without a decoupled bath both strengths coincide.
                self.assertEqual(extended[7], default[4])

    def test_decoupled_bath_enters_only_the_sound_wave_strength(self):
        coupled = build(decoupled_sm_bath=True, coupled_hydrodynamics=True)
        uncoupled = build(decoupled_sm_bath=True, coupled_hydrodynamics=False)
        for T in TEMPERATURES:
            with self.subTest(T=T):
                hyd_c, inf_c, eq_c, wall_c = alphas(coupled, T, return_wall_strength=True)[4:]
                hyd_u, inf_u, eq_u, wall_u = alphas(uncoupled, T, return_wall_strength=True)[4:]
                # The wall is pushed by the transitioning sector alone.
                self.assertEqual(wall_c, wall_u)
                self.assertEqual((inf_c, eq_c), (inf_u, eq_u))
                self.assertEqual(hyd_u, wall_u)
                # Hydrodynamically coupled: the SM bath dilutes the sound-wave strength.
                self.assertLess(hyd_c, 0.5 * wall_c)

    def test_friction_pressures_are_independent_of_bath_assignment(self):
        """Radiation that does not change mass across the wall exerts no pressure on it.

        alpha_inf and alpha_eq are the leading- and next-to-leading-order friction pressures
        divided by the same enthalpy. Their ratio is a ratio of pressures on the wall, and
        the Standard Model radiation contributes none of it either way, so moving it to the
        decoupled bath must leave that ratio alone. Before the normalisation was fixed,
        alpha_inf and alpha_eq were normalised differently from alpha and gamma_eq changed
        by a factor 2.4 at T = 20.

        The wall strength alpha_hyd_wall is NOT invariant, and should not be. Moving the
        Standard Model to the decoupled bath is not a pure relabelling: a decoupled bath is
        not in thermal contact with the fields of the potential and is not reheated inside
        the bubbles, so it is no longer part of the broken phase's equation of state. That
        changes the broken-phase sound speed by 12 % at T = 20, and the pseudo-trace
        e - p / c_{s,b}^2 depends on it by construction. The dependence is checked
        quantitatively in test_pseudotrace_sound_speed.py.
        """
        reference = build()
        relabelled = build(decoupled_sm_bath=True, coupled_hydrodynamics=True)
        for T in TEMPERATURES:
            with self.subTest(T=T):
                a = alphas(reference, T, return_wall_strength=True)
                b = alphas(relabelled, T, return_wall_strength=True)
                self.assertTrue(np.isclose(a[5] / a[6], b[5] / b[6], rtol=1e-10, atol=0),
                                "alpha_inf / alpha_eq must not depend on the bath label")

    def test_friction_pressure_ratio(self):
        # alpha_inf / alpha_eq = Delta P_LO / Delta P_NLO with Delta P_LO = sum_i c_i N_i Delta m_i^2 T^2 / 24
        # (c_i = 1 for bosons, 1/2 for fermions) and Delta P_NLO = sum_V g_V^2 N_V Delta m_V T^3,
        # eqs. (2.6)-(2.8) of arXiv:1903.09642. The ratio does not depend on the normalisation.
        pot = build()
        for T in TEMPERATURES:
            with self.subTest(T=T):
                broken = pot.findMinimum(np.array([1000.0]), T)
                m2_b_bro, n_b, _, physical = pot.boson_massSq(broken, 0)
                m2_b_sym = pot.boson_massSq(np.array([0.0]), 0)[0]
                m2_f_bro, n_f = pot.fermion_massSq(broken)
                m2_f_sym = pot.fermion_massSq(np.array([0.0]))[0]
                delta_m2 = (np.sum(n_b * physical * np.maximum(m2_b_bro - m2_b_sym, 0))
                            + 0.5 * np.sum(n_f * np.maximum(m2_f_bro - m2_f_sym, 0)))
                delta_m_v = np.maximum(np.sqrt(np.maximum(m2_b_bro, 0)) - np.sqrt(np.maximum(m2_b_sym, 0)), 0)
                g2_delta_m_v = np.sum(pot.mass_spectrum.boson_gauge_couplings**2 * n_b * physical * delta_m_v)
                alpha_inf, alpha_eq = alphas(pot, T)[5:7]
                self.assertTrue(np.isclose(alpha_inf / alpha_eq, delta_m2 * T**2 / 24 / (g2_delta_m_v * T**3),
                                           rtol=1e-10, atol=0))

    def test_fixed_step_solver_uses_the_same_normalisation(self):
        pot = build()
        for T in TEMPERATURES:
            with self.subTest(T=T):
                adaptive = alphas(pot, T, return_wall_strength=True)
                fixed = alphas(pot, T, module=bubbledynamics_fixedstep)
                self.assertTrue(np.allclose(fixed[4:6], adaptive[5:7], rtol=1e-12, atol=0))


class CalcKappasTests(unittest.TestCase):
    ARGS = dict(alpha_inf=0.01, alpha_eq=1e-3, vw=0.95, cs=1 / np.sqrt(3), Rsep=1e3, R0=1.0)

    @staticmethod
    def config(mode):
        config = GWConf()
        config.bw_collisions = mode
        return config

    def test_alpha_wall_defaults_to_alphaN(self):
        for mode in ("off", "full", "NLO"):
            with self.subTest(mode=mode):
                config = self.config(mode)
                # "full" gives alpha_eff = 0 and a NaN kappa_sw, which assert_equal treats as equal.
                np.testing.assert_equal(calc_kappas(0.5, **self.ARGS, config=config),
                                        calc_kappas(0.5, **self.ARGS, config=config, alpha_wall=0.5))

    def test_alpha_wall_sets_the_collision_efficiency(self):
        config = self.config("NLO")
        kappa_phi, kappa_sw, _ = calc_kappas(0.5, **self.ARGS, config=config, alpha_wall=1.0)
        # gamma_eq = (1.0 - 0.01) / 1e-3 = 990 exceeds gamma_star = 2 Rsep / (3 R0) = 667.
        self.assertAlmostEqual(kappa_phi, 1 - 0.01 / 1.0, places=12)
        # The sound-wave strength alphaN does not enter kappa_phi.
        self.assertEqual(calc_kappas(0.2, **self.ARGS, config=config, alpha_wall=1.0)[0], kappa_phi)
        # It does enter kappa_sw, through alpha_eff = alphaN (1 - kappa_phi).
        self.assertNotEqual(calc_kappas(0.2, **self.ARGS, config=config, alpha_wall=1.0)[1], kappa_sw)


if __name__ == "__main__":
    unittest.main()
