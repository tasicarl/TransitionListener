"""A decoupled radiation bath enters the Hubble rate and the redshift, not the reheating of the bubbles."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import percolation_adaptivestepsize as pas
from transitionlistener import percolation_fixedstepsize as pfs
from transitionlistener import thermodynamics as td

DELTA_V = 1.0   # vacuum energy released by the transition
A_PT = 1.0      # radiation of the transitioning sector: e = A_PT T^4
H_DS = 5.0      # its entropy degrees of freedom


class CoupledRadiationEntropyTests(unittest.TestCase):
    @staticmethod
    def pot(e_geff, p_geff):
        return types.SimpleNamespace(kin_coupled_e_geff=e_geff, kin_coupled_p_geff=p_geff, conversionFactor=1.0)

    def test_the_default_standard_model_bath_uses_the_entropy_table(self):
        pot = self.pot(td.e_geffSM, td.p_geffSM)
        for T in (0.01, 1.0, 50.0, 300.0):
            self.assertEqual(bd.h_eff_coupled_radiation(T, pot), td.s_geffSM(T, 1.0))

    def test_the_general_formula_agrees_with_the_standard_model_table(self):
        pot = self.pot(lambda T, cf: td.e_geffSM(T, cf), lambda T, cf: td.p_geffSM(T, cf))
        for T in (0.01, 1.0, 50.0, 300.0):
            self.assertAlmostEqual(bd.h_eff_coupled_radiation(T, pot) / td.s_geffSM(T, 1.0), 1.0, places=12)

    def test_no_coupled_radiation_carries_no_entropy(self):
        pot = self.pot(lambda T, cf: 0.0 * T, lambda T, cf: 0.0 * T)
        self.assertEqual(bd.h_eff_coupled_radiation(10.0, pot), 0.0)


class Step3ProfileTests(unittest.TestCase):
    """Synthetic transition: e_sym = DELTA_V + A_PT T^4, e_bro = A_PT T^4, plus a bath e_dec = b T^4."""

    def profile(self, b_decoupled):
        sym = types.SimpleNamespace(name="sym")
        bro = types.SimpleNamespace(name="bro")

        def energy_density(pot, phase, T, include_decoupled=True):
            e = (DELTA_V if phase is sym else 0.0) + A_PT * T**4
            return e + (b_decoupled * T**4 if include_decoupled else 0.0)

        TSYM = np.linspace(1.0, 0.8, 40)
        P = np.clip((1.0 - TSYM) / 0.15, 0.0, 1.0) ** 2 * 0.95
        n = TSYM.size
        state = bd.PercolationState(TSYM=TSYM, Sr=np.zeros(n), Hr=np.zeros(n), Pr=np.zeros(n), Tb=np.zeros(n))
        settings = types.SimpleNamespace(f_final=0.99, time_temperature_mode="sound_speed", integral_method="ode")
        with mock.patch.object(bd, "energyDensity", energy_density), \
                mock.patch.object(bd, "calcAction", lambda *args: 100.0), \
                mock.patch.object(bd, "h_eff_DS", lambda T, pot, phase: H_DS), \
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda T, pot: 0.0, create=True), \
                mock.patch.object(bd, "percIntegralODE_full_sweep", lambda TSYM, *a, **k: (None, P)):
            pas._compute_step3_profile(state, TSYM, P, settings, {}, types.SimpleNamespace(conversionFactor=1.0), sym, bro, 1.0, 1.0,
                                       TBROmin=0.5, TBROmax=5.0, verbose=False)
        return TSYM, P, state

    def test_the_bath_does_not_change_the_temperature_inside_the_bubbles(self):
        _, P, alone = self.profile(0.0)
        _, _, with_bath = self.profile(50.0)
        self.assertTrue(np.any(P > 0.0))
        np.testing.assert_allclose(with_bath.Tb, alone.Tb, rtol=1e-12, equal_nan=True)

    def test_the_bath_enters_the_hubble_rate_at_the_background_temperature(self):
        b = 50.0
        TSYM, P, state = self.profile(b)
        e_sym = DELTA_V + A_PT * TSYM**4
        e_bro = A_PT * np.where(np.isfinite(state.Tb), state.Tb, 0.0) ** 4
        rho = P * e_bro + (1.0 - P) * e_sym + b * TSYM**4
        np.testing.assert_allclose(state.Hr, bd.HubbleParameter(rho, 1.0), rtol=1e-12)

    def test_the_first_bubbles_reheat_to_energy_conservation_of_the_transitioning_sector(self):
        TSYM, P, state = self.profile(50.0)
        first = int(np.flatnonzero(P > 1e-12)[0])
        expected = (TSYM[first] ** 4 + DELTA_V / A_PT) ** 0.25
        self.assertAlmostEqual(state.Tb[first] / expected, 1.0, places=10)


class FixedStepProfileTests(unittest.TestCase):
    """The same synthetic transition in the fixed-step refinement reached through calcPercAndEvolve."""

    def refine(self, b_decoupled):
        sym = types.SimpleNamespace(name="sym")
        bro = types.SimpleNamespace(name="bro")

        def energy_density(pot, phase, T, include_decoupled=True):
            e = (DELTA_V if phase is sym else 0.0) + A_PT * T**4
            return e + (b_decoupled * T**4 if include_decoupled else 0.0)

        TSYM = np.linspace(1.0, 0.8, 40)
        P = np.clip((1.0 - TSYM) / 0.15, 0.0, 1.0) ** 2 * 0.999
        Tperc = float(np.interp(0.3, np.sort(P), TSYM[np.argsort(P)]))
        n = TSYM.size
        state = bd.PercolationState(TSYM=TSYM.copy(), Sr=np.zeros(n), Hr=np.zeros(n), Pr=P.copy(), Tb=np.zeros(n))
        grid = bd.PercolationGrid(TSYM=TSYM, Tstart=1.0, TpercApprox=Tperc, tmin=0.5, tmax=1.5, TBROmin=0.5, TBROmax=5.0)
        settings = types.SimpleNamespace(f_start=1e-3, f_final=0.99, f_perc=0.3, max_boundary_n=2, maxit=10,
                                         rel_increment=0.1)
        with mock.patch.object(bd, "energyDensity", energy_density), \
                mock.patch.object(pfs, "energyDensity", energy_density), \
                mock.patch.object(pfs, "calcAction", lambda *args: 100.0), \
                mock.patch.object(pfs, "percIntegral", lambda TS, H, S, vw=1.0: -np.log(1.0 - P[len(TS) - 1])), \
                mock.patch.object(bd, "h_eff_DS", lambda T, pot, phase: H_DS), \
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda T, pot: 0.0, create=True):
            _, state = pfs._refine_percolation_temperature(state, grid, settings, {}, types.SimpleNamespace(conversionFactor=1.0),
                                                           sym, bro, 1.0, 1.0, Tperc, 1e-6, False)
        return TSYM, P, state

    def test_the_bath_does_not_change_the_temperature_inside_the_bubbles(self):
        _, P, alone = self.refine(0.0)
        _, _, with_bath = self.refine(50.0)
        np.testing.assert_allclose(with_bath.Tb, alone.Tb, rtol=1e-12, equal_nan=True)

    def test_the_bath_enters_the_hubble_rate_at_the_background_temperature(self):
        b = 50.0
        TSYM, P, state = self.refine(b)
        active = np.isfinite(state.Sr) & (np.arange(TSYM.size) > 0)
        e_sym = DELTA_V + A_PT * TSYM**4
        rho = P * A_PT * state.Tb**4 + (1.0 - P) * e_sym + b * TSYM**4
        np.testing.assert_allclose(state.Hr[active], bd.HubbleParameter(rho, 1.0)[active], rtol=1e-12)


class ReheatingDegreesOfFreedomTests(unittest.TestCase):
    """g and h after reheating: transitioning sector at T_DS, SM coupled to it or decoupled at T_dec."""

    G_DS, H_DS = 4.0, 3.5
    G_SM = 90.0  # a bath of constant degrees of freedom: g_e = g_p = g gives h = g

    def pot(self, sm_decoupled):
        const = lambda value: (lambda T, cf: value + 0.0 * T)
        zero = const(0.0)
        pot = types.SimpleNamespace(
            conversionFactor=1.0, SM_bath="decoupled" if sm_decoupled else None,
            boson_massSq=lambda X, T: np.zeros(2), fermion_massSq=lambda X: np.zeros(1),
            mass_spectrum=types.SimpleNamespace(is_SM_bosons=np.zeros(2, bool), is_SM_fermions=np.zeros(1, bool)),
            kin_coupled_e_geff=zero, kin_coupled_p_geff=zero,
            kin_decoupled_e_geff=zero, kin_decoupled_p_geff=zero)
        if sm_decoupled:
            pot.kin_decoupled_e_geff, pot.kin_decoupled_p_geff = const(self.G_SM), const(self.G_SM)
        else:
            pot.kin_coupled_e_geff, pot.kin_coupled_p_geff = const(self.G_SM), const(self.G_SM)
        return pot

    def dof(self, sm_decoupled, T_DS, T_dec):
        with mock.patch.object(td, "e_geffDS", lambda b, f, T: self.G_DS), \
                mock.patch.object(td, "s_geffDS", lambda b, f, T: self.H_DS):
            return td.reheating_geff(self.pot(sm_decoupled), None, T_DS, T_dec)

    def test_a_decoupled_standard_model_gives_the_temperature_ratio_formula(self):
        # arXiv:2311.06346: g = g_SM + g_DS xi^4, h = h_SM + h_DS xi^3 with xi = T_DS/T_SM, and
        # the decoupled Standard Model is still at T_perc.
        T_DS, T_perc = 132.4, 53.9
        g, h, T_SM = self.dof(True, T_DS, T_perc)
        xi = T_DS / T_perc
        self.assertEqual(T_SM, T_perc)
        self.assertAlmostEqual(g / (self.G_SM + self.G_DS * xi**4), 1.0, places=12)
        self.assertAlmostEqual(h / (self.G_SM + self.H_DS * xi**3), 1.0, places=12)

    def test_the_redshift_does_not_depend_on_the_reference_bath(self):
        T_DS, T_perc = 132.4, 53.9
        g, h, T_SM = self.dof(True, T_DS, T_perc)
        r = T_perc / T_DS
        g_DS_ref, h_DS_ref = self.G_DS + self.G_SM * r**4, self.H_DS + self.G_SM * r**3
        self.assertAlmostEqual(T_SM * g**0.5 * h**(-1 / 3) / (T_DS * g_DS_ref**0.5 * h_DS_ref**(-1 / 3)), 1.0, places=12)
        self.assertAlmostEqual(g * h**(-4 / 3) / (g_DS_ref * h_DS_ref**(-4 / 3)), 1.0, places=12)

    def test_a_coupled_standard_model_is_reheated_with_the_transitioning_sector(self):
        g, h, T_SM = self.dof(False, 61.4, 53.9)
        self.assertEqual((g, h, T_SM), (self.G_DS + self.G_SM, self.H_DS + self.G_SM, 61.4))

    def test_the_standard_model_bath_is_detected_and_can_be_set(self):
        zero = lambda T, cf: 0.0 * T
        pot = types.SimpleNamespace(SM_bath=None, kin_decoupled_e_geff=zero)
        self.assertEqual(td.sm_bath(pot), "coupled")
        pot.kin_decoupled_e_geff = td.e_geffSM
        self.assertEqual(td.sm_bath(pot), "decoupled")
        pot.kin_decoupled_e_geff, pot.SM_bath = (lambda T, cf: td.e_geffSM(T, cf)), "decoupled"
        self.assertEqual(td.sm_bath(pot), "decoupled")
        pot.SM_bath = "photon bath"
        with self.assertRaises(ValueError):
            td.sm_bath(pot)

if __name__ == "__main__":
    unittest.main()
