"""A decoupled radiation bath enters the Hubble rate and the redshift, not the reheating of the bubbles."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

import contextlib
import io

from transitionlistener import bubbledynamics as bd
from transitionlistener import bubbledynamics_fixedstep as bdf
from transitionlistener import percolation_adaptivestepsize as pas
from transitionlistener import percolation_fixedstepsize as pfs
from transitionlistener import constants as cn
from transitionlistener.generic_potential import generic_potential
from transitionlistener.gwfopt import FOPTspectrum
from transitionlistener.helper_functions import load_potential
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

    phases = (types.SimpleNamespace(name="sym"), types.SimpleNamespace(name="bro"))

    def energy_density(self, b_decoupled):
        sym = self.phases[0]

        def energy_density(pot, phase, T, include_decoupled=True):
            e = (DELTA_V if phase is sym else 0.0) + A_PT * T**4
            return e + (b_decoupled * T**4 if include_decoupled else 0.0)
        return energy_density

    def profile(self, b_decoupled):
        sym, bro = self.phases
        energy_density = self.energy_density(b_decoupled)

        TSYM = np.linspace(1.0, 0.8, 40)
        P = np.clip((1.0 - TSYM) / 0.15, 0.0, 1.0) ** 2 * 0.95
        n = TSYM.size
        state = bd.PercolationState(TSYM=TSYM, Sr=np.zeros(n), Hr=np.zeros(n), Pr=np.zeros(n), Tb=np.zeros(n))
        settings = types.SimpleNamespace(f_final=0.99, time_temperature_mode="sound_speed", integral_method="ode")
        with mock.patch.object(bd, "energyDensity", energy_density), \
                mock.patch.object(bd, "calcAction", lambda *args: 100.0), \
                mock.patch.object(bd, "h_eff_DS", lambda T, pot, phase: H_DS), \
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda T, pot: 0.0), \
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

    def test_the_self_consistency_target_matches_the_profile(self):
        # The step-3 residual compares 3 M_Pl^2 H^2 / (8 pi) with the energy density the
        # profile was built from; with the bath counted once at T it vanishes.
        b = 50.0
        TSYM, P, state = self.profile(b)
        sym, bro = self.phases
        with mock.patch.object(bd, "energyDensity", self.energy_density(b)):
            target = pas._profile_energy_density(None, sym, bro, TSYM, np.where(np.isfinite(state.Tb), state.Tb, TSYM), P)
        rho_from_h = 3.0 * state.Hr**2 * (cn.Mpl_GeV / 1.0) ** 2 / (8.0 * np.pi)
        np.testing.assert_allclose(rho_from_h, target, rtol=1e-12)

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
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda T, pot: 0.0):
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


class PipelineFixedStepProfileTests(unittest.TestCase):
    """The synthetic transition in the fixed step size solver of the pipeline (bubbledynamics_fixedstep)."""

    def refine(self, b_decoupled):
        sym = types.SimpleNamespace(name="sym", valAt=lambda T: np.zeros(1))
        bro = types.SimpleNamespace(name="bro", valAt=lambda T: np.zeros(1))

        def energy_density(pot, phase, T, include_decoupled=True):
            e = (DELTA_V if phase is sym else 0.0) + A_PT * T**4
            return e + (b_decoupled * T**4 if include_decoupled else 0.0)

        TSYM = np.linspace(1.0, 0.8, 40)
        P = np.clip((1.0 - TSYM) / 0.15, 0.0, 1.0) ** 2 * 0.999
        Tperc = float(np.interp(0.3, np.sort(P), TSYM[np.argsort(P)]))
        n = TSYM.size
        state = bdf.PercolationState(TSYM=TSYM.copy(), Sr=np.zeros(n), Hr=np.zeros(n), Pr=P.copy(), Pr_exp=np.zeros(n),
                                     scalef_ratio=np.zeros(n), soundSpSq=np.zeros(n), Tb=np.zeros(n))
        grid = bdf.PercolationGrid(TSYM=TSYM, TpercApprox=Tperc, tmin=0.5, tmax=1.5, TBROmin=0.5, TBROmax=5.0)
        settings = bdf.PercolationSettings(f_perc=0.3, f_start=1e-3, f_final=0.99, weight=1.0, maxit=10,
                                           rel_increment=0.1, max_boundary_n=2)
        integral = lambda TS, *args, **kwargs: -np.log(1.0 - P[len(TS) - 1])
        with mock.patch.object(bdf, "energyDensity", energy_density), \
                mock.patch.object(bdf, "calcAction", lambda *args: 100.0), \
                mock.patch.object(bdf, "percIntegral", integral), \
                mock.patch.object(bdf, "percIntegralwExp", integral), \
                mock.patch.object(bdf, "calcSoundSpeedSq", lambda *args: 1.0 / 3.0), \
                mock.patch.object(bdf, "scalefactorRatio", lambda *args: 1.0), \
                mock.patch.object(bdf, "h_eff_DS", lambda T, pot, phase: H_DS), \
                mock.patch.object(bdf, "h_eff_coupled_radiation", lambda T, pot: 0.0):
            _, state = bdf._refine_percolation_temperature(state, grid, settings, {}, types.SimpleNamespace(conversionFactor=1.0),
                                                           sym, bro, 1.0, 1.0, Tperc, 1e-6, False)
        return TSYM, P, state

    def test_the_bath_does_not_change_the_temperature_inside_the_bubbles(self):
        _, P, alone = self.refine(0.0)
        _, _, with_bath = self.refine(50.0)
        self.assertTrue(np.any(P > 0.0))
        np.testing.assert_allclose(with_bath.Tb, alone.Tb, rtol=1e-12)

    def test_the_bath_enters_the_hubble_rate_at_the_background_temperature(self):
        b = 50.0
        TSYM, P, state = self.refine(b)
        active = np.isfinite(state.Sr) & (np.arange(TSYM.size) > 0)
        rho = P * A_PT * state.Tb**4 + (1.0 - P) * (DELTA_V + A_PT * TSYM**4) + b * TSYM**4
        np.testing.assert_allclose(state.Hr[active], bd.HubbleParameter(rho, 1.0)[active], rtol=1e-12)


class ReheatingTemperatureTests(unittest.TestCase):
    """integrate_broken_temperature follows the transitioning sector only."""

    def treh(self, b_decoupled):
        sym = types.SimpleNamespace(name="sym", Tmin=0.5, Tmax=5.0)
        bro = types.SimpleNamespace(name="bro", Tmin=0.5, Tmax=5.0)

        def energy_density(pot, phase, T, include_decoupled=True):
            e = (DELTA_V if phase is sym else 0.0) + A_PT * T**4
            return e + (b_decoupled * T**4 if include_decoupled else 0.0)

        TSYM = np.linspace(1.0, 0.8, 40)
        P = np.clip((1.0 - TSYM) / 0.15, 0.0, 1.0) ** 2 * 0.999
        with mock.patch.object(bd, "energyDensity", energy_density), \
                mock.patch.object(bd, "h_eff_DS", lambda T, pot, phase: H_DS), \
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda T, pot: 0.0):
            return bd.integrate_broken_temperature(None, sym, bro, TSYM, P, 0.9)

    def test_the_bath_does_not_change_the_reheating_temperature(self):
        alone, with_bath = self.treh(0.0), self.treh(50.0)
        self.assertIsNotNone(alone)
        self.assertAlmostEqual(with_bath / alone, 1.0, places=12)


class SpectrumAndModelTests(unittest.TestCase):
    GW = dict(alpha=0.5, RH=1e-2, Treh_SM_GeV=0.05, g_eff_tot_reh=10.0, h_eff_tot_reh=10.0,
              kappa_phi=0.0, kappa_sw=0.3, kappa_turb=0.03, v_wall=0.9)

    def peak(self, D):
        spec = FOPTspectrum(dict(self.GW, D=D), {}, verbose=False)
        f = np.logspace(-12, -4, 8001)
        h2O = np.array([spec.h2Omega_0_sum(x) for x in f])
        i = int(np.argmax(h2O))
        self.assertTrue(0 < i < f.size - 1, "the peak must lie inside the frequency grid")
        return f[i], h2O[i]

    def test_dilution_shifts_frequencies_by_the_third_and_amplitudes_by_the_four_thirds_power(self):
        f1, A1 = self.peak(1.0)
        f8, A8 = self.peak(8.0)
        self.assertAlmostEqual(f8 / f1, 0.5, delta=0.5 * 0.0024)   # one step of the frequency grid, 10^(1/1000)
        self.assertAlmostEqual(A8 / A1, 8.0 ** (-4.0 / 3.0), places=4)

    def test_the_bsmpt_energy_density_includes_a_decoupled_bath(self):
        pot = load_potential("models/TL_2HDM_BSMPT.py", "R2HDM")(
            {"lambda1": 0.28894, "lambda2": 0.26237, "lambda3": 5.7702, "lambda4": -2.17494, "lambda5": -2.2417,
             "m12_sq_GeV2": 1573.171, "tan_beta": 21.3949, "yukawa_type": 1, "v_GeV": 246.22})
        X, T = np.zeros((3, pot.Ndim)), np.linspace(20.0, 120.0, 3) / pot.conversionFactor
        default = pot.radiationEnergyDensity(X, T)
        np.testing.assert_array_equal(default, pot.radiationEnergyDensity(X, T, include_decoupled=False))
        pot.kin_decoupled_e_geff = td.e_geffSM
        added = pot.radiationEnergyDensity(X, T) - default
        np.testing.assert_allclose(added, np.pi**2 / 30 * td.e_geffSM(T, pot.conversionFactor) * T**4, rtol=1e-12)

    def test_a_standard_model_in_both_baths_is_flagged(self):
        pot = types.SimpleNamespace(SM_bath=None, kin_coupled_e_geff=td.e_geffSM, kin_decoupled_e_geff=td.e_geffSM)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            generic_potential._check_radiation_baths(pot)
        self.assertIn("counted twice", out.getvalue())
        pot.kin_coupled_e_geff = lambda T, cf: 0.0 * T
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            generic_potential._check_radiation_baths(pot)
        self.assertEqual(out.getvalue(), "")


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

    def test_standard_model_fields_of_the_potential_count(self):
        # The SM tables are capped below the SM fields of the potential (e.g. W, Z, t, h in
        # the 2HDM), so those must be counted from the potential, not masked out.
        pot = self.pot(False)
        pot.mass_spectrum = types.SimpleNamespace(is_SM_bosons=np.array([True, False]), is_SM_fermions=np.array([True]))
        pot.boson_massSq = lambda X, T: np.array([7.0, 1.0])
        pot.fermion_massSq = lambda X: np.array([3.0])
        with mock.patch.object(td, "e_geffDS", lambda b, f, T: float(np.sum(b) + np.sum(f))), \
                mock.patch.object(td, "s_geffDS", lambda b, f, T: float(np.sum(b) + np.sum(f))):
            g, h, _ = td.reheating_geff(pot, None, 61.4, 53.9)
        self.assertEqual((g, h), (11.0 + self.G_SM, 11.0 + self.G_SM))

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
