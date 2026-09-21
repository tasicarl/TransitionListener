"""A decoupled radiation bath enters the Hubble rate and the redshift, not the reheating of the bubbles."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

import contextlib
import importlib.util
import inspect
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


class TimeTemperatureFactorsTests(unittest.TestCase):
    """The scale factor of the percolation integral follows entropy conservation."""

    @staticmethod
    def bath(masses, dofs, gauge_bosons=0):
        """A potential whose modes have the given masses and degrees of freedom."""
        masses = np.asarray(masses, dtype=float)
        dofs = np.asarray(dofs, dtype=float)
        zero = lambda T, cf: 0.0 * np.asarray(T, dtype=float)
        return types.SimpleNamespace(
            conversionFactor=1.0, T_eps=1.0e-12, X0=np.zeros(1),
            boson_massSq=lambda X, T: (masses**2, dofs, np.zeros_like(dofs),
                                       np.ones_like(dofs, dtype=bool)),
            fermion_massSq=lambda X: (np.zeros(0), np.zeros(0)),
            kin_coupled_e_geff=zero, kin_coupled_p_geff=zero,
            mass_spectrum=types.SimpleNamespace(
                number_gauge_bosons=gauge_bosons,
                is_SM_bosons=np.zeros(dofs.size, dtype=bool), is_SM_fermions=np.zeros(0, dtype=bool)),
        )

    PHASE = types.SimpleNamespace(valAt=lambda T: np.zeros(1))

    def test_the_scale_factor_is_the_entropy_ratio(self):
        # Massless modes: the entropy of the transitioning sector is proportional to T^3, so
        # a ~ 1/T exactly. A grid that spans twelve decades with few points, as the percolation
        # solver builds it, defeats the trapezoidal integration of -1/(3 c_s^2 T), which
        # overestimates every interval; entropy conservation is exact whatever the grid.
        T = np.logspace(2, -10, 25)
        cs2, a = bd._time_temperature_factors(self.bath([0.0], [100.0]), self.PHASE, T)
        np.testing.assert_allclose(a, T[0] / T, rtol=1e-10)
        self.assertEqual(cs2.shape, T.shape)

    def test_degrees_of_freedom_that_freeze_out_slow_the_expansion(self):
        # A mode of mass 1 leaves the plasma as the temperature falls below it, so the entropy
        # falls faster than T^3 and a grows faster than 1/T, by the cube root of the ratio of
        # the degrees of freedom.
        T = np.array([100.0, 1.0e-3])
        pot = self.bath([0.0, 1.0], [10.0, 90.0])
        _, a = bd._time_temperature_factors(pot, self.PHASE, T)
        h_hot = float(bd.h_eff_DS(T[0], pot, self.PHASE))
        h_cold = float(bd.h_eff_DS(T[1], pot, self.PHASE))
        self.assertAlmostEqual(h_cold, 10.0, places=6)
        self.assertAlmostEqual(a[1] / (T[0] / T[1]), (h_hot / h_cold) ** (1 / 3), places=6)

    def test_an_unusable_entropy_falls_back_to_the_bag_relation(self):
        # Ghosts are subtracted as massless modes, so once every mode of the potential is
        # heavy the count of degrees of freedom would turn negative; it is floored at zero,
        # and the bag relation continues the chain where the entropy vanishes.
        T = np.logspace(2, -10, 25)
        pot = self.bath([0.0, 10.0], [1.0, 6.0], gauge_bosons=2)
        self.assertEqual(float(bd.h_eff_DS(1.0e-6, pot, self.PHASE)), 0.0)   # floored, not negative
        cs2, a = bd._time_temperature_factors(pot, self.PHASE, T)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue(np.all(np.diff(a) > 0.0), "the scale factor must grow as T falls")
        np.testing.assert_allclose(a[-1] / a[-2], T[-2] / T[-1], rtol=1e-10)

    def test_the_universe_never_contracts_as_it_cools(self):
        # An entropy that rises while the temperature falls would give a shrinking scale
        # factor; the bag relation replaces such a step.
        T = np.array([10.0, 1.0, 0.1])
        entropies = {10.0: 1.0, 1.0: 1.0e6, 0.1: 1.0e12}
        pot = self.bath([0.0], [1.0])
        with mock.patch.object(bd, "h_eff_DS", lambda t, p, ph: entropies[round(float(t), 6)] / t**3), \
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda t, p: 0.0):
            _, a = bd._time_temperature_factors(pot, self.PHASE, T)
        np.testing.assert_allclose(a, T[0] / T, rtol=1e-10)


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

    def test_a_potential_with_standard_model_fields_cannot_decouple_the_standard_model(self):
        base = load_potential("models/TL_2HDM_BSMPT.py", "R2HDM")
        params = {"lambda1": 0.28894, "lambda2": 0.26237, "lambda3": 5.7702, "lambda4": -2.17494,
                  "lambda5": -2.2417, "m12_sq_GeV2": 1573.171, "tan_beta": 21.3949, "yukawa_type": 1, "v_GeV": 246.22}

        class DecoupledSM(base):
            def setConfigParameters(self):
                super().setConfigParameters()
                self.kin_coupled_e_geff = self.kin_coupled_p_geff = lambda T, cf: 0.0 * T
                self.kin_decoupled_e_geff, self.kin_decoupled_p_geff = td.e_geffSM, td.p_geffSM

        class DarkRadiation(base):
            def setConfigParameters(self):
                super().setConfigParameters()
                self.kin_decoupled_e_geff = self.kin_decoupled_p_geff = lambda T, cf: 2.0 + 0.0 * T

        with self.assertRaises(ValueError):
            DecoupledSM(params)
        pot = DarkRadiation(params)
        self.assertEqual(td.sm_bath(pot), "coupled")

    def test_a_coupled_bath_without_the_standard_model_table_is_flagged(self):
        # the Standard Model fields of the potential are subtracted from the coupled bath, so a
        # coupled bath that does not contain them would come out too small
        pot = types.SimpleNamespace(
            SM_bath=None, kin_coupled_e_geff=(lambda T, cf: 2.0 + 0.0 * T),
            kin_decoupled_e_geff=(lambda T, cf: 0.0 * T),
            mass_spectrum=types.SimpleNamespace(is_SM_bosons=np.array([True]), is_SM_fermions=np.zeros(0, bool)))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            generic_potential._check_radiation_baths(pot)
        self.assertIn("counted once", out.getvalue())
        pot.kin_coupled_e_geff = td.e_geffSM
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            generic_potential._check_radiation_baths(pot)
        self.assertEqual(out.getvalue(), "")

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


class PotentialEntropyTests(unittest.TestCase):
    """The entropy of the fields of the potential counts Standard Model fields (e.g. of the 2HDM)."""

    @staticmethod
    def mock_potential():
        # boson_massSq gives (m^2, dof, c, physical): one scalar mode of 6 degrees of freedom and
        # two gauge bosons of 6 together, whose 2 ghosts are subtracted, plus one fermion mode.
        return types.SimpleNamespace(
            X0=np.zeros(1), conversionFactor=1.0,
            boson_massSq=lambda X, T: (np.array([1.0, 4.0]), np.array([6.0, 6.0]),
                                       np.array([0.5, 1.5]), np.array([True, True])),
            fermion_massSq=lambda X: (np.array([9.0]), np.array([12.0])),
            mass_spectrum=types.SimpleNamespace(
                number_gauge_bosons=2, Nscalars=1, dof_bosons=np.array([6.0, 6.0]),
                is_SM_bosons=np.array([True, True]), is_SM_fermions=np.array([True])),
        )

    @staticmethod
    def landau_sum(fn, T):
        return (fn(1.0, T, 6.0, "b") + fn(2.0, T, 6.0, "b") + fn(3.0, T, 12.0, "f")
                - fn(0.0, T, 2.0, "b"))

    def test_every_mode_of_the_potential_counts_and_the_ghosts_are_subtracted(self):
        # Standard Model fields are no longer masked out, so every mode counts; the ghosts of the
        # two gauge bosons are massless in the Landau gauge and are removed.
        pot, T = self.mock_potential(), 3.0
        phase = types.SimpleNamespace(valAt=lambda T: np.zeros(1))
        for kind, fn, module_fn in (("e", td.e_geff, "g_eff_DS"), ("s", td.s_geff, "h_eff_DS")):
            expected = self.landau_sum(fn, T)
            for module in (bd, bdf):
                with self.subTest(kind=kind, module=module.__name__):
                    self.assertAlmostEqual(float(getattr(module, module_fn)(T, pot, phase)),
                                           float(expected), places=12)

    def test_the_standard_model_fields_are_removed_from_the_tabulated_bath(self):
        # What is subtracted from the table are the physical degrees of freedom of the Standard
        # Model fields of the potential: their modes in the Landau gauge, minus their ghosts. Here
        # every field is one, so the potential adds exactly what the table loses.
        pot, T = self.mock_potential(), 3.0
        # the subtraction is interpolated in log T, so it matches the direct sum to the
        # accuracy of that table rather than to machine precision
        for kind, fn in (("e", td.e_geff), ("s", td.s_geff)):
            with self.subTest(kind=kind):
                self.assertAlmostEqual(float(td.sm_fields_in_potential_geff(pot, T, kind)),
                                       float(self.landau_sum(fn, T)), places=7)
        # only the fields flagged is_SM are removed, ghosts included for the gauge bosons among them
        pot.mass_spectrum.is_SM_bosons = np.array([True, False])
        pot.mass_spectrum.is_SM_fermions = np.array([False])
        self.assertAlmostEqual(float(td.sm_fields_in_potential_geff(pot, T, "e")),
                               float(td.e_geff(1.0, T, 6.0, "b")), places=7)
        # a potential without Standard Model fields leaves the table alone
        pot.mass_spectrum.is_SM_bosons = np.zeros(2, bool)
        self.assertEqual(td.sm_fields_in_potential_geff(pot, T, "s"), 0.0)

    def test_the_tabulated_subtraction_follows_a_changed_spectrum(self):
        # the subtraction is tabulated once per model, so it has to notice when the flags,
        # the vacuum or the degrees of freedom of that model change afterwards
        pot, T = self.mock_potential(), 3.0
        first = float(td.sm_fields_in_potential_geff(pot, T, "e"))
        pot.mass_spectrum.is_SM_bosons = np.array([True, False])
        second = float(td.sm_fields_in_potential_geff(pot, T, "e"))
        expected = float(td.e_geff(1.0, T, 6.0, "b") + td.e_geff(3.0, T, 12.0, "f"))
        self.assertAlmostEqual(second, expected, places=7)   # the gauge modes are no longer SM
        self.assertNotAlmostEqual(first, second, places=3)
        pot.mass_spectrum.is_SM_bosons = np.array([True, True])
        self.assertAlmostEqual(float(td.sm_fields_in_potential_geff(pot, T, "e")), first, places=12)
        # a changed model parameter enters the key as well, even when the flags do not move
        pot.model_parameters = {"lambda3": {"value": 1.0}}
        pot.boson_massSq = lambda X, T: (np.array([4.0, 4.0]), np.array([6.0, 6.0]),
                                         np.array([0.5, 1.5]), np.array([True, True]))
        self.assertNotAlmostEqual(float(td.sm_fields_in_potential_geff(pot, T, "e")), first, places=3)

    def test_setting_model_parameters_drops_the_tabulated_subtraction(self):
        # set_modelparams used to reset the temperature cap of the Standard Model tables; it
        # now drops the tables that replaced it, so a reused potential cannot keep a stale one
        spec = importlib.util.spec_from_file_location("tl_2hdm_for_cache_test", "models/TL_2HDM.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = [c for _, c in inspect.getmembers(module, inspect.isclass)
               if issubclass(c, generic_potential) and c is not generic_potential
               and c.__module__ == module.__name__][0]
        with contextlib.redirect_stdout(io.StringIO()):
            pot = cls({})
            td.sm_fields_in_potential_geff(pot, 100.0 / pot.conversionFactor, "e")
            self.assertTrue(pot._sm_fields_geff_splines)
            pot.set_modelparams({})
        self.assertEqual(pot._sm_fields_geff_splines, {})

    def test_the_degrees_of_freedom_at_zero_temperature(self):
        # the coefficients are the limits: a massless mode keeps its count, a massive one does
        # not, and the pressure itself vanishes either way through the T^4 the callers apply
        for fn in (td.e_geff, td.p_geff):
            with self.subTest(fn=fn.__name__):
                self.assertAlmostEqual(float(fn(0.0, 0.0, 4.0, "b")), 4.0, places=12)
                self.assertAlmostEqual(float(fn(1.0, 0.0, 4.0, "b")), 0.0, places=12)

    def test_the_pressure_degrees_of_freedom_take_vectors(self):
        # potential_fields_geff runs inside Vtot, which passes arrays of temperatures.
        T = np.array([0.5, 2.0, 50.0])
        for m, g, ptype in ((0.0, 4.0, "b"), (7.0, 1.0, "b"), (3.0, 12.0, "f")):
            with self.subTest(m=m, ptype=ptype):
                np.testing.assert_allclose(td.p_geff(m, T, g, ptype),
                                           [float(td.p_geff(m, float(t), g, ptype)) for t in T],
                                           rtol=1e-12)
        self.assertAlmostEqual(float(td.p_geff(0.0, 1.0, 4.0, "b")), 4.0, places=12)

    def test_the_2hdm_entropy_matches_the_standard_model_tables(self):
        # In the zero-temperature vacuum the Standard Model fields of the potential carry their
        # Standard Model masses, so what the potential adds cancels what the table loses: the total
        # is the Standard Model table plus the additional scalars of the 2HDM, exactly.
        spec = importlib.util.spec_from_file_location("tl_2hdm_for_test", "models/TL_2HDM.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = [c for _, c in inspect.getmembers(module, inspect.isclass)
               if issubclass(c, generic_potential) and c is not generic_potential and c.__module__ == module.__name__][0]
        with contextlib.redirect_stdout(io.StringIO()):
            pot = cls({})
        CF = pot.conversionFactor
        X = pot.X0 if np.ndim(pot.X0) == 1 else pot.X0[0]
        phase = types.SimpleNamespace(valAt=lambda T: X)
        spectrum = pot.mass_spectrum
        b, f = pot.boson_massSq(X, 0.0), pot.fermion_massSq(X)
        not_SM = (~np.asarray(spectrum.is_SM_bosons, bool), ~np.asarray(spectrum.is_SM_fermions, bool))
        for T_GeV in (10.0, 50.0, 100.0):
            T = T_GeV / CF
            with self.subTest(T_GeV=T_GeV):
                for kind, bath, table in (("s", bd.h_eff_DS, td.s_geffSM), ("e", bd.g_eff_DS, td.e_geffSM)):
                    total = bath(T, pot, phase) + (bd.h_eff_coupled_radiation(T, pot) if kind == "s"
                                                   else pot.kin_coupled_e_geff(T, CF) - td.sm_fields_in_potential_geff(pot, T, "e"))
                    reference = table(T, CF) + td.potential_fields_geff(b, f, T, kind, not_SM)
                    # exact up to the accuracy of the tabulated subtraction, which is
                    # interpolated in log T once per model rather than recomputed inside Vtot
                    self.assertAlmostEqual(float(total) / float(reference), 1.0, places=8)
        # at temperatures far above the Standard Model masses the subtracted piece is the physical
        # count of h, W, Z, the photon and t, b, tau: 1 + 6 + 3 + 2 + 7/8 * 28
        T_high = 1.0e5 / CF
        self.assertAlmostEqual(float(td.sm_fields_in_potential_geff(pot, T_high, "e")), 36.5, places=3)
        # the same counting reproduces radiationEnergyDensity, which uses the full thermal
        # integrals, wherever the modes of the potential are not tachyonic
        ghosts = spectrum.number_gauge_bosons
        for T_GeV in (50.0, 100.0):
            T = T_GeV / CF
            with self.subTest(T_GeV=T_GeV):
                counted = (td.potential_fields_geff(b, f, T, "e") - td.e_geff(0.0, T, ghosts, "b")
                           + pot.kin_coupled_e_geff(T, CF) - td.sm_fields_in_potential_geff(pot, T, "e"))
                integrated = pot.radiationEnergyDensity(np.asarray(X, float), T) / (np.pi**2 / 30 * T**4)
                self.assertAlmostEqual(float(counted) / float(integrated), 1.0, places=6)
        # Vtot and radiationEnergyDensity pass arrays of temperatures through the same routines
        T_grid = np.linspace(20.0, 200.0, 5) / CF
        X_grid = np.tile(np.asarray(X, float), (T_grid.size, 1))
        self.assertEqual(np.shape(pot.radiationEnergyDensity(X_grid, T_grid)), (5,))
        self.assertEqual(np.shape(pot.constantTerms(T_grid)), (5,))


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
            mass_spectrum=types.SimpleNamespace(number_gauge_bosons=0, is_SM_bosons=np.zeros(2, bool),
                                                is_SM_fermions=np.zeros(1, bool)),
            kin_coupled_e_geff=zero, kin_coupled_p_geff=zero,
            kin_decoupled_e_geff=zero, kin_decoupled_p_geff=zero)
        if sm_decoupled:
            pot.kin_decoupled_e_geff, pot.kin_decoupled_p_geff = const(self.G_SM), const(self.G_SM)
        else:
            pot.kin_coupled_e_geff, pot.kin_coupled_p_geff = const(self.G_SM), const(self.G_SM)
        return pot

    def dof(self, sm_decoupled, T_DS, T_dec):
        dof_of = {"e": self.G_DS, "s": self.H_DS}
        with mock.patch.object(td, "potential_fields_geff",
                               lambda b, f, T, kind="e", mask=None: dof_of[kind]):
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

    def test_a_standard_model_field_of_the_potential_is_counted_once(self):
        # A massless Standard Model field of the potential contributes its 2 degrees of freedom
        # from the potential and is removed from the table, so the total is the table alone.
        pot = self.pot(False)
        pot.X0 = np.zeros(1)
        pot.mass_spectrum = types.SimpleNamespace(
            number_gauge_bosons=0, Nscalars=1, dof_bosons=np.array([2.0]),
            is_SM_bosons=np.array([True]), is_SM_fermions=np.zeros(0, bool))
        pot.conversionFactor = 1.0
        pot.boson_massSq = lambda X, T: (np.array([0.0]), np.array([2.0]), np.array([0.0]), np.array([True]))
        pot.fermion_massSq = lambda X: (np.zeros(0), np.zeros(0))
        g, h, _ = td.reheating_geff(pot, np.zeros(1), 61.4, 61.4)
        self.assertAlmostEqual(g, self.G_SM, places=12)
        self.assertAlmostEqual(h, self.G_SM, places=12)

    def test_a_frozen_out_sector_does_not_redshift_with_negative_degrees_of_freedom(self):
        # every mode of the potential heavy against the temperature: the modes die away, the
        # massless ghosts do not, so the difference would be negative without the floor
        pot = self.pot(False)
        pot.X0 = np.zeros(1)
        pot.mass_spectrum = types.SimpleNamespace(
            number_gauge_bosons=2, Nscalars=0, dof_bosons=np.array([6.0]),
            is_SM_bosons=np.zeros(1, bool), is_SM_fermions=np.zeros(0, bool))
        pot.boson_massSq = lambda X, T: (np.array([1.0e6]), np.array([6.0]),
                                         np.array([0.0]), np.array([True]))
        pot.fermion_massSq = lambda X: (np.zeros(0), np.zeros(0))
        g, h, _ = td.reheating_geff(pot, np.zeros(1), 1.0, 1.0)
        self.assertAlmostEqual(g, self.G_SM, places=12)   # the bath alone, nothing negative
        self.assertAlmostEqual(h, self.G_SM, places=12)

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
