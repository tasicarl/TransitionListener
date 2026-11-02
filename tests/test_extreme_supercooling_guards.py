"""Guards for a broken phase with no thermal pressure left to double precision.

At percolation temperatures far below the mass scale the thermal part of the potential
underflows in the broken phase. Its enthalpy and its sound speed then come back as zero or
as NaN, which two places used to carry into an integrator or a division.
"""

from __future__ import annotations

import types
import unittest
import warnings
from unittest import mock

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import errors
from transitionlistener import bubbledynamics_fixedstep as bdf
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
        # against; c_b^2 = 0 is the same statement through the sound speed. A vanishing
        # symmetric-phase sound speed would divide by zero in mu = 1 + 1/c_s^2.
        self.assertEqual(self._find_vw(0.1, 1 / 3, 1 / 3, 0.0), 1)
        self.assertEqual(self._find_vw(0.1, 0.0, 1 / 3, 0.9), 1)
        self.assertEqual(self._find_vw(0.1, 1 / 3, 0.0, 0.9), 1)
        self.assertEqual(self._find_vw(0.1, 1 / 3, -0.2, 0.9), 1)

    def test_usable_inputs_still_solve(self):
        vw = self._find_vw(0.1, 1 / 3, 1 / 3, 0.9)
        self.assertTrue(0.0 < vw < 1.0)
        # A very strong transition is still the runaway branch, as before.
        self.assertEqual(self._find_vw(1e17, 1 / 3, 1 / 3, 0.0), 1)


class _EmptyBrokenPhasePotential:
    """Potential whose broken phase has no thermal pressure left.

    ``dV/dT`` and ``d2V/dT2`` both vanish there, as they do once ``m/T`` is large enough for
    the Boltzmann factors to underflow, so the broken-phase enthalpy is zero and its sound
    speed is ``0/0``. ``numpy_valued`` switches between plain floats, where that division
    raises, and numpy scalars, where it gives a not-a-number and a warning.
    """

    conversionFactor = 1.0
    T_eps = 1e-3
    X0 = np.array([1.0])

    def __init__(self, numpy_valued: bool):
        self.numpy_valued = numpy_valued
        self.config = types.SimpleNamespace(
            gwConf=types.SimpleNamespace(coupled_hydrodynamics=True))

    def _wrap(self, value):
        return np.array([value]) if self.numpy_valued else value

    @staticmethod
    def _is_broken(X):
        return bool(np.allclose(np.atleast_1d(X), 1.0))

    def dVdT(self, X, T, dT=None, include_decoupled=True, include_radiation=True):
        return self._wrap(0.0 if self._is_broken(X) else -1.0)

    def d2VdT2(self, X, T, dT=None, include_decoupled=True):
        return self._wrap(0.0 if self._is_broken(X) else -3.0 / T)

    # `calcWallVelocityLTE` takes the enthalpy and the sound speeds from the thermal part, so
    # a stand-in that answers only `dVdT`/`d2VdT2` would raise before reaching the guard.
    def dV_thermal_dT(self, X, T, dT=None, include_decoupled=True):
        return self.dVdT(X, T, dT=dT, include_decoupled=include_decoupled)

    def d2V_thermal_dT2(self, X, T, dT=None, include_decoupled=True):
        return self.d2VdT2(X, T, dT=dT, include_decoupled=include_decoupled)

    def Vtot(self, X, T, include_decoupled=True):
        return self._wrap(-0.25 if self._is_broken(X) else -1.0)


class SoundSpeedOutputTests(unittest.TestCase):
    """calc_cs fills the reported sound-speed columns and divides by T d2V/dT2 as well."""

    def _calc_cs(self, mode, numpy_valued):
        class Pot:
            T_eps = 1e-3
            X0 = np.array([1.0])
            config = types.SimpleNamespace(
                gwConf=types.SimpleNamespace(coupled_hydrodynamics=True))

            @staticmethod
            def _is_broken(X):
                return bool(np.allclose(np.atleast_1d(X), 1.0))

            def _wrap(self, value):
                return np.array([value]) if numpy_valued else value

            def dVdT(self, X, T, dT=None, include_decoupled=True, include_radiation=True):
                if self._is_broken(X):
                    return self._wrap(0.0 if mode == "frozen" else 1.0)
                return self._wrap(-1.0)

            def d2VdT2(self, X, T, dT=None, include_decoupled=True):
                if self._is_broken(X):
                    # "frozen": both vanish, so the ratio is 0/0. "negative": opposite signs,
                    # so the ratio is negative and its square root does not exist either.
                    return self._wrap(0.0 if mode == "frozen" else -3.0 / T)
                return self._wrap(-3.0 / T)

            # `calc_cs` differentiates the temperature-dependent part of the potential, so a
            # stand-in has to offer those derivatives. The values are the same: what does not
            # depend on the temperature drops out of either.
            def dV_thermal_dT(self, X, T, dT=None):
                return self.dVdT(X, T, dT=dT)

            def d2V_thermal_dT2(self, X, T, dT=None):
                return self.d2VdT2(X, T, dT=dT)

        class Phase:
            def __init__(self, x):
                self.x = np.array([x])

            def valAt(self, T):
                return self.x

        hydro = Hydrodynamics.__new__(Hydrodynamics)
        hydro.pot, hydro.high_phase, hydro.low_phase = Pot(), Phase(0.0), Phase(1.0)
        hydro.verbose = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return hydro.calc_cs(100.0, sym=False), hydro.calc_cs(100.0, sym=True)

    def test_a_phase_without_a_plasma_has_no_sound_speed(self):
        for mode in ("frozen", "negative"):
            for numpy_valued in (False, True):
                with self.subTest(mode=mode, numpy_valued=numpy_valued):
                    broken, symmetric = self._calc_cs(mode, numpy_valued)
                    self.assertTrue(np.isnan(broken))
                    # The healthy phase is untouched: c_s = 1/sqrt(3) for radiation.
                    self.assertAlmostEqual(symmetric, 1.0 / np.sqrt(3.0), places=12)


class WallVelocityEntryPointTests(unittest.TestCase):
    """The production entry point divides by the sound speed before find_vw is reached."""

    def _wall_velocity(self, numpy_valued):
        class Phase:
            def __init__(self, x):
                self.x = np.array([x])

            def valAt(self, T):
                return self.x

        hydro = Hydrodynamics.__new__(Hydrodynamics)
        hydro.pot = _EmptyBrokenPhasePotential(numpy_valued)
        hydro.high_phase, hydro.low_phase, hydro.verbose = Phase(0.0), Phase(1.0), False
        return hydro.calcWallVelocityLTE(100.0)

    def test_an_empty_broken_phase_gives_a_runaway(self):
        # c_b^2 = (dV/dT)/(T d2V/dT2) is formed, and DTheta divides by it, before find_vw is
        # called, so guarding find_vw alone leaves this path broken.
        for numpy_valued in (False, True):
            with self.subTest(numpy_valued=numpy_valued):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    self.assertEqual(self._wall_velocity(numpy_valued), 1)

    def test_a_sound_speed_that_underflows_is_caught(self):
        # A subnormal dV/dT against a large second derivative: the enthalpy stays positive and
        # the denominator is nonzero, so checking only those lets a zero sound speed through.
        class Pot:
            T_eps = 1e-3
            X0 = np.array([1.0])
            config = types.SimpleNamespace(
                gwConf=types.SimpleNamespace(coupled_hydrodynamics=True))

            @staticmethod
            def _is_broken(X):
                return bool(np.allclose(np.atleast_1d(X), 1.0))

            def dVdT(self, X, T, dT=None, include_radiation=True, include_decoupled=True):
                return -5e-320 if self._is_broken(X) else -1.0

            def d2VdT2(self, X, T, dT=None, include_radiation=True, include_decoupled=True):
                return 1.0e10 if self._is_broken(X) else -3.0 / T

            # `calcWallVelocityLTE` takes the enthalpy and the two sound speeds from the
            # thermal part now, so the stand-in has to answer there too or the test would be
            # exercising an attribute error rather than the guard.
            def dV_thermal_dT(self, X, T, dT=None, include_decoupled=True):
                return self.dVdT(X, T, dT=dT, include_decoupled=include_decoupled)

            def d2V_thermal_dT2(self, X, T, dT=None, include_decoupled=True):
                return self.d2VdT2(X, T, dT=dT, include_decoupled=include_decoupled)

            def Vtot(self, X, T, include_decoupled=True):
                return -0.25 if self._is_broken(X) else -1.0

        class Phase:
            def __init__(self, x):
                self.x = np.array([x])

            def valAt(self, T):
                return self.x

        hydro = Hydrodynamics.__new__(Hydrodynamics)
        hydro.pot, hydro.high_phase, hydro.low_phase = Pot(), Phase(0.0), Phase(1.0)
        hydro.verbose = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.assertEqual(hydro.calcWallVelocityLTE(100.0), 1)


class _FrozenPlasmaPotential:
    """Both temperature derivatives vanish, as once every mode has frozen out."""

    T_eps = 1e-3
    X0 = np.array([1.0])

    def dVdT(self, X, T, dT=None, include_decoupled=True, include_radiation=True):
        return 0.0

    def d2VdT2(self, X, T, dT=None, include_decoupled=True):
        return 0.0

    def V_thermal(self, X, T):
        # Constant in T, so both thermal derivatives vanish, which is the point of the stub.
        return 0.0

    def dV_thermal_dT(self, X, T, dT=None):
        return 0.0

    def d2V_thermal_dT2(self, X, T, dT=None):
        return 0.0


class SoundSpeedHelperTests(unittest.TestCase):
    """calcSoundSpeedSq itself divides by T d2V/dT2, before any caller can check it."""

    def test_both_solvers_return_a_number_without_warning(self):
        for label, fn in (("adaptive", bd.calcSoundSpeedSq),
                          ("fixed step", bdf.calcSoundSpeedSq)):
            with self.subTest(label):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    value = fn(_FrozenPlasmaPotential(), np.array([1.0]), 100.0)
                # 0/0 has no value; it must come back as not a number rather than raise,
                # so that the callers' guards are the ones that decide.
                self.assertFalse(np.isfinite(value))


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

    def _alphas(self, broken_cs_sq, symmetric_cs_sq=None):
        real = bd.calcSoundSpeedSq

        def fake(pot, X, t):
            if np.allclose(np.atleast_1d(X), self.low.valAt(t)):
                return broken_cs_sq
            if symmetric_cs_sq is not None:
                return symmetric_cs_sq
            return real(pot, X, t)

        with mock.patch.object(bd, "calcSoundSpeedSq", side_effect=fake):
            values = bd.calcAlphas(self.T, self.pot, self.high, self.low, verbose=False)
        return dict(zip(self.names, (float(np.squeeze(v)) for v in values)))

    def test_only_alpha_thetabar_needs_the_symmetric_sound_speed(self):
        # c_s,sym^2 appears only in the denominator of alpha_thetabar; the pseudo-trace
        # itself and the hydrodynamic strengths are built from the broken-phase value, so
        # they must survive a symmetric-phase value that does not exist.
        for bad in (0.0, float("nan")):
            with self.subTest(symmetric_cs_sq=bad):
                alphas = self._alphas(1.0 / 3.0, symmetric_cs_sq=bad)
                self.assertTrue(np.isnan(alphas["alpha_thetabar"]))
                self.assertTrue(np.isfinite(alphas["alpha_hyd"]))

    def test_a_vanishing_broken_sound_speed_does_not_raise(self):
        for cs_sq in (0.0, float("nan")):
            with self.subTest(broken_cs_sq=cs_sq):
                alphas = self._alphas(cs_sq)
                # The two definitions built on the pseudo-trace are unavailable ...
                self.assertTrue(np.isnan(alphas["alpha_thetabar"]))
                self.assertTrue(np.isnan(alphas["alpha_hyd"]))
                # ... and the ones that need no sound speed survive.
                for name in ("alpha_p", "alpha_theta", "alpha_e", "alpha_inf", "alpha_eq"):
                    self.assertTrue(np.isfinite(alphas[name]), name)

    def test_a_usable_sound_speed_still_gives_the_pseudo_trace(self):
        alphas = self._alphas(1.0 / 3.0)
        self.assertTrue(np.isfinite(alphas["alpha_thetabar"]))
        self.assertTrue(np.isfinite(alphas["alpha_hyd"]))


class FixedStepPseudoTraceGuardTests(PseudoTraceGuardTests):
    """The fixed step size solver has its own calcAlphas and needed the same guard.

    Its naming differs: there ``alpha_theta`` is the pseudo-trace strength, while in the
    adaptive solver that name holds the bag-model one and the pseudo-trace is
    ``alpha_thetabar``.
    """

    def setUp(self):
        super().setUp()
        self.names = ("alpha_p", "alpha_theta", "alpha_e", "alpha_hyd",
                      "alpha_inf", "alpha_eq")

    def test_only_alpha_thetabar_needs_the_symmetric_sound_speed(self):
        self.skipTest("the fixed step size solver has no alpha_thetabar")

    def _alphas(self, broken_cs_sq):
        from transitionlistener import bubbledynamics_fixedstep as bdf

        real = bdf.calcSoundSpeedSq

        def fake(pot, X, t):
            if np.allclose(np.atleast_1d(X), self.low.valAt(t)):
                return broken_cs_sq
            return real(pot, X, t)

        with mock.patch.object(bdf, "calcSoundSpeedSq", side_effect=fake):
            values = bdf.calcAlphas(self.T, self.pot, self.high, self.low, verbose=False)
        return dict(zip(self.names, (float(np.squeeze(v)) for v in values)))

    def test_a_vanishing_broken_sound_speed_does_not_raise(self):
        for cs_sq in (0.0, float("nan")):
            with self.subTest(broken_cs_sq=cs_sq):
                # Unguarded, this path divides a numpy float by zero: the result is inf,
                # then inf - inf = nan, with a RuntimeWarning on the way. The strengths
                # come out the same either way, so the warning is what the test watches.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    alphas = self._alphas(cs_sq)
                # Here alpha_theta is the pseudo-trace strength, so it is the one that goes.
                self.assertTrue(np.isnan(alphas["alpha_theta"]))
                self.assertTrue(np.isnan(alphas["alpha_hyd"]))
                for name in ("alpha_p", "alpha_e", "alpha_inf", "alpha_eq"):
                    self.assertTrue(np.isfinite(alphas[name]), name)

    def test_a_usable_sound_speed_still_gives_the_pseudo_trace(self):
        alphas = self._alphas(1.0 / 3.0)
        self.assertTrue(np.isfinite(alphas["alpha_theta"]))
        self.assertTrue(np.isfinite(alphas["alpha_hyd"]))


class EnthalpyNormalisationTests(unittest.TestCase):
    """The strength must not be normalised by a difference of the vacuum energy with itself.

    Measured on the conformal dark U(1) model of `examples/example_line.yaml`, whose full
    coupling range the release paper plots the strength over: at g = 0.5 and its percolation
    temperature of 2.94e-6 GeV the energy density either side of the derivative step is the
    same double, 1.48570035040305108e+08 in internal units, because the thermal part the
    derivative is after is about 1e-15 of the vacuum energy. The difference was exactly zero
    and the strength came back as infinity, and at g = 0.517 it kept one digit and still
    looked like a number.
    """

    #: (coupling, percolation temperature in GeV) of the two leftmost points of that scan,
    #: as the solver computed them, and two couplings where the old form still had digits.
    POINTS = ((0.500, 2.937788918807139e-06), (0.517, 1.031688466184511e-05),
              (0.600, 1.0e-3), (0.800, 1.0e-2))

    def _conformal(self, g):
        from pathlib import Path

        from transitionlistener.helper_functions import load_potential

        repo = Path(__file__).resolve().parents[1]
        return load_potential(
            str(repo / "models/TL_conformal_dark_u1.py"), "specific_potential"
        )({"g": g, "y": 0.01, "v_GeV": 0.14}, verbose=False)

    def test_the_hot_part_is_the_energy_density_without_a_constant(self):
        """What the helper leaves out may not depend on the temperature.

        The decomposition is the whole claim: if a temperature-dependent term sat outside
        the helper, differencing the helper would silently drop it.
        """
        for module in (bd, bdf):
            pot = self._conformal(0.8)
            X = np.zeros(pot.Ndim)
            T = 1.0e-2 / pot.conversionFactor
            left = [float(np.squeeze(pot.energyDensity(X, t)))
                    - module.hotEnergyDensity(pot, X, t)
                    for t in (T, 2.0 * T, 5.0 * T)]
            with self.subTest(solver=module.__name__):
                self.assertAlmostEqual(left[0], left[1], delta=abs(left[0]) * 1e-12)
                self.assertAlmostEqual(left[0], left[2], delta=abs(left[0]) * 1e-12)

    def test_the_hot_part_differences_where_the_whole_energy_cannot(self):
        """At the two leftmost points of the shipped scan, and still agrees at the others."""
        from transitionlistener.bubbledynamics import temperatureDerivativeStep

        expected_zero = {0.500}
        for g, T_GeV in self.POINTS:
            pot = self._conformal(g)
            X = np.zeros(pot.Ndim)
            T = T_GeV / pot.conversionFactor
            dT = temperatureDerivativeStep(pot, T, X, include_decoupled=True, two_point=True)
            whole = (float(np.squeeze(pot.energyDensity(X, T + dT)))
                     - float(np.squeeze(pot.energyDensity(X, T - dT)))) / (2 * dT)
            hot = (bd.hotEnergyDensity(pot, X, T + dT)
                   - bd.hotEnergyDensity(pot, X, T - dT)) / (2 * dT)
            with self.subTest(g=g):
                self.assertTrue(np.isfinite(hot) and hot > 0.0)
                if g in expected_zero:
                    # The measured case: the whole energy differences to exactly zero, which
                    # is what sent the strength to infinity.
                    self.assertEqual(0.0, whole)
                elif g >= 0.6:
                    # Where the old form still had digits the two agree, so no number that
                    # was trustworthy moves.
                    self.assertAlmostEqual(hot, whole, delta=abs(hot) * 1e-9)

    def test_an_ordinary_point_still_gives_a_strength(self):
        """The 2HDM-scale case, where the two forms are bit-identical, still computes."""
        from pathlib import Path

        from transitionlistener.helper_functions import load_potential

        repo = Path(__file__).resolve().parents[1]
        pot = load_potential(
            str(repo / "models/TL_dark_U1_g_parameterization.py"), "specific_potential"
        )({"g": 1.0, "v_GeV": 0.1, "lambda": 0.03}, verbose=False)

        class Phase:
            def __init__(self, x):
                self.x = np.array([x])

            def valAt(self, T):
                return self.x

        for module, index in ((bd, 2), (bdf, 1)):
            with self.subTest(solver=module.__name__):
                values = module.calcAlphas(3.4e2, pot, Phase(0.0), Phase(0.5), verbose=False)
                self.assertTrue(np.isfinite(float(np.squeeze(values[index]))))

    def test_both_solvers_normalise_through_the_helper(self):
        """The decomposition only helps where the solver's own number comes from it.

        Each helper is replaced by one that returns twice its value, which has to halve the
        strength. A solver that builds its enthalpy inline calls the helper and ignores it,
        and that is what this catches.
        """
        from pathlib import Path

        from transitionlistener.helper_functions import load_potential

        repo = Path(__file__).resolve().parents[1]
        pot = load_potential(
            str(repo / "models/TL_dark_U1_g_parameterization.py"), "specific_potential"
        )({"g": 1.0, "v_GeV": 0.1, "lambda": 0.03}, verbose=False)

        class Phase:
            def __init__(self, x):
                self.x = np.array([x])

            def valAt(self, T):
                return self.x

        # The adaptive solver normalises by a temperature derivative of the hot energy, the
        # fixed-step one by the enthalpy built from it. Doubling either halves the strength.
        for module, name, index in ((bd, "hotEnergyDensity", 2),
                                    (bdf, "symmetricPhaseEnthalpy", 1)):
            with self.subTest(solver=module.__name__):
                real = getattr(module, name)
                args = (3.4e2, pot, Phase(0.0), Phase(0.5))
                before = float(np.squeeze(module.calcAlphas(*args, verbose=False)[index]))
                with mock.patch.object(module, name,
                                       lambda *a, **k: 2.0 * real(*a, **k)):
                    after = float(np.squeeze(module.calcAlphas(*args, verbose=False)[index]))
                self.assertTrue(np.isfinite(before) and before != 0.0)
                self.assertAlmostEqual(0.5 * before, after, delta=abs(before) * 1e-9)

    def test_an_enthalpy_that_cannot_normalise_refuses_the_point(self):
        """Driven through `calcAlphas`, not by constructing the error.

        Zero, not-a-number and infinity all produce a strength no reader could use, and the
        guard tested only for zero until this case was written. The enthalpy is replaced at
        its source, which is what a phase with no plasma in it does to these numbers.
        """
        from pathlib import Path

        from transitionlistener.helper_functions import load_potential

        repo = Path(__file__).resolve().parents[1]
        pot = load_potential(
            str(repo / "models/TL_dark_U1_g_parameterization.py"), "specific_potential"
        )({"g": 1.0, "v_GeV": 0.1, "lambda": 0.03}, verbose=False)

        class Phase:
            def __init__(self, x):
                self.x = np.array([x])

            def valAt(self, T):
                return self.x

        for module in (bd, bdf):
            for value in (0.0, float("nan"), float("inf")):
                with self.subTest(solver=module.__name__, enthalpy=value):
                    with mock.patch.object(module, "symmetricPhaseEnthalpy",
                                           lambda *a, **k: value):
                        with self.assertRaises(errors.EnthalpyUnderflowError) as caught:
                            module.calcAlphas(3.4e2, pot, Phase(0.0), Phase(0.5),
                                              verbose=False)
                    self.assertEqual(20, caught.exception.errorcode)

    def test_the_refusal_names_its_code_and_its_terms(self):
        """The error carries the terms the verdict was taken on."""
        error = errors.EnthalpyUnderflowError(2.9e-6, {"hot e": 0.0, "V_thermal": 0.0})
        self.assertEqual(20, error.errorcode)
        self.assertIn("hot e", str(error))


class UnmockedEntryPointTests(unittest.TestCase):
    """calcAlphas end to end with a real potential whose broken phase has frozen out.

    Nothing here mocks ``calcSoundSpeedSq``: the potential's own temperature derivatives are
    made to vanish in the broken phase, which is what happens once ``m/T`` is large enough
    for the Boltzmann factors to underflow, and the sound speed is computed from them.
    """

    def _run(self, module):
        from pathlib import Path

        from transitionlistener.helper_functions import load_potential

        repo = Path(__file__).resolve().parents[1]
        pot = load_potential(
            str(repo / "models/TL_dark_U1_g_parameterization.py"), "specific_potential"
        )({"g": 1.0, "v_GeV": 0.1, "lambda": 0.03}, verbose=False)

        class Phase:
            def __init__(self, x):
                self.x = np.array([x])

            def valAt(self, T):
                return self.x

        high, low = Phase(0.0), Phase(1.0)
        real_dVdT, real_d2VdT2 = pot.dVdT, pot.d2VdT2

        # The signatures mirror generic_potential exactly, including the optional dT of the
        # second derivative, which one caller omits, and include_radiation, which the callers
        # here leave at its default. Everything is forwarded by keyword.
        def frozen_dVdT(X, T, dT, include_radiation=True, include_decoupled=True):
            if np.allclose(np.atleast_1d(X), low.valAt(T)):
                return 0.0
            return real_dVdT(X, T, dT, include_radiation=include_radiation,
                             include_decoupled=include_decoupled)

        def frozen_d2VdT2(X, T, dT=None, include_radiation=True, include_decoupled=True):
            if np.allclose(np.atleast_1d(X), low.valAt(T)):
                return 0.0
            return real_d2VdT2(X, T, dT, include_radiation=include_radiation,
                               include_decoupled=include_decoupled)

        # `calcSoundSpeedSq` reads the thermal part of the potential, not the whole of it, so
        # freezing only `dVdT`/`d2VdT2` would leave the sound speed computable and the test
        # would no longer be about a frozen phase. A phase that has frozen out has a thermal
        # part with no temperature dependence left, so a constant is what to return.
        real_V_thermal = pot.V_thermal

        def frozen_V_thermal(X, T, include_decoupled=False):
            if np.allclose(np.atleast_1d(X), low.valAt(T)):
                return 0.0
            return real_V_thermal(X, T, include_decoupled=include_decoupled)

        pot.dVdT, pot.d2VdT2 = frozen_dVdT, frozen_d2VdT2
        pot.V_thermal = frozen_V_thermal
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return module.calcAlphas(3.4e2, pot, high, low, verbose=False,
                                     **({"return_wall_strength": True}
                                        if module is bd else {}))

    def test_the_adaptive_solver_survives_a_frozen_broken_phase(self):
        values = self._run(bd)
        # return_wall_strength=True adds alpha_hyd_wall as an eighth entry; the observables
        # stage asks for it, and it is built on the same pseudo-trace difference.
        names = ("alpha_p", "alpha_theta", "alpha_thetabar", "alpha_e", "alpha_hyd",
                 "alpha_inf", "alpha_eq", "alpha_hyd_wall")
        self.assertEqual(len(values), len(names))
        alphas = dict(zip(names, (float(np.squeeze(v)) for v in values)))
        for name in ("alpha_thetabar", "alpha_hyd", "alpha_hyd_wall"):
            self.assertTrue(np.isnan(alphas[name]), name)
        for name in ("alpha_p", "alpha_theta", "alpha_e", "alpha_inf", "alpha_eq"):
            self.assertTrue(np.isfinite(alphas[name]), name)

    def test_the_fixed_step_solver_survives_a_frozen_broken_phase(self):
        values = self._run(bdf)
        names = ("alpha_p", "alpha_theta", "alpha_e", "alpha_hyd", "alpha_inf", "alpha_eq")
        self.assertEqual(len(values), len(names))
        alphas = dict(zip(names, (float(np.squeeze(v)) for v in values)))
        # There alpha_theta is the pseudo-trace strength; that solver has no wall strength.
        for name in ("alpha_theta", "alpha_hyd"):
            self.assertTrue(np.isnan(alphas[name]), name)
        for name in ("alpha_p", "alpha_e", "alpha_inf", "alpha_eq"):
            self.assertTrue(np.isfinite(alphas[name]), name)


if __name__ == "__main__":
    unittest.main()
