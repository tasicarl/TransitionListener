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


class _FrozenPlasmaPotential:
    """Both temperature derivatives vanish, as once every mode has frozen out."""

    T_eps = 1e-3
    X0 = np.array([1.0])

    def dVdT(self, X, T, dT=None, include_decoupled=True, include_radiation=True):
        return 0.0

    def d2VdT2(self, X, T, dT=None, include_decoupled=True):
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

        def frozen_dVdT(X, T, dT, include_radiation=True, include_decoupled=True):
            if np.allclose(np.atleast_1d(X), low.valAt(T)):
                return 0.0
            return real_dVdT(X, T, dT, include_radiation, include_decoupled)

        def frozen_d2VdT2(X, T, dT, include_decoupled=True):
            if np.allclose(np.atleast_1d(X), low.valAt(T)):
                return 0.0
            return real_d2VdT2(X, T, dT, include_decoupled)

        pot.dVdT, pot.d2VdT2 = frozen_dVdT, frozen_d2VdT2
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return module.calcAlphas(3.4e2, pot, high, low, verbose=False)

    def test_the_adaptive_solver_survives_a_frozen_broken_phase(self):
        values = self._run(bd)
        names = ("alpha_p", "alpha_theta", "alpha_thetabar", "alpha_e", "alpha_hyd",
                 "alpha_inf", "alpha_eq")
        alphas = dict(zip(names, (float(np.squeeze(v)) for v in values)))
        self.assertTrue(np.isnan(alphas["alpha_thetabar"]))
        self.assertTrue(np.isnan(alphas["alpha_hyd"]))
        self.assertTrue(np.isfinite(alphas["alpha_p"]))

    def test_the_fixed_step_solver_survives_a_frozen_broken_phase(self):
        values = self._run(bdf)
        names = ("alpha_p", "alpha_theta", "alpha_e", "alpha_hyd", "alpha_inf", "alpha_eq")
        alphas = dict(zip(names, (float(np.squeeze(v)) for v in values)))
        # There alpha_theta is the pseudo-trace strength.
        self.assertTrue(np.isnan(alphas["alpha_theta"]))
        self.assertTrue(np.isnan(alphas["alpha_hyd"]))
        self.assertTrue(np.isfinite(alphas["alpha_p"]))


if __name__ == "__main__":
    unittest.main()
