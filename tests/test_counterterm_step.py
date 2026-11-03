"""The counterterm derivatives must not depend on the phase-tracing accuracy."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import mpmath as mp
import numpy as np

from transitionlistener import helper_functions
from transitionlistener.helper_functions import load_potential

REPO = Path(__file__).resolve().parents[1]
ABELIAN = (REPO / "models/TL_dark_U1_g_parameterization.py", "specific_potential",
           {"g": 1.0, "lambda": 0.03, "v_GeV": 0.1})


def build(spec, **overrides):
    path, cls, params = spec
    return load_potential(str(path), cls)(dict(params, **overrides), verbose=False)


class CountertermStepTests(unittest.TestCase):
    def test_counterterms_independent_of_tracing_accuracy(self):
        # The counterterm derivatives used to take the phase-tracing accuracy as
        # their finite-difference step, so tighter tracing changed the potential.
        def counterterms(**overrides):
            pot = build(ABELIAN, **overrides)
            return pot._dV1atvev, pot._d2V1atvev, pot.dl, pot.dmu2

        reference = counterterms()
        for overrides in ({"precision_mode": "benchmark"}, {"precision_mode": "xtrace"},
                          {"precision_mode": "robust"}, {"precision_trace_field_accuracy": 1e-5}):
            with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                self.assertEqual(counterterms(**overrides), reference)

    def test_model_may_override_the_step(self):
        pot = build(ABELIAN)
        self.assertEqual(pot.counterterm_derivative_step, 1.0e-3)
        dark_u1 = load_potential(str(REPO / "models/TL_dark_U1.py"), "specific_potential")(
            {"g_tilde": 2.65, "l": 1.5e-3, "v_GeV": 1e7}, verbose=False)
        self.assertEqual(dark_u1.counterterm_derivative_step, 0.1)


class RoundingOfTheCounterterms(unittest.TestCase):
    """The finite difference at the vev of a sum of order 1e9 is free of rounding, and the step still regulates."""

    @staticmethod
    def model(g=1.0, **attributes):
        cls = load_potential(str(REPO / "models/TL_dark_U1_g_parameterization.py"), "specific_potential")
        original = {name: getattr(cls, name) for name in attributes}
        for name, value in attributes.items():
            setattr(cls, name, value)
        try:
            return cls({"g": g, "lambda": 0.018012121212121212, "v_GeV": 0.1}, verbose=False)
        finally:
            for name, value in original.items():
                setattr(cls, name, value)

    def test_a_change_of_rounding_does_not_reach_the_counterterms(self):
        ref = self.model()
        for k in (-2, -1, 1, 2):
            other = self.model(1.0 + k * 1e-14)
            self.assertLess(abs(other.dmu2 / ref.dmu2 - 1), 1e-8)
            self.assertLess(abs(other.dl / ref.dl - 1), 1e-8)

    def test_the_second_derivative_is_the_stencil_in_exact_arithmetic(self):
        """The five-point stencil at the step 1e-3, in 60 digits; the heavy species agree with any step."""
        mp.mp.dps = 60
        g, lam, v = mp.mpf(1), mp.mpf("0.018012121212121212"), mp.mpf(1000)
        lam = mp.mpf(float(lam))
        mu2, Q2 = lam * v**2, v**2

        pot = self.model()
        _, dof, constants, _ = pot.boson_massSq(np.array([[1000.0]]), 0.0)

        def V1(phi):
            masses = [-mu2 + 3 * lam * phi**2, -mu2 + lam * phi**2, g**2 * phi**2, g**2 * phi**2]
            return sum(mp.mpf(float(n)) * m2**2 * (mp.log(abs(m2 / Q2) + mp.mpf("1e-100")) - mp.mpf(float(c)))
                       for n, m2, c in zip(np.ravel(dof), masses, np.ravel(constants))) / (64 * mp.pi**2)

        h = mp.mpf(float(1e-3))
        exact = (-V1(v - 2 * h) + 16 * V1(v - h) - 30 * V1(v) + 16 * V1(v + h) - V1(v + 2 * h)) / (12 * h**2)
        self.assertAlmostEqual(float(pot.d2V1atvev()) / float(exact), 1.0, delta=1e-9)

    def test_the_step_still_regulates_the_goldstone_boson(self):
        small, large = self.model(), self.model(counterterm_derivative_step=0.1)
        self.assertGreater(float(large.d2V1atvev()) - float(small.d2V1atvev()), 15.0)

    def test_a_species_at_the_step_of_the_model_when_its_mass_changes(self):
        pot = self.model(counterterm_derivative_step=1e-3)
        steps = []
        original = helper_functions.hessianFunction

        def spy(f, eps, Ndim, order):
            steps.append(eps)
            return original(f, eps=eps, Ndim=Ndim, order=order)

        del pot._d2V1atvev
        with mock.patch.object(helper_functions, "hessianFunction", spy):
            pot.d2V1atvev()
        self.assertEqual(steps.count(1e-3), 1)
        self.assertEqual(steps.count(1.0), len(steps) - 1)


if __name__ == "__main__":
    unittest.main()
