"""The counterterm derivatives must not depend on the phase-tracing accuracy."""

from __future__ import annotations

import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
