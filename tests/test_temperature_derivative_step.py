"""The step for temperature derivatives has to grow with the vacuum offset.

``dVdT`` and ``d2VdT2`` difference the full effective potential. In the broken phase of a
supercooled transition that potential is dominated by a temperature-independent vacuum
piece, so the stencil cancels most of its own value and the result is round-off unless the
step is large enough. The fixed rule ``max(T_eps, 1e-4 T)`` does not know about that offset
and lands in the noisy region at low temperature.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

import numpy as np

from transitionlistener.helper_functions import load_potential, temperatureDerivativeStep

REPO = Path(__file__).resolve().parents[1]
CONFORMAL = (REPO / "models/TL_conformal_dark_u1.py", "specific_potential",
             {"g": 0.62, "y": 0.01, "v_GeV": 0.14})


def build():
    path, cls, params = CONFORMAL
    base = load_potential(str(path), cls)
    with contextlib.redirect_stdout(io.StringIO()):
        return base(dict(params), verbose=False)


def old_step(pot, T):
    """The rule this replaces."""
    T_abs = abs(float(T))
    dT = max(float(getattr(pot, "T_eps", 1.0e-3)), T_abs * 1.0e-4)
    return min(dT, 0.25 * T_abs) if T_abs > 0.0 else dT


class StepRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pot = build()

    def test_it_stays_inside_its_bounds(self):
        T = 20.0
        phi = self.pot.findMinimum(np.array([1000.0]), T)
        for X in (phi, np.array([0.0]), None):
            with self.subTest(X=None if X is None else "field"):
                dT = temperatureDerivativeStep(self.pot, T, X)
                self.assertGreater(dT, 0.0)
                self.assertLessEqual(dT, 0.25 * T)
                self.assertGreaterEqual(dT / T, 1.0e-4 - 1e-12)
                self.assertLessEqual(dT / T, 3.0e-2 + 1e-12)

    def test_without_a_field_value_it_falls_back(self):
        self.assertAlmostEqual(temperatureDerivativeStep(self.pot, 20.0) / 20.0, 1.0e-2)

    def test_it_grows_with_the_vacuum_offset(self):
        """The broken phase needs a larger step than the symmetric one."""
        T = 20.0
        phi = self.pot.findMinimum(np.array([1000.0]), T)
        self.assertGreater(temperatureDerivativeStep(self.pot, T, phi),
                           temperatureDerivativeStep(self.pot, T, np.array([0.0])))


class SoundSpeedAccuracyTests(unittest.TestCase):
    """Where the offset dominates, the old step misses the converged value."""

    @classmethod
    def setUpClass(cls):
        cls.pot = build()

    def cs_sq(self, phi, T, dT):
        dVdT = float(np.squeeze(self.pot.dVdT(phi, T, dT=dT, include_decoupled=False)))
        d2 = float(np.squeeze(self.pot.d2VdT2(phi, T, dT=dT, include_decoupled=False)))
        return dVdT / (T * d2)

    def test_the_step_lands_on_the_converged_plateau(self):
        """At large steps the stencil is truncation limited and flat; the rule sits there."""
        for T in (0.5, 5.0, 20.0):
            with self.subTest(T=T):
                phi = self.pot.findMinimum(np.array([1000.0]), T)
                reference = self.cs_sq(phi, T, 0.1 * T)   # deep in the flat region
                chosen = self.cs_sq(phi, T, temperatureDerivativeStep(self.pot, T, phi))
                self.assertLess(abs(chosen / reference - 1), 5e-3,
                                f"chosen {chosen} vs reference {reference}")

    def test_the_old_rule_misses_it_at_low_temperature(self):
        """The regression this fixes: at T = 0.5 the old step is 0.9 percent off."""
        T = 0.5
        phi = self.pot.findMinimum(np.array([1000.0]), T)
        reference = self.cs_sq(phi, T, 0.1 * T)
        old = self.cs_sq(phi, T, old_step(self.pot, T))
        chosen = self.cs_sq(phi, T, temperatureDerivativeStep(self.pot, T, phi))
        self.assertGreater(abs(old / reference - 1), 5e-3)
        self.assertLess(abs(chosen / reference - 1), 5e-3)

    def test_both_rules_agree_where_there_is_no_cancellation(self):
        """High temperature: the offset is not dominant and the step does not matter."""
        T = 100.0
        phi = self.pot.findMinimum(np.array([1000.0]), T)
        old = self.cs_sq(phi, T, old_step(self.pot, T))
        chosen = self.cs_sq(phi, T, temperatureDerivativeStep(self.pot, T, phi))
        self.assertLess(abs(chosen / old - 1), 1e-4)


if __name__ == "__main__":
    unittest.main()
