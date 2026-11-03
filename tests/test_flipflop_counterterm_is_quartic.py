"""The second counterterm coupling of the dark flip-flop multiplies a quartic.

`DarkFlipFlop.calcCounterTerms` leaves `dlambda2` at zero, so the term never
contributes to a run and no observable changes with it. The docstring of `Vct`
states the term as `dlambda2 / 4 * phi_2 ** 4`, and that is what the Lagrangian
needs it to be: a counterterm for the quartic self-coupling of the second field
has to carry the same field power as the coupling it renormalises, or it shifts
the mass instead. The test sets the coupling by hand and reads off the power.
"""

import contextlib
import io
import unittest
from pathlib import Path

import numpy as np

from transitionlistener.helper_functions import load_potential

_MODEL = Path(__file__).resolve().parents[1] / "models" / "TL_dark_flipflop.py"
_POINT = dict(lambda0=0.005098, lambda1=0.002144, lambda12=0.003078,
              v_GeV=3.728330741601088, y=0.972319, gamma=0.7532)


def _load_flipflop():
    with contextlib.redirect_stdout(io.StringIO()):
        return load_potential(str(_MODEL), "DarkFlipFlop")(_POINT, verbose=False)


class TheSecondCountertermIsQuartic(unittest.TestCase):

    def setUp(self):
        self.pot = _load_flipflop()
        # Isolate the term: everything else in Vct is switched off.
        self.pot.dmu1 = 0.0
        self.pot.dmu2 = 0.0
        self.pot.dlambda1 = 0.0
        self.pot.dlambda12 = 0.0
        self.pot.dlambda2 = 1.0

    def test_the_term_scales_as_the_fourth_power_of_the_second_field(self):
        phi2 = np.array([0.3, 0.6, 1.2])
        X = np.stack([np.zeros_like(phi2), phi2], axis=-1)
        Vct = np.asarray(self.pot.Vct(X), dtype=float)
        # Doubling phi_2 multiplies a quartic by 16 and a quadratic by 4.
        for a, b in zip(Vct[:-1], Vct[1:]):
            self.assertAlmostEqual(b / a, 16.0, places=10)

    def test_the_term_is_the_coupling_over_four_times_the_quartic(self):
        phi2 = 0.7
        X = np.array([0.0, phi2])
        self.assertAlmostEqual(
            float(self.pot.Vct(X)), 0.25 * phi2 ** 4, places=14,
            msg="Vct does not reproduce dlambda2 / 4 * phi_2 ** 4")

    def test_the_first_field_still_carries_its_own_quartic(self):
        self.pot.dlambda2 = 0.0
        self.pot.dlambda1 = 1.0
        phi1 = 0.7
        X = np.array([phi1, 0.0])
        self.assertAlmostEqual(float(self.pot.Vct(X)), 0.25 * phi1 ** 4, places=14)


if __name__ == "__main__":
    unittest.main()
