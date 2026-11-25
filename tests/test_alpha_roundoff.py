"""alpha_thetabar at deep supercooling: d(energy density)/dT without round-off.

The line scan at g = 0.50 (y = 0.01, v = 0.173 GeV) percolates at T/v ~ 5e-5
in the 3D treatment, where the pipeline reported alpha = inf.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from transitionlistener import bubbledynamics as bd  # noqa: E402
import TL_conformal_u1_fermions as m4  # noqa: E402

CASES = [  # g, T_perc in GeV from the line scan (3D, leading order)
    (0.50, 9.149e-06),
    (0.55, 8.329e-05),
    (0.60, 3.301e-04),
    (0.80, 3.182e-03),
]


def _pot(g):
    return m4.specific_potential({"g": g, "y": 0.01, "v_GeV": 0.173}, verbose=False)


@pytest.mark.parametrize("g,T_GeV", CASES)
def test_the_derivative_is_finite_and_matches_the_radiation_scaling(g, T_GeV):
    """In the symmetric phase of this model the thermal energy density is the
    radiation bath, so its T-derivative is close to 4 rho_rad / T (g_* varies
    slowly). The old central difference of the total gave exactly 0 at
    g = 0.50; the new one must be finite and within a few per cent of that."""
    pot = _pot(g)
    T = T_GeV / pot.conversionFactor
    X = np.array([0.0])
    dedT = float(np.squeeze(bd.energyDensityTemperatureDerivative(pot, X, T, T * 1e-5)))
    expect = 4.0 * float(np.squeeze(pot.radiationEnergyDensity(X, T))) / T
    assert np.isfinite(dedT) and dedT > 0.0
    assert dedT == pytest.approx(expect, rel=0.05)


def test_the_old_form_really_failed_where_the_new_one_does_not():
    """Guard that the test above exercises the actual failure: the total's
    central difference is exactly zero at g = 0.50."""
    pot = _pot(0.50)
    T = 9.149e-06 / pot.conversionFactor
    X = np.array([0.0])
    dT = T * 1e-5
    old = (pot.energyDensity(X, T + dT) - pot.energyDensity(X, T - dT)) / (2 * dT)
    assert float(np.squeeze(old)) == 0.0


def test_where_the_old_form_was_accurate_the_two_agree():
    """At g = 0.80 the old difference kept ~8 digits; the new one must agree."""
    pot = _pot(0.80)
    T = 3.182e-03 / pot.conversionFactor
    X = np.array([0.0])
    dT = T * 1e-5
    old = (pot.energyDensity(X, T + dT) - pot.energyDensity(X, T - dT)) / (2 * dT)
    new = bd.energyDensityTemperatureDerivative(pot, X, T, dT)
    assert float(np.squeeze(new)) == pytest.approx(float(np.squeeze(old)), rel=1e-6)
