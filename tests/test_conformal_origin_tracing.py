"""The symmetric phase of the 3D conformal models stays at the origin in deep supercooling.

The origin is a minimum at every temperature, with the thermal mass as its curvature. The tracer's
scales are fixed, though, and at T = 1e-12 GeV they are ten orders of magnitude above the
temperature: its minimiser no longer moves the field, its predictor grows any offset as T^-2, and
its Hessian step of 3e-6 GeV reaches past the barrier. Tracing then ended at the inflection of V,
at T = 4.4e-5 GeV for (g, y) = (0.4, 0), and a transition percolating at 2.6e-12 GeV was lost.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "models"))

import TL_conformal_u1_fermions_3d as dark3d  # noqa: E402
from transitionlistener.phases import Phases  # noqa: E402

G, Y, V_GEV = 0.4, 0.0, 2.9534010321641144


def _potential(rate_mode="model"):
    pot = dark3d.specific_potential({"g": G, "y": Y, "v_GeV": V_GEV}, verbose=False)
    pot.config.nucleationConf.rate_mode = rate_mode
    return pot


@pytest.mark.parametrize("T_GeV", [1e-3, 4.4e-5, 1e-7, 1e-10, 1e-12])
def test_the_curvature_at_the_origin_is_the_thermal_mass_at_every_temperature(T_GeV):
    pot = _potential()
    T = T_GeV / pot.conversionFactor
    curvature = float(np.squeeze(pot.d2V(np.zeros(1), T)))
    # 0.92 is the one-loop value measured at 1e-1 GeV, where the fixed step is accurate
    assert curvature == pytest.approx(0.92 * (G * T / 2.0) ** 2, rel=2e-3, abs=0.0)


def test_a_small_offset_is_restored_and_not_extrapolated():
    pot = _potential()
    T = 1e-10 / pot.conversionFactor
    offset = np.array([1e-3 * T])
    # Newton's step from the offset lands on the origin, so gradV carries the same curvature
    newton = float(np.squeeze(pot.gradV(offset, T))) / float(np.squeeze(pot.d2V(offset, T)))
    assert newton == pytest.approx(offset[0], rel=1e-2, abs=0.0)
    assert float(np.squeeze(pot.dgradV_dT(offset, T))) == 0.0


def test_the_symmetric_phase_is_traced_to_the_cold_limit_at_the_origin():
    pot = _potential()
    phases = Phases(pot, False).phases
    phases = phases.values() if isinstance(phases, dict) else phases
    symmetric = min(phases, key=lambda phase: abs(float(np.ravel(phase.X[-1])[0])))
    assert symmetric.Tmin == pytest.approx(pot.Tmin, rel=1e-6, abs=0.0)
    assert np.max(G * np.abs(symmetric.X[:, 0]) / symmetric.T) < 0.1


def test_the_derivatives_stay_the_base_class_s_outside_a_3d_run_and_away_from_the_origin():
    pot, base = _potential("S3overT_4D"), dark3d.base.specific_potential(
        {"g": G, "y": Y, "v_GeV": V_GEV}, verbose=False)
    T = 1e-7 / pot.conversionFactor
    near, far = np.array([1e-3 * T]), np.array([1.0])
    for X in (near, far):
        for name in ("gradV", "d2V", "dgradV_dT"):
            assert np.array_equal(getattr(pot, name)(X, T), getattr(base, name)(X, T))
    model = _potential()
    for name in ("gradV", "d2V", "dgradV_dT"):
        assert np.array_equal(getattr(model, name)(far, T), getattr(base, name)(far, T))
