"""The symmetric phase stays on the fixed subspace of the symmetry down to the lowest temperature.

In the classically conformal U(1) the origin is held in place only by the thermal masses, so the
potential is flat there to within the tracing tolerance and the trace drifts off it at low
temperature; its last node, at today's CMB temperature, was an extrapolation to a field value of
about 0.017 (internal units, v = 1000), where the potential is not even at a minimum. At the
line point g = 0.45, v = 2.62525 GeV, the transition percolates at T_p = 2.8 keV (T_p/v ~ 1e-6),
where the spline through that node put the dark photon at m = 6 T. The counted entropy of the
false vacuum, the "dof_table" scheme of the time-temperature relation, then counted 3.0 of the 4
dark degrees of freedom at 3 T_p, 0.86 at T_p and none at T_p/3, and d ln s/d ln T came out 3.25
instead of 3, which moved T_p by +4% and (beta/H)_RH by -8%.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from transitionlistener import bubbledynamics as bd  # noqa: E402
from transitionlistener import phases as ph  # noqa: E402

Z2 = [np.array([[-1.0]]), np.identity(1)]


def _phase(key, T, X):
    X = np.asarray(X, dtype=float).reshape(len(T), -1)
    return ph.PhaseInfo(key, X, np.asarray(T, dtype=float), np.gradient(X, T, axis=0))


def test_phase_that_is_its_own_mirror_image_is_projected():
    T = np.array([1e-10, 0.0167, 0.0667, 0.1167, 0.1667, 0.2223])
    X = [0.0169, 6.7e-4, 2.5e-4, 1.3e-4, 7.9e-5, 5.1e-5]      # the traced values of the U(1)
    phases = {1: _phase(1, T, X)}
    assert ph.symmetrizeInvariantPhases(phases, Z2, tol=0.2) == [1]
    assert np.all(phases[1].X == 0.0) and np.all(phases[1].dXdT == 0.0)
    assert np.all(phases[1].valAt(np.geomspace(1e-9, 0.2, 50)) == 0.0)


def test_broken_and_departing_phases_are_left_alone():
    T = np.linspace(0.1, 1.0, 8)
    broken = _phase(0, T, 1000.0 - 10 * T)
    departing = _phase(1, T, np.where(T > 0.5, 0.0, 30.0 * (0.5 - T)))  # leaves the origin
    phases = {0: broken, 1: departing}
    before = {k: p.valAt(T).copy() for k, p in phases.items()}
    assert ph.symmetrizeInvariantPhases(phases, Z2, tol=0.2) == []
    for k, p in phases.items():
        assert np.array_equal(p.valAt(T), before[k])
    # only the identity: nothing to project onto
    sym = _phase(2, T, 1e-3 * T)
    assert ph.symmetrizeInvariantPhases({2: sym}, [np.identity(1)], tol=0.2) == []


def test_only_the_components_odd_under_the_symmetry_are_projected():
    T = np.linspace(0.1, 1.0, 8)
    X = np.stack([500.0 + 0.0 * T, 1e-3 * np.sin(7 * T)], axis=-1)   # (even, odd)
    flip_second = np.diag([1.0, -1.0])
    phases = {0: _phase(0, T, X)}
    assert ph.symmetrizeInvariantPhases(phases, [flip_second, np.identity(2)], tol=0.2) == [0]
    assert np.allclose(phases[0].X[:, 0], 500.0) and np.all(phases[0].X[:, 1] == 0.0)


DRIFTED_T = np.array([1e-10, 0.0167, 0.0667, 0.1167, 0.1667, 0.2223])
DRIFTED_X = np.array([0.0169, 6.7e-4, 2.5e-4, 1.3e-4, 7.9e-5, 5.1e-5]).reshape(-1, 1)


@pytest.fixture(scope="module")
def conformal_potential():
    """The classically conformal dark U(1) at the point where the drift was found.

    Its own symmetric phase traces to the origin exactly, so the drifted trace below is put in by
    hand: what is tested is what a drifted false vacuum does to the quantities read off it, and
    that the projection undoes it.
    """
    import TL_conformal_dark_u1 as model

    return model.specific_potential({"g": 0.45, "y": 0.01, "v_GeV": 2.62525}, verbose=False)


def _drifted_and_projected():
    drifted = _phase(1, DRIFTED_T, DRIFTED_X)
    projected = _phase(1, DRIFTED_T, DRIFTED_X)
    assert ph.symmetrizeInvariantPhases({1: projected}, Z2, tol=0.2) == [1]
    return drifted, projected


def test_a_drifted_false_vacuum_loses_the_light_degrees_of_freedom(conformal_potential):
    """Four dark modes and the two Weyl fermions count 7.5 while they are light.

    On the drifted trace the dark photon is heavy against the temperature, so the counted entropy
    of the false vacuum falls away as the temperature drops, which is what feeds the scale factor
    of the percolation integral.
    """
    pot = conformal_potential
    drifted, projected = _drifted_and_projected()
    counted = []
    for T_keV in (0.3, 1.0, 2.8, 9.0, 30.0):
        T = T_keV * 1e-6 / pot.conversionFactor
        counted.append(bd.h_eff_DS(T, pot, drifted))
        assert bd.h_eff_DS(T, pot, projected) == pytest.approx(7.5, abs=1e-12)
    # the drifted trace loses more than two thirds of them at the coldest point, and the loss
    # grows as the temperature falls, so the counted entropy does not scale as T^3
    assert counted[0] < 2.5
    assert counted[-1] > 7.0
    assert all(a < b for a, b in zip(counted, counted[1:]))


def test_a_drifted_false_vacuum_moves_the_sound_speed_off_one_third(conformal_potential):
    pot = conformal_potential
    drifted, projected = _drifted_and_projected()
    T = np.array([2.8e-6 / pot.conversionFactor])
    cs2_drifted, _ = bd._time_temperature_factors(pot, drifted, T)
    cs2_projected, _ = bd._time_temperature_factors(pot, projected, T)
    assert abs(float(cs2_drifted[0]) - 1.0 / 3.0) > 5e-3
    assert float(cs2_projected[0]) == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_the_traced_symmetric_phase_ends_on_the_origin(conformal_potential):
    """End to end: the conformal model traced here does drift off the origin, and the projection
    puts it back, so that the false vacuum keeps all of its light degrees of freedom."""
    pot = conformal_potential
    pot.config.tracingConf.tracing_field_accuracy = 1e-3
    pot.config.tracingConf.tracing_temp_accuracy = 5e-4
    traced = ph.Phases(pot, verbose=False)
    symmetric = min((traced[k] for k in traced.keys()),
                    key=lambda q: float(np.abs(np.atleast_1d(q.X[-1])).max()))
    assert np.all(symmetric.X == 0.0)
    assert np.all(symmetric.dXdT == 0.0)
    assert bd.h_eff_DS(symmetric.Tmin, pot, symmetric) == pytest.approx(7.5, abs=1e-12)
