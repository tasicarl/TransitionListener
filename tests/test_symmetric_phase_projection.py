"""The symmetric phase stays on the fixed subspace of the symmetry down to the lowest temperature.

Where only the thermal masses hold the symmetric minimum in place, as in a classically conformal
model, the potential is flat there to within the tracing tolerance and the trace drifts off it as
the temperature falls; its last node, at the lowest tracing temperature, is an extrapolation to a
field value of about 0.017 in internal units where the scale is 1000, at which the potential is not
even at a minimum.

Everything read off the false vacuum below that point inherits the error. On a trace drifted that
way, the counted entropy of the false vacuum of the conformal dark U(1) falls from its 7.5 light
modes to 2.1 between 30 and 0.3 keV instead of holding at 7.5, so it no longer scales as T^3, and
the sound speed of the symmetric phase comes out c_s^2 = 0.324 rather than 1/3. Those two are what
the tests below measure, on a drifted trace put in by hand so that they do not depend on the tracer
reproducing the drift.
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


# a two-field model whose potential is invariant under each flip separately, so the group has
# three non-identity elements rather than one
Z2xZ2 = [np.diag([-1.0, 1.0]), np.diag([1.0, -1.0]), np.diag([-1.0, -1.0]), np.identity(2)]


def test_a_phase_invariant_under_one_element_of_several_is_projected_onto_that_subspace():
    """With more than one symmetry, only the components the matching elements are odd in go."""
    T = np.linspace(0.1, 1.0, 8)
    # large first component, second below the tolerance: invariant under the flip of the second
    X = np.stack([500.0 + 0.0 * T, 1e-3 * np.sin(7 * T)], axis=-1)
    phases = {0: _phase(0, T, X)}
    assert ph.symmetrizeInvariantPhases(phases, Z2xZ2, tol=0.2) == [0]
    assert np.allclose(phases[0].X[:, 0], 500.0)
    assert np.all(phases[0].X[:, 1] == 0.0)


def test_a_phase_at_the_origin_is_projected_by_every_element():
    T = np.linspace(0.1, 1.0, 8)
    X = np.stack([1e-3 * np.sin(7 * T), 1e-3 * np.cos(5 * T)], axis=-1)
    phases = {0: _phase(0, T, X)}
    assert ph.symmetrizeInvariantPhases(phases, Z2xZ2, tol=0.2) == [0]
    assert np.all(phases[0].X == 0.0)


def test_mirrors_with_several_symmetries_keep_one_copy_of_each_distinct_image():
    """The image under the flip of an even component duplicates one of the others."""
    T = np.linspace(0.1, 1.0, 8)
    # on the fixed subspace of the second flip, so that element gives back the phase itself
    phases = {0: _phase(0, T, np.stack([500.0 + 0.0 * T, np.zeros_like(T)], axis=-1))}
    ph.generateMirrorPhases(phases, 1e-3, Z2xZ2)
    keys = sorted(phases, key=str)
    assert keys == [0, "0-m1"], keys
    assert np.allclose(phases["0-m1"].X[:, 0], -500.0)
    assert np.all(phases["0-m1"].X[:, 1] == 0.0)


def test_a_phase_on_every_fixed_subspace_gets_no_mirror_at_all():
    T = np.linspace(0.1, 1.0, 8)
    phases = {1: _phase(1, T, np.zeros((len(T), 2)))}
    ph.generateMirrorPhases(phases, 1e-3, Z2xZ2)
    assert sorted(phases, key=str) == [1]


def test_a_phase_on_the_fixed_subspace_gets_no_mirror():
    """Its image under the symmetry is itself, so a mirror would be a second copy of it."""
    T = np.linspace(0.1, 1.0, 8)
    phases = {1: _phase(1, T, np.zeros_like(T))}
    ph.generateMirrorPhases(phases, 1e-3, Z2)
    assert sorted(phases, key=str) == [1]


def _linked(key, X, low=(), high=()):
    phase = _phase(key, np.linspace(0.1, 1.0, 8), X)
    phase.low_trans.update(low)
    phase.high_trans.update(high)
    return phase


def _dangling(phases):
    keys = {str(k) for k in phases}
    return {str(t) for p in phases.values()
            for t in list(p.low_trans) + list(p.high_trans)} - keys


def test_links_of_a_mirror_name_phases_that_exist():
    T = np.linspace(0.1, 1.0, 8)
    phases = {0: _linked(0, 1000.0 - 10 * T, low=[1], high=[1]),
              1: _linked(1, 500.0 + 0 * T)}
    ph.generateMirrorPhases(phases, 1e-3, Z2)
    assert sorted(phases, key=str) == [0, "0-m1", 1, "1-m1"]
    assert {str(t) for t in phases["0-m1"].low_trans} == {"1-m1"}
    assert {str(t) for t in phases["0-m1"].high_trans} == {"1-m1"}
    assert _dangling(phases) == set()


def test_a_link_to_a_phase_that_is_its_own_image_points_at_that_phase():
    """Its image is never created, so the link has to name the phase itself."""
    T = np.linspace(0.1, 1.0, 8)
    phases = {0: _linked(0, 1000.0 - 10 * T, low=[1]),
              1: _linked(1, np.zeros_like(T))}          # on the fixed subspace
    ph.generateMirrorPhases(phases, 1e-3, Z2)
    assert sorted(phases, key=str) == [0, "0-m1", 1]
    # the key itself, not a string of it: the links are read with `in` against a phase key
    assert phases["0-m1"].low_trans == {1}
    assert all(not isinstance(t, str) for t in phases["0-m1"].low_trans)
    assert phases[1].key in phases["0-m1"].low_trans
    assert _dangling(phases) == set()


def test_high_links_come_from_the_high_transitions():
    T = np.linspace(0.1, 1.0, 8)
    phases = {0: _linked(0, 1000.0 - 10 * T, low=["LOW"], high=["HIGH"]),
              "LOW": _linked("LOW", 400.0 + 0 * T), "HIGH": _linked("HIGH", 600.0 + 0 * T)}
    ph.generateMirrorPhases(phases, 1e-3, Z2)
    assert {str(t) for t in phases["0-m1"].low_trans} == {"LOW-m1"}
    assert {str(t) for t in phases["0-m1"].high_trans} == {"HIGH-m1"}


def test_a_phase_with_only_high_transitions_is_mirrored():
    T = np.linspace(0.1, 1.0, 8)
    phases = {0: _linked(0, 1000.0 - 10 * T, high=[1]), 1: _linked(1, 500.0 + 0 * T)}
    ph.generateMirrorPhases(phases, 1e-3, Z2)          # must not raise
    assert list(phases["0-m1"].low_trans) == []
    assert {str(t) for t in phases["0-m1"].high_trans} == {"1-m1"}


def test_a_phase_that_leaves_its_image_in_between_keeps_its_mirror():
    """Meeting its image at both ends is not enough: the trace must stay there throughout."""
    T = np.linspace(0.1, 1.0, 9)
    X = np.zeros_like(T)
    X[len(T) // 2] = 5.0                      # the two ends coincide, the middle does not
    phases = {1: _phase(1, T, X)}
    ph.generateMirrorPhases(phases, 1e-3, Z2)
    assert sorted(phases, key=str) == [1, "1-m1"]
    assert np.allclose(phases["1-m1"].X.ravel(), -X)


def test_a_broken_phase_still_gets_its_mirror():
    T = np.linspace(0.1, 1.0, 8)
    phases = {0: _phase(0, T, 1000.0 - 10 * T)}
    ph.generateMirrorPhases(phases, 1e-3, Z2)
    assert sorted(phases, key=str) == [0, "0-m1"]
    assert np.allclose(phases["0-m1"].X, -phases[0].X)


def test_mirrors_of_several_phases_are_kept_apart():
    """Skipping one phase's images must not lose another's."""
    T = np.linspace(0.1, 1.0, 8)
    phases = {0: _phase(0, T, 1000.0 - 10 * T), 1: _phase(1, T, np.zeros_like(T))}
    ph.generateMirrorPhases(phases, 1e-3, Z2)
    assert sorted(phases, key=str) == [0, "0-m1", 1]


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


def test_the_model_traces_with_mirror_phases_switched_on(conformal_potential):
    """With `gen_mirror_phases` on, the projected symmetric phase must not be mirrored onto
    itself: a second phase at the same highest temperature is rejected by the tracer."""
    pot = conformal_potential
    pot.config.tracingConf.tracing_field_accuracy = 1e-3
    pot.config.tracingConf.tracing_temp_accuracy = 5e-4
    pot.config.tracingConf.gen_mirror_phases = True
    try:
        traced = ph.Phases(pot, verbose=False)
    finally:
        pot.config.tracingConf.gen_mirror_phases = False
    keys = sorted(traced.keys(), key=str)
    # the broken phase keeps its mirror, the symmetric one gets none
    assert keys == [0, "0-m1", 1], keys
    # and the symmetric phase got there by being projected, not by having been traced there
    assert np.all(traced[1].X == 0.0)
    assert np.all(traced[1].dXdT == 0.0)
