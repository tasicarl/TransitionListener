"""A phase projected onto the fixed subspace must not keep the range its drift ended at.

`symmetrizeInvariantPhases` repairs the field values of a phase whose trace drifted off a
minimum the thermal masses alone hold in place, and keeps the temperatures the drifting trace
reached. `bubbledynamics` then takes the overlap of the two phases as the floor of the
percolation search, so a symmetric phase that stops above the percolation temperature makes the
search report the transition as too strongly supercooled to percolate. Measured on the
conformal dark U(1) at g = 0.48363636: the traced Tmin of that phase is 2.35e-13 GeV on one
machine and 2.14e-06 GeV on another, and the second fails.
"""

import unittest

import numpy as np

from transitionlistener.phases import (
    PhaseInfo,
    curvatureVerdictWithStepScaling,
    extendProjectedPhaseRanges,
)


def _origin_phase(T, key=1, ndim=1):
    """A phase sitting exactly on the origin over the temperatures ``T``."""
    T = np.asarray(T, dtype=float)
    X = np.zeros((T.size, ndim))
    return PhaseInfo(key, X, T, np.zeros_like(X))


def _one_loop_log_potential():
    """A potential with the one-loop logarithm and no quadratic term at the origin.

    ``V1`` carries ``m**4 log(m**2)``, and with ``m**2`` proportional to the field squared that
    is ``x**4 log(x**2)``. The fourth-order stencil leaves a residue there, so the curvature
    comes out negative and falls by a factor four per halving of the step, which is what was
    measured on the conformal dark U(1).
    """
    def V(X, T):
        s = np.sum(np.asarray(X, dtype=float) ** 2, axis=-1)
        return s * s * np.log(np.abs(s) + 1e-300)
    return V


def _cubic_potential():
    """A potential with a cubic in the field, as the daisy term leaves at a vanishing mass."""
    def V(X, T):
        return -np.abs(np.asarray(X, dtype=float)[..., 0]) ** 3
    return V


def _thermal_mass_potential(T_saddle, ndim=1):
    """A potential whose origin is a minimum above ``T_saddle`` and a saddle below it.

    The curvature at the origin is ``T**2 - T_saddle**2``, which a finite difference returns
    exactly, so the step-scaling test sees a step-independent number of the right sign.
    """
    def V(X, T):
        X = np.asarray(X, dtype=float)
        return 0.5 * (float(T) ** 2 - float(T_saddle) ** 2) * np.sum(X * X, axis=-1)
    return V


class CurvatureVerdictTests(unittest.TestCase):
    """The three verdicts of the step-scaling test, one case each."""

    def test_a_positive_curvature_is_a_minimum(self):
        V = _thermal_mass_potential(1.0)
        self.assertEqual(
            curvatureVerdictWithStepScaling(V, np.zeros(1), 4.0, 1, 1e-3), "minimum")

    def test_a_step_independent_negative_curvature_is_a_saddle(self):
        V = _thermal_mass_potential(10.0)
        self.assertEqual(
            curvatureVerdictWithStepScaling(V, np.zeros(1), 1.0, 1, 1e-3), "saddle")

    def test_a_one_loop_logarithm_origin_is_flat_and_not_a_saddle(self):
        """The conformal case the tracer's own comment measured: the curvature at the origin is
        negative and falls by a factor four per halving of the step."""
        self.assertEqual(
            curvatureVerdictWithStepScaling(
                _one_loop_log_potential(), np.zeros(1), 1.0, 1, 1e-3), "flat")

    def test_a_cubic_origin_is_flat_and_not_a_saddle(self):
        """The shape the daisy term leaves where a resummed mass vanishes: the curvature halves
        with the step rather than holding its value."""
        self.assertEqual(
            curvatureVerdictWithStepScaling(_cubic_potential(), np.zeros(1), 1.0, 1, 1e-3),
            "flat")

    def test_a_quartic_origin_is_flat_and_not_a_saddle(self):
        """A potential with no quadratic term at the origin. The fourth-order stencil is exact
        for a quartic, so both steps return zero and neither is a negative curvature."""
        def V(X, T):
            X = np.asarray(X, dtype=float)
            return -0.25 * np.sum(X * X, axis=-1) ** 2
        self.assertEqual(
            curvatureVerdictWithStepScaling(V, np.zeros(1), 1.0, 1, 1e-3), "flat")

    def test_the_fixtures_really_do_what_the_tests_above_assume(self):
        """The two flat fixtures fall with the step and the thermal mass holds its value, so
        neither of the tests above can pass for the wrong reason."""
        from transitionlistener import helper_functions
        from scipy import linalg

        def smallest(V, eps, T=1.0):
            hess = helper_functions.hessianFunction(
                lambda Y: V(Y, T), eps=eps, Ndim=1, order=4)
            return float(linalg.eigvalsh(np.atleast_2d(np.squeeze(hess(np.zeros(1)))))[0])

        for V, expected in ((_one_loop_log_potential(), 4.0), (_cubic_potential(), 2.0)):
            coarse, fine = smallest(V, 1e-3), smallest(V, 0.5e-3)
            self.assertLess(coarse, 0.0)
            self.assertLess(fine, 0.0)
            self.assertAlmostEqual(abs(coarse) / abs(fine), expected, delta=0.05)

        mass = _thermal_mass_potential(10.0)
        coarse, fine = smallest(mass, 1e-3), smallest(mass, 0.5e-3)
        self.assertLess(coarse, 0.0)
        self.assertAlmostEqual(abs(coarse) / abs(fine), 1.0, places=6)


class ExtendProjectedPhaseRangeTests(unittest.TestCase):
    """What the extension does to a truncated projected phase."""

    def test_a_truncated_origin_phase_is_extended_to_the_tracing_floor(self):
        phases = {1: _origin_phase(np.geomspace(2.0, 100.0, 20))}
        extended = extendProjectedPhaseRanges(
            phases, [1], _thermal_mass_potential(1.0e-6), 1, 1e-3, 1.0e-3, 100.0)
        self.assertIn(1, extended)
        self.assertLess(phases[1].Tmin, 2.0e-3,
                        "the cold end was not carried down to the tracing floor")
        self.assertAlmostEqual(phases[1].Tmax, 100.0, places=10)

    def test_the_extension_stops_where_the_origin_becomes_a_saddle(self):
        """Walking down past the temperature where the thermal mass vanishes must stop."""
        T_saddle = 5.0
        phases = {1: _origin_phase(np.geomspace(20.0, 100.0, 20))}
        extendProjectedPhaseRanges(
            phases, [1], _thermal_mass_potential(T_saddle), 1, 1e-3, 1.0e-3, 100.0)
        self.assertGreater(phases[1].Tmin, T_saddle,
                           "the phase was extended below the temperature where its origin "
                           "stops being a minimum")
        self.assertLess(phases[1].Tmin, 20.0, "no temperature was added at all")

    def test_a_phase_whose_minimum_moves_is_left_alone(self):
        """A fixed subspace of more than one point: the range is the phase's own."""
        T = np.geomspace(2.0, 100.0, 20)
        X = np.linspace(1.0, 2.0, T.size).reshape(-1, 1)
        phases = {1: PhaseInfo(1, X, T, np.zeros_like(X))}
        before = (phases[1].Tmin, phases[1].Tmax, len(phases[1].T))
        extended = extendProjectedPhaseRanges(
            phases, [1], _thermal_mass_potential(1.0e-6), 1, 1e-3, 1.0e-3, 100.0)
        self.assertEqual(extended, {})
        self.assertEqual((phases[1].Tmin, phases[1].Tmax, len(phases[1].T)), before)

    def test_a_phase_already_spanning_the_range_is_not_touched(self):
        phases = {1: _origin_phase(np.geomspace(1.0e-3, 100.0, 20))}
        before = len(phases[1].T)
        extended = extendProjectedPhaseRanges(
            phases, [1], _thermal_mass_potential(1.0e-6), 1, 1e-3, 1.0e-3, 100.0)
        self.assertEqual(extended, {})
        self.assertEqual(len(phases[1].T), before)

    def test_extending_keeps_the_links_and_the_interpolation(self):
        phase = _origin_phase(np.geomspace(2.0, 100.0, 20))
        phase.low_trans.add(7)
        phase.high_trans.add(9)
        phase.mirrorPhase = 3
        phases = {1: phase}
        extendProjectedPhaseRanges(
            phases, [1], _thermal_mass_potential(1.0e-6), 1, 1e-3, 1.0e-3, 100.0)
        self.assertEqual(phase.low_trans, {7})
        self.assertEqual(phase.high_trans, {9})
        self.assertEqual(phase.mirrorPhase, 3)
        self.assertTrue(np.all(np.diff(phase.T) > 0.0), "the trace is not sorted")
        self.assertEqual(len(phase.T), len(phase.X))
        self.assertEqual(len(phase.T), len(phase.dXdT))
        # The projected field value is the fixed point everywhere, so the interpolant is zero.
        for T in (3.0e-3, 1.0, 50.0):
            np.testing.assert_allclose(np.atleast_1d(phase.valAt(T)), 0.0, atol=0.0)


class ExtendTraceTests(unittest.TestCase):
    """The mechanics `extendProjectedPhaseRanges` is built on."""

    def test_a_duplicate_temperature_is_refused_by_name(self):
        """The interpolator also refuses a repeated temperature, with a message about its own
        x array, so the refusal is asserted by what it says: without the check here the caller
        is told the interpolation failed rather than what it did wrong."""
        phase = _origin_phase(np.array([1.0, 2.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "already carries"):
            phase.extendTrace(np.array([2.0]), np.zeros((1, 1)), np.zeros((1, 1)))

    def test_mismatched_lengths_are_refused_by_name(self):
        phase = _origin_phase(np.array([1.0, 2.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "one field vector and one slope"):
            phase.extendTrace(np.array([0.5, 0.25]), np.zeros((1, 1)), np.zeros((1, 1)))

    def test_nodes_on_both_sides_are_merged_in_order(self):
        phase = _origin_phase(np.array([1.0, 2.0, 3.0]))
        phase.extendTrace(np.array([0.5, 4.0]), np.zeros((2, 1)), np.zeros((2, 1)))
        np.testing.assert_allclose(phase.T, [0.5, 1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(phase.Tmin, 0.5)
        self.assertAlmostEqual(phase.Tmax, 4.0)


if __name__ == "__main__":
    unittest.main()
