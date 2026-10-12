"""The adaptive percolation's temperature bookkeeping must not depend on the unit of temperature.

Temperatures are in the model's internal units. A strongly supercooled transition happens far below
one in those units: the classically conformal U(1) of the 3D-rate studies percolates at internal
temperatures of 1e-9 to 1e-4. The tolerances that decide whether a temperature is already present
in the support had a floor of one, so they were 1e-6 (presence) and 1e-10 (deduplication) in
absolute terms. At an internal temperature of 6e-6 the first is a sixth of the temperature: every
point the jump refinement proposed inside a rise of P(T) counted as already present, the
controller found nothing to add with most of its budget unspent, and the acceptance gate rejected
the profile with "sharp unresolved jump in P(T)". Above one the floor never acted, so nothing
changes there.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import percolation_adaptive_rate as par
from transitionlistener import percolation_adaptivestepsize as pas


def settings(**overrides):
    values = dict(n_action_max=60, n_action_increment=5, maxit=20)
    values.update(overrides)
    return types.SimpleNamespace(**values)


# The support of the failing run, internal units: the jump P = 0.105 -> 0.236 between the two
# middle points, and the 13 points the jump refinement proposed inside it.
ACTIVE = np.array([1.9203e-05, 1.2920e-05, 8.0918e-06, 6.9371e-06, 5.7825e-06, 4.9170e-06, 2.3210e-06])
PROPOSED = np.linspace(6.9371e-06, 5.7825e-06, 15)[1:-1]


def _admitted(active, proposed):
    new_active, _, dropped = pas._apply_dynamiczoomwindow_support_update(
        active, proposed, active.copy(), np.asarray([], dtype=float), settings(), max_new_points=60)
    return np.setdiff1d(new_active, active), dropped


class ColdSupportTests(unittest.TestCase):
    def test_refinement_points_inside_a_cold_jump_are_admitted(self):
        added, dropped = _admitted(ACTIVE, PROPOSED)
        self.assertEqual(dropped.size, 0)
        self.assertEqual(added.size, PROPOSED.size)

    def test_the_support_update_is_the_same_in_any_unit_of_temperature(self):
        warm, _ = _admitted(ACTIVE * 1e6, PROPOSED * 1e6)
        cold, _ = _admitted(ACTIVE, PROPOSED)
        np.testing.assert_allclose(np.sort(cold) * 1e6, np.sort(warm), rtol=1e-12)

    def test_distinct_cold_temperatures_are_not_merged(self):
        temps = np.array([3.05e-9, 3.0e-9, 2.9e-9])
        self.assertEqual(bd._temperature_grid(temps).size, 3)
        # and a genuine duplicate still is
        self.assertEqual(bd._temperature_grid(np.array([3.0e-9, 3.0e-9 * (1 + 1e-12)])).size, 1)

    def test_a_cold_temperature_is_present_only_where_it_is(self):
        self.assertTrue(par._temperature_present(ACTIVE, 6.9371e-06 * (1 + 1e-9)))
        self.assertFalse(par._temperature_present(ACTIVE, 6.5e-06))


class UnchangedAboveOneTests(unittest.TestCase):
    """Wherever a temperature of at least one is involved, the floor never acted."""

    def test_the_presence_tolerance_is_the_old_one(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            values = 10 ** rng.uniform(0.0, 4.0, size=rng.integers(1, 20))
            candidate = float(10 ** rng.uniform(-3.0, 4.0))
            old = 1e-6 * max(abs(candidate), float(np.max(np.abs(values))), 1.0)
            self.assertEqual(par._temperature_tolerance(values, candidate), old)

    def test_the_deduplication_is_the_old_one(self):
        rng = np.random.default_rng(2)
        values = np.sort(10 ** rng.uniform(0.0, 3.0, size=400))
        values = np.concatenate([values, values * (1 + 5e-11), values * (1 + 5e-10)])
        grid = bd._temperature_grid(values)
        # the old rule: merge neighbours closer than 1e-10 max(|a|, |b|, 1)
        ordered = np.sort(values)[::-1]
        unique = [float(ordered[0])]
        for c in ordered[1:]:
            if abs(unique[-1] - c) > 1e-10 * max(abs(unique[-1]), abs(c), 1.0):
                unique.append(float(c))
        np.testing.assert_array_equal(grid, np.asarray(unique))


if __name__ == "__main__":
    unittest.main()
