"""The Jouguet velocity's distance from the speed of light decides which regime of the
sound-wave efficiency fit applies, so its sign has to survive every transition strength a
double can hold."""

from __future__ import annotations

import math
import unittest

from transitionlistener.hydrodynamics import jouguetRootExcess, kappa_sw


def naive_excess(alpha: float) -> float:
    """The textbook expression, accurate below a strength of about 1e3."""
    return math.sqrt(alpha ** 2 + 2.0 * alpha / 3.0) - alpha


def rationalised_with_alpha(alpha: float) -> float:
    """The form this test file exists to rule out: rationalised, but still carrying alpha."""
    root = math.sqrt(alpha) * math.sqrt(alpha + 2.0 / 3.0)
    denominator = root + alpha
    return 0.0 if math.isinf(denominator) else (2.0 / 3.0) * alpha / denominator


class JouguetRootExcessTests(unittest.TestCase):
    def test_it_agrees_with_the_textbook_expression_where_that_one_is_accurate(self):
        for alpha in (1e-8, 1e-4, 1e-2, 0.5, 1.0, 10.0, 1e3):
            with self.subTest(alpha=alpha):
                self.assertAlmostEqual(
                    jouguetRootExcess(alpha) / naive_excess(alpha), 1.0, delta=1e-12)

    def test_it_stays_finite_and_tends_to_one_third_at_the_top_of_the_range(self):
        for alpha in (1e100, 1e200, 1e307, 1.7e308):
            with self.subTest(alpha=alpha):
                excess = jouguetRootExcess(alpha)
                self.assertTrue(math.isfinite(excess))
                self.assertAlmostEqual(excess, 1.0 / 3.0, delta=1e-50)

    def test_the_form_carrying_alpha_is_the_one_that_fails_there(self):
        # Guard on the premise: without it the test above could pass on either form.
        self.assertEqual(rationalised_with_alpha(1e308), 0.0)
        with self.assertRaises(OverflowError):
            naive_excess(1e308)

    def test_it_never_leaves_the_interval_the_exact_difference_lives_in(self):
        for alpha in (1e-300, 1e-30, 1e-3, 1.0, 1e3, 1e30, 1e300):
            with self.subTest(alpha=alpha):
                self.assertGreater(jouguetRootExcess(alpha), 0.0)
                self.assertLessEqual(jouguetRootExcess(alpha), 1.0 / 3.0)

    def test_zero_strength_has_no_excess(self):
        self.assertEqual(jouguetRootExcess(0.0), 0.0)


class JouguetBranchAtExtremeStrengthTests(unittest.TestCase):
    """A wall below the Jouguet velocity must not be sent through the detonation branch."""

    def test_a_sound_speed_above_two_thirds_keeps_the_jouguet_velocity_above_one(self):
        # vJ - 1 = (cs - 1 + excess) / (1 + alpha), so its sign is that of cs - 2/3.
        for alpha in (1e8, 1e100, 1e308):
            with self.subTest(alpha=alpha):
                self.assertGreater((0.70 - 1.0 + jouguetRootExcess(alpha)), 0.0)

    def test_a_sound_speed_below_two_thirds_keeps_it_below_one(self):
        for alpha in (1e8, 1e100, 1e308):
            with self.subTest(alpha=alpha):
                self.assertLess((0.60 - 1.0 + jouguetRootExcess(alpha)), 0.0)

    def test_the_efficiency_stays_finite_for_a_wall_just_below_the_speed_of_light(self):
        for alpha in (1e8, 1e100, 1e308):
            with self.subTest(alpha=alpha):
                value = kappa_sw(alpha, 0.999, 0.70)
                self.assertTrue(math.isfinite(value))
                self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()
