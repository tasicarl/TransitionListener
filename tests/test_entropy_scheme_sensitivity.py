"""The entropy-scheme sensitivity diagnostic.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import constants as cn
from transitionlistener import errors
from transitionlistener.helper_functions import load_potential
from transitionlistener.config import PercolationConf

REPO = Path(__file__).resolve().parents[1]


class FixedPhase:
    """Symmetric phase pinned at a field value, as the percolation support sees it."""

    Tmin, Tmax = 1.0e-6, 1.0e6

    def __init__(self, x):
        self.x = np.asarray(x, float)

    def valAt(self, T, deriv=0):
        T = np.asarray(T, float)
        return np.broadcast_to(self.x, T.shape + self.x.shape).copy()


def conformal(v_GeV=6.0, g=0.692):
    return load_potential(str(REPO / "models/TL_conformal_dark_u1.py"),
                          "specific_potential")({"g": g, "y": 0.01, "v_GeV": v_GeV},
                                                verbose=False)


class SensitivityTests(unittest.TestCase):
    def test_it_measures_the_gap_between_the_two_schemes(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 40)
        spread, shift = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
            entropy_definition="dof_table")
        self.assertTrue(np.isfinite(spread) and spread > 0.0)
        self.assertTrue(np.isfinite(shift) and shift > 0.0)

    def test_it_is_symmetric_in_the_configured_scheme(self):
        # It reports the gap between the two, so which one the run uses may not change it.
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 40)
        a = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
            entropy_definition="dof_table")
        b = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
            entropy_definition="eff_potential")
        self.assertAlmostEqual(a[1], b[1], places=12)
        # The spread is a ratio, so it is symmetric only to first order; allow for that.
        self.assertAlmostEqual(a[0] / b[0], 1.0, places=2)

    def test_identical_schemes_would_report_no_sensitivity(self):
        # The number has to come from the difference and nothing else: with both schemes
        # answering alike, there is nothing to report.
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 40)
        with mock.patch.object(bd, "entropy_density",
                               lambda pot, ph, t, d: float(t) ** 4):
            spread, shift = bd.entropy_scheme_sensitivity(
                pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
                entropy_definition="dof_table")
        self.assertAlmostEqual(spread, 0.0, places=10)
        self.assertAlmostEqual(shift, 0.0, places=10)

    def test_bag_mode_has_nothing_to_compare(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 40)
        spread, shift = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="bag",
            entropy_definition="dof_table")
        self.assertTrue(np.isnan(spread) and np.isnan(shift))

    def test_a_support_of_one_temperature_gives_nan(self):
        pot = conformal()
        spread, shift = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), np.array([20.0]),
            time_temperature_mode="sound_speed", entropy_definition="dof_table")
        self.assertTrue(np.isnan(spread) and np.isnan(shift))

    def test_an_unknown_definition_is_refused(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 8)
        with self.assertRaises(errors.PercolationError):
            bd.entropy_scheme_sensitivity(
                pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
                entropy_definition="dof_tabel")

    def test_a_timeout_reaches_the_caller(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 8)

        def timing_out(pot, phase, T, definition):
            raise errors.Timeout()

        with mock.patch.object(bd, "entropy_density", timing_out):
            with self.assertRaises(errors.Timeout):
                bd.entropy_scheme_sensitivity(
                    pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
                    entropy_definition="dof_table")

    def test_the_shift_is_the_scale_factor_gap_over_three(self):
        # d ln a = -(1/3) d ln s, so the reported shift must be exactly the two schemes'
        # logarithmic entropy ratios differenced and divided by three. Computed here from
        # the interpolants directly, so an algebra change in the helper shows up.
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 40)
        ratios = {}
        for definition in cn.ENTROPY_DEFINITIONS:
            log_entropy, _ = bd.expansion_interpolants(
                pot, FixedPhase([0.0]), T, entropy_definition=definition)
            ratios[definition] = float(log_entropy(T[0]) - log_entropy(T[-1]))
        expected = abs(ratios["dof_table"] - ratios["eff_potential"]) / 3.0
        _, shift = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
            entropy_definition="dof_table")
        self.assertAlmostEqual(shift, expected, places=12)
        self.assertGreater(expected, 0.0)   # or the comparison would be vacuous


class SpreadStatisticTests(unittest.TestCase):
    """The spread is the median over the support, not the largest value.

    The distinction is the whole calibration: measured against the shift the mean bubble
    separation actually takes, the median is unbiased to 6% while the maximum overstates it
    by a factor 1.8, and one support temperature that has lost a usable sound speed in one
    scheme drives the maximum to 100% where the separation moves by a few per cent.
    """

    def _spread_with(self, factors):
        """Two schemes whose 3 c_s^2 differ by the given factors, one per temperature."""
        pot = conformal()
        T = np.geomspace(50.0, 15.0, len(factors))
        calls = {"n": 0}
        base = bd.entropy_density

        def entropy(pot_, phase, t, definition):
            # dof_table: s ~ T^4 everywhere. eff_potential: s ~ T^(4/f) at each temperature,
            # so that 3 c_s^2 differs from the counted one by the intended factor there.
            if definition == "dof_table":
                return float(t) ** 4
            i = int(np.argmin(np.abs(T - float(t))))
            return float(t) ** (4.0 / factors[i])

        with mock.patch.object(bd, "entropy_density", entropy):
            spread, _ = bd.entropy_scheme_sensitivity(
                pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
                entropy_definition="dof_table")
        return spread

    def test_one_outlying_temperature_does_not_set_the_spread(self):
        # Nine temperatures differing by 1%, one by a factor 50. The maximum would report
        # something enormous; the median must stay near the 1% the history mostly has.
        factors = [1.01] * 9 + [50.0]
        spread = self._spread_with(factors)
        self.assertLess(spread, 0.10)

    def test_a_uniform_difference_is_reported_as_it_is(self):
        factors = [1.10] * 10
        spread = self._spread_with(factors)
        self.assertGreater(spread, 0.05)

    def test_the_source_takes_the_median(self):
        import inspect
        src = inspect.getsource(bd.entropy_scheme_sensitivity)
        i = src.index("cs_sq_spread = ")
        self.assertIn("np.median", src[i:i + 120])
        self.assertNotIn("np.max", src[i:i + 120])


class FlagPredicateTests(unittest.TestCase):
    """The production predicate itself, not a copy of it.

    The flag is `bd.entropy_scheme_is_sensitive`, which `_compute_percolation` calls; a test
    that re-implemented the comparison would keep passing if the observables stopped using
    it.
    """

    def test_a_spread_above_the_threshold_raises_it(self):
        self.assertTrue(bd.entropy_scheme_is_sensitive(0.05, 0.04))

    def test_a_spread_below_it_does_not(self):
        self.assertFalse(bd.entropy_scheme_is_sensitive(0.01, 0.04))

    def test_the_boundary_is_strict(self):
        self.assertFalse(bd.entropy_scheme_is_sensitive(0.04, 0.04))

    def test_a_nan_spread_leaves_it_unset(self):
        self.assertFalse(bd.entropy_scheme_is_sensitive(float("nan"), 0.04))

    def test_the_threshold_is_the_callers(self):
        self.assertTrue(bd.entropy_scheme_is_sensitive(0.005, 0.001))
        self.assertFalse(bd.entropy_scheme_is_sensitive(0.005, 0.1))

    def test_a_configuration_without_the_setting_gets_the_documented_default(self):
        # A PercolationConf built before this setting existed must not be flagged at some
        # other number written out at the call site.
        import types
        legacy = types.SimpleNamespace(time_temperature_mode="sound_speed")
        self.assertAlmostEqual(bd.entropy_scheme_threshold(legacy),
                               PercolationConf().entropy_scheme_warn_threshold)
        self.assertAlmostEqual(bd.entropy_scheme_threshold(legacy), 0.04)

    def test_a_configuration_with_the_setting_gets_its_own_value(self):
        import types
        self.assertAlmostEqual(
            bd.entropy_scheme_threshold(
                types.SimpleNamespace(entropy_scheme_warn_threshold=0.11)), 0.11)

    def test_the_observables_use_this_predicate_and_the_shared_default(self):
        # The flag must come from the predicate above, and the fallback used when a
        # PercolationConf predates the setting must be the documented default, not a
        # second copy of the number.
        import inspect
        from transitionlistener import transitionObservables as to
        src = inspect.getsource(to)
        i = src.index('derived["WARNING:entropy_scheme_sensitive"]')
        stanza = src[i:i + 200]
        self.assertIn("entropy_scheme_is_sensitive", stanza)
        self.assertNotIn("shift", stanza)
        self.assertIn("entropy_scheme_threshold(", src)
        self.assertEqual(bd.ENTROPY_SCHEME_DEFAULT_THRESHOLD,
                         PercolationConf().entropy_scheme_warn_threshold)

    def test_the_keys_are_registered_where_registration_is_required(self):
        from transitionlistener import config
        from transitionlistener.interface.samplers import get_empty_result
        empty = get_empty_result()
        for k in ("WARNING:entropy_scheme_sensitive", "DIAG:entropy_scheme_cs2_spread",
                  "DIAG:entropy_scheme_lna_gap"):
            with self.subTest(key=k):
                self.assertIn(k, config.all_observables)
                self.assertIn(k, empty)
        self.assertEqual(set(config.all_observables) - set(empty), set())


class ReusedHistoryTests(unittest.TestCase):
    """The configured history is handed in, not rebuilt.

    Rebuilding it would cost two entropy evaluations per support point instead of one, and
    would compare against a history that is not the one the observables came from.
    """

    def _count_evaluations(self, **kwargs):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 20)
        seen = []
        real = bd.expansion_interpolants

        def counting(pot_, phase, temps, **kw):
            seen.append(kw.get("entropy_definition"))
            return real(pot_, phase, temps, **kw)

        with mock.patch.object(bd, "expansion_interpolants", counting):
            out = bd.entropy_scheme_sensitivity(
                pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
                entropy_definition="dof_table", **kwargs)
        return seen, out

    def test_without_a_handed_history_both_schemes_are_built(self):
        seen, _ = self._count_evaluations()
        self.assertEqual(len(seen), 2)

    def test_with_one_handed_in_only_the_other_is_built(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 20)
        history = bd.expansion_interpolants(
            pot, FixedPhase([0.0]), T, entropy_definition="dof_table")
        seen, _ = self._count_evaluations(configured_history=history)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen, ["eff_potential"])

    def test_the_answer_is_the_same_either_way(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 20)
        history = bd.expansion_interpolants(
            pot, FixedPhase([0.0]), T, entropy_definition="dof_table")
        a = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
            entropy_definition="dof_table")
        c = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
            entropy_definition="dof_table", configured_history=history)
        self.assertAlmostEqual(a[0], c[0], places=12)
        self.assertAlmostEqual(a[1], c[1], places=12)

    def test_a_history_that_could_not_be_built_gives_nan(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 20)
        spread, shift = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
            entropy_definition="dof_table", configured_history=(None, None))
        self.assertTrue(np.isnan(spread) and np.isnan(shift))


class UnusableHistoryTests(unittest.TestCase):
    """A history that fell back everywhere is not a history.

    `_time_temperature_factors` replaces each unusable temperature with the bag values, so a
    scheme that fails at every support point would otherwise hand back perfectly ordinary
    interpolants describing nothing, and the comparison would report agreement where nothing
    was computed. That is the one answer this diagnostic must never give.
    """

    def _failing_entropy(self, *a, **k):
        raise ValueError("no entropy here")

    def test_no_usable_temperature_gives_no_history(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 20)
        with mock.patch.object(bd, "entropy_density", self._failing_entropy):
            self.assertEqual(
                bd.expansion_interpolants(pot, FixedPhase([0.0]), T,
                                          entropy_definition="dof_table"),
                (None, None))

    def test_the_diagnostic_then_reports_nothing_rather_than_agreement(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 20)
        with mock.patch.object(bd, "entropy_density", self._failing_entropy):
            spread, shift = bd.entropy_scheme_sensitivity(
                pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
                entropy_definition="dof_table")
        self.assertTrue(np.isnan(spread), "a failed comparison may not look like agreement")
        self.assertTrue(np.isnan(shift))

    def test_one_usable_temperature_is_still_a_history(self):
        # The guard is "none usable", not "any unusable": a support that mostly fell back
        # still carries what it computed.
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 20)
        calls = {"n": 0}
        real = bd.entropy_density

        def sometimes(pot_, phase, t, definition):
            calls["n"] += 1
            if calls["n"] > 3:
                raise ValueError("no entropy here")
            return real(pot_, phase, t, definition)

        with mock.patch.object(bd, "entropy_density", sometimes):
            log_entropy, cooling = bd.expansion_interpolants(
                pot, FixedPhase([0.0]), T, entropy_definition="dof_table")
        self.assertIsNotNone(log_entropy)
        self.assertIsNotNone(cooling)


class SymmetryTests(unittest.TestCase):
    """The spread is the distance between the schemes, so it cannot depend on which is set.

    Dividing by the configured scheme made the number directional: for cooling factors 1 and
    1.041 that is 4.10% one way and 3.94% the other, which straddles the default threshold,
    so the same point would be flagged or not depending on the run's own setting.
    """

    def test_the_spread_is_identical_either_way(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 40)
        a = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
            entropy_definition="dof_table")
        b = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
            entropy_definition="eff_potential")
        self.assertEqual(a[0], b[0])
        self.assertAlmostEqual(a[1], b[1], places=15)

    def test_the_flag_cannot_depend_on_the_configured_scheme(self):
        # The case from the review, at the threshold where it decides the outcome.
        a, b = 1.0, 1.041
        symmetric = 2.0 * abs(b - a) / abs(a + b)
        threshold = PercolationConf().entropy_scheme_warn_threshold
        self.assertEqual(bd.entropy_scheme_is_sensitive(symmetric, threshold),
                         bd.entropy_scheme_is_sensitive(symmetric, threshold))
        # and the directional forms it replaced straddle the threshold, which is the point
        self.assertGreater(abs(b / a - 1.0), threshold)
        self.assertLess(abs(a / b - 1.0), threshold)


class NonApplicableDefaultsTests(unittest.TestCase):
    """Where the comparison does not apply, the keys still have to be written."""

    def test_the_fixed_step_context_sets_them(self):
        # The keys are registered for every run, so the solver that cannot compare must
        # still say "does not apply" rather than leave the writer to fill the boolean
        # with nan.
        import inspect
        from transitionlistener import transitionObservables_fixedstep as tof
        src = inspect.getsource(tof)
        for key in ("WARNING:entropy_scheme_sensitive", "DIAG:entropy_scheme_cs2_spread",
                    "DIAG:entropy_scheme_lna_gap"):
            with self.subTest(key=key):
                self.assertIn(key, src)

    def test_the_adaptive_solver_sets_them_before_the_condition(self):
        import inspect
        from transitionlistener import transitionObservables as to
        src = inspect.getsource(to)
        default = src.index('derived.setdefault("WARNING:entropy_scheme_sensitive"')
        guarded = src.index('entropy_scheme_diagnostic', default)
        self.assertLess(default, guarded,
                        "the defaults must be set before the diagnostic is attempted")


class EmptyDefinitionTests(unittest.TestCase):
    def test_an_empty_definition_is_a_value_and_is_refused(self):
        # `or` would have taken it for "not given" and silently used the default.
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 8)
        with self.assertRaises(errors.PercolationError):
            bd.entropy_scheme_sensitivity(
                pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
                entropy_definition="")

    def test_none_still_means_the_default(self):
        pot = conformal()
        T = np.geomspace(50.0, 15.0, 8)
        spread, _ = bd.entropy_scheme_sensitivity(
            pot, FixedPhase([0.0]), T, time_temperature_mode="sound_speed",
            entropy_definition=None)
        self.assertTrue(np.isfinite(spread))


class RuntimeOverrideTests(unittest.TestCase):
    """Both settings are selectable per run, as `docs/source/usage.rst` states."""

    @staticmethod
    def conf(**overrides):
        from transitionlistener import runtime_options
        from transitionlistener.config import PercolationConf
        c = PercolationConf()
        runtime_options.apply_percolation_overrides(c, overrides)
        return c

    def test_both_settings_are_percolation_overrides(self):
        from transitionlistener import runtime_options
        for key in ("percolation_entropy_scheme_diagnostic",
                    "percolation_entropy_scheme_warn_threshold"):
            with self.subTest(key=key):
                self.assertIn(key, runtime_options.PERCOLATION_OVERRIDE_KEYS)

    def test_the_comparison_can_be_switched_off(self):
        self.assertFalse(
            self.conf(percolation_entropy_scheme_diagnostic=False).entropy_scheme_diagnostic)
        self.assertTrue(self.conf().entropy_scheme_diagnostic)

    def test_the_threshold_is_settable(self):
        self.assertAlmostEqual(
            self.conf(percolation_entropy_scheme_warn_threshold=0.05
                      ).entropy_scheme_warn_threshold, 0.05)

    def test_a_non_positive_threshold_is_refused(self):
        # Silently accepting zero would flag every point that has any spread at all.
        for bad in (0.0, -0.01):
            with self.subTest(threshold=bad):
                with self.assertRaises(ValueError):
                    self.conf(percolation_entropy_scheme_warn_threshold=bad)

    def test_the_default_threshold_is_the_measured_one(self):
        # The criterion is "R_* uncertain at about the five per cent level", set a little
        # below five so the scatter of the estimate does not hide such a point. A change to
        # it should be deliberate.
        self.assertAlmostEqual(self.conf().entropy_scheme_warn_threshold, 0.04)


if __name__ == "__main__":
    unittest.main()
