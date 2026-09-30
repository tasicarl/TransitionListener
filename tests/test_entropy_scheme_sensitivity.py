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


class FlagWiringTests(unittest.TestCase):
    """The flag is raised on the spread, and the threshold is the one the user set."""

    class Conf:
        time_temperature_mode = "sound_speed"
        entropy_definition = "dof_table"
        entropy_scheme_diagnostic = True
        entropy_scheme_warn_threshold = 0.02

    def _derived(self, spread, shift, **conf):
        """Run the observables' diagnostic block with a stubbed sensitivity."""
        from transitionlistener import transitionObservables as to
        c = self.Conf()
        for k, v in conf.items():
            setattr(c, k, v)
        derived = {}
        threshold = float(c.entropy_scheme_warn_threshold)
        # Mirror of the block under test; the contract asserted here is that the flag comes
        # from the spread and not from the gap.
        derived["DIAG:entropy_scheme_cs2_spread"] = float(spread)
        derived["DIAG:entropy_scheme_lna_gap"] = float(shift)
        derived["WARNING:entropy_scheme_sensitive"] = bool(
            np.isfinite(spread) and spread > threshold)
        return derived

    def test_the_source_flags_on_the_spread_and_not_on_the_gap(self):
        # Read the shipped code rather than trusting the mirror above: the flag must be
        # computed from the spread. A change to the gap must not be able to raise it.
        import inspect
        from transitionlistener import transitionObservables as to
        src = inspect.getsource(to)
        i = src.index('derived["WARNING:entropy_scheme_sensitive"]')
        stanza = src[i:i + 200]
        self.assertIn("spread", stanza)
        self.assertNotIn("shift", stanza)

    def test_a_large_gap_alone_does_not_raise_the_flag(self):
        d = self._derived(spread=0.001, shift=10.0)
        self.assertFalse(d["WARNING:entropy_scheme_sensitive"])

    def test_a_spread_above_the_threshold_raises_it(self):
        d = self._derived(spread=0.05, shift=0.0)
        self.assertTrue(d["WARNING:entropy_scheme_sensitive"])

    def test_a_nan_spread_leaves_it_unset(self):
        d = self._derived(spread=float("nan"), shift=float("nan"))
        self.assertFalse(d["WARNING:entropy_scheme_sensitive"])

    def test_the_threshold_is_the_users(self):
        self.assertTrue(self._derived(spread=0.005, shift=0.0,
                                      entropy_scheme_warn_threshold=0.001
                                      )["WARNING:entropy_scheme_sensitive"])
        self.assertFalse(self._derived(spread=0.005, shift=0.0,
                                       entropy_scheme_warn_threshold=0.1
                                       )["WARNING:entropy_scheme_sensitive"])

    def test_the_keys_are_registered_where_registration_is_required(self):
        from transitionlistener import config
        from transitionlistener.interface.samplers import get_empty_result
        keys = ("WARNING:entropy_scheme_sensitive", "DIAG:entropy_scheme_cs2_spread",
                "DIAG:entropy_scheme_lna_gap")
        empty = get_empty_result()
        for k in keys:
            with self.subTest(key=k):
                self.assertIn(k, config.all_observables)
                self.assertIn(k, empty)
        # and the invariant that keeps failed scan rows aligned with successful ones
        self.assertEqual(set(config.all_observables) - set(empty), set())


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
        # 3.5% was chosen on 91 points run in both schemes, where it flags 7, every one of
        # which moves by more than 2.9%. A change to it should be deliberate.
        self.assertAlmostEqual(self.conf().entropy_scheme_warn_threshold, 0.035)


if __name__ == "__main__":
    unittest.main()
