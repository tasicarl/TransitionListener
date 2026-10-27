"""The broken-phase temperature may not sit below the symmetric one, within an accuracy.

`T_bro < T_sym` is thermodynamically wrong: the broken phase is reheated by the latent heat it
releases. The guard against it used to be an exact `<` raising a bare `ValueError`, which
carries no `errorcode`, so a point that tripped it was reported as an unclassified failure.

Measured on the 2HDM at `lambda4 = -2.0807692307692305`, the shortfall was 31.7 parts per
million against the 1 per cent `acc_tperc` the run worked to, and the root solve behind it was
sound: the energy criterion changes sign exactly once in its bracket. The real fault was a jump
of the true-vacuum fraction from 0 to 1 across an interval 1.17 parts per million wide, which
the solver diagnoses itself once it is allowed to continue.

So the rule under test is: below the accuracy the temperatures carry, clamp and warn; above it,
refuse the profile with an error code rather than without one.
"""

from __future__ import annotations

import unittest
import warnings

from transitionlistener import errors
from transitionlistener import helper_functions as hf


class TheGuardComparesAgainstTheAccuracyTheTemperaturesCarryTests(unittest.TestCase):
    """The three outcomes of `enforce_broken_not_below_symmetric`."""

    #: The measured 2HDM case: T_bro short of T_sym by 3.167e-05 relative.
    T_SYM = 110.7296778
    T_BRO = 110.7261707

    def test_a_broken_temperature_above_the_symmetric_one_is_returned_unchanged(self):
        value, shortfall = hf.enforce_broken_not_below_symmetric(
            self.T_SYM + 1.0, self.T_SYM, rtol=1e-2)
        self.assertEqual(value, self.T_SYM + 1.0)
        self.assertEqual(shortfall, 0.0)

    def test_equal_temperatures_are_not_a_shortfall(self):
        """The guard is `<`, not `<=`: a profile that just touches is not short of anything."""
        value, shortfall = hf.enforce_broken_not_below_symmetric(
            self.T_SYM, self.T_SYM, rtol=1e-2)
        self.assertEqual(value, self.T_SYM)
        self.assertEqual(shortfall, 0.0)

    def test_the_measured_2hdm_shortfall_is_clamped_rather_than_raised(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value, shortfall = hf.enforce_broken_not_below_symmetric(
                self.T_BRO, self.T_SYM, rtol=1e-2)
        self.assertEqual(value, self.T_SYM, "the clamp must return the symmetric temperature")
        self.assertAlmostEqual(shortfall, 3.167262896169389e-05, places=12)
        self.assertEqual(len(caught), 1, "a clamp that says nothing is the silent fix this "
                                         "guard was changed to avoid")
        self.assertIn("clamping", str(caught[0].message))

    def test_a_shortfall_beyond_the_accuracy_is_refused_with_an_error_code(self):
        with self.assertRaises(errors.PercolationError) as raised:
            hf.enforce_broken_not_below_symmetric(100.0, 110.0, rtol=1e-2)
        self.assertEqual(raised.exception.errorcode, 10,
                         "the point of the change is that this failure carries a code; a bare "
                         "ValueError is reported as -1, unclassified")
        message = " ".join(str(raised.exception).split())
        self.assertIn("9.091e-02", message, "the message states the measured shortfall")
        self.assertIn("reheated", message)

    def test_the_tolerance_is_the_argument_and_not_a_constant(self):
        """A shortfall of 1e-3 passes a 1e-2 accuracy and fails a 1e-4 one."""
        value, _ = hf.enforce_broken_not_below_symmetric(
            110.0 * (1.0 - 1.0e-3), 110.0, rtol=1.0e-2)
        self.assertEqual(value, 110.0)
        with self.assertRaises(errors.PercolationError):
            hf.enforce_broken_not_below_symmetric(
                110.0 * (1.0 - 1.0e-3), 110.0, rtol=1.0e-4)

    def test_a_zero_symmetric_temperature_does_not_divide_by_zero(self):
        with self.assertRaises(errors.PercolationError):
            hf.enforce_broken_not_below_symmetric(-1.0, 0.0, rtol=1e-2)

    def test_the_call_site_is_named_in_both_the_warning_and_the_error(self):
        """Three modules call this; a message that does not say which is not actionable."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hf.enforce_broken_not_below_symmetric(self.T_BRO, self.T_SYM, rtol=1e-2,
                                                  where="the adaptive step-3 profile")
        self.assertIn("the adaptive step-3 profile", str(caught[0].message))
        with self.assertRaises(errors.PercolationError) as raised:
            hf.enforce_broken_not_below_symmetric(100.0, 110.0, rtol=1e-2,
                                                  where="the adaptive step-3 profile")
        self.assertIn("the adaptive step-3 profile", str(raised.exception))


class EveryCopyOfTheGuardUsesTheSharedRuleTests(unittest.TestCase):
    """The solver keeps this profile loop in three modules, and they drift apart.

    `AGENTS.md` warns that many functions exist twice, once in the adaptive module and once in
    its `_fixedstep` twin, and that a guard added to the copy the executing path does not call
    is dead code. Here there are three copies. This test reads the sources rather than running
    them, because reaching all three through the solver needs three different models.
    """

    MODULES = ("percolation_adaptivestepsize", "percolation_fixedstepsize",
               "bubbledynamics_fixedstep")

    @staticmethod
    def _source(name):
        import importlib
        import inspect

        return inspect.getsource(importlib.import_module(f"transitionlistener.{name}"))

    def test_no_copy_still_raises_a_bare_value_error_for_this_condition(self):
        for name in self.MODULES:
            with self.subTest(module=name):
                src = self._source(name)
                self.assertNotIn("T_bro < T_sym", src,
                                 "this copy still carries the old uncoded guard")

    def test_every_copy_calls_the_shared_rule(self):
        for name in self.MODULES:
            with self.subTest(module=name):
                self.assertIn("enforce_broken_not_below_symmetric", self._source(name),
                              "this copy does not go through the shared rule, so a change to "
                              "the rule would not reach it")

    def test_every_copy_recomputes_the_energy_density_after_a_clamp(self):
        """The Hubble rate reads `eBRO`, evaluated before the guard runs.

        All three copies compute the broken-phase energy density from `TBRO` and then use it,
        after the guard, to build `state.Hr[i]`. A clamp moves the temperature, so without a
        recomputation the recorded `Tb[i]` and the energy density behind `Hr[i]` would
        describe different temperatures. Found by reading the diff, not by a failing test:
        the shift is at most `acc_tperc` and no fixture exercises it.
        """
        for name in self.MODULES:
            with self.subTest(module=name):
                src = self._source(name)
                i = src.index("enforce_broken_not_below_symmetric")
                window = src[i:i + 900]
                self.assertIn("_clamped", window,
                              "this copy ignores whether the clamp fired")
                self.assertIn("energyDensity(pot, phase_broken, TBRO", window,
                              "this copy does not re-evaluate the energy density at the "
                              "clamped temperature, so Hr would use the pre-clamp one")

    def test_every_copy_passes_the_run_s_own_accuracy(self):
        """A hard-coded tolerance here would ignore a user who tightened `acc_tperc`."""
        for name in self.MODULES:
            with self.subTest(module=name):
                src = self._source(name)
                i = src.index("enforce_broken_not_below_symmetric")
                self.assertIn("settings.acc_tperc", src[i:i + 400],
                              "this copy does not pass the configured accuracy")


class TheFixedStepSettingsCarryTheAccuracyTests(unittest.TestCase):
    """The fixed-step solver had no accuracy field; the guard needs one there too.

    `bubbledynamics_fixedstep.PercolationSettings` is a public dataclass, so where the new
    field sits is part of its contract: inserted among the released fields it would shift
    every later positional argument onto the wrong attribute, silently, because they all
    carry defaults.
    """

    RELEASED_ORDER = ("f_perc", "f_start", "f_final", "weight", "maxit", "rel_increment",
                      "max_boundary_n")

    def test_the_new_field_is_appended_and_does_not_shift_the_released_ones(self):
        import dataclasses

        from transitionlistener import bubbledynamics_fixedstep as bdf

        names = [f.name for f in dataclasses.fields(bdf.PercolationSettings)]
        self.assertEqual(tuple(names[:len(self.RELEASED_ORDER)]), self.RELEASED_ORDER,
                         "a field was inserted among the released ones, so a positional "
                         "construction now lands on the wrong attribute")
        self.assertEqual(names[-1], "acc_tperc")

    def test_a_positional_construction_still_means_what_it_did(self):
        from transitionlistener import bubbledynamics_fixedstep as bdf

        s = bdf.PercolationSettings(0.3, 1e-3, 0.99, 1.0, 10, 0.1, 2)
        self.assertEqual(s.max_boundary_n, 2)
        self.assertEqual(s.acc_tperc, 1e-2, "the default must match the configured default")

    def test_the_accuracy_is_carried_from_the_configuration(self):
        """A default that silently replaced a user's tightened `acc_tperc` would be a bug."""
        import types

        from transitionlistener import bubbledynamics_fixedstep as bdf
        from transitionlistener import constants as cn

        conf = types.SimpleNamespace(
            f_perc=0.3, f_start=1e-3, f_final=0.99, weight=1.0, maxit=10, rel_increment=0.1,
            max_boundary_ratio=0.5, acc_tperc=4.2e-4,
            entropy_definition=cn.ENTROPY_DEFINITIONS[0])
        pot = types.SimpleNamespace(
            config=types.SimpleNamespace(percolationConf=conf))
        built = bdf._build_percolation_settings(pot, 8)
        self.assertEqual(built.acc_tperc, 4.2e-4,
                         "the configured accuracy did not reach the settings the guard reads")
        self.assertEqual(built.f_perc, 0.3, "the control: the other fields still arrive")


if __name__ == "__main__":
    unittest.main()
