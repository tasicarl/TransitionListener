"""The broken phase is the warmer of the two, and the guard says which phases when it is not.

The transition releases energy and happens below the critical temperature, so the broken phase
into which the bubbles grow is reheated: `T_bro > T_sym` is a property of the physics, not an
accuracy the solver achieves. The guard therefore asserts the ordering rather than comparing it
against a configured tolerance, and the only slack it allows is the floating-point grid.

What that slack has to be was measured, not chosen. Over 2313 step-3 support points on the 2HDM
line `lambda3 = 5.5 .. 5.87`, across the four phase pairings the line visits, the relative
latent heat `(e_sym - e_bro) / |e_sym|` has a median of +9.74e-04, is exactly zero at 22 points,
and where it comes out negative the worst case is -5.91e-14, 266 units in the last place. That
is round-off on a difference of two nearly equal energy densities, and the ordering it produces
has to be clamped, not refused.

The one point whose ordering was reversed for a physical reason, `lambda4 =
-2.0807692307692305`, had a relative temperature shortfall of 3.167e-05 and a relative latent
heat of -1.264e-04: a third of a typical latent heat, carrying the wrong sign, because the
phase-chain walk had paired the symmetric phase with a broken phase other than the one the
transition ends in. That one has to be refused, with a message naming the pair.

So the rule under test is: clamp what double precision cannot resolve, refuse what it can, and
never take the threshold from `acc_tperc`, which at 1 per cent is seven orders too loose and
admitted the real inversion silently.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import pathlib
import types
import unittest
import warnings
from unittest import mock

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import bubbledynamics_fixedstep as bf
from transitionlistener import percolation_adaptivestepsize as pas
from transitionlistener import errors
from transitionlistener import helper_functions as hf


def a_phase(key):
    """A stand-in carrying only the `key` the guard names in its messages."""
    return type("PhaseStub", (), {"key": key})()


#: The measured `lambda4 = -2.0807692307692305` point: T_bro short of T_sym by 3.167e-05.
LAMBDA4_T_SYM = 110.7296778
LAMBDA4_T_BRO = 110.7261707
#: Its energy densities at T_sym, in GeV^4. Both are measured at the SAME temperature: the
#: broken phase carries more energy there, which is the defect. Reading the broken one at the
#: temperature the root solve returned instead gives e_bro = e_sym exactly, because that is
#: what the solve sets, and the latent heat then reads zero and says nothing.
LAMBDA4_E_SYM = 5.488013e9
LAMBDA4_E_BRO_AT_SYM = 5.488707e9

#: The worst round-off inversion in the survey: a relative latent heat of -5.91e-14. Since `e`
#: follows `T**4` closely enough here that `de / e = 4 dT / T`, that is a relative temperature
#: shortfall of about 1.5e-14.
ROUNDOFF_SHORTFALL = 1.5e-14


class TheGuardAssertsTheOrderingTests(unittest.TestCase):
    """The outcomes of `enforce_broken_not_below_symmetric` on the measured cases."""

    def test_a_broken_temperature_above_the_symmetric_one_is_returned_unchanged(self):
        value, shortfall = hf.enforce_broken_not_below_symmetric(120.0, 110.0)
        self.assertEqual(value, 120.0)
        self.assertEqual(shortfall, 0.0)

    def test_equal_temperatures_are_not_a_shortfall(self):
        """22 of the 2313 surveyed points had a latent heat of exactly zero.

        They are healthy points, so they must pass without a warning: a clamp warning on each
        of them would report round-off that did not happen.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value, shortfall = hf.enforce_broken_not_below_symmetric(110.0, 110.0)
        self.assertEqual(value, 110.0)
        self.assertEqual(shortfall, 0.0)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_a_temperature_above_the_symmetric_one_is_not_warned_about_either(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hf.enforce_broken_not_below_symmetric(120.0, 110.0)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_the_measured_roundoff_inversion_is_clamped_rather_than_refused(self):
        t_sym = 110.0
        t_bro = t_sym * (1.0 - ROUNDOFF_SHORTFALL)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value, shortfall = hf.enforce_broken_not_below_symmetric(t_bro, t_sym)
        self.assertEqual(value, t_sym)
        self.assertGreater(shortfall, 0.0)
        self.assertEqual(len(caught), 1)
        self.assertIn("round-off", str(caught[0].message))

    def test_the_measured_lambda4_inversion_is_refused_with_an_error_code(self):
        with self.assertRaises(errors.PercolationError) as raised:
            hf.enforce_broken_not_below_symmetric(LAMBDA4_T_BRO, LAMBDA4_T_SYM)
        self.assertEqual(raised.exception.errorcode,
                         errors.PercolationError().errorcode)

    def test_the_threshold_separates_the_two_measured_cases_in_both_directions(self):
        """Any threshold that clamps one and refuses the other satisfies the rule.

        The two measured numbers are nine orders of magnitude apart, so this is the whole
        design: a threshold that fails either half is wrong, whatever its value.
        """
        t_sym = 110.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clamped, _ = hf.enforce_broken_not_below_symmetric(
                t_sym * (1.0 - ROUNDOFF_SHORTFALL), t_sym)
        self.assertEqual(clamped, t_sym, "the measured round-off must be clamped")
        with self.assertRaises(errors.PercolationError):
            hf.enforce_broken_not_below_symmetric(LAMBDA4_T_BRO, LAMBDA4_T_SYM)

    def test_the_run_s_percolation_accuracy_would_not_separate_them(self):
        """Why the threshold is not `acc_tperc`: at 1 per cent it clamps the real inversion."""
        shortfall = (LAMBDA4_T_SYM - LAMBDA4_T_BRO) / LAMBDA4_T_SYM
        self.assertLess(shortfall, 1e-2)
        self.assertGreater(shortfall, hf.BROKEN_TEMPERATURE_ROUNDOFF)

    def test_a_zero_symmetric_temperature_does_not_divide_by_zero(self):
        with self.assertRaises(errors.PercolationError):
            hf.enforce_broken_not_below_symmetric(-5.0, 0.0)

    def test_the_call_site_is_named_in_both_the_warning_and_the_error(self):
        t_sym = 110.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hf.enforce_broken_not_below_symmetric(
                t_sym * (1.0 - ROUNDOFF_SHORTFALL), t_sym, where="a named profile")
        self.assertIn("a named profile", str(caught[0].message))
        with self.assertRaises(errors.PercolationError) as raised:
            hf.enforce_broken_not_below_symmetric(
                LAMBDA4_T_BRO, LAMBDA4_T_SYM, where="a named profile")
        self.assertIn("a named profile", str(raised.exception))


class NonFiniteTemperaturesCannotBeOrderedTests(unittest.TestCase):
    """`not (t_bro < t_sym)` is True for a nan, so finiteness has to be checked first."""

    def test_a_nan_broken_temperature_is_refused_rather_than_returned(self):
        with self.assertRaises(errors.PercolationError) as raised:
            hf.enforce_broken_not_below_symmetric(np.nan, 110.0)
        self.assertIn("finite", str(raised.exception))

    def test_a_nan_symmetric_temperature_is_refused(self):
        with self.assertRaises(errors.PercolationError):
            hf.enforce_broken_not_below_symmetric(110.0, np.nan)

    def test_an_infinite_temperature_is_refused(self):
        for pair in ((np.inf, 110.0), (110.0, np.inf), (-np.inf, 110.0)):
            with self.subTest(pair=pair):
                with self.assertRaises(errors.PercolationError):
                    hf.enforce_broken_not_below_symmetric(*pair)

    def test_the_error_is_classified_and_not_a_bare_value_error(self):
        """A bare `ValueError` has no `errorcode` and reached the user as code -1."""
        with self.assertRaises(errors.PercolationError) as raised:
            hf.enforce_broken_not_below_symmetric(np.nan, 110.0)
        self.assertTrue(hasattr(raised.exception, "errorcode"))
        self.assertNotEqual(raised.exception.errorcode, -1)


class TheErrorNamesThePairAndItsLatentHeatTests(unittest.TestCase):
    """What points at a mispairing rather than at the root solve."""

    def raise_lambda4(self, **kwargs):
        with self.assertRaises(errors.PercolationError) as raised:
            hf.enforce_broken_not_below_symmetric(
                LAMBDA4_T_BRO, LAMBDA4_T_SYM, **kwargs)
        return str(raised.exception)

    def test_both_phase_keys_appear_in_the_error(self):
        message = self.raise_lambda4(
            phase_symmetric=a_phase("1"), phase_broken=a_phase("8-6-7-5-4-3-2"))
        self.assertIn("1 -> 8-6-7-5-4-3-2", message)

    def test_the_latent_heat_is_quoted_with_the_released_energy_sign(self):
        """Energy released is `e_sym - e_bro`, so the defect shows as a NEGATIVE number."""
        message = self.raise_lambda4(
            e_sym=LAMBDA4_E_SYM,
            measure_broken_energy=lambda: LAMBDA4_E_BRO_AT_SYM)
        self.assertIn("-1.2646e-04", message)

    def test_the_message_says_the_latent_heat_was_measured_at_the_symmetric_temperature(self):
        """At the broken-phase temperature the root solve returned it is identically zero."""
        message = self.raise_lambda4(
            e_sym=LAMBDA4_E_SYM,
            measure_broken_energy=lambda: LAMBDA4_E_BRO_AT_SYM)
        self.assertIn("at T_sym", message)

    def test_both_energy_densities_are_quoted_with_their_unit(self):
        message = self.raise_lambda4(
            e_sym=LAMBDA4_E_SYM,
            measure_broken_energy=lambda: LAMBDA4_E_BRO_AT_SYM)
        self.assertIn("GeV^4", message)

    def test_the_error_still_raises_when_the_energies_were_not_passed(self):
        message = self.raise_lambda4()
        self.assertIn("not measured", message)

    def test_a_non_finite_energy_does_not_hide_the_error(self):
        message = self.raise_lambda4(
            e_sym=np.nan, measure_broken_energy=lambda: LAMBDA4_E_BRO_AT_SYM)
        self.assertIn("nan", message)

    def test_a_zero_symmetric_energy_does_not_divide_by_zero(self):
        message = self.raise_lambda4(
            e_sym=0.0, measure_broken_energy=lambda: LAMBDA4_E_BRO_AT_SYM)
        self.assertIn("0.0", message)

    def test_a_measurement_that_fails_does_not_replace_the_error_being_reported(self):
        """Losing one number from the message beats losing the whole diagnosis."""
        def explode():
            raise RuntimeError("the potential could not be evaluated there")

        message = self.raise_lambda4(e_sym=LAMBDA4_E_SYM, measure_broken_energy=explode)
        self.assertIn("not measured", message)
        self.assertNotIn("could not be evaluated", message)

    def test_the_measurement_is_not_made_on_a_healthy_point(self):
        """It costs an energy-density evaluation, so it belongs on the error path only."""
        calls = []

        def measure():
            calls.append(1)
            return LAMBDA4_E_BRO_AT_SYM

        hf.enforce_broken_not_below_symmetric(
            120.0, 110.0, e_sym=LAMBDA4_E_SYM, measure_broken_energy=measure)
        self.assertEqual(calls, [])

    def test_the_measurement_is_not_made_when_the_guard_only_clamps(self):
        calls = []

        def measure():
            calls.append(1)
            return LAMBDA4_E_BRO_AT_SYM

        t_sym = 110.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hf.enforce_broken_not_below_symmetric(
                t_sym * (1.0 - ROUNDOFF_SHORTFALL), t_sym,
                e_sym=LAMBDA4_E_SYM, measure_broken_energy=measure)
        self.assertEqual(calls, [])


class EveryCopyOfTheProfileUsesTheSharedRuleTests(unittest.TestCase):
    """The step-3 profile exists in three copies and all three must share one rule."""

    MODULES = ("percolation_adaptivestepsize", "percolation_fixedstepsize",
               "bubbledynamics_fixedstep")

    def source_of(self, module):
        path = (pathlib.Path(hf.__file__).parent / f"{module}.py")
        return path.read_text()

    def guard_calls_in(self, module):
        """The `enforce_broken_not_below_symmetric` calls, parsed rather than grepped."""
        tree = ast.parse(self.source_of(module))
        return [node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "enforce_broken_not_below_symmetric"]

    def test_every_copy_calls_the_shared_rule_exactly_once(self):
        for module in self.MODULES:
            with self.subTest(module=module):
                self.assertEqual(len(self.guard_calls_in(module)), 1)

    def test_every_copy_passes_both_phases_and_both_energy_measurements(self):
        """Without them the error cannot name the pair, which is the whole diagnosis."""
        expected = {"e_sym", "measure_broken_energy", "phase_symmetric", "phase_broken",
                    "where"}
        for module in self.MODULES:
            with self.subTest(module=module):
                call, = self.guard_calls_in(module)
                self.assertEqual({kw.arg for kw in call.keywords}, expected)

    def test_every_copy_measures_the_broken_energy_at_the_symmetric_temperature(self):
        """At the solved broken-phase temperature the latent heat is identically zero."""
        for module in self.MODULES:
            with self.subTest(module=module):
                call, = self.guard_calls_in(module)
                measure, = [kw.value for kw in call.keywords
                            if kw.arg == "measure_broken_energy"]
                self.assertIsInstance(measure, ast.Lambda)
                source = ast.unparse(measure)
                self.assertIn("TSYM[i]", source)
                self.assertNotIn("TBRO", source)

    def test_no_copy_takes_the_threshold_from_the_run_s_accuracy(self):
        """`acc_tperc` is seven orders too loose; passing it again would re-hide the defect."""
        for module in self.MODULES:
            with self.subTest(module=module):
                call, = self.guard_calls_in(module)
                self.assertNotIn("rtol", {kw.arg for kw in call.keywords})
                self.assertNotIn("acc_tperc", ast.unparse(call))

    def test_every_copy_recomputes_the_energy_density_after_a_clamp(self):
        """The Hubble rate below the clamp reads `eBRO`, evaluated before the clamp moved T."""
        for module in self.MODULES:
            with self.subTest(module=module):
                source = self.source_of(module)
                after = source[source.index("enforce_broken_not_below_symmetric"):]
                block = after[:after.index("\n\n")]
                self.assertIn("if _clamped:", block)
                self.assertIn("energyDensity(pot, phase_broken, TBRO", block)

    def test_no_copy_still_raises_a_bare_value_error_for_this_condition(self):
        for module in self.MODULES:
            with self.subTest(module=module):
                source = self.source_of(module)
                self.assertNotIn("raise ValueError(\"T_bro", source)
                self.assertNotIn("raise ValueError('T_bro", source)


class TheThresholdIsNotARunSettingTests(unittest.TestCase):
    """It is a property of double precision, so it does not belong in the configuration."""

    def test_the_guard_takes_no_tolerance_argument(self):
        parameters = inspect.signature(hf.enforce_broken_not_below_symmetric).parameters
        self.assertNotIn("rtol", parameters)

    def test_the_guard_takes_the_broken_energy_as_a_measurement_not_a_value(self):
        """A value would have to be the one at T_bro, which the root solve makes meaningless."""
        parameters = inspect.signature(hf.enforce_broken_not_below_symmetric).parameters
        self.assertIn("measure_broken_energy", parameters)
        self.assertNotIn("e_bro", parameters)

    def test_the_percolation_settings_no_longer_carry_an_unused_accuracy_field(self):
        """It was added for the old tolerance and nothing reads it now."""
        names = [field.name for field in dataclasses.fields(bf.PercolationSettings)]
        self.assertNotIn("acc_tperc", names)

    def test_the_released_positional_order_of_the_settings_is_unchanged(self):
        names = [field.name for field in dataclasses.fields(bf.PercolationSettings)]
        self.assertEqual(names, ["f_perc", "f_start", "f_final", "weight", "maxit",
                                 "rel_increment", "max_boundary_n"])


class TheClampIsCarriedIntoTheEnergyDensityTests(unittest.TestCase):
    """The Hubble rate below the clamp reads `eBRO`, evaluated before the clamp moved T.

    Driven through `_compute_step3_profile` with a synthetic transition rather than read from
    the source, so a call site that stopped recomputing the energy density is caught by a
    number and not by a string match. The synthetic sector is `e_sym = DELTA_V + T**4` and
    `e_bro = T**4`, so the step-3 root solve returns `T_bro = (T_sym**4 + DELTA_V)**0.25` and
    the sign of `DELTA_V` decides the ordering: positive is a transition that releases energy,
    negative is one that absorbs it, which is the defect the guard exists for.
    """

    phases = (types.SimpleNamespace(name="sym", key="1"),
              types.SimpleNamespace(name="bro", key="0"))

    def profile(self, delta_v):
        """Run the adaptive step-3 profile on the synthetic sector and return its state."""
        sym, bro = self.phases

        def energy_density(pot, phase, T, include_decoupled=True):
            return (delta_v if phase is sym else 0.0) + T**4

        TSYM = np.linspace(1.0, 0.8, 40)
        P = np.clip((1.0 - TSYM) / 0.15, 0.0, 1.0) ** 2 * 0.95
        n = TSYM.size
        state = bd.PercolationState(TSYM=TSYM, Sr=np.zeros(n), Hr=np.zeros(n),
                                    Pr=np.zeros(n), Tb=np.zeros(n))
        settings = types.SimpleNamespace(f_final=0.99, time_temperature_mode="sound_speed",
                                         integral_method="ode")
        with mock.patch.object(bd, "energyDensity", energy_density), \
                mock.patch.object(bd, "calcAction", lambda *args: 100.0), \
                mock.patch.object(bd, "h_eff_DS", lambda T, pot, phase: 5.0), \
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda T, pot: 0.0), \
                mock.patch.object(bd, "percIntegralODE_full_sweep",
                                  lambda TSYM, *a, **k: (None, P)):
            pas._compute_step3_profile(
                state, TSYM, P, settings, {},
                types.SimpleNamespace(conversionFactor=1.0), sym, bro, 1.0, 1.0,
                TBROmin=0.5, TBROmax=5.0, verbose=False)
        return TSYM, P, state

    #: `T_bro / T_sym - 1 = -2.5e-10`, inside the 1e-9 the guard treats as round-off. From
    #: `T_bro = (T_sym**4 + DELTA_V)**0.25` with `T_sym` near 1, that needs `DELTA_V = -1e-9`.
    ROUNDOFF_DELTA_V = -1e-9

    def test_a_transition_that_releases_energy_reheats_the_bubbles(self):
        """The control: with DELTA_V positive the guard never fires."""
        TSYM, P, state = self.profile(1.0)
        warm = np.flatnonzero(P > 1e-12)
        self.assertGreater(warm.size, 0)
        self.assertTrue(np.all(state.Tb[warm] >= TSYM[warm]))

    def test_no_point_is_left_below_the_symmetric_temperature(self):
        """The guarantee the guard provides, whatever the root solve returned."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            TSYM, P, state = self.profile(self.ROUNDOFF_DELTA_V)
        warm = np.flatnonzero(P > 1e-12)
        self.assertGreater(warm.size, 0)
        self.assertTrue(np.all(state.Tb[warm] >= TSYM[warm]))

    def test_the_round_off_sized_inversion_is_clamped_where_the_solve_inverted_it(self):
        """Not every support point inverts: at small bubble fraction the root solve's target
        is not `e_sym(T_sym)`, so there the ordering comes out right on its own. Where it does
        invert, the clamp sets the two temperatures exactly equal."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            TSYM, P, state = self.profile(self.ROUNDOFF_DELTA_V)
        warm = np.flatnonzero(P > 1e-12)
        clamped = state.Tb[warm] == TSYM[warm]
        self.assertGreater(int(clamped.sum()), 0)

    def test_the_hubble_rate_is_built_from_the_clamped_temperature(self):
        """A call site that left `eBRO` at the pre-clamp temperature fails this by 1.0e-09."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            TSYM, P, state = self.profile(self.ROUNDOFF_DELTA_V)
        warm = np.flatnonzero(P > 1e-12)
        e_sym = self.ROUNDOFF_DELTA_V + TSYM**4
        e_bro_clamped = state.Tb**4
        rho = P * e_bro_clamped + (1.0 - P) * e_sym
        np.testing.assert_allclose(state.Hr[warm],
                                   bd.HubbleParameter(rho, 1.0)[warm], rtol=1e-12)

    def test_the_clamped_and_unclamped_energy_densities_are_far_enough_apart_to_see(self):
        """Without this the previous test could pass on a tolerance rather than on the clamp."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            TSYM, P, state = self.profile(self.ROUNDOFF_DELTA_V)
        warm = np.flatnonzero(P > 1e-12)
        t_unclamped = (TSYM**4 + self.ROUNDOFF_DELTA_V) ** 0.25
        separation = np.abs(state.Tb[warm] ** 4 - t_unclamped[warm] ** 4) / state.Tb[warm] ** 4
        self.assertGreater(separation.max(), 1e-10)

    def test_a_real_inversion_refuses_the_profile_instead_of_clamping_it(self):
        """DELTA_V = -0.1 is 1e8 times the round-off threshold: energy absorbed, not released."""
        with self.assertRaises(errors.PercolationError) as raised:
            self.profile(-0.1)
        self.assertIn("1 -> 0", str(raised.exception))
        self.assertIn("relative latent heat", str(raised.exception))

if __name__ == "__main__":
    unittest.main()
