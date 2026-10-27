"""The saturated tail of the step-3 profile must stay finite, or the integral stops accumulating.

Percolation step 3 stops computing the action once step 2's fraction has passed `f_final`, on
the grounds that nothing new nucleates below there. That is right: on the 2HDM benchmark
`S_3/T` is already 133 at the saturation point and only grows colder, so the rate is
`exp(-133)` or less whatever the action is.

What it missed is that the region carries the growth of the bubbles already nucleated, and
`percIntegralODE` needs valid samples across it to integrate that growth. Writing `nan` there
fails its validity test, floors the source and transport terms, and the integral stops
accumulating: the benchmark coupling `lambda3 = 5.7918389018` had `I` fall from 50 to 9.7e-38,
so `P = 1 - exp(-I)` rounded to exactly 0 at all 174 support points and a transition whose
percolation temperature step 2 had already found was reported as never percolating.

Note on what this file does not test. A minimal synthetic case does not reproduce the collapse:
with a constant action and a smooth scale factor the integral reaches the same value whether
the tail is finite or not, because the warm region alone already saturates it. The collapse
needs the real `S_3(T)` and the real `a(T)` of a strongly supercooled transition, where the
warm region alone reaches only a fifth of the total. The evidence for the mechanism is
therefore the recorded arrays of that benchmark run, replayed with the inputs crossed, not a
constructed case here. Tuning a synthetic case until it agreed would be fitting the test to the
answer.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import percolation_adaptivestepsize as pas


class SaturatedTailTests(unittest.TestCase):
    """Drive the step-3 profile with a fraction that saturates partway down the support."""

    phases = (types.SimpleNamespace(name="sym"), types.SimpleNamespace(name="bro"))

    def profile(self, nonfinite_call=None, zero_at_last_computed=False):
        """Run the step-3 profile loop.

        With `nonfinite_call` set, the mocked `calcAction` returns `nan` on that call, counting
        from zero, and records the `nan` like any other value. That is what a cached discarded
        sample does in production: `calcAction` returns `outdict[T]["action"]` whenever the
        temperature is already in the cache, and `_rescue_action_rate_jitter` writes `nan` into
        that payload for a sample it drops.
        """
        sym, bro = self.phases

        def energy_density(pot, phase, T, include_decoupled=True):
            return (1.0 if phase is sym else 0.0) + T**4

        # An action that varies with temperature, so "the last computed one" is identifiable,
        # and a record of every value the solver actually asked for. The record is the
        # reference the assertions use: it is built by the mock, so the implementation cannot
        # satisfy it by carrying some other constant.
        computed: list[float] = []

        def action(pot, T, ps, pb, outdict, *a, **k):
            if zero_at_last_computed and len(computed) == 21:
                computed.append(0.0)
                return 0.0
            if nonfinite_call is not None and len(computed) == nonfinite_call:
                computed.append(np.nan)
                return np.nan
            value = 100.0 + float(np.squeeze(T))
            computed.append(value)
            return value

        TSYM = np.linspace(1.0, 0.5, 40)
        # P passes f_final = 0.99 at index 20 and stays there, so the walk saturates
        P = np.where(np.arange(TSYM.size) < 20, 0.5, 0.999)
        n = TSYM.size
        state = bd.PercolationState(TSYM=TSYM, Sr=np.zeros(n), Hr=np.zeros(n),
                                    Pr=np.zeros(n), Tb=np.zeros(n))
        settings = types.SimpleNamespace(f_final=0.99, acc_tperc=1e-2, time_temperature_mode="sound_speed",
                                         integral_method="ode")
        with mock.patch.object(bd, "energyDensity", energy_density), \
                mock.patch.object(bd, "calcAction", action), \
                mock.patch.object(bd, "h_eff_DS", lambda T, pot, phase: 5.0), \
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda T, pot: 0.0), \
                mock.patch.object(bd, "percIntegralODE_full_sweep",
                                  lambda TSYM, *a, **k: (None, P)):
            pas._compute_step3_profile(
                state, TSYM, P, settings, {}, types.SimpleNamespace(conversionFactor=1.0),
                sym, bro, 1.0, 1.0, TBROmin=0.1, TBROmax=5.0, verbose=False)
        return TSYM, state, computed

    def test_the_saturated_tail_carries_a_finite_action(self):
        _, state, _ = self.profile()
        self.assertTrue(np.all(np.isfinite(state.Sr)),
                        "the saturated branch left a non-finite action in Sr, which fails "
                        "percIntegralODE's validity test and floors the integrand")

    def test_the_tail_carries_the_last_action_the_solver_computed(self):
        """The carried value must be the last computed action, not merely some constant.

        The previous version of this test asserted only that the tail was constant and above
        100, which the fixture's first action also satisfies. It would have passed an
        implementation that carried the warmest action forward instead of the coldest, and the
        two differ by the whole temperature range. The reference here is the list the mocked
        `calcAction` filled, so the assertion fails for any other value.
        """
        _, state, computed = self.profile()
        self.assertGreater(len(computed), 1, "the fixture must compute more than one action")
        # The spread has to be wider than the tolerance of the np.isclose check below, not
        # merely nonzero: 101.0 against 100.73 is 2.6e-3 relative against an rtol of 1e-5.
        self.assertFalse(
            np.isclose(computed[0], computed[-1]),
            "the fixture cannot discriminate if the first and last action agree to within "
            "the tolerance the assertions below use")

        carried = state.Sr[len(computed):]
        self.assertGreater(carried.size, 0, "the fixture must saturate before the last point")
        np.testing.assert_allclose(
            carried, computed[-1], rtol=0, atol=0,
            err_msg="the saturated tail must hold the last action the solver computed")
        self.assertFalse(np.any(np.isclose(carried, computed[0])),
                         "carrying the first action forward is not the same fix")

    def test_the_computed_entries_are_left_alone(self):
        """The carry must not overwrite the actions that were computed."""
        _, state, computed = self.profile()
        np.testing.assert_allclose(state.Sr[:len(computed)], computed, rtol=0, atol=0)

    def test_the_carry_skips_a_non_positive_action_not_just_a_nan(self):
        """What `percIntegralODE` calls usable, which is narrower than finite.

        Its validity mask is `np.isfinite(S) & (S > 0.0) & ...` (`bubbledynamics.py:1250`), so
        a carried action of zero or less is dropped from the integral exactly as a `nan` is,
        and the tail would stop accumulating again. A non-positive action is not physical for a
        bounce; it means that solve did not succeed.

        Here the last computed action is zero, so the carry has to reach past it to the last
        positive one. The fixture computes 22 actions before it saturates, so the zero is
        placed on the 22nd.
        """
        _, state, computed = self.profile(zero_at_last_computed=True)
        self.assertEqual(computed[-1], 0.0, "the fixture must end on a non-positive action")
        carried = state.Sr[len(computed):]
        self.assertGreater(carried.size, 0)
        self.assertTrue(np.all(carried > 0.0),
                        "a carried action must satisfy the same validity test as a computed "
                        "one, or the integral floors it away")
        positive = [v for v in computed if v > 0.0]
        np.testing.assert_allclose(carried, positive[-1], rtol=0, atol=0)

    def test_step_3_leaves_no_hole_between_the_computed_part_and_the_tail(self):
        """The shape of the array the percolation integral is handed.

        `percIntegralODE` drops non-finite actions from its validity test, so a non-finite
        entry anywhere below the warm region floors the source and transport terms from there
        down. This pins the one thing `_compute_step3_profile` controls: everything from its
        first computed action to the coldest point comes back finite, with no gap where the
        carry starts.

        The claim is about step 3's own output and not about the profile for the rest of the
        solve. `calcAction` can hand back a non-finite action for a sample the jitter rescue
        discarded (`bubbledynamics.py:2705` writes `nan` into the cached payload and
        `bubbledynamics.py:900` returns it), and that rescue also writes `nan` straight into
        `state.Sr`, so a later stage can reintroduce a non-finite tail. Asserting otherwise
        here would be asserting something this function cannot promise.
        """
        _, state, _ = self.profile()
        finite = np.isfinite(np.asarray(state.Sr, dtype=float))
        self.assertTrue(finite.any(), "the fixture must compute at least one action")
        first = int(np.argmax(finite))
        self.assertTrue(
            np.all(finite[first:]),
            "a non-finite action appears after a finite one, so the percolation integral "
            "stops accumulating from there down")

    def test_the_carry_takes_the_last_finite_action_not_the_last_entry(self):
        """The `np.isfinite` filter in the carry, which nothing else exercises.

        A computed sample can itself be non-finite: `calcAction` returns
        `outdict[T]["action"]` for any temperature already in the cache, and
        `_rescue_action_rate_jitter` writes `nan` into that payload for a sample it discards.
        Here the non-finite action lands on the last temperature whose action is computed,
        which is the one position where the filter changes the answer -- anywhere further up
        the profile the entry immediately above the saturation point is finite anyway, and the
        unfiltered and filtered carries agree.

        Without the filter the tail would hold `nan`, `percIntegralODE` would mark every one
        of those samples invalid, and both its source and its transport term would be floored
        at `_LOG_GAMMA_SOURCE_FLOOR` over the whole tail: exactly the collapse this branch is
        about, reintroduced by the fix itself.
        """
        _, state, computed = self.profile(nonfinite_call=21)
        self.assertEqual(len(computed) - 1, 21,
                         "the fixture must put the non-finite action on the last computed "
                         "temperature, or the filter is not being asked to skip anything")
        self.assertFalse(np.isfinite(state.Sr[21]),
                         "the non-finite action must reach Sr, otherwise the carry below "
                         "never sees it")

        last_finite = computed[-2]
        self.assertTrue(np.isfinite(last_finite))
        carried = state.Sr[22:]
        self.assertGreater(carried.size, 0)
        np.testing.assert_allclose(
            carried, last_finite, rtol=0, atol=0,
            err_msg="the saturated tail must carry the last FINITE action, not the "
                    "non-finite entry that sits directly above it")


if __name__ == "__main__":
    unittest.main()
