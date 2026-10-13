"""The adaptive percolation's temperature bookkeeping must not depend on the unit of temperature.

Temperatures are in the model's internal units, where the stable minimum sits at ``v_stable``, so an
internal temperature is ``v_stable`` times ``T/v``: below one wherever the transition is supercooled
by more than 1/v_stable, which is 4e-3 for the 2HDM and 1e-3 for the models built on
``internal_scale = 1000``. The 2HDM point at ``lambda3 = 5.8197`` on the marginal line reached
internal 2.4e-13 - the 2HDM's ``conversionFactor`` is one, so that is 2.4e-13 GeV as well - while
its hot end stayed at 76.2, and 32 of its 144 support points sat below internal one. Two shapes of
tolerance fail on such a support.

An absolute floor of one made the tolerances that decide whether a temperature is already present
1e-6 (presence) and 1e-10 (deduplication) in absolute terms, larger than the temperature itself
below internal 1e-6. Scaling by the largest entry of the array instead makes the tolerance a large
fraction of the cold end whenever the support spans decades: on the 144-point 2HDM support above it
was 7.6e-5, and 26 of the 143 adjacent pairs of its own support points fell inside it, 22 of them
more than a tenth apart and the median one 172 per cent apart. Either way
every point the jump refinement proposed inside a rise of P(T) counted as already present, the
controller found nothing to add with most of its budget unspent, and the acceptance gate rejected
the profile with "sharp unresolved jump in P(T)".

Which of the two bites depends on the model, and the array maximum is the one that reaches the
models shipped here. Their percolation temperatures are not extreme in internal units: across the
four dark U(1) regression datasets of this repository's validation scans, internal ``Tperc`` runs
from 0.21 to 162 and internal ``Tcrit`` from 84 to 316, so the floor of one is nearly inert there -
22 of their 1080 points have an internal ``Tperc`` below one - while a support reaching from
``Tcrit`` down to ``Tperc`` still spans a factor 2 to 790 and the array-scaled presence tolerance is
5e-5 to 1.5e-3 of the cold end.

Removing the floor changes nothing wherever a temperature of at least one is involved. Judging each
entry against its own size rather than against the array maximum is not confined to cold supports:
it changes any support that spans decades, in one direction only, because the pairwise tolerance is
never larger than the array-scaled one, so points can only be separated, never merged.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types
import unittest

import numpy as np
from scipy import optimize

from transitionlistener import bubbledynamics as bd
from transitionlistener import percolation_adaptive_gridbuilders as gridbuilders
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
    """Wherever a temperature of at least one is involved, the floor never acted.

    Scaling by the array maximum, on the other hand, acted there too, so the first test below
    separates the two: above one it asserts that the floor is dead and that the array maximum
    is not, on the same draws.
    """

    def test_the_presence_rule_is_relative_to_each_entry(self):
        """The presence predicate itself, not a tolerance helper nothing calls any more.

        Drawing the support above one keeps the removed floor out of the comparison, so what
        this pins is the surviving difference: each entry is judged against its own size. The
        support spans four decades, which is where the array maximum and the entry part
        company; with a support of one decade the two rules agree on almost every draw and
        the test would pass against either.
        """
        rng = np.random.default_rng(1)
        changed = 0
        for _ in range(400):
            values = 10 ** rng.uniform(0.0, 4.0, size=int(rng.integers(2, 20)))
            candidate = float(values[0] * (1.0 + 10 ** rng.uniform(-9.0, -3.0)))
            pairwise = bool(np.any(np.abs(values - candidate)
                                   <= 1e-6 * np.maximum(np.abs(values), abs(candidate))))
            array_max = bool(np.any(np.abs(values - candidate)
                                    <= 1e-6 * max(float(np.max(np.abs(values))), abs(candidate))))
            floored = bool(np.any(np.abs(values - candidate)
                                  <= 1e-6 * np.maximum(np.maximum(np.abs(values), abs(candidate)), 1.0)))
            self.assertEqual(par._temperature_present(values, candidate), pairwise)
            # the floor is dead above one; the array maximum is not
            self.assertEqual(pairwise, floored)
            changed += int(pairwise != array_max)
        self.assertGreater(changed, 20,
                           "the draws never separated the two rules, so this test would pass "
                           "against the array-scaled one as well")

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


class TemperaturesAreJudgedAgainstTheirOwnSizeTests(unittest.TestCase):
    """Each temperature is compared with its own magnitude, not the array's largest.

    Scaling the tolerance by `max(|candidate|, max|values|)` makes it a large fraction of the
    cold end whenever the support spans decades, which is exactly what a strongly supercooled
    transition produces: the support bank reaches up to the critical temperature while the
    transition happens many decades below it. Every refinement point proposed inside the rise
    of P(T) then counts as already present and is dropped, the candidate grid comes back
    unchanged, and the controller accepts a profile that its own acceptance gate rejects.

    Dropping the absolute floor of one is not enough on its own: a support bank holding hot
    points brings the same failure back, which is what the hot-start experiment hit.
    """

    def test_a_hot_support_point_does_not_swallow_a_cold_candidate(self):
        """The case that defeats an array-scaled tolerance."""
        support = np.array([1.0e-6, 4.3e-1])          # transition scale and the critical end
        candidate = 1.0e-6 * (1.0 + 1.0e-3)           # 0.1% away: a distinct temperature
        self.assertFalse(
            par._temperature_present(support, candidate),
            "a candidate 0.1% from the cold support point was counted as already present, "
            "which is what the array-scaled tolerance did")

    def test_a_genuine_duplicate_is_still_recognised(self):
        """Otherwise the refinement would add the same temperature forever."""
        support = np.array([1.0e-6, 4.3e-1])
        self.assertTrue(par._temperature_present(support, 1.0e-6 * (1.0 + 1.0e-9)))
        self.assertTrue(par._temperature_present(support, 4.3e-1))

    # Scaling by `max(|v|, |c|)` rather than by `|c|` alone is deliberate but is an equivalent
    # mutant on any input: writing `v = c (1 + d)`, the two differ only for `d` in
    # `(relative, relative / (1 - relative)]`, a window of relative width `relative**2`, and
    # for distant temperatures both refuse the match. The same holds for `min(|v|, |c|)` and
    # for `|v|` alone: over 300000 draws across 17 decades, half of them placed deliberately
    # inside the tolerance boundary, none of the three disagreed with the shipped form on a
    # single draw, while the array maximum disagreed on 121332 of them. There is therefore no
    # test here that fails when the scale is taken from one side of the comparison instead of
    # both. What matters, and what the tests do pin, is that the array's largest entry never
    # sets it, and that `relative` reaches the comparison at all.

    def test_it_still_separates_points_a_decade_apart_at_any_scale(self):
        """The property has to hold wherever the transition sits, not just near one."""
        for scale in (1.0e-12, 1.0e-6, 1.0, 1.0e3):
            support = np.array([scale, 10.0 * scale, 100.0 * scale])
            with self.subTest(scale=scale):
                self.assertFalse(par._temperature_present(support, 3.0 * scale),
                                 f"a point between two decades was swallowed at scale {scale}")

    def test_an_empty_support_matches_nothing(self):
        self.assertFalse(par._temperature_present(np.array([]), 1.0))
        self.assertEqual(par._temperatures_match(None, 1.0).size, 0)

    def test_the_relative_argument_reaches_the_comparison(self):
        """The three deduplication callers ask for 1e-10, four orders tighter than presence.

        Without this, `_temperatures_match` could ignore `relative` and serve every caller the
        presence tolerance of 1e-6: nothing else in the suite distinguishes the two, so
        `_probability_at`, `_controller_added_support_points` and
        `_finalize_percolation_state` would have no coverage of the value they pass.
        """
        values = np.array([1.0e-6, 4.3e-1])
        probe = 1.0e-6 * (1.0 - 1.0e-8)       # inside 1e-6, outside 1e-10
        self.assertTrue(np.any(par._temperatures_match(values, probe)))
        self.assertFalse(np.any(par._temperatures_match(values, probe, relative=1e-10)))

        # and through the two callers that reach it with 1e-10. The probe sits just below the
        # coldest support point, so only the stored-value branch can answer: interpolation is
        # outside its range and returns nothing.
        temps = np.array([4.3e-1, 1.0e-6])
        probs = np.array([0.9, 0.1])
        self.assertEqual(par._probability_at(1.0e-6, temps, probs), 0.1)
        self.assertIsNone(par._probability_at(probe, temps, probs),
                          "a temperature 1e-8 below the coldest support point was served that "
                          "point's stored probability, so the tolerance was the loose one")
        self.assertEqual(
            pas._controller_added_support_points(np.array([probe]), np.array([1.0e-6])), 1,
            "a support point 1e-8 from the free bank was counted as already paid for")


def _action_for_log10_rate(temperature: float, hubble: float, target: float) -> float:
    """The action whose ``log10(Gamma/H^4)`` at ``temperature`` is ``target``."""

    def residual(action: float) -> float:
        value = par._log10_gamma_h4_array(
            np.array([temperature]), np.array([action]), np.array([hubble]))
        return float(value[0]) - target

    return float(optimize.brentq(residual, 1.0e-6, 1.0e6, xtol=1e-14, rtol=1e-15))


class InertIntervalLookupTests(unittest.TestCase):
    """``_rate_interval_is_post_peak_inert`` must judge the interval it was asked about.

    Its endpoints are two entries of the support array, handed over by
    ``_largest_probability_jump_interval``. The lookup that turns them back into indices used
    an absolute tolerance scaled by the hottest support point, so on a support spanning
    decades it resolved a cold endpoint to a point several grid positions too hot. Measured on
    the 144-point 2HDM support that reached internal 2.4e-13 with its hot end at 76.2: 28 of
    the 143 adjacent pairs came back with at least one wrong index, 26 of them with
    ``high_index >= low_index``.

    A wrong index here is not a rounding error. The verdict vetoes the repair of a jump in
    P(T), so it decides whether the controller spends action budget inside the jump or accepts
    a profile the acceptance gate will reject.
    """

    def _profile(self, temperatures, rates, hubble=1.0e-16):
        hub = np.full(len(temperatures), hubble)
        acts = np.array([_action_for_log10_rate(float(T), hubble, float(r))
                         for T, r in zip(temperatures, rates)])
        return np.asarray(temperatures, dtype=float), acts, hub

    def test_the_cold_interval_is_read_off_its_own_rates(self):
        # A support spanning six decades: a hot shoulder, a resolved rate peak, and a cold
        # tail whose spacing is far below 1e-6 times the hot end.
        cold = np.array([5.0e-4, 4.0e-4, 3.2e-4, 2.56e-4, 2.048e-4, 1.6384e-4])
        temps = np.concatenate((np.array([76.4, 41.0, 20.0, 1.0e-2]), cold))
        # peak at 41.0; the cold tail is deep post-peak except for the pair under test
        rates = np.array([-30.0, -1.0, -8.0, -40.0, -40.0, -40.0, -40.0, -2.0, -2.0, -40.0])
        T, S, H = self._profile(temps, rates)
        log10 = par._log10_gamma_h4_array(T, S, H)
        np.testing.assert_allclose(log10, rates, rtol=0.0, atol=1e-8)
        # premise: the pair under test is two adjacent entries of the array
        high, low = float(cold[3]), float(cold[4])
        self.assertEqual(float(T[7]), high)
        self.assertEqual(float(T[8]), low)
        # premise: the array maximum makes the lookup tolerance wider than the cold gaps
        self.assertGreater(2.0e-6 * float(np.max(T)), abs(high - low))
        # its own endpoint rates are -2, above the inert threshold of -10, so the interval is
        # not inert and the repair must not be vetoed
        self.assertFalse(
            par._rate_interval_is_post_peak_inert(T, S, H, high, low, settings(), vw=1.0),
            "the interval was called inert on rates that belong to colder support points, so "
            "the repair of a jump in P(T) would be vetoed on the wrong evidence")

    def test_two_near_duplicate_endpoints_keep_their_own_indices(self):
        """The real support carries near-duplicates; the first match is not the right one.

        The measured 2HDM support holds pairs such as 61.642055 and 61.642054, 1.6e-8 apart in
        relative terms. Both are inside any sane presence tolerance of each other, so taking
        the first match resolves both endpoints to the hotter one, ``high_index >= low_index``
        fires and the function returns "not inert" without looking at a rate. Taking the
        closest match instead gives each endpoint its own index.
        """
        temps = np.array([76.4, 41.0, 61.642055e-6, 61.642054e-6, 20.0e-6, 1.0e-6])
        temps = np.sort(temps)[::-1]
        rates = np.array([-1.0 if abs(T - 41.0) < 1e-9 else -40.0 for T in temps])
        T, S, H = self._profile(temps, rates)
        self.assertTrue(
            par._rate_interval_is_post_peak_inert(
                T, S, H, 61.642055e-6, 61.642054e-6, settings(), vw=1.0),
            "the two near-duplicate endpoints collapsed onto one index, so the interval was "
            "refused before its rates were read")

    def test_a_genuinely_inert_cold_interval_is_still_inert(self):
        """Control: the fix must not turn the veto off altogether."""
        cold = np.array([5.0e-4, 4.0e-4, 3.2e-4, 2.56e-4, 2.048e-4, 1.6384e-4])
        temps = np.concatenate((np.array([76.4, 41.0, 20.0, 1.0e-2]), cold))
        rates = np.full(temps.size, -40.0)
        rates[1] = -1.0
        T, S, H = self._profile(temps, rates)
        self.assertTrue(
            par._rate_interval_is_post_peak_inert(
                T, S, H, float(cold[3]), float(cold[4]), settings(), vw=1.0),
            "an interval whose own rates are 30 orders below the peak is no longer inert")


class FinalizedActionTemperaturesTests(unittest.TestCase):
    """``_finalize_percolation_state`` is the third caller of the 1e-10 rule, and the only one
    that runs on both backends.

    ``calcPercAndEvolve`` calls it outside the ``algorithm_mode`` branch, so the fixed-step
    solver reaches it too. It reports which action temperatures the solve added that were not
    cached before, and the two forms this replaces both scaled that comparison by something
    other than the temperature being judged: the floor of one made it 1e-10 absolute, and the
    array maximum made it 1e-10 times the hottest entry of the previous bank. Either way a
    genuinely new action temperature at a cold support point was reported as already cached and
    the count of added points came back short.
    """

    def _finalize(self, previous, added):
        state = bd.PercolationState(
            TSYM=np.asarray(previous, dtype=float),
            Sr=np.zeros(len(previous)), Hr=np.zeros(len(previous)),
            Pr=np.zeros(len(previous)), Tb=np.zeros(len(previous)),
            support_bank=np.asarray(previous, dtype=float),
            free_support_bank=None,
        )
        outdict = {float(T): {"action": 150.0} for T in list(previous) + list(added)}
        bd._finalize_percolation_state(state, outdict, np.asarray(previous, dtype=float))
        return np.asarray(state.action_temperatures, dtype=float)

    def test_a_new_cold_action_temperature_is_reported_as_new(self):
        # The previous bank spans from the transition scale up to the critical end, which is
        # what makes an array-scaled tolerance 1e-10 * 4.3e-1 = 4.3e-11 at a support point of
        # 1e-6, and the floor of one makes it 1e-10. The new temperature is 1e-8 away in
        # relative terms, so 1e-14 in absolute terms: inside both of the old windows and
        # outside 1e-10 times its own size.
        previous = [4.3e-1, 1.0e-6]
        fresh = 1.0e-6 * (1.0 - 1.0e-8)
        self.assertGreater(abs(fresh - 1.0e-6), 1.0e-10 * abs(fresh))
        self.assertLess(abs(fresh - 1.0e-6), 1.0e-10 * max(max(previous), 1.0))
        added = self._finalize(previous, [fresh])
        self.assertEqual(added.size, 1,
                         "an action temperature 1e-8 from a cold support point was reported as "
                         "already cached, so the solve's own work went unrecorded")
        self.assertAlmostEqual(float(added[0]) / fresh, 1.0, places=12)

    def test_a_temperature_that_was_already_cached_is_not_reported_twice(self):
        """Control: the rule must still recognise what is genuinely there."""
        previous = [4.3e-1, 1.0e-6]
        self.assertEqual(self._finalize(previous, [1.0e-6 * (1.0 + 1.0e-13)]).size, 0)
        self.assertEqual(self._finalize(previous, []).size, 0)


class ColdWindowsAreRelativeTests(unittest.TestCase):
    """Two window tolerances that still carried a floor of one.

    Both are 1e-10 times a scale floored at one, so both are 1e-10 absolute for any window
    below one in internal units - a tenth of an internal temperature of 1e-9. Both are latent
    rather than active: they need an internal window below about 1e-8 to be a per cent of it,
    and no recorded run places a step-1 window that cold. Latent is not unreachable - the
    supercooled 2HDM support reaches internal 2.4e-13, where 1e-10 is 400 times the
    temperature - so the claim is about where the windows have been observed, not about where
    they can go. They are tested because the expression is the one the presence tolerance had,
    and the scale at which each stops working is worth recording rather than rediscovering.
    """

    def _step1_profile(self, scale):
        # Points on both sides of the startup centre and close to it, which is what a grid
        # refined around the percolation temperature looks like: 35.3 and 34.7 are 0.86% from
        # the centre, so at a scale of 1e-10 they sit 3e-11 away and a dead band of 1e-10
        # absolute swallows them.
        temperatures = np.asarray(
            [62.0, 50.0, 36.0, 35.3, 35.0, 34.7, 33.0, 30.0, 28.0], dtype=float) * scale
        return gridbuilders._Step1ApproxProfile(
            sampled_temperatures=temperatures,
            temperatures=temperatures,
            actions=150.0 * temperatures,
            gamma=np.full(temperatures.size, 1.0e-40),
            hubble_sym=np.full(temperatures.size, 1.0e-16),
            log_gamma_h4_sym=np.full(temperatures.size, -92.0),
            i_approx=np.zeros(temperatures.size),
            probabilities=np.zeros(temperatures.size),
            tperc_approx=35.0 * scale,
            startup_center=35.0 * scale,
            window_high=62.0 * scale,
            window_low=28.0 * scale,
            window_source="test",
        )

    def test_the_step1_active_grid_is_the_same_in_any_unit_of_temperature(self):
        """The dead band around the startup centre must scale with the window.

        With a floor of one it is 1e-10 wide in absolute terms, so at an internal temperature
        of 1e-9 ``_count_high`` and ``_count_low`` stop seeing the points that are there and
        the fill loops add more, spending action budget on a grid that was already built.
        """
        opts = settings(n_action_min=8, weight=0.5, f_start=1.0e-3)
        warm = gridbuilders._build_step1_active_grid(self._step1_profile(1.0), opts)
        cold = gridbuilders._build_step1_active_grid(self._step1_profile(1.0e-10), opts)
        self.assertIsNotNone(warm)
        self.assertIsNotNone(cold)
        self.assertEqual(np.asarray(cold).size, np.asarray(warm).size,
                         "the cold window produced a different number of support points")
        np.testing.assert_allclose(np.asarray(cold) * 1.0e10, np.asarray(warm), rtol=1e-9)

    def test_a_cached_action_below_a_cold_window_stays_out_of_it(self):
        """The cached-action window must not reach a tenth of the way below its own floor."""
        outdict = {1.0e-9: {"action": 150.0, "trantype": 1},
                   9.5e-10: {"action": 151.0, "trantype": 1},
                   5.0e-9: {"action": 152.0, "trantype": 1}}
        inside = bd._transition_action_outdict_temperatures(
            outdict, None, None, tmin=1.0e-9, tmax=1.0e-8)
        self.assertEqual(sorted(float(t) for t in inside), [1.0e-9, 5.0e-9],
                         "a cached action 5% below the requested window was reported inside "
                         "it, which an absolute tolerance of 1e-10 admits")
        # control: a warm window is unaffected, the floor never acted there
        warm = bd._transition_action_outdict_temperatures(
            {10.0: {"action": 1.0, "trantype": 1}, 9.5: {"action": 2.0, "trantype": 1}},
            None, None, tmin=10.0, tmax=100.0)
        self.assertEqual([float(t) for t in warm], [10.0])


if __name__ == "__main__":
    unittest.main()
