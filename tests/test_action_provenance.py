"""A carried action must not read as a solved one.

Percolation step 3 stops solving the bounce once the fraction has passed `f_final` and carries
the last usable action across the rest of the support, because `percIntegralODE` needs valid
samples there to integrate the growth of the bubbles already nucleated. That made the tail
finite on purpose.

`_cold_walk_has_left_the_bounce_region` reads the same array to answer a different question:
where did the bounce stop existing, so that the controller stops expanding the cold range into
temperatures no solve can reach. It answered it by looking for a trailing run of non-finite
actions, which was the right test only while "not solved" and "not finite" were the same thing.

`PercolationState.Sr_computed` records which entries a solve produced. The test below is the
situation the two readings disagree about: a tail that is finite because it was carried, with
no solve behind it.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from transitionlistener import percolation_adaptive_gridbuilders as gb


class ColdWalkReadsProvenanceTests(unittest.TestCase):
    """The guard's inputs, driven directly."""

    def test_a_carried_tail_still_reads_as_past_the_bounce_region(self):
        """The case the carry created, and the reason this field exists.

        Ten solved actions followed by five carried copies of the last one. Every entry is
        finite, so the old reading sees no trailing failure and reports that the walk is still
        inside the bounce region. With the provenance it reports the truth: nothing was solved
        over those five points.
        """
        actions = np.concatenate([np.linspace(140.0, 150.0, 10), np.full(5, 150.0)])
        computed = np.concatenate([np.ones(10, bool), np.zeros(5, bool)])
        self.assertFalse(gb._cold_walk_has_left_the_bounce_region(actions),
                         "without provenance a carried tail looks solved, which is the "
                         "weakness this change removes")
        self.assertTrue(gb._cold_walk_has_left_the_bounce_region(actions, computed),
                        "five carried points are five points no solve reached")

    def test_a_solved_tail_is_not_mistaken_for_a_carried_one(self):
        """The guard must not fire merely because provenance is present."""
        actions = np.linspace(140.0, 150.0, 15)
        computed = np.ones(15, bool)
        self.assertFalse(gb._cold_walk_has_left_the_bounce_region(actions, computed))

    def test_a_non_finite_tail_still_fires_without_provenance(self):
        """The pre-existing behaviour, which step 2 relies on and which must not change.

        Step 2 never carries, so it passes no provenance, and its `nan` tail has to keep
        meaning what it always meant.
        """
        actions = np.concatenate([np.linspace(140.0, 150.0, 10), np.full(4, np.nan)])
        self.assertTrue(gb._cold_walk_has_left_the_bounce_region(actions))
        self.assertTrue(gb._cold_walk_has_left_the_bounce_region(actions, np.ones(14, bool)),
                        "a solve was attempted and produced nothing, which is the original "
                        "meaning and is independent of provenance")

    def test_a_mismatched_provenance_array_is_ignored_rather_than_trusted(self):
        """Defensive, because the two arrays are filled in different places.

        A provenance array of the wrong length cannot be aligned with the actions, so using it
        would silently mislabel points. Falling back to finiteness is the behaviour before this
        change rather than a guess.
        """
        actions = np.concatenate([np.linspace(140.0, 150.0, 10), np.full(5, 150.0)])
        self.assertFalse(gb._cold_walk_has_left_the_bounce_region(actions, np.ones(3, bool)))

    def test_a_tail_that_is_both_carried_and_unsolved_fires_once(self):
        """Carried entries and non-finite entries in the same tail."""
        actions = np.concatenate([np.linspace(140.0, 150.0, 8), np.full(3, 150.0),
                                  np.full(3, np.nan)])
        computed = np.concatenate([np.ones(8, bool), np.zeros(3, bool), np.ones(3, bool)])
        self.assertTrue(gb._cold_walk_has_left_the_bounce_region(actions, computed))


class ProfileRecordsProvenanceTests(unittest.TestCase):
    """The field is filled where the profile is built, not merely declared."""

    phases = (types.SimpleNamespace(name="sym"), types.SimpleNamespace(name="bro"))

    def profile(self):
        """Drive the real step-3 walk with a fraction that saturates partway down."""
        from unittest import mock
        from transitionlistener import bubbledynamics as bd
        from transitionlistener import percolation_adaptivestepsize as pas
        sym, bro = self.phases

        def energy_density(pot, phase, T, include_decoupled=True):
            return (1.0 if phase is sym else 0.0) + T**4

        def action(pot, T, ps, pb, outdict, *a, **k):
            return 100.0 + float(np.squeeze(T))

        TSYM = np.linspace(1.0, 0.5, 40)
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
        return state

    def test_the_walk_marks_solved_and_carried_entries_apart(self):
        """The mask must agree with where the solves actually stopped."""
        state = self.profile()
        self.assertIsNotNone(state.Sr_computed)
        self.assertEqual(state.Sr_computed.size, state.Sr.size)
        self.assertTrue(state.Sr_computed[0], "the first point is always solved")
        self.assertFalse(state.Sr_computed[-1], "the cold end is carried, not solved")
        # the carried entries are exactly the ones holding a repeat of the last solved action
        solved = np.flatnonzero(state.Sr_computed)
        carried = np.flatnonzero(~state.Sr_computed)
        self.assertGreater(carried.size, 0, "the fixture must saturate")
        self.assertEqual(carried.min(), solved.max() + 1,
                         "provenance must split the array in one place, not scatter")
        np.testing.assert_allclose(state.Sr[carried], state.Sr[solved.max()], rtol=0, atol=0)

    def test_the_guard_sees_the_carried_tail_through_the_state(self):
        """The two halves joined: what the walk records is what the guard reads."""
        state = self.profile()
        self.assertFalse(
            gb._cold_walk_has_left_the_bounce_region(state.Sr),
            "every entry is finite, so finiteness alone cannot see the carry")
        self.assertTrue(
            gb._cold_walk_has_left_the_bounce_region(state.Sr, state.Sr_computed),
            "with the provenance the carried tail is correctly read as unsolved")

    def test_the_state_carries_a_mask_the_same_length_as_the_support(self):
        from transitionlistener import bubbledynamics as bd
        n = 12
        state = bd.PercolationState(TSYM=np.linspace(90.0, 40.0, n), Sr=np.zeros(n),
                                    Hr=np.zeros(n), Pr=np.zeros(n), Tb=np.zeros(n))
        self.assertIsNone(state.Sr_computed,
                          "None until a profile is built, so a consumer can tell that it has "
                          "no provenance rather than reading an all-False array as fact")


if __name__ == "__main__":
    unittest.main()
