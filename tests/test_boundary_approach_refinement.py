"""Refinement of the interval that ends where the bounce stops solving.

The coldest end of a traced phase pair is a boundary, not a sample: below the boundary the
symmetric phase is gone and the bounce has nothing to tunnel out of, so the action is not a
number there. The nucleation rate is still rising as the boundary is approached, which means
the support points that decide the percolation integral lie inside the last interval. No
other part of the controller places them: the step-size test compares two rates, and an
interval with a rate at one end only never registers as unresolved.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import percolation_adaptive_gridbuilders as gridbuilders
from transitionlistener import percolation_adaptivestepsize as pas


# The measured interval of the dark U(1) point that failed on figure 13's grid:
# l = 1e-4, v_GeV = 1.2689610031679234e7, g_tilde = 2.69, benchmark precision. At the rebuild
# where the approach first fires, the coldest support point carrying an action sits at
# T_int = 5.701880 in the model's internal temperature units and the traced phase pair ends
# at T_int = 5.553053. The percolation integral reaches f_perc only once the support extends
# to T_int = 5.680, where log10(Gamma/H^4) = 11.5. The sixth decimal of both temperatures
# moves between runs, so the prose quotes four; the fixtures keep the digits as measured.
WARM_END = 5.701880
BOUNDARY = 5.553053
INTEGRAL_NEEDS = 5.680


def settings(**overrides):
    values = dict(n_action_max=10, n_action_increment=5, maxit=10, f_start=1e-3, f_final=0.99,
                  max_log10_rate_step=12.0)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def boundary_grid(log10_rate, temperatures, *, max_new_points=8):
    """Run the candidate builder on a prescribed rate history."""

    log10_rate = np.asarray(log10_rate, dtype=float)
    temps = np.asarray(temperatures, dtype=float)
    with mock.patch.object(gridbuilders, "_log10_gamma_h4_array", lambda t, a, h: log10_rate):
        return gridbuilders.build_boundary_approach_candidates(
            temps, np.zeros(temps.size), np.ones(temps.size), settings(),
            max_new_points=max_new_points,
        )


class BoundaryApproachGridTests(unittest.TestCase):
    """The placement of the points between the coldest sample and the boundary."""

    def test_the_interval_losing_its_action_is_sampled(self):
        grid = boundary_grid([-2.0, np.nan], [WARM_END, BOUNDARY])
        self.assertIsNotNone(grid)
        self.assertEqual(grid.size, 8)
        self.assertTrue(np.all((grid < WARM_END) & (grid > BOUNDARY)),
                        f"points outside the interval: {grid}")

    def test_the_points_spread_over_the_interval_rather_than_crowding_the_boundary(self):
        grid = boundary_grid([-2.0, np.nan], [WARM_END, BOUNDARY])
        # A ladder that halves the remaining distance to the boundary puts every point
        # within a thousandth of the boundary and samples none of the body of the interval,
        # which is the part where the bounce still solves. The geometric spread keeps the
        # samples separated: the widest logarithmic gap between neighbours is no larger than
        # twice the narrowest.
        gaps = np.diff(np.log(grid[::-1]))
        self.assertLess(float(np.max(gaps)) / float(np.min(gaps)), 2.0,
                        f"uneven spacing: {gaps}")

    def test_one_approach_reaches_the_temperature_the_integral_needs(self):
        """Eight points on the measured interval sample below T_int = 5.680.

        The recorded run places them at T_int = 5.6851 down to 5.5694 and the three warmest
        return an action, the coldest of those three at T_int = 5.6518. So the approach has
        to put several points below T_int = 5.680, not merely its coldest one, or the ones
        that solve all sit above the temperature the integral needs.
        """
        grid = boundary_grid([-2.0, np.nan], [WARM_END, BOUNDARY])
        below = grid[grid < INTEGRAL_NEEDS]
        self.assertGreaterEqual(below.size, 3,
                                f"only {below.size} of 8 points are below {INTEGRAL_NEEDS}: {grid}")
        self.assertAlmostEqual(float(np.max(grid)), 5.685, places=3)
        self.assertAlmostEqual(float(np.min(grid)), 5.569, places=3)

    def test_the_interval_with_the_highest_rate_is_served_first(self):
        temps = [10.0, 9.0, 8.0, 7.0]
        grid = boundary_grid([-5.0, np.nan, -2.0, np.nan], temps)
        self.assertIsNotNone(grid, "no interval was served")
        self.assertTrue(np.all((grid < 8.0) & (grid > 7.0)),
                        f"the colder, livelier boundary was not chosen: {grid}")

    def test_the_builder_follows_the_temperature_order_not_the_array_order(self):
        # The support arrays reach the builder in whatever order the bank holds them. Read in
        # array order these three samples hold no boundary interval at all: the first pair has
        # its missing rate at the warmer end and the second pair carries a rate at both ends.
        # Sorted hot to cold the interval from 8.0 to 5.6 is the candidate.
        grid = boundary_grid([np.nan, -5.0, -2.0], [5.6, 10.0, 8.0])
        self.assertIsNotNone(grid, "the samples were read in array order")
        self.assertTrue(np.all((grid < 8.0) & (grid > 5.6)), f"wrong interval: {grid}")

    def test_nothing_while_every_sample_carries_an_action(self):
        self.assertIsNone(boundary_grid([-3.0, -2.0, -1.0], [10.0, 8.0, 6.0]))

    def test_nothing_where_the_rate_is_already_dead(self):
        # A cold tail whose rate sits a thousand decades below the peak can no longer
        # change the true vacuum fraction, so refining towards it buys nothing.
        self.assertIsNone(boundary_grid([0.0, -1500.0, np.nan], [10.0, 8.0, 6.0]))

    def test_no_points_without_a_budget(self):
        # Removing the budget guard is an equivalent mutation: a geometric spread asked for
        # two points and stripped of its endpoints is already empty, and an empty grid is
        # reported as no candidate. The test records the behaviour, not a live guard.
        self.assertIsNone(boundary_grid([-2.0, np.nan], [WARM_END, BOUNDARY], max_new_points=0))

    def test_no_points_across_a_vanishing_interval(self):
        self.assertIsNone(boundary_grid([-2.0, np.nan], [BOUNDARY, BOUNDARY * (1.0 - 1e-14)]))

    def test_no_points_below_a_boundary_at_zero_temperature(self):
        self.assertIsNone(boundary_grid([-2.0, np.nan], [WARM_END, 0.0]))

    def test_a_vanishing_action_at_the_colder_end_is_not_a_boundary(self):
        # `logGamma` returns `+inf` where the action vanishes and not a number where no
        # action was obtained. A colder end carrying `+inf` has an overwhelming rate, not a
        # missing one, so the integral needs no extra support above it.
        self.assertIsNone(boundary_grid([-2.0, np.inf], [WARM_END, BOUNDARY]))

    def test_a_refused_candidate_does_not_hide_a_usable_one(self):
        # Two intervals lose their action. The livelier of the two ends at zero temperature,
        # where no grid can be placed, so the other one has to be served instead of the
        # builder reporting that it found nothing.
        grid = boundary_grid([-5.0, np.nan, -2.0, np.nan], [10.0, 9.0, 8.0, 0.0])
        self.assertIsNotNone(grid, "the usable interval from 10.0 to 9.0 was not sampled")
        self.assertTrue(np.all((grid < 10.0) & (grid > 9.0)), f"wrong interval: {grid}")


def _allowance_from_the_guard(opts):
    """The fewest spent points at which the guard refuses a further boundary approach.

    Read off the guard rather than recomputed, so a cap written somewhere other than
    `_BOUNDARY_APPROACH_ROUNDS` times the batch is still measured here.
    """

    state = bd.PercolationState(TSYM=np.zeros(1), Sr=np.zeros(1), Hr=np.zeros(1),
                                Pr=np.zeros(1), Tb=np.zeros(1))
    for spent in range(0, 1000):
        state.boundary_refine_points = spent
        if pas._boundary_approach_is_exhausted(state, opts):
            return spent
    raise AssertionError("the guard never reports the allowance as spent")


class BoundaryApproachAllowanceTests(unittest.TestCase):
    """How many support points a single scan point may spend on boundary approaches."""

    def test_the_allowance_is_two_approaches_worth_of_points(self):
        batch = pas._boundary_approach_batch(settings())
        state = bd.PercolationState(TSYM=np.zeros(1), Sr=np.zeros(1), Hr=np.zeros(1),
                                    Pr=np.zeros(1), Tb=np.zeros(1))
        for spent, exhausted in ((0, False), (batch - 1, False), (batch, False),
                                 (2 * batch - 1, False), (2 * batch, True),
                                 (2 * batch + 1, True)):
            state.boundary_refine_points = spent
            with self.subTest(spent=spent):
                self.assertEqual(
                    pas._boundary_approach_is_exhausted(state, settings()), exhausted,
                    f"{spent} points spent of an allowance of {2 * batch}")

    def test_the_allowance_is_the_batch_the_refinement_asks_for(self):
        """The cap and the request are the same number, for every setting.

        A cap that does not divide into the request admits a further approach and then
        overruns it, which is how an allowance documented as two approaches becomes three.
        """
        for increment in (1, 3, 5, 8, 12, 16):
            with self.subTest(n_action_increment=increment):
                opts = settings(n_action_increment=increment)
                batch = pas._boundary_approach_batch(opts)
                self.assertEqual(
                    pas._BOUNDARY_APPROACH_ROUNDS * batch,
                    _allowance_from_the_guard(opts),
                    "the cap the guard enforces is not a whole number of approaches")

    def test_the_shipped_batch_is_the_one_the_measurement_used(self):
        # `n_action_increment` ships at 5 and the batch has a floor of 8, which is the count
        # figure 13's grid was measured with. A batch that followed the setting down would
        # place a different grid from the one the recovered points were measured on.
        from transitionlistener.config import PercolationConf
        self.assertEqual(PercolationConf.n_action_increment, 5)
        self.assertEqual(
            pas._boundary_approach_batch(settings(n_action_increment=5)), 8)

    def test_the_refinement_asks_for_exactly_one_approach_of_points(self):
        asked = {}

        def capture(temperatures, actions, hubble, opts, *, max_new_points):
            asked["max_new_points"] = int(max_new_points)
            return None

        opts = settings(n_action_increment=12)
        with mock.patch.object(pas, "_build_rate_step_grid", return_value=None), \
                mock.patch.object(pas, "build_boundary_approach_candidates", capture):
            pas._try_dynamiczoomwindow_rate_refinement(
                np.linspace(10.0, 5.0, 4), np.ones(4), np.ones(4), opts,
                lambda grid, reason, **kwargs: False)
        self.assertEqual(asked["max_new_points"], pas._boundary_approach_batch(opts))

    def test_both_support_solvers_consult_the_allowance(self):
        """Scope: this covers reach, not the guard's effect.

        Both adaptive support solvers carry their own copy of `_apply_candidate_grid`, and a
        guard placed in one of the two is dead on the path the other drives. The test reads
        the two closures' code objects and asserts each loads
        `_boundary_approach_is_exhausted`. It says nothing about what the closures do with
        the answer; `test_the_allowance_is_two_approaches_worth_of_points` covers that for
        the guard itself. The count of solvers is measured here rather than assumed, so a
        third copy fails the test instead of going uncovered.
        """
        solvers = [pas._initial_percolation_scan_dynamiczoomwindow,
                   pas._refine_percolation_temperature_dynamiczoomwindow]
        found = 0
        for solver in solvers:
            closures = [c for c in solver.__code__.co_consts
                        if getattr(c, "co_name", None) == "_apply_candidate_grid"]
            self.assertEqual(len(closures), 1,
                             f"{solver.__name__} does not hold exactly one "
                             "_apply_candidate_grid closure")
            found += 1
            self.assertIn("_boundary_approach_capacity", closures[0].co_names,
                          f"{solver.__name__} admits a boundary approach without asking "
                          "for its capacity, which is where the allowance is enforced")
        self.assertEqual(found, 2, "the number of support solvers changed")

    def test_an_exhausted_allowance_refuses_the_grid(self):
        """The real decision, not a stand-in: `None` refuses, a number is the capacity.

        The test above reads the closures' code objects, so a change that calls
        `_boundary_approach_is_exhausted` and discards the answer passes it. This drives
        `_boundary_approach_capacity`, which both closures call, so the answer has to decide.
        """
        cfg = settings(n_action_increment=5, n_action_max=60)
        batch = pas._boundary_approach_batch(cfg)
        cap = pas._BOUNDARY_APPROACH_ROUNDS * batch
        for spent, refused in ((0, False), (batch, False), (cap - 1, False),
                               (cap, True), (cap + batch, True)):
            with self.subTest(points_spent=spent):
                state = _state(spent)
                granted = pas._boundary_approach_capacity(state, cfg, 0)
                self.assertEqual(granted is None, refused,
                                 f"with {spent} of {cap} allowed points spent the approach "
                                 f"was {'refused' if granted is None else 'granted ' + str(granted)}")

    def test_the_capacity_is_the_batch_the_builder_asked_for(self):
        """A large increment must not arrive truncated by a smaller `n_action_max`."""
        cfg = settings(n_action_increment=200, n_action_max=60)
        state = _state(0)
        granted = pas._boundary_approach_capacity(state, cfg, 0)
        batch = pas._boundary_approach_batch(cfg)
        self.assertEqual(batch, 200)
        self.assertIsNotNone(granted)
        self.assertGreaterEqual(granted, batch,
                                f"the builder asks for {batch} points and the update grants "
                                f"{granted}, so the grid arrives truncated and the allowance "
                                f"is spent on approaches that never install it")


def _state(boundary_refine_points):
    """A `PercolationState` whose only field that matters here is the boundary counter."""

    empty = np.asarray([], dtype=float)
    return bd.PercolationState(TSYM=empty, Sr=empty, Hr=empty, Pr=empty, Tb=empty,
                               boundary_refine_points=int(boundary_refine_points))


def rate_step_grid(log10_rate, temperatures, *, max_new_points):
    """Run the step-size builder on a prescribed rate history."""

    log10_rate = np.asarray(log10_rate, dtype=float)
    temps = np.asarray(temperatures, dtype=float)
    with mock.patch.object(gridbuilders, "_log10_gamma_h4_array", lambda t, a, h: log10_rate):
        return gridbuilders._build_rate_step_grid(
            temps, np.zeros(temps.size), np.ones(temps.size), settings(),
            max_new_points=max_new_points)


class RateStepHandsOverTheBoundaryIntervalTests(unittest.TestCase):
    """The step-size builder leaves the boundary interval to the builder above.

    The premise the boundary approach rests on. Both fixtures give the builder more budget
    than it can spend, because a builder that claimed the interval and then ran out of points
    before reaching it would look the same as one that left it alone.
    """

    def test_an_interval_without_a_cold_rate_is_not_claimed_as_a_rate_step(self):
        # 12 decades sit between 10 and 9, which does not exceed max_log10_rate_step, so the
        # interval from 9 to 8, whose cold end has no rate, is the only one that could be
        # claimed. The builder has to find nothing at all.
        self.assertIsNone(rate_step_grid([-14.0, -2.0, np.nan], [10.0, 9.0, 8.0],
                                         max_new_points=60))

    def test_a_genuine_rate_step_is_still_served_beside_such_an_interval(self):
        # 198 decades between 10 and 9 make that interval the rate step. With a budget of 60
        # the builder can serve it and still have points left, so any point it places in the
        # interval from 9 to 8 is a claim on the boundary interval rather than a shortage.
        grid = rate_step_grid([-200.0, -2.0, np.nan], [10.0, 9.0, 8.0], max_new_points=60)
        self.assertIsNotNone(grid)
        self.assertTrue(np.all((grid < 10.0) & (grid > 9.0)), f"wrong interval: {grid}")


class ControllerReachesTheBoundaryBuilderTests(unittest.TestCase):
    """The wiring: the controller asks for a boundary approach once the rate step is served."""

    def test_the_boundary_approach_is_tried_when_no_rate_step_is_left(self):
        T = np.linspace(10.0, 1.0, 6)
        # The true vacuum fraction does not reach f_final on this grid, so the controller
        # looks for something to refine.
        P = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.5])
        calls = []

        def apply_candidate_grid(grid, reason, **kwargs):
            calls.append(reason)
            return reason == "boundary_approach"

        common = dict(stage="test", iterator=1, previous_estimate=None, vw=1.0, hot_bound=20.0,
                      apply_candidate_grid=apply_candidate_grid,
                      promote_cached_action_points=lambda reason: False,
                      promote_hot_head_cached_points=lambda: False)
        with mock.patch.object(pas, "_hot_head_underresolved", return_value=(False, None, None)), \
                mock.patch.object(pas, "_build_rate_step_grid", return_value=None), \
                mock.patch.object(pas, "build_boundary_approach_candidates",
                                  return_value=np.linspace(6.0, 5.6, 8)):
            decision, *_ = pas._dynamiczoomwindow_post_sweep_controller(
                T, np.ones_like(T), P, np.full_like(T, 100.0), settings(), None,
                remaining_budget=0, **common)
        self.assertEqual(decision, "retry")
        self.assertIn("boundary_approach", calls)

    def test_the_controller_gives_up_when_the_boundary_builder_finds_nothing(self):
        T = np.linspace(10.0, 1.0, 6)
        P = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.5])
        common = dict(stage="test", iterator=1, previous_estimate=None, vw=1.0, hot_bound=20.0,
                      apply_candidate_grid=lambda grid, reason, **kwargs: False,
                      promote_cached_action_points=lambda reason: False,
                      promote_hot_head_cached_points=lambda: False)
        with mock.patch.object(pas, "_hot_head_underresolved", return_value=(False, None, None)), \
                mock.patch.object(pas, "_build_rate_step_grid", return_value=None), \
                mock.patch.object(pas, "build_boundary_approach_candidates", return_value=None):
            decision, *_ = pas._dynamiczoomwindow_post_sweep_controller(
                T, np.ones_like(T), P, np.full_like(T, 100.0), settings(), None,
                remaining_budget=0, **common)
        self.assertEqual(decision, "limit")


if __name__ == "__main__":
    unittest.main()
