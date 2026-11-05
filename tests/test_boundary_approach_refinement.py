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
from pathlib import Path
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
        the two closures' code objects and asserts each loads `_boundary_approach_capacity`.
        It says nothing about what the closures do with the answer, which
        `BoundaryApproachCounterTests` covers by running each of them on a real candidate
        grid. The count of solvers is measured here rather than assumed, so a third copy
        fails the test instead of going uncovered.
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


#: The two adaptive support solvers. Each carries its own `_apply_candidate_grid` closure, so
#: a counter or a refusal written into one of them is dead on the path the other drives.
SUPPORT_SOLVERS = (
    pas._initial_percolation_scan_dynamiczoomwindow,
    pas._refine_percolation_temperature_dynamiczoomwindow,
)


def _closure_code(solver):
    """The `_apply_candidate_grid` code object the named support solver defines."""

    found = [c for c in solver.__code__.co_consts
             if getattr(c, "co_name", None) == "_apply_candidate_grid"]
    if len(found) != 1:
        raise AssertionError(
            f"{solver.__name__} holds {len(found)} _apply_candidate_grid closures, not one")
    return found[0]


def apply_candidate_grid_of(solver, state, opts, active_grid, rebuilds):
    """The solver's own `_apply_candidate_grid`, ready to call outside a percolation run.

    The closure is the module's code object, rebuilt over cells holding the state, the
    settings, the support window and the two callbacks a run would give it, so the body that
    grants the capacity, calls the real support update and advances the counter is the one
    that executes. The solvers set up that state by solving bounces over a traced phase pair,
    which no unit test can do, and a stand-in closure would leave the counter and the refusal
    uncovered, which is what reaching only `_boundary_approach_capacity` already did.

    A code object carries no keyword defaults, so the rebuilt closure is given the zero the
    source declares. `test_the_boundary_approach_passes_no_extra_budget` pins that the
    refinement helper passes nothing and therefore uses that default.
    """

    code = _closure_code(solver)

    def note_rebuild(reason, old_grid, new_grid):
        rebuilds.append((reason, np.asarray(new_grid, dtype=float)))

    def reset_arrays(new_grid):
        state.TSYM = np.asarray(new_grid, dtype=float)

    cells = {"TSYM": np.asarray(active_grid, dtype=float),
             "_note_rebuild": note_rebuild,
             "grid": types.SimpleNamespace(tmin=0.0, tmax=1.0e4),
             "outdict": {},
             "phase_broken": None,
             "phase_symmetric": None,
             "settings": opts,
             "state": state}
    # The two solvers name their array reset differently (`_reset_active_arrays` in step 2,
    # `_reset_state_arrays` in step 3); every other free variable is shared.
    closure = tuple(types.CellType(cells[name] if name in cells else reset_arrays)
                    for name in code.co_freevars)
    function = types.FunctionType(code, vars(pas), "_apply_candidate_grid", None, closure)
    function.__kwdefaults__ = {"extra_budget": 0}
    return function


def _support_state(active_grid, *, spent=0, free_support_bank=()):
    """A `PercolationState` holding the support banks the closure reads and writes."""

    active = np.asarray(active_grid, dtype=float)
    return bd.PercolationState(
        TSYM=active.copy(), Sr=np.zeros_like(active), Hr=np.zeros_like(active),
        Pr=np.zeros_like(active), Tb=np.zeros_like(active),
        support_bank=active.copy(),
        free_support_bank=np.asarray(free_support_bank, dtype=float),
        boundary_refine_points=int(spent),
    )


def run_boundary_approach(solver, state, opts, active_grid, rebuilds,
                          *, log10_rate=(-2.0, np.nan), temperatures=(WARM_END, BOUNDARY)):
    """Drive one boundary approach through the path a running controller takes.

    `_try_dynamiczoomwindow_rate_refinement` is what the post-sweep controller calls: it
    builds the candidate grid with `build_boundary_approach_candidates` and hands it to the
    solver's `_apply_candidate_grid` under the reason `boundary_approach`. The prescribed rate
    history carries a step of a single decade, below `max_log10_rate_step`, so the ordinary
    rate repair finds nothing and the boundary approach is the only refinement offered.
    """

    rates = np.asarray(log10_rate, dtype=float)
    temps = np.asarray(temperatures, dtype=float)
    apply_grid = apply_candidate_grid_of(solver, state, opts, active_grid, rebuilds)
    with mock.patch.object(gridbuilders, "_log10_gamma_h4_array", lambda t, a, h: rates):
        return pas._try_dynamiczoomwindow_rate_refinement(
            temps, np.zeros(temps.size), np.ones(temps.size), opts, apply_grid)


class BoundaryApproachCounterTests(unittest.TestCase):
    """What the closures do with the allowance, run as the controller runs them.

    `BoundaryApproachAllowanceTests` covers `_boundary_approach_capacity` on its own and reads
    the closures' code objects for the call. Neither reaches the lines that spend the
    allowance: with the counter increment deleted every rebuild ran past the documented cap
    and the suite stayed green, and a closure that asked for the capacity and dropped the
    answer refused nothing. These tests run each closure on a real candidate grid instead.
    """

    def test_the_number_of_support_solvers_is_the_number_covered(self):
        # Measured rather than assumed: a third adaptive solver fails this instead of going
        # uncovered by every test below, each of which iterates over `SUPPORT_SOLVERS`.
        names = [name for name, value in vars(pas).items()
                 if callable(value) and getattr(value, "__module__", None) == pas.__name__
                 and any(getattr(c, "co_name", None) == "_apply_candidate_grid"
                         for c in getattr(getattr(value, "__code__", None), "co_consts", ()))]
        self.assertCountEqual(names, [solver.__name__ for solver in SUPPORT_SOLVERS])

    def test_the_boundary_approach_passes_no_extra_budget(self):
        """The capacity the closure grants is the batch, not a budget the caller supplied.

        Every test here runs the closure at `extra_budget` = 0, and that is the value a run
        uses only as long as the refinement helper passes nothing.
        """
        asked = {}

        def apply_grid(grid, reason, **kwargs):
            asked[reason] = dict(kwargs)
            return True

        with mock.patch.object(pas, "_build_rate_step_grid", return_value=None), \
                mock.patch.object(pas, "build_boundary_approach_candidates",
                                  return_value=np.linspace(6.0, 5.6, 8)):
            pas._try_dynamiczoomwindow_rate_refinement(
                np.linspace(10.0, 5.0, 4), np.ones(4), np.ones(4), settings(), apply_grid)
        self.assertEqual(asked.get("boundary_approach"), {})

    def test_the_increment_is_the_only_thing_that_writes_the_counter(self):
        """Nothing else in the module may set `boundary_refine_points`.

        The tests below give each closure the two callbacks a run would give it, so a
        statement that zeroed the counter from somewhere else in the solver, the array reset
        on a rebuild being the obvious place, would not show up in them. Read off the module's
        syntax tree instead: two writes, the increment in each of the two closures.
        """
        import ast

        source = Path(pas.__file__).read_text()
        writes = []
        for node in ast.walk(ast.parse(source)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr == "boundary_refine_points":
                    writes.append((node.lineno, type(node).__name__))
        self.assertEqual(
            [kind for _, kind in writes], ["AugAssign", "AugAssign"],
            f"the counter is written at lines {[line for line, _ in writes]}, and the only "
            "writes allowed are the increment in each support solver's _apply_candidate_grid")

    def test_the_counter_advances_by_the_points_the_approach_added(self):
        for solver in SUPPORT_SOLVERS:
            with self.subTest(solver=solver.__name__):
                opts = settings(n_action_increment=5, n_action_max=10)
                expected = pas._boundary_approach_batch(opts)
                active = np.linspace(10.0, WARM_END, 3)
                state = _support_state(active)
                rebuilds = []
                self.assertTrue(run_boundary_approach(solver, state, opts, active, rebuilds),
                                "the approach was not applied at all")
                self.assertEqual(len(rebuilds), 1, "the support was not rebuilt once")
                self.assertEqual(rebuilds[0][0], "boundary_approach")
                self.assertEqual(rebuilds[0][1].size, active.size + expected)
                self.assertEqual(int(state.boundary_refine_points), expected,
                                 f"{expected} points were admitted and the counter stands at "
                                 f"{int(state.boundary_refine_points)}, so the allowance is "
                                 "not being spent")

    def test_a_point_whose_action_already_ran_does_not_spend_the_allowance(self):
        """The counter follows the points that need a new action, not the grid size.

        A candidate temperature already in the free support bank has had its action computed,
        so admitting it costs nothing and must not count against the two approaches.
        """
        for solver in SUPPORT_SOLVERS:
            with self.subTest(solver=solver.__name__):
                opts = settings(n_action_increment=5, n_action_max=10)
                grid = boundary_grid([-2.0, np.nan], [WARM_END, BOUNDARY],
                                     max_new_points=pas._boundary_approach_batch(opts))
                self.assertEqual(grid.size, 8)
                active = np.linspace(10.0, WARM_END, 3)
                state = _support_state(active, free_support_bank=grid[:2])
                rebuilds = []
                self.assertTrue(run_boundary_approach(solver, state, opts, active, rebuilds))
                self.assertEqual(rebuilds[0][1].size, active.size + grid.size,
                                 "the free-bank points were not admitted")
                self.assertEqual(int(state.boundary_refine_points), grid.size - 2,
                                 "the two points whose action had already run were charged "
                                 "to the allowance")

    def test_an_exhausted_point_is_refused_the_next_grid(self):
        """The spent allowance stops the approach where it is enforced.

        A closure that asks `_boundary_approach_capacity` and discards the answer passes the
        code-object test and fails this one: the grid is applied, the support grows and the
        counter runs past the cap.
        """
        for solver in SUPPORT_SOLVERS:
            with self.subTest(solver=solver.__name__):
                opts = settings(n_action_increment=5, n_action_max=10)
                cap = pas._BOUNDARY_APPROACH_ROUNDS * pas._boundary_approach_batch(opts)
                active = np.linspace(10.0, WARM_END, 3)
                state = _support_state(active, spent=cap)
                bank_before = np.asarray(state.support_bank, dtype=float).copy()
                rebuilds = []
                self.assertFalse(run_boundary_approach(solver, state, opts, active, rebuilds),
                                 "an exhausted point was granted a further approach")
                self.assertEqual(rebuilds, [], "the support was rebuilt anyway")
                np.testing.assert_array_equal(state.support_bank, bank_before)
                self.assertEqual(int(state.boundary_refine_points), cap,
                                 "the counter moved on a refused approach")

    def test_the_last_point_of_the_allowance_is_still_admitted(self):
        # The refusal has to sit at the cap and not below it: a guard that refused one
        # approach early would pass the test above for the wrong reason.
        for solver in SUPPORT_SOLVERS:
            with self.subTest(solver=solver.__name__):
                opts = settings(n_action_increment=5, n_action_max=10)
                cap = pas._BOUNDARY_APPROACH_ROUNDS * pas._boundary_approach_batch(opts)
                active = np.linspace(10.0, WARM_END, 3)
                state = _support_state(active, spent=cap - 1)
                rebuilds = []
                self.assertTrue(run_boundary_approach(solver, state, opts, active, rebuilds),
                                "the last approach of the allowance was refused")
                self.assertEqual(len(rebuilds), 1)
                self.assertEqual(int(state.boundary_refine_points),
                                 cap - 1 + pas._boundary_approach_batch(opts))

    def test_the_whole_batch_is_installed_when_the_increment_exceeds_the_maximum(self):
        """The capacity reaches the support update, and it is the batch the builder asked for.

        `n_action_increment` and `n_action_max` are independent settings. At an increment of
        200 and a maximum of 10 the builder asks for 200 points, and a capacity taken from
        `n_action_max` caps the support update at 20 points without a cached action, counted
        over the whole support. On this fixture's three-point active grid that admits 17 of
        the 200, blocks 183 and leaves the counter at 17, so the allowance of 400 is spent on
        approaches that never install the grid the builder designed.
        """
        for solver in SUPPORT_SOLVERS:
            with self.subTest(solver=solver.__name__):
                opts = settings(n_action_increment=200, n_action_max=10)
                grid = boundary_grid([-2.0, np.nan], [WARM_END, BOUNDARY],
                                     max_new_points=pas._boundary_approach_batch(opts))
                self.assertEqual(grid.size, 200, "the builder did not ask for the batch")
                active = np.linspace(10.0, WARM_END, 3)
                state = _support_state(active)
                rebuilds = []
                self.assertTrue(run_boundary_approach(solver, state, opts, active, rebuilds))
                self.assertEqual(rebuilds[0][1].size, active.size + grid.size,
                                 "the grid the builder designed arrived truncated")
                self.assertEqual(int(state.boundary_refine_points), grid.size)


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
