"""Support points far below the hot end of the grid must be able to enter it.

"Is this temperature already on the grid?" used a tolerance proportional to the hottest
temperature, so everything six decades below it counted as a duplicate of the cold end.
At g = 0.45, y = 0.3, mu0 = 1 GeV, step 2 stopped at 6.6e-7 GeV with the next point at
2.4e-13 GeV and step 3 then rejected the run on an unresolved jump in P(T), with 120
actions unspent, although the transition percolates at 3e-8 GeV.
"""
import numpy as np

from transitionlistener.bubbledynamics import _temperature_grid
from transitionlistener.percolation_adaptive_gridbuilders import _build_dynamiczoomwindow_jump_grid
from transitionlistener.percolation_adaptive_rate import _temperature_present
from transitionlistener.percolation_adaptivestepsize import (
    _apply_dynamiczoomwindow_support_update,
    _controller_added_support_points,
)


class Settings:
    n_action_max = 60


HOT, COLD = 100.0, 1e-10


def test_a_cold_candidate_is_not_a_duplicate_of_the_cold_end():
    grid = np.array([HOT, 1.0, COLD])
    for candidate in (1e-5, 1e-7, 1e-9):
        assert not _temperature_present(grid, candidate)


def test_a_real_duplicate_is_still_found_at_every_scale():
    for value in (HOT, 1.0, 1e-5, COLD):
        grid = np.array([HOT, 1.0, 1e-5, COLD])
        assert _temperature_present(grid, value * (1 + 1e-8))
        assert not _temperature_present(grid, value * (1 + 1e-4))


def test_the_grid_keeps_distinct_cold_temperatures():
    assert _temperature_grid([1e-11, 2e-11, 3e-11]).size == 3
    assert _temperature_grid([1e-11, 1e-11 * (1 + 1e-12)]).size == 1


def test_the_jump_ladder_enters_the_active_grid():
    active = np.array([HOT, 1.0, 2.7e-4, COLD])
    ladder = _build_dynamiczoomwindow_jump_grid(2.7e-4, COLD, max_new_points=15)
    new_active, _, dropped = _apply_dynamiczoomwindow_support_update(active, ladder, active, active, Settings())
    assert dropped.size == 0
    assert new_active.size == active.size + ladder.size


def test_cold_points_with_no_action_count_against_the_budget():
    assert _controller_added_support_points(np.array([HOT, 1e-9, 1e-11]), np.array([HOT])) == 2
