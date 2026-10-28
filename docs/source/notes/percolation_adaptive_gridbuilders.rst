.. _notes-percolation-adaptive-gridbuilders:

``percolation_adaptive_gridbuilders.py``
========================================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/percolation_adaptive_gridbuilders.py``, which now carry
the summary and point here.

.. _note-percolation-adaptive-gridbuilders-nucleation-is-dead:

``_nucleation_is_dead``
-----------------------

The cold actions of a completed transition do not fail - they *saturate*. Measured on the 2HDM edge line with the full action cache: the action settles at its zero-temperature value near 4357 and ``S_3/T`` runs up to 1.15e+09 by T = 3.8e-06 (internal). Such a point has a perfectly finite action, so it is promoted into the active grid as a legitimate cached temperature, the grid's cold boundary lands there, and the rate-step refinement then fills six orders of magnitude in temperature with support that cannot influence ``P`` or the rate peak.

Guarding on action finiteness cannot catch this, which is why two earlier attempts produced bit-identical runs.

.. _note-percolation-adaptive-gridbuilders-cold-walk-has-left-the-bounce-region:

``_cold_walk_has_left_the_bounce_region``
-----------------------------------------

Below the transition the symmetric phase stops being metastable, so the bounce stops existing and every action there comes back non-finite. Those samples are dropped by the percolation integral and by the profile rebuild, so they tell the controller nothing, yet each one costs a bounce solve. Measured on the 2HDM edge line, 38-57% of all support points in every one of five dumps had no computable action, in a single contiguous run at the cold end: at lambda3 = 5.8109479409, 75 finite-action points from 76.4 down to 25.4 GeV followed by 56 non-finite ones down to 3.8e-06 GeV.

The test is deliberately a *trailing* run rather than a count: a hole in the middle of the grid is a solver hiccup, not the edge of the bounce region. It also requires some finite action to exist, so a grid that has not started yet is never mistaken for one that has run past the edge, and tolerates ``_COLD_WALK_MAX_FAILED_ACTIONS - 1`` isolated failures so that a transition lying just below one bad solve is still found.

.. _note-percolation-adaptive-gridbuilders-acceptance-jump-is-exempted-inert:

``_acceptance_jump_is_exempted_inert``
--------------------------------------

A jump past the rate peak cannot usually be repaired by more support -- the action there is dead -- so the inert veto skips it. An interval selected by the *acceptance* criterion is exempt, because leaving it unrepaired is fatal. Such an exempted repair must not preempt the anchor remesh: at lambda3 = 5.8136778037 the direct repair took the refinement slot the remesh needed and the run ended one support update short at Delta P = 0.0578, while at lambda3 = 5.8177725978 the same repair is the only thing that resolves the profile. The caller uses this to try the remesh first and fall back to the direct repair, so both cases are served.

