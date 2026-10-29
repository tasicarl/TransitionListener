.. _notes-transitions:

``transitions.py``
==================


.. _note-transitions-transitions-find-transition-path-from-tnuc-comment:

``Transitions.findTransitionPathFromTnuc``, implementation comment
------------------------------------------------------------------

Without the test at the top, the walk asks the zero-temperature phase for one more transition, is refused the cycle's closing link, and raises -- discarding the transitions it had already found and reporting the run as too supercooled to nucleate. Tested at the top, on the phase this iteration begins at, because `startPhase` is reassigned in two places below: once by the second-order branch inside the child loop, and once at the end of the iteration. Reading it after either of those compares a phase against itself and reports a valid second-order step as a cycle. Completion is tested before the cycle guard. A chain can reach the zero-temperature phase before it has found any transition, advance from it by a second-order step, and come back to it with transitions in hand. That chain is complete and must stop here; testing the guard first would report it as a cycle and discard it, throwing away a complete set of transitions and reporting the run as too supercooled to nucleate.

