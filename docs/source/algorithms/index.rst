.. _algorithms:

The algorithms
==============

What TransitionListener computes at each stage, which choice the implementation makes where
more than one is defensible, and what it refuses rather than guessing.

The pages follow the order of a run. A model gives an effective potential; the potential gives
phases and, between them, candidate transitions; a transition gives a bounce action at each
temperature; the action gives a nucleation rate and, integrated, a percolation history; the
history gives the transition strength and duration; and those give a spectrum.

Where the behaviour changed in version 2.2, the page says so at the paragraph that changed. A
box marked as a caution means the earlier version did not merely differ but could return a
number that was wrong without saying so.

.. toctree::
   :maxdepth: 1

   the_effective_potential
   the_plasma
   phases_and_transitions
   the_bounce_action
   nucleation_and_percolation
   hydrodynamics
   the_spectrum
