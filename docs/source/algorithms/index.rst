.. _algorithms:

The algorithms
==============

Each page describes one stage of a run: the quantity it computes, the choice the
implementation makes where more than one is defensible, and the cases it refuses rather than
guessing.

The pages follow the order of a run. A model gives an effective potential; the potential gives
phases and, between them, candidate transitions; a transition gives a bounce action at each
temperature; the action gives a nucleation rate and, integrated, a percolation history; the
history gives the transition strength and duration; and those give a spectrum.

Where the behaviour changed in version 2.2, the page says so at the paragraph that changed.

.. toctree::
   :maxdepth: 1

   the_effective_potential
   the_plasma
   phases_and_transitions
   the_bounce_action
   nucleation_and_percolation
   hydrodynamics
   the_spectrum
