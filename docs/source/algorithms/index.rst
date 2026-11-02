.. _implementation-notes:

Implementation notes
====================

These pages describe what TransitionListener computes at each stage of a run: the quantity,
the equations the module solves, the choice the implementation makes where more than one is
defensible, and the settings that reach it. Each page also lists the cases in which the stage
stops and returns an error code instead of a number, with what each code means, because a
point that cannot be computed to the accuracy asked for is reported as such and not filled in
with a value from a fallback.

A run passes through six stages. The model file defines the effective potential. The minima of
the potential, followed in temperature, are the phases, and a pair of phases that tunnelling
can connect is a candidate transition. For each temperature the critical bubble of such a
transition is computed, and its Euclidean action sets the rate at which bubbles nucleate.
Integrating that rate over the expansion history produces the fraction of space in the true
vacuum, from which the strength and the duration of the transition follow, and those two fix
the gravitational wave spectrum and its signal-to-noise ratio in each detector. The module
:py:mod:`transitionlistener.interface.pipeline` runs the stages in this order, and the pages
below follow them.

.. figure:: ../_static/pipeline.png
   :width: 100%

   A run goes from a model file to an observable in six steps.

The run from end to end
-----------------------

The starting point is a model file: the tree-level scalar potential, the field content, the
counterterms and the field-dependent mass spectrum. Models inherit from
:py:mod:`transitionlistener.generic_potential`, which evaluates the one-loop, daisy-corrected
effective potential and its derivatives from them, so a model file carries parameters rather
than machinery. :doc:`the_effective_potential` describes what is assembled and
:doc:`the_plasma` the degrees of freedom that go with it.

With the potential in hand, :py:mod:`transitionlistener.phases` follows every minimum as the
universe cools and :py:mod:`transitionlistener.transitions` reads off which pairs of them could
be connected by tunnelling. :doc:`phases_and_transitions` describes both.

For each candidate, the tunnelling path through field space is found by
:py:mod:`transitionlistener.pathDeformation` and the three-dimensional Euclidean action along
it by :py:mod:`transitionlistener.tunneling1D`. The bounce is the field configuration of the critical
bubble, the one that is as likely to grow as to shrink, and its three-dimensional Euclidean
action :math:`S_3` is what the nucleation rate depends on exponentially. The page
:doc:`the_bounce_action` describes how the bounce is found, by shooting along one direction
and deforming the path in the others.

The action fixes the rate at which bubbles nucleate, and
:py:mod:`transitionlistener.bubbledynamics` integrates that bubble nucleation rate into a
history of the true-vacuum fraction, feeding the energy released back into the expansion as
it goes.
:doc:`nucleation_and_percolation` describes the integral, its two solvers and the temperatures
that come out of it.

From that history :py:mod:`transitionlistener.transitionObservables` computes the quantities a
gravitational wave prediction needs. :py:mod:`transitionlistener.hydrodynamics` computes the
fraction of the released vacuum energy that ends up as bulk motion of the plasma, for the
three ways a wall can move through it: a deflagration, a hybrid and a detonation.
:doc:`the_observables` and :doc:`hydrodynamics` describe both.

Finally :py:mod:`transitionlistener.gwfopt` builds the spectrum and
:py:mod:`transitionlistener.observability` compares it with what detectors can see, which is
:doc:`the_spectrum`.

.. toctree::
   :maxdepth: 1

   the_effective_potential
   the_plasma
   phases_and_transitions
   the_bounce_action
   nucleation_and_percolation
   hydrodynamics
   the_observables
   the_spectrum
   the_plots
   error_reference
