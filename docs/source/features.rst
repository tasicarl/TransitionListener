.. _features:

========
Features
========

TransitionListener takes a particle physics model and gives back a gravitational wave
spectrum, with every step in between done for you: the effective potential, the phases, the
bounce, the nucleation history, the hydrodynamics and the signal. You write down the model,
and the package does the rest.

.. figure:: _static/pipeline.png
   :width: 100%

   A run goes from a model file to a signal-to-noise ratio in six steps.

Models
------

A model file holds the parameters of the model and nothing else. You give the tree-level
potential, the field content and the mass spectrum, and the one-loop corrections at zero and
finite temperature, the counterterms and the daisy resummation all come from the base class. Any number of scalar fields
is supported, and eight models ship with the package, among them three dark sector scenarios
and the two Higgs doublet model.

.. figure:: _static/algorithm_potential_vs_temperature.png
   :width: 100%

   A barrier appearing as the universe cools, which is what makes a transition first order.

Transitions at any degree of supercooling
-----------------------------------------

Phases are traced in temperature and the candidate transitions between them are found
automatically. The bounce action is computed along a path that is deformed until it is the
real tunnelling path, and the machinery is built for the hard cases: transitions supercooled
by many orders of magnitude, actions that are not monotonic, vacuum-dominated expansion, and
scales anywhere from the MeV range to the GUT scale.

.. figure:: _static/algorithm_shooting.png
   :width: 100%

   Finding the bounce by overshooting and undershooting.

A self-consistent false vacuum evolution and Hubble expansion
-------------------------------------------------------------

The converted fraction, the expansion rate and the temperature of the plasma inside the
bubbles all depend on each other, and the package solves them together rather than one after
another. It reports the percolation, completion and reheating temperatures, the transition
strength, the mean bubble separation and both definitions of the inverse duration, with the
wall velocity in local thermal equilibrium.

Every run also carries a row of warnings and diagnostics: whether the sound speed survived a
change of the differencing step, whether the two entropy schemes agree, whether the accuracy
targets were met, whether the wall velocity is a choice among branches. Nothing is silently
patched.

The gravitational wave signal and its detectability
---------------------------------------------------

The gravitational wave spectrum is built from the bubble collisions, the sound waves and the
turbulence, with the templates of the LISA Cosmology Working Group and efficiency factors
taken from the wall velocity and the strength. It is then compared against the sensitivities
of LIGO-Virgo-KAGRA, LISA, DECIGO, BBO, the Einstein Telescope, Cosmic Explorer and muAres,
and against the pulsar timing arrays, with likelihoods for the NANOGrav and IPTA data sets.

.. figure:: _static/algorithm_spectrum_sources.png
   :width: 100%

   The three sources, their sum, and what LISA can see.

Scanning
--------

A run is configured by one readable file. Single points, grids, lines, random scans and nested
sampling with UltraNest all use the same format, and the plots come out with the results.

At a glance
-----------

================================  ============================================================
Fields                            any number of scalars, with mixing
Resummation                       Arnold-Espinosa, Parwani, or none
Phase tracing                     adaptive in temperature, with the symmetry of the potential
Bounce                            path deformation in several field directions
Percolation                       two solvers, one adaptive and one on a fixed grid
Expansion history                 self-consistent, with reheating and a computed sound speed
Wall velocity                     local thermal equilibrium
Spectrum                          collisions, sound waves, turbulence
Detectors                         ground, space and pulsar timing
Scans                             point, grid, line, random, nested sampling
================================  ============================================================

Further reading
---------------

:doc:`getting_started` runs your first point. :doc:`usage` is the configuration file.
:doc:`algorithms/index` is what the package actually computes, stage by stage.
