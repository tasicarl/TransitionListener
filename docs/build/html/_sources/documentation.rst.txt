Documentation
=============

`Full API reference on one page <api/TransitionListener.html>`_

The main interface is :py:mod:`transitionlistener.interface`. TransitionListener
also provides several other modules for specific tasks, which are documented
below.


Essential modules:
--------------------------------------------------------------------------------
- :py:mod:`transitionlistener.bubbledynamics`: Computation of the transition dynamics,
  including bubble nucleation rate, bubble growth, and phase transition completion.
- :py:mod:`transitionlistener.generic_potential`: Model-independent potential
  definitions and utilities.
- :py:mod:`transitionlistener.gwfopt`: Gravitational wave spectra from
  first-order phase transitions.
- :py:mod:`transitionlistener.hydrodynamics`: Computation of bubble wall velocity
  and fluid profiles for efficiency factor calculations.
- :py:mod:`transitionlistener.interface`: Main interface class for TransitionListener.
- :py:mod:`transitionlistener.observability`: Computation of the observability of
  gravitational wave signals at various detectors.
- :py:mod:`transitionlistener.pathDeformation`: Tunneling pathfinding and bubble
  profile calculations using the path deformation method.
- :py:mod:`transitionlistener.phases`: Definition of the phases object
  containing phase information at different temperatures.
- :py:mod:`transitionlistener.thermodynamics`: Computation of thermodynamic
  quantities like energy density and pressure.
- :py:mod:`transitionlistener.transitionObservables`: Coordination of the computation
  of all derived observables from the phase transition.
- :py:mod:`transitionlistener.transitions`: Characterization and analysis of
  first-order phase transitions given a phases object.
- :py:mod:`transitionlistener.tunneling1D`: Bounce action calculations for single-field potentials.

Internally used modules and utilities:
--------------------------------------------------------------------------------

- :py:mod:`transitionlistener.colors`: Utility functions and definitions of often used colors.
- :py:mod:`transitionlistener.config`: Standard configuration handling for TransitionListener.
- :py:mod:`transitionlistener.constants`: Physical and mathematical constants used throughout TransitionListener.
- :py:mod:`transitionlistener.errors`: Custom error and exception classes for handling various errors.
- :py:mod:`transitionlistener.finiteT`: Numerics of the finite temperature potential calculations.
- :py:mod:`transitionlistener.interface.check`: Installation self-check entry point used by ``tl --check``.
- :py:mod:`transitionlistener.gridplots`: Plotting utilities for grid scans and line scans.
- :py:mod:`transitionlistener.helper_functions`: General helper functions used throughout TransitionListener.
- :py:mod:`transitionlistener.particles`: Particle content definitions and utilities.
- :py:mod:`transitionlistener.plot_settings`: General plot settings and styles for TransitionListener plots.
- :py:mod:`transitionlistener.plots`: Plotting utilities for potentials, tunneling paths,
  bubble profiles, thermodynamic parameters, and gravitational wave spectra.
