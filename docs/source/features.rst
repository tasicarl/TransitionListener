.. _features:

========
Features
========

TransitionListener offers a comprehensive suite of features for simulating and analyzing
cosmological first-order phase transitions and their associated gravitational wave signals.
Key features include:

- Model-independent framework for analyzing first-order phase transitions, including custom multi-scalar potentials, one-loop thermal corrections with daisy resummation, and several built-in benchmark models such as three dark-sector scenarios and the 2HDM.

- Phase tracing based on CosmoTransitions, with adaptive temperature stepping, automated identification of critical temperatures and transitions, tunneling pathfinding via path deformation, bounce-profile computation, and user-adjustable accuracies.

- Bubble nucleation rate calculations and nucleation-temperature determination that remain applicable for extremely supercooled transitions, u-shaped bounce actions, vacuum-dominated transitions, and arbitrary physical scales through the ``internal_scale`` conversion.

- State-of-the-art bubble wall velocity calculations in the local-thermal-equilibrium approximation, following the method of van de Vis et al. (2023), available at https://arxiv.org/pdf/2303.10171.

- Self-consistent computation of the Hubble rate and false vacuum fraction, including reheating from local energy conservation across the bubble wall and a sound-speed-based time-temperature relation for the percolation integrals.

- Computation of the percolation, completion, and reheating temperatures with both the default ``adaptive_step_size`` solver and an optional ``fixed_step_size`` solver for numerically noisy actions, together with ODE-based and diagnostic double-integral implementations.

- Calculation of both the mean bubble separation and the inverse duration from the logarithmic temperature derivative of :math:`S_3/T`; the numerical precision is high enough to validate analytical relations between them at the sub-percent level, and the mean separation is used as the default gravitational wave length scale.

- Configurable precision bundles for tracing and tunnelling, allowing users to switch between the ``default``, ``robust``, ``xtrace``, ``tunneltight``, and ``benchmark`` presets instead of tuning every low-level control by hand.

- Prediction of the gravitational wave spectrum, including bubble-collision, sound-wave, and MHD-turbulence contributions, state-of-the-art simulation fits, and automatic efficiency-factor calculations based on the wall velocity and transition strength.

- Observability forecasts for current and future detectors, including LIGO-Virgo-KAGRA, LISA, DECIGO, BBO, ET, CE, muAres, and pulsar timing arrays.

- Pulsar-timing-array likelihood support based on the Ceffyl code, interfaced through PTArcade, with access to the NANOGrav 12.5 yr, NANOGrav 15 yr, and IPTA datasets.

- Multiple scan modes, including grid scans, random scans, line scans, nested sampling via UltraNest, and flexible YAML-based configuration.

- Comprehensive plotting utilities for potentials, tunneling paths, bubble profiles, thermodynamic parameters, and gravitational wave spectra.

- Modular and extensible code organisation that makes it straightforward to add new models and numerical workflows.
