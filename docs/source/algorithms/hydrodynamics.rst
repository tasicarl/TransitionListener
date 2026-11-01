.. _algorithm-hydrodynamics:

Hydrodynamics of the bubble wall
================================

The gravitational wave signal is not produced by the scalar field directly but by the motion it
drives in the plasma. How much of the released energy ends up as bulk motion, and in what form,
is a hydrodynamic question that depends on how fast the wall moves and how strong the
transition is.

The transition strength
-----------------------

The strength is the energy released, measured against the energy already present:

.. math::
   \alpha = \frac{\Delta\theta}{3 w_{\rm sym}} ,

where :math:`\theta = (e - 3p)/4` is the trace anomaly, :math:`\Delta` denotes the difference
between the two phases and :math:`w` is the enthalpy of the symmetric phase. Several strengths
appear in a run and they answer different questions: one drives the sound waves, one is the
pressure available to the wall, one is the pressure at which the wall would stop accelerating.

.. admonition:: They must share a normalisation
   :class: important

   The efficiency with which bubble collisions radiate depends on ratios of these strengths.
   A ratio of pressures must not depend on what they are divided by, as long as all of them
   are divided by the same thing.

.. versionchanged:: 2.2
   They did not share it. The strength that drives the sound waves was normalised to the
   enthalpy of the symmetric phase, while the two that describe the wall were normalised to the
   radiation energy density of the transitioning sector with zero-temperature masses. They are
   now all referred to the same enthalpy, so that the ratios mean what they are supposed to
   mean. This was found and fixed by Nico Benincasa and Kristjan Müürsepp.

The sound speed
---------------

The speed of sound in each phase follows from the effective potential,

.. math::
   c_s^2 = \frac{\partial V/\partial T}{T\, \partial^2 V/\partial T^2} ,

and it enters in several places: the speed at which bubbles expand is the larger of the wall
velocity and the sound speed, the efficiency factors depend on it, and the spectrum divides by
the difference between the wall velocity and it.

A sound speed of one or more is not a speed, so nothing computed from it means anything. The
code refuses such a point rather than substituting a value, and it reports which phase produced
it and which of two known mechanisms is responsible: a loss of significant digits in the
temperature derivatives, or a resummation used outside its range of validity.

.. versionchanged:: 2.2
   Under supercooling the sound speed returned values that depended on the step used for the
   temperature derivatives, values above the speed of light, and at low enough temperature
   nothing at all. The cause lay in the potential rather than in the sound speed: two
   quantities in it are built by subtracting nearly equal numbers, and dividing by a second
   temperature derivative is where the lost digits become visible. The same expressions are
   evaluated, arranged so that they keep their accuracy, and what remains is reported.

The wall velocity
-----------------

In local thermal equilibrium the wall velocity follows from matching the flow across the wall
together with entropy conservation, and the code solves that matching. The result is one of
three kinds of solution: a deflagration, in which the plasma is pushed ahead of a wall slower
than the sound speed; a detonation, in which the wall outruns the fluid; or a hybrid.

Where the matching has no solution the wall runs away, and the code returns a velocity of one.
That is the expected outcome on a strongly supercooled transition, where there is little plasma
to push against, and it is not flagged as a failure. A velocity of zero is a different
statement, meaning the strength is below the threshold at which any solution exists, and it is
flagged, because every efficiency factor is proportional to it and the spectrum would
otherwise come out as a confident prediction of silence.

.. versionchanged:: 2.2
   At percolation temperatures far below the mass scale the thermal part of the potential
   underflows in the broken phase, so its enthalpy is exactly zero and its sound speed is zero
   divided by zero. Two places downstream carried that forward instead of stopping, in both
   cases after the percolation itself had succeeded. They now stop.

.. admonition:: A limitation to be aware of
   :class: important

   The percolation history is computed with a wall velocity of one, whatever velocity is
   configured, because the velocity in local equilibrium is evaluated at the percolation
   temperature and so is not available until the history has been computed. Where the velocity
   comes out as one, which is the common case on a supercooled transition, the two agree. Where
   a velocity below one is configured, the history and the spectrum are built on different
   values.

The efficiency factors
----------------------

What fraction of the released energy goes into each source is given by fits to hydrodynamic
simulations: one for the bulk motion that drives the sound waves, one for the fraction that
remains in the scalar field and radiates when bubbles collide, and one for the turbulence.
These are fits over a range of strength and wall velocity, and outside that range they are
extrapolations rather than results.

In the code
-----------

``hydrodynamics.Hydrodynamics`` holds the matching: ``calc_cs`` for the sound speed,
``sound_speed_is_step_dependent`` for the diagnostic that says whether a value depends on the
step it was taken with, ``calcWallVelocityLTE`` and ``find_vw`` for the wall velocity,
``find_vJ`` for the Jouguet velocity that separates the branches, and ``find_kappa`` for the
efficiency. ``kappa_sw`` and ``calc_kappas`` are the fits.
