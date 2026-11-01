.. _algorithm-hydrodynamics:

Hydrodynamics of the bubble wall
================================

The gravitational wave signal is not produced by the scalar field directly but by the motion it
drives in the plasma. How fast the wall moves, how the fluid flows around it, and how much of
the released energy ends up as bulk motion are hydrodynamic questions, and this page is about
the equations that answer them.

The transition strength
-----------------------

The energy released, measured against the energy already present, is

.. math::
   \alpha = \frac{\Delta\theta}{3 w_{\rm sym}},
   \qquad \theta = \frac{e - 3p}{4},

where :math:`\theta` is the trace anomaly, :math:`e` and :math:`p` are the energy density and
the pressure, :math:`w = e + p` is the enthalpy, and :math:`\Delta` is the difference between
the symmetric and the broken phase. Several strengths appear in a run: one evaluated at
nucleation, one at the wall, and the asymptotic ones that fix how much pressure the wall feels.

.. admonition:: They must share a normalisation
   :class: important

   The efficiency with which bubble collisions radiate depends on ratios of these strengths. A
   ratio of pressures must not depend on what they are divided by, as long as all of them are
   divided by the same thing.

.. versionchanged:: 2.2
   They did not share it. The strength that drives the sound waves was normalised to the
   enthalpy of the symmetric phase, while the two describing the wall were normalised to the
   radiation energy density of the transitioning sector with zero-temperature masses. They are
   now all referred to the same enthalpy. Found and fixed by Nico Benincasa and Kristjan
   Müürsepp.

The sound speed
---------------

The speed of sound in each phase follows from the effective potential,

.. math::
   c_s^2 = \frac{\partial V/\partial T}{T\,\partial^2 V/\partial T^2},

evaluated separately in the symmetric and the broken phase. It is not a constant: only for a
bag equation of state does it take the radiation value :math:`1/3`. It enters the matching
below, the speed at which bubbles expand, which is :math:`\max(v_w, c_s)`, and the thickness of
the sound shell that sets the peak of the spectrum.

A value of one or more is not a speed, so nothing computed from it means anything. Such a point
is refused rather than patched, and the message names the phase and which of two known
mechanisms produced it: a loss of significant digits in the temperature derivatives, or a
resummation used outside its range.

.. versionchanged:: 2.2
   Under supercooling the sound speed returned values that depended on the differencing step,
   values above the speed of light, and at low enough temperature nothing at all. The cause lay
   in the potential rather than in the sound speed: two quantities in it are built by
   subtracting nearly equal numbers, and dividing by a second temperature derivative is where
   the lost digits become visible.

The wall velocity in local equilibrium
--------------------------------------

The wall is a surface across which the plasma flows. Conservation of the energy-momentum
tensor across it gives the junction conditions, which relate the fluid velocity and enthalpy
just ahead of the wall, :math:`v_+` and :math:`w_+`, to those just behind it, :math:`v_-` and
:math:`w_-`, both in the wall's rest frame. Eliminating the enthalpies leaves one relation
between the velocities and the strength at the wall,

.. math::
   v_+ = \frac{1}{2}\,\frac{c_b^2 + v_-^2 \;\pm\; \sqrt{\mathcal{D}}}
                           {v_-\left(1 + 3 c_b^2 \alpha\right)},
   \qquad
   \mathcal{D} = v_-^4 - 2c_b^2 v_-^2 (1-6\alpha)
                 + c_b^4\left[1 - 12 v_-^2 \alpha (1-3\alpha)\right],

with :math:`c_b` the sound speed behind the wall. The two signs are the two branches,
deflagration and detonation, separated by the Jouguet velocity.

Away from the wall the fluid obeys the relativistic hydrodynamic equations for a self-similar
flow, which depend only on :math:`\xi = r/t`. Writing
:math:`\mu(\xi,v) = (\xi - v)/(1-\xi v)` for the Lorentz-transformed velocity, they reduce to
two ordinary differential equations along the velocity,

.. math::
   \frac{d\xi}{dv} = \frac{\xi\,(1-v\xi)}{2v\,(1-v^2)}
                     \left(\frac{\mu^2}{c_s^2} - 1\right),
   \qquad
   \frac{dw}{dv} = w\left(1 + \frac{1}{c_s^2}\right)\frac{\mu}{1-v^2},

which the code integrates outward from the wall. The first changes sign where
:math:`\mu^2 = c_s^2`, and that is the sound shell at which the profile ends.

Closing the problem
~~~~~~~~~~~~~~~~~~~

The junction conditions alone do not fix the wall velocity: they hold for any :math:`v_w` and
determine only how the fluid responds to it. What fixes it, in local thermal equilibrium, is
that entropy is conserved across the wall. The solver turns that into a shooting problem in
:math:`v_w`:

1. For a trial :math:`v_w`, solve for the strength at the wall consistent with the strength at
   nucleation.
2. Take :math:`v_- = \min(c_b, v_w)` and obtain :math:`v_+` from the junction condition above.
3. Integrate the fluid equations outward from the wall to the shock front.
4. At the front the fluid must satisfy the junction condition of a shock in a medium that does
   not change phase. The residual

   .. math::
      \frac{v_+^{\rm sh}}{v_-^{\rm sh}}
      - \frac{(\mu-1)\,w_-^{\rm sh} + 1}{(\mu-1) + w_-^{\rm sh}},
      \qquad \mu = 1 + \frac{1}{c_s^2},

   is what the shooting drives to zero.

A root of that residual is the wall velocity. The implementation follows [vandeVis2023]_.

.. figure:: ../_static/algorithm_fluid_profiles.png
   :width: 100%
   :align: center

   The flow the equations above produce, integrated with the module's own ``dfdv`` and its
   own shock condition. A wall slower than the sound speed drives the plasma ahead of it and
   the disturbance ends at a shock front, which is supersonic and so sits outside the sound
   speed, while the fluid inside the bubble stays at rest. A wall faster than the Jouguet
   velocity outruns any signal, so the plasma ahead is undisturbed and a rarefaction behind the
   wall carries the velocity back to zero exactly at the speed of sound. The shooting above
   drives the residual at the shock front to zero. The two curves are two different
   transitions, as they have to be: the junction condition puts the deflagration at
   :math:`\alpha = 0.18` and the detonation at :math:`\alpha = 0.087`, each with a root on its
   own branch and none on the other, and :math:`v_{\rm J} = 0.77` at the second of those.

Runaway, and no solution
~~~~~~~~~~~~~~~~~~~~~~~~

The residual has no root when the strength at nucleation exceeds the largest value admitting a
subsonic solution. The plasma then cannot hold the wall back and it runs away, and the code
returns :math:`v_w = 1`. That is the expected outcome on a strongly supercooled transition,
where there is little plasma to push against, and it is not flagged as a failure.

A strength below the threshold at which any solution exists is a different statement and
returns zero, which is flagged, because every efficiency factor is proportional to the wall
velocity and the spectrum would otherwise come out as a confident prediction of silence.

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
   comes out as one, which is the common case on a supercooled transition, the two agree.

The efficiency factors
----------------------

What fraction of the released energy reaches each source is given by fits to hydrodynamic
simulations [Espinosa2010]_. The sound-wave efficiency is interpolated between four limiting
regimes, chosen by the wall velocity relative to the sound speed and the Jouguet velocity, each
with its own closed form in :math:`\alpha`. The bubble-collision efficiency [EllisLewicki2019]_
depends on the ratios of the strengths discussed above, which is why they must share a
normalisation.

These are fits over a range of strength and wall velocity. Outside that range they are
extrapolations rather than results.

In the code
-----------

``hydrodynamics.Hydrodynamics`` holds the problem. ``calc_cs`` is the sound speed and
``sound_speed_is_step_dependent`` the diagnostic that says whether a value depends on the step
it was taken with. ``get_vp`` is the junction condition, ``dfdv`` the fluid equations,
``integrate_plasma`` their integration, ``shooting`` the residual, ``find_vJ`` the Jouguet
velocity, ``max_al`` the largest strength admitting a solution, and ``find_vw`` the root.
``kappa_sw`` and ``calc_kappas`` are the efficiency fits.

References
----------

.. [Espinosa2010] J. R. Espinosa, T. Konstandin, J. M. No and G. Servant, *Energy budget of
   cosmological first-order phase transitions*, JCAP **06** (2010) 028,
   `arXiv:1004.4187 <https://arxiv.org/abs/1004.4187>`_.

.. [vandeVis2023] J. van de Vis et al., *The bubble wall velocity in local thermal
   equilibrium*, `arXiv:2303.10171 <https://arxiv.org/abs/2303.10171>`_.

.. [EllisLewicki2019] J. Ellis, M. Lewicki, J. M. No and V. Vaskonen, *Gravitational wave
   energy budget in strongly supercooled phase transitions*,
   `arXiv:1903.09642 <https://arxiv.org/abs/1903.09642>`_.
