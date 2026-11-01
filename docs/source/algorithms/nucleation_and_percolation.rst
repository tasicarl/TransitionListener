.. _algorithm-percolation:

Nucleation, percolation and the expansion history
=================================================

A transition does not happen at a temperature; it happens over a range of them. Bubbles appear
at a rate that rises steeply as the universe cools, they grow, and at some point enough of
space has converted that the transition is under way. This page describes how that history is
computed and which temperature is reported as which.

The fraction of space that has converted
----------------------------------------

The probability that a given point is still in the false vacuum at temperature :math:`T` is
:math:`e^{-I(T)}`, where :math:`I` counts the bubbles whose growth has reached that point:

.. math::
   I(T) = \int_{T}^{T_{\rm start}} \frac{dT'}{T'}\,
          \frac{\Gamma(T')}{H(T')^4}\,
          \frac{4\pi}{3}\, v_w^3 \left[\int_{T}^{T'} \frac{dT''}{H(T'')\,T''}\right]^3 ,

with :math:`\Gamma` the nucleation rate from :doc:`the_bounce_action`, :math:`H` the expansion
rate and :math:`v_w` the speed at which a bubble wall advances. The inner integral is the
distance a bubble nucleated at :math:`T'` has grown by the time the universe has cooled to
:math:`T`, and the cube of it is its volume. The converted fraction is

.. math::
   P(T) = 1 - e^{-I(T)} .

The percolation temperature is where :math:`P` reaches a fixed fraction, conventionally about
a third, at which the bubbles form a connected network. The nucleation temperature, the more
familiar quantity, is where roughly one bubble exists per Hubble volume. Neither is a property
of the potential alone: both depend on the expansion rate, which in turn depends on how much of
the universe has already converted.

Three passes
------------

That circularity is handled by computing the history three times, each with a better expansion
rate.

**The first pass** locates the temperature range in which anything happens at all, so that the
later passes do not spend bounce actions far away from it. Actions are expensive and this is
what makes the cost bearable.

**The second pass** computes :math:`P(T)` using the expansion rate of a universe that has not
yet converted. This neglects the energy the transition releases.

**The third pass** feeds the converted fraction back into the expansion rate and into the
temperature of the phase the bubbles grow into, which is heated by the energy released. It is
iterated to consistency, and it is the one that gives the reheating temperature and decides
whether the transition completes.

Choosing where to evaluate the action
-------------------------------------

Every evaluation of the integrand needs a bounce action, and a bounce action costs far more
than everything else combined. The whole design question is therefore which temperatures to
spend them on, and the solver chooses them adaptively.

The difficulty is that the rate rises through many orders of magnitude over a narrow range of
temperature. A grid that steps across that rise without resolving it gets the integral wrong
without anything looking wrong: the result is a smooth curve and a plausible number. So the
solver locates the rise, places points inside it, and refines until the integral stops moving.

.. versionchanged:: 2.2
   The support points are chosen against the rise of the rate, the integral is evaluated on a
   sub-grid of them, and the point at which the solver gives up is reported as a refusal rather
   than as a result. On the couplings nearest the edge where a transition stops percolating,
   where every point had previously failed with a numerical complaint, points now resolve.

.. admonition:: What this got wrong before 2.2
   :class: caution

   Near the end of a strongly supercooled transition the solver ran out of its budget of bounce
   actions and reported that as a physical verdict. A point that it could not resolve and a
   point whose transition genuinely never percolates came out of a scan looking the same, so a
   boundary in parameter space could be an artefact of the solver rather than a statement about
   the physics.

The expansion history
---------------------

The scale factor is needed in two places: in the growth of a bubble, and in the dilution of the
energy density. It follows from entropy conservation, :math:`a^3 s = \text{constant}`, where
:math:`s` is the entropy density of the phase being traced.

.. versionchanged:: 2.2
   The scale factor comes from entropy conservation rather than from integrating
   :math:`d\ln a/dT = -1/(3 c_s^2 T)` across a grid that can span ten or more decades of
   temperature. The entropy and the sound speed used anywhere in the history are now the same
   one, selected by a single setting, rather than one coming from the counted degrees of
   freedom and the other from the effective potential.

Temperatures the run reports
----------------------------

``Tcrit``
   Where the two minima are degenerate. A property of the potential alone.

``Tnuc``
   Where about one bubble has nucleated per Hubble volume.

``Tperc``
   Where the converted fraction reaches the percolation threshold.

``Tf``
   Where the transition completes, in the sense that the false-vacuum fraction has fallen to a
   small final value.

``Treh``
   The temperature after the released energy has been shared out.

The inverse duration
--------------------

How fast the transition proceeds is quoted as :math:`\beta/H`, and there are two of them,
which answer different questions.

The first is local: the logarithmic derivative of the rate at the percolation temperature,
:math:`\beta/H = T\,d(S_3/T)/dT`. The second is integral: it is read off the mean separation
between bubbles, :math:`R_*`, which follows from the whole nucleation history rather than from
one temperature.

The two agree when the transition is fast, because then the rate is well described by its
slope at one point. They do not agree when it is slow, and the difference is physical rather
than numerical. A feature in one of them is not evidence of anything until it appears in the
other.

.. versionchanged:: 2.2
   The mean bubble separation is obtained from the expansion history rather than from a
   relation that assumes radiation domination, and the local derivative is obtained from a fit
   to the action rather than from a two-point difference.

What it refuses
---------------

- The converted fraction never reaches the percolation threshold, so there is no percolation
  temperature on the range searched.
- The transition percolates and then stalls, never reaching completion. This is reported
  separately from the previous case, because the physics differs: a transition that stalls
  leaves a mixture of phases.
- The estimate that brackets the percolation temperature does not converge.
- The sampled action implies a rate that is not smooth, which means the action itself is not
  resolved.
- The run exceeded its time limit. This is a statement about the clock and not about the
  physics, and it is reported as its own failure rather than as a verdict.

Where it is in the code
-----------------------

``bubbledynamics.calcPercAndEvolve`` is the entry point and ``percIntegral`` the integral
itself. The adaptive solver lives in ``percolation_adaptivestepsize`` for the controllers and
the state, ``percolation_adaptive_gridbuilders`` for turning arrays into the next set of
candidate temperatures, and ``percolation_adaptive_rate`` for locating the rise of the rate. A
fixed-step solver exists alongside it in the modules ending ``_fixedstep``.
