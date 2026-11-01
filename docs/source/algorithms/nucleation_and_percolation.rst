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

Placing the support points
--------------------------

Every evaluation of the integrand needs a bounce action, and a bounce action costs far more
than everything else combined. The solver is therefore built around a budget: it may compute
only a few dozen actions, and the whole question is where to spend them. Its internal name is
the dynamic zoom window, which is what the ``_dynamiczoomwindow_`` functions in the source
refer to.

The difficulty is the shape of the integrand. The rate goes as :math:`e^{-S_3/T}` and the
action falls steeply as the universe cools, so :math:`\Gamma/H^4` rises through tens of orders
of magnitude over a narrow range of temperature, is appreciable over a short interval, and
dies again. A grid that steps across that rise without resolving it returns a smooth curve and
a plausible number that is wrong by orders of magnitude, with nothing in the output to show it.

Scouting
~~~~~~~~

The first pass works on the logarithm of the rate in Hubble units,
:math:`\log_{10}(\Gamma/H^4)`, because that is the quantity whose shape matters and it is
well behaved over the whole range. It computes a sparse set of actions, reconstructs an
approximate profile from them, and asks two questions: is the hot end far enough out that
nucleation has not yet begun, and is the cold end far enough down that it can no longer
contribute. The second has a quantitative answer: once the rate has fallen a thousand decades
below its peak, no colder point can change the converted fraction, and the tail below that is
dead.

If either end fails, the window is widened and more actions are spent, up to a ceiling on how
many the scout may use. When the transition is bracketed, the interior peak of the rate is
located and a window is built around it. That window is the starting grid for the next pass.

Refinement
~~~~~~~~~~

The later passes add support points where the profile is not yet resolved. Several strategies
exist because there are several ways for a profile to be unresolved, and each has its own
signature:

**A jump in the converted fraction.** If two neighbouring support points differ by more than
:math:`\Delta P = 0.1`, the rise between them is not resolved and intermediate temperatures
are added. This is the commonest case and the most important: it is the direct statement that
the grid has stepped over the transition.

**The rise of the rate.** Points are added across the interval where
:math:`\log(\Gamma/H^4)` climbs, located from the sampled rate rather than from the
converted fraction, so that the rise is resolved before it has had any visible effect on
:math:`P`.

**The peak of the rate**, and the approach to it, refined where the sampled maximum is a
single point rather than a resolved shape.

**The hot shoulder**, where the converted fraction is still near zero but the rate is already
appreciable, so the integral is accumulating in a region the grid treats as empty.

**The hot head**, the hottest retained support point, which has to sit where nothing has
happened yet. If the profile still rises at its hot end, the window did not start early enough.

**Remeshing from anchors.** Rather than adding points to an existing grid, a fresh batch is
placed at fixed values of :math:`P`, which distributes effort by how much of the transition
each interval carries rather than by temperature.

**Extending the range**, in small batches, when the window itself is too narrow, with the cold
boundary chosen from how much the last extension gained.

Acceptance
~~~~~~~~~~

A profile is accepted only if it passes two tests that the refinement might not have reached:

- No two neighbouring support points differ by more than :math:`\Delta P = 0.2`. A profile
  that still has a jump that large has not resolved the transition, whatever else converged.
- The hot head is not underresolved, in the sense above.

An interval is exempted from the first test where the rate there is so far past its peak that
the interval cannot contribute, which is the same dead-tail criterion the scout uses.

If neither the refinement budget nor the range can fix a failure, the point is refused, and
the message says which resource ran out: the number of actions, the number of iterations, or
the temperature range.

.. versionchanged:: 2.2
   The refinement strategies above, the acceptance tests, and the distinction between running
   out of budget and a transition that genuinely does not percolate are new. Before, the solver
   could exhaust its budget on the couplings nearest the percolation edge and report that in a
   way indistinguishable from a physical verdict.

Settings
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 10 56

   * - Setting
     - Default
     - Meaning
   * - ``n_action_min``
     - 15
     - Support points built before the first refinement sweep.
   * - ``n_action_increment``
     - 5
     - Support points added per refinement iteration.
   * - ``n_action_max``
     - 60
     - Most new support points in one update; a point whose action is cached is free.
   * - ``max_action_temperatures``
     - 100
     - Most action evaluations the scout may spend bracketing the transition.
   * - ``large_delta_p_refine_threshold``
     - 0.1
     - Neighbouring jump in :math:`P` above which extra refinement is triggered.
   * - ``large_delta_p_success_threshold``
     - 0.2
     - Neighbouring jump in :math:`P` above which a profile is refused.
   * - ``percolation_maxit``
     - 20
     - Refinement iterations before the solver gives up.
   * - ``f_perc``
     - 0.28957
     - Converted fraction defining the percolation temperature.
   * - ``max_boundary_ratio``
     - 0.45
     - Most of the temperature range that may sit on a plateau where :math:`P` is zero or one.

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

The entropy used
~~~~~~~~~~~~~~~~

The entropy can be taken in two ways, selected by ``entropy_definition``. ``dof_table``, the
default, uses the counted degrees of freedom; ``eff_potential`` uses :math:`-\partial V/\partial T`
of the effective potential. They are two forms of the same quantity and agree where the
counting and the potential describe the same plasma. The important thing is that one of them is
used throughout: the scale factor and the sound speed must come from the same entropy, or the
expansion history is not consistent with the thermodynamics it is built from.

The temperatures reported
-------------------------

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

Refusals
--------

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

In the code
-----------

``bubbledynamics.calcPercAndEvolve`` is the entry point and ``percIntegral`` the integral
itself. The adaptive solver lives in ``percolation_adaptivestepsize`` for the controllers and
the state, ``percolation_adaptive_gridbuilders`` for turning arrays into the next set of
candidate temperatures, and ``percolation_adaptive_rate`` for locating the rise of the rate. A
fixed-step solver exists alongside it in the modules ending ``_fixedstep``.
