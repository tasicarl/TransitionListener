.. _algorithm-percolation:

Nucleation, percolation and the expansion history
=================================================

A transition does not happen at a temperature, it happens over a range of them. Bubbles appear
at a rate that rises steeply as the universe cools, they grow, and at some point enough of
space has converted that the transition is under way. This page describes how that history is
computed and which temperature is reported as which.

The code is in
:py:mod:`transitionlistener.bubbledynamics` and in the modules whose names begin with
``percolation_``.

The fraction of space that has converted
----------------------------------------

The probability that a given point is still in the false vacuum at temperature :math:`T` is
:math:`\mathrm{e}^{-I(T)}`, where :math:`I` counts the bubbles whose growth has reached that point:

.. math::
   I(T) = \int_{T}^{T_\mathrm{start}} \frac{\mathrm{d}T'}{T'}\,
          \frac{\Gamma(T')}{H(T')^4}\,
          \frac{4\pi}{3}\, v_\mathrm{w}^3 \left[\int_{T}^{T'} \frac{\mathrm{d}T''}{H(T'')\,T''}\right]^3 ,

with :math:`\Gamma` the nucleation rate from :doc:`the_bounce_action`, :math:`H` the expansion
rate and :math:`v_\mathrm{w}` the speed at which a bubble wall advances. The upper limit
:math:`T_\mathrm{start}` can be any temperature above which no bubble nucleates, such as the
temperature at which the two phases become degenerate. The code starts the integral at the
hottest support point it keeps, which the scout described below places where :math:`P` is
still below :math:`f_\mathrm{start} = 10^{-3}`. The inner integral is the
distance a bubble nucleated at :math:`T'` has grown by the time the universe has cooled to
:math:`T`, and the cube of it is its volume. The true-vacuum fraction is

.. math::
   P(T) = 1 - \mathrm{e}^{-I(T)} .

Written as it stands, the inner integral has to be redone for every value of the outer one, so
one evaluation of :math:`I` costs the square of the number of support points and a sweep over
the grid costs the cube. The same quantity is the end of a chain of four first-order equations,

.. math::
   \frac{\mathrm{d}J_0}{\mathrm{d}T} = -\frac{\Gamma}{T^4 H}, \qquad
   \frac{\mathrm{d}J_1}{\mathrm{d}T} = -\frac{J_0}{H}, \qquad
   \frac{\mathrm{d}J_2}{\mathrm{d}T} = -\frac{2 J_1}{H}, \qquad
   \frac{\mathrm{d}J_3}{\mathrm{d}T} = -\frac{3 J_2}{H},

each starting at zero at the hot end, with :math:`I = (4\pi/3)\,v_\mathrm{w}^3\,J_3`. One sweep from the
hot end to the cold one produces :math:`I` at every support point, at a cost linear in their
number, which is what makes a support of several hundred temperatures affordable. Where the
sound speed is not that of radiation the same chain carries :math:`a(T)` and
:math:`3 c_\mathrm{s}^2` explicitly, and in the radiation limit they cancel.

The percolation temperature is where :math:`P` reaches the fixed fraction
:math:`f_\mathrm{perc} = 0.28957` at which the bubbles form a connected network
[Guth:1981uk]_. The nucleation temperature, the more familiar quantity, is where one
bubble has nucleated per Hubble volume [Enqvist:1991xw]_ [Turner:1992tz]_. Neither is a property
of the potential alone: both depend on the expansion rate, which in turn depends on how much of
the universe has already converted.

Three passes
------------

That circularity is handled by computing the history three times, each with a better expansion
rate.

The first pass locates the temperature range in which anything happens at all, so that the
later passes do not spend bounce actions far away from it. Actions are expensive and this is
what makes the cost bearable.

The second pass computes :math:`P(T)` using the expansion rate of a universe that has not
yet converted. This neglects the energy the transition releases.

The third pass feeds the true-vacuum fraction back into the expansion rate and into the
temperature of the phase the bubbles grow into, which is heated by the energy released. It is
iterated to consistency, and it is the one that sets the reheating temperature and decides
whether the transition completes.

Two backends
------------

The history above can be computed by either of two solvers, chosen with
``percolation_algorithm_mode``. Their main difference is how the temperatures at which an
action is evaluated are chosen, which is the whole cost of a run and most of its accuracy.
They are separate implementations rather than one solver with two samplers, and the
paragraph after the table says where that shows.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``fixed_step_size``
     - ``adaptive_step_size`` (default)
   * - Support
     - A fixed number of temperatures, evenly spaced over the current range.
     - A budget of temperatures, placed where the profile is not yet resolved.
   * - When it refines
     - By moving or rescaling the whole grid and recomputing it.
     - By adding points inside the intervals that need them, keeping what is computed.
   * - Cost
     - Known in advance: ``n_action`` actions per range it tries.
     - Bounded by the budget. A temperature whose action is cached is free.
   * - Strength
     - Simple, predictable, and adequate when the transition is not supercooled.
     - Resolves a rise that occupies a small fraction of the range.
   * - Weakness
     - A rise narrower than the spacing is stepped over, and widening the range makes that
       worse.
     - More machinery, and its refinement decisions are harder to follow.

Both build on the same physics: the rate, the integral over it, the energy densities and the
expansion history follow the same definitions. They do not share the code that evaluates
them. The percolation integral, the strengths, the inverse duration, the mean bubble
separation and the Hubble rate each exist twice, once in
:py:mod:`transitionlistener.bubbledynamics` for the adaptive backend and once in
:py:mod:`transitionlistener.bubbledynamics_fixedstep` for the fixed-step one, so the two can and
sometimes do differ beyond their sampling. The transition strength is the case to know: both
report the pseudo-trace value under ``alpha``, and the fixed-step backend writes that same
value to ``alpha_theta`` as well, where the adaptive one puts the bag-model strength, as
:doc:`hydrodynamics` sets out.

The fixed-step backend
----------------------

It lays ``n_action`` temperatures evenly across a trial range and computes the profile on them.
If the result is unusable it changes the range rather than the sampling: it moves the window
when the transition sits outside it, narrows it around a jump in the true-vacuum fraction when
one is found, and tries again with the same number of points. The grid is always uniform and
always the same size, so the cost of each attempt is known before it starts.

This is adequate when the rise of the rate occupies a decent fraction of the temperature range,
which is the ordinary case. It fails when the rise is narrow compared with the range, because
then a uniform grid steps over it: the profile comes back as a jump from zero converted to
fully converted between two neighbouring points, and the percolation temperature is read off a
curve that was never resolved.

The adaptive backend
--------------------

Its internal name is the dynamic zoom window, which is what the ``_dynamiczoomwindow_``
functions in the source refer to. It treats the number of actions as a budget and spends it
where the profile needs it.

.. figure:: ../_static/algorithm_rate_and_support.png
   :width: 100%

   The rate in Hubble units is appreciable over a short interval and negligible elsewhere. A
   grid spread evenly over the temperature range steps across the rise and sees the true-vacuum
   fraction jump from zero to one with nothing in between.

Scouting
~~~~~~~~

The first pass works on :math:`\log_{10}(\Gamma/H^4)` rather than on the true-vacuum fraction,
because that is the quantity whose shape matters and it is well behaved over the whole range.
It computes a sparse set of actions, reconstructs an approximate profile, and asks two
questions: whether the hot end is far enough out that nucleation has not begun, and whether the
cold end is far enough down that it can no longer contribute. The second has a quantitative
answer: once the rate has fallen a thousand decades below its peak no colder point can change
the true-vacuum fraction, and the tail below that is dead.

If either end fails the window is widened and more actions are spent, up to a ceiling. When the
transition is bracketed the interior peak of the rate is located and a window is built around
it, and that window is the starting grid for the next pass.

Refinement
~~~~~~~~~~

Several strategies exist because there are several ways for a profile to be unresolved, and
each has its own signature:

- A jump in the true-vacuum fraction is the most common signature. Two neighbouring support
  points differing by more than :math:`\Delta P = 0.1` mean the rise between them is
  unresolved, and intermediate temperatures are added. Nothing else says as directly that the
  grid has stepped over the transition.
- The rise of the rate gets points across any interval where :math:`\log_{10}(\Gamma/H^4)`
  changes by more than ``percolation_max_log10_rate_step`` (12 decades by default) between
  neighbouring support points. The interval is located from the sampled rate rather than from
  the true-vacuum fraction, so that the rise is resolved before it has had any visible effect
  on :math:`P`.
- The peak of the rate is refined where the sampled maximum is a single point rather than a
  resolved shape.
- The hot shoulder is the region where the true-vacuum fraction is still near zero but the rate
  is already appreciable, so the integral is accumulating where the grid treats the transition
  as not yet started.
- The hot head is the hottest retained support point, which has to sit where nothing has
  happened yet. If the profile still rises there, the window did not start early enough.
- Remeshing from anchors places a fresh batch at fixed values of :math:`P` rather than adding
  to the existing grid, which distributes effort by how much of the transition each interval
  carries.
- Extending the range adds small batches beyond the cold end, with the new boundary chosen
  from how much the last extension gained.

.. versionchanged:: 2.2
   The last interval before the coldest end of a traced phase pair is sampled as well. The rate
   is still rising as that boundary is approached, so the support points that decide the
   integral lie inside it, and it was never refined. The refinement has an allowance of its own.

Acceptance
~~~~~~~~~~

A profile is accepted only if no two neighbouring support points differ by more than
:math:`\Delta P = 0.2`, and the hot head is resolved. An interval is exempt from the first
test where it cannot change :math:`P`: the rate has a resolved peak, the interval lies on a
monotonic flank of that peak, :math:`\Gamma/H^4` is below :math:`10^{-10}` at both of its
ends, and an upper bound on what the interval adds to :math:`I` is below :math:`10^{-4}`.

If neither the refinement budget nor the range can fix a failure, the point is refused, and the
message says which resource ran out: the number of actions, the number of iterations, or the
temperature range.

.. versionchanged:: 2.2
   The refinement strategies above, the acceptance tests, and the distinction between running
   out of budget and a transition that genuinely does not percolate are new. Before, the solver
   could exhaust its budget on the couplings nearest the percolation edge and report that in a
   way indistinguishable from a physical verdict.

Settings
~~~~~~~~

The keys below are set per run in the input dictionary. Each sets the attribute of
``PercolationConf`` with the same name without the ``percolation_`` prefix. For
``percolation_maxit`` that attribute is ``maxit``. The last three rows have no input key and
are attributes of ``PercolationConf`` only.

.. list-table::
   :header-rows: 1
   :widths: 40 10 50

   * - Setting
     - Default
     - Meaning
   * - ``percolation_algorithm_mode``
     - ``adaptive_step_size``
     - Which backend computes the history.
   * - ``percolation_n_action``
     - 30
     - Support points of the fixed-step backend.
   * - ``percolation_n_action_min``
     - 15
     - Support points built before the first adaptive refinement sweep.
   * - ``percolation_n_action_increment``
     - 5
     - Support points added per adaptive refinement iteration.
   * - ``percolation_n_action_max``
     - 60
     - Most new support points in one update. A cached action is free.
   * - ``percolation_max_action_temperatures``
     - 100
     - Most actions the scout may spend bracketing the transition.
   * - ``percolation_maxit``
     - 20
     - Support updates per pass, in the second and the third pass each. At the limit the
       solver keeps the best grid it has and refuses the point if that grid fails the
       acceptance tests.
   * - ``percolation_max_log10_rate_step``
     - 12
     - Largest change of :math:`\log_{10}(\Gamma/H^4)` between neighbouring support points
       before the interval is refined. Zero switches the test off.
   * - ``percolation_f_perc``
     - 0.28957
     - Converted fraction defining the percolation temperature.
   * - ``large_delta_p_refine_threshold``
     - 0.1
     - Neighbouring jump in :math:`P` above which refinement is triggered.
   * - ``large_delta_p_success_threshold``
     - 0.2
     - Neighbouring jump in :math:`P` above which a profile is refused.
   * - ``max_boundary_ratio``
     - 0.45
     - Fixed-step backend only. Most of the range that may sit on a plateau where :math:`P`
       is zero or one.

The expansion history
---------------------

The scale factor is needed in two places: in the growth of a bubble, and in the dilution of the
energy density. It follows from entropy conservation, :math:`a^3 s = \mathrm{constant}`, where
:math:`s` is the entropy density of the phase being traced.

.. versionchanged:: 2.2
   The scale factor comes from entropy conservation rather than from integrating
   :math:`\mathrm{d}\ln a/\mathrm{d}T = -1/(3 c_\mathrm{s}^2 T)` across a grid that can span ten or more decades of
   temperature. The entropy and the sound speed used anywhere in the history are now the same
   one, selected by a single setting, rather than one coming from the counted degrees of
   freedom and the other from the effective potential.

The entropy used
~~~~~~~~~~~~~~~~

The entropy can be taken in two ways, selected by ``percolation_entropy_definition``.
``dof_table``, the default, uses the counted degrees of freedom, and ``eff_potential`` uses
:math:`-\partial V/\partial T` of the effective potential. They are two forms of the same quantity and agree where the
counting and the potential describe the same plasma. The important thing is that one of them is
used throughout: the scale factor and the sound speed must come from the same entropy, or the
expansion history is not consistent with the thermodynamics it is built from.

The setting applies to the adaptive backend. The fixed-step backend has its own
time-temperature relation, which takes the entropy and the sound speed from the potential.
It accepts the default ``dof_table`` without using it and refuses ``eff_potential``.

The time-temperature relation itself is selected by ``percolation_time_temperature_mode``:
``sound_speed``, the default, uses :math:`\mathrm{d}T/\mathrm{d}t = -3 c_\mathrm{s}^2 H T`
with the sound speed of this entropy, and ``bag`` uses :math:`\mathrm{d}T/\mathrm{d}t = -HT`.

The temperatures reported
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Key
     - What it is
   * - ``Tcrit``
     - Where the two minima are degenerate. A property of the potential alone.
   * - ``Tnuc``
     - Where about one bubble has nucleated per Hubble volume.
   * - ``Tperc``
     - Where the true-vacuum fraction reaches the percolation threshold.
   * - ``Tf``
     - Where the transition completes, in the sense that the false-vacuum fraction has fallen
       to a small final value.
   * - ``Treh``
     - The temperature after the released energy has been shared out.

Completion
----------

Percolating is not the same as finishing. The false vacuum occupies a physical volume
:math:`\mathcal{V} \propto a^3 \mathrm{e}^{-I}`, which has two competing factors: the expansion inflates
it and the growth of the bubbles eats it. The transition finishes only where the second wins,

.. math::
   \frac{1}{\mathcal{V}}\frac{\mathrm{d}\mathcal{V}}{\mathrm{d}t}
   = 3H\left(1 + c_\mathrm{s}^2\, T\,\frac{\mathrm{d}I}{\mathrm{d}T}\right) < 0 ,

which is the criterion of [Turner:1992tz]_ in the form of [Ellis:2018mja]_, written for the
time-temperature relation :math:`\mathrm{d}T/\mathrm{d}t = -3 c_\mathrm{s}^2 H T` rather than the radiation one. The
code evaluates it at the percolation temperature, and where it fails there the run sets
``WARNING:false_vacuum_not_shrinking`` and keeps its results. A transition whose true-vacuum
fraction stays below :math:`f_\mathrm{final} = 0.99` down to the coldest temperature searched
is refused as stalled.

The mean separation between bubbles
-----------------------------------

The number density of bubbles that have nucleated by temperature :math:`T`, each diluted by the
expansion since it appeared, is

.. math::
   n_\mathrm{b}(T) = \int_T^{T_\mathrm{start}} \frac{\mathrm{d}T'}{T'}\,\frac{\Gamma(T')}{H(T')}\,
            \mathrm{e}^{-I(T')} \left[\frac{a(T')}{a(T)}\right]^3 ,

and the mean separation is :math:`R_* = n_\mathrm{b}^{-1/3}`. The factor :math:`\mathrm{e}^{-I}` is there because
a bubble can only nucleate in space that is still in the false vacuum. :math:`R_*` is what sets
the frequency at which the spectrum peaks, and it is the quantity the second of the two inverse
durations below is read from.

The reheating temperature
-------------------------

The phase the bubbles grow into is heated by the energy the transition releases, so it is not at
the temperature of the phase they nucleate in. Its temperature follows from an entropy-conserving
step plus an energy correction for the fraction converted during that step, which is a
discretisation of

.. math::
   \frac{\mathrm{d}T_\mathrm{b}}{\mathrm{d}T} = \frac{3\,S(T_\mathrm{b})}{T\,S'(T_\mathrm{b})}
   + \frac{P'(T)}{P(T)}\,\frac{e_\mathrm{s}(T) - E(T_\mathrm{b})}{E'(T_\mathrm{b})} ,

Here :math:`T_\mathrm{b}` is the temperature of the broken phase, the plasma inside the
bubbles, against the temperature :math:`T` of the symmetric phase outside them. :math:`S` and
:math:`E` are the entropy and the energy density of the broken phase and
:math:`e_\mathrm{s}` the energy density of the symmetric one. The first term is the cooling
that the expansion alone would produce. The second is the heating by the energy the
transition releases.

The integration starts where the true-vacuum fraction first reaches :math:`10^{-6}`, with
the instantaneous-reheating condition as its initial value. The reheating temperature
:math:`T_\mathrm{reh}` is the value :math:`T_\mathrm{b}` has reached when the symmetric phase
is at the percolation temperature. It is the temperature of the plasma inside the bubbles
once the released energy is spread through it, and it is the one that enters the redshift of
the spectrum, in place of the percolation temperature of the phase outside. For a strong
transition the reheating and the percolation temperature differ by a factor of
:math:`(1+\alpha)^{1/4}` or more, which is why the package integrates the equation above
rather than using that factor. Where the integration fails, the code reads
:math:`T_\mathrm{b}` off the tabulated trajectory, and then solves the instantaneous-reheating
condition at the percolation temperature. A Standard Model bath that is decoupled from the
transitioning sector is not reheated, and its reheating temperature is reported as the
percolation temperature.

.. versionchanged:: 2.2
   The equation is integrated with an error-controlled integrator rather than read off the
   tabulated trajectory. Read off the table, the reheating temperature depends on where the
   support points happen to sit, which shows up as point-to-point scatter along scans that are
   otherwise smooth.

The inverse duration
--------------------

How fast the transition proceeds is quoted as an inverse duration in units of the Hubble
rate, and there are two of them, which answer different questions. They are written
:math:`(\beta/H)_{S_3}` and :math:`(\beta/H)_\mathrm{RH}` here and everywhere else in this
documentation, because the symbol without a subscript does not say which of the two is
meant.

The local one is the logarithmic derivative of the rate at the percolation temperature,
:math:`(\beta/H)_{S_3} = 3 c_\mathrm{s}^2\,T\,\mathrm{d}(S_3/T)/\mathrm{d}T`, with
:math:`c_\mathrm{s}` the sound speed of the symmetric phase from the entropy of the expansion
history. The factor :math:`3 c_\mathrm{s}^2` converts the temperature derivative into a time
derivative and is one in the bag limit. The integral one is read off the
mean separation between bubbles, :math:`R_*`, which follows from the whole nucleation history
rather than from one temperature. Its subscript is the one the output key ``betaH_RH``
carries, and it stands for :math:`R_*` measured in Hubble radii.

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

- The true-vacuum fraction never reaches the percolation threshold, so there is no percolation
  temperature on the range searched.
- The transition percolates and then stalls, never reaching completion. This is reported
  separately from the previous case, because the physics differs: a transition that stalls
  leaves a mixture of phases.
- The estimate that brackets the percolation temperature does not converge.
- The sampled action implies a rate that is not smooth, which means the action itself is not
  resolved.
- The run exceeded its time limit. This is a statement about the clock and not about the
  physics, and it is reported as its own failure rather than as a verdict.

References
----------

.. [Guth:1981uk] A. H. Guth and E. J. Weinberg, *Cosmological consequences of a first
   order phase transition in the SU(5) grand unified model*, Phys. Rev. D **23** (1981) 876.
   The percolation criterion.

.. [Enqvist:1991xw] K. Enqvist, J. Ignatius, K. Kajantie and K. Rummukainen, *Nucleation and
   bubble growth in a first order cosmological electroweak phase transition*,
   Phys. Rev. D **45** (1992) 3415.

.. [Turner:1992tz] M. S. Turner, E. J. Weinberg and L. M. Widrow, *Bubble nucleation in first
   order inflation and other cosmological phase transitions*, Phys. Rev. D **46** (1992) 2384.

.. [Ellis:2018mja] J. Ellis, M. Lewicki and J. M. No, *On the maximal strength of a
   first-order electroweak phase transition and its gravitational wave signal*, JCAP **04**
   (2019) 003, `arXiv:1809.08242 <https://arxiv.org/abs/1809.08242>`_.
