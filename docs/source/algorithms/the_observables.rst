.. _algorithm-observables:

The transition observables
==========================

The percolation history is a curve. What a gravitational wave prediction needs from it is a
handful of numbers: when the transition happened, how much energy it released, how long it
took, and how fast the walls moved. This page is about those numbers, how each is defined, and
what the code reports when one of them cannot be trusted.

The temperatures
----------------

Five temperatures come out of a run, and they answer different questions.

:math:`T_\mathrm{c}`
   Where the two minima are degenerate. A property of the potential alone, computed before any
   bounce, and the only one of the five that does not depend on the expansion history.

:math:`T_\mathrm{nuc}`
   Where roughly one bubble has nucleated per Hubble volume, the condition
   :math:`\Gamma/H^4 = 1`. The familiar quantity, and the weakest of the five: on a strongly
   supercooled transition it can sit far from where the universe actually converts, and on a
   very slow one the condition is never met at all, which is reported rather than patched.

:math:`T_\mathrm{perc}`
   Where the true-vacuum fraction reaches :math:`f_\mathrm{perc} = 0.2896`, at which the
   bubbles form a connected network. This is the temperature the strength and the duration are
   evaluated at.

:math:`T_\mathrm{f}`
   Where the transition is finished, taken as the true-vacuum fraction reaching
   :math:`f_\mathrm{final} = 0.99`. One per cent of space is still in the false vacuum there,
   so this is a convention and not a sharp boundary; the value is a setting.

:math:`T_\mathrm{reh}`
   The temperature after the energy released has been shared out. It is not a point on the
   percolation curve but the end of a separate integration, described below.

.. versionchanged:: 2.2
   The nucleation temperature is read from a spline of the rate rather than from a bracketing
   search, and where the condition is not met on the range explored the run says so instead of
   returning the edge of that range.

The reheating temperature
-------------------------

The bubbles grow into plasma that the released energy has already heated, so the temperature
inside a bubble is not the temperature of the phase it is nucleating from. Stepping from one
support point to the next, an entropy-conserving update followed by an energy correction for
the fraction converted during the step, is a discretisation of

.. math::
   \frac{\mathrm{d}T_\mathrm{b}}{\mathrm{d}T} = \frac{3\,S(T_\mathrm{b})}{T\,S'(T_\mathrm{b})}
   + \frac{P_\mathrm{t}'(T)}{P_\mathrm{t}(T)}\,
     \frac{e_\mathrm{s}(T) - E(T_\mathrm{b})}{E'(T_\mathrm{b})} ,

with :math:`S` and :math:`E` the entropy and energy density of the broken phase and
:math:`e_\mathrm{s}` that of the symmetric one. The first term is the cooling the expansion
alone would give and the second is the heating. The integration starts where the converted
fraction first becomes non-negligible, with the instantaneous-reheating condition as its
initial value, and the reheating temperature is its value at the end of the transition.

.. versionchanged:: 2.2
   The equation is integrated with an error-controlled integrator rather than read off the
   tabulated trajectory. Read off the table, the answer depends on where the support points
   happen to sit, which shows up as point-to-point scatter along scans that are otherwise
   smooth.

The strength
------------

How much energy the transition releases, measured against the energy already present, is
:math:`\alpha`; it is defined from the trace anomaly in :doc:`hydrodynamics`, where the wall
velocity it feeds is computed. Several strengths appear in a run, evaluated at different
places, and the important thing is that they share a normalisation, because the efficiency
with which bubble collisions radiate depends on ratios of them.

The two inverse durations
-------------------------

How fast the transition proceeds is quoted as :math:`\beta/H`, and two different quantities go
by that name. They answer different questions and they are not interchangeable.

**The local one** is the logarithmic derivative of the rate at the percolation temperature,

.. math::
   \left(\frac{\beta}{H}\right)_{S_\mathrm{3} = T\,\frac{d}{\mathrm{d}T}\left(\frac{S_\mathrm{3}}{T}\right)
   \;\Bigg|_{T_\mathrm{perc}} ,

read off the action at one temperature. It is what the exponential approximation to the rate
gives, and it is the quantity most of the literature means.

**The integral one** is read off the mean separation between the bubbles,

.. math::
   \left(\frac{\beta}{H}\right)_{R_*} =
   \left(\frac{8\pi}{f_\mathrm{perc}}\right)^{1/3}
   \frac{\max(v_\mathrm{w}, c_\mathrm{s})}{R_*} ,

where :math:`R_*` follows from the whole nucleation history, as in
:doc:`nucleation_and_percolation`, and not from any single temperature.

The two agree when the transition is fast, because then the rate really is well described by
its slope at one point. They do not agree when it is slow, and the difference is physical
rather than numerical. Two consequences are worth stating plainly. A feature that appears in
one of them and not the other is not evidence of anything. And the local one can be negative:
where the rate peaks above the percolation temperature, the action is rising again by the time
the universe converts, and a negative value there is the correct answer rather than a defect.

.. admonition:: They must not be mixed inside one quantity
   :class: important

   The mean bubble separation is the length scale the spectrum is built on, so the inverse
   duration that enters the spectrum must be the integral one. Taking the amplitude from one
   definition and the peak frequency from the other produces a spectrum that belongs to no
   transition.

.. versionchanged:: 2.2
   The mean bubble separation is obtained from the expansion history rather than from a
   relation that assumes radiation domination, and the local derivative from a least-squares
   fit to the action samples rather than from a two-point difference. The adaptive support can
   place samples :math:`10^{-6}` of a temperature apart, and a difference across two of those
   turns the irregular :math:`10^{-2}` error that the path deformation leaves in
   :math:`S_\mathrm{3}/T` into errors of hundreds.

Completion
----------

Percolating is not the same as finishing. The false vacuum occupies a physical volume that the
expansion inflates and the growth of the bubbles eats, and the transition completes only where
the second wins; the criterion is in :doc:`nucleation_and_percolation`. A transition that
percolates and then stalls is reported as such, and is a different outcome from one that never
percolated.

What the code reports when a number cannot be trusted
-----------------------------------------------------

Every run writes a row of warnings and diagnostics beside the observables. A warning is a
statement that something about this point needs a human; a diagnostic is a measured number
that says how far from ordinary it is. None of them stops a run, and none of them changes a
result: the point of reporting is that the number is still there to be looked at.

The accuracy targets
~~~~~~~~~~~~~~~~~~~~

``WARNING:accuracy_targets_missed``, with ``DIAG:accuracy_tperc_uncertainty``,
``DIAG:accuracy_tfinal_uncertainty`` and ``DIAG:accuracy_rh_uncertainty``.

The adaptive solver refines until the percolation temperature, the completion temperature and
the mean bubble separation are resolved to their configured targets, and accepts the profile
either way once refinement has nothing left to try. The three diagnostics are the uncertainty
it measured in each, and the warning says a target was measured and missed.

The three states are worth telling apart. A false warning beside finite uncertainties means
every target that could be measured was met. A false warning beside an infinite separation
uncertainty means that one was never measured rather than met, which happens whenever there is
no previous estimate to difference against. A false warning beside ``nan`` means no verdict was
formed at all, which is what the fixed-step backend writes, since it has no such controller.

The wall velocity
~~~~~~~~~~~~~~~~~

``WARNING:v_wall_branch_ambiguous`` and ``WARNING:v_wall_no_solution``.

The velocity in local thermal equilibrium is found by driving a matching residual to zero, and
the search assumes that residual crosses zero once below the Jouguet velocity. For a strong
transition it can cross more than once, and the velocity is then a choice among branches rather
than a determination; the first warning says so. The second says no root exists at all, which
is a different statement from a wall that runs away: a runaway is the expected outcome of a
strongly supercooled transition and is not flagged.

The inverse duration
~~~~~~~~~~~~~~~~~~~~

``WARNING:betaH_small``, ``WARNING:betaH_very_small``, ``WARNING:betaH_mismatch``,
``WARNING:betaH_nonfinite`` and ``WARNING:betaH_S3_fit_unstable``.

The first two say the transition is slow enough that the templates built for fast ones are
being extrapolated. The mismatch says the two definitions above disagree, which on a slow
transition is expected and on a fast one is a reason to look. The last says the least-squares
fit to the action moved when it was repeated on a smaller set of samples, so the derivative is
not resolved by the support that is there.

The thermodynamics
~~~~~~~~~~~~~~~~~~

``WARNING:noisy_c_s`` with ``DIAG:c_s_step_change`` says the sound speed moved when the
differencing step was changed by a factor ten in each direction. A value can be round-off and
still lie between zero and one, so no test on the value itself finds this; on a healthy point
the change is of order :math:`10^{-7}`.

``WARNING:daisy_outside_validity`` with ``DIAG:daisy_over_radiation`` says the Arnold-Espinosa
daisy term is being used where it does not apply, as described in
:doc:`the_effective_potential`. It reports a modelling choice and nothing is corrected.

``WARNING:entropy_scheme_sensitive`` with ``DIAG:entropy_scheme_cs2_spread`` and
``DIAG:entropy_scheme_lna_gap`` says the two ways of counting the entropy, and so the two
expansion histories they give, differ enough that the mean bubble separation of this point
depends on which was chosen. Nothing in the literature chooses between them, so the difference
is carried as an uncertainty.

The nucleation history
~~~~~~~~~~~~~~~~~~~~~~

``WARNING:nucleationRate_nonexponential`` says the rate does not behave like an exponential
over the support, which is the assumption behind the local inverse duration.
``WARNING:spline_tnuc_unavailable``, ``WARNING:spline_tnuc_not_reached`` and
``WARNING:spline_tnuc_failed`` distinguish three ways the nucleation temperature can be
absent: no spline could be built, the condition was never met on the range explored, or the
root-finding failed. ``WARNING:false_vacuum_not_shrinking`` says the completion criterion is
not satisfied at the percolation temperature, so this transition percolates but has not been
shown to finish.

The phases
~~~~~~~~~~

``WARNING:not_T0_global_min`` says the chain of phases does not end in the minimum the model
declares as its zero-temperature vacuum, which usually means the phase tracing lost the phase
rather than that the model is wrong.

The bounce
~~~~~~~~~~

``WARNING:action_rescue_attempted``, ``WARNING:action_rescue_failed`` and
``DIAG:action_rescue_attempts`` record that an action looked unstable against its neighbours
and was recomputed at a tighter path deformation, and whether that helped.

In the code
-----------

``transitionObservables`` assembles the row, with ``build_action_spline`` for the action
interpolation and ``theChainEndedElsewhere`` for the phase check. ``bubbledynamics`` holds
``calc_betaH_S3`` for the local inverse duration, ``integrate_broken_temperature`` for the
reheating temperature, ``falseVacuumVolumeGrowthRate`` for the completion criterion and
``entropy_scheme_sensitivity`` for the entropy comparison. The registry of every warning and
diagnostic key, with the label each carries in a plot, is in ``config.py``.
