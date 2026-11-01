.. _upgrading-2-2:

Upgrading from 2.1
==================

Version 2.2 changes numbers that 2.1 would have given. This page is for a reader who has
results from 2.1 and wants to know what to re-run and what to expect. It is not a list of
changes; that is the :doc:`/changelog`.

Does it affect me?
------------------

The effect is concentrated on strongly supercooled transitions and on the parameter points at
the edge of a scan, where a transition is marginal. Away from those it is small to absent.

Two checks tell you quickly. If your transitions percolate well above the temperature at which
the barrier disappears, and if none of your points returned a warning or an error, the changes
below will move your numbers by well under a per cent, and re-running is a formality. If you
have points that failed, points at the edge of a region, or transitions that supercool by more
than an order of magnitude in temperature, re-run them and read the rest of this page.

What moves, and by how much
---------------------------

**The expansion history and everything downstream of it.** The scale factor now comes from
entropy conservation and uses one entropy throughout, where before it mixed two definitions of
the same quantity. This touches the percolation temperature, the mean bubble separation and the
inverse duration on every run.

**The transition strengths.** The three strengths that describe the wall are now normalised
consistently with each other, so the efficiency of bubble collisions changes wherever the wall
runs away.

**The counting of the plasma.** Models whose potential contains Standard Model fields had those
fields counted twice in the energy density and almost not at all in the entropy. Fixing it
moves the reheating temperature and, through the redshift, the peak frequency and amplitude of
the spectrum.

**The spectrum's redshift.** It is applied from the temperature of the bath that carries the
entropy rather than from the sector that underwent the transition. The two differ only when the
sectors are not in equilibrium.

**The sound speed.** Where it previously returned a value above the speed of light or
not-a-number, it now either returns a usable value or refuses the point.

What used to be wrong rather than different
-------------------------------------------

These are the cases where 2.1 could return a number that was wrong without saying so. If any of
them applies to your parameter space, the old numbers should not be used.

.. admonition:: A bounce that was never resolved
   :class: caution

   On a strongly supercooled transition the profile integration could accept a trial that had
   not reached the false vacuum, and return an action of zero, which reads as no tunnelling.
   Points that reported no transition in a supercooled region are the ones to re-run.

.. admonition:: A path between images rather than a tunnelling channel
   :class: caution

   Where the potential has a symmetry, the two phases of a transition could be traced onto
   different copies of the same physical state, and the bounce was computed along a path
   between the copies. The action came out more than twice too large on a measured case, which
   in an exponent is an enormous factor, and nothing downstream could detect it.

.. admonition:: A solver failure reported as physics
   :class: caution

   Running out of the budget of bounce actions near the percolation edge was reported in a way
   that could not be told from a transition that genuinely never percolates. A boundary drawn
   in a scan from 2.1 may be a property of the solver.

.. admonition:: A run that hit its time limit
   :class: caution

   A point that exceeded its walltime was reported as a finished point with a verdict.

What is new to look at
----------------------

Ten diagnostic columns say how far a point can be trusted, rather than leaving the user to
infer it. They are described in the :doc:`/error_reference` and in the usage page. A point that
carries none of them was computed without the solver having to rescue itself.

What this version still does not do
-----------------------------------

The zero-temperature bounce is not computed, so a transition that supercools far enough for the
four-dimensional action to govern the rate is outside what this version covers.

The percolation history is computed with a wall velocity of one whatever is configured, because
the velocity in local equilibrium is only available once the history exists. Where the velocity
comes out as one, which is the common case on a supercooled transition, this makes no
difference.
