.. _upgrading-2-2:

Upgrading from 2.1
==================

Version 2.2 gives different numbers from 2.1. This page is for a reader who has results from
2.1 and wants to know what to re-run. The full list of changes is the :doc:`/changelog`; the
reasoning behind each is on the algorithm pages, marked at the paragraph it applies to.

Scope
-----

The differences are concentrated on strongly supercooled transitions and on parameter points at
the edge of a scan, where a transition is marginal. Elsewhere they are small.

Two checks settle it quickly. If your transitions percolate well above the temperature at which
the barrier disappears, and no point returned a warning or an error, the changes move your
numbers by well under a per cent. If you have points that failed, points at the edge of a
region, or transitions that supercool by more than an order of magnitude in temperature,
re-run them.

Quantities that move
--------------------

**The expansion history, and everything downstream of it.** The scale factor comes from entropy
conservation and uses one entropy throughout, where it previously mixed two definitions of the
same quantity. This touches the percolation temperature, the mean bubble separation and the
inverse duration on every run. See :doc:`nucleation_and_percolation`.

**The transition strengths.** The three strengths that describe the wall share a normalisation,
so the efficiency of bubble collisions changes wherever the wall runs away. See
:doc:`hydrodynamics`.

**The counting of the plasma.** Models whose potential contains Standard Model fields had those
fields counted in both inventories in one place and in neither in another. Correcting it moves
the reheating temperature and, through the redshift, the peak frequency and the amplitude. See
:doc:`the_plasma`.

**The redshift of the spectrum**, now applied from the temperature of the bath that carries the
entropy. The two differ only when the sectors are not in equilibrium. See :doc:`the_spectrum`.

**The sound speed**, which now either returns a usable value or refuses the point. See
:doc:`hydrodynamics`.

**The bounce**, whose tolerances are measured against the bounce rather than against the
distance between the minima, and which refuses an action it has not resolved. See
:doc:`the_bounce_action`.

Points worth re-running first
-----------------------------

Four situations in which a 2.1 result is most likely to differ by more than a little:

- A supercooled point that reported no transition. The profile integration could accept a trial
  that had not reached the false vacuum and return an action of zero, which reads as no
  tunnelling.
- Any model whose potential has a symmetry. Two phases of one transition could be traced onto
  different images of the same physical state, and the bounce was then computed along a path
  between the images rather than along a tunnelling channel.
- A boundary in a scan near the percolation edge, which may be a property of the solver rather
  than of the physics.
- Any point that hit its walltime, which was reported as a finished point with a verdict.

New diagnostics
---------------

Ten columns say how far a point can be trusted rather than leaving it to be inferred. They are
listed in the :doc:`/error_reference`. A point carrying none of them was computed without the
solver having to rescue itself.

Limitations
-----------

The zero-temperature bounce is not computed, so a transition that supercools far enough for the
four-dimensional action to govern the rate is outside what this version covers.

The percolation history is computed with a wall velocity of one whatever is configured, because
the velocity in local equilibrium is only available once the history exists. Where that
velocity comes out as one, which is the common case on a supercooled transition, this makes no
difference.
