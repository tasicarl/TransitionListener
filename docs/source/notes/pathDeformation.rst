.. _notes-pathDeformation:

``pathDeformation.py``
======================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/pathDeformation.py``, which now carry
the summary and point here.

.. _note-path-deformation-deformation-points:

``Deformation_Points``
----------------------

Unlike :class:`Deformation_Spline`, this class changes the points themselves rather than fitting a spline to the points. It is a more straightforward implementation, and when run with comparable inputs (i.e., the number of basis splines is about the same as the number of points), this method tends to be somewhat faster. The individual stepsizes here change with the total number of points, whereas in the spline implementation they mostly depend on the number of basis functions. However, as long as the path is fairly smooth, the total number of splines in that class can probably be smaller than the total number of points in this class, so this class will tend to be somewhat slower.

The two implementations should converge upon the same answer when the number of points and basis functions get large.

.. _note-path-deformation-extension-minimum:

``_extension_minimum``
----------------------

The path is extended past each end until it reaches a minimum. ``scipy.optimize.fmin`` was used for that, and it starts from ``x0 = 0`` with an initial simplex step of 2.5e-4 whatever the problem looks like. That step is in units of the knot index, so in field units it is 2.5e-4 times the spacing between the first two knots, and for a path whose ends are far apart it is large. On the conformal U(1) at g = 0.54 with the minima 1000 field units apart, the two-knot initial path has a knot spacing of the whole path, and the end extension settles a full knot out, at field -1000. That potential is even in the field, so this is the mirror of the broken minimum: the extended path runs from 1000 to -1000, is 2000 long instead of 1000, and connects two degenerate minima. The solver compares its ends, finds them equal, and reports "stable, not metastable", which comes back as an infinite action.

So the search walks outward instead of starting with a step of its own choosing, and two things bound it.

``floor`` is the potential at the opposite end of the path, passed when extending the metastable end. Tunnelling needs that end to stay the higher of the two, so an extension that would take it to or below the other end is the one that destroys the problem being solved, and the walk stops there. This is what separates the mirror minimum above, which is exactly degenerate with the broken minimum and is refused, from a real minimum a little way past the end, which is kept: on the 2HDM benchmark at T = 36.56 the potential at the end is -2.4347e+07, there is a genuine minimum 11.01 knots out at -2.4348e+07, and the broken minimum is far below at -6.79e+07, so the extension is allowed and the action stays at the converged 143.69.

The other bound is the rounding of ``V`` itself. Near a path end the potential can be flat to machine precision over many decades of ``x``: at that same 2HDM end the potential is -2.4e+07, so consecutive probes differ by rounding alone, about 5e-9, for ``x`` below 1e-4 knots. Taking the first such rise for a minimum stopped the search at x = 4e-10 and lost the real one at 11.01, which moved the action to 119.21 against the converged 143.69. A rise only counts when it clears that noise.

.. _note-path-deformation-extension-minimum-clears-the-floor:

``_extension_minimum.clears_the_floor``
---------------------------------------

A degenerate mirror minimum only touches the other end's value, it does not go below it, so this is a margin rather than a comparison: the two are the same number to every digit in that case, while a minimum that is genuinely allowed sits far above, -2.4348e+07 against -6.79e+07 on the 2HDM benchmark.

The margin is the rounding of the two values and nothing more, the same rule the walk above uses for a rise. A margin of a fixed relative size instead -- 1e-9 of the floor, as this was -- is not a property of the potential at all: the bounce depends only on differences of ``V``, so adding a constant to it must not change which paths are extended, and a margin proportional to ``|floor|`` changes with that constant. It also refuses minima that are genuinely metastable: an end 500 above a floor of -1e12 would have had to clear 1000.

What this cannot do is recover a gap the two values no longer carry. At a floor of -1e12 the spacing of doubles is 1.2e-4, so a gap of 1e-3 is eight of them and the margin, fifteen, refuses it, while the same gap at a floor of zero is accepted. That is not the rule changing its mind: it is the offset having thrown the information away before the comparison sees it, and a potential evaluated by summing terms is good to several of those spacings at best. The answer when the gap is inside the rounding is to leave the end where it is, which is what this returns and what the old search did on a potential it could not read. Nothing shipped here is near that limit -- the 2HDM compares -2.4348e+07 against -6.79e+07, a gap of 0.6 of the value against a margin of 1e-15 of it.

.. _note-path-deformation-barrier-is-resolved:

``_barrier_is_resolved``
------------------------

The barrier is the region between the two minima where the potential rises above the higher of them. Counting the samples in it is a direct measure of whether the grid can carry it, and it needs no second evaluation of the potential: the values are already in hand.

Measured counts, which is what the threshold is set from:

* conformal U(1) at g = 0.54 with 10000 samples, so a spacing of 0.1 on a path of 1000: 1 sample inside the barrier at T_int = 0.1, and 0 at 0.01 and below; * the 2HDM benchmark, whose barrier spans a generic part of a path 240 to 247 units long and stands 3.8e+06 to 1.5e+07 above the minima: 400 to 760 at every temperature; * `fullTunneling`'s own default of 100 samples on that same conformal path at T_int = 15: 2.

The threshold sits below that last one deliberately. A grid of 100 knots cannot carry the re-placed spacing, so the sample count is checked as well, and the two together mean a path is only re-sampled when its grid is blind to the barrier and has the knots to do better.

.. _note-path-deformation-clustered-v-samples:

``_clustered_V_samples``
------------------------

Both the spacing of the uniform samples and how far they reach past the ends are fixed fractions of the path length, while the barrier is a feature whose width is set by the temperature. In a strongly supercooled model it is a vanishing fraction of the path, so the uniform grid misses it and the action is wrong: measured on the conformal U(1) at g = 0.54, too high by 1.1e-2 relative where the spacing is 0.86 barrier widths, 6.6e-2 at 1.67 and 65% at 4.3.

So the samples approach the ends geometrically, which is where the barrier goes as the temperature falls: the barrier top moves towards the metastable minimum, and so does the escape point. The overhang has to shrink with them. Leaving it at 0.2 L is what dominates the error once the grid is clustered, because the potential is -2.0e+08 at the broken minimum and -9.3e-17 at the symmetric one, and 0.2 L past the symmetric end it is -2.4e+06: one cubic spline through all of that is bent, just inside the end, towards a value some 24 orders of magnitude above the barrier it has to resolve. With the overhang counted in samples at the local spacing instead, the action at T_int = 1e-4 goes from 98.40 to 79.97724 against a true 79.97725, and at 1e-5 from 583.8 to 69.468 against 69.465.

This grid is used only where the uniform one cannot see the barrier at all. It is not a better grid in general: its spacing ratios reach 1e-9, which an interpolating cubic spline does not take well on a potential with the 2HDM's curvature, where it returns an infinite action at temperatures the uniform grid resolves.

.. _note-path-deformation-spline-path:

``SplinePath``
--------------

The spline-fitting happens in several steps:

  1. The derivatives of the input points are found, and used to
     determine the path length and direction at each point.
  2. If `extend_to_minima` is True, additional points are added at each end
     of the path such that ends lie on local minima.
  3. The points are fit to a spline, with the knots given by the path
     distances from the first point.
  4. If `reeval_distances` is True, the distances to each point are
     re-evaluated using the spline. A new spline is fit with more accurate
     knots.

The potential as a function of distance can be defined in one of two ways. If `V_spline_samples` is None, the potential as a function of distance `x` along the path is given by `V[pts(x)]`, where `pts(x)` is the spline function that defines the path. If `V_spline_samples` is not None, the potential is first evaluated `V_spline_samples` times along the path, and another spline is fit to the output. In other words, when `V_spline_samples` is None, the input potential `V` is evaluated for every value `x` passed to to the class method :meth:`V`, whereas if `V_spline_samples` is not None, the input potential is only evaluated during initialization.

.. _note-path-deformation-is-acheaper-pairing:

``_isACheaperPairing``
----------------------

Written as a pair of refused comparisons rather than as ``action < best``, because every comparison against a `nan` is false: ``action < best`` rejects a `nan` candidate, which is right, but also keeps a `nan` incumbent against every later candidate, which is not. ``+inf`` needs the same care in the other direction, since a positivity test alone accepts it. The Euclidean action of a bounce is finite and non-negative, and :func:`bounceAction` stores 0 for a path whose two points coincide and ``+inf`` for a pairing it could not resolve, so a value that is not finite and non-negative is not an action: it never wins, and it is displaced by the first value that is one.

.. _note-path-deformation-images-of-the-false-vacuum:

``imagesOfTheFalseVacuum``
--------------------------

Where the potential is invariant under a group, each minimum stands for a set of images, all of them the same physics. Which image a phase was traced onto is settled by the sign of a numerical step and not by the potential, so the two phases of a transition can be stored on different images, and the bounce is then computed along a path between images rather than the path the field takes. The field separation and the action of that path are those of no physical channel, and the action can come out a factor of two wrong without anything downstream being able to tell. That defect is what this exists for, and it is worth the one specific reference: on commit 065ef81 with `models/TL_2HDM.py` class `R2HDM` at shipped settings, with the benchmark parameters of `examples/bp_lisa.yaml` at `lambda3 = 5.868` the path between images gave `S_3/T = 336.84` at T = 30.0 GeV where the path the field takes gives 152.97.

The bubble nucleates along whichever pairing carries the smallest action, so all of the images are candidates and :func:`bounceActionOverImages` computes the bounce for each of them. The transformation is applied to the false vacuum, leaving the true vacuum where the phase tracing put it, so that every candidate ends at the same point.

The images are returned with the one nearest the true vacuum first, and the point as it was traced wins a tie, which makes the order independent of the order the group is given in. The order decides nothing: every candidate is computed and the smallest action is kept, so it only settles which bounce is solved first and which entry a tie goes to. It is the order it is because a shorter path usually carries the smaller action, so the pairing solved first is usually the one kept. The two pairings approach each other as the false vacuum approaches a fixed point of the group, which is the point they share, and where the false vacuum sits at such a point there is one pairing and the group costs nothing.

An image equal to the traced point is dropped, so the identity, which most models' groups contain, costs nothing, and a false vacuum sitting at a fixed point of the group, the origin in particular, yields one candidate and costs one bounce however large the group is.

.. _note-path-deformation-images-to-pair-over:

``imagesToPairOver``
--------------------

Empty unless `pair_transition_over_images` is on, which leaves one bounce per temperature. A model with no symmetry group gives the same thing and costs the same either way.

Off by default because the defect it guards against is now fixed where it arises. The bounce used to be computed between two images because the phase tracing could put the two phases on different ones, and `phases.canonicalImageElement` now puts a whole phase on its canonical image, so the pairing recomputes a bounce whose answer is rejected. Measured on commit 065ef81 with `models/TL_2HDM.py` class `R2HDM` at shipped settings, with the benchmark parameters of `examples/bp_lisa.yaml` at `lambda3 = 5.868`, the coupling the pairing was added for: of 76 temperatures, 66 have two images genuinely apart, between 0.12 and 0.25 of the path length, and at none of the 66 did the image the tracing did not produce carry the smaller action. Its action was larger by a median factor 2.1, and the second bounce took 66 per cent of all the time spent on bounces, raising the whole point from 426.96 s to 1075.64 s. At `lambda3 = 5.6518`, where the false vacuum is the symmetric phase, the two images are the same point to the tracer's resolution and the switch costs 5 per cent.

Turning it on is the way to check that on a new model or a new group, which is why the switch is here rather than the code removed.

.. _note-path-deformation-bounce-action-over-images:

``bounceActionOverImages``
--------------------------

Where the potential is invariant under a group, the false vacuum stands for a set of images, all of them the same physical state, and which of them the phase tracing landed on is settled by the sign of a numerical step rather than by the potential. The paths from those images to the true vacuum are different paths and carry different actions, and the bubble nucleates along the one with the smallest action, so that is the action the bubble nucleation rate is built from.

Each image is offered to :func:`bounceAction` in turn, nearest to the true vacuum first, and the entry with the smallest action is the one stored. A pairing whose bounce the solver could not resolve comes back as ``trantype`` 0 with an infinite action, which loses every comparison, so it does not cost the temperature when another pairing resolved; where no pairing resolved, the first one's entry is kept and the caller sees the infinite action it would have seen before any of the images were tried. A pairing whose two points coincide comes back as ``trantype`` 2 with action 0, and only the nearest image can do that, since an image equal to the true vacuum is at distance zero and so is ordered first.

Nothing here catches an exception. Every ``transitionlistener`` error the bounce code raises about a path is already absorbed by :func:`bounceAction` into one of those two ``trantype`` values, except the one that reports a fault in the code, and the two modules' ``raise`` statements are pinned against that census in the tests, each entry measured by injecting the failure where it is really raised. So there is nothing left for a handler here to pass over, and what would still reach it is the run deadline, a fault in the code, or a failure from outside the package such as a ``ValueError`` out of scipy. None of those may be read as a pairing that happened not to resolve, and the last of them ends the temperature, which is what the single pairing already did.

