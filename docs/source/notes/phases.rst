.. _notes-phases:

``phases.py``
=============

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/phases.py``, which now carry
the summary and point here.

.. _note-phases-phase-info-build-interpolator:

``PhaseInfo._build_interpolator``
---------------------------------

A phase is a curve of minima that the tracer has already found; between two traced temperatures the field value has to stay in the range the trace establishes. An interpolating cubic does not guarantee that. The traced temperatures are not evenly spaced -- the tracer clusters them where it takes small steps and leaves gaps where it takes large ones -- and a cubic through spacing ratios of that size oscillates.

The failure mode is a displaced phase rather than a slightly wrong one. The bounce solver reads the phase through `valAt`, so an excursion is handed to it as the point it tunnels from; starting from a displaced symmetric phase shortens the path and lowers the barrier, and the action comes out too small over the temperatures either side.

A shape preserving interpolator cannot leave the data's range: between two traced temperatures each field component stays between the two values traced there, so the largest field value anywhere is the largest traced one. A trace that exhibits the excursion is the fixture of `tests/test_phase_interpolation_does_not_overshoot.py`, which compares the two interpolations on it directly.

Outside the traced range it extrapolates, as `splev` on this spline also did. `bubbledynamics` reads a raise from `valAt` as a phase that cannot be evaluated at that temperature, and clamps its stencils rather than range-checking them, so a refusal here would be taken for a missing phase and a NaN would pass that test unnoticed. Shape preservation says nothing about the extrapolation: it is a statement about the interval between two nodes.

.. _note-phases-descend-from-saddle:

``descendFromSaddle``
---------------------

A potential with a reflection symmetry has a stationary point at the origin at every temperature, so a minimiser seeded there returns it whether or not it is a minimum: the gradient vanishes by symmetry and there is no direction for the minimiser to take. The thermal mass term tilts the origin over below some temperature, because ``m_eff^2(T) = -|m^2| + c T^2`` changes sign, and the field then sits in a displaced minimum instead. In a strongly supercooled transition the barrier is nearly temperature independent, so the universe reaches that temperature while still in the high-temperature phase, and the displaced minimum is the false vacuum the bounce has to start from.

Stepping along the eigenvector of the most negative curvature and re-minimising finds it. Both directions are tried at each step size and the nearer of what they find is returned, since the two are equivalent under the symmetry only until the minimiser has moved and the sign of an eigenvector is arbitrary; the step grows from the tracer's own resolution, so the first step size that finds anything gives the nearest minimum. `max_displacement` bounds the search, so that a saddle whose descent leads far away does not deliver a distant minimum, such as the true vacuum, in place of the phase being traced.

Returns `x` unchanged where the Hessian there is already positive definite, and where no minimum is found within the bound; the caller decides what to do with a point that is still not a minimum.

.. _note-phases-canonical-image-element:

``canonicalImageElement``
-------------------------

Below the temperature at which the origin tilts over, which image of the displaced minimum the field sits in is settled by the sign of a numerical step, not by the potential: the images are exactly degenerate. Tracing therefore returns a phase on either image, and two seeds that reach the same physical phase on opposite images give two phases that `removeRedundantPhases` cannot tell are one, because it compares field values. The pair is then carried through as separate phases and the transition can be built between a high phase on one image and a low phase on the other, whose field separation and bounce action are those of no physical path. On the 2HDM at `lambda3 = 5.868` that gave `S_3/T = 336.6` at `T = 30.0` GeV against `152.6` and `153.3` at the neighbouring couplings, and no percolation at all.

`applySymmetries` chooses the image of a single point. Applied node by node it can send neighbouring nodes of one trace to different images, putting a step into a phase that is smooth, so the transformation is chosen once here and applied to every node. It is read off the node furthest from the origin, where the sign of each field component is resolved and the choice is therefore least ambiguous: near the origin every image of a node is the same point to within the tracing accuracy.

.. _note-phases-symmetrize-invariant-phases:

``symmetrizeInvariantPhases``
-----------------------------

Near a minimum that only the thermal masses hold in place, as the symmetric phase of a classically conformal model, the potential is flat to within the tracing tolerance and the trace drifts off the minimum. ``fmin`` (Nelder-Mead, initial simplex 5% of the starting point) leaves every point within about 20 ``xeps`` of the origin where it is, the step predictor evaluated there pushes the trace further out as the temperature falls, and the last node, at the lowest tracing temperature, is an extrapolation that is not a minimum. The spline through it puts the false vacuum at an arbitrary field value over the lowest decades in temperature, where an extremely supercooled transition happens: in the conformal U(1) at T/v ~ 1e-6 the dark photon came out several times heavier than T there, and the counted entropy of the false vacuum (``bubbledynamics.h_eff_DS``) varied with T.

A phase that coincides with its image under an element ``g`` of the symmetry group of the potential at every traced temperature, within ``tol``, is invariant under ``g``; its minimum lies exactly on the fixed subspace of ``g``, and it is replaced by its projection ``(X + X g)/2``. Phases that differ from their image anywhere, broken phases and phases that leave the fixed subspace at a second-order transition, are left alone.

.. _note-phases-merge-phases-meeting-at-apoint:

``mergePhasesMeetingAtAPoint``
------------------------------

A minimum that moves continuously into another as the temperature rises is one phase. The tracer stops where the Hessian loses positivity, which is exactly that point, and whether it then steps across the merge or restarts just below it is decided by where its step happens to land. On the 2HDM the same structure comes out as one phase at lambda3 = 5.813 and as two pieces 4.5 MeV apart at 5.8145.

Leaving the two separate does not make the answer imprecise, it loses the coupling: a pair of phases has to be degenerate somewhere in its common temperature range for a critical temperature to exist, and the degeneracy with the broken minimum sits at 96.97 GeV, well above the 39.60 GeV where the split falls. No tunnelling candidate is then offered and the chain ends with `NoViablePhasePairError`.

Two pieces are joined when the gap between them is a thousandth of the range they span and their endpoints across it agree to better than `diftol`. The field test is what separates this from a first-order transition, where the two phases differ by far more than `diftol`: 246 GeV against 1 GeV on the 2HDM. The gap test only keeps the join local.

.. _note-phases-trace-minimum-check-comment:

``traceMinimum.check``, implementation comment
----------------------------------------------

Applying it was tried and withdrawn. A conformal potential has no quadratic term at its origin, and `d2V` is a finite difference of step `x_eps`, so what the difference returns there is the step squared times the quartic: a small negative number, which `(eigs <= 0).any()` calls a saddle. The trace then stops short of T = 0 and the run dies with `WrongT0MinimumError` on a symmetric phase that ends at the origin, which is where it should end. Halving `x_eps` divides that number by four, which is how it can be told from a curvature.

Scaling the test by the curvature at the node before it does not help, because the quantity compared is still step-dependent: it passed locally and failed on the CI runner at the same threshold. Measuring against the largest eigenvalue is wrong in the other direction, admitting a genuinely negative endpoint wherever the spectrum spans many decades.

One reference, since the withdrawal rests on a measurement: commit 065ef81 with `models/TL_conformal_dark_u1.py` at g = 0.7, v_GeV = 0.1, in the internal units the tracer works in. At the internal temperature 1e-6 the origin's `d2V` reads -6.771e-08, -1.693e-08, -4.232e-09, -1.058e-09 and -2.645e-10 as `x_eps` halves from 4e-3 to 2.5e-4, extrapolating to the zero a conformal potential has there, while the broken phase's endpoint holds +9160 across the same steps.

So the endpoint is accepted on the slope alone, and a trace can still finish on a point that is not a minimum. The check that would decide it is not implemented: evaluate the curvature at two steps and ask whether it scales with the step, which separates a flat direction from a saddle without thresholding an absolute number.

