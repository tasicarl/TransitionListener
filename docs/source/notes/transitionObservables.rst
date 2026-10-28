.. _notes-transitionObservables:

``transitionObservables.py``
============================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/transitionObservables.py``, which now carry
the summary and point here.

.. _note-transition-observables-build-action-spline:

``build_action_spline``
-----------------------

Above the hottest temperature whose action was computed there is no nucleation, not a constant action. Holding the action constant lets S/T fall as T rises, so the rate *grows* towards high temperature, which is backwards: the barrier grows there and the rate has to vanish. Measured on the 2HDM at lambda3 = 5.7727298627, where the action solve failed at the hot end of the support: the action was held at its value at T = 49.28, S3/T fell from 151.0 to 130.2 by T = 57.18, and the bubble number density came out 7340 times too large, which put the mean bubble separation 19 times too low and betaH_RH at 1883 against 99 and 96 at the neighbouring couplings. An infinite action gives a rate of zero there, which is the limit, and is what the estimator's own spline in `percolation_adaptivestepsize` already returns.

There is deliberately no `_fixedstep` twin of this. That backend builds its own `Sint` as an `interp1d` over the finite-action temperatures with `fill_value=(-inf, +inf)`, so it already returns an infinite action above its data, and its bubble-number integral runs to the hottest support temperature only because the integrand is identically zero above the last computed action.

.. _note-transition-observables-the-chain-ended-elsewhere:

``theChainEndedElsewhere``
--------------------------

Two labels for the same minimum have the same potential whatever their coordinates differ by, and a genuinely different vacuum is deeper or shallower by far more than any tracing residual, so the depth is the question. A pair of degenerate minima has no difference in depth to find, so the distance in field space is kept as a second condition and either one reports.

`valAt` interpolates the nodes the phase tracer put down, so it carries the tracer's resolution and not the potential's. On the real two Higgs doublet model at `lambda3 = 5.8672` the traced endpoint sits 1.2e-04 GeV from `X0` in the first field component and 6.5e-04 GeV in the second, which is 2.7e-06 of the field scale, and the potential at the two points agrees to eight digits. `np.allclose`, which this replaces, refused them on the smaller component alone and by a factor 1.04, because a test applied component by component gives the 11.5 GeV component a tolerance 21 times tighter than the 246.2 GeV one while the tracer's error is the same absolute size in both.

Both routes call this, the adaptive one and the fixed step size one, so that the two cannot drift apart.

