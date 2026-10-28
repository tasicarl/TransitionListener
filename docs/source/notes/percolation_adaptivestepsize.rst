.. _notes-percolation-adaptivestepsize:

``percolation_adaptivestepsize.py``
===================================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/percolation_adaptivestepsize.py``, which now carry
the summary and point here.

.. _note-percolation-adaptivestepsize-dynamiczoomwindow-accuracy-verdict:

``_dynamiczoomwindow_accuracy_verdict``
---------------------------------------

None means undecided, which is not the same claim as missed. Tperc and Tf can always be judged: when there is no previous estimate to difference against, each falls back to its own bracket width, so a single profile still measures them.

RH has no bracket measure, but since `RH_half_support` it has a self-contained one: the same samples re-integrated on every second support point give a second RH, and the relative difference between the two is an uncertainty a single profile can report. So RH is normally judged like the other two now. Its uncertainty is still infinite where that halved estimate cannot be built -- a support too short to halve, or one whose samples are unusable -- and an infinite one still means nobody measured it. A profile that meets every target it could measure and could not measure RH is returned as None rather than False, and that case is now the exception rather than the rule.

An RH that could not be determined at all is a different matter and counts as missed, because the value is absent rather than its reference.

Relation to `_dynamiczoomwindow_accuracy_reached`, which reads the same three settings a few lines above: that one answers whether the controller may stop refining and so has to be the strict conjunction, and this one only reports. They agree wherever RH was measured, `True` here exactly when `reached` is True, and differ only in that this function returns None instead of False where `RH_uncertainty` is infinite. Keep the two thresholds in step; `test_accuracy_verdict_producer.py` pins the agreement. The True return is not reachable through today's caller: with a previous estimate the caller has already returned True from the `reached` branch above, and without one `_estimate_dynamiczoomwindow_accuracy` leaves `RH_uncertainty` infinite, so the verdict is None. It is written out in full because the separate change that gives RH a second estimate on the accepted profile makes it live.

.. _note-percolation-adaptivestepsize-compute-step3-profile-comment:

``_compute_step3_profile``, implementation comment
--------------------------------------------------

The truncation above is right that nothing new nucleates below the saturation point: `S_3/T` is already 133 there on the 2HDM benchmark and only grows as the temperature falls, reaching 412 a decade below, so the rate is `exp(-133)` or smaller and no choice of action in this region can contribute to the integral. What the truncation missed is that this region is where the bubbles *already* nucleated keep growing, and `percIntegralODE` needs valid samples across it to carry that growth. A `nan` fails its validity test, which floors both the source and the transport term at `_LOG_GAMMA_SOURCE_FLOOR`, and the integral stops accumulating: on that benchmark `I` collapsed from 50 to 9.7e-38 and `P = 1 - exp(-I)` rounded to exactly 0 at all 174 support points, so a transition whose percolation temperature step 2 had already found was reported as one that never percolates.

Dropping the points instead does not work: it removes the growth along with the nucleation and leaves `I` at 10.4. The last computed action is the cheapest choice and costs nothing, since the rate there is negligible whatever the action is.

It has to be the last USABLE action, not merely the last finite one. `percIntegralODE`'s validity mask is `np.isfinite(S) & (S > 0.0) & ...`, so carrying a zero or negative action forward would leave the sample invalid and the integral floored exactly as a `nan` does, which is the failure this branch exists to prevent. A non-positive action is not physical for a bounce and means that solve did not succeed.

