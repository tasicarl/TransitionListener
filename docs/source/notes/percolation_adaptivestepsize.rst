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

