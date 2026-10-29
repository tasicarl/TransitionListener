.. _notes-percolation-adaptive-rate:

``percolation_adaptive_rate.py``
================================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/percolation_adaptive_rate.py``, which now carry
the summary and point here.

.. _note-percolation-adaptive-rate-temperatures-match:

``_temperatures_match``
-----------------------

Relative to the two temperatures compared, with no absolute floor, and pairwise, so each entry is judged against its own size rather than against the largest temperature in the array. Temperatures are in the model's internal units, where the stable minimum sits at ``v_stable``, so an internal temperature is ``v_stable`` times ``T/v`` and falls below one wherever the transition is supercooled by more than ``1/v_stable`` - 4e-3 for the 2HDM, 1e-3 for the models built on ``internal_scale = 1000``. Both of the forms this replaces were then a large fraction of the temperature they were applied to.

An absolute floor of one made the presence tolerance 1e-6 absolute, which is larger than the temperature itself wherever the support reaches below internal 1e-6. Scaling by the array's maximum instead makes it a large fraction of the cold end whenever the support spans decades: on the 144-point support of the 2HDM point at ``lambda3 = 5.8197``, which reached internal 2.4e-13 while its hot end stayed at 76.2, the tolerance was 7.6e-5 and 26 of the 143 adjacent pairs of its own support points fell inside it - 22 of them more than a tenth apart, the median one 172 per cent apart, and the three coldest 110 per cent apart each. The rule here merges four of the same 143, all of them duplicates to nine digits. Either way, every refinement point proposed inside the rise of P(T) counts as already present and is dropped, the controller finds nothing to add with most of its budget unspent, and the acceptance gate then rejects the profile the controller accepted.

``max(|v|, |candidate|)`` rather than ``|candidate|`` alone is deliberate but makes no measurable difference: writing ``v = candidate (1 + d)``, the two disagree only for ``d`` in ``(relative, relative / (1 - relative)]``, a window of relative width ``relative**2``.

.. _note-percolation-adaptive-rate-rate-interval-is-post-peak-inert-comment:

``_rate_interval_is_post_peak_inert``, implementation comment
-------------------------------------------------------------

``relative`` is 2e-6 rather than the 1e-6 of ``_temperature_present`` because the tolerance it replaces was the sum of the presence tolerances of the two endpoints. The width is therefore the one this lookup already had; only its scale changed. Both values answer the same question - is this the support point the caller named - so the wider one is kept here only to leave the acceptance behaviour of a warm support untouched.

Taking the closest entry rather than the first makes the width of this window inert on every path that reaches here: all three callers hand over two entries of ``temps`` itself, either an adjacent pair from a loop over the array or the endpoints ``_largest_probability_jump_interval`` read off it, so the matching entry is at distance zero and wins any argmin. No test can therefore separate this tolerance from the one it replaces, and none here tries to; what the tests pin is the index selection. The scale is made relative anyway, because an endpoint that has been through arithmetic would be looked up against the hottest support point instead of against itself.

