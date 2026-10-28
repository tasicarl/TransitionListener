.. _notes-plots:

``plots.py``
============

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/plots.py``, which now carry
the summary and point here.

.. _note-plots-tlplots-bounce-false-vacuum-at-temperature:

``TLPlots._bounce_false_vacuum_at_temperature``
-----------------------------------------------

The profile drawn beside this marker starts at the image `bounceActionOverImages` chose, which is the traced point unless `pair_transition_over_images` is on and a transformed image carried the smaller action. Evaluating the traced phase alone would then put the marker a reflection away from the curve it labels, and on a model whose fields are hundreds of GeV that is the whole width of the plot.

The phase is evaluated at `temperature`, because the plot is not always at `Tnuc` and `tr.high_vev` is, and the group element that took the traced point to the one used is then applied to it. Where no element of the potential's group reproduces that map, the phase point is returned unmapped: a marker in the traced place is wrong in the same way as before this, and inventing a transformation the potential does not have would be worse.

