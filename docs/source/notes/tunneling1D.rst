.. _notes-tunneling1D:

``tunneling1D.py``
==================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/tunneling1D.py``, which now carry
the summary and point here.

.. _note-tunneling1-d-bracket-spread:

``_bracketSpread``
------------------

A branch whose action is not finite has not resolved the bounce either, and a relative spread cannot say so: `inf - inf` and `inf / inf` are both `nan`, and `nan > tol` is false, so the caller would accept the sample on its worst input. Non-finite is reported as an infinite spread, which the caller refuses.

A branch whose action is not positive is refused the same way, and for the same reason: agreement is not the only thing that makes a bounce resolved. The Euclidean action of a bounce is positive, so a branch returning zero or less has not found one, and two such branches agree perfectly with each other. Two equal negative actions give a spread of zero and would pass; `bounceAction`'s caller then replaces a negative action with zero and labels the point a second-order transition, so a solver failure would be delivered as a physical result. The same measurement that set the tolerance saw it: at lambda3 = 5.816407666, T = 9.8546 GeV one of the two branches returns a negative action.

