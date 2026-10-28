.. _error-reference:

Error reference
===============

Why each failure is classified the way it is, and what was measured to decide it.
This is the detail that used to sit in the exception docstrings in
``src/transitionlistener/errors.py``; the docstrings now carry the summary and point
here.

.. _err-superluminal-sound-speed-error:

SuperluminalSoundSpeedError
---------------------------

The effective potential gave a sound speed of one or more.

``c_s^2 = (dV/dT)/(T d2V/dT2)`` sets the speed at which the bubbles grow,
``max(v_wall, c_s)``, the efficiency factors and the spectrum. A value of one or more is
not a speed, so nothing downstream of it has a meaning: for a runaway wall it replaces
the wall velocity and multiplies ``(beta/H)_RH`` by ``c_s``, and the spectrum divides by
``v_wall - c_s``. Rather than substitute a number the model did not give, the point is
refused and the computation that produced it is reported.
Which phase it came from is part of the report. Both are refused: the symmetric-phase
value is what the hydrodynamic matching across the wall is solved with, so a value of
one or more there is no more usable than in the broken phase.
``step_change``, ``daisy_ratio`` and ``daisy_outside``, where the caller has them, say
which of the two known mechanisms produced it. Neither is a property of the physics.
A large ``step_change`` means the finite differences have lost their significant digits
and the number is round-off.
``daisy_ratio`` is a size and ``daisy_outside`` is the verdict; the size alone does not
imply the verdict. Where the Arnold-Espinosa resummation is used outside its range of
validity, ``m << T``, the resummed term falls as a power of the temperature where the
mode it stands for should be Boltzmann suppressed, it comes to dominate the thermal
potential, and ``c_s^2`` drifts towards the ``1/2`` of a ``T^3`` potential rather than
the ``1/3`` of radiation. But a daisy term larger than the radiation is not by itself
that failure: at high temperature it legitimately exceeds the radiation while every mode
is still lighter than the temperature. ``daisy_outside`` is
:func:`generic_potential.daisy_outside_validity`, which requires both, and it is what the
message draws its conclusion from.

.. _err-too-much-supercooling-error:

TooMuchSupercoolingError
------------------------

Raised with CLI error code 8 from either of two stages, and the message says which.

At the transition-chain walk it means a candidate below the current phase did reach the
nucleation solver and the solve did not succeed. **That test is reaching the solver, not
what the solver concluded**: the transition may be genuinely too supercooled, or the solve
may have failed numerically, and the raise does not tell those two apart. The claim is
about the one step the walk had reached; steps already completed are not implicated.
At percolation it means the true vacuum fraction never reaches `f_perc`, so no percolation
temperature exists on the temperature range that was searched.
Distinct from `NoViablePhasePairError`, code 19, which comes only from the first stage and
means no candidate was ever offered to the solver, so nothing about the rate was measured
at that step.

.. _err-no-viable-phase-pair-error:

NoViablePhasePairError
----------------------

Raised when no phase pair below the current phase was ever a tunnelling candidate.

**Every claim here is about one step of the chain**, the phase the walk had reached, and
about the candidates offered below it. The transitions already found are counted in the
message, because this error discards them, and nothing here reassesses them or says how
they were obtained. In particular they need not have been solved: a second-order step
records a transition with no nucleation solve, and in the adaptive backend
`seedlessFirstOrderCandidate` accepts a pair without computing a nucleation temperature at
all, leaving that to percolation.
Distinct from `TooMuchSupercoolingError`, which at this same point means a candidate did
reach the nucleation solver and the solve did not succeed, and which is also raised from a
later stage, where percolation found no temperature at which the true vacuum fraction
reaches `f_perc`. **The test is reaching the solver, not what the solver concluded**: a
failed solve may mean the transition is genuinely too supercooled, or that the solve
failed numerically, and the raise does not tell those two apart. This class is the
narrower statement, that no candidate below this phase reached the solver at all, so
nothing about the rate at this step was computed either way.

.. _err-eternal-inflation-error:

EternalInflationError
---------------------

Raised when the eternal-inflation criterion is fulfilled.

``calcTf`` also raises it when the true vacuum fraction never reaches ``f_final``, which
is a different statement and the far more common one on a strongly supercooled line: the
transition *percolated* and then stalled. Measured on the 2HDM marginal line, the 29-point
zoom at the no-percolation edge from ``lambda3 = 5.8193`` to 5.8222, all 14 such points
reached ``Pint`` between 0.3215 and 0.7640, every one above ``f_perc = 0.28957``,
so each had a well-defined percolation temperature at the moment it was refused.
``Tperc``, ``Pint_max``, ``Tlow`` and ``f_final`` carry those numbers, because the
alternative is running the point again to recover what the solver already had.
``Tperc`` and ``Tlow`` are in the solver's internal temperature units, as the raising
code holds them; multiply by the model's ``conversionFactor`` for GeV. All four default
to ``None``, so a bare ``raise EternalInflationError()`` keeps working and a consumer
must treat ``None`` as "not reported" rather than as a value.
A transition that stalls at ``Pint`` well below one leaves a mixture of phases, so
``Tperc`` is meaningful there while ``Treh`` and the spectrum are not ordinary
predictions.

