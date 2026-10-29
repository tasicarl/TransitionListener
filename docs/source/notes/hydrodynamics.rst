.. _notes-hydrodynamics:

``hydrodynamics.py``
====================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/hydrodynamics.py``, which now carry
the summary and point here.

.. _note-hydrodynamics-resolve-configured-sound-speed:

``resolve_configured_sound_speed``
----------------------------------

``"1/3"`` names ``c_s^2 = 1/3``, the massless plasma, and gives ``1/sqrt(3)``. Anything else is read as ``c_s`` itself. The two solvers historically accepted different halves of this, the adaptive one only the label and the fixed-step one only a number, so the value the sound-speed refusal in :class:`errors.SuperluminalSoundSpeedError` recommends was rejected by one of them. Both go through here now.

The result is strictly between zero and one: the spectrum divides by ``v_wall - c_s``, so a configured value of exactly one with a runaway wall is a division by zero, and the guard on the computed value calls one not a speed.

.. _note-hydrodynamics-hydrodynamics-sound-speed-is-step-dependent:

``Hydrodynamics.sound_speed_is_step_dependent``
-----------------------------------------------

A value can be round-off and still be a speed, so no test on the value itself can find it. Under extreme supercooling the thermal part of the potential sits far below the vacuum energy and the finite differences lose their significant digits; what comes back is then a number that moves when the step moves. Measured on the conformal dark U(1), at a temperature where the thermal part is 2.8e-16 of the vacuum energy, ``c_s`` comes out as 0.82 while ``c_s^2`` ranges over -0.001 to 0.89 across steps from ``dT/T = 1e-3`` to ``2e-1``: perfectly subluminal, and meaningless.

Returns ``(noisy, relative_change)``, the change being the larger of the two the step variation produces.

A value that is refused at one step and not at another is noisy: it has stopped being a speed somewhere inside the step range, which no single step can show. A value that is refused at *every* step is not: it is not a speed, but it does not depend on the step, and the refusal already reports what is wrong with it. Refused values are compared like any others, so one that is refused everywhere and also moves is noisy too. A value that cannot be computed at the base step is not called noisy; it is already reported as not-a-number.

