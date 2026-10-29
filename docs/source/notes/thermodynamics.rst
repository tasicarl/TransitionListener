.. _notes-thermodynamics:

``thermodynamics.py``
=====================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/thermodynamics.py``, which now carry
the summary and point here.

.. _note-thermodynamics-reheating-geff:

``reheating_geff``
------------------

Three baths contribute: the fields of the potential (the transitioning sector, "DS") in the broken phase at ``X_broken``, reheated to ``T_DS``; the coupled bath, reheated with them; and the decoupled bath, still at ``T_dec``. Each enters with its temperature ratio to the bath that holds the Standard Model, ``g = sum g_i (T_i/T_SM)^4`` and ``h = sum h_i (T_i/T_SM)^3`` (arXiv:2109.06208, 2311.06346). Returns ``(g, h, T_SM)``.

The fields of the potential count in the Landau gauge counting, every mode including the Goldstone modes and the ghosts of the gauge bosons subtracted, as in ``radiationEnergyDensity``. Fields flagged ``is_SM`` count there and are removed from the tabulated bath by ``sm_fields_in_potential_geff``, so that each is counted once.

