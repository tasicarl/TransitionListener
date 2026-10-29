.. _notes-generic-potential:

``generic_potential.py``
========================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/generic_potential.py``, which now carry
the summary and point here.

.. _note-generic-potential-daisy-mass-cubed-difference:

``_daisy_mass_cubed_difference``
--------------------------------

The daisy term of the Arnold-Espinosa resummation is proportional to this difference. Written out directly it subtracts two nearly equal numbers whenever the thermal correction is small against the mass itself, which is the whole low-temperature regime: on the conformal dark U(1) of ``examples/example_point.yaml`` at an internal temperature of 0.02, the two cubes are about ``1e9`` in internal units while their difference is about ``0.3``, so ten of the sixteen digits are gone before anything else happens. The potential itself tolerates that; its second temperature derivative does not, and the sound speed divides by exactly that second derivative. Measured there, the second derivative of the daisy term used to change by 60% when the derivative step was changed by a factor ten, and now changes by 2e-11.

Two things are needed. The difference is written as ``(m_0^2)^{3/2} [(1 + u)^{3/2} - 1]`` with ``u = Pi/m_0^2``, and the bracket is evaluated as ``expm1(1.5 log1p(u))``, which has no cancellation. And that evaluation is kept in real arithmetic: numpy's complex ``log1p`` and ``expm1`` lose accuracy for small arguments, by a relative ``8e-8`` at ``u = 1e-9`` against ``1e-16`` for the real ones, so the ``+0j`` that exists only to admit tachyonic modes must not be applied to the modes that are fine. Negative ``m_0^2``, and a thermal correction that is not small, keep the direct complex form, where there is nothing to cancel.

.. _note-generic-potential-generic-potential:

``generic_potential``
---------------------

This class acts as the skeleton around which different scalar field models can be formed. At a bare minimum, subclasses must implement :func:`init`, :func:`V0`, and they must provide a :class:`~transitionlistener.particles.MassSpectrum` instance via ``self.mass_spectrum``. Subclasses will also likely implement :func:`approxZeroTMin`. Once the tree-level potential and particle spectrum are defined, the one-loop zero-temperature potential (using MS-bar renormalization) and finite-temperature potential can be used without any further modification.

If one wishes to rewrite the effective potential from scratch (that is, using a different method to calculate one-loop and finite-temperature corrections), this class and its various helper functions can still be used. In that case, one would need to override :func:`Vtot` (used by most of the helper functions) and :func:`V1T_from_X` (which should only return the temperature-dependent part of Vtot; used in temperature derivative calculations), and possibly override :func:`V0` (used by :func:`massSqMatrix` and for plotting at tree level).

Such a model must also override :func:`V_thermal`, which is the temperature-dependent part of the potential on its own and is what the sound speed, the transition strengths and the wall velocity are taken from. It is a separate hook from :func:`V1T_from_X` because the two differ in what they include: :func:`V_thermal` carries the radiation bath that its ``include_decoupled`` argument selects and the daisy term, while the base :func:`V1T_from_X` carries neither. Overriding the potential without it raises :class:`errors.PotentialError` rather than reconstructing a thermal part from the mass spectrum, which would not be the model's own.

The `__init__` function performs initialization specific for this abstract class. Subclasses should either override this initialization *but make sure to call the parent implementation*, or, more simply, override the :func:`init` method. In the base implementation, the former calls the latter and the latter does nothing. At a bare minimum, subclasses must set the `Ndim` parameter to the number of dynamic field dimensions in the model.

One of the main jobs of this class is to provide an easy interface for calculating the phase structure and phase transitions. These are given by the methods :func:`getPhases`, :func:`calcTcTrans`, and :func:`findAllTransitions`.

The attributes that can (and should!) be set during initialisation are listed in the class
docstring of :class:`generic_potential`.

.. _note-generic-potential-generic-potential-daisy-outside-validity:

``generic_potential.daisy_outside_validity``
--------------------------------------------

The Arnold-Espinosa term resums the zero Matsubara mode of the bosons, which is justified where that mode is infrared-enhanced, meaning ``m << T``. Where instead ``m >> T`` a mode should be Boltzmann suppressed, and the one-loop thermal integral is: ``V1T`` goes to exactly zero. The daisy term does not. With a thermal correction ``Pi`` small against ``m^2`` it tends to ``-(T^3/8 pi) sum n_i c_i m_i``, a power law with no ``exp(-m/T)`` in it, so it survives where the physics says it should not and eventually exceeds the radiation that is still relativistic.

Measured on the conformal dark U(1) of ``examples/example_point.yaml``: the daisy term is 7% of the field-independent radiation at ``T/v = 1e-1``, equal to it at about ``1e-2``, thirteen times it at ``1e-3`` and ten million times it at ``1e-9``, while the lightest mode runs from ``m/T = 1.3`` to ``1.3e8``. Where it dominates, the free energy goes as ``T^3`` rather than ``T^4`` and the sound speed of the plasma tends to ``1/sqrt(2)`` instead of the ``1/sqrt(3)`` of the radiation that is actually there. A sound speed above ``1/sqrt(3)`` in a late, cold, heavily supercooled phase is therefore this, and not a property of the plasma.

This is reported and not corrected. Which resummation a model uses is the user's choice, and a prescription that is Boltzmann suppressed at low temperature is a change to the thermodynamics rather than a guard; it is left to a change of its own.

Returns ``(outside, ratio)`` with the ratio of the daisy term to the field-independent radiation. ``(False, nan)`` where it cannot be decided, and where the model is not using this resummation at all: with ``daisy`` set to ``"Parwani"`` or ``"off"`` there is no Arnold-Espinosa term in the potential, so building one here and reporting its size would describe a prescription the run does not use.

.. _note-generic-potential-generic-potential-debye-mass-sq:

``generic_potential.debye_massSq``
----------------------------------

Returns an array shaped like the boson mass spectrum, or ``None``. ``None`` is not the default: the base class measures the thermal part, as described below, and returns ``None`` only where that measurement fails its own check. A model that states its Debye masses in closed form should override this; where neither the override nor the measurement is available, the daisy term falls back to the difference of the spectra at ``T`` and at zero, which is what it used before.

That difference is badly conditioned, and the daisy term is the one place it matters. On the conformal dark U(1) of ``examples/example_point.yaml``, at an internal temperature of 0.02, the masses squared are about ``5e4`` while their thermal part is about ``5e-5``, so the subtraction keeps seven of sixteen digits; by an internal temperature of ``1e-7`` it keeps none and the thermal part underflows to exactly zero, which silently removes the daisy term altogether. A model that builds its masses as ``m^2(X) + Pi(T)`` already holds ``Pi`` and can hand it over exactly, and should override this with the closed form.

The default measures it instead of declining. Where the thermal part is ``Pi = c(X) T^2``, which is what the Arnold-Espinosa Debye masses are, the coefficient can be read off at a reference temperature high enough for the subtraction to keep its digits and then evaluated at any temperature. The reference is the model's own scale ``v_stable``, where ``Pi`` is of order the masses themselves.

That the thermal part goes as ``T^2`` is assumed by nothing here: it is checked, at a second reference temperature and at two field points, and the default declines whenever the check fails.

The check is not a formality, and neither is doing it at two field points. Of the eight models shipped in ``models/``, four pass and four fail, and the failures are all the same physics: where the bosonic masses are eigenvalues of a mass matrix whose *entries* go as ``T^2``, the eigenvalues do not. Measured on ``models/TL_2HDM.py``, the two longitudinal gauge modes move their apparent coefficient by 4.6% and 11% between ``T`` and ``T/2``. ``models/TL_dark_flipflop.py`` fails only where both of its fields are on: along ``X0``, where the second vanishes and the matrix is diagonal, it looks quadratic to ``7e-18``, and at a generic point it is off by ``5e-3``. A check at one field point would have accepted it and handed the daisy term a ``Pi`` that is wrong by half a per cent.

For those models there is no ``Pi`` to hand over: the daisy term needs the two spectra themselves and the subtraction is the only route, so declining is the correct answer rather than a missing feature.

Returns an array shaped like the boson mass spectrum, or ``None``.

.. _note-generic-potential-generic-potential-debye-coefficient:

``generic_potential._debye_coefficient``
----------------------------------------

Separate from :func:`debye_massSq` so the verification is paid once rather than on every call, and decided at two field points fixed by the model rather than at whatever field value happens to arrive first. ``Vtot`` is called with random field points while the model is being built, and a verdict read off those would differ from run to run.

Three outcomes. A model whose thermal part is not quadratic never measures again. One whose coefficient is the same at both reference points is taken to be field independent and never measures again either. The rest measure on every call, two extra spectrum evaluations. None of the models shipped here takes that third path: the four that pass the check all have a field-independent coefficient, and the four whose coefficient does move with the field turn out not to be quadratic either, which is the same mixing in both cases. It is kept for models that are.

