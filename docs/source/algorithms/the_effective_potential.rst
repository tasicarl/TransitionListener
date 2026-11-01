.. _algorithm-potential:

The effective potential
=======================

Everything a run computes follows from one function: the free energy of the scalar fields at a
given temperature. The phases are its minima, the transition is the passage between two of
them, the bounce is a solution of its equations of motion, and the energy released is a
difference of its values. A model file supplies the tree-level potential and the mass spectrum;
this page describes what is built on top of them.

What is computed
----------------

The effective potential is assembled as

.. math::
   V(\phi, T) = V_0(\phi) + V_1(\phi) + V_{\rm ct}(\phi) + V_T(\phi, T)
                + V_{\rm daisy}(\phi, T) + V_{\rm rad}(T),

where :math:`V_0` is the tree-level potential from the model, :math:`V_1` the one-loop
correction at zero temperature, :math:`V_{\rm ct}` the counterterms that hold the tree-level
vacuum and masses in place, :math:`V_T` the one-loop thermal correction,
:math:`V_{\rm daisy}` the resummation of the infrared-sensitive modes, and
:math:`V_{\rm rad}` the field-independent radiation, which does not move the minima but does
set the energy density and so the expansion rate.

The one-loop correction at zero temperature, in the modified minimal subtraction scheme, is

.. math::
   V_1 = \frac{1}{64\pi^2}\left[ \sum_{i\in\text{bosons}} n_i\, m_i^4
         \left( \ln\frac{|m_i^2|}{\mu^2} - c_i \right)
         - \sum_{f\in\text{fermions}} n_f\, m_f^4
         \left( \ln\frac{|m_f^2|}{\mu^2} - \frac{3}{2} \right) \right],

with :math:`n` the number of degrees of freedom of each species, :math:`m^2(\phi)` its
field-dependent mass squared, and :math:`\mu` the renormalisation scale. The thermal part is
the standard one-loop integral, evaluated from splined forms of the bosonic and fermionic
thermal functions, which take the mass over temperature squared as their argument so that a
negative mass squared can be handled as the real part of the defining integral.

Counterterms
------------

The one-loop correction shifts the minimum of the tree-level potential and the masses at it.
The counterterms undo that, so that the parameters a user supplies remain the physical vacuum
expectation value and the physical masses rather than tree-level inputs that happen to be near
them. They are fixed by requiring that the first and second derivatives of
:math:`V_1 + V_{\rm ct}` vanish at the zero-temperature vacuum.

.. versionchanged:: 2.2
   The derivatives that fix the counterterms used to take their step from the tolerance of the
   phase tracer, so tightening an unrelated accuracy setting changed the potential itself. They
   now take their own step, whose default is the value that was in effect in the default setup.

Resummation of the infrared modes
---------------------------------

At high temperature the one-loop expansion fails for the longitudinal modes of the gauge
bosons and for the scalars, because their thermal masses are small against the temperature.
The remedy is to resum the leading thermal corrections into the masses. Two prescriptions are
offered, selected per model:

``ArnoldEspinosa``
   The resummation is added as a separate cubic term, built from the difference between the
   thermally corrected and the uncorrected masses, and applied to the longitudinal modes.

``Parwani``
   The thermal masses are substituted into the one-loop thermal integral directly, so that all
   modes are resummed.

``off``
   No resummation, which is correct only where no mode is light against the temperature.

Neither prescription is uniformly better, and they differ by terms of the order that is being
resummed. Where a result depends on which is chosen, that dependence is an uncertainty of the
calculation and not a detail of the implementation.

.. admonition:: Where the resummation stops being valid
   :class: important

   The Arnold-Espinosa term is derived for modes that are light against the temperature. Used
   outside that range the resummed term falls as a power of the temperature where the mode it
   stands for should be exponentially suppressed, and it comes to dominate the thermal
   potential. The sound speed then drifts towards the value for a potential cubic in the
   temperature rather than that of radiation, which is how the failure shows up downstream.

   The code reports this: ``daisy_outside_validity`` requires both that the daisy term exceeds
   the radiation and that a mode is no longer light, since at high temperature the first alone
   is unremarkable.

Field-independent radiation
---------------------------

The species that do not appear in the potential are counted from tabulated baths and enter as
a field-independent term. It cannot move a minimum, but it sets the energy density and so the
expansion rate, and it has to avoid double counting the species that are in the potential. That
bookkeeping is the subject of :doc:`the_plasma`.

Temperature derivatives
-----------------------

The entropy, the energy density and the sound speed all come from derivatives of the potential
with respect to temperature, taken by finite differences.

.. versionchanged:: 2.2
   The step is now chosen from the size of the temperature-independent part of the potential
   rather than as a fixed fraction of the temperature.

.. admonition:: What this got wrong before 2.2
   :class: caution

   In the broken phase of a strongly supercooled transition the potential is dominated by a
   large, temperature-independent vacuum term. A finite difference then subtracts two nearly
   equal numbers, and with a step chosen as a fraction of the temperature the result was
   round-off rather than a derivative. The sound speed, which divides by a second temperature
   derivative, is where that first became visible: it came out dependent on the step, sometimes
   above the speed of light, and at low enough temperature it came out as not-a-number.

Settings
--------

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Setting
     - Default
     - What it controls
   * - ``daisy``
     - model
     - ``ArnoldEspinosa``, ``Parwani`` or ``off``.
   * - ``renormScaleSq``
     - model
     - The renormalisation scale squared in the one-loop correction.
   * - ``counterterm_derivative_step``
     - 1e-3
     - Step used for the derivatives that fix the counterterms.
   * - ``entropy_definition``
     - setting
     - Which entropy the expansion history and the sound speed are built from; the two must
       agree.

Where it is in the code
-----------------------

``generic_potential.Vtot`` assembles the pieces above. ``V1`` and ``V1T`` are the zero- and
finite-temperature loop corrections, ``Vct`` the counterterms, ``Vdaisy`` the resummation, and
``radiationEnergyDensity`` the field-independent part. ``dVdT`` and ``d2VdT2`` take the
temperature derivatives, and ``daisy_outside_validity`` reports when the resummation is being
used where it does not hold. A model file supplies ``V0``, ``boson_massSq`` and
``fermion_massSq``.
