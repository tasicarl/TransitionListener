.. _algorithm-plasma:

The plasma and its degrees of freedom
=====================================

The expansion rate, the energy released by the transition, the reheating that follows it and
the redshift of the spectrum all depend on how much radiation the universe contains. That is
counted as an effective number of degrees of freedom, and the counting has to be done once:
every species has to appear exactly once, and no species may be left out.

This is harder than it sounds, because TransitionListener holds the plasma in two separate
inventories that have to be kept from overlapping.

Two inventories
---------------

The fields that appear in the effective potential are counted mode by mode, with the masses
they have at the field value and temperature in question. Their contribution changes as the
field moves, which is exactly what makes the two phases differ, so they cannot be taken from a
table.

Everything else is counted from tabulated radiation baths: a kinetically coupled bath, which by
default holds the Standard Model, and a kinetically decoupled bath, which can hold a secluded
sector. These are functions of temperature alone.

Every particle must appear in exactly one of the two. For a model whose potential contains no
Standard Model fields the division is clean. For one that does, such as a two-Higgs-doublet
model, the same fields are in the potential and in the table, and something must remove the
overlap.

Counting the fields of the potential
------------------------------------

In the Landau gauge the modes of the potential are not the physical particles
[ArnoldEspinosaPlasma]_. Each gauge boson
contributes two transverse modes and one longitudinal mode, each broken generator adds a
Goldstone mode, and each gauge boson is accompanied by a ghost that removes one bosonic degree
of freedom. The gauge-dependent pieces cancel between themselves, as they must, and what is
left is the physical count. For the Standard Model fields of a two-Higgs-doublet model at the
zero-temperature vacuum,

.. math::
   \underbrace{1}_{h} + \underbrace{1+2}_{G^0,\,G^\pm}
   + \underbrace{2+1}_{Z_{\rm T},\,Z_{\rm L}}
   + \underbrace{2+1}_{\gamma_{\rm T},\,\gamma_{\rm L}}
   + \underbrace{4+2}_{W_{\rm T},\,W_{\rm L}}
   - \underbrace{4}_{\rm ghosts} \;=\; 12
   \;=\; \underbrace{1}_{h} + \underbrace{3}_{Z} + \underbrace{6}_{W^\pm}
   + \underbrace{2}_{\gamma},

and with the top, bottom and tau included this is
:math:`12 + \tfrac{7}{8}\times 28 = 36.5` at temperatures far above the masses.

That number is the bridge between the two inventories.

Removing the overlap
--------------------

Exactly those degrees of freedom are subtracted from the tabulated bath, at the masses the
fields have in the zero-temperature vacuum. The books then close: in that vacuum the potential
adds precisely what the table loses, and the total is the full tabulated bath plus whatever
additional fields the model carries. Away from the vacuum the fields of the potential follow
their field-dependent masses, which is what makes the two phases differ, while the table
continues to supply the light species at the temperature in question.

.. versionchanged:: 2.2
   Before, the overlap was removed by freezing the tabulated Standard Model above half the mass
   of the lightest massive field that the potential carried. That froze the photon, the gluons,
   the light fermions and the quantum chromodynamic crossover along with the fields it was
   meant to exclude, and it was stored for the whole module rather than for one model, so
   constructing one model changed the counting for every other model in the same session.

   The degrees of freedom of the fields of the potential were also counted incorrectly: fields
   flagged as belonging to the Standard Model were masked out entirely, the Goldstone modes
   were skipped, and the ghosts were kept. All modes are now counted and one ghost per gauge
   boson is subtracted, which is the gauge counting the energy density had always used.

.. figure:: ../_static/algorithm_plasma_dof.png
   :width: 100%
   :align: center

   The two inventories. Above, the Standard Model bath as the package tabulates it: the step
   near 0.2 GeV is the quark-gluon plasma turning into hadrons and the one near 100 GeV is the
   electroweak scale. The plateau at high temperature sits at 104.1 rather than the free-field
   106.75, because the fit carries the interaction correction to the quark-gluon free energy.
   The two counts differ wherever a species is neither relativistic nor gone, which is at and
   above the quark-hadron step, by at most 6.7 per cent at 0.2 GeV; below about 0.5 MeV they
   part permanently, since the neutrinos no longer share the photons' temperature after
   electrons and positrons annihilate, and the inset shows that, since 3.38 against 3.93 is
   half a per cent of the main panel. The two fits meet at 0.12 GeV, and the small
   discontinuity there belongs to the fits and not to the physics.

   Below, the fields of the two Higgs doublet model's own potential, at the benchmark of
   ``examples/bp_lisa.yaml``, counted separately in each of its two phases. In the symmetric
   phase every mode is massless, so the count sits at its relativistic value of 40.5 at every
   temperature. In the broken phase the modes take their masses from the field, and the count
   falls as each of them leaves the plasma in turn: the arrows mark the temperature at which a
   mode has lost half of its contribution, solved from the same integral the counting uses
   rather than read off :math:`m = T`. The two phases meet where the broken minimum
   disappears. A model with a transition therefore has two different plasmas on the two sides
   of its bubble walls, which is why the count has to be taken at a field value and not only
   at a temperature.

In the code
-----------

``thermodynamics.potential_fields_geff`` counts the fields of a potential in the gauge counting
above, for the energy density, the pressure or the entropy. ``sm_fields_in_potential_geff``
returns what those fields contribute to the tables, ghosts included, and zero for a model whose
potential carries none of them. ``generic_potential.radiationEnergyDensity`` subtracts the one
from the other. ``generic_potential._check_radiation_baths`` warns where a potential carrying
Standard Model fields is given a coupled bath that is not the Standard Model table, since the
subtraction then removes species the bath may not contain.

A longer account, including the thermal functions and the resummation, is in
:doc:`the_effective_potential`.

References
----------

.. [ArnoldEspinosaPlasma] P. Arnold and O. Espinosa, *The effective potential and first order
   phase transitions: beyond leading-order*, Phys. Rev. D **47** (1993) 3546. The resummation
   whose counting of the longitudinal modes this bookkeeping has to match.
