.. _algorithm-plasma:

The plasma and its degrees of freedom
=====================================

The expansion rate, the energy released by the transition, the reheating that follows it and
the redshift of the spectrum all depend on how much radiation the universe contains. The
radiation is counted as an effective number of degrees of freedom, the number of massless
bosonic modes that would carry the same energy density at that temperature, and every species
has to enter that number exactly once.

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
   + \underbrace{2+1}_{Z_\mathrm{T},\,Z_\mathrm{L}}
   + \underbrace{2+1}_{\gamma_\mathrm{T},\,\gamma_\mathrm{L}}
   + \underbrace{4+2}_{W_\mathrm{T},\,W_\mathrm{L}}
   - \underbrace{4}_\mathrm{ghosts} \;=\; 12
   \;=\; \underbrace{1}_{h} + \underbrace{3}_{Z} + \underbrace{6}_{W^\pm}
   + \underbrace{2}_{\gamma},

and with the top, bottom and tau included this is
:math:`12 + \tfrac{7}{8}\times 28 = 36.5` at temperatures far above the masses. The same
36.5 is what the tabulated bath holds for those fields, which is what makes the subtraction
below possible.

Removing the overlap
--------------------

Those degrees of freedom are subtracted from the tabulated bath, at the masses the fields have
in the zero-temperature vacuum. In that vacuum the potential then adds exactly what the table
loses, and the total is the full tabulated bath plus whatever additional fields the model
carries. Away from the vacuum the fields of the potential follow
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

   The two inventories of the plasma, on one temperature axis. Above, the tabulated Standard
   Model bath. Below, the whole plasma of the two Higgs doublet model at the benchmark of
   ``examples/bp_lisa.yaml``, counted separately in each of its two phases. Each curve ends in
   an open circle at the temperature at which its phase stops existing.

Reading the two panels
----------------------

The upper panel is the Standard Model bath on its own. The step near 0.2 GeV is the
quark-gluon plasma turning into hadrons and the one near 100 GeV is the electroweak scale. The
plateau at high temperature sits at 104.1 rather than the free-field 106.75, because the fit
carries the interaction correction to the quark-gluon free energy. The energy and entropy
counts differ wherever a species is neither relativistic nor gone, which is at and above the
quark-hadron step, by at most 6.7 per cent at 0.2 GeV. Below about 0.5 MeV they part
permanently, since the neutrinos no longer share the photons' temperature once electrons and
positrons have annihilated, and the inset is the only place that can be seen, because 3.38
against 3.93 is half a per cent of the main panel. The two fits meet at 0.12 GeV, and the
small discontinuity there belongs to the fits and not to the physics.

The lower panel carries the tabulated bath as well as the fields of the potential, with the
Standard Model fields that appear in both counted once, so it is what the expansion rate is
built from. In the broken phase the fields of the potential take their masses from the field
and leave the plasma one after another, and the arrows mark where each has lost half of its
contribution, solved from the same integral the counting uses rather than read off
:math:`m = T`. The heavy scalars :math:`A` and :math:`H^\pm` of this benchmark reach that
point at 154 GeV, above the end of the broken phase, so they have no arrow. The broken-phase
curve ends at 121 GeV, the last temperature of this grid at which the broken minimum still
exists. Above it the field falls to the origin. A temperature grid ten times finer puts that
end at 121.5 GeV, so it is where the phase ends and not where the grid happens to lie.

In the symmetric phase those fields are massless and contribute their full 40.5 at every
temperature, so the slope of that curve is not theirs. It is the bath around them, which loses
degrees of freedom as it cools whether or not a transition happens. The curve ends in an open
circle at the temperature below which the origin of this potential stops being a minimum, so
that there is no symmetric phase to count. At the other end it is drawn to the top of the
range: the origin is certainly a minimum at high temperature, where the thermal masses
dominate, although the test cannot confirm it there, because the curvature is a finite
difference of the full potential and above about a TeV the field-independent radiation
dominates that potential and is differenced away. The two curves meet where the broken minimum
disappears. A model with a transition therefore has two different plasmas on the two sides of
its bubble walls, which is why the count is taken at a field value and not only at a
temperature.

A longer account, including the thermal functions and the resummation, is in
:doc:`the_effective_potential`.

References
----------

.. [ArnoldEspinosaPlasma] P. Arnold and O. Espinosa, *The effective potential and first order
   phase transitions: beyond leading-order*, Phys. Rev. D **47** (1993) 3546. The resummation
   whose counting of the longitudinal modes this bookkeeping has to match.
