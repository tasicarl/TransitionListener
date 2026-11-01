.. _algorithm-spectrum:

The gravitational wave spectrum
===============================

The transition leaves a stochastic background of gravitational waves. Its amplitude and the
frequency at which it peaks follow from four numbers computed earlier: the strength of the
transition, its inverse duration, the temperature at which it finishes, and the speed of the
bubble walls.

The three sources
-----------------

Three processes radiate, and the code adds their spectra:

**Sound waves.** The bulk motion left in the plasma after the walls have passed. On most
transitions this dominates.

**Bubble collisions.** The energy that remains in the scalar field itself, which radiates when
walls meet. This matters when the wall runs away, so that a large part of the energy never
reaches the plasma.

**Turbulence.** The fraction of the bulk motion that becomes vortical.

Each is written as a peak amplitude times a shape in frequency, with both taken from fits to
simulations. The shape is a broken power law whose break is at the peak, and the peak is set by
the length scale of the source, which is the mean separation between bubbles.

From the source to today
------------------------

A spectrum produced at temperature :math:`T` is redshifted by the expansion since then. The
amplitude scales with the fourth power of the scale factor and the frequency with its inverse,
so both depend on the degrees of freedom at the time of production through entropy
conservation. The redshift is applied from the reheating temperature, after the energy released
by the transition has been shared out, and not from the percolation temperature.

.. versionchanged:: 2.2
   The spectrum is redshifted from the temperature of the bath that carries the entropy, rather
   than from the temperature of the sector that underwent the transition. The two differ when
   the sectors are not in equilibrium with one another.

.. admonition:: The fits have a range
   :class: important

   The sound-wave template rests on simulations of transitions whose strength is at most of
   order a tenth. A scan that reaches far stronger transitions is extrapolating, and the
   amplitude it reports there is a statement about the template as much as about the model.
   Where the strength is large, a template built for strongly supercooled transitions is the
   appropriate one.

Observability
-------------

The spectrum is compared with the sensitivity of a detector by the signal-to-noise ratio,

.. math::
   \mathrm{SNR} = \sqrt{\mathcal{T} \int df
                  \left[\frac{h^2\Omega_{\rm GW}(f)}{h^2\Omega_{\rm noise}(f)}\right]^2 } ,

with :math:`\mathcal{T}` the observation time. Sensitivity curves for the interferometers and
the pulsar timing arrays are shipped with the package.

For pulsar timing arrays the comparison is made differently, against the measured free spectrum
rather than against a noise curve. The data is a posterior for the amplitude in each frequency
bin, and the likelihood of a model is the product of those posteriors evaluated at the
amplitudes it predicts. What the code reports is the distance from the best fit any such model
achieves on that data, expressed in standard deviations.

.. versionchanged:: 2.2
   The free spectra ship with the package and are selected by name, where they were previously
   downloaded while the run was in progress. Eight are available, and which one a run used is
   recorded. The reference against which the distance is measured belongs to the data set, so
   there is one per data set; a data set with no reference reports no distance rather than a
   distance from a reference obtained on different data.

Where it is in the code
-----------------------

``gwfopt.FOPTspectrum`` assembles the spectrum from the four numbers and the efficiency
factors. ``observability.Observability`` computes signal-to-noise ratios and the pulsar timing
likelihoods. ``pta_free_spectra`` holds the shipped free spectra and the reference likelihoods.
