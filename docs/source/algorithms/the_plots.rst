.. _algorithm-plots:

Plots a run can produce
=======================

Besides the numbers, a run can draw what it computed. Each plot is switched on in the
``additional_plots`` section of the configuration file, and each is the state of one stage of
the pipeline rather than a summary of the result, so they are the first place to look when a
point behaves oddly.

The code is in :py:mod:`transitionlistener.plots` for a single point and in
:py:mod:`transitionlistener.gridplots` for a scan.

Every figure of the single-point gallery below was produced by the configuration block shown
above it, run on the conformal dark :math:`U(1)` model of ``examples/example_point.yaml`` at
:math:`g = 0.7`, :math:`v = 0.1` GeV and :math:`y = 0.01`. The phases are numbered in the
order the tracer found them and are labelled P1 and P2 in the panels: P1 is the broken phase,
with the field away from the origin, and P2 the symmetric one.

The potential
-------------

.. code-block:: yaml

   additional_plots:
     potential:
       plot?: True
       T_GeV: 0.02
       phi_ranges_GeV: [0, 0.15]
       n: 400

The effective potential along the field directions at one temperature. This is the plot that
shows whether there is a barrier at all, and how deep the broken minimum has become. A
potential that is not smooth here is a model problem and everything downstream of it is
unreliable.

.. figure:: ../_static/plots/potential.png
   :width: 100%

   The potential at 0.020 GeV, just below the critical temperature of 0.0218 GeV. The
   barrier at :math:`\phi = 0.042` GeV separates the symmetric minimum at the origin from the
   broken one at :math:`\phi = 0.0965` GeV, and is 81 per cent as high as the broken minimum
   is deep.

The phases
----------

.. code-block:: yaml

     phases:
       plot?: True
       include_transitions?: False
       Tmin_GeV: 0
       Tmax_GeV: 0.03

Every minimum the tracer followed, against temperature, which is the output of
:doc:`phases_and_transitions`. A phase that stops early, jumps, or wanders away from a smooth
curve is the tracer losing it, and that is visible here and almost nowhere else.
``include_transitions?`` adds the candidate transitions between them.

.. figure:: ../_static/plots/phases.png
   :width: 100%

   The two phases the tracer followed. The symmetric phase, with the field at the origin,
   exists at every temperature. The broken phase ends at 0.025 GeV, where its minimum stops
   being a minimum. A phase that ends where the potential says it should is the tracer
   working, and a phase that ends in the middle of a smooth curve is the tracer losing it.

The action
----------

.. code-block:: yaml

     action:
       plot?: True
       Tmin_GeV: 0
       Tmax_GeV: 0.022
       phase_indices: [0, 1]
       n: 100

The bounce action :math:`S_3/T` against temperature for one pair of phases, which is what
:doc:`the_bounce_action` produces and what the nucleation rate is built from. The rate depends
on it exponentially, so a wobble of one per cent here is a factor of several in the rate: this
is the plot on which an unresolved bounce shows up.

.. figure:: ../_static/plots/action.png
   :width: 100%

   The action against temperature.

The energy density
------------------

.. code-block:: yaml

     energy_density:
       plot?: True
       Tmin_GeV: 0.002
       Tmax_GeV: 0.025

The energy density of each phase against temperature. The gap between the two is the energy
the transition has to release, so this is where a transition strength that looks surprising
can be checked against the thermodynamics it came from.

.. figure:: ../_static/plots/rho.png
   :width: 100%

   The energy density of the two phases.

The degrees of freedom
----------------------

.. code-block:: yaml

     dofs:
       plot?: True
       Tmin_GeV: 0.002
       Tmax_GeV: 0.025

The effective degrees of freedom of the plasma against temperature, in the sense of
:doc:`the_plasma`, which fix the expansion rate and the redshift of the signal.

.. figure:: ../_static/plots/dofs.png
   :width: 100%

   The counts and the energy densities of this benchmark point, split into the fields of the
   potential and the tabulated baths. Each quantity is drawn once per phase, so the pair of
   curves of one colour is the same quantity in the symmetric and in the broken phase, and
   the percolation and reheating temperatures are marked.

The percolation history
-----------------------

.. code-block:: yaml

     percolation:
       plot?: True
       Tmin_GeV: 0.00235
       Tmax_GeV: 0.00285

The true-vacuum fraction :math:`P_\mathrm{t}(T)` against temperature, with the nucleation and
percolation temperatures marked, which is the subject of :doc:`nucleation_and_percolation`.
The fraction is recomputed for the plot rather than read back from the run, so
``Tmin_GeV`` and ``Tmax_GeV`` are worth setting to a window around the transition: over the
whole range of the scan the rise is one vertical line. A window wider than the support of the
run is cut back to it, so that the curve reaches both ends of the panel.

.. figure:: ../_static/plots/percolation_0.png
   :width: 100%

   The true-vacuum fraction of this benchmark point, with temperature falling to the left.
   The dashed line is the percolation temperature, where the fraction reaches
   :math:`f_\mathrm{perc} = 0.28957`.

The bubble profile
------------------

.. code-block:: yaml

     profile:
       plot?: True

     profileV:
       plot?: True
       field_index_1: 0
       field_index_2: 1

The field through the wall of the critical bubble at the nucleation temperature, and the path
it takes through field space over the potential. The second is the one that shows whether the
path deformation found a sensible path or cut a corner.

.. figure:: ../_static/plots/profile.png
   :width: 100%

   The bounce of this benchmark point. The field sits at the true vacuum inside the bubble,
   falls through the wall at a radius of about 1000 inverse GeV, and reaches the false vacuum
   outside. A profile that stops before it reaches the false vacuum is an unresolved bounce.

The spectrum and the detectors
------------------------------

.. code-block:: yaml

     gw_spectrum:
       plot?: True

     sensitivities:
       plot?: True

The gravitational wave spectrum with its three sources, and the same spectrum against the
sensitivity curves of the detectors that ship with the package, as in :doc:`the_spectrum`.

.. figure:: ../_static/plots/GW_spectrum.png
   :width: 100%

   The spectrum of this benchmark point against the detectors, with the three sources drawn
   separately. The sound waves carry almost all of the signal here, so their curve lies under
   the total.

.. figure:: ../_static/plots/GW_sensitivities.png
   :width: 100%

   The detector sensitivities on their own, which is what ``sensitivities`` draws.



Plots a scan can produce
------------------------

A scan over a grid of model parameters draws one map per output. The gallery below comes from
``examples/example_grid.yaml``, a ten by ten grid of the abelian dark Higgs model of
``models/TL_dark_U1.py``, run with

.. code-block:: console

   $ tl -c examples/example_grid.yaml -j 10

Each panel is a thumbnail, and the full-size file is behind it.

Transition strength and the characteristic temperatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The panels below show the strength :math:`\alpha`, the inverse duration
:math:`(\beta/H)_\mathrm{RH}` and the critical, nucleation, percolation, completion and
reheating temperatures across a two-dimensional grid of model parameters.

.. image:: ../_static/example_output/log_plot_alpha.png
   :width: 31%
   :target: ../_static/example_output/log_plot_alpha.png

.. image:: ../_static/example_output/log_plot_betaH_RH.png
   :width: 31%
   :target: ../_static/example_output/log_plot_betaH_RH.png

.. image:: ../_static/example_output/log_plot_Tcrit_SM_GeV.png
   :width: 31%
   :target: ../_static/example_output/log_plot_Tcrit_SM_GeV.png

.. image:: ../_static/example_output/log_plot_Tnuc_SM_GeV.png
   :width: 31%
   :target: ../_static/example_output/log_plot_Tnuc_SM_GeV.png

.. image:: ../_static/example_output/log_plot_Tperc_SM_GeV.png
   :width: 31%
   :target: ../_static/example_output/log_plot_Tperc_SM_GeV.png

.. image:: ../_static/example_output/log_plot_Tf_SM_GeV.png
   :width: 31%
   :target: ../_static/example_output/log_plot_Tf_SM_GeV.png

.. image:: ../_static/example_output/log_plot_Treh_SM_GeV.png
   :width: 31%
   :target: ../_static/example_output/log_plot_Treh_SM_GeV.png

The plasma and its thermodynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sound speed :math:`c_\mathrm{s}` in each phase, the total degrees of freedom at the
reheating temperature, and :math:`\xi_\mathrm{c} = v(T_\mathrm{c})/T_\mathrm{c}`, the
field value at the critical temperature in units of that temperature, which is the usual
measure of how strongly the symmetry is broken there.

.. image:: ../_static/example_output/lin_plot_c_s_sym.png
   :width: 31%
   :target: ../_static/example_output/lin_plot_c_s_sym.png

.. image:: ../_static/example_output/lin_plot_c_s_bro.png
   :width: 31%
   :target: ../_static/example_output/lin_plot_c_s_bro.png

.. image:: ../_static/example_output/lin_plot_g_eff_tot_reh.png
   :width: 31%
   :target: ../_static/example_output/lin_plot_g_eff_tot_reh.png

.. image:: ../_static/example_output/lin_plot_xi_crit.png
   :width: 31%
   :target: ../_static/example_output/lin_plot_xi_crit.png

The peak of the spectrum
~~~~~~~~~~~~~~~~~~~~~~~~

The frequency :math:`f_\mathrm{peak}` in Hz and the amplitude
:math:`h^2\Omega_\mathrm{GW}` at which the predicted spectrum peaks, both as base-ten
logarithms, across the scan.

.. image:: ../_static/example_output/Add_info_log10_f_peak_Hz.png
   :width: 31%
   :target: ../_static/example_output/Add_info_log10_f_peak_Hz.png

.. image:: ../_static/example_output/Add_info_log10_h2OmegaGW_peak.png
   :width: 31%
   :target: ../_static/example_output/Add_info_log10_h2OmegaGW_peak.png

Signal-to-noise ratios
~~~~~~~~~~~~~~~~~~~~~~

The signal-to-noise ratio of the predicted spectrum in three planned detectors: BBO, DECIGO
and the Einstein Telescope.

.. image:: ../_static/example_output/SNR_BBO.png
   :width: 31%
   :target: ../_static/example_output/SNR_BBO.png

.. image:: ../_static/example_output/SNR_DECIGO.png
   :width: 31%
   :target: ../_static/example_output/SNR_DECIGO.png

.. image:: ../_static/example_output/SNR_ET.png
   :width: 31%
   :target: ../_static/example_output/SNR_ET.png

