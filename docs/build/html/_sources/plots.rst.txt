Plots
=====

TransitionListener offers a variety of additional plots to help you
visualize and analyze the results of your phase transition simulations.
These plots can be enabled or disabled via the ``additional_plots`` section
in your input YAML file. Here is a brief overview of the available plots and
their configuration options:

- **Action Plot**: Visualizes the Euclidean action as a function of temperature.

  - ``plot?``: Set to ``True`` to enable the plot.
  - ``Tmin_GeV`` and ``Tmax_GeV``: Define the temperature range for the plot.
  - ``phase_indices``: Specify which phases to include in the plot.
  - ``n``: Number of points to compute.

- **Potential Plot**: Displays the effective potential at a specified temperature.

  - ``plot?``: Set to ``True`` to enable the plot.
  - ``T_GeV``: Temperature at which to plot the potential.
  - ``phi_ranges_GeV``: Range of field values to plot.
  - ``n``: Number of points to compute.

- **Phases Plot**: Shows the evolution of different phases over temperature.

  - ``Tmin_GeV`` and ``Tmax_GeV``: Define the temperature range for the plot.
  - ``plot?``: Set to ``True`` to enable the plot.
  - ``include_transitions?``: Whether to include transition regions in the plot.

- **Bubble Profile Plot**: Illustrates the bubble profile for specified field indices.

  - ``plot?``: Set to ``True`` to enable the plot.
  - ``field_index_1`` and ``field_index_2``: Indices of the fields to plot.

- **Energy Density Plot**: Depicts the energy density as a function of temperature.

  - ``plot?``: Set to ``True`` to enable the plot.
  - ``Tmin_GeV`` and ``Tmax_GeV``: Define the temperature range for the plot.
  - ``n``: Number of points to compute.

- **Degrees of Freedom Plot**: Shows the effective degrees of freedom over temperature.

  - ``plot?``: Set to ``True`` to enable the plot.
  - ``Tmin_GeV`` and ``Tmax_GeV``: Define the temperature range for the plot.
  - ``n``: Number of points to compute.

- **Sensitivity Curves Plot**: Compares the predicted gravitational wave spectrum with detector sensitivity curves.

  - ``plot?``: Set to ``True`` to enable the plot.
