Usage
=====

After ``pip``-installing TransitionListener, you can easily start using it
from your terminal. It is really simple to get started.

Configuration files
-------------------

All it takes is a human-readable configuration file, a ``.yaml`` file, which
specifies the model parameters, scanning ranges, and desired outputs.

You can find example configuration files in the ``examples/`` directory of the
TransitionListener repository.


Single point evaluation
-----------------------

Let's say you have created a configuration file named ``example_point.yaml`` in the
``examples/`` directory,
which specifies a benchmark point which you would like to know of whether it features a
first-order phase transition and what gravitational wave signal it produces.

You can run TransitionListener using the command line interface as follows:

.. code-block:: console

   $ tl -c examples/example_point.yaml -v

This command will execute TransitionListener with the specified configuration
file and enable verbose output. This will
print detailed information about the progress of the computation to the terminal.
When the computation is finished, the results
will be saved to the scans directory to a subfolder as specified in the configuration
file. In there you will find 
various output files, including ``0_Input_params.txt`` reminding you of the chosen input parameters,
``1_All_params.txt`` showing all computed parameters during the performed run, as well as
``2_Observability.txt`` containing the computed observability metrics. Further, additional
plots and log files including information about the computation process are saved there.


Grid and line scans
-------------------

If you want to run a small grid scan using the command line interface, you can use the following command:

.. code-block:: console

   $ tl -c examples/example_grid.yaml -j 10

The -j flag specifies the number of parallel jobs to run. Here we took 10 jobs.
This can speed up the scan significantly on multi-core machines. The above scan takes
only a couple of minutes on a modern machine.
After the scan is complete, the results will be saved in the specified output
directory. You will find various output files, including an overview plot of the grid for
each output quantity, of the kind :doc:`algorithms/the_plots` shows.

On top of that, you will find close to 200 intermediate results saved in the output directory,
in the form of ``.txt`` files as well as ``.pdf`` plots.

Alternatively, you can also run an example line scan using the following command:

.. code-block:: console

   $ tl -c examples/example_line.yaml -j 5

This will perform a line scan over the specified parameter range with 5 parallel jobs.
The results will be saved in the output directory as specified in the configuration file.
The output is very similar to the grid scan case. For instance, you will find the
efficiency factor for converting released vacuum energy to bulk fluid motion among them.


Choosing the percolation solver
-------------------------------

Two solvers compute the history of the true-vacuum fraction, selected with
``percolation_algorithm_mode``:

.. code-block:: yaml

   percolation_algorithm_mode: adaptive_step_size   # the default

``adaptive_step_size`` decides for itself at which temperatures to evaluate the bounce action,
and needs no seed. ``fixed_step_size`` works on a temperature grid you fix in advance, and is
worth trying if a point behaves oddly. Most runs never need either setting touched.

What each solver does, which settings reach it, and what its warnings mean is in
:doc:`algorithms/nucleation_and_percolation` and :doc:`algorithms/the_observables`.


Precision modes
---------------

The tracing and tunnelling precision presets are configurable through
``precision_mode``:

- ``default`` -- vanilla settings.
- ``robust`` -- tighter phase tracing only.
- ``xtrace`` -- still tighter tracing (very small ``dtstart``, ``dtmin``).
- ``tunneltight`` -- default tracing plus tighter bounce path-deformation
  tolerances.
- ``benchmark`` -- tightens both tracing and tunnelling. Note: While being the
  most precise mode, it can be computationally expensive, with runtimes of a single
  computation of over one hour, in the case of multi-dimensional scalar potentials.


The radiation that takes part in the transition
-----------------------------------------------

A model declares which radiation is in thermal contact with the fields of its potential and
which is not, because the two are reheated differently and enter the expansion rate
differently. The defaults put the Standard Model in contact, which is what most models want.
The settings, and what to change for a dark sector that is decoupled from the Standard Model,
are in :doc:`algorithms/the_plasma`.


Random and nested sampling scans
--------------------------------
You can also run a random scan by using the following command:

.. code-block:: console

   $ tl -c examples/example_random_conformal.yaml -j 10

This will perform a random scan over the specified parameter ranges with 10
parallel jobs. The results will be saved in the output directory in a ``.csv`` file.
You can use this file 
to analyze the results further or visualize them using your preferred plotting tools.

If you are interested in performing a nested sampling scan using ``mpi`` for
parallelization you can use the following command:

.. code-block:: console

   $ mpiexec -n 16 tl -c examples/example_ultranest.yaml

This will run a nested sampling scan with 16 parallel processes.
The results will again be saved in a ``.csv`` file in the specified output directory.

For detailed installation instructions and usage examples please refer to the following sections:

- Installation: :doc:`installation`
- Usage: :doc:`usage`


For more information, visit the documentation at
https://tasillo.de/TransitionListener or check out the code
repository at https://github.com/tasicarl/TransitionListener.
