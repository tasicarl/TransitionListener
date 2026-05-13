Installation Guide
==================

The core package can be installed directly with ``pip``. PTA likelihood support
is optional and is best provisioned through a dedicated conda environment
because the ``PTArcade`` / ``enterprise`` stack may require additional native libraries.

Installation Instructions
-------------------------

For the core package, clone the repository and install it directly:

.. code-block:: bash

    git clone https://github.com/tasicarl/TransitionListener.git
    cd TransitionListener
    pip install -e .

To enable PTA likelihood calculations, create the recommended ``conda-forge``
environment on a Linux machine:

.. code-block:: bash

    conda create -n TL -c conda-forge python=3.10 ptarcade ultranest tqdm corner getdist mpi4py

If you are working on macOS, use instead

.. code-block:: bash

    conda create -n TL --platform osx-64 -c conda-forge python=3.10 ptarcade ultranest tqdm corner getdist mpi4py

Once the environment is created, activate it before installing TransitionListener:

.. code-block:: bash

    conda activate TL

Now, clone the TransitionListener repository from GitHub if you haven't already:

.. code-block:: bash

    git clone https://github.com/tasicarl/TransitionListener.git

Now, install TransitionListener using pip. Navigate to the root folder of TransitionListener and run:

.. code-block:: bash

    pip install .

If you want ``pip`` to record the optional PTA dependency explicitly, install the extra instead:

.. code-block:: bash

    pip install ".[pta]"

Alternatively, you can also install TransitionListener directly from the GitHub repository using:

.. code-block:: bash

    pip install git+https://github.com/carlotasillo/TransitionListener.git

Verifying the Installation
--------------------------

To verify that TransitionListener is installed correctly, you can run the following command in your terminal:

.. code-block:: bash

    python -c "import transitionlistener; print(transitionlistener.__version__)"

To see if the necessary dependencies are installed correctly, and in order to run a
quick (1 - 2 minute) test, you can also run

.. code-block:: bash

    tl --check

You can also check how the command line interface is working by running:

.. code-block:: bash

    tl --help

This should display the help message for the ``tl`` command line tool,
confirming that TransitionListener is installed and ready to use.

If you encounter any issues during installation, please feel free to reach out
via the GitHub issues page at https://github.com/tasicarl/TransitionListener/issues!
