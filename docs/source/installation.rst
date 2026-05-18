Installation Guide
==================

Choose the path that matches your use case:

- :ref:`core` — Linux or macOS, no PTA likelihood
- :ref:`pta-linux` — Linux, with PTA likelihood
- :ref:`pta-mac` — macOS, with PTA likelihood

.. _core:

Core installation
-----------------

Clone the repository and install with ``pip``:

.. code-block:: bash

    git clone https://github.com/tasicarl/TransitionListener.git
    cd TransitionListener
    pip install -e .

.. _pta-linux:

Installation with PTA support on Linux
---------------------------------------

PTA likelihood support requires ``PTArcade`` and its ``enterprise`` stack,
which depend on native libraries best managed through conda.

First, create a dedicated conda environment:

.. code-block:: bash

    conda create -n TL -c conda-forge python ptarcade ultranest tqdm corner getdist mpi4py

Activate the environment and reinstall ``setuptools`` via pip (conda's build of
setuptools does not always expose ``pkg_resources`` correctly on Python 3.12):

.. code-block:: bash

    conda activate TL
    pip install "setuptools<72"

Then clone and install TransitionListener:

.. code-block:: bash

    git clone https://github.com/tasicarl/TransitionListener.git
    cd TransitionListener
    pip install -e .

.. note::
    The ``pip install "setuptools<72"`` step works around a known incompatibility
    in ``PTArcade``: it relies on ``pkg_resources``, which was removed from
    setuptools ≥ 72. The root fix has already been applied to ``enterprise``
    (`nanograv/enterprise@a6d504d <https://github.com/nanograv/enterprise/commit/a6d504d69254c45b530471efe0c7cc7a3c772c3f>`_)
    and a corresponding PTArcade release is expected soon, after which this
    workaround will no longer be needed.
    Pip may report version-conflict warnings during this step — they are harmless
    and can be ignored.

.. note::
    The first invocation of ``tl`` after a fresh PTA installation (e.g. ``tl --help``)
    may take several minutes while ``enterprise`` compiles and caches internal data.
    This is normal and only happens once.

.. _pta-mac:

Installation with PTA support on macOS
---------------------------------------

On macOS, the ``ceffyl`` package required by ``PTArcade`` is only available
as an x86-64 build, so the conda environment must be created with the
``osx-64`` platform flag:

.. code-block:: bash

    conda create -n TL --platform osx-64 -c conda-forge python ptarcade ultranest tqdm corner getdist mpi4py

Activate the environment and reinstall ``setuptools`` via pip (conda's build of
setuptools does not always expose ``pkg_resources`` correctly on Python 3.12):

.. code-block:: bash

    conda activate TL
    pip install "setuptools<72"

Then clone and install TransitionListener:

.. code-block:: bash

    git clone https://github.com/tasicarl/TransitionListener.git
    cd TransitionListener
    pip install -e .

.. note::
    The ``pip install "setuptools<72"`` step works around a known incompatibility
    in ``PTArcade``: it relies on ``pkg_resources``, which was removed from
    setuptools ≥ 72. The root fix has already been applied to ``enterprise``
    (`nanograv/enterprise@a6d504d <https://github.com/nanograv/enterprise/commit/a6d504d69254c45b530471efe0c7cc7a3c772c3f>`_)
    and a corresponding PTArcade release is expected soon, after which this
    workaround will no longer be needed.
    Pip may report version-conflict warnings during this step — they are harmless
    and can be ignored.

.. note::
    The first invocation of ``tl`` after a fresh PTA installation (e.g. ``tl --help``)
    may take several minutes while ``enterprise`` compiles and caches internal data.
    This is normal and only happens once.

Verifying the Installation
--------------------------

Run the built-in check to verify all dependencies and the physics engine:

.. code-block:: bash

    tl --check

This runs a quick dependency test (~1 s) followed by an end-to-end physics
smoke test (~1–2 min). You can also inspect the command-line interface with:

.. code-block:: bash

    tl --help

If you encounter any issues, please open an issue at
https://github.com/tasicarl/TransitionListener/issues.
