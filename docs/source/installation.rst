Installation Guide
==================

TransitionListener can be installed in three ways, listed in order of
convenience:

1. :ref:`install-conda` — a fully resolved scientific stack from
   ``conda-forge``,
2. :ref:`install-pip` — a ``pip`` install from PyPI,
3. :ref:`install-clone` — an editable install from a GitHub clone, for
   developers and contributors.

Each route works out of the box for the *core* gravitational wave
workflow. Each route also supports the optional PTA likelihood layer
(``PTArcade`` / ``enterprise``), enabled by installing the
``ptarcade`` package alongside ``transitionlistener``.

If you do not have a conda flavour installed yet, any of
`conda <https://docs.conda.io/projects/conda/en/stable/>`_,
`mamba <https://mamba.readthedocs.io/en/latest/>`_, or
`micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_
work; the commands below assume ``conda`` but ``mamba`` /
``micromamba`` accept the exact same syntax.

.. note::
    **Apple Silicon (M-series Macs) — read this first.**
    The PTA likelihood layer depends on ``ceffyl``, which ships only
    as an ``osx-64`` build on ``conda-forge``. As a consequence, any
    environment that includes ``ptarcade`` has to be created with the
    ``--platform osx-64`` flag. The *core* TransitionListener install
    (no PTA layer) works natively on ``osx-arm64`` via any of the
    three routes below.

.. _install-conda:

Option 1: conda from conda-forge (recommended)
----------------------------------------------

The ``conda-forge`` package ``transitionlistener`` ships a fully
resolved scientific stack — no compiler step, no source build of
``mpi4py``:

.. code-block:: bash

    # Core install (no PTA likelihood)
    conda create -n TL -c conda-forge transitionlistener
    conda activate TL

    # With the optional PTA likelihood
    conda create -n TL -c conda-forge transitionlistener ptarcade
    conda activate TL

On Apple Silicon, the PTA-enabled environment must be created under
``osx-64``:

.. code-block:: bash

    conda create -n TL --platform osx-64 -c conda-forge transitionlistener ptarcade
    conda activate TL

(The core install above works natively on Apple Silicon without any
extra flag.)

If you intend to use the PTA likelihood, you may also need the
``setuptools`` workaround described in :ref:`install-pta-caveat`.

.. _install-pip:

Option 2: pip from PyPI
-----------------------

The package is published on PyPI as ``transitionlistener``:

.. code-block:: bash

    pip install transitionlistener            # core
    pip install transitionlistener ptarcade   # with the optional PTA likelihood

On Apple Silicon, the second command triggers a source build of
``enterprise-pulsar`` and its dependencies, which is brittle in
practice — for the PTA workflow on M-series Macs, prefer
:ref:`install-conda`.

If you install ``ptarcade``, see :ref:`install-pta-caveat`.

.. _install-clone:

Option 3: editable install from a GitHub clone (developers)
-----------------------------------------------------------

If you want to track ``main`` or contribute changes, install in
editable mode from a local clone:

.. code-block:: bash

    git clone https://github.com/tasicarl/TransitionListener.git
    cd TransitionListener
    pip install -e .                # core
    pip install -e . ptarcade       # with the optional PTA likelihood

The same Apple Silicon caveat applies to the ``ptarcade`` line.

If you install ``ptarcade``, see :ref:`install-pta-caveat`.

.. _install-pta-caveat:

PTArcade ``setuptools`` workaround (applies to all three routes)
----------------------------------------------------------------

PTArcade currently still relies on ``pkg_resources``, which was
removed from setuptools ≥ 72. The symptom in a PTA-enabled run is a
line like

.. code-block:: text

    PTA likelihood dependencies unavailable; continuing with SNR-only observability outputs.
      Import error: No module named 'pkg_resources'

printed during ``tl -c examples/example_point.yaml -v`` (or any other
PTA-touching invocation): the SNR table still renders, but the
**LOG LIKELIHOODS FOR PULSAR TIMING ARRAYS** table is silently
skipped.

The workaround is to install an older ``setuptools`` inside the
PTA-enabled environment:

.. code-block:: bash

    pip install "setuptools<72"

After this step, rerun the same ``tl -c ...`` command — the PTA
likelihood table should now populate alongside the SNR table.

The root fix has already landed in ``enterprise``
(`nanograv/enterprise@a6d504d <https://github.com/nanograv/enterprise/commit/a6d504d69254c45b530471efe0c7cc7a3c772c3f>`_)
and a corresponding PTArcade release is expected soon, after which
this workaround will no longer be needed. Pip may report
version-conflict warnings during this step — they are harmless and
can be ignored.

.. note::
    The first invocation of ``tl`` after a fresh PTA installation
    (e.g. ``tl --help``) may take several minutes while
    ``enterprise`` compiles and caches internal data. This is normal
    and only happens once.

Verifying the Installation
--------------------------

Whichever route you used, TransitionListener exposes a ``tl`` console
script with a built-in self-test:

.. code-block:: bash

    tl --check

This runs a battery of quick imports plus an end-to-end physics
smoke test on ``examples/example_point.yaml`` (the conformal
:math:`U(1)` benchmark) and verifies that the key observables
(``Tperc_SM_GeV``, ``alpha``, ``RH``) land in their expected
physical bands.

.. note::
    The first ``tl --check`` after a fresh install typically takes
    2–3 minutes while NumPy / SciPy / matplotlib warm their library
    and font caches and the phase tracer allocates its internal
    buffers. Subsequent runs finish in well under a minute.

For a faster sanity check that does not run the full smoke test:

.. code-block:: bash

    tl -V
    tl --help

If you cloned the repository (Option 3) and want to run the full
release test suite:

.. code-block:: bash

    pip install pytest
    pytest tests/test_release_smoke.py

If you encounter any issues, please open an issue at
https://github.com/tasicarl/TransitionListener/issues.
