.. _algorithm-compiled-code:

Compiled code
=============

Most of the time of a point goes into a few loops that run many thousands of times: the potential
and its derivatives during the tracing of the phases, the shooting of the bounce, and the
percolation integral. These loops run in machine code, compiled by `numba
<https://numba.pydata.org>`_, which is a required dependency. The compiled code computes the
same expressions as the Python code it replaces, and the numbers agree to rounding, except where a
kernel takes other integration steps. The first run compiles the kernels, which takes about
twenty seconds, and later runs read them from a cache on disk.

The kernels live in :py:mod:`transitionlistener.compiled`. This page lists what is compiled for
which models, how to see which path a model takes, and how a model with several fields gets a
compiled potential.

The compiled potential
----------------------

A model with one field compiles its potential by itself at the end of its construction, from
its own functions, however it was constructed.

What compiles
   A model with one field whose tree-level potential and counterterms together are a
   polynomial in the field up to :math:`\phi^6`, and whose masses squared are quadratic
   in the field, each boson with a thermal mass :math:`c_T T^2`:

   .. math::

      m_b^2 = a_0 + a_1 \phi + a_2 \phi^2 + c_T T^2, \qquad
      m_f^2 = a_0 + a_1 \phi + a_2 \phi^2 .

   The one-loop potential may use its own renormalisation conditions per mode. All three
   daisy schemes are covered. The compiled potential is compared with the Python
   potential on a grid of fields and temperatures before it is used.

What does not compile
   A model with more than one field that does not supply a compiled model (see below), a
   mass that is not of this form, a tree-level potential of higher degree or not polynomial,
   and a model that replaces one of ``Vtot``, ``V1T``, ``V1T_from_X``, ``Vdaisy``,
   ``V_thermal``, ``gradV``, ``d2V`` or ``dgradV_dT``. Such a model runs on the numpy code of
   :py:mod:`transitionlistener.generic_potential`, with the same results and about thirty times more slowly.

How to see which it is
   ``pot.compiled_status`` is ``"compiled"``, or says why not. With ``verbose=True`` the
   model prints the same line when it is constructed.

Changing a model after its construction
   Assigning to an attribute of the model or of its class, and ``set_modelparams``, make the
   model read its compiled potential again at the next evaluation. This also holds for
   replacing a method, on the instance or on the class. A change inside a list, a dictionary or
   an array is not seen: assign the new object to its attribute instead.

Other compiled parts
--------------------

The energy and entropy density along a phase are compiled in the same way, unless the model
replaces ``energyDensity``, ``radiationEnergyDensity``, ``energyDensityDaisy`` or ``V1_from_X``,
or has Standard Model fields in its potential.

The 1D shots of the four-dimensional bounce are compiled for every model, with any number of
fields. To use the Python integrator, pass ``tunneling_class=tunneling1D.SingleFieldInstanton``
to :py:func:`transitionlistener.pathDeformation.fullTunneling`.

A compiled model with several fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mass eigenvalues of several fields cannot be read from a model, so a model that wants a
compiled potential writes them out once more as plain arithmetic. This is optional: the
Python potential of the model stays the reference, and without these lines the model runs
on it. Write three functions at the top level of the model file and return them, with the
numbers they read, from the method ``compiled_model``:

.. code-block:: python

   def compiled_tree(x, p):
       return 0.25 * p[0] * (x[0]**2 - p[2])**2 + 0.5 * p[1] * x[0]**2 * x[1]**2

   def compiled_bosons(x, T, p, m2):
       m2[0] = p[0] * (3 * x[0]**2 - p[2]) + p[1] * x[1]**2 + p[0] / 4 * T**2
       m2[1] = p[1] * x[0]**2 + p[1] / 12 * T**2

   def compiled_fermions(x, p, m2):
       m2[0] = (p[3] * x[1])**2

   class MyModel(generic_potential.generic_potential):
       ...
       def compiled_model(self):
           return compiled_tree, compiled_bosons, compiled_fermions, (
               self.lam, self.portal, self.v_sq, self.y)

``x[i]`` are the fields, ``p[i]`` the numbers in the order of the tuple, and ``m2[i]`` the
masses squared, in the order of the spectrum and including their thermal part if the model
has one. ``compiled_tree`` returns ``V0 + Vct``. The first three are pointers, so index them
one number at a time. Array operations such as ``x * x`` do not compile. The functions may
use ``numpy`` functions of single numbers such as ``np.sqrt``, and
``transitionlistener.compiled.multifield.eigenvalues_2x2(a, b, c)`` returns the two
eigenvalues of a mixing block, small one first. A model without fermions passes ``None`` for
``compiled_fermions``. No decorator is needed. ``models/TL_dark_flipflop.py`` and
``models/TL_2HDM.py`` are worked examples.

The numbers are read from the model each time it changes, so ``compiled_model`` takes them
from the attributes the Python potential uses, counterterms included, and nothing has to be
stated twice. The potential itself, its one-loop and thermal parts, the daisy term, gradient,
Hessian and temperature derivative come from the package, with the same expressions as the
Python code, and the simplex search of the phase tracing runs on them. The gradient is the
chain rule: the derivative of the potential with respect to each mass squared, read from the
thermal splines, times a finite difference of the functions above, so these functions should
be smooth in the fields, and two modes that cross should have the same degrees of freedom and
the same renormalisation constant. The first run compiles each function in less
than a second and stores it in the cache on disk. Later processes read it from there.

Before the compiled potential is used, its tree level, masses and total potential are
compared with the Python ones on a grid of fields and temperatures, which takes a few
milliseconds. If they differ, or if a function does not compile, or if the model replaces one
of the entry points above, the model runs on the Python potential with a warning, and
``pot.compiled_status`` says why. The energy and entropy density of such a model run on the
Python expressions.
