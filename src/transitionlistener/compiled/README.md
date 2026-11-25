# Compiled kernels

The inner loops of TransitionListener in machine code, compiled with [numba](https://numba.pydata.org). Each kernel
computes the same expressions as the Python code it replaces; the Python code stays the reference, and every compiled
potential is compared with it before it is used. The first run compiles the kernels (about twenty seconds) and stores
them in a cache on disk. The implementation note "Compiled code" of the documentation describes which models get which
kernels.

| file | what it does | used by |
|---|---|---|
| `splines.py` | FITPACK `splev`/`splder`, scipy's `BSpline` and `PPoly` evaluated at one point, and a PCHIP interpolant with a compiled scalar call: the thermal functions `Jb`/`Jf`, the degrees of freedom of the Standard Model and the traced phases | `finiteT`, `bubbledynamics`, `onefield_potential.py` |
| `roots.py` | scipy's `brentq`, one step at a time (numba cannot cache code that takes a function as an argument) | `tunneling1D`, `onefield_potential.py` |
| `shooting.py` | the overshoot/undershoot method of the 1D bounce: the start from the linearised solution near the centre and the integrator of one shot | `tunneling1D` |
| `deformation.py` | the arithmetic of one step of the path deformation of a multi-field bounce, and the numbers of a spline path | `pathDeformation`, `helper_functions` |
| `percolation.py` | the percolation integral on every prefix of the temperature grid, and the checks of the adaptive solver on its support | `bubbledynamics`, `percolation_adaptive*` |
| `onefield.py` | the one-loop effective potential of one field, its derivatives and its thermodynamics, from a table of numbers | `onefield_potential.py` |
| `onefield_potential.py` | reads that table from a one-field model at the end of its construction, and checks it against the model's Python potential | `generic_potential`, `bubbledynamics` |
| `multifield.py` | the one-loop effective potential of any number of fields and its derivatives, from three native functions of a model (tree level, boson and fermion masses) | `multifield_potential.py` |
| `multifield_potential.py` | compiles those functions from a model's `compiled_model` and checks the result against the model's Python potential | `generic_potential`, `bubbledynamics` |
| `bounce_table.py` | the interpolant of the table of the leading-order 3D action and the edge of its domain | `eft_rate/abelian_lo_bounce_table.py` |
| `mixing.py` | one partial wave of the coupled 3 x 3 fluctuation determinant of the Goldstone mode and the gauge field, with scipy's Radau step control | `eft_rate/abelian_determinants.py` |

`onefield.py`, `multifield.py`, `splines.py`, `roots.py`, `shooting.py`, `deformation.py` and `percolation.py` work
for every model. `bounce_table.py` and `mixing.py` belong to the three-dimensional rate of `eft_rate/` and assume
one scalar field with an abelian gauge field.
