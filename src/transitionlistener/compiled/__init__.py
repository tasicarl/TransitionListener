"""Compiled kernels of TransitionListener: numba functions on numbers and arrays.

    splines       FITPACK, BSpline and PPoly at one point; a PCHIP interpolant with a compiled scalar call
    roots         scipy's brentq
    shooting      the start of a 1D shot and the integrator of the overshoot / undershoot method
    bounce_table  the interpolant of the table of the leading-order 3D action
    mixing        one partial wave of a coupled 3 x 3 fluctuation determinant
    percolation   the percolation integral on every prefix of a grid and the checks on the support of the solver
    onefield      a one-field effective potential, its derivatives and thermodynamics, from a table of numbers
    multifield    an effective potential of any number of fields and its derivatives, from native mass functions
    deformation   the arithmetic of one step of the deformation of a path
    onefield_potential, multifield_potential   read the tables and functions of these kernels from a model

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
