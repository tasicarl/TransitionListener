"""The compiled potential of a model that supplies its masses and tree level as native functions.

generic_potential calls :func:`build` for a model whose ``compiled_model`` returns ``(tree, bosons, fermions, numbers)``.
The functions are compiled once per process, each on its own and with a cache on disk, and reach the kernels of
compiled.multifield as addresses. The numbers are read from the model at every build, which follows every change of the
model (generic_potential._update_kernels). The stencils of the derivatives are those of helper_functions: on the potential
for the Hessian, on the masses and the tree level for the gradient, which is a chain rule.

A model can make these functions its only definition, with :func:`tree_at`, :func:`bosons_at` and :func:`fermions_at` for
its V0, Vct and mass functions. Tree level, masses and potential are compared with the model's Python methods on a grid
of fields and temperatures before the kernels are used, and a mismatch keeps the model on the Python potential, with a
warning. The entropy degrees of freedom along a phase read the masses at T = 0 of the same functions. Everything else
that reads the spectrum runs in Python on the model's own mass functions, as does the radiation bath, which is added to the
kernels from constantTerms.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import types
from pathlib import Path

import numpy as np
from numba import cfunc

from transitionlistener import helper_functions, thermodynamics
from transitionlistener.compiled import multifield as kernels
from transitionlistener.compiled.onefield import DAISY_ALL_MODES, DAISY_OFF, DAISY_ZERO_MODES
from transitionlistener.compiled.onefield_potential import _shared_tables

_DAISY = {"off": DAISY_OFF, "Parwani": DAISY_ALL_MODES, "ArnoldEspinosa": DAISY_ZERO_MODES}
#: Relative differences above rounding, on the size of each number, that the check refuses.
_SPECTRUM_TOLERANCE, _POTENTIAL_TOLERANCE = 1e-12, 1e-10

#: The order of the finite difference of masses and tree level in the chain rule of kernels._gradient. These are smooth
#: functions of the fields, so that the two points of this order reach the rounding error of the gradient.
_MASS_STENCIL_ORDER = 2

#: The compiled functions of the process, which keep the code behind the addresses in the kernels alive.
_native = {}


def _native_function(function, signature):
    """``function`` compiled for the kernels, once per process. ``function`` may carry an njit decorator.

    numba imports the module of a function when it reads the function back from its cache on disk, and a model file is
    loaded under no importable name, so the copy compiled here belongs to the module of the kernels. Its name holds the
    file, which keeps functions of the same name in two models apart."""
    compiled = _native.get(function)
    if compiled is None:
        code = getattr(function, "py_func", function)
        plain = types.FunctionType(code.__code__, code.__globals__, code.__name__, code.__defaults__)
        plain.__module__ = kernels.__name__
        plain.__qualname__ = f"{Path(code.__code__.co_filename).stem}.{code.__name__}"
        try:
            compiled = cfunc(signature, cache=True, error_model="numpy")(plain)
        except RuntimeError:        # no place for a cache: a function that was not defined in a file
            compiled = cfunc(signature, error_model="numpy")(plain)
        _native[function] = compiled
    return compiled


def _flat(Xa, ndim):
    """The field points as a writable (N, ndim) array in C order, the one type the kernels are compiled for: a row of it
    is what the native functions of a model read through a pointer."""
    X = Xa.reshape(-1, ndim)
    if not (X.flags.c_contiguous and X.flags.writeable):
        X = np.array(X, order="C")
    return X


def _numbers(numbers):
    return np.ascontiguousarray(numbers, dtype=float).reshape(-1)


def tree_at(tree, X, numbers):
    """The native function ``tree`` of a model at the fields X of any shape (..., Ndim)."""
    X = np.asarray(X, dtype=float)
    y = kernels.tree_at(_native_function(tree, kernels.TREE_SIGNATURE).address, _flat(X, X.shape[-1]), _numbers(numbers))
    return y.reshape(X.shape[:-1])[()]


def bosons_at(bosons, count, X, T, numbers):
    """The ``count`` boson masses of the native function ``bosons`` at the fields X and temperatures T, which broadcast
    to the fields, in an array of shape (..., count)."""
    X = np.asarray(X, dtype=float)
    temperature = np.ascontiguousarray(np.broadcast_to(np.asarray(T, dtype=float), X.shape[:-1]), dtype=float)
    m2 = np.empty((temperature.size, count))
    kernels.bosons_at(_native_function(bosons, kernels.BOSONS_SIGNATURE).address, _flat(X, X.shape[-1]),
                      temperature.reshape(-1), _numbers(numbers), m2)
    return m2.reshape(X.shape[:-1] + (count,))


def fermions_at(fermions, count, X, numbers):
    """The ``count`` fermion masses of the native function ``fermions`` at the fields X, of shape (..., count)."""
    X = np.asarray(X, dtype=float)
    flat = _flat(X, X.shape[-1])
    m2 = np.empty((flat.shape[0], count))
    kernels.fermions_at(_native_function(fermions, kernels.FERMIONS_SIGNATURE).address, flat, _numbers(numbers), m2)
    return m2.reshape(X.shape[:-1] + (count,))


class MultiFieldPotential:
    """The kernels of one model, with the calling conventions of generic_potential.

    Each method answers for the inputs it covers and returns None for the others, which then take the generic code.
    """

    #: The energy and entropy density of the model are the Python ones.
    thermal = None

    def __init__(self, table, addresses, ndim, daisy, order, T_eps, gradient, hessian, ghosts):
        self.P = (table, *_shared_tables()["pieces"], addresses)
        self.G = _shared_tables()["G"]
        self.ghosts = ghosts
        self.ndim = ndim
        self.daisy = daisy
        self.order = order
        self.T_eps = float(T_eps)
        self.gdx, self.gcoef = gradient
        self.hdx, self.hcoef, self.hstarts = hessian
        self.radiation = {}

    def _radiation(self, pot, T, include_decoupled):
        key = (T, include_decoupled)
        value = self.radiation.get(key)
        if value is None:
            value = float(np.asarray(pot.constantTerms(np.asanyarray(T), include_decoupled=include_decoupled)))
            if len(self.radiation) > 4096:
                self.radiation.clear()
            self.radiation[key] = value
        return value

    def _points(self, X, Ta):
        """(field points, temperatures per point, shape of the result without the field axis), or None."""
        Xa = X if (type(X) is np.ndarray and X.dtype == np.float64) else np.asarray(X, dtype=float)
        if Xa.ndim < 1 or Xa.shape[-1] != self.ndim or type(Ta) is not np.ndarray or Ta.dtype != np.float64:
            return None
        if Ta.ndim == 0:
            shape = Xa.shape[:-1]
            field = _flat(Xa, self.ndim)
            return field, np.full(field.shape[0], float(Ta)), shape
        shape = np.broadcast_shapes(Xa.shape[:-1], Ta.shape)
        field = _flat(np.broadcast_to(Xa, shape + (self.ndim,)), self.ndim)
        return field, np.array(np.broadcast_to(Ta, shape)).reshape(-1), shape

    def vtot(self, pot, X, T, include_radiation, include_decoupled):
        Ta = np.asanyarray(T)
        points = self._points(X, Ta)
        if points is None:
            return None
        field, temperature, shape = points
        if shape == () and Ta.ndim == 0:
            y = kernels.veff(field[0], float(Ta), self.P, self.daisy)
            if include_radiation:
                y += self._radiation(pot, float(Ta), bool(include_decoupled))
            return np.float64(y)
        y = kernels.vtot_many(field, temperature, self.P, self.daisy).reshape(shape)
        if include_radiation:
            if Ta.ndim == 0:
                y += self._radiation(pot, float(Ta), bool(include_decoupled))
            else:
                y += pot.constantTerms(Ta, include_decoupled=include_decoupled)
        return y

    def gradv(self, X, T):
        points = self._points(X, np.asanyarray(T))
        if points is None:
            return None
        field, temperature, shape = points
        return kernels.gradv_many(field, temperature, self.P, self.daisy, self.gdx, self.gcoef).reshape(shape + (self.ndim,))

    def d2v(self, X, T):
        points = self._points(X, np.asanyarray(T))
        if points is None:
            return None
        field, temperature, shape = points
        out = kernels.d2v_many(field, temperature, self.P, self.daisy, self.hdx, self.hcoef, self.hstarts)
        return out.reshape(shape + (self.ndim, self.ndim))

    def dgradv_dt(self, X, T):
        points = self._points(X, np.asanyarray(T, dtype=float))
        if points is None:
            return None
        field, temperature, shape = points
        out = kernels.dgradv_dt_many(field, temperature, self.T_eps, self.order, self.P, self.daisy, self.gdx, self.gcoef)
        return out.reshape(shape + (self.ndim,))

    def minimum(self, x, T, xtol):
        """optimize.fmin of Vtot(X, T, False, False) from x, or None where the search ends without an answer."""
        ok, found = kernels.nelder_mead(np.array(x, dtype=float, order="C"), float(T), xtol, self.P, self.daisy)
        return found if ok else None

    def newton_minimum(self, X, T, phitol, step_size, origin):
        return None

    def h_eff(self, spline, T):
        """bubbledynamics.h_eff_DS along a phase, with ``spline`` from PhaseInfo.compiled_trace."""
        return kernels.h_eff(T, self.ghosts, *spline, self.P, self.G)


def _hessian_stencils(hessian):
    """The stencils of helper_functions.hessianFunction, one after the other, in the numbering of kernels.d2v_many."""
    dx = [block for row in hessian.dx for block in row]
    coef = [block for row in hessian.coef for block in row]
    return (np.ascontiguousarray(np.concatenate(dx), dtype=float), np.ascontiguousarray(np.concatenate(coef), dtype=float),
            np.cumsum([0] + [len(c) for c in coef], dtype=np.int64))


def _grid(pot, s):
    """Fields and temperatures of the check: the origin, the axes, the vacuum, small and generic points, each at the
    temperatures of a run. Generic points have negative masses squared in a model with a broken phase."""
    ndim = pot.Ndim
    points = [np.zeros(ndim), *(s * np.eye(ndim)), *(-0.7 * s * np.eye(ndim)), np.asarray(pot.X0, dtype=float).reshape(ndim),
              *(s * np.array([1e-5, 1e-2, 1.0]) * np.ones((ndim, 3))).T,
              *(s * np.random.default_rng(0).uniform(-1.7, 1.7, (4, ndim)))]
    temperatures = s * np.array([1e-8, 1e-3, 0.1, 2.0])
    field = np.array(points)
    return (np.repeat(field, temperatures.size, axis=0), np.tile(temperatures, field.shape[0]))


def _close(a, b, size, tolerance):
    return bool(np.all(np.abs(a - b) <= tolerance * size))


def _agrees(compiled, pot, s):
    """The name of what differs between the kernels and the Python potential on the grid of the check, or None."""
    field, temperature = _grid(pot, s)
    python = {"tree level": np.asarray(pot.V0(field) + pot.Vct(field), dtype=float),
              "boson masses": pot.boson_massSq(field, 0.0 * temperature)[0],
              "thermal boson masses": pot.boson_massSq(field, temperature)[0],
              "fermion masses": pot.fermion_massSq(field)[0]}
    if any(np.asarray(m).dtype != np.float64 for m in python.values()):
        return "complex masses"
    table = compiled.P[0]
    nb, nf = int(table[kernels.NBOS]), int(table[kernels.NFER])
    ours = {"tree level": np.empty(field.shape[0]), "boson masses": np.empty((field.shape[0], nb)),
            "thermal boson masses": np.empty((field.shape[0], nb)), "fermion masses": np.empty((field.shape[0], nf))}
    if nf == 0:
        python["fermion masses"] = ours["fermion masses"]
    kernels.spectrum_many(field, temperature, compiled.P, ours["tree level"], ours["boson masses"],
                          ours["thermal boson masses"], ours["fermion masses"])
    for name, value in python.items():
        if value.shape != ours[name].shape:
            return name
        size = np.abs(value) + s**4 if value.ndim == 1 else np.sqrt(np.sum(value * value, axis=-1, keepdims=True))
        if not _close(value, ours[name], size, _SPECTRUM_TOLERANCE):
            return name
    potential = np.asarray(pot._Vtot_generic(field, temperature, False, True), dtype=float)
    ours_potential = kernels.vtot_many(field, temperature, compiled.P, compiled.daisy)
    if not _close(potential, ours_potential, np.abs(potential) + temperature**4 + s**4 * 1e-6, _POTENTIAL_TOLERANCE):
        return "the potential"
    return None


def build(pot, model):
    """The :class:`MultiFieldPotential` of ``pot``. Raises NotImplementedError, saying why, where the kernels would not
    be the model. ``model`` is what ``pot.compiled_model()`` returned."""
    if pot.daisy not in _DAISY:
        raise NotImplementedError(f"no kernel for the daisy scheme {pot.daisy!r}")
    if pot.deriv_order not in (2, 4):
        raise NotImplementedError(f"no kernel for derivatives of order {pot.deriv_order}")
    tree, bosons, fermions, numbers = model
    ndim = pot.Ndim
    X0 = np.asarray(pot.X0, dtype=float).reshape(ndim)
    s = float(np.max(np.abs(X0))) or float(getattr(pot, "v_stable", 0.0) or 0.0)
    if not np.isfinite(s) or s <= 0.0:
        raise NotImplementedError("no field scale")

    spectrum = pot.boson_massSq(X0, 0.0)
    nb, c = np.asarray(spectrum[1], dtype=float).reshape(-1), np.asarray(spectrum[2], dtype=float).reshape(-1)
    nf = np.asarray(pot.fermion_massSq(X0)[1], dtype=float).reshape(-1)
    if fermions is None:
        if np.any(nf != 0.0):
            raise NotImplementedError("the model has fermions and no function for their masses")
        nf = nf[:0]
    pot._debye_coefficient(X0)
    constant = isinstance(pot._debye_state, np.ndarray)
    debye = pot._debye_state.reshape(-1) if constant else np.zeros(nb.size)

    parameters = np.asarray(numbers, dtype=float).reshape(-1)
    limits = _shared_tables()["limits"]
    start = kernels.MODES + kernels.BW * nb.size + nf.size
    table = np.zeros(start + parameters.size)
    table[kernels.NBOS], table[kernels.NFER] = nb.size, nf.size
    table[kernels.MUSQ] = pot.renormScaleSq
    table[kernels.CONSTANT_DEBYE] = constant
    table[kernels.PARAMETERS] = start
    table[kernels.XBMIN:kernels.MODES] = limits
    table[kernels.MODES:kernels.MODES + kernels.BW * nb.size] = np.column_stack((nb, c, debye)).reshape(-1)
    table[start - nf.size:start] = nf
    table[start:] = parameters

    addresses = np.zeros(3, dtype=np.int64)
    addresses[kernels.TREE] = _native_function(tree, kernels.TREE_SIGNATURE).address
    addresses[kernels.BOSONS] = _native_function(bosons, kernels.BOSONS_SIGNATURE).address
    if fermions is not None:
        addresses[kernels.FERMIONS] = _native_function(fermions, kernels.FERMIONS_SIGNATURE).address

    gradient = helper_functions.gradientFunction(None, pot.x_eps, ndim, _MASS_STENCIL_ORDER)
    hessian = helper_functions.hessianFunction(None, pot.x_eps, ndim, pot.deriv_order)
    compiled = MultiFieldPotential(table, addresses, ndim, _DAISY[pot.daisy], pot.deriv_order, pot.T_eps,
                                   (np.ascontiguousarray(gradient.dx, dtype=float),
                                    np.ascontiguousarray(gradient.coef, dtype=float)), _hessian_stencils(hessian),
                                   float(thermodynamics.s_geff(0.0, 1.0, pot.mass_spectrum.number_gauge_bosons, "b")))
    different = _agrees(compiled, pot, s)
    if different is not None:
        raise NotImplementedError(f"compiled_model does not reproduce the Python potential: {different} differ")
    return compiled
