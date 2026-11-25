"""The one-loop effective potential of any number of fields in numba, from three native functions and a table of numbers.

The functions come from a model, compiled alone with the C signatures

    tree(x, p) -> V0 + Vct        double f(double *x, double *p)
    bosons(x, T, p, m2)           void f(double *x, double T, double *p, double *m2)
    fermions(x, p, m2)            void f(double *x, double *p, double *m2)

and arrive here as addresses, so that the kernels are the same machine code for every model and their cache on disk does
not depend on one. A dispatcher among the arguments would make the kernels depend on the model and miss the cache in
every process. A first-class function argument keeps the cache, but its unboxing costs tens of microseconds per call from
Python; an integer costs nothing.

P is the tuple (C, knots, polynomial pieces and a knot table of Jb, the same of Jf, F). F holds the addresses of the three
functions. The table C holds the numbers listed in the constants below, then n, c and the Debye coefficient of each
boson, then n of each fermion, then the parameters p of the model. V1 and V1T are the expressions of generic_potential:

    V1      sum_b n m^4 (ln|m^2/mu^2| - c) / (64 pi^2) - sum_f n m^4 (ln|m^2/mu^2| - 3/2) / (64 pi^2)
    V1T     sum_i n_i J(m_i^2 / T^2) T^4 / (2 pi^2)
    daisy   none, all modes resummed (Parwani), or the zero modes alone (Arnold and Espinosa), where the thermal part of
            a mass is c T^2 if the model has field-independent Debye coefficients and m(T)^2 - m(0)^2 otherwise.

The gradient is the chain rule, dV/dx = dV0/dx + sum_a dV/dm_a^2 dm_a^2/dx, with J' from the same splines. The derivatives
of tree level and masses are finite differences of the functions of the model, with the stencil the caller passes, and
modes that cross keep the sum over the pair smooth where they have the same n and c, as the two eigenvalues of a block do.

The array ``work`` holds the masses of one evaluation: the bosons at T = 0, the bosons at T, the fermions. The array
``guess`` holds the knot interval of the last evaluation of each mode, bosons first; _guesses starts it.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic

from .onefield import DAISY_ALL_MODES, DAISY_ZERO_MODES, _DAISY_NORM, _V1_NORM, _V1T_NORM, _re_pow15, _s_mode
from .splines import _dpiece, _find_interval, _jpiece

(NBOS, NFER, MUSQ, CONSTANT_DEBYE, PARAMETERS, XBMIN, XBMAX, XFMIN, XFMAX, YBMIN, YFMIN, MODES) = range(12)
BW = 3      # entries per boson
TREE, BOSONS, FERMIONS = 0, 1, 2

#: The C signatures of the functions of a model, for numba.cfunc.
_DOUBLES = types.CPointer(types.double)
TREE_SIGNATURE = types.double(_DOUBLES, _DOUBLES)
BOSONS_SIGNATURE = types.void(_DOUBLES, types.double, _DOUBLES, _DOUBLES)
FERMIONS_SIGNATURE = types.void(_DOUBLES, _DOUBLES, _DOUBLES)


def _native_call(builder, address, result, arguments):
    """Call the function at an integer address. An integer argument is a pointer to doubles, a double stays one."""
    double = ir.DoubleType()
    pointer = double.as_pointer()
    values = [v if v.type == double else builder.inttoptr(v, pointer) for v in arguments]
    function = ir.FunctionType(result, [v.type for v in values])
    return builder.call(builder.inttoptr(address, function.as_pointer()), values)


@intrinsic
def _call_tree(typingctx, address, x, p):
    def codegen(context, builder, signature, args):
        return _native_call(builder, args[0], ir.DoubleType(), args[1:])
    return types.float64(address, x, p), codegen


@intrinsic
def _call_bosons(typingctx, address, x, T, p, m2):
    def codegen(context, builder, signature, args):
        _native_call(builder, args[0], ir.VoidType(), args[1:])
        return context.get_dummy_value()
    return types.none(address, x, T, p, m2), codegen


@intrinsic
def _call_fermions(typingctx, address, x, p, m2):
    def codegen(context, builder, signature, args):
        _native_call(builder, args[0], ir.VoidType(), args[1:])
        return context.get_dummy_value()
    return types.none(address, x, p, m2), codegen


@njit(cache=True, error_model="numpy")
def eigenvalues_2x2(a, b, c):
    """helper_functions.eigenvalues_2x2 at one point: (small, large) of the symmetric matrix [[a, b], [b, c]]."""
    trace = a + c
    discriminant = trace * trace - 4.0 * (a * c - b * b)
    root = np.sqrt(0.0 if discriminant < 0.0 else discriminant)
    return 0.5 * (trace - root), 0.5 * (trace + root)


@njit(cache=True, error_model="numpy")
def _guesses(C):
    return np.full(int(C[NBOS]) + int(C[NFER]), 3)


@njit(cache=True, error_model="numpy")
def _parameters(C):
    """The address of the parameters of the model."""
    return C.ctypes.data + 8 * int(C[PARAMETERS])


@njit(cache=True, error_model="numpy")
def _zero_temperature_masses(x, P, work):
    C, F = P[0], P[7]
    nb = int(C[NBOS])
    p = _parameters(C)
    _call_bosons(F[BOSONS], x.ctypes.data, 0.0, p, work.ctypes.data)
    if C[NFER] > 0.0:
        _call_fermions(F[FERMIONS], x.ctypes.data, p, work.ctypes.data + 16 * nb)


@njit(cache=True, error_model="numpy")
def _thermal_masses(x, T, P, work):
    C, F = P[0], P[7]
    _call_bosons(F[BOSONS], x.ctypes.data, T, _parameters(C), work.ctypes.data + 8 * int(C[NBOS]))


@njit(cache=True, error_model="numpy")
def _vacuum(x, P, work):
    """V0 + Vct + V1, the masses at T = 0 being in work."""
    C, F = P[0], P[7]
    nb, nf = int(C[NBOS]), int(C[NFER])
    mu2 = C[MUSQ]
    s = 0.0
    for i in range(nb):
        b = MODES + i * BW
        m2 = work[i]
        s += C[b] * m2 * m2 * (np.log(abs(m2 / mu2) + 1e-100) - C[b + 1])
    for i in range(nf):
        m2 = work[2 * nb + i]
        s -= C[MODES + nb * BW + i] * m2 * m2 * (np.log(abs(m2 / mu2) + 1e-100) - 1.5)
    return _call_tree(F[TREE], x.ctypes.data, _parameters(C)) + s / _V1_NORM


@njit(cache=True, error_model="numpy")
def _one_loop_thermal(x, T, P, daisy, work, guess):
    """generic_potential.V1T_from_X, the masses at T = 0 being in work."""
    C, tb, pb, tabb, tf, pf, tabf, F = P
    nb, nf = int(C[NBOS]), int(C[NFER])
    start = 0
    if daisy == DAISY_ALL_MODES:
        _thermal_masses(x, T, P, work)
        start = nb
    T2 = T * T + 1e-100
    s = 0.0
    for i in range(nb):
        y, guess[i] = _jpiece(work[start + i] / T2, guess[i], tb, pb, tabb, C[XBMIN], C[XBMAX], C[YBMIN])
        s += C[MODES + i * BW] * y
    for i in range(nf):
        y, guess[nb + i] = _jpiece(work[2 * nb + i] / T2, guess[nb + i], tf, pf, tabf, C[XFMIN], C[XFMAX], C[YFMIN])
        s += C[MODES + nb * BW + i] * y
    return s * ((T * T) * (T * T)) / _V1T_NORM


@njit(cache=True, error_model="numpy")
def _zero_modes(x, T, P, work):
    """generic_potential.Vdaisy with the real part of the difference of the two powers, as daisy_difference_array."""
    C = P[0]
    nb = int(C[NBOS])
    constant = C[CONSTANT_DEBYE] != 0.0
    if not constant:
        _thermal_masses(x, T, P, work)
    T2 = T * T
    w = 0.0
    for i in range(nb):
        b = MODES + i * BW
        a0 = work[i]
        Pi = C[b + 2] * T2 if constant else work[nb + i] - a0
        if Pi == 0.0:
            continue
        if a0 > 0.0 and abs(Pi / a0) < 1.0e-2:      # the difference of the two powers loses its digits
            d = a0 * np.sqrt(a0) * np.expm1(1.5 * np.log1p(Pi / a0))
        else:
            d = _re_pow15(a0 + Pi) - _re_pow15(a0)
        w += C[b] * d
    return -(T / _DAISY_NORM) * w


@njit(cache=True, error_model="numpy")
def _veff(x, T, P, daisy, work, guess):
    """generic_potential.Vtot(X, T, include_radiation=False) at one field point."""
    _zero_temperature_masses(x, P, work)
    y = _vacuum(x, P, work) + _one_loop_thermal(x, T, P, daisy, work, guess)
    if daisy == DAISY_ZERO_MODES:
        y += _zero_modes(x, T, P, work)
    return y


@njit(cache=True, error_model="numpy")
def veff(x, T, P, daisy):
    C = P[0]
    return _veff(x, T, P, daisy, np.empty(2 * int(C[NBOS]) + int(C[NFER])), _guesses(C))


@njit(cache=True, error_model="numpy")
def vtot_many(X, T, P, daisy):
    """Vtot at the points X[n] and the temperatures T[n]."""
    C = P[0]
    work = np.empty(2 * int(C[NBOS]) + int(C[NFER]))
    guess = _guesses(C)
    out = np.empty(X.shape[0])
    for n in range(X.shape[0]):
        out[n] = _veff(X[n], T[n], P, daisy, work, guess)
    return out


@njit(cache=True, error_model="numpy")
def _slopes(T, P, daisy, thermal, work, guess, slope):
    """dV/dm^2 of every entry of work, with V the expression of _veff, or of _one_loop_thermal if thermal. The masses of
    the point are in work: both sets of bosons if _uses_thermal_masses, else those at T = 0."""
    C, tb, pb, tabb, tf, pf, tabf, F = P
    nb, nf = int(C[NBOS]), int(C[NFER])
    mu2 = C[MUSQ]
    slope[:] = 0.0
    if not thermal:
        for i in range(nb):
            b = MODES + i * BW
            m2 = work[i]
            slope[i] = C[b] * m2 * (2.0 * (np.log(abs(m2 / mu2) + 1e-100) - C[b + 1]) + 1.0) / _V1_NORM
        for i in range(nf):
            m2 = work[2 * nb + i]
            slope[2 * nb + i] = -C[MODES + nb * BW + i] * m2 * (2.0 * (np.log(abs(m2 / mu2) + 1e-100) - 1.5) + 1.0) / _V1_NORM
    T2 = T * T + 1e-100
    weight = ((T * T) * (T * T)) / (_V1T_NORM * T2)
    start = nb if daisy == DAISY_ALL_MODES else 0
    for i in range(nb):
        y, guess[i] = _dpiece(work[start + i] / T2, guess[i], tb, pb, tabb, C[XBMIN], C[XBMAX])
        slope[start + i] += weight * C[MODES + i * BW] * y
    for i in range(nf):
        y, guess[nb + i] = _dpiece(work[2 * nb + i] / T2, guess[nb + i], tf, pf, tabf, C[XFMIN], C[XFMAX])
        slope[2 * nb + i] += weight * C[MODES + nb * BW + i] * y
    if daisy == DAISY_ZERO_MODES and not thermal:
        constant = C[CONSTANT_DEBYE] != 0.0
        for i in range(nb):
            b = MODES + i * BW
            a0 = work[i]
            Pi = C[b + 2] * T * T if constant else work[nb + i] - a0
            if Pi == 0.0:
                continue
            weight = -1.5 * (T / _DAISY_NORM) * C[b]
            if a0 > 0.0:
                slope[i] -= weight * np.sqrt(a0)
            top = a0 + Pi
            if top > 0.0:
                slope[nb + i if not constant else i] += weight * np.sqrt(top)


@njit(cache=True, error_model="numpy")
def _uses_thermal_masses(C, daisy):
    return daisy == DAISY_ALL_MODES or (daisy == DAISY_ZERO_MODES and C[CONSTANT_DEBYE] == 0.0)


@njit(cache=True, error_model="numpy")
def _gradient(x, T, P, daisy, dx, coef, thermal, y, work, guess, slope, out):
    """gradientFunction of Vtot, or of V1T_from_X if thermal, at one field point, by the chain rule: dV/dm^2 at x times the
    stencil of the masses, plus the stencil of the tree level. Only the masses are differenced, on the stencil of the
    gradient, and the spline of J and the logarithm are read once."""
    C, F = P[0], P[7]
    masses_at_T = _uses_thermal_masses(C, daisy) and not (thermal and daisy != DAISY_ALL_MODES)
    p = _parameters(C)
    nb = int(C[NBOS])
    _zero_temperature_masses(x, P, work)
    if masses_at_T:
        _thermal_masses(x, T, P, work)
    else:
        work[nb:2 * nb] = 0.0
    _slopes(T, P, daisy, thermal, work, guess, slope)
    ndim = x.size
    for j in range(ndim):
        g = 0.0
        for o in range(dx.shape[0]):
            for k in range(ndim):
                y[k] = x[k] + dx[o, j, k]
            _zero_temperature_masses(y, P, work)
            if masses_at_T:
                _thermal_masses(y, T, P, work)
            s = 0.0 if thermal else _call_tree(F[TREE], y.ctypes.data, p)
            for i in range(slope.size):
                s += slope[i] * work[i]
            g += s * coef[o, j]
        out[j] = g


@njit(cache=True, error_model="numpy")
def gradv_many(X, T, P, daisy, dx, coef):
    """generic_potential.gradV of X[n] at T[n]."""
    C = P[0]
    work = np.empty(2 * int(C[NBOS]) + int(C[NFER]))
    slope = np.empty(work.size)
    guess = _guesses(C)
    y = np.empty(X.shape[1])
    out = np.empty(X.shape)
    for n in range(X.shape[0]):
        _gradient(X[n], T[n], P, daisy, dx, coef, False, y, work, guess, slope, out[n])
    return out


@njit(cache=True, error_model="numpy")
def d2v_many(X, T, P, daisy, dx, coef, starts):
    """generic_potential.d2V of X[n] at T[n]. The pair (i, j <= i) of derivatives has the stencil number
    i (i + 1) / 2 + j, at the rows starts[number] .. starts[number + 1] of dx and coef."""
    C = P[0]
    ndim = X.shape[1]
    work = np.empty(2 * int(C[NBOS]) + int(C[NFER]))
    guess = _guesses(C)
    y = np.empty(ndim)
    out = np.empty((X.shape[0], ndim, ndim))
    for n in range(X.shape[0]):
        x = X[n]
        number = 0
        for i in range(ndim):
            for j in range(i + 1):
                h = 0.0
                for row in range(starts[number], starts[number + 1]):
                    for k in range(ndim):
                        y[k] = x[k] + dx[row, k]
                    h += _veff(y, T[n], P, daisy, work, guess) * coef[row]
                out[n, i, j] = h
                out[n, j, i] = h
                number += 1
    return out


@njit(cache=True, error_model="numpy")
def dgradv_dt_many(X, T, e, order, P, daisy, dx, coef):
    """generic_potential.dgradV_dT of X[n] at T[n]."""
    C = P[0]
    ndim = X.shape[1]
    work = np.empty(2 * int(C[NBOS]) + int(C[NFER]))
    slope = np.empty(work.size)
    guess = _guesses(C)
    y = np.empty(ndim)
    a = np.empty(ndim)
    b = np.empty(ndim)
    c = np.empty(ndim)
    d = np.empty(ndim)
    out = np.empty(X.shape)
    for n in range(X.shape[0]):
        x = X[n]
        if order == 2:
            _gradient(x, T[n] + e, P, daisy, dx, coef, True, y, work, guess, slope, a)
            _gradient(x, T[n] - e, P, daisy, dx, coef, True, y, work, guess, slope, b)
            for j in range(ndim):
                out[n, j] = (a[j] - b[j]) * (1.0 / (2 * e))
        else:
            _gradient(x, T[n] - 2 * e, P, daisy, dx, coef, True, y, work, guess, slope, a)
            _gradient(x, T[n] - e, P, daisy, dx, coef, True, y, work, guess, slope, b)
            _gradient(x, T[n] + e, P, daisy, dx, coef, True, y, work, guess, slope, c)
            _gradient(x, T[n] + 2 * e, P, daisy, dx, coef, True, y, work, guess, slope, d)
            for j in range(ndim):
                out[n, j] = (a[j] - 8 * b[j] + 8 * c[j] - d[j]) * (1.0 / (12 * e))
    return out


@njit(cache=True, error_model="numpy")
def _sort_simplex(sim, fsim, row):
    """Sort the simplex by value, a stable insertion sort."""
    for i in range(1, fsim.size):
        value = fsim[i]
        row[:] = sim[i]
        j = i - 1
        while j >= 0 and fsim[j] > value:
            fsim[j + 1] = fsim[j]
            sim[j + 1] = sim[j]
            j -= 1
        fsim[j + 1] = value
        sim[j + 1] = row


@njit(cache=True, error_model="numpy")
def nelder_mead(x0, T, xatol, P, daisy):
    """scipy's _minimize_neldermead with ftol = inf and the defaults of optimize.fmin, on _veff. Returns (ok, x); ok is
    False where scipy's budget of 200 N evaluations or iterations is reached or a value is not finite, so that the caller
    can take scipy's answer for those."""
    C = P[0]
    N = x0.size
    work = np.empty(2 * int(C[NBOS]) + int(C[NFER]))
    guess = _guesses(C)
    sim = np.empty((N + 1, N))
    fsim = np.empty(N + 1)
    xbar, xr, xe, row = np.empty(N), np.empty(N), np.empty(N), np.empty(N)
    budget = 200 * N
    sim[0] = x0
    for k in range(N):
        sim[k + 1] = x0
        sim[k + 1, k] = 1.05 * x0[k] if x0[k] != 0.0 else 0.00025
    for k in range(N + 1):
        fsim[k] = _veff(sim[k], T, P, daisy, work, guess)
    calls = N + 1
    iterations = 1
    _sort_simplex(sim, fsim, row)
    while calls < budget and iterations < budget:
        if not np.all(np.isfinite(fsim)):
            return False, sim[0]
        if np.max(np.abs(sim[1:] - sim[0])) <= xatol:
            return True, sim[0]
        xbar[:] = 0.0
        for k in range(N):
            xbar += sim[k]
        xbar /= N
        xr[:] = 2 * xbar - sim[N]
        fr = _veff(xr, T, P, daisy, work, guess)
        calls += 1
        if fr < fsim[0]:
            xe[:] = 3 * xbar - 2 * sim[N]
            fe = _veff(xe, T, P, daisy, work, guess)
            calls += 1
            if fe < fr:
                sim[N] = xe
                fsim[N] = fe
            else:
                sim[N] = xr
                fsim[N] = fr
        elif fr < fsim[N - 1]:
            sim[N] = xr
            fsim[N] = fr
        else:
            shrink = False
            if fr < fsim[N]:
                xe[:] = 1.5 * xbar - 0.5 * sim[N]
                fc = _veff(xe, T, P, daisy, work, guess)
                calls += 1
                if fc <= fr:
                    sim[N] = xe
                    fsim[N] = fc
                else:
                    shrink = True
            else:
                xe[:] = 0.5 * xbar + 0.5 * sim[N]
                fc = _veff(xe, T, P, daisy, work, guess)
                calls += 1
                if fc < fsim[N]:
                    sim[N] = xe
                    fsim[N] = fc
                else:
                    shrink = True
            if shrink:
                for j in range(1, N + 1):
                    sim[j] = sim[0] + 0.5 * (sim[j] - sim[0])
                    fsim[j] = _veff(sim[j], T, P, daisy, work, guess)
                calls += N
        iterations += 1
        _sort_simplex(sim, fsim, row)
    return False, sim[0]


@njit(cache=True, error_model="numpy")
def tree_at(address, X, p):
    """A model's tree function at the points X[n], with its numbers p."""
    y = np.empty(X.shape[0])
    for n in range(X.shape[0]):
        x = X[n]
        y[n] = _call_tree(address, x.ctypes.data, p.ctypes.data)
    return y


@njit(cache=True, error_model="numpy")
def bosons_at(address, X, T, p, m2):
    """A model's boson masses at the points X[n] and temperatures T[n], into m2[n], with its numbers p."""
    for n in range(X.shape[0]):
        x = X[n]
        row = m2[n]
        _call_bosons(address, x.ctypes.data, T[n], p.ctypes.data, row.ctypes.data)


@njit(cache=True, error_model="numpy")
def fermions_at(address, X, p, m2):
    """A model's fermion masses at the points X[n], into m2[n], with its numbers p."""
    for n in range(X.shape[0]):
        x = X[n]
        row = m2[n]
        _call_fermions(address, x.ctypes.data, p.ctypes.data, row.ctypes.data)


@njit(cache=True, error_model="numpy")
def spectrum_many(X, T, P, tree, m2b, m2T, m2f):
    """What the model supplies at X[n]: tree level, bosons at T = 0 and at T[n], fermions, for the check against Python."""
    C, F = P[0], P[7]
    nb, nf = int(C[NBOS]), int(C[NFER])
    work = np.empty(2 * nb + nf)
    for n in range(X.shape[0]):
        x = X[n]
        _zero_temperature_masses(x, P, work)
        _thermal_masses(x, T[n], P, work)
        tree[n] = _call_tree(F[TREE], x.ctypes.data, _parameters(C))
        for i in range(nb):
            m2b[n, i] = work[i]
            m2T[n, i] = work[nb + i]
        for i in range(nf):
            m2f[n, i] = work[2 * nb + i]


@njit(cache=True, error_model="numpy")
def h_eff(T, ghosts, c, xp, P, G):
    """bubbledynamics.h_eff_DS at the field of a phase: the piecewise cubics of the fields are c[:, i, field] on xp."""
    teb, ceb, xe0, xe1, tpb, cpb, xp0, xp1, tef, cef, xf0, xf1, tpf, cpf, xq0, xq1 = G
    C = P[0]
    nb, nf = int(C[NBOS]), int(C[NFER])
    i = _find_interval(xp, T, True)
    d = T - xp[i]
    x = ((c[0, i] * d + c[1, i]) * d + c[2, i]) * d + c[3, i]
    work = np.empty(2 * nb + nf)
    _zero_temperature_masses(x, P, work)
    total = 0.0
    for k in range(nb):
        m = np.sqrt(max(work[k], 0.0))
        total += _s_mode(m, T, C[MODES + k * BW], 1.0, teb, ceb, xe0, xe1, tpb, cpb, xp0, xp1)
    for k in range(nf):
        m = np.sqrt(max(work[2 * nb + k], 0.0))
        total += _s_mode(m, T, C[MODES + nb * BW + k], 7.0 / 8.0, tef, cef, xf0, xf1, tpf, cpf, xq0, xq1)
    r = total - ghosts
    return r if r > 0.0 else 0.0
