"""The overshoot / undershoot method of tunneling1D.SingleFieldInstanton in numba.

Every shot starts from the exact solution of the linearised equation near r = 0, with Gamma and the Bessel functions
I_nu and J_nu, and for most shots a root search for the radius where the field has left the true vacuum by a given
distance. Those run here for every instanton. The integration of the shot itself, a Cash-Karp step with the step control
of helper_functions.rkqs, needs a compiled dV/dphi, which _force reads from P: either a potential known in closed form or
the spline of a potential along a path. A shot ends in an undershoot or an overshoot, located on the cubic Bezier through
the last step.

Gamma, I_nu and J_nu are scipy.special.cython_special's double versions, the functions behind the ufuncs, through their
C capsules.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import llvmlite.binding as llvm
import numpy as np
from numba import njit, types
from numba.extending import get_cython_function_address

from . import roots

_EPS = np.finfo(float).eps


def _bind(name, nargs):
    # registered with LLVM under a fixed symbol name, so that the compiled code can be cached on disk (a ctypes
    # pointer would bake the address of this process into it)
    symbol = "tl_cython_special" + name
    llvm.add_symbol(symbol, get_cython_function_address("scipy.special.cython_special", name))
    return types.ExternalFunction(symbol, types.float64(*([types.float64] * nargs), types.intc))


_gamma = _bind("__pyx_fuse_1gamma", 1)
_iv = _bind("__pyx_fuse_1iv", 2)
_jv = _bind("__pyx_fuse_1jv", 2)


@njit(cache=True)
def exact_solution(r, phi0, dV, d2V, nu):
    """phi(r) and dphi/dr for a quadratic potential around phi0, see SingleFieldInstanton.exactSolution."""
    beta = np.sqrt(abs(d2V))
    beta_r = beta * r
    if beta_r < 1e-2:
        # the series of the Bessel function at small r
        s = 1 if d2V > 0 else -1
        phi = 0.0
        dphi = 0.0
        for k in range(1, 4):
            t = (0.5 * beta_r) ** (2 * k - 2) * s ** k / (_gamma(k + 1.0, 0) * _gamma(k + 1 + nu, 0))
            phi += t
            dphi += t * (2 * k)
        phi *= 0.25 * _gamma(nu + 1, 0) * r * r * dV * s
        dphi *= 0.25 * _gamma(nu + 1, 0) * r * dV * s
        phi += phi0
    elif d2V > 0:
        p = (0.5 * beta_r) ** -nu
        phi = (_gamma(nu + 1, 0) * p * _iv(nu, beta_r, 0) - 1) * dV / d2V
        dphi = -nu * (p / r) * _iv(nu, beta_r, 0)
        dphi += p * 0.5 * beta * (_iv(nu - 1, beta_r, 0) + _iv(nu + 1, beta_r, 0))
        dphi *= _gamma(nu + 1, 0) * dV / d2V
        phi += phi0
    else:
        p = (0.5 * beta_r) ** -nu
        phi = (_gamma(nu + 1, 0) * p * _jv(nu, beta_r, 0) - 1) * dV / d2V
        dphi = -nu * (p / r) * _jv(nu, beta_r, 0)
        dphi += p * 0.5 * beta * (_jv(nu - 1, beta_r, 0) - _jv(nu + 1, beta_r, 0))
        dphi *= _gamma(nu + 1, 0) * dV / d2V
        phi += phi0
    return phi, dphi


#: initial_conditions: the conditions at rmin or r0, the field moving the wrong way at rmin, and two failures
START_FOUND, WRONG_DIRECTION, START_NOT_A_NUMBER, START_SAME_SIGN = 0, 1, 2, 3


@njit(cache=True)
def initial_conditions(delta_phi0, rmin, cutoff, phi0, dV, d2V, absmin, nu):
    """(status, r0, phi, dphi) of SingleFieldInstanton.initialConditions."""
    phi_r0, dphi_r0 = exact_solution(rmin, phi0, dV, d2V, nu)
    if abs(phi_r0 - absmin) > abs(cutoff):
        return START_FOUND, rmin, phi_r0, dphi_r0
    if np.sign(dphi_r0) != np.sign(delta_phi0):
        return WRONG_DIRECTION, rmin, phi_r0, dphi_r0
    r = rmin
    rlast = r
    while np.isfinite(r):
        rlast = r
        r *= 10
        phi = exact_solution(r, phi0, dV, d2V, nu)[0]
        if abs(phi - absmin) > abs(cutoff):
            break
    state = np.empty(roots.STATE)
    value = abs(exact_solution(r, phi0, dV, d2V, nu)[0] - absmin) - abs(cutoff)
    start = abs(exact_solution(rlast, phi0, dV, d2V, nu)[0] - absmin) - abs(cutoff)
    r0, status = roots.start(state, rlast, start, r, value)
    for _ in range(roots.MAXITER):
        if status != roots.RUNNING:
            break
        r0, status = roots.step(state, value)
        if status == roots.RUNNING:
            value = abs(exact_solution(r0, phi0, dV, d2V, nu)[0] - absmin) - abs(cutoff)
    if status == roots.SAME_SIGN:
        return START_SAME_SIGN, r0, np.nan, np.nan
    if status == roots.NOT_A_NUMBER:
        return START_NOT_A_NUMBER, r0, np.nan, np.nan
    phi_r0, dphi_r0 = exact_solution(r0, phi0, dV, d2V, nu)
    return START_FOUND, r0, phi_r0, dphi_r0


@njit(cache=True)
def _force(x, P):
    """-dV/dphi at x from P = (closed, breakpoints, coefficients).

    With a non-empty ``closed`` this is the potential reflected about its true minimum closed[0], at
    phi = closed[0] - x, with

        dV/dphi = closed[1] phi + closed[2] phi^3 - closed[3] phi |phi| - closed[4] phi sqrt(closed[5] + closed[6] phi^2)
                  + closed[7] (|phi|/closed[8])^closed[9] sign(phi);

    the last term is the wall of an unbounded potential, and closed[7] = 0 means there is none. Otherwise the piecewise
    polynomial with coefficients c[k, i], the highest power first, gives dV/dx itself, extrapolating at the ends."""
    closed, brk, coef = P
    if closed.size > 0:
        p = closed[0] - x
        dv = closed[1] * p + closed[2] * p * p * p - closed[3] * p * abs(p)
        if closed[4] != 0.0:
            dv -= closed[4] * p * np.sqrt(closed[5] + closed[6] * p * p)
        if closed[7] != 0.0:
            dv += closed[7] * (abs(p) / closed[8]) ** closed[9] * np.sign(p)
        return -dv
    n = brk.size - 1
    i = np.searchsorted(brk, x, side="right") - 1
    if i < 0:
        i = 0
    elif i > n - 1:
        i = n - 1
    d = x - brk[i]
    v = coef[0, i]
    for k in range(1, coef.shape[0]):
        v = v * d + coef[k, i]
    return v


@njit(cache=True)
def _step_error(err, y, epsfrac, epsabs):
    absolute = abs(err / epsabs)
    fractional = abs(err) / ((abs(y) + 1e-300) * epsfrac)
    return absolute if absolute < fractional else fractional


@njit(cache=True)
def _cash_karp(P, y0, y1, k0, k1, r0, h, epsfrac, epsabs, alpha):
    """One accepted step of helper_functions.rkqs on phi'' = force(phi) - alpha phi' / r from (y0, y1) with the
    derivatives (k0, k1). Returns (dy0, dy1, h, h_next, ok); ok False where the step size rounds to zero."""
    b21, b31, b32, b41, b42, b43, b51, b52, b53, b54, b61, b62, b63, b64, b65 = (
        0.2, 3.0 / 40.0, 9.0 / 40.0, 0.3, -0.9, 1.2, -11.0 / 54.0, 2.5, -70.0 / 27.0, 35.0 / 27.0,
        1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0)
    c1, c3, c4, c6 = 37.0 / 378.0, 250.0 / 621.0, 125.0 / 594.0, 512.0 / 1771.0
    dc1 = c1 - 2825.0 / 27648.0
    dc3 = c3 - 18575.0 / 48384.0
    dc4 = c4 - 13525.0 / 55296.0
    dc5 = -277.00 / 14336.0
    dc6 = c6 - 0.25
    while True:
        s20 = y1 + b21 * h * k1
        s21 = _force(y0 + b21 * h * k0, P) - alpha * s20 / (r0 + 0.2 * h)
        s30 = y1 + h * (b31 * k1 + b32 * s21)
        s31 = _force(y0 + h * (b31 * k0 + b32 * s20), P) - alpha * s30 / (r0 + 0.3 * h)
        s40 = y1 + h * (b41 * k1 + b42 * s21 + b43 * s31)
        s41 = _force(y0 + h * (b41 * k0 + b42 * s20 + b43 * s30), P) - alpha * s40 / (r0 + 0.6 * h)
        s50 = y1 + h * (b51 * k1 + b52 * s21 + b53 * s31 + b54 * s41)
        s51 = _force(y0 + h * (b51 * k0 + b52 * s20 + b53 * s30 + b54 * s40), P) - alpha * s50 / (r0 + h)
        s60 = y1 + h * (b61 * k1 + b62 * s21 + b63 * s31 + b64 * s41 + b65 * s51)
        s61 = _force(y0 + h * (b61 * k0 + b62 * s20 + b63 * s30 + b64 * s40 + b65 * s50), P) - alpha * s60 / (r0 + 0.875 * h)
        dy0 = h * (c1 * k0 + c3 * s30 + c4 * s40 + c6 * s60)
        dy1 = h * (c1 * k1 + c3 * s31 + c4 * s41 + c6 * s61)
        err0 = h * (dc1 * k0 + dc3 * s30 + dc4 * s40 + dc5 * s50 + dc6 * s60)
        err1 = h * (dc1 * k1 + dc3 * s31 + dc4 * s41 + dc5 * s51 + dc6 * s61)
        s0 = _step_error(err0, y0, epsfrac[0], epsabs[0])
        s1 = _step_error(err1, y1, epsfrac[1], epsabs[1])
        # a nan accepts the step: _step_error_estimate of helper_functions turns it into 0
        errmax = 0.0 if (s0 != s0 or s1 != s1) else max(s0, s1)
        if errmax < 1.0:
            return dy0, dy1, h, (0.9 * h * errmax ** -0.2 if errmax > 1.89e-4 else 5 * h), True
        shrunk = 0.9 * h * errmax ** -0.25
        h = max(shrunk, h * 0.1) if h > 0 else min(shrunk, h * 0.1)
        if r0 + h == r0:
            return dy0, dy1, h, h, False


#: integrate: how a shot ended
SHOT_CONVERGED, SHOT_UNDERSHOOT, SHOT_OVERSHOOT, SHOT_PAST_RMAX, SHOT_STEP_TOO_SMALL, SHOT_STEP_ROUNDS_TO_ZERO = (
    0, 1, 2, -1, -2, -3)


@njit(cache=True)
def integrate(P, r0, y0, y1, dr, epsfrac, epsabs, drmin, rmax, phi_meta, alpha):
    """The loop of SingleFieldInstanton.integrateProfile. Returns (code, r0, y[2], dy/dr[2] at r0, r1, y[2], dy/dr[2] at
    r1, last step); the two points are the ends of the last step."""
    k0 = y1
    k1 = _force(y0, P) - alpha * y1 / r0
    ysign = np.sign(y0 - phi_meta)
    rmax = rmax + r0
    while True:
        dy0, dy1, h, drnext, ok = _cash_karp(P, y0, y1, k0, k1, r0, dr, epsfrac, epsabs, alpha)
        if not ok:
            return SHOT_STEP_ROUNDS_TO_ZERO, r0, y0, y1, k0, k1, 0.0, 0.0, 0.0, 0.0, 0.0, h
        r1 = r0 + h
        a0 = y0 + dy0
        a1 = y1 + dy1
        d0 = a1
        d1 = _force(a0, P) - alpha * a1 / r1
        code = 99
        if r1 > rmax:
            code = SHOT_PAST_RMAX
        elif h < drmin:
            code = SHOT_STEP_TOO_SMALL
        elif abs(a0 - phi_meta) < 3 * epsabs[0] and abs(a1) < 3 * epsabs[1]:
            code = SHOT_CONVERGED
        elif a1 * ysign > 0:
            code = SHOT_UNDERSHOOT
        elif (a0 - phi_meta) * ysign < 0:
            code = SHOT_OVERSHOOT
        if code != 99:
            return code, r0, y0, y1, k0, k1, r1, a0, a1, d0, d1, h
        r0, y0, y1, k0, k1, dr = r1, a0, a1, d0, d1, drnext


@njit(cache=True)
def integrate_and_save(P, R, ya, yb, dr, epsfrac, epsabs, drmin, alpha):
    """The loop of SingleFieldInstanton.integrateAndSaveProfile. Returns (Y, ok)."""
    N = R.size
    Y = np.zeros((N, 2))
    y0, y1 = ya, yb
    Y[0, 0], Y[0, 1] = y0, y1
    r0 = R[0]
    k0 = y1
    k1 = _force(y0, P) - alpha * y1 / r0
    i = 1
    while i < N:
        dy0, dy1, h, hnext, ok = _cash_karp(P, y0, y1, k0, k1, r0, dr, epsfrac, epsabs, alpha)
        if not ok:
            return Y, False
        if h >= drmin:
            a0 = y0 + dy0
            a1 = y1 + dy1
        else:
            a0 = y0 + dy0 * drmin / h
            a1 = y1 + dy1 * drmin / h
            h = hnext = drmin
        r1 = r0 + h
        d0 = a1
        d1 = _force(a0, P) - alpha * a1 / r1
        while i < N and r0 < R[i] <= r1:
            # the cubic Bezier between (y, h dy) at the two ends, as helper_functions.cubicInterpFunction
            x = (R[i] - r0) / h
            Y[i, 0] = _bezier(x, y0, y0 + h * k0 / 3.0, a0 - h * d0 / 3.0, a0)
            Y[i, 1] = _bezier(x, y1, y1 + h * k1 / 3.0, a1 - h * d1 / 3.0, a1)
            i += 1
        r0, y0, y1, k0, k1, dr = r1, a0, a1, d0, d1, hnext
    return Y, True


@njit(cache=True)
def _bezier(t, p0, p1, p2, p3):
    mt = 1 - t
    return p0 * mt * mt * mt + 3 * p1 * mt * mt * t + 3 * p2 * mt * t * t + p3 * t * t * t


@njit(cache=True)
def bezier_root(p0, p1, p2, p3, target):
    """The root in [0, 1] of the cubic Bezier minus target, as (root, status of roots)."""
    state = np.empty(roots.STATE)
    value = p3 - target
    x, status = roots.start(state, 0.0, p0 - target, 1.0, value)
    for _ in range(roots.MAXITER):
        if status != roots.RUNNING:
            break
        x, status = roots.step(state, value)
        if status == roots.RUNNING:
            value = _bezier(x, p0, p1, p2, p3) - target
    return x, status


@njit(cache=True, error_model="numpy")
def interior(R, phi0, dV, d2V, nu):
    """phi and dphi/dr of exact_solution on the radii R[1:], and phi0 and 0 at R[0] = 0."""
    phi = np.empty(R.size)
    dphi = np.empty(R.size)
    phi[0], dphi[0] = phi0, 0.0
    for i in range(1, R.size):
        phi[i], dphi[i] = exact_solution(R[i], phi0, dV, d2V, nu)
    return phi, dphi


@njit(cache=True)
def _spline_slopes(x, brk, coef):
    """dV/dx and d2V/dx2 at x from the piecewise polynomial of dV/dx, extrapolating at the ends."""
    i = np.searchsorted(brk, x, side="right") - 1
    if i < 0:
        i = 0
    elif i > brk.size - 2:
        i = brk.size - 2
    d = x - brk[i]
    value = coef[0, i]
    slope = 0.0
    for k in range(1, coef.shape[0]):
        slope = slope * d + value
        value = value * d + coef[k, i]
    return value, slope


#: bisect: how a call ended, and what the last trial was
TRIAL_FOR_PYTHON, BISECTION_DONE = 0, 1
TRIAL_NONE, TRIAL_CONVERGED, TRIAL_UNDERSHOOT, TRIAL_OVERSHOOT = 0, 1, 2, 3
#: bisect: rows of ``kept``, the trial integrated last on each side and the last one of all, and its columns
KEPT_UNDERSHOOT, KEPT_OVERSHOOT, KEPT_LAST = 0, 1, 2
KEPT_R0, KEPT_PHI, KEPT_DPHI, KEPT_RF, KEPT_EPSABS_PHI, KEPT_EPSABS_DPHI, KEPT_DELTA_PHI0, KEPT_FILLED = range(8)


@njit(cache=True, error_model="numpy")
def _keep(kept, row, r0, phi, dphi, rf, epsabs, delta_phi0):
    kept[row, KEPT_R0] = r0
    kept[row, KEPT_PHI] = phi
    kept[row, KEPT_DPHI] = dphi
    kept[row, KEPT_RF] = rf
    kept[row, KEPT_EPSABS_PHI] = epsabs[0]
    kept[row, KEPT_EPSABS_DPHI] = epsabs[1]
    kept[row, KEPT_DELTA_PHI0] = delta_phi0
    kept[row, KEPT_FILLED] = 1.0


@njit(cache=True, error_model="numpy")
def bisect(P, x, xmin, xmax, xincrease, xtol, absmin, delta_phi, metamin, phi_eps, cutoff, rscale, rmin, dr0, drmin, rmax,
           phitol, min_slope, alpha, kept):
    """The loop of SingleFieldInstanton.findProfile on the spline of P, for as long as its trials are ordinary.

    A trial with no usable slope, which a non-finite x has none of, no start, or a shot that failed changes nothing and
    comes back with TRIAL_FOR_PYTHON, for the caller to take the same trial. Returns (status, x, xmin, xmax, last trial), with the
    trials to keep in ``kept``."""
    brk, coef = P[1], P[2]
    nu = 0.5 * (alpha - 1)
    epsfrac = np.array([phitol, phitol])
    epsabs = np.empty(2)
    last = TRIAL_NONE
    while True:
        delta_phi0 = np.exp(-x) * delta_phi
        phi_scale = max(abs(delta_phi0 - delta_phi), abs(delta_phi) * _EPS)
        epsabs[0] = abs(phi_scale * phitol)
        epsabs[1] = abs(phi_scale / rscale * phitol)
        phi0 = absmin + delta_phi0
        dV, d2V = _spline_slopes(phi0, brk, coef)
        if phi_eps > 0:
            blend = np.exp(-(delta_phi0 / phi_eps) ** 2)
            dV = d2V * delta_phi0 * blend + dV * (1 - blend)
        slope = dV * np.sign(delta_phi0)
        if not (np.isfinite(slope) and slope > min_slope):
            return TRIAL_FOR_PYTHON, x, xmin, xmax, last
        status, r0, phi, dphi = initial_conditions(delta_phi0, rmin, cutoff, phi0, dV, d2V, absmin, nu)
        if (status != START_FOUND and status != WRONG_DIRECTION) or not np.isfinite(r0):
            return TRIAL_FOR_PYTHON, x, xmin, xmax, last
        code, ra, ya0, ya1, da0, da1, rb, yb0, yb1, db0, db1, dr = integrate(
            P, r0, phi, dphi, dr0, epsfrac, epsabs, drmin, rmax, metamin, alpha)
        if code < 0:
            return TRIAL_FOR_PYTHON, x, xmin, xmax, last
        if code == SHOT_CONVERGED:
            rf, yf0, yf1 = rb, yb0, yb1
            trial = TRIAL_CONVERGED
        else:
            p0, p1, p2, p3 = ya0, ya0 + dr * da0 / 3.0, yb0 - dr * db0 / 3.0, yb0
            q0, q1, q2, q3 = ya1, ya1 + dr * da1 / 3.0, yb1 - dr * db1 / 3.0, yb1
            if code == SHOT_UNDERSHOOT:
                root, found = bezier_root(q0, q1, q2, q3, 0.0)
                trial = TRIAL_UNDERSHOOT
            else:
                root, found = bezier_root(p0, p1, p2, p3, metamin)
                trial = TRIAL_OVERSHOOT
            if found != roots.CONVERGED:
                return TRIAL_FOR_PYTHON, x, xmin, xmax, last
            rf = ra + dr * root
            yf0 = _bezier(root, p0, p1, p2, p3)
            yf1 = _bezier(root, q0, q1, q2, q3)
        if abs(yf0 - metamin) < 3 * epsabs[0] and abs(yf1) < 3 * epsabs[1]:
            trial = TRIAL_CONVERGED
        last = trial
        _keep(kept, KEPT_LAST, r0, phi, dphi, rf, epsabs, delta_phi0)
        if trial == TRIAL_CONVERGED:
            return BISECTION_DONE, x, xmin, xmax, last
        if trial == TRIAL_UNDERSHOOT:
            _keep(kept, KEPT_UNDERSHOOT, r0, phi, dphi, rf, epsabs, delta_phi0)
            xmin = x
            x = x * xincrease if xmax == np.inf else 0.5 * (xmin + xmax)
        else:
            _keep(kept, KEPT_OVERSHOOT, r0, phi, dphi, rf, epsabs, delta_phi0)
            xmax = x
            x = 0.5 * (xmin + xmax)
        if x == xmin or x == xmax or (xmax - xmin) < xtol * min(max(abs(x), abs(xmin)), 1.0):
            return BISECTION_DONE, x, xmin, xmax, last
