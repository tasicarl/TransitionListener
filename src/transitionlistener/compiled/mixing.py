"""One partial wave of a coupled 3 x 3 fluctuation determinant in numba (see abelian_determinants.log_gy_mixed).

_chunk is scipy's Radau (scipy/integrate/_ivp/radau.py) line by line, with the same step control and the same tolerances,
on the linear system Y' = A(r) Y with the analytic A(r); _log_det adds the chunks and the QR between them. The stage
matrices are 6 x 6 per solution column. The functions in A(r) are read from piecewise cubics, see
_coefficients.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np
from numba import njit

_CF_TERMS = 400        # as abelian_determinants._iv_log_deriv

_S6 = 6.0 ** 0.5
_C = np.array([(4.0 - _S6) / 10.0, (4.0 + _S6) / 10.0, 1.0])
_E = np.array([-13.0 - 7.0 * _S6, -13.0 + 7.0 * _S6, -1.0]) / 3.0
_MU_REAL = 3.0 + 3.0 ** (2.0 / 3.0) - 3.0 ** (1.0 / 3.0)
_MU_COMPLEX = (3.0 + 0.5 * (3.0 ** (1.0 / 3.0) - 3.0 ** (2.0 / 3.0))
               - 0.5j * (3.0 ** (5.0 / 6.0) + 3.0 ** (7.0 / 6.0)))
_T = np.array([[0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
               [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
               [1.0, 1.0, 0.0]])
_TI = np.array([[4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
                [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
                [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
_TI_REAL = _TI[0].copy()
_TI_COMPLEX = _TI[1] + 1j * _TI[2]
_P = np.array([[13.0 / 3.0 + 7.0 * _S6 / 3.0, -23.0 / 3.0 - 22.0 * _S6 / 3.0, 10.0 / 3.0 + 5.0 * _S6],
               [13.0 / 3.0 - 7.0 * _S6 / 3.0, -23.0 / 3.0 + 22.0 * _S6 / 3.0, 10.0 / 3.0 - 5.0 * _S6],
               [1.0 / 3.0, -8.0 / 3.0, 10.0 / 3.0]])
_NEWTON_MAXITER = 6
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0

OK, FAILED_STEP, FAILED_VALUE, FAILED_DEPENDENT, FAILED_DET = 0, 1, 2, 3, 4


@njit(cache=True)
def _horner(c, i, d):
    return ((c[0, i] * d + c[1, i]) * d + c[2, i]) * d + c[3, i]


@njit(cache=True)
def _coefficients(r, x0, ix, n, wc, pc, dwc, ix2, n2, hc, qc, cm, cp, l, out):
    """out = [w, a_SG, a_LG, a_GS, a_GL, w + dWG, U_S, U_L, U_G]: the matrix A(r) of fluctuations._mixed_rhs_core."""
    x = np.log(r)
    u = (x - x0) * ix
    i = int(u)
    if i < 0:
        i = 0
    elif i > n - 2:
        i = n - 2
    d = x - (x0 + i / ix)
    w = _horner(wc, i, d)
    p = _horner(pc, i, d)
    dw = _horner(dwc, i, d)
    u = (x - x0) * ix2
    i = int(u)
    if i < 0:
        i = 0
    elif i > n2 - 2:
        i = n2 - 2
    d = x - (x0 + i / ix2)
    hh = np.exp(_horner(hc, i, d))
    q = _horner(qc, i, d)
    out[0] = w
    out[1] = 2.0 * cm * p * (r * hh)
    out[2] = -2.0 * cp * p * (hh / r)
    out[3] = 2.0 * cm * p * (1.0 / (r * hh))
    out[4] = -2.0 * cp * p * (r / hh)
    out[5] = dw + w
    out[6] = 2.0 * l / r
    out[7] = 2.0 * (l + 2.0) / r
    out[8] = (1.0 + 2.0 * q) / r


@njit(cache=True)
def _rhs(k, Y, F):
    """F = A Y for the 3 solution columns; returns False if an entry is not finite."""
    ok = True
    for c in range(3):
        F[0, c] = Y[3, c]
        F[1, c] = Y[4, c]
        F[2, c] = Y[5, c]
        F[3, c] = -k[6] * Y[3, c] + k[0] * Y[0, c] + k[1] * Y[2, c]
        F[4, c] = -k[7] * Y[4, c] + k[0] * Y[1, c] + k[2] * Y[2, c]
        F[5, c] = -k[8] * Y[5, c] + k[5] * Y[2, c] + k[3] * Y[0, c] + k[4] * Y[1, c]
        for j in range(6):
            if not np.isfinite(F[j, c]):
                ok = False
    return ok


@njit(cache=True)
def _lu(k, mu, M, piv):
    """LU with partial pivoting of mu I - A(k), real or complex by the type of mu and M."""
    M[:, :] = 0.0
    for j in range(6):
        M[j, j] = mu
    M[0, 3] = M[1, 4] = M[2, 5] = -1.0
    M[3, 3] += k[6]
    M[4, 4] += k[7]
    M[5, 5] += k[8]
    M[3, 0] = -k[0]
    M[3, 2] = -k[1]
    M[4, 1] = -k[0]
    M[4, 2] = -k[2]
    M[5, 2] = -k[5]
    M[5, 0] = -k[3]
    M[5, 1] = -k[4]
    for j in range(6):
        p = j
        big = abs(M[j, j])
        for i in range(j + 1, 6):
            if abs(M[i, j]) > big:
                big = abs(M[i, j])
                p = i
        piv[j] = p
        if p != j:
            for c in range(6):
                M[j, c], M[p, c] = M[p, c], M[j, c]
        for i in range(j + 1, 6):
            f = M[i, j] / M[j, j]
            M[i, j] = f
            if f != 0.0:
                for c in range(j + 1, 6):
                    M[i, c] -= f * M[j, c]


@njit(cache=True)
def _solve(M, piv, b):
    """b <- M^-1 b, column by column; M, piv from _lu_real or _lu_complex."""
    for c in range(3):
        for j in range(6):              # the rows of M were swapped whole, so the swaps come first
            p = piv[j]
            if p != j:
                b[j, c], b[p, c] = b[p, c], b[j, c]
        for j in range(6):
            for i in range(j + 1, 6):
                b[i, c] -= M[i, j] * b[j, c]
        for j in range(5, -1, -1):
            s = b[j, c]
            for i in range(j + 1, 6):
                s -= M[j, i] * b[i, c]
            b[j, c] = s / M[j, j]


@njit(cache=True)
def _norm(a, s):
    t = 0.0
    for j in range(6):
        for c in range(3):
            q = a[j, c] / s[j, c]
            t += q * q
    return np.sqrt(t / 18.0)


@njit(cache=True)
def _predict_factor(h_abs, h_abs_old, error_norm, error_norm_old):
    """h_abs_old < 0 stands for scipy's None."""
    if error_norm_old < 0.0 or h_abs_old < 0.0 or error_norm == 0.0:
        multiplier = 1.0
    else:
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25
    if error_norm == 0.0:
        return np.inf
    return min(1.0, multiplier) * error_norm ** -0.25


@njit(cache=True)
def _chunk(y, a, b, rtol, atol, x0, ix, n, wc, pc, dwc, ix2, n2, hc, qc, cm, cp, l, C, E, T, TI, TI_REAL, TI_COMPLEX,
           P, MU_REAL, MU_COMPLEX, stats):
    """solve_ivp(rhs, (a, b), y, method="Radau", rtol=rtol, atol=atol) for the matrix y (6, 3), in place.
    Returns 0, or 1 if the step fell below the spacing of r. Line by line scipy's Radau._step_impl and
    solve_collocation_system."""
    newton_tol = max(10.0 * 2.220446049250313e-16 / rtol, min(0.03, rtol ** 0.5))
    k = np.empty(9)
    args = (x0, ix, n, wc, pc, dwc, ix2, n2, hc, qc, cm, cp, l)
    f = np.empty((6, 3))
    t = a
    _coefficients(t, *args, k)
    _rhs(k, y, f)

    # select_initial_step
    interval = abs(b - a)
    scale = atol + np.abs(y) * rtol
    d0 = _norm(y, scale)
    d1 = _norm(f, scale)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    h0 = min(h0, interval)
    y1 = y + h0 * f
    k1 = np.empty(9)
    _coefficients(t + h0, *args, k1)
    f1 = np.empty((6, 3))
    _rhs(k1, y1, f1)
    d2 = _norm(f1 - f, scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.25
    h_abs = min(100.0 * h0, h1, interval)

    h_abs_old = -1.0
    error_norm_old = -1.0
    kj = k.copy()                       # the Jacobian: A at the point it was taken
    current_jac = True
    have_lu = False
    MR = np.empty((6, 6))
    MC = np.empty((6, 6), dtype=np.complex128)
    pr = np.empty(6, dtype=np.int64)
    pi = np.empty(6, dtype=np.int64)
    Z = np.zeros((3, 6, 3))
    W = np.zeros((3, 6, 3))
    dW = np.zeros((3, 6, 3))
    Fs = np.zeros((3, 6, 3))
    Ks = np.empty((3, 9))
    Q = np.zeros((3, 6, 3))
    y_old = y.copy()
    has_sol = False
    h_sol = 1.0
    t_old = t
    Yt = np.empty((6, 3))
    Fi = np.empty((6, 3))
    dreal = np.empty((6, 3))
    dcplx = np.empty((6, 3), dtype=np.complex128)
    err = np.empty((6, 3))
    ZE = np.empty((6, 3))
    dn_s = np.empty((6, 3))

    while True:
        if t >= b:
            return 0
        min_step = 10.0 * abs(np.nextafter(t, np.inf) - t)
        h_abs_entry = h_abs             # scipy's self.h_abs, which becomes h_abs_old below
        h_abs_prev, err_prev = h_abs_old, error_norm_old
        if h_abs < min_step:
            h_abs = min_step
            h_abs_prev = -1.0
            err_prev = -1.0
        rejected = False
        accepted = False
        n_iter = 0
        rate = -1.0
        error_norm = 0.0
        safety = 0.0
        t_new = t
        y_new = y
        h = h_abs
        while not accepted:
            if not h_abs >= min_step:   # also for a nan, which would otherwise never leave this loop
                return 1
            h = h_abs
            t_new = t + h
            if t_new - b > 0.0:
                t_new = b
            h = t_new - t
            h_abs = abs(h)
            for i in range(3):
                Z[i] = 0.0
            if has_sol:
                for i in range(3):
                    xx = ((t + h * C[i]) - t_old) / h_sol
                    p1, p2, p3 = xx, xx * xx, xx * xx * xx
                    Z[i] = Q[0] * p1 + Q[1] * p2 + Q[2] * p3 + y_old - y
            scale = atol + np.abs(y) * rtol
            for i in range(3):
                kt = Ks[i]
                _coefficients(t + h * C[i], *args, kt)

            converged = False
            while not converged:
                if not have_lu:
                    _lu(kj, MU_REAL / h, MR, pr)
                    _lu(kj, MU_COMPLEX / h, MC, pi)
                    stats[2] += 1
                    have_lu = True
                # solve_collocation_system
                M_real = MU_REAL / h
                M_complex = MU_COMPLEX / h
                Z0 = Z
                for i in range(3):
                    W[i] = TI[i, 0] * Z0[0] + TI[i, 1] * Z0[1] + TI[i, 2] * Z0[2]
                Zn = Z0.copy()
                dW_norm_old = -1.0
                rate = -1.0
                nk = 0
                conv = False
                for it in range(_NEWTON_MAXITER):
                    nk = it + 1
                    finite = True
                    for i in range(3):
                        Yt[:, :] = y + Zn[i]
                        F_i = Fs[i]
                        if not _rhs(Ks[i], Yt, F_i):
                            finite = False
                    stats[1] += 3
                    if not finite:
                        break
                    for j in range(6):
                        for c in range(3):
                            dreal[j, c] = (Fs[0, j, c] * TI_REAL[0] + Fs[1, j, c] * TI_REAL[1]
                                           + Fs[2, j, c] * TI_REAL[2] - M_real * W[0, j, c])
                            dcplx[j, c] = (Fs[0, j, c] * TI_COMPLEX[0] + Fs[1, j, c] * TI_COMPLEX[1]
                                           + Fs[2, j, c] * TI_COMPLEX[2]
                                           - M_complex * (W[1, j, c] + 1j * W[2, j, c]))
                    _solve(MR, pr, dreal)
                    _solve(MC, pi, dcplx)
                    dW[0] = dreal
                    dW[1] = dcplx.real
                    dW[2] = dcplx.imag
                    tt = 0.0
                    for i in range(3):
                        for j in range(6):
                            for c in range(3):
                                q = dW[i, j, c] / scale[j, c]
                                tt += q * q
                    dW_norm = np.sqrt(tt / 54.0)
                    if dW_norm_old >= 0.0:
                        rate = dW_norm / dW_norm_old
                    if rate >= 0.0 and (rate >= 1.0
                                        or rate ** (_NEWTON_MAXITER - it) / (1.0 - rate) * dW_norm > newton_tol):
                        break
                    W += dW
                    for i in range(3):
                        Zn[i] = T[i, 0] * W[0] + T[i, 1] * W[1] + T[i, 2] * W[2]
                    if dW_norm == 0.0 or (rate >= 0.0 and rate / (1.0 - rate) * dW_norm < newton_tol):
                        conv = True
                        break
                    dW_norm_old = dW_norm
                converged = conv
                n_iter = nk
                if not converged:
                    if current_jac:
                        break
                    _coefficients(t, *args, kj)
                    stats[3] += 1
                    current_jac = True
                    have_lu = False
                else:
                    Z[:] = Zn
            if not converged:
                h_abs *= 0.5
                have_lu = False
                continue
            y_new = y + Z[2]
            for j in range(6):
                for c in range(3):
                    ZE[j, c] = (Z[0, j, c] * E[0] + Z[1, j, c] * E[1] + Z[2, j, c] * E[2]) / h
            err[:, :] = f + ZE
            _solve(MR, pr, err)
            for j in range(6):
                for c in range(3):
                    dn_s[j, c] = atol + max(abs(y[j, c]), abs(y_new[j, c])) * rtol
            error_norm = _norm(err, dn_s)
            safety = 0.9 * (2.0 * _NEWTON_MAXITER + 1.0) / (2.0 * _NEWTON_MAXITER + n_iter)
            if rejected and error_norm > 1.0:
                Yt[:, :] = y + err
                _rhs(k, Yt, Fi)
                stats[1] += 1
                err[:, :] = Fi + ZE
                _solve(MR, pr, err)
                error_norm = _norm(err, dn_s)
            if error_norm > 1.0:
                factor = _predict_factor(h_abs, h_abs_prev, error_norm, err_prev)
                h_abs *= max(_MIN_FACTOR, safety * factor)
                have_lu = False
                rejected = True
            else:
                accepted = True

        recompute_jac = n_iter > 2 and rate > 1e-3
        factor = _predict_factor(h_abs, h_abs_prev, error_norm, err_prev)
        factor = min(_MAX_FACTOR, safety * factor)
        if not recompute_jac and factor < 1.2:
            factor = 1.0
        else:
            have_lu = False
        _coefficients(t_new, *args, k)
        _rhs(k, y_new, f)
        if recompute_jac:
            kj[:] = k
            stats[3] += 1
            current_jac = True
        else:
            current_jac = False
        h_abs_old = h_abs_entry
        error_norm_old = error_norm
        h_abs = h_abs * factor
        y_old[:, :] = y
        t_old = t
        h_sol = t_new - t
        t = t_new
        y[:, :] = y_new
        for i in range(3):
            Q[i] = Z[0] * P[0, i] + Z[1] * P[1, i] + Z[2] * P[2, i]
        has_sol = True
        stats[0] += 1


@njit(cache=True)
def _log_det(edges, rtol, atol, x0, ix, n, wc, pc, dwc, ix2, n2, hc, qc, l, C, E, T, TI, TI_REAL, TI_COMPLEX, P,
             MU_REAL, MU_COMPLEX, stats):
    """The chunks and the QR of abelian_determinants.log_gy_mixed. Returns (code, ln|det|, index of the failing chunk)."""
    cm = np.sqrt(l / (2.0 * l + 1.0))
    cp = np.sqrt((l + 1.0) / (2.0 * l + 1.0))
    Y = np.zeros((6, 3))
    for j in range(3):
        Y[j, j] = 1.0
    log_scale = 0.0
    for e in range(edges.size - 1):
        code = _chunk(Y, edges[e], edges[e + 1], rtol, atol, x0, ix, n, wc, pc, dwc, ix2, n2, hc, qc, cm, cp, l,
                      C, E, T, TI, TI_REAL, TI_COMPLEX, P, MU_REAL, MU_COMPLEX, stats)
        if code != 0:
            return FAILED_STEP, np.nan, e
        for j in range(6):
            for c in range(3):
                if not np.isfinite(Y[j, c]):
                    return FAILED_VALUE, np.nan, e
        Q, R = np.linalg.qr(Y)
        for j in range(3):
            d = R[j, j]
            if d == 0.0:
                return FAILED_DEPENDENT, np.nan, e
            s = 1.0 if d > 0.0 else -1.0
            for c in range(6):
                Y[c, j] = Q[c, j] * s
            log_scale += np.log(abs(d))
    det_scaled = np.linalg.det(Y[:3].copy())
    if det_scaled == 0.0:
        return FAILED_DET, np.nan, edges.size - 2
    return OK, np.log(abs(det_scaled)) + log_scale, -1


@njit(cache=True)
def _q_nodes(nu, z):
    """z I'_nu(z)/I_nu(z) at the nodes, abelian_determinants._iv_log_deriv times z."""
    out = np.empty(z.size)
    for i in range(z.size):
        frac = 0.0
        zi = z[i]
        for k in range(_CF_TERMS, 0, -1):
            frac = 1.0 / (2.0 * (nu + k) / zi + frac)
        out[i] = zi * frac + nu
    return out
