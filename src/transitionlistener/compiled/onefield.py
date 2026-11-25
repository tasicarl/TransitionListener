"""The one-loop effective potential of one field in numba, from a table of numbers.

    tree level     a polynomial in phi up to phi^6;
    boson masses   m^2 = a0 + a1 phi + a2 phi^2 + cT T^2 for each mode;
    fermion masses m^2 = a0 + a1 phi + a2 phi^2;
    V1             sum_i (+-) n_i M_i^4 (ln|M_i^2| - k_i) / (64 pi^2), with M_i^2 = m_i^2 + d_i, where d_i is the
                   resummation of the Goldstone bosons, generic_potential.setGoldstoneShift;
    V1T            sum_i n_i J(m_i^2 / T^2) T^4 / (2 pi^2), with the Jb and Jf splines of finiteT;
    daisy          none, all modes resummed (Parwani), or the zero modes alone (Arnold and Espinosa).

The table C holds, in this order, the number of bosons and fermions, the polynomial p0..p6, the ranges of the thermal
functions and then the modes: a0, a1, a2, cT, n, k, d per boson and a0, a1, a2, n, k per fermion. P is the tuple
(C, knots and coefficients of Jb and a knot table, the same of Jf). onefield_potential reads the table from a model.

The second half of the file is the thermodynamics along a traced phase: energy and entropy density, from the same table,
the masses at their zero-temperature values, the tables of thermodynamics and the piecewise polynomial (cp, xp) of the
phase.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np
from numba import njit

from . import roots
from .splines import _bspline, _djspline, _jspline, _ppeval

DAISY_OFF, DAISY_ALL_MODES, DAISY_ZERO_MODES = 0, 1, 2

(NBOS, NFER, POLY, XBMIN, XBMAX, XFMIN, XFMAX, YBMIN, YFMIN, MODES) = (0, 1, 2, 9, 10, 11, 12, 13, 14, 15)
BW = 7      # entries per boson
FW = 5      # entries per fermion

_V1_NORM = 64.0 * np.pi * np.pi
_V1T_NORM = 2.0 * np.pi * np.pi
_DAISY_NORM = 12.0 * np.pi


@njit(cache=True, error_model="numpy")
def _mass(C, b, phi):
    """a0 + a1 phi + a2 phi^2 of the mode whose entries start at b."""
    return C[b] + phi * (C[b + 1] + phi * C[b + 2])


@njit(cache=True, error_model="numpy")
def _polynomial(C, phi):
    y = 0.0
    for j in range(6, -1, -1):
        y = y * phi + C[POLY + j]
    return y


@njit(cache=True, error_model="numpy")
def _vacuum(phi, C):
    """The tree-level polynomial and V1 at one field value."""
    nb = int(C[NBOS])
    s = 0.0
    for i in range(nb):
        b = MODES + i * BW
        m2 = _mass(C, b, phi) + C[b + 6]
        if m2 != 0.0:
            s += C[b + 4] * m2 * m2 * (np.log(abs(m2)) - C[b + 5])
    for i in range(int(C[NFER])):
        b = MODES + nb * BW + i * FW
        m2 = _mass(C, b, phi)
        if m2 != 0.0:
            s -= C[b + 3] * m2 * m2 * (np.log(abs(m2)) - C[b + 4])
    return _polynomial(C, phi) + s / _V1_NORM


@njit(cache=True, error_model="numpy")
def _thermal(phi, T, P, daisy):
    """generic_potential.V1T_from_X at one field value."""
    C, tb, cb, tabb, tf, cf, tabf = P
    nb = int(C[NBOS])
    T2 = T * T + 1e-100
    s = 0.0
    for i in range(nb):
        b = MODES + i * BW
        m2 = _mass(C, b, phi)
        if daisy == DAISY_ALL_MODES:
            m2 += C[b + 3] * (T * T)
        s += C[b + 4] * _jspline(m2 / T2, tb, cb, tabb, C[XBMIN], C[XBMAX], C[YBMIN])
    for i in range(int(C[NFER])):
        b = MODES + nb * BW + i * FW
        s += C[b + 3] * _jspline(_mass(C, b, phi) / T2, tf, cf, tabf, C[XFMIN], C[XFMAX], C[YFMIN])
    return s * ((T * T) * (T * T)) / _V1T_NORM


@njit(cache=True, error_model="numpy")
def _re_pow15(x):
    """The real part of (x + 0j)**1.5. For x < 0 that is |x|^1.5 cos(3 pi / 2) = 0."""
    if x > 0.0:
        return x * np.sqrt(x)
    return 0.0 if x <= 0.0 else x


@njit(cache=True, error_model="numpy")
def daisy_difference_array(m2, Pi, out):
    """The real part of (m^2 + Pi)^(3/2) - (m^2)^(3/2) for every entry."""
    for i in range(m2.size):
        a0 = m2[i]
        if a0 > 0.0 and abs(Pi[i] / a0) < 1.0e-2:
            out[i] = a0 * np.sqrt(a0) * np.expm1(1.5 * np.log1p(Pi[i] / a0))
        else:
            out[i] = _re_pow15(a0 + Pi[i]) - _re_pow15(a0)


@njit(cache=True, error_model="numpy")
def veff(phi, T, P, daisy):
    """generic_potential.Vtot(X, T, include_radiation=False) at one field value."""
    C = P[0]
    y = _vacuum(phi, C) + _thermal(phi, T, P, daisy)
    if daisy == DAISY_ZERO_MODES:
        T2 = T * T
        w = 0.0
        for i in range(int(C[NBOS])):
            b = MODES + i * BW
            a0 = _mass(C, b, phi)
            Pi = C[b + 3] * T2
            if a0 > 0.0 and abs(Pi / a0) < 1.0e-2:      # the difference of the two powers loses its digits
                d = a0 * np.sqrt(a0) * np.expm1(1.5 * np.log1p(Pi / a0))
            else:
                d = _re_pow15(a0 + Pi) - _re_pow15(a0)
            w += C[b + 4] * d
        y += -(T / _DAISY_NORM) * w
    return y


@njit(cache=True, error_model="numpy")
def vtot_many(X, T, P, daisy):
    out = np.empty(X.shape[0])
    for n in range(X.shape[0]):
        out[n] = veff(X[n], T[n], P, daisy)
    return out


@njit(cache=True, error_model="numpy")
def _derivative(x, T, P, daisy, dx, coef):
    g = 0.0
    for o in range(dx.shape[0]):
        g += veff(x + dx[o], T, P, daisy) * coef[o]
    return g


@njit(cache=True, error_model="numpy")
def gradv_many(X, T, P, daisy, dx, coef):
    out = np.empty(X.shape[0])
    for n in range(X.shape[0]):
        out[n] = _derivative(X[n], T, P, daisy, dx, coef)
    return out


@njit(cache=True, error_model="numpy")
def d2v(x, T, P, daisy, dx, coef):
    """generic_potential.d2V of one field point."""
    return _derivative(x, T, P, daisy, dx, coef)


@njit(cache=True, error_model="numpy")
def _gradv1t(x, T, P, daisy, dx, coef):
    g = 0.0
    for o in range(dx.shape[0]):
        g += _thermal(x + dx[o], T, P, daisy) * coef[o]
    return g


@njit(cache=True, error_model="numpy")
def dgradv_dt_many(X, T, e, order, P, daisy, dx, coef):
    """generic_potential.dgradV_dT of X[n] at T[n]."""
    out = np.empty(X.shape[0])
    for n in range(X.shape[0]):
        if order == 2:
            out[n] = (_gradv1t(X[n], T[n] + e, P, daisy, dx, coef)
                      - _gradv1t(X[n], T[n] - e, P, daisy, dx, coef)) * (1.0 / (2 * e))
        else:
            out[n] = (_gradv1t(X[n], T[n] - 2 * e, P, daisy, dx, coef) - 8 * _gradv1t(X[n], T[n] - e, P, daisy, dx, coef)
                      + 8 * _gradv1t(X[n], T[n] + e, P, daisy, dx, coef)
                      - _gradv1t(X[n], T[n] + 2 * e, P, daisy, dx, coef)) * (1.0 / (12 * e))
    return out


@njit(cache=True, error_model="numpy")
def nelder_mead(x0, T, xatol, P, daisy):
    """scipy's _minimize_neldermead for one variable with ftol = inf and the defaults of optimize.fmin, on veff. Returns
    (ok, x); ok False, where scipy's budget of 200 evaluations or iterations is reached or a value is not finite."""
    x1 = 1.05 * x0 if x0 != 0.0 else 0.00025
    f0 = veff(x0, T, P, daisy)
    f1 = veff(x1, T, P, daisy)
    calls = 2
    iterations = 1
    if f1 < f0:
        x0, x1 = x1, x0
        f0, f1 = f1, f0
    while calls < 200 and iterations < 200:
        if not (np.isfinite(f0) and np.isfinite(f1)):
            return False, x0
        if abs(x1 - x0) <= xatol:
            return True, x0
        xr = 2 * x0 - 1 * x1
        fr = veff(xr, T, P, daisy)
        calls += 1
        if fr < f0:
            xe = 3 * x0 - 2 * x1
            fe = veff(xe, T, P, daisy)
            calls += 1
            if fe < fr:
                x1, f1 = xe, fe
            else:
                x1, f1 = xr, fr
        else:
            shrink = False
            if fr < f1:
                xc = 1.5 * x0 - 0.5 * x1
                fc = veff(xc, T, P, daisy)
                calls += 1
                if fc <= fr:
                    x1, f1 = xc, fc
                else:
                    shrink = True
            else:
                xc = 0.5 * x0 + 0.5 * x1
                fc = veff(xc, T, P, daisy)
                calls += 1
                if fc < f1:
                    x1, f1 = xc, fc
                else:
                    shrink = True
            if shrink:
                x1 = x0 + 0.5 * (x1 - x0)
                f1 = veff(x1, T, P, daisy)
                calls += 1
        iterations += 1
        if f1 < f0:
            x0, x1 = x1, x0
            f0, f1 = f1, f0
    return False, x0


@njit(cache=True, error_model="numpy")
def newton_minimum(x, T, P, daisy, hdx, hcoef, gdx, gcoef, phitol, step_size, mass, om, ostep):
    """phases.findLocalMinimum for one field. With om >= 0, where mass |phi| < om T, the first and second derivative
    are central differences of veff with the step ostep T."""
    niter = 0
    while True:
        niter += 1
        if om >= 0.0 and T > 0.0 and mass * abs(x) < om * T:
            h = ostep * T
            vp = veff(x + h, T, P, daisy)
            vm = veff(x - h, T, P, daisy)
            H = (vp + vm - 2.0 * veff(x, T, P, daisy)) / (h * h)
            g = (vp - vm) / (2.0 * h)
        else:
            H = _derivative(x, T, P, daisy, hdx, hcoef)
            g = _derivative(x, T, P, daisy, gdx, gcoef)
        if H == 0.0:
            break
        eps = g / (H + 1e-5)
        x = x - eps * step_size
        if abs(eps) <= phitol:
            break
        if niter >= 200:
            break
    return x


# ------------------------------------------------------------------------------------------- along a traced phase
@njit(cache=True, error_model="numpy")
def _dof_entry(x, x0, x1, low, t, c):
    """e_geff / p_geff per degree of freedom: the massless limit below the table, 0 above it."""
    if x < x0:
        return low
    if x0 <= x < x1:
        return _bspline(t, c, x)
    return 0.0


@njit(cache=True, error_model="numpy")
def _s_mode(m, T, dof, low, teb, ceb, xe0, xe1, tpb, cpb, xp0, xp1):
    """thermodynamics.s_geff for scalars."""
    x = m / (T + 1e-100)
    e = _dof_entry(x, xe0, xe1, low, teb, ceb) * dof
    p = _dof_entry(x, xp0, xp1, low, tpb, cpb) * dof
    return (3.0 * e + p) / 4.0


@njit(cache=True, error_model="numpy")
def h_eff(T, ghosts, cp, xp, C, G):
    """bubbledynamics.h_eff_DS."""
    teb, ceb, xe0, xe1, tpb, cpb, xp0, xp1, tef, cef, xf0, xf1, tpf, cpf, xq0, xq1 = G
    phi = _ppeval(cp, xp, T, True)
    nb = int(C[NBOS])
    total = 0.0
    for i in range(nb):
        k = MODES + i * BW
        m = np.sqrt(max(_mass(C, k, phi), 0.0))
        total = total + _s_mode(m, T, C[k + 4], 1.0, teb, ceb, xe0, xe1, tpb, cpb, xp0, xp1)
    for i in range(int(C[NFER])):
        k = MODES + nb * BW + i * FW
        m = np.sqrt(max(_mass(C, k, phi), 0.0))
        total = total + _s_mode(m, T, C[k + 3], 7.0 / 8.0, tef, cef, xf0, xf1, tpf, cpf, xq0, xq1)
    r = total - ghosts
    return r if r > 0.0 else 0.0


@njit(cache=True, error_model="numpy")
def energy_density(phi, T, coupled, use_table, lg, decoupled, Ng, V0ref, ce, xe, C, J):
    """generic_potential.energyDensity for a scalar T != 0 (radiationEnergyDensity inlined). decoupled is the value of the
    decoupled bath, 0.0 when it is not included; coupled that of the coupled bath, unless use_table."""
    tb, cb, tabb, wb, tf, cf, tabf, wf, dmin = J
    v0 = _vacuum(phi, C)
    dv0 = v0 - V0ref
    if dv0 / abs(v0) < 1e-10:
        dv0 = 0.0
    T2 = T * T
    nb = int(C[NBOS])
    vt = 0.0
    for i in range(nb):
        k = MODES + i * BW
        m2 = _mass(C, k, phi)
        x = m2 / T2
        jv = _jspline(x, tb, cb, tabb, C[XBMIN], C[XBMAX], C[YBMIN])
        dj = _djspline(x, tb, wb, tabb, C[XBMIN], C[XBMAX], dmin[0])
        vt += -3.0 * C[k + 4] * T2 * T2 * jv + 2.0 * m2 * C[k + 4] * T2 * dj
    for i in range(int(C[NFER])):
        k = MODES + nb * BW + i * FW
        m2 = _mass(C, k, phi)
        x = m2 / T2
        jv = _jspline(x, tf, cf, tabf, C[XFMIN], C[XFMAX], C[YFMIN])
        dj = _djspline(x, tf, wf, tabf, C[XFMIN], C[XFMAX], dmin[1])
        vt += -3.0 * C[k + 3] * T2 * T2 * jv + 2.0 * m2 * C[k + 3] * T2 * dj
    geff = _ppeval(ce, xe, lg, True) if use_table else coupled
    geff = geff - Ng + decoupled
    return dv0 + vt / _V1T_NORM + np.pi * np.pi / 30.0 * geff * (T2 * T2)


@njit(cache=True, error_model="numpy")
def energy_density_on_phase(T, coupled, use_table, lg, decoupled, Ng, V0ref, cp, xp, ce, xe, C, J):
    """bubbledynamics.energyDensity: energy_density at the field of the phase."""
    return energy_density(_ppeval(cp, xp, T, True), T, coupled, use_table, lg, decoupled, Ng, V0ref, ce, xe, C, J)


@njit(cache=True, error_model="numpy")
def entropy(t, ghosts, cp, xp, C, G, cs0, cs1, CF):
    """bubbledynamics.entropy_density(pot, phase, t, "dof_table") on the tabulated Standard Model bath."""
    return (h_eff(t, ghosts, cp, xp, C, G) + _ppeval(cs0, cs1, np.log10(t * CF), True)) * (t * t * t)


@njit(cache=True, error_model="numpy")
def entropy_and_sound_speed(T, Tmin, Tmax, rel, ghosts, cp, xp, C, G, cs0, cs1, CF, cs_out, entropy_out):
    """The loop of bubbledynamics._time_temperature_factors. Returns the number of usable temperatures."""
    usable = 0
    for i in range(T.size):
        t = T[i]
        here = entropy(t, ghosts, cp, xp, C, G, cs0, cs1, CF)
        h = t * rel
        t_lo = t - h if t - h > Tmin else Tmin
        t_hi = t + h if t + h < Tmax else Tmax
        s_lo = entropy(t_lo, ghosts, cp, xp, C, G, cs0, cs1, CF)
        s_hi = entropy(t_hi, ghosts, cp, xp, C, G, cs0, cs1, CF)
        cs = np.nan
        if s_lo > 0.0 and s_hi > 0.0 and t_hi > t_lo:
            cs = (np.log(t_hi) - np.log(t_lo)) / (np.log(s_hi) - np.log(s_lo))
        cs_out[i] = 1.0 / 3.0
        entropy_out[i] = np.nan
        if np.isfinite(cs) and 0.0 < cs < 1.0 and np.isfinite(here) and here > 0.0:
            cs_out[i] = cs
            entropy_out[i] = here
            usable += 1
    return usable


@njit(cache=True, error_model="numpy")
def _criterion(x, kind, c0, c1, c2, c3, Ng, V0ref, ghosts, cp, xp, ce0, ce1, cs0, cs1, C, J, G, CF):
    """Tb_criterion (0), energy_criterion_BRO (1), entropy_criterion_SYM_BRO (2) of bubbledynamics, with what does not
    depend on x in c."""
    if kind == 2:
        return entropy(x, ghosts, cp, xp, C, G, cs0, cs1, CF) - c0
    e = energy_density_on_phase(x, 0.0, True, np.log10(x * CF), 0.0, Ng, V0ref, cp, xp, ce0, ce1, C, J)
    if kind == 0:
        return e / c0 - 1
    return e * c2 - (c1 * (c2 - c3) + c3 * c0)


@njit(cache=True, error_model="numpy")
def criterion_root(a, b, kind, c0, c1, c2, c3, Ng, V0ref, ghosts, cp, xp, ce0, ce1, cs0, cs1, C, J, G, CF):
    """The root of _criterion in [a, b], as (root, status of roots)."""
    state = np.empty(roots.STATE)
    value = _criterion(b, kind, c0, c1, c2, c3, Ng, V0ref, ghosts, cp, xp, ce0, ce1, cs0, cs1, C, J, G, CF)
    first = _criterion(a, kind, c0, c1, c2, c3, Ng, V0ref, ghosts, cp, xp, ce0, ce1, cs0, cs1, C, J, G, CF)
    x, status = roots.start(state, a, first, b, value)
    for _ in range(roots.MAXITER):
        if status != roots.RUNNING:
            break
        x, status = roots.step(state, value)
        if status == roots.RUNNING:
            value = _criterion(x, kind, c0, c1, c2, c3, Ng, V0ref, ghosts, cp, xp, ce0, ce1, cs0, cs1, C, J, G, CF)
    return x, status
