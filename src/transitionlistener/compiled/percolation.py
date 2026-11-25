"""The percolation integral and the bookkeeping of the adaptive solver in numba.

percIntegralODE_full_sweep needs the double integral on every prefix T[:i+1] of the temperature grid. Every quantity per
support point (the logarithm of the source, S/T, ln H, ln a) is the same in all prefixes, since only ratios to T[0]
enter; only the peak of the source, and with it the refined intervals and the spline nodes, change. _subgrid does for
one prefix what bubbledynamics._double_integral_subgrid describes, on those numbers, and the double integral follows on
the sub-grid. The splines are CubicSpline (not-a-knot) and, on fewer than four nodes, PchipInterpolator of scipy.

percIntegralODE solves the chain of four integrals J_n with chain_integral: scipy's DOP853 (solve_ivp with dense output
and events), line by line, on the two interpolants of the logarithms of the source and of the transport factor. The
right-hand side is _chain_rhs.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np
from numba import njit
from scipy.integrate._ivp import dop853_coefficients as _dop853

from . import roots
from .splines import _find_interval

_STAGES = _dop853.N_STAGES
_A = _dop853.A[:_STAGES + 4]
_B = _dop853.B
_C = _dop853.C[:_STAGES + 4]
_E3 = _dop853.E3
_E5 = _dop853.E5
_D = _dop853.D
_SAFETY, _MIN_FACTOR, _MAX_FACTOR, _ERROR_EXPONENT = 0.9, 0.2, 10.0, -1.0 / 8.0

#: What chain_integral returns as its status: the cold end, the stop value of the integral, or a step below the spacing
#: of the floating-point numbers.
CHAIN_FINISHED, CHAIN_STOPPED, CHAIN_FAILED = 0, 1, -1


@njit(cache=True, error_model="numpy")
def _plan(n, ls, finite, change, want, T, u, peak, negligible, cap, steps, nodes, capped):
    """The sub-steps of each interval of the prefix of n support points, and the spline nodes. peak is the largest
    finite log_source in it. Returns the number of nodes, 0 without refinement. An interval that the cap on sub-steps
    limits is counted in capped: [intervals of the worst prefix, largest change of the logarithm of the source]."""
    any_refined = False
    first = -1
    last = -1
    bound = peak - negligible
    limited = 0
    for k in range(n - 1):
        m = 1
        if finite[k] and finite[k + 1] and T[k] > T[k + 1]:
            top = ls[k] if ls[k] >= ls[k + 1] else ls[k + 1]
            if top > bound:
                w = want[k]
                if w > cap:
                    limited += 1
                    w = cap
                    capped[1] = max(capped[1], change[k])
                m = int(w) if w >= 1.0 else 1
        steps[k] = m
        if m > 1:
            if not any_refined:
                first = k
            last = k
            any_refined = True
    capped[0] = max(capped[0], limited)
    if not any_refined:
        return 0
    lo = first - 1 if first > 0 else 0
    hi = last + 2 if last + 2 < n - 1 else n - 1
    nn = 0
    ascending = True
    for j in range(hi, lo - 1, -1):      # ascending in u on a descending grid, the order of np.unique
        if finite[j]:
            nodes[nn] = j
            if nn > 0 and not u[j] > u[nodes[nn - 1]]:
                ascending = False
            nn += 1
    if not ascending:                    # a grid out of order or with repeated temperatures: np.unique, by hand
        order = np.argsort(u[nodes[:nn]], kind="mergesort")
        taken = nodes[:nn].copy()
        kept = 0
        for i in order:
            if kept > 0 and u[taken[i]] == u[nodes[kept - 1]]:
                nodes[kept - 1] = taken[i]
            else:
                nodes[kept] = taken[i]
                kept += 1
        nn = kept
    return nn


@njit(cache=True, error_model="numpy")
def _cubic(c, r, i, d):
    """Row r of the cubic of interval i at the distance d from its left end."""
    return ((c[r, 0, i] * d + c[r, 1, i]) * d + c[r, 2, i]) * d + c[r, 3, i]


@njit(cache=True, error_model="numpy")
def _system(nn, nodes, u, y, dl, d, du, b):
    """The tridiagonal system of CubicSpline (not-a-knot at both ends) on the nodes, for the 4 rows of y (support
    index in the columns): x and slopes as in CubicSpline.__init__, fills dl, d, du (length nn-1, nn, nn-1), b."""
    x = u[nodes[:nn]]
    dx = x[1:] - x[:-1]
    for i in range(nn - 2):
        d[i + 1] = 2 * (dx[i] + dx[i + 1])
        du[i + 1] = dx[i]
        dl[i] = dx[i + 1]
    d[0] = dx[1]
    du[0] = x[2] - x[0]
    d0 = x[2] - x[0]
    d[nn - 1] = dx[nn - 3]
    dl[nn - 2] = x[nn - 1] - x[nn - 3]
    d1 = x[nn - 1] - x[nn - 3]
    p0 = dx[0] * dx[0]
    p1 = dx[nn - 2] * dx[nn - 2]
    slope = np.empty((4, nn - 1))
    for r in range(4):
        for i in range(nn - 1):
            slope[r, i] = (y[r, nodes[i + 1]] - y[r, nodes[i]]) / dx[i]
        for i in range(nn - 2):
            b[r, i + 1] = 3 * (dx[i + 1] * slope[r, i] + dx[i] * slope[r, i + 1])
        b[r, 0] = ((dx[0] + 2 * d0) * dx[1] * slope[r, 0] + p0 * slope[r, 1]) / d0
        b[r, nn - 1] = (p1 * slope[r, nn - 3] + (2 * d1 + dx[nn - 2]) * dx[nn - 3] * slope[r, nn - 2]) / d1
    return slope



@njit(cache=True, error_model="numpy")
def _spline_on_steps(n, nn, nodes, u, y, s, slope, steps, sT, cool, uo, ho, ao, so, co):
    """CubicHermiteSpline's coefficients from the slopes s, and the interior points of every refined interval:
    u (np.linspace), lnH and ln a (spline), S/T and 3 c_s^2 (spline, held between the two support values, as
    _clamp_between). Returns the number of interior points."""
    x = u[nodes[:nn]]
    c = np.empty((4, 4, nn - 1))
    for r in range(4):
        for i in range(nn - 1):
            dxi = x[i + 1] - x[i]
            t = (s[r, i] + s[r, i + 1] - 2 * slope[r, i]) / dxi
            c[r, 0, i] = t / dxi
            c[r, 1, i] = (slope[r, i] - s[r, i]) / dxi - t
            c[r, 2, i] = s[r, i]
            c[r, 3, i] = y[r, nodes[i]]
    q = 0
    for k in range(n - 1):
        m = steps[k]
        if m <= 1:
            continue
        start = u[k]
        step = (u[k + 1] - start) / m
        a0 = sT[k]
        a1 = sT[k + 1]
        lo_s = a1 if a1 < a0 else a0
        hi_s = a1 if a0 < a1 else a0     # python's min(a, b) / max(a, b)
        b0 = cool[k]
        b1 = cool[k + 1]
        lo_c = b1 if b1 < b0 else b0
        hi_c = b1 if b0 < b1 else b0
        for j in range(1, m):
            v = j * step + start
            i = _find_interval(x, v, True)
            d = v - x[i]
            uo[q] = v
            so[q] = _cubic(c, 0, i, d)
            ho[q] = _cubic(c, 1, i, d)
            co[q] = _cubic(c, 2, i, d)
            ao[q] = _cubic(c, 3, i, d)
            so[q] = so[q] if so[q] <= hi_s else hi_s
            so[q] = so[q] if so[q] >= lo_s else lo_s
            co[q] = co[q] if co[q] <= hi_c else hi_c
            co[q] = co[q] if co[q] >= lo_c else lo_c
            q += 1
    return q



@njit(cache=True, error_model="numpy")
def _assemble(n, T, H, S, cool, scale, steps, q, Tk, Hk, so, co, ak, To, Ho, So, Co, Ao):
    """The support points of the prefix with the q interior points between them. Returns the length."""
    w = 0
    p = 0
    for k in range(n - 1):
        To[w] = T[k]
        Ho[w] = H[k]
        So[w] = S[k]
        Co[w] = cool[k]
        Ao[w] = scale[k]
        w += 1
        if q > 0:
            for _ in range(steps[k] - 1):
                To[w] = Tk[p]
                Ho[w] = Hk[p]
                So[w] = so[p] * Tk[p]
                Co[w] = co[p]
                Ao[w] = ak[p]
                w += 1
                p += 1
    To[w] = T[n - 1]
    Ho[w] = H[n - 1]
    So[w] = S[n - 1]
    Co[w] = cool[n - 1]
    Ao[w] = scale[n - 1]
    return w + 1



@njit(cache=True, error_model="numpy")
def _transport(M, To, Ho, Co, Ao, base, radius, dT):
    """a(T') (int_T^T' dT'' base/a) on the sub-grid, from the cold end, in the ratios of neighbouring scale factors."""
    for i in range(M):
        base[i] = 1.0 / (Co[i] * Ho[i] * To[i])
    radius[M - 1] = 0.0
    for i in range(M - 2, -1, -1):
        ratio = Ao[i] / Ao[i + 1]
        d = To[i] - To[i + 1]
        dT[i] = d
        step = 0.5 * (base[i] + base[i + 1] * ratio) * d
        radius[i] = ratio * radius[i + 1] + step


@njit(cache=True, error_model="numpy")
def _pchip_edge(h0, h1, m0, m1):
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(d) > 3.0 * abs(m0):
        return 3.0 * m0
    return d


@njit(cache=True, error_model="numpy")
def _pchip_slopes(nn, nodes, u, y, s):
    """The derivatives of PchipInterpolator on two or three nodes, for _spline_on_steps. Returns the secant slopes."""
    slope = np.empty((4, nn - 1))
    x = u[nodes[:nn]]
    for r in range(4):
        for i in range(nn - 1):
            slope[r, i] = (y[r, nodes[i + 1]] - y[r, nodes[i]]) / (x[i + 1] - x[i])
        if nn == 2:
            s[r, 0] = slope[r, 0]
            s[r, 1] = slope[r, 0]
            continue
        h0 = x[1] - x[0]
        h1 = x[2] - x[1]
        m0 = slope[r, 0]
        m1 = slope[r, 1]
        s[r, 1] = 0.0
        if np.sign(m0) == np.sign(m1) and m0 != 0.0 and m1 != 0.0:
            w1 = 2 * h1 + h0
            w2 = h1 + 2 * h0
            s[r, 1] = 1.0 / ((w1 / m0 + w2 / m1) / (w1 + w2))
        s[r, 0] = _pchip_edge(h0, h1, m0, m1)
        s[r, 2] = _pchip_edge(h1, h0, m1, m0)
    return slope



@njit(cache=True, error_model="numpy")
def _gtsv(n, dl, d, du, b, s):
    """LAPACK dgtsv (partial pivoting) on the 4 right-hand sides in the rows of b; the slopes go to s. Overwrites
    dl, d, du, b. The matrix of a spline on strictly ascending nodes has no vanishing pivot."""
    for i in range(n - 1):
        if abs(d[i]) >= abs(dl[i]):
            fact = dl[i] / d[i]
            d[i + 1] -= fact * du[i]
            for r in range(4):
                b[r, i + 1] -= fact * b[r, i]
            if i < n - 2:
                dl[i] = 0.0
        else:
            fact = d[i] / dl[i]
            d[i] = dl[i]
            temp = d[i + 1]
            d[i + 1] = du[i] - fact * temp
            if i < n - 2:
                dl[i] = du[i + 1]
                du[i + 1] = -fact * dl[i]
            du[i] = temp
            for r in range(4):
                temp = b[r, i]
                b[r, i] = b[r, i + 1]
                b[r, i + 1] = temp - fact * b[r, i + 1]
    for r in range(4):
        s[r, n - 1] = b[r, n - 1] / d[n - 1]
        s[r, n - 2] = (b[r, n - 2] - du[n - 2] * s[r, n - 1]) / d[n - 2]
        for j in range(n - 3, -1, -1):
            s[r, j] = (b[r, j] - du[j] * s[r, j + 1] - dl[j] * s[r, j + 2]) / d[j]



@njit(cache=True, error_model="numpy")
def _integrate(M, To, So, base, radius, dT):
    """The trapezoid of np.nan_to_num(Gamma * base * radius**3), with bubbledynamics.Gamma."""
    big = np.finfo(np.float64).max
    total = 0.0
    previous = 0.0
    for i in range(M):
        t = To[i]
        a = So[i]
        if a == 0.0:
            rate = np.inf
        elif np.isinf(a):
            rate = np.nan
        else:
            root = np.sqrt(a / (2 * np.pi * t))
            rate = t * t * t * t * (root * root * root) * np.exp(-a / t)
        r = radius[i]
        value = rate * base[i] * (r * r * r)
        if np.isnan(value):
            value = 0.0
        elif value == np.inf:
            value = big
        elif value == -np.inf:
            value = -big
        if i > 0:
            total += 0.5 * (previous + value) * dT[i - 1]
        previous = value
    return total



@njit(cache=True, error_model="numpy")
def _workspace(N, size):
    """The arrays of _subgrid for N support points and at most `size` interior points."""
    return (np.ones(N - 1, dtype=np.int64), np.zeros(N, dtype=np.int64), np.empty(N), np.empty(N), np.empty(N),
            np.empty((4, N)), np.empty((4, N)), np.empty(size), np.empty(size), np.empty(size), np.empty(size),
            np.empty(size), np.empty(size), np.empty(size), np.empty(size), np.empty(size + N), np.empty(size + N),
            np.empty(size + N), np.empty(size + N), np.empty(size + N))


@njit(cache=True, error_model="numpy")
def _subgrid(n, count, peak, T, H, S, cool, scale, ls, finite, change, want, u, y, sT, negligible, cap, work, capped):
    """The sub-grid of the first n support points, in the last five arrays of work. count is the number of finite
    log_source among them and peak their largest. Returns its length."""
    steps, nodes, dl, d, du, b, s, uo, ho, ao, so, co, Tk, Hk, ak, To, Ho, So, Co, Ao = work
    nn = 0
    if count >= 2:
        nn = _plan(n, ls, finite, change, want, T, u, peak, negligible, cap, steps, nodes, capped)
    q = 0
    if nn > 0:
        if nn >= 4:
            slope = _system(nn, nodes, u, y, dl, d, du, b)
            _gtsv(nn, dl, d, du, b, s)
        else:
            slope = _pchip_slopes(nn, nodes, u, y, s)
        q = _spline_on_steps(n, nn, nodes, u, y, s, slope, steps, sT, cool, uo, ho, ao, so, co)
        for k in range(q):
            Tk[k] = np.exp(uo[k])
            Hk[k] = np.exp(ho[k])
            ak[k] = np.exp(ao[k])
    return _assemble(n, T, H, S, cool, scale, steps, q, Tk, Hk, so, co, ak, To, Ho, So, Co, Ao)


@njit(cache=True, error_model="numpy")
def subgrid(T, H, S, cool, scale, ls, finite, change, want, u, y, sT, negligible, cap, size, capped):
    """(T, H, S, 3 c_s^2, a) on the sub-grid of the whole array."""
    N = T.size
    work = _workspace(N, size)
    count = 0
    peak = -np.inf
    for i in range(N):
        if finite[i]:
            count += 1
            peak = max(peak, ls[i])
    M = _subgrid(N, count, peak, T, H, S, cool, scale, ls, finite, change, want, u, y, sT, negligible, cap, work, capped)
    return work[15][:M].copy(), work[16][:M].copy(), work[17][:M].copy(), work[18][:M].copy(), work[19][:M].copy()


@njit(cache=True, error_model="numpy")
def double_integral(M, T, H, S, cool, scale):
    """int dT' Gamma (dt/dT') a^3 (int dT''/a)^3 on the first M points of a sub-grid, without the factors of the wall."""
    base = np.empty(M)
    radius = np.empty(M)
    dT = np.empty(M)
    _transport(M, T, H, cool, scale, base, radius, dT)
    return _integrate(M, T, S, base, radius, dT)


@njit(cache=True, error_model="numpy")
def prefix_integrals(T, H, S, cool, scale, ls, finite, change, want, u, y, sT, negligible, cap, size, prefactor, capped):
    """prefactor * double_integral of the sub-grid of T[:i+1], for all i."""
    N = T.size
    out = np.zeros(N)
    work = _workspace(N, size)
    base = np.empty(size + N)
    radius = np.empty(size + N)
    dT = np.empty(size + N)
    count = 0
    peak = -np.inf
    if finite[0]:
        peak = ls[0]
        count = 1
    for i in range(1, N):
        if finite[i]:
            count += 1
            if ls[i] > peak:
                peak = ls[i]
        M = _subgrid(i + 1, count, peak, T, H, S, cool, scale, ls, finite, change, want, u, y, sT, negligible, cap, work,
                     capped)
        To, Ho, So, Co, Ao = work[15:]
        _transport(M, To, Ho, Co, Ao, base, radius, dT)
        out[i] = prefactor * _integrate(M, To, So, base, radius, dT)
    return out


@njit(cache=True, error_model="numpy")
def scale_factor_from_entropy(T, entropy):
    """a/a_hot from a^3 s = const. Where the entropy is not usable the bag relation a ~ 1/T continues the chain, and a
    step that would shrink the universe as it cools is replaced by it as well."""
    N = T.size
    out = np.ones(N)
    ln_a = 0.0
    for i in range(1, N):
        if T[i - 1] > 0.0 and T[i] > 0.0:
            bag = -(np.log(T[i]) - np.log(T[i - 1]))
            if np.isfinite(entropy[i - 1]) and np.isfinite(entropy[i]):
                step = -(np.log(entropy[i]) - np.log(entropy[i - 1])) / 3.0
            else:
                step = bag
            if step * bag < 0.0:
                step = bag
            ln_a += step
        out[i] = np.exp(min(max(ln_a, -700.0), 700.0))
    return out


@njit(cache=True, error_model="numpy")
def distinct(ordered):
    """A descending array without its near-duplicates, relative to the two temperatures: a floor of one in the
    internal units of a model would merge distinct temperatures of a strongly supercooled transition."""
    out = np.empty_like(ordered)
    out[0] = ordered[0]
    n = 1
    for i in range(1, ordered.size):
        c = ordered[i]
        a = out[n - 1]
        scale = abs(a) if abs(a) > abs(c) else abs(c)
        if not (a == c or abs(a - c) <= 1e-10 * scale):
            out[n] = c
            n += 1
    return out[:n].copy()



@njit(cache=True, error_model="numpy")
def count_absent(bank, free):
    """How many entries of bank match no entry of free, relative to the two temperatures, within 1e-10."""
    count = 0
    for v in bank:
        found = False
        for w in free:
            size = abs(w) if abs(w) > abs(v) else abs(v)
            if abs(w - v) <= 1e-10 * size:
                found = True
                break
        if not found:
            count += 1
    return count



@njit(cache=True, error_model="numpy")
def rate_interval_is_inert(temps_in, acts_in, hub_in, high, low, vw, min_peak, inert_log10, max_dI, min_monotone, tolerance, relative):
    """See percolation_adaptive_rate._rate_interval_is_post_peak_inert: the test, from the sort of the support on."""
    order = np.argsort(-temps_in)
    n = order.size
    temps = temps_in[order]
    acts = acts_in[order]
    hub = hub_in[order]
    ln10 = np.log(10.0)
    rate = np.empty(n)
    peak_index = -1
    count = 0
    for i in range(n):
        t = temps[i]
        a = acts[i]
        h = hub[i]
        r = np.nan
        if np.isfinite(t) and np.isfinite(a) and np.isfinite(h) and t > 0.0 and h > 0.0:
            if a == 0.0:
                r = np.inf
            else:
                r = (4.0 * np.log(t) + 1.5 * np.log(a / (2 * np.pi * t)) - a / t - 4.0 * np.log(h)) / ln10
        rate[i] = r
        if np.isfinite(r) and t > 0.0:
            count += 1
            if peak_index < 0 or r > rate[peak_index]:
                peak_index = i
    if count < 3 or peak_index <= 0 or peak_index >= n - 1 or rate[peak_index] < min_peak:
        return False
    if not (np.isfinite(high) and np.isfinite(low) and high > low):
        return False
    high_index = -1
    low_index = -1
    high_gap = np.inf
    low_gap = np.inf
    for i in range(n):
        t = temps[i]
        size = abs(t)
        gap = abs(t - high)
        if gap <= relative * (size if size > abs(high) else abs(high)) and gap < high_gap:
            high_gap = gap
            high_index = i
        gap = abs(t - low)
        if gap <= relative * (size if size > abs(low) else abs(low)) and gap < low_gap:
            low_gap = gap
            low_index = i
    if high_index < 0 or low_index < 0 or high_index >= low_index:
        return False
    if not (np.isfinite(rate[high_index]) and np.isfinite(rate[low_index])):
        return False
    if max(rate[high_index], rate[low_index]) > inert_log10:
        return False
    if max_dI > 0.0:
        ends = np.empty(2)
        for j, i in enumerate((high_index, low_index)):
            r = rate[i]
            h = hub[i]
            t = temps[i]
            valid = np.isfinite(r) and np.isfinite(h) and h > 0.0 and np.isfinite(t) and t > 0.0
            ends[j] = r * ln10 + 3.0 * np.log(h) - 4.0 * np.log(t) if valid else -np.inf
        if not (np.isfinite(ends[0]) and np.isfinite(ends[1])):
            return False
        max_log_a = max(ends[0], ends[1])
        volume = 0.0
        volume_low = 0.0
        volume_high = 0.0
        for i in range(n - 2, -1, -1):
            h0 = hub[i]
            h1 = hub[i + 1]
            inv0 = abs(1.0 / h0) if np.isfinite(h0) and h0 > 0.0 else 0.0
            inv1 = abs(1.0 / h1) if np.isfinite(h1) and h1 > 0.0 else 0.0
            volume += 0.5 * abs(temps[i] - temps[i + 1]) * (inv0 + inv1)
            if i == low_index:
                volume_low = volume
            if i == high_index:
                volume_high = volume
        max_volume = max(volume_low, volume_high)
        width = abs(high - low)
        wall = max(vw, 0.0)
        if max_volume <= 0.0 or width <= 0.0 or wall <= 0.0:
            bound = 0.0
        else:
            log_bound = (np.log(4.0 * np.pi / 3.0) + 3.0 * np.log(wall) + max_log_a + 3.0 * np.log(max_volume)
                         + np.log(width))
            if log_bound > 700.0:
                bound = np.inf
            elif log_bound < -745.0:
                bound = 0.0
            else:
                bound = np.exp(log_bound)
        if not np.isfinite(bound) or bound > max_dI:
            return False
    if high_index > peak_index and low_index > peak_index:
        first, last, sign = peak_index, low_index, 1.0
    elif high_index < peak_index and low_index < peak_index:
        first, last, sign = high_index, peak_index, -1.0
    else:
        return False
    used = 0
    previous = 0.0
    monotone = True
    for i in range(first, last + 1):
        r = rate[i]
        if np.isfinite(r):
            if used > 0 and sign * (r - previous) > tolerance:
                monotone = False
            previous = r
            used += 1
    return used >= min_monotone + 1 and monotone


@njit(cache=True, error_model="numpy")
def _chain_rhs(u, y, x, source, transport, floor, f):
    """dJ_n/du at u = ln T: J0' = -gamma T and Jn' = -n J_(n-1) nu T, from the cubics of ln gamma and ln nu on the nodes x."""
    i = _find_interval(x, u, True)
    d = u - x[i]
    a = ((source[0, i] * d + source[1, i]) * d + source[2, i]) * d + source[3, i]
    b = ((transport[0, i] * d + transport[1, i]) * d + transport[2, i]) * d + transport[3, i]
    t = np.exp(u)
    gamma = np.exp(a) if a > floor else 0.0
    nu = np.exp(b) * t if b > floor else 0.0
    f[0] = -gamma * t
    f[1] = -y[0] * nu
    f[2] = -2.0 * y[1] * nu
    f[3] = -3.0 * y[2] * nu


@njit(cache=True, error_model="numpy")
def _chain_norm(v, scale):
    s = 0.0
    for j in range(4):
        q = v[j] / scale[j]
        s += q * q
    return np.sqrt(s) / 2.0


@njit(cache=True, error_model="numpy")
def _chain_first_step(t0, y, f, bound, x, source, transport, floor, rtol, atol, work, f1):
    """scipy.integrate._ivp.common.select_initial_step for a decreasing variable, with Python's rule for a nan in min and max."""
    interval = abs(bound - t0)
    scale = np.abs(y) * rtol + atol
    d0 = _chain_norm(y, scale)
    d1 = _chain_norm(f, scale)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    if interval < h0:
        h0 = interval
    work[:] = y - h0 * f
    _chain_rhs(t0 - h0, work, x, source, transport, floor, f1)
    d2 = _chain_norm(f1 - f, scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = h0 * 1e-3 if h0 * 1e-3 > 1e-6 else 1e-6
    else:
        h1 = (0.01 / (d2 if d2 > d1 else d1)) ** (1.0 / 8.0)
    h = 100.0 * h0
    if h1 < h:
        h = h1
    if interval < h:
        h = interval
    return h


@njit(cache=True, error_model="numpy")
def _chain_interpolate(F, y_old, a):
    """Dop853DenseOutput of the last component at the fraction a of the step."""
    y = 0.0
    for i in range(6, -1, -1):
        y += F[i, 3]
        y *= a if i % 2 == 0 else 1.0 - a
    return y + y_old


@njit(cache=True, error_model="numpy")
def _chain_event(F, y_old, t_old, t, h, g_old, g_new, level, prefactor):
    """The u where prefactor * J3 crosses level inside the step, or a nan."""
    state = np.empty(roots.STATE)
    x, status = roots.start(state, t_old, g_old, t, g_new)
    value = g_new
    for _ in range(roots.MAXITER):
        if status != roots.RUNNING:
            break
        x, status = roots.step(state, value)
        if status == roots.RUNNING:
            value = prefactor * _chain_interpolate(F, y_old, (x - t_old) / h) - level
    return x if status == roots.CONVERGED else np.nan


@njit(cache=True, error_model="numpy")
def chain_integral(x, source, transport, floor, grid, prefactor, target, stop, rtol, atol, J3):
    """solve_ivp of the J_n from grid[0] down to grid[-1] with y(grid[0]) = 0: J3 on the descending grid, up to where
    prefactor * J3 reaches stop. target is a value of prefactor * J3 to locate, or a nan for none.

    Returns (status, points of the grid reached, the last u with prefactor * J3 = target or a nan, J3 at the stop). The
    event at target is reported as solve_ivp reports it: not after the stop."""
    t = grid[0]
    bound = grid[grid.size - 1]
    y = np.zeros(4)
    y_new = np.empty(4)
    y_old = np.empty(4)
    f = np.empty(4)
    stage = np.empty(4)
    K = np.empty((_STAGES + 4, 4))
    F = np.empty((7, 4))
    scale = np.empty(4)
    _chain_rhs(t, y, x, source, transport, floor, f)
    h_abs = _chain_first_step(t, y, f, bound, x, source, transport, floor, rtol, atol, stage, K[1])
    g_target = -target
    g_stop = -stop
    has_target = target == target
    crossing = np.nan
    j3_stop = np.nan
    reached = 0
    while True:
        min_step = 10.0 * abs(np.nextafter(t, -np.inf) - t)
        if h_abs < min_step:
            h_abs = min_step
        rejected = False
        while True:
            if h_abs < min_step:
                return CHAIN_FAILED, reached, crossing, j3_stop
            t_new = t - h_abs
            if t_new < bound:
                t_new = bound
            h = t_new - t
            h_abs = abs(h)
            K[0] = f
            for s in range(1, _STAGES):
                stage[:] = 0.0
                for r in range(s):
                    a = _A[s, r]
                    if a != 0.0:
                        for j in range(4):
                            stage[j] += K[r, j] * a
                for j in range(4):
                    stage[j] = y[j] + stage[j] * h
                _chain_rhs(t + _C[s] * h, stage, x, source, transport, floor, K[s])
            y_new[:] = 0.0
            for r in range(_STAGES):
                for j in range(4):
                    y_new[j] += K[r, j] * _B[r]
            for j in range(4):
                y_new[j] = y[j] + h * y_new[j]
            _chain_rhs(t_new, y_new, x, source, transport, floor, K[_STAGES])
            error5 = 0.0
            error3 = 0.0
            for j in range(4):
                scale[j] = atol + max(abs(y[j]), abs(y_new[j])) * rtol
                e5 = 0.0
                e3 = 0.0
                for r in range(_STAGES + 1):
                    e5 += K[r, j] * _E5[r]
                    e3 += K[r, j] * _E3[r]
                e5 /= scale[j]
                e3 /= scale[j]
                error5 += e5 * e5
                error3 += e3 * e3
            if error5 == 0.0 and error3 == 0.0:
                error_norm = 0.0
            else:
                error_norm = abs(h) * error5 / np.sqrt((error5 + 0.01 * error3) * 4.0)
            if error_norm < 1.0:
                if error_norm == 0.0:
                    factor = _MAX_FACTOR
                else:
                    factor = _SAFETY * error_norm ** _ERROR_EXPONENT
                    if _MAX_FACTOR < factor:
                        factor = _MAX_FACTOR
                if rejected and 1.0 < factor:
                    factor = 1.0
                h_abs *= factor
                break
            factor = _SAFETY * error_norm ** _ERROR_EXPONENT
            h_abs *= factor if factor > _MIN_FACTOR else _MIN_FACTOR
            rejected = True

        g_target_new = prefactor * y_new[3] - target
        g_stop_new = prefactor * y_new[3] - stop
        hit_target = has_target and ((g_target <= 0.0 and g_target_new >= 0.0) or (g_target >= 0.0 and g_target_new <= 0.0))
        hit_stop = g_stop <= 0.0 and g_stop_new >= 0.0
        end = t_new
        needs_dense = hit_target or hit_stop or (reached < grid.size and grid[reached] >= t_new)
        if needs_dense:
            for s in range(_STAGES + 1, _STAGES + 4):
                stage[:] = 0.0
                for r in range(s):
                    a = _A[s, r]
                    if a != 0.0:
                        for j in range(4):
                            stage[j] += K[r, j] * a
                for j in range(4):
                    stage[j] = y[j] + stage[j] * h
                _chain_rhs(t + _C[s] * h, stage, x, source, transport, floor, K[s])
            for j in range(4):
                delta = y_new[j] - y[j]
                F[0, j] = delta
                F[1, j] = h * K[0, j] - delta
                F[2, j] = 2.0 * delta - h * (K[_STAGES, j] + K[0, j])
            for i in range(4):
                for j in range(4):
                    v = 0.0
                    for r in range(_STAGES + 4):
                        v += _D[i, r] * K[r, j]
                    F[3 + i, j] = h * v
        stopped = False
        if hit_stop:
            root = _chain_event(F, y[3], t, t_new, h, g_stop, g_stop_new, stop, prefactor)
            if root != root:
                return CHAIN_FAILED, reached, crossing, j3_stop
            stopped = True
            end = root
            j3_stop = _chain_interpolate(F, y[3], (root - t) / h)
        if hit_target:
            root = _chain_event(F, y[3], t, t_new, h, g_target, g_target_new, target, prefactor)
            if root != root:
                return CHAIN_FAILED, reached, crossing, j3_stop
            if root >= end:
                crossing = root
        while reached < grid.size and grid[reached] >= end:
            J3[reached] = _chain_interpolate(F, y[3], (grid[reached] - t) / h)
            reached += 1
        if stopped:
            return CHAIN_STOPPED, reached, crossing, j3_stop
        t = t_new
        y[:] = y_new
        f[:] = K[_STAGES]
        g_target = g_target_new
        g_stop = g_stop_new
        if t <= bound:
            return CHAIN_FINISHED, reached, crossing, j3_stop
