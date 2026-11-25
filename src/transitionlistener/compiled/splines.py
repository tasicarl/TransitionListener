"""Compiled splines: FITPACK splev and splder, scipy's BSpline and PPoly at one point, and a PCHIP interpolant whose
scalar calls run in numba.

The Jb and Jf tables of finiteT, the e_geff and p_geff tables of thermodynamics and the traced phases are all splines
evaluated one point at a time, so these are the innermost loops of a run. A number from an uncommon input (a nan, a
point outside the range) is what the library itself gives.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np
from numba import njit
from scipy import interpolate


@njit(cache=True)
def _upd(hh_i, h_im1, t_li, t_lj, x):
    """One inner step of fpbspl.f: returns the new h(i), h(i+1)."""
    if t_li != t_lj:
        f = hh_i / (t_li - t_lj)
        return h_im1 + f * (t_li - x), f * (x - t_lj)
    return h_im1, 0.0


@njit(cache=True)
def _splev(t, c, tab, x):
    """FITPACK splev.f + fpbspl.f for one point, ext=0 (extrapolate); cubic (k = 3) only.

    tab = [x0, 1/dx, M, L_0, ..., L_{M-1}]: a lookup table on a uniform grid in x giving a
    starting knot interval; the two while loops then make the interval exact, so the result does
    not depend on the table (it only saves the binary search).
    """
    if x != x:
        return np.nan
    n = t.size
    lo = 3
    hi = n - 5
    # largest L in [lo, hi] with t[L] <= x, else lo (splev's interval search, clamped)
    if x < t[lo + 1]:
        L = lo
    elif x >= t[hi]:
        L = hi
    else:
        M = int(tab[2])
        b = int((x - tab[0]) * tab[1])
        if b < 0:
            b = 0
        elif b > M - 1:
            b = M - 1
        L = int(tab[3 + b])
        while L > lo and t[L] > x:
            L -= 1
        while L < hi and t[L + 1] <= x:
            L += 1
    return _splev_at(t, c, L, x)


@njit(cache=True)
def _splev_at(t, c, L, x):
    """fpbspl.f for k = 3 with scalar temporaries, on the knot interval L.
    scipy's BSpline (__fitpack.h _deBoor_D with m = 0) is the same recursion, so interp1d(kind="cubic") is this too."""
    h0, h1 = _upd(1.0, 0.0, t[L + 1], t[L], x)
    h2 = 0.0
    h3 = 0.0
    hh0, hh1 = h0, h1
    h0, h1 = _upd(hh0, 0.0, t[L + 1], t[L - 1], x)
    h1, h2 = _upd(hh1, h1, t[L + 2], t[L], x)
    hh0, hh1, hh2 = h0, h1, h2
    h0, h1 = _upd(hh0, 0.0, t[L + 1], t[L - 2], x)
    h1, h2 = _upd(hh1, h1, t[L + 2], t[L - 1], x)
    h2, h3 = _upd(hh2, h2, t[L + 3], t[L], x)
    sp = 0.0
    sp = sp + c[L - 3] * h0
    sp = sp + c[L - 2] * h1
    sp = sp + c[L - 1] * h2
    sp = sp + c[L] * h3
    return sp


@njit(cache=True)
def _jspline(x, t, c, tab, xmin, xmax, ymin):
    # y = splev(x); y[x < xmin] = splev(xmin); y[x > xmax] = 0
    if x < xmin:
        return ymin
    if x > xmax:
        return 0.0
    return _splev(t, c, tab, x)


@njit(cache=True)
def _interval(x, L, t, tab):
    """The knot interval of x, starting from the guess L in [3, t.size - 5]."""
    if not t[L] <= x < t[L + 1]:
        lo = 3
        hi = t.size - 5
        if x < t[lo + 1]:
            L = lo
        elif x >= t[hi]:
            L = hi
        else:
            M = int(tab[2])
            b = int((x - tab[0]) * tab[1])
            if b > M - 1:
                b = M - 1
            L = int(tab[3 + b])
            while t[L] > x:
                L -= 1
            while t[L + 1] <= x:
                L += 1
    return L


@njit(cache=True)
def _jpiece(x, L, t, pieces, tab, xmin, xmax, ymin):
    """_jspline from the polynomial of the knot interval (_piece_table), as (value, interval): the same spline to rounding,
    at a quarter of the cost of the recursion of _splev_at. L is a guess of the interval, in [3, t.size - 5]. The interval
    of the last evaluation of the mode is the right one almost always, since the points of a finite-difference stencil lie
    close together, and it saves the look-up in a table too large for the fast caches."""
    if x != x:
        return np.nan, L
    if x < xmin:
        return ymin, L
    if x > xmax:
        return 0.0, L
    L = _interval(x, L, t, tab)
    u = x - t[L]
    return pieces[L, 3] + u * (pieces[L, 2] + u * (pieces[L, 1] + u * pieces[L, 0])), L


@njit(cache=True)
def _dpiece(x, L, t, pieces, tab, xmin, xmax):
    """The derivative of the function _jpiece evaluates, as (value, interval): zero where that function is constant."""
    if x != x:
        return np.nan, L
    if x < xmin or x > xmax:
        return 0.0, L
    L = _interval(x, L, t, tab)
    u = x - t[L]
    return pieces[L, 2] + u * (2.0 * pieces[L, 1] + u * 3.0 * pieces[L, 0]), L


@njit(cache=True, error_model="numpy")
def jspline_array(x, t, c, tab, xmin, xmax, ymin, out):
    """_jspline at every entry of x; nan stays nan."""
    for i in range(x.size):
        out[i] = np.nan if x[i] != x[i] else _jspline(x[i], t, c, tab, xmin, xmax, ymin)


@njit(cache=True)
def _bspline(t, c, x):
    """BSpline.__call__ of a cubic interp1d at x inside its range."""
    L = np.searchsorted(t, x, side="right") - 1
    if L < 3:
        L = 3
    elif L > t.size - 5:
        L = t.size - 5
    return _splev_at(t, c, L, x)


@njit(cache=True)
def _splder1(t, w, tab, x):
    """FITPACK splder.f with nu = 1 on a cubic spline; w holds the coefficients of the derivative, from _der_coef."""
    n = t.size
    lo = 3
    hi = n - 5
    if x < t[lo + 1]:
        L = lo
    elif x >= t[hi]:
        L = hi
    else:
        M = int(tab[2])
        b = int((x - tab[0]) * tab[1])
        if b < 0:
            b = 0
        elif b > M - 1:
            b = M - 1
        L = int(tab[3 + b])
        while L > lo and t[L] > x:
            L -= 1
        while L < hi and t[L + 1] <= x:
            L += 1
    h0, h1 = _upd(1.0, 0.0, t[L + 1], t[L], x)
    hh0, hh1 = h0, h1
    h0, h1 = _upd(hh0, 0.0, t[L + 1], t[L - 1], x)
    h1, h2 = _upd(hh1, h1, t[L + 2], t[L], x)
    sp = 0.0
    sp = sp + w[L - 3] * h0
    sp = sp + w[L - 2] * h1
    sp = sp + w[L - 1] * h2
    return sp


def _der_coef(t, c):
    """The first step of splder.f: w[i] = 3 (c[i + 1] - c[i]) / (t[i + 4] - t[i + 1]), kept where the knots differ."""
    w = np.array(c, dtype=float)
    nk1 = t.size - 4
    for i in range(nk1 - 1):
        fac = t[i + 4] - t[i + 1]
        if fac > 0.0:
            w[i] = 3.0 * (w[i + 1] - w[i]) / fac
    return w


@njit(cache=True, error_model="numpy")
def _djspline(x, t, w, tab, xmin, xmax, ymin):
    # y = splev(x, der=1); y[x < xmin] = splev(xmin, der=1); y[x > xmax] = 0
    if x < xmin:
        return ymin
    if x > xmax:
        return 0.0
    return _splder1(t, w, tab, x)



def _piece_table(t, c):
    """The cubic of each knot interval L of a FITPACK spline: row L holds the coefficients of (x - t[L])^3 .. (x - t[L])^0."""
    return np.ascontiguousarray(interpolate.PPoly.from_spline((t, c, 3)).c.T, dtype=float)


def _knot_table(t, M=65536):
    """Starting knot intervals for _splev on a grid uniform in x (bin start -> interval)."""
    lo, hi = 3, t.size - 5
    x0 = float(t[lo])
    inv_dx = M / float(t[hi] - t[lo])
    edges = x0 + np.arange(M) / inv_dx
    L = np.searchsorted(t[lo:hi + 1], edges, side="right") - 1 + lo
    L = np.clip(L, lo, hi)
    return np.concatenate([[x0, inv_dx, M], L.astype(float)])



@njit(cache=True)
def _find_interval(x, xval, extrapolate):
    n = x.size
    a = x[0]
    b = x[n - 1]
    if not (a <= xval <= b):
        if xval < a and extrapolate:
            return 0
        if xval > b and extrapolate:
            return n - 2
        return -1
    if xval == b:
        return n - 2
    low = 0
    high = n - 2
    if xval < x[low + 1]:
        high = low
    while low < high:
        mid = (high + low) // 2
        if xval < x[mid]:
            high = mid
        elif xval >= x[mid + 1]:
            low = mid + 1
        else:
            low = mid
            break
    return low


@njit(cache=True)
def _ppeval(c, x, xval, extrapolate):
    i = _find_interval(x, xval, extrapolate)
    if i < 0:
        return np.nan
    s = xval - x[i]
    res = 0.0
    z = 1.0
    k = c.shape[0]
    for kp in range(k):
        res = res + c[k - kp - 1, i] * z * 1.0
        if kp < k - 1:
            z *= s
    return res



class PchipInterpolator(interpolate.PchipInterpolator):
    """scipy's interpolant, with the call at one float in numba. Every other call is scipy's."""

    def __call__(self, x, nu=0, extrapolate=None):
        if (type(x) is float or type(x) is np.float64) and nu == 0 and extrapolate is None:
            state = self.__dict__.get("_scalar_state")
            if state is None or state[0] is not self.c or state[1] is not self.x:
                ok = (self.c.ndim == 2 and self.axis == 0 and self.x[0] < self.x[-1]
                      and self.extrapolate in (True, False) and self.c.dtype == np.float64
                      and self.c.flags.c_contiguous and self.x.flags.c_contiguous)
                state = (self.c, self.x, ok, bool(self.extrapolate))
                self._scalar_state = state
            if state[2]:
                return np.array(_ppeval(state[0], state[1], float(x), state[3]))
        return super().__call__(x, nu, extrapolate)
