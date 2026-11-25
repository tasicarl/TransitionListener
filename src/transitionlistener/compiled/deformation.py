"""The arithmetic of one step of Deformation_Spline, and the numbers of a SplinePath.

A path has shape (points, fields), the spline coefficients (basis functions, fields) and the basis functions at the
points (points, basis functions). The loop over the steps stays with the caller, which holds the gradient of any
potential; these kernels know the numbers only.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def basis(t, x, k):
    """The B-splines of degree k on the knots t at the points x, with their first and second derivatives."""
    m = len(t) - 1
    N = np.zeros((len(x), m))
    dN = np.zeros((len(x), m))
    d2N = np.zeros((len(x), m))
    inverse = np.empty(m + 1)
    for i in range(len(x)):
        for j in range(m):
            if x[i] > t[j] and x[i] <= t[j + 1]:
                N[i, j] = 1.0
    for order in range(1, k + 1):
        width = len(t) - order
        for j in range(width):
            step = t[j + order] - t[j]
            inverse[j] = 1.0 / step if step != 0.0 else 0.0
        for i in range(len(x)):
            for j in range(width - 1):
                left = inverse[j]
                right = inverse[j + 1]
                a = (x[i] - t[j]) * left
                b = (x[i] - t[j + order + 1]) * right
                d2N[i, j] = d2N[i, j] * a - d2N[i, j + 1] * b + 2.0 * (dN[i, j] * left - dN[i, j + 1] * right)
                dN[i, j] = dN[i, j] * a - dN[i, j + 1] * b + N[i, j] * left - N[i, j + 1] * right
                N[i, j] = N[i, j] * a - N[i, j + 1] * b
    return N[:, :m - k].copy(), dN[:, :m - k].copy(), d2N[:, :m - k].copy()


@njit(cache=True, error_model="numpy")
def normal_force(beta, dX, d2X, phi, dV, v2):
    """The normal force on the path, v2 d2phi/ds2 - dV_perp, with s the path length."""
    n, fields = phi.shape
    F = np.empty((n, fields))
    dphi = np.empty(fields)
    d2phi = np.empty(fields)
    for i in range(n):
        dphi_sq = 0.0
        dphi_d2phi = 0.0
        for a in range(fields):
            first = phi[n - 1, a] - phi[1, a]
            second = 0.0
            for b in range(beta.shape[0]):
                first += beta[b, a] * dX[i, b]
                second += beta[b, a] * d2X[i, b]
            dphi[a] = first
            d2phi[a] = second
            dphi_sq += first * first
            dphi_d2phi += first * second
        length = np.sqrt(dphi_sq)
        dV_along = 0.0
        for a in range(fields):
            dV_along += dV[i, a] * dphi[a] / length
        for a in range(fields):
            d2phids2 = (d2phi[a] - dphi[a] * dphi_d2phi / dphi_sq) / dphi_sq
            F[i, a] = d2phids2 * v2[i, 0] - dV[i, a] + dV_along * dphi[a] / length
    return F


@njit(cache=True, error_model="numpy")
def largest_norm(vectors):
    """The largest length of the rows, nan if one of them is nan, as np.max returns."""
    largest = -np.inf
    for i in range(vectors.shape[0]):
        square = 0.0
        for a in range(vectors.shape[1]):
            square += vectors[i, a] * vectors[i, a]
        length = np.sqrt(square)
        if length != length:
            return length
        if length > largest:
            largest = length
    return largest


@njit(cache=True, error_model="numpy")
def reversed_points(F, F_last):
    """The number of points at which the force turned against the last one."""
    count = 0
    for i in range(F.shape[0]):
        product = 0.0
        for a in range(F.shape[1]):
            product += F[i, a] * F_last[i, a]
        if product < 0.0:
            count += 1
    return count


@njit(cache=True, error_model="numpy")
def fit(inverse, target):
    """The spline coefficients of target; False as the second value if target is not finite."""
    beta = np.zeros((inverse.shape[0], target.shape[1]))
    finite = True
    for i in range(target.shape[0]):
        for a in range(target.shape[1]):
            value = target[i, a]
            if not np.isfinite(value):
                finite = False
            for b in range(inverse.shape[0]):
                beta[b, a] += inverse[b, i] * value
    return beta, finite


@njit(cache=True, error_model="numpy")
def spline_step(phi, F, stepsize, t, X, inverse, length):
    """Move phi by stepsize F and fit the result: the new path, its coefficients, the ratio of the fitted force to
    the length, and False in the fourth place if the moved path is not finite. phi and F are not changed."""
    n, fields = phi.shape
    moved = phi + F * stepsize
    start = moved[0].copy()
    span = moved[n - 1] - start
    line = np.empty((n, fields))
    for i in range(n):
        for a in range(fields):
            line[i, a] = start[a] + span[a] * t[i, 0]
    beta, finite = fit(inverse, moved - line)
    new = line + X @ beta
    return new, beta, largest_norm((new - phi) / stepsize) / length, finite


_EPS = np.finfo(float).eps
_TINY = np.finfo(float).tiny


@njit(cache=True, error_model="numpy")
def _speed(brk, coef, j, x):
    """|dp/du| at u = x on the interval j of the piecewise polynomial of the path's derivative."""
    d = x - brk[j]
    square = 0.0
    for a in range(coef.shape[2]):
        value = coef[0, j, a]
        for k in range(1, coef.shape[0]):
            value = value * d + coef[k, j, a]
        square += value * value
    return np.sqrt(square)


@njit(cache=True, error_model="numpy")
def arc_length(brk, coef, u, nodes, weights):
    """The length of the path from u[0] to each u, with coef[order, interval, field] the piecewise polynomial of its
    derivative: Gauss-Legendre on every piece between neighbouring points and breakpoints."""
    length = np.zeros(u.size)
    j = 0
    last = brk.size - 2
    for i in range(1, u.size):
        total = length[i - 1]
        a = u[i - 1]
        while a < u[i]:
            while j < last and brk[j + 1] <= a:
                j += 1
            b = min(u[i], brk[j + 1]) if j < last else u[i]
            half = 0.5 * (b - a)
            for n in range(nodes.size):
                total += half * weights[n] * _speed(brk, coef, j, a + half * (nodes[n] + 1.0))
            a = b
        length[i] = total
    return length


@njit(cache=True, error_model="numpy")
def first_rise(ladder, probes):
    """Walk outward along the ladder of potentials. Returns the index of the first value above the lowest one so far,
    -1 if the walk ends without one, and the two rungs that bracket the minimum."""
    best_x, best_v, previous_x = 0.0, probes[0], 0.0
    for i in range(1, probes.size):
        current = probes[i]
        if not np.isfinite(current):
            break
        noise = 8.0 * _EPS * max(abs(current), abs(best_v), _TINY)
        if current > best_v + noise:
            return i, previous_x, best_x
        if current < best_v:
            previous_x, best_x, best_v = best_x, ladder[i], current
    return -1, previous_x, best_x
