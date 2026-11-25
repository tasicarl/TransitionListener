"""The table of the universal function of eft_rate.abelian_lo_bounce_table in numba: its edge in the (A, B) plane at given c and its interpolant.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def edge_B(A, c=1.0):
    """B_E(A) for A > A_S = c/(4 pi): Newton iteration on A = h(f) - f h'(f)/2 in ln f, from the two limits of the curve.

    h(f) = (2 f + c (w + 1/(1 + w))) / (6 pi) with w = sqrt(1 + c f^2); c = h3 / (2 M2) is the temporal mass's
    coupling over its leading-order value."""
    a = A - c / (4.0 * np.pi)
    sixpi = 6.0 * np.pi
    u = math.log(sixpi * a) if a < 0.1 else math.log(12.0 * np.pi * A / (2.0 + c * math.sqrt(c)))
    d = 0.0
    f = 1.0
    for _ in range(60):
        f = math.exp(u)
        w = math.sqrt(1.0 + c * f * f)
        q = 1.0 - 1.0 / ((1.0 + w) * (1.0 + w))
        d = (2.0 + c * c * f / w * q) / sixpi
        dd = c * c * (q / (w * w * w) + 2.0 * c * f * f / (w * w * (1.0 + w) ** 3)) / sixpi
        # A(f) - A_S = (h - h(0)) - f h'/2; h(0) = c/(4 pi) taken out so that small f keeps its digits
        h_minus = (2.0 * f + (c * c * f * f / (w + 1.0)) * (1.0 - 1.0 / ((1.0 + w) * 2.0))) / sixpi
        res = h_minus - 0.5 * f * d - a
        step = res / (0.5 * f * (d - f * dd))
        u -= step
        if abs(step) < 1e-13:
            break
    return d / f


@njit(cache=True)
def _basis(t, x, i, N):
    """The six non-zero quintic B-splines at x on [t[i], t[i+1]] (de Boor's recursion)."""
    left = np.empty(6)
    right = np.empty(6)
    N[0] = 1.0
    for j in range(1, 6):
        left[j] = x - t[i + 1 - j]
        right[j] = t[i + j] - x
        saved = 0.0
        for r in range(j):
            tmp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        N[j] = saved


@njit(cache=True)
def _spline3(x, y, z, x0, hx, y0, hy, z0, hz, tx, ty, tz, c):
    """The tensor-product quintic spline at (x, y, z), all inside the grid."""
    nx, ny, nz = c.shape
    i = int(math.floor((x - x0) / hx)) + 3
    i = 5 if i < 5 else (nx - 1 if i > nx - 1 else i)
    j = int(math.floor((y - y0) / hy)) + 3
    j = 5 if j < 5 else (ny - 1 if j > ny - 1 else j)
    k = int(math.floor((z - z0) / hz)) + 3
    k = 5 if k < 5 else (nz - 1 if k > nz - 1 else k)
    Nx = np.empty(6)
    Ny = np.empty(6)
    Nz = np.empty(6)
    _basis(tx, x, i, Nx)
    _basis(ty, y, j, Ny)
    _basis(tz, z, k, Nz)
    out = 0.0
    for a in range(6):
        plane = 0.0
        for b in range(6):
            row = 0.0
            for e in range(6):
                row += Nz[e] * c[i - 5 + a, j - 5 + b, k - 5 + e]
            plane += Ny[b] * row
        out += Nx[a] * plane
    return out


@njit(cache=True)
def _coordinates(A, B, c=1.0):
    """(code, s, tau). code: 0 inside the region with a bounce, 1 no bounce."""
    a = A - c / (4.0 * np.pi)
    if not a > 0.0:
        return 1, 0.0, 0.0
    ratio = B / edge_B(A, c)
    if not ratio < 1.0:
        return 1, 0.0, 0.0
    return 0, math.log(a), -math.log1p(-ratio)


@njit(cache=True)
def log_F(A, B, c, x0, x1, hx, y0, y1, hy, z0, z1, hz, tx, ty, tz, coef):
    """(code, ln F). code: 0 from the table, 1 no bounce, 2 outside the grid."""
    code, s, tau = _coordinates(A, B, c)
    if code != 0:
        return code, 0.0
    if not (x0 <= s <= x1 and y0 <= tau <= y1 and z0 <= c <= z1):
        return 2, 0.0
    return 0, _spline3(s, tau, c, x0, hx, y0, hy, z0, hz, tx, ty, tz, coef)

