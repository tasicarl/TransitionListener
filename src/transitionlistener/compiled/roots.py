"""scipy's brentq (Zeros/brentq.c) in numba, one step at a time.

numba does not cache code that takes a function as an argument, so the search takes the values of the function from its
caller, in this loop::

    state = np.empty(STATE)
    value = f(b)
    x, status = start(state, a, f(a), b, value)
    for _ in range(MAXITER):
        if status != RUNNING:
            break
        x, status = step(state, value)
        if status == RUNNING:
            value = f(x)

status is then CONVERGED, NOT_A_NUMBER or SAME_SIGN, and RUNNING if the search ran out of iterations. Any but CONVERGED
leaves the decision to the caller, in general for the error of optimize.brentq.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np
from numba import njit

#: The defaults of optimize.brentq.
XTOL = 2e-12
RTOL = 4 * np.finfo(float).eps
MAXITER = 100

STATE = 7
RUNNING, CONVERGED, NOT_A_NUMBER, SAME_SIGN = 0, 1, 2, 3
_XPRE, _XCUR, _XBLK, _FPRE, _FBLK, _SPRE, _SCUR = range(STATE)


@njit(cache=True)
def start(state, xa, fa, xb, fb):
    """Begin the search in [xa, xb] from the values at the ends. Returns (x, status)."""
    if fa != fa:
        return xa, NOT_A_NUMBER
    if fb != fb:
        return xb, NOT_A_NUMBER
    if fa == 0.0:
        return xa, CONVERGED
    if fb == 0.0:
        return xb, CONVERGED
    if np.signbit(fa) == np.signbit(fb):
        return 0.0, SAME_SIGN
    state[_XPRE], state[_XCUR], state[_XBLK] = xa, xb, 0.0
    state[_FPRE], state[_FBLK] = fa, 0.0
    state[_SPRE], state[_SCUR] = 0.0, 0.0
    return xb, RUNNING


@njit(cache=True)
def step(state, fcur):
    """One iteration, with the value at the last x. Returns (x, status)."""
    xpre = state[_XPRE]
    xcur = state[_XCUR]
    xblk = state[_XBLK]
    fpre = state[_FPRE]
    fblk = state[_FBLK]
    spre = state[_SPRE]
    scur = state[_SCUR]
    if fcur != fcur:
        return xcur, NOT_A_NUMBER
    if fpre != 0.0 and fcur != 0.0 and np.signbit(fpre) != np.signbit(fcur):
        xblk = xpre
        fblk = fpre
        spre = scur = xcur - xpre
    if abs(fblk) < abs(fcur):
        xpre = xcur
        xcur = xblk
        xblk = xpre
        fpre = fcur
        fcur = fblk
        fblk = fpre
    delta = (XTOL + RTOL * abs(xcur)) / 2
    sbis = (xblk - xcur) / 2
    if fcur == 0.0 or abs(sbis) < delta:
        return xcur, CONVERGED
    if abs(spre) > delta and abs(fcur) < abs(fpre):
        if xpre == xblk:
            stry = -fcur * (xcur - xpre) / (fcur - fpre)
        else:
            dpre = (fpre - fcur) / (xpre - xcur)
            dblk = (fblk - fcur) / (xblk - xcur)
            stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
        if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
            spre = scur
            scur = stry
        else:
            spre = sbis
            scur = sbis
    else:
        spre = sbis
        scur = sbis
    xpre = xcur
    fpre = fcur
    if abs(scur) > delta:
        xcur += scur
    else:
        xcur += delta if sbis > 0.0 else -delta
    state[_XPRE], state[_XCUR], state[_XBLK] = xpre, xcur, xblk
    state[_FPRE], state[_FBLK] = fpre, fblk
    state[_SPRE], state[_SCUR] = spre, scur
    return xcur, RUNNING
