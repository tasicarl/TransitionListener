"""The couplings of a model at every temperature from one solution of its RG equations.

A run asks for the couplings at some 180 scales, each time from the same input at mu0. One solution with dense output
serves them all: it is computed once per input and direction, to twice the distance asked for, and agrees with a direct
solve to 7e-9 in the quartic and 5e-12 in g and y. A pole of the quartic ends the solution; flow returns None beyond it,
and the caller then integrates directly and reports the failure.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np
from scipy.integrate import solve_ivp

_FLOWS = {}
_MAX_FLOWS = 64
_MIN_REACH = 20.0
_TIGHTER = 1e-2         # the interpolant is one order below the integrator
_MAX_STEP = 0.5          # e-folds of the scale: the interpolant needs short steps where the running is slow


def flow(key, rhs, start, u0, target, rtol, atol):
    """u(target) of du/dt = rhs(t, u), u(start) = u0, or None beyond a pole. key identifies the input and the equations."""
    if not (np.isfinite(start) and np.isfinite(target)):
        # a nan scale made the end of the integration nan, and solve_ivp then stepped for ever
        raise ValueError(f"the RG flow needs finite scales, got ln mu = {start} -> {target}")
    side = target > start
    known = _FLOWS.get((key, side))
    if known is None or (known[2] and abs(target - start) > abs(known[0] - start)):
        reach = start + (1.0 if side else -1.0) * max(2.0 * abs(target - start), _MIN_REACH)
        sol = solve_ivp(rhs, (start, reach), u0, rtol=_TIGHTER * rtol, atol=_TIGHTER * atol, dense_output=True,
                        method="RK45", max_step=_MAX_STEP)
        if sol.sol is None:
            return None
        if len(_FLOWS) >= _MAX_FLOWS:
            _FLOWS.clear()
        known = _FLOWS[(key, side)] = (float(sol.t[-1]), sol.sol, bool(sol.success))
    if abs(target - start) > abs(known[0] - start):
        return None
    return np.asarray(known[1](target), dtype=float)
