"""This module contains the phases object. It contains
an interface to the phase tracing routines, stores
the phase information of the potential and provides
interpolated functions.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np
from dataclasses import dataclass
from scipy import optimize
from scipy import interpolate
from scipy import linalg
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Iterable

from transitionlistener import generic_potential
from transitionlistener.errors import WrongHighTPhaseError

from . import console, print
import rich
from rich.columns import Columns as RichColumns

if TYPE_CHECKING:
    from transitionlistener.transitions import TransitionInfo

EXTRA_ACCURATE_TRACING = False


def findLocalMinimum(
    X: np.ndarray,
    T: float,
    dV: Callable,
    d2V: Callable,
    phitol: float = 1e-6,
    step_size: float = 0.1,
) -> np.ndarray:
    r"""Find a nearby local minimum by Newton stepping in field space."""

    hess_offset = np.diag(np.ones_like(X) * 1e-5)
    niter = 0
    while True:
        niter += 1
        Hess = d2V(X, T)
        if np.linalg.det(Hess) == 0:
            break
        eps = np.matmul(np.linalg.inv(Hess + hess_offset), dV(X, T))
        X = X - eps * step_size
        if np.sqrt(np.sum(eps * eps)) <= phitol:
            break
        if niter >= 200:
            break
    return X


if EXTRA_ACCURATE_TRACING:
    def traceLocalMinimum(f, x, t, df_dx, d2f_dx2, xeps):
        return findLocalMinimum(x, t, df_dx, d2f_dx2)
else:
    def traceLocalMinimum(f, x, t, df_dx, d2f_dx2, xeps):
        return optimize.fmin(f, x, args=(t,), xtol=xeps, ftol=np.inf, disp=False)


class PhaseInfo:
    """Describes a temperature-dependent minimum, plus second-order transitions
    to and from that minimum.

    Attributes
    ----------
    key : hashable
        A unique identifier for the phase (usually an int).
    X, T, dXdT : array_like
        The minima and its derivative at different temperatures.
    tck : tuple
        Spline knots and coefficients, used in `interpolate.splev`.
    low_trans : set
        Phases (identified by keys) which are joined by a second-order
        transition to this phase.
    high_trans : set
        Phases (identified by keys) which are joined by a second-order
        transition to this phase.
    """

    def __init__(self, key: int | str, X: np.ndarray, T: np.ndarray, dXdT: np.ndarray):
        """Create the spline representation of a phase trajectory in field space.
        
        Parameters
        ----------
        key : hashable
            A unique identifier for the phase (usually an int).
        X : np.ndarray
            The minima of the potential at different temperatures.
        T : np.ndarray
            The temperatures at which the minima occur.
        dXdT : np.ndarray
            The derivative of the minima with respect to temperature.

        .. note::
            This class is not intended to be used directly. Instead, use the
            `Phase` class, which provides a more convenient interface.
            
        .. todo::
            Include check if key is hashable. Should it be int or str only?
        """
        self.key = key
        # We shouldn't ever really need to sort the array, but there must be
        # some bug in the above code that makes it so that occasionally the last
        # step goes backwards. This should fix that.
        i = np.argsort(T)
        T, X, dXdT = T[i], X[i], dXdT[i]
        self.X = X
        self.T = T
        self.dXdT = dXdT
        self.Tmax = T[-1]
        self.Tmin = T[0]
        # Make the spline:
        k = 3 if len(T) > 3 else 1
        tck, u = interpolate.splprep(X.T, u=T, s=0, k=k)
        self.tck = tck
        # Make default connections
        self.low_trans = set()
        self.high_trans = set()
        self.parentPhases = []  # empty if high-T phase
        self.childPhases = []
        # if not none, but a phase key, the phase is generated
        # by symmetry transformations of the potential
        self.mirrorPhase = None

    def valAt(self, T: float | np.ndarray, deriv: int = 0) -> np.ndarray:
        """Find the minimum at the value `T` using a spline.

        Parameters
        ----------
        T : float | np.ndarray
            The temperature at which to eval the minimum
        deriv : int
            If deriv > 0, instead return the derivative of the minimum with
            respect to `T`. Can return up to the third derivative for cubic
            splines (when ``len(X) > 3``) or first derivative for linear
            splines.

        Returns
        -------
        np.ndarray :
            The minimum at the temperature `T`."""
        T = np.asanyarray(T).T
        y = interpolate.splev(T, self.tck, der=deriv)
        return np.asanyarray(y).T

    def addLinkFrom(self, other_phase) -> None:
        """Add a link from `other_phase` to this phase, checking to see if there
        is a second-order transition.

        Parameters
        ----------
        other_phase : PhaseInfo
            Information of the other phase
        Returns
        -------
        None."""
        if np.min(self.T) >= np.max(other_phase.T):
            self.low_trans.add(other_phase.key)
            other_phase.high_trans.add(self.key)
        if np.max(self.T) <= np.min(other_phase.T):
            self.high_trans.add(other_phase.key)
            other_phase.low_trans.add(self.key)

    def __repr__(self):
        """Return a concise string representation summarising the phase track."""
        popts = np.get_printoptions()
        np.set_printoptions(formatter={"float": lambda x: "%0.4g" % x})
        if len(self.X) > 1:
            Xstr = "[%s, ..., %s]" % (self.X[0], self.X[-1])
        else:
            Xstr = "[%s]" % self.X[0]
        if len(self.T) > 1:
            Tstr = "[%0.4g, ..., %0.4g]" % (self.T[0], self.T[-1])
        else:
            Tstr = "[%0.4g]" % self.T[0]
        if len(self.dXdT) > 1:
            dXdTstr = "[%s, ..., %s]" % (self.dXdT[0], self.dXdT[-1])
        else:
            dXdTstr = "[%s]" % self.dXdT[0]
        s = "Phase(key=%s, X=%s, T=%s, dXdT=%s" % (self.key, Xstr, Tstr, dXdTstr)
        np.set_printoptions(**popts)
        return s


def findApproxLocalMin(f: Callable, x1: np.ndarray, x2: np.ndarray, args=(), n=100, edge=0.05) -> list:
    """
    Find minima on a straight line between two points.

    When jumping between phases, we want to make sure that we
    don't jump over an intermediate phase. This function does a rough
    calculation to find any such intermediate phases.

    Parameters
    ----------
    f : callable
        The function `f(x)` to minimize.
    x1, x2 : array_like
        The points between which to find minima.
    args : tuple, optional
        Extra arguments to pass to `f`.
    n : int, optional
        Number of points to test for local minima.
    edge : float, optional
        Don't test for minima directly next to the input points. If ``edge==0``,
        the minima potentially go all the way to input points. If ``edge==0.5``,
        the range of tested minima shrinks to a single point at the center of
        the two points.

    Returns
    -------
    list
        A list of approximate minima, with each minimum having the same shape
        as `x1` and `x2`.
    """
    x1, x2 = np.array(x1), np.array(x2)
    x = x1 + (x2 - x1) * np.linspace(edge, 1 - edge, n).reshape(n, 1)
    y = f(x, *args)
    i = (y[2:] > y[1:-1]) & (y[:-2] > y[1:-1])
    return x[1:-1][i]


_traceMinimum_rval = namedtuple("traceMinimum_rval", "X T dXdT overX overT")


def traceMinimum(
    f : Callable, df_dx : Callable, d2f_dxdt : Callable, d2f_dx2 : Callable, 
    x0 : np.ndarray, t0 : float, tstop : float, dtstart : float, deltaX_target : float,
    dtabsMax: float = 20.0, dtfracMax: float = 0.25, dtmin : float = 1e-3, deltaX_tol : float = 1.2,
    minratio : float = 1e-2, verbose : bool = False, conversionFactor : float = 1,
) -> _traceMinimum_rval:

    """
    Trace the minimum `xmin(t)` of the function `f(x,t)`, starting at `x0, t0`.

    Parameters
    ----------
    f : callable
        The scalar function `f(x,t)` which needs to be minimized. The input will
        be of the same type as `(x0,t0)`.
    d2f_dxdt, d2f_dx2 : callable
        Functions which return returns derivatives of `f(x)`. `d2f_dxdt` should
        return the derivative of the gradient of `f(x)` with respect to `t`, and
        `d2f_dx2` should return the Hessian matrix of `f(x)` evaluated at `t`.
        Both should take as inputs `(x,t)`.
    x0 : array_like
        The initial starting point. Must be an array even if the potential is
        one-dimensional (in which case the array should have length 1).
    t0 : float
        The initial starting parameter `t`.
    tstop : float
        Stop the trace when `t` reaches `tstop`.
    dtstart : float
        Initial stepsize.
    deltaX_target : float
        The target error in x at each step. Determines the
        stepsize in t by extrapolation from last error.
    dtabsMax : float, optional
    dtfracMax : float, optional
        The largest stepsize in t will be the LARGEST of
        ``abs(dtstart)*dtabsMax`` and ``t*dtfracMax``.
    dtmin : float, optional
        The smallest stepsize we'll allow before assuming the transition ends,
        relative to `dtstart`
    deltaX_tol : float, optional
        ``deltaX_tol*deltaX_target`` gives the maximum error in x
        before we want to shrink the stepsize and recalculate the minimum.
    minratio : float, optional
        The smallest ratio between smallest and largest eigenvalues in the
        Hessian matrix before treating the smallest eigenvalue as zero (and
        thus signaling a saddle point and the end of the minimum).

    Returns
    -------
      X, T, dXdT : array_like
        Arrays of the minimum at different values of t, and
        its derivative with respect to t.
      overX : array_like
        The point beyond which the phase seems to disappear.
      overT : float
        The t-value beyond which the phase seems to disappear.

    Notes
    -----
    In prior versions, `d2f_dx2` was optional and called `d2f`, while `d2f_dxdt`
    was calculated from an optional parameter `df` using finite differences. If
    Neither of these were supplied, they would be calculated directly from
    `f(x,t)` using finite differences. This lead to a messier calling signature,
    since additional parameters were needed to find the finite differences. By
    instead requiring that the derivatives be supplied, the task of creating the
    derivative functions can be delegated to more general purpose routines
    (see e.g. :class:`helper_functions.gradientFunction` and
    :class:`helper_functions.hessianFunction`).

    Also new in this version, `dtmin` and `dtabsMax` are now relative to
    `dtstart`. The idea here is that there should be some required parameter
    that sets the scale, and then optional parameters can set the tolerances
    relative to this scale. `deltaX_target` is now not optional for the same
    reasoning.
    """
    Ndim = len(x0)
    M0 = d2f_dx2(x0, t0)
    minratio *= min(abs(linalg.eigvalsh(M0))) / max(abs(linalg.eigvalsh(M0)))

    def dxmindt(x, t):
        M = d2f_dx2(x, t)
        if abs(linalg.det(M)) < (1e-3 * np.max(abs(M))) ** Ndim:
            # Assume matrix is singular
            return None, False
        b = -d2f_dxdt(x, t)
        eigs = linalg.eigvalsh(M)
        try:
            dxdt = linalg.solve(M, b, overwrite_a=False, overwrite_b=False)
            isneg = (eigs <= 0).any() or min(eigs) / max(eigs) < minratio
        except Exception:
            dxdt = None
            isneg = False
        return dxdt, isneg

    xeps = deltaX_target * 1e-2

    def fmin(x, t):
        return traceLocalMinimum(f, x, t, df_dx, d2f_dx2, xeps)

    deltaX_tol = deltaX_tol * deltaX_target
    tscale = abs(dtstart)
    dtabsMax = dtabsMax * tscale
    dtmin = dtmin * tscale

    x, t, dt, xerr = x0, t0, dtstart, 0.0
    dxdt, negeig = dxmindt(x, t)
    X, T, dXdT = [x], [t], [dxdt]
    overX = overT = None

    def trace():
        nonlocal x, t, dt, dxdt, xerr, overX, overT

        # Get the values at the next step
        tnext = t + dt
        xnext = fmin(x + dxdt * dt, tnext)
        dxdt_next, negeig = dxmindt(xnext, tnext)
        if dxdt_next is None or negeig is True:
            # We got stuck on a saddle, so there must be a phase transition
            # there.
            dt *= 0.5
            overX, overT = xnext, tnext
        else:
            # The step might still be too big if it's outside of our error
            # tolerance.
            xerr = max(np.sum((x + dxdt * dt - xnext) ** 2), np.sum((xnext - dxdt_next * dt - x) ** 2)) ** 0.5
            if xerr < deltaX_tol:  # Normal step, error is small
                T.append(tnext)
                X.append(xnext)
                dXdT.append(dxdt_next)
                if overT is None:
                    # change the stepsize only if the last step wasn't
                    # troublesome
                    dt *= deltaX_target / (xerr + 1e-100)
                x, t, dxdt = xnext, tnext, dxdt_next
                overX = overT = None
            else:
                # Either stepsize was too big, or we hit a transition.
                # Just cut the step in half.
                dt *= 0.5
                overX, overT = xnext, tnext

    def check():
        nonlocal x, t, dt, dxdt, xerr, overX, overT, negeig
        # Now do some checks on dt.
        if abs(dt) < abs(dtmin):
            # Found a transition! Or at least a point where the step is really
            # small.
            return True

        if dt > 0 and t >= tstop or dt < 0 and t <= tstop:
            # Reached tstop, but we want to make sure we stop right at tstop.
            dt = tstop - t
            xnext = fmin(x + dxdt * dt, tstop)
            dxdt_next, negeig = dxmindt(x, tstop)
            tnext = tstop
            if dxdt_next is None:
                # Cannot evaluate the next slope reliably; stop the trace here.
                X[-1], T[-1], dXdT[-1] = xnext, tnext, dxdt
                return True
            # Here it previously doesn't check if we have a phase transition between
            # tstop and t. This can happend. Therefore implement a
            # check for this:
            xerr = max(np.sum((x + dxdt * dt - xnext) ** 2), np.sum((xnext - dxdt_next * dt - x) ** 2)) ** 0.5
            if xerr < deltaX_tol:
                # good step, found endpoint of phase
                X[-1], T[-1], dXdT[-1] = xnext, tnext, dxdt_next
                return True
            else:
                # remove wrong entries:
                _, _, _ = X.pop(), T.pop(), dXdT.pop()
                # reset t, etc
                t, x, _ = T[-1], X[-1], dXdT[-1]
                # Either stepsize was too big, cut the step in half.
                dt *= 0.5
                overX, overT = xnext, tnext

        dtmax = max(t * dtfracMax, dtabsMax)
        if abs(dt) > dtmax:
            dt = np.sign(dt) * dtmax
        return False

    if verbose:
        with console.status("Tracing phase...", spinner="bouncingBall"):
            while dxdt is not None:
                trace()
                if check():
                    break
        console.print("[bold green]Phase traced :heavy_check_mark:")
    else:
        while dxdt is not None:
            trace()
            if check():
                break

    if overT is None:
        overX, overT = X[-1], T[-1]

    X = np.array(X)
    T = np.array(T)
    dXdT = np.array(dXdT)
    return _traceMinimum_rval(X, T, dXdT, overX, overT)


def traceMultiMin(
    f: Callable, df_dx: Callable, d2f_dxdt: Callable, d2f_dx2: Callable,
    points: list, tLow: float, tHigh: float, deltaX_target: float,
    verbose: bool = False, conversionFactor: float = 1, dtstart: float = 1e-3,
    tjump: float = 1e-3, forbidCrit: Callable | None = None, applySymmetries: Callable | None = None,
    single_trace_args: dict = {}, local_min_args: dict = {},
) -> dict[int, PhaseInfo]:
    """
    Trace multiple minima `xmin(t)` of the function `f(x,t)`.

    This function will trace the minima starting from the initial `(x,t)` values
    given in `points`. When a phase disappears, the function will search for
    new nearby minima, and trace them as well. In this way, if each minimum
    corresponds to a different phase, this function can find the (possibly)
    complete phase structure of the potential.

    Parameters
    ----------
    f : callable
        The scalar function `f(x,t)` which needs to be minimized. The input will
        be of the same type as each entry in the `points` parameter.
    d2f_dxdt, d2f_dx2 : callable
        Functions which return returns derivatives of `f(x)`. `d2f_dxdt` should
        return the derivative of the gradient of `f(x)` with respect to `t`, and
        `d2f_dx2` should return the Hessian matrix of `f(x)` evaluated at `t`.
        Both should take as inputs `(x,t)`.
    points : list
        A list of points [(x1,t1), (x2,t2),...] that we want to trace, where
        `x1`, `x2`, etc. are each a one-dimensional array.
    tLow, tHigh : float
        Lowest and highest temperatures between which to trace.
    deltaX_target : float
        Passed to :func:`traceMinimum` and used to set the tolerance in
        minimization.
    dtstart : float, optional
        The starting stepsize, relative to ``tHigh-tLow``.
    tjump : float, optional
        The jump in `t` from the end of one phase to the initial tracing point
        in another. If this is too large, intermediate phases may be skipped.
        Relative to ``tHigh-tLow``.
    forbidCrit : callable or None, optional
        A function that determines whether or not to forbid a phase with a given
        starting point. Should take a point `x` as input, and return True (if
        the phase should be discarded) or False (if the phase should be kept).
    single_trace_args : dict, optional
        Arguments to pass to :func:`traceMinimum`.
    local_min_args : dict, optoinal
        Arguments to pass to :func:`findApproxLocalMinima`.

    Returns
    -------
    phases : list[PhaseInfo]
        A dictionary of :class:`Phase` instances. The keys in the dictionary
        are integers corresponding to the order in which the phases were
        constructed.
    """
    # We want the minimization here to be very accurate so that we don't get
    # stuck on a saddle or something. This isn't much of a bottle neck.
    xeps = deltaX_target * 1e-2
    CF = conversionFactor

    def fmin(x, t):
        return traceLocalMinimum(f, x, t, df_dx, d2f_dx2, xeps)

    dtstart = dtstart * (tHigh - tLow)
    tjump = tjump * (tHigh - tLow)
    phases = {}
    nextPoint = []
    for p in points:
        x, t = p
        nextPoint.append([t, dtstart, fmin(x, t), None])

    while len(nextPoint) != 0:
        t1, dt1, x1, linkedFrom = nextPoint.pop()
        # check that we are in the right quadrant of field space.
        if applySymmetries is not None:
            x1 = applySymmetries(x1)
        x1 = fmin(x1, t1)  # make sure we start as accurately as possible.
        # Check to see if this point is outside the bounds
        if t1 < tLow or (t1 == tLow and dt1 < 0):
            continue
        if t1 > tHigh or (t1 == tHigh and dt1 > 0):
            continue
        if forbidCrit is not None and forbidCrit(x1) is True:
            continue
        # Check to see if it's redudant with another phase
        for i, phase in enumerate(phases.values()):
            if t1 < min(phase.Tmin, phase.Tmax) or t1 > max(phase.Tmin, phase.Tmax):
                continue
            x = fmin(phase.valAt(t1), t1)
            if np.sum((x - x1) ** 2) ** 0.5 < 2 * deltaX_target:
                # The point is already covered
                # Skip this phase and change the linkage.
                if linkedFrom != i and linkedFrom is not None:
                    phase.addLinkFrom(phases[linkedFrom])
                break
        else:
            # The point is not yet covered. Trace the phase.
            if verbose:
                print("\nTracing phase starting at phi =", x1 * CF, "GeV; T =", t1 * CF, "GeV")
            phase_key = len(phases)
            oldNumPoints = len(nextPoint)
            if t1 > tLow:
                if verbose:
                    print("Tracing minimum down")
                down_trace = traceMinimum(
                    f,
                    df_dx,
                    d2f_dxdt,
                    d2f_dx2,
                    x1,
                    t1,
                    tLow,
                    -dt1,
                    deltaX_target,
                    **single_trace_args,
                    verbose=verbose,
                    conversionFactor=conversionFactor,
                )
                X_down, T_down, dXdT_down, nX, nT = down_trace
                t2, dt2 = nT - tjump, 0.1 * tjump
                x2 = fmin(nX, t2)
                nextPoint.append([t2, dt2, x2, phase_key])
                if np.sum((X_down[-1] - x2) ** 2) > deltaX_target**2:
                    for point in findApproxLocalMin(
                        f, X_down[-1], x2, (t2,), **local_min_args
                    ):
                        nextPoint.append([t2, dt2, fmin(point, t2), phase_key])
                X_down = X_down[::-1]
                T_down = T_down[::-1]
                dXdT_down = dXdT_down[::-1]
            if t1 < tHigh:
                if verbose:
                    print("Tracing minimum up")
                up_trace = traceMinimum(
                    f,
                    df_dx,
                    d2f_dxdt,
                    d2f_dx2,
                    x1,
                    t1,
                    tHigh,
                    +dt1,
                    deltaX_target,
                    **single_trace_args,
                    verbose=verbose,
                    conversionFactor=conversionFactor,
                )
                X_up, T_up, dXdT_up, nX, nT = up_trace
                t2, dt2 = nT + tjump, 0.1 * tjump
                x2 = fmin(nX, t2)
                nextPoint.append([t2, dt2, x2, phase_key])
                if np.sum((X_up[-1] - x2) ** 2) > deltaX_target**2:
                    for point in findApproxLocalMin(
                        f, X_up[-1], x2, (t2,), **local_min_args
                    ):
                        nextPoint.append([t2, dt2, fmin(point, t2), phase_key])
            # Then join the two together
            if t1 <= tLow:
                X, T, dXdT = X_up, T_up, dXdT_up
            elif t1 >= tHigh:
                X, T, dXdT = X_down, T_down, dXdT_down
            else:
                X = np.append(X_down, X_up[1:], 0)
                T = np.append(T_down, T_up[1:], 0)
                dXdT = np.append(dXdT_down, dXdT_up[1:], 0)
            if forbidCrit is not None and (forbidCrit(X[0]) or forbidCrit(X[-1])):
                # The phase is forbidden.
                # Don't add it, and make it a dead-end.
                nextPoint = nextPoint[:oldNumPoints]
            elif len(X) > 1:
                newphase = PhaseInfo(phase_key, X, T, dXdT)
                if linkedFrom is not None:
                    newphase.addLinkFrom(phases[linkedFrom])
                phases[phase_key] = newphase
            else:
                # The phase is just a single point.
                # Don't add it, and make it a dead-end.
                nextPoint = nextPoint[:oldNumPoints]

    return phases


def generateMirrorPhases(phases: dict[int | str, PhaseInfo],
                         diftol: float, invGroupElements: list[np.ndarray]):
    """Use the transformations of the potential
    to generate the mirror phases that have not been traced.

    Parameters
    ----------
    phases : list[PhaseInfo]
        List of the phases
    diftol : float
        The tolerance in field space for which to consider two phases equal
    invGroupElements : list[np.ndarray]
        Matrix transformations which leave the potential invariant

    Returns
    -------
    """
    if invGroupElements == []:
        return None

    mPhases = []

    for phase in phases.values():
        new_mphases = []
        for i, g in enumerate(invGroupElements):
            if np.sum(g) == len(g):
                # skip the identity
                break
            mkey = str(phase.key) + "-m" + str(i + 1)
            mirrorPhase = PhaseInfo(mkey, phase.X @ g, phase.T, phase.dXdT @ g)
            for lk in phase.low_trans:
                mirrorPhase.low_trans.add(str(lk) + "-m" + str(i))
            for hk in phase.high_trans:
                mirrorPhase.high_trans.add(str(lk) + "-m" + str(i))

            mirrorPhase.mirrorPhase = phase.key
            new_mphases.append(mirrorPhase)

        while True:
            redundant = False
            mp = new_mphases.pop()
            # check if we created one phase twice by 2 different
            # transformations
            for op in new_mphases:
                DXmin = mp.X[0] - op.X[0]
                DXmax = mp.X[-1] - op.X[-1]
                if (np.sqrt(np.dot(DXmin, DXmin)) < diftol and 
                    np.sqrt(np.dot(DXmax, DXmax)) < diftol):
                    redundant = True
                    break

            if not redundant:
                mPhases.append(mp)
            if new_mphases == []:
                break

    for mp in mPhases:
        phases[mp.key] = mp


def _removeRedundantPhase(phases: dict[int | str, PhaseInfo],
                          removed_phase: PhaseInfo,
                          redundant_with_phase: PhaseInfo):
    """
    Remove a phase from the dictionary of phases, and update the links
    between the remaining phases.

    Parameters
    ----------
    phases : dict
        The dictionary of phases.

    removed_phase : PhaseInfo
        The phase to be removed.

    redundant_with_phase : PhaseInfo
        The phase with which the removed phase is redundant.
    """
    for key in removed_phase.low_trans:
        if key != redundant_with_phase.key:
            p = phases[key]
            p.high_trans.discard(removed_phase.key)
            redundant_with_phase.addLinkFrom(p)

    for key in removed_phase.high_trans:
        if key != redundant_with_phase.key:
            p = phases[key]
            p.low_trans.discard(removed_phase.key)
            redundant_with_phase.addLinkFrom(p)
    del phases[removed_phase.key]


@dataclass
class PhaseComparison:
    """Summary information about the overlap between two phases.

    The object stores the temperature interval where the phases overlap and the
    corresponding positions of the minima.  This allows helper routines to
    decide whether phases should be merged or kept separate without re-running
    the minimiser.
    """

    phase1: PhaseInfo
    phase2: PhaseInfo
    tmin: float
    tmax: float
    x1_min: np.ndarray
    x2_min: np.ndarray
    x1_max: np.ndarray
    x2_max: np.ndarray
    same_at_tmin: bool
    same_at_tmax: bool


def _phase_overlap_summary(
    phase1: PhaseInfo,
    phase2: PhaseInfo,
    fmin: Callable[[np.ndarray, float], np.ndarray],
    diftol: float,
    verbose: bool,
    conversionFactor: float,
) -> PhaseComparison | None:
    """Compute the overlap and equality checks between two phases."""
    tmax = min(phase1.Tmax, phase2.Tmax)  # lowest of the two highest temperatures
    tmin = max(phase1.Tmin, phase2.Tmin)  # highest of the two lowest temperatures
    if tmin > tmax:  # no overlap in the phases
        return None

    # Last point in the phase 1
    x1_max = phase1.X[-1] if tmax == phase1.Tmax else fmin(phase1.valAt(tmax), tmax)
    # Last point in the phase 2
    x2_max = phase2.X[-1] if tmax == phase2.Tmax else fmin(phase2.valAt(tmax), tmax)
    dif_max = np.linalg.norm(x1_max - x2_max)
    same_at_tmax = dif_max < diftol

    # First point in the phase 1
    x1_min = phase1.X[0] if tmin == phase1.Tmin else fmin(phase1.valAt(tmin), tmin)
    # First point in the phase 2
    x2_min = phase2.X[0] if tmin == phase2.Tmin else fmin(phase2.valAt(tmin), tmin)
    dif_min = np.linalg.norm(x1_min - x2_min)
    same_at_tmin = dif_min < diftol

    if verbose:
        print("Tmax = {:} GeV".format(tmax * conversionFactor))
        print("Phase {:} at Tmax = {:} GeV".format(phase1.key, x1_max * conversionFactor))
        print("Phase {:} at Tmax = {:} GeV".format(phase2.key, x2_max * conversionFactor))
        print(
            "Difference between phase {:} and {:} at Tmax = {:} GeV".format(
                phase1.key, phase2.key, dif_max * conversionFactor
            )
        )
        if same_at_tmax:
            print(
                "This is smaller than the tolerance",
                diftol * conversionFactor,
                "GeV. The phases are considered equal.\n",
            )
        else:
            print("This is larger than the tolerance", diftol * conversionFactor, "GeV.\n")

        print("Tmin = {:} GeV".format(tmin * conversionFactor))
        print("Phase {:} at Tmin = {:} GeV".format(phase1.key, x1_min * conversionFactor))
        print("Phase {:} at Tmin = {:} GeV".format(phase2.key, x2_min * conversionFactor))
        print(
            "Difference between phase {:} and {:} at Tmin = {:} GeV".format(
                phase1.key, phase2.key, dif_min * conversionFactor
            )
        )
        if same_at_tmin:
            print(
                "This is smaller than the tolerance",
                diftol * conversionFactor,
                "GeV. The phases are considered equal.\n",
            )
        else:
            print("This is larger than the tolerance", diftol * conversionFactor, "GeV.\n")

    return PhaseComparison(
        phase1=phase1,
        phase2=phase2,
        tmin=tmin,
        tmax=tmax,
        x1_min=x1_min,
        x2_min=x2_min,
        x1_max=x1_max,
        x2_max=x2_max,
        same_at_tmin=same_at_tmin,
        same_at_tmax=same_at_tmax,
    )


def _merge_full_overlap(
    phases: dict[int | str, PhaseInfo],
    comp: PhaseComparison,
    verbose: bool,
) -> bool:
    """Merge two fully overlapping phases when they coincide at tmin and tmax.

    This mirrors the original CosmoTransitions logic: if two phases agree
    throughout their overlap, keep a single representative that stretches
    across the combined temperature range.  The verbose block recreates the
    informative output from the previous implementation.
    """
    if not (comp.same_at_tmin and comp.same_at_tmax):
        return False

    if verbose:
        print("Found two phases that are the same at tmin and tmax.")
        print("Phases {:} and {:} are redundant.\n".format(comp.phase1.key, comp.phase2.key))

    p_low = comp.phase1 if comp.phase1.Tmin <= comp.phase2.Tmin else comp.phase2
    p_high = comp.phase1 if comp.phase1.Tmax > comp.phase2.Tmax else comp.phase2

    if p_low is p_high:
        p_reject = comp.phase1 if p_low is comp.phase2 else comp.phase2
        _removeRedundantPhase(phases, p_reject, p_low)
        return True

    mask_low = p_low.T <= comp.tmax
    T_low = p_low.T[mask_low]
    X_low = p_low.X[mask_low]
    dXdT_low = p_low.dXdT[mask_low]

    mask_high = p_high.T > comp.tmax
    T_high = p_high.T[mask_high]
    X_high = p_high.X[mask_high]
    dXdT_high = p_high.dXdT[mask_high]

    T = np.append(T_low, T_high, axis=0)
    X = np.append(X_low, X_high, axis=0)
    dXdT = np.append(dXdT_low, dXdT_high, axis=0)

    newkey = f"{p_low.key}-{p_high.key}"
    newphase = PhaseInfo(newkey, X, T, dXdT)
    phases[newkey] = newphase
    _removeRedundantPhase(phases, p_low, newphase)
    _removeRedundantPhase(phases, p_high, newphase)
    return True


def _merge_branch_overlap(
    phases: dict[int | str, PhaseInfo],
    comp: PhaseComparison,
    verbose: bool,
    conversionFactor: float,
) -> bool:
    """Handle the case where phases coincide at tmax but differ at tmin."""
    if not comp.same_at_tmax or comp.same_at_tmin:
        return False

    p_low = comp.phase1 if comp.phase1.Tmin <= comp.phase2.Tmin else comp.phase2
    p_high = comp.phase1 if comp.phase1.Tmax > comp.phase2.Tmax else comp.phase2

    if verbose:
        msg = (
            f"The phases {p_low.key} and {p_high.key} merge at "
            f"T = {comp.tmax * conversionFactor} GeV. "
            "They are identical there, but develop into two different branches at lower temperatures. "
            "This might lead to domain walls, which our code cannot handle yet. "
            "We just remove the branch which dies the quickest.\n"
        )
        console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")

    mask_low = p_low.T <= comp.tmax
    T_low = p_low.T[mask_low]
    X_low = p_low.X[mask_low]
    dXdT_low = p_low.dXdT[mask_low]

    mask_high = p_high.T > comp.tmax
    T_high = p_high.T[mask_high]
    X_high = p_high.X[mask_high]
    dXdT_high = p_high.dXdT[mask_high]

    T = np.append(T_low, T_high, axis=0)
    X = np.append(X_low, X_high, axis=0)
    dXdT = np.append(dXdT_low, dXdT_high, axis=0)

    newkey = f"{p_low.key}-{p_high.key}"
    newphase = PhaseInfo(newkey, X, T, dXdT)
    phases[newkey] = newphase
    _removeRedundantPhase(phases, comp.phase1, newphase)
    _removeRedundantPhase(phases, comp.phase2, newphase)
    return True


def removeRedundantPhases(
    f: Callable,
    df_dx: np.ndarray,
    d2f_dx2: np.ndarray,
    phases: dict[int | str, PhaseInfo],
    xeps=1e-5,
    diftol=1e-2,
    verbose=False,
    conversionFactor=1,
):
    """
    Remove redundant phases from a dictionary output by :func:`traceMultiMin`.

    Although :func:`traceMultiMin` attempts to only trace each phase once, there
    are still instances where a single phase gets traced twice. If a phase is
    included twice, the routines for finding transition regions and tunnelling
    get very confused. This routine follows the original CosmoTransitions
    philosophy while splitting the individual steps into small helpers so they
    can be tested independently.

    Parameters
    ----------
    f : callable
        The scalar potential ``f(x, T)`` that was passed to :func:`traceMultiMin`.
    df_dx, d2f_dx2 : np.ndarray
        First and second derivatives of ``f`` with respect to the fields.
    phases : dict
        Dictionary of :class:`PhaseInfo` objects returned by :func:`traceMultiMin`.
    xeps : float, optional
        Minimisation tolerance in field space when re-minimising at the overlap.
    diftol : float, optional
        Maximum separation between potential minima before they are considered identical.
    verbose : bool, optional
        Emit diagnostic output that mirrors the original implementation.
    conversionFactor : float, optional
        Conversion factor from internal units to GeV for verbose messages.

    Notes
    -----
    If two phases are merged to get rid of redundancy, the resulting phase has
    a key that is a string combination of the two prior keys.
    """

    # We want to make the logic extremely simple at the cost of checking the
    # same thing multiple times.
    # There's just no way this function is going to be the bottle neck.

    def fmin(x, t):  # noqa: ANN202 - NumPy array signature is clear from context
        return np.array(traceLocalMinimum(f, x, t, df_dx, d2f_dx2, xeps))

    while True:
        merged = False  # Track whether any change was made in this pass
        phase_list = list(phases.values())
        for phase1 in phase_list:
            for phase2 in phase_list:
                if phase1.key == phase2.key:
                    continue
                summary = _phase_overlap_summary(
                    phase1,
                    phase2,
                    fmin,
                    diftol,
                    verbose,
                    conversionFactor,
                )
                if summary is None:
                    continue
                if _merge_full_overlap(phases, summary, verbose):
                    merged = True
                    break
                if _merge_branch_overlap(phases, summary, verbose, conversionFactor):
                    merged = True
                    break
                if summary.same_at_tmin:
                    # These phases may still allow tunneling even if they merge
                    # at lower temperatures, so keep both.
                    pass
            if merged:
                break
        if not merged:
            break


class Phases:
    """This class contains the phase tracing routines
    and stores the phase information.
    """

    def __init__(self, pot: generic_potential, verbose=False):
        """Trace all phases of ``pot`` and build their transition graph."""
        self.pot = pot
        self.verbose = verbose
        self.Tmin = pot.Tmin
        self.Tmax = pot.Tmax
        self.x_eps = pot.config.tracingConf.tracing_field_accuracy
        self.T_eps = pot.config.tracingConf.tracing_temp_accuracy
        self.tracingArgs = pot.config.tracingConf.tracing_args
        self.genMirrorPhases = pot.config.tracingConf.gen_mirror_phases
        self.diftol = pot.config.tracingConf.diftol
        self.conversionFactor = pot.conversionFactor
        self.rootPhase = None
        self.T0Phase = None

        self.phases = self.findPhases(self.genMirrorPhases)
        self.testTmax()
        self.buildPhaseGraph(self.phases, pot.Vtot)
        if verbose:
            console.print("\nFound {:} phases:\n".format(len(self.phases)))
            console.print(self.__str__())

    def __str__(self):
        """Return a formatted overview of all phases and their temperature range."""
        tables = []
        for i in self.phases.keys():
            table = rich.table.Table(title="Phase " + str(i), title_justify="left", box=rich.box.ROUNDED)
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="orange1")
            table.add_row("Tmin / GeV", str(self.phases[i].Tmin * self.conversionFactor))
            table.add_row("Tmax / GeV", str(self.phases[i].Tmax * self.conversionFactor))
            table.add_row("phi(Tmin) / GeV", str(self.phases[i].X[0] * self.conversionFactor))
            table.add_row("phi(Tmax) / GeV", str(self.phases[i].X[-1] * self.conversionFactor))
            tables.append(table)
        console.print(RichColumns(tables))
        return ""

    def __getitem__(self, key: str) -> PhaseInfo:
        """Return the :class:`PhaseInfo` object associated with ``key``."""
        return self.phases[key]

    def keys(self) -> list[str]:
        """Expose the keys of the internally stored phases."""
        return self.phases.keys()

    def phase_alias(self, key: int | str) -> str:
        """Return a short user-facing alias such as ``P1`` for one phase key."""
        for idx, candidate in enumerate(self.phases.keys(), start=1):
            if candidate == key:
                return f"P{idx}"
        return str(key)

    def resolve_phase_key(self, key_or_alias: int | str) -> int | str:
        """Resolve a user-facing alias such as ``P1`` back to the internal key."""
        if isinstance(key_or_alias, str):
            alias = key_or_alias.strip().upper()
            if alias.startswith("P") and alias[1:].isdigit():
                idx = int(alias[1:]) - 1
                keys = list(self.phases.keys())
                if 0 <= idx < len(keys):
                    return keys[idx]
        return key_or_alias

    def __len__(self) -> int:
        """Return the number of phases that were traced."""
        return len(self.phases)
   
    def testTmax(self):
        """Test that there is only one phase with Tmax = self.Tmax

        Raises
        ------
        ValueError
            If more than one phase has Tmax = self.Tmax
        """
        Tmax_phases = []
        for p in self.phases.values():
            if abs(p.Tmax - self.Tmax) < self.T_eps:
                Tmax_phases.append(p.key)
        if len(Tmax_phases) > 1:
            msg = f"More than one phase has Tmax = {self.Tmax * self.conversionFactor} GeV: {Tmax_phases}. "
            msg += "This is not allowed. Please increase the Tmax_factor in the model file."
            raise WrongHighTPhaseError(msg)

    def printPhaseTree(self):
        """description

        Parameters
        ----------

        Returns
        -------

        """
        rootPhase = self.rootPhase
        print(f"High-T Phase = {rootPhase}")
        for p in self.phases.values():
            print(f"Phase {p.key} -> {p.childPhases}")

    def findPhases(self, genMirrorPhases: bool = True) -> list[PhaseInfo]:
        """Find different phases as functions of temperature

        Parameters
        ----------
        genMirrorPhases : bool, optional
            If true, generate phases by applying the symmetry transformations
            of the potential to the existing phases.

        Returns
        -------
        list[PhaseInfo]
            Each item in the returned dictionary is an instance of
            :class:`transitionFinder.Phase`, and each phase is
            identified by a unique key. This value is also stored in
            `self.phases`.
        """
        pot = self.pot
        verbose = self.verbose
        # `tracing_args` is primarily forwarded to `traceMinimum` via
        # `single_trace_args`, but we also allow selected `traceMultiMin`
        # controls (dtstart/tjump/local_min_args) to be configured from the
        # same model-level dictionary.
        single_trace_args = dict(self.tracingArgs)
        trace_multi_overrides = {}
        for key in ("dtstart", "tjump", "local_min_args"):
            if key in single_trace_args:
                trace_multi_overrides[key] = single_trace_args.pop(key)

        tracingArgs = dict(
            single_trace_args=single_trace_args,
            forbidCrit=self.pot.forbidPhaseCrit,
            **trace_multi_overrides,
        )
        tstart = self.Tmin
        tstop = self.Tmax

        points = []
        for x0 in pot.approxZeroTMin():
            points.append([x0, tstart])
        tracingArgs_ = dict(forbidCrit=pot.forbidPhaseCrit)
        tracingArgs_.update(tracingArgs)
        
        if verbose:
            console.print(
                f"\nPhase tracing between T = {tstart * self.conversionFactor}" +
                f" GeV and T = {tstop * self.conversionFactor} GeV",
                style="bold green",
            )

        def V_(X, T):
            # speedup: no call to interpolated geff needed here
            return pot.Vtot(X, T, include_radiation=False, include_decoupled=False)

        phases = traceMultiMin(
            V_,  # pot.Vtot,
            pot.gradV,
            pot.dgradV_dT,
            pot.d2V,
            points,
            verbose=verbose,
            conversionFactor=self.conversionFactor,
            applySymmetries=pot.applySymmetries,
            tLow=tstart,
            tHigh=tstop,
            deltaX_target=100 * self.x_eps,
            **tracingArgs_,
        )

        if verbose:
            console.print(f"\nInitial tracing found {len(phases)} phases."
                          f" Removing redundant phases...",
                          style="bold green")

            
        removeRedundantPhases(
            pot.Vtot, pot.gradV, pot.d2V, phases, self.x_eps * 1e-2, diftol=self.diftol,
            verbose=verbose, conversionFactor=self.conversionFactor
        )
        
        if verbose:
            console.print("After removing redundant phases, {:} phases remain.".format(len(phases)),
                          style="bold green")
        
        if genMirrorPhases:
            generateMirrorPhases(phases, self.diftol, pot.invGroupElements)
        return phases

    def buildPhaseGraph(self, in_phases: dict[int | str, PhaseInfo], V: Callable):
        """Construct a tree of the phase history starting from the high
        temperature phase.

        Parameters
        ----------
        in_phases : list[PhaseInfo]
            List of phases
        V : callable
            The temperature dependent effective potential V(X, T)

        Returns
        -------
        None."""
        # First find the start and end phases
        rootPhase = self.getStartPhase(in_phases, V)
        self.rootPhase = rootPhase
        T0Phase = self.getT0Phase(in_phases, V)
        self.T0Phase = T0Phase

        phases = []
        for key in in_phases.keys():
            phases.append(key)
        for startPhase in phases:
            for endPhase in phases:
                if startPhase == endPhase:
                    continue

                tmax = min(in_phases[startPhase].Tmax, in_phases[endPhase].Tmax)
                tmin = max(in_phases[startPhase].Tmin, in_phases[endPhase].Tmin)

                if tmin >= tmax:
                    # no overlap, except if 2nd order transition
                    if in_phases[endPhase].key in in_phases[startPhase].low_trans:
                        in_phases[endPhase].parentPhase = startPhase
                        in_phases[startPhase].childPhases.append(endPhase)
                    continue

                def DV(T):
                    return V(in_phases[startPhase].valAt(T), T) - V(in_phases[endPhase].valAt(T), T)

                DVmin = DV(tmin)

                if DVmin < 0:
                    # No tunneling possible
                    continue

                in_phases[startPhase].childPhases.append(endPhase)
                in_phases[endPhase].parentPhases.append(startPhase)

    def getStartPhase(self, phases: dict[int | str, PhaseInfo], V: Callable = None) -> int | str:
        """
        Find the key for the high-T phase.

        Parameters
        ----------
        phases : dict
            Output from :func:`traceMultiMin`.
        V : callable
            The potential V(x,T). Only necessary if there are
            multiple phases with the same Tmax.
            
        Returns
        -------
        int | str :
            The key of the high-T phase.
        """
        startPhases = []
        startPhase = None
        Tmax = None
        assert len(phases) > 0
        for i in list(phases.keys()):
            if phases[i].T[-1] == Tmax:
                # add this to the startPhases list.
                startPhases.append(i)
            elif Tmax is None or phases[i].T[-1] > Tmax:
                startPhases = [i]
                Tmax = phases[i].T[-1]
        if len(startPhases) == 1 or V is None:
            startPhase = startPhases[0]
        else:
            # more than one phase have the same maximum temperature
            # Pick the stable one at high temp.
            Vmin = None
            for i in startPhases:
                V_ = V(phases[i].X[-1], phases[i].T[-1])
                if Vmin is None or V_ < Vmin:
                    Vmin = V_
                    startPhase = i
        assert startPhase in phases
        return startPhase

    def getT0Phase(self, phases : dict[int | str, PhaseInfo], V: Callable) -> int | str:
        """Find the stable phase at T = 0.

        Parameters
        ----------
        phases : list[PhaseInfo]
            List of the phases
        V : callable
            The effective potential as V(X, T)

        Returns
        -------
        int | str:
            The key of the stable phase at T = 0."""
        endPhases = []
        endPhase = None
        Tmin = None
        assert len(phases) > 0
        for i in list(phases.keys()):
            if Tmin is None or phases[i].Tmin < Tmin:
                Tmin = phases[i].Tmin

        for i in list(phases.keys()):
            if phases[i].Tmin == Tmin:
                endPhases.append(i)

        if len(endPhases) == 1:
            endPhase = endPhases[0]
        else:
            Vmin = None
            for i in endPhases:
                V_ = V(phases[i].valAt(Tmin), Tmin)
                if Vmin is None or V_ < Vmin:
                    Vmin = V_
                    endPhase = i
        assert endPhase in phases
        return endPhase

# ==================================================
# Routines for plotting the phase history
# ==================================================


@dataclass(frozen=True)
class PhaseSegment:
    """Piecewise representation of the thermal history across a single phase."""

    index: int
    phase_key: int | str
    phase: PhaseInfo
    entry_temperature: float
    exit_temperature: float

    def bounds(self) -> tuple[float, float]:
        """Return the temperature interval for which this phase is active."""
        low = min(self.entry_temperature, self.exit_temperature)
        high = max(self.entry_temperature, self.exit_temperature)
        return low, high

    def temperature_grid(
        self,
        lower: float | None = None,
        upper: float | None = None,
        min_points: int = 50,
    ) -> np.ndarray:
        """Create a temperature grid within the active interval for evaluation."""
        low, high = self.bounds()
        if lower is not None:
            low = max(low, lower)
        if upper is not None:
            high = min(high, upper)
        if high <= low:
            return np.array([])

        raw = self.phase.T
        mask = (raw >= low) & (raw <= high)
        temps = raw[mask]  # temperatures already in the phase trace
        if temps.size >= 2:  # enough points already
            temps = np.concatenate((temps, [low, high]))
            temps = np.unique(temps)
        else:
            temps = np.linspace(low, high, min_points)
        return temps

    def __call__(self, T: float | np.ndarray, deriv: int = 0) -> np.ndarray:
        """Evaluate the phase minimum at temperature ``T``."""
        return self.phase.valAt(T, deriv=deriv)


class TransitionHistory:
    """Ordered collection of ``PhaseSegment`` objects describing the evolution."""

    def __init__(self, segments: Iterable[PhaseSegment]):
        """Store the ordered list of phase segments traversed during cooling."""
        self._segments = list(segments)

    def __len__(self) -> int:
        """Return the number of stored phase segments."""
        return len(self._segments)

    def __iter__(self):
        """Iterate over the ordered phase segments."""
        return iter(self._segments)

    def __getitem__(self, index: int) -> PhaseSegment:
        """Return the phase segment at ``index``."""
        return self._segments[index]

    def keys(self):
        """Expose indices for iteration, matching list semantics."""
        return range(len(self._segments))

    def items(self):
        """Yield ``(index, PhaseSegment)`` pairs in order."""
        for idx, segment in enumerate(self._segments):
            yield idx, segment

    def values(self):
        """Return the ordered list of segments."""
        return list(self._segments)

    def phase_keys(self) -> list[int | str]:
        """Return the ordered phase identifiers along the history."""
        return [segment.phase_key for segment in self._segments]

    @property
    def temperature_span(self) -> tuple[float, float]:
        """Return the overall minimum and maximum temperature that are covered."""
        if not self._segments:
            return (0.0, 0.0)
        lows, highs = zip(*(segment.bounds() for segment in self._segments))
        return (min(lows), max(highs))


def reconstructTransitionHistory(phases: "Phases",
                                 transitions: Iterable["TransitionInfo"]) -> TransitionHistory:
    """Construct the ordered thermal history using the traced phase graph.

    Parameters
    ----------
    phases : Phases
        The traced phases together with their transition graph.
    transitions : Iterable[TransitionInfo]
        The ordered list of realised transitions starting from the high-T phase.

    Returns
    -------
    TransitionHistory
        An ordered, piecewise representation of the vacuum expectation values
        that can be evaluated via the individual :class:`PhaseSegment` objects.
    """
    if phases is None:
        raise ValueError("No phases supplied to reconstructTransitionHistory.")

    transition_list = list(transitions)

    if not hasattr(phases, "rootPhase"):
        msg = "reconstructTransitionHistory now expects a Phases instance with a built phase graph."
        raise AttributeError(msg)

    if len(transition_list) == 0:
        root_key = phases.rootPhase
        root_phase = phases[root_key]
        segment = PhaseSegment(
            index=0,
            phase_key=root_phase.key,
            phase=root_phase,
            entry_temperature=float(root_phase.Tmax),
            exit_temperature=float(root_phase.Tmin),
        )
        return TransitionHistory([segment])

    start_key = phases.rootPhase
    if transition_list[0].high_phase != start_key:
        msg = (
            "Transition list does not start at the high-temperature phase. "
            f"Expected high phase {start_key}, received {transition_list[0].high_phase}."
        )
        raise ValueError(msg)

    phase_keys: list[int | str] = [start_key]
    current_key = start_key
    for tr in transition_list:
        if tr.high_phase != current_key:
            msg = (
                "Inconsistent transition ordering: "
                f"expected high phase {current_key}, received {tr.high_phase}."
            )
            raise ValueError(msg)
        phase_keys.append(tr.low_phase)
        current_key = tr.low_phase

    def _entry_temperature(tr: "TransitionInfo") -> float:
        if getattr(tr, "type", None) == 2:
            return float(tr.Tcrit)
        return float(tr.derived_params.get("Treh", tr.Tcrit))

    def _exit_temperature(tr: "TransitionInfo") -> float:
        if getattr(tr, "type", None) == 2:
            return float(tr.Tcrit)
        return float(tr.derived_params.get("Tperc", tr.Tcrit))

    segments: list[PhaseSegment] = []
    for idx, phase_key in enumerate(phase_keys):
        phase = phases[phase_key]
        if idx == 0:
            entry_temp = float(phase.Tmax)
        else:
            entry_temp = _entry_temperature(transition_list[idx - 1])
        if idx == len(phase_keys) - 1:
            exit_temp = float(phase.Tmin)
        else:
            exit_temp = _exit_temperature(transition_list[idx])

        entry_temp = float(np.clip(entry_temp, phase.Tmin, phase.Tmax))
        exit_temp = float(np.clip(exit_temp, phase.Tmin, phase.Tmax))

        segments.append(
            PhaseSegment(
                index=idx,
                phase_key=phase.key,
                phase=phase,
                entry_temperature=entry_temp,
                exit_temperature=exit_temp,
            )
        )

    return TransitionHistory(segments)
