"""The energy ordering a transition's two phases must satisfy.

A first-order transition releases energy and happens below the critical temperature, so the
broken phase into which the bubbles grow is reheated. The checks here assert that, separately
from the numerical helpers in :mod:`helper_functions` and from the profile solvers that call
them.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import warnings

import numpy as np

from transitionlistener import errors

#: Relative slack on the broken-phase temperature, measured rather than configured.
#:
#: ``T_bro > T_sym`` holds by construction, so the only way a sound transition can come out
#: reversed is that the root solve behind ``T_bro`` sets two nearly equal energy densities
#: against each other and double precision loses the sign of the difference.
#:
#: Measured over 2313 step-3 support points on the 2HDM line ``lambda3 = 5.5 .. 5.87``: where
#: the relative latent heat comes out negative on a correctly ordered pair, the worst case is
#: ``-5.91e-14``. The one genuinely reversed point carried ``-1.264e-04``. ``1e-09`` sits five
#: orders above the round-off and four and a half below the real inversion.
BROKEN_TEMPERATURE_ROUNDOFF = 1e-9


def enforce_broken_not_below_symmetric(T_bro, T_sym, *,
                                       where: str = "the percolation profile",
                                       e_sym=None, measure_broken_energy=None,
                                       phase_symmetric=None, phase_broken=None):
    """Assert that the broken phase is the warmer of the two, and say which phases if not.

    ``T_bro > T_sym`` is a property of the physics, not an accuracy the solver achieves, so a
    reversed ordering means one of two things and the size of the shortfall tells them apart.
    Below :data:`BROKEN_TEMPERATURE_ROUNDOFF` the latent heat was too small for double
    precision to keep its sign and ``T_bro`` is clamped with a warning; above it the profile is
    refused as a :class:`errors.PercolationError` naming both phases and the latent heat.

    Parameters
    ----------
    T_bro, T_sym : float
        Broken- and symmetric-phase temperatures at the same support point, in GeV.
    where : str
        Named in the warning and the error, so the call sites stay distinguishable.
    e_sym : float, optional
        Energy density of the symmetric phase at ``T_sym``, in GeV^4.
    measure_broken_energy : callable, optional
        Returns the broken phase's energy density **at** ``T_sym``, in GeV^4. It must be
        measured there: the root solve sets ``e_bro(T_bro) = e_sym(T_sym)``, so read at the
        temperature that solve returned the latent heat is identically zero. Called only when
        the profile is refused, and its own failure cannot replace the error being reported.
    phase_symmetric, phase_broken : optional
        The two phases being paired; their ``key`` is named in the error, which is what points
        at a mispairing rather than at the root solve.

    Returns
    -------
    tuple[float, float]
        The temperature to use, and the relative shortfall that was clamped, zero if none.

    Raises
    ------
    errors.PercolationError
        If either temperature is not finite, or the shortfall exceeds
        :data:`BROKEN_TEMPERATURE_ROUNDOFF`.
    """
    t_bro = float(T_bro)
    t_sym = float(T_sym)
    # `not (t_bro < t_sym)` is True for a nan, so a nan would pass an ordering test unnoticed.
    if not (np.isfinite(t_bro) and np.isfinite(t_sym)):
        raise errors.PercolationError(
            f"{_pair_description(phase_symmetric, phase_broken)} reached {where} with "
            f"T_bro = {t_bro!r} and T_sym = {t_sym!r}, at least one of which is not a finite "
            "temperature, so the two cannot be ordered. The step-3 root solve for the "
            "reheated broken-phase temperature is the usual source."
        )
    if t_bro >= t_sym:
        return t_bro, 0.0
    scale = abs(t_sym) if t_sym != 0.0 else 1.0
    shortfall = (t_sym - t_bro) / scale
    if shortfall <= BROKEN_TEMPERATURE_ROUNDOFF:
        warnings.warn(
            f"T_bro fell {shortfall:.3e} below T_sym in {where}, which is round-off on a "
            f"latent heat too small for double precision to keep the sign of (the threshold "
            f"is {BROKEN_TEMPERATURE_ROUNDOFF:.3g} relative); clamping T_bro to T_sym.",
            RuntimeWarning,
            stacklevel=2,
        )
        return t_sym, shortfall
    raise errors.PercolationError(
        f"T_bro = {t_bro!r} GeV is below T_sym = {t_sym!r} GeV in {where}, short by "
        f"{shortfall:.3e} relative, {shortfall / BROKEN_TEMPERATURE_ROUNDOFF:.3g} times the "
        f"{BROKEN_TEMPERATURE_ROUNDOFF:.3g} that round-off on the energy difference behind "
        "T_bro can account for. "
        "The transition releases energy and happens below the critical temperature, so the "
        "broken phase is reheated and cannot be the colder of the two. "
        f"{_pair_description(phase_symmetric, phase_broken)} carries "
        f"{_latent_heat_description(e_sym, _measured(measure_broken_energy))} at T_sym, so "
        "this pair of phases does not release energy there and cannot be the transition. "
        "A phase-chain walk that "
        "paired the symmetric phase with a broken phase other than the one the transition "
        "ends in is the cause seen so far."
    )


def _measured(measure):
    """Call `measure` for the error message, returning None rather than raising.

    `errors.Timeout` still propagates: it is raised from the `SIGALRM` handler and is an
    ordinary `Exception`, so swallowing it would report an aborted run as a physics failure.
    """
    if measure is None:
        return None
    try:
        return float(measure())
    except errors.Timeout:
        raise
    except Exception:
        return None


def _pair_description(phase_symmetric, phase_broken) -> str:
    """Name the two phases being paired, for the guard's messages."""
    sym = getattr(phase_symmetric, "key", None)
    bro = getattr(phase_broken, "key", None)
    if sym is None and bro is None:
        return "The phase pair"
    return f"The phase pair {sym} -> {bro}"


def _latent_heat_description(e_sym, e_bro) -> str:
    """Describe the relative latent heat of a phase pair, for the guard's error message."""
    if e_sym is None or e_bro is None:
        return "a latent heat that was not measured here"
    e_sym = float(e_sym)
    e_bro = float(e_bro)
    if not (np.isfinite(e_sym) and np.isfinite(e_bro)) or e_sym == 0.0:
        return f"energy densities e_sym = {e_sym!r} GeV^4 and e_bro = {e_bro!r} GeV^4"
    relative = (e_sym - e_bro) / abs(e_sym)
    return (f"a relative latent heat (e_sym - e_bro) / |e_sym| = {relative:+.4e} "
            f"(e_sym = {e_sym!r} GeV^4, e_bro = {e_bro!r} GeV^4)")
