"""
This module is used to calculate the bubble dynamics in the dark sector.
It contains functions to calculate the energy density, effective degrees of freedom,
the Hubble parameter, the bubble nucleation rate, and from this the nucleation
and percolation temperature.

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
from scipy import integrate

from transitionlistener import thermodynamics as td
from transitionlistener import constants as cn
from transitionlistener.helper_functions import derivative
from transitionlistener import errors
from transitionlistener.pathDeformation import bounceAction
from transitionlistener.finiteT import Jb_spline as Jb
from transitionlistener.finiteT import Jf_spline as Jf


def g_eff_DS(T_DS: float, pot, phase) -> float:
    """Calculate the effective energy degrees of freedom in the dark sector.

    Parameters
    ----------
    T : float
        Temperature at which to evaluate g_eff
    pot : generic_potential
        The effetive potential object
    phase : PhaseInfo
        The phase of the system, either symmetric or broken.

    Returns
    ----------
    float
        Effective degrees of freedom in the dark sector."""
    try:
        vevT = phase.valAt(T_DS)
    except BaseException:
        print("Warning: TBRO is too low for interpolation of vev, using T = 0 value")
        # Temperature is to low for interpolation of vev, use T = 0 value
        vevT = pot.X0
    bosons = pot.boson_massSq(vevT, 0)*(~pot.mass_spectrum.is_SM_bosons)
    fermions = pot.fermion_massSq(vevT)*(~pot.mass_spectrum.is_SM_fermions)
    geff = td.e_geffDS(bosons, fermions, T_DS)
    return geff


def h_eff_DS(T_DS: float, pot, phase) -> float:
    """Calculate the effective entropy degrees of freedom in the dark sector.

    Parameters
    ----------
    T : float
        Temperature at which to evaluate g_eff
    pot : generic_potential
        The effetive potential object
    phase : PhaseInfo
        The phase of the system, either symmetric or broken.

    Returns
    ----------
    geff : float
        Effective degrees of freedom in the dark sector."""
    try:
        vevT = phase.valAt(T_DS)
    except BaseException:
        print("Warning: TBRO is too low for interpolation of vev, using T = 0 value")
        # Temperature is to low for interpolation of vev, use T = 0 value
        vevT = pot.X0

    # Set SM masses to zero:
    bosons = pot.boson_massSq(vevT, 0)*(~pot.mass_spectrum.is_SM_bosons)
    fermions = pot.fermion_massSq(vevT)*(~pot.mass_spectrum.is_SM_fermions)
    geff = td.s_geffDS(bosons, fermions, T_DS)
    return geff


def energyDensity(pot, phase, T: float | np.ndarray, include_decoupled=True) -> float | np.ndarray:
    r"""This function calls the implementation in the effective potential.

    Parameters
    ----------
    pot : generic_potential
        Effective potential
    phase : PhaseInfo
        The phase information
    T : float|np.ndarray
        The temperature at which to compute the energy density
    include_decoupled : bool, optional
        If true, also account for the energy density in a decoupled sector

    Returns
    -------
    float|np.ndarray :
        The energy density at `T`."""
    X = phase.valAt(T)
    return pot.energyDensity(X, T, include_decoupled=include_decoupled)


def Gamma(T: float | np.ndarray, S: float | np.ndarray) -> np.ndarray:
    """Calculate the bubble nucleation rate.

    Parameters
    ----------
    T : float | np.ndarray
        Symmetric phase temperature
    S : float | np.ndarray
        The action at temperature T

    Returns
    ----------
    Gamma : np.ndarray
        The bubble nucleation rate."""
    S = np.atleast_1d(S)
    T = np.atleast_1d(T)
    result = np.zeros_like(T, dtype=float)

    mask_zero = S == 0
    mask_inf = np.isinf(S)
    mask_valid = ~(mask_zero | mask_inf)

    result[mask_zero] = np.inf
    result[mask_inf] = np.nan
    # Ignore RuntimeWarnings in sqrt and exp (e.g., for invalid or overflow values)
    with np.errstate(invalid="ignore", over="ignore"):
        result[mask_valid] = (
            T[mask_valid] ** 4
            * np.sqrt(S[mask_valid] / (2 * np.pi * T[mask_valid])) ** 3
            * np.exp(-S[mask_valid] / T[mask_valid])
        )

    return result


def logGamma(T: float | np.ndarray, S: float | np.ndarray) -> np.ndarray:
    """Calculate the log10 of the bubble nucleation rate.

    Parameters
    ----------
    T : float | np.ndarray
        Symmetric phase temperature
    S : float | np.ndarray
        The action at temperature T

    Returns
    ----------
    Gamma : np.ndarray
        The bubble nucleation rate."""
    S = np.atleast_1d(S)
    T = np.atleast_1d(T)
    result = np.zeros_like(T, dtype=float)

    mask_zero = S == 0
    mask_inf = np.isinf(S)
    mask_valid = ~(mask_zero | mask_inf)

    result[mask_zero] = np.inf
    result[mask_inf] = np.nan
    # Ignore RuntimeWarnings in sqrt and exp (e.g., for invalid or overflow values)
    with np.errstate(invalid="ignore", over="ignore"):
        result[mask_valid] = (
            4 * np.log(T[mask_valid])
            + 3 / 2 * np.log(S[mask_valid] / (2 * np.pi * T[mask_valid]))
            - S[mask_valid] / T[mask_valid]
        )

    return result


def HubbleParameter(rho: float | np.ndarray, CF: float) -> float:
    """Hubble parameter.

    Parameters
    ----------
    rho : float
        The energy density of the unverse.
    CF : float
        Conversion factor to convert internal units to GeV.
    Returns
    -------
    float :
        The Hubble parameter."""
    return np.sqrt(8 * np.pi / 3 * rho) / (cn.Mpl_GeV / CF)


def scalefactorRatio(pot, T1, T2, X) -> float:
    """Compute the scalefactor ratio a(T1)/a(T2)

    Parameters
    ----------

    Returns
    -------

    """
    if T1 == T2:
        return 1
    n = 100
    Trange = np.linspace(T1, T2, n)
    dVdT = np.array([pot.dVdT(X, T, dT=0.1) for T in Trange])
    d2VdT2 = np.array([pot.d2VdT2(X, T) for T in Trange])

    intgr = integrate.trapezoid(d2VdT2/(3*dVdT), x=Trange)
    return np.exp(intgr)

def calcAction(pot, T: float, start_phase, end_phase, outdict: dict, verbose: bool = False,
               phitol: float = 1e-6) -> float:
    """Calculate the action at temperature `T`

    Parameters
    ----------
    pot : generic_potential
        The effective potential object
    T : float
        The temperature
    start_phase : PhaseInfo
        The information about the high temperature phase
    end_phase : PhaseInfo
        The information about the low temperature phase
    outdict : dict
        The dictionary storing the action evaluations with the key `T`.
    verbose : bool, optional
        Set the output level
    phitol : float, optional
        Set the accuracy of the minimisation of the potential minima.

    Returns
    -------
    float :
        The action at temperature `T`."""

    x0 = findLocalMinimum(start_phase.valAt(T), T, pot.gradV, pot.d2V)
    x1 = findLocalMinimum(end_phase.valAt(T), T, pot.gradV, pot.d2V)
    tdict = dict(low_vev=x1, high_vev=x0)

    if T in outdict:
        return outdict[T]["action"]

    outdict = bounceAction(
        T, pot.Vtot, pot.gradV, outdict, tdict, verbose, pot.conversionFactor,
        **pot.config.tracingConf.tunneling_params)
    return outdict[T]["action"]


def approxPercIntegral(
    Tperc: float,
    logG: float,
    Tnuc: float,
    beta: float,
    alphasq: float,
    pot,
    phase,
    verbose: bool = False,
) -> float:
    r"""Calculate the percolation integral up to the gaussian
    approximation.

    .. math::
      I = \int_{-\infty}^{t_p} d t (t_p - t) \Gamma_p
      \exp(\beta (t - t_p) - a^2/2(t - t_p)^2)

    Parameters
    ----------
    Tperc : float,
        The percolation temperature
    logG : float
        :math:`\log(\Gamma_p)`, the log of the nucleation rate
    Tnuc : float
        The nucleation temperature
    beta : float
        First derivative of the action w.r.t. time: :math:`d/dt (S_3/T)`
    alphasq : float
        Second derivative of the action w.r.t. time :math:`d^2/dt^2 (S_3/T)`
    pot : "generic_potential"
        The effective potential
    phase: PhaseInfo
        The current phase
    verbose: bool, optional
        Set the output level
    Returns
    -------
    float:
        The percolation interal.
    """
    assert Tperc <= Tnuc

    # T to temperature:
    Trange = np.linspace(Tperc, Tnuc, num=200)
    rho_range = np.zeros_like(Trange)
    for i, T in enumerate(Trange):
        rho_range[i] = energyDensity(pot, phase, T)
    integr = 1 / (Trange * HubbleParameter(rho_range, pot.conversionFactor))
    trange = integrate.cumulative_trapezoid(integr, x=Trange, initial=0)
    trange = trange - trange[-1]  # set tperc = 0

    def fG(t):
        res = (-t) ** 3 * np.exp(logG + beta * t - 1 / 2 * alphasq * t**2)
        return res

    Iperc = integrate.trapezoid(fG(trange), x=trange)
    return Iperc


def percIntegral(T: np.ndarray, H: np.ndarray, S: np.ndarray, vw=1.0) -> float:
    """Perform the percolation integral beteen Tstart and Tend.

    Parameters
    ----------
    T : np.ndarray
        Temperature of the dark sector symmetric phase. It must start with the
        nucleation temperature and go down to the percolation temp.
    H : np.ndarray
        Hubble rate evaluated at T
    S : np.ndarray
        S3 Euclidian bounce action evaluated at T
    vw : float

    Returns
    -------
    float
        The integral evaluated at the last temperature ``T[-1]``.
    """
    # See eq. (4.57) in 2305.02357
    # The ordering of the array is important
    if len(T) > 1:
        if not T[0] >= T[1]:
            raise errors.PercolationError("T is not decreasing in the percolation integral.")

    vol_int = np.array([integrate.trapezoid(1 / H[i:], x=T[i:]) for i in range(len(T))])
    with np.errstate(invalid="ignore"):
        integrant = Gamma(T, S) / T**4 / H * vol_int**3
    y = integrate.trapezoid(np.nan_to_num(integrant), x=T)
    return 4 * np.pi / 3 * vw**3 * y


def percIntegralwExp(T: np.ndarray, H: np.ndarray, S: np.ndarray,
                              scaleF_ratio: np.ndarray,
                              soundSpeedSq: np.ndarray,
                              vw=1.0) -> float:
    """Perform the percolation integral beteen Tstart and Tend.
    Take the expansion of the universe into account.

    Parameters
    ----------
    T : np.ndarray
        Temperature of the dark sector symmetric phase. It must start with the
        nucleation temperature and go down to the percolation temp.
    H : np.ndarray
        Hubble rate evaluated at T
    S : np.ndarray
        S3 Euclidian bounce action evaluated at T
    vw : float

    Returns
    -------
    float
        The integral evaluated at the last temperature ``T[-1]``.
    """
    # See eq. (4.57) in 2305.02357
    # The ordering of the array is important
    if len(T) > 1:
        if not T[0] >= T[1]:
            raise errors.PercolationError("T is not decreasing in the percolation integral.")

    vol_int = np.zeros_like(T)
    for i in range(len(T)):
        vol_int[i] = integrate.trapezoid(1/(3*H[i:]*T[i:]*soundSpeedSq[i:]*scaleF_ratio[i:]),
                                         x=T[i:])
    with np.errstate(invalid="ignore"):
        integrant = Gamma(T, S) * scaleF_ratio**3 / (3*H*T*soundSpeedSq) * vol_int**3
    y = integrate.trapezoid(np.nan_to_num(integrant), x=T)
    return 4 * np.pi / 3 * vw**3 * y


def approxNucleationCriterion(T_DS: float, S: float, pot, phase_sym, phase_bro) -> float:
    r"""
    Calculate the nucleation criterion for a given temperature and action.
    This function computes the nucleation criterion based on the following equation:

    .. math::

        \frac{\Gamma}{H^4} = 1

    where :math:`\Gamma` denotes the bubble nucleation rate and :math:`H`
    is the Hubble parameter during radiation domination.

    Parameters
    ----------
    T_DS : float
        Temperature of the DS.
    S : float
        Action at temperature TDS
    pot : generic_potential
        The effective potential object
    phase_sym : PhaseInfo
        The information about the high temperature phase
    phase_bro : PhaseInfo
        The information about the low temperature phase

    Returns
    -------
    float
        Result of the nucleation criterion, 0 if fulfilled.
    """
    T_DS += 1e-100

    if S == -np.inf:
        return np.inf
    elif S == np.inf:
        return -np.inf
    # This approximation identifies the PT-sector temperature with T_DS and
    # keeps any hidden-sector temperature ratio temperature independent.
    T = T_DS

    rho = energyDensity(pot, phase_sym, T)
    H = HubbleParameter(rho, pot.conversionFactor)
    logG = logGamma(T_DS, S)[0]  # Returnvalue is an array
    crit = logG - 4 * np.log(H)
    # crit is > 0 if more than one bubble is nucleated until T
    # crit is < 0 if less than one bubble is nucleated until T

    return crit


def ApproxPercCritSecondOrder(
    T: float,
    Tnuc: float,
    outdict: dict,
    pot,
    phase_sym,
    phase_bro,
    Tmax: float,
    Tmin: float,
    vw: float,
    verbose: bool = False,
) -> float:
    r"""Calculate a an approximation of the percolation temperature.
    This uses the saddlepoint approximation of

    .. math::
        \Gamma = \Gamma_p e^{\beta (t - t_p) - \alpha^2 / 2 (t - t_p)^2}

    Important: The criterion returns a value < 0 if the temperature is too high
    and a value > 0 if the temperature is too small. It returns 0 if the criterion
    is met.

    If the action becomes inf at a temperature, this has to be because
    the S/T function is u-shaped and must occur at the Tmin end, since
    at Tnuc the action is finite. Therefore also return np.inf for
    S/T = inf.

    Parameters
    ----------
    T : float
        The temperature.
    Tnuc : float
        The nucleation temperature
    outdict : dict
        Dictionary storing the action evaluation
    pot : generic_potential
        The effective potential object
    phase_sym : PhaseInfo
        The high temperature phase information
    phase_bro : PhaseInfo
        The low temperature phase information
    Tmax : float
        The maximal temperature, where to compute the nucleation criterion (usually Tnuc)
    Tmin : float
        The minimal temperature where to compute the nucleation criterion, usually
        the minimal temp. of coexistence of the phases.
    vw : float
        The bubble wall velocity.
    verbose : bool, optional
        Set the output level

    Returns
    -------
    float
        Zero if the criterion is met for ``T``; negative if ``T`` is too large, positive if it is too small."""
    fperc = 0.34  # - log(P = 0.7) = 0.34
    dT = np.maximum(T * 1e-3, 1e-3)  # Try this for numerical stability
    Tr = np.zeros(3)
    Sr = np.zeros(3)
    logGr = np.zeros(3)


    Sr[0] = calcAction(pot, T + dT, phase_sym, phase_bro, outdict)
    Sr[1] = calcAction(pot, T, phase_sym, phase_bro, outdict)
    Sr[2] = calcAction(pot, T - dT, phase_sym, phase_bro, outdict)

    # If S/T is inf then the temperature must be too small
    if np.isinf(Sr[1] / T):
        return np.inf

    if np.isinf(Sr[0] / T):
        # It could be that the temperature interval was chosen too large
        while np.isinf(Sr[0]) and dT / T > 1e-10:
            dT *= 0.1
            Sr[0] = calcAction(pot, T + dT, phase_sym, phase_bro, outdict)
        if np.isinf(Sr[0] / T):
            # If we still have an inf action, then the temperature is too small
            return np.inf

    Tr[0] = T + dT
    Tr[1] = T
    Tr[2] = T - dT
    logGr[0] = logGamma(Tr[0], Sr[0])
    logGr[1] = logGamma(Tr[1], Sr[1])
    logGr[2] = logGamma(Tr[2], Sr[2])
    rho = energyDensity(pot, phase_sym, T)
    H = HubbleParameter(rho, pot.conversionFactor)
    betaH = -T * (logGr[0] - logGr[1]) / dT

    dSTdT = (Sr[0] / Tr[0] - Sr[1] / Tr[1]) / dT
    d2STdT2 = (Sr[0] / Tr[0] - 2 * Sr[1] / Tr[1] + Sr[2] / Tr[2]) / dT**2

    dHdT = (HubbleParameter(energyDensity(pot, phase_sym, T + dT), pot.conversionFactor) - H) / dT
    alphaH2 = -(T**2) * d2STdT2 - (T**2 * dHdT / H + T) * dSTdT

    beta = betaH * H
    asq = -alphaH2 * H**2
    Iperc = 4 * np.pi / 3 * vw**3
    Iperc *= approxPercIntegral(T, logGr[1], Tnuc, beta, asq, pot, phase_sym, verbose)
    crit = np.log(fperc / (Iperc + 1e-300))

    return crit


def ApproxPercolCriterion(
    T: float, outdict: dict, pot, phase_sym, phase_bro, Tmax: float, Tmin: float, vw: float, verbose: bool = False
) -> float:
    r"""Calculate a rough approximation of the percolation temperature.
    This uses the saddlepoint approximation of

    .. math::

        \Gamma = \Gamma_0 \exp\bigl[-\beta (t - t_0)\bigr]

    Important: The criterion returns a value < 0 if the temperature is too high
    and a value > 0 if the temperature is too small. It returns 0 if the criterion
    is met.

    If the action becomes inf at a temperature, this has to be because
    the S/T function is u-shaped and must occur at the Tmin end, since
    at Tnuc the action is finite. Therefore also return np.inf for
    S/T = inf.

    Parameters
    ----------
    T : float
        The temperature.
    outdict : dict
        Dictionary storing the action evaluation
    pot : generic_potential
        The effective potential object
    phase_sym : PhaseInfo
        The high temperature phase information
    phase_bro : PhaseInfo
        The low temperature phase information
    Tmax : float
        The maximal temperature, where to compute the nucleation criterion (usually Tnuc)
    Tmin : float
        The minimal temperature where to compute the nucleation criterion, usually
        the minimal temp. of coexistence of the phases.
    vw : float
        The bubble wall velocity.
    verbose : bool, optional
        Set the output level

    Returns
    -------
    float
        Zero if the criterion is met for ``T``; negative if ``T`` is too large, positive if it is too small.
    """
    # Check if T is in the range of the potential
    if T >= Tmax:
        return -np.inf
    elif T <= Tmin:
        return np.inf

    # Calculate beta/H
    fperc = 0.34  # - log(P = 0.7) = 0.34
    dT = np.maximum(T * 1e-2, 1e-3)  # Try this for numerical stability
    if T + dT > Tmax:
        dT = (Tmax - T) / 100.0
    Tr = np.zeros(2)
    Sr = np.zeros(2)
    Gr = np.zeros(2)
    Sr[0] = calcAction(pot, T + dT, phase_sym, phase_bro, outdict)
    Sr[1] = calcAction(pot, T, phase_sym, phase_bro, outdict)

    # If S/T is inf then the temperature must be too small
    if np.isinf(Sr[1] / T):
        return np.inf

    if np.isinf(Sr[0] / T):
        # It could be that the temperature interval was chosen too large
        while np.isinf(Sr[0]) and dT / T > 1e-10:
            dT *= 0.1
            Sr[0] = calcAction(pot, T + dT, phase_sym, phase_bro, outdict)
        if np.isinf(Sr[0] / T):
            # If we still have an inf action, then the temperature is too small
            return np.inf

    Tr[0] = T + dT
    Tr[1] = T
    Gr[0] = Gamma(Tr[0], Sr[0])
    Gr[1] = Gamma(Tr[1], Sr[1])
    rho = energyDensity(pot, phase_sym, T)
    H = HubbleParameter(rho, pot.conversionFactor)
    betaH = -T * (np.log(Gr[0]) - np.log(Gr[1])) / dT

    # if betaH becomes negative, then the temperature
    # is too small. Return positive infinity
    if betaH < 0:
        return np.inf

    crit = np.power(Gr[1], 1 / 4.0) / H / betaH / np.power(fperc / (8 * np.pi * vw**3), 1 / 4.0)
    if verbose:
        print(
            f"T = {T * pot.conversionFactor} GeV, S3 = {Sr[1] * pot.conversionFactor} GeV,"
            f"S3/T = {Sr[1] / T}, Gamma = {Gr[1] * pot.conversionFactor**4} GeV^4, betaH = {betaH:2.5g}"
        )
        print("crit = ", crit)
    return crit - 1


def ApproxPercolCriterion2(
    T: float, outdict: dict, pot, phase_sym, phase_bro, Tmax: float, Tmin: float, vw: float, verbose: bool = False
) -> float:
    """Alternative method to estimate the percolation criterion. This uses a
    smaller dT for the derivatives and computes betaH from the action not the
    nucleation rate.

    Parameters
    ----------
    T : float
        The temperature.
    outdict : dict
        Dictionary storing the action evaluation
    pot : generic_potential
        The effective potential object
    phase_sym : PhaseInfo
        The high temperature phase information
    phase_bro : PhaseInfo
        The low temperature phase information
    Tmax : float
        The maximal temperature, where to compute the nucleation criterion (usually Tnuc)
    Tmin : float
        The minimal temperature where to compute the nucleation criterion, usually
        the minimal temp. of coexistence of the phases.
    vw : float
        The bubble wall velocity.
    verbose : bool, optional
        Set the output level

    Returns
    ----------
    float :
        0 if the criterion is met for T, < 0 if T is to large, > 0 if T is to small."""
    try:
        T = T[0]  # need this when the function is run from optimize.newton, because outdict needs a hashable key
    except Exception:
        pass

    if T >= Tmax:
        return -np.inf
    if T <= Tmin:
        return np.inf

    dT = T * 1e-3
    if T + dT > Tmax:
        dT = (Tmax - T) / 100.0

    if T + dT >= Tmax:
        return -np.inf
    if T + dT <= Tmin:
        return np.inf

    fperc = 0.34  # - log(P = 0.7) = 0.34
    SdT = calcAction(pot, T + dT, phase_sym, phase_bro, outdict)
    S = calcAction(pot, T, phase_sym, phase_bro, outdict)
    if np.isinf(SdT) or np.isinf(S):
        return np.inf

    if S == 0 or SdT == 0:
        return -np.inf

    betaH = (SdT - S) / dT - S / T
    rho = energyDensity(pot, phase_sym, T)
    H = HubbleParameter(rho, pot.conversionFactor)
    G = Gamma(T, S)
    if not np.isfinite(betaH) or abs(betaH) < 1e-30:
        if verbose:
            print("Warning: betaH is not finite or zero in ApproxPercolCriterion2. betaH =", betaH)
        return np.inf
    if not np.isfinite(H) or H <= 0:
        if verbose:
            print("Warning: H is not finite or zero in ApproxPercolCriterion2. H =", H)
        return np.inf
    if not np.isfinite(vw) or vw <= 0:
        if verbose:
            print("Warning: vw is not finite or zero in ApproxPercolCriterion2. vw =", vw)
        return np.inf
    crit = np.power(G, 1 / 4.0) / H / betaH / np.power(fperc / (8 * np.pi * vw**3), 1 / 4.0)
    return float(np.squeeze(crit)) - 1


def calcApproxPercolation(
    outdict: dict,
    pot,
    Tnuc: float,
    phase_sym,
    phase_bro,
    vw: float,
    verbose: bool = True,
    tmin: float | None = None,
    tmax: float | None = None,
) -> float:
    """Estimate the percolation temperature assuming equal SM/DS temperatures.

    Parameters
    ----------
    outdict : dict
        Dictionary storing the action evaluation.
    pot : generic_potential
        The effective potential object.
    Tnuc : float
        The nucleation temperature.
    phase_sym : PhaseInfo
        High-temperature phase information.
    phase_bro : PhaseInfo
        Low-temperature phase information.
    vw : float
        Bubble wall velocity.
    verbose : bool, optional
        Whether to emit diagnostic messages.

    Returns
    -------
    float
        The approximate percolation temperature.
    """
    if tmin is None or tmax is None:
        tmin, tmax = _phase_overlap_interval(pot, phase_sym, phase_bro, Tnuc=Tnuc)

    dT = np.maximum(Tnuc * 1e-3, 1e-3)  # Try this for numerical stability
    Tmin = max(float(tmin) + 2 * dT, float(tmin))
    # Shift Tmax to make derivative possible, take 2*dT to be sure.
    # Keep the search inside the phase-overlap interval.
    Tmax = min(Tnuc - 2 * dT, float(tmax) - 2 * dT)
    if Tmax <= Tmin:
        raise errors.PercolationApproximation1Error(
            "No valid temperature bracket for percolation approximation inside "
            f"the phase-overlap interval [{tmin:.8g}, {tmax:.8g}] at Tnuc={Tnuc:.8g}."
        )

    try:
        # The calculation at the Tmin failes some times for u-shaped actions
        args = (outdict, pot, phase_sym, phase_bro, Tmax, Tmin, vw, verbose)
        # rtol is large, dont waste time on precision here
        Tperc = optimize.brentq(ApproxPercolCriterion2, Tmax, Tmin, rtol=1e-3, args=args)

    except ValueError as err:
        if verbose:
            print("Brentq failed in percolation approximation 1: ", err)
        # This error is likely due to the range of the search not being large enough
        # to find the root. We can try to increase the range and see if it works.
        try:
            Tperc = optimize.brentq(
                ApproxPercolCriterion,
                float(tmax),
                Tmin,
                rtol=1e-3,
                args=(outdict, pot, phase_sym, phase_bro, float(tmax), Tmin, vw, verbose),
            )
        except ValueError as err:
            if verbose:
                print("Brentq failed in percolation approximation 1, even after increasing temperature range: ", err)
            # This error may indicate that the error was not due to the range of the search.
            # Try with another method now:
            try:
                Tperc = optimize.newton(
                    ApproxPercolCriterion2,
                    Tnuc,
                    args=(outdict, pot, phase_sym, phase_bro, float(tmax), float(tmin), vw, verbose),
                )
                if verbose:
                    print("Newton method used in percolation approximation 1")
                if Tperc < float(tmin) or Tperc > float(tmax):
                    raise ValueError("Newton method failed to find a root in the range")
            except Exception as newton_err:
                if verbose:
                    print("Newton method failed in percolation approximation 1: ", newton_err)
                raise errors.PercolationApproximation1Error(newton_err)
    return np.squeeze(Tperc)


def Tb_criterion(TBRO: float, TSYM: float, phase_sym, phase_broken, pot) -> float:
    """Criterion to find the broken phase temperature, it
    uses energy conservation.
    Reheat to SM + DS in the broken phase. Assume instant heating.

    Parameters
    ----------
    TBRO : float
        Broken phase temperature to solve for
    TSYM : float
        Temperature in the symmetric phase
    phase_sym : PhaseInfo
        Field value in the broken phase as a function of temperature
    phase_broken : PhaseInfo
        Field value in the broken phase as a function of temperature
    pot : generic_potential
        The effective potential object

    Returns
    -------
    float
        Criterion: 0 if ``TBRO`` is the correct broken phase temperature.
    """

    eSYM = energyDensity(pot, phase_sym, TSYM, include_decoupled=False)
    eBRO = energyDensity(pot, phase_broken, TBRO, include_decoupled=False)

    return eBRO / eSYM - 1


def energy_criterion_BRO(TBRO: float, eSYM: float, eBRO_start: float, P: float, dP: float, phase_broken, pot) -> float:
    """Calculate the temperature in the broken phase that results
    from the reheating by converting dP of the false vacuum into true vacuum.

    Parameters
    ----------
    TBRO : float
        Temperature to solve for.
    eSYM : float
        Energy density in the symmetric phase
    eBRO_start : float
        Initial energy density in broken phase
    P : float
        True vacuum fraction
    dP : float
        Change in the true vacuum fraction
    phase_broken : PhaseInfo
        Information about the end phase.
    pot : generic_potential

    Returns
    -------
    float
        Zero when the criterion is fulfilled.
    """
    eBRO = energyDensity(pot, phase_broken, TBRO, include_decoupled=False)
    crit = eBRO * P - (eBRO_start * (P - dP) + dP * eSYM)
    return crit


def entropy_criterion_SYM_BRO(Tb: float, TBRO_ref: float, TSYM: float, TSYM_ref: float, phase_broken, pot) -> float:
    """Calculate the temperature of the dark sector broken phase in terms
    of the symmetric phase temperature. Only valid when entropy is
    conserved between reference temperature and TSYM.

    Parameters
    ----------
    Tb : float
        Temperature in the broken phase.
    TBRO_ref : float
        Reference temperature of DS from which on entropy is conserved.
    TSYM : float
        SYM temperature for which we want to know TDS(TSM).
    TSYM_ref : float
        Reference temperature of SYM from which on entropy is conserved.
    phase_broken : PhaseInfo
        Phase information about the end (broken) phase.
    pot : generic_potential

    Returns
    -------
    float
        Zero when the condition is met.
    """
    heff_ds_pt = h_eff_DS(TBRO_ref, pot, phase_broken)
    heff_SM_pt = td.s_geffSM(TBRO_ref, pot.conversionFactor)
    heff_ds = h_eff_DS(Tb, pot, phase_broken)
    heff_SM = td.s_geffSM(Tb, pot.conversionFactor)
    crit = (heff_ds + heff_SM) * Tb**3
    crit -= (heff_ds_pt + heff_SM_pt) * TBRO_ref**3 * TSYM**3 / TSYM_ref**3
    return crit


@dataclass
class PercolationSettings:
    """Container for frequently used percolation configuration parameters."""

    f_perc: float
    f_start: float
    f_final: float
    weight: float
    maxit: int
    rel_increment: float
    max_boundary_n: int


@dataclass
class PercolationGrid:
    """Shared inputs needed throughout the percolation workflow."""

    TSYM: np.ndarray
    TpercApprox: float
    tmin: float
    tmax: float
    TBROmin: float
    TBROmax: float


@dataclass
class PercolationState:
    """Mutable arrays updated during the percolation iterations."""

    TSYM: np.ndarray
    Sr: np.ndarray
    Hr: np.ndarray
    Pr: np.ndarray
    Pr_exp: np.ndarray
    scalef_ratio: np.ndarray
    soundSpSq: np.ndarray
    Tb: np.ndarray


def _build_percolation_settings(pot, nAction: int) -> PercolationSettings:
    """Read commonly used configuration parameters once."""
    conf = pot.config.percolationConf
    max_boundary_n = int(conf.max_boundary_ratio * nAction)
    return PercolationSettings(
        f_perc=conf.f_perc,
        f_start=conf.f_start,
        f_final=conf.f_final,
        weight=conf.weight,
        maxit=conf.maxit,
        rel_increment=conf.rel_increment,
        max_boundary_n=max_boundary_n,
    )


def _phase_overlap_interval(
    pot,
    phase_symmetric,
    phase_broken,
    Tnuc: float | None = None,
) -> tuple[float, float]:
    """Return the common traced temperature interval of both phases.

    The percolation and observable pipeline assumes all thermodynamic
    quantities are evaluated inside this overlap. If this is violated, we
    surface a hard error instead of silently extrapolating phase splines.
    """
    tmin = max(float(phase_symmetric.Tmin), float(phase_broken.Tmin))
    tmax = min(float(phase_symmetric.Tmax), float(phase_broken.Tmax))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmin >= tmax:
        raise errors.PercolationError(
            "No overlapping temperature interval between traced phases: "
            f"Tsym in [{phase_symmetric.Tmin:.8g}, {phase_symmetric.Tmax:.8g}], "
            f"Tbro in [{phase_broken.Tmin:.8g}, {phase_broken.Tmax:.8g}]."
        )

    if Tnuc is not None:
        tracing_conf = getattr(getattr(pot, "config", None), "tracingConf", None)
        nuc_tol = float(getattr(tracing_conf, "nucleation_Ttol", 1e-8))
        scale = max(abs(tmin), abs(tmax), abs(float(Tnuc)), 1.0)
        temp_tol = max(nuc_tol, 1e-10 * scale)
        if Tnuc < tmin - temp_tol or Tnuc > tmax + temp_tol:
            raise errors.PercolationError(
                "Nucleation temperature is outside the traced phase-overlap interval: "
                f"Tnuc={Tnuc:.8g}, overlap=[{tmin:.8g}, {tmax:.8g}], tol={temp_tol:.3g}. "
                "This indicates inconsistent phase tracing / tunnelling data."
            )

    return tmin, tmax


def _initial_temperature_grid(
    outdict: dict,
    pot,
    Tnuc: float,
    phase_symmetric,
    phase_broken,
    vw: float,
    nAction: int,
    settings: PercolationSettings,
    overlap_tmin: float,
    overlap_tmax: float,
    verbose: bool,
) -> PercolationGrid:
    """Step 1: build the temperature grid around an approximate percolation point."""
    if verbose:
        print("\nPercolation step 1: calculating Tperc using the saddlepoint approximation")
    try:
        TpercApprox = calcApproxPercolation(
            outdict,
            pot,
            Tnuc,
            phase_symmetric,
            phase_broken,
            vw,
            verbose,
            tmin=overlap_tmin,
            tmax=overlap_tmax,
        )
    except errors.PercolationApproximation1Error as err:
        if verbose:
            print("Percolation approximation 1 failed: ", err, ". We will continue with TpercApprox = Tnuc.")
        TpercApprox = Tnuc
    if verbose:
        print(f"Approximate percolation temperature: {TpercApprox * pot.conversionFactor:2.5g} GeV")

    dT = (Tnuc - TpercApprox) / int(nAction * settings.weight)
    tmin = overlap_tmin
    tmax = overlap_tmax

    if TpercApprox < Tnuc:
        upper = np.maximum(TpercApprox - int(nAction * (1 - settings.weight)) * dT, tmin)
        TSYM = np.linspace(Tnuc, upper, nAction)
    else:
        if verbose:
            print("Warning: TpercApprox > Tnuc, use interval around TpercApprox")
        # here we set couple_hydrodynamics to false, since we are only interested in alpha_tot anyways
        _, alpha_perc_approx, _, _, _, _ = calcAlphas(TpercApprox, pot, phase_symmetric, phase_broken,
                                                      coupled_hydrodynamics=False, verbose=verbose)

        # The following is a heuristic to set the temperature range. The values are
        # chose somewhat arbitrarily, but they should be reasonable for most cases.
        # This step is not critical, but it should be chosen such that
        # a percolation temperature is found within a reasonable range.
        if alpha_perc_approx < 1:
            Tmin = 0.999 * TpercApprox
            Tmax = 1.001 * TpercApprox
        elif alpha_perc_approx < 10:
            Tmin = 0.9 * TpercApprox
            Tmax = 1.1 * TpercApprox
        elif alpha_perc_approx < 100:
            Tmin = 0.8 * TpercApprox
            Tmax = 1.2 * TpercApprox
        else:
            Tmin = 0.5 * TpercApprox
            Tmax = 1.5 * TpercApprox
        Tmax = min(Tmax, tmax)
        Tmin = max(Tmin, tmin)
        TSYM = np.linspace(Tmax, Tmin, nAction)

    return PercolationGrid(
        TSYM=TSYM,
        TpercApprox=TpercApprox,
        tmin=tmin,
        tmax=tmax,
        TBROmin=phase_broken.Tmin,
        TBROmax=phase_broken.Tmax,
    )


def _initial_percolation_scan(
    grid: PercolationGrid,
    settings: PercolationSettings,
    outdict: dict,
    pot,
    phase_symmetric,
    phase_broken,
    CF: float,
    vw: float,
    verbose: bool,
) -> PercolationState:
    """Step 2: compute P(T) with a P-independent Hubble rate.
    
    Calculate the action and Hubble parameter at each temperature
    to find a first guess for the true vacuum fraction P and use the latter
    to calculate the percolation temperature."""
    TSYM = grid.TSYM
    Sr = np.zeros_like(TSYM)
    Hr = np.zeros_like(TSYM)
    Pr = np.zeros_like(TSYM)
    Tb = np.zeros_like(TSYM)
    scalef_ratio = np.zeros_like(TSYM)
    soundSpSq = np.zeros_like(TSYM)
    Pr_exp = np.zeros_like(TSYM)
    TBROmin = grid.TBROmin
    TBROmax = grid.TBROmax

    iterator = 1
    iteration_condition = True
    if verbose:
        print("\nPercolation step 2: calculating Tperc assuming P = 0 in the Hubble rate")

    while iteration_condition:
        Sr.fill(0)
        Hr.fill(0)
        Pr.fill(0)
        Tb.fill(0)
        Pr_exp.fill(0)
        soundSpSq.fill(0)
        scalef_ratio.fill(0)
        is_Pr_converged = False

        for i, T in enumerate(TSYM):
            if not is_Pr_converged:
                Sr[i] = calcAction(pot, T, phase_symmetric, phase_broken, outdict)
                rho = energyDensity(pot, phase_symmetric, T)
                scalef_ratio[i] = scalefactorRatio(pot, TSYM[0], T, phase_symmetric.valAt(T))
                soundSpSq[i] = calcSoundSpeedSq(pot, phase_symmetric.valAt(T), T)
                Hr[i] = HubbleParameter(rho, CF)
                Pr[i] = 1 - np.exp(-percIntegral(TSYM[0:i+1], Hr[0:i+1], Sr[0:i+1], vw=vw))
                Pr_exp[i] = 1 - np.exp(-percIntegralwExp(TSYM[0:i+1], Hr[0:i+1],
                                                                  Sr[0:i+1], scalef_ratio[0:i+1],
                                                                  soundSpSq[0:i+1], vw=vw))
            else:
                Sr[i] = np.nan
                rho = energyDensity(pot, phase_symmetric, T)
                Hr[i] = HubbleParameter(rho, CF)
                scalef_ratio[i] = scalefactorRatio(pot, TSYM[0], T, phase_symmetric.valAt(T))
                soundSpSq[i] = calcSoundSpeedSq(pot, phase_symmetric.valAt(T), T)
                Pr[i] = Pr[i - 1]
                Pr_exp[i] = Pr_exp[i - 1]

            if verbose:
                print(
                    f"Step {i+1}/{len(TSYM)}, T = {TSYM[i] * CF:2.8g} GeV, "
                    f"P_true = {Pr[i]:2.5g}, S_3 = {Sr[i] * CF:2.5g} GeV, "
                    f"S_3/T = {Sr[i] / T:2.5g}"
                )

            if i >= 1 and Pr[i] > settings.f_final and Pr[i - 1] > settings.f_final:
                if verbose and not is_Pr_converged:
                    print(
                        "WARNING: P > ",
                        settings.f_final,
                        " at T = ",
                        TSYM[i] * CF,
                        " GeV detected. Stop computation of action from here on "
                        "and only fill in the values for the Hubble rate.",
                    )
                is_Pr_converged = True

        boundary_n = min(settings.max_boundary_n, len(Pr) - 1)
        if Pr[0] <= settings.f_start and Pr[-1] >= settings.f_final:
            # This is good already! P_true goes from < f_start to > f_final.
            # The temperature range could maybe be improved:

            # Check if Pr includes a jump from 0 to 1
            jump_indices = np.where((Pr[:-1] == 0) & (Pr[1:] == 1))[0]
            has_jump_0_to_1 = jump_indices.size > 0
            if has_jump_0_to_1:
                # If there is a jump from 0 to 1, adjust the temperature range to cover only this jump.
                max_index = jump_indices[0]
                min_index = jump_indices[0] + 1
                tsym_jump_max = TSYM[max_index]
                tsym_jump_min = TSYM[min_index]
                if verbose:
                    print(
                        "Warning: Pr includes a 0 followed by 1, i.e. a "
                        "discontinuous jump in true vacuum fraction."
                    )
                    print(
                        f"Jump occurs between the indices {max_index} and {min_index}, "
                        f"TSYM values: {tsym_jump_min} and {tsym_jump_max}. "
                        "Set these as new boundaries."
                    )
                if tsym_jump_max / tsym_jump_min - 1 > 1e-10:
                    TSYM = np.linspace(tsym_jump_max, tsym_jump_min, len(TSYM))
                    grid.TSYM = TSYM
                    Sr = np.zeros_like(TSYM)
                    Hr = np.zeros_like(TSYM)
                    Pr = np.zeros_like(TSYM)
                    Tb = np.zeros_like(TSYM)
                    iterator += 1
                    iteration_condition = iterator < settings.maxit
                else:
                    if verbose:
                        print(
                            "WARNING: Even though there is a jump in P_true, the "
                            "temperature range is already very small. We will not "
                            "adjust it further and call the percolation temperature "
                            "calculation converged."
                        )
                    # The difference in the temperatures is already tiny,
                    # so further shrinking the range would only make the code unstable.
                    iteration_condition = False

            elif Pr[boundary_n] == 0 or Pr[-boundary_n] == 1:
                # There are too many points in the temperature range where P_true = 0 or P_true = 1.
                if verbose and iterator != settings.maxit:
                    print(
                        "Warning: the temperature range for the percolation "
                        "temperature computation was chosen too wide in iteration ",
                        iterator,
                        ". Decrease it!",
                    )

                if Pr[boundary_n] == 0:
                    # redo step 2 with a lower highest temperature
                    Tmax_new = TSYM[np.where(Pr < settings.f_start)[0][-1]]
                else:
                    Tmax_new = TSYM[0]

                if Pr[-boundary_n] == 1:
                    # redo step 2 with a higher lowest temperature
                    Tmin_new = TSYM[np.where(Pr > settings.f_final)[0][0]]
                else:
                    Tmin_new = TSYM[-1]

                if Tmax_new < Tmin_new:
                    raise errors.PercolationError(
                        "The temperature range for the percolation temperature "
                        "computation was messed up. This should not happen. Please report this as a bug."
                    )
                TSYM = np.linspace(Tmax_new, Tmin_new, len(TSYM))
                grid.TSYM = TSYM
                Sr = np.zeros_like(TSYM)
                Hr = np.zeros_like(TSYM)
                Pr = np.zeros_like(TSYM)
                Tb = np.zeros_like(TSYM)
                iterator += 1
                iteration_condition = iterator < settings.maxit
            else:
                # The temperature range is good!
                iteration_condition = False
        else:
            # The temperature range is not good, we need to adjust it.
            if verbose and iterator != settings.maxit:
                print(
                    "Warning: the temperature range for the percolation "
                    "temperature computation was chosen too narrow in iteration ",
                    iterator,
                    ". Increase it!",
                )
            if verbose and iterator == settings.maxit:
                print(
                    "Warning: the temperature range for the percolation "
                    "temperature computation was chosen too narrow. This is "
                    "the last iteration, so we will use the current "
                    "temperature range anyway."
                )
            if Pr[0] > settings.f_start:
                Tmax_new = TSYM[0] + (grid.tmax - TSYM[0]) * settings.rel_increment
            else:
                Tmax_new = TSYM[0]

            if Pr[-1] < settings.f_final:
                # redo step 2 with a lower lowest temperature
                if iterator >= settings.maxit - 2:
                    # If we are at the last iteration, we use the tmin value
                    # to be sure that we cover the whole possible temperature
                    # range before concluding that percolation is not possible.
                    Tmin_new = grid.tmin
                else:
                    Tmin_new = TSYM[-1] - (TSYM[-1] - grid.tmin) * settings.rel_increment
            else:
                Tmin_new = TSYM[-1]

            # The temperature must be decreasing in this array
            TSYM = np.linspace(Tmax_new, Tmin_new, len(TSYM))
            grid.TSYM = TSYM
            Sr = np.zeros_like(TSYM)
            Hr = np.zeros_like(TSYM)
            Pr = np.zeros_like(TSYM)
            Tb = np.zeros_like(TSYM)
            iterator += 1
            iteration_condition = iterator < settings.maxit

    return PercolationState(TSYM=grid.TSYM, Sr=Sr, Hr=Hr, Pr=Pr, Tb=Tb,
                            Pr_exp=Pr_exp, soundSpSq=soundSpSq, scalef_ratio=scalef_ratio)


def _solve_for_initial_Tperc(
    state: PercolationState,
    settings: PercolationSettings,
    pot,
    verbose: bool,
) -> float:
    """Solve for the percolation temperature after step 2."""
    Pint = interpolate.interp1d(state.TSYM, state.Pr)
    try:
        Tperc_prev = optimize.brentq(lambda T: Pint(T) - settings.f_perc, state.TSYM[0], state.TSYM[-1])
    except ValueError as err:
        msg = err.args[0]
        if msg.startswith("f(a) and f(b) must have different signs"):
            if Pint(state.TSYM[-1]) < settings.f_perc:
                raise errors.TooMuchSupercoolingError(
                    "The percolation temperature could not be found because the true vacuum "
                    "fraction only reaches "
                    f"{Pint(state.TSYM[-1])} < {settings.f_perc} at Tmin = "
                    f"{state.TSYM[-1] * pot.conversionFactor} GeV."
                )
        raise errors.PercolationError(err)
    except Exception as err:
        msg = (
            "Error in calculating Tperc_prev after the second approximation step "
            "with P = 0 in the Hubble rate. This might be because the nucleation "
            "criterion could be fulfilled but not the percolation one due to "
            f"strong supercooling: {err}"
        )
        if verbose:
            print(msg)
        raise errors.PercolationError(err)
    if verbose:
        print(f"Tperc_prev = {Tperc_prev * pot.conversionFactor:2.8g} GeV")
    return Tperc_prev


def _refine_percolation_temperature(
    state: PercolationState,
    grid: PercolationGrid,
    settings: PercolationSettings,
    outdict: dict,
    pot,
    phase_symmetric,
    phase_broken,
    CF: float,
    vw: float,
    Tperc_prev: float,
    rtol: float,
    verbose: bool,
) -> tuple[float, PercolationState]:
    """Step 3: iterate with the P-dependent Hubble rate until convergence.
    
    Refine the percolation temperature by solving the
    percolation integral with the Hubble rate and the true vacuum fraction
    calculated in the previous step. The function iteratively
    improves the true vacuum fraction and the Hubble rate until
    convergence is reached."""
    TBROmin = grid.TBROmin
    TBROmax = grid.TBROmax
    TSYM = state.TSYM
    iterator = 1
    iteration_condition = True

    if verbose:
        print("\nPercolation step 3: calculating Tperc with P-dependent Hubble rate")

    while iteration_condition:
        if verbose:
            print(f"Iteration {iterator}, T_perc = {Tperc_prev * CF:2.8g} GeV")

        Pprev = state.Pr.copy()
        state.Hr.fill(0)
        state.Pr.fill(0)
        state.Pr_exp.fill(0)
        state.scalef_ratio.fill(0)
        state.soundSpSq.fill(0)
        state.Tb.fill(0)
        state.Sr.fill(0)
        is_Pr_converged = False

        for i, T in enumerate(TSYM):
            if not is_Pr_converged:
                eSYM = energyDensity(pot, phase_symmetric, T)
                if i == 0: # High temperature bin
                    # Pprev must be zero here
                    state.Hr[i] = HubbleParameter(eSYM, CF)
                    state.Sr[i] = calcAction(pot, T, phase_symmetric, phase_broken, outdict)
                    state.scalef_ratio[i] = 1
                    state.soundSpSq[i] = calcSoundSpeedSq(pot, phase_symmetric.valAt(T), T)
                    try:
                        TBRO = optimize.brentq(
                            Tb_criterion,
                            TBROmin,
                            TBROmax,
                            args=(T, phase_symmetric, phase_broken, pot),
                        )
                        state.Tb[i] = TBRO
                    except ValueError as err:
                        # Sometimes, brentq fails to find a root here. This is usually due to
                        # an inaccuracy in phase tracing, which can be solved by increasing
                        # the diftol parameter in the config file.
                        raise errors.PercolationError(
                            "Error in Tb_criterion: "
                            f"{err}. Please check the input parameters again by hand. "
                            "It might be that the phase tracing is not accurate enough, "
                            "try increasing the diftol parameter in the config file."
                        )
                    eBRO = 0
                else:
                    # Use entropy conservation in the symmetric phase between two time steps
                    # to compute the broken phase energy density in step i based on the
                    # broken phase energy density in step i-1.
                    TBRO = optimize.brentq(
                        entropy_criterion_SYM_BRO, TBROmin, TBROmax,
                        args=(state.Tb[i - 1], T, TSYM[i - 1], phase_broken, pot)
                    )
                    eBRO = energyDensity(pot, phase_broken, TBRO)

                    P = Pprev[i]
                    dP = Pprev[i] - Pprev[i - 1]
                    energy_release = eBRO * (P - dP) + dP * eSYM
                    relative_energy_release = energy_release / eBRO
                    if relative_energy_release < 1e-50:
                        # If the energy release between two steps is too small, we don't need to
                        # calculate the broken phase temperature. The following computation in that
                        # case would just try to find an arbitrary argument T in front of a factor zero
                        # being the energy release. It is thus safer to just set TBRO to the value obtained
                        # in the previous step in the entropy conservation.
                        if verbose:
                            print("Energy release is too small to use energy criterion, using T_bro from entropy step")
                        state.Tb[i] = TBRO
                    else:
                        # Use the energy conservation to find the broken phase temperature
                        # based on the energy release between two steps.
                        try:
                            TBRO = optimize.brentq(
                                energy_criterion_BRO,
                                TBROmin,
                                TBROmax,
                                args=(eSYM, eBRO, Pprev[i], Pprev[i] - Pprev[i - 1], phase_broken, pot),
                            )
                            state.Tb[i] = TBRO
                        except ValueError as err:
                            # Sometimes, the energy criterion fails to find a root,
                            # because the energy release is too small.
                            # We try to limit this case by checking the relative
                            # energy release beforehand, but sometimes it still happens.
                            # In that case, we just use the value from the previous step
                            # and print a warning. This is not critical, as the energy
                            # release is small anyway.
                            if verbose:
                                print(
                                    "Warning: energy_criterion failed. This is usually "
                                    "because the relative energy release "
                                    f"{relative_energy_release} is too small. We set it "
                                    "to 0 and continue: ",
                                    err,
                                )
                            state.Tb[i] = state.Tb[i - 1]
                            TBRO = state.Tb[i]
                        except Exception as err:
                            raise errors.PercolationError(
                                "Error in energy_criterion_BRO: "
                                f"{err}. Please check this input parameters again by hand."
                            )

                        eBRO = energyDensity(pot, phase_broken, TBRO)

                    if state.Tb[i] < TSYM[i]:
                        raise ValueError(
                            "There was a problem with the T_bro calculation, "
                            "T_bro < T_sym. This is (usually) not allowed by the "
                            "second law of thermodynamics."
                        )

                state.Sr[i] = calcAction(pot, T, phase_symmetric, phase_broken, outdict)
                state.Hr[i] = HubbleParameter(Pprev[i] * eBRO + (1 - Pprev[i]) * eSYM, CF)
                state.soundSpSq[i] = calcSoundSpeedSq(pot, phase_symmetric.valAt(T), T)
                state.scalef_ratio[i] = scalefactorRatio(pot, TSYM[0], T,
                                                         phase_symmetric.valAt(T))
                state.Pr[i] = 1 - np.exp(-percIntegral(TSYM[0:i+1], state.Hr[0:i+1],
                                                       state.Sr[0:i+1], vw=vw))
                state.Pr_exp[i] = 1 - np.exp(-percIntegralwExp(TSYM[0:i+1], state.Hr[0:i+1],
                                                               state.Sr[0:i+1],
                                                               state.scalef_ratio[0:i+1],
                                                               state.soundSpSq[0:i+1],
                                                               vw=vw))
            else:
                state.Sr[i] = np.nan
                rho = energyDensity(pot, phase_symmetric, T)
                state.soundSpSq[i] = calcSoundSpeedSq(pot, phase_symmetric.valAt(T), T) 
                state.scalef_ratio[i] = scalefactorRatio(pot, TSYM[0], T,
                                                         phase_symmetric.valAt(T))
                state.Hr[i] = HubbleParameter(rho, CF)
                state.Pr[i] = state.Pr[i-1]
                state.Pr_exp[i] = state.Pr_exp[i-1]

            if verbose:
                print(
                    f"Step {i+1}/{len(TSYM)}, T = {TSYM[i] * CF:2.8g} GeV, "
                    f"P_true = {state.Pr[i]:2.5g}, S_3 = {state.Sr[i] * CF:2.5g} GeV, "
                    f"S_3/T = {state.Sr[i] / T:2.5g}"
                )

            if i >= 1 and state.Pr[i] > settings.f_final and state.Pr[i - 1] > settings.f_final:
                if verbose and not is_Pr_converged:
                    print(
                        "WARNING: P > ",
                        settings.f_final,
                        " at T = ",
                        TSYM[i] * CF,
                        " GeV detected. Stop computation of action from here on "
                        "and only fill in the values for the Hubble rate.",
                    )
                is_Pr_converged = True

        Pint = interpolate.interp1d(TSYM, state.Pr)
        Pexpint = interpolate.interp1d(TSYM, state.Pr_exp)
        try:
            Tperc = optimize.brentq(lambda T: Pint(T) - settings.f_perc, TSYM[0], TSYM[-1])
        except ValueError as err:
            msg = err.args[0]
            if msg.startswith("f(a) and f(b) must have different signs"):
                if Pint(TSYM[-1]) < settings.f_perc:
                    raise errors.TooMuchSupercoolingError(
                        "The percolation temperature could not be found because the "
                        "true vacuum fraction only reaches "
                        f"{Pint(TSYM[-1])} < {settings.f_perc} at Tmin = "
                        f"{TSYM[-1] * CF} GeV."
                    )
            raise errors.PercolationError(err)

        except Exception as err:
            msg = (
                "Warning: could not compute Tperc after the third approximation step "
                "with variable P in Hubble rate: "
                f"{err}. Set the percolation temperature to the one from the previous step."
            )
            if verbose:
                print(msg)
            Tperc = Tperc_prev

        if np.abs(Tperc - Tperc_prev) / Tperc < rtol:
            # A percolation temperature was found!
            if state.Pr[0] <= settings.f_start and state.Pr[-1] >= settings.f_final:
                # This is good already! P_true goes from < f_start to > f_final.
                # The temperature range could maybe be improved:

                # Check if Pr includes a jump from 0 to 1
                jump_indices = np.where((state.Pr[:-1] == 0) & (state.Pr[1:] == 1))[0]
                has_jump_0_to_1 = jump_indices.size > 0
                boundary_n = min(settings.max_boundary_n, len(state.Pr) - 1)
                if has_jump_0_to_1:
                    # If there is a jump from 0 to 1, we need to adjust the temperature range to cover only this jump.
                    max_index = jump_indices[0]
                    min_index = jump_indices[0] + 1
                    tsym_jump_max = TSYM[max_index]
                    tsym_jump_min = TSYM[min_index]
                    if verbose:
                        print(
                            "Warning: Pr includes a 0 followed by 1, i.e. a "
                            "discontinuous jump in true vacuum fraction."
                        )
                        print(
                            f"Jump occurs between the indices {max_index} and {min_index}, "
                            f"TSYM values: {tsym_jump_min} and {tsym_jump_max}. "
                            "Set these as new boundaries."
                        )
                    if tsym_jump_max / tsym_jump_min - 1 > 1e-10:
                        TSYM = np.linspace(tsym_jump_max, tsym_jump_min, len(TSYM))
                        state.TSYM = TSYM
                        state.Sr = np.zeros_like(TSYM)
                        state.Hr = np.zeros_like(TSYM)
                        state.Pr = np.zeros_like(TSYM)
                        state.Tb = np.zeros_like(TSYM)
                        Tperc_prev = Tperc
                        iterator += 1
                        iteration_condition = iterator < settings.maxit
                    else:
                        if verbose:
                            print(
                                "WARNING: Even though there is a jump in P_true, the "
                                "temperature range is already very small. We will not "
                                "adjust it further and call the percolation temperature "
                                "calculation converged."
                            )
                            print(
                                "\nConverged after "
                                f"{iterator} iterations to Tperc = "
                                f"{Tperc * CF:2.8g} GeV"
                            )
                        # The temperature difference is already tiny,
                        # so further shrinking the range would only make the code unstable.
                        iteration_condition = False
                elif state.Pr[boundary_n] == 0 or state.Pr[-boundary_n] == 1:
                    # There are too many points in the temperature range where P_true = 0 or P_true = 1.
                    if verbose and iterator != settings.maxit:
                        print(
                            "Warning: the temperature range for the percolation "
                            "temperature computation was chosen too wide in iteration ",
                            iterator,
                            ". Decrease it!",
                        )
                    if state.Pr[boundary_n] == 0:
                        # redo step 2 with a lower highest temperature
                        Tmax_new = TSYM[np.where(state.Pr < settings.f_start)[0][-1]]
                    else:
                        Tmax_new = TSYM[0]

                    if state.Pr[-boundary_n] == 1:
                        # redo step 2 with a higher lowest temperature
                        Tmin_new = TSYM[np.where(state.Pr > settings.f_final)[0][0]]
                    else:
                        Tmin_new = TSYM[-1]

                    if Tmax_new < Tmin_new:
                        raise errors.PercolationError(
                            "The temperature range for the percolation temperature "
                            "computation was messed up. It is not clear how this could "
                            "have happened."
                        )
                    TSYM = np.linspace(Tmax_new, Tmin_new, len(TSYM))
                    state.TSYM = TSYM
                    state.Sr = np.zeros_like(TSYM)
                    state.Hr = np.zeros_like(TSYM)
                    state.Pr = np.zeros_like(TSYM)
                    state.Tb = np.zeros_like(TSYM)
                    state.Pr_exp = np.zeros_like(TSYM)
                    state.soundSpSq = np.zeros_like(TSYM)
                    state.scalef_ratio = np.zeros_like(TSYM)
                    Tperc_prev = Tperc
                    iterator += 1
                    iteration_condition = iterator < settings.maxit
                else:
                    # The temperature range is good!
                    if verbose:
                        print(
                            "\nConverged after "
                            f"{iterator} iterations to Tperc = "
                            f"{Tperc * CF:2.8g} GeV"
                        )
                    iteration_condition = False # This breaks the while loop
            else:
                # The temperature range is not large enough to find a percolation temperature, we need to increase it
                if verbose:
                    print(
                        "Warning: even though a percolation temperature was previously "
                        f"found (Tperc = {Tperc * CF:2.8g} GeV), it is not clear "
                        "whether the phase transition can finalize. The temperature "
                        "range will therefore be increased."
                    )
                    print("This was iteration", iterator, "of the third step of the percolation calculation.")
                    if iterator == settings.maxit - 1:
                        print(
                            "This was the penultimate iteration, the temperature range "
                            "will be set to the maximum possible range."
                        )
                    elif iterator == settings.maxit:
                        print("This was the last iteration. We stop here and return an error.")
                    else:
                        print("We will increase the temperature range and try again.")

                if state.Pr[0] > settings.f_start:
                    Tmax_new = TSYM[0] + (grid.tmax - TSYM[0]) * settings.rel_increment
                else:
                    Tmax_new = TSYM[0]

                if state.Pr[-1] < settings.f_final:
                    if iterator == settings.maxit - 1:
                        # If we are at the last iteration, we set the lower temperature boundary
                        # all the way down to the minimal possible temperature allowed by the
                        # potential. This is to ensure that we do not miss the temperature at which
                        # the transition completes.
                        Tmin_new = grid.tmin
                    elif iterator == settings.maxit:
                        raise errors.PercolationError(
                            "The true vacuum fraction at Tmin = "
                            f"{grid.tmin * CF:2.5g} GeV computed in step 3 "
                            "of the percolation computation is too low for the "
                            "transition to be considered complete: "
                            f"P(Tmin) = {state.Pr[-1]:2.5g}."
                        )
                    else:
                        Tmin_new = TSYM[-1] - (TSYM[-1] - grid.tmin) * settings.rel_increment
                else:
                    Tmin_new = TSYM[-1]

                TSYM = np.linspace(Tmax_new, Tmin_new, len(TSYM))
                state.TSYM = TSYM
                state.Sr = np.zeros_like(TSYM)
                state.Hr = np.zeros_like(TSYM)
                state.Pr = np.zeros_like(TSYM)
                state.Tb = np.zeros_like(TSYM)
                Tperc_prev = Tperc
                iterator += 1
                iteration_condition = iterator <= settings.maxit
        else:
            if verbose:
                print(
                    "Percolation temperature T_perc = ",
                    Tperc,
                    "was found, but it is not close enough to the previous one (",
                    Tperc_prev,
                    ") at iteration ",
                    iterator,
                    ". Iterate one more time with the P(t) from the previous step and see if T_perc converges.",
                )
            Tperc_prev = Tperc
            iterator += 1
            iteration_condition = iterator <= settings.maxit

    return Tperc, state


def calcPercAndEvolve(outdict: dict, Tnuc: float, phase_symmetric,
                      phase_broken, pot, vw=1.0, rtol=1e-4, nAction=50,
                      verbose=False):
    """
    Calculate the percolation temperature iteratively. The function
    assumes that the dark sector and the SM are in thermal equilibrium
    during the phase transition. First, the function calculates the
    approximate percolation temperature using the function
    calcApproxPercolation assuming P = 0 in the computation of the
    Hubble parameter and using the saddlepoint approximation, i.e.
    assuming a quickly growing bubble nucleation rate. Then, it
    computes the percolation temperature by actually solving the
    percolation integral, assuming that P = 0 at all times, such that
    the computation of the Hubble rate is simplified. Next, the
    function refines the percolation temperature by solving the
    percolation integral with the Hubble rate with the previously
    calculated P. The function returns the percolation temperature,
    if it converged, the symmetric phase temperature, the Hubble
    parameter, the true vacuum fraction, the broken phase temperature,
    and the action.

    Parameters
    ----------
    outdict : dict
        Dict storing the tunneling information.
    Tnuc : float
        Nucleation temperature of the dark sector
    phase_symmetric : PhaseInfo
        Phi of the broken phase as function of TDS
    phase_broken : PhaseInfo
        Phi of the broken phase as function of TDS
    pot : generic_potential
        The effective potential object
    vw : float, optional
        The bubble wall velocity
    rtol : float, optional
        Desired relative error in the percolation temperature
    nAction : int, optional
        Number of action evaluations
    verbose : bool, optional
        Set the output level.

    Returns
    ----------
    Tperc : float
        The percolation temperature in the DS.
    TSYM : ndarray
        The symmetric phase temperature from Tnuc til Tperc
    Hr : ndarray
        The Hubble parameter from Tnuc til Tperc
    Pr : ndarray
        The true vacuum fraction from Tnuc til Tperc
    Tb : ndarray
        The broken phase temperature (also SM temperature)
    Sr : ndarray
        The action from Tnuc til Tperc"""

    CF = pot.conversionFactor
    settings = _build_percolation_settings(pot, nAction)
    overlap_tmin, overlap_tmax = _phase_overlap_interval(
        pot,
        phase_symmetric,
        phase_broken,
        Tnuc=Tnuc,
    )
    grid = _initial_temperature_grid(
        outdict,
        pot,
        Tnuc,
        phase_symmetric,
        phase_broken,
        vw,
        nAction,
        settings,
        overlap_tmin,
        overlap_tmax,
        verbose,
    )
    state = _initial_percolation_scan(
        grid,
        settings,
        outdict,
        pot,
        phase_symmetric,
        phase_broken,
        CF,
        vw,
        verbose,
    )
    Tperc_prev = _solve_for_initial_Tperc(state, settings, pot, verbose)
    Tperc, state = _refine_percolation_temperature(
        state,
        grid,
        settings,
        outdict,
        pot,
        phase_symmetric,
        phase_broken,
        CF,
        vw,
        Tperc_prev,
        rtol,
        verbose,
    )
    return Tperc, state.TSYM, state.Hr, state.Pr, state.Tb, state.Sr, \
        state.Pr_exp, state.scalef_ratio, state.soundSpSq


def calc_betaH_S3(T: float, Sint: interpolate.interp1d, outdict: dict, pot, phase_sym, phase_bro, verbose=False) -> float:
    """Calculate the phase transition speed from the action derivative.

    Parameters
    ----------
    T : float
        The temperature at which to evaluate beta/H
    Sint: interpolate.interp1d
        Interpolation function of the action
    outdict : dict
        Dictionary storing the action evaluations, key is `T`
    pot : generic_potential
        The effective potential object
    phase_sym : PhaseInfo
        Information about the high temperature (symmetric) phase
    phase_bro : PhaseInfo
        Information about the low temperature (broken) phaes

    Returns
    ----------
    float :
        The transition speed beta/H."""
    dT = T * 1e-5
    tmin = max(float(phase_sym.Tmin), float(phase_bro.Tmin))
    tmax = min(float(phase_sym.Tmax), float(phase_bro.Tmax))
    if tmin >= tmax or T <= tmin or T >= tmax:
        if verbose:
            print(
                "Warning: cannot evaluate betaH outside phase overlap. "
                f"T={T:.8g}, overlap=[{tmin:.8g}, {tmax:.8g}]"
            )
        return np.nan
    # Check if the derivative can actually be calculated
    # in the range of interest. If not, make the points
    # support for the derivative closer to each other.
    if T - 2 * dT < tmin or T + 2 * dT > tmax:
        dT = np.minimum((T - tmin) / 100, (tmax - T) / 100)

    if len(Sint.x) >= 10 and Sint.x[0] < T - 2 * dT and Sint.x[-1] > T + 2 * dT:
        dSdT = derivative(Sint, T, dT)
        betaH = dSdT - Sint(T) / T
    else:
        # not enough evaluations of S for accuracy
        Tr = []
        Sr = []
        for i in range(-2, 3, 1):
            S = calcAction(pot, T - dT * i, phase_sym, phase_bro, outdict)
            Tr.append(T - dT * i)
            Sr.append(S)
        T_ar = np.array(Tr[::-1])  # We need increasing T for the spline interpolation
        S_ar = np.array(Sr[::-1])
        if np.all(np.isinf(S_ar)):
            if verbose:
                print("WARNING: All elements of S_ar are infinite. Unable to proceed with calculation on beta/H.")
                print(
                    "Most likely the transition is so weak that the action is infinite. "
                    "Correspondingly, beta/H is also set to inf."
                )
            return np.inf
        tckS = interpolate.splrep(T_ar, S_ar, s=0)
        S = interpolate.splev(T, tckS, der=0)
        dSdT = interpolate.splev(T, tckS, der=1)
        betaH = dSdT - S / T
    return betaH


def calcSoundSpeedSq(pot, X, T) -> float:
    """Compute the sound speed squared.

    Note: Decoupled degrees of freedom do not enter here.

    Parameters
    ----------

    Returns
    -------

    """
    dT = T*1e-3
    dVdT = pot.dVdT(X, T, dT=dT, include_decoupled=False)
    d2VdT2 = pot.d2VdT2(X, T, dT=dT, include_decoupled=False)
    csSq = dVdT/(T*d2VdT2)
    return csSq


def calcAlphas(T: float, pot, high_phase, low_phase, verbose=False) -> tuple[float]:
    """Calculate the total transition strenght of the PT.
    Use several definitions.

    The last 3 alphas are normalised to the radiation energy density
    of only the relevant sector for the bubble expansion. I.e.
    when `pot.kin_coupled_e/p_geff = 0`, they are normalised to
    radiation energy density in the PT sector only.

    Parameters
    ----------
    T : float
        Temperature at which to evaluate alpha
    pot : generic_potential
        Effective potential
    high_phase : phaseInfo
        Phase info of the initial phase
    low_phase : phaseInfo
        Phase info of the end phase

    Returns
    ----------
    tuple : float
        Tuple of floats, the 3 definitions of alpha and the alphas used for the
        calculation the efficency factors kappa."""

    high_phi = high_phase.valAt(T)  # Start phase phi values
    low_phi = low_phase.valAt(T)  # End phase phi values
    DeltaV = np.abs(pot.Vtot(high_phi, T) - pot.Vtot(low_phi, T))

    # Derivative of the potential with respect to T
    dT = T * 1e-5
    dDeltaV_p = np.abs(pot.Vtot(high_phi, T + dT / 2) - pot.Vtot(low_phi, T + dT / 2))
    dDeltaV_m = np.abs(pot.Vtot(high_phi, T - dT / 2) - pot.Vtot(low_phi, T - dT / 2))
    dDeltaVdT = (dDeltaV_p - dDeltaV_m) / dT

    # previously we used the broken phase geff of the dark sector,
    # This seemed wrong to me, so I put here the symmetric phase
    # Note that this gives slightly different alpha values for T < 100 MeV

    # Rad. energy density of the sector with the PT
    rho_rad_PTsector = pot.radiationEnergyDensity(high_phi, T, include_decoupled=False)
    # Note: include_decoupled = False means that a secluded sector is not included

    # Here we assume that the decoupled sector has the same temperature as the PT sector
    rho_rad_tot = pot.radiationEnergyDensity(high_phi, T, include_decoupled=True)

    # Energy density:
    DeltaE = DeltaV - T * dDeltaVdT

    # Alpha definitions:
    alpha_p = DeltaV / rho_rad_tot
    # Evaluate the enthalpy-normalized trace-anomaly definition separately
    # from the bag-model alpha used above.
    csSq_sym = calcSoundSpeedSq(pot, high_phi, T)
    V0 = pot.Vtot(pot.X0, pot.Tmin)
    theta_sym = pot.energyDensity(high_phi, T) + (pot.Vtot(high_phi, T)- V0)/csSq_sym
    csSq_bro = calcSoundSpeedSq(pot, low_phi, T)
    theta_bro = pot.energyDensity(low_phi, T) + (pot.Vtot(low_phi, T) - V0)/csSq_bro
    alpha_theta = (theta_sym - theta_bro)/(3 * (-pot.Vtot(high_phi, T) + V0 + pot.energyDensity(high_phi, T)))
    

    alpha_e = DeltaE / rho_rad_tot

    bosons_low = pot.boson_massSq(low_phi, 0)  # low-T phase masses
    bosons_high = pot.boson_massSq(high_phi, 0)  # high-T phase masses 
    fermions_low = pot.fermion_massSq(low_phi)
    fermions_high = pot.fermion_massSq(high_phi)

    # alpha_inf
    gauge_coupling = pot.mass_spectrum.boson_gauge_couplings
    m2_bos_after, dof_bos, _, is_physical = bosons_low
    m2_bos_before, _, _, _ = bosons_high
    m2_fer_after, dof_fer = fermions_low
    m2_fer_before, _ = fermions_high

    delta_m2_bos = np.maximum(m2_bos_after - m2_bos_before, 0)
    m2factor = np.sum(dof_bos * is_physical * delta_m2_bos, axis=-1) / 24.0

    delta_m2_fer = np.maximum(m2_fer_after - m2_fer_before, 0)
    m2factor += np.sum(dof_fer * delta_m2_fer, axis=-1) / 48.0

    m_bos_after = np.sqrt(np.where(m2_bos_after > 0, m2_bos_after, 0))
    # Avoid sqrt of negative mass squares by setting negatives to zero.
    # This only occurs for Goldstones, which are excluded via is_physical = 0.
    m_bos_before = np.sqrt(np.where(m2_bos_before > 0, m2_bos_before, 0))
    delta_m_bos = np.maximum(m_bos_after - m_bos_before, 0)

    # alpha_eq, see eq. (2.16) in 1903.09642
    # the hydrodynamic alphas do not depend on the decoupled radiation bath!
    alpha_eq = T**3 / rho_rad_PTsector * np.sum(delta_m_bos * gauge_coupling**2 * dof_bos * is_physical, axis=-1)
    alpha_inf = T**2 / (24 * rho_rad_PTsector) * m2factor
    # alpha_hyd = (DeltaE + 3 * DeltaV) / (4 * rho_rad_PTsector)
    alpha_hyd = (theta_sym - theta_bro) / \
        (3 * (V0-pot.Vtot(high_phi, T, include_decoupled=False) +
              pot.energyDensity(high_phi, T, include_decoupled=False)))

    return alpha_p, alpha_theta, alpha_e, alpha_hyd, alpha_inf, alpha_eq


def calc_betaH_S3_approx(T, outdict, pot, phase_sym, phase_bro, tmin, tmax, verbose=False):
    """
    Calculate the betaH parameter at temperature T
    using the derivative of the bounce action with respect to
    the temperature.
    """
    S = calcAction(pot, T, phase_sym, phase_bro, outdict)
    if np.isinf(S):
        if verbose:
            print("Warning: S(T) is inf")
        return np.nan

    dT = T * 1e-3
    # if dT is too large, set it to a smaller value
    if T + dT > tmax:
        dT = (tmax - T) / 100

    # Try to calculate the derivative to the right of T
    SdT = calcAction(pot, T + dT, phase_sym, phase_bro, outdict)
    betaH = (SdT - S) / dT - S / T
    while np.isinf(SdT) and dT / T > 1e-15:
        if verbose:
            print("SdT is inf, reducing dT to ", dT, " and trying again")
        dT *= 0.1
        SdT = calcAction(pot, T + dT, phase_sym, phase_bro, outdict)
        betaH = (SdT - S) / dT - S / T

    # if SdT is inf, we cannot calculate betaH
    # if beta/H is negative, there was a numerical issue
    if np.isinf(SdT) or betaH < 0:
        # Try to calculate the derivative to the left of T
        dT = T * 1e-3
        if T - dT < tmin:
            dT = (T - tmin) / 100
        SdT_ = calcAction(pot, T - dT, phase_sym, phase_bro, outdict)
        betaH = (S - SdT_) / dT - S / T
        while np.isinf(SdT_) and dT / T > 1e-15:
            if verbose:
                print("SdT is still inf, reducing dT to ", dT, " and trying again")
            dT *= 0.1
            SdT_ = calcAction(pot, T - dT, phase_sym, phase_bro, outdict)
            betaH = (S - SdT_) / dT - S / T
        if np.isinf(SdT_):
            if verbose:
                print("Warning: SdT_ is inf, cannot calculate betaH")
            return np.nan

    if np.isinf(SdT):
        if verbose:
            print("Warning: S(T+dT) is inf, cannot calculate betaH")
        return np.nan

    if betaH <= 0 and verbose:
        print("Warning: betaH is negative!")

    return betaH


def calcMeanBubbleSeparation(T, Tmax, Sint, Pint, Hint):
    """Calculate the mean bubble separation at temperature T.

    Parameters
    ----------
    T : float
        Symmetric phase temperature
    Tmax : float
        Upper limit of the integral, usually the nucleation temperature
    phase_sym : scipy.interpolate.interp1d object
        The symmetric phase
    Sint : scipy.interpolate.interp1d object
        The action
    Pint : scipy.interpolate.interp1d object
        The true vacuum fraction.
    Hint : scipy.interpolate.interp1d object
        The Hubble rate
    Returns
    ----------
    R : float
        Mean bubble separation."""

    # See eq. (5.42) in 2305.02357
    Tr = np.linspace(T, Tmax, 10_000)
    integrant = Gamma(Tr, Sint(Tr)) * (1 - Pint(Tr)) / (Tr * Hint(Tr)) * (T / Tr) ** 3
    integrant[np.isnan(integrant)] = 0  # Replace NaNs (due to infinite action) with 0
    res = integrate.trapezoid(integrant, x=Tr)
    res = np.power(res, -1 / 3)
    return res


def calcMeanBubbleSeparationwExp(T, Tmax, Sint, Pint, Hint, CSint, SFRint):
    """Calculate the mean bubble separation at temperature T.
    Include the effect of cosmic expansion.

    Parameters
    ----------
    T : float
        Symmetric phase temperature
    Tmax : float
        Upper limit of the integral, usually the nucleation temperature
    phase_sym : scipy.interpolate.interp1d object
        The symmetric phase
    Sint : scipy.interpolate.interp1d object
        The action
    Pint : scipy.interpolate.interp1d object
        The true vacuum fraction.
    Hint : scipy.interpolate.interp1d object
        The Hubble rate
    CSint: scipy.interpolate.interp1d object
        The sound speed squared
    SRFint: scipy.interpolate.interp1d object
        The scale factor ratio
    Returns
    ----------
    R : float
        Mean bubble separation."""

    # See eq. (5.42) in 2305.02357
    Tr = np.linspace(T, Tmax, 10_000)
    integrant = Gamma(Tr, Sint(Tr)) * (1 - Pint(Tr)) / (3 * Hint(Tr) * Tr * CSint(Tr) ) \
        * SFRint(Tr)**3
    integrant[np.isnan(integrant)] = 0  # Replace NaNs (due to infinite action) with 0
    res = integrate.trapezoid(integrant, x=Tr)
    res = np.power(res, -1 / 3)
    return res



def calcTf(Tperc, Tlow, Pint, pot, verbose=False):
    """Calculate the final temperature at which the transition ends.

    Parameters
    ----------
    Tperc : float
        Symmetric phase percolation temperature, i.e. maximally possible value of the final temperature
    Tlow : float
        Lowest temperature considered in the percolation computation. At this temperature, the true vacuum
        fraction Pint is >= f_final ~ 0.99. The final temperature must thus be <= Tlow.
    phase_sym : scipy.interpolate.interp1d object
        The symmetric phase
    Sint : scipy.interpolate.interp1d object
        The action
    Pint : scipy.interpolate.interp1d object
        The true vacuum fraction.
    Returns
    ----------
    Tf : float
        Final temperature."""

    f_final = pot.config.percolationConf.f_final  # true vacuum fraction at final temperature

    # Step 1: Check if the true vacuum fraction at Tlow is >= f_final
    if Pint(Tlow) < f_final:
        # It looks like the transition cannot completely finish
        msg = "The transition cannot completely finish, Pint(Tlow) < f_final: " + str(Pint(Tlow)) + " < " + str(f_final)
        raise errors.EternalInflationError(msg)

    # Step 2: Find Tf by solving Pint(Tf) = f_final
    try:
        Tf = optimize.brentq(lambda TSYM: Pint(TSYM) - f_final, Tlow, Tperc)
    except Exception as e:
        msg = "Error in calculating Tf: " + str(e)
        if verbose:
            print(msg)
        raise errors.PercolationError(e)

    return Tf


# ==================================================
# Nucleation computation
# ==================================================


def findTunnelingTmax(DV: callable, Tmin: float, Tmax: float, tol=1e-15, maxiter=100):
    """Finds the largest temperature Tmax in [Tmin, Tmax] such that
    V(start_phase.valAt(T), T) > V(end_phase.valAt(T), T).

    Important: `tol=1e-15` is required to to find the right Tmax,
               `tol=1e-8` is not enough!

    Parameters
    ----------
    DV : callable
    Tmin : float
    Tmax : float
    tol : float, optional
    maxiter : int, optional

    Returns
    -------
    float
        The maximal temperature where V(start_phase, Tmax) > V(end_phase, Tmax).
    """

    if DV(Tmax) > 0:
        return Tmax
    # Search for the root of DV(T) = 0 in [Tmin, Tmax]
    try:
        Tmax_new = optimize.brentq(DV, Tmin, Tmax, xtol=tol, maxiter=maxiter)
        return Tmax_new
    except ValueError:
        # Maybe DV(Tmin) < 0 for numerical reasons:
        try:
            Tmax_new = optimize.brentq(DV, (Tmin + Tmax)/2.0, Tmax, xtol=tol, maxiter=maxiter)
            return Tmax_new
        except ValueError:
            # No crossing found; fall back to Tmin.
            return Tmin


def findTminFromUShapedNucl(Tmin, Tmax, args, start_phase, end_phase, outdict,
                            Ttol, maxiter, CF, verbose, nuclCriterion):
    """Find the minimum of the nucleation criterion between `Tmin` and `Tmax`

    Parameters
    ----------

    Returns
    -------
    float | None:
        Temperature of the minimum of the nucleation criterion.
        If it is a hidden second order transition, return None
    """
    ntest = 100
    Tmin_new = 0
    try:
        # Try to find the minimum of the tunneling action
        # between Tmin and Tmax
        def maximizer(x, *args):
            # This function is used to find the maximum of the
            # nucleation criterion, corresponding to the minimum of
            # a U-shaped action.
            return -evalNucleationCriterion(x, *args)

        Ttest = np.nan

        # This skips Tmin and Tmax, which have already been checked and would
        # yield boundary errors in the bracketing algorithm
        Trange = np.linspace(Tmin, Tmax, ntest, endpoint=False)[1:]
        for t in Trange:
            nc = evalNucleationCriterion(t, *args)
            if not np.isinf(nc):
                Ttest = t
                # usually this it the case after one or two iterations
                break

        if np.isnan(Ttest):
            # It could be that the action is actually very hard to compute
            # because the transitions is extremely weak. In that case the
            # action jumps between -inf and +inf. We can then take the highest
            # temperature for which the nucleation criterion was found to be
            # fulfilled, even if it is infinitely large due to the numerical
            # precision.
            if any([outdict[T]["trantype"] == 2 for T in outdict]):
                max_key = max(T for T in outdict if outdict[T]["trantype"] == 2)
                if verbose:
                    print(
                        "Found a second-order transition at T = ",
                        max_key * CF,
                        "GeV. This was a little hard to find because it was "
                        "treated as a very weak first-order transition.",
                    )
                rdict = outdict[max_key]
                rdict["Tcrit"] = rdict["Tnuc"] = max_key
                return None

            if all([evalNucleationCriterion(T, *args) < 0 if T <= Tmax and T >= Tmin else True for T in outdict]):
                # This means the nucleation criterion is never fulfilled between Tmin and Tmax.
                # The barrier is too high to allow for the nucleation to be possible.
                raise errors.NucleationError("Nucleation criterion failed: too low at all temperatures.")
            raise errors.NucleationError(
                "Nucleation criterion failed due to unknown reason: No finite nucleation "
                "criterion found in the range between Tmin and Tmax."
            )
        result = optimize.minimize_scalar(
            maximizer,
            bracket=(Tmin, Ttest, Tmax),
            args=args,
            method="brent",
            options={"xtol": (Tmax - Tmin) * 1e-5, "maxiter": maxiter, "disp": 0},
        )
        if result.success:
            # We found a minimum of the action! Let's check if the nucleation
            # criterion is fulfilled at this temperature.
            Tmin_u = float(np.squeeze(result.x))  # this is a hack to make the np array hashable
            action_u = outdict[Tmin_u]["action"]
            ncMin_u = nuclCriterion(action_u, Tmin_u, start_phase, end_phase)

            Tmin_new = Tmin_u

            if verbose:
                print("Finding Tmin_u done! Tmin_u = ", result.x * CF, "GeV")
            # Note: it can also happen that a minimum is found in
            # the range of Tmin and Tmax, even though there is no
            # actual minimum of the action. This can happen if the
            # action is u-shaped and so large that the nucleation
            # criterion is not fulfilled at all. In this case, one
            # still obtains a minimum of the action at an arbitrary
            # temperature because the nucleation criterion evaluates
            # to - inf everywhere, such that the "maximizer" is inf
            # everywhere. This is not a problem, because in any case
            # we are checking again if the nucleation criterion is fulfilled
            # at the found minimum of the action.

            if verbose:
                print("action_u = ", action_u * CF, "GeV")
                print("ncMin_u = ", ncMin_u)
            if ncMin_u < 0:
                if verbose:
                    print("Nucleation criterion not fulfilled at the minimum of the u-shaped action.")
                raise errors.NucleationError(
                    "Nucleation criterion failed because the nucleation rate is too low at all temperatures."
                )
            if verbose:
                print("Nucleation criterion fulfilled at the minimum of the u-shaped action.")
        else:
            if verbose:
                print("Finding Tmin_u failed: ", result.message)
            raise errors.NucleationError("Nucleation criterion failed in the finding of Tmin_u:" + str(result.message))
    except Exception as err:
        raise errors.NucleationError("Nucleation calculation failed: " + str(err))

    return Tmin_new


def computeNucleationTemperature(V, dV, d2V, start_phase, end_phase,
                                 CF, Ttol=1e-8, maxiter=100, phitol=1e-8,
                                 overlapAngle=45.0, nuclCriterion=lambda S, T: 140 - S / (T + 1e-100),
                                 approximate_strength_threshold=1e-4, verbose=False,
                                 fullTunneling_params={}):
    """Find the instanton and nucleation temperature for tunneling from
    `start_phase` to `end_phase`.

    Parameters
    ----------
    V, dV : callable
        The potential V(x,T) and its gradient.
    start_phase : PhaseInfo
        The metastable phase from which tunneling occurs.
    end_phase : PhaesInfo
        The destination phase.
    Tmax : float
        The highest temperature at which to try tunneling.
    Ttol : float, optional
        Tolerance for finding the nucleation temperature.
    maxiter : int, optional
        Maximum number of times to try tunneling.
    phitol : float, optional
        Tolerance for finding the minima.
    overlapAngle : float, optional
        If two phases are in the same direction, only try tunneling to the
        closer one. Set to zero to always try tunneling to all available phases.
    nuclCriterion : callable
        Function of the action *S* and temperature *T*. Should return 0 for the
        correct nucleation rate, > 0 for a low rate and < 0 for a high rate.
        Defaults to ``S/T - 140``.
    verbose : bool
        If true, print a message before each attempted tunneling.
    fullTunneling_params : dict
        Parameters to pass to :func:`pathDeformation.fullTunneling`.

    Returns
    -------
    dict or None
    A description of the tunneling solution at the nucleation temperature,
    or None if there is no found solution. Has the following keys:

    - *Tnuc* : the nucleation temperature
    - *low_vev, high_vev* : vevs for the low-T phase (the phase that the
        instanton tunnels to) and high-T phase (the phase that the instanton
        tunnels from).
    - *low_phase, high_phase* : identifier keys for the low-T and high-T
        phases.
    - *action* : The Euclidean action of the instanton.
    - *instanton* : Output from :func:`pathDeformation.fullTunneling`, or
        None for a second-order transition.
    - *trantype* : 1 or 2 for first or second-order transitions.
    """
    outdict = {}  # keys are T values
    args = (start_phase, end_phase, V, dV, d2V, phitol, overlapAngle, nuclCriterion,
            fullTunneling_params, verbose, outdict, CF)
    Tmin = max(start_phase.Tmin, end_phase.Tmin)

    # We want Tmax to be the highest temperature at which tunneling is theoretically
    # possible, i.e. where the potential difference between the two phases is at least
    # DVmin. Find the largest Tmax for which V(start_phase) > V(end_phase)

    def DV(T):
        # Be very accurate about the minima!
        x0 = findLocalMinimum(start_phase.valAt(T), T, dV, d2V)
        x1 = findLocalMinimum(end_phase.valAt(T), T, dV, d2V)
        # This below is necessary because sometimes the phase tracing is not
        # accurate enough.
        if np.sqrt(np.sum((x0 - start_phase.valAt(T))**2)) > 1:
            x0 = start_phase.valAt(T)
        if np.sqrt(np.sum((x1 - end_phase.valAt(T))**2)) > 1:
            x1 = end_phase.valAt(T)
        x = x0 - x1
        dist = np.sqrt(x.dot(x))
        # Add small epsilon to pot. value in end phase to ensure breaking of degeneracy!
        # This might cause problems for very weak transitions,but on the otherhand, they
        # are not interesting
        eps = dist * 1e-3
        return V(x0, T) - (V(x1, T) + eps)

    # Always check that the Tmax is where DV is > 0
    Tmax = findTunnelingTmax(DV, Tmin, end_phase.Tmax)

    # Check for "hidden"" second order transitions
    if not checkNucleationPossibility(Tmin, Tmax, args, outdict, V, dV,
                                      start_phase, end_phase, CF,
                                      approximate_strength_threshold, verbose):
        rdict = dict(
            Tcrit=Tmax,
            Tnuc=Tmax,
            Tperc=Tmax,
            low_vev=end_phase.valAt(Tmax),
            high_vev=start_phase.valAt(Tmax),
            low_phase=end_phase.key,
            high_phase=start_phase.key,
            action=np.nan,
            full_tunneling_info=np.nan,
            instanton=np.nan,
            trantype=2,
        )
        rdict["Tcrit"] = rdict["Tnuc"] = Tmax
        return rdict

    ncrit_Tmin = evalNucleationCriterion(Tmin, *args)
    ncrit_Tmax = evalNucleationCriterion(Tmax, *args)

    # Is it a second order transition?
    # Does the nucleation happen already at Tmax?
    # This could be a second order transition or an
    # immediate transition after a another one
    if ncrit_Tmax >= 0:
        if verbose:
            print(f"Nucl. crit fullfilled at Tmax = {Tmax*CF}!")
        Tnuc = Tmax
        rdict = outdict[Tmax]
        rdict["Tcrit"] = Tmax
        if rdict["trantype"] != 2:
            filtered_outdict = {k: {kk: vv for kk, vv in v.items() if kk != "instanton"} for k, v in outdict.items()}
            filtered_outdict = dict(sorted(filtered_outdict.items()))
            rdict["full_tunneling_info"] = filtered_outdict
        # assert rdict['trantype'] == 2
        return rdict

    # Is the nucleation criterion fulfilled at Tmin?
    if ncrit_Tmin < 0:
        if verbose:
            print("Nucleation criterion not fulfilled at Tmin")
        # The nucleation criterion is not fulfilled at the boundaries of
        # the temperature interval. It could be that the action is
        # U-shaped such that the nucleation criterion is not fulfilled
        # at Tmin and Tmax, but maybe somewhere in between.
        # In this case we need to find the minimum of the action
        # between Tmin and Tmax. This is done by minimizing the
        # tunneling action at T.
        if verbose:
            print("The bounce action might be U-shaped. Try to find the minimum of it")
            print("between Tmin = ", Tmin * CF, "GeV and Tmax = ", Tmax * CF, "GeV.")
        Tmin = findTminFromUShapedNucl(
            Tmin, Tmax, args, start_phase, end_phase, outdict, Ttol, maxiter, CF, verbose, nuclCriterion
        )
        # Hidden second order transition?
        if Tmin is None:
            rdict = dict(Tcrit=Tmax, Tnuc=Tmax, Tperc=Tmax,
                         low_vev=end_phase.valAt(Tmax),
                         high_vev=start_phase.valAt(Tmax),
                         low_phase=end_phase.key,
                         high_phase=start_phase.key,
                         action=np.nan, full_tunneling_info=np.nan,
                         instanton=np.nan, trantype=2)
            rdict["Tcrit"] = rdict["Tnuc"] = Tmax
            return rdict

    # Check if the tunneling is energetically allowed between Tmin and Tmax
    try:
        Tnuc = optimize.brentq(evalNucleationCriterion, Tmin, Tmax, args=args, xtol=Ttol, maxiter=maxiter, disp=False)
        if verbose:
            print("Brentq done after first try. Tnuc = ", Tnuc * CF, "GeV")
    except ValueError:
        print("Brentq failed for first try, retrying with larger Ttol.")
        try:
            Tnuc = optimize.brentq(
                evalNucleationCriterion, Tmin, Tmax, args=args, xtol=Ttol * 10, maxiter=maxiter, disp=False
            )
            if verbose:
                print("Brentq done after second try! Tnuc = ", Tnuc * CF, "GeV")
        except Exception:
            if verbose:
                print("Brentq failed again after increasing the numerical tolerance by a factor 10.")
            raise errors.NucleationError("Nucleation criterion failed due to unknown reason.")

    try:
        if Tnuc in outdict:
            rdict = outdict[Tnuc]
            # Remove 'instanton' key from each sub-dictionary in outdict before saving, because
            # it is no longer needed and could cause memory issues, becasuse it can become quite
            # large.
            filtered_outdict = {k: {kk: vv for kk, vv in v.items() if kk != "instanton"} for k, v in outdict.items()}
            # This is new and allows to re-use the computed tunneling solutions also at a
            # later part of the code, especially the percolation computation
            filtered_outdict = dict(sorted(filtered_outdict.items()))
            rdict["full_tunneling_info"] = filtered_outdict
        else:
            raise errors.NucleationError(
                "Tnuc not in outdict. This can only happen if the nucleation "
                "criterion could not be fulfilled at any temperature."
            )
    except UnboundLocalError as err:
        if verbose:
            print(
                "Tnuc was never defined. This happens if the nucleation "
                "criterion could not be fulfilled at any temperature, but a "
                "np.nan was returned. This is likely a numerical bug."
            )
        raise errors.NucleationError(
            "Nucleation criterion failed, probably due to a numerical error: "
            + str(err)
        )

    if "trantype" in rdict and rdict["trantype"] > 0:
        if rdict["trantype"] == 2:
            # This is not a first-order transition, but actually a second-order one. This was
            # however only found here because the barrier was tiny and the nucleation
            # criterion is fulfilled trivially. We create a second-order transition dictionary
            # and return it.
            if verbose:
                print(
                    "The transition is actually a second-order transition, "
                    "because the action is zero at the transition."
                )
            rdict["Tcrit"] = rdict["Tnuc"] = Tnuc
            return rdict
        rdict["Tcrit"] = Tmax
        return rdict
    elif "trantype" in rdict and rdict["trantype"] == 0:
        # Check if there is a second-order transition just below the found Tnuc
        keys = sorted(key for key in outdict.keys() if key != "full_tunneling_info" and key < Tnuc)
        max_key = keys[-1] if keys else None
        if max_key is not None and outdict[max_key]["trantype"] == 2:
            # If there is a second-order transition just below the found Tnuc, return it
            if verbose:
                print(
                    "The found transition had an infinite action. But found a "
                    "second-order transition just below the found Tnuc. "
                    "Returning it. Tcrit = ",
                    max_key * CF,
                    "GeV",
                )
            rdict = outdict[max_key]
            rdict["Tcrit"] = rdict["Tnuc"] = max_key
            return rdict
        else:
            if verbose:
                print(
                    "The transition has action = 0 and trantype = 0, which means "
                    "that it is a second-order transition, but no second-order "
                    "transition was found just below the found Tnuc. This is "
                    "likely a numerical issue."
                )
            return rdict
    else:
        raise errors.NucleationError(
            "The nucleation computation failed: Either the barrier is too high or the transition is second order."
        )


def checkNucleationPossibility(Tmin, Tmax, args, outdict, V, dV, start_phase,
                               end_phase, conversionFactor, apprx_strength_thr,
                               verbose) -> tuple[bool, dict]:
    """Do some checks to see if tunneling is possible.

    Parameters
    ----------
    tr : TransitionInfo
        The transition information
    Returns
    -------
    tuple[bool, dict] :
        (True, {}) if possible otherwise, (False, {returndict})."""
    # We want Tmax to be the highest temperature at which tunneling is theoretically
    # possible, i.e. where the potential difference between the two phases is at least
    # DVmin. Find the largest Tmax for which V(start_phase) > V(end_phase)

    rdict = {}
    # Check 1 for "hidden" second-order transitions: action is zero at Tmax
    nuclCritTmax = evalNucleationCriterion(Tmax, *args)
    if np.isinf(nuclCritTmax):
        if Tmax in outdict:
            rdict = outdict[Tmax]
            if "trantype" in rdict and rdict["trantype"] == 2:
                if verbose:
                    print(
                        "The transition is actually a second-order transition "
                        "because the action is zero at the transition."
                    )
                return False

    # Check 2 for "hidden" second-order transitions: the strength of the transition would in any case be tiny
    def approx_strength(T):
        # Approximate strength parameter of the transition, if it happened at temperature T
        DV = V(start_phase.valAt(T), T) - (V(end_phase.valAt(T), T))
        return np.abs(DV) / T**4

    approx_strength_range = np.array([approx_strength(T) for T in np.linspace(Tmin, Tmax, 100)])
    if np.all(approx_strength_range < apprx_strength_thr):
        if verbose:
            print(f"PT strength {np.max(approx_strength_range)} below threshold: {apprx_strength_thr}.")
            print(f"This is treated as a second-order transition at Tmax = {Tmax * conversionFactor} GeV.")
            # Create a second-order transition dictionary
        return False

    # Tunneling possible, return true and empty dict.
    return True


def computeNucleationTemperature2(
    tr,
    V,
    dV,
    d2V,
    start_phase,
    end_phase,
    Tmax,
    conversionFactor,
    Ttol=1e-8,
    maxiter=100,
    phitol=1e-8,
    overlapAngle=45.0,
    nuclCriterion=lambda S, T: 140 - S / (T + 1e-100),
    approximate_strength_threshold=1e-4,
    verbose=False,
    fullTunneling_params={},
):
    """Refactored version of `computeNucleationTemperature`
    Find the instanton and nucleation temeprature for tunneling from
    `start_phase` to `end_phase`.

    Parameters
    ----------
    tr: TransitionInfo
        Object containing the transition informations
    V, dV, d2V : callable
        The potential V(x,T) and its gradient.
    start_phase : PhaseInfo
        The metastable phase from which tunneling occurs.
    end_phase : PhaesInfo
        The destination phase.
    Tmax : float
        The highest temperature at which to try tunneling.
    Ttol : float, optional
        Tolerance for finding the nucleation temperature.
    maxiter : int, optional
        Maximum number of times to try tunneling.
    phitol : float, optional
        Tolerance for finding the minima.
    overlapAngle : float, optional
        If two phases are in the same direction, only try tunneling to the
        closer one. Set to zero to always try tunneling to all available phases.
    nuclCriterion : callable
        Function of the action *S* and temperature *T*. Should return 0 for the
        correct nucleation rate, > 0 for a low rate and < 0 for a high rate.
        Defaults to ``S/T - 140``.
    verbose : bool
        If true, print a message before each attempted tunneling.
    fullTunneling_params : dict
        Parameters to pass to :func:`pathDeformation.fullTunneling`.

    Returns
    -------
    dict or None
    A description of the tunneling solution at the nucleation temperature,
    or None if there is no found solution. Has the following keys:

    - *Tnuc* : the nucleation temperature
    - *low_vev, high_vev* : vevs for the low-T phase (the phase that the
        instanton tunnels to) and high-T phase (the phase that the instanton
        tunnels from).
    - *low_phase, high_phase* : identifier keys for the low-T and high-T
        phases.
    - *action* : The Euclidean action of the instanton.
    - *instanton* : Output from :func:`pathDeformation.fullTunneling`, or
        None for a second-order transition.
    - *trantype* : 1 or 2 for first or second-order transitions.
    """
    # If we need to calculate the bounce action brute-force if it is U-shaped, this is
    # the maximum number of test points to use.
    ntest = 100
    CF = conversionFactor
    outdict = {}  # keys are T values
    # args for the tunneling function
    args = (start_phase, end_phase, V, dV, d2V, phitol, overlapAngle,
            nuclCriterion, fullTunneling_params, verbose, outdict, CF)
    # usually: lowest temperature of starting phase. But it can happen that for some
    # other reason, the start and end phases have been swapped. In that case, we don't
    # want to make an error here.
    Tmin = max(start_phase.Tmin, end_phase.Tmin)
    # usually highest temperature of end phase, unless it is below Tmin (which is bad)
    T_highest_other = max(Tmin, end_phase.Tmax)
    Tmax = min(Tmax, T_highest_other)

    def DV(T):
        # Add small epsilon to pot. value in end phase to ensure breaking of degeneracy!
        x0 = start_phase.valAt(T)
        x1 = end_phase.valAt(T)
        dist = np.sqrt((x0-x1)**2)
        eps = dist * 1e-3
        return V(start_phase.valAt(T), T) - (V(end_phase.valAt(T), T) + eps)

    # Check if the Potential difference is large enough to tunnel (numerically)
    Tmax = findTunnelingTmax(DV, Tmin, Tmax)

    # check if tunneling is possible
    tPossible, rdict = checkNucleationPossibility(Tmin, Tmax, args, outdict, V, dV,
                                                  start_phase, end_phase, CF,
                                                  approximate_strength_threshold, verbose)
    if not tPossible:
        return rdict

    ncrit_Tmin = evalNucleationCriterion(Tmin, *args)
    ncrit_Tmax = evalNucleationCriterion(Tmax, *args)

    # Is it a second order transition?
    if ncrit_Tmax > 0:
        if verbose:
            print(f"Nucl. crit fullfilled at Tmax = {Tmax*CF}!")
        Tnuc = Tmax
        rdict = outdict[Tmax]
        assert rdict["trantype"] == 2
        return rdict

    # Is there more than one bubble nucleated at Tmin
    # Or is the nucleation criterion u-shaped?
    if ncrit_Tmin < 0:
        # find a temperature where the the criterion becomes positive (> 1 bubble)
        def abort_fmin(T, outdict=outdict, nc=evalNucleationCriterion):
            val = nc(T, *args)
            if val > 0 and not np.isnan(val):
                raise StopIteration(T)

        Tguess = np.nan
        # first search for a non-inf value of the nucleation criterion:
        for t in np.linspace(Tmin, Tmax, ntest, endpoint=False)[1:]:
            nc = evalNucleationCriterion(t, *args)
            if not np.isinf(nc):
                Tguess = t
        # Is the nucleation rate to low everywhere?
        if np.isnan(Tguess):
            if any([outdict[T]["trantype"] == 2 for T in outdict]):
                max_key = max(T for T in outdict if outdict[T]["trantype"] == 2)
                rdict = outdict[max_key]
                rdict["Tcrit"] = rdict["Tnuc"] = max_key
                return rdict
            if all([evalNucleationCriterion(T, *args) < 0 if T <= Tmax and T >= Tmin else True for T in outdict]):
                # This means the nucleation criterion is never fulfilled between Tmin and Tmax. The barrier is too high
                # to allow for the nucleation to be possible.
                raise errors.NucleationError("Nucleation rate too low at all temperatures.")

        try:
            Tmin = abs(
                optimize.fmin(
                    lambda x: -evalNucleationCriterion(abs(x), *args), Tguess, callback=abort_fmin, maxiter=20, disp=0
                )[0]
            )
            if evalNucleationCriterion(Tmin, *args) < 0:
                # The nucleation criterion is not fullfilled anywhere!
                raise errors.NucleationError("Nucleation rate to low at all temperatures!")
        except StopIteration as err:
            Tmin = abs(err.args[0])

    try:
        Tnuc = optimize.brentq(evalNucleationCriterion, Tmin, Tmax, args=args, xtol=Ttol, maxiter=maxiter, disp=False)
    except Exception as e:
        print("Nucleation calculation failed!")
        print(e)
        raise errors.NucleationError("Unknown nulceation error!")

    rdict = outdict[Tnuc]
    filtered_outdict = {k: {kk: vv for kk, vv in v.items() if kk != "instanton"} for k, v in outdict.items()}
    # This is new and allows to re-use the computed tunneling solutions also at a
    # later part of the code, especially the percolation computation
    filtered_outdict = dict(sorted(filtered_outdict.items()))
    rdict["full_tunneling_info"] = filtered_outdict
    return rdict


def evalNucleationCriterion(T, start_phase, end_phase, V, dV,
                            d2V, phitol, overlapAngle, nuclCriterion,
                            fullTunneling_params, verbose, outdict,
                            conversionFactor):
    """Find the lowest action tunneling solution.

    Return ``nuclCriterion(S,T)``, and store a dictionary describing the
    transition in outdict for key `T`."""
    try:
        T = T[0]  # need this when the function is run from optimize.fmin
    except Exception:
        pass

    if T in outdict:
        return nuclCriterion(outdict[T]["action"], T, start_phase, end_phase)

    # Loop through all the phases, adding acceptable minima
    x0 = findLocalMinimum(start_phase.valAt(T), T, dV, d2V)
    # Check if we accidentally hopped into the other minimum
    if np.sqrt(np.sum((x0 - start_phase.valAt(T))**2)) > 1:
        x0 = start_phase.valAt(T)
    V0 = V(x0, T)
    p = end_phase
    if p.Tmin > T:
        if verbose:
            print(
                "Phase %s not valid at T = %g" % (p.key, T * conversionFactor),
                "GeV < Tmin = ",
                p.Tmin * conversionFactor,
                "GeV",
            )
        # raise Exception("Tunneling not possible, low phase does not exist at T = ", T)
        return np.inf

    if p.Tmax < T:
        if verbose:
            print(
                "Phase %s not valid at T = %g" % (p.key, T * conversionFactor),
                "GeV > Tmax = ",
                p.Tmax * conversionFactor,
                "GeV",
            )
            # raise Exception("Tunneling not possible, low phase does not exist at T = ", T)
        return -np.inf

    x1 = findLocalMinimum(p.valAt(T), T, dV, d2V)
    if np.sqrt(np.sum((x1 - end_phase.valAt(T))**2)) > 1:
        x1 = end_phase.valAt(T)
    V1 = V(x1, T)

    if V1 >= V0:
        if verbose:
            print(
                "Tunneling energetically not possible."
                + "Tmax might be too high or the numerical accuracy is not sufficient."
                + " This is not a problem, just a warning."
            )

    tdict = dict(low_vev=x1, high_vev=x0, Tnuc=T, low_phase=end_phase.key, high_phase=start_phase.key)
    # tunnel_list.append(tdict)
    # This is a check to see if the two phases are in the same direction.
    # Future work, focusing on multiple discrete symmetries might
    # act here.
    # If two phases are in the same direction, we only want to try tunneling to the closer one:
    # if overlapAngle > 0:
    #     excluded = []
    #     cos_overlap = np.cos(overlapAngle * np.pi/180)
    #     for i in range(1, len(tunnel_list)):
    #         for j in range(i):
    #             xi = tunnel_list[i]['low_vev']
    #             xj = tunnel_list[j]['low_vev']
    #             xi2 = np.sum((xi-x0)**2)
    #             xj2 = np.sum((xj-x0)**2)
    #             dotij = np.sum((xj-x0)*(xi-x0))
    #             if dotij >= np.sqrt(xi2*xj2) * cos_overlap:
    #                 excluded.append(i if xi2 > xj2 else j)
    #     for i in sorted(excluded)[::-1]:
    #         del tunnel_list[i]

    outdict = bounceAction(
        T, V, dV, outdict, tdict, verbose=verbose, conversionFactor=conversionFactor, **fullTunneling_params
    )

    S = outdict[T]["action"]
    crit = nuclCriterion(S, T, start_phase, end_phase)

    return crit


def makeSecondOrderDict(transition, conversionFactor: float) -> dict:
    """Create a dictionary describing a second-order phase transition."""
    transition_dict = dict(
        Tcrit=transition.Tcrit,
        Tcrit_SM_GeV=transition.Tcrit * conversionFactor,
        Tnuc=transition.Tnuc,
        Tperc=transition.Tperc,
        low_vev=transition.low_vev,
        high_vev=transition.high_vev,
        low_phase=transition.low_phase,
        high_phase=transition.high_phase,
        action=np.nan,
        trantype=2,
    )
    return transition_dict


def findLocalMinimum(
    X: np.ndarray, T: float, dV: callable, d2V: callable, phitol: float = 1e-6, step_size: float = 0.1
) -> np.ndarray:
    r"""Find the local minimum with the algorithm developed
    in BSMPT

    .. math::
        \vec{\epsilon} = H^{-1}\nabla V

        \vec{\phi}_{min}^{i+1} = \vec{\phi}_{min}^i - \vec{\epsilon}

    Parameters
    ----------
    X : np.ndarray
        The initial guess for the location of the minimum
    T : float
        Temperature
    dV : callable
        Gradient of the potenital :math:`\nabla V(X, T)`
    d2V : callable
        Hessian matrix of the potenital :math:`\partial^2 V(X,T)/\partial \phi_i\partial\phi_j`
    phitol : float
        The desired accuracy of the minimum
    step_size : float
        The multiplyier controlling the step size in the direction of the minimum.

    Returns
    -------
    np.ndarray:
        The minimum."""
    hess_offset = np.diag(np.ones_like(X) * 1e-5)
    niter = 0
    while True:
        niter += 1
        Hess = d2V(X, T)
        # Stop if zero eigenvalues
        if np.linalg.det(Hess) == 0:
            break
        Hinv = np.linalg.inv(Hess + hess_offset)
        eps = np.matmul(Hinv, dV(X, T))
        X = X - eps * step_size
        if np.sqrt(np.sum(eps * eps)) <= phitol:
            break
        if niter >= 200:
            break
    return X
