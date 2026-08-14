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

import copy
from dataclasses import dataclass
import math
import numpy as np
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
from transitionlistener.nucleation import computeNucleationTemperature


def _maybe_dump_percolation_spline(state, stage: str, pot=None) -> None:
    """Dump the (T, P) spline of the current percolation state to disk.

    Active only when the environment variable ``TL_PERC_DUMP_DIR`` is set.
    Saves a linspace of ``TL_PERC_DUMP_N`` (default 10000) temperature points
    and a monotone-preserving interpolation of the true-vacuum fraction
    ``P_t(T)`` as ``<stage>.npz`` inside the dump directory.
    """
    import os
    dump_dir = os.environ.get("TL_PERC_DUMP_DIR")
    if not dump_dir or state is None:
        return
    try:
        n_support = int(os.environ.get("TL_PERC_DUMP_N", "10000"))
        T = np.asarray(state.TSYM, dtype=float)
        P = np.asarray(state.Pr, dtype=float)
        finite = np.isfinite(T) & np.isfinite(P)
        T = T[finite]
        P = P[finite]
        if T.size < 4:
            return
        order = np.argsort(T)
        T = T[order]
        P = P[order]
        _, unique_idx = np.unique(T, return_index=True)
        T = T[unique_idx]
        P = P[unique_idx]
        P = np.clip(P, 0.0, 1.0)
        Pint = interpolate.PchipInterpolator(T, P, extrapolate=True)
        T_grid = np.linspace(float(T.min()), float(T.max()), n_support)
        P_grid = np.clip(Pint(T_grid), 0.0, 1.0)
        os.makedirs(dump_dir, exist_ok=True)
        extra = {}
        if pot is not None:
            if hasattr(pot, "conversionFactor"):
                extra["CF"] = float(pot.conversionFactor)
            if hasattr(pot, "g"):
                extra["g"] = float(pot.g)
            if hasattr(pot, "l"):
                extra["lambda"] = float(pot.l)
        np.savez(os.path.join(dump_dir, f"{stage}.npz"),
                 T=T_grid, P=P_grid, T_support=T, P_support=P, **extra)
    except Exception:
        pass


@dataclass
class PercolationGrid:
    """Initial temperature grid and traced phase-overlap bounds for percolation."""

    TSYM: np.ndarray
    Tstart: float
    TpercApprox: float
    tmin: float
    tmax: float
    TBROmin: float
    TBROmax: float
    free_support_bank: np.ndarray | None = None


@dataclass
class PercolationState:
    """Current percolation profile plus support-bank diagnostics."""

    TSYM: np.ndarray
    Sr: np.ndarray
    Hr: np.ndarray
    Pr: np.ndarray
    Tb: np.ndarray
    explored_tmin: float | None = None
    rebuild_count: int = 0
    support_bank: np.ndarray | None = None
    free_support_bank: np.ndarray | None = None
    action_temperatures: np.ndarray | None = None
    cold_tail_last_temperature: float | None = None
    cold_tail_last_probability: float | None = None
    cold_tail_last_integral: float | None = None
    cold_tail_last_log_gamma_h4: float | None = None
    cold_tail_post_peak_streak: int = 0


def _finalize_percolation_state(
    state: PercolationState,
    outdict: dict,
    action_temperatures_before: np.ndarray,
) -> None:
    """Freeze the support-bank bookkeeping after a percolation solve.

    The adaptive solver stores both temperatures already sampled by the
    percolation ODE and additional temperatures at which the tunneling action
    had to be reevaluated. This helper normalises both sets onto sorted,
    duplicate-free temperature grids so the diagnostics layer can report them
    consistently.
    """
    state.support_bank = _temperature_grid(state.support_bank if state.support_bank is not None else state.TSYM)
    if state.free_support_bank is not None:
        state.action_temperatures = _temperature_grid(
            np.concatenate((state.support_bank, np.asarray(state.free_support_bank, dtype=float)))
        )
        return

    previous = _temperature_grid(action_temperatures_before)
    current = _temperature_grid(_action_outdict_temperatures(outdict))
    if previous.size == 0:
        state.action_temperatures = current
        return
    state.action_temperatures = np.asarray(
        [
            float(value)
            for value in current
            if not np.any(
                np.isclose(
                    previous,
                    float(value),
                    atol=1e-10 * max(np.max(np.abs(previous)), abs(float(value)), 1.0),
                    rtol=0.0,
                )
            )
        ],
        dtype=float,
    )


def calcPercAndEvolve(
    outdict: dict,
    Tnuc: float | None,
    phase_symmetric,
    phase_broken,
    pot,
    vw=1.0,
    rtol=1e-4,
    nAction=50,
    verbose=False,
    return_metadata: bool = False,
):
    """Compute the percolation profile with the configured percolation algorithm.

    The public return shape is kept for compatibility:
    ``Tperc, TSYM, H, P, Tb, S`` plus optional diagnostics metadata.
    """

    CF = pot.conversionFactor
    action_temperatures_before = _action_outdict_temperatures(outdict)
    settings = _build_percolation_settings(pot, int(nAction))
    state: PercolationState | None = None
    try:
        Tstart = None if Tnuc is None else float(Tnuc)
        if Tstart is not None and not np.isfinite(Tstart):
            Tstart = None
        overlap_tmin, overlap_tmax = _phase_overlap_interval(
            pot,
            phase_symmetric,
            phase_broken,
        )
        from transitionlistener.percolation_adaptivestepsize import _initial_temperature_grid

        if settings.algorithm_mode == "adaptive_step_size":
            # The adaptive step size solver keeps the same physical pass structure:
            # coarse approximate Tperc, then a step-2 solve with P = 0 in the
            # Hubble rate, then a step-3 solve with the P-dependent Hubble
            # rate.  Only the support-point management changes, which now lives
            # in the dedicated adaptive step size module.
            from transitionlistener.percolation_adaptivestepsize import (
                _initial_percolation_scan_dynamiczoomwindow,
                _refine_percolation_temperature_dynamiczoomwindow,
            )

            grid = _initial_temperature_grid(
                outdict,
                pot,
                Tstart,
                phase_symmetric,
                phase_broken,
                vw,
                int(nAction),
                settings,
                overlap_tmin,
                overlap_tmax,
                verbose,
            )

            state = _initial_percolation_scan_dynamiczoomwindow(
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
            _maybe_dump_percolation_spline(state, "step2", pot)
            Tperc_prev = _solve_for_initial_Tperc(state, settings, pot, verbose)
            Tperc, state = _refine_percolation_temperature_dynamiczoomwindow(
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
            _maybe_dump_percolation_spline(state, "step3", pot)
        else:
            from transitionlistener.percolation_fixedstepsize import (
                _initial_percolation_scan,
                _refine_percolation_temperature,
            )

            grid = _initial_temperature_grid(
                outdict,
                pot,
                Tstart,
                phase_symmetric,
                phase_broken,
                vw,
                int(nAction),
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
            _maybe_dump_percolation_spline(state, "step2", pot)
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
            _maybe_dump_percolation_spline(state, "step3", pot)
        _finalize_percolation_state(state, outdict, action_temperatures_before)
        jitter_diagnostic = _raise_if_action_rate_jitter_unresolved(
            state,
            settings,
            CF,
            pot=pot,
            phase_symmetric=phase_symmetric,
            phase_broken=phase_broken,
            outdict=outdict,
        )
        tnuc_estimate = estimate_spline_tnuc_from_rate_history(
            state.TSYM,
            state.Hr,
            state.Sr,
            state.Pr,
            vw=vw,
        )
        metadata = PercolationDiagnostics.from_state(state, grid, jitter_diagnostic, tnuc_estimate)
        if return_metadata:
            return Tperc, state.TSYM, state.Hr, state.Pr, state.Tb, state.Sr, metadata
        return Tperc, state.TSYM, state.Hr, state.Pr, state.Tb, state.Sr
    except Exception as err:
        if state is not None:
            _finalize_percolation_state(state, outdict, action_temperatures_before)
            try:
                setattr(err, "percolation_state", state)
                setattr(err, "percolation_rebuild_count", int(state.rebuild_count))
            except Exception:
                pass
        raise err


def _build_percolation_settings(pot, nAction: int):
    """Validate ``pot.config.percolationConf`` and attach runtime-only values."""

    conf = pot.config.percolationConf
    static_n_action = max(int(nAction), 1)
    algorithm_mode = str(getattr(conf, "algorithm_mode", "adaptive_step_size"))
    if algorithm_mode not in {"adaptive_step_size", "fixed_step_size"}:
        raise errors.PercolationError(
            "percolation_algorithm_mode must be 'adaptive_step_size' or "
            f"'fixed_step_size', got {algorithm_mode!r}."
        )
    integral_method = str(getattr(conf, "integral_method", "ode"))
    if integral_method not in {"ode", "double_integral"}:
        raise errors.PercolationError(
            "percolation_integral_method must be 'ode' or "
            f"'double_integral', got {integral_method!r}."
        )
    time_temperature_mode = str(getattr(conf, "time_temperature_mode", "sound_speed"))
    if time_temperature_mode not in {"sound_speed", "bag"}:
        raise errors.PercolationError(
            "percolation_time_temperature_mode must be 'sound_speed' or "
            f"'bag', got {time_temperature_mode!r}."
        )

    n_action_min = max(int(getattr(conf, "n_action_min", 20)), 2)
    n_action_increment = max(int(getattr(conf, "n_action_increment", 5)), 1)
    n_action_max = max(int(getattr(conf, "n_action_max", 50)), n_action_min)
    boundary_basis = n_action_min if algorithm_mode == "adaptive_step_size" else static_n_action

    conf.algorithm_mode = algorithm_mode
    conf.integral_method = integral_method
    conf.time_temperature_mode = time_temperature_mode
    conf.n_action_min = n_action_min
    conf.n_action_increment = n_action_increment
    conf.n_action_max = n_action_max
    conf.max_action_temperatures = max(int(getattr(conf, "max_action_temperatures", 100)), 1)
    conf.max_boundary_n = int(conf.max_boundary_ratio * boundary_basis)
    conf.large_delta_p_refine_threshold = float(getattr(conf, "large_delta_p_refine_threshold", 0.1))
    conf.large_delta_p_success_threshold = float(getattr(conf, "large_delta_p_success_threshold", 0.2))
    conf.jitter_GH4_threshold = max(float(getattr(conf, "jitter_GH4_threshold", 1.0)), 0.0)
    conf.jitter_rescue = bool(getattr(conf, "jitter_rescue", False))
    conf.n_jitter_save = max(int(getattr(conf, "n_jitter_save", 20)), 0)
    conf.acc_tperc = float(getattr(conf, "acc_tperc", 1e-2))
    conf.acc_tfinal = float(getattr(conf, "acc_tfinal", 1e-2))
    conf.acc_rh = float(getattr(conf, "acc_rh", 1e-2))
    return conf


def _phase_overlap_interval(
    pot,
    phase_symmetric,
    phase_broken,
    Tnuc: float | None = None,
) -> tuple[float, float]:
    """Return the common traced temperature interval of both phases."""

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


def _temperature_grid(
    values: np.ndarray | list[float] | tuple[float, ...] | None,
) -> np.ndarray:
    """Return finite temperatures sorted descending with near-duplicates removed."""

    if values is None:
        return np.asarray([], dtype=float)
    raw = np.asarray(values, dtype=float).reshape(-1)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        return np.asarray([], dtype=float)

    ordered = np.sort(raw)[::-1]
    unique = [float(ordered[0])]
    for candidate in ordered[1:]:
        candidate = float(candidate)
        scale = max(abs(unique[-1]), abs(candidate), 1.0)
        if not math.isclose(unique[-1], candidate, abs_tol=1e-10 * scale, rel_tol=0.0):
            unique.append(candidate)
    return np.asarray(unique, dtype=float)


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


def calcSoundSpeedSq(pot, X, T) -> float:
    """Compute the symmetric-phase sound speed squared at ``(X, T)``.

    Decoupled radiation is excluded because it does not participate in the
    local time-temperature relation of the transitioning plasma.
    """
    T_abs = abs(float(T))
    dT = max(float(getattr(pot, "T_eps", 1.0e-3)), T_abs * 1.0e-4)
    if T_abs > 0.0:
        dT = min(dT, 0.25 * T_abs)
    dVdT = pot.dVdT(X, T, dT=dT, include_decoupled=False)
    d2VdT2 = pot.d2VdT2(X, T, dT=dT, include_decoupled=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs_sq = np.asarray(dVdT, dtype=float) / (float(T) * np.asarray(d2VdT2, dtype=float))
    return float(np.squeeze(cs_sq))


def _time_temperature_factors(
    pot,
    phase,
    T: np.ndarray,
    mode: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(c_s^2, a/a_hot)`` arrays for the generalized integral.

    ``None, None`` is returned for the bag-limit mode so the historical ODE is
    used without any extra thermodynamic finite-difference cost.
    """
    mode = "sound_speed" if mode is None else str(mode)
    if mode not in {"sound_speed", "bag"}:
        raise errors.PercolationError(
            f"Unknown percolation time-temperature mode {mode!r}. "
            "Supported modes are 'sound_speed' and 'bag'."
        )
    if mode == "bag" or pot is None or phase is None:
        return None, None

    temperatures = np.asarray(T, dtype=float)
    sound_speed_sq = np.full_like(temperatures, 1.0 / 3.0, dtype=float)
    for i, temp in enumerate(temperatures):
        try:
            cs_sq = calcSoundSpeedSq(pot, phase.valAt(float(temp)), float(temp))
        except Exception:
            cs_sq = np.nan
        if np.isfinite(cs_sq) and cs_sq > 0.0:
            sound_speed_sq[i] = float(cs_sq)

    # d ln a / dT = -1 / (3 c_s^2 T), normalized to a(T_hot)=1.
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = -1.0 / (3.0 * sound_speed_sq * temperatures)
    integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)
    ln_a = integrate.cumulative_trapezoid(integrand, x=temperatures, initial=0.0)
    ln_a = np.clip(ln_a, -700.0, 700.0)
    scale_factor = np.exp(ln_a)
    return sound_speed_sq, scale_factor


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

    if T in outdict:
        return outdict[T]["action"]

    from transitionlistener.phases import findLocalMinimum

    x0 = findLocalMinimum(start_phase.valAt(T), T, pot.gradV, pot.d2V)
    x1 = findLocalMinimum(end_phase.valAt(T), T, pot.gradV, pot.d2V)
    tdict = dict(low_vev=x1, high_vev=x0)

    outdict = bounceAction(
        T, pot.Vtot, pot.gradV, outdict, tdict, verbose, pot.conversionFactor,
        **pot.config.tracingConf.tunneling_params)
    return outdict[T]["action"]


def percIntegral(
    T: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    vw=1.0,
    *,
    entropy_density: np.ndarray | None = None,
    cooling_factor: np.ndarray | None = None,
) -> float:
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

    T = np.asarray(T, dtype=float)
    H = np.asarray(H, dtype=float)
    S = np.asarray(S, dtype=float)
    vol_int = np.array([integrate.trapezoid(1 / H[i:], x=T[i:]) for i in range(len(T))])
    with np.errstate(invalid="ignore"):
        integrant = Gamma(T, S) / T**4 / H * vol_int**3
    y = integrate.trapezoid(np.nan_to_num(integrant), x=T)
    return 4 * np.pi / 3 * vw**3 * y


# ---------------------------------------------------------------------------
# ODE-based percolation integral
# ---------------------------------------------------------------------------
#
# The standard ``percIntegral`` computes I(T) via a nested double integral
# at O(N^2) per evaluation (O(N^3) for a full sweep over the grid).  The
# functions below reformulate I(T) as a 4-component ODE chain that can be
# integrated in a single O(N) sweep from T_hot to T_cold:
#
#   A(T)  = Gamma(T) / (T^4 * H(T))          (nucleation-rate density)
#   J_0'  = -A              J_0(T_hot) = 0
#   J_1'  = -J_0 / H        J_1(T_hot) = 0
#   J_2'  = -2 J_1 / H      J_2(T_hot) = 0
#   J_3'  = -3 J_2 / H      J_3(T_hot) = 0
#
#   I(T)  = (4 pi / 3) * v_w^3 * J_3(T)
#   P(T)  = 1 - exp(-I(T))
#
# Here ' = d/dT (with T *decreasing* toward the cold boundary).
# See the derivation in the module-level docstring of
# ``percolation_adaptivestepsize.py``.
# ---------------------------------------------------------------------------

_LOG_GAMMA_SOURCE_FLOOR = -700.0  # exp(-700) ~ 1e-304, safely above denormal threshold


def _log_bag_gamma_source_array(
    T: np.ndarray,
    S: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    r"""Compute the bag-limit ``ln gamma(T)`` source term element-wise.

    Works entirely in log-space so the exponentially suppressed hot tail
    never underflows.

    Parameters
    ----------
    T, S, H : np.ndarray
        Temperature, bounce action, and Hubble rate arrays (internal units,
        same length, T decreasing).

    Returns
    -------
    np.ndarray
        ``ln gamma`` at each grid point.  Entries where the action is zero or
        infinite are set to ``-inf``.
    """
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    H = np.asarray(H, dtype=float)
    result = np.full_like(T, -np.inf)
    valid = np.isfinite(S) & (S > 0.0) & np.isfinite(H) & (H > 0.0) & (T > 0.0)
    if not np.any(valid):
        return result
    # ln(Gamma) = 4 ln T + 3/2 ln(S/(2 pi T)) - S/T
    # Bag-limit gamma = Gamma / (H T^4), so the explicit 4 ln(T)
    # prefactor in Gamma cancels.
    Tv = T[valid]
    Sv = S[valid]
    Hv = H[valid]
    result[valid] = (
        1.5 * np.log(Sv / (2.0 * np.pi * Tv))
        - Sv / Tv
        - np.log(Hv)
    )
    return result


def percIntegralODE(
    T: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    vw: float = 1.0,
    *,
    sound_speed_sq: np.ndarray | None = None,
    scale_factor: np.ndarray | None = None,
) -> np.ndarray:
    r"""Compute I(T_i) at every grid point via the J_n ODE chain.

    With ``sound_speed_sq`` and ``scale_factor`` this solves the generalized
    EOS form using
    ``gamma = Gamma a^3 / (H T 3 c_s^2)`` and
    ``nu = 1 / (H T 3 c_s^2 a)``.  Without those arrays it falls back to the
    historical bag-limit relation, where ``c_s^2 = 1/3`` and ``a(T)`` cancels
    to the simpler transport factor ``1/H``.

    Parameters
    ----------
    T : np.ndarray
        Temperatures in descending order (T[0] = hottest).
    H : np.ndarray
        Hubble rate at each temperature.
    S : np.ndarray
        Bounce action S_3 at each temperature.
    vw : float
        Bubble-wall velocity.

    Returns
    -------
    np.ndarray
        ``I(T_i)`` at each grid point (same length as *T*).
        ``P(T_i) = 1 - exp(-I(T_i))``.
    """
    T = np.asarray(T, dtype=float)
    H = np.asarray(H, dtype=float)
    S = np.asarray(S, dtype=float)
    N = T.size
    if N == 0:
        return np.zeros(0)
    if N == 1:
        return np.zeros(1)
    if T[0] < T[-1]:
        raise errors.PercolationError(
            "T is not decreasing in percIntegralODE."
        )

    use_general_eos = sound_speed_sq is not None and scale_factor is not None
    if use_general_eos:
        sound_speed_sq = np.asarray(sound_speed_sq, dtype=float)
        scale_factor = np.asarray(scale_factor, dtype=float)
        if sound_speed_sq.shape != T.shape or scale_factor.shape != T.shape:
            raise errors.PercolationError(
                "sound_speed_sq and scale_factor must have the same shape as T."
            )
        valid = (
            np.isfinite(S)
            & (S > 0.0)
            & np.isfinite(H)
            & (H > 0.0)
            & np.isfinite(sound_speed_sq)
            & (sound_speed_sq > 0.0)
            & np.isfinite(scale_factor)
            & (scale_factor > 0.0)
            & (T > 0.0)
        )
        log_gamma_source = np.full_like(T, -np.inf)
        if np.any(valid):
            log_gamma_source[valid] = (
                np.asarray(logGamma(T[valid], S[valid]), dtype=float)
                + 3.0 * np.log(scale_factor[valid])
                - np.log(H[valid] * T[valid] * 3.0 * sound_speed_sq[valid])
            )
        log_transport = np.full_like(T, -np.inf)
        log_transport[valid] = -np.log(H[valid] * T[valid] * 3.0 * sound_speed_sq[valid] * scale_factor[valid])
    else:
        # Bag-limit source and transport factors.
        log_gamma_source = _log_bag_gamma_source_array(T, S, H)
        inv_H = np.where(np.isfinite(H) & (H > 0.0), 1.0 / H, 0.0)
        log_transport = np.full_like(T, -np.inf)
        valid_h = inv_H > 0.0
        log_transport[valid_h] = np.log(inv_H[valid_h])

    # PCHIP requires finite values: clamp -inf to a floor so the source term
    # evaluates to zero in the RHS without upsetting the interpolant.
    log_gamma_source = np.maximum(log_gamma_source, _LOG_GAMMA_SOURCE_FLOOR)
    log_transport = np.maximum(log_transport, _LOG_GAMMA_SOURCE_FLOOR)

    # --- Trim clamped hot-end outliers ----------------------------------------
    # The support bank occasionally places one or a few temperatures far above
    # the nucleation peak (e.g. T[0] = 70 GeV while T[1] = 0.8 GeV).  Those
    # hot-end points have a floored source term (no contribution to the
    # integral), but their presence creates a huge interval in u = ln(T) space
    # (Δu ~ 4.4 vs. Δu ~ 0.01 everywhere else).  The PCHIP interpolant across
    # such a stretched interval produces a large slope that causes DOP853's
    # initial step-size estimate to collapse to sub-machine-epsilon values.
    #
    # Because gamma ≈ 0 in the clamped region, J_n(T_first_non_clamped) = 0 is a
    # valid initial condition — trimming these points changes I(T_cold) by at
    # most exp(_LOG_GAMMA_SOURCE_FLOOR) * ΔT ≈ 0.
    #
    # The caller expects len(I_values) == len(T_input), so we restore the
    # trimmed hot-end positions with I = 0 before returning.
    _FLOOR_MARGIN = 1.0
    n_trim = int(np.argmax(log_gamma_source > _LOG_GAMMA_SOURCE_FLOOR + _FLOOR_MARGIN))
    if n_trim > 0 and n_trim < N - 1:
        T = T[n_trim:]
        log_gamma_source = log_gamma_source[n_trim:]
        log_transport = log_transport[n_trim:]
        N = T.size
    else:
        n_trim = 0
    # --------------------------------------------------------------------------

    # Build PCHIP interpolants in log(T) space.
    #
    # The temperatures are log-spaced, so u = ln(T) is *linearly* spaced —
    # the PCHIP nodes are equidistant in u, which gives much better
    # conditioning than T-space where the hot tail is enormously stretched.
    # More importantly, solving the ODE in u-space avoids the "Required step
    # size is less than spacing between numbers" failure that DOP853 can hit
    # when T is very small (~1e-5 GeV): in T-space the minimum allowed step
    # size is |h| > eps * |T| ~ 2e-21, whereas in u-space the span is
    # O(10) and machine precision is never a bottleneck.
    T_asc = T[::-1].copy()
    u_asc = np.log(T_asc)           # ascending (T_asc is ascending → u_asc ascending)
    log_gamma_source_asc = log_gamma_source[::-1].copy()
    log_transport_asc = log_transport[::-1].copy()

    log_gamma_source_interp = interpolate.PchipInterpolator(u_asc, log_gamma_source_asc, extrapolate=True)
    log_transport_interp = interpolate.PchipInterpolator(u_asc, log_transport_asc, extrapolate=True)

    # The ODE chain in u = ln(T) coordinates.  Chain rule: d/du = T · d/dT,
    # so each RHS term acquires a factor of T = exp(u):
    #
    #   dJ0/du = -gamma(T) · T
    #   dJ1/du = -J0 · nu(T) · T
    #   dJ2/du = -2 J1 · nu(T) · T
    #   dJ3/du = -3 J2 · nu(T) · T
    #
    # We integrate from u_hot = ln(T[0]) downward to u_cold = ln(T[-1]).
    # solve_ivp supports a decreasing t_span, so we pass (u_hot, u_cold)
    # and it steps in the negative-u direction.
    #
    # Absolute tolerance: J0 starts at values as small as exp(-S_3/T_hot)
    # which can be ~exp(-800) ~ 1e-348.  atol=1e-300 keeps the solver from
    # treating J0 as zero in the early (very suppressed) hot-tail phase.
    u_eval = np.log(T)   # decreasing (mirrors T which is decreasing)

    def rhs(u, y):
        t = np.exp(u)
        log_gamma_source_value = float(log_gamma_source_interp(u))
        gamma_source = (
            np.exp(log_gamma_source_value)
            if log_gamma_source_value > _LOG_GAMMA_SOURCE_FLOOR
            else 0.0
        )
        lt = float(log_transport_interp(u))
        transport = np.exp(lt) if lt > _LOG_GAMMA_SOURCE_FLOOR else 0.0
        J0, J1, J2, J3 = y
        return [
            -gamma_source * t,          # dJ0/du
            -J0 * transport * t,        # dJ1/du
            -2.0 * J1 * transport * t,  # dJ2/du
            -3.0 * J2 * transport * t,  # dJ3/du
        ]

    sol = integrate.solve_ivp(
        rhs,
        t_span=(float(u_eval[0]), float(u_eval[-1])),
        y0=[0.0, 0.0, 0.0, 0.0],
        method="DOP853",
        t_eval=u_eval,
        rtol=1e-10,
        atol=1e-300,
        dense_output=False,
    )
    if sol.status != 0:
        raise errors.PercolationError(
            "percIntegralODE: the percolation integral solver could not reach "
            "the cold end of the temperature grid. The action S_3(T) is most "
            "likely non-smooth on the scale of the support points (numerical "
            "jitter from path deformation), so the adaptive ODE step shrinks "
            "below floating-point spacing and the solver gives up. "
            "Suggested fixes: (a) rerun with precision_mode: tunneltight or "
            "benchmark to tighten the bounce-action path deformation; "
            "(b) probe the action with tests/probe_bounce_stability.py to "
            "localise the jittery T-range; "
            "(c) if the model is multi-field, lower precision_deform_fRatioConv "
            "(default 1e-2 for Ndim>=2) further. "
            f"Underlying solver message: {sol.message}"
        )

    J3 = sol.y[3]  # J3 at each T_i
    I_trimmed = (4.0 * np.pi / 3.0) * vw**3 * J3
    # Restore trimmed hot-end points with I = 0 (A ≈ 0 there).
    if n_trim > 0:
        I_values = np.concatenate([np.zeros(n_trim), I_trimmed])
    else:
        I_values = I_trimmed
    return np.asarray(I_values, dtype=float)


def percIntegralODE_full_sweep(
    T: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    vw: float = 1.0,
    *,
    pot=None,
    phase_symmetric=None,
    time_temperature_mode: str | None = None,
    integral_method: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""One-shot ODE sweep: compute both I(T_i) and P(T_i) at every grid point.

    This replaces the per-point ``percIntegral`` calls in the step-2 / step-3
    inner loops.

    Parameters
    ----------
    T, H, S : np.ndarray
        Temperatures (descending), Hubble rates, bounce actions.
    vw : float
        Bubble-wall velocity.

    Returns
    -------
    I_values : np.ndarray
        Percolation integral ``I(T_i)`` at each grid point.
    P_values : np.ndarray
        True-vacuum fraction ``P(T_i) = 1 - exp(-I(T_i))``.
    """
    method = "ode" if integral_method is None else str(integral_method)
    if method == "double_integral":
        I_values = np.asarray(
            [percIntegral(T[: i + 1], H[: i + 1], S[: i + 1], vw=vw) for i in range(len(T))],
            dtype=float,
        )
    elif method == "ode":
        sound_speed_sq, scale_factor = _time_temperature_factors(
            pot,
            phase_symmetric,
            np.asarray(T, dtype=float),
            time_temperature_mode,
        )
        I_values = percIntegralODE(
            T,
            H,
            S,
            vw=vw,
            sound_speed_sq=sound_speed_sq,
            scale_factor=scale_factor,
        )
    else:
        raise errors.PercolationError(
            f"Unknown percolation integral method {integral_method!r}. "
            "Supported methods are 'ode' and 'double_integral'."
        )
    P_values = 1.0 - np.exp(-I_values)
    return I_values, P_values


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


def _approx_percolation_criterion(
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

    f_perc = float(pot.config.percolationConf.f_perc)
    Iperc = -np.log1p(-f_perc)
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
            print("Warning: betaH is not finite or zero in percolation approximation. betaH =", betaH)
        return np.inf
    if not np.isfinite(H) or H <= 0:
        if verbose:
            print("Warning: H is not finite or zero in percolation approximation. H =", H)
        return np.inf
    if not np.isfinite(vw) or vw <= 0:
        if verbose:
            print("Warning: vw is not finite or zero in percolation approximation. vw =", vw)
        return np.inf
    crit = np.power(G, 1 / 4.0) / H / betaH / np.power(Iperc / (8 * np.pi * vw**3), 1 / 4.0)
    return float(np.squeeze(crit)) - 1


def calcApproxPercolation(
    outdict: dict,
    pot,
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
        tmin, tmax = _phase_overlap_interval(pot, phase_sym, phase_bro)

    Tmin = float(tmin)
    Tmax = float(tmax)
    if Tmax <= Tmin:
        raise errors.PercolationApproximation1Error(
            "No valid temperature bracket for percolation approximation inside "
            f"the phase-overlap interval [{tmin:.8g}, {tmax:.8g}]."
        )

    try:
        # The approximation is only used to place the first adaptive step size grid.
        # If it fails to bracket a root, fail fast and let the caller choose a
        # controlled fallback instead of spending many hidden action evaluations.
        Tperc = optimize.brentq(
            _approx_percolation_criterion,
            Tmax,
            Tmin,
            rtol=1e-3,
            args=(outdict, pot, phase_sym, phase_bro, Tmax, Tmin, vw, verbose),
        )
    except ValueError as err:
        if verbose:
            print("Brentq failed in percolation approximation 1: ", err)
        raise errors.PercolationApproximation1Error(
            "Standard percolation approximation failed to bracket a root in "
            f"[{Tmin:.8g}, {Tmax:.8g}]: {err}"
        ) from err
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
    # Working assumption: the broken bubble reheats the DS and the coupled SM
    # plasma to one local common temperature Tb, while the cosmological scale
    # factor between successive steps is still inferred from the background
    # symmetric-phase temperatures TSYM / TSYM_ref. Revisit this hybrid
    # assumption if the TBRO evolution remains suspect.
    heff_ds_pt = h_eff_DS(TBRO_ref, pot, phase_broken)
    heff_SM_pt = td.s_geffSM(TBRO_ref, pot.conversionFactor)
    heff_ds = h_eff_DS(Tb, pot, phase_broken)
    heff_SM = td.s_geffSM(Tb, pot.conversionFactor)
    crit = (heff_ds + heff_SM) * Tb**3
    crit -= (heff_ds_pt + heff_SM_pt) * TBRO_ref**3 * TSYM**3 / TSYM_ref**3
    return crit




def _solve_for_initial_Tperc(
    state: PercolationState,
    settings,
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
                explored_tmin = float(state.TSYM[-1]) if state.explored_tmin is None else float(state.explored_tmin)
                current_tmin = float(state.TSYM[-1])
                collapse_tol = 1e-12 * max(abs(explored_tmin), abs(current_tmin), 1.0)
                if explored_tmin + collapse_tol < current_tmin:
                    raise errors.PercolationError(
                        "The percolation-temperature search became numerically unstable: "
                        "the support grid previously explored lower temperatures down to "
                        f"{explored_tmin * pot.conversionFactor:2.8g} GeV, but the final "
                        "step-2 support grid shrank back to Tmin = "
                        f"{current_tmin * pot.conversionFactor:2.8g} GeV before failing "
                        f"with P(Tmin) = {Pint(state.TSYM[-1]):2.5g} < {settings.f_perc:2.5g}. "
                        "This usually indicates a spurious 0->1 jump in P(T); rerun with "
                        "higher percolation resolution."
                    )
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


def _action_outdict_temperatures(outdict: dict | None) -> np.ndarray:
    """Return the finite action temperatures currently cached in ``outdict``."""
    if not isinstance(outdict, dict):
        return np.asarray([], dtype=float)
    temperatures: list[float] = []
    for key, payload in outdict.items():
        if not isinstance(payload, dict) or "action" not in payload:
            continue
        try:
            temperature = float(key)
        except Exception:
            continue
        if np.isfinite(temperature):
            temperatures.append(float(temperature))
    return _temperature_grid(temperatures)


def _transition_action_outdict_temperatures(
    outdict: dict | None,
    phase_symmetric,
    phase_broken,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> np.ndarray:
    """Return cached action temperatures that clearly belong to one transition."""
    if not isinstance(outdict, dict):
        return np.asarray([], dtype=float)

    def phase_point_matches(reference_point, candidate_point) -> bool:
        try:
            reference = np.asarray(reference_point, dtype=float)
            candidate = np.asarray(candidate_point, dtype=float)
        except Exception:
            return False
        if reference.shape != candidate.shape or reference.size == 0:
            return False
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
            return False
        difference = float(np.linalg.norm(reference - candidate))
        scale = max(float(np.linalg.norm(reference)), float(np.linalg.norm(candidate)), 1.0)
        return difference <= max(1e-6, 2e-2 * scale)

    lower = -math.inf if tmin is None else float(min(tmin, tmax))
    upper = math.inf if tmax is None else float(max(tmin, tmax))
    scale = max(
        abs(lower) if np.isfinite(lower) else 0.0,
        abs(upper) if np.isfinite(upper) else 0.0,
        1.0,
    )
    interval_tol = 1e-10 * scale

    temperatures: list[float] = []
    for key, payload in outdict.items():
        if not isinstance(payload, dict):
            continue
        try:
            temperature = float(key)
        except Exception:
            continue
        if not np.isfinite(temperature):
            continue
        if temperature < lower - interval_tol or temperature > upper + interval_tol:
            continue

        try:
            action = float(payload.get("action", np.nan))
        except Exception:
            continue
        if not np.isfinite(action):
            continue

        high_vev = payload.get("high_vev")
        low_vev = payload.get("low_vev")
        if high_vev is not None and low_vev is not None:
            try:
                symmetric_point = phase_symmetric.valAt(temperature)
                broken_point = phase_broken.valAt(temperature)
            except Exception:
                pass
            else:
                if not (
                    phase_point_matches(high_vev, symmetric_point)
                    and phase_point_matches(low_vev, broken_point)
                ):
                    continue
        temperatures.append(float(temperature))
    return _temperature_grid(temperatures)


@dataclass
class PercolationDiagnostics:
    """Percolation diagnostics returned together with the final support bank.

    The solver keeps only diagnostics that are still used by the production
    adaptive step size percolation workflow and its current validation scripts.
    """

    start_temperature: float | None = None
    final_active_support_points: int = 0
    total_unique_percolation_support_points: int = 0
    total_unique_action_temperatures: int = 0
    rebuild_count: int = 0
    support_bank_temperatures: list[float] | None = None
    action_temperatures: list[float] | None = None
    spline_tnuc: float | None = None
    spline_tnuc_warning: str | None = None
    spline_tnuc_max_N: float = math.nan
    spline_tnuc_integral_prefactor: float = math.nan
    action_jitter_max_residual_oom: float = math.nan
    action_jitter_temperature: float | None = None
    action_jitter_probability: float | None = None
    action_jitter_log10_gamma_h4: float | None = None
    action_jitter_log10_gamma_h4_smooth: float | None = None
    action_jitter_rescue_attempts: int = 0
    action_jitter_rescue_success: bool = False
    action_jitter_rescue_temperatures: list[float] | None = None

    @classmethod
    def from_state(
        cls,
        state: PercolationState,
        grid: PercolationGrid,
        jitter_diagnostic: dict[str, float | int | bool | list[float] | None] | None,
        tnuc_estimate: dict[str, float | str | None],
    ) -> "PercolationDiagnostics":
        """Build a serialisable diagnostics snapshot from the final solver state."""
        support_bank = _temperature_grid(state.support_bank)
        action_temperatures = _temperature_grid(state.action_temperatures)
        metadata = cls(
            start_temperature=float(grid.Tstart),
            final_active_support_points=int(len(np.asarray(state.TSYM, dtype=float))),
            total_unique_percolation_support_points=int(len(support_bank)),
            total_unique_action_temperatures=int(len(action_temperatures)),
            rebuild_count=int(state.rebuild_count),
            support_bank_temperatures=[float(value) for value in support_bank.tolist()],
            action_temperatures=[float(value) for value in action_temperatures.tolist()],
        )
        if jitter_diagnostic:
            max_residual = jitter_diagnostic.get("max_residual_oom", math.nan)
            metadata.action_jitter_max_residual_oom = (
                float(max_residual) if max_residual is not None else math.nan
            )
            metadata.action_jitter_temperature = (
                float(jitter_diagnostic["temperature"])
                if jitter_diagnostic.get("temperature") is not None
                else None
            )
            metadata.action_jitter_probability = (
                float(jitter_diagnostic["probability"])
                if jitter_diagnostic.get("probability") is not None
                else None
            )
            metadata.action_jitter_log10_gamma_h4 = (
                float(jitter_diagnostic["log10_gamma_h4"])
                if jitter_diagnostic.get("log10_gamma_h4") is not None
                else None
            )
            metadata.action_jitter_log10_gamma_h4_smooth = (
                float(jitter_diagnostic["log10_gamma_h4_smooth"])
                if jitter_diagnostic.get("log10_gamma_h4_smooth") is not None
                else None
            )
            metadata.action_jitter_rescue_attempts = int(jitter_diagnostic.get("rescue_attempts", 0) or 0)
            metadata.action_jitter_rescue_success = bool(jitter_diagnostic.get("rescue_success", False))
            metadata.action_jitter_rescue_temperatures = [
                float(value) for value in (jitter_diagnostic.get("rescue_temperatures") or [])
            ]

        if tnuc_estimate.get("Tnuc") is not None:
            metadata.spline_tnuc = float(tnuc_estimate["Tnuc"])
        metadata.spline_tnuc_warning = (
            str(tnuc_estimate["warning"]) if tnuc_estimate.get("warning") is not None else None
        )
        max_N = tnuc_estimate.get("max_N", math.nan)
        prefactor = tnuc_estimate.get("integral_prefactor", math.nan)
        metadata.spline_tnuc_max_N = float(max_N) if max_N is not None else math.nan
        metadata.spline_tnuc_integral_prefactor = float(prefactor) if prefactor is not None else math.nan
        return metadata


def _action_rate_jitter_diagnostic(
    T: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    P: np.ndarray,
) -> dict[str, float | int | None]:
    """Measure non-smooth jumps in the active-band rate samples."""
    from transitionlistener.percolation_adaptive_rate import _log10_gamma_h4_array

    p_low = 0.01
    p_high = 0.9
    T = np.asarray(T, dtype=float)
    H = np.asarray(H, dtype=float)
    S = np.asarray(S, dtype=float)
    P = np.asarray(P, dtype=float)
    log10_rate = _log10_gamma_h4_array(T, S, H)
    mask = (
        np.isfinite(T)
        & np.isfinite(H)
        & np.isfinite(S)
        & np.isfinite(P)
        & np.isfinite(log10_rate)
        & (T > 0.0)
        & (H > 0.0)
        & (P >= p_low)
        & (P <= p_high)
    )
    active_indices = np.flatnonzero(mask)
    if active_indices.size < 5:
        return {"n_active": int(active_indices.size), "max_residual_oom": math.nan}

    x = np.log(T[active_indices])
    y = log10_rate[active_indices]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    indices = active_indices[order]
    if np.ptp(x) <= 1.0e-12:
        return {"n_active": int(active_indices.size), "max_residual_oom": math.nan}

    degree = 2 if x.size >= 3 else 1
    try:
        coeff = np.polyfit(x, y, degree)
        smooth = np.polyval(coeff, x)
    except Exception:
        return {"n_active": int(active_indices.size), "max_residual_oom": math.nan}
    residual = y - smooth
    if residual.size == 0 or not np.any(np.isfinite(residual)):
        return {"n_active": int(active_indices.size), "max_residual_oom": math.nan}
    local_idx = int(np.nanargmax(np.abs(residual)))
    source_idx = int(indices[local_idx])
    return {
        "n_active": int(active_indices.size),
        "max_residual_oom": float(abs(residual[local_idx])),
        "signed_residual_oom": float(residual[local_idx]),
        "temperature": float(T[source_idx]),
        "probability": float(P[source_idx]),
        "log10_gamma_h4": float(log10_rate[source_idx]),
        "log10_gamma_h4_smooth": float(smooth[local_idx]),
        "p_low": float(p_low),
        "p_high": float(p_high),
    }


def _raise_if_action_rate_jitter_unresolved(
    state: PercolationState,
    settings,
    CF: float,
    pot=None,
    phase_symmetric=None,
    phase_broken=None,
    outdict: dict | None = None,
) -> dict[str, float | int | None]:
    """Reject accepted profiles whose active-band rate data is visibly unstable."""
    diagnostic = _action_rate_jitter_diagnostic(state.TSYM, state.Hr, state.Sr, state.Pr)
    threshold = float(settings.jitter_GH4_threshold)
    residual = diagnostic.get("max_residual_oom")
    if (
        residual is not None
        and np.isfinite(float(residual))
        and float(residual) > threshold
        and bool(getattr(settings, "jitter_rescue", False))
        and int(getattr(settings, "n_jitter_save", 0)) > 0
        and pot is not None
        and phase_symmetric is not None
        and phase_broken is not None
        and isinstance(outdict, dict)
    ):
        diagnostic = _try_action_jitter_tunneltight_rescue(
            state,
            settings,
            diagnostic,
            threshold,
            pot,
            phase_symmetric,
            phase_broken,
            outdict,
        )
        residual = diagnostic.get("max_residual_oom")
    if residual is None or not np.isfinite(float(residual)) or float(residual) <= threshold:
        return diagnostic

    temperature_value = diagnostic.get("temperature")
    probability_value = diagnostic.get("probability")
    log10_rate_value = diagnostic.get("log10_gamma_h4")
    smooth_value = diagnostic.get("log10_gamma_h4_smooth")
    temperature = float(temperature_value) if temperature_value is not None else math.nan
    probability = float(probability_value) if probability_value is not None else math.nan
    log10_rate = float(log10_rate_value) if log10_rate_value is not None else math.nan
    smooth = float(smooth_value) if smooth_value is not None else math.nan
    message = (
        "Detected non-smooth bounce-rate data in the active percolation band "
        f"{diagnostic.get('p_low', 0.01):.3g} <= P(T) <= "
        f"{diagnostic.get('p_high', 0.9):.3g}: "
        f"the largest quadratic-fit residual in log10(Gamma/H^4) is "
        f"{float(residual):.3g} orders of magnitude, above the configured "
        f"threshold {threshold:.3g}. The offending support point is at "
        f"T = {temperature * CF:.6g} GeV (internal T = {temperature:.8g}), "
        f"P = {probability:.6g}, log10(Gamma/H^4) = {log10_rate:.6g}, while the "
        f"local smooth fit predicts {smooth:.6g}. This indicates an instability "
        "of the bounce computation or a wrong tunnelling branch, not a trustworthy "
        "percolation feature. Suggested fixes: rerun with more robust/benchmark "
        "tunnelling accuracy, improve phase tracing near this temperature, inspect "
        "the action bank, or increase the bounce solver precision."
    )
    err = errors.ActionRateJitterError(message)
    setattr(err, "action_rate_jitter_diagnostic", diagnostic)
    raise err


def _matching_outdict_key(outdict: dict, temperature: float) -> object | None:
    """Return an outdict key matching ``temperature`` up to floating roundoff."""
    if temperature in outdict:
        return temperature
    for key, payload in outdict.items():
        if not isinstance(payload, dict) or "action" not in payload:
            continue
        try:
            key_float = float(key)
        except Exception:
            continue
        if np.isclose(key_float, temperature, rtol=1.0e-11, atol=1.0e-12):
            return key
    return None


def _try_action_jitter_tunneltight_rescue(
    state: PercolationState,
    settings,
    diagnostic: dict[str, float | int | None],
    threshold: float,
    pot,
    phase_symmetric,
    phase_broken,
    outdict: dict,
) -> dict[str, float | int | None]:
    """Opt-in recomputation of jitter outlier actions with tunneltight settings."""
    max_attempts = max(int(getattr(settings, "n_jitter_save", 0)), 0)
    if max_attempts <= 0:
        return diagnostic
    attempted_temperatures: list[float] = []
    attempted_keys: set[float] = set()
    original_tunneling = copy.deepcopy(pot.config.tracingConf.tunneling_params)
    tight_tunneling = copy.deepcopy(original_tunneling)
    deform = dict(tight_tunneling.get("deformation_deform_params", {}))
    deform["converge_0"] = 1.0
    deform["fRatioConv"] = 5.0e-3
    tight_tunneling["deformation_deform_params"] = deform

    for _attempt in range(max_attempts):
        residual = diagnostic.get("max_residual_oom")
        temperature_value = diagnostic.get("temperature")
        if (
            residual is None
            or not np.isfinite(float(residual))
            or float(residual) <= threshold
            or temperature_value is None
        ):
            break
        temperature = float(temperature_value)
        rounded_key = round(temperature, 12)
        if rounded_key in attempted_keys:
            break
        attempted_keys.add(rounded_key)
        attempted_temperatures.append(temperature)
        idx = int(np.nanargmin(np.abs(np.asarray(state.TSYM, dtype=float) - temperature)))
        old_key = _matching_outdict_key(outdict, temperature)
        old_payload = outdict.pop(old_key, None) if old_key is not None else None
        if old_payload is not None:
            unstable = outdict.setdefault("_unstable_action_entries", [])
            if isinstance(unstable, list):
                unstable.append(
                    {
                        "T": float(temperature),
                        "action": old_payload.get("action") if isinstance(old_payload, dict) else None,
                        "reason": "action_rate_jitter_tunneltight_rescue",
                        "max_residual_oom": float(residual),
                    }
                )
        try:
            pot.config.tracingConf.tunneling_params = tight_tunneling
            state.Sr[idx] = calcAction(pot, temperature, phase_symmetric, phase_broken, outdict)
        except Exception:
            if old_key is not None and old_payload is not None:
                outdict[old_key] = old_payload
            break
        finally:
            pot.config.tracingConf.tunneling_params = original_tunneling
        diagnostic = _action_rate_jitter_diagnostic(state.TSYM, state.Hr, state.Sr, state.Pr)

    diagnostic = dict(diagnostic)
    diagnostic["rescue_attempts"] = int(len(attempted_temperatures))
    diagnostic["rescue_temperatures"] = [float(value) for value in attempted_temperatures]
    residual = diagnostic.get("max_residual_oom")
    diagnostic["rescue_success"] = bool(
        residual is not None and np.isfinite(float(residual)) and float(residual) <= threshold
    )
    return diagnostic


def estimate_spline_tnuc_from_rate_history(
    T: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    P: np.ndarray,
    vw: float = 1.0,
) -> dict[str, float | str | None]:
    r"""Estimate ``Tnuc`` from the saved rate/P/H splines without new actions.

    The diagnostic solves

    ``N(T) = 4*pi/3 * v_w^3 * int_T^Tinit Gamma/H^4 * (1-P) dT/T = 1``

    on a dense spline grid.  It is intentionally warning-only: if the final
    action bank does not bracket ``N=1`` the caller can still return a valid
    percolation result and expose the failure category in metadata.
    """
    try:
        T = np.asarray(T, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float).reshape(-1)
        S = np.asarray(S, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float).reshape(-1)
        log_rate = np.asarray(logGamma(T, S), dtype=float) - 4.0 * np.log(H)
        mask = (
            np.isfinite(T)
            & np.isfinite(H)
            & np.isfinite(S)
            & np.isfinite(P)
            & np.isfinite(log_rate)
            & (T > 0.0)
            & (H > 0.0)
        )
        if np.count_nonzero(mask) < 3:
            return {"Tnuc": None, "warning": "spline_tnuc_unavailable", "max_N": math.nan}

        u = np.log(T[mask])
        rate = log_rate[mask]
        prob = np.clip(P[mask], 0.0, 1.0)
        order = np.argsort(u)
        u = u[order]
        rate = rate[order]
        prob = prob[order]
        unique = np.concatenate(([True], np.diff(u) > 1.0e-12))
        u = u[unique]
        rate = rate[unique]
        prob = prob[unique]
        if u.size < 3 or np.ptp(u) <= 1.0e-12:
            return {"Tnuc": None, "warning": "spline_tnuc_unavailable", "max_N": math.nan}

        rate_spline = interpolate.PchipInterpolator(u, rate, extrapolate=False)
        prob_spline = interpolate.PchipInterpolator(u, prob, extrapolate=False)
        n_grid = max(1000, 20 * int(u.size))
        u_dense = np.linspace(float(u[0]), float(u[-1]), n_grid)
        rate_dense = np.asarray(rate_spline(u_dense), dtype=float)
        prob_dense = np.clip(np.asarray(prob_spline(u_dense), dtype=float), 0.0, 1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            integrand = np.exp(np.clip(rate_dense, -745.0, 700.0)) * (1.0 - prob_dense)
        integrand[~np.isfinite(integrand)] = 0.0
        cumulative_from_cold = integrate.cumulative_trapezoid(integrand, x=u_dense, initial=0.0)
        total = float(cumulative_from_cold[-1])
        prefactor = 4.0 * np.pi / 3.0 * float(vw) ** 3
        N_dense = prefactor * (total - cumulative_from_cold)
        max_N = float(np.nanmax(N_dense)) if N_dense.size else math.nan
        if not np.isfinite(max_N):
            return {"Tnuc": None, "warning": "spline_tnuc_failed", "max_N": math.nan}
        if max_N < 1.0:
            return {
                "Tnuc": None,
                "warning": "spline_tnuc_not_reached",
                "max_N": max_N,
                "integral_prefactor": prefactor,
            }

        u_hot_to_cold = u_dense[::-1]
        N_hot_to_cold = N_dense[::-1]
        crossing = np.flatnonzero(N_hot_to_cold >= 1.0)
        if crossing.size == 0:
            return {"Tnuc": None, "warning": "spline_tnuc_failed", "max_N": max_N}
        idx = int(crossing[0])
        if idx == 0:
            u_cross = float(u_hot_to_cold[0])
        else:
            n0 = float(N_hot_to_cold[idx - 1])
            n1 = float(N_hot_to_cold[idx])
            u0 = float(u_hot_to_cold[idx - 1])
            u1 = float(u_hot_to_cold[idx])
            if abs(n1 - n0) <= 1.0e-300:
                u_cross = u1
            else:
                frac = (1.0 - n0) / (n1 - n0)
                u_cross = u0 + frac * (u1 - u0)
        return {
            "Tnuc": float(np.exp(u_cross)),
            "warning": None,
            "max_N": max_N,
            "integral_prefactor": prefactor,
        }
    except Exception:
        return {"Tnuc": None, "warning": "spline_tnuc_failed", "max_N": math.nan}



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


def calcAlphas(T: float, pot, high_phase, low_phase, verbose=False) -> tuple[float]:
    """Calculate the total transition strenght of the PT.
    Use several definitions.

    The last 3 alphas are normalised to the radiation energy density of only
    the relevant sector for the bubble expansion. The hydro coupling is read
    from ``pot.config.gwConf.coupled_hydrodynamics``.

    Parameters
    ----------
    T : float
        Temperature at which to evaluate alpha

    Returns
    ----------
    tuple : float
        ``(alpha_p, alpha_theta, alpha_thetabar, alpha_e, alpha_hyd, alpha_inf,
        alpha_eq)``. ``alpha_theta`` is the bag-model strength fed to the
        GW-signal / kappa() pipeline; ``alpha_thetabar`` is the beyond-bag
        pseudo-trace-anomaly definition from arXiv:2004.06995."""

    high_phi = high_phase.valAt(T)  # Start phase phi values
    low_phi = low_phase.valAt(T)  # End phase phi values
    DeltaV = np.abs(pot.Vtot(high_phi, T) - pot.Vtot(low_phi, T))

    # Derivative of the potential with respect to T
    dT = T * 1e-5
    dDeltaV_p = np.abs(pot.Vtot(high_phi, T + dT / 2) - pot.Vtot(low_phi, T + dT / 2))
    dDeltaV_m = np.abs(pot.Vtot(high_phi, T - dT / 2) - pot.Vtot(low_phi, T - dT / 2))
    dDeltaVdT = (dDeltaV_p - dDeltaV_m) / dT

    # Use the symmetric-phase radiation bath when normalizing the release.
    # This differs slightly from the older broken-phase normalization below
    # roughly 100 MeV.
    rho_rad_PTsector = pot.radiationEnergyDensity(high_phi, T, include_decoupled=False)
    # With include_decoupled=False, sectors thermally decoupled from the PT
    # sector are excluded from this normalization.

    # Here we assume that the decoupled sector has the same temperature as the PT sector
    rho_rad_tot = pot.radiationEnergyDensity(high_phi, T, include_decoupled=True)

    # Energy density:
    DeltaE = DeltaV - T * dDeltaVdT

    # Alpha definitions:
    alpha_p = DeltaV / rho_rad_tot
    alpha_theta = (DeltaE + 3 * DeltaV) / (4 * rho_rad_tot)
    alpha_e = DeltaE / rho_rad_tot

    # Beyond-bag pseudo-trace-anomaly alpha from arXiv:2004.06995 (Giese-KKS).
    # Use the same effective-potential normalization as energyDensity(); otherwise
    # theta_bar and its enthalpy normalization depend on the arbitrary Vtot offset.
    # V0_ref = pot.V0(pot.X0) + pot.Vct(pot.X0) + pot.V1_from_X(pot.X0)
    # Veff_sym = pot.Vtot(high_phi, T) - V0_ref
    # Veff_bro = pot.Vtot(low_phi, T) - V0_ref
    # csSq_sym = calcSoundSpeedSq(pot, high_phi, T)
    # theta_sym = pot.energyDensity(high_phi, T) + Veff_sym / csSq_sym
    # csSq_bro = calcSoundSpeedSq(pot, low_phi, T)
    # theta_bro = pot.energyDensity(low_phi, T) + Veff_bro / csSq_bro
    # alpha_thetabar_old = (theta_sym - theta_bro) / (3 * (-Veff_sym + pot.energyDensity(high_phi, T)))

    V0_ref = pot.V0(pot.X0) + pot.Vct(pot.X0) + pot.V1_from_X(pot.X0)
    Veff_sym = pot.Vtot(high_phi, T) - V0_ref
    Veff_bro = pot.Vtot(low_phi, T) - V0_ref
    csSq_sym = calcSoundSpeedSq(pot, high_phi, T)
    theta_sym = -T*pot.dVdT(high_phi, T, dT=dT) + Veff_sym * (1 + 1/ csSq_sym)
    csSq_bro = calcSoundSpeedSq(pot, low_phi, T)
    theta_bro = -T*pot.dVdT(low_phi, T, dT=dT) + Veff_bro * (1 + 1/ csSq_bro)
    dedT = (pot.energyDensity(high_phi, T + dT) - pot.energyDensity(high_phi, T - dT))/(2*dT)
    alpha_thetabar = (theta_sym - theta_bro) / (3* csSq_sym * T * dedT)

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
	Veff_sym_w = pot.Vtot(high_phi, T, include_decoupled=False) - V0_ref
	e_sym_w = pot.energyDensity(high_phi, T, include_decoupled=False)
	alpha_hyd_coupled = (theta_sym - theta_bro) / (3 * (-Veff_sym_w + e_sym_w))

	alpha_eq = T**3 / (3/4 * (-Veff_sym_w + e_sym_w)) * np.sum(delta_m_bos * gauge_coupling**2 * dof_bos * is_physical, axis=-1)
	alpha_inf = T**2 / (18 * (-Veff_sym_w + e_sym_w)) * m2factor
	
	include_decoupled_in_enthalpy = bool(pot.config.gwConf.coupled_hydrodynamics)
	Veff_sym_w = pot.Vtot(high_phi, T, include_decoupled=include_decoupled_in_enthalpy) - V0_ref
	e_sym_w = pot.energyDensity(high_phi, T, include_decoupled=include_decoupled_in_enthalpy)
	alpha_hyd_config = (theta_sym - theta_bro) / (3 * (-Veff_sym_w + e_sym_w))

	return alpha_p, alpha_theta, alpha_thetabar, alpha_e, [alpha_hyd_coupled, alpha_hyd_config], alpha_inf, alpha_eq




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


def calcMeanBubbleSeparation(
    T,
    Tmax,
    Sint,
    Pint,
    Hint,
    entropyInt=None,
    coolingInt=None,
    verbose=False,
):
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

    # See eq. (5.42) in 2305.02357.
    if not (np.isfinite(T) and np.isfinite(Tmax)) or Tmax <= T:
        return np.inf
    if entropyInt is None and coolingInt is None:
        # Preserve the bag-mode integration path exactly so RH/betaH_from_RH
        # remain comparable to fixed-grid reference runs.
        Tr = np.linspace(T, Tmax, 10_000)
    else:
        # For entropy-aware histories, use a relative-temperature grid so the
        # numerical resolution is stable under overall temperature rescalings.
        if T > 0.0 and Tmax > 0.0 and Tmax / T > 1.0 + 1e-10:
            Tr = np.geomspace(T, Tmax, 10_000)
        else:
            Tr = np.linspace(T, Tmax, 10_000)
        entropy_T = max(float(entropyInt(T)), 1e-300)
        entropy_Tr = np.maximum(np.asarray(entropyInt(Tr), dtype=float), 1e-300)
        entropy_ratio = np.clip(entropy_T / entropy_Tr, 0.0, np.inf)
    if entropyInt is None:
        entropy_ratio = (T / Tr) ** 3
    if coolingInt is None:
        cooling_factor = 1.0
    else:
        cooling_factor = np.maximum(np.asarray(coolingInt(Tr), dtype=float), 1e-300)
    integrant = Gamma(Tr, Sint(Tr)) * (1 - Pint(Tr)) / (cooling_factor * Tr * Hint(Tr)) * entropy_ratio
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

    # Here, an eternal inflation criterion after Lewicki et al. 1809.08242 eq. (2.26)
    # could be implemented. We decided against it so far after some consideration
    # and preliminary tests: dP/dT < -3*(1 - P(T)) /T seems to be fulfilled at
    # T_final already in all relevant cases.
    return Tf
