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
    # Support points added to resolve the rise of the nucleation rate; they have
    # their own allowance and do not use up n_action_max.
    rate_refine_points: int = 0
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
    """Calculate the effective energy degrees of freedom of the fields of the potential.

    Every mode counts, the Goldstone modes included, and the ghosts of the gauge bosons are
    subtracted, as in ``radiationEnergyDensity``. Fields flagged ``is_SM`` (e.g. W, Z, t and
    h in the 2HDM) count here and are removed from the tabulated bath in turn
    (``thermodynamics.sm_fields_in_potential_geff``), so that each of them is counted once.

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
    ghosts = td.e_geff(0.0, T_DS, pot.mass_spectrum.number_gauge_bosons, "b")
    geff = td.potential_fields_geff(pot.boson_massSq(vevT, 0), pot.fermion_massSq(vevT), T_DS, "e")
    # The ghosts are massless in the Landau gauge while the Goldstone modes they cancel are
    # not, so once every mode of the potential is frozen out the difference would turn
    # negative. A sector that has left the plasma contributes nothing, not less than nothing.
    return max(float(geff - ghosts), 0.0)


def h_eff_DS(T_DS: float, pot, phase) -> float:
    """Calculate the effective entropy degrees of freedom of the fields of the potential.

    Every mode counts, the Goldstone modes included, and the ghosts of the gauge bosons are
    subtracted, as in ``radiationEnergyDensity``. Fields flagged ``is_SM`` (e.g. W, Z, t and
    h in the 2HDM) count here and are removed from the tabulated bath in turn
    (``thermodynamics.sm_fields_in_potential_geff``), so that each of them is counted once.

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

    ghosts = td.s_geff(0.0, T_DS, pot.mass_spectrum.number_gauge_bosons, "b")
    geff = td.potential_fields_geff(pot.boson_massSq(vevT, 0), pot.fermion_massSq(vevT), T_DS, "s")
    # The ghosts are massless in the Landau gauge while the Goldstone modes they cancel are
    # not, so once every mode of the potential is frozen out the difference would turn
    # negative. A sector that has left the plasma contributes nothing, not less than nothing.
    return max(float(geff - ghosts), 0.0)


def h_eff_coupled_radiation(T: float, pot) -> float:
    """Entropy degrees of freedom of the radiation coupled to the transitioning sector.

    That radiation (``pot.kin_coupled_*``, by default the Standard Model) is reheated
    inside the bubbles together with the transitioning sector, a decoupled bath is not.
    For the default Standard Model bath the tabulated entropy degrees of freedom are
    used; otherwise ``s = (e + p)/T`` gives ``h = (3 g_e + g_p)/4``.
    """
    table = td.h_eff_radiation(pot.kin_coupled_e_geff, pot.kin_coupled_p_geff, T, pot.conversionFactor)
    # Standard Model fields of the potential are counted there, not in the tabulated bath.
    return table - td.sm_fields_in_potential_geff(pot, T, "s")


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
    entropy = np.full_like(temperatures, np.nan, dtype=float)
    for i, temp in enumerate(temperatures):
        t = float(temp)
        try:
            X = phase.valAt(t)
        except Exception:
            continue
        # The two quantities are taken separately: a sound speed that could be computed stays
        # in use even where the entropy cannot be, and the other way round.
        try:
            cs_sq = calcSoundSpeedSq(pot, X, t)
        except Exception:
            cs_sq = np.nan
        try:
            # The entropy of the transitioning sector from its degrees of freedom, not from
            # -dV/dT: the Arnold-Espinosa daisy term tends to -T/(12 pi) sum n m^3 once the
            # modes are heavy, so its contribution to -dV/dT tends to a constant and the
            # entropy taken from the potential stops falling like T^3. At 1 GeV that already
            # overstates the entropy of the 2HDM plasma by a factor 3.7.
            entropy_density = float(h_eff_DS(t, pot, phase) + h_eff_coupled_radiation(t, pot)) * t**3
        except Exception:
            entropy_density = np.nan
        # A sound speed outside (0, 1] means the traced phase has stopped being a sensible
        # equilibrium, which happens far below completion, where the grid still reaches but
        # the false vacuum no longer describes a plasma. The bag value is the fallback there.
        if np.isfinite(cs_sq) and 0.0 < cs_sq <= 1.0:
            sound_speed_sq[i] = float(cs_sq)
        if np.isfinite(entropy_density) and entropy_density > 0.0:
            entropy[i] = entropy_density

    # The scale factor follows from entropy conservation, a^3 s = const, which is the integral
    # form of d ln a / dT = -1 / (3 c_s^2 T). Taking the ratio of entropies instead of
    # integrating the sound speed keeps a(T) accurate on the coarse grids of the percolation
    # solver, which can span ten or more decades in temperature: one interval alone could
    # otherwise contribute hundreds of e-folds, because the trapezoidal rule multiplies the
    # width of the interval by a sound speed that is already meaningless at its cold end.
    # Where the entropy is not usable, the bag relation a ~ 1/T continues the chain, and a
    # step that would shrink the universe as it cools is replaced by it as well.
    ln_a = np.zeros_like(temperatures)
    for i in range(1, temperatures.size):
        t_prev, t_cur = float(temperatures[i - 1]), float(temperatures[i])
        s_prev, s_cur = entropy[i - 1], entropy[i]
        if t_prev <= 0.0 or t_cur <= 0.0:
            step = 0.0
            ln_a[i] = ln_a[i - 1] + step
            continue
        bag_step = -(np.log(t_cur) - np.log(t_prev))
        if np.isfinite(s_prev) and np.isfinite(s_cur):
            step = -(np.log(s_cur) - np.log(s_prev)) / 3.0
        else:
            step = bag_step
        if step * bag_step < 0.0:
            step = bag_step
        ln_a[i] = ln_a[i - 1] + step
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


# Value of the percolation integral I at which the ODE sweep stops: the false-vacuum
# fraction exp(-I) is then 2e-22, far below anything an observable can resolve.
_PERC_INTEGRAL_STOP = 50.0


def percIntegralODE(
    T: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    vw: float = 1.0,
    *,
    sound_speed_sq: np.ndarray | None = None,
    scale_factor: np.ndarray | None = None,
    i_target: float | None = None,
) -> np.ndarray | tuple[np.ndarray, float | None]:
    r"""Compute I(T_i) at every grid point via the J_n ODE chain.

    With ``i_target`` the solver additionally records the temperature at which
    the percolation integral reaches that value, and the return value becomes
    the pair ``(I_values, crossing_temperature)``, where the crossing is
    ``None`` if the target is not reached. Taking the crossing from the
    integrator avoids reading it back off an interpolation of the sampled
    ``I`` values.

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
    np.ndarray or tuple
        ``I(T_i)`` at each grid point (same length as *T*).
        ``P(T_i) = 1 - exp(-I(T_i))``. With ``i_target``, the pair
        ``(I_values, crossing_temperature)``.
    """
    T = np.asarray(T, dtype=float)
    H = np.asarray(H, dtype=float)
    S = np.asarray(S, dtype=float)
    N = T.size
    if N <= 1:
        I_values = np.zeros(N)
        return I_values if i_target is None else (I_values, None)
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

    events = []
    if i_target is not None:
        def _reaches_target(u, y):
            return (4.0 * np.pi / 3.0) * vw**3 * y[3] - float(i_target)

        _reaches_target.terminal = False
        _reaches_target.direction = 0.0
        events.append(_reaches_target)

    # Stop once the false vacuum is gone for all practical purposes. Nothing
    # colder can change Tperc, Tf, Treh or R*, and the grid can extend far below
    # completion into regions where the traced false vacuum is no longer a
    # sensible equilibrium (its sound speed passes through zero, the scale
    # factor blows up by e^700), which used to make the integrator fail on
    # an otherwise healthy transition (2HDM benchmark, lambda3 = 5.7175).
    def _false_vacuum_gone(u, y):
        return (4.0 * np.pi / 3.0) * vw**3 * y[3] - _PERC_INTEGRAL_STOP

    _false_vacuum_gone.terminal = True
    _false_vacuum_gone.direction = 1.0
    events.append(_false_vacuum_gone)

    sol = integrate.solve_ivp(
        rhs,
        t_span=(float(u_eval[0]), float(u_eval[-1])),
        y0=[0.0, 0.0, 0.0, 0.0],
        method="DOP853",
        t_eval=u_eval,
        rtol=1e-10,
        atol=1e-300,
        dense_output=False,
        events=events,
    )
    if sol.status == -1:
        raise errors.PercolationError(
            "percIntegralODE: the percolation integral solver could not reach "
            "the cold end of the temperature grid. The action S_3(T) is most "
            "likely non-smooth on the scale of the support points (numerical "
            "jitter from path deformation), so the adaptive ODE step shrinks "
            "below floating-point spacing and the solver gives up. "
            "Suggested fixes: (a) rerun with precision_mode: tunneltight or "
            "benchmark to tighten the bounce-action path deformation; "
            "(b) if the model is multi-field, lower precision_deform_fRatioConv "
            "(default 1e-2 for Ndim>=2) further. "
            f"Underlying solver message: {sol.message}"
        )

    J3 = sol.y[3]  # J3 at each T_i reached before a terminal event
    if J3.size < u_eval.size:
        # Stopped at _PERC_INTEGRAL_STOP: the colder points keep that value, so
        # the true-vacuum fraction there is 1 to within exp(-_PERC_INTEGRAL_STOP).
        j3_stop = float(sol.y_events[-1][0][3]) if len(sol.y_events[-1]) else float(J3[-1])
        J3 = np.concatenate([J3, np.full(u_eval.size - J3.size, max(j3_stop, float(J3[-1]) if J3.size else 0.0))])
    I_trimmed = (4.0 * np.pi / 3.0) * vw**3 * J3
    # Restore trimmed hot-end points with I = 0 (A ≈ 0 there).
    if n_trim > 0:
        I_values = np.concatenate([np.zeros(n_trim), I_trimmed])
    else:
        I_values = I_trimmed
    I_values = np.asarray(I_values, dtype=float)
    if i_target is None:
        return I_values
    crossing = None
    if sol.t_events and len(sol.t_events[0]) > 0:
        crossing = float(np.exp(float(sol.t_events[0][-1])))
    return I_values, crossing


def percolation_temperature_from_ode(
    T: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    *,
    vw: float,
    pot,
    phase_symmetric,
    time_temperature_mode: str | None,
    f_perc: float,
):
    """Return the percolation temperature straight from the integrator.

    The profile sweep tabulates the true-vacuum fraction on the support grid,
    and the percolation temperature is normally recovered by interpolating
    those samples and root-finding on them. That read-off moves when the
    support points move. Here the same ODE is integrated once more with an
    event at the target value of the percolation integral, so the crossing is
    located by the integrator itself. Returns ``None`` if it cannot be found.
    """
    try:
        target = -np.log(max(1.0 - float(f_perc), 1e-300))
        sound_speed_sq, scale_factor = _time_temperature_factors(
            pot,
            phase_symmetric,
            np.asarray(T, dtype=float),
            time_temperature_mode,
        )
        _, crossing = percIntegralODE(
            T,
            H,
            S,
            vw=vw,
            sound_speed_sq=sound_speed_sq,
            scale_factor=scale_factor,
            i_target=target,
        )
    except Exception:
        return None
    if crossing is None or not np.isfinite(crossing) or crossing <= 0.0:
        return None
    return float(crossing)


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
    # Working assumption: the broken bubble reheats the DS and the coupled
    # radiation to one local common temperature Tb, while the cosmological scale
    # factor between successive steps is still inferred from the background
    # symmetric-phase temperatures TSYM / TSYM_ref. A decoupled bath is not
    # reheated and does not enter.
    heff_ds_pt = h_eff_DS(TBRO_ref, pot, phase_broken)
    heff_SM_pt = h_eff_coupled_radiation(TBRO_ref, pot)
    heff_ds = h_eff_DS(Tb, pot, phase_broken)
    heff_SM = h_eff_coupled_radiation(Tb, pot)
    crit = (heff_ds + heff_SM) * Tb**3
    crit -= (heff_ds_pt + heff_SM_pt) * TBRO_ref**3 * TSYM**3 / TSYM_ref**3
    return crit


def _finite_difference(func, x: float, rel_step: float = 1e-6) -> float:
    """Central difference of ``func`` at ``x`` with a relative step."""
    h = rel_step * max(abs(float(x)), 1e-30)
    return (func(float(x) + h) - func(float(x) - h)) / (2.0 * h)


def integrate_broken_temperature(
    pot,
    phase_symmetric,
    phase_broken,
    TSYM: np.ndarray,
    P: np.ndarray,
    T_target: float,
    *,
    p_seed: float = 1e-6,
    rtol: float = 1e-8,
    atol: float = 0.0,
):
    r"""Integrate the broken-phase temperature down to ``T_target``.

    The step-3 profile obtains the broken-phase temperature by stepping from one
    support point to the next: an entropy-conserving update followed by an energy
    correction for the volume fraction converted during the step. Those steps are
    a first-order discretisation of

    .. math::
        \frac{dT_b}{dT} = \frac{3\,S(T_b)}{T\,S'(T_b)}
        + \frac{P'(T)}{P(T)}\,\frac{e_s(T) - E(T_b)}{E'(T_b)} ,

    with :math:`S(x) = [h_{\rm ds}(x) + h_{\rm SM}(x)]\,x^3` proportional to the
    entropy density of the broken phase and :math:`E` its energy density. Reading
    the reheating temperature off that tabulated trajectory makes it depend on
    where the support points happen to sit, which shows up as point-to-point
    scatter along smooth parameter scans. Solving the equation with an
    error-controlled integrator makes the result depend on a tolerance instead.

    The true-vacuum fraction enters through the source term. Its derivative is
    taken on a spline of :math:`\ln I`, with :math:`I = -\ln(1-P)` the
    percolation integral, which is smooth and monotone, rather than on a spline
    of :math:`P`, which sweeps from 0 to 1 and whose derivative depends strongly
    on the node placement. The integration starts where the true-vacuum fraction
    first reaches ``p_seed``, located in the same variable, with the
    instantaneous-reheating condition as initial value.

    Returns the broken-phase temperature at ``T_target``, or ``None`` if the
    integration could not be carried out.
    """
    temps = np.asarray(TSYM, dtype=float)
    probs = np.asarray(P, dtype=float)
    finite = np.isfinite(temps) & np.isfinite(probs)
    if np.count_nonzero(finite) < 4:
        return None
    order = np.argsort(temps[finite])
    t_asc, p_asc = temps[finite][order], probs[finite][order]
    t_asc, unique_idx = np.unique(t_asc, return_index=True)
    p_asc = p_asc[unique_idx]
    if t_asc.size < 4:
        return None

    try:
        p_spline = interpolate.CubicSpline(t_asc, p_asc, extrapolate=False)
    except Exception:
        return None
    dp_spline = p_spline.derivative()

    # dI/dT = I * dlnI/dT and dP/dT = (1-P) * dI/dT, with I = -ln(1-P).
    with np.errstate(all="ignore"):
        i_all = -np.log(np.clip(1.0 - p_asc, 1e-300, None))
    ok = np.isfinite(i_all) & (i_all > 0.0)
    log_i_spline = None
    if np.count_nonzero(ok) >= 4:
        try:
            log_i_spline = interpolate.CubicSpline(t_asc[ok], np.log(i_all[ok]),
                                                   extrapolate=False)
            dlog_i_spline = log_i_spline.derivative()
        except Exception:
            log_i_spline = None

    def profile_at(T):
        """Return (P, dP/dT) at temperature ``T``."""
        if log_i_spline is not None:
            li = float(log_i_spline(T))
            if np.isfinite(li):
                i_val = float(np.exp(li))
                p_val = -np.expm1(-i_val)
                dp = (1.0 - p_val) * i_val * float(dlog_i_spline(T))
                if np.isfinite(p_val) and np.isfinite(dp):
                    return p_val, dp
        return float(p_spline(T)), float(dp_spline(T))

    target = float(T_target)
    if not np.isfinite(target) or target <= float(t_asc[0]) or target >= float(t_asc[-1]):
        return None

    hotter = t_asc[t_asc > target]
    if hotter.size == 0:
        return None
    seed_floor = float(hotter[0])

    # Seed: where the true-vacuum fraction reaches p_seed. In the hot tail P runs
    # over many orders of magnitude, so a spline through P can overshoot and give a
    # spurious crossing; ln I between neighbouring support points is monotone, so
    # interpolate linearly in (T, ln I) instead.
    t_seed = None
    if np.count_nonzero(ok) >= 2:
        t_good, log_i = t_asc[ok], np.log(i_all[ok])
        log_i_seed = np.log(-np.log(max(1.0 - p_seed, 1e-300)))
        for hi, lo in zip(t_good[::-1][:-1], t_good[::-1][1:]):
            if lo <= target:
                break
            a_hi = float(np.interp(hi, t_good, log_i))
            a_lo = float(np.interp(lo, t_good, log_i))
            if (a_hi - log_i_seed) * (a_lo - log_i_seed) <= 0.0 and a_hi != a_lo:
                w = (log_i_seed - a_hi) / (a_lo - a_hi)
                t_seed = float(hi + w * (lo - hi))
                break
        if t_seed is not None and (not np.isfinite(t_seed) or t_seed <= target):
            t_seed = None
    if t_seed is None:
        for hi, lo in zip(t_asc[::-1][:-1], t_asc[::-1][1:]):
            if lo <= target:
                break
            p_hi, p_lo = float(p_spline(hi)), float(p_spline(lo))
            if (p_hi - p_seed) * (p_lo - p_seed) <= 0.0 and p_hi != p_lo:
                try:
                    t_seed = float(optimize.brentq(lambda x: float(p_spline(x)) - p_seed, lo, hi))
                except Exception:
                    t_seed = None
                break
    if t_seed is None or not np.isfinite(t_seed) or t_seed <= target:
        t_seed = seed_floor
    if t_seed <= target:
        return None

    def e_sym(T):
        return energyDensity(pot, phase_symmetric, float(T), include_decoupled=False)

    def e_bro(x):
        return energyDensity(pot, phase_broken, float(x), include_decoupled=False)

    def s_bro(x):
        x = float(x)
        return (h_eff_DS(x, pot, phase_broken) + h_eff_coupled_radiation(x, pot)) * x**3

    try:
        tb_seed = optimize.brentq(
            Tb_criterion,
            float(phase_broken.Tmin),
            float(phase_broken.Tmax),
            args=(t_seed, phase_symmetric, phase_broken, pot),
        )
    except Exception:
        return None

    # Guard against a pathological right-hand side stalling the integrator; the
    # caller falls back to the tabulated trajectory.
    calls = {"n": 0}

    def rhs(T, y):
        calls["n"] += 1
        if calls["n"] > 200000:
            raise RuntimeError("integrate_broken_temperature: RHS evaluation cap reached")
        tb = float(y[0])
        if not np.isfinite(tb) or tb <= 0.0:
            return [0.0]
        ds = _finite_difference(s_bro, tb)
        de = _finite_difference(e_bro, tb)
        drift = 3.0 * s_bro(tb) / (T * ds) if ds != 0.0 else 0.0
        source = 0.0
        p_val, dp_val = profile_at(T)
        if np.isfinite(p_val) and p_val > 0.0 and de != 0.0 and np.isfinite(dp_val):
            source = dp_val * (e_sym(T) - e_bro(tb)) / (p_val * de)
        total = drift + source
        return [total if np.isfinite(total) else 0.0]

    try:
        sol = integrate.solve_ivp(
            rhs,
            (t_seed, target),
            [float(tb_seed)],
            method="LSODA",
            rtol=rtol,
            atol=atol if atol > 0.0 else 1e-12 * max(abs(float(tb_seed)), 1.0),
            dense_output=False,
        )
    except Exception:
        return None
    if not sol.success or sol.y.shape[1] == 0:
        return None
    result = float(sol.y[0, -1])
    return result if np.isfinite(result) and result > 0.0 else None


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



def _fit_action_slope(T_samples, S_samples, T: float, n_points: int) -> float:
    """Return T d(S3/T)/dT at ``T`` from a least-squares quadratic in S3/T.

    The quadratic is fitted to the ``n_points`` samples nearest to ``T`` in the
    variable ``T - T_ref``, so that the slope at ``T`` is its linear coefficient.
    """
    order = np.argsort(np.abs(T_samples - T))[:n_points]
    x = T_samples[order] - T
    y = S_samples[order] / T_samples[order]
    coeffs = np.polyfit(x, y, 2)
    return float(T * coeffs[1])


def calc_betaH_S3(T: float, Sint: interpolate.interp1d, outdict: dict, pot, phase_sym, phase_bro, verbose=False,
                  diagnostics: dict | None = None) -> float:
    """Calculate the phase transition speed from the action derivative.

    beta/H = T d(S3/T)/dT at ``T``, estimated by a local least-squares quadratic
    through the action samples of the percolation support rather than from the
    slope of an interpolating spline. The adaptive support places samples as
    close as 1e-6 T apart; an interpolant forced exactly through them converts
    an irregular 1e-2 error in S3/T (path-deformation noise) into errors of
    hundreds in beta/H, while a fit over the nearest samples averages it out.

    The sample selection is controlled by ``percolationConf``:

    * ``betaH_S3_fit_points`` nearest samples are used;
    * at least ``betaH_S3_fit_min_per_side`` of them must lie on each side of
      ``T`` and none further than ``betaH_S3_fit_max_rel_span`` in |T'/T - 1|,
      otherwise five new actions are computed at T (1 + k step), k = -2..2,
      with step ``betaH_S3_fallback_rel_step``, and the same fit is applied;
    * the fit is repeated with the ``betaH_S3_fit_check_points`` nearest samples,
      and a relative difference above ``betaH_S3_fit_rel_tol``, measured against
      the larger of the two, is reported through ``diagnostics["fit_unstable"]``.

    Both sample counts are clamped: a quadratic needs three samples, and the
    check fit uses at most as many as the fit itself, in which case it is skipped.

    See the CAUTION note in ``PercolationConf``: the defaults were validated on
    the 2HDM BSMPT benchmark only.

    Parameters
    ----------
    T : float
        The temperature at which to evaluate beta/H
    Sint: interpolate.interp1d
        Interpolation of the action; its nodes ``Sint.x`` are the support samples
    outdict : dict
        Dictionary storing the action evaluations, key is `T`
    pot : generic_potential
        The effective potential object
    phase_sym : PhaseInfo
        Information about the high temperature (symmetric) phase
    phase_bro : PhaseInfo
        Information about the low temperature (broken) phase
    diagnostics : dict, optional
        Filled with the samples used, the fallback flag, the check-fit value
        and ``fit_unstable``.

    Returns
    ----------
    float :
        The transition speed beta/H."""
    conf = pot.config.percolationConf
    # A quadratic needs three samples, and the check fit has to be the smaller one;
    # runtime overrides enforce that, a hand-edited config does not.
    n_fit = max(int(getattr(conf, "betaH_S3_fit_points", 11)), 3)
    n_check = min(max(int(getattr(conf, "betaH_S3_fit_check_points", 7)), 3), n_fit)
    rel_tol = float(getattr(conf, "betaH_S3_fit_rel_tol", 0.03))
    min_side = int(getattr(conf, "betaH_S3_fit_min_per_side", 2))
    max_span = float(getattr(conf, "betaH_S3_fit_max_rel_span", 0.02))
    fallback_step = float(getattr(conf, "betaH_S3_fallback_rel_step", 2e-3))
    diag = diagnostics if diagnostics is not None else {}

    tmin = max(float(phase_sym.Tmin), float(phase_bro.Tmin))
    tmax = min(float(phase_sym.Tmax), float(phase_bro.Tmax))
    if tmin >= tmax or T <= tmin or T >= tmax:
        if verbose:
            print(
                "Warning: cannot evaluate betaH outside phase overlap. "
                f"T={T:.8g}, overlap=[{tmin:.8g}, {tmax:.8g}]"
            )
        return np.nan

    T_samples = np.asarray(getattr(Sint, "x", []), dtype=float)
    S_samples = np.asarray(Sint(T_samples), dtype=float).reshape(-1) if T_samples.size else np.array([])
    usable = np.isfinite(T_samples) & np.isfinite(S_samples)
    T_samples, S_samples = T_samples[usable], S_samples[usable]

    use_support = T_samples.size >= n_fit
    if use_support:
        nearest = T_samples[np.argsort(np.abs(T_samples - T))[:n_fit]]
        below = int(np.sum(nearest < T))
        above = int(np.sum(nearest > T))
        span = float(np.max(np.abs(nearest / T - 1.0)))
        use_support = below >= min_side and above >= min_side and span <= max_span
        diag.update(n_below=below, n_above=above, max_rel_offset=span)

    if not use_support:
        # The support does not bracket T densely enough: compute a small symmetric
        # set of actions of our own, kept inside the phase overlap.
        step = fallback_step * T
        step = min(step, 0.2 * (T - tmin), 0.2 * (tmax - T))
        T_samples = T + step * np.arange(-2, 3)
        S_samples = np.array([calcAction(pot, float(t), phase_sym, phase_bro, outdict) for t in T_samples])
        if np.all(np.isinf(S_samples)):
            if verbose:
                print("WARNING: All actions around T are infinite; beta/H is set to inf.")
            return np.inf
        finite = np.isfinite(S_samples)
        T_samples, S_samples = T_samples[finite], S_samples[finite]
        if T_samples.size < 3:
            return np.nan
        n_fit = n_check = int(T_samples.size)
        diag.update(fallback=True)
        if verbose:
            print("betaH_S3: support too sparse around T, used five fresh actions for the fit.")
    else:
        diag.update(fallback=False)

    betaH = _fit_action_slope(T_samples, S_samples, T, n_fit)
    diag.update(betaH_fit=betaH, n_fit=int(min(n_fit, T_samples.size)))
    if n_check < n_fit and T_samples.size >= max(n_check, 3):
        check = _fit_action_slope(T_samples, S_samples, T, n_check)
        # Symmetric scale: near a zero of beta/H the two fits must not look
        # inconsistent only because the larger one is in the denominator.
        rel = abs(check - betaH) / max(abs(betaH), abs(check), 1e-300)
        diag.update(betaH_check=check, check_rel_diff=rel, fit_unstable=bool(rel > rel_tol))
    else:
        diag.update(fit_unstable=False)
    return betaH


def calcAlphas(T: float, pot, high_phase, low_phase, verbose=False,
               return_wall_strength: bool = False) -> tuple[float, ...]:
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
        alpha_eq)``, and ``alpha_hyd_wall`` as an eighth entry if
        ``return_wall_strength`` is true. ``alpha_theta`` is the bag-model strength fed to the
        GW-signal / kappa() pipeline; ``alpha_thetabar`` is the beyond-bag
        pseudo-trace-anomaly definition from arXiv:2004.06995, which divides the
        pressure of both phases by the sound speed of the broken phase."""

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
    # roughly 100 MeV. Here we assume that the decoupled sector has the same
    # temperature as the PT sector.
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

    # theta_bar = e - p / c_s^2 with ONE sound speed for both phases, that of the broken
    # phase: arXiv:2004.06995 eq. (2.13) with the definition below its eq. (2.11),
    # arXiv:2010.09744, and arXiv:2206.01130 sec. 2, which writes it out as
    # D theta_bar(T_n) = theta_s(T_n) - theta_b(T_n) with theta_bar = e - p / c_{s,b}^2.
    # Only then does everything the two phases share cancel in the difference; one sound
    # speed per phase leaves a radiation bath, and the zero point of the potential, behind
    # as (const) x (1/cs_sym^2 - 1/cs_bro^2). derived["c_s"] is already the broken-phase
    # value. With p = -Veff, e - p/c_s^2 = (Veff - T dV/dT) + Veff/c_s^2, hence (1 + 1/c_s^2).
    V0_ref = pot.V0(pot.X0) + pot.Vct(pot.X0) + pot.V1_from_X(pot.X0)
    Veff_sym = pot.Vtot(high_phi, T, include_decoupled=False) - V0_ref
    Veff_bro = pot.Vtot(low_phi, T, include_decoupled=False) - V0_ref
    # csSq_sym does not enter theta; it converts de/dT into the enthalpy 3 w below,
    # which normalises alpha_thetabar
    csSq_sym = calcSoundSpeedSq(pot, high_phi, T)
    csSq_bro = calcSoundSpeedSq(pot, low_phi, T)
    theta_sym = -T*pot.dVdT(high_phi, T, dT=dT, include_decoupled=False) + Veff_sym * (1 + 1/ csSq_bro)
    theta_bro = -T*pot.dVdT(low_phi, T, dT=dT, include_decoupled=False) + Veff_bro * (1 + 1/ csSq_bro)
    dedT = (pot.energyDensity(high_phi, T + dT) - pot.energyDensity(high_phi, T - dT))/(2*dT)
    alpha_thetabar = (theta_sym - theta_bro) / (3* csSq_sym * T * dedT)

    bosons_low = pot.boson_massSq(low_phi, 0)  # low-T phase masses
    bosons_high = pot.boson_massSq(high_phi, 0)  # high-T phase masses 
    fermions_low = pot.fermion_massSq(low_phi)
    fermions_high = pot.fermion_massSq(high_phi)

    # Delta m^2 = sum_i c_i N_i Delta m_i^2 with c_i = 1 (1/2) for bosons (fermions), eq. (2.8) of
    # 1903.09642; the 1/24 of the leading-order pressure enters alpha_inf below.
    gauge_coupling = pot.mass_spectrum.boson_gauge_couplings
    m2_bos_after, dof_bos, _, is_physical = bosons_low
    m2_bos_before, _, _, _ = bosons_high
    m2_fer_after, dof_fer = fermions_low
    m2_fer_before, _ = fermions_high

    delta_m2_bos = np.maximum(m2_bos_after - m2_bos_before, 0)
    m2factor = np.sum(dof_bos * is_physical * delta_m2_bos, axis=-1)

    delta_m2_fer = np.maximum(m2_fer_after - m2_fer_before, 0)
    m2factor += np.sum(dof_fer * delta_m2_fer, axis=-1) / 2.0

    m_bos_after = np.sqrt(np.where(m2_bos_after > 0, m2_bos_after, 0))
    # Avoid sqrt of negative mass squares by setting negatives to zero.
    # This only occurs for Goldstones, which are excluded via is_physical = 0.
    m_bos_before = np.sqrt(np.where(m2_bos_before > 0, m2_bos_before, 0))
    delta_m_bos = np.maximum(m_bos_after - m_bos_before, 0)

    # Enthalpy w = e + p of the symmetric phase of the transitioning sector alone,
    # without a decoupled radiation bath. The vacuum parts cancel in e + p, and Veff_sym
    # above is already the transitioning sector's effective potential.
    e_sym_PT = pot.energyDensity(high_phi, T, include_decoupled=False)
    w_sym_PT = -Veff_sym + e_sym_PT

    # alpha_inf and alpha_eq as defined in section 2 of 1903.09642: the leading- and
    # next-to-leading-order friction pressures on the wall divided by the radiation energy
    # density of the plasma the wall moves through, taken here as 3 w / 4, which equals
    # it for a relativistic plasma and is consistent with the enthalpy normalisation of
    # the hydrodynamic strengths below.
    rho_R_PT = 0.75 * w_sym_PT
    alpha_eq = T**3 / rho_R_PT * np.sum(delta_m_bos * gauge_coupling**2 * dof_bos * is_physical, axis=-1)
    alpha_inf = T**2 / (24 * rho_R_PT) * m2factor

    # Giese-KKS strengths, 3 w_sym in the denominator. The bubble wall is pushed only by
    # the transitioning sector, so the strength that enters the wall dynamics (gamma_eq and
    # kappa_col in calc_kappas) excludes the decoupled bath. The strength that enters the
    # sound waves includes it when the two sectors are hydrodynamically coupled.
    alpha_hyd_wall = (theta_sym - theta_bro) / (3 * w_sym_PT)
    include_decoupled_in_enthalpy = bool(pot.config.gwConf.coupled_hydrodynamics)
    Veff_sym_w = pot.Vtot(high_phi, T, include_decoupled=include_decoupled_in_enthalpy) - V0_ref
    e_sym_w = pot.energyDensity(high_phi, T, include_decoupled=include_decoupled_in_enthalpy)
    alpha_hyd = (theta_sym - theta_bro) / (3 * (-Veff_sym_w + e_sym_w))

    alphas = (alpha_p, alpha_theta, alpha_thetabar, alpha_e, alpha_hyd, alpha_inf, alpha_eq)
    if return_wall_strength:
        return alphas + (alpha_hyd_wall,)
    return alphas


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


def falseVacuumVolumeGrowthRate(TSYM, P, T, cs_sq: float = 1.0 / 3.0) -> float:
    r"""Growth rate of the physical false-vacuum volume, in units of ``3H``.

    The false vacuum occupies a physical volume
    :math:`\mathcal{V}_{\rm false} \propto a^3 (1 - P_{\rm true}) = a^3 e^{-I}`,
    which has to shrink for the transition to complete rather than to inflate
    forever. This requirement goes back to Turner, Weinberg and Widrow,
    Phys. Rev. D 46 (1992) 2384; the temperature form used here is eq. (2.26)
    of Ellis, Lewicki and No, arXiv:1809.08242, and we refer to it as the
    Lewicki criterion,

    .. math::
        \frac{1}{\mathcal{V}_{\rm false}}
        \frac{{\rm d}\mathcal{V}_{\rm false}}{{\rm d}t}
        = H\left(3 + T\frac{{\rm d}I}{{\rm d}T}\right) < 0 \,.

    That form assumes the radiation relation :math:`{\rm d}T/{\rm d}t = -HT`.
    TransitionListener integrates the percolation history with the generalized
    relation :math:`{\rm d}T/{\rm d}t = -3 c_{\rm s}^2 H T`, for which

    .. math::
        \frac{1}{\mathcal{V}_{\rm false}}
        \frac{{\rm d}\mathcal{V}_{\rm false}}{{\rm d}t}
        = 3H\left(1 + c_{\rm s}^2 T \frac{{\rm d}I}{{\rm d}T}\right) \,,

    and the returned quantity is the bracket. With :math:`c_{\rm s}^2 = 1/3` it
    reduces to :math:`(3 + T\,{\rm d}I/{\rm d}T)/3`, i.e. to the published
    criterion divided by three, so the sign is unchanged.

    The derivative is taken on a monotonic spline of :math:`\ln I` rather than
    of :math:`P`, because :math:`I = -\ln(1 - P_{\rm true})` spans decades over
    the sampled range while :math:`P` saturates.

    Parameters
    ----------
    TSYM : array_like
        Symmetric-phase temperatures of the percolation history.
    P : array_like
        True-vacuum fraction at those temperatures.
    T : float
        Temperature at which to evaluate the criterion, typically ``Tperc``.
    cs_sq : float, optional
        Sound speed squared entering the time-temperature relation. Pass
        ``1/3`` for the bag limit, which reproduces the published form.

    Returns
    -------
    float
        ``1 + cs_sq * T * dI/dT``. Negative means the false-vacuum volume is
        shrinking, i.e. the criterion is fulfilled. ``np.nan`` when the
        percolation history is too sparse to differentiate.
    """
    temperatures = np.asarray(TSYM, dtype=float)
    fractions = np.asarray(P, dtype=float)
    if temperatures.size != fractions.size or not np.isfinite(T):
        return float("nan")

    integral = -np.log1p(-np.clip(fractions, 0.0, 1.0 - 1e-15))
    usable = np.isfinite(temperatures) & np.isfinite(integral) & (integral > 0.0)
    temperatures = temperatures[usable]
    integral = integral[usable]
    if temperatures.size < 3:
        return float("nan")

    order = np.argsort(temperatures)
    temperatures = temperatures[order]
    integral = integral[order]
    keep = np.concatenate(([True], np.diff(temperatures) > 0))
    temperatures = temperatures[keep]
    integral = integral[keep]
    if temperatures.size < 3 or not (temperatures[0] <= T <= temperatures[-1]):
        return float("nan")

    try:
        log_integral = interpolate.PchipInterpolator(
            temperatures, np.log(integral), extrapolate=False
        )
        dIdT = float(np.exp(log_integral(T)) * log_integral(T, 1))
    except Exception:
        return float("nan")
    if not np.isfinite(dIdT):
        return float("nan")
    return 1.0 + float(cs_sq) * float(T) * dIdT


def percolation_sound_speed_sq(
    pot,
    phase_symmetric,
    T: float,
    *,
    time_temperature_mode: str | None = None,
    integral_method: str | None = None,
) -> float:
    """Sound speed squared of the time-temperature relation used for the percolation history.

    Mirrors ``percIntegralODE_full_sweep``: the double integral and the ``bag``
    mode integrate with ``dT/dt = -H T``, i.e. ``c_s^2 = 1/3``; the ODE in
    ``sound_speed`` mode uses the symmetric-phase value from
    ``_time_temperature_factors``, including its fallback to ``1/3``.
    """
    method = "ode" if integral_method is None else str(integral_method)
    if method == "double_integral":
        return 1.0 / 3.0
    sound_speed_sq, _ = _time_temperature_factors(
        pot, phase_symmetric, np.array([float(T)]), time_temperature_mode
    )
    if sound_speed_sq is None:
        return 1.0 / 3.0
    return float(sound_speed_sq[0])


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
