"""
This module is used to calculate the bubble dynamics in the dark sector.
It contains functions to calculate the energy density, effective degrees of freedom,
the Hubble parameter, the bubble nucleation rate, and from this the nucleation
and percolation temperature.

What this module computes is described in the documentation under
Nucleation, percolation and the expansion history: :doc:`/algorithms/nucleation_and_percolation`.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import copy
from dataclasses import dataclass
import math
import warnings
import numpy as np
from scipy import optimize
from scipy import interpolate
from scipy import integrate

from transitionlistener import thermodynamics as td
from transitionlistener import constants as cn
from transitionlistener import errors
from transitionlistener.compiled import multifield_potential
from transitionlistener.compiled import onefield_potential
from transitionlistener.compiled import percolation as compiled_percolation
from transitionlistener.compiled.splines import PchipInterpolator
from transitionlistener.helper_functions import (temperatureDerivativeStep,
                                                 thermalDerivativeStep)
from transitionlistener.pathDeformation import (bounceActionOverImages,
                                                bounceFreeParams,
                                                imagesToPairOver,
                                                modelAction)
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
        Pint = PchipInterpolator(T, P, extrapolate=True)
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
    # True where `Sr` holds an action a bounce solve produced, False where it was carried
    # forward from a warmer point because the fraction had already saturated and step 3
    # stopped solving. `np.isfinite(Sr)` cannot answer this since the carry made the tail
    # finite, and two consumers want different things from that array: `percIntegralODE`
    # needs a usable number wherever it integrates, while
    # `_cold_walk_has_left_the_bounce_region` needs to know where a solve was attempted and
    # failed, which is what marks the edge of the bounce region. None where no profile has
    # been built yet, and for the fixed-step backend, which never carries.
    Sr_computed: np.ndarray | None = None
    cold_tail_last_temperature: float | None = None
    cold_tail_last_probability: float | None = None
    cold_tail_last_integral: float | None = None
    cold_tail_last_log_gamma_h4: float | None = None
    cold_tail_post_peak_streak: int = 0
    # The accuracy estimate the post-sweep controller took on the profile it accepted, kept
    # so the verdict survives the solve. The controller accepts the profile either way -- by
    # the time it estimates, refinement has nothing left to try, so retrying would be futile
    # -- so without this field a profile that missed acc_rh is indistinguishable from one
    # that met it. Left None when the controller stopped at its iteration limit, because the
    # estimate it hands back there was measured on an earlier grid and does not describe the
    # returned profile.
    accuracy_estimate: dict | None = None
    # Support points spent approaching a boundary at which the bounce stops solving.
    # Its own allowance, like `rate_refine_points`: by the time the support reaches such a
    # boundary the rate allowance is long spent, and without a bound of its own the approach
    # repeats every rebuild. A new field goes last so no positional call site shifts.
    boundary_refine_points: int = 0


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
    from transitionlistener.percolation_adaptive_rate import _temperatures_match

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
            if not np.any(_temperatures_match(previous, float(value), relative=1e-10))
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
    entropy_definition = str(getattr(conf, "entropy_definition", ENTROPY_DEFINITIONS[0]))
    if entropy_definition not in ENTROPY_DEFINITIONS:
        raise errors.PercolationError(
            f"percolation_entropy_definition must be {ENTROPY_DEFINITIONS[0]!r} or "
            f"{ENTROPY_DEFINITIONS[1]!r}, got {entropy_definition!r}."
        )
    if algorithm_mode == "fixed_step_size" and entropy_definition != ENTROPY_DEFINITIONS[0]:
        # That solver keeps its own time-temperature relation, which takes both factors from the
        # potential: `scalefactorRatio` integrates d2V/dT2 / (3 dV/dT), i.e. s = -dV/dT, and the
        # sound speed comes from `calcSoundSpeedSq` on the same entropy. It never reads this
        # setting, so the combination it is given is whatever it hard-codes; silently ignoring a
        # thermodynamic choice is worse than refusing it. The default `dof_table` is accepted
        # there only for backwards compatibility, and is the combination that is mislabelled.
        raise errors.PercolationError(
            f"percolation_entropy_definition={entropy_definition!r} is not implemented for "
            "percolation_algorithm_mode='fixed_step_size', whose time-temperature relation is "
            "its own. Use the adaptive step size solver, or leave the definition at "
            f"{ENTROPY_DEFINITIONS[0]!r}."
        )

    n_action_min = max(int(getattr(conf, "n_action_min", 20)), 2)
    n_action_increment = max(int(getattr(conf, "n_action_increment", 5)), 1)
    n_action_max = max(int(getattr(conf, "n_action_max", 50)), n_action_min)
    boundary_basis = n_action_min if algorithm_mode == "adaptive_step_size" else static_n_action

    conf.algorithm_mode = algorithm_mode
    conf.integral_method = integral_method
    conf.time_temperature_mode = time_temperature_mode
    conf.entropy_definition = entropy_definition
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

    return compiled_percolation.distinct(np.sort(raw)[::-1])


def _on_kernels(pot, phase):
    """The compiled potential of ``pot`` and the spline of a one-field ``phase``, or ``(None, None)`` if either is missing."""
    read = getattr(pot, "thermal_kernels", None)
    kernels = read() if read is not None else None
    if kernels is None:
        return None, None
    spline = getattr(phase, "compiled_spline", None)
    spline = spline() if spline is not None else None
    return (kernels, spline) if spline is not None else (None, None)


def _on_multifield_kernels(pot, phase):
    """The compiled potential of a model with several fields and the spline of ``phase``, or ``(None, None)``."""
    read = getattr(pot, "current_kernels", None)
    kernels = read() if read is not None else None
    if not isinstance(kernels, multifield_potential.MultiFieldPotential):
        return None, None
    spline = getattr(phase, "compiled_trace", None)
    spline = spline() if spline is not None else None
    return (kernels, spline) if spline is not None and spline[0].shape[2] == kernels.ndim else (None, None)


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
    except errors.Timeout:
        raise
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
    if (type(T_DS) is float or type(T_DS) is np.float64) and T_DS > 0.0:
        kernels, spline = _on_kernels(pot, phase)
        if kernels is not None:
            return kernels.h_eff(spline, float(T_DS))
        kernels, spline = _on_multifield_kernels(pot, phase)
        if kernels is not None:
            return kernels.h_eff(spline, float(T_DS))
    try:
        vevT = phase.valAt(T_DS)
    except errors.Timeout:
        raise
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
    if (type(T) is float or type(T) is np.float64) and T > 0.0:
        read = getattr(pot, "thermal_kernels", None)
        kernels = read() if read is not None else None
        if kernels is not None and kernels.thermal["sm_bath"]:
            return kernels.h_eff_coupled(float(T))
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
    if (type(T) is float or type(T) is np.float64) and T > 0.0:
        kernels, spline = _on_kernels(pot, phase)
        if kernels is not None:
            return kernels.energy_density_on_phase(spline, float(T), include_decoupled)
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
    """Calculate the natural logarithm of the bubble nucleation rate.

    Parameters
    ----------
    T : float | np.ndarray
        Symmetric phase temperature
    S : float | np.ndarray
        The action at temperature T

    Returns
    ----------
    logGamma : np.ndarray
        ``ln Gamma``, i.e. ``4 ln T + (3/2) ln(S/2 pi T) - S/T``. It is ``+inf`` where the
        action vanishes and not a number where the action is infinite."""
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


def hotEnergyDensity(pot, X, T, include_decoupled: bool = True) -> float:
    """The part of ``pot.energyDensity`` that depends on the temperature.

    ``energyDensity`` is this plus ``DV0``, the vacuum energy measured from the zero-
    temperature minimum, which does not depend on the temperature. Anything built from a
    temperature difference of the energy density has to difference this part alone: under
    supercooling the vacuum energy is the larger by up to fifteen orders of magnitude, and a
    difference of the sum then has no significant digits left, or none at all: the two
    energies either side of a small temperature step can be the same double, with a
    difference of exactly zero, while this part differences cleanly.

    The vacuum energy also carries no enthalpy, entering ``e`` as ``+rho_vac`` and ``p`` as
    ``-rho_vac``, so an enthalpy formed as ``e + p`` leaves it out the same way.
    """
    return float(np.squeeze(pot.energyDensityDaisy(X, T)
                            + pot.radiationEnergyDensity(X, T, include_decoupled)))


def symmetricPhaseEnthalpy(pot, X, T, include_decoupled: bool = True) -> float:
    """The enthalpy ``w = e + p`` of the phase at ``X``, without the vacuum energy.

    ``include_decoupled`` selects the same plasma the caller's other thermodynamics uses: the
    strength that drives the wall is normalised to the transitioning sector alone, the one
    that drives the sound waves to the whole plasma where the two sectors are coupled.

    The vacuum energy enters ``e`` as ``+rho_vac`` and ``p`` as ``-rho_vac``, so it cancels
    analytically. Numerically the two are equal to every digit a double holds once the
    transition is supercooled far enough: written as ``-Veff + e`` this enthalpy was 2 per
    cent wrong at g = 0.5 of ``examples/example_line.yaml``. Built from the
    temperature-dependent parts alone it is exact there and identical from g = 0.6 up.
    """
    # pot.enthalpyDensity forms e + p from the temperature-dependent parts, and for a model that rewrites its
    # potential or energy density from the two with the vacuum energy removed (generic_potential).
    return float(np.squeeze(pot.enthalpyDensity(X, T, include_decoupled=include_decoupled)))



def requireUsableEnthalpy(value: float, T: float, terms: dict) -> float:
    """Return the enthalpy, or refuse the point if it cannot normalise a strength.

    Zero and not-a-number both produce a strength that is not a number, and infinity produces
    zero, so none of the three may reach a reported value. Testing only for zero let the other
    two through.
    """
    if not np.isfinite(value) or value == 0.0:
        raise errors.EnthalpyUnderflowError(T, terms, value=value)
    return value


def calcSoundSpeedSq(pot, X, T) -> float:
    """Compute the symmetric-phase sound speed squared at ``(X, T)``.

    Decoupled radiation is excluded because it does not participate in the
    local time-temperature relation of the transitioning plasma.
    """
    # Differentiate the temperature-dependent part on its own, not the whole potential. The
    # latter carries a large temperature-independent vacuum offset which cancels analytically
    # and not numerically, and under supercooling it is the larger by ten or more orders of
    # magnitude, so the finite differences lose every significant digit. `V_thermal` excludes
    # the decoupled bath, as this function requires.
    dT = thermalDerivativeStep(pot, T, X)
    dVdT = pot.dV_thermal_dT(X, T, dT=dT)
    d2VdT2 = pot.d2V_thermal_dT2(X, T, dT=dT)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs_sq = np.asarray(dVdT, dtype=float) / (float(T) * np.asarray(d2VdT2, dtype=float))
    return float(np.squeeze(cs_sq))


ENTROPY_DEFINITIONS = cn.ENTROPY_DEFINITIONS

# Relative step of the logarithmic entropy derivative of the "dof_table" scheme. The counted
# entropy is a smooth function of the temperature (the Standard Model part is a spline through a
# Savitzky-Golay-filtered table), so the choice is not delicate over several decades. Steps from
# 1e-3 down to 1e-6 leave c_s^2 within the structure the table itself contributes, even across
# the QCD crossover where that structure is strongest; only a step as coarse as 1e-2 is visible
# in the result.
_ENTROPY_REL_STEP = 1.0e-4


def entropy_density(pot, phase, T: float, definition: str = "dof_table") -> float:
    """Entropy density of the plasma of the traced phase, in the chosen scheme.

    Only ratios and logarithmic derivatives of this quantity are used, so the constant
    ``2 pi^2/45`` is dropped from the counted form and the two schemes share no absolute
    normalisation.

    Both describe the same plasma, the fields of the potential and the coupled radiation bath,
    and they differ in how they treat the fields. ``"dof_table"`` counts them as free particles
    at their zero-temperature masses and adds the tabulated entropy degrees of freedom of the
    bath. ``"eff_potential"`` takes ``-dV/dT``, which gives the fields their thermal masses and
    the daisy term, and carries the bath with its perturbative corrections, but not those
    corrections for the fields themselves. Neither route contains the other, which is why the
    two are compared and the difference reported rather than one of them corrected.

    """
    t = float(T)
    if definition == "eff_potential":
        X = phase.valAt(t)
        dT = temperatureDerivativeStep(pot, t, X)
        return -float(np.squeeze(pot.dVdT(X, t, dT=dT, include_decoupled=False)))
    if definition == ENTROPY_DEFINITIONS[0]:
        return float(h_eff_DS(t, pot, phase) + h_eff_coupled_radiation(t, pot)) * t**3
    # Both branches are named explicitly. Falling through to the counted route would answer a
    # misspelled scheme with a number instead of an error; callers inside the package validate
    # first, but this is a public function and the tests call it directly.
    raise errors.PercolationError(
        f"Unknown percolation entropy definition {definition!r}. "
        f"Supported definitions are {ENTROPY_DEFINITIONS[0]!r} and {ENTROPY_DEFINITIONS[1]!r}."
    )


def _entropy_history(pot, phase, temperatures, definition, sound_speed_sq, entropy) -> int:
    """Fill the sound speed and the entropy at each temperature; returns how many are usable."""
    usable_points = 0
    for i, temp in enumerate(temperatures):
        t = float(temp)
        # `valAt` extrapolates its spline rather than raising, so this is not a range check; it
        # only skips a temperature at which the traced phase cannot be evaluated at all. The
        # range itself is the caller's: the grid comes from `_phase_overlap_interval`.
        try:
            phase.valAt(t)
        except errors.Timeout:
            raise
        except Exception:
            continue
        try:
            entropy_here = entropy_density(pot, phase, t, definition)
        except errors.Timeout:
            raise
        except Exception:
            entropy_here = np.nan
        try:
            # One stencil for both schemes, differentiating along the phase the same entropy
            # that fixes a(T). A fixed-field derivative drops the term the phase contributes,
            # d2V/dXdT * dX/dT: along a phase dV/dX = 0, so dX/dT = -(d2V/dX2)^-1 d2V/dXdT,
            # and that term vanishes only where the phase does not move. It is exactly zero
            # for a symmetric phase pinned at the origin, which is why the fixed-field route
            # agrees with this one there. The discrepancy grows with the drift, reaching about a
            # tenth of a per cent where dX/dT = 0.5.
            h = t * _ENTROPY_REL_STEP
            # Keep the stencil inside the traced range. `valAt` interpolates a spline and
            # happily extrapolates instead of raising, and where it does raise, `h_eff_DS`
            # falls back to the zero-temperature vev; either way the derivative would be
            # taken on a field configuration that is not the traced phase. Clamping turns
            # the edges into a one-sided difference.
            t_lo = max(t - h, float(getattr(phase, "Tmin", t - h)))
            t_hi = min(t + h, float(getattr(phase, "Tmax", t + h)))
            s_lo = entropy_density(pot, phase, t_lo, definition)
            s_hi = entropy_density(pot, phase, t_hi, definition)
            if s_lo > 0.0 and s_hi > 0.0 and t_hi > t_lo:
                # c_s^2 = 1 / (d ln s / d ln T).
                cs_sq = (np.log(t_hi) - np.log(t_lo)) / (np.log(s_hi) - np.log(s_lo))
            else:
                # A non-positive entropy has no logarithm and no sound speed. The potential
                # route can reach that where dV/dT changes sign; the bag fallback applies.
                cs_sq = np.nan
        except errors.Timeout:
            raise
        except Exception:
            cs_sq = np.nan
        # A sound speed outside (0, 1) means the traced phase has stopped being a sensible
        # equilibrium, which happens far below completion, where the grid still reaches but
        # the false vacuum no longer describes a plasma. The bag value is the fallback there.
        # Strictly below one, as everywhere else: `calc_cs` refuses one or more and
        # `resolve_configured_sound_speed` will not accept it, and the spectrum divides by
        # `v_wall - c_s`, which is zero for a runaway wall at exactly one.
        # The two are one entropy read twice, so a temperature contributes both or neither:
        # keeping a sound speed where the entropy is unusable, or the other way round, would
        # build the scale factor and the time-temperature relation from different amounts of
        # information, which is the mixing this setting exists to end.
        usable = (np.isfinite(cs_sq) and 0.0 < cs_sq < 1.0
                  and np.isfinite(entropy_here) and entropy_here > 0.0)
        if usable:
            sound_speed_sq[i] = float(cs_sq)
            entropy[i] = entropy_here
            usable_points += 1

    return usable_points


def _time_temperature_factors(
    pot,
    phase,
    T: np.ndarray,
    mode: str | None = None,
    definition: str | None = None,
    usable_out: list | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(c_s^2, a/a_hot)`` arrays for the generalized integral.

    Both come from one entropy density, because ``d ln a/dT = -1/(3 c_s^2 T)`` and
    ``s a^3 = const`` are the same relation: a constant ``g_*s``, ``c_s^2 = 1/3`` and
    ``a ~ 1/T`` are one assumption, not three. Which entropy is used is the
    ``entropy_definition`` setting.

    ``None, None`` is returned for the bag-limit mode, whose ODE needs neither.
    """
    mode = "sound_speed" if mode is None else str(mode)
    if mode not in {"sound_speed", "bag"}:
        raise errors.PercolationError(
            f"Unknown percolation time-temperature mode {mode!r}. "
            "Supported modes are 'sound_speed' and 'bag'."
        )
    definition = "dof_table" if definition is None else str(definition)
    if definition not in ENTROPY_DEFINITIONS:
        raise errors.PercolationError(
            f"Unknown percolation entropy definition {definition!r}. "
            f"Supported definitions are {ENTROPY_DEFINITIONS[0]!r} and {ENTROPY_DEFINITIONS[1]!r}."
        )
    if mode == "bag" or pot is None or phase is None:
        return None, None

    temperatures = np.ascontiguousarray(T, dtype=float)
    sound_speed_sq = np.full_like(temperatures, 1.0 / 3.0, dtype=float)
    entropy = np.full_like(temperatures, np.nan, dtype=float)
    kernels, spline = _on_kernels(pot, phase)
    if (kernels is not None and definition == ENTROPY_DEFINITIONS[0] and kernels.thermal["sm_bath"]
            and temperatures.ndim == 1 and temperatures.size > 0 and hasattr(phase, "Tmin") and hasattr(phase, "Tmax")
            and np.all(np.isfinite(temperatures)) and np.all(temperatures > 0.0)):
        usable_points = kernels.entropy_history(spline, temperatures, float(phase.Tmin), float(phase.Tmax),
                                                _ENTROPY_REL_STEP, sound_speed_sq, entropy)
    else:
        usable_points = _entropy_history(pot, phase, temperatures, definition, sound_speed_sq, entropy)

    # The scale factor follows from entropy conservation, a^3 s = const, which is the integral
    # form of d ln a / dT = -1 / (3 c_s^2 T). Taking the ratio of entropies instead of
    # integrating the sound speed keeps a(T) accurate on the coarse grids of the percolation
    # solver, which can span ten or more decades in temperature: one interval alone could
    # otherwise contribute hundreds of e-folds, because the trapezoidal rule multiplies the
    # width of the interval by a sound speed that is already meaningless at its cold end.
    # Where the entropy is not usable, the bag relation a ~ 1/T continues the chain, and a
    # step that would shrink the universe as it cools is replaced by it as well.
    scale_factor = compiled_percolation.scale_factor_from_entropy(temperatures, entropy)
    if usable_out is not None:
        # The arrays hold the bag values where a temperature fell back, so a caller cannot
        # tell a history that was computed from one that was not. Ask for the count to find
        # out; it is a parameter rather than module state so that concurrent calls cannot
        # read each other's answer.
        usable_out.append(usable_points)
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

    # Over every image of the false vacuum, since the images are the same physical state and
    # the bubble nucleates along whichever pairing carries the smallest action.
    outdict = bounceActionOverImages(
        T, pot.Vtot, pot.gradV, outdict, tdict,
        imagesToPairOver(pot), verbose, pot.conversionFactor,
        **pot.config.tracingConf.tunneling_params, **bounceFreeParams(pot))

    # Give the model the chance to supply its own action. The default returns
    # what it is handed, so the four-dimensional path is untouched. Both are
    # kept: everything downstream reads "action", and "action_4d" preserves the
    # tunnelling solver's own answer for comparison. See
    # generic_potential.actionNuc for why this seam exists alongside the rate
    # one -- overriding the rate alone does not reach the percolation
    # integrands, which is how the 3D mode came to change nothing.
    entry = outdict[T]
    if "action_4d" not in entry:                 # the model has not supplied it already
        raw = entry["action"]
        entry["action_4d"] = raw
        entry["action"] = modelAction(pot, T, raw, instanton=entry.get("instanton"))
    return entry["action"]


def calcLogGamma(pot, T: float, outdict: dict) -> float:
    """Return ``ln(Gamma/V)`` at temperature `T`, caching it in `outdict`.

    The rate is evaluated through :meth:`generic_potential.logGammaNuc`, at the
    one point where the bounce profile stored by the tunnelling solver is still
    available alongside the action. The value is cached in
    ``outdict[T]["log_gamma"]`` so that repeated queries at the same temperature
    -- of which the adaptive percolation machinery makes many -- do not repeat
    the work.

    With the default ``rate_mode`` this returns exactly ``logGamma(T, S)``.

    Parameters
    ----------
    pot : generic_potential
        The effective potential object, which owns the rate.
    T : float
        The temperature, in internal units.
    outdict : dict
        The tunnelling dictionary, keyed by temperature. Must already contain
        an entry for `T`; call :func:`calcAction` first if it might not.

    Returns
    -------
    float
        ``ln(Gamma/V)`` in internal units.
    """
    entry = outdict[T]
    if "log_gamma" not in entry:
        entry["log_gamma"] = pot.logGammaNuc(
            T, entry["action"], instanton=entry.get("instanton")
        )
    return entry["log_gamma"]


def rateArrays(pot, temperatures: np.ndarray, actions: np.ndarray,
               outdict: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Gamma, ln Gamma)`` on a temperature grid, routed through the model.

    In the default ``rate_mode`` this delegates to :func:`Gamma` and
    :func:`logGamma`. In ``"model"`` mode the logarithm comes from the model
    -- which is the only way a rate depending on the bounce profile or on the
    three-dimensional matching can be obtained -- and ``Gamma`` is its
    exponential.

    Parameters
    ----------
    pot : generic_potential
        The effective potential object.
    temperatures, actions : np.ndarray
        The support grid and the bounce action on it.
    outdict : dict, optional
        Required in ``"model"`` mode, where the per-temperature entries carry
        the bounce profiles.

    Returns
    -------
    tuple of np.ndarray
        ``(Gamma, ln Gamma)``.
    """
    # a potential without a configuration (a stand-in in the grid-builder tests) takes the
    # default four-dimensional rate
    conf = getattr(getattr(pot, "config", None), "nucleationConf", None)
    mode = getattr(conf, "rate_mode", "S3overT_4D")
    if mode == "S3overT_4D":
        return (np.asarray(Gamma(temperatures, actions), dtype=float),
                np.asarray(logGamma(temperatures, actions), dtype=float))

    if outdict is None:
        raise ValueError(
            f"rate_mode {mode!r} needs the tunnelling dictionary, because the "
            "rate depends on more than (T, S): pass outdict."
        )
    log_gamma = np.asarray(
        [calcLogGamma(pot, float(t), outdict) for t in np.atleast_1d(temperatures)],
        dtype=float,
    )
    return np.exp(log_gamma), log_gamma


def percIntegral(
    T: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    vw=1.0,
    *,
    entropy_density: np.ndarray | None = None,
    cooling_factor: np.ndarray | None = None,
    scale_factor: np.ndarray | None = None,
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
    entropy_density : np.ndarray, optional
        Entropy density of the expansion history, ``s ~ a^-3``. Only its ratios enter.
        With both this and ``scale_factor`` left at ``None`` the bag relation ``s ~ T^3``
        is used.
    cooling_factor : np.ndarray, optional
        ``3 c_s^2``, the divisor of the transport factor, since
        ``dt = -dT / (3 c_s^2 H T)``. ``None`` means the bag relation ``3 c_s^2 = 1``.
    scale_factor : np.ndarray, optional
        ``a(T)`` itself, in place of ``entropy_density``; only ratios to ``a(T[0])``
        enter. Preferred where the history spans many e-folds, since ``a^-3`` underflows
        to zero once ``ln a`` exceeds about 236 and the ratio then has to be rebuilt from
        values that no longer carry it, while ``a`` stays representable to ``ln a`` of
        709. Passing both is an error.

    Returns
    -------
    float
        The integral evaluated at the last temperature ``T[-1]``.
    """
    if entropy_density is not None and scale_factor is not None:
        raise errors.PercolationError(
            "percIntegral takes either entropy_density or scale_factor, not both."
        )
    # See eq. (4.57) in 2305.02357
    # The ordering of the array is important
    if len(T) > 1:
        if not T[0] >= T[1]:
            raise errors.PercolationError("T is not decreasing in the percolation integral.")

    T = np.asarray(T, dtype=float)
    H = np.asarray(H, dtype=float)
    S = np.asarray(S, dtype=float)
    if T.size < 2:
        return 0.0
    # I = 4 pi/3 v^3 int dt' Gamma a(t')^3 (int_t'^t dt''/a)^3 with dt = -dT/(3 c_s^2 H T).
    # Only ratios of the scale factor enter; the bag limit is a ~ 1/T and 3 c_s^2 = 1.
    # A constant 3 c_s^2 may be given as a scalar; the sub-grid reads it per support point.
    cooling = (np.ones_like(T) if cooling_factor is None
               else np.asarray(cooling_factor, dtype=float) * np.ones_like(T))
    if scale_factor is not None:
        a = np.asarray(scale_factor, dtype=float)
        scale = a / a[0]
    elif entropy_density is not None:
        entropy = np.asarray(entropy_density, dtype=float)
        scale = (entropy / entropy[0]) ** (-1.0 / 3.0)
    else:
        scale = T[0] / T
    T, H, S, cooling, scale = _double_integral_subgrid(T, H, S, cooling, scale)
    # a(T') (int_T^T' dT'' base/a), built from the cold end with the ratios of neighbouring
    # scale factors only: a^3 and the transport factor never appear separately, so a history
    # of hundreds of e-folds neither overflows nor underflows. The ratio never exceeds one,
    # since a grows as T falls.
    y = compiled_percolation.double_integral(T.size, T, H, S, cooling, scale)
    return 4 * np.pi / 3 * vw**3 * y


# The support grid of the adaptive solver resolves the rate where it is large but can leave
# gaps of up to several e-folds of temperature on the hot shoulder. Across such a gap the
# trapezoidal rule overestimates an exponentially falling source by about beta du / 2, and the
# bubbles nucleated there, few but large, enter I with the cube of their radius: on grids the
# solver produced, I at the percolation point came out 40 to 60 % high and Tperc a few per cent
# high. The double integral is therefore evaluated on a sub-grid that interpolates the smooth
# ingredients between the support points.
_SUBGRID_MAX_LOG_STEP = 0.1       # largest change of ln(source) per sub-step
_SUBGRID_MAX_STEPS = 2000         # sub-steps per support interval
_SUBGRID_NEGLIGIBLE = 40.0        # intervals this far below the peak of ln(source) stay whole


def _clamp_between(values, first, second):
    """Hold ``values`` between ``first`` and ``second``, in either order."""
    return np.clip(values, min(first, second), max(first, second))


def _subgrid_source(T, H, S, cooling, scale):
    """What the kernels of the sub-grid read per support point, as contiguous arrays: the five inputs, the logarithm of
    the source with the mask of its finite entries, its change per interval and the sub-steps that change asks for,
    ln T, the four interpolated quantities, S/T, and the room needed for the interior points."""
    T, H, S, cooling, scale = (np.ascontiguousarray(np.broadcast_to(values, np.shape(T)), dtype=float)
                               for values in (T, H, S, cooling, scale))
    ok = (np.isfinite(S) & (S > 0.0) & np.isfinite(H) & (H > 0.0) & np.isfinite(cooling)
          & (cooling > 0.0) & np.isfinite(scale) & (scale > 0.0) & (T > 0.0))
    log_source = np.full_like(T, -np.inf)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        log_source[ok] = (np.asarray(logGamma(T[ok], S[ok]), dtype=float) + 3.0 * np.log(scale[ok])
                          - np.log(cooling[ok] * H[ok] * T[ok]))
        finite = np.isfinite(log_source)
        change = np.abs(np.diff(np.where(finite, log_source, 0.0)))
        want = np.ceil(change / _SUBGRID_MAX_LOG_STEP)
        S_over_T = S / T
        interpolated = np.array([S_over_T, np.log(np.where(ok, H, 1.0)), cooling, np.log(np.where(ok, scale, 1.0))])
        u = np.log(np.where(T > 0.0, T, np.nan))
    size = int(np.sum(np.clip(want, 1.0, _SUBGRID_MAX_STEPS) - 1.0)) + 1
    return (T, H, S, cooling, scale, log_source, finite, change, want, u, interpolated, S_over_T,
            float(_SUBGRID_NEGLIGIBLE), float(_SUBGRID_MAX_STEPS), size)


def _warn_of_the_cap(capped) -> None:
    """``capped`` is what the kernels report: the support intervals the cap limits, and the largest change among them."""
    # The cap bounds the memory one support interval can ask for, but it must not degrade the
    # resolution without saying so. It binds only where a single interval spans more than
    # _SUBGRID_MAX_STEPS * _SUBGRID_MAX_LOG_STEP in the logarithm of the source; the step then
    # achieved is change / _SUBGRID_MAX_STEPS, and the trapezoidal error on that interval grows
    # as its square over twelve.
    if capped[0] > 0:
        achieved = float(capped[1] / _SUBGRID_MAX_STEPS)
        warnings.warn(
            f"percolation sub-grid limited by _SUBGRID_MAX_STEPS on "
            f"{int(capped[0])} support interval(s): the logarithm of the source "
            f"changes by up to {float(capped[1]):.3g} across one of them, so it is "
            f"resolved to {achieved:.3g} instead of the requested {_SUBGRID_MAX_LOG_STEP:.3g}.",
            RuntimeWarning,
            stacklevel=3,
        )


def _double_integral_subgrid(T, H, S, cooling, scale):
    """Subdivide the support intervals of the double integral where the source varies.

    The source ``Gamma a^3 / (3 c_s^2 H T)`` is resolved to a change of at most
    ``_SUBGRID_MAX_LOG_STEP`` in its logarithm per sub-step, wherever it is within
    ``exp(_SUBGRID_NEGLIGIBLE)`` of its peak. Between the support points ``S/T``, ``ln H``,
    ``3 c_s^2`` and ``ln a`` are interpolated in ``u = ln T`` by a not-a-knot cubic spline,
    which reproduces a Gaussian turnover to 0.1 % across gaps of 0.7 e-folds where PCHIP is
    off by a few per cent. Intervals with a non-finite end, and grids that are fine already,
    are returned unchanged.

    Parameters
    ----------
    T : np.ndarray
        Descending temperatures of the support grid.
    H, S, cooling, scale : np.ndarray
        Hubble rate, action, ``3 c_s^2`` and scale factor on the support grid.

    Returns
    -------
    tuple of np.ndarray
        ``(T, H, S, cooling, scale)`` on the sub-grid, still descending in temperature.
    """
    capped = np.zeros(2)
    fine = compiled_percolation.subgrid(*_subgrid_source(T, H, S, cooling, scale), capped)
    _warn_of_the_cap(capped)
    return fine


# ---------------------------------------------------------------------------
# ODE-based percolation integral
# ---------------------------------------------------------------------------
#

# Floor on the logarithm of the source and the transport term of the percolation integral.
# `exp(-700)` is about 1e-304, which is above the denormal threshold, so a floored sample
# contributes nothing to the integral without underflowing to a value the integrator cannot
# divide by. A `nan` sample is floored here rather than dropped, so that the growth of bubbles
# already nucleated is still carried across the region where no new ones appear.
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

# Spacing in u = ln T of the lattice on which `percIntegralODE` evaluates its integrand, and
# the largest number of lattice nodes it builds (the spacing is doubled until it fits).
_RATE_LATTICE_DU = 2.0 ** -11
_RATE_LATTICE_MAX_NODES = 2 ** 17
# Samples closer than this in ln T are thinned out of `ActionExponentSpline`.
_ACTION_SPLINE_MIN_DU = 2.0 ** -9


def _rate_lattice(u_low: float, u_high: float, du: float = _RATE_LATTICE_DU) -> np.ndarray:
    """Nodes ``k du`` for integer ``k`` inside ``[u_low, u_high]``, plus the two ends.

    The nodes are fixed in ``u = ln T``, not placed relative to the support, so two supports
    that cover the same range share every interior node. A lattice node closer than ``du/8``
    to an end is dropped so that no interval is much shorter than the others.
    """
    span = float(u_high) - float(u_low)
    while span / du > _RATE_LATTICE_MAX_NODES:
        du *= 2.0
    k_low = math.ceil(float(u_low) / du)
    k_high = math.floor(float(u_high) / du)
    inner = np.arange(k_low, k_high + 1, dtype=float) * du
    inner = inner[(inner - u_low > du / 8.0) & (u_high - inner > du / 8.0)]
    return np.concatenate(([float(u_low)], inner, [float(u_high)]))


class ActionExponentSpline:
    r"""``S/T`` as a function of ``u = ln T``: a cubic spline over each run of usable samples.

    ``u`` ascending and ``S`` the action at those points; ``usable`` marks the samples that
    may enter. Every maximal run of consecutive usable samples gets its own not-a-knot cubic
    spline of ``S/T``, provided it keeps at least four samples; the spline is only defined
    from the first to the last sample of its run, and `covers` says where that is.

    Within a run a sample closer than ``min_du`` in ``u`` to the previous one kept is left
    out, the last sample of the run always kept. The adaptive support places samples as close
    as ``1e-6 T``, and a cubic through two samples that close takes its slope from their
    difference, i.e. from the noise of the bounce solver; thinned, the spline cannot amplify
    that noise beyond its value. Smooth actions lose nothing, since the dropped sample carries
    the same information as its neighbour.

    Used by `percIntegralODE` for the rate, and by the action interpolant of the observables
    (``transitionObservables.build_action_spline``) for the mean bubble separation, so that
    both read one action between the support points.
    """

    def __init__(self, u, S, usable=None, min_du: float = None):
        u = np.asarray(u, dtype=float)
        S = np.asarray(S, dtype=float)
        min_du = _ACTION_SPLINE_MIN_DU if min_du is None else float(min_du)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            exponent = S / np.exp(u)
        ok = np.isfinite(u) & np.isfinite(S) & (S > 0.0) & np.isfinite(exponent)
        if usable is not None:
            ok &= np.asarray(usable, dtype=bool)
        self.runs: list[tuple[int, int, float, float, interpolate.CubicSpline]] = []
        n = u.size
        start = 0
        while start < n:
            if not ok[start]:
                start += 1
                continue
            stop = start
            while stop + 1 < n and ok[stop + 1]:
                stop += 1
            kept = [start]
            for i in range(start + 1, stop + 1):
                if u[i] - u[kept[-1]] >= min_du:
                    kept.append(i)
                elif i == stop and len(kept) > 1:
                    kept[-1] = i
            if len(kept) >= 4:
                kept = np.asarray(kept)
                spline = interpolate.CubicSpline(u[kept], exponent[kept], bc_type="not-a-knot")
                self.runs.append((int(start), int(stop), float(u[start]), float(u[stop]), spline))
            start = stop + 1

    def covers(self, u) -> np.ndarray:
        """Whether each ``u`` lies inside one of the runs."""
        u = np.asarray(u, dtype=float)
        inside = np.zeros(u.shape, dtype=bool)
        for _, _, low, high, _ in self.runs:
            inside |= (u >= low) & (u <= high)
        return inside

    def __call__(self, u) -> np.ndarray:
        """``S/T`` at ``u``, ``nan`` outside every run."""
        u = np.asarray(u, dtype=float)
        out = np.full(u.shape, np.nan)
        for _, _, low, high, spline in self.runs:
            inside = (u >= low) & (u <= high)
            if np.any(inside):
                out[inside] = spline(u[inside])
        return out


def _rate_on_log_lattice(
    u: np.ndarray,
    S: np.ndarray,
    log_source: np.ndarray,
    log_transport: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""The integrand of `percIntegralODE` on a lattice fixed in ``u = ln T``.

    ``u`` ascending, ``S`` the action, ``log_source`` and ``log_transport`` the logarithms of
    the source and transport factors at the same support points. Returns the lattice and both
    factors on it.

    Interpolating ``ln gamma`` itself between the support points would make the result
    depend on where those points landed. ``ln gamma = b - S/T`` has a maximum at the peak of
    the nucleation rate, and a shape-preserving interpolant puts a zero slope at each sample
    next to an extremum, so it flattens the peak whenever no sample sits on it, and ``T_p``
    then moves when a support point is added or removed near the peak.

    The two parts of ``ln gamma`` are therefore interpolated apart. ``S/T`` carries the
    curvature of the peak (with a slowly varying Hubble rate the peak of the rate sits close
    to the minimum of ``S/T``) and is interpolated by `ActionExponentSpline`, a cubic spline,
    which follows an extremum between samples to the fourth power of the spacing instead of
    flattening it; the remainder ``b = ln gamma + S/T`` varies slowly (``ln S``, ``ln H``,
    ``ln a``) and keeps the shape-preserving interpolant. Both are evaluated on the lattice,
    and the integrator interpolates between lattice nodes, which are dense enough
    (``du = 2^-11``) for that step to be negligible. Adding or moving a support point then
    changes the integrand only by the change of the spline.

    Where the source is floored or the action is unusable the samples carry no rate, and the
    old interpolant of ``ln gamma`` is kept on every interval touching such a sample, as is
    any run of fewer than four usable samples. The transport factor has no extremum and keeps
    its interpolant everywhere; it is only evaluated on the lattice, because the integrator
    takes both factors on one set of nodes.
    """
    u = np.asarray(u, dtype=float)
    n = u.size
    if n < 4:
        return u, np.asarray(log_source, dtype=float), np.asarray(log_transport, dtype=float)
    S = np.asarray(S, dtype=float)
    log_source = np.asarray(log_source, dtype=float)
    old_source = PchipInterpolator(u, log_source, extrapolate=True)
    old_transport = PchipInterpolator(u, np.asarray(log_transport, dtype=float), extrapolate=True)
    nodes = _rate_lattice(u[0], u[-1])
    source = np.asarray(old_source(nodes), dtype=float)
    transport = np.asarray(old_transport(nodes), dtype=float)

    exponent = ActionExponentSpline(
        u, S, usable=np.isfinite(log_source) & (log_source > _LOG_GAMMA_SOURCE_FLOOR + 1.0))
    if exponent.runs:
        with np.errstate(invalid="ignore", divide="ignore"):
            exponent_samples = S / np.exp(u)
        for start, stop, low, high, _ in exponent.runs:
            run = slice(start, stop + 1)
            remainder = PchipInterpolator(u[run], log_source[run] + exponent_samples[run])
            inside = (nodes >= low) & (nodes <= high)
            source[inside] = remainder(nodes[inside]) - exponent(nodes[inside])
    source = np.maximum(source, _LOG_GAMMA_SOURCE_FLOOR)
    transport = np.maximum(transport, _LOG_GAMMA_SOURCE_FLOOR)
    return nodes, source, transport


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

    The nested double integral costs O(N^2) per evaluation; written as a chain of four
    first-order equations in ``J_0..J_3`` it is one sweep from the hot to the cold end,

    ::

       A    = Gamma / (T^4 H)      J_0' = -A        J_1' = -J_0 / H
       J_2' = -2 J_1 / H           J_3' = -3 J_2 / H

    with every ``J_n`` zero at the hot end and ``I = (4 pi/3) v_w^3 J_3``, where ``'`` is
    ``d/dT`` and ``T`` decreases along the sweep.

    With ``i_target`` the crossing temperature is taken from the integrator rather than read
    back off an interpolation of the sampled ``I``. With ``sound_speed_sq`` and
    ``scale_factor`` the generalised equation of state is used, ``gamma = Gamma a^3/(H T
    3 c_s^2)`` and ``nu = 1/(H T 3 c_s^2 a)``; without them it falls back to the bag limit,
    where ``c_s^2 = 1/3`` and ``a(T)`` cancels.

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
        # Every factor can be positive and finite on its own while their product still
        # underflows to exactly zero. The adaptive cold boundary walks T toward 0, so
        # `H * T * 3 c_s^2` reaches the denormal range and then 0, and `-log(0)` is
        # `+inf`. The floor clamp further down uses `np.maximum`, which removes `-inf`
        # but leaves `+inf` untouched, so the value survives into the PCHIP interpolants
        # and surfaces as a bare `ValueError: y must contain only finite values` with no
        # typed error and no diagnostic. A sample whose denominator has underflowed
        # carries no information, so mark it invalid here and let the existing
        # floor-and-trim path drop it like any other unusable point.
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            denominator = H * T * 3.0 * sound_speed_sq
            representable = (
                np.isfinite(denominator)
                & (denominator > 0.0)
                & np.isfinite(denominator * scale_factor)
                & (denominator * scale_factor > 0.0)
            )
        valid = valid & representable
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
    S_asc = S[n_trim:][::-1].copy()

    # The integrand is evaluated on a lattice fixed in u = ln T rather than interpolated
    # between the support points, see `_rate_on_log_lattice`.
    u_nodes, source_nodes, transport_nodes = _rate_on_log_lattice(
        u_asc, S_asc, log_gamma_source_asc, log_transport_asc)
    log_gamma_source_interp = PchipInterpolator(u_nodes, source_nodes, extrapolate=True)
    log_transport_interp = PchipInterpolator(u_nodes, transport_nodes, extrapolate=True)

    # The ODE chain in u = ln(T) coordinates.  Chain rule: d/du = T · d/dT,
    # so each RHS term acquires a factor of T = exp(u):
    #
    #   dJ0/du = -gamma(T) · T
    #   dJ1/du = -J0 · nu(T) · T
    #   dJ2/du = -2 J1 · nu(T) · T
    #   dJ3/du = -3 J2 · nu(T) · T
    #
    # We integrate from u_hot = ln(T[0]) downward to u_cold = ln(T[-1]) with
    # compiled_percolation.chain_integral, which is DOP853.
    #
    # Absolute tolerance: J0 starts at values as small as exp(-S_3/T_hot)
    # which can be ~exp(-800) ~ 1e-348.  atol=1e-300 keeps the solver from
    # treating J0 as zero in the early (very suppressed) hot-tail phase.
    #
    # The integration stops once the false vacuum is gone for all practical
    # purposes. Nothing colder can change Tperc, Tf, Treh or R*, and the grid can
    # extend far below completion into regions where the traced false vacuum is no
    # longer a sensible equilibrium (its sound speed passes through zero, the scale
    # factor blows up by e^700), which makes the integrator fail on an otherwise
    # healthy transition.
    u_eval = np.log(T)   # decreasing (mirrors T which is decreasing)
    prefactor = (4.0 * np.pi / 3.0) * vw**3
    J3 = np.empty(u_eval.size)
    status, reached, u_crossing, j3_stop = compiled_percolation.chain_integral(
        np.ascontiguousarray(log_gamma_source_interp.x),
        np.ascontiguousarray(log_gamma_source_interp.c),
        np.ascontiguousarray(log_transport_interp.c),
        _LOG_GAMMA_SOURCE_FLOOR,
        np.ascontiguousarray(u_eval),
        float(prefactor),
        np.nan if i_target is None else float(i_target),
        _PERC_INTEGRAL_STOP,
        1e-10,
        1e-300,
        J3,
    )
    if status == compiled_percolation.CHAIN_FAILED:
        raise errors.PercolationError(
            "percIntegralODE: the percolation integral solver could not reach "
            "the cold end of the temperature grid. The action S_3(T) is most "
            "likely non-smooth on the scale of the support points (numerical "
            "jitter from path deformation), so the adaptive ODE step shrinks "
            "below floating-point spacing and the solver gives up. "
            "Suggested fixes: (a) rerun with precision_mode: tunneltight or "
            "benchmark to tighten the bounce-action path deformation; "
            "(b) if the model is multi-field, lower precision_deform_fRatioConv "
            "(default 1e-2 for Ndim>=2) further."
        )

    if reached < u_eval.size:
        # Stopped at _PERC_INTEGRAL_STOP: the colder points keep that value, so
        # the true-vacuum fraction there is 1 to within exp(-_PERC_INTEGRAL_STOP).
        J3[reached:] = max(j3_stop, J3[reached - 1]) if reached > 0 else j3_stop
    I_trimmed = (4.0 * np.pi / 3.0) * vw**3 * J3
    # Restore trimmed hot-end points with I = 0 (A ≈ 0 there).
    if n_trim > 0:
        I_values = np.concatenate([np.zeros(n_trim), I_trimmed])
    else:
        I_values = I_trimmed
    I_values = np.asarray(I_values, dtype=float)
    if i_target is None:
        return I_values
    crossing = None if np.isnan(u_crossing) else float(np.exp(u_crossing))
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
    entropy_definition: str | None = None,
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
            entropy_definition,
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
    except errors.Timeout:
        raise
    except Exception:
        return None
    if crossing is None or not np.isfinite(crossing) or crossing <= 0.0:
        return None
    return float(crossing)


def _prefix_integrals(T, H, S, vw, cooling_factor, scale_factor) -> np.ndarray:
    """``percIntegral(T[:i+1], ...)`` for every ``i``."""
    T = np.asarray(T, dtype=float)
    if T.size < 2:
        return np.zeros(T.size)
    if not T[0] >= T[1]:
        raise errors.PercolationError("T is not decreasing in the percolation integral.")
    cooling = np.ones_like(T) if cooling_factor is None else np.asarray(cooling_factor, dtype=float) * np.ones_like(T)
    scale = T[0] / T if scale_factor is None else np.asarray(scale_factor, dtype=float) / scale_factor[0]
    capped = np.zeros(2)
    I_values = compiled_percolation.prefix_integrals(
        *_subgrid_source(T, np.asarray(H, dtype=float), np.asarray(S, dtype=float), cooling, scale),
        4 * np.pi / 3 * vw**3, capped)
    _warn_of_the_cap(capped)
    return I_values


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
    entropy_definition: str | None = None,
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
        # The double integral uses the same expansion history as the ODE, so both methods
        # read the same time-temperature relation.
        sound_speed_sq, scale_factor = _time_temperature_factors(
            pot,
            phase_symmetric,
            np.asarray(T, dtype=float),
            time_temperature_mode,
            entropy_definition,
        )
        if sound_speed_sq is None:
            cooling_factor = None
        else:
            cooling_factor = 3.0 * sound_speed_sq
        I_values = _prefix_integrals(T, H, S, vw, cooling_factor, scale_factor)
    elif method == "ode":
        sound_speed_sq, scale_factor = _time_temperature_factors(
            pot,
            phase_symmetric,
            np.asarray(T, dtype=float),
            time_temperature_mode,
            entropy_definition,
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


#: Step of the geometric walk that brackets the hottest root of the percolation criterion.
#: Small enough not to step over the root where it sits close to `Tmax`, large enough that
#: the walk costs a handful of action evaluations rather than a scan of the whole interval.
_APPROX_PERCOLATION_SCAN_RATIO = 1.15

#: How many steps of that walk may pass without a computable action before the estimate
#: declines. Such a step cannot bracket anything -- the criterion returns `+inf` for it, so
#: it carries no sign -- and `S_3/T` cannot be read from it either, so the nucleation-dead
#: stop below cannot bound the walk on its own: with every action non-finite the walk pays for
#: the whole interval, which runs from the critical temperature down to the integrator's floor
#: and costs a few hundred action evaluations. The bound counts *consecutive* such steps, so the
#: stretch above the critical temperature, where the broken phase is still the metastable one
#: and `bounceAction` returns `inf` by construction, is walked through rather than stopped at.
#: The walk starts just above the critical temperature, so that stretch is a step or two, and
#: twenty steps is more than an order of magnitude in temperature beyond it.
_APPROX_PERCOLATION_MAX_BLIND_STEPS = 20


def _bracket_sign(value) -> float:
    """The sign of a criterion value for bracketing: +1, -1, or 0 for no information.

    The two infinities are not symmetric. ``-inf`` is a definite answer: the criterion
    returns it above the top of the range and where an action is exactly zero, in both cases
    meaning "above the root", so discarding it skips the sign test exactly where the criterion
    is most emphatic. ``+inf`` carries no information, since it is returned at the bottom of
    the range by construction and wherever an action is not computable, and bracketing against
    it manufactures a root at the cold end of whatever interval it is handed. A stretch with no
    computable action is bounded by ``_APPROX_PERCOLATION_MAX_BLIND_STEPS`` instead.

    Returns
    -------
    float
        ``+1.0`` or ``-1.0`` where the value carries a definite sign, ``0.0`` where it carries
        none. ``-inf`` is a definite answer and gives ``-1.0``; ``+inf`` and ``nan`` carry no
        information and give ``0.0``, so that bracketing against either cannot manufacture a
        root.
    """
    value = float(np.squeeze(value))
    if np.isnan(value) or value == np.inf:
        return 0.0
    return 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)


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

    # The approximation is only used to place the first adaptive step size grid. If it
    # fails to bracket a root, fail fast and let the caller choose a controlled fallback
    # instead of spending many hidden action evaluations.
    #
    # Find the *hottest* temperature at which the criterion is met, by walking down from
    # `Tmax` until it changes sign and only then handing that sub-bracket to `brentq`.
    #
    # Handing `brentq` the whole interval does not work, because the interval carries more
    # than one root and `brentq` assumes one: the criterion is positive below the percolation
    # point and negative above it, but as `T` falls the rate vanishes, so it turns negative
    # again at the cold end. Percolation is the first temperature the universe meets on the way
    # down, so the hottest root is the one wanted and the others are artefacts of the bracket.
    #
    # The walk is geometric at `_APPROX_PERCOLATION_SCAN_RATIO` per step, two action
    # evaluations each, and reaches the root in a handful of steps because the root sits within
    # a factor of a few of `Tmax`. It stops where no bubble can nucleate at any Hubble rate,
    # and separately after `_APPROX_PERCOLATION_MAX_BLIND_STEPS` consecutive steps whose action
    # could not be computed, which the first stop cannot see because it has no `S_3/T` to read.
    args = (outdict, pot, phase_sym, phase_bro, Tmax, Tmin, vw, verbose)
    entered_live = False
    blind_steps = 0
    from transitionlistener.percolation_adaptive_gridbuilders import (
        _nucleation_is_dead,
    )  # imported here because that module imports this one at load time
    hot = float(Tmax)
    f_hot = _approx_percolation_criterion(hot, *args)
    cold = None
    probe = hot
    while probe > Tmin:
        probe = max(probe / _APPROX_PERCOLATION_SCAN_RATIO, Tmin)
        f_probe = _approx_percolation_criterion(probe, *args)
        # A probe that lands exactly on the root is the answer, and must be taken here.
        # `_bracket_sign` maps an exact zero to "no information", which is right for a
        # bracket -- a zero cannot sit on either side of itself -- but leaving it to the
        # sign test would discard the root and, worse, carry the zero into `f_hot`, after
        # which every later comparison is a product with zero and the walk can never
        # bracket anything again. It would then run to a dead or blind stop and raise,
        # having had the answer in hand.
        if float(np.squeeze(f_probe)) == 0.0:
            return np.squeeze(probe)
        if _bracket_sign(f_hot) * _bracket_sign(f_probe) < 0:
            cold = probe
            break
        # Stop once the walk leaves the band where a bubble can nucleate at all, but only
        # on the cold side of it. `S_3/T` is large at both ends for opposite reasons: the
        # action diverges as `T` approaches the critical temperature from below, so the hot
        # end is large because the transition has not started, and the cold end is large
        # because the rate has died. Testing it before the walk has entered the live band
        # stops at the first probe, which is what reading `Tmax` as "too cold" would mean.
        # Read the action the criterion has just cached rather than asking for it again:
        # `calcAction` keys `outdict` by `T`, and the criterion evaluated this `probe` on
        # the line above. A criterion that returned before computing one, at or outside the
        # bracket ends, leaves no entry, and the walk then carries on rather than treating
        # the absence as a dead temperature, counting it instead as a step without a
        # computable action, which is what it is.
        entry = outdict.get(probe)
        S_probe = entry.get("action") if isinstance(entry, dict) else None
        computable = S_probe is not None and np.isfinite(S_probe)
        # `_nucleation_is_dead` is the same test the controller's grid uses, read from
        # where it is defined rather than repeated here, so the threshold cannot drift
        # between the two. It reads a non-finite action as "not dead", which is right for
        # a grid point and not what this walk needs, so finiteness is required separately.
        live = computable and probe > 0 and not _nucleation_is_dead(probe, S_probe)
        # The two stops answer different questions and must not be reached through each
        # other. Leaving the live band is a statement about the *physics* at a temperature
        # whose action is in hand: `S_3/T` has passed the threshold, so nothing colder can
        # nucleate. A probe whose action could not be computed says nothing about the
        # physics there, and reaching the dead stop through one made a single transient
        # failed solve end the walk the moment it followed a live sample, which is the
        # opposite of what `_APPROX_PERCOLATION_MAX_BLIND_STEPS` is for and would step over
        # any sign change below it.
        if computable:
            blind_steps = 0
            if live:
                entered_live = True
            elif entered_live:
                break
        else:
            blind_steps += 1
            if blind_steps >= _APPROX_PERCOLATION_MAX_BLIND_STEPS:
                break
        # Only a probe that carries a sign becomes the bracket's hot end. A `+inf` from an
        # action that could not be computed has `_bracket_sign` zero, and letting it take
        # `f_hot` makes every later comparison a product with zero, so the walk steps over
        # the next sign change exactly as the exact-zero case above would: a finite
        # negative sample, then a failed solve, then a finite positive sample brackets
        # nothing, and the root between the first and the third is lost. Keeping the last
        # informative endpoint costs nothing -- the blind step is still counted above, so a
        # run of uncomputable probes still ends the walk -- and leaves `brentq` a bracket
        # whose ends are both signed.
        if _bracket_sign(f_probe) != 0.0:
            hot, f_hot = probe, f_probe
    if cold is None:
        raise errors.PercolationApproximation1Error(
            "The percolation approximation found no sign change between "
            f"{Tmax:.8g} and {Tmin:.8g} GeV."
        )
    try:
        Tperc = optimize.brentq(
            _approx_percolation_criterion,
            hot,
            cold,
            rtol=1e-3,
            args=args,
        )
    except ValueError as err:
        if verbose:
            print("Brentq failed in percolation approximation 1: ", err)
        raise errors.PercolationApproximation1Error(
            "Standard percolation approximation failed to bracket a root between "
            f"{cold:.8g} and {hot:.8g} GeV, the step of the walk that changed sign "
            f"inside the phase-overlap interval [{Tmin:.8g}, {Tmax:.8g}]: {err}"
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


def entropy_criterion_SYM_BRO(Tb: float, TBRO_ref: float, volume_ratio: float, phase_broken, pot) -> float:
    """Calculate the temperature of the reheated broken phase after an adiabatic expansion
    from a reference point. Only valid when entropy is conserved in between.

    Parameters
    ----------
    Tb : float
        Temperature in the broken phase.
    TBRO_ref : float
        Broken-phase temperature at the reference point, from which on entropy is conserved.
    volume_ratio : float
        ``(a_ref/a)^3``, the comoving volume at the reference point over that at the point
        of interest. It has to come from the expansion history of the percolation integral,
        ``s_sym(T)/s_sym(T_ref)`` for the entropy density ``s_sym`` of the false-vacuum plasma
        (``comoving_volume_ratios``), so that the broken plasma is redshifted with the same
        scale factor that fixes the percolation temperature.
    phase_broken : PhaseInfo
        Phase information about the end (broken) phase.
    pot : generic_potential

    Returns
    -------
    float
        Zero when the condition is met.
    """
    # The broken bubble reheats the DS and the coupled radiation to one local common
    # temperature Tb, and its entropy is diluted by the background expansion. A decoupled
    # bath is not reheated and does not enter.
    heff_ds_pt = h_eff_DS(TBRO_ref, pot, phase_broken)
    heff_SM_pt = h_eff_coupled_radiation(TBRO_ref, pot)
    heff_ds = h_eff_DS(Tb, pot, phase_broken)
    heff_SM = h_eff_coupled_radiation(Tb, pot)
    crit = (heff_ds + heff_SM) * Tb**3
    crit -= (heff_ds_pt + heff_SM_pt) * TBRO_ref**3 * volume_ratio
    return crit


def comoving_volume_ratios(pot, phase_symmetric, T, time_temperature_mode=None,
                           entropy_definition=None) -> np.ndarray:
    """``(a(T[i-1])/a(T[i]))^3`` between neighbouring temperatures of a descending grid.

    The scale factor is the one of the percolation integral, ``_time_temperature_factors``
    on the same grid: entropy conservation in the false-vacuum plasma, ``s_sym a^3 = const``,
    with the bag relation ``a ~ 1/T`` in ``bag`` mode and wherever the entropy is not usable.
    The first entry, which has no hotter neighbour, is one.
    """
    temperatures = np.asarray(T, dtype=float)
    ratios = np.ones(temperatures.size)
    if temperatures.size < 2:
        return ratios
    _, scale_factor = _time_temperature_factors(
        pot, phase_symmetric, temperatures, time_temperature_mode, entropy_definition)
    if scale_factor is None:
        ratios[1:] = (temperatures[1:] / temperatures[:-1]) ** 3
    else:
        scale_factor = np.asarray(scale_factor, dtype=float)
        ratios[1:] = (scale_factor[:-1] / scale_factor[1:]) ** 3
    return ratios


_CRITERIA = {Tb_criterion: onefield_potential.TB_CRITERION, energy_criterion_BRO: onefield_potential.ENERGY_CRITERION,
             entropy_criterion_SYM_BRO: onefield_potential.ENTROPY_CRITERION}


def criterion_root(criterion, a: float, b: float, args: tuple = ()) -> float:
    """``optimize.brentq(criterion, a, b, args=args)`` for one of the three criteria above.

    On a compiled potential with the tabulated Standard Model bath the search runs in the kernels; anything but a
    converged root goes to scipy, for its error.
    """
    kind = _CRITERIA.get(criterion)
    kernels, spline = (None, None) if kind is None else _on_kernels(args[-1], args[-2])
    if kernels is None or not kernels.thermal["sm_bath"] or not (a > 0.0 and b > 0.0):
        return optimize.brentq(criterion, a, b, args=args)
    constants = [0.0, 0.0, 0.0, 0.0]
    if kind == onefield_potential.TB_CRITERION:
        constants[0] = float(energyDensity(args[-1], args[1], args[0], include_decoupled=False))
    elif kind == onefield_potential.ENERGY_CRITERION:
        constants[:4] = [float(value) for value in args[:4]]
    else:
        reference, volume_ratio = float(args[0]), float(args[1])
        constants[0] = kernels.entropy_on_phase(spline, reference) * volume_ratio
    root, converged = kernels.criterion_root(kind, spline, float(a), float(b), constants)
    return root if converged else optimize.brentq(criterion, a, b, args=args)


def _convert_half_at_interval_start(
    pot, phase_symmetric, phase_broken, T_hot, Tb_hot, P_hot, dP, TBROmin, TBROmax
) -> tuple[float, bool]:
    r"""Broken-phase temperature after converting half of an interval's ``dP`` at its hot end.

    Step 3 of the adaptive solver carries the reheated broken-phase temperature ``T_b``
    from one support point to the next: the broken plasma is redshifted adiabatically from ``T_hot`` to the next support
    temperature, and the false vacuum converted in between, ``dP``, is added at the energy
    density of the symmetric phase there. Adding all of it at the cold end leaves the energy
    released inside the interval unredshifted, an error of first order in the width of the
    interval, so ``H(T)`` and with it ``T_p`` would change whenever the support changes.
    Converting half at the hot end, redshifting, and the other half at the cold end is the
    symmetric splitting, of second order.

    Returns the temperature to redshift from and whether the half was converted; on no
    conversion (``dP`` not positive, or the root not bracketed) it returns ``Tb_hot`` and
    ``False``, and the caller converts the whole ``dP`` at the cold end.

    The fixed-step solver keeps the end-point conversion: the copy a run executes is
    ``bubbledynamics_fixedstep._refine_percolation_temperature``, reached through
    ``transitionObservables_fixedstep``.
    """
    if not (np.isfinite(dP) and dP > 0.0 and np.isfinite(Tb_hot)):
        return float(Tb_hot), False
    e_sym_hot = energyDensity(pot, phase_symmetric, T_hot, include_decoupled=False)
    e_bro_hot = energyDensity(pot, phase_broken, Tb_hot, include_decoupled=False)
    try:
        Tb = criterion_root(
            energy_criterion_BRO,
            TBROmin,
            TBROmax,
            args=(e_sym_hot, e_bro_hot, P_hot + 0.5 * dP, 0.5 * dP, phase_broken, pot),
        )
    except ValueError:
        return float(Tb_hot), False
    if not np.isfinite(Tb):
        return float(Tb_hot), False
    return float(Tb), True


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
    log_entropy=None,
):
    r"""Integrate the broken-phase temperature down to ``T_target``.

    Stepping from one support point to the next, an entropy-conserving update followed by an
    energy correction for the volume fraction converted during the step, is a first-order
    discretisation of

    .. math::
        \frac{dT_b}{dT} = \frac{d\ln s_{\rm sym}}{dT}\,\frac{S(T_b)}{S'(T_b)}
        + \frac{P'(T)}{P(T)}\,\frac{e_s(T) - E(T_b)}{E'(T_b)} ,

    with :math:`S(x) = [h_{\rm ds}(x) + h_{\rm SM}(x)]\,x^3` proportional to the entropy
    density of the broken phase and :math:`E` its energy density. The entropy density of the
    false-vacuum plasma, :math:`s_{\rm sym} \propto a^{-3}`, sets the dilution by the
    expansion: ``log_entropy`` is :math:`\ln s_{\rm sym}(T)` of the expansion history of the
    percolation integral (``expansion_interpolants``), and ``None`` stands for the bag
    relation :math:`s_{\rm sym} \propto T^3`, the history of a percolation integral that uses
    it. Reading the reheating temperature off the tabulated trajectory instead makes it depend
    on where the support points happen to sit, which shows up as point-to-point scatter along
    smooth scans.

    The true-vacuum fraction enters through the source term, and its derivative is taken on a
    spline of :math:`\ln I`, with :math:`I = -\ln(1-P)`, which is smooth and monotone, rather
    than of :math:`P`, which sweeps from 0 to 1 and whose derivative depends strongly on the
    node placement. The integration starts where the fraction first reaches ``p_seed``, with
    the instantaneous-reheating condition as initial value.

    Returns
    -------
    float or None
        The broken-phase temperature at ``T_target``, in GeV, or ``None`` where the integration
        could not be carried out. Callers have to handle the ``None``.
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

    def dlog_entropy(T):
        return _finite_difference(lambda x: float(log_entropy(x)), T)

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
        dilution = 3.0 / T if log_entropy is None else dlog_entropy(T)
        drift = dilution * s_bro(tb) / ds if ds != 0.0 else 0.0
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
    except errors.Timeout:
        raise
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
    require_finite_action: bool = True,
) -> np.ndarray:
    """Return cached action temperatures that clearly belong to one transition.

    ``require_finite_action`` controls which question is being asked. The default asks
    which temperatures carry an action that can be *used*, so entries whose solve
    returned a non-finite value are dropped. Passing ``False`` asks instead which
    temperatures have already been *handed to the solver*: a solve that ran and came
    back non-finite is settled, its cost is sunk and asking again only re-reads the
    cache. The controller's support budget needs the second question, because a
    temperature counted as pending for the rest of the run holds budget that nothing
    can ever release.
    """
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
    # Relative to the requested window, without a floor of one: the floor made this 1e-10
    # absolute, which for a window at an internal temperature of 1e-9 reaches a tenth of the
    # way below it and admits cached actions from outside the window the caller asked for.
    # That needs a window below about internal 1e-8, colder than anything shipped here
    # reaches, so this is the same expression made safe rather than a defect observed.
    scale = max(
        abs(lower) if np.isfinite(lower) else 0.0,
        abs(upper) if np.isfinite(upper) else 0.0,
    )
    interval_tol = 1e-10 * scale

    # kept on the broken phase, and valid for these two traces: PhaseInfo.replaceTrace builds a new interpolator
    symmetric_trace = getattr(phase_symmetric, "interpolator", phase_symmetric)
    broken_trace = getattr(phase_broken, "interpolator", phase_broken)
    matched_points = getattr(phase_broken, "_matched_action_points", None)
    if matched_points is None:
        matched_points = {}
        try:
            phase_broken._matched_action_points = matched_points
        except AttributeError:
            pass

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

        if "action" not in payload:
            # No attempt has been recorded for this temperature at all, so it is
            # neither usable nor settled.
            continue
        try:
            action = float(payload["action"])
        except Exception:
            continue
        if require_finite_action and not np.isfinite(action):
            continue

        high_vev = payload.get("high_vev")
        low_vev = payload.get("low_vev")
        if high_vev is not None and low_vev is not None:
            # This scan runs hundreds of times over the same entries, and its comparisons were 9 %
            # of a 3D-rate run. An entry's answer is kept for as long as it is the same entry.
            traced = payload.get("high_vev_as_traced")
            known = matched_points.get(temperature)
            if (known is not None and known[0] is high_vev and known[1] is low_vev and known[2] is traced
                    and known[3] is symmetric_trace and known[4] is broken_trace):
                if not known[5]:
                    continue
                temperatures.append(float(temperature))
                continue
            try:
                symmetric_point = phase_symmetric.valAt(temperature)
                broken_point = phase_broken.valAt(temperature)
            except Exception:
                pass
            else:
                # `high_vev` is the image of the false vacuum the bounce was computed on,
                # which is not the point the phase tracing produced where a transformed image
                # carried the smaller action. For a group that reverses the fields the two are
                # a reflection apart, far outside this tolerance, so the entry would not be
                # recognised as this transition's and its action would be computed again. The
                # as-traced point is
                # recorded beside it for exactly this comparison; entries written before it
                # existed carry only `high_vev`.
                high_matches = phase_point_matches(high_vev, symmetric_point) or (
                    traced is not None and phase_point_matches(traced, symmetric_point))
                matched = bool(high_matches and phase_point_matches(low_vev, broken_point))
                matched_points[temperature] = (high_vev, low_vev, traced, symmetric_trace, broken_trace, matched)
                if not matched:
                    continue
        temperatures.append(float(temperature))
    return _temperature_grid(temperatures)


def _uncertainty_or_nan(value) -> float:
    """One relative uncertainty for the diagnostics snapshot, or nan if it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


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
    # The temperatures whose action the rescue gave up on and voided. Separate from
    # `action_jitter_rescue_success`, which is measured on the profile that remains and is
    # therefore True after a discard: voiding the offending sample is exactly what removes
    # the residual it was flagged for. Without this list the only record of a support point
    # dropped from the profile is the `_unstable_action_entries` note in the action cache,
    # which no consumer reads.
    action_jitter_rescue_discarded_temperatures: list[float] | None = None
    # Whether the accepted profile met acc_tperc, acc_tfinal and acc_rh, with the three
    # relative uncertainties behind that verdict. True means all three targets were measured
    # and met. False means a measured uncertainty exceeded its target, or a required observable
    # -- Tperc, Tf or RH -- could not be determined at all, an absent value being a different
    # matter from an absent reference. None means everything measurable was met and `acc_rh`
    # was never measured, or that no verdict reached these diagnostics at all: the
    # fixed_step_size backend, which has no such controller, and a profile the controller
    # handed back at its iteration limit rather than accepting. The three uncertainties tell
    # those two None cases apart, being all nan in the second. The targets do not gate
    # acceptance, so these fields are the only record of whether they were honoured.
    # `transitionObservables` reads them into the `WARNING:accuracy_targets_missed` and
    # `DIAG:accuracy_*_uncertainty` columns; a caller that keeps the metadata from
    # `calcPercAndEvolve(return_metadata=True)` reads them here directly.
    accuracy_targets_met: bool | None = None
    accuracy_tperc_uncertainty: float = math.nan
    accuracy_tfinal_uncertainty: float = math.nan
    accuracy_rh_uncertainty: float = math.nan

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
        # `getattr` rather than attribute access: the fixed_step_size solver has its own
        # `PercolationState` without this field, and although that one never reaches this
        # function today, the snapshot is taken after the solve has already succeeded and
        # so nothing in it may raise. The estimate is typed `dict[str, float | None]`, and a
        # None or a missing key is reported as nan rather than turning a completed
        # percolation solve into a failed point.
        estimate = getattr(state, "accuracy_estimate", None)
        if isinstance(estimate, dict):
            met = estimate.get("targets_met")
            metadata.accuracy_targets_met = None if met is None else bool(met)
            metadata.accuracy_tperc_uncertainty = _uncertainty_or_nan(estimate.get("Tperc_uncertainty"))
            metadata.accuracy_tfinal_uncertainty = _uncertainty_or_nan(estimate.get("Tf_uncertainty"))
            metadata.accuracy_rh_uncertainty = _uncertainty_or_nan(estimate.get("RH_uncertainty"))
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
            metadata.action_jitter_rescue_discarded_temperatures = [
                float(value)
                for value in (jitter_diagnostic.get("rescue_discarded_temperatures") or [])
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


# Decades below the in-band rate peak beyond which a sample is no longer judged for
# smoothness. This answers a different question from the grid builders' floor at 1e3
# decades: that one decides where to *place* support, where being permissive costs only a
# few wasted evaluations, while this one decides whether a wobble is evidence of an
# unstable bounce solve, and one quadratic in ln T cannot describe a range of hundreds of
# decades at all.
#
# Both ends of the window are constrained, and by different things. Open it much wider and
# the gate goes back to rejecting the dead tail it was written to ignore, since every false
# rejection seen came from samples forty decades or more below the peak. Close it much further
# and the window starves the fit, which needs five active samples for its quadratic. 30 sits
# between the two, with 1.35x margin to the shallowest observed false positive.
#
# This does not claim that a sample below the floor cannot matter: intervals worth more than a
# millionth of the percolation integral have been recorded hundreds of decades below a
# profile's own peak. It says that below the floor the gate cannot tell a bad solve from its
# own inability to fit, so it reports the exclusion in `n_dead_rate_excluded` rather than a
# verdict.
_JITTER_DEAD_RATE_DECADES = 30.0


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
    # P rises monotonically as T falls, so every sample colder than the one that reached
    # p_low is inside the band whether or not its rate still matters. Fitting one quadratic
    # in ln T across a dead tail that spans seven decades of temperature measures the fit's
    # inadequacy, not the data's: it produced residuals of billions of orders of magnitude and
    # advised the user to increase the bounce precision for samples far colder than percolation.
    #
    # The grid builders' floor, 1e3 decades below the peak, is far too loose to use here: it
    # is sized for deciding where to *place* support, where being permissive costs only a few
    # wasted evaluations, and no sample on the measured line ever reached it, so applying it
    # excluded nothing and the gate still judged the dead tail. P rises past f_perc while the
    # rate falls through hundreds of decades, so a sample can be well inside the probability
    # band and still carry no weight at all: a flagged point can sit hundreds of decades below
    # the peak with the transition already past percolation, where its apparent jump is a small
    # fraction of the range the quadratic has to span. Judge only the samples whose rate is
    # still live, with the threshold measured for this gate.
    band_idx = np.flatnonzero(mask)
    dead_in_band = 0
    if band_idx.size:
        order = band_idx[np.argsort(-T[band_idx])]
        rate_hot_to_cold = log10_rate[order]
        live_floor = float(np.max(rate_hot_to_cold)) - _JITTER_DEAD_RATE_DECADES
        # Liveness is decided by the warmest rate still to come at or below each sample --
        # the running maximum over the cold-side suffix -- and not by the sample's own value.
        # Judging a sample by its own rate lets a downward spike exempt itself: the defect
        # pushes it below the floor, and the gate it should have tripped never sees it. Both
        # signs occur, and a dropout tens of decades below its own local fit is the case that
        # would exempt itself. The suffix maximum is non-increasing towards the cold end, so
        # this keeps a contiguous warm stretch, and a lone dropout stays in it as long as the
        # samples just colder than it are still live.
        suffix_max = np.maximum.accumulate(rate_hot_to_cold[::-1])[::-1]
        dead = order[suffix_max < live_floor]
        dead_in_band = int(dead.size)
        if dead_in_band:
            mask = mask.copy()
            mask[dead] = False
    active_indices = np.flatnonzero(mask)
    if active_indices.size < 5:
        return {
            "n_active": int(active_indices.size),
            "max_residual_oom": math.nan,
            "n_dead_rate_excluded": dead_in_band,
        }

    x = np.log(T[active_indices])
    y = log10_rate[active_indices]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    indices = active_indices[order]
    if np.ptp(x) <= 1.0e-12:
        return {
            "n_active": int(active_indices.size),
            "max_residual_oom": math.nan,
            "n_dead_rate_excluded": dead_in_band,
        }

    degree = 2 if x.size >= 3 else 1
    try:
        coeff = np.polyfit(x, y, degree)
        smooth = np.polyval(coeff, x)
    except errors.Timeout:
        raise
    except Exception:
        return {
            "n_active": int(active_indices.size),
            "max_residual_oom": math.nan,
            "n_dead_rate_excluded": dead_in_band,
        }
    residual = y - smooth
    if residual.size == 0 or not np.any(np.isfinite(residual)):
        return {
            "n_active": int(active_indices.size),
            "max_residual_oom": math.nan,
            "n_dead_rate_excluded": dead_in_band,
        }
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
        "n_dead_rate_excluded": dead_in_band,
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
    discarded_temperatures: list[float] = []
    original_tunneling = copy.deepcopy(pot.config.tracingConf.tunneling_params)
    # the same tightening as the betaH_S3 fit rescue, which keeps a stricter setting if the run
    # already asked for one
    tight_tunneling = _tight_tunneling_params(pot)

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
        except Exception as exc:
            if old_key is not None and old_payload is not None:
                outdict[old_key] = old_payload
            if isinstance(exc, errors.Timeout):
                pot.config.tracingConf.tunneling_params = original_tunneling
                raise
            break
        finally:
            pot.config.tracingConf.tunneling_params = original_tunneling
        diagnostic = _action_rate_jitter_diagnostic(state.TSYM, state.Hr, state.Sr, state.Pr)

        # The retry has had its turn. If this same sample is still the worst offender and
        # still outside the threshold, the solver cannot compute it consistently, and
        # keeping it injects a rate feature that is orders of magnitude wrong into P(T).
        # Discard it and let the profile stand on the samples that are trustworthy.
        #
        # What the discard does and does not do. The non-finite action is dropped by the
        # action interpolants built from the returned `Sr`, so the rate feature it carried no
        # longer reaches `R_*`, `betaH_RH`, `betaH_S3`, the spline nucleation-temperature
        # estimate or the jitter diagnostic. What it does not do is re-derive the profile: the
        # percolation integral has already run with the sample in it, so `Pr`, `Tb`, `Tperc`
        # and the support bank still describe the profile that included it, and the returned
        # `Sr` and `Pr` therefore describe slightly different supports.
        #
        # That is tolerated rather than hidden, because both alternatives measured worse.
        # Re-deriving the profile from the reduced support and re-running the gates leaves the
        # coupling's inverse duration far from the trend its neighbours set, and refusing the
        # point instead costs every coupling that discards at all.
        #
        # The couplings that reached the discard were sick for a reason outside this file: the
        # traced symmetric phase was interpolated with an overshooting cubic, so the bounce
        # tunnelled from a displaced point and returned too small an action, and that is what
        # tripped the gate. `PhaseInfo._build_interpolator` now uses a shape-preserving
        # interpolator, which removes the cause. The discard stays, because nothing stops
        # another model from tripping the gate for a reason of its own.
        still_worst = diagnostic.get("temperature")
        residual_now = diagnostic.get("max_residual_oom")
        if (
            still_worst is not None
            and round(float(still_worst), 12) == rounded_key
            and residual_now is not None
            and np.isfinite(float(residual_now))
            and float(residual_now) > threshold
        ):
            discarded_temperatures.append(float(temperature))
            state.Sr[idx] = np.nan
            dead_key = _matching_outdict_key(outdict, temperature)
            if dead_key is not None:
                payload = outdict.get(dead_key)
                if isinstance(payload, dict):
                    payload["action"] = np.nan
                    payload["trantype"] = 0
            unstable = outdict.setdefault("_unstable_action_entries", [])
            if isinstance(unstable, list):
                unstable.append(
                    {
                        "T": float(temperature),
                        "action": None,
                        "reason": "action_rate_jitter_discarded_after_failed_retry",
                        "max_residual_oom": float(residual_now),
                    }
                )
            diagnostic = _action_rate_jitter_diagnostic(
                state.TSYM, state.Hr, state.Sr, state.Pr
            )

    diagnostic = dict(diagnostic)
    diagnostic["rescue_discarded_temperatures"] = [
        float(value) for value in discarded_temperatures
    ]
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
) -> dict[str, float | str | None]:
    r"""Estimate ``Tnuc`` from the saved rate/P/H splines without new actions.

    The diagnostic solves

    ``N(T) = 4*pi/3 * int_T^Tinit Gamma/H^4 * (1-P) dT/T = 1``

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

        # A cubic spline, as the double integral uses: N crosses one on the hot side, where the support points are
        # sparse, and the slopes of a monotone interpolant there are only first-order accurate and move with them.
        rate_spline = interpolate.CubicSpline(u, rate, extrapolate=False)
        prob_spline = PchipInterpolator(u, prob, extrapolate=False)
        n_grid = max(1000, 20 * int(u.size))
        u_dense = np.linspace(float(u[0]), float(u[-1]), n_grid)
        rate_dense = np.asarray(rate_spline(u_dense), dtype=float)
        prob_dense = np.clip(np.asarray(prob_spline(u_dense), dtype=float), 0.0, 1.0)
        with np.errstate(divide="ignore"):
            log_integrand = (rate_dense + np.log1p(-prob_dense))[::-1]
        # Summed from the hot end: N runs over many decades, and a difference from the total would
        # lose the hot end, where N crosses one, to the rounding of the total. Each cell is
        # integrated as the exponential its integrand is. The trapezoid of an exponential over a
        # cell exceeds the exact integral by a share (step in ln f)^2/12 while that step is small,
        # and the step reaches 57.6 in the steepest cell of the conformal benchmark of the
        # documentation, with a median of 0.158, where the trapezoid is wrong by a factor.
        prefactor = 4.0 * np.pi / 3.0      # the volume of a sphere of Hubble radius 1/H
        step = float(u_dense[1] - u_dense[0])
        # The cell is integrated in the logarithm, so that a cell whose log integrand rises
        # by hundreds is still the exponential it is. Writing it as the larger endpoint
        # times a factor below one keeps every intermediate finite: clipping the exponent
        # instead undercounts such a cell by the factor it was clipped by, and a cell whose
        # colder endpoint underflows came out as zero even when the warmer endpoint carries
        # the whole count.
        lower = log_integrand[:-1]
        upper = log_integrand[1:]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            growth = (upper - lower) / step
            rise = np.abs(upper - lower)
            larger = np.maximum(lower, upper)
            cells = np.exp(larger + np.log(-np.expm1(-rise)) - np.log(rise)) * step
            # A flat cell is the rectangle, which is the limit of the line above.
            cells = np.where(rise <= 1.0e-8, np.exp(larger) * step, cells)
            # An endpoint of zero, which is what a true-vacuum fraction that rounds to one
            # leaves behind, has no logarithm and no exponential model. The cell is finite
            # and not zero, so it is integrated in ordinary space, where a linear integrand
            # is exact.
            no_logarithm = ~np.isfinite(lower) | ~np.isfinite(upper)
            cells = np.where(no_logarithm,
                             0.5 * step * (np.exp(lower) + np.exp(upper)), cells)
        cells[~np.isfinite(cells)] = 0.0
        # The grid may stop just above the crossing, so the contribution above the hottest node is
        # added from the slope of the rate there. The extrapolation is cut off one grid range in
        # ln(T) above that node: the slope of a cubic spline at its last node can be arbitrarily
        # close to zero, and the integral to infinite temperature, exp(ln f)/(-slope), then grows
        # without bound and counts bubbles no support point of the rate history reports.
        hot_slope = float(rate_spline(u[-1], 1))
        u_above = float(u[-1] - u[0])
        if np.isfinite(hot_slope) and hot_slope <= 0.0:
            # A slope of exactly zero is the series limit of the expression below and not an
            # absent contribution: at `decay` of zero the factor is 1 and the term is the
            # rectangle `exp(ln f) * u_above`. Excluding it dropped a term the same size as
            # the ones being kept, so a flat hot tail could make the count appear to cross
            # inside the history when the bounded count above the hottest node already
            # exceeds one. The `abs(decay) > 1e-8` test below already carries the limit, so
            # zero only has to be let through to it.
            decay = hot_slope * u_above
            above_grid = float(np.exp(np.minimum(log_integrand[0], 700.0))) * u_above * (
                np.expm1(decay) / decay if abs(decay) > 1.0e-8 else 1.0
            )
        else:
            # A positive slope means the rate rises towards the hot end, where it must fall,
            # and a non-finite one means the spline could not be differentiated there.
            above_grid = 0.0
        N_hot_to_cold = above_grid + np.concatenate(([0.0], np.cumsum(cells)))
        N_hot_to_cold *= prefactor
        # A non-finite total is still an answer. On a rate that spans hundreds of e-folds the
        # integrand overflows in the cells below the crossing, which the zeroing of the non-finite
        # cells drops, and the crossing is reached from the hot end before the integrand grows
        # that far, so an infinite total costs the max_N diagnostic and not the temperature.
        max_N = float(np.nanmax(N_hot_to_cold)) if N_hot_to_cold.size else math.nan
        if np.isnan(max_N):
            return {"Tnuc": None, "warning": "spline_tnuc_failed", "max_N": math.nan}
        if max_N < 1.0:
            return {
                "Tnuc": None,
                "warning": "spline_tnuc_not_reached",
                "max_N": max_N,
                "integral_prefactor": prefactor,
            }

        crossing = np.flatnonzero(N_hot_to_cold >= 1.0)
        if crossing.size == 0:
            return {"Tnuc": None, "warning": "spline_tnuc_failed", "max_N": max_N}
        idx = int(crossing[0])
        if idx == 0:
            # One bubble is already counted at the hottest temperature of the rate history, so the
            # crossing lies above every support point and this history cannot say where it is.
            return {"Tnuc": None, "warning": "spline_tnuc_failed", "max_N": max_N}
        # The place of the crossing inside its cell is solved in the logarithm, for the same
        # reason the cell integral is. The cell's hot endpoint can be below the smallest
        # normal double, as it is when the rate jumps by hundreds of e-folds inside one cell;
        # dividing by it in ordinary space gave an infinite distance and returned a nucleation
        # temperature of exactly zero with no warning at all.
        shortfall = (1.0 - float(N_hot_to_cold[idx - 1])) / prefactor
        start = float(lower[idx - 1])
        exponent = float(growth[idx - 1])
        if not (shortfall > 0.0) or not np.isfinite(start) or not np.isfinite(exponent):
            return {"Tnuc": None, "warning": "spline_tnuc_failed", "max_N": max_N}
        log_shortfall = math.log(shortfall)
        if exponent == 0.0:
            # A flat cell is the rectangle, so the distance is the shortfall over the integrand.
            log_below = log_shortfall - start
            below = math.exp(log_below) if log_below < 709.0 else math.inf
        else:
            # The logarithm of the dimensionless `shortfall * exponent / exp(start)`, which is
            # what the exponential model has to reach inside the cell.
            scale = log_shortfall + math.log(abs(exponent)) - start
            if exponent > 0.0:
                # log1p(exp(scale)), which stays finite however large `scale` is.
                below = float(np.logaddexp(0.0, scale)) / exponent
            elif scale >= 0.0:
                # The cell falls towards the cold end and cannot supply the missing count.
                return {"Tnuc": None, "warning": "spline_tnuc_failed", "max_N": max_N}
            else:
                below = math.log1p(-math.exp(scale)) / exponent
        if not np.isfinite(below) or below < 0.0:
            return {"Tnuc": None, "warning": "spline_tnuc_failed", "max_N": max_N}
        # The crossing lies in this cell, so the distance cannot exceed its width.
        below = min(below, step)
        u_cross = float(u_dense[-idx]) - below
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


# The rate-jitter rescue divides the path-deformation convergence target by this factor,
# after the absolute cap. A fixed absolute target on its own cannot work: a model
# that already declares that target gets no tightening and the rescue becomes a no-op.
_JITTER_RESCUE_TIGHTEN_FACTOR = 5.0


def _tight_tunneling_params(pot):
    """A copy of the tunneling parameters with the path deformation tightened.

    The same two settings the ``tunneltight`` precision mode applies, and the same ones the
    rate-jitter rescue uses.
    """
    tight = copy.deepcopy(pot.config.tracingConf.tunneling_params)
    deform = dict(tight.get("deformation_deform_params", {}))
    # Smaller is stricter for both, so a configuration already stricter than this is kept:
    # the point is to tighten the deformation, never to loosen what the run was asked for.
    for name, value in (("converge_0", 1.0), ("fRatioConv", 5.0e-3)):
        current = deform.get(name)
        try:
            deform[name] = min(float(current), value)
        except (TypeError, ValueError):
            deform[name] = value
    # `min(current, constant)` leaves a model that already asks for the constant with no
    # tightening at all, which makes the rescue recompute the same action with the same
    # settings, which is the state of any run that already configures the two constants above:
    # every deformation parameter of the "tight" set is then bit-identical to the run's and the
    # rate-jitter rescue is a no-op. Divide by a further factor instead, so there is always
    # real margin: with the cap above, the tight target is 1e-3 for any run at or above 5e-3,
    # and one fifth of the configured value below that (6e-4 at the multi-field default of
    # 3e-3). `converge_0` keeps the plain clamp, so the margin comes from
    # `fRatioConv` alone, where halving the target was measured to cut the action jitter by
    # about a factor three, so a further factor buys a comparable amount; the cost is bounded
    # because the rescue only
    # recomputes the handful of outlier temperatures it targets.
    try:
        deform["fRatioConv"] = float(deform["fRatioConv"]) / _JITTER_RESCUE_TIGHTEN_FACTOR
    except (KeyError, TypeError, ValueError):
        pass
    tight["deformation_deform_params"] = deform
    return tight


def _recompute_actions_tight(pot, temperatures, phase_sym, phase_bro, outdict, progress):
    """Recompute the actions at ``temperatures`` with the path deformation tightened.

    ``progress`` is the caller's record: the cache entries removed under ``dropped``, so that
    the cache can be put back as it was, and the temperatures actually recomputed under
    ``done``. Both matter where this raises part of the way through.

    ``calcAction`` hands back whatever is already in ``outdict`` for a temperature, and the
    percolation temperature is always in there, so an action computed at the loose deformation
    would be returned unchanged and the recomputation would be silent about it. Every entry the
    stencil lands on is therefore dropped first and kept for the record.

    """
    original = pot.config.tracingConf.tunneling_params
    try:
        pot.config.tracingConf.tunneling_params = _tight_tunneling_params(pot)
        actions = []
        for temperature in temperatures:
            temperature = float(temperature)
            stale_key = _matching_outdict_key(outdict, temperature)
            if stale_key is not None:
                stale = outdict.pop(stale_key, None)
                if isinstance(stale, dict):
                    progress["dropped"].append((stale_key, stale))
                    replaced = outdict.setdefault("_unstable_action_entries", [])
                    if isinstance(replaced, list):
                        replaced.append({"T": temperature, "action": stale.get("action"),
                                         "reason": "betaH_S3_fit_rescue"})
            actions.append(calcAction(pot, temperature, phase_sym, phase_bro, outdict))
            progress["done"].append(temperature)
        return np.array(actions, dtype=float)
    finally:
        pot.config.tracingConf.tunneling_params = original


def _restore_dropped_actions(outdict, dropped) -> None:
    """Put back the actions a retry removed, for a retry whose answer is not used.

    The value that is reported then came from the samples as they were, and the cache has to
    say the same thing. Entries recomputed successfully are left in place: they are the better
    number for whatever asks next.
    """
    for key, payload in dropped or ():
        outdict[key] = payload
    replaced = outdict.get("_unstable_action_entries")
    if isinstance(replaced, list):
        outdict["_unstable_action_entries"] = [
            entry for entry in replaced if entry.get("reason") != "betaH_S3_fit_rescue"
        ]


def calc_betaH_S3(T: float, Sint: interpolate.interp1d, outdict: dict, pot, phase_sym, phase_bro, verbose=False,
                  diagnostics: dict | None = None) -> float:
    """Calculate the phase transition speed from the action derivative.

    The derivative is estimated by a local least-squares quadratic through the action samples
    of the percolation support, not from the slope of an interpolating spline. The adaptive
    support places samples as close as 1e-6 T apart, and an interpolant forced exactly through
    them turns an irregular 1e-2 error in ``S3/T``, which is what the path deformation leaves,
    into errors of hundreds in beta/H; a fit over the nearest samples averages it out.

    ``percolationConf`` controls the sample selection: how many of the nearest samples are
    used, how many of them must lie on each side of ``T`` and how far they may reach, and what
    to do when those conditions are not met, which is to compute five new actions around ``T``
    and fit those. The fit is repeated on a smaller set and a disagreement is reported through
    ``diagnostics["fit_unstable"]`` rather than corrected. See the caution in
    ``PercolationConf``: the defaults were validated on the 2HDM benchmark only.

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
        and ``fit_unstable``. ``fit_temperatures`` holds the temperatures of the
        samples behind the returned value: the support samples, the fallback
        actions, or the stencil of a fit rescue whose answer was used.

    Returns
    ----------
    float :
        The transition speed beta/H.
    """
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

    def fit_and_check(T_values, S_values):
        """The slope over the nearest samples, and how far the smaller fit disagrees."""
        slope = _fit_action_slope(T_values, S_values, T, n_fit)
        if n_check < n_fit and T_values.size >= max(n_check, 3):
            other = _fit_action_slope(T_values, S_values, T, n_check)
            # Symmetric scale: near a zero of beta/H the two fits must not look
            # inconsistent only because the larger one is in the denominator.
            return slope, other, abs(other - slope) / max(abs(slope), abs(other), 1e-300)
        return slope, None, None

    betaH, check, rel = fit_and_check(T_samples, S_samples)
    diag.update(betaH_fit=betaH, n_fit=int(min(n_fit, T_samples.size)),
                fit_temperatures=T_samples[np.argsort(np.abs(T_samples - T))[:n_fit]])
    if rel is None:
        diag.update(fit_unstable=False)
        return betaH
    diag.update(betaH_check=check, check_rel_diff=rel, fit_unstable=bool(rel > rel_tol))
    if rel <= rel_tol or not bool(getattr(conf, "betaH_S3_fit_rescue", False)):
        return betaH

    # The two fits disagree. Two things can do that, and they need different answers. The
    # actions can be noisy: at the default path-deformation tolerance the scatter of S3/T
    # reaches a few tenths, against a physical variation across the window of the same size.
    # Or the samples can sit too close together: the adaptive support places them as little as
    # 1e-6 T apart, and over a baseline that short even an exact action leaves the slope
    # undetermined. Recomputing the support samples would only cure the first, so the fit is
    # repeated on a stencil of its own, spaced as the sparse-support fallback spaces one, with
    # the deformation tightened as the rate-jitter rescue tightens it.
    half = max(int(n_fit) // 2, max(n_check, 3) // 2 + 1)
    # The outermost sample sits at half steps from T, so the cap on the step has to know how
    # many there are: the sparse-support fallback takes two and can afford a fifth of the
    # distance to the edge of the overlap each way.
    step = fallback_step * T
    step = min(step, 0.8 * (T - tmin) / half, 0.8 * (tmax - T) / half)
    T_stencil = T + step * np.arange(-half, half + 1, dtype=float)
    diag.update(fit_rescue_rel_span=float(np.max(np.abs(T_stencil / T - 1.0))))
    # the record is ours, so a failure part of the way through still says what was taken from the
    # cache and how many actions were really recomputed, rather than how many were planned
    progress: dict = {"dropped": [], "done": []}
    try:
        tight_S = _recompute_actions_tight(pot, T_stencil, phase_sym, phase_bro, outdict, progress)
    except Exception as exc:                                   # pragma: no cover - solver failure
        diag.update(fit_rescue_attempts=len(progress["done"]),
                    fit_rescue_error=f"{type(exc).__name__}: {exc}", fit_rescue_success=False)
        _restore_dropped_actions(outdict, progress["dropped"])
        if isinstance(exc, errors.Timeout):
            # The alarm that raises this is for the run as a whole and can go off anywhere,
            # here included. Carrying on with the slope the retry was called to replace would
            # let a point run past the time it was given and report a number as if nothing had
            # happened.
            raise
        return betaH
    diag.update(fit_rescue_attempts=len(progress["done"]))
    finite = np.isfinite(tight_S)
    if int(np.count_nonzero(finite)) < max(n_check, 3):
        diag.update(fit_rescue_success=False)
        _restore_dropped_actions(outdict, progress["dropped"])
        return betaH
    T_tight, S_tight = T_stencil[finite], tight_S[finite]
    n_fit_tight = min(n_fit, int(T_tight.size))
    betaH_tight, check_tight, rel_tight = fit_and_check(T_tight, S_tight)
    if rel_tight is None:
        diag.update(fit_rescue_success=False)
        _restore_dropped_actions(outdict, progress["dropped"])
        return betaH
    diag.update(betaH_fit=betaH_tight, betaH_check=check_tight, check_rel_diff=rel_tight,
                fit_unstable=bool(rel_tight > rel_tol),
                fit_rescue_success=bool(rel_tight <= rel_tol),
                betaH_before_rescue=betaH, n_fit=n_fit_tight,
                fit_temperatures=T_tight[np.argsort(np.abs(T_tight - T))[:n_fit]])
    if verbose:
        print(f"betaH_S3: the two fits differed by {100 * rel:.2f} %; refitted on "
              f"{int(T_stencil.size)} fresh actions spanning {100 * diag['fit_rescue_rel_span']:.2f} % "
              f"in T with the deformation tightened, {100 * rel_tight:.2f} % after, "
              f"beta/H {betaH:.6g} -> {betaH_tight:.6g}.")
    return betaH_tight


def calc_rate_log_slope(T: float, Sint: interpolate.interp1d, outdict: dict, pot, phase_sym, phase_bro,
                        verbose=False, diagnostics: dict | None = None,
                        temperatures=None) -> float:
    """Return ``-d ln Gamma / d ln T`` at ``T``, the slope of the nucleation rate itself.

    The rate is the model's, :func:`calcLogGamma`, at the actions the model supplies. It
    differs from the slope of the action alone, :func:`calc_betaH_S3`, by the prefactor
    ``T^4 (S/2 pi T)^(3/2)``, and by whatever a model's ``logGammaNuc`` adds beyond it. For
    the rates in this repository ``logGammaNuc`` is ``logGamma(T, S)``, the rate of the
    percolation integrals, and the three-dimensional model carries its determinant
    correction in the action, so :func:`calc_betaH_S3` contains that correction as well.

    The slope is the linear coefficient of a least-squares quadratic in ``ln Gamma``
    against ``T - T_ref``. ``temperatures`` are the samples to fit, normally
    ``diagnostics["fit_temperatures"]`` of :func:`calc_betaH_S3` at the same ``T``, so that
    both speeds rest on the same actions, including the stencil of its fit rescue when that
    rescue replaced the support fit. Without them the samples are selected as
    :func:`calc_betaH_S3` selects them before any rescue: the support samples nearest to
    ``T`` (``betaH_S3_fit_points``, at least ``betaH_S3_fit_min_per_side`` on each side,
    none further than ``betaH_S3_fit_max_rel_span``), otherwise five actions at
    ``T (1 + k step)``, k = -2..2, with ``betaH_S3_fallback_rel_step``. Neither the
    stability check nor the fit rescue of :func:`calc_betaH_S3` is repeated here. With the
    samples of :func:`calc_betaH_S3` every action is already in ``outdict``; the rate is
    evaluated at most once per sample and cached there, which is the cost to watch for a
    model whose ``logGammaNuc`` is expensive.

    Converted to time with the factor of :meth:`_beta_time_temperature_factor`, this is
    the observable ``betaH_Gamma = H^-1 d ln Gamma/dt``.

    Returns
    -------
    float
        ``-d ln Gamma / d ln T``, or ``nan`` if fewer than three finite rates are available.
    """
    conf = pot.config.percolationConf
    n_fit = max(int(getattr(conf, "betaH_S3_fit_points", 11)), 3)
    min_side = int(getattr(conf, "betaH_S3_fit_min_per_side", 2))
    max_span = float(getattr(conf, "betaH_S3_fit_max_rel_span", 0.02))
    fallback_step = float(getattr(conf, "betaH_S3_fallback_rel_step", 2e-3))
    diag = diagnostics if diagnostics is not None else {}

    tmin = max(float(phase_sym.Tmin), float(phase_bro.Tmin))
    tmax = min(float(phase_sym.Tmax), float(phase_bro.Tmax))
    if tmin >= tmax or T <= tmin or T >= tmax:
        return np.nan

    if temperatures is not None:
        T_samples = np.asarray(temperatures, dtype=float).reshape(-1)
        T_samples = T_samples[np.isfinite(T_samples)]
        diag.update(shared_samples=True, fallback=False)
    else:
        T_samples = np.asarray(getattr(Sint, "x", []), dtype=float)
        T_samples = T_samples[np.isfinite(T_samples)]
        use_support = T_samples.size >= n_fit
        if use_support:
            nearest = T_samples[np.argsort(np.abs(T_samples - T))[:n_fit]]
            use_support = (int(np.sum(nearest < T)) >= min_side and int(np.sum(nearest > T)) >= min_side
                           and float(np.max(np.abs(nearest / T - 1.0))) <= max_span)
            T_samples = nearest
        if not use_support:
            step = min(fallback_step * T, 0.2 * (T - tmin), 0.2 * (tmax - T))
            T_samples = T + step * np.arange(-2, 3)
        diag.update(shared_samples=False, fallback=not use_support)

    log_gamma = []
    for t in T_samples:
        t = float(t)
        action = calcAction(pot, t, phase_sym, phase_bro, outdict)
        log_gamma.append(calcLogGamma(pot, t, outdict) if np.isfinite(action) else np.nan)
    log_gamma = np.asarray(log_gamma, dtype=float)
    finite = np.isfinite(log_gamma)
    if int(np.count_nonzero(finite)) < 3:
        return np.nan
    x = T_samples[finite] - T
    coeffs = np.polyfit(x, log_gamma[finite], 2)
    diag.update(n_fit=int(np.count_nonzero(finite)))
    return float(-T * coeffs[1])


def energyDensityTemperatureDerivative(pot, X, T: float, dT: float,
                                       include_decoupled: bool = True) -> float:
    """``d(energy density)/dT`` at ``(X, T)``, by central difference of its
    temperature-dependent part only.

    ``pot.energyDensity`` is a temperature-independent vacuum offset (the
    potential difference to the broken vacuum ``X0``) plus the daisy and
    radiation terms. Differencing the total subtracts two numbers dominated by
    that offset, and in a strongly supercooled transition the thermal part
    falls below double precision against it: at ``T/v ~ 5e-5`` the difference
    is exactly zero and ``alpha_thetabar`` becomes infinite, at ``T/v ~ 5e-4``
    only about two significant digits survive. The offset does not depend on
    ``T``, so leaving it out changes nothing mathematically and removes the
    cancellation.
    """
    return (hotEnergyDensity(pot, X, T + dT, include_decoupled)
            - hotEnergyDensity(pot, X, T - dT, include_decoupled)) / (2 * dT)


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
    # A decoupled bath is field independent and cancels between the phases, so leaving it
    # out changes nothing analytically and removes a cancellation the step is not sized for.
    DeltaV = np.abs(pot.Vtot(high_phi, T, include_decoupled=False)
                    - pot.Vtot(low_phi, T, include_decoupled=False))

    # Derivative of the potential with respect to T
    # One step per phase: the broken phase carries a large vacuum offset that the stencil
    # has to difference away, the symmetric phase does not, so a step that is accurate for
    # one is not for the other. The difference of the two potentials has to cancel the
    # larger of the two offsets, so it takes the larger step.
    dT_sym = temperatureDerivativeStep(pot, T, high_phi)
    dT_bro = temperatureDerivativeStep(pot, T, low_phi)
    # dDeltaVdT below is a two-point central difference, not one of the five-point
    # stencils, so it balances at the cube root of the same ratio and takes a much
    # smaller step; the five-point step would cost 2e-4 of truncation error here.
    dT = max(temperatureDerivativeStep(pot, T, high_phi, two_point=True),
             temperatureDerivativeStep(pot, T, low_phi, two_point=True))
    dDeltaV_p = np.abs(pot.Vtot(high_phi, T + dT / 2, include_decoupled=False)
                       - pot.Vtot(low_phi, T + dT / 2, include_decoupled=False))
    dDeltaV_m = np.abs(pot.Vtot(high_phi, T - dT / 2, include_decoupled=False)
                       - pot.Vtot(low_phi, T - dT / 2, include_decoupled=False))
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
    # The pseudo-trace is defined through the sound speed, so it does not exist where there
    # is no plasma to carry sound. At extreme supercooling the thermal part of the potential
    # underflows in the broken phase, dV/dT and d2V/dT2 are both zero and the sound speed
    # comes back as 0 rather than as a number; dividing by it raised ZeroDivisionError.
    # The bag-model strengths below need no sound speed and stay available. The two sound
    # speeds are checked separately because they enter in different places: only the broken
    # one builds the pseudo-trace, while the symmetric one appears solely in the denominator
    # of alpha_thetabar. A usable broken-phase value therefore still gives alpha_hyd.
    broken_usable = np.isfinite(csSq_bro) and csSq_bro > 0.0
    sym_usable = np.isfinite(csSq_sym) and csSq_sym > 0.0
    if not broken_usable:
        if verbose:
            print(
                "No usable broken-phase sound speed for the pseudo-trace "
                f"(c_s,bro^2 = {csSq_bro}); alpha_thetabar and alpha_hyd are set to nan."
            )
        theta_sym = theta_bro = np.nan
    else:
        theta_sym = -T*pot.dVdT(high_phi, T, dT=dT_sym, include_decoupled=False) + Veff_sym * (1 + 1/ csSq_bro)
        theta_bro = -T*pot.dVdT(low_phi, T, dT=dT_bro, include_decoupled=False) + Veff_bro * (1 + 1/ csSq_bro)
    # This energy density keeps the decoupled bath, so its step is sized with it too. That
    # the sound speed beside it in the denominator is computed without the bath is the
    # separate question of which plasma 3 w refers to; it is not decided here. The
    # temperature-independent vacuum offset is left out of the difference, which removes
    # the cancellation that the step alone cannot at extreme supercooling.
    dT_e = temperatureDerivativeStep(pot, T, high_phi, include_decoupled=True,
                                     two_point=True)
    dedT = energyDensityTemperatureDerivative(pot, high_phi, T, dT_e)
    # The vacuum offset is out of the difference; what can still fail is a plasma with no temperature dependence left,
    # a phase with nothing in it rather than a loss of precision.
    if sym_usable:
        requireUsableEnthalpy(3 * csSq_sym * T * dedT, T, {"de/dT": dedT, "dT": dT_e})
    if broken_usable and sym_usable:
        alpha_thetabar = (theta_sym - theta_bro) / (3* csSq_sym * T * dedT)
    else:
        if broken_usable and verbose:
            print(
                "No usable symmetric-phase sound speed "
                f"(c_s,sym^2 = {csSq_sym}); alpha_thetabar is set to nan, "
                "the hydrodynamic strengths are unaffected."
            )
        alpha_thetabar = np.nan

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
    w_sym_PT = requireUsableEnthalpy(symmetricPhaseEnthalpy(pot, high_phi, T, include_decoupled=False), T,
                                     {"plasma": "the transitioning sector alone"})

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
    w_sym_sw = requireUsableEnthalpy(
        symmetricPhaseEnthalpy(pot, high_phi, T, include_decoupled=include_decoupled_in_enthalpy), T,
        {"plasma": "with the decoupled bath" if include_decoupled_in_enthalpy else "the transitioning sector alone"})
    alpha_hyd = (theta_sym - theta_bro) / (3 * w_sym_sw)

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

    The false vacuum occupies a physical volume :math:`\mathcal{V} \propto a^3 e^{-I}`, which
    has to shrink for the transition to complete rather than inflate forever. The requirement
    goes back to Turner, Weinberg and Widrow, Phys. Rev. D 46 (1992) 2384; the temperature form
    is eq. (2.26) of Ellis, Lewicki and No, arXiv:1809.08242,

    .. math::
        \frac{1}{\mathcal{V}}\frac{{\rm d}\mathcal{V}}{{\rm d}t}
        = H\left(3 + T\frac{{\rm d}I}{{\rm d}T}\right) < 0 \,,

    which assumes :math:`{\rm d}T/{\rm d}t = -HT`. The percolation history here is integrated
    with :math:`{\rm d}T/{\rm d}t = -3 c_{\rm s}^2 H T`, for which the rate is
    :math:`3H(1 + c_{\rm s}^2 T\,{\rm d}I/{\rm d}T)`, and the returned quantity is the
    bracket. At :math:`c_{\rm s}^2 = 1/3` it is the published criterion divided by three, so
    the sign is unchanged. The derivative is taken on a monotonic spline of :math:`\ln I`
    rather than of :math:`P`, because :math:`I` spans decades over the sampled range while
    :math:`P` saturates.

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
        log_integral = PchipInterpolator(
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
    entropy_definition: str | None = None,
) -> float:
    """Sound speed squared of the time-temperature relation used for the percolation history.

    Mirrors ``percIntegralODE_full_sweep``: the ``bag`` mode integrates with
    ``dT/dt = -H T``, i.e. ``c_s^2 = 1/3``; ``sound_speed`` mode uses the symmetric-phase
    value from ``_time_temperature_factors``, including its fallback to ``1/3``. The
    integral method no longer enters, both integrate the same relation.
    """
    if not percolation_uses_sound_speed(time_temperature_mode):
        return 1.0 / 3.0
    sound_speed_sq, _ = _time_temperature_factors(
        pot, phase_symmetric, np.array([float(T)]), time_temperature_mode, entropy_definition
    )
    if sound_speed_sq is None:
        return 1.0 / 3.0
    return float(sound_speed_sq[0])


def percolation_uses_sound_speed(time_temperature_mode: str | None) -> bool:
    """Whether the percolation history uses the sound speed of the plasma.

    Both integral methods do in ``sound_speed`` mode (the default); the ``bag`` mode
    integrates with ``dT/dt = -H T``, i.e. ``c_s^2 = 1/3``. Every quantity read off that
    history (``Tperc``, the mean bubble separation, ``beta/H``) must use the same relation.
    """
    mode = "sound_speed" if time_temperature_mode is None else str(time_temperature_mode)
    if mode not in {"sound_speed", "bag"}:
        raise errors.PercolationError(
            f"Unknown percolation time-temperature mode {mode!r}. "
            "Supported modes are 'sound_speed' and 'bag'."
        )
    return mode == "sound_speed"


def expansion_interpolants(pot, phase_symmetric, T, *, time_temperature_mode: str | None = None,
                           entropy_definition: str | None = None):
    """Interpolants of the expansion history for ``calcMeanBubbleSeparation``.

    Built from the same ``_time_temperature_factors`` as the percolation integral, on the
    temperatures ``T`` of the percolation support, so that the mean bubble separation uses
    the history that fixed ``Tperc``. Adiabatic expansion gives
    ``d ln a / dT = -1/(3 c_s^2 T)``, hence ``dt = -dT / (3 c_s^2 H T)`` and an entropy
    density ``s ~ a^-3``.

    Returns
    -------
    tuple
        ``(logEntropyInt, coolingInt)`` with ``logEntropyInt(T) = ln s(T) = -3 ln a(T)``
        up to an additive constant, and ``coolingInt(T) = 3 c_s^2(T)``; or ``(None, None)``
        when the history is the bag limit, ``s ~ T^3`` and ``3 c_s^2 = 1``.

        The logarithm is handed over rather than ``s`` itself because
        ``_time_temperature_factors`` allows ``ln a`` up to 700, where ``s ~ a^-3``
        underflows to zero and the ratio that the separation needs can no longer be
        recovered from it.
    """
    if not percolation_uses_sound_speed(time_temperature_mode):
        return None, None
    temperatures = np.unique(np.asarray(T, dtype=float))
    temperatures = temperatures[np.isfinite(temperatures) & (temperatures > 0.0)][::-1]
    if temperatures.size < 2:
        return None, None
    usable_points: list[int] = []
    sound_speed_sq, scale_factor = _time_temperature_factors(
        pot, phase_symmetric, temperatures, time_temperature_mode, entropy_definition,
        usable_out=usable_points,
    )
    if sound_speed_sq is None:
        return None, None
    if usable_points and usable_points[0] == 0:
        # Every temperature fell back, so these arrays hold the bag relation throughout and
        # describe no plasma. Interpolating them would give a history indistinguishable from
        # one that was computed, and a caller comparing two schemes would read agreement
        # where nothing was compared. The callers of this function treat `None` as "no
        # history" and fall back to a ~ 1/T, which is what these arrays already say. The
        # percolation integral reads `_time_temperature_factors` directly and is unaffected.
        return None, None
    ascending = temperatures[::-1]
    cs_sq_int = PchipInterpolator(ascending, sound_speed_sq[::-1], extrapolate=True)
    ln_a_int = PchipInterpolator(ascending, np.log(scale_factor[::-1]), extrapolate=True)

    def logEntropyInt(x):
        return -3.0 * ln_a_int(np.asarray(x, dtype=float))

    def coolingInt(x):
        return 3.0 * cs_sq_int(np.asarray(x, dtype=float))

    return logEntropyInt, coolingInt


ENTROPY_SCHEME_DEFAULT_THRESHOLD = 0.04
"""Default for ``percolation_entropy_scheme_warn_threshold``; see ``config.PercolationConf``."""

def validate_entropy_definition(entropy_definition) -> str:
    """Return the entropy definition to use, refusing anything unknown.

    Only ``None`` means "not given": an empty string is a value, and a wrong one.
    """
    configured = (ENTROPY_DEFINITIONS[0] if entropy_definition is None
                  else str(entropy_definition))
    if configured not in ENTROPY_DEFINITIONS:
        raise errors.PercolationError(
            f"Unknown percolation entropy definition {configured!r}. "
            f"Supported definitions are {ENTROPY_DEFINITIONS[0]!r} and "
            f"{ENTROPY_DEFINITIONS[1]!r}."
        )
    return configured


def entropy_scheme_threshold(conf) -> float:
    """The spread above which a point is flagged, for a percolation configuration.

    A configuration built before the setting existed does not carry it, and must then get
    the documented default rather than a second number written out at the call site.

    The value is validated here and not only in the run-time overrides, because a model is
    free to assign `config.percolationConf` attributes directly and most of them do. A
    threshold of zero or below would flag every point that has any spread at all, which is
    not "flag nothing"; a run that should flag nothing switches the comparison off with
    `entropy_scheme_diagnostic`.
    """
    threshold = float(getattr(conf, "entropy_scheme_warn_threshold",
                              ENTROPY_SCHEME_DEFAULT_THRESHOLD))
    if not threshold > 0.0:
        raise errors.PercolationError(
            f"percolation_entropy_scheme_warn_threshold={threshold!r} must be positive. A run "
            "that should not flag anything switches the comparison off with "
            "percolation_entropy_scheme_diagnostic instead."
        )
    return threshold


def entropy_scheme_is_sensitive(cs_sq_spread, threshold) -> bool:
    """Whether a point's expansion history depends appreciably on the entropy scheme.

    The criterion is the spread of ``3 c_s^2`` between the two schemes and never the gap in
    ``ln a``: the spread estimates the relative shift the mean bubble separation takes
    between them to about thirty per cent, while the gap is biased by a factor that depends
    on the model. A spread that could not be computed is not a clean point, so it leaves the
    flag unset and the ``nan`` visible beside it rather than raising anything.
    """
    spread = float(cs_sq_spread)
    return bool(np.isfinite(spread) and spread > float(threshold))


def record_entropy_scheme_diagnostics(
    derived: dict,
    pot,
    phase_symmetric,
    T,
    conf,
    configured_history=None,
    verbose: bool = False,
    console=None,
) -> None:
    """Write the entropy-scheme diagnostics and flag into ``derived``.

    All three keys are set on every path. Where the comparison does not apply, the flag is
    ``False`` and both numbers are ``nan``: the keys are registered for every run, so leaving
    them unset would have the writer report them as unimplemented and fill all three with
    ``nan``, the boolean included.

    A timeout reaches the caller. Any other failure of the comparison leaves the ``nan`` and
    must not invalidate the point, which was computed before the comparison was asked for.

    """
    derived.setdefault("WARNING:entropy_scheme_sensitive", False)
    derived.setdefault("DIAG:entropy_scheme_cs2_spread", float("nan"))
    derived.setdefault("DIAG:entropy_scheme_lna_gap", float("nan"))

    # Both settings are validated before anything can return early, for the same reason the
    # definition is validated before the exits inside `entropy_scheme_sensitivity`: a bad
    # value is a configuration error whether or not this point would have been compared, and
    # these exits are not the place to discover that it was ignored.
    threshold = entropy_scheme_threshold(conf)
    validate_entropy_definition(getattr(conf, "entropy_definition", None))

    history_available = not (configured_history is None
                             or configured_history[0] is None)
    if not (history_available and getattr(conf, "entropy_scheme_diagnostic", True)):
        return

    try:
        spread, shift = entropy_scheme_sensitivity(
            pot,
            phase_symmetric,
            T,
            time_temperature_mode=getattr(conf, "time_temperature_mode", None),
            entropy_definition=getattr(conf, "entropy_definition", None),
            configured_history=configured_history,
        )
    except errors.Timeout:
        raise
    except errors.PercolationError:
        # A configuration error, not a solver failure: absorbing it would answer a bad
        # setting with "the two agree here". The validation above already refuses the cases
        # that arise today, so this guards a configuration error raised deeper in the
        # comparison, which the ordinary-failure handler below would otherwise hide.
        raise
    except Exception as err:
        spread = shift = float("nan")
        if verbose:
            print("Error in comparing the entropy schemes: ", err)

    derived["DIAG:entropy_scheme_cs2_spread"] = float(spread)
    derived["DIAG:entropy_scheme_lna_gap"] = float(shift)
    # The flag is raised on the median spread of 3 c_s^2 over the support, which estimates
    # the relative shift the mean bubble separation takes between the schemes to about 30%.
    # The gap in ln a is recorded beside it but is not the criterion: it is the linearised
    # estimate of the same shift and is biased by a model-dependent factor that varies by two
    # orders of magnitude between the models this was checked on.
    derived["WARNING:entropy_scheme_sensitive"] = entropy_scheme_is_sensitive(
        spread, threshold)
    if derived["WARNING:entropy_scheme_sensitive"] and verbose and console is not None:
        console.print(
            "[bold yellow]WARNING:[/bold yellow] the two entropy schemes differ by "
            f"{spread * 100.0:.2f} % in 3 c_s^2 in the median over this support, above the "
            f"{threshold * 100.0:.2f} % threshold, so the mean bubble separation of this "
            "point moves by roughly that much between them. Say which "
            "percolation_entropy_definition produced it."
        )


# The potential's entropy takes three derivatives per temperature, and the comparison is a median over the support, which
# needs no more temperatures than this. Both ends always stay.
_ENTROPY_SCHEME_POINTS = 24


def entropy_scheme_sensitivity(
    pot,
    phase_symmetric,
    T: np.ndarray,
    *,
    time_temperature_mode: str | None = None,
    entropy_definition: str | None = None,
    configured_history=None,
):
    """How much the choice of entropy scheme matters on this percolation support.

    The two schemes of ``ENTROPY_DEFINITIONS`` are two approximations to one entropy density
    and neither contains the other, so the difference between them is an uncertainty of the
    expansion history rather than a numerical error. It is reported and not corrected, because
    nothing in the literature chooses between them.

    Two numbers, both on the support that fixed the percolation temperature. ``cs_sq_spread``
    is the median relative difference between the two schemes' ``3 c_s^2``, which over 91
    points of four scan lines estimated the shift in the mean bubble separation to within a
    factor of two at 93 % of points. ``scale_factor_shift`` is the gap between the schemes in
    ``|delta ln a|`` across the support; since ``d ln a/d ln T = -1/(3 c_s^2)`` and
    ``ln a = -(1/3) ln s`` up to a constant, it is the difference of the two logarithmic
    entropy ratios over three, and the first-order estimate of the same shift.

    Pass ``configured_history`` as the ``(logEntropyInt, coolingInt)`` pair already built for
    the configured scheme and only the other scheme is evaluated: one further history rather
    than two, the comparison made against the history the observables came from, and half the
    cost, since one history is three ``entropy_density`` evaluations per support point.

    Returns
    -------
    tuple
        ``(cs_sq_spread, scale_factor_shift)``, or ``(nan, nan)`` where the comparison
        cannot be made: in bag mode, on a support of fewer than two temperatures, or where
        one of the schemes has no usable history.
    """
    # Validated before anything can return early: a misspelling is a configuration error
    # whether or not this support would have been compared.
    configured = validate_entropy_definition(entropy_definition)
    other = [d for d in ENTROPY_DEFINITIONS if d != configured][0]

    if not percolation_uses_sound_speed(time_temperature_mode):
        # In bag mode the history is 3 c_s^2 = 1 by construction and reads no entropy.
        return float("nan"), float("nan")

    temperatures = np.asarray(T, dtype=float)
    if temperatures.size < 2:
        return float("nan"), float("nan")
    if temperatures.size > _ENTROPY_SCHEME_POINTS:
        temperatures = temperatures[np.unique(np.round(
            np.linspace(0, temperatures.size - 1, _ENTROPY_SCHEME_POINTS)).astype(np.int64))]

    histories = {}
    if configured_history is not None:
        # The caller has the configured history already: it is the one the observables were
        # computed from, and rebuilding it would double the cost of the comparison and risk
        # comparing against a history that is not the one that produced `Tperc`.
        log_entropy, cooling = configured_history
        if log_entropy is None or cooling is None:
            return float("nan"), float("nan")
        histories[configured] = (log_entropy, cooling)
    needed = [d for d in (configured, other) if d not in histories]
    for definition in needed:
        log_entropy, cooling = expansion_interpolants(
            pot,
            phase_symmetric,
            temperatures,
            time_temperature_mode=time_temperature_mode,
            entropy_definition=definition,
        )
        if log_entropy is None or cooling is None:
            return float("nan"), float("nan")
        histories[definition] = (log_entropy, cooling)

    hot, cold = float(temperatures[0]), float(temperatures[-1])
    ratios, coolings = {}, {}
    for definition, (log_entropy, cooling) in histories.items():
        ratios[definition] = float(log_entropy(hot) - log_entropy(cold))
        coolings[definition] = np.array(
            [float(cooling(float(t))) for t in temperatures], dtype=float)

    a = coolings[configured]
    b = coolings[other]
    usable = np.isfinite(a) & np.isfinite(b) & ((a + b) != 0.0)
    if not np.any(usable):
        return float("nan"), float("nan")

    # The median over the support, not the largest value. The shift in the mean bubble
    # separation integrates the difference over the whole history, so a typical value
    # predicts it and an extreme one does not: measured against the shift it produces, the
    # median is unbiased to 6% while the maximum overstates it by a factor 1.8, and a single
    # support temperature that has lost a usable sound speed in one scheme can drive the
    # maximum to 100% where the separation moves by a few per cent.
    # Symmetric in the two schemes: the difference is taken relative to their mean, not to
    # the configured one. Dividing by the configured scheme would make the number, and with
    # it the flag, depend on which scheme the run happened to use: for factors 1 and 1.041
    # that is 4.10% one way and 3.94% the other, which straddles the default threshold.
    # For a difference d the two agree to O(d^2), so the calibration is unaffected at the
    # per cent level where it was measured.
    cs_sq_spread = float(np.median(
        2.0 * np.abs(b[usable] - a[usable]) / np.abs(a[usable] + b[usable])))
    scale_factor_shift = abs(ratios[configured] - ratios[other]) / 3.0
    if not np.isfinite(scale_factor_shift):
        return cs_sq_spread, float("nan")
    return cs_sq_spread, float(scale_factor_shift)


def calcMeanBubbleSeparation(
    T,
    Tmax,
    Sint,
    Pint,
    Hint,
    entropyInt=None,
    coolingInt=None,
    verbose=False,
    *,
    logEntropyInt=None,
):
    """Calculate the mean bubble separation at temperature T.

    Parameters
    ----------
    T : float
        Symmetric phase temperature
    Tmax : float
        Upper limit of the integral, usually the nucleation temperature
    Sint : scipy.interpolate.interp1d object
        The action
    Pint : scipy.interpolate.interp1d object
        The true vacuum fraction.
    Hint : scipy.interpolate.interp1d object
        The Hubble rate
    entropyInt : callable, optional
        Entropy density of the expansion history, ``entropyInt(T) ~ a(T)^-3``. Only its
        ratios enter, so any normalisation will do. ``None`` falls back to the bag
        relation ``s ~ T^3``, i.e. ``a ~ 1/T``. It is limited to histories of at most
        about 236 e-folds, beyond which ``a^-3`` underflows; use ``logEntropyInt`` there.
    logEntropyInt : callable, keyword only, optional
        ``ln s(T) = -3 ln a(T)``, up to an additive constant, in place of ``entropyInt``.
        The ratio is then formed as a difference of logarithms and stays exact over the
        whole range ``_time_temperature_factors`` allows. This is what
        ``expansion_interpolants`` returns. It takes precedence over ``entropyInt``.
    coolingInt : callable, optional
        ``coolingInt(T) = 3 c_s^2(T)``, the factor the integrand is *divided* by, since
        ``dt = -dT / (3 c_s^2 H T)``. It is ``3 c_s^2``, not ``1 / (3 c_s^2)``. ``None``
        falls back to the bag relation, ``3 c_s^2 = 1``.

        Both come from ``expansion_interpolants``, built from the same factors as the
        percolation integral, so that the separation uses the history that fixed
        ``Tperc``. With both ``None`` the integral is that of the bag limit.

    Returns
    ----------
    R : float
        Mean bubble separation."""

    # See eq. (5.42) in 2305.02357.
    if not (np.isfinite(T) and np.isfinite(Tmax)) or Tmax <= T:
        return np.inf
    # A grid uniform in ln T, whatever the expansion history: the nucleation history spans a few
    # e-folds of T above T, while Tmax can lie orders of magnitude higher, so that a linear grid
    # puts the whole history inside its first cell when the transition is supercooled.
    if T > 0.0 and Tmax > 0.0 and Tmax / T > 1.0 + 1e-10:
        Tr = np.geomspace(T, Tmax, 10_000)
    else:
        Tr = np.linspace(T, Tmax, 10_000)
    if logEntropyInt is not None:
        # s(T)/s(T') as a difference of logarithms: T' >= T and a falls with rising T, so
        # the exponent is never positive and deep histories underflow to zero, which is
        # the right limit. Clamping s itself instead would make both operands equal and
        # silently return an unexpanding universe.
        log_ratio = float(logEntropyInt(T)) - np.asarray(logEntropyInt(Tr), dtype=float)
        entropy_ratio = np.exp(np.minimum(log_ratio, 0.0))
    elif entropyInt is None:
        entropy_ratio = (T / Tr) ** 3
    else:
        entropy_T = float(entropyInt(T))
        entropy_Tr = np.asarray(entropyInt(Tr), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_ratio = entropy_T / entropy_Tr
        entropy_ratio = np.nan_to_num(entropy_ratio, nan=0.0, posinf=0.0, neginf=0.0)
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
    Pint : scipy.interpolate.interp1d object
        The true vacuum fraction.
    pot : generic_potential
        The effective potential. Only ``pot.config.percolationConf.f_final`` is read.
    verbose : bool
        Print the error message if the solve for the final temperature fails, just before
        :class:`PercolationError` is raised. Nothing is printed on success.
    Returns
    ----------
    Tf : float
        Final temperature."""

    f_final = pot.config.percolationConf.f_final  # true vacuum fraction at final temperature

    # Step 1: Check if the true vacuum fraction at Tlow is >= f_final
    if Pint(Tlow) < f_final:
        # It looks like the transition cannot completely finish
        # Report the percolation temperature with the refusal. Pint(Tlow) here is routinely
        # well above f_perc, i.e. the transition percolated and then stalled, so Tperc is
        # already determined. The scan output records the error code alone, so the message is
        # what survives, and without Tperc in it the only way to recover the number is to run
        # the point again.
        pint_max = float(Pint(Tlow))
        msg = (
            "The transition cannot completely finish, Pint(Tlow) < f_final: "
            + str(pint_max) + " < " + str(f_final)
            + ". The percolation temperature of this profile was already determined: "
            + "Tperc = " + str(Tperc)
            + ", and Pint reached " + str(pint_max) + " at Tlow = " + str(Tlow)
            + " (both temperatures in the solver's internal units; multiply by the model's"
            " conversionFactor for GeV), so the percolation observables are defined"
            " while T_final is not."
            " A transition stalling below f_final leaves a phase mixture, so Treh and the"
            " spectrum are not ordinary predictions here."
        )
        raise errors.EternalInflationError(
            msg, Pint_max=pint_max, Tperc=Tperc, Tlow=Tlow, f_final=f_final)

    # Step 2: Find Tf by solving Pint(Tf) = f_final
    try:
        Tf = optimize.brentq(lambda TSYM: Pint(TSYM) - f_final, Tlow, Tperc)
    except errors.Timeout:
        raise
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
