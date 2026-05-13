"""Grid-building helpers for the adaptive_step_size solver.

This module owns the self-contained support-point and candidate-grid
construction logic used by
:mod:`transitionlistener.percolation_adaptivestepsize`. It takes numpy
arrays and settings in and produces candidate temperature grids; the
top-level scan controllers (and their :class:`PercolationState`
bookkeeping) stay in the parent module.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from dataclasses import dataclass
import math
import numpy as np

from transitionlistener import errors
from transitionlistener.bubbledynamics import (
    PercolationGrid,
    _temperature_grid,
)
from transitionlistener.percolation_adaptive_rate import (
    HOT_PROBABILITY_ANCHORS,
    HOT_RATE_ACCEPT_LOG,
    _build_peak_centered_step1_window,
    _build_step1_hot_probability_anchor_grid,
    _build_step1_hot_rate_anchor_grid,
    _find_value_crossing,
    _hot_head_underresolved,
    _log_gamma_h4_array,
    _probability_anchor_presence_tolerance,
    _probability_at,
    _rate_interval_is_post_peak_inert,
    _remesh_dp_limit,
    _temperature_midpoint,
    _temperature_present,
)

# Fixed probability anchors used by the simplified step-2/step-3 remeshing
# controller.  The idea is to approximate a near-uniform mesh in the physically
# interesting part of P(T) without requiring the caller to micromanage several
# different refinement heuristics.  Anchors are only activated once the current
# profile actually reaches the corresponding probability.
_PROBABILITY_REMESH_ANCHORS = (
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    2.0e-1,
    3.5e-1,
    5.0e-1,
    6.5e-1,
    8.0e-1,
    9.0e-1,
    9.5e-1,
    9.8e-1,
    9.9e-1,
)

_STARTUP_WINDOW_HIGH = 0.999
_COLD_TAIL_TARGET_DP = 0.08
_COLD_TAIL_TARGET_DI = 0.25
_COLD_TAIL_INTEGRAL_SWITCH_P = 0.8
_COLD_TAIL_MIN_LOG_STEP = 0.25
_COLD_TAIL_MAX_LOG_STEP = 5.0
_COLD_TAIL_STEP_SHRINK_LIMIT = 0.35
_COLD_TAIL_STEP_GROWTH_LIMIT = 3.0
_COLD_TAIL_ZERO_GAIN_SCALE = 2.0
_COLD_TAIL_POST_PEAK_MIN_STREAK = 2
_COLD_TAIL_POST_PEAK_BOOST_COEFF = 0.25
_COLD_TAIL_POST_PEAK_MAX_BOOST = 3.0
_STRICT_JUMP_REFINE_THRESHOLD = 0.05
_STRICT_JUMP_SUCCESS_THRESHOLD = 0.05
_STRICT_JUMP_P_MAX = 0.2
_STRICT_JUMP_APPLY_GLOBALLY = False


def _tb_root_bracketed_for_tsym(
    TSYM: float,
    TBROmin: float,
    TBROmax: float,
    phase_symmetric,
    phase_broken,
    pot,
) -> bool:
    """Return whether the reheating criterion changes sign across ``[TBROmin, TBROmax]``."""
    from transitionlistener import bubbledynamics as bd

    crit_min = float(bd.Tb_criterion(TBROmin, TSYM, phase_symmetric, phase_broken, pot))
    crit_max = float(bd.Tb_criterion(TBROmax, TSYM, phase_symmetric, phase_broken, pot))
    if not (np.isfinite(crit_min) and np.isfinite(crit_max)):
        return False
    if crit_min == 0.0 or crit_max == 0.0:
        return True
    return bool(crit_min * crit_max < 0.0)


def _highest_tsym_with_tb_root(
    tmin: float,
    tmax: float,
    TBROmin: float,
    TBROmax: float,
    phase_symmetric,
    phase_broken,
    pot,
) -> float:
    """Find the hottest traced temperature that still admits a bracketed reheating root."""
    tmin = float(tmin)
    tmax = float(tmax)
    if _tb_root_bracketed_for_tsym(tmax, TBROmin, TBROmax, phase_symmetric, phase_broken, pot):
        return tmax
    if not _tb_root_bracketed_for_tsym(tmin, TBROmin, TBROmax, phase_symmetric, phase_broken, pot):
        raise errors.PercolationError(
            "Tb_criterion has no bracketed reheating root anywhere inside the traced "
            f"phase-overlap interval [{tmin:.8g}, {tmax:.8g}]."
        )

    low = tmin
    high = tmax
    scale = max(abs(tmin), abs(tmax), abs(TBROmin), abs(TBROmax), 1.0)
    tol = 1e-8 * scale
    for _ in range(80):
        if abs(high - low) <= tol:
            break
        mid = 0.5 * (low + high)
        if _tb_root_bracketed_for_tsym(mid, TBROmin, TBROmax, phase_symmetric, phase_broken, pot):
            low = mid
        else:
            high = mid
    return float(low)


def _next_untried_scout_temperature(
    high: float,
    low: float,
    tried: np.ndarray,
    *,
    prefer: str,
) -> float | None:
    """Bisect an interval until a temperature not already present in ``tried`` is found."""
    high = float(high)
    low = float(low)
    tried = np.asarray(tried, dtype=float)
    if not (np.isfinite(high) and np.isfinite(low)) or high <= low:
        return None

    current_high = high
    current_low = low
    for _ in range(32):
        candidate = _temperature_midpoint(current_high, current_low)
        if candidate is None or not np.isfinite(float(candidate)):
            return None
        if not _temperature_present(tried, float(candidate)):
            return float(candidate)
        if prefer == "hotter":
            current_low = float(candidate)
        else:
            current_high = float(candidate)
        if current_high <= current_low:
            return None
    return None


@dataclass
class _Step1ApproxProfile:
    """Approximate percolation profile reconstructed from cached step-1 actions."""

    sampled_temperatures: np.ndarray
    temperatures: np.ndarray
    actions: np.ndarray
    gamma: np.ndarray
    hubble_sym: np.ndarray
    log_gamma_h4_sym: np.ndarray
    i_approx: np.ndarray
    probabilities: np.ndarray
    tperc_approx: float
    startup_center: float
    window_high: float | None
    window_low: float | None
    window_source: str
    peak_temperature: float | None = None
    peak_log_gamma_h4: float | None = None


def _build_step1_approx_profile(
    outdict: dict,
    pot,
    phase_symmetric,
    phase_broken,
    vw: float,
    tmin: float,
    tmax: float,
    tperc_approx: float,
    settings,
    sampled_temperatures: np.ndarray | None = None,
) -> _Step1ApproxProfile | None:
    """Reconstruct an approximate percolation profile from cached action temperatures."""
    from transitionlistener import bubbledynamics as bd

    cached_temperatures = bd._transition_action_outdict_temperatures(
        outdict,
        phase_symmetric,
        phase_broken,
        tmin=tmin,
        tmax=tmax,
    )
    all_sampled_temperatures = _temperature_grid(
        np.concatenate(
            (
                np.asarray(cached_temperatures, dtype=float),
                np.asarray([] if sampled_temperatures is None else sampled_temperatures, dtype=float),
            )
        )
    )
    if cached_temperatures.size < 2:
        return None

    actions = np.asarray(
        [float(bd.calcAction(pot, float(temp), phase_symmetric, phase_broken, outdict)) for temp in cached_temperatures],
        dtype=float,
    )
    mask = np.isfinite(cached_temperatures) & np.isfinite(actions)
    temperatures = np.asarray(cached_temperatures[mask], dtype=float)
    actions = np.asarray(actions[mask], dtype=float)
    if temperatures.size < 2:
        return None

    hubble_sym = np.asarray(
        [
            float(bd.HubbleParameter(bd.energyDensity(pot, phase_symmetric, float(temp)), pot.conversionFactor))
            for temp in temperatures
        ],
        dtype=float,
    )
    gamma = np.asarray(bd.Gamma(temperatures, actions), dtype=float)
    log_gamma_h4_sym = np.asarray(bd.logGamma(temperatures, actions), dtype=float) - 4.0 * np.log(hubble_sym)
    if not (np.all(np.isfinite(hubble_sym)) and np.all(np.isfinite(gamma)) and np.all(np.isfinite(log_gamma_h4_sym))):
        return None

    i_approx = np.zeros_like(temperatures, dtype=float)
    probabilities = np.zeros_like(temperatures, dtype=float)
    i_approx_vals, probabilities[:] = bd.percIntegralODE_full_sweep(
        temperatures,
        hubble_sym,
        actions,
        vw=vw,
        pot=pot,
        phase_symmetric=phase_symmetric,
        time_temperature_mode=settings.time_temperature_mode,
        integral_method=settings.integral_method,
    )
    i_approx[:] = i_approx_vals

    window_high, _, _ = _find_value_crossing(
        temperatures,
        probabilities,
        float(settings.f_start),
    )
    window_low, _, _ = _find_value_crossing(
        temperatures,
        probabilities,
        _STARTUP_WINDOW_HIGH,
    )
    startup_center = float(tperc_approx)
    window_source = "probability"
    peak_temperature = None
    peak_log_gamma_h4 = None

    if window_high is None or window_low is None:
        peak_temperature, peak_log_gamma_h4, peak_window_high, peak_window_low = _build_peak_centered_step1_window(
            temperatures,
            log_gamma_h4_sym,
            settings,
        )
        if peak_window_high is not None and peak_window_low is not None:
            window_high = float(peak_window_high)
            window_low = float(peak_window_low)
            startup_center = float(peak_temperature)
            window_source = "peak_log_gamma_h4"

    return _Step1ApproxProfile(
        sampled_temperatures=all_sampled_temperatures,
        temperatures=temperatures,
        actions=actions,
        gamma=gamma,
        hubble_sym=hubble_sym,
        log_gamma_h4_sym=log_gamma_h4_sym,
        i_approx=i_approx,
        probabilities=probabilities,
        tperc_approx=float(tperc_approx),
        startup_center=float(startup_center),
        window_high=None if window_high is None else float(window_high),
        window_low=None if window_low is None else float(window_low),
        window_source=str(window_source),
        peak_temperature=None if peak_temperature is None else float(peak_temperature),
        peak_log_gamma_h4=None if peak_log_gamma_h4 is None else float(peak_log_gamma_h4),
    )


def _scout_needs_hot_extension(profile: _Step1ApproxProfile | None, settings) -> bool:
    """Decide whether the approximate step-1 profile must be extended to hotter temperatures."""
    if profile is None:
        return True
    if profile.probabilities.size == 0:
        return True
    if float(profile.probabilities[0]) > float(settings.f_start):
        return True
    hot_head_unresolved, _, _ = _hot_head_underresolved(
        profile.temperatures,
        profile.probabilities,
        profile.actions,
        profile.hubble_sym,
        settings,
        for_acceptance=False,
    )
    if hot_head_unresolved:
        return True
    if profile.window_high is not None and profile.window_low is not None:
        return False
    return profile.window_high is None


def _scout_needs_cold_extension(profile: _Step1ApproxProfile | None, settings) -> bool:
    """Decide whether the approximate step-1 profile must be extended to colder temperatures."""
    if profile is None:
        return True
    if profile.window_high is not None and profile.window_low is not None:
        return False
    if profile.probabilities.size == 0:
        return True
    if float(profile.probabilities[-1]) < float(settings.f_final):
        return True
    return profile.window_low is None


def _remaining_scout_action_budget(
    outdict: dict,
    phase_symmetric,
    phase_broken,
    settings,
    *,
    tmin: float,
    tmax: float,
) -> int:
    """Return how many more unique transition action temperatures may be added."""
    from transitionlistener import bubbledynamics as bd

    used = int(
        len(
            bd._transition_action_outdict_temperatures(
                outdict,
                phase_symmetric,
                phase_broken,
                tmin=tmin,
                tmax=tmax,
            )
        )
    )
    return max(int(settings.max_action_temperatures) - int(used), 0)


def _extend_step1_scout_until_bracketed(
    outdict: dict,
    pot,
    phase_symmetric,
    phase_broken,
    vw: float,
    hot_grid_start: float,
    tmin: float,
    tmax: float,
    tperc_approx: float,
    settings,
    verbose: bool,
) -> _Step1ApproxProfile | None:
    """Enlarge the step-1 scout bank until the approximate transition window is bracketed."""
    from transitionlistener import bubbledynamics as bd

    attempted_scout_temperatures = np.asarray([], dtype=float)

    profile = _build_step1_approx_profile(
        outdict,
        pot,
        phase_symmetric,
        phase_broken,
        vw,
        tmin,
        tmax,
        tperc_approx,
        settings,
        sampled_temperatures=attempted_scout_temperatures,
    )
    max_extensions = max(int(settings.maxit), 1) * 4
    for _ in range(max_extensions):
        need_hot = _scout_needs_hot_extension(profile, settings)
        need_cold = _scout_needs_cold_extension(profile, settings)
        if not need_hot and not need_cold:
            return profile

        remaining_budget = _remaining_scout_action_budget(
            outdict,
            phase_symmetric,
            phase_broken,
            settings,
            tmin=tmin,
            tmax=tmax,
        )
        if remaining_budget <= 0:
            break

        all_cached = bd._transition_action_outdict_temperatures(
            outdict,
            phase_symmetric,
            phase_broken,
            tmin=tmin,
            tmax=tmax,
        )
        occupied_scout_points = _temperature_grid(
            np.concatenate((np.asarray(all_cached, dtype=float), attempted_scout_temperatures))
        )
        candidate_temperatures: list[float] = []
        if need_hot:
            hottest = float(occupied_scout_points[0]) if occupied_scout_points.size else (
                float(profile.temperatures[0]) if profile is not None and profile.temperatures.size else float(tperc_approx)
            )
            hot_scout_bound = max(float(hot_grid_start), float(tmax))
            scout_temp = _next_untried_scout_temperature(
                hot_scout_bound,
                hottest,
                occupied_scout_points,
                prefer="hotter",
            )
            if scout_temp is not None and scout_temp < hot_scout_bound and scout_temp > hottest:
                candidate_temperatures.append(float(scout_temp))
        if need_cold:
            coldest = float(occupied_scout_points[-1]) if occupied_scout_points.size else (
                float(profile.temperatures[-1]) if profile is not None and profile.temperatures.size else float(tperc_approx)
            )
            scout_temp = _next_untried_scout_temperature(
                coldest,
                float(tmin),
                occupied_scout_points,
                prefer="colder",
            )
            if scout_temp is not None and scout_temp < coldest and scout_temp > float(tmin):
                candidate_temperatures.append(float(scout_temp))

        if not candidate_temperatures:
            break
        unique_candidates = _temperature_grid(candidate_temperatures)
        attempted_scout_temperatures = _temperature_grid(
            np.concatenate((attempted_scout_temperatures, unique_candidates))
        )
        for temperature in unique_candidates[:remaining_budget]:
            if verbose:
                print(
                    "Step-1 scout extension:",
                    f"T = {float(temperature) * pot.conversionFactor:2.5g} GeV",
                )
            bd.calcAction(pot, float(temperature), phase_symmetric, phase_broken, outdict)

        profile = _build_step1_approx_profile(
            outdict,
            pot,
            phase_symmetric,
            phase_broken,
            vw,
            tmin,
            tmax,
            tperc_approx,
            settings,
            sampled_temperatures=attempted_scout_temperatures,
        )

    return profile


def _fill_largest_gap(points: np.ndarray, high: float, low: float) -> float | None:
    """Return the midpoint of the largest gap inside one ordered interval."""
    anchors = [float(high), float(low)]
    for value in np.asarray(points, dtype=float):
        if float(low) <= float(value) <= float(high):
            anchors.append(float(value))
    anchors = np.asarray(sorted(set(anchors), reverse=True), dtype=float)
    if anchors.size < 2:
        return None
    gaps = anchors[:-1] - anchors[1:]
    if gaps.size == 0:
        return None
    gap_index = int(np.argmax(gaps))
    upper = float(anchors[gap_index])
    lower = float(anchors[gap_index + 1])
    return _temperature_midpoint(upper, lower)


def _cached_promotion_bounds(active_grid: np.ndarray) -> tuple[float, float] | None:
    """Return a narrow interval around the current active grid for cache promotion.

    Dormant step-1 scout points remain available in the support bank, but only
    the cached temperatures close to the currently active transition band should
    be promoted automatically. Larger range changes are handled by the explicit
    range-expansion logic.
    """
    temps = _temperature_grid(active_grid)
    if temps.size == 0:
        return None
    high = float(temps[0])
    low = float(temps[-1])
    span = max(high - low, 0.0)
    mean_spacing = span / max(len(temps) - 1, 1)
    padding = max(0.15 * span, 1.5 * mean_spacing)
    return float(high + padding), float(low - padding)


def _build_step1_active_grid(
    profile: _Step1ApproxProfile,
    settings,
) -> np.ndarray | None:
    """Build the compact active TSYM grid from the approximate step-1 profile."""
    if profile.window_high is None or profile.window_low is None:
        return None

    temperatures = _temperature_grid(profile.temperatures)
    scale = max(
        abs(float(profile.window_high)),
        abs(float(profile.window_low)),
        abs(float(profile.startup_center)),
        1.0,
    )
    atol = 1e-10 * scale
    active = np.asarray(
        [
            float(value)
            for value in temperatures
            if float(profile.window_low) - atol <= float(value) <= float(profile.window_high) + atol
        ],
        dtype=float,
    )
    active = _temperature_grid(active)

    boundary_candidates = [float(profile.window_high), float(profile.window_low)]
    for candidate in boundary_candidates:
        if not np.any(np.isclose(active, candidate, atol=atol, rtol=0.0)):
            active = _temperature_grid(np.append(active, candidate))

    hot_probability_grid = _build_step1_hot_probability_anchor_grid(profile, settings, active)
    if hot_probability_grid is not None:
        for value in hot_probability_grid:
            active = _temperature_grid(np.append(active, float(value)))

    hot_rate_grid = _build_step1_hot_rate_anchor_grid(profile, settings, active)
    if hot_rate_grid is not None:
        for value in hot_rate_grid:
            active = _temperature_grid(np.append(active, float(value)))

    requested = max(int(settings.n_action_min), 2)
    n_hi = max(int(math.ceil(requested * float(settings.weight))), 1)
    n_lo = max(int(requested - n_hi), 1)

    def _count_high(points: np.ndarray) -> int:
        return int(np.count_nonzero(np.asarray(points, dtype=float) > float(profile.startup_center) + atol))

    def _count_low(points: np.ndarray) -> int:
        return int(np.count_nonzero(np.asarray(points, dtype=float) < float(profile.startup_center) - atol))

    while _count_high(active) < n_hi:
        midpoint = _fill_largest_gap(active, float(profile.window_high), float(profile.startup_center))
        if midpoint is None:
            break
        updated_active = _temperature_grid(np.append(active, midpoint))
        if updated_active.size == active.size:
            break
        active = updated_active

    while _count_low(active) < n_lo:
        midpoint = _fill_largest_gap(active, float(profile.startup_center), float(profile.window_low))
        if midpoint is None:
            break
        updated_active = _temperature_grid(np.append(active, midpoint))
        if updated_active.size == active.size:
            break
        active = updated_active

    return _temperature_grid(active)


def _initial_temperature_grid(
    outdict: dict,
    pot,
    Tstart: float,
    phase_symmetric,
    phase_broken,
    vw: float,
    nAction: int,
    settings,
    overlap_tmin: float,
    overlap_tmax: float,
    verbose: bool,
) -> PercolationGrid:
    """Step 1: build the initial TSYM grid around the approximate percolation point."""
    from transitionlistener import bubbledynamics as bd
    tmin = float(overlap_tmin)
    raw_tmax = float(overlap_tmax)
    hot_grid_start = _highest_tsym_with_tb_root(
        tmin,
        raw_tmax,
        float(phase_broken.Tmin),
        float(phase_broken.Tmax),
        phase_symmetric,
        phase_broken,
        pot,
    )
    tmax = raw_tmax
    if verbose and hot_grid_start < raw_tmax:
        print(
            "Initial active adaptive step size grid starts at the hottest TSYM that still "
            "brackets Tb_criterion, but hotter pure-symmetric support remains available:",
            f"{raw_tmax * pot.conversionFactor:2.5g} -> {hot_grid_start * pot.conversionFactor:2.5g} GeV",
        )

    if verbose:
        print("\nPercolation step 1: calculating Tperc using the saddlepoint approximation")
    try:
        TpercApprox = bd.calcApproxPercolation(
            outdict,
            pot,
            phase_symmetric,
            phase_broken,
            vw,
            verbose,
            tmin=overlap_tmin,
            tmax=overlap_tmax,
        )
    except errors.PercolationApproximation1Error as err:
        if verbose:
            print(
                "Percolation approximation 1 failed: ",
                err,
                ". We will continue with TpercApprox = hot_grid_start.",
            )
        TpercApprox = hot_grid_start
    if verbose:
        print(f"Approximate percolation temperature: {TpercApprox * pot.conversionFactor:2.5g} GeV")

    initial_n_action = int(nAction)
    grid_n = int(settings.n_action_min)

    scout_profile = _extend_step1_scout_until_bracketed(
        outdict,
        pot,
        phase_symmetric,
        phase_broken,
        vw,
        hot_grid_start,
        tmin,
        tmax,
        TpercApprox,
        settings,
        verbose,
    )
    if scout_profile is not None:
        active_grid = _build_step1_active_grid(scout_profile, settings)
        if active_grid is not None and active_grid.size >= 2:
            if verbose:
                print(
                    "Step-1 scout resolved an approximate active window:",
                    f"[{float(active_grid[-1]) * pot.conversionFactor:2.5g}, "
                    f"{float(active_grid[0]) * pot.conversionFactor:2.5g}] GeV with "
                    f"{len(active_grid)} active TSYM support points and "
                    f"{len(scout_profile.temperatures)} cached scout temperatures.",
                    f"(source: {scout_profile.window_source})",
                )
            return PercolationGrid(
                TSYM=active_grid,
                Tstart=float(active_grid[0]),
                TpercApprox=float(scout_profile.tperc_approx),
                tmin=tmin,
                tmax=tmax,
                TBROmin=phase_broken.Tmin,
                TBROmax=phase_broken.Tmax,
                free_support_bank=scout_profile.sampled_temperatures.copy(),
            )

    if verbose:
        print(
            "Warning: step-1 scout profile could not define any compact startup grid. "
            "Using the broad startup-grid heuristic."
        )

    dT = (hot_grid_start - TpercApprox) / int(initial_n_action * settings.weight)

    if TpercApprox < hot_grid_start:
        upper = np.maximum(TpercApprox - int(initial_n_action * (1 - settings.weight)) * dT, tmin)
        TSYM = np.linspace(hot_grid_start, upper, grid_n)
    else:
        if verbose:
            print("Warning: TpercApprox > overlap Tmax, use interval around TpercApprox")
        _, alpha_perc_approx, _, _, _, _, _ = bd.calcAlphas(
            TpercApprox,
            pot,
            phase_symmetric,
            phase_broken,
            verbose=verbose,
        )
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
        if Tmax <= Tmin:
            # If clipping the local interval around TpercApprox collapses or inverts
            # the bracket, fall back to a narrow valid window at the hot overlap edge.
            # This keeps the startup grid descending and lets step 2 decide whether
            # the transition history has to expand colder.
            Tmax = float(tmax)
            Tmin = max(float(tmin), float(tmax) * (1.0 - 1e-3))
        TSYM = np.linspace(Tmax, Tmin, grid_n)

    TSYM = _temperature_grid(TSYM)

    return PercolationGrid(
        TSYM=TSYM,
        Tstart=hot_grid_start,
        TpercApprox=TpercApprox,
        tmin=tmin,
        tmax=tmax,
        TBROmin=phase_broken.Tmin,
        TBROmax=phase_broken.Tmax,
        free_support_bank=None if scout_profile is None else scout_profile.sampled_temperatures.copy(),
    )


def _build_dynamiczoomwindow_jump_grid(
    jump_high: float,
    jump_low: float,
    *,
    max_new_points: int,
) -> np.ndarray | None:
    """Resolve a sharp P(T) jump by adding a few intermediate temperatures.

    Large late-time jumps often span several decades in temperature. In that
    regime a geometric ladder is more informative than a linear spacing, while
    modest intervals are still sampled linearly.
    """
    if max_new_points <= 0:
        return None
    jump_high = float(jump_high)
    jump_low = float(jump_low)
    if not (np.isfinite(jump_high) and np.isfinite(jump_low)) or jump_high <= jump_low:
        return None
    if jump_high / max(jump_low, 1e-30) - 1 <= 1e-10:
        return None
    if jump_low > 0.0 and jump_high / jump_low >= 2.0:
        refined = np.geomspace(jump_high, jump_low, int(max_new_points) + 2, dtype=float)[1:-1]
    else:
        refined = np.linspace(jump_high, jump_low, int(max_new_points) + 2, dtype=float)[1:-1]
    refined = _temperature_grid(refined)
    return refined if refined.size > 0 else None


def _largest_probability_jump_interval(
    temperatures: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_delta_p: float,
    p_max: float | None = None,
    apply_globally: bool = True,
    prefer_hottest: bool = False,
) -> tuple[float, float, float, float, float] | None:
    """Return the largest neighbouring interval with Delta P above the threshold.

    The return values are ``(jump_high, jump_low, probability_high,
    probability_low, delta_p)`` in the descending-temperature convention used by
    the support bank. Intervals that contain NaNs or non-descending
    temperatures are ignored. When ``apply_globally`` is false, only intervals
    touching ``P < p_max`` are considered.
    """
    temps = np.asarray(temperatures, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if temps.size < 2 or probs.size < 2:
        return None

    best_interval: tuple[float, float, float, float, float] | None = None
    best_delta_p = float(min_delta_p)
    for i in range(min(len(temps), len(probs)) - 1):
        jump_high = float(temps[i])
        jump_low = float(temps[i + 1])
        probability_high = float(probs[i])
        probability_low = float(probs[i + 1])
        if not (
            np.isfinite(jump_high)
            and np.isfinite(jump_low)
            and np.isfinite(probability_high)
            and np.isfinite(probability_low)
            and jump_high > jump_low
        ):
            continue
        if not apply_globally:
            if p_max is None or not np.isfinite(float(p_max)) or float(p_max) <= 0.0:
                continue
            if min(probability_high, probability_low) >= float(p_max):
                continue
        delta_p = abs(probability_low - probability_high)
        if delta_p <= float(min_delta_p):
            continue
        if prefer_hottest:
            return (
                jump_high,
                jump_low,
                probability_high,
                probability_low,
                float(delta_p),
            )
        if delta_p > best_delta_p:
            best_delta_p = float(delta_p)
            best_interval = (
                jump_high,
                jump_low,
                probability_high,
                probability_low,
                float(delta_p),
            )
    return best_interval


def _preferred_probability_jump_interval(
    temperatures: np.ndarray,
    probabilities: np.ndarray,
    settings,
    *,
    for_acceptance: bool,
) -> tuple[tuple[float, float, float, float, float] | None, float, str]:
    """Return the jump interval the controller should refine or enforce first.

    The stricter optional threshold is tried before the historical global one.
    This allows threshold studies to demand tighter neighbouring-``Delta P``
    control either globally or only in the early-rise part of the curve.
    """

    strict_threshold = _STRICT_JUMP_SUCCESS_THRESHOLD if for_acceptance else _STRICT_JUMP_REFINE_THRESHOLD
    generic_threshold = float(
        settings.large_delta_p_success_threshold
        if for_acceptance
        else settings.large_delta_p_refine_threshold
    )
    generic_interval = _largest_probability_jump_interval(
        temperatures,
        probabilities,
        min_delta_p=generic_threshold,
    )

    if strict_threshold > 0.0:
        strict_interval = _largest_probability_jump_interval(
            temperatures,
            probabilities,
            min_delta_p=strict_threshold,
            p_max=_STRICT_JUMP_P_MAX,
            apply_globally=_STRICT_JUMP_APPLY_GLOBALLY,
            prefer_hottest=(not _STRICT_JUMP_APPLY_GLOBALLY and not bool(for_acceptance)),
        )
        if strict_interval is not None:
            if (
                not bool(for_acceptance)
                and not _STRICT_JUMP_APPLY_GLOBALLY
                and generic_interval is not None
                and generic_threshold > 0.0
            ):
                strict_severity = float(strict_interval[-1]) / max(float(strict_threshold), 1.0e-30)
                generic_severity = float(generic_interval[-1]) / max(float(generic_threshold), 1.0e-30)
                if generic_severity > max(1.5 * strict_severity, strict_severity + 0.5):
                    return generic_interval, generic_threshold, "generic"
            strict_mode = "strict global" if _STRICT_JUMP_APPLY_GLOBALLY else f"strict P<{_STRICT_JUMP_P_MAX:2.3g}"
            return strict_interval, strict_threshold, strict_mode
    return generic_interval, generic_threshold, "generic"


def _raise_if_large_delta_p_underresolved(
    temperatures: np.ndarray,
    probabilities: np.ndarray,
    settings,
    *,
    stage: str,
    outcome: str,
) -> None:
    """Reject a final profile when the remaining neighbouring jump is too large.

    The controller already treats large neighbouring ``Delta P`` intervals as a
    refinement target. This helper enforces a stricter final acceptance rule:
    once the solver is about to return a physical verdict (success or eternal
    inflation), the remaining largest neighbouring jump must be small enough to
    trust that verdict numerically.
    """

    jump_interval, threshold, threshold_mode = _preferred_probability_jump_interval(
        temperatures,
        probabilities,
        settings,
        for_acceptance=True,
    )
    if threshold <= 0.0:
        return
    if jump_interval is None:
        return
    jump_high, jump_low, probability_high, probability_low, delta_p = jump_interval
    raise errors.PercolationError(
        "Numerical accuracy error: detected a sharp unresolved jump in P(T) at "
        f"T = [{jump_low:2.5g}, {jump_high:2.5g}] where "
        f"P = {probability_high:2.5g} -> {probability_low:2.5g} "
        f"(Delta P = {delta_p:2.5g} > {threshold:2.5g}). "
        f"{stage} cannot accept the {outcome} profile because this jump remained "
        "after the controller already spent its remaining support budget on local "
        f"jump refinement ({threshold_mode} criterion). This indicates a "
        "non-smooth or unstable bounce/action input in that interval, so the "
        "transition profile is not trustworthy."
    )


def _raise_if_hot_head_underresolved(
    temperatures: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    hubble: np.ndarray,
    settings,
    *,
    stage: str,
    outcome: str,
) -> None:
    """Reject a final profile when the hottest retained support point starts too late."""

    hot_head_unresolved, probability_hot, log_gamma_h4_hot = _hot_head_underresolved(
        temperatures,
        probabilities,
        actions,
        hubble,
        settings,
        for_acceptance=True,
    )
    if not hot_head_unresolved:
        return

    probability_gate = float(settings.f_start)
    threshold = HOT_RATE_ACCEPT_LOG
    raise errors.PercolationError(
        "Numerical accuracy error: the hot onset of P(T) is still underresolved. "
        f"The hottest retained support point has P = {probability_hot:2.5g} <= {probability_gate:2.5g}, "
        f"but log10(Gamma/H^4) = {log_gamma_h4_hot:2.5g} > {threshold:2.5g}. "
        f"{stage} cannot accept the {outcome} profile because the support still starts "
        "too late on the hot side. This misses part of the early nucleation tail and "
        "can underestimate RH / betaH_from_RH."
    )


def _build_dynamiczoomwindow_unresolved_jump_grid(
    temperatures: np.ndarray,
    probabilities: np.ndarray,
    settings,
    *,
    max_new_points: int,
    actions: np.ndarray | None = None,
    hubble: np.ndarray | None = None,
    vw: float = 1.0,
) -> np.ndarray | None:
    """Return mandatory jump-refinement points for unresolved-tail profiles.

    Even before the full transition is bracketed, a broad neighbouring jump in
    ``P(T)`` can already invalidate any later eternal-inflation / completion
    verdict. In that situation the controller should spend support directly
    inside the sharp interval before extending the cold tail further.
    """

    jump_interval, _, threshold_mode = _preferred_probability_jump_interval(
        temperatures,
        probabilities,
        settings,
        for_acceptance=True,
    )
    if jump_interval is None:
        return None
    jump_high, jump_low, _, _, delta_p = jump_interval
    if _rate_interval_is_post_peak_inert(
        temperatures,
        actions,
        hubble,
        jump_high,
        jump_low,
        settings,
        vw=vw,
    ):
        return None
    _, threshold, _ = _preferred_probability_jump_interval(
        temperatures,
        probabilities,
        settings,
        for_acceptance=True,
    )
    requested_points = int(max(1, max_new_points))
    if threshold > 0.0 and np.isfinite(delta_p):
        base_points = max(int(settings.n_action_increment), 1)
        severity = max(float(delta_p) / float(threshold), 1.0)
        boosted_points = int(math.ceil(base_points * severity))
        requested_points = min(
            int(max_new_points),
            max(base_points, min(boosted_points, 3 * base_points)),
        )
    return _build_dynamiczoomwindow_jump_grid(
        jump_high,
        jump_low,
        max_new_points=requested_points,
    )


def _range_expand_batch_size(settings, remaining_budget: int) -> int:
    """Return the number of points one unresolved-edge range expansion may add.

    Range growth is only meant to probe the next unresolved edge, not to flood
    the whole tail with support. Large batches were one of the main causes of
    the 150-250 point active grids in the first v2 tests, especially for low-g
    points that still needed many cold expansions.
    """
    if remaining_budget <= 0:
        return 0
    return max(1, min(int(settings.n_action_increment), int(remaining_budget), 5))


def _build_dynamiczoomwindow_anchor_remesh_grid(
    temperatures: np.ndarray,
    probabilities: np.ndarray,
    settings,
    *,
    max_new_points: int | None = None,
    actions: np.ndarray | None = None,
    hubble: np.ndarray | None = None,
    vw: float = 1.0,
) -> tuple[np.ndarray | None, str | None]:
    """Return one simplified remeshing batch from fixed ``P(T)`` anchors.

    The controller is intentionally simple:

    1. add temperatures that correspond to a fixed set of probability anchors
       already reached by the current profile,
    2. then repeatedly split the largest neighbouring interval whose
       ``Delta P`` still exceeds the allowed band-dependent threshold.

    This gives a near-uniform mesh in probability space, which is close to the
    original goal of distributing support roughly evenly over the whole rise of
    ``P(T)`` without stacking many separate refinement heuristics.
    """

    temps = np.asarray(temperatures, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    finite_mask = np.isfinite(temps) & np.isfinite(probs)
    temps = temps[finite_mask]
    probs = probs[finite_mask]
    if temps.size < 2:
        return None, None
    temps = _temperature_grid(temps)
    if temps.size != probs.size:
        # Preserve the original descending ordering when finite masks shortened
        # the arrays asymmetrically.
        probs = np.asarray(
            [float(value) for _, value in sorted(zip(np.asarray(temperatures, dtype=float)[finite_mask], probs), reverse=True)],
            dtype=float,
        )
        probs = probs[: temps.size]
    if temps.size < 2 or probs.size < 2:
        return None, None

    requested_new_points = max(int(settings.n_action_increment if max_new_points is None else max_new_points), 1)
    candidate = np.asarray([], dtype=float)
    occupied = temps.copy()
    added_labels: list[str] = []
    occupied_probs_cache = np.asarray(probs, dtype=float).copy()

    def update_occupied_probabilities() -> np.ndarray:
        values: list[float] = []
        for value in occupied:
            probability = _probability_at(value, temps, probs)
            values.append(float("nan") if probability is None else float(probability))
        return np.asarray(values, dtype=float)

    def add_candidate(temperature: float | None, label: str) -> bool:
        nonlocal candidate, occupied, occupied_probs_cache
        if temperature is None or not np.isfinite(float(temperature)):
            return False
        value = float(temperature)
        if _temperature_present(occupied, value) or _temperature_present(candidate, value):
            return False
        updated_occupied = _temperature_grid(np.append(occupied, value))
        candidate = _temperature_grid(np.append(candidate, value))
        occupied = updated_occupied
        occupied_probs_cache = update_occupied_probabilities()
        added_labels.append(label)
        return True

    p_min = float(np.min(probs))
    p_max = float(np.max(probs))
    anchor_targets = sorted(
        {
            float(value)
            for value in _PROBABILITY_REMESH_ANCHORS
            if np.isfinite(float(value))
        }
        | {
            float(value)
            for value in HOT_PROBABILITY_ANCHORS
            if np.isfinite(float(value))
        }
        | {float(settings.f_perc), float(settings.f_final), 1.0e-2}
    )

    for target in anchor_targets:
        if candidate.size >= requested_new_points:
            break
        if not (p_min + 1e-12 < target < p_max - 1e-12):
            continue
        target_tolerance = _probability_anchor_presence_tolerance(target, settings)
        if np.isfinite(target_tolerance) and np.any(
            np.isfinite(occupied_probs_cache)
            & (np.abs(occupied_probs_cache - float(target)) <= float(target_tolerance))
        ):
            continue
        crossing, _, crossing_index = _find_value_crossing(temps, probs, target)
        if crossing_index is None:
            continue
        add_candidate(crossing, f"P={target:2.3g}")

    while candidate.size < requested_new_points:
        occupied_probs = occupied_probs_cache

        best_interval: tuple[float, float, float, float, float, float] | None = None
        best_score = 1.0
        for i in range(min(len(occupied), len(occupied_probs)) - 1):
            upper = float(occupied[i])
            lower = float(occupied[i + 1])
            probability_high = float(occupied_probs[i])
            probability_low = float(occupied_probs[i + 1])
            if not (
                np.isfinite(upper)
                and np.isfinite(lower)
                and np.isfinite(probability_high)
                and np.isfinite(probability_low)
                and upper > lower
            ):
                continue
            delta_p = abs(probability_low - probability_high)
            threshold = _remesh_dp_limit(probability_high, probability_low, settings)
            if threshold <= 0.0:
                continue
            score = delta_p / threshold
            if _rate_interval_is_post_peak_inert(
                temps,
                actions,
                hubble,
                upper,
                lower,
                settings,
                vw=vw,
            ):
                continue
            if score > best_score:
                best_score = score
                best_interval = (
                    upper,
                    lower,
                    probability_high,
                    probability_low,
                    delta_p,
                    threshold,
                )

        if best_interval is None:
            break

        upper, lower, probability_high, probability_low, delta_p, threshold = best_interval
        midpoint = _temperature_midpoint(upper, lower)
        if not add_candidate(
            midpoint,
            f"dP={delta_p:2.3g}>{threshold:2.3g} at {probability_high:2.3g}->{probability_low:2.3g}",
        ):
            break

    if candidate.size == 0:
        return None, None

    reason = (
        "anchor_remesh from fixed P(T) anchors and largest remaining Delta P gaps; added "
        + ", ".join(added_labels)
    )
    return candidate, reason


def _build_dynamiczoomwindow_range_grid(
    temperatures: np.ndarray,
    *,
    new_high: float,
    new_low: float,
    max_new_points: int,
) -> np.ndarray | None:
    """Extend the sampled temperature range with only a few new support points."""
    if max_new_points <= 0:
        return None

    temps = _temperature_grid(temperatures)
    if temps.size == 0:
        return None

    current_high = float(temps[0])
    current_low = float(temps[-1])
    span_high = max(float(new_high) - current_high, 0.0)
    span_low = max(current_low - float(new_low), 0.0)
    if span_high <= 0.0 and span_low <= 0.0:
        return None

    remaining = int(max_new_points)
    n_high = 0
    n_low = 0
    if span_high > 0.0:
        n_high = 1
        remaining -= 1
    if span_low > 0.0 and remaining > 0:
        n_low = 1
        remaining -= 1

    total_span = span_high + span_low
    while remaining > 0 and total_span > 0.0:
        if span_high > 0.0 and (
            span_low <= 0.0
            or (n_high - 1 + 1e-12) / max(span_high, 1e-30)
            <= (n_low - 1 + 1e-12) / max(span_low, 1e-30)
        ):
            n_high += 1
        elif span_low > 0.0:
            n_low += 1
        remaining -= 1

    candidate = np.asarray([], dtype=float)
    if n_high > 0:
        for value in np.linspace(float(new_high), current_high, n_high + 1, dtype=float)[:-1]:
            candidate = _temperature_grid(np.append(candidate, float(value)))
    if n_low > 0:
        low_endpoint = float(new_low)
        low_ratio = (
            current_low / low_endpoint
            if current_low > 0.0 and low_endpoint > 0.0
            else 1.0
        )
        if low_ratio > 10.0:
            low_values = np.geomspace(current_low, low_endpoint, n_low + 1, dtype=float)[1:]
        else:
            low_values = np.linspace(current_low, low_endpoint, n_low + 1, dtype=float)[1:]
        for value in low_values:
            candidate = _temperature_grid(np.append(candidate, float(value)))

    candidate = np.asarray(
        [
            float(value)
            for value in _temperature_grid(candidate)
            if not _temperature_present(temps, float(value))
        ],
        dtype=float,
    )
    return candidate if candidate.size > 0 else None


def _adaptive_cold_boundary(
    *,
    current_low: float,
    target_low: float,
    current_probability: float,
    current_action: float,
    current_hubble: float,
    settings,
    last_temperature: float | None,
    last_probability: float | None,
    last_integral: float | None,
    last_log_gamma_h4: float | None,
    post_peak_streak: int,
) -> tuple[float, float, float | None, int]:
    """Choose the next cold boundary from the observed gain of the last batch.

    The controller measures how much the previous cold-side extension increased
    ``P(Tmin)`` (or, close to completion, ``I(Tmin) = -ln(1-P)``).  Large gains
    imply the next step should shrink; tiny gains imply that the remaining cold
    tail must be explored with a larger logarithmic jump.  Once
    ``log(Gamma/H^4)`` is already falling on the cold side, the controller
    boosts the step size further because the rate has passed its peak.
    """
    current = float(current_low)
    target = float(target_low)
    if np.isfinite(current_probability):
        p_clipped = min(max(float(current_probability), 0.0), 1.0 - 1e-15)
        current_integral = float(-math.log1p(-p_clipped)) if p_clipped > 0.0 else 0.0
    else:
        current_integral = 0.0
    current_log_gamma_h4_raw = _log_gamma_h4_array(
        np.asarray([current], dtype=float),
        np.asarray([current_action], dtype=float),
        np.asarray([current_hubble], dtype=float),
    )[0]
    current_log_gamma_h4 = float(current_log_gamma_h4_raw) if np.isfinite(current_log_gamma_h4_raw) else None

    if not (np.isfinite(current) and np.isfinite(target) and current > 0.0 and target > 0.0 and target < current):
        return target, current_integral, current_log_gamma_h4, 0

    remaining_log_span = math.log(current / target)
    dlnT_next = settings.rel_increment * remaining_log_span
    dlnT_next = min(max(dlnT_next, _COLD_TAIL_MIN_LOG_STEP), _COLD_TAIL_MAX_LOG_STEP)

    updated_streak = 0
    last_temp_finite = bool(last_temperature is not None and np.isfinite(last_temperature) and float(last_temperature) > current)
    if last_temp_finite:
        previous_log_step = math.log(float(last_temperature) / current)
        if previous_log_step > 0.0:
            if (
                current_log_gamma_h4 is not None
                and last_log_gamma_h4 is not None
                and np.isfinite(last_log_gamma_h4)
            ):
                if current_log_gamma_h4 < float(last_log_gamma_h4):
                    updated_streak = int(post_peak_streak) + 1
                else:
                    updated_streak = 0
            else:
                updated_streak = 0

            if current_probability >= _COLD_TAIL_INTEGRAL_SWITCH_P:
                previous_metric = 0.0 if last_integral is None or not np.isfinite(last_integral) else float(last_integral)
                metric_gain = current_integral - previous_metric
                target_gain = _COLD_TAIL_TARGET_DI
            else:
                previous_metric = 0.0 if last_probability is None or not np.isfinite(last_probability) else float(last_probability)
                metric_gain = float(current_probability) - previous_metric
                target_gain = _COLD_TAIL_TARGET_DP

            if not np.isfinite(metric_gain) or metric_gain <= 1e-8:
                # If the previous cold-side extension produced essentially no
                # gain in P(Tmin), do not keep exploring decade-by-decade. Jump
                # by at least two decades when possible; later remeshing is much
                # cheaper than repeatedly adding almost equivalent cold points.
                dlnT_next = max(
                    previous_log_step * _COLD_TAIL_ZERO_GAIN_SCALE,
                    math.log(100.0),
                )
            else:
                dlnT_next = previous_log_step * target_gain / metric_gain

            dlnT_next = min(
                max(dlnT_next, previous_log_step * _COLD_TAIL_STEP_SHRINK_LIMIT),
                previous_log_step * _COLD_TAIL_STEP_GROWTH_LIMIT,
            )
            dlnT_next = min(max(dlnT_next, _COLD_TAIL_MIN_LOG_STEP), _COLD_TAIL_MAX_LOG_STEP)

            if (
                updated_streak >= _COLD_TAIL_POST_PEAK_MIN_STREAK
                and current_log_gamma_h4 is not None
                and last_log_gamma_h4 is not None
                and np.isfinite(last_log_gamma_h4)
            ):
                slope = abs((current_log_gamma_h4 - float(last_log_gamma_h4)) / max(previous_log_step, 1e-12))
                boost = 1.0 + _COLD_TAIL_POST_PEAK_BOOST_COEFF * slope
                boost = min(boost, _COLD_TAIL_POST_PEAK_MAX_BOOST)
                dlnT_next *= boost
                dlnT_next = min(max(dlnT_next, _COLD_TAIL_MIN_LOG_STEP), _COLD_TAIL_MAX_LOG_STEP)

    dlnT_next = min(dlnT_next, remaining_log_span)
    if dlnT_next <= 0.0:
        return target, current_integral, current_log_gamma_h4, updated_streak

    if dlnT_next >= remaining_log_span * (1.0 - 1e-12):
        return target, current_integral, current_log_gamma_h4, updated_streak
    return float(current * math.exp(-dlnT_next)), current_integral, current_log_gamma_h4, updated_streak
