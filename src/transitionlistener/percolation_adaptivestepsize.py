"""Seedless dynamic zoom-window percolation (adaptive step size) algorithm.

Grid-construction helpers (step-1 scouting, jump/range/anchor-remesh builders,
adaptive cold-boundary picker, jump-underresolution guards) live in
:mod:`transitionlistener.percolation_adaptive_gridbuilders`. This module keeps the
top-level scan controllers and their :class:`PercolationState` bookkeeping.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import math
import numpy as np
from scipy import interpolate
from scipy import optimize

from transitionlistener import constants as cn
from transitionlistener import errors
from transitionlistener.bubbledynamics import (
    PercolationGrid,
    PercolationState,
    _temperature_grid,
)
from transitionlistener.percolation_adaptive_rate import (
    _build_dynamiczoomwindow_hot_head_grid,
    _build_dynamiczoomwindow_hot_shoulder_grid,
    _build_dynamiczoomwindow_rate_peak_grid,
    _find_value_crossing,
    _hot_head_underresolved,
    _temperature_present,
)
from transitionlistener.percolation_adaptive_gridbuilders import (
    _adaptive_cold_boundary,
    _build_dynamiczoomwindow_anchor_remesh_grid,
    _build_dynamiczoomwindow_jump_grid,
    _build_dynamiczoomwindow_range_grid,
    _build_dynamiczoomwindow_unresolved_jump_grid,
    _cached_promotion_bounds,
    _initial_temperature_grid,
    _raise_if_hot_head_underresolved,
    _raise_if_large_delta_p_underresolved,
    _range_expand_batch_size,
)


def _controller_added_support_points(
    support_bank: np.ndarray | None,
    free_support_bank: np.ndarray | None,
) -> int:
    """Count support points that require new controller-requested actions."""

    bank = _temperature_grid(support_bank)
    free_bank = _temperature_grid(free_support_bank)
    if bank.size == 0:
        return 0
    if free_bank.size == 0:
        return int(bank.size)
    count = 0
    for value in bank:
        scale = max(np.max(np.abs(free_bank)), abs(float(value)), 1.0)
        if not np.any(np.isclose(free_bank, float(value), atol=1e-10 * scale, rtol=0.0)):
            count += 1
    return int(count)


def _log_support_rebuild(
    state: PercolationState,
    stage: str,
    reason: str,
    old_grid: np.ndarray,
    new_grid: np.ndarray,
    verbose: bool,
) -> None:
    """Bump ``state.rebuild_count`` and print a support-rebuild line when verbose."""
    state.rebuild_count = int(state.rebuild_count or 0) + 1
    if verbose:
        print(
            "Support rebuild:",
            f"stage={stage}, reason={reason}, "
            f"old=[{float(old_grid[-1]):2.5g}, {float(old_grid[0]):2.5g}], "
            f"new=[{float(new_grid[-1]):2.5g}, {float(new_grid[0]):2.5g}], "
            f"n={len(new_grid)}",
        )


def _apply_dynamiczoomwindow_support_update(
    current_active_grid: np.ndarray,
    candidate_grid: np.ndarray,
    support_bank: np.ndarray | None,
    free_support_bank: np.ndarray | None,
    settings,
    *,
    max_new_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one support-grid update while keeping dormant scout points inactive."""
    current_active = _temperature_grid(current_active_grid)
    bank = _temperature_grid(support_bank)
    free_bank = _temperature_grid(free_support_bank)
    if bank.size == 0:
        bank = current_active
    else:
        bank = _temperature_grid(np.concatenate((bank, current_active)))

    known_active = _temperature_grid(np.concatenate((bank, current_active)))
    new_candidates: list[float] = []
    for value in _temperature_grid(candidate_grid):
        numeric = float(value)
        if _temperature_present(known_active, numeric) or _temperature_present(new_candidates, numeric):
            continue
        new_candidates.append(numeric)

    controller_added = _controller_added_support_points(bank, free_bank)
    updated_bank = bank
    dropped_new_points: list[float] = []
    limit = int(settings.n_action_max if max_new_points is None else max_new_points)
    admitted_candidates: list[float] = []
    for value in new_candidates:
        if _temperature_present(free_bank, float(value)):
            updated_bank = _temperature_grid(np.append(updated_bank, float(value)))
            admitted_candidates.append(float(value))
            continue
        if controller_added >= limit:
            dropped_new_points.append(float(value))
            continue
        updated_bank = _temperature_grid(np.append(updated_bank, float(value)))
        admitted_candidates.append(float(value))
        controller_added += 1

    active = current_active
    if admitted_candidates:
        active = _temperature_grid(np.concatenate((active, np.asarray(admitted_candidates, dtype=float))))
    if active.size == 0:
        active = current_active
    return active, updated_bank, np.asarray(dropped_new_points, dtype=float)


def _estimate_dynamiczoomwindow_accuracy(
    temperatures: np.ndarray,
    hubble: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    settings,
    pot,
    *,
    previous: dict[str, float | None] | None = None,
) -> dict[str, float | None]:
    """Estimate Tperc, Tf and RH together with conservative convergence indicators."""
    from transitionlistener import bubbledynamics as bd

    estimate: dict[str, float | None] = {
        "Tperc": math.nan,
        "Tperc_uncertainty": math.inf,
        "Tf": math.nan,
        "Tf_uncertainty": math.inf,
        "RH": math.nan,
        "RH_uncertainty": math.inf,
        "Tperc_bracket_width": math.inf,
        "Tf_bracket_width": math.inf,
    }

    Tperc, Tperc_width, _ = _find_value_crossing(temperatures, probabilities, settings.f_perc)
    estimate["Tperc"] = math.nan if Tperc is None else float(Tperc)
    estimate["Tperc_bracket_width"] = math.inf if Tperc_width is None else float(Tperc_width)

    Tf_cross, Tf_width, _ = _find_value_crossing(temperatures, probabilities, settings.f_final)
    estimate["Tf_bracket_width"] = math.inf if Tf_width is None else float(Tf_width)

    temps = np.asarray(temperatures, dtype=float)
    hubble_values = np.asarray(hubble, dtype=float)
    probability_values = np.asarray(probabilities, dtype=float)
    action_values = np.asarray(actions, dtype=float)

    finite_prob = np.isfinite(probability_values)
    finite_action = np.isfinite(action_values)
    finite_hubble = np.isfinite(hubble_values)
    if (
        np.count_nonzero(finite_prob) >= 4
        and np.count_nonzero(finite_action) >= 4
        and np.count_nonzero(finite_hubble) >= 4
    ):
        try:
            Pint = interpolate.interp1d(
                temps[finite_prob],
                probability_values[finite_prob],
                kind="cubic",
                fill_value=(0.0, 1.0),
                bounds_error=False,
            )
            Sint = interpolate.interp1d(
                temps[finite_action],
                action_values[finite_action],
                kind="cubic",
                fill_value=(-np.inf, np.inf),
                bounds_error=False,
            )
            Hint = interpolate.interp1d(
                temps[finite_hubble],
                hubble_values[finite_hubble],
                kind="cubic",
                fill_value="extrapolate",
                bounds_error=False,
            )
        except Exception:
            Pint = Sint = Hint = None
    else:
        Pint = Sint = Hint = None

    if Tf_cross is not None:
        estimate["Tf"] = float(Tf_cross)
    if Pint is not None and np.isfinite(float(estimate["Tperc"])):
        try:
            estimate["Tf"] = bd.calcTf(
                float(estimate["Tperc"]),
                float(temps[-1]),
                Pint,
                pot,
                verbose=False,
            )
        except Exception:
            pass

    if (
        Pint is not None
        and Sint is not None
        and Hint is not None
        and np.isfinite(float(estimate["Tperc"]))
    ):
        try:
            bubble_separation = bd.calcMeanBubbleSeparation(
                float(estimate["Tperc"]),
                float(temps[0]),
                Sint,
                Pint,
                Hint,
                None,
                None,
                verbose=False,
            )
            rh_indicator = float(bubble_separation * float(Hint(float(estimate["Tperc"]))))
            if np.isfinite(rh_indicator) and rh_indicator > 0.0:
                estimate["RH"] = rh_indicator
        except Exception:
            pass

    previous_Tperc = None if previous is None else previous.get("Tperc")
    previous_Tf = None if previous is None else previous.get("Tf")
    previous_RH = None if previous is None else previous.get("RH")

    if np.isfinite(float(estimate["Tperc"])):
        bracket_uncertainty = float(estimate["Tperc_bracket_width"]) / max(abs(float(estimate["Tperc"])), 1e-30)
        if previous_Tperc is None or not np.isfinite(previous_Tperc) or not np.isfinite(float(estimate["Tperc"])):
            estimate["Tperc_uncertainty"] = bracket_uncertainty
        else:
            estimate["Tperc_uncertainty"] = max(
                bracket_uncertainty,
                abs(float(estimate["Tperc"]) - float(previous_Tperc))
                / max(abs(float(estimate["Tperc"])), 1e-30),
            )
    if np.isfinite(float(estimate["Tf"])):
        bracket_uncertainty = float(estimate["Tf_bracket_width"]) / max(abs(float(estimate["Tf"])), 1e-30)
        if previous_Tf is None or not np.isfinite(previous_Tf) or not np.isfinite(float(estimate["Tf"])):
            estimate["Tf_uncertainty"] = bracket_uncertainty
        else:
            estimate["Tf_uncertainty"] = max(
                bracket_uncertainty,
                abs(float(estimate["Tf"]) - float(previous_Tf))
                / max(abs(float(estimate["Tf"])), 1e-30),
            )
    if np.isfinite(float(estimate["RH"])):
        estimate["RH_uncertainty"] = (
            0.0
            if previous_RH is None
            or not np.isfinite(previous_RH)
            or not np.isfinite(float(estimate["RH"]))
            else abs(float(estimate["RH"]) - float(previous_RH))
            / max(abs(float(estimate["RH"])), 1e-30)
        )

    return estimate


def _dynamiczoomwindow_accuracy_reached(
    estimate: dict[str, float | None],
    settings,
) -> bool:
    """Return whether all requested default-path accuracy targets were met."""
    return (
        np.isfinite(float(estimate["Tperc"]))
        and np.isfinite(float(estimate["Tf"]))
        and np.isfinite(float(estimate["RH"]))
        and float(estimate["Tperc_uncertainty"]) <= settings.acc_tperc
        and float(estimate["Tf_uncertainty"]) <= settings.acc_tfinal
        and float(estimate["RH_uncertainty"]) <= settings.acc_rh
    )


def _dynamiczoomwindow_accuracy_warning_message(
    stage: str,
    estimate: dict[str, float | None],
    settings,
) -> str:
    """Describe which accuracy goals prevented further early stopping."""
    reasons: list[str] = []
    if not np.isfinite(float(estimate["Tperc"])):
        reasons.append("Tperc could not be bracketed")
    elif float(estimate["Tperc_uncertainty"]) > settings.acc_tperc:
        reasons.append(
            f"Tperc uncertainty {100.0 * float(estimate['Tperc_uncertainty']):2.3g}% > "
            f"{100.0 * settings.acc_tperc:2.3g}%"
        )

    if not np.isfinite(float(estimate["Tf"])):
        reasons.append("Tf could not be determined")
    elif float(estimate["Tf_uncertainty"]) > settings.acc_tfinal:
        reasons.append(
            f"Tf uncertainty {100.0 * float(estimate['Tf_uncertainty']):2.3g}% > "
            f"{100.0 * settings.acc_tfinal:2.3g}%"
        )

    if not np.isfinite(float(estimate["RH"])):
        reasons.append("RH could not be determined")
    elif float(estimate["RH_uncertainty"]) > settings.acc_rh:
        reasons.append(
            f"RH uncertainty {100.0 * float(estimate['RH_uncertainty']):2.3g}% > "
            f"{100.0 * settings.acc_rh:2.3g}%"
        )

    if not reasons:
        reasons.append("the requested accuracy could not be certified")
    return (
        f"Warning: {stage} stopped before reaching all requested accuracy goals. "
        f"Increasing n_action_max above {settings.n_action_max} might help. "
        "Remaining issue(s): " + "; ".join(reasons) + "."
    )


def _try_dynamiczoomwindow_local_refinement(
    temperatures: np.ndarray,
    hubble: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    settings,
    *,
    remaining_budget: int,
    vw: float,
    apply_candidate_grid,
) -> tuple[bool, int]:
    """Spend one refinement batch on the most local unresolved feature."""
    if remaining_budget <= 0:
        return False, int(remaining_budget)

    batch = min(int(settings.n_action_increment), int(remaining_budget))
    hot_shoulder_grid, _ = _build_dynamiczoomwindow_hot_shoulder_grid(
        temperatures,
        probabilities,
        actions,
        hubble,
        settings,
        max_new_points=batch,
    )
    if hot_shoulder_grid is not None and apply_candidate_grid(hot_shoulder_grid, "hot_shoulder_refine"):
        return True, int(remaining_budget)

    jump_grid = _build_dynamiczoomwindow_unresolved_jump_grid(
        temperatures,
        probabilities,
        settings,
        max_new_points=int(remaining_budget),
        actions=actions,
        hubble=hubble,
        vw=vw,
    )
    if jump_grid is not None and apply_candidate_grid(jump_grid, "jump_refine"):
        return True, int(remaining_budget)

    return False, int(remaining_budget)


def _try_dynamiczoomwindow_completed_refinement(
    temperatures: np.ndarray,
    hubble: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    settings,
    pot,
    *,
    remaining_budget: int,
    previous_estimate: dict[str, float | None] | None,
    iterator: int,
    stage: str,
    vw: float,
    apply_candidate_grid,
    promote_cached_action_points,
) -> tuple[bool, dict[str, float | None] | None]:
    """Run the simplified remeshing order once the current range brackets the transition.

    The simplified controller still gives priority to the most obviously
    untrustworthy local feature before rebuilding the general anchor mesh:

    1. repair a still-underresolved hot-shoulder interval directly,
    2. repair a still-large neighbouring ``Delta P`` jump directly,
    3. otherwise rebuild the next support batch from fixed probability anchors
       plus the largest remaining neighbouring ``Delta P`` violations.
    """
    if iterator < settings.maxit:
        refined, remaining_budget = _try_dynamiczoomwindow_local_refinement(
            temperatures,
            hubble,
            probabilities,
            actions,
            settings,
            remaining_budget=remaining_budget,
            vw=vw,
            apply_candidate_grid=apply_candidate_grid,
        )
        if refined:
            return True, None

    remesh_grid, remesh_reason = _build_dynamiczoomwindow_anchor_remesh_grid(
        temperatures,
        probabilities,
        settings,
        max_new_points=min(int(settings.n_action_increment), int(remaining_budget)) if remaining_budget > 0 else 0,
        actions=actions,
        hubble=hubble,
        vw=vw,
    )
    if remesh_grid is not None and remesh_reason is not None and iterator < settings.maxit and remaining_budget > 0:
        if apply_candidate_grid(remesh_grid, "anchor_remesh"):
            return True, None

    estimate = _estimate_dynamiczoomwindow_accuracy(
        temperatures,
        hubble,
        probabilities,
        actions,
        settings,
        pot,
        previous=previous_estimate,
    )
    if promote_cached_action_points("cached_action_points"):
        return True, estimate
    if previous_estimate is not None and _dynamiczoomwindow_accuracy_reached(estimate, settings):
        return False, estimate

    if previous_estimate is not None or remaining_budget <= 0:
        print(_dynamiczoomwindow_accuracy_warning_message(stage, estimate, settings))
    return False, estimate


def _try_dynamiczoomwindow_ordered_support_refinement(
    temperatures: np.ndarray,
    hubble: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    settings,
    *,
    remaining_budget: int,
    vw: float,
    hot_bound: float,
    hot_head_needs_support: bool,
    apply_candidate_grid,
) -> tuple[bool, int]:
    """Try the shared adaptive step size support-refinement order before range expansion."""

    refined_locally, remaining_budget = _try_dynamiczoomwindow_local_refinement(
        temperatures,
        hubble,
        probabilities,
        actions,
        settings,
        remaining_budget=remaining_budget,
        vw=vw,
        apply_candidate_grid=apply_candidate_grid,
    )
    if refined_locally or remaining_budget <= 0:
        return bool(refined_locally), int(remaining_budget)

    if hot_head_needs_support:
        hot_extend_high = float(temperatures[0]) + (float(hot_bound) - float(temperatures[0])) * max(
            float(settings.rel_increment),
            0.35,
        )
        hot_head_grid, _ = _build_dynamiczoomwindow_hot_head_grid(
            temperatures,
            probabilities,
            actions,
            hubble,
            settings,
            new_high=min(float(hot_bound), float(hot_extend_high)),
            max_new_points=min(int(settings.n_action_increment), int(remaining_budget)),
        )
        if hot_head_grid is not None and apply_candidate_grid(hot_head_grid, "hot_head_refine"):
            return True, int(remaining_budget)

    cold_tail_unresolved = bool(probabilities[-1] < settings.f_final)
    if cold_tail_unresolved and remaining_budget > 0:
        rate_peak_grid, _ = _build_dynamiczoomwindow_rate_peak_grid(
            temperatures,
            actions,
            hubble,
            settings,
            max_new_points=min(int(settings.n_action_increment), int(remaining_budget)),
        )
        if rate_peak_grid is not None and apply_candidate_grid(rate_peak_grid, "rate_peak_refine"):
            return True, int(remaining_budget)

    if probabilities[0] <= settings.f_start and probabilities[-1] >= settings.f_perc and not hot_head_needs_support:
        remesh_grid, _ = _build_dynamiczoomwindow_anchor_remesh_grid(
            temperatures,
            probabilities,
            settings,
            max_new_points=min(int(settings.n_action_increment), int(remaining_budget)),
            actions=actions,
            hubble=hubble,
            vw=vw,
        )
        if remesh_grid is not None and apply_candidate_grid(remesh_grid, "anchor_remesh"):
            return True, int(remaining_budget)

    return False, int(remaining_budget)


def _dynamiczoomwindow_post_sweep_controller(
    temperatures: np.ndarray,
    hubble: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    settings,
    pot,
    *,
    stage: str,
    iterator: int,
    remaining_budget: int,
    previous_estimate: dict[str, float | None] | None,
    vw: float,
    hot_bound: float,
    apply_candidate_grid,
    promote_cached_action_points,
    promote_hot_head_cached_points,
    profile_residual: float = 0.0,
    profile_rho_tol: float = math.inf,
    verbose: bool = False,
) -> tuple[str, dict[str, float | None] | None, int, bool]:
    """Shared adaptive step size controller after one computed P(T) sweep.

    The stage-specific code is responsible for computing the profile.  Once the
    arrays exist, both step 2 and step 3 follow the same support-policy order:
    repair hot onset, accept/refine completed profiles, stop on budget/iteration
    limits, try local support refinement, otherwise ask the caller to range
    expand.
    """

    hot_head_needs_support, _, _ = _hot_head_underresolved(
        temperatures,
        probabilities,
        actions,
        hubble,
        settings,
        for_acceptance=True,
    )

    if hot_head_needs_support and promote_hot_head_cached_points():
        return "retry", None, int(remaining_budget), hot_head_needs_support
    if promote_cached_action_points("cached_action_points"):
        return "retry", None, int(remaining_budget), hot_head_needs_support

    completed = (
        float(probabilities[0]) <= float(settings.f_start)
        and float(probabilities[-1]) >= float(settings.f_final)
        and not hot_head_needs_support
    )
    if completed:
        refinement_previous_estimate = previous_estimate if profile_residual <= profile_rho_tol else None
        if profile_residual > profile_rho_tol and verbose:
            print(
                f"{stage} profile residual remains above tolerance even after same-grid "
                "Picard sweeps. Spending any remaining controller budget on support "
                "refinement before accepting the profile."
            )
        should_retry, previous_estimate = _try_dynamiczoomwindow_completed_refinement(
            temperatures,
            hubble,
            probabilities,
            actions,
            settings,
            pot,
            remaining_budget=remaining_budget,
            previous_estimate=refinement_previous_estimate,
            iterator=iterator,
            stage=stage,
            vw=vw,
            apply_candidate_grid=apply_candidate_grid,
            promote_cached_action_points=promote_cached_action_points,
        )
        if should_retry:
            return "retry", previous_estimate, int(remaining_budget), hot_head_needs_support
        return "accept", previous_estimate, int(remaining_budget), hot_head_needs_support

    if iterator >= settings.maxit or remaining_budget <= 0:
        print(
            f"Warning: {stage} reached the support-update limit before the full "
            "completion range was resolved. Continuing with the best available grid."
        )
        return "limit", previous_estimate, int(remaining_budget), hot_head_needs_support

    refined, remaining_budget = _try_dynamiczoomwindow_ordered_support_refinement(
        temperatures,
        hubble,
        probabilities,
        actions,
        settings,
        remaining_budget=remaining_budget,
        vw=vw,
        hot_bound=hot_bound,
        hot_head_needs_support=hot_head_needs_support,
        apply_candidate_grid=apply_candidate_grid,
    )
    if refined:
        return "retry", None, int(remaining_budget), hot_head_needs_support

    return "range_expand", previous_estimate, int(remaining_budget), hot_head_needs_support


def _dynamiczoomwindow_range_targets(
    temperatures: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    hubble: np.ndarray,
    settings,
    *,
    target_low: float,
    target_high: float,
    iterator: int,
    hot_head_needs_support: bool,
    maxit_margin: int,
    cold_tail_last_temperature: float | None,
    cold_tail_last_probability: float | None,
    cold_tail_last_integral: float | None,
    cold_tail_last_log_gamma_h4: float | None,
    cold_tail_post_peak_streak: int,
) -> tuple[float, float, float | None, float | None, float | None, float | None, int, bool]:
    """Choose the next range-expansion bounds for the shared adaptive step size controller."""

    temperatures = np.asarray(temperatures, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    actions = np.asarray(actions, dtype=float)
    hubble = np.asarray(hubble, dtype=float)
    cold_tail_unresolved = bool(probabilities[-1] < settings.f_final)

    if (probabilities[0] > settings.f_start or hot_head_needs_support) and not cold_tail_unresolved:
        high = float(temperatures[0]) + (float(target_high) - float(temperatures[0])) * settings.rel_increment
    else:
        high = float(temperatures[0])

    if not cold_tail_unresolved:
        return (
            high,
            float(temperatures[-1]),
            cold_tail_last_temperature,
            cold_tail_last_probability,
            cold_tail_last_integral,
            cold_tail_last_log_gamma_h4,
            int(cold_tail_post_peak_streak),
            False,
        )

    if iterator >= settings.maxit - int(maxit_margin):
        low = float(target_low)
    else:
        low, current_integral, current_log_gamma_h4, cold_tail_post_peak_streak = _adaptive_cold_boundary(
            current_low=float(temperatures[-1]),
            target_low=float(target_low),
            current_probability=float(probabilities[-1]),
            current_action=float(actions[-1]),
            current_hubble=float(hubble[-1]),
            settings=settings,
            last_temperature=cold_tail_last_temperature,
            last_probability=cold_tail_last_probability,
            last_integral=cold_tail_last_integral,
            last_log_gamma_h4=cold_tail_last_log_gamma_h4,
            post_peak_streak=cold_tail_post_peak_streak,
        )
        cold_tail_last_temperature = float(temperatures[-1])
        cold_tail_last_probability = float(probabilities[-1])
        cold_tail_last_integral = float(current_integral)
        cold_tail_last_log_gamma_h4 = current_log_gamma_h4
        low = min(float(low), float(temperatures[-1]))

    return (
        high,
        low,
        cold_tail_last_temperature,
        cold_tail_last_probability,
        cold_tail_last_integral,
        cold_tail_last_log_gamma_h4,
        int(cold_tail_post_peak_streak),
        True,
    )


def _apply_dynamiczoomwindow_candidate_grid(
    active_grid: np.ndarray,
    candidate_grid: np.ndarray,
    support_bank: np.ndarray,
    free_support_bank: np.ndarray,
    settings,
    outdict: dict,
    phase_symmetric,
    phase_broken,
    *,
    tmin: float,
    tmax: float,
    stage: str,
    reset_arrays,
    note_rebuild,
    extra_budget: int = 0,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Refresh the cached support bank and merge one candidate refinement grid."""
    from transitionlistener import bubbledynamics as bd

    old_grid = np.asarray(active_grid, dtype=float).copy()
    cached_points = bd._transition_action_outdict_temperatures(
        outdict,
        phase_symmetric,
        phase_broken,
        tmin=tmin,
        tmax=tmax,
    )
    if cached_points.size > 0:
        free_support_bank = _temperature_grid(
            np.concatenate((free_support_bank, cached_points))
        )

    new_active, updated_bank, dropped_new_points = _apply_dynamiczoomwindow_support_update(
        active_grid,
        candidate_grid,
        support_bank,
        free_support_bank,
        settings,
        max_new_points=int(settings.n_action_max) + max(int(extra_budget), 0),
    )
    if dropped_new_points.size > 0:
        print(
            f"Warning: n_action_max = {settings.n_action_max} blocked "
            f"{len(dropped_new_points)} new controller-added support points during {stage}."
        )

    if (
        np.asarray(new_active, dtype=float).shape == np.asarray(active_grid, dtype=float).shape
        and (
            np.asarray(new_active, dtype=float).size == 0
            or np.allclose(
                np.asarray(new_active, dtype=float),
                np.asarray(active_grid, dtype=float),
                rtol=0.0,
                atol=1e-14,
            )
        )
    ):
        return False, active_grid, updated_bank, free_support_bank

    reset_arrays(new_active)
    note_rebuild(old_grid, new_active)
    return True, new_active, updated_bank, free_support_bank


def _promote_dynamiczoomwindow_cached_points(
    active_grid: np.ndarray,
    support_bank: np.ndarray,
    free_support_bank: np.ndarray,
    settings,
    outdict: dict,
    phase_symmetric,
    phase_broken,
    *,
    tmin: float,
    tmax: float,
    promotion_high: float | None = None,
    promotion_low: float | None = None,
    stage: str,
    reason: str,
    reset_arrays,
    note_rebuild,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Promote newly cached action temperatures into the active support bank."""
    from transitionlistener import bubbledynamics as bd

    cached_points = bd._transition_action_outdict_temperatures(
        outdict,
        phase_symmetric,
        phase_broken,
        tmin=tmin,
        tmax=tmax,
    )
    pending_points = np.asarray(
        [
            float(value)
            for value in _temperature_grid(cached_points)
            if not _temperature_present(support_bank, float(value))
        ],
        dtype=float,
    )
    if (
        pending_points.size > 0
        and promotion_high is not None
        and promotion_low is not None
        and np.isfinite(float(promotion_high))
        and np.isfinite(float(promotion_low))
    ):
        high = float(max(promotion_high, promotion_low))
        low = float(min(promotion_high, promotion_low))
        scale = max(abs(high), abs(low), 1.0)
        atol = 1e-10 * scale
        pending_points = np.asarray(
            [
                float(value)
                for value in np.asarray(pending_points, dtype=float)
                if low - atol <= float(value) <= high + atol
            ],
            dtype=float,
        )
        pending_points = _temperature_grid(pending_points)
    if pending_points.size == 0:
        return False, active_grid, support_bank, free_support_bank

    return _apply_dynamiczoomwindow_candidate_grid(
        active_grid,
        pending_points,
        support_bank,
        free_support_bank,
        settings,
        outdict,
        phase_symmetric,
        phase_broken,
        tmin=tmin,
        tmax=tmax,
        stage=stage,
        reset_arrays=reset_arrays,
        note_rebuild=lambda old_grid, new_grid: note_rebuild(reason, old_grid, new_grid),
    )


def _compute_step2_profile(
    state: PercolationState,
    TSYM: np.ndarray,
    previous_probability: np.ndarray,
    settings,
    outdict: dict,
    pot,
    phase_symmetric,
    phase_broken,
    CF: float,
    vw: float,
    verbose: bool,
) -> None:
    """Compute the P=0-Hubble profile used by adaptive step size, percolation step 2."""
    from transitionlistener import bubbledynamics as bd

    TSYM = np.asarray(TSYM, dtype=float)
    previous_probability = np.asarray(previous_probability, dtype=float)
    state.Sr.fill(0)
    state.Hr.fill(0)
    state.Pr.fill(0)
    state.Tb.fill(0)
    action_tail_saturated = False

    for i, T in enumerate(TSYM):
        rho = bd.energyDensity(pot, phase_symmetric, T)
        state.Hr[i] = bd.HubbleParameter(rho, CF)
        state.Sr[i] = (
            bd.calcAction(pot, T, phase_symmetric, phase_broken, outdict)
            if not action_tail_saturated
            else np.nan
        )
        if (
            i >= 1
            and previous_probability[i] > settings.f_final
            and previous_probability[i - 1] > settings.f_final
        ):
            if verbose and not action_tail_saturated:
                print(
                    "WARNING: P > ",
                    settings.f_final,
                    " at T = ",
                    TSYM[i] * CF,
                    " GeV detected. Stop computation of action from here on "
                    "and only fill in the values for the Hubble rate.",
                )
            action_tail_saturated = True

    _, Pr_ode = bd.percIntegralODE_full_sweep(
        TSYM,
        state.Hr,
        state.Sr,
        vw=vw,
        pot=pot,
        phase_symmetric=phase_symmetric,
        time_temperature_mode=settings.time_temperature_mode,
        integral_method=settings.integral_method,
    )
    state.Pr[:] = Pr_ode

    if verbose:
        for i, T in enumerate(TSYM):
            print(
                f"Step {i+1}/{len(TSYM)}, T = {TSYM[i] * CF:2.8g} GeV, "
                f"P_true = {state.Pr[i]:2.5g}, S_3 = {state.Sr[i] * CF:2.5g} GeV, "
                f"S_3/T = {state.Sr[i] / T:2.5g}"
            )


def _compute_step3_profile(
    state: PercolationState,
    TSYM: np.ndarray,
    previous_probability: np.ndarray,
    settings,
    outdict: dict,
    pot,
    phase_symmetric,
    phase_broken,
    CF: float,
    vw: float,
    TBROmin: float,
    TBROmax: float,
    verbose: bool,
) -> None:
    """Compute the P-dependent Hubble/Tb profile used by adaptive step size, percolation step 3."""
    from transitionlistener import bubbledynamics as bd

    previous_probability = np.asarray(previous_probability, dtype=float)
    state.Hr.fill(0)
    state.Pr.fill(0)
    state.Tb.fill(np.nan)
    state.Sr.fill(0)
    profile_saturated = False

    for i, T in enumerate(TSYM):
        if not profile_saturated:
            eSYM = bd.energyDensity(pot, phase_symmetric, T)
            state.Sr[i] = bd.calcAction(pot, T, phase_symmetric, phase_broken, outdict)
            P = float(previous_probability[i]) if np.isfinite(previous_probability[i]) else 0.0

            if P <= 1e-12:
                state.Hr[i] = bd.HubbleParameter(eSYM, CF)
                state.Tb[i] = np.nan
            elif i == 0:
                try:
                    TBRO = optimize.brentq(
                        bd.Tb_criterion,
                        TBROmin,
                        TBROmax,
                        args=(T, phase_symmetric, phase_broken, pot),
                    )
                except ValueError as err:
                    raise errors.PercolationError(
                        "Error in Tb_criterion: "
                        f"{err}. The phase tracing or tunneling precision may be too low."
                    )
                state.Tb[i] = TBRO
                eBRO = bd.energyDensity(pot, phase_broken, TBRO)
                state.Hr[i] = bd.HubbleParameter(P * eBRO + (1.0 - P) * eSYM, CF)
            else:
                P_prev = float(previous_probability[i - 1]) if np.isfinite(previous_probability[i - 1]) else 0.0
                previous_tb = float(state.Tb[i - 1]) if np.isfinite(state.Tb[i - 1]) else np.nan
                if P_prev < 1e-6 or not np.isfinite(previous_tb):
                    try:
                        TBRO = optimize.brentq(
                            bd.Tb_criterion,
                            TBROmin,
                            TBROmax,
                            args=(T, phase_symmetric, phase_broken, pot),
                        )
                    except ValueError as err:
                        if not np.isfinite(previous_tb):
                            raise errors.PercolationError(
                                "Error in Tb_criterion for the first nonzero-P support point: "
                                f"{err}. The hot pure-symmetric tail may be too close to the "
                                "broken-branch tracing boundary."
                            )
                        TBRO = optimize.brentq(
                            bd.entropy_criterion_SYM_BRO,
                            TBROmin,
                            TBROmax,
                            args=(previous_tb, T, TSYM[i - 1], phase_broken, pot),
                        )
                else:
                    try:
                        TBRO = optimize.brentq(
                            bd.entropy_criterion_SYM_BRO,
                            TBROmin,
                            TBROmax,
                            args=(previous_tb, T, TSYM[i - 1], phase_broken, pot),
                        )
                    except ValueError:
                        TBRO = optimize.brentq(
                            bd.Tb_criterion,
                            TBROmin,
                            TBROmax,
                            args=(T, phase_symmetric, phase_broken, pot),
                        )
                eBRO = bd.energyDensity(pot, phase_broken, TBRO)

                dP = P - P_prev
                energy_release = eBRO * (P - dP) + dP * eSYM
                if energy_release / eBRO < 1e-50:
                    state.Tb[i] = TBRO
                else:
                    try:
                        TBRO = optimize.brentq(
                            bd.energy_criterion_BRO,
                            TBROmin,
                            TBROmax,
                            args=(eSYM, eBRO, P, dP, phase_broken, pot),
                        )
                        state.Tb[i] = TBRO
                    except ValueError:
                        state.Tb[i] = state.Tb[i - 1]
                        TBRO = state.Tb[i]
                    eBRO = bd.energyDensity(pot, phase_broken, TBRO)

                if state.Tb[i] < TSYM[i]:
                    raise ValueError(
                        "There was a problem with the T_bro calculation, "
                        "T_bro < T_sym. This is usually not allowed."
                    )

                state.Hr[i] = bd.HubbleParameter(P * eBRO + (1.0 - P) * eSYM, CF)
        else:
            state.Sr[i] = np.nan
            state.Hr[i] = bd.HubbleParameter(bd.energyDensity(pot, phase_symmetric, T), CF)

        if i >= 1 and previous_probability[i] > settings.f_final and previous_probability[i - 1] > settings.f_final:
            if verbose and not profile_saturated:
                print(
                    "WARNING: P > ",
                    settings.f_final,
                    " at T = ",
                    TSYM[i] * CF,
                    " GeV detected. Stop computation of action from here on "
                    "and only fill in the values for the Hubble rate.",
                )
            profile_saturated = True

    _, Pr_ode = bd.percIntegralODE_full_sweep(
        TSYM,
        state.Hr,
        state.Sr,
        vw=vw,
        pot=pot,
        phase_symmetric=phase_symmetric,
        time_temperature_mode=settings.time_temperature_mode,
        integral_method=settings.integral_method,
    )
    state.Pr[:] = Pr_ode

    if verbose:
        for i, T in enumerate(TSYM):
            print(
                f"Step {i+1}/{len(TSYM)}, T = {T * CF:2.8g} GeV, "
                f"P_true = {state.Pr[i]:2.5g}, S_3 = {state.Sr[i] * CF:2.5g} GeV, "
                f"S_3/T = {state.Sr[i] / T:2.5g}"
            )


def _initial_percolation_scan_dynamiczoomwindow(
    grid: PercolationGrid,
    settings,
    outdict: dict,
    pot,
    phase_symmetric,
    phase_broken,
    CF: float,
    vw: float,
    verbose: bool,
) -> PercolationState:
    """Dynamic zoom-window step 2 with a coarse scan and local window refinement."""
    from transitionlistener import bubbledynamics as bd

    original_grid = _temperature_grid(grid.TSYM)
    TSYM = original_grid.copy()
    # Free support points are cached/scout action temperatures. They may be
    # promoted into the active grid later, but they do not consume the
    # controller's new-action budget.
    free_support_bank = (
        _temperature_grid(grid.free_support_bank)
        if grid.free_support_bank is not None
        else bd._transition_action_outdict_temperatures(
            outdict,
            phase_symmetric,
            phase_broken,
            tmin=grid.tmin,
            tmax=grid.tmax,
        )
    )
    state = PercolationState(
        TSYM=TSYM,
        Sr=np.zeros_like(TSYM),
        Hr=np.zeros_like(TSYM),
        Pr=np.zeros_like(TSYM),
        Tb=np.zeros_like(TSYM),
        explored_tmin=float(TSYM[-1]),
        rebuild_count=0,
        support_bank=_temperature_grid(TSYM),
        free_support_bank=free_support_bank,
    )

    iterator = 1
    previous_estimate = None

    def _note_rebuild(reason: str, old_grid: np.ndarray, new_grid: np.ndarray) -> None:
        _log_support_rebuild(state, "step2", reason, old_grid, new_grid, verbose)

    def _reset_active_arrays(new_grid: np.ndarray) -> None:
        nonlocal TSYM
        TSYM = np.asarray(new_grid, dtype=float)
        state.TSYM = TSYM
        state.Sr = np.zeros_like(TSYM)
        state.Hr = np.zeros_like(TSYM)
        state.Pr = np.zeros_like(TSYM)
        state.Tb = np.zeros_like(TSYM)

    def _apply_candidate_grid(candidate_grid: np.ndarray, reason: str, *, extra_budget: int = 0) -> bool:
        nonlocal TSYM
        changed, TSYM, state.support_bank, state.free_support_bank = _apply_dynamiczoomwindow_candidate_grid(
            TSYM,
            candidate_grid,
            state.support_bank,
            state.free_support_bank,
            settings,
            outdict,
            phase_symmetric,
            phase_broken,
            tmin=grid.tmin,
            tmax=grid.tmax,
            stage="step 2",
            reset_arrays=_reset_active_arrays,
            note_rebuild=lambda old_grid, new_grid: _note_rebuild(reason, old_grid, new_grid),
            extra_budget=extra_budget,
        )
        return changed

    def _promote_cached_action_points(reason: str) -> bool:
        nonlocal TSYM
        promotion_bounds = _cached_promotion_bounds(TSYM)
        changed, TSYM, state.support_bank, state.free_support_bank = _promote_dynamiczoomwindow_cached_points(
            TSYM,
            state.support_bank,
            state.free_support_bank,
            settings,
            outdict,
            phase_symmetric,
            phase_broken,
            tmin=grid.tmin,
            tmax=grid.tmax,
            promotion_high=None if promotion_bounds is None else promotion_bounds[0],
            promotion_low=None if promotion_bounds is None else promotion_bounds[1],
            stage="step 2",
            reason=reason,
            reset_arrays=_reset_active_arrays,
            note_rebuild=_note_rebuild,
        )
        return changed

    def _promote_hot_head_cached_points() -> bool:
        nonlocal TSYM
        changed, TSYM, state.support_bank, state.free_support_bank = _promote_dynamiczoomwindow_cached_points(
            TSYM,
            state.support_bank,
            state.free_support_bank,
            settings,
            outdict,
            phase_symmetric,
            phase_broken,
            tmin=grid.tmin,
            tmax=grid.tmax,
            promotion_high=float(grid.tmax),
            promotion_low=float(TSYM[0]),
            stage="step 2",
            reason="hot_head_cached_points",
            reset_arrays=_reset_active_arrays,
            note_rebuild=_note_rebuild,
        )
        return changed

    if verbose:
        print("\nPercolation step 2: calculating Tperc assuming P = 0 in the Hubble rate")

    while True:
        state.explored_tmin = min(float(state.explored_tmin), float(TSYM[-1]))
        _compute_step2_profile(
            state,
            TSYM,
            np.zeros_like(TSYM),
            settings,
            outdict,
            pot=pot,
            phase_symmetric=phase_symmetric,
            phase_broken=phase_broken,
            CF=CF,
            vw=vw,
            verbose=verbose,
        )

        remaining_budget = max(
            int(settings.n_action_max) - _controller_added_support_points(state.support_bank, state.free_support_bank),
            0,
        )
        decision, previous_estimate, remaining_budget, hot_head_needs_support = (
            _dynamiczoomwindow_post_sweep_controller(
                TSYM,
                state.Hr,
                state.Pr,
                state.Sr,
                settings,
                pot,
                stage="step 2",
                iterator=iterator,
                remaining_budget=remaining_budget,
                previous_estimate=previous_estimate,
                vw=vw,
                hot_bound=grid.tmax,
                apply_candidate_grid=_apply_candidate_grid,
                promote_cached_action_points=_promote_cached_action_points,
                promote_hot_head_cached_points=_promote_hot_head_cached_points,
                verbose=verbose,
            )
        )
        if decision == "retry":
            iterator += 1
            continue
        if decision in {"accept", "limit"}:
            break

        (
            Tmax_new,
            Tmin_new,
            state.cold_tail_last_temperature,
            state.cold_tail_last_probability,
            state.cold_tail_last_integral,
            state.cold_tail_last_log_gamma_h4,
            state.cold_tail_post_peak_streak,
            _,
        ) = _dynamiczoomwindow_range_targets(
            TSYM,
            state.Pr,
            state.Sr,
            state.Hr,
            settings,
            target_low=grid.tmin,
            target_high=grid.tmax,
            iterator=iterator,
            hot_head_needs_support=hot_head_needs_support,
            maxit_margin=2,
            cold_tail_last_temperature=state.cold_tail_last_temperature,
            cold_tail_last_probability=state.cold_tail_last_probability,
            cold_tail_last_integral=state.cold_tail_last_integral,
            cold_tail_last_log_gamma_h4=state.cold_tail_last_log_gamma_h4,
            cold_tail_post_peak_streak=state.cold_tail_post_peak_streak,
        )

        candidate_grid = _build_dynamiczoomwindow_range_grid(
            TSYM,
            new_high=Tmax_new,
            new_low=Tmin_new,
            max_new_points=_range_expand_batch_size(settings, int(remaining_budget)),
        )
        if candidate_grid is None or not _apply_candidate_grid(candidate_grid, "range_expand"):
            break
        previous_estimate = None
        iterator += 1

    return state


def _refine_percolation_temperature_dynamiczoomwindow(
    state: PercolationState,
    grid: PercolationGrid,
    settings,
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
    """Dynamic zoom-window step 3 with the same focused refinement as step 2."""
    from transitionlistener import bubbledynamics as bd

    TBROmin = grid.TBROmin
    TBROmax = grid.TBROmax
    TSYM = np.asarray(state.TSYM, dtype=float)
    state.support_bank = _temperature_grid(state.support_bank)
    state.free_support_bank = _temperature_grid(state.free_support_bank)
    state.rebuild_count = int(state.rebuild_count or 0)
    state.cold_tail_post_peak_streak = int(state.cold_tail_post_peak_streak or 0)
    previous_estimate = None
    iterator = 1
    profile_picard_retries = 0
    profile_rho_tol = max(5.0 * float(settings.acc_rh), 0.02)

    def _note_rebuild(reason: str, old_grid: np.ndarray, new_grid: np.ndarray) -> None:
        _log_support_rebuild(state, "step3", reason, old_grid, new_grid, verbose)

    def _reset_state_arrays(new_grid: np.ndarray) -> None:
        nonlocal TSYM, profile_picard_retries
        TSYM = np.asarray(new_grid, dtype=float)
        state.TSYM = TSYM
        state.Sr = np.zeros_like(TSYM)
        state.Hr = np.zeros_like(TSYM)
        state.Pr = np.zeros_like(TSYM)
        state.Tb = np.zeros_like(TSYM)
        profile_picard_retries = 0

    def _profile_self_consistency_residual() -> float:
        """Return the current step-3 fixed-point residual on the active broken branch.

        Step 3 updates H(T) from the previous Picard iterate Pprev(T).  Once the
        current iterate P(T) is known, a fully self-consistent solution should
        satisfy rho(H) ~= P * eBRO(TBRO) + (1 - P) * eSYM(TSYM) over the region
        where TBRO is still actively evolved. Large deviations mean that Tperc
        may already look stable while the underlying P/H/TBRO profile has not
        yet caught up.
        """
        active_mask = (
            np.isfinite(state.Tb)
            & (state.Tb > 0.0)
            & np.isfinite(state.Pr)
            & np.isfinite(state.Hr)
        )
        if not np.any(active_mask):
            return 0.0

        rho_from_h = 3.0 * np.asarray(state.Hr[active_mask], dtype=float) ** 2
        rho_from_h *= (cn.Mpl_GeV / CF) ** 2 / (8.0 * np.pi)

        tsym_active = np.asarray(TSYM[active_mask], dtype=float)
        tbro_active = np.asarray(state.Tb[active_mask], dtype=float)
        e_sym = np.asarray(
            [bd.energyDensity(pot, phase_symmetric, float(temp)) for temp in tsym_active],
            dtype=float,
        )
        e_bro = np.asarray(
            [bd.energyDensity(pot, phase_broken, float(temp)) for temp in tbro_active],
            dtype=float,
        )
        rho_target = np.asarray(state.Pr[active_mask], dtype=float) * e_bro
        rho_target += (1.0 - np.asarray(state.Pr[active_mask], dtype=float)) * e_sym
        scale = np.maximum(np.maximum(np.abs(rho_target), np.abs(rho_from_h)), 1.0e-30)
        return float(np.nanmax(np.abs(rho_from_h - rho_target) / scale))

    def _apply_candidate_grid(candidate_grid: np.ndarray, reason: str, *, extra_budget: int = 0) -> bool:
        nonlocal TSYM
        changed, TSYM, state.support_bank, state.free_support_bank = _apply_dynamiczoomwindow_candidate_grid(
            TSYM,
            candidate_grid,
            state.support_bank,
            state.free_support_bank,
            settings,
            outdict,
            phase_symmetric,
            phase_broken,
            tmin=grid.tmin,
            tmax=grid.tmax,
            stage="step 3",
            reset_arrays=_reset_state_arrays,
            note_rebuild=lambda old_grid, new_grid: _note_rebuild(reason, old_grid, new_grid),
            extra_budget=extra_budget,
        )
        return changed

    def _promote_cached_action_points(reason: str) -> bool:
        nonlocal TSYM
        promotion_bounds = _cached_promotion_bounds(TSYM)
        changed, TSYM, state.support_bank, state.free_support_bank = _promote_dynamiczoomwindow_cached_points(
            TSYM,
            state.support_bank,
            state.free_support_bank,
            settings,
            outdict,
            phase_symmetric,
            phase_broken,
            tmin=grid.tmin,
            tmax=grid.tmax,
            promotion_high=None if promotion_bounds is None else promotion_bounds[0],
            promotion_low=None if promotion_bounds is None else promotion_bounds[1],
            stage="step 3",
            reason=reason,
            reset_arrays=_reset_state_arrays,
            note_rebuild=_note_rebuild,
        )
        return changed

    def _promote_hot_head_cached_points() -> bool:
        nonlocal TSYM
        changed, TSYM, state.support_bank, state.free_support_bank = _promote_dynamiczoomwindow_cached_points(
            TSYM,
            state.support_bank,
            state.free_support_bank,
            settings,
            outdict,
            phase_symmetric,
            phase_broken,
            tmin=grid.tmin,
            tmax=grid.tmax,
            promotion_high=float(grid.tmax),
            promotion_low=float(TSYM[0]),
            stage="step 3",
            reason="hot_head_cached_points",
            reset_arrays=_reset_state_arrays,
            note_rebuild=_note_rebuild,
        )
        return changed

    if verbose:
        print("\nPercolation step 3: calculating Tperc with P-dependent Hubble rate")

    while True:
        if state.explored_tmin is None:
            state.explored_tmin = float(TSYM[-1])
        else:
            state.explored_tmin = min(float(state.explored_tmin), float(TSYM[-1]))

        if verbose:
            print(f"Iteration {iterator}, T_perc = {Tperc_prev * CF:2.8g} GeV")

        Pprev = np.asarray(state.Pr, dtype=float).copy()
        _compute_step3_profile(
            state,
            TSYM,
            Pprev,
            settings,
            outdict,
            pot,
            phase_symmetric,
            phase_broken,
            CF,
            vw,
            TBROmin,
            TBROmax,
            verbose,
        )

        Pint = interpolate.interp1d(TSYM, state.Pr)
        try:
            Tperc = optimize.brentq(lambda T: Pint(T) - settings.f_perc, TSYM[0], TSYM[-1])
        except ValueError as err:
            msg = err.args[0]
            if msg.startswith("f(a) and f(b) must have different signs") and Pint(TSYM[-1]) < settings.f_perc:
                raise errors.TooMuchSupercoolingError(
                    "The percolation temperature could not be found because the "
                    "true vacuum fraction only reaches "
                    f"{Pint(TSYM[-1])} < {settings.f_perc} at Tmin = "
                    f"{TSYM[-1] * CF} GeV."
                )
            raise errors.PercolationError(err)

        # First demand consistency of the P-dependent Hubble update itself: if
        # Tperc still moves noticeably between successive step-3 iterations,
        # continue iterating before judging the support accuracy.
        if np.isfinite(Tperc) and abs(Tperc - Tperc_prev) / max(abs(Tperc), 1e-30) >= rtol:
            if iterator >= settings.maxit:
                print(
                    "Warning: step 3 reached the iteration limit before Tperc fully "
                    "stopped moving under the P-dependent Hubble update. Continuing."
                )
            else:
                Tperc_prev = Tperc
                iterator += 1
                continue

        profile_residual = _profile_self_consistency_residual()
        if profile_residual > profile_rho_tol and profile_picard_retries < 3 and iterator < settings.maxit:
            if verbose:
                print(
                    "Step-3 profile not yet self-consistent on the current support grid: "
                    f"max |rho(H) - rho_mix| / rho = {profile_residual:2.5g} > {profile_rho_tol:2.5g}. "
                    "Recomputing P/H/TBRO once more with the same cached actions."
                )
            Tperc_prev = Tperc
            iterator += 1
            profile_picard_retries += 1
            continue
        if profile_residual <= profile_rho_tol:
            profile_picard_retries = 0

        remaining_budget = max(
            int(settings.n_action_max) - _controller_added_support_points(state.support_bank, state.free_support_bank),
            0,
        )
        decision, previous_estimate, remaining_budget, hot_head_needs_support = (
            _dynamiczoomwindow_post_sweep_controller(
                TSYM,
                state.Hr,
                state.Pr,
                state.Sr,
                settings,
                pot,
                stage="step 3",
                iterator=iterator,
                remaining_budget=remaining_budget,
                previous_estimate=previous_estimate,
                vw=vw,
                hot_bound=grid.tmax,
                apply_candidate_grid=_apply_candidate_grid,
                promote_cached_action_points=_promote_cached_action_points,
                promote_hot_head_cached_points=_promote_hot_head_cached_points,
                profile_residual=profile_residual,
                profile_rho_tol=profile_rho_tol,
                verbose=verbose,
            )
        )
        if decision == "retry":
            Tperc_prev = Tperc
            iterator += 1
            continue
        if decision == "accept":
            _raise_if_hot_head_underresolved(
                TSYM,
                state.Pr,
                state.Sr,
                state.Hr,
                settings,
                stage="step 3",
                outcome="successful",
            )
            _raise_if_large_delta_p_underresolved(
                TSYM,
                state.Pr,
                settings,
                stage="step 3",
                outcome="successful",
            )
            return float(Tperc), state

        if decision == "limit":
            _raise_if_hot_head_underresolved(
                TSYM,
                state.Pr,
                state.Sr,
                state.Hr,
                settings,
                stage="step 3",
                outcome="best-available unresolved-tail",
            )
            _raise_if_large_delta_p_underresolved(
                TSYM,
                state.Pr,
                settings,
                stage="step 3",
                outcome="best-available unresolved-tail",
            )
            return float(Tperc), state

        cold_tail_unresolved = bool(state.Pr[-1] < settings.f_final)
        if cold_tail_unresolved:
            lower_bound_reached = abs(float(TSYM[-1]) - float(grid.tmin)) <= 1e-12 * max(abs(float(grid.tmin)), 1.0)
            if lower_bound_reached:
                _raise_if_hot_head_underresolved(
                    TSYM,
                    state.Pr,
                    state.Sr,
                    state.Hr,
                    settings,
                    stage="step 3",
                    outcome="eternal-inflation",
                )
                _raise_if_large_delta_p_underresolved(
                    TSYM,
                    state.Pr,
                    settings,
                    stage="step 3",
                    outcome="eternal-inflation",
                )
                raise errors.EternalInflationError(
                    "The transition cannot completely finish because the true vacuum "
                    "fraction at the lowest evaluated temperature stays below f_final: "
                    f"P(Tmin) = {state.Pr[-1]:2.5g} < {settings.f_final:2.5g} at "
                    f"Tmin = {TSYM[-1] * CF:2.5g} GeV."
                )

        (
            Tmax_new,
            Tmin_new,
            state.cold_tail_last_temperature,
            state.cold_tail_last_probability,
            state.cold_tail_last_integral,
            state.cold_tail_last_log_gamma_h4,
            state.cold_tail_post_peak_streak,
            _,
        ) = _dynamiczoomwindow_range_targets(
            TSYM,
            state.Pr,
            state.Sr,
            state.Hr,
            settings,
            target_low=grid.tmin,
            target_high=grid.tmax,
            iterator=iterator,
            hot_head_needs_support=hot_head_needs_support,
            maxit_margin=1,
            cold_tail_last_temperature=state.cold_tail_last_temperature,
            cold_tail_last_probability=state.cold_tail_last_probability,
            cold_tail_last_integral=state.cold_tail_last_integral,
            cold_tail_last_log_gamma_h4=state.cold_tail_last_log_gamma_h4,
            cold_tail_post_peak_streak=state.cold_tail_post_peak_streak,
        )

        candidate_grid = _build_dynamiczoomwindow_range_grid(
            TSYM,
            new_high=Tmax_new,
            new_low=Tmin_new,
            max_new_points=_range_expand_batch_size(settings, int(remaining_budget)),
        )
        if candidate_grid is None or not _apply_candidate_grid(candidate_grid, "range_expand"):
            _raise_if_large_delta_p_underresolved(
                TSYM,
                state.Pr,
                settings,
                stage="step 3",
                outcome="best-available unresolved-tail",
            )
            return float(Tperc), state
        Tperc_prev = Tperc
        iterator += 1
