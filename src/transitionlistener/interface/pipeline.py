"""Core execution pipeline for evaluating a single parameter point.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import signal
from typing import Callable, Dict, Optional

import numpy as np

from transitionlistener.phases import Phases
from transitionlistener.transitions import Transitions
from transitionlistener.transitionObservables import (
    TransitionObservables as _TransitionObservablesAdaptiveStepSize,
)
from transitionlistener.transitionObservables_fixedstep import (
    TransitionObservables as _TransitionObservablesFixedStepSize,
)


def _select_observables_class(pot):
    """Return the TransitionObservables class for the configured algorithm_mode."""
    if pot.config.percolationConf.algorithm_mode == "fixed_step_size":
        return _TransitionObservablesFixedStepSize
    return _TransitionObservablesAdaptiveStepSize
from transitionlistener.gwfopt import FOPTspectrum
from transitionlistener.observability import Observability

from transitionlistener import console
from rich.panel import Panel

from . import state
from .logging_utils import Logger


def _find_strongest_transition(transition_observables: dict) -> dict:
    """Return the transition entry with the largest alpha among FOPTs."""
    max_alpha = -np.inf
    strongest_transition = None
    fallback_transition = None
    for obs in transition_observables.values():
        if not obs or obs.get("trantype", 1) != 1:
            continue
        if fallback_transition is None and np.isfinite(obs.get("Treh_SM_GeV", np.nan)):
            fallback_transition = obs
        alpha = obs.get("alpha", np.nan)
        if np.isfinite(alpha) and alpha > max_alpha:
            max_alpha = alpha
            strongest_transition = obs
    return strongest_transition or fallback_transition or {}


def _extract_zero_temperature_mass_spectrum(potential) -> dict:
    """Return observability-style entries describing the T=0 mass spectrum."""

    if potential is None or not hasattr(potential, "get_zero_temperature_mass_spectrum"):
        return {}

    entries = potential.get_zero_temperature_mass_spectrum()
    if not entries:
        return {}

    spectrum_entries: Dict[str, object] = {}
    for entry in entries:
        kind = entry.get("kind", "boson")
        index = entry.get("index", 0)
        base_name = f"mass_spectrum_T0_{kind}_{index:02d}"
        mass_value = entry.get("mass_GeV", np.nan)
        log_value = entry.get("log10_mass", np.nan)
        latex_label = entry.get("latex")
        text_label = entry.get("text")

        if isinstance(mass_value, (int, float, np.integer, np.floating)) and np.isfinite(mass_value):
            spectrum_entries[f"{base_name}_lin"] = float(mass_value)
        else:
            spectrum_entries[f"{base_name}_lin"] = np.nan
        if isinstance(log_value, (int, float, np.integer, np.floating)) and np.isfinite(log_value):
            spectrum_entries[f"{base_name}_log10"] = float(log_value)
        else:
            spectrum_entries[f"{base_name}_log10"] = np.nan
        if latex_label is not None:
            spectrum_entries[f"{base_name}_latex"] = latex_label
        if text_label is not None:
            spectrum_entries[f"{base_name}_text"] = text_label
        spectrum_entries[f"{base_name}_kind"] = kind

    return spectrum_entries


def _compute_single_point(
    inputparams_dict: dict,
    potential: object,
    timeout: float,
    verbose: bool,
    include_smbhb: bool,
    max_stage: int = 4,
    stage_callback: Optional[Callable[[str, dict], None]] = None,
):
    """Execute the single-point pipeline up to ``max_stage`` and optionally
    notify after each stage."""

    context: Dict[str, object] = {
        "pot": None,
        "phases": None,
        "transitions": None,
        "transition_observables": None,
        "strongest_transition_observables": None,
        "gwspectrum": None,
        "observability": None,
    }

    def notify(stage_name: str):
        if stage_callback is not None:
            stage_callback(stage_name, context)

    if include_smbhb:
        A = inputparams_dict.get("A_smbhb", 1e-15)
        gamma = inputparams_dict.get("gamma_smbhb", 2 / 3)
        astroparams_dict = {"A": A, "gamma": gamma}
    else:
        astroparams_dict = {"A": 0, "gamma": 0}

    if timeout > 0 and not state.DEBUGMODE:
        signal.alarm(timeout)
    try:
        context["pot"] = potential(inputparams_dict, verbose=verbose)
        notify("potential")

        if max_stage >= 1:
            context["phases"] = Phases(context["pot"], verbose)
            notify("phases")

        if max_stage >= 2:
            context["transitions"] = Transitions(context["phases"],
                                                 context["pot"], verbose)
            notify("transitions")

        if max_stage >= 3:
            TransitionObservables = _select_observables_class(context["pot"])
            transition_observables = TransitionObservables(
                context["pot"], context["phases"],
                context["transitions"], verbose
            )
            context["transition_observables"] = transition_observables
            context["strongest_transition_observables"] = _find_strongest_transition(
                transition_observables.transitionObservables
            )
            notify("observables")

        if max_stage >= 4:
            try:
                gwspectrum = FOPTspectrum(
                    context["strongest_transition_observables"],
                    astroparams_dict, verbose
                )
                observability = Observability(
                    gwspectrum, verbose, include_smbhb=include_smbhb
                )
                context["gwspectrum"] = gwspectrum
                context["observability"] = observability
                notify("gw")
            except Exception as err:
                if context.get("transition_observables") is None:
                    raise
                if verbose:
                    print(f"Skipping GW observability after transition observables: {err}")

        signal.alarm(0)
        return context, None
    except Exception as err:
        signal.alarm(0)
        if state.DEBUGMODE:
            raise
        return context, err


def _build_result_from_context(context: dict) -> dict:
    """Construct the result dictionary returned by ``run_TL``
    from the computation context."""
    transition_observables = context.get("transition_observables")
    strongest = context.get("strongest_transition_observables") or {}
    gwspectrum = context.get("gwspectrum")
    observability = context.get("observability")
    potential = context.get("pot")

    observability_dict = (
        observability.observability_dict.copy()
        if observability is not None
        else {}
    )

    return {
        "transitionObservables": transition_observables.transitionObservables
        if transition_observables is not None
        else {},
        "strongestTransitionObservables": strongest,
        "gwspectrum": gwspectrum.gwspec_dict if gwspectrum is not None else {},
        "observability": observability_dict,
        "error": np.nan,
    }


def _handle_single_point_error(
    error: Exception,
    inputparams_dict: dict,
    errorlogger: Logger,
    context: dict,
):
    """Log the error and shape a result dictionary containing
    available partial results."""
    error_code = getattr(error, "errorcode", -1)
    message = getattr(error, "message", str(error))
    line = (
        "ERROR of type "
        + str(error_code)
        + " for input parameters "
        + str(inputparams_dict)
        + ": "
        + message
    )
    errorlogger.log(line + "\n")
    console.print(Panel.fit(line, border_style="red"))

    result = {
        "error": np.array([error_code]),
    }
    transition_observables = context.get("transition_observables")
    if transition_observables is not None:
        result["transitionObservables"] = transition_observables.transitionObservables
    strongest = context.get("strongest_transition_observables")
    if strongest:
        result["strongestTransitionObservables"] = strongest
    gwspectrum = context.get("gwspectrum")
    if gwspectrum is not None:
        result["gwspectrum"] = gwspectrum.gwspec_dict
    observability = context.get("observability")
    potential = context.get("pot")
    observability_dict = (
        observability.observability_dict.copy()
        if observability is not None
        else {}
    )
    spectrum_entries = _extract_zero_temperature_mass_spectrum(potential)
    observability_dict.update(spectrum_entries)
    if observability_dict:
        result["observability"] = observability_dict

    return result


def run_TL(
    inputparams_dict: dict,
    potential: object,
    errorlogger: Logger,
    resultlogger: Logger,
    timeout: float,
    verbose: bool = False,
    call_from_sampler: bool = False,
    include_smbhb: bool = False,
    return_context: bool = False,
):
    """Core execution pipeline that evaluates observables for a
    single parameter point."""

    def return_result(params):
        observability = params.get("observability", {})
        line = str(inputparams_dict) + ":" + str(observability)
        resultlogger.log(line + "\n")

    context, error = _compute_single_point(
        inputparams_dict,
        potential,
        timeout,
        verbose,
        include_smbhb,
        max_stage=4,
    )

    if error is not None:
        if call_from_sampler:
            raise error
        result = _handle_single_point_error(error, inputparams_dict,
                                            errorlogger, context)
        if return_context:
            return result, context
        return result

    all_params_dict = _build_result_from_context(context)
    if resultlogger is not None:
        return_result(all_params_dict)
    if return_context:
        return all_params_dict, context
    return all_params_dict
