"""Single-point evaluation logic with staged plotting callbacks.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path

from transitionlistener import plots
from transitionlistener import observability as obs
from transitionlistener.helper_functions import load_potential
from transitionlistener import console

from .logging_utils import Logger
from .pipeline import (
    _compute_single_point,
    _build_result_from_context,
    _handle_single_point_error,
)


_STAGE_NAME_TO_LEVEL = {
    "potential": 0,
    "phases": 1,
    "transitions": 2,
    "observables": 3,
    "gw": 4,
}


_MASS_GROUP_PREFIX = "mass_spectrum_T0_"
_MASS_KIND_ORDER = {"boson": 0, "fermion": 1}


def _plot_required_stage(plot_key: str, spec: dict) -> int:
    """Return the minimum pipeline stage needed for a given plot."""
    if plot_key == "potential":
        return 0
    if plot_key == "action":
        return 1
    if plot_key == "energy_density":
        return 1
    if plot_key == "phases":
        include = bool(spec.get("include_transitions?", True))
        return 3 if include else 1
    if plot_key in {"profile", "profileV", "dofs", "percolation"}:
        return 3
    if plot_key == "sensitivities":
        return 0
    if plot_key == "gw_spectrum":
        return 4
    return 4


def _format_numeric_value(key: str, value) -> str:
    """Render numeric output in a consistent fashion."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "-"
        if key in {"step", "total_steps"}:
            return str(int(round(value)))
        return f"{float(value):.4e}"
    return str(value)


def _format_warning_value(value) -> str:
    """Render warning flags as booleans."""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "False"
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return str(bool(value))
    return str(bool(value))


@dataclass
class PlotPlan:
    """Information about requested plots and their pipeline requirements."""

    specs: dict[str, dict]
    ordered_keys: list[str]
    wants_gw_spectrum: bool
    stage_requirements: dict[str, int]


def _build_plot_plan(conf) -> PlotPlan:
    """Extract plot configuration from the user configuration."""
    plot_specs = conf.additional_plots or {}
    active_plot_keys = [
        key
        for key, spec in plot_specs.items()
        if isinstance(spec, dict) and spec.get("plot?", False)
    ]
    wants_gw_spectrum = "gw_spectrum" in active_plot_keys
    ordered_non_gw_keys = [
        key for key in plot_specs.keys() if key in active_plot_keys and key != "gw_spectrum"
    ]
    stage_requirements = {
        key: _plot_required_stage(key, plot_specs[key])
        for key in ordered_non_gw_keys
    }
    return PlotPlan(
        specs=plot_specs,
        ordered_keys=ordered_non_gw_keys,
        wants_gw_spectrum=wants_gw_spectrum,
        stage_requirements=stage_requirements,
    )


class PlotExecutor:
    """Manage additional plots and spectrum rendering during the pipeline."""

    def __init__(self, conf, input_params: dict, plan: PlotPlan, outpath: str, *, verbose: bool = False):
        """Store plot requests and the pipeline context needed to satisfy them."""
        self._conf = conf
        self._input_params = input_params
        self._plan = plan
        self._outpath = outpath
        self._verbose = verbose
        self._executed: set[str] = set()

    def stage_callback(self, stage_name: str, ctx: dict):
        """Pipeline hook that triggers plots once prerequisites are met."""
        current_level = _STAGE_NAME_TO_LEVEL.get(stage_name, -1)
        for key in self._plan.ordered_keys:
            if key in self._executed:
                continue
            required_stage = self._plan.stage_requirements.get(key, 4)
            if required_stage <= current_level:
                self._run_additional_plot(key, ctx)

        if stage_name == "gw" and self._plan.wants_gw_spectrum and "gw_spectrum" not in self._executed:
            self._run_gw_spectrum(ctx)

    def mark_gw_skipped(self, message: str, colour: str = "red"):
        """Helper to emit a message when the GW spectrum cannot be produced."""
        if self._plan.wants_gw_spectrum and "gw_spectrum" not in self._executed:
            console.print(f"[bold {colour}]{message}[/bold {colour}]")
            self._executed.add("gw_spectrum")

    def _run_additional_plot(self, key: str, ctx: dict):
        """Execute one configured additional plot and degrade gracefully on failure."""
        self._executed.add(key)
        spec = self._plan.specs.get(key, {})
        try:
            self._run_additional_plot_impl(key, spec, ctx)
        except Exception as err:
            console.print(
                f"[bold yellow]Skipping additional plot '{key}': {err}[/bold yellow]"
            )

    def _run_additional_plot_impl(self, key: str, spec: dict, ctx: dict):
        """Dispatch one additional-plot request to the matching plotting routine."""
        if key == "sensitivities":
            console.print(f"[bold green]Create {key} plot...[/bold green]")
            gw = obs.GW_Sensitivity_Data()
            gw.foldername = self._outpath
            plots.plotSensitivities(gw, showplot=False, call_from_spectrum=False)
            console.print(f"[bold green]Done: {key} plot.[/bold green]")
            return

        plotter = plots.TLPlots(
            self._input_params,
            self._conf.modelfile,
            self._conf.potential_name,
            self._outpath,
            self._conf.plot_description,
            potential=ctx.get("pot"),
            phases=ctx.get("phases"),
            transitions=ctx.get("transitions"),
            transition_observables=ctx.get("transition_observables"),
            verbose=self._verbose,
        )

        console.print(f"[bold green]Create {key} plot...[/bold green]")
        if key == "action":
            phase_indices_cfg = spec.get("phase_indices", [0, 1])
            if not isinstance(phase_indices_cfg, (list, tuple)) or len(phase_indices_cfg) != 2:
                raise ValueError(
                    "additional_plots.action.phase_indices must be a list of two phase keys."
                )
            phase_indices = [phase_indices_cfg[0], phase_indices_cfg[1]]
            plotter.plotAction(
                np.array(spec.get("Tmin_GeV", 0), dtype=float),
                np.array(spec.get("Tmax_GeV", 0), dtype=float),
                phase_indices,
                np.array(spec.get("n", 100), dtype=int),
            )
        elif key == "potential":
            plotter.plotPotential(
                np.array(spec.get("T_GeV", 0), dtype=float),
                np.array(spec.get("phi_ranges_GeV", [0, 1]), dtype=float),
                np.array(spec.get("n", 100), dtype=int),
            )
        elif key == "phases":
            include_transitions = bool(spec.get("include_transitions?", True))
            plotter.plotPhases(
                Tmin_GeV=np.array(spec.get("Tmin_GeV", np.nan), dtype=float),
                Tmax_GeV=np.array(spec.get("Tmax_GeV", np.nan), dtype=float),
                include_transitions=include_transitions,
                plot_squaresum=spec.get("plot_squaresum", True)
            )
        elif key == "profileV":
            plotter.plotProfileV(
                np.array(spec.get("field_index_1", 0), dtype=int),
                np.array(spec.get("field_index_2", 1), dtype=int),
            )
        elif key == "energy_density":
            plotter.plotEnergyDensity(
                np.array(spec.get("Tmin_GeV", np.nan), dtype=float),
                np.array(spec.get("Tmax_GeV", np.nan), dtype=float),
            )
        elif key == "dofs":
            plotter.plotDOFs(
                np.array(spec.get("Tmin_GeV", np.nan), dtype=float),
                np.array(spec.get("Tmax_GeV", np.nan), dtype=float),
            )
        elif key == "profile":
            plotter.plotProfile()
        elif key == "percolation":
            n_action_cfg = spec.get("n_action", None)
            n_action = None if n_action_cfg is None else int(np.asarray(n_action_cfg, dtype=int))
            plotter.plotPercolation(
                Tmin_GeV=np.array(spec.get("Tmin_GeV", np.nan), dtype=float),
                Tmax_GeV=np.array(spec.get("Tmax_GeV", np.nan), dtype=float),
                n_action=n_action,
            )
        console.print(f"[bold green]Done: {key} plot.[/bold green]")

    def _run_gw_spectrum(self, ctx: dict):
        """Render the GW spectrum for the strongest transition when available."""
        strongest = ctx.get("strongest_transition_observables") or {}
        if not strongest:
            self.mark_gw_skipped("Skipping GW spectrum plot because no valid transition was found.", colour="yellow")
            return
        console.print(f"[bold green]Create the gw_spectrum plot[/bold green]")
        plots.plotGWSpectrum(strongest, showplot=False, foldername=self._outpath, legendfontsize=5)
        console.print(f"[bold green]Done: gw_spectrum plot.[/bold green]")
        self._executed.add("gw_spectrum")


def _write_input_parameters(outpath: str, input_params: dict):
    """Write the per-point input parameter dictionary to disk.

    Numeric entries are rendered with the tabular ``{:e}`` format we use
    throughout. Non-numeric entries (string-valued runtime options like
    ``precision_mode`` or ``percolation_algorithm_mode``) are written as
    plain ``str(value)``; without this fallback the writer would die with
    ``could not convert string to float`` whenever such an option lands in
    the input dict alongside the physical model parameters.
    """
    with open(outpath + "0_Input_params.txt", "w", encoding="utf-8") as file:
        for name, value in input_params.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                file.write("{:<22} {}\n".format(name, str(value)))
                continue
            file.write("{:<22} {:e}\n".format(name, numeric))


def _write_transition_outputs(outpath: str, conf, result: dict) -> float:
    """Write the strongest-transition observables and warning flags to disk."""
    transition_names = list(conf.derived_params.keys())
    strongest = result.get("strongestTransitionObservables", {}) if isinstance(result, dict) else {}
    transition_values = [strongest.get(name, np.nan) for name in transition_names]

    raw_error_value = result.get("error", np.nan) if isinstance(result, dict) else np.nan
    if isinstance(raw_error_value, np.ndarray):
        error_scalar = raw_error_value.flatten()[0] if raw_error_value.size else np.nan
    else:
        error_scalar = raw_error_value

    all_names = transition_names + ["error"]
    all_values = transition_values + [error_scalar]
    entries = {name: value for name, value in zip(all_names, all_values)}

    warning_keys = sorted(
        (k for k in entries if k.upper().startswith("WARNING")),
        key=lambda k: (k.lower(), k),
    )
    error_value = entries.get("error", np.nan)
    numerical_keys = sorted(
        (k for k in entries if k not in warning_keys and k != "error"),
        key=lambda k: (k.lower(), k),
    )

    with open(outpath + "1_All_params.txt", "w", encoding="utf-8") as file:
        for key in numerical_keys:
            file.write("{:<22} {}\n".format(key, _format_numeric_value(key, entries[key])))

        if numerical_keys and (warning_keys or "error" in entries):
            file.write("\n")

        if warning_keys:
            file.write("Warnings:\n")
            warning_labels = [
                key.split("WARNING:", 1)[1] if ":" in key else key
                for key in warning_keys
            ]
            max_len = max(len(label) for label in warning_labels)
            for key in warning_keys:
                label = key.split("WARNING:", 1)[1] if ":" in key else key
                file.write(f"    {label:<{max_len}} {_format_warning_value(entries[key])}\n")

        if warning_keys and "error" in entries:
            file.write("\n")

        if "error" in entries:
            if isinstance(error_value, (float, np.floating)):
                error_str = "-" if not np.isfinite(error_value) else f"{float(error_value):.4e}"
            elif isinstance(error_value, (int, np.integer)):
                error_str = str(int(error_value))
            else:
                error_str = str(error_value)
            file.write("{:<22} {}\n".format("error", error_str))

    return error_scalar


def _write_observability_outputs(outpath: str, result: dict, had_error: bool):
    """Write detector-by-detector SNR and observability summaries to disk."""
    if had_error:
        msg = "Skipping observability output because errors were detected."
        console.print(f"[bold red]{msg}[/bold red]")
        return

    observability = result.get("observability", {}) if isinstance(result, dict) else {}
    if not observability:
        return

    detectors: list[str] = []
    for key in observability.keys():
        detector = None
        if key.endswith("_SNR"):
            detector = key[:-4]
        elif key.endswith("_detectable"):
            detector = key[:-11]
        if detector and detector not in detectors:
            detectors.append(detector)

    with open(outpath + "2_Observability.txt", "w", encoding="utf-8") as file:
        file.write(
            "{:<20} {:<15} {:<31} {:<12}\n".format(
                "Observatory",
                "SNR",
                "Should have been observed already?",
                "Observable?",
            )
        )
        file.write("-" * 90 + "\n")
        for det in detectors:
            snr = observability.get(f"{det}_SNR", float("nan"))
            detectable = observability.get(f"{det}_detectable", False)
            is_existing = det in obs.EXISTING_OBSERVATORIES
            should_str = (
                _format_numeric_value("detectable", detectable)
                if is_existing
                else "-"
            )
            observable_str = (
                _format_numeric_value("detectable", detectable)
                if not is_existing
                else "-"
            )
            file.write(
                "{:<20} {:<15} {:<31} {:<12}\n".format(
                    det,
                    _format_numeric_value("snr", snr),
                    should_str,
                    observable_str,
                )
            )
        file.write("\n")

        logL_keys = [k for k in observability if k.startswith("logL_")]
        printed_keys = set()
        if logL_keys:
            file.write("PTA likelihood summary:\n")
            file.write("{:<16} {:>6} {:<12} {:>12} {:>12} {:>10} {:>18}\n".format(
                "PTA", "Bins", "Variant", "lnL", "Delta lnL", "Sigma", "Within 3 sigma?"
            ))
            file.write("-" * 90 + "\n")
            for label, settings in obs.PTA_LIKELIHOOD_SETTINGS.items():
                bins = int(settings["nbins"])
                display_name = obs.PTA_DISPLAY_NAMES.get(label, label)
                first_row = True
                for variant in obs.PTA_LIKELIHOOD_VARIANTS:
                    lnL_key = f"lnL_{variant}_{label}"
                    delta_key = f"delta_lnL_{variant}_{label}"
                    sigma_key = f"sigma_{variant}_{label}"
                    within_key = f"within_3sigma_{variant}_{label}"
                    printed_keys.update({lnL_key, delta_key, sigma_key, within_key})

                    lnL_val = observability.get(lnL_key, np.nan)
                    delta_val = observability.get(delta_key, np.nan)
                    sigma_val = observability.get(sigma_key, np.nan)
                    within_val = observability.get(within_key, np.nan)

                    lnL_str = f"{lnL_val:.2f}" if np.isfinite(lnL_val) else "nan"
                    delta_str = f"{delta_val:.2f}" if np.isfinite(delta_val) else "nan"
                    sigma_str = f"{sigma_val:.2f}" if np.isfinite(sigma_val) else "nan"
                    within_str = ""
                    if variant == "PTArcade":
                        if isinstance(within_val, (bool, np.bool_)):
                            within_str = str(bool(within_val))
                        elif np.isfinite(within_val):
                            within_str = str(bool(within_val < obs.DELTA_LNL_WITHIN_THRESHOLD))

                    name_col = display_name if first_row else ""
                    file.write("{:<16} {:>6} {:<12} {:>12} {:>12} {:>10} {:>18}\n".format(
                        name_col, bins, variant, lnL_str, delta_str, sigma_str, within_str
                    ))
                    first_row = False
                file.write("\n")

        written_keys = {
            f"{det}_SNR" for det in detectors
        } | {f"{det}_detectable" for det in detectors}
        if logL_keys:
            written_keys.update(logL_keys)
            written_keys.update(printed_keys)

        additional_rows: list[tuple[str, str]] = []
        additional_written: set[str] = set()
        mass_rows: list[tuple[tuple[int, int], str, str, str]] = []
        for key, value in observability.items():
            if not key.endswith("_lin"):
                continue
            base_key = key[:-4]
            additional_written.add(key)
            log_key = f"{base_key}_log10"
            latex_key = f"{base_key}_latex"
            text_key = f"{base_key}_text"
            kind_key = f"{base_key}_kind"
            for meta_key in (log_key, latex_key, text_key, kind_key, base_key):
                if meta_key in observability:
                    additional_written.add(meta_key)

            if base_key.startswith(_MASS_GROUP_PREFIX):
                suffix = base_key[len(_MASS_GROUP_PREFIX):]
                kind_hint = observability.get(kind_key, "boson")
                try:
                    kind_suffix, index_str = suffix.split("_", 1)
                except ValueError:
                    kind_suffix, index_str = kind_hint, "0"
                try:
                    index_val = int(index_str)
                except ValueError:
                    index_val = 0
                kind = str(kind_hint or kind_suffix)
                order = (_MASS_KIND_ORDER.get(kind, 99), index_val)
                text_label = observability.get(text_key)
                latex_label = observability.get(latex_key, text_label or base_key)
                formatted_mass = _format_numeric_value(base_key, value)
                mass_rows.append((order, str(text_label or base_key), str(latex_label), formatted_mass))
                continue

            additional_rows.append((base_key, _format_numeric_value(base_key, value)))

        if additional_rows:
            file.write("Additional information on GW spectrum:\n")
            file.write("{:<30} {:>15}\n".format("Quantity", "Value"))
            file.write("-" * 48 + "\n")
            for name, formatted in additional_rows:
                file.write("{:<30} {:>15}\n".format(name, formatted))
            file.write("\n")

        written_keys.update(additional_written)

        if mass_rows:
            file.write("Zero-temperature mass spectrum (GeV):\n")
            file.write("{:<30} {:>15}\n".format("Particle", "Mass"))
            file.write("-" * 48 + "\n")
            for _order, text_label, latex_label, formatted in sorted(mass_rows, key=lambda item: item[0]):
                display = text_label
                file.write("{:<30} {:>15}\n".format(display, formatted))
            file.write("\n")

        remaining_keys = [k for k in observability if k not in written_keys]

        warning_keys = sorted(
            (k for k in remaining_keys if k.upper().startswith("WARNING")),
            key=lambda k: (k.lower(), k),
        )
        error_key = "error" if "error" in remaining_keys else None
        numerical_keys = sorted(
            (k for k in remaining_keys if k not in warning_keys and k != error_key),
            key=lambda k: (k.lower(), k),
        )

        for key in numerical_keys:
            file.write("{:<22} {}\n".format(key, _format_numeric_value(key, observability.get(key))))

        if numerical_keys and (warning_keys or error_key):
            file.write("\n")

        if warning_keys:
            file.write("Warnings:\n")
            warning_labels = [
                key.split("WARNING:", 1)[1] if ":" in key else key
                for key in warning_keys
            ]
            max_len = max(len(label) for label in warning_labels)
            for key in warning_keys:
                label = key.split("WARNING:", 1)[1] if ":" in key else key
                file.write(f"    {label:<{max_len}} {_format_warning_value(observability.get(key))}\n")

        if warning_keys and error_key:
            file.write("\n")

        if error_key:
            value = observability.get(error_key)
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                value_str = "-"
            elif isinstance(value, (float, np.floating)):
                value_str = f"{float(value):.4e}"
            elif isinstance(value, (int, np.integer)):
                value_str = str(int(value))
            else:
                value_str = str(value)
            file.write("{:<22} {}\n".format("error", value_str))


def single(conf, verbose: bool = False):
    """Evaluate a single parameter point and generate the requested plots."""
    outpath = conf.output_path
    Path(outpath).mkdir(parents=True, exist_ok=True)

    input_params = {key: val for key, val in conf.params.items()}
    plot_plan = _build_plot_plan(conf)
    plot_executor = PlotExecutor(conf, input_params, plot_plan, outpath, verbose=verbose)

    potential = load_potential(conf.modelfile, conf.potential_name)
    errorlogger = Logger(outpath + "error.log")
    resultlogger = Logger(outpath + "result.log")

    stage_callback = (
        plot_executor.stage_callback
        if (plot_plan.ordered_keys or plot_plan.wants_gw_spectrum)
        else None
    )

    context, error = _compute_single_point(
        input_params,
        potential,
        conf.timeout,
        verbose,
        include_smbhb=False,
        max_stage=4,
        stage_callback=stage_callback,
    )

    if error is not None:
        result = _handle_single_point_error(error, input_params, errorlogger, context)
        plot_executor.mark_gw_skipped(
            "Skipping GW spectrum plot because errors were detected before the spectrum stage."
        )
    else:
        result = _build_result_from_context(context)
        observability = result.get("observability", {})
        line = str(input_params) + ":" + str(observability)
        resultlogger.log(line + "\n")

    console.print("[bold green]Phase transition analysis done. Saving the results...[/bold green]")
    _write_input_parameters(outpath, input_params)

    error_scalar = _write_transition_outputs(outpath, conf, result)
    had_error = not np.isnan(error_scalar)
    _write_observability_outputs(outpath, result, had_error)
