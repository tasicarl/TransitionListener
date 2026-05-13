"""Configuration loader for TransitionListener scan YAML files.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import os
import yaml

from rich.panel import Panel
from rich.table import Table

from transitionlistener.config import all_observables
from transitionlistener.helper_functions import import_file, load_potential
from transitionlistener import console

from . import state
from ..observability import GW_Sensitivity_Data


class ScanConfig:
    """Read and store all settings required for a TransitionListener scan."""

    def __init__(self, file: str):
        """Parse the YAML configuration and expose the fields as attributes."""
        with open(file, "r", encoding="utf-8") as cfile:
            config = yaml.safe_load(cfile)

        self.type = config["Scan"]

        state.TIMEOUT = config.get("timeout", 300)
        self.timeout = state.TIMEOUT

        self.modelfile = config["Modelfile"]
        self.potential_name = config["Potential"]
        state.model = import_file(self.modelfile)
        state.potential = load_potential(self.modelfile, self.potential_name)
        self.model_params = self._resolve_model_parameters(state.potential)

        self.derived_params = all_observables.copy()

        self.output_path = config["output_path"]
        if self.output_path[-1] != "/":
            self.output_path += "/"

        self.format = config["format"]
        if self.format not in ["hdf5", "txt"]:
            raise Exception("Output format must be either 'hdf5' or 'txt'")

        self.description = config["description"]
        self.plot_description = config["plot_description"]

        if self.type == "SinglePoint":
            self.params = config["Parameters"]
            self.additional_plots = config.get("additional_plots", {})
            self.plots = None

        elif self.type == "TableScan":
            self.params = config["Parameters"]
            self.scan_params = config["Parameters"]["scan_params"]
            self.other_params = config["Parameters"]["other_params"] or {}
            try:
                self.N = config["Parameters"]["N"]
            except KeyError:
                raise Exception("Number of random points N must be specified.")
            if self.N < 1 or int(self.N) != self.N:
                raise Exception("Number of random points N must be a positive integer.")
            self.table_file = config["Tablefile"]
            if not os.path.isfile(self.table_file):
                raise Exception(f"Table file {self.table_file} does not exist.")

        elif self.type == "RandomScan":
            self.params = config["Parameters"]
            self.scan_params = config["Parameters"]["scan_params"]
            self.other_params = config["Parameters"]["other_params"] or {}
            try:
                self.N = config["Parameters"]["N"]
            except KeyError:
                raise Exception("Number of random points N must be specified.")
            if self.N < 1 or int(self.N) != self.N:
                raise Exception("Number of random points N must be a positive integer.")

        elif self.type == "UltranestScan":
            self.params = config["Parameters"]
            self.scan_params = config["Parameters"]["scan_params"]
            self.other_params = config["Parameters"]["other_params"]
            self.transformation_file = config["Parameters"].get("transformation_file", None)
            self.resume_mode = config.get("resume_mode", "resume")
            valid_resume_modes = {True, "overwrite", "subfolder", "resume", "resume-similar"}
            if self.resume_mode not in valid_resume_modes:
                raise ValueError(
                    "For UltranestScan, resume_mode must be one of "
                    f"{sorted(repr(mode) for mode in valid_resume_modes)}; "
                    f"got {self.resume_mode!r}."
                )

        elif self.type == "LineScan":
            self.plots = config["plots"]
            if len(list(config["Parameters"]["line_param"].keys())) != 1:
                raise Exception("Line scan requires exactly one line parameter.")

            self.line_param = list(config["Parameters"]["line_param"].keys())[0]
            if self.line_param not in self.model_params.keys():
                raise Exception(f"Line scan parameter {self.line_param} not found in model parameters.")

            try:
                self.scale = config["Parameters"]["line_param"][self.line_param]["scale"]
                self.minmax = config["Parameters"]["line_param"][self.line_param]["range"]
            except KeyError:
                raise Exception("Line scan parameter must have 'range' and 'scale' specified.")

            self.other_params = config["Parameters"]["other_params"] or {}
            try:
                self.N = config["Parameters"]["N"]
            except KeyError:
                raise Exception("Number of line points N must be specified.")
            if self.N < 2 or int(self.N) != self.N:
                raise Exception("Number of line points N must be an integer >= 2.")

            self.overview_detector_name = config["overview_detector_name"]
            self._validate_overview_detector(self.overview_detector_name)
            self.overview_param_names = config["overview_param_names"]
            self.show_scan_points = config.get("show_scan_points", True)

        elif self.type == "GridScan":
            self.plots = config["plots"]
            grid_keys = list(config["Parameters"]["grid_params"].keys())
            if set(grid_keys) != {"x", "y"} or len(grid_keys) != 2:
                raise Exception("Grid scan requires exactly two grid parameters named 'x' and 'y'.")

            try:
                self.N = config["Parameters"]["N"]
            except KeyError:
                raise Exception("Number of grid points N must be specified.")
            if self.N < 2 or int(self.N) != self.N:
                raise Exception("Number of grid points N must be an integer >= 2.")

            self.minmax = {}
            self.scales = {}
            self.grid_names = {}

            try:
                for key in grid_keys:
                    entry = config["Parameters"]["grid_params"][key]
                    self.minmax[key] = [float(val) for val in entry["range"]] # Support 1e-3
                    self.scales[key] = entry["scale"]
                    self.grid_names[key] = entry["name"]
            except KeyError:
                raise Exception("Grid parameters must have 'name', 'scale' and 'range' specified.")

            self.other_params = config["Parameters"]["other_params"] or {}
            self.overview_detector_name = config["overview_detector_name"]
            self._validate_overview_detector(self.overview_detector_name)
            self.overview_param_names = config["overview_param_names"]
            self.show_scan_points = config.get("show_scan_points", True)

        else:
            raise Exception(f"Unknown scan type {self.type}")

    def _validate_overview_detector(self, detector_name: str) -> None:
        """Ensure the requested overview detector exists in the observability data."""
        available = GW_Sensitivity_Data().det_names
        if detector_name not in available:
            raise ValueError(
                "overview_detector_name must be one of "
                + ", ".join(sorted(available))
                + f"; got '{detector_name}'"
            )

    @staticmethod
    def _resolve_model_parameters(potential):
        """Return the parameter metadata defined by the imported model module."""
        try:
            params = potential.model_parameters
        except Exception:
            raise AttributeError(
                "The provided model does not expose model_parameters either as a module-level "
                "attribute or on the specific_potential class."
            )

        return {name: params[name].copy() for name in params}

    def display(self):
        """Pretty-print the loaded configuration to the console."""
        try:
            summary = Table(show_header=False, box=None, pad_edge=False)
            summary.add_column("Setting", style="cyan")
            summary.add_column("Value")
            summary.add_row("Type", str(self.type))
            summary.add_row("Model", str(self.modelfile))
            summary.add_row("Output", str(self.output_path))
            summary.add_row("Format", str(self.format))
            plots_value = getattr(self, "plots", None)
            if plots_value is not None:
                summary.add_row("Plots", str(plots_value))
            if getattr(self, "description", None):
                summary.add_row("Description", str(self.description))
            console.print(Panel(summary, title="Config", border_style="green"))

            params_tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
            params_tbl.add_column("Parameter", style="cyan")
            params_tbl.add_column("Value")

            if self.type == "SinglePoint":
                for k, v in self.params.items():
                    try:
                        val = f"{float(v):g}"
                    except Exception:
                        val = str(v)
                    params_tbl.add_row(k, val)
            elif self.type == "LineScan":
                pname = self.line_param
                rmin, rmax = self.minmax
                params_tbl.add_row("mode", "Line")
                params_tbl.add_row(pname, f"[{rmin:g}, {rmax:g}] ({self.scale})")
                params_tbl.add_row("N", f"{self.N}")
            elif self.type == "GridScan":
                xname = self.grid_names.get("x", "x")
                yname = self.grid_names.get("y", "y")
                xmin, xmax = self.minmax["x"]
                ymin, ymax = self.minmax["y"]
                xscale = self.scales["x"]
                yscale = self.scales["y"]
                params_tbl.add_row("mode", "Grid")
                params_tbl.add_row("x", f"{xname} [{xmin:g}, {xmax:g}] ({xscale})")
                params_tbl.add_row("y", f"{yname} [{ymin:g}, {ymax:g}] ({yscale})")
                params_tbl.add_row("N per axis", f"{self.N}")
            else:
                if hasattr(self, "scan_params") and isinstance(self.scan_params, dict):
                    for k, spec in self.scan_params.items():
                        try:
                            rmin, rmax = spec.get("range", [None, None])
                            scale = spec.get("scale", "")
                            if rmin is not None and rmax is not None:
                                spec_str = f"[{float(rmin):g}, {float(rmax):g}] ({scale})"
                            else:
                                spec_str = str(spec)
                        except Exception:
                            spec_str = str(spec)
                        params_tbl.add_row(k, spec_str)
                if hasattr(self, "N"):
                    params_tbl.add_row("N", f"{self.N}")
            console.print(Panel(params_tbl, title="Scan parameters", border_style="green"))

            fixed = getattr(self, "other_params", None)
            if fixed:
                fixed_tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
                fixed_tbl.add_column("Fixed Parameter", style="cyan")
                fixed_tbl.add_column("Value")
                for k, v in fixed.items():
                    try:
                        val = f"{float(v):g}"
                    except Exception:
                        val = str(v)
                    fixed_tbl.add_row(k, val)
                console.print(Panel(fixed_tbl, title="Fixed Parameters", border_style="green"))
        except Exception:
            pass
