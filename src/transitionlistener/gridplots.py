"""Helpers to visualise grid scans of TransitionListener observables.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import re

from .observability import GW_Sensitivity_Data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.ticker import FuncFormatter, MaxNLocator
from .colors import *
from .plot_settings import plot_settings as TL_plot_settings
from .plot_settings import add_TL_logo
from .plot_settings import TLcmap

_MASS_GROUP_PREFIX = "mass_spectrum_T0_"
_MASS_KIND_ORDER = {"boson": 0, "fermion": 1}

plt.rcParams.update(TL_plot_settings)


def _as_scalar_or_nan(value) -> float:
    """Convert an arbitrary value to a scalar float when possible."""
    try:
        arr = np.asarray(value)
    except Exception:
        return np.nan
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        try:
            return float(np.squeeze(arr))
        except Exception:
            return np.nan
    return np.nan


def _detectable_from_snr(snr: float, threshold: float) -> bool:
    """Infer detectability from SNR when legacy scan exports omit the flag."""
    return bool(
        np.isfinite(snr)
        and np.isfinite(threshold)
        and snr > 0.0
        and threshold > 0.0
        and snr > threshold
    )


def _prepare_axes(ax : plt.Axes):
    """Apply common styling to the axes of a grid plot."""
    ax.tick_params(axis='both', which='both', 
                   direction="in", width=0.5, zorder=1000)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    for axis in ["top", 'bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
        ax.spines[axis].set_zorder(10000)


def _pow10_tick_formatter(value: float, _pos: float) -> str:
    """Format a log10-space value as 10^x."""
    if not np.isfinite(value):
        return ""
    rounded = int(np.round(value))
    if abs(value - rounded) < 1e-8:
        return rf"$10^{{{rounded}}}$"
    return rf"$10^{{{value:.1f}}}$"


def _style_log_colorbar(cbar) -> None:
    """Display colorbar ticks (stored in log10 space) as powers of ten."""
    cbar.locator = MaxNLocator(nbins=5, integer=True)
    cbar.formatter = FuncFormatter(_pow10_tick_formatter)
    cbar.update_ticks()
        
def _finish_plot(fig : plt.Figure, ax: plt.Axes, xy_plot_names: tuple[str, str],
                 im : QuadMesh, scale: str, title: str, save_path: str):
    """Configure the axes of a grid plot and save the figure."""
    _prepare_axes(ax)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xy_plot_names[0])
    ax.set_ylabel(xy_plot_names[1])
    if scale == 'linlog':
        ax.set_yscale('log')
    if scale == 'loglin':
        ax.set_xscale('log')
    if scale == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')
    add_TL_logo(loc="upper right", ax=ax)
    plt.savefig(save_path)
    plt.close(fig)

        
class plot2dData(GW_Sensitivity_Data):
    """Render two-dimensional scan outputs as heatmaps and diagnostic plots."""

    def __init__(
        self,
        data_dict_array,
        x,
        y,
        scale,
        xy_plot_names,
        foldername,
        derived_params_names,
        derived_params_plot_names,
        overview_title,
        show_scan_points=True,
    ):
        """Cache the scan data and pre-compute helper arrays for plotting.

        Parameters
        ----------
        data_dict_array : numpy.ndarray
            ``(N, N)`` grid of dictionaries produced by ``run_TL``.
        x, y : numpy.ndarray
            Grid coordinates for the scan axes.
        scale : str
            Axis scaling mode. One of ``\"linlin\"``, ``\"linlog\"``, ``\"loglin\"`` or
            ``\"loglog\"``.
        xy_plot_names : tuple[str, str]
            Labels for the x and y axes (LaTeX strings supported).
        foldername : str
            Output directory where exported figures and tables are written.
        derived_params_names : Iterable[str]
            Keys of derived observables to extract from ``strongestTransitionObservables``.
        derived_params_plot_names : Iterable[str]
            Human-readable labels corresponding to ``derived_params_names``.
        overview_title : str
            Title used in the overview figure.
        """
        super().__init__()
        self.foldername = foldername
        self.data_dict_array = data_dict_array
        self.scale = scale
        self.xy_plot_names = xy_plot_names
        self.derived_params_names = derived_params_names
        self.derived_params_plot_names = derived_params_plot_names
        self.overview_title = overview_title
        self.show_scan_points = show_scan_points

        if self.scale == "linlin":
            self.x = np.append(x, 2*x[-1] - x[-2])
            self.y = np.append(y, 2*y[-1] - y[-2])
        elif self.scale == "linlog":
            self.x = np.append(x, 2*x[-1] - x[-2])
            self.y = np.append(
                y, y[-1] + 10**(np.log10(y[-1]) - np.log10(y[0])) * (y[1] - y[0]))
        elif self.scale == "loglin":
            self.x = np.append(
                x, x[-1] + 10**(np.log10(x[-1]) - np.log10(x[0])) * (x[1] - x[0]))
            self.y = np.append(y, 2*y[-1] - y[-2])
        elif self.scale == "loglog":
            self.x = np.append(
                x, x[-1] + 10**(np.log10(x[-1]) - np.log10(x[0])) * (x[1] - x[0]))
            self.y = np.append(
                y, y[-1] + 10**(np.log10(y[-1]) - np.log10(y[0])) * (y[1] - y[0]))
            
        self.gridsize = len(x)
        self.SNR, self.detectable, self.logL = self.retrieve_observability_data()
        self.derived_params_array_dict = self.retrieve_derived_params()
        self.add_spec_info = self.retrieve_add_spec_info()
        
        self.log_params = dict()
        self.lin_params = dict()

        for name in self.derived_params_names:
            with np.errstate(invalid="ignore", divide="ignore"):
                self.log_params[name] = np.log10(self.derived_params_array_dict[name])
                self.lin_params[name] = self.derived_params_array_dict[name]

    def retrieve_observability_data(self):
        """Extract detector SNRs, detectability flags and PTA log-likelihoods."""
        n = self.gridsize
        SNR = np.full((n, n, len(self.det_names)), np.nan)
        detectable = np.full((n, n, len(self.det_names)), False)
        logL = np.full((n, n, len(self.PTA_names)), np.nan)

        for i in range(n):
            for j in range(n):
                if "observability" in self.data_dict_array[i, j]:
                    obs_dict = self.data_dict_array[i, j]["observability"]
                    for k, dn in enumerate(self.det_names):
                        if dn + "_SNR" in obs_dict:
                            SNR[i, j, k] = _as_scalar_or_nan(
                                obs_dict[dn + "_SNR"]
                            )
                        if dn + "_detectable" in obs_dict:
                            detectable[i, j, k] = bool(
                                _as_scalar_or_nan(
                                    obs_dict[dn + "_detectable"]
                                )
                            )
                        else:
                            detectable[i, j, k] = _detectable_from_snr(
                                SNR[i, j, k],
                                self.det[dn]["thr"],
                            )
                    for l, pl in enumerate(self.PTA_labels):
                        if "logL_" + pl in obs_dict:
                            logL[i, j, l] = _as_scalar_or_nan(
                                obs_dict["logL_" + pl]
                            )
        return SNR, detectable, logL
    
    def retrieve_derived_params(self):
        """Collect derived parameters and error codes from the scan output."""
        n = self.gridsize
        derived_params = dict()
        for name in self.derived_params_names:
            derived_params[name] = np.full((n, n), np.nan, dtype=float)
        derived_params["error"] = np.full((n, n), np.nan, dtype=float)

        for i in range(n):
            for j in range(n):
                for name in self.derived_params_names:
                    if "strongestTransitionObservables" in self.data_dict_array[i, j]:
                        if name in self.data_dict_array[i, j]["strongestTransitionObservables"]:
                            derived_params[name][i, j] = _as_scalar_or_nan(
                                self.data_dict_array[i, j]["strongestTransitionObservables"][name]
                            )
                if "error" in self.data_dict_array[i, j]:
                    derived_params["error"][i, j] = _as_scalar_or_nan(
                        self.data_dict_array[i, j]["error"]
                    )
        return derived_params
    
    def retrieve_add_spec_info(self):
        """Gather additional observability metadata that is not an SNR or log-likelihood."""
        n = self.gridsize

        # Identify which auxiliary quantities are available across the grid.
        add_spec_names = set()
        latex_labels: dict[str, str | None] = {}
        text_labels: dict[str, str | None] = {}
        kind_labels: dict[str, str | None] = {}
        for i in range(n):
            for j in range(n):
                obs_dict = self.data_dict_array[i, j].get("observability", {})
                for key in obs_dict:
                    if any(marker in key for marker in ["SNR", "logL", "detectable"]):
                        continue
                    if key.endswith("_lin"):
                        base_name = key[:-4]
                        add_spec_names.add(base_name)
                        latex_labels.setdefault(base_name, obs_dict.get(f"{base_name}_latex"))
                        text_labels.setdefault(base_name, obs_dict.get(f"{base_name}_text"))
                        kind_labels.setdefault(base_name, obs_dict.get(f"{base_name}_kind"))

        mass_group_members: list[tuple[int, int, str, str | None]] = []
        for name in add_spec_names:
            if not name.startswith(_MASS_GROUP_PREFIX):
                continue
            suffix = name[len(_MASS_GROUP_PREFIX):]
            parts = suffix.split("_", 1)
            if len(parts) != 2:
                continue
            kind, index_str = parts
            try:
                index = int(index_str)
            except ValueError:
                continue
            order = _MASS_KIND_ORDER.get(kind, 99)
            mass_group_members.append((order, index, name, kind))

        groups = {}
        if mass_group_members:
            groups["mass_spectrum_T0"] = [
                {"name": name, "index": index, "kind": kind}
                for order, index, name, kind in sorted(mass_group_members)
            ]

        add_spec_info_lin = {
            name: np.full((n, n), np.nan, dtype=float) for name in add_spec_names
        }
        add_spec_info_log10 = {
            name: np.full((n, n), np.nan, dtype=float) for name in add_spec_names
        }

        for i in range(n):
            for j in range(n):
                obs_dict = self.data_dict_array[i, j].get("observability", {})
                for name in add_spec_names:
                    lin_key = f"{name}_lin"
                    log_key = f"{name}_log10"
                    if lin_key in obs_dict:
                        add_spec_info_lin[name][i, j] = _as_scalar_or_nan(obs_dict[lin_key])
                    if log_key in obs_dict:
                        add_spec_info_log10[name][i, j] = _as_scalar_or_nan(obs_dict[log_key])
                    if name not in latex_labels or latex_labels[name] is None:
                        latex_labels[name] = obs_dict.get(f"{name}_latex", latex_labels.get(name))
                    if name not in text_labels or text_labels[name] is None:
                        text_labels[name] = obs_dict.get(f"{name}_text", text_labels.get(name))
                    if name not in kind_labels or kind_labels[name] is None:
                        kind_labels[name] = obs_dict.get(f"{name}_kind", kind_labels.get(name))

        add_spec_info = {
            "lin": add_spec_info_lin,
            "log10": add_spec_info_log10,
            "latex": {name: latex_labels.get(name, name) for name in add_spec_names},
            "text": {name: text_labels.get(name) for name in add_spec_names},
            "kind": {name: kind_labels.get(name) for name in add_spec_names},
            "groups": groups,
        }
        return add_spec_info

    def plot_SNRs(self):
        """Save heatmaps of the signal-to-noise ratio for every detector."""
        for dn in self.det_names:
            detector_index = self.det_names.index(dn)
            logSNR = np.log10(self.SNR[:, :, detector_index])
            detectable = self.detectable[:, :, detector_index]
            blackdots = np.full_like(logSNR, 1)
            blackdots[~detectable] = 0
            logSNR[~detectable] = np.nan

            fig, ax = plt.subplots()
            ax.pcolormesh(self.x, self.y, blackdots.T, cmap="gist_gray")
            im = ax.pcolormesh(self.x, self.y, logSNR.T, cmap=TLcmap)
            _finish_plot(fig, ax, self.xy_plot_names,
                         im, self.scale,
                         R"$\mathrm{log}_{10}$ SNR for "+self.det[dn]["plot_name"],
                         self.foldername+"SNR_"+dn+".pdf")
        
    def plot_logLs(self):
        """Save heatmaps of the PTA log-likelihood for every pulsar collaboration."""
        for pl in self.PTA_labels:
            detector_index = self.PTA_labels.index(pl)
            logL = self.logL [:, :, detector_index]
            fig, ax = plt.subplots()
            im = ax.pcolormesh(self.x, self.y, logL.T, cmap=TLcmap)
            _finish_plot(fig, ax, self.xy_plot_names,
                         im, self.scale,
                         R"$\log L$ for "+pl,
                         self.foldername+"logL_"+pl+".pdf")
            
    def save_SNRs(self):
        """Export the log10 SNR grids as plain-text tables."""
        for dn in self.det_names:
            detector_index = self.det_names.index(dn)
            np.seterr(invalid="ignore")
            logSNR = np.log10(self.SNR[:, :, detector_index])
            np.seterr(invalid="warn")
            np.savetxt(self.foldername+"log_SNR_"+dn+".txt", logSNR)
    
    def save_logLs(self):
        """Export the PTA log-likelihood grids as plain-text tables."""
        for i, pl in enumerate(self.PTA_labels):
            logL = self.logL[:, :, i]
            np.savetxt(self.foldername+"log_L_"+pl+".txt", logL)
            
    def plot_add_infos(self):
        """Visualise any auxiliary observability fields that were stored."""
        # This method plots both the additional spectra information and the error keys
        add_spec_info_lin = self.add_spec_info["lin"]
        add_spec_info_log10 = self.add_spec_info["log10"]
        add_spec_info_latex = self.add_spec_info["latex"]
        mass_group = {
            member.get("name")
            for member in self.add_spec_info.get("groups", {}).get("mass_spectrum_T0", [])
        }

        for pname in add_spec_info_lin.keys():
            if pname in mass_group:
                continue
            param_lin = add_spec_info_lin[pname]
            param_log10 = add_spec_info_log10.get(pname, np.full_like(param_lin, np.nan))
            param_latex = add_spec_info_latex.get(pname, pname)

            if np.all(np.isnan(param_lin)) and np.all(np.isnan(param_log10)):
                continue
            fig, ax = plt.subplots()
            im = ax.pcolormesh(self.x, self.y, param_lin.T, cmap=TLcmap)
            _finish_plot(fig, ax, self.xy_plot_names,
                         im, self.scale,
                         param_latex,
                         self.foldername+"Add_info_"+pname+".pdf")

            if not np.all(np.isnan(param_log10)):
                fig, ax = plt.subplots()
                im = ax.pcolormesh(self.x, self.y, param_log10.T, cmap=TLcmap)
                _finish_plot(fig, ax, self.xy_plot_names,
                             im, self.scale,
                             r"$\log_{10}$" + param_latex,
                             self.foldername+"Add_info_log10_"+pname+".pdf")

    def plot_mass_spectrum(self):
        """Render the zero-temperature mass spectrum as per-particle grids."""
        groups = self.add_spec_info.get("groups", {})
        mass_group = groups.get("mass_spectrum_T0", [])
        if not mass_group:
            return

        latex_labels = self.add_spec_info["latex"]
        text_labels = self.add_spec_info.get("text", {})

        def _math_content(label: str) -> str:
            if label.startswith("$") and label.endswith("$") and len(label) >= 2:
                return label[1:-1]
            return label

        def _safe_component(component: str | None, fallback: str) -> str:
            if not component:
                return fallback
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", component.strip())
            safe = safe.strip("._")
            return safe or fallback

        for member in mass_group:
            name = member.get("name")
            index = member.get("index", 0)
            kind = member.get("kind", "boson")
            latex_label = latex_labels.get(name, name)
            math_label = _math_content(latex_label)
            text_label = text_labels.get(name)
            file_stub = _safe_component(text_label, f"{kind}_{index:02d}")

            linear_title = rf"${math_label} / \text{{GeV}}$"
            log_title = rf"$\log_{{10}} {math_label} / \text{{GeV}}$"

            lin_values = self.add_spec_info["lin"].get(name)
            if lin_values is not None and not np.all(np.isnan(lin_values)):
                fig, ax = plt.subplots()
                im = ax.pcolormesh(self.x, self.y, lin_values.T, cmap=TLcmap)
                _finish_plot(
                    fig,
                    ax,
                    self.xy_plot_names,
                    im,
                    self.scale,
                    linear_title,
                    self.foldername + f"Mass_spectrum_T0_lin_{file_stub}.pdf",
                )

            log_values = self.add_spec_info["log10"].get(name)
            if log_values is not None and not np.all(np.isnan(log_values)):
                fig, ax = plt.subplots()
                im = ax.pcolormesh(self.x, self.y, log_values.T, cmap=TLcmap)
                _finish_plot(
                    fig,
                    ax,
                    self.xy_plot_names,
                    im,
                    self.scale,
                    log_title,
                    self.foldername + f"Mass_spectrum_T0_log10_{file_stub}.pdf",
                )
            
    def plot_errors(self):
        """Plot the error-code grid released by the CLI."""
        errors = self.derived_params_array_dict["error"]
        fig, ax = plt.subplots()
        im = ax.pcolormesh(self.x, self.y, errors.T, cmap=TLcmap)
        
        _finish_plot(fig, ax, self.xy_plot_names,
                     im, self.scale,
                     "Errorcodes",
                     self.foldername+"Errors.pdf")
            
    def save_add_infos(self):
        """Save the additional observability information as plain-text tables."""
        add_spec_info_lin = self.add_spec_info["lin"]
        add_spec_info_log10 = self.add_spec_info["log10"]
        if not add_spec_info_lin:
            return

        mass_group = {
            member.get("name")
            for member in self.add_spec_info.get("groups", {}).get("mass_spectrum_T0", [])
        }

        for pname, grid in add_spec_info_lin.items():
            if pname in mass_group:
                continue
            np.savetxt(self.foldername+"Add_info_lin_"+pname+".txt", grid)
            if pname in add_spec_info_log10:
                np.savetxt(
                    self.foldername+"Add_info_log10_"+pname+".txt",
                    add_spec_info_log10[pname],
                )

    def plot_overview(self, overview_detector_name="LISA",
        overview_param_names= ("alpha", "RH", "Treh_SM_GeV"),
        overview_param_plot_names=(R"$\alpha_\mathrm{tot}$",
        R"$RH_\mathrm{perc}$", R"$T^{\mathrm{reh}}_{\mathrm{SM}}$ / $\mathrm{GeV}$"),
        figsize=None):
        """Create a four-panel summary containing three observables and one SNR."""

        # Define parameters to plot (3 GW params + 1 SNR)
        param_names = overview_param_names
        param_plot_names = list(overview_param_plot_names)
        SNR_name = overview_detector_name
        detector_label = str(overview_detector_name).replace("_", " ")
        SNR_plot_name = f"SNR for {detector_label}"

        # prepare 3 params
        log_overview_params = dict()
        for name in param_names:
            log_overview_params[name] = self.log_params[name]

        # prepare SNR
        detector_index = self.det_names.index(SNR_name)
        logSNR = np.log10(self.SNR[:, :, detector_index])
        detectable = self.detectable[:, :, detector_index]
        blackdots = np.full_like(logSNR, 1)
        blackdots[~detectable] = 0
        logSNR[~detectable] = np.nan

        # Prepare axes
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]
        ax4 = axs[1, 1]
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlabel(self.xy_plot_names[0])
            ax.set_ylabel(self.xy_plot_names[1])
            if self.scale == 'linlog':
                ax.set_yscale('log')
            if self.scale == 'loglin':
                ax.set_xscale('log')
            if self.scale == 'loglog':
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.tick_params(axis='both', which='both', direction="in", width=0.5, zorder=1000)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            for axis in ["top", 'bottom','left','right']:
                ax.spines[axis].set_linewidth(0.5)
                ax.spines[axis].set_zorder(10000)

        # Plot 3 params
        GW_axes = (ax1, ax2, ax3)
        for i, ax in enumerate(GW_axes):
            im = GW_axes[i].pcolormesh(
                self.x, self.y, log_overview_params[param_names[i]].T, cmap=TLcmap)
            cbar = fig.colorbar(im, ax=GW_axes[i])
            _style_log_colorbar(cbar)
            GW_axes[i].set_title(param_plot_names[i])

        # Plot SNR
        ax4.set_title(SNR_plot_name)
        ax4.pcolormesh(self.x, self.y, blackdots.T, cmap="gist_gray")
        im4 = ax4.pcolormesh(self.x, self.y, logSNR.T, cmap=TLcmap)
        cbar4 = fig.colorbar(im4, ax=ax4)
        _style_log_colorbar(cbar4)
        fig.suptitle(self.overview_title)
        plt.tight_layout()

        # Savefig
        add_TL_logo(loc="upper right", ax=ax4)
        plt.savefig(self.foldername+"overview_plot.pdf")
        plt.close(fig)

    def plot_log_params(self):
        """Plot the logarithmic version of every derived parameter."""
        # Log plots
        for name, plotname in zip(self.derived_params_names, self.derived_params_plot_names):
            # skip parameters whose name contains "WARNING"
            if "WARNING" in name or "step" in name:
                continue
            
            fig, ax = plt.subplots()
            im = ax.pcolormesh(self.x, self.y, self.log_params[name].T, cmap=TLcmap)
            _finish_plot(fig, ax, self.xy_plot_names,
                        im, self.scale, R"$\mathrm{log}_{10}$ "+plotname,
                        self.foldername+"log_plot_"+name+".pdf")

    def plot_lin_params(self):
        """Plot the linear-scale version of every derived parameter."""
        for name, plotname in zip(self.derived_params_names, self.derived_params_plot_names):
            fig, ax = plt.subplots()
            im = ax.pcolormesh(self.x, self.y, self.lin_params[name].T, cmap=TLcmap)
            if "WARNING" in name or "step" in name:
                _finish_plot(fig, ax, self.xy_plot_names,
                         im, self.scale, plotname,
                         self.foldername+name+".pdf")
            else:
                _finish_plot(fig, ax, self.xy_plot_names,
                             im, self.scale, plotname,
                             self.foldername+"lin_plot_"+name+".pdf")
