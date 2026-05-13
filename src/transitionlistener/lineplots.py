"""Helpers to visualise line scans of TransitionListener observables.

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
from matplotlib.ticker import FixedLocator, FuncFormatter
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
    """Apply common styling to the axes of a line plot."""
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


def _style_log10_y_axis(ax: plt.Axes) -> None:
    """Display y-axis values (stored in log10 space) as powers of ten."""
    y0, y1 = ax.get_ylim()
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return
    lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)

    major_start = int(np.floor(lo))
    major_end = int(np.ceil(hi))
    major = np.arange(major_start, major_end + 1, dtype=float)
    if major.size == 0:
        major = np.array([np.round(0.5 * (lo + hi))], dtype=float)

    minor: list[float] = []
    for exp in range(major_start - 1, major_end + 1):
        for digit in range(2, 10):
            val = exp + np.log10(digit)
            if lo <= val <= hi:
                minor.append(val)

    ax.yaxis.set_major_locator(FixedLocator(major))
    if minor:
        ax.yaxis.set_minor_locator(FixedLocator(minor))
    ax.yaxis.set_major_formatter(FuncFormatter(_pow10_tick_formatter))
    ax.tick_params(axis="y", which="major", length=3.5)
    ax.tick_params(axis="y", which="minor", length=2.0)


def _show_every_second_major_ylabel(ax: plt.Axes) -> None:
    """Keep all major ticks but label only every second one."""
    for idx, tick in enumerate(ax.yaxis.get_major_ticks()):
        visible = (idx % 2 == 0)
        tick.label1.set_visible(visible)
        # Keep right-side ticks but avoid a second labeled y-axis.
        tick.label2.set_visible(False)


def _finish_plot(fig : plt.Figure, ax: plt.Axes, x_plot_name: str, scale: str,
                 x: np.ndarray, title: str, save_path: str):
    """Configure the axes of a line plot and save the figure."""
    _prepare_axes(ax)
    ax.set_xlabel(x_plot_name)
    if scale == 'log':
        ax.set_xscale('log')
    ax.set_xlim(x[0], x[-1])
    ax.set_title(title)
    add_TL_logo(loc="upper right", ax=ax)
    plt.savefig(save_path)
    plt.close(fig)


def _plot_scan_series(ax: plt.Axes, x: np.ndarray, values: np.ndarray, *,
                      show_points: bool = True, **kwargs):
    """Draw scan curves with small markers so isolated finite points are visible."""
    style = dict(
        marker="o" if show_points else None,
        markersize=2.4,
        linewidth=1.2,
        zorder=2,
    )
    style.update(kwargs)
    ax.plot(x, values, **style)


class plot1dData(GW_Sensitivity_Data):
    """Render one-dimensional scan outputs as line plots and diagnostic summaries."""

    def __init__(
        self,
        data_dict_array,
        x,
        scale,
        x_plot_name,
        foldername,
        derived_params_names,
        derived_params_plot_names,
        overview_title,
        show_scan_points=True,
    ):
        """Cache the scan data and prepare intermediate arrays for plotting.

        Parameters
        ----------
        data_dict_array : Sequence[dict]
            List of dictionaries produced by ``run_TL`` along the scan axis.
        x : numpy.ndarray
            Sample points of the scan axis.
        scale : str
            Axis scaling mode, either ``\"lin\"`` or ``\"log\"``.
        x_plot_name : str
            Label for the horizontal axis (LaTeX strings supported).
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
        self.x_plot_name = x_plot_name
        self.derived_params_names = derived_params_names
        self.derived_params_plot_names = derived_params_plot_names
        self.overview_title = overview_title
        self.show_scan_points = show_scan_points
        self.x = x
        self.N = len(x)
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
        n = self.N
        SNR = np.full((n, len(self.det_names)), np.nan)
        detectable = np.full((n, len(self.det_names)), False)
        logL = np.full((n, len(self.PTA_names)), np.nan)

        for i in range(n):
            if "observability" in self.data_dict_array[i]:
                obs_dict = self.data_dict_array[i]["observability"]
                for k, dn in enumerate(self.det_names):
                    if dn + "_SNR" in obs_dict:
                        SNR[i, k] = _as_scalar_or_nan(
                            obs_dict[dn + "_SNR"]
                        )
                    if dn + "_detectable" in obs_dict:
                        detectable[i, k] = bool(
                            _as_scalar_or_nan(
                                obs_dict[dn + "_detectable"]
                            )
                        )
                    else:
                        detectable[i, k] = _detectable_from_snr(
                            SNR[i, k],
                            self.det[dn]["thr"],
                        )
                for l, pl in enumerate(self.PTA_labels):
                    if "logL_" + pl in obs_dict:
                        logL[i, l] = _as_scalar_or_nan(
                            obs_dict["logL_" + pl]
                        )
        return SNR, detectable, logL
    
    def retrieve_derived_params(self):
        """Collect derived parameters and error codes from the scan output."""
        n = self.N
        derived_params = dict()
        for name in self.derived_params_names:
            derived_params[name] = np.full(n, np.nan, dtype=float)
        derived_params["error"] = np.full(n, np.nan, dtype=float)

        for i in range(n):
            for name in self.derived_params_names:
                if "strongestTransitionObservables" in self.data_dict_array[i]:
                    if name in self.data_dict_array[i]["strongestTransitionObservables"]:
                        derived_params[name][i] = _as_scalar_or_nan(
                            self.data_dict_array[i]["strongestTransitionObservables"][name]
                        )
            if "error" in self.data_dict_array[i]:
                derived_params["error"][i] = _as_scalar_or_nan(
                    self.data_dict_array[i]["error"]
                )
        return derived_params
    
    def retrieve_add_spec_info(self):
        """Gather additional observability metadata that is not an SNR or log-likelihood."""
        n = self.N

        add_spec_names = set()
        latex_labels: dict[str, str | None] = {}
        text_labels: dict[str, str | None] = {}
        kind_labels: dict[str, str | None] = {}
        for i in range(n):
            obs_dict = self.data_dict_array[i].get("observability", {})
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

        add_spec_info_lin = {name: np.full(n, np.nan, dtype=float) for name in add_spec_names}
        add_spec_info_log10 = {name: np.full(n, np.nan, dtype=float) for name in add_spec_names}

        for i in range(n):
            obs_dict = self.data_dict_array[i].get("observability", {})
            for name in add_spec_names:
                lin_key = f"{name}_lin"
                log_key = f"{name}_log10"
                if lin_key in obs_dict:
                    add_spec_info_lin[name][i] = _as_scalar_or_nan(obs_dict[lin_key])
                if log_key in obs_dict:
                    add_spec_info_log10[name][i] = _as_scalar_or_nan(obs_dict[log_key])
                if name not in latex_labels or latex_labels[name] is None:
                    latex_labels[name] = obs_dict.get(f"{name}_latex", latex_labels.get(name))
                if name not in text_labels or text_labels[name] is None:
                    text_labels[name] = obs_dict.get(f"{name}_text", text_labels.get(name))
                if name not in kind_labels or kind_labels[name] is None:
                    kind_labels[name] = obs_dict.get(f"{name}_kind", kind_labels.get(name))

        return {
            "lin": add_spec_info_lin,
            "log10": add_spec_info_log10,
            "latex": {name: latex_labels.get(name, name) for name in add_spec_names},
            "text": {name: text_labels.get(name) for name in add_spec_names},
            "kind": {name: kind_labels.get(name) for name in add_spec_names},
            "groups": groups,
        }
    
    def plot_SNRs(self):
        """Generate line plots of the signal-to-noise ratio for every detector."""
        for dn in self.det_names:
            detector_index = self.det_names.index(dn)
            SNR = self.SNR[:, detector_index]
            thr = self.det[dn]["thr"]
            
            fig, ax = plt.subplots()
            _plot_scan_series(ax, self.x, SNR, show_points=self.show_scan_points)
            ax.axhline(thr, color='red', linestyle='--', zorder=1)
            ax.set_yscale('log')
            _finish_plot(fig, ax, self.x_plot_name, self.scale, self.x,
                R"SNR for "+self.det[dn]["plot_name"], self.foldername+"SNR_"+dn+".pdf")
        
    def plot_logLs(self):
        """Generate line plots of the PTA log-likelihood for every collaboration."""
        for pl in self.PTA_labels:
            detector_index = self.PTA_labels.index(pl)
            logL = self.logL[:, detector_index]
            fig, ax = plt.subplots()
            _plot_scan_series(ax, self.x, logL, show_points=self.show_scan_points)
            _finish_plot(fig, ax, self.x_plot_name, self.scale, self.x,
                R"$\log L$ for "+pl, self.foldername+"logL_"+pl+".pdf")
            
    def save_SNRs(self):
        """Export the log10 SNR array as plain-text tables."""
        for dn in self.det_names:
            detector_index = self.det_names.index(dn)
            np.seterr(invalid="ignore")
            logSNR = np.log10(self.SNR[:, detector_index])
            np.seterr(invalid="warn")
            np.savetxt(self.foldername+"log_SNR_"+dn+".txt", logSNR)
    
    def save_logLs(self):
        """Export the PTA log-likelihood arrays as plain-text tables."""
        for i, pl in enumerate(self.PTA_labels):
            logL = self.logL[:, i]
            np.savetxt(self.foldername+"log_L_"+pl+".txt", logL)
            
    def plot_add_infos(self):
        """Visualise auxiliary observability quantities along the scan axis."""
        add_spec_info_lin = self.add_spec_info["lin"]
        add_spec_info_log10 = self.add_spec_info["log10"]
        add_spec_info_latex = self.add_spec_info["latex"]

        mass_group = {
            member.get("name")
            for member in self.add_spec_info.get("groups", {}).get("mass_spectrum_T0", [])
        }

        for pname, values in add_spec_info_lin.items():
            if pname in mass_group:
                continue
            if np.all(np.isnan(values)):
                continue
            fig, ax = plt.subplots()
            _plot_scan_series(ax, self.x, values, show_points=self.show_scan_points)
            _finish_plot(fig, ax, self.x_plot_name, self.scale, self.x,
                add_spec_info_latex.get(pname, pname),
                self.foldername+"Add_info_"+pname+".pdf")

            log_values = add_spec_info_log10.get(pname)
            if log_values is None or np.all(np.isnan(log_values)):
                continue
            fig, ax = plt.subplots()
            _plot_scan_series(ax, self.x, log_values, show_points=self.show_scan_points)
            _finish_plot(fig, ax, self.x_plot_name, self.scale, self.x,
                r"$\log_{10}$ " + add_spec_info_latex.get(pname, pname),
                self.foldername+"Add_info_log10_"+pname+".pdf")

    def plot_mass_spectrum(self):
        """Render the zero-temperature mass spectrum as per-particle plots."""
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
            pname = member.get("name")
            index = member.get("index", 0)
            kind = member.get("kind", "boson")
            latex_label = latex_labels.get(pname, pname)
            math_label = _math_content(latex_label)
            text_label = text_labels.get(pname)
            file_stub = _safe_component(text_label, f"{kind}_{index:02d}")

            linear_title = rf"${math_label} / \text{{GeV}}$"
            log_title = rf"$\log_{{10}} {math_label} / \text{{GeV}}$"

            lin_values = self.add_spec_info["lin"].get(pname)
            if lin_values is not None and not np.all(np.isnan(lin_values)):
                fig, ax = plt.subplots()
                _plot_scan_series(ax, self.x, lin_values, show_points=self.show_scan_points)
                _finish_plot(
                    fig,
                    ax,
                    self.x_plot_name,
                    self.scale,
                    self.x,
                    linear_title,
                    self.foldername + f"Mass_spectrum_T0_lin_{file_stub}.pdf",
                )

            log_values = self.add_spec_info["log10"].get(pname)
            if log_values is not None and not np.all(np.isnan(log_values)):
                fig, ax = plt.subplots()
                _plot_scan_series(ax, self.x, log_values, show_points=self.show_scan_points)
                _finish_plot(
                    fig,
                    ax,
                    self.x_plot_name,
                    self.scale,
                    self.x,
                    log_title,
                    self.foldername + f"Mass_spectrum_T0_log10_{file_stub}.pdf",
                )

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

        for pname, values in add_spec_info_lin.items():
            if pname in mass_group:
                continue
            np.savetxt(self.foldername+"Add_info_lin_"+pname+".txt", values)
            if pname in add_spec_info_log10:
                np.savetxt(
                    self.foldername+"Add_info_log10_"+pname+".txt",
                    add_spec_info_log10[pname],
                )

    def plot_log_params(self):
        """Plot the logarithmic version of the derived parameters."""
        # Log plots
        for name, plotname in zip(self.derived_params_names, self.derived_params_plot_names):
            # skip parameters whose name contains "WARNING"
            if "WARNING" in name or "step" in name:
                continue
            fig, ax = plt.subplots()
            _plot_scan_series(ax, self.x, self.log_params[name], show_points=self.show_scan_points)
            _finish_plot(fig, ax, self.x_plot_name, self.scale, self.x,
                R"$\mathrm{log}_{10}$ "+plotname, self.foldername+"log_plot_"+name+".pdf")

    def plot_lin_params(self):
        """Plot the linear-scale version of the derived parameters."""
        for name, plotname in zip(self.derived_params_names, self.derived_params_plot_names):
            fig, ax = plt.subplots()
            _plot_scan_series(ax, self.x, self.lin_params[name], show_points=self.show_scan_points)
            if "WARNING" in name or "step" in name:
                _finish_plot(fig, ax, self.x_plot_name, self.scale, self.x,
                    plotname, self.foldername+name+".pdf")
            else:
                _finish_plot(fig, ax, self.x_plot_name, self.scale, self.x,
                    plotname, self.foldername+"lin_plot_"+name+".pdf")
            
    def plot_overview(self, overview_detector_name="LISA",
        overview_param_names= ("alpha", "RH", "Treh_SM_GeV"),
        overview_param_plot_names=(R"$\alpha_\mathrm{tot}$",
        R"$RH_\mathrm{perc}$", R"$T^{\mathrm{reh}}_{\mathrm{SM}}$ / $\mathrm{GeV}$"),
        figsize=(6.4, 4.2)):
        """Create a four-panel summary containing three observables and the SNR."""


        # Define parameters to plot (3 GW params + 1 SNR)
        param_names = overview_param_names
        param_plot_names = list(overview_param_plot_names)
        SNR_name = overview_detector_name
        detector_label = str(overview_detector_name).replace("_", " ")
        SNR_plot_name = f"SNR ({detector_label})"

        # Prepare overview parameters: keep Treh linear, others in log10-space.
        lin_overview_params = {"Treh_SM_GeV"}
        overview_params = {}
        for name in param_names:
            if name in lin_overview_params:
                overview_params[name] = self.lin_params[name]
            else:
                overview_params[name] = self.log_params[name]

        # prepare SNR
        detector_index = self.det_names.index(SNR_name)
        SNR = self.SNR[:, detector_index]
        logSNR = np.full_like(SNR, np.nan, dtype=float)
        valid = np.isfinite(SNR) & (SNR > 0)
        logSNR[valid] = np.log10(SNR[valid])
        thr = self.det[SNR_name]["thr"]
        log_thr = np.log10(thr) if np.isfinite(thr) and thr > 0 else np.nan

        # Prepare axes
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]
        ax4 = axs[1, 1]

        # Plot 3 params
        GW_axes = (ax1, ax2, ax3)
        for i, ax in enumerate(GW_axes):
            pname = param_names[i]
            _plot_scan_series(GW_axes[i], self.x, overview_params[pname],
                              show_points=self.show_scan_points)
            _prepare_axes(ax)
            if self.scale == 'log':
                ax.set_xscale('log')
            ax.set_xlim(self.x[0], self.x[-1])
            ax.set_ylabel(param_plot_names[i])
            if pname in lin_overview_params:
                pass
            else:
                _style_log10_y_axis(ax)
            if str(pname) == "alpha":
                _show_every_second_major_ylabel(ax)


        # Plot SNR
        _plot_scan_series(ax4, self.x, logSNR, show_points=self.show_scan_points)
        if np.isfinite(log_thr):
            ax4.axhline(log_thr, color='red', linestyle='--', zorder=1)
        _prepare_axes(ax4)
        if self.scale == 'log':
            ax4.set_xscale('log')
        ax4.set_xlim(self.x[0], self.x[-1])
        ax4.set_ylabel(SNR_plot_name)
        _style_log10_y_axis(ax4)

        # Keep x-labels only in the bottom row.
        ax1.set_xlabel("")
        ax2.set_xlabel("")
        ax1.tick_params(axis="x", labelbottom=False)
        ax2.tick_params(axis="x", labelbottom=False)
        ax3.set_xlabel(self.x_plot_name)
        ax4.set_xlabel(self.x_plot_name)
        
        if str(self.overview_title).strip():
            fig.suptitle(self.overview_title)
        plt.tight_layout()

        # Savefig
        add_TL_logo(loc="upper right", ax=ax4)
        plt.savefig(self.foldername+"overview_plot.pdf")
        plt.close(fig)
        
    def plot_errors(self):
        """Plot the error-code trace along the scan axis."""
        errors = self.derived_params_array_dict["error"]
        fig, ax = plt.subplots()
        ax.plot(self.x, errors, "X")
        _finish_plot(fig, ax, self.x_plot_name, self.scale, self.x,
            R"Errorcodes", self.foldername+"Errors.pdf")
