"""Visualization utilities for TransitionListener potentials, spectra and scans.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.random import choice as _choice

import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedLocator, FixedFormatter)

from transitionlistener import generic_potential, console
from .bubbledynamics import calcAction, Gamma, HubbleParameter, calcPercAndEvolve, energyDensity
from .transitions import Transitions
from .transitionObservables import (
    TransitionObservables as _TransitionObservablesAdaptiveStepSize,
    _extract_instanton_plot_arrays,
    _select_instanton_near_temperature,
)
from .transitionObservables_fixedstep import TransitionObservables as _TransitionObservablesFixedStepSize


def _select_observables_class(pot):
    """Pick the legacy seeded class when algorithm_mode == 'fixed_step_size'."""
    if pot.config.percolationConf.algorithm_mode == "fixed_step_size":
        return _TransitionObservablesFixedStepSize
    return _TransitionObservablesAdaptiveStepSize
from .phases import Phases, reconstructTransitionHistory
from .plot_settings import plot_settings as TL_plot_settings
from .plot_settings import add_TL_logo
from .colors import *
import transitionlistener.constants as cn
from .observability import Observability
from .gwfopt import FOPTspectrum
from . import thermodynamics as td
from .helper_functions import import_file, load_potential

plt.rcParams.update(TL_plot_settings)

DATA_DIR = Path(__file__).resolve().parent / "tab_data"

class TLPlots():
    """Collection of high-level plotting routines for a single potential instance."""

    def __init__(self, inputparam_dict: dict, modelfile: str,
                 potential_name: str, output_folder_name: str = "",
                 plot_description: str = "",
                 potential=None, phases=None, transitions=None,
                 transition_observables=None, verbose: bool = False):
        """Load the model and keep bookkeeping information for subsequent plots.

        Parameters
        ----------
        inputparam_dict : dict
            Dictionary of input parameters for the potential.
        modelfile : str
            Path to the model file defining the potential.
        output_folder_name : str, optional
            Folder where to save the plots.
        plot_description : str, optional
            Description to include in the plot titles.
        """
        self.inputparam_dict = inputparam_dict
        self.modelfile = modelfile
        self.output_folder_name = output_folder_name
        self.plot_description = plot_description
        self.verbose = verbose

        if potential is not None:
            self.pot = potential
        else:
            self.pot = load_potential(modelfile, potential_name)

        if phases is not None:
            self.phases = phases
        if transitions is not None:
            self.transitions = transitions
        if transition_observables is not None:
            self.derived_params_dict = transition_observables

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_phase_data(self, require_transitions: bool = False):
        """Lazily populate phases, transitions, and observables needed by a plot."""
        if not hasattr(self, "phases") or self.phases is None:
            self.phases = Phases(self.pot, verbose=self.verbose)
        if require_transitions:
            if not hasattr(self, "transitions") or self.transitions is None:
                self.transitions = Transitions(self.phases, self.pot, verbose=self.verbose)
            if not hasattr(self, "derived_params_dict") or self.derived_params_dict is None:
                self.derived_params_dict = _select_observables_class(self.pot)(
                    self.pot, self.phases, self.transitions, self.verbose
                )

    def _display_phase_label(self, phase_key) -> str:
        """Return the short user-facing label for one phase."""
        if hasattr(self, "phases") and self.phases is not None:
            return self.phases.phase_alias(phase_key)
        return str(phase_key)

    def _resolve_phase_key(self, phase_key):
        """Resolve a short user-facing phase label to the internal phase key."""
        if hasattr(self, "phases") and self.phases is not None:
            return self.phases.resolve_phase_key(phase_key)
        return phase_key

    @staticmethod
    def _style_axes(ax):
        """Apply the project-wide axis styling used by most TransitionListener plots."""
        ax.tick_params(axis="both", which="both", labelsize=10, direction="in", width=0.5)
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_linewidth(0.5)

    @staticmethod
    def _format_temperature_label(value: float, prefix: str = "T") -> str:
        """Format a temperature label in GeV, switching to scientific notation when needed."""
        if value < 1e-2 or value > 1e4:
            exponent = int(np.floor(np.log10(value)))
            mantissa = value / 10 ** exponent
            return rf"{prefix} = {mantissa:.3f} \cdot 10^{{{exponent}}}\,\mathrm{{GeV}}"
        return rf"{prefix} = {value:g}\,\mathrm{{GeV}}"

    def _prepare_profile_axes(self, count: int, height: float = 3.0):
        """Create a vertically stacked figure with consistent styling."""
        fig, axes = plt.subplots(count, 1, figsize=(6.4, height * count))
        if count == 1:
            axes = [axes]
        for ax in axes:
            self._style_axes(ax)
        return fig, axes

    @staticmethod
    def _normalize_profile_plot_arrays(radius, phi):
        """Validate and normalize cached bounce-profile arrays for plotting."""
        if radius is None or phi is None:
            return None, None

        radius_arr = np.asarray(radius, dtype=float).reshape(-1)
        phi_arr = np.asarray(phi, dtype=float)
        if phi_arr.ndim == 1:
            phi_arr = phi_arr[:, np.newaxis]

        if phi_arr.ndim != 2:
            return None, None
        if radius_arr.size == 0 or phi_arr.shape[0] == 0 or radius_arr.size != phi_arr.shape[0]:
            return None, None
        if not np.all(np.isfinite(radius_arr)) or not np.all(np.isfinite(phi_arr)):
            return None, None
        return radius_arr, phi_arr

    def _transition_derived_params(self, transition_index: int, tr) -> dict:
        """Return the observable dictionary associated with one transition."""
        if hasattr(self, "derived_params_dict") and self.derived_params_dict is not None:
            try:
                derived = self.derived_params_dict[transition_index]
            except Exception:  # noqa: BLE001
                derived = None
            if isinstance(derived, dict):
                return derived
        if isinstance(getattr(tr, "derived_params", None), dict):
            return tr.derived_params
        return {}

    # The adaptive/fixed-step observable containers cache extra per-transition
    # data that plotting can reuse instead of relying on ``tr.instanton`` alone.
    def _transition_percolation_result(self, transition_index: int):
        """Return cached percolation metadata for one transition when available."""
        percolation_results = getattr(getattr(self, "derived_params_dict", None), "percolationResults", None)
        if isinstance(percolation_results, dict):
            return percolation_results.get(transition_index)
        return None

    def _transition_outdict(self, transition_index: int):
        """Return the cached action dictionary for one transition when available."""
        outdicts = getattr(getattr(self, "derived_params_dict", None), "outdicts", None)
        if isinstance(outdicts, dict):
            return outdicts.get(transition_index)
        return None

    def _profile_temperature_candidates(self, transition_index: int, tr) -> list[float]:
        """Return preferred temperatures at which to look up cached instantons."""
        candidates: list[float] = []
        if tr.Tnuc is not None and np.isfinite(tr.Tnuc):
            candidates.append(float(tr.Tnuc))

        derived = self._transition_derived_params(transition_index, tr)
        tnuc_gev = derived.get("Tnuc_SM_GeV", np.nan)
        if np.isfinite(tnuc_gev):
            candidates.append(float(tnuc_gev) / self.pot.conversionFactor)

        percolation = self._transition_percolation_result(transition_index)
        tperc = getattr(percolation, "Tperc", np.nan)
        if np.isfinite(tperc):
            candidates.append(float(tperc))

        unique: list[float] = []
        for candidate in candidates:
            if not any(np.isclose(candidate, seen) for seen in unique):
                unique.append(candidate)
        return unique

    # Use the best available physical reference temperature for labeling and
    # for choosing which cached instanton to recover.
    def _profile_reference_temperature(self, transition_index: int, tr) -> tuple[float | None, str]:
        """Return the best available temperature and label prefix for profile plots."""
        if tr.Tnuc is not None and np.isfinite(tr.Tnuc):
            return float(tr.Tnuc), r"T_\mathrm{nuc}"

        derived = self._transition_derived_params(transition_index, tr)
        tnuc_gev = derived.get("Tnuc_SM_GeV", np.nan)
        if np.isfinite(tnuc_gev):
            return float(tnuc_gev) / self.pot.conversionFactor, r"T_\mathrm{nuc}"

        percolation = self._transition_percolation_result(transition_index)
        tperc = getattr(percolation, "Tperc", np.nan)
        if np.isfinite(tperc):
            return float(tperc), r"T_\mathrm{perc}"

        return None, "T"

    def _profile_title(self, transition_index: int, tr) -> str:
        """Build a title that reflects the temperature actually available."""
        high_label = self._display_phase_label(tr.high_phase)
        low_label = self._display_phase_label(tr.low_phase)
        temperature, prefix = self._profile_reference_temperature(transition_index, tr)
        if temperature is None:
            return f"Bounce from {high_label} to {low_label}"

        label = self._format_temperature_label(
            float(temperature) * self.pot.conversionFactor,
            prefix=prefix,
        )
        return fr"Bounce at ${label}$ from {high_label} to {low_label}"

    def _phase_point_at_temperature(self, phase_key, fallback_point, temperature):
        """Evaluate one phase minimum at ``temperature`` with a safe fallback."""
        if (
            temperature is not None
            and np.isfinite(temperature)
            and hasattr(self, "phases")
            and self.phases is not None
        ):
            try:
                phase = self.phases[phase_key]
                point = np.asarray(phase.valAt(float(temperature)), dtype=float)
                if np.all(np.isfinite(point)):
                    return point
            except Exception:  # noqa: BLE001
                pass
        return np.asarray(fallback_point, dtype=float)

    @staticmethod
    def _expanded_axis_limits(values, *, rel_pad: float, fallback_scale: float, min_pad: float):
        """Return ``(low, high)`` with robust padding, even for zero-width data."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            pad = max(float(min_pad), rel_pad * max(float(fallback_scale), 1.0))
            return -pad, pad

        low = float(np.min(arr))
        high = float(np.max(arr))
        span = high - low
        if span > 0:
            pad = max(rel_pad * span, float(min_pad))
        else:
            center = 0.5 * (low + high)
            ref = max(abs(center), float(fallback_scale), float(min_pad))
            pad = max(rel_pad * ref, float(min_pad))
        return low - pad, high + pad

    # Bounce profiles can come either from the transition object itself or from
    # cached outdict entries keyed by temperature in the observables pipeline.
    def _resolve_profile_plot_arrays(self, tr, transition_index: int):
        """Resolve radius/field arrays for plotting, including seedless caches."""
        candidates = [tr.instanton]
        outdict = self._transition_outdict(transition_index)
        for temperature in self._profile_temperature_candidates(transition_index, tr):
            instanton = _select_instanton_near_temperature(outdict, temperature)
            if instanton is not None:
                candidates.append(instanton)

        seen: set[int] = set()
        for instanton in candidates:
            if instanton is None:
                continue
            marker = id(instanton)
            if marker in seen:
                continue
            seen.add(marker)
            radius, phi = _extract_instanton_plot_arrays(instanton)
            radius, phi = self._normalize_profile_plot_arrays(radius, phi)
            if radius is not None and phi is not None:
                return radius, phi

        raise ValueError("No cached bubble profile is available for this transition.")

    # Only first-order transitions with a recoverable cached profile should end
    # up in ``profile`` / ``profileV``.
    def _collect_profile_entries(self):
        """Return FOPTs with resolved bounce profiles, skipping unavailable ones."""
        entries = []
        for transition_index, tr in enumerate(self.transitions):
            if getattr(tr, "type", None) != 1:
                continue
            try:
                radius, phi = self._resolve_profile_plot_arrays(tr, transition_index)
            except ValueError as err:
                high_label = self._display_phase_label(tr.high_phase)
                low_label = self._display_phase_label(tr.low_phase)
                console.print(
                    "[bold yellow]Skipping transition "
                    f"{high_label}->{low_label} in profile plots: {err}[/bold yellow]"
                )
                continue
            entries.append((transition_index, tr, radius, phi))
        return entries

    def _plot_profile_axis(self, ax, tr, radius, phi, transition_index: int, ylabel: str):
        """Render a single 1D bounce profile on the provided axes."""
        for i in range(phi.shape[-1]):
            lbl = r"$\phi_" + f"{i}" + r"$"
            ax.semilogx(
                radius / self.pot.conversionFactor,
                phi[:, i] * self.pot.conversionFactor, label=lbl,
            )
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_title(self._profile_title(transition_index, tr))

    def plotAction(self, Tmin_GeV : float, Tmax_GeV : float, phase_indices : list, n : int = 100,
                   show : bool = False, save : bool = True, savetxt : bool = True):
        """Visualise the Euclidean action and nucleation criterion across temperatures.
        
        Parameters
        ----------
        Tmin_GeV : float
            Minimum temperature in GeV.
        Tmax_GeV : float
            Maximum temperature in GeV.
        phase_indices : list
            List of phase indices to consider.
        n : int, optional
            Number of points to sample in the temperature range.
        show : bool, optional
            Whether to show the plot.
        save : bool, optional
            Whether to save the plot.

        """
        if not hasattr(self, "phases"):
            self.phases = Phases(self.pot, verbose=self.verbose)
        phase_indices = [self._resolve_phase_key(phase_indices[0]), self._resolve_phase_key(phase_indices[1])]
            
        print("\nPlot action in the range of temperatures from %.2f GeV to %.2f GeV." % (Tmin_GeV, Tmax_GeV))
        Trange = np.linspace(Tmin_GeV, Tmax_GeV, n) / self.pot.conversionFactor
        S3 = np.zeros_like(Trange)
        GH = np.zeros_like(Trange)

        for i, T in enumerate(Trange):
            print(f"Calculating action at T = {T * self.pot.conversionFactor:.2e} GeV.")
            try:
                S3_raw = calcAction(
                    self.pot,
                    T,
                    self.phases[phase_indices[1]],
                    self.phases[phase_indices[0]],
                    {},
                    verbose=True,
                )
                S3_val = float(np.squeeze(np.asarray(S3_raw, dtype=float)))
                S3[i] = S3_val

                G_raw = Gamma(T, S3_val)
                H_raw = HubbleParameter(
                    energyDensity(self.pot, self.phases[phase_indices[1]], T),
                    self.pot.conversionFactor,
                )
                G = float(np.squeeze(np.asarray(G_raw, dtype=float)))
                H = float(np.squeeze(np.asarray(H_raw, dtype=float)))
                GH[i] = np.nan if H == 0 else G / H**4

                print(
                    f"T / GeV: {T * self.pot.conversionFactor:.2e}, "
                    f"S3 / GeV: {float(S3_val * self.pot.conversionFactor):.2e}, "
                    f"Gamma / GeV^4: {float(G * self.pot.conversionFactor**4):.2e}, "
                    f"H / GeV: {float(H * self.pot.conversionFactor):.2e}, "
                    f"Gamma / H^4: {float(GH[i]):.2e}"
                )
            except Exception as e:
                print(f"Error at T = {T * self.pot.conversionFactor:.2e} GeV: {e}")
                S3[i] = np.nan
                GH[i] = np.nan
                continue
        fig, axes = plt.subplots(2,1, sharex=True)
        for ax in axes:
            ax.tick_params(axis='both', which='both', labelsize=10, direction="in", width=0.5)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(0.5)
            
        ax = axes[0]
        ax2 = axes[1]
        ax.plot(Trange * self.pot.conversionFactor, S3 / Trange, label="Action")
        ax2.plot(Trange * self.pot.conversionFactor, GH, label=r"$\Gamma / H^4$")
        ax2.plot(Trange * self.pot.conversionFactor, np.ones_like(Trange), "k--")
        ax2.set_xlabel(R"$T$ / $\mathrm{GeV}$")
        ax.set_ylabel("Euclidean action $S_3/T$")
        ax2.set_ylabel(r"$\Gamma / H^4$")
        ax.set_title(self.plot_description)
        ax.set_xlim(Tmin_GeV, Tmax_GeV)
        ax.set_yscale("log")
        ax2.set_yscale("log")
        add_TL_logo(loc="lower left", magnification=2, ax=ax2)
        fig.tight_layout()
        if save: fig.savefig(self.output_folder_name + "action.pdf")
        if show: plt.show()
        if savetxt:
            output_array = np.vstack((Trange * self.pot.conversionFactor,
                                      S3 / Trange,
                                      GH)).T
            np.savetxt(self.output_folder_name + "action.txt", output_array,
                       header="T_GeV S3_over_T Gamma_over_H4", fmt="%.6e")

    def plotPotential(self, plot_T_GeV : float, phi_ranges_GeV : list, n : int = 100,
                      show : bool = False, save : bool = True, log : bool = False):
        """Plot the potential energy landscape at a fixed temperature.
        
        Parameters
        ----------
        plot_T_GeV : float
            Temperature in GeV at which to plot the potential.
        phi_ranges_GeV : list
            List defining the plotting ranges for each field dimension in GeV.            
        n : int, optional
            Number of points to use for the plot.
        show : bool, optional
            Whether to show the plot.
        save : bool, optional
            Whether to save the plot.
        log : bool, optional
            Whether to use a logarithmic color scale (only for 2D plots).
        """
        if plot_T_GeV == 0.0:
            T_str = r"$T = {:g}$".format(plot_T_GeV)
        elif plot_T_GeV < 1e-2 or plot_T_GeV > 1e4:
            exponent = int(np.floor(np.log10(plot_T_GeV)))
            mantissa = plot_T_GeV / 10**exponent
            T_str = r"$T = {:.3f} \cdot 10^{{{}}}$".format(mantissa, exponent)
        else:
            T_str = r"$T = {:g}$".format(plot_T_GeV)
            
        if self.pot.Ndim ==1:
            fig, ax = plot1d(
                self.pot,
                phi_ranges_GeV[0] / self.pot.conversionFactor,
                phi_ranges_GeV[1] / self.pot.conversionFactor,
                T=plot_T_GeV / self.pot.conversionFactor,
                subtract=True,
                n=n,
            )
        else:
            fig, ax = plot2d(
                self.pot,
                np.array(
                    [phi_ranges_GeV[i] / self.pot.conversionFactor for i in range(len(phi_ranges_GeV))]
                ),
                T=plot_T_GeV / self.pot.conversionFactor,
                treelevel=False,
                offset=0,
                xaxis=0,
                yaxis=1,
                n=n,
                clevs=200,
                cfrac=.8,
                log=log,
            )
        ax.set_title(self.plot_description + f": Potential at {T_str}" + r"$\,\mathrm{GeV}$")
        add_TL_logo(loc="upper left", ax=ax)
        fig.tight_layout() 
        if save: fig.savefig(self.output_folder_name + "potential.pdf")
        if show: plt.show()
        
    def plotPercolation(
        self,
        show: bool = False,
        Tmin_GeV: float = np.nan,
        Tmax_GeV: float = np.nan,
        n_action: int | None = None,
    ):
        """Plot the true-vacuum fraction to illustrate percolation."""
        if not hasattr(self, "phases"):
            self.phases = Phases(self.pot, verbose=self.verbose)
        if not hasattr(self, "transitions"):
            self.transitions = Transitions(self.phases, self.pot, self.verbose)
        if not hasattr(self, "derived_params_dict"):
            self.derived_params_dict = _select_observables_class(self.pot)(self.pot, self.phases, self.transitions, self.verbose)
        
        nTransitions = len(self.transitions)        
        print("Found", nTransitions, "transitions.")

        # Loop over transitions
        for i, tr in enumerate(self.transitions):
            print("Transition "+str(i+1)+":")
            try:
                if tr is not None:
                    if tr.type != 1:
                        print("Not a first-order transition, skipping.")
                        continue
                    Tnuc = tr.Tnuc
                    phase_symmetric = self.phases[tr.high_phase]
                    phase_broken = self.phases[tr.low_phase]
                    # Respect configured support points unless explicitly
                    # overridden for the percolation plot in the YAML file.
                    if n_action is None:
                        n_action_eff = int(getattr(self.pot.config.gwConf, "n_action", 25))
                    else:
                        n_action_eff = int(n_action)
                    if n_action_eff < 5:
                        n_action_eff = 5
                    if self.verbose:
                        plot_range_msg = (
                            f"[{Tmin_GeV}, {Tmax_GeV}] GeV"
                            if (np.isfinite(Tmin_GeV) or np.isfinite(Tmax_GeV))
                            else "automatic"
                        )
                        print(
                            "Running percolation diagnostics with "
                            f"n_action={n_action_eff}, "
                            f"T range={plot_range_msg}, "
                            f"Tnuc={Tnuc * self.pot.conversionFactor:.6g} GeV."
                        )
                    Tperc, TSYM, H, P, TBRO, S, Pr_exp, scalef_ratio, soundSpSq = calcPercAndEvolve(
                        {},
                        Tnuc,
                        phase_symmetric,
                        phase_broken,
                        self.pot,
                        vw=1,
                        nAction=n_action_eff,
                        verbose=self.verbose,
                    )
                    
                    # Keep percolation diagnostics in a strip-like layout.
                    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6.4, 2.35))

                    ax.tick_params(axis='both', which='both', labelsize=10, direction="in", width=0.5)
                    ax.xaxis.set_ticks_position('both')
                    ax.yaxis.set_ticks_position('both')
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(0.5)

                    TSYM_GeV = np.asarray(TSYM, dtype=float) * self.pot.conversionFactor
                    P_arr = np.asarray(P, dtype=float)
                    mask = np.isfinite(TSYM_GeV) & np.isfinite(P_arr)
                    TSYM_GeV = TSYM_GeV[mask]
                    P_arr = P_arr[mask]

                    if TSYM_GeV.size == 0:
                        if self.verbose:
                            print("No finite percolation data points; skipping this transition.")
                        continue

                    # Keep the lower bound at the last temperature where
                    # percolation data exists.
                    Tmin_data_GeV = float(TSYM_GeV[-1])

                    Tnuc_GeV = Tnuc * self.pot.conversionFactor
                    Tperc_GeV = Tperc * self.pot.conversionFactor
                    if np.isfinite(Tmin_GeV):
                        Tmin_plot_GeV = float(Tmin_GeV)
                        if Tmin_plot_GeV > Tmin_data_GeV:
                            Tmin_plot_GeV = Tmin_data_GeV
                    else:
                        Tmin_plot_GeV = Tmin_data_GeV
                    Tmax_plot_GeV = Tnuc_GeV if not np.isfinite(Tmax_GeV) else float(Tmax_GeV)
                    if Tmax_plot_GeV <= Tmin_plot_GeV:
                        if self.verbose:
                            print(
                                "Invalid percolation plotting range "
                                f"[{Tmin_plot_GeV}, {Tmax_plot_GeV}] GeV; "
                                "falling back to automatic range."
                            )
                        Tmin_plot_GeV = Tmin_data_GeV
                        Tmax_plot_GeV = Tnuc_GeV

                    order = np.argsort(TSYM_GeV)
                    TSYM_GeV = TSYM_GeV[order]
                    P_arr = P_arr[order]

                    window = (TSYM_GeV >= Tmin_plot_GeV) & (TSYM_GeV <= Tmax_plot_GeV)
                    if np.count_nonzero(window) >= 2:
                        TSYM_plot = TSYM_GeV[window]
                        P_plot = P_arr[window]
                    else:
                        TSYM_plot = TSYM_GeV
                        P_plot = P_arr

                    if TSYM_plot.size >= 2:
                        n_dense = max(1200, 8 * TSYM_plot.size)
                        T_dense_GeV = np.linspace(TSYM_plot[0], TSYM_plot[-1], n_dense)
                        P_dense = np.interp(T_dense_GeV, TSYM_plot, P_plot)
                    else:
                        T_dense_GeV = TSYM_plot
                        P_dense = P_plot

                    ax.plot(T_dense_GeV, P_dense, label="$P_\\mathrm{t}(T)$")
                    ax.set_title(self.plot_description)
                    ax.set_ylabel("True vacuum fraction")
                    ax.axvline(Tperc_GeV, ls="--", color="black", label="$T_\\mathrm{perc}$")
                    ax.axvline(Tnuc_GeV, ls="-.", color="grey", label="$T_\\mathrm{nuc}$")
                    x_right_GeV = Tmax_plot_GeV
                    if not np.isfinite(Tmax_GeV):
                        xpad_GeV = max(0.02 * max(Tnuc_GeV - Tmin_plot_GeV, 0.0), 5.0e-5)
                        x_right_GeV = Tnuc_GeV + xpad_GeV
                    ax.set_xlim(Tmin_plot_GeV, x_right_GeV)
                    ax.set_xlabel(r"$T$ / $\mathrm{GeV}$")
                    ax.legend(loc="upper right")
                    add_TL_logo(loc="lower left", magnification=1.5, ax=ax)
                    fig.tight_layout()
                    if show:
                        plt.show()
                    else:
                        fig.savefig(self.output_folder_name + "percolation_" + str(i) + ".pdf")
            except Exception as e:
                print("Error in transition "+str(i)+":", e)
                print("Skipping this transition.")
                continue
            
    def plotProfile(self):
        """Plot bounce profiles along the tunnelling path for all FOPTs."""
        self._ensure_phase_data(require_transitions=True)
        has_fopts = any(getattr(tr, "type", None) == 1 for tr in self.transitions)
        profile_entries = self._collect_profile_entries()
        if not profile_entries:
            message = (
                "No cached bubble profiles available for profile plot."
                if has_fopts
                else "No first-order transitions available for profile plot."
            )
            console.print(f"[bold yellow]{message}[/bold yellow]")
            return

        fig, axes = self._prepare_profile_axes(len(profile_entries))
        ylabel = r"$\phi_i$ / $\mathrm{GeV}$"
        for ax, (transition_index, tr, radius, phi) in zip(axes, profile_entries):
            self._plot_profile_axis(ax, tr, radius, phi, transition_index, ylabel)

        axes[-1].set_xlabel(r"Bubble radius / $\mathrm{GeV}^{-1}$")
        add_TL_logo(loc="lower right", magnification=2, ax=axes[-1])
        fig.tight_layout()
        fig.savefig(self.output_folder_name + "profile.pdf")

    def plotProfileV(self, phi1 : int, phi2 : int):
        """Show the potential along the tunnelling path between two field values.
    
        Parameters
        ----------
        phi1 : int
            Index of the first field dimension to plot.
        phi2 : int
            Index of the second field dimension to plot.
        
        """
        if self.pot.Ndim == 1:
            raise ValueError("plotProfileV only works for Ndim > 1.")
        if not hasattr(self, "phases"):
            self.phases = Phases(self.pot, verbose=self.verbose)
        if not hasattr(self, "transitions"):
            self.transitions = Transitions(self.phases, self.pot, verbose=self.verbose)
        if not hasattr(self, "derived_params_dict"):
            self.derived_params_dict = _select_observables_class(self.pot)(self.pot, self.phases, self.transitions, self.verbose)

        has_fopts = any(getattr(tr, "type", None) == 1 for tr in self.transitions)
        profile_entries = self._collect_profile_entries()
        if not profile_entries:
            message = (
                "No cached bubble profiles available for profileV plot."
                if has_fopts
                else "No first-order transitions available for profileV plot."
            )
            console.print(f"[bold yellow]{message}[/bold yellow]")
            return

        nFOPTs = len(profile_entries)

        # the labels of the field axes we want to plot
        xaxis=phi1
        yaxis=phi2
        
        fig, axes = plt.subplots(nFOPTs, 1, figsize=(6.4, 3 * nFOPTs))
        if nFOPTs == 1:
            axes = [axes]
            
        for ax in axes:
            ax.tick_params(axis='both', which='both', labelsize=10, direction="in", width=0.5)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(0.5)
        
        for idx, (transition_index, tr, _, phi) in enumerate(profile_entries):
            mag = 0.2
            plot_temperature, _ = self._profile_reference_temperature(transition_index, tr)
            if plot_temperature is None:
                plot_temperature = tr.Tcrit
            start = self._phase_point_at_temperature(tr.high_phase, tr.high_vev, plot_temperature)
            end = self._phase_point_at_temperature(tr.low_phase, tr.low_vev, plot_temperature)
            fallback_scale = float(
                max(
                    np.nanmax(np.abs(phi)),
                    np.nanmax(np.abs(start)),
                    np.nanmax(np.abs(end)),
                    1.0 / max(self.pot.conversionFactor, 1.0),
                )
            )
            min_pad = 0.05 / max(self.pot.conversionFactor, 1.0)
            x0, x1 = self._expanded_axis_limits(
                np.concatenate((phi[:, xaxis], [start[xaxis], end[xaxis]])),
                rel_pad=mag,
                fallback_scale=fallback_scale,
                min_pad=min_pad,
            )
            y0, y1 = self._expanded_axis_limits(
                np.concatenate((phi[:, yaxis], [start[yaxis], end[yaxis]])),
                rel_pad=mag,
                fallback_scale=fallback_scale,
                min_pad=min_pad,
            )
            phi_ranges_GeV = np.array([x0, x1, y0, y1]) * self.pot.conversionFactor
            plot2d(
                self.pot,
                np.array([x0, x1, y0, y1]),
                axes[idx],
                T=plot_temperature,
                treelevel=False,
                offset=0,
                xaxis=xaxis,
                yaxis=yaxis,
                n=100,
                clevs=200,
                cfrac=.8,
            )
            axes[idx].plot(
                start[xaxis] * self.pot.conversionFactor,
                start[yaxis] * self.pot.conversionFactor,
                label="Start",
                marker="^",
                lw=0,
                markersize=8,
                color="C3",
            )
            axes[idx].plot(
                end[xaxis] * self.pot.conversionFactor,
                end[yaxis] * self.pot.conversionFactor,
                label="End",
                marker="x",
                lw=0,
                markersize=8,
                color="C5",
            )
            axes[idx].plot(
                phi[:, xaxis] * self.pot.conversionFactor,
                phi[:, yaxis] * self.pot.conversionFactor,
                color="C2",
                lw=3,
                alpha=1,
            )
            axes[idx].legend(loc="upper right")
            axes[idx].set_title(self._profile_title(transition_index, tr))
            axes[idx].set_xlim(phi_ranges_GeV[0], phi_ranges_GeV[1])
            axes[idx].set_ylim(phi_ranges_GeV[2], phi_ranges_GeV[3])
        add_TL_logo(loc="lower right", magnification=1, ax=axes[-1])
        fig.tight_layout()
        fig.savefig(self.output_folder_name + "profileV.pdf")

    def plotPhases(self, Tmin_GeV: float = np.nan, Tmax_GeV: float = np.nan,
                   include_transitions: bool = True, save: bool = True,
                   show: bool = False, plot_squaresum: bool = True):
        """Plot the traced phase trajectories and optionally their transitions.

        ``plot_squaresum`` only matters in multi-field models; when ``False``
        the panel for ``sqrt(sum_i phi_i^2)`` is omitted, giving a 2-panel
        figure that matches the 1-field default layout.
        """
        if include_transitions:
            # First produce the plot immediately after phase tracing.
            self._ensure_phase_data(require_transitions=False)
            self._render_phase_plot(
                include_transitions=False,
                Tmin_GeV=Tmin_GeV,
                Tmax_GeV=Tmax_GeV,
                save_path=self.output_folder_name + "phases.pdf" if save else None,
                show=show,
                plot_squaresum=plot_squaresum,
            )
            # Then ensure transitions exist and overlay the full history.
            self._ensure_phase_data(require_transitions=True)
            self._render_phase_plot(
                include_transitions=True,
                Tmin_GeV=Tmin_GeV,
                Tmax_GeV=Tmax_GeV,
                save_path=self.output_folder_name + "phases_with_transitions.pdf" if save else None,
                show=show,
                plot_squaresum=plot_squaresum,
            )
        else:
            self._ensure_phase_data(require_transitions=False)
            self._render_phase_plot(
                include_transitions=False,
                Tmin_GeV=Tmin_GeV,
                Tmax_GeV=Tmax_GeV,
                save_path=self.output_folder_name + "phases.pdf" if save else None,
                show=show,
                plot_squaresum=plot_squaresum,
            )

    def _render_phase_plot(self, include_transitions: bool, Tmin_GeV: float, Tmax_GeV: float,
                           save_path: Optional[str], show: bool,
                           plot_squaresum: bool = True) -> None:
        """Render a single phase plot with or without transition overlays."""
        CF = self.pot.conversionFactor
        color = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        ls = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-."]

        if self.pot.Ndim == 1 or not plot_squaresum:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.4, 5), edgecolor="white")
        else:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.4, 7), edgecolor="white")

        for ax in axes:
            self._style_axes(ax)

        for i, key in enumerate(self.phases.keys()):
            p = self.phases[key]
            DeltaV = self.pot.Vtot(p.X, p.T) - self.pot.Vtot(p.X * 0, p.T)
            for j in range(self.pot.Ndim):
                axes[0].plot(p.T * CF, p.X[:, j] * CF, color=color[i % len(color)], ls=ls[j],
                             label=fr"{self._display_phase_label(key)}, $\phi_{j+1}$")
            if self.pot.Ndim > 1 and plot_squaresum:
                squaresum = np.sum(p.X ** 2, axis=1)
                axes[1].plot(p.T * CF, np.sqrt(squaresum) * CF, color=color[i % len(color)], ls=ls[0])
            axes[-1].plot(p.T * CF, DeltaV * CF ** 4, color=color[i % len(color)], ls=ls[0])

        if include_transitions:
            if not hasattr(self, "phi_min"):
                self.phi_min = reconstructTransitionHistory(self.phases, self.transitions)

            lower_bound = None if np.isnan(Tmin_GeV) else Tmin_GeV / CF
            upper_bound = None if np.isnan(Tmax_GeV) else Tmax_GeV / CF
            phi_label_added = False
            for idx, segment in self.phi_min.items():
                temps = segment.temperature_grid(
                    lower=lower_bound,
                    upper=upper_bound,
                    min_points=max(50, segment.phase.T.size),
                )
                if temps.size < 2:
                    continue
                phi_vals = np.asarray(segment(temps))
                if phi_vals.ndim == 1:
                    phi_vals = phi_vals.reshape(-1, 1)
                label = r"$\phi_{\min}^\mathrm{glob}(T)$" if not phi_label_added else None
                for j in range(self.pot.Ndim):
                    axes[0].plot(
                        temps * CF,
                        phi_vals[:, j] * CF,
                        color="black",
                        lw=1.3,
                        label=label if j == 0 else None,
                        zorder=5,
                        ls="dotted",
                    )
                if self.pot.Ndim > 1 and plot_squaresum:
                    magnitude = np.linalg.norm(phi_vals, axis=1)
                    axes[1].plot(
                        temps * CF,
                        magnitude * CF,
                        color="black",
                        lw=1.3,
                        zorder=5,
                        ls="dotted",
                    )
                zero_field = np.zeros_like(phi_vals)
                delta_v = self.pot.Vtot(phi_vals, temps) - self.pot.Vtot(zero_field, temps)
                axes[-1].plot(
                    temps * CF,
                    delta_v * CF**4,
                    color="black",
                    lw=1.3,
                    zorder=5,
                    ls="dotted",
                )
                phi_label_added = True

            for tr in self.transitions:
                if getattr(tr, "type", 0) == 1:
                    Tperc_GeV = tr.derived_params.get("Tperc_SM_GeV", np.nan)
                    Treh_GeV = tr.derived_params.get("Treh_SM_GeV", np.nan)
                    for ax in axes:
                        ax.axvline(Tperc_GeV, ls="--", color=DESYpetrol, alpha=0.5)
                        ax.axvline(Treh_GeV, ls="-.", color=DESYmagenta, alpha=0.5)
                if getattr(tr, "type", 0) == 2:
                    Tcrit_GeV = tr.Tcrit * CF
                    for ax in axes:
                        ax.axvline(Tcrit_GeV, ls="-", color=DESYgelb, alpha=0.5)

            axes[-1].legend(loc="upper right", handles=[
                plt.Line2D([0], [0], color=DESYpetrol, linestyle="--", label=r"$T_\mathrm{perc}$", alpha=0.5),
                plt.Line2D([0], [0], color=DESYmagenta, linestyle="-.", label=r"$T_\mathrm{reh}$", alpha=0.5),
                plt.Line2D([0], [0], color=DESYgelb, linestyle="-", label=r"$T_\mathrm{crit}$", alpha=0.5),
            ])

        add_TL_logo(loc="lower right", magnification=2, ax=axes[-1])
        axes[0].set_title(self.plot_description)
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel(r"$\phi_i$ / $\mathrm{GeV}$")
        if self.pot.Ndim > 1 and plot_squaresum:
            axes[1].set_ylabel(r"$\sqrt{\sum_i \phi_i^2}$ / $\mathrm{GeV}$")
        axes[-1].set_ylabel(r"$[V(\phi_{min}(T), T) - V(0, T)] / \mathrm{GeV^4}$")

        if not np.isnan(Tmin_GeV) and not np.isnan(Tmax_GeV):
            axes[-1].set_xlim(Tmin_GeV, Tmax_GeV)
        else:
            axes[-1].set_xlim(self.pot.Tmin * CF, self.pot.Tmax * CF)
        axes[-1].set_xlabel(r"$T$ / $\mathrm{GeV}$")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    def plotDOFs(self, Tmin_GeV : float =np.nan, Tmax_GeV : float =np.nan):
        """Plot the effective degrees of freedom in the SM and dark sector.
        
        Parameters
        ----------
        Tmin_GeV : float, optional
            Minimum temperature in GeV for the x-axis.
        Tmax_GeV : float, optional
            Maximum temperature in GeV for the x-axis.
        """
        if not hasattr(self, "phases"):
            self.phases = Phases(self.pot, verbose=self.verbose)
        if not hasattr(self, "transitions"):
            self.transitions = Transitions(self.phases, self.pot, verbose=self.verbose)
        if not hasattr(self, "derived_params_dict"):
            self.derived_params_dict = _select_observables_class(self.pot)(self.pot, self.phases, self.transitions, self.verbose)
        if not hasattr(self, "phi_min"):
            self.phi_min = reconstructTransitionHistory(self.phases, self.transitions)

        CF = self.pot.conversionFactor
        history = self.phi_min
        history_Tmin, history_Tmax = history.temperature_span

        Tmin = history_Tmin if np.isnan(Tmin_GeV) else Tmin_GeV / CF
        Tmax = history_Tmax if np.isnan(Tmax_GeV) else Tmax_GeV / CF

        if Tmax <= Tmin:
            raise ValueError("Requested temperature range for plotDOFs has zero or negative span.")

        fig, axes = plt.subplots(2,1, sharex=True, figsize=(6.4, 6), edgecolor="white")
        for a in axes:
            a.tick_params(axis='both', which='both', labelsize=10, direction="in", width=0.5)
            a.xaxis.set_ticks_position('both')
            a.yaxis.set_ticks_position('both')
            for axis in ['top','bottom','left','right']:
                a.spines[axis].set_linewidth(0.5)

        label_usage_geff = {"BSM": False, "SM": False, "Total": False}
        label_usage_rho = {"BSM": False, "SM": False, "Total": False}
        for idx, segment in history.items():
            seg_low, seg_high = segment.bounds()
            seg_low = max(seg_low, Tmin)
            seg_high = min(seg_high, Tmax)
            if seg_high <= seg_low:
                continue

            # Use a denser sampling for smoother g_eff and rho(T) curves.
            min_points = int(np.clip(max(400, 2 * segment.phase.T.size), 400, 2000))
            temps = segment.temperature_grid(lower=Tmin, upper=Tmax, min_points=min_points)
            if temps.size < min_points:
                temps = np.linspace(seg_low, seg_high, min_points)

            phi_vals = np.asarray(segment(temps))
            if phi_vals.ndim == 1:
                phi_vals = phi_vals.reshape(-1, 1)

            geffDS = []
            geffSM = []
            for phi_vec, T in zip(phi_vals, temps):
                # Use the mass-spectrum tuple API directly; this is supported
                # by all model files and avoids relying on optional sector wrappers.
                bosons = self.pot.boson_massSq(phi_vec, 0)
                fermions = self.pot.fermion_massSq(phi_vec)
                geffDS.append(td.e_geffDS(bosons, fermions, T))
                geffSM.append(td.e_geffSM(T, CF))

            geffDS = np.asarray(geffDS)
            geffSM = np.asarray(geffSM)
            rhoDS = np.pi**2 / 30 * geffDS * temps**4
            rhoSM = np.pi**2 / 30 * geffSM * temps**4

            temps_plot = temps * CF

            label = None if label_usage_geff["BSM"] else "BSM"
            axes[0].plot(temps_plot, geffDS, color="C0", label=label)
            if label:
                label_usage_geff["BSM"] = True

            label = None if label_usage_geff["SM"] else "SM"
            axes[0].plot(temps_plot, geffSM, color="C1", label=label)
            if label:
                label_usage_geff["SM"] = True

            label = None if label_usage_geff["Total"] else "Total"
            axes[0].plot(temps_plot, geffDS + geffSM, color="black", label=label)
            if label:
                label_usage_geff["Total"] = True

            label = None if label_usage_rho["BSM"] else "BSM"
            axes[1].plot(temps_plot, rhoDS * CF**4, color="C0", label=label)
            if label:
                label_usage_rho["BSM"] = True

            label = None if label_usage_rho["SM"] else "SM"
            axes[1].plot(temps_plot, rhoSM * CF**4, color="C1", label=label)
            if label:
                label_usage_rho["SM"] = True

            label = None if label_usage_rho["Total"] else "Total"
            axes[1].plot(temps_plot, (rhoDS + rhoSM) * CF**4, color="black", label=label)
            if label:
                label_usage_rho["Total"] = True

        no_label1 = False # these two make sure we only label the lines once
        no_label2 = False
        for tr in self.transitions:
            if tr.type == 1:
                Tperc_GeV = tr.derived_params["Tperc_SM_GeV"]
                Treh_GeV = tr.derived_params["Treh_SM_GeV"]
                if not no_label1:
                    for ax in axes:
                        ax.axvline(Tperc_GeV, ls="--", color=DESYpetrol, alpha=0.5, label=r"$T_\mathrm{perc}$")
                        ax.axvline(Treh_GeV, ls="-.", color=DESYmagenta, alpha=0.5, label=r"$T_\mathrm{reh}$")
                else:
                    for ax in axes:
                        ax.axvline(Tperc_GeV, ls="--", color=DESYpetrol, alpha=0.5)
                        ax.axvline(Treh_GeV, ls="-.", color=DESYmagenta, alpha=0.5)
                no_label1 = True
            if tr.type == 2:
                Tcrit_GeV = tr.Tcrit * self.pot.conversionFactor
                if not no_label2:
                    for ax in axes:
                        ax.axvline(Tcrit_GeV, ls="-", color=DESYgelb, alpha=0.5, label=r"$T_\mathrm{crit}$")
                else:
                    for ax in axes:
                        ax.axvline(Tcrit_GeV, ls="-", color=DESYgelb, alpha=0.5)
                no_label2 = True

        add_TL_logo(loc="lower right", ax=axes[1], magnification=2)
        axes[0].set_title(self.plot_description)
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel(R"$g_{\mathrm{eff}}(T)$")
        axes[1].set_ylabel(R"$\rho(T) / \mathrm{GeV}^4$")
        axes[1].set_yscale("log")
        axes[0].set_xlim(Tmin * CF, Tmax * CF)
        axes[1].set_xlim(Tmin * CF, Tmax * CF)
        axes[1].set_xlabel(R"$T$ / $\mathrm{GeV}$")
        fig.tight_layout()
        fig.savefig(self.output_folder_name + "dofs.pdf")

    def plotEnergyDensity(self, Tmin_GeV : float =np.nan, Tmax_GeV : float =np.nan,
                          show_plot=False):
        """Plot the energy density evolution for each phase.
        
        Parameters
        ----------
        Tmin_GeV : float, optional
            Minimum temperature in GeV for the x-axis.
        Tmax_GeV : float, optional
            Maximum temperature in GeV for the x-axis.
        """
        if not hasattr(self, "phases"):
            self.phases = Phases(self.pot, verbose=self.verbose)

        fig, ax = plt.subplots(1,1, sharex=True, figsize=(6.4, 4.8), edgecolor="white")
        for a in (ax,):
            a.tick_params(axis='both', which='both', labelsize=10, direction="in", width=0.5)
            a.xaxis.set_ticks_position('both')
            a.yaxis.set_ticks_position('both')
            for axis in ['top','bottom','left','right']:
                a.spines[axis].set_linewidth(0.5)

        Tmin = self.pot.Tmin if np.isnan(Tmin_GeV) else Tmin_GeV / self.pot.conversionFactor
        Tmax = self.pot.Tmax if np.isnan(Tmax_GeV) else Tmax_GeV / self.pot.conversionFactor
        Trange = np.exp(np.linspace(np.log(Tmin), np.log(Tmax), 1000))

        for i, phase_key in enumerate(self.phases.keys()):
            Trange_phase = Trange[(Trange >= self.phases[phase_key].Tmin) & (Trange <= self.phases[phase_key].Tmax)]
            if len(Trange_phase) == 0:
                continue
            rho = np.array([energyDensity(self.pot, self.phases[phase_key], T) for T in Trange_phase])
            ax.plot(
                Trange_phase * self.pot.conversionFactor,
                rho * self.pot.conversionFactor**4,
                label=self._display_phase_label(phase_key),
            )

        add_TL_logo(loc="lower right", ax=ax)
        ax.set_title(self.plot_description)
        ax.legend(loc="upper right")
        ax.set_ylabel(R"$\rho$ / $\mathrm{GeV}^4$")
        ax.set_yscale("log")
        ax.set_xscale("log")
        #ax.set_ylim(1e-2, 1e11)
        ax.set_xlim(Trange[0] * self.pot.conversionFactor, Trange[-1] * self.pot.conversionFactor)
        ax.set_xlabel(R"$T$ / $\mathrm{GeV}$")
        fig.tight_layout()
        if show_plot:
            plt.show()
        else:
            fig.savefig(self.output_folder_name + "rho.pdf")

def plotSensitivities(gw, showplot : bool =False, call_from_spectrum=False, nfreq : int =5,
                      fig=None, ax=None):
    """Plot ground- and space-based detector sensitivities alongside the spectrum.
    
    Parameters
    ----------
    gw : GWSpectrum
        The GWSpectrum object containing the sensitivity data.
    showplot : bool, optional
        Whether to display the plot.
    call_from_spectrum : bool, optional
        Whether the function is called from within a spectrum plot.
        If True, the function returns the figure and axis objects instead
        of showing or saving the plot.
    nfreq : int, optional
        Number of frequency bins for the PTA violins.
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=(6.4, 3.5))
    transparency = 1
    
    for det_name in gw.det:
        if gw.det[det_name]["no_plot"]:
            continue

        is_existing = det_name in {"NANOGrav_15_yrs", "HLV_O2"}

        ax.fill_between(
            np.log10(gw.det[det_name]["f_PLI"]),
            np.log10(gw.det[det_name]["PLI"]),
            0,
            alpha=1,
            color="white",
            zorder=gw.det[det_name]["zorder"],
            lw=0,
        )
        
        ax.fill_between(
            np.log10(gw.det[det_name]["f_PLI"]),
            np.log10(gw.det[det_name]["PLI"]),
            0,
            alpha=0.8,
            color=gw.det[det_name]["color"],
            zorder=gw.det[det_name]["zorder"],
            lw=0,
        )
        if is_existing:
            ax.fill_between(
                np.log10(gw.det[det_name]["f_PLI"]),
                np.log10(gw.det[det_name]["PLI"]),
                0,
                alpha=transparency,
                color=gw.det[det_name]["color"],
                zorder=gw.det[det_name]["zorder"],
                label=gw.det[det_name]["plot_name"],
                lw=0,
            )
        else:
            # Predicted sensitivity: draw as hatched region, no solid fill
            ax.fill_between(
                np.log10(gw.det[det_name]["f_PLI"]),
                np.log10(gw.det[det_name]["PLI"]),
                0,
                alpha=0,
                color="none",
                hatch="\\\\",
                edgecolor=gw.det[det_name]["color"],
                linewidth=0.0,
                zorder=gw.det[det_name]["zorder"],
                label=gw.det[det_name]["plot_name"],
            )
        

    
    add_violins(ax, violincolor="#C0105A", nfreq=nfreq)
    add_NeffHatch(ax, Neffcolor="grey", h2OmegaNeff=2.48e-7) 
    if not call_from_spectrum:
        ax.set_title("PLI sensitivities and PTA violins")
    ax.set_xlabel(R"$f$ / $\mathrm{Hz}$")
    ax.set_ylabel(R'$h^2 \Omega_\mathrm{GW} (f)$')
    
    ax.tick_params(axis='both', which='both', direction="in", width=0.5, zorder=1000)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    for axis in ["top", 'bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
        ax.spines[axis].set_zorder(10000)
        
    xlim = [-10,4]
    xa = np.linspace(int(xlim[0]), int(xlim[1]), int(xlim[1] - xlim[0] + 1))
    xtMajor = [j for j in xa]
    xtMinor = [np.log10(i*10**j) for j in xa for i in range(10)[1:10]]
    label_map = {-9: "nHz", -6: "µHz", -3: "mHz", 0: "Hz", 3: "kHz"}
    xlMajor = [label_map.get(i, "") for i in xa]
    xMajorLocator = FixedLocator(xtMajor)
    xMinorLocator = FixedLocator(xtMinor)
    xMajorFormatter = FixedFormatter(xlMajor)
    ax.xaxis.set_major_locator(xMajorLocator)
    ax.xaxis.set_minor_locator(xMinorLocator)
    ax.xaxis.set_major_formatter(xMajorFormatter)
    ax.set_xlim(-9.5, 3.5)

    ylim = [-18, -6]
    ya = np.linspace(ylim[0], ylim[1], int(ylim[1] - ylim[0] + 1))
    ytMajor = [j for j in ya]
    ytMinor = [np.log10(i*10**j) for i in range(10)[1:10] for j in ya]
    ylMajor = [r"$10^{" + str(int(i)) + "}$" if i%2==1 else "" for i in ya]
    yMajorLocator = FixedLocator(ytMajor)
    yMinorLocator = FixedLocator(ytMinor)
    yMajorFormatter = FixedFormatter(ylMajor)
    ax.yaxis.set_major_locator(yMajorLocator)
    ax.yaxis.set_minor_locator(yMinorLocator)
    ax.yaxis.set_major_formatter(yMajorFormatter)
    ax.set_ylim(ylim[0], ylim[1])
    ax.yaxis.set_zorder(1000)
    ax.xaxis.set_zorder(1000)
    ax.set_ylim(-17.5, -6)

    #ax.grid()
    add_TL_logo(loc="outside", ax=ax, small_logo=False,
                magnification=1.4)
    ax.legend(bbox_to_anchor=(1, 0.7), ncol=1)
    if not call_from_spectrum:
        plt.tight_layout()
        if showplot:
            plt.show()
        else:
            from pathlib import Path
            Path(gw.foldername).mkdir(parents=True, exist_ok=True)
            plt.savefig(gw.foldername+"GW_sensitivities.pdf")
        plt.close(fig)
    else:
        return fig, ax

def plotSensitivitiesPTA(gw, showplot : bool = False, call_from_spectrum : bool = False, nfreq : int = 5):
    """Plot PTA sensitivity curves alongside the spectrum.
    
    Parameters
    ----------
    gw : GWSpectrum
        The GWSpectrum object containing the sensitivity data.
    showplot : bool, optional
        Whether to display the plot.
    call_from_spectrum : bool, optional
        Whether the function is called from within a spectrum plot.
        If True, the function returns the figure and axis objects instead
        of showing or saving the plot.
    nfreq : int, optional
        Number of frequency bins for the PTA violins.
    """
    fig, ax = plt.subplots()    
    transparency = 0.5
    
    for det_name in gw.det:
        if gw.det[det_name]["no_plot"]:
            continue

        is_existing = det_name in {"NANOGrav_15_yrs", "HLV_O2"}

        ax.fill_between(
            np.log10(gw.det[det_name]["f_PLI"]),
            np.log10(gw.det[det_name]["PLI"]),
            0,
            alpha=1,
            color="white",
            zorder=gw.det[det_name]["zorder"],
        )
        if is_existing:
            ax.fill_between(
                np.log10(gw.det[det_name]["f_PLI"]),
                np.log10(gw.det[det_name]["PLI"]),
                0,
                alpha=transparency,
                color=gw.det[det_name]["color"],
                zorder=gw.det[det_name]["zorder"],
                label=gw.det[det_name]["plot_name"],
            )
        else:
            ax.fill_between(
                np.log10(gw.det[det_name]["f_PLI"]),
                np.log10(gw.det[det_name]["PLI"]),
                0,
                alpha=0,
                color="none",
                hatch="\\\\",
                edgecolor=gw.det[det_name]["color"],
                linewidth=0.0,
                zorder=gw.det[det_name]["zorder"],
                label=gw.det[det_name]["plot_name"],
            )
        

    
    add_violins(ax, violincolor="#C0105A", nfreq=nfreq)
    add_NeffHatch(ax, Neffcolor="grey", h2OmegaNeff=2.48e-7) 
    if not call_from_spectrum:
        ax.set_title("PLI sensitivities and PTA violins")
    ax.set_xlabel(R"$f$ / $\mathrm{Hz}$")
    ax.set_ylabel(R'$h^2 \Omega_\mathrm{GW} (f)$')
    
    ax.tick_params(axis='both', which='both', direction="in", width=0.5, zorder=1000)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    for axis in ["top", 'bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
        ax.spines[axis].set_zorder(10000)
        
    xlim = [-10,-6]
    xa = np.linspace(int(xlim[0]), int(xlim[1]), int(xlim[1] - xlim[0] + 1))
    xtMajor = [j for j in xa]
    xtMinor = [np.log10(i*10**j) for j in xa for i in range(10)[1:10]]
    label_map = {-9: "nHz", -6: "µHz", -3: "mHz", 0: "Hz", 3: "kHz"}
    xlMajor = [label_map.get(i, "") for i in xa]
    xMajorLocator = FixedLocator(xtMajor)
    xMinorLocator = FixedLocator(xtMinor)
    xMajorFormatter = FixedFormatter(xlMajor)
    ax.xaxis.set_major_locator(xMajorLocator)
    ax.xaxis.set_minor_locator(xMinorLocator)
    ax.xaxis.set_major_formatter(xMajorFormatter)
    ax.set_xlim(-9.5, -5.5)

    ylim = [-18, -6]
    ya = np.linspace(ylim[0], ylim[1], int(ylim[1] - ylim[0] + 1))
    ytMajor = [j for j in ya]
    ytMinor = [np.log10(i*10**j) for i in range(10)[1:10] for j in ya]
    ylMajor = [r"$10^{" + str(int(i)) + "}$" if i%2==1 else "" for i in ya]
    yMajorLocator = FixedLocator(ytMajor)
    yMinorLocator = FixedLocator(ytMinor)
    yMajorFormatter = FixedFormatter(ylMajor)
    ax.yaxis.set_major_locator(yMajorLocator)
    ax.yaxis.set_minor_locator(yMinorLocator)
    ax.yaxis.set_major_formatter(yMajorFormatter)
    ax.set_ylim(ylim[0], ylim[1])
    ax.yaxis.set_zorder(1000)
    ax.xaxis.set_zorder(1000)
    ax.set_ylim(-16, -6)

    #ax.grid()
    add_TL_logo(loc="outside", ax=ax, small_logo=False)
    #fig.legend(loc="lower right", bbox_to_anchor=(1.0, 0.7), ncol=1)
    if not call_from_spectrum:
        plt.tight_layout()
        if showplot:
            plt.show()
        else:
            from pathlib import Path
            Path(gw.foldername).mkdir(parents=True, exist_ok=True)
            plt.savefig(gw.foldername+"GW_sensitivities.pdf")
        plt.close(fig)
    else:
        return fig, ax

def plotSpectrum(gw, showplot : bool=True):
    """Plot the total spectrum and its components together with sensitivities.
    
    Parameters
    ----------
    gw : GWSpectrum
        The GWSpectrum object containing the spectrum data.
    showplot : bool, optional
        Whether to display the plot.
    """
    fig, ax = plotSensitivities(gw, showplot=False, call_from_spectrum=True)
    frequencies_Hz = np.logspace(-10, 4, 1000)  # Hz
    sourcecolors = [DESYcyan, DESYorange, DESYmagenta]
    with np.errstate(divide='ignore'):
        for i, source in enumerate(gw.fopt_spectrum.sources):
            ax.plot(np.log10(frequencies_Hz), np.log10(gw.my_spec(
                frequencies_Hz, source)), label=source, color=sourcecolors[i], zorder=100+i)
        ax.plot(np.log10(frequencies_Hz), np.log10(gw.my_spec(
            frequencies_Hz)), label="Total", color="black", ls="dashed", zorder=100+i+1)

    fig.tight_layout()
    ax.legend(bbox_to_anchor=(1, 0.7), ncol=1)
    
    if showplot:
        plt.show()
    else:
        from pathlib import Path
        Path(gw.foldername).mkdir(parents=True, exist_ok=True)
        plt.savefig(gw.foldername+"GW_spectrum.pdf")
    plt.close(fig)


def plotProfile2(transitions, conversionFactor):
    """Standalone function to plot the bubble profile.

    Parameters
    ----------
    transitions : list[TransitionInfo]
        List of the transition objects
    conversionFactor : float
        Conversion factor to go from internal units to GeV
    Returns
    -------
    None."""
    fopts = [tr for tr in transitions if getattr(tr, "type", None) == 1]
    if not fopts:
        console.print("[bold yellow]No first-order transitions available for profile plot.[/bold yellow]")
        return

    height = 3.0
    fig, axes = plt.subplots(len(fopts), 1, figsize=(6.4, height * len(fopts)))
    if len(fopts) == 1:
        axes = [axes]
    for ax in axes:
        ax.tick_params(axis="both", which="both", labelsize=10, direction="in", width=0.5)
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_linewidth(0.5)
    ylabel = r"$\phi-\phi_\mathrm{min}^\mathrm{false}$ / $\mathrm{GeV}$"
    for ax, tr in zip(axes, fopts):
        instanton = tr.instanton
        Phis = instanton.Phi
        for i in range(Phis.shape[-1]):
            lbl = r"$\phi_" + f"{i}" + r"$"
            ax.plot(
                instanton.profile1D.R / conversionFactor,
                Phis[...,i] * conversionFactor, label=lbl,
            )
        ax.set_ylabel(ylabel)
        tnuc_gev = tr.Tnuc * conversionFactor
        prefix = r"T_\mathrm{nuc}"
        if tnuc_gev < 1e-2 or tnuc_gev > 1e4:
            exponent = int(np.floor(np.log10(tnuc_gev)))
            mantissa = tnuc_gev / 10 ** exponent
            label = rf"{prefix} = {mantissa:.3f} \cdot 10^{{{exponent}}}\,\mathrm{{GeV}}"
        else:
            label = rf"{prefix} = {tnuc_gev:g}\,\mathrm{{GeV}}"
        ax.set_title(
            fr"Bounce at ${label}$ from phase {tr.high_phase} to {tr.low_phase}"
        )
    axes[-1].set_xlabel(r"Bubble radius / $\mathrm{GeV}^{-1}$")
    add_TL_logo(loc="lower right", magnification=2, ax=axes[-1])
    fig.tight_layout()
    plt.legend()
    # fig.savefig(self.output_folder_name + "profile.pdf")
    plt.show()

def plotGWSpectrum(gwparams_dict: dict, showplot: bool=False, saveplot: bool=True,
                  foldername: str="", filename: str="", fig=None, ax=None,
                  legendfontsize: int=8):
    """Build the spectrum from parameters and plot it for the core detectors.
    
    Parameters
    ----------
    gwparams_dict : dict
        Dictionary containing the parameters for the GW spectrum.
    showplot : bool, optional
        Whether to display the plot.
    foldername : str, optional
        Folder to save the plot if showplot is False.
    filename : str, optional
        Filename to save the plot if showplot is False.
    
    """
    astroparams_dict = {}
    gwspec = FOPTspectrum(gwparams_dict, astroparams_dict, verbose=False)
    gw = Observability(gwspec, verbose=False, include_smbhb=False)
    figwasnone = False
    if fig == None:
        figwasnone = True
    if ax == None:
        fig, ax = plotSensitivities(gw, showplot=False, call_from_spectrum=True, fig=fig, ax=ax)
    frequencies_Hz = np.logspace(-10, 4, 1000)  # Hz
    sourcecolors = [DESYcyan, DESYorange, DESYmagenta]
    with np.errstate(divide='ignore'):
        for i, source in enumerate(gwspec.sources):
            ax.plot(np.log10(frequencies_Hz), np.log10(gw.my_spec(
                frequencies_Hz, source)), label=source, color=sourcecolors[i], zorder=100+i)
        ax.plot(np.log10(frequencies_Hz), np.log10(gw.my_spec(
            frequencies_Hz)), label="Total", color="black", ls="dashed", zorder=100+i+1)

    if figwasnone:
        fig.tight_layout()
        ax.legend(bbox_to_anchor=(1.05, 0.6), ncol=1, fontsize=legendfontsize)

    if showplot:
        plt.show()
        plt.close(fig)
    elif saveplot:
        from pathlib import Path
        Path(foldername).mkdir(parents=True, exist_ok=True)
        if filename != "":
            plt.savefig(foldername+filename)
        else:
            plt.savefig(foldername+"GW_spectrum.pdf")
        plt.close(fig)
    else:
        return fig, ax


def plotSpectrum2PTA(gwparams_dict : dict, showplot : bool=True,
                     foldername : str="", filename : str="", title : str=""):
    """Build the spectrum and plot it together with PTA sensitivities."""
    astroparams_dict = {}
    gwspec = FOPTspectrum(gwparams_dict, astroparams_dict, verbose=False)
    gw = Observability(gwspec, verbose=False, include_smbhb=False)
    fig, ax = plotSensitivitiesPTA(gw, showplot=False, call_from_spectrum=True, nfreq=14)
    frequencies_Hz = np.logspace(-10, -5, 1000)  # Hz
    sourcecolors = [DESYcyan, DESYorange, DESYmagenta]
    with np.errstate(divide='ignore'):
        for i, source in enumerate(gwspec.sources):
            ax.plot(np.log10(frequencies_Hz), np.log10(gw.my_spec(
                frequencies_Hz, source)), label=source, color=sourcecolors[i], zorder=100+i)
        ax.plot(np.log10(frequencies_Hz), np.log10(gw.my_spec(
            frequencies_Hz)), label="Total", color="black", ls="dashed", zorder=100+i+1)

    fig.tight_layout()
    fig.legend(bbox_to_anchor=(0.99, 0.55), ncol=1, fontsize=7)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.92)
    if showplot:
        plt.show()
    else:
        from pathlib import Path
        Path(foldername).mkdir(parents=True, exist_ok=True)
        if filename != "":
            plt.savefig(foldername+filename)
        else:
            plt.savefig(foldername+"GW_spectrum_PTA.pdf")
    plt.close(fig)

def plot2d(pot : generic_potential, box : tuple, ax : Optional[plt.Axes] = None,
           T : float = 0, treelevel : bool = False, offset : float = 0,
            xaxis : int = 0, yaxis : int = 1, n : int = 50, clevs : int = 200,
            cfrac : float = .8, log : bool = False, **contourParams):
    """
    Makes a countour plot of the potential.

    Parameters
    ----------
    box : tuple
        The bounding box for the plot, (xlow, xhigh, ylow, yhigh).
    T : float, optional
        The temperature
    offset : array_like
        A constant to add to all coordinates. Especially
        helpful if Ndim > 2.
    xaxis, yaxis : int, optional
        The integers of the axes that we want to plot.
    n : int
        Number of points evaluated in each direction.
    clevs : int
        Number of contour levels to draw.
    cfrac : float
        The lowest contour is always at ``min(V)``, while the highest is
        at ``min(V) + cfrac*(max(V)-min(V))``. If ``cfrac < 1``, only part
        of the plot will be covered. Useful when the minima are more
        important to resolve than the maximum.
    log : bool, optional
        Log scale of the colorbar.
    contourParams :
        Any extra parameters to be passed to :func:`plt.contour`.
    """
    xmin, xmax, ymin, ymax = box

    if ax is None:
        fig = plt.figure(figsize=(5.31,3),  edgecolor="white")
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(axis='both', which='both', labelsize=10, direction="in", width=0.5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
    else:
        fig = ax.figure

    X = np.linspace(xmin, xmax, n).reshape(n, 1) * np.ones((1, n))
    Y = np.linspace(ymin, ymax, n).reshape(1, n) * np.ones((n, 1))
    XY = np.zeros((n, n, pot.Ndim))
    XY[..., xaxis], XY[..., yaxis] = X, Y
    XY += offset
    Z = pot.V0(XY) if treelevel else pot.DVtot(XY, T)
    X_plot = X * pot.conversionFactor
    Y_plot = Y * pot.conversionFactor
    Z_plot = Z * pot.conversionFactor**4
    if log:
        Z_plot = np.log10(Z_plot - np.min(Z_plot) + 1)
    minZ, maxZ = min(Z_plot.ravel()), max(Z_plot.ravel())
    N = np.linspace(minZ, minZ + (maxZ - minZ) * cfrac, clevs)
    contour = ax.contour(X_plot, Y_plot, Z_plot, N, cmap=cmapTL2, **contourParams)
    cbar = fig.colorbar(contour, ax=ax)
    if log:
        cbar.set_label(r"$log_10(V - V_{\mathrm{min}} + 1)$")
    else:
        cbar.set_label(r"$V / \mathrm{GeV}^4$")
    ax.set_xlabel(r"$\phi_1$ / $\mathrm{GeV}$")
    ax.set_ylabel(r"$\phi_2$ / $\mathrm{GeV}$")
    ax.yaxis.set_zorder(1000)
    ax.xaxis.set_zorder(1000)
    ax.set_xlim(xmin*pot.conversionFactor, xmax*pot.conversionFactor)
    ax.set_ylim(ymin*pot.conversionFactor, ymax*pot.conversionFactor)
    # Matplotlib compatibility: avoid switching layout engines when the figure
    # already uses one (e.g. constrained layout).
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    return fig, ax


def plot1d(pot : generic_potential, x1 : float, x2 : float, ax : Optional[plt.Axes] = None,
           T : float = 0, treelevel : bool = False, subtract : bool = True, n : int = 500,
           **plotParams):
    """Plot the 1D projection of the potential between two field values.
    
    Parameters
    ----------
    x1 : float
        The starting field value.
    x2 : float
        The ending field value.
    T : float, optional
        The temperature.
    treelevel : bool, optional
        Whether to plot the tree-level potential or the full one-loop
        potential including thermal corrections.
    subtract : bool, optional
        Whether to subtract the potential at zero field value.
    n : int, optional
        Number of points to evaluate.
    plotParams :
        Any extra parameters to be passed to :func:`plt.plot`.
    """
    if pot.Ndim == 1:
        x = np.linspace(x1, x2, n)
        X = x[:, np.newaxis]
    else:
        dX = np.array(x2) - np.array(x1)
        X = dX * np.linspace(0, 1, n)[:, np.newaxis] + x1
        x = np.linspace(0, 1, n) * np.sum(dX**2)**.5
    if treelevel:
        y = pot.V0(X) - pot.V0(X * 0) if subtract else pot.V0(X)
    else:
        y = pot.DVtot(X, T) if subtract else pot.Vtot(X, T)

    if ax is None:
        fig = plt.figure(figsize=(5.31,3),  edgecolor="white")    
        ax = fig.add_subplot(1,1,1)
        ax.tick_params(axis='both', which='both', labelsize=10, direction="in", width=0.5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
            
    ax.plot(x * pot.conversionFactor, y * pot.conversionFactor**4, **plotParams)
    ax.set_xlabel(R"$\phi$ / $\mathrm{GeV}$")
    ax.set_ylabel(R"$V(\phi) / \mathrm{GeV}^4$")
    ax.yaxis.set_zorder(1000)
    ax.xaxis.set_zorder(1000)
    ax.set_xlim(x1*pot.conversionFactor, x2*pot.conversionFactor)
    fig.tight_layout()
    return fig, ax

def add_violins(ax : plt.Axes, violincolor : str = DESYdunkelrot, nfreq : int = 14):
    """Add stylistic violin markers to highlight PTA frequency bins.
    
    Parameters
    ----------
    ax : plt.Axes
        The axes to add the violins to.
    violincolor : str, optional
        The color of the violins.
    nfreq : int, optional
        Number of frequency bins for the violins. Default is 14.
    """
    directory = DATA_DIR / "30f_fs{hd}_100fDMGP_ceffyl"

    def log10_h2O(log10_rho, f, T):
        return np.log10(8 * np.pi**4 / cn.H100_Hz**2 * 10**(2 * log10_rho) * f**5 * T)

    N_violin = int(1e5)
    density = np.load(directory / 'density.npy')
    freqs = np.load(directory / 'freqs.npy')
    log10rhogrid = np.load(directory / 'log10rhogrid.npy')

    for i in range(nfreq):
        samplebin = np.array(
            log10_h2O(
                _choice(
                    a=log10rhogrid,
                    p=np.exp(density[0,i,:])/np.sum(np.exp(density[0,i,:])),
                    size=N_violin
                ),
                freqs[i],
                1/freqs[0]
                )
        )

        violin_parts = ax.violinplot(dataset=(samplebin),
            positions=[np.log10(freqs[i])]
            , widths=[np.log10(freqs[i])*.01], showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set(color=violincolor, edgecolor=None, alpha=1, zorder=100)
        if i == 0:
            pc.set(label="NG15 (violins)")

def add_NeffHatch(ax : plt.Axes, Neffcolor : str = DESYpetrol, h2OmegaNeff: float = 2.48e-7):
    """Overlay a hatched region indicating the :math:`\\Delta N_\\mathrm{eff}` bound.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the hatch to.
    Neffcolor : str, optional
        The color of the hatch.
    h2OmegaNeff : float, optional
        The upper bound on :math:`h^2 \\Omega_\\mathrm{GW}` from :math:`\\Delta N_\\mathrm{eff}`.
        Default is ``2.48e-7``.
    """
    fNeff = 1.5e-11
    frequencies_Hz = np.linspace(-12, 5, 3)  # Hz
    ax.hlines(y=np.log10(h2OmegaNeff), xmin=np.log10(fNeff), xmax=20, linewidth=1, color=Neffcolor, zorder=20)
    ax.fill_between(
        frequencies_Hz,
        np.ones_like(frequencies_Hz) * np.log10(h2OmegaNeff),
        1,
        color="white",
        linewidth=0.0,
        zorder=10,
        alpha=1,
    )
    ax.fill_between(
        frequencies_Hz,
        np.ones_like(frequencies_Hz) * np.log10(h2OmegaNeff),
        1,
        color=Neffcolor,
        linewidth=0.0,
        zorder=10,
        alpha=0.5,
    )
    ax.fill_between(
        frequencies_Hz,
        np.ones_like(frequencies_Hz) * np.log10(h2OmegaNeff),
        1,
        color="none",
        hatch="//",
        edgecolor=Neffcolor,
        linewidth=0.0,
        zorder=10,
        label=R"$N_\mathrm{eff}$",
    )
