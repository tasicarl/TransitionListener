"""Sensitivity curves and observability metrics for gravitational wave experiments.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import os
from collections import OrderedDict

import numpy as np
from scipy import integrate
try:  # Optional PTA dependencies; skipped if environment lacks tempo2/ptarcade stack.
    from ptarcade import signal_builder  # type: ignore
    from enterprise.signals.parameter import Uniform
    _PTA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    signal_builder = None  # type: ignore
    Uniform = None  # type: ignore
    _PTA_IMPORT_ERROR = exc

from . import console, print
from .colors import *
from .constants import *

import rich


def _require_pta_dependencies() -> None:
    """Raise a clear error if PTA dependencies are unavailable."""
    if _PTA_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PTA likelihood evaluation requires ptarcade/enterprise/tempo2; "
            "set up tempo2 and ptarcade or disable PTA calculations."
        ) from _PTA_IMPORT_ERROR

# Mapping of PTA likelihood settings keyed by the label used throughout the codebase.
PTA_LIKELIHOOD_SETTINGS = OrderedDict(
    # Best-fit likelihoods were obtained from PTArcade ultranest scans of
    # first-order phase-transition GW spectra following arXiv:2403.03723.
    [
        (
            "NG15_14bins",
            {
                "pta_name": "NG15",
                "nbins": 14,
                "threshold": -110.0,
                "bestfit_lnL": -103.12759208991564,
            },
        ),
        (
            "NG15_5bins",
            {
                "pta_name": "NG15",
                "nbins": 5,
                "threshold": -35.0,
                "bestfit_lnL": -29.55312949665563,
            },
        ),
        (
            "NG12_5bins",
            {
                "pta_name": "NG12",
                "nbins": 5,
                "threshold": -45.0,
                "bestfit_lnL": -36.89179733242301,
            },
        ),
        (
            "IPTA2_13bins",
            {
                "pta_name": "IPTA2",
                "nbins": 13,
                "threshold": -115.0,
                "bestfit_lnL": -105.66902507253809,
            },
        ),
    ]
)

PTA_LIKELIHOOD_VARIANTS = ("PTArcade", "mock", "smooth")

PTA_DISPLAY_NAMES = {
    "NG15_14bins": "NANOGrav 15yr",
    "NG15_5bins": "NANOGrav 15yr",
    "NG12_5bins": "NANOGrav 12.5yr",
    "IPTA2_13bins": "IPTA DR2",
}

MOCK_LNL_REFERENCE = {
    "f_peak_Hz_log10": -7.70241036708,
    "h2OmegaGW_peak_log10": -8.134671889762169,
}

EXISTING_OBSERVATORIES = {
    "EPTA_18_yrs",
    "NANOGrav_11_yrs",
    "NANOGrav_15_yrs",
    "HLV_O2"
}

DELTA_LNL_WITHIN_THRESHOLD = 4.5

class GW_Sensitivity_Data:
    """
    Class used to import the sensitivity data curves of GW observaories.

    Attributes
    ----------
    self.det_f : float array
            Frequencies from the detector data
    self.det_data : float array
            Detector sensitivity curves
    self.det_names : string tuple
            Deterctor names, ordered as in input file
    self.det_colors : string tuple
            Colors for the detector sensitivities in plots, ordered as in input file
    self.det : dictionary
            Dicitonary for the sensitivitie with the detector name as key
    self.det_thr : float tuple
            Expected detector threshold SNRs
    self.det_tobs : float tuple
            Expected detector observation times
    """

    def __init__(self):
        """Load sensitivity and power-law integrated curves for supported detectors."""
        data_path = os.path.join(os.path.dirname(__file__), "tab_data")

        def _csv_path(filename: str) -> str:
            return os.path.join(data_path, filename)

        detectors = self._load_ground_space_detectors(_csv_path)

        det_names = (
            "SKA_5_yrs",
            "SKA_10_yrs",
            "SKA_20_yrs",
            "EPTA_18_yrs",
            "NANOGrav_11_yrs",
            "NANOGrav_15_yrs",
            "LISA",
            "B-DECIGO",
            "DECIGO",
            "BBO",
            "ET",
            "muAres",
            "HLV_O2",
            "HLVK_design",
        )
        mu_ares = self._load_mu_ares(_csv_path)
        ground_based = self._load_ground_based(_csv_path)
        ng15 = self._load_ng15(_csv_path)

        combined_curves = {}
        for curves in (detectors["curves"], mu_ares["curves"], ground_based["curves"], ng15["curves"]):
            combined_curves.update(curves)
        det_files = {name: combined_curves[name] for name in det_names}

        det_plot_names = (
            r"SKA $5 \, \mathrm{yr}$",
            r"SKA $10 \, \mathrm{yr}$",
            r"SKA $20 \, \mathrm{yr}$",
            r"EPTA $18 \, \mathrm{yr}$",
            r"NANOGrav $11 \, \mathrm{yr}$",
            r"NANOGrav $15 \, \mathrm{yr}$",
            "LISA",
            "B-DECIGO",
            "DECIGO",
            "BBO",
            "ET",
            r"$\mu$Ares",
            "LIGO-VIRGO O2",
            "HLVK design",
        )
        det_single_obs_names = {"LISA", "B-DECIGO", "ET", "muAres"}
        det_no_plot = {"EPTA_18_yrs", "NANOGrav_11_yrs", "B-DECIGO", "DECIGO", "HLVK_design"}
        det_colors = (
            "#FFD700",  # SKA_5_yrs       – bright gold
            "#FF7F00",  # SKA_10_yrs      – vivid orange
            "#E8000B",  # SKA_20_yrs      – pure red
            "white",    # EPTA_18_yrs (hidden)
            "white",    # NANOGrav_11_yrs (hidden)
            "#C0105A",  # NANOGrav_15_yrs – crimson-magenta
            "#4477AA",  # LISA            – blue
            "white",    # B-DECIGO (hidden)
            "white",    # DECIGO (hidden)
            "#66CCEE",  # BBO             – cyan
            "#AA4499",  # ET              – purple
            "#00C2AB",  # muAres          – teal
            "#002147",  # HLV_O2          – very dark navy blue
            "white",    # HLVK_design (hidden)
        )
        det_zorder = (3, 2, 1, 5, 6, 4, 5, 3, 1, 2, 3, 1, 7, 8)
        det_thr = (4, 4, 4, 1.19, 0.697, 5, 10, 8, 10, 10, 5, 10, 1, 1)
        det_tobs = (5*yr, 10*yr, 20*yr, 18*yr, 11*yr, 15*yr, 4*yr, 4*yr, 4*yr, 4*yr, 5*yr, 7*yr, 1*yr, 1*yr)

        self.det = {}
        for i, name in enumerate(det_names):
            f_noise, noise, f_PLI, PLI = det_files[name]
            self.det[name] = {
                "plot_name": det_plot_names[i],
                "f_noise": f_noise,
                "noise": noise,
                "f_PLI": f_PLI,
                "PLI": PLI,
                "thr": det_thr[i],
                "tobs": det_tobs[i],
                "color": det_colors[i],
                "zorder": det_zorder[i],
                "no_plot": name in det_no_plot,
                "single_obs": name in det_single_obs_names,
            }

        self.det_names = det_names
        self.PTA_labels = tuple(PTA_LIKELIHOOD_SETTINGS.keys())
        self.PTA_names = tuple(
            str(cfg["pta_name"]) for cfg in PTA_LIKELIHOOD_SETTINGS.values()
        )
        self.PTA_bins = tuple(int(cfg["nbins"]) for cfg in PTA_LIKELIHOOD_SETTINGS.values())
        self.foldername = ""

    def _load_ground_space_detectors(self, path_func):
        """Load SKA/LISA-style detector curves (Breitbach & Schwaller et al., 1811.11175)."""
        det_PLI_f, *det_PLI_data = np.genfromtxt(path_func("PLI-dataset.csv"), delimiter=",", unpack=True, skip_header=1)
        det_noise_f, *det_noise_data = np.genfromtxt(path_func("noise-dataset.csv"), delimiter=",", unpack=True, skip_header=1)
        names = (
            "SKA_5_yrs",
            "SKA_10_yrs",
            "SKA_20_yrs",
            "EPTA_18_yrs",
            "NANOGrav_11_yrs",
            "NANOGrav_15_yrs",
            "LISA",
            "B-DECIGO",
            "DECIGO",
            "BBO",
            "ET",
        )
        curves = {
            "SKA_5_yrs": (det_noise_f, det_noise_data[0], det_PLI_f, det_PLI_data[0]),
            "SKA_10_yrs": (det_noise_f, det_noise_data[1], det_PLI_f, det_PLI_data[1]),
            "SKA_20_yrs": (det_noise_f, det_noise_data[2], det_PLI_f, det_PLI_data[2]),
            "EPTA_18_yrs": (det_noise_f, det_noise_data[3], det_PLI_f, det_PLI_data[3]),
            "NANOGrav_11_yrs": (det_noise_f, det_noise_data[4], det_PLI_f, det_PLI_data[4]),
            "LISA": (det_noise_f, det_noise_data[5], det_PLI_f, det_PLI_data[5]),
            "B-DECIGO": (det_noise_f, det_noise_data[6], det_PLI_f, det_PLI_data[6]),
            "DECIGO": (det_noise_f, det_noise_data[7], det_PLI_f, det_PLI_data[7]),
            "BBO": (det_noise_f, det_noise_data[8], det_PLI_f, det_PLI_data[8]),
            "ET": (det_noise_f, det_noise_data[9], det_PLI_f, det_PLI_data[9]),
        }
        return {"names": names, "curves": curves}

    def _load_mu_ares(self, path_func):
        """Load μAres sensitivity and PLI curves (Sesana et al., 1908.11391)."""
        det_muAres_PLI_f, det_muAres_PLI_data = np.genfromtxt(path_func("muAres_f_Hz_PLI_h2OmegaGW.csv"), delimiter=";", unpack=True)
        det_muAres_noise_f, det_muAres_noise_data = np.genfromtxt(path_func("muAres_f_Hz_h2Omega_noise.csv"), delimiter=";", unpack=True)
        curves = {
            "muAres": (det_muAres_noise_f, det_muAres_noise_data, det_muAres_PLI_f, det_muAres_PLI_data)
        }
        return {"names": ("muAres",), "curves": curves}

    def _load_ground_based(self, path_func):
        """Load the LIGO/Virgo design and O(2) curves (Schmitz et al., 2002.04615)."""
        det_HLV_O2_PLI_f, det_HLV_O2_PLI_data = np.genfromtxt(path_func("HLV-O2-PLI.csv"), delimiter=";", unpack=True)
        det_HLV_O2_noise_f, det_HLV_O2_noise_data = np.genfromtxt(path_func("HLV-O2-noise.csv"), delimiter=";", unpack=True)
        det_HLVK_design_PLI_f, det_HLVK_design_PLI_data = np.genfromtxt(path_func("HLVK-design-PLI.csv"), delimiter=";", unpack=True)
        det_HLVK_design_noise_f, det_HLVK_design_noise_data = np.genfromtxt(path_func("HLVK-design-noise.csv"), delimiter=";", unpack=True)
        curves = {
            "HLV_O2": (det_HLV_O2_noise_f, det_HLV_O2_noise_data, det_HLV_O2_PLI_f, det_HLV_O2_PLI_data),
            "HLVK_design": (
                det_HLVK_design_noise_f,
                det_HLVK_design_noise_data,
                det_HLVK_design_PLI_f,
                det_HLVK_design_PLI_data,
            ),
        }
        return {"names": ("HLV_O2", "HLVK_design"), "curves": curves}

    def _load_ng15(self, path_func):
        """Load the NANOGrav 15yr data and convert it to h^2Ω_GW (NANOGrav Collab., 2306.16218)."""
        det_NG15_PLI_f, det_NG15_PLI_data_hc = np.genfromtxt(path_func("NG15_PLI_hc.csv"), delimiter=",", unpack=True, skip_header=1)
        det_NG15_noise_f, det_NG15_noise_data_hc = np.genfromtxt(path_func("NG15_noise_hc.csv"), delimiter=",", unpack=True, skip_header=1)

        def _hc_to_h2OmegaGW(f_Hz, h_c):
            return 2 * np.pi**2 / 3.0 * (f_Hz**2) * (h_c**2) / H100_Hz**2

        det_NG15_PLI_f = det_NG15_PLI_f[::-1]
        det_NG15_noise_f = det_NG15_noise_f[::-1]
        det_NG15_PLI_data_hc = det_NG15_PLI_data_hc[::-1]
        det_NG15_noise_data_hc = det_NG15_noise_data_hc[::-1]
        det_NG15_PLI_data = _hc_to_h2OmegaGW(det_NG15_PLI_f, det_NG15_PLI_data_hc)
        det_NG15_noise_data = _hc_to_h2OmegaGW(det_NG15_noise_f, det_NG15_noise_data_hc)

        curves = {
            "NANOGrav_15_yrs": (det_NG15_noise_f, det_NG15_noise_data, det_NG15_PLI_f, det_NG15_PLI_data)
        }
        return {"names": ("NANOGrav_15_yrs",), "curves": curves}

# ==================================================
# PTArcade compatible config and signal models
# ==================================================

class PTAConfig:
    """Minimal PTArcade-style configuration container used by the PTA likelihood wrappers."""

    def __init__(self, nfreqBins=14, PTA="NG15"):
        """Store the PTA dataset label and the number of GW frequency bins."""
        self.pta_data = PTA  # "NG15" or "NG12" or "IPTA2"
        self.corr = False
        self.red_components = 30
        self.gwb_components = nfreqBins
        self.bhb_th_prior = True
        self.mod_sel = False
        self.A_bhb_logmin = False
        self.A_bhb_logmax = False

class PTAModel:
    """Adapter exposing a TransitionListener spectrum in the format expected by PTArcade."""

    def __init__(self, spectrum):
        """Store the stochastic background spectrum callable and metadata."""
        self.spectrum = spectrum
        self.smbhb = False
        self.parameters = {}

class PTAModel_SMBHB:
    """PTArcade adapter including the nuisance parameters for an SMBHB foreground."""

    def __init__(self, spectrum):
        """Store the spectrum and freeze the astrophysical nuisance priors."""
        self.spectrum = spectrum
        self.smbhb = False
        self.parameters = {
            # The ranges are irrelevant for the likelihood calculation
            'A': Uniform(0, 0)('A'),
            'gamma': Uniform(0, 0)('gamma')
        }



class Observability(GW_Sensitivity_Data):
    """Compute SNRs, PTA likelihoods and derived quantities for a GW spectrum."""

    def __init__(self, gwspec_dict: dict, verbose: bool=False,
                 include_smbhb: bool=False):
        """Store the supplied spectrum and immediately evaluate all observables."""
        super(Observability, self).__init__()
        self.gwspec_dict = gwspec_dict
        self.verbose = verbose

        if include_smbhb:
            self.my_spec = gwspec_dict["h2OmegaGW_with_astrobg_fun"]
        else:
            self.my_spec = gwspec_dict["h2OmegaGW_fun"]
        self.my_spec_with_astrobg = gwspec_dict["h2OmegaGW_with_astrobg_fun"]
        self.add_spec_info = gwspec_dict["additional_spec_info"]

        self.h2OmegaGW = gwspec_dict["h2OmegaGW"]
        self.h2OmegaGW_with_astrobg = gwspec_dict["h2OmegaGW_with_astrobg"]

        self.calc_SNR()
        self.pta_likelihoods = {label: {} for label in self.PTA_labels}
        self.ptas = {}
        if signal_builder is not None and _PTA_IMPORT_ERROR is None:
            for label, settings in PTA_LIKELIHOOD_SETTINGS.items():
                if include_smbhb:
                    input_payload = {
                        "config": PTAConfig(
                            int(settings["nbins"]),
                            str(settings["pta_name"]),
                        ),
                        "model": PTAModel_SMBHB(self.my_spec_with_astrobg),
                    }
                else:
                    input_payload = {
                        "config": PTAConfig(
                            nfreqBins=int(settings["nbins"]),
                            PTA=str(settings["pta_name"]),
                        ),
                        "model": PTAModel(self.my_spec),
                    }
                self.ptas[label] = signal_builder.ceffyl_builder(input_payload)
            if include_smbhb:
                self.calc_PTA_logL_with_astronoise(
                    gwspec_dict.astroparams_dict["A"],
                    gwspec_dict.astroparams_dict["gamma"],
                )
            else:
                self.calc_PTA_logL()
        elif self.verbose:
            console.print(
                "[yellow]PTA likelihood dependencies unavailable; "
                "continuing with SNR-only observability outputs.[/yellow]"
            )
        self.calc_DNeff_GW()
        self.observability_dict = self.return_gw_info_dict()

        if self.verbose:
            self.print_additional_spec_info()

    def __iter__(self):
        """Enable iteration over the assembled observability dictionary."""
        return iter(self.observability_dict)

    def __getitem__(self, key: str):
        """Access individual observables by key."""
        return self.observability_dict[key]

    def __len__(self):
        """Return the number of entries in the observability dictionary."""
        return len(self.observability_dict)

    def keys(self):
        """Expose the keys present in the observability dictionary."""
        return self.observability_dict.keys()

    def _get_additional_value(self, key: str, variant: str) -> float:
        """Helper to access derived additional-spectrum values safely."""
        entry = self.add_spec_info.get(key, {})
        if isinstance(entry, dict):
            value = entry.get(variant, np.nan)
            if isinstance(value, (float, np.floating, int, np.integer)):
                return float(value)
        return float("nan")

    def _flatten_additional_spec_info(self) -> OrderedDict[str, float | str]:
        """Return an ordered mapping with lin/log/latex variants for output."""
        flattened: OrderedDict[str, float | str] = OrderedDict()
        for key, entry in self.add_spec_info.items():
            if not isinstance(entry, dict):
                continue
            lin = entry.get("lin", np.nan)
            log10 = entry.get("log10", np.nan)
            latex = entry.get("latex")

            if np.isscalar(lin):
                flattened[f"{key}_lin"] = float(lin)
            else:
                flattened[f"{key}_lin"] = lin

            if np.isscalar(log10):
                flattened[f"{key}_log10"] = float(log10)
            else:
                flattened[f"{key}_log10"] = log10

            if latex is not None:
                flattened[f"{key}_latex"] = str(latex)

        return flattened


    def calc_SNR(self):
        """Compute the signal-to-noise ratio for each detector sensitivity curve."""
        def calc_SNR_for_det(detector_name):
            sensitivity = self.det[detector_name]["noise"]
            freq = self.det[detector_name]["f_noise"]
            frac = np.array(
                (self.my_spec(freq)/sensitivity)**2)
            integral = integrate.simpson(frac, freq)
            tobs = self.det[detector_name]["tobs"]
            snr2 = 2. * tobs * integral
            if self.det[detector_name]["single_obs"]:
                snr2 /= 2.
            return np.sqrt(snr2)

        SNR = np.zeros(len(self.det_names))
        detectable = np.zeros(len(self.det_names), dtype=bool)
        for i, name in enumerate(self.det_names):
            SNR[i] = calc_SNR_for_det(name)
            log_thr = np.log10(self.det[name]["thr"])
            detectable[i] = bool(np.log10(SNR[i]) > log_thr)

        if self.verbose:
            existing_detectors = {
                "EPTA_18_yrs",
                "NANOGrav_11_yrs",
                "NANOGrav_15_yrs",
                "HLV_O2",
            }
            table = rich.table.Table(
                title="SNR VALUES AND OBSERVABILITY",
                title_justify="left",
                box=rich.box.ROUNDED,
            )
            table.add_column("Detector", style="cyan", no_wrap=True)
            table.add_column("SNR", style="orange1", justify="right")
            table.add_column(
                "Should have been observed already?",
                style="green",
                justify="center",
            )
            table.add_column("Observable?", style="green", justify="center")

            for n, v, d in zip(self.det_names, SNR, detectable):
                existing = n in EXISTING_OBSERVATORIES
                mark = "[green]\u2714[/green]" if d else "[red]\u274C[/red]"
                observed_col = mark if existing else ""
                future_col = mark if not existing else ""
                table.add_row(n, f"{v:.4e}", observed_col, future_col)
            console.print("\n", table)

        self.SNR = SNR
        self.detectable = detectable
        return SNR
    
    def calc_DNeff_GW(self):
        """Compute the contribution to the effective number of relativistic species from GWs."""
        def integrand(x):
            return self.my_spec(x)/x  # h2OmegaGW(f) / f as a function of f_Hz

        f_min = 1.5e-11  # Hz
        f_max = 1e10     # Hz
        freqs = np.logspace(np.log(f_min), np.log(f_max), 100000)
        integrand_values = integrand(freqs)
        h2OmegaGW = integrate.simpson(integrand_values, freqs)
        DNeff_GW = (8/7) * (11/4)**(4/3) * h2OmegaGW / Omega_gamma_h2
        self.DNeff_GW = DNeff_GW
        self.add_spec_info["DNeff_GW"] = {
            "lin": float(DNeff_GW),
            "log10": self._safe_log10(DNeff_GW),
            "latex": r"$\Delta N_\mathrm{eff}^{\mathrm{GW}}$",
            "label": "DNeff_GW",
        }
        return DNeff_GW

    def calc_PTA_logL(self):
        """Evaluate PTA log-likelihoods using PTArcade for the stored spectrum."""
        _require_pta_dependencies()

        logL_values: dict[str, float] = {}
        for label in PTA_LIKELIHOOD_SETTINGS:
            pta = self.ptas[label]
            logL_values[label] = float(pta.ln_likelihood(np.array([])))

        self._populate_pta_likelihoods(logL_values)

        if self.verbose:
            self._print_pta_likelihood_table()

        return self.logL_PTAs

    def _populate_pta_likelihoods(self, logL_values: dict[str, float]):
        """Store PTA likelihood variants, mock approximations, and derived metrics."""
        self.logL_PTAs = np.array(
            [logL_values.get(label, np.nan) for label in PTA_LIKELIHOOD_SETTINGS]
        )

        pta_likelihoods = {}
        for label, settings in PTA_LIKELIHOOD_SETTINGS.items():
            actual = logL_values.get(label, np.nan)
            threshold = float(settings["threshold"])
            bestfit = float(settings["bestfit_lnL"])
            mock = self._compute_mock_likelihood(threshold)
            smooth = self._compute_smoothened_likelihood(actual, mock, threshold)

            variant_data = {}
            for variant, value in (
                ("PTArcade", actual),
                ("mock", mock),
                ("smooth", smooth),
            ):
                delta = self._compute_delta_lnL(value, bestfit)
                sigma = self._delta_to_sigma(delta)
                within = self._within_three_sigma(delta) if variant == "PTArcade" else None
                variant_data[variant] = {
                    "lnL": value,
                    "delta_lnL": delta,
                    "sigma": sigma,
                    "within_3sigma": within,
                }

            pta_likelihoods[label] = variant_data

        self.pta_likelihoods = pta_likelihoods

    def _compute_mock_likelihood(self, threshold: float) -> float:
        """Compute the mock likelihood using peak frequency and amplitude."""
        log10_h2 = self._get_additional_value("h2OmegaGW_peak", "log10")
        log10_f = self._get_additional_value("f_peak_Hz", "log10")

        if not (np.isfinite(log10_h2) and np.isfinite(log10_f)):
            return np.nan

        diff_h2 = log10_h2 - MOCK_LNL_REFERENCE["h2OmegaGW_peak_log10"]
        diff_f = log10_f - MOCK_LNL_REFERENCE["f_peak_Hz_log10"]
        return threshold - diff_h2 ** 2 - diff_f ** 2

    @staticmethod
    def _compute_smoothened_likelihood(actual: float, mock: float, threshold: float) -> float:
        """Return PTArcade likelihood when above threshold, else the mock approximation."""
        if np.isfinite(actual) and actual >= threshold:
            return actual
        return mock if np.isfinite(mock) else actual

    @staticmethod
    def _compute_delta_lnL(value: float, bestfit: float) -> float:
        """Return the log-likelihood distance to the best-fit value."""
        if not np.isfinite(value):
            return np.inf
        return bestfit - value

    @staticmethod
    def _delta_to_sigma(delta_lnL: float) -> float:
        """Translate a log-likelihood difference into an approximate sigma distance."""
        if not np.isfinite(delta_lnL):
            return np.inf
        return np.sqrt(2.0 * max(delta_lnL, 0.0))

    @staticmethod
    def _within_three_sigma(delta_lnL: float) -> bool:
        """Return True if the delta log-likelihood is within the 3-sigma criterion."""
        return bool(np.isfinite(delta_lnL) and delta_lnL < DELTA_LNL_WITHIN_THRESHOLD)

    def _print_pta_likelihood_table(self):
        """Pretty-print PTA likelihood variants and their derived metrics."""
        table = rich.table.Table(
            title="LOG LIKELIHOODS FOR PULSAR TIMING ARRAYS",
            title_justify="left",
            box=rich.box.ROUNDED,
        )
        table.add_column("PTA", style="cyan", no_wrap=True)
        table.add_column("Bins", style="cyan", justify="center")
        table.add_column("Variant", style="magenta", no_wrap=True)
        table.add_column("lnL", style="orange1", justify="right")
        table.add_column("Δ lnL", style="orange1", justify="right")
        table.add_column("σ", style="orange1", justify="right")
        table.add_column("Within 3σ?", style="green", justify="center")

        for label, settings in PTA_LIKELIHOOD_SETTINGS.items():
            variant_data = self.pta_likelihoods.get(label, {})
            bins = int(settings["nbins"])
            display_name = PTA_DISPLAY_NAMES.get(label, label)
            for idx, variant in enumerate(PTA_LIKELIHOOD_VARIANTS):
                metrics = variant_data.get(variant, {})
                lnL_val = metrics.get("lnL", np.nan)
                delta_val = metrics.get("delta_lnL", np.nan)
                sigma_val = metrics.get("sigma", np.nan)
                within = metrics.get("within_3sigma") if variant == "PTArcade" else None

                lnL_str = self._format_float(lnL_val, decimals=2)
                delta_str = self._format_float(delta_val, decimals=2)
                sigma_str = self._format_float(sigma_val, decimals=2)
                mark = ""
                if variant == "PTArcade" and isinstance(within, bool):
                    mark = "[green]\u2714[/green]" if within else "[red]\u274C[/red]"

                end_section = variant == PTA_LIKELIHOOD_VARIANTS[-1]
                table.add_row(
                    display_name if idx == 0 else "",
                    str(bins),
                    variant,
                    lnL_str,
                    delta_str,
                    sigma_str,
                    f"{mark}",
                    end_section=end_section,
                )
        console.print("\n", table)

    @staticmethod
    def _format_float(value: float, decimals: int = 10) -> str:
        """Format floating-point values for readable table output."""
        if not np.isfinite(value):
            return "nan"
        return f"{value:.{decimals}f}"

    @staticmethod
    def _format_scientific(value: float) -> str:
        """Format a floating-point value using scientific notation."""
        if not np.isfinite(value):
            return "nan"
        return f"{value:.4e}"

    @staticmethod
    def _safe_log10(value: float) -> float:
        """Safely compute the base-10 logarithm of a positive value."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float("nan")
        if numeric > 0:
            return float(np.log10(numeric))
        return float("nan")

    def calc_PTA_logL_with_astronoise(self, A: float, gamma: float):
        """Evaluate PTA log-likelihoods including an astrophysical background."""
        _require_pta_dependencies()

        logL_values: dict[str, float] = {}
        for label in PTA_LIKELIHOOD_SETTINGS:
            pta = self.ptas[label]
            logL_values[label] = float(pta.ln_likelihood(np.array([A, gamma])))

        self._populate_pta_likelihoods(logL_values)

        if self.verbose:
            self._print_pta_likelihood_table()

        return self.logL_PTAs

    def print_additional_spec_info(self):
        """Render a table summarising derived GW spectrum properties."""
        table = rich.table.Table(
            title="Additional information on GW spectrum",
            title_justify="left",
            show_header=True,
            header_style="bold",
            box=rich.box.ROUNDED,
        )
        table.add_column("Quantity", style="cyan", no_wrap=True)
        table.add_column("Value", style="orange1", justify="right")

        for key, entry in self.add_spec_info.items():
            if not isinstance(entry, dict):
                continue
            label = entry.get("label", key)
            lin = entry.get("lin", np.nan)
            table.add_row(
                str(label),
                self._format_scientific(lin),
            )

        console.print("\n", table)

    def initialise_gw_info_dict(self):
        """Initialise an empty return object.

        Parameters
        ----------

        Returns
        -------

        """
        gw_info_dict = {}
        # Add SNR and detectable for each detector
        for name in self.det_names:
            gw_info_dict[f"{name}_SNR"] = np.nan
            gw_info_dict[f"{name}_detectable"] = False
        # Add logL for each PTA
        for label in self.PTA_labels:
            variant_data = self.pta_likelihoods.get(label, {})
            actual_metrics = variant_data.get("PTArcade", {})
            gw_info_dict[f"lnL_{label}"] = np.nan
            for variant in PTA_LIKELIHOOD_VARIANTS:
                metrics = variant_data.get(variant, {})
                gw_info_dict[f"lnL_{variant}_{label}"] = np.nan
                gw_info_dict[f"delta_lnL_{variant}_{label}"] = np.nan
                gw_info_dict[f"sigma_{variant}_{label}"] = np.nan
                gw_info_dict[f"within_3sigma_{variant}_{label}"] = np.nan
        # Add additional spectrum info
        for key, value in self._flatten_additional_spec_info().items():
            gw_info_dict[key] = value
        # Add DNeff
        gw_info_dict["DNeff_GW"] = np.nan
        return gw_info_dict


    def return_gw_info_dict(self):
        """Collect SNRs, likelihoods and auxiliary info into a result dictionary."""
        gw_info_dict = {}
        # Add SNR and detectable for each detector
        for name, snr, detectable in zip(self.det_names, self.SNR, self.detectable):
            gw_info_dict[f"{name}_SNR"] = snr
            gw_info_dict[f"{name}_detectable"] = detectable
        # Add logL for each PTA
        for label in self.PTA_labels:
            variant_data = self.pta_likelihoods.get(label, {})
            actual_metrics = variant_data.get("PTArcade", {})
            gw_info_dict[f"lnL_{label}"] = actual_metrics.get("lnL", np.nan)
            for variant in PTA_LIKELIHOOD_VARIANTS:
                metrics = variant_data.get(variant, {})
                gw_info_dict[f"lnL_{variant}_{label}"] = metrics.get("lnL", np.nan)
                gw_info_dict[f"delta_lnL_{variant}_{label}"] = metrics.get("delta_lnL", np.nan)
                gw_info_dict[f"sigma_{variant}_{label}"] = metrics.get("sigma", np.nan)
                within_val = metrics.get("within_3sigma", np.nan)
                if isinstance(within_val, (bool, np.bool_)):
                    gw_info_dict[f"within_3sigma_{variant}_{label}"] = bool(within_val)
                else:
                    gw_info_dict[f"within_3sigma_{variant}_{label}"] = np.nan
        # Add additional spectrum info
        for key, value in self._flatten_additional_spec_info().items():
            gw_info_dict[key] = value
        # Add DNeff
        gw_info_dict["DNeff_GW"] = self.DNeff_GW
        return gw_info_dict
