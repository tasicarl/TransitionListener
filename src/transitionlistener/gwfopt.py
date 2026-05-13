"""
Module to calculate the GW energy spectrum from the phase transition parameters.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from collections import OrderedDict

import numpy as np
from scipy.special import gamma
import transitionlistener.constants as c
from . import errors
from scipy.optimize import minimize_scalar

from . import console, print
import rich


class FOPTspectrum():
    """
    Class to compute the gravitational wave spectrum from a first order phase transition.
    """

    def __init__(self, gwparams_dict: dict, astroparams_dict: dict=
                 {"A": 0.0, "gamma": 0.0}, verbose: bool = False):
        """Validate inputs, cache parameters and pre-compute spectra grids."""
        self.gwparams_dict = gwparams_dict
        self.astroparams_dict = astroparams_dict
        if verbose:
            self.print_params()

        try:
            self.alpha = gwparams_dict["alpha"]
            self.RH = gwparams_dict["RH"]
            self.Treh_SM_GeV = gwparams_dict["Treh_SM_GeV"]
            self.g_eff_tot_reh = gwparams_dict["g_eff_tot_reh"]
            self.h_eff_tot_reh = gwparams_dict["h_eff_tot_reh"]
            self.kappa_phi = gwparams_dict["kappa_phi"]
            self.kappa_sw = gwparams_dict["kappa_sw"]
            self.kappa_turb = gwparams_dict["kappa_turb"]
            self.v_wall = gwparams_dict.get("v_wall", 1.0)  # Default to 1.0 if not provided
            self.c_s = gwparams_dict.get("c_s", 1/np.sqrt(3))  # Default to 1/sqrt(3) if not provided
            self.D = gwparams_dict.get("D", 1.0)  # Default to 1.0 if not provided
            self.alpha_hyd = gwparams_dict.get("alpha_hyd", self.alpha)  # Default to alpha if not provided
            self.g0 = gwparams_dict.get("g0", 2)  # Default to 2 if not provided
            self.h0 = gwparams_dict.get("h0", 3.9309363)  # Default to h0 from LISA update paper
            self.betaH = gwparams_dict.get(
                "betaH",
                (8 * np.pi / 0.28957) ** (1 / 3) / self.RH * max(self.v_wall, self.c_s),
            )  # Default to betaH_RH if not provided
        except Exception as err:
            raise errors.GWParamsError(f"Missing necessary GW parameters in gwparams_dict. Error: {err}")

        self.verbose = verbose
        self.sources = ["Wall collisions", "Sound waves", "Turbulence"]

        self.h2OmegaGW = self.h2Omega_0_sum
        self.h2OmegaGW_with_astrobg = self.h2Omega_PT_and_astro
        self.my_additional_spec_info = self.additional_spec_info(source="all")

        f_Hz = np.logspace(-10, 1, 1000)  # Frequency range in Hz for the spectrum
        self.gwspec_dict = {
            "h2OmegaGW_fun": self.h2OmegaGW,
            "h2OmegaGW_with_astrobg_fun": self.h2OmegaGW_with_astrobg,
            "f_Hz": f_Hz,
            "h2OmegaGW": self.h2OmegaGW(f_Hz),
            "h2OmegaGW_with_astrobg": self.h2OmegaGW_with_astrobg(f_Hz),
            "additional_spec_info": self.my_additional_spec_info,
        }
        

    def __iter__(self):
        """Allow iteration over the underlying spectrum dictionary."""
        return iter(self.gwspec_dict)

    def __getitem__(self, key: str):
        """Expose dictionary-style access to precomputed spectrum components."""
        return self.gwspec_dict[key]

    def __len__(self):
        """Return the number of entries stored in the spectrum dictionary."""
        return len(self.gwspec_dict)

    def keys(self):
        """Return the keys of the spectrum dictionary."""
        return self.gwspec_dict.keys()

    def print_params(self):
        """Render a table with the gravitational wave and astrophysical parameters."""
        table = rich.table.Table(title="FULL GW PARAMETERS", title_justify="left", box=rich.box.ROUNDED)
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="orange1")

        def _format_numeric_value(key: str, value) -> str:
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
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return "False"
            if isinstance(value, (int, np.integer, bool, np.bool_)):
                return str(bool(value))
            return str(bool(value))

        regular_items = []
        warning_items = []
        for key, value in self.gwparams_dict.items():
            if isinstance(key, str) and key.upper().startswith("WARNING"):
                warning_items.append((key, value))
            else:
                regular_items.append((key, value))

        for key, value in sorted(regular_items, key=lambda kv: (kv[0].lower(), kv[0])):
            table.add_row(key, _format_numeric_value(key, value))

        if warning_items:
            table.add_section()
            table.add_row("Warnings", "")
            sorted_warnings = sorted(warning_items, key=lambda kv: (kv[0].lower(), kv[0]))
            warning_labels = [
                key.split("WARNING:", 1)[1] if ":" in key else key
                for key, _ in sorted_warnings
            ]
            max_warning_len = max(len(label) for label in warning_labels)
            for label, (_, value) in zip(warning_labels, sorted_warnings):
                table.add_row(f"    {label:<{max_warning_len}}", _format_warning_value(value))

        console.print("\n", table)

    def BPL(self, f : float, fb: float, Omegab: float, n1: float, n2: float, a1: float) -> float:
        """
        A broken power law function, see 2403.03723 eq. (2.4)
        
        Parameters
        ----------
        f : float
            Frequency in Hz
        fb : float
            Break frequency in Hz
        Omegab : float
            Amplitude at fb
        n1 : float
            Exponent before the break
        n2 : float
            Exponent after the break
        a1 : float
            Transition parameter at the break
        
        Returns
        -------
        float
            Broken power law function value at frequency ``f``.
        """
        return Omegab * (f / fb)**n1 * (1/2 + 1/2 * (f / fb)**a1)**((n2 - n1) / a1)

    def DBPL(self, f : float, f1: float, f2: float, Omega2: float,
             n1: float, n2: float, n3: float, a1: float, a2: float) -> float:
        """
        A double broken power law function, see 2403.03723 eq. (2.8)
        
        Parameters
        ----------
        f : float
            Frequency in Hz
        f1 : float
            First break frequency in Hz
        f2 : float
            Second break frequency in Hz
        Omega2 : float
            Amplitude at f2
        n1 : float
            Exponent before the first break
        n2 : float
            Exponent between the first and second break
        n3 : float
            Exponent after the second break
        a1 : float
            Transition parameter at the first break
        a2 : float
            Transition parameter at the second break

        Returns
        -------
        float
            Double broken power law function value at frequency ``f``.
        """
        S = (f / f1)**n1 * (1 + (f / f1)**a1)**((n2 - n1) / a1) * (1 + (f / f2)**a2)**((n3 - n2) / a2)
        S_at_f2 = (f2 / f1)**n1 * (1 + (f2 / f1)**a1)**((n2 - n1) / a1) * 2**((n3 - n2) / a2)
        S2 = S / S_at_f2 # both are normalized to the same arbitrary constant which drops out here
        return Omega2 * S2

    def additional_spec_info(
        self,
        source: str = "all",
        f_pivot_Hz: float = 1e-8,
    ) -> OrderedDict[str, dict[str, float | str]]:
        """Calculate summary information about the GW spectrum.

        Parameters
        ----------
        source : str
            Source to consider for the GW spectrum. Options are ``"all"``,
            ``"Wall collisions"``, ``"Sound waves"``, ``"Turbulence"``.
        f_pivot_Hz : float
            Pivot frequency in Hz to evaluate the spectral amplitude.

        Returns
        -------
        OrderedDict[str, dict[str, float | str]]
            Ordered dictionary containing linear, logarithmic and LaTeX labelled
            representations of each quantity describing the spectrum.
        """

        # Takes log10 frequency in Hertz as input and returns minus the total h2Omega_GW amplitude
        minush2O = lambda log10f: -self.h2Omega_0_sum(10**log10f, source)

        # Use log10 of the frequency for better numerical stability
        res = minimize_scalar(minush2O, method="brent", options={'maxiter': 1000, 'xtol': 1e-6})
        peak_freq_Hz = 10**res.x
        peak_amplitude = -res.fun

        # Compute the spectral amplitude at pivot frequency
        h2Omega_at_pivot = self.h2Omega_0_sum(f_pivot_Hz, source)
        
        def _safe_log10(value: float) -> float:
            try:
                if value > 0:
                    return float(np.log10(value))
            except Exception:
                pass
            return float("nan")

        def _make_entry(value: float, latex: str, label: str) -> dict[str, float | str]:
            lin_value = float(value)
            return {
                "lin": lin_value,
                "log10": _safe_log10(lin_value),
                "latex": latex,
                "label": label,
            }

        add_info: OrderedDict[str, dict[str, float | str]] = OrderedDict()
        add_info["f_peak_Hz"] = _make_entry(
            peak_freq_Hz,
            r"$f_\mathrm{peak} / \mathrm{Hz}$",
            "f_peak_Hz",
        )
        add_info["h2OmegaGW_peak"] = _make_entry(
            peak_amplitude,
            r"$h^2 \Omega_\mathrm{GW}(f_\mathrm{peak})$",
            "h2OmegaGW_peak",
        )
        add_info["f_pivot_Hz"] = _make_entry(
            f_pivot_Hz,
            r"$f_\mathrm{pivot} / \mathrm{Hz}$",
            "f_pivot_Hz",
        )
        add_info["h2OmegaGW_at_pivot"] = _make_entry(
            h2Omega_at_pivot,
            r"$h^2 \Omega_\mathrm{GW}(f_\mathrm{pivot})$",
            "h2OmegaGW_at_pivot",
        )
        return add_info

    def h2Omega_PT_and_astro(self, f_Hz : float) -> float:
        """Calculate the total GW energy density spectrum from a first order
        phase transition and an astrophysical background.

        Parameters
        ----------
        f_Hz : float
            Frequency in Hz

        Returns
        -------
        float
            Total GW energy density spectrum value.
        """
        return self.h2Omega_0_sum(f_Hz) + self.h2Omega_astro_noise(f_Hz)
    
    def h2Omega_astro_noise(self, f_Hz : float) -> float:
        """
        Calculate the astrophysical background noise spectrum in the power-law approximation, e.g. for
        supermassive black hole mergers in the nHz frequency range. Note that the below parameterization
        is specific to the PTA band and should not be used for other frequencies.
        
        Parameters
        ----------
        f_Hz : float
            Frequency in Hz

        Returns
        -------
        float
            Astrophysical noise spectrum value.
        """
        A = self.astroparams_dict.get("A", 1e-15)  # Default to 1e-15 if not provided
        gamma = self.astroparams_dict.get("gamma", 13/3)  # Default to 13/3 if not provided
        return 2 * np.pi**2 / (3 * c.H100_Hz**2) * A**2 * (f_Hz/c.fyr_Hz)**(5-gamma) * c.fyr_Hz**2

    def h2Omega_0_sum(self, f_Hz : float, source : str ="all") -> float:
        """State-of-the-art model for the GW spectrum from a FOPT.

        Parameters
        ----------
        f_Hz : float
            Frequency in Hz
        source : str
            Source to consider for the GW spectrum. Options are "all",
            "Wall collisions", "Sound waves", "Turbulence".

        Returns
        -------
        float
            GW energy density spectrum value at frequency ``f_Hz``.
        """

        self.sources = ["Wall collisions", "Sound waves", "Turbulence"]

        # Amplitude prefactor due to redshift
        Rh2 = c.Omega_gamma_h2 / self.D**(4./3.) * (self.h0 / self.h_eff_tot_reh)**(4./3.) \
            * self.g_eff_tot_reh / self.g0

        # Hubble parameter at the time of the phase transition redshifted to today in Hertz
        H_p0_Hz = 1.65e-5 * self.D**(-4/3) * (self.Treh_SM_GeV / 100)
        H_p0_Hz *= (self.g_eff_tot_reh / 100)**(1/2)
        H_p0_Hz *= (100 / self.h_eff_tot_reh)**(1/3)

        # Kinetic energy fractions.
        # Note: The factor 0.6 accounts for the efficiency in producing kinetic energy
        # in the bulk fluid motion with respect to the single bubble case, as found in Ref.
        # 2209.04369 (Higgsless simulations) by Jinno, Konstandin, Rubira, Stomberg
        _k = self.alpha / (1 + self.alpha)
        Kphi = self.kappa_phi * _k
        Ksw = 0.6 * self.kappa_sw * _k
        Kturb = 0.6 * self.kappa_turb * _k

        # Peak frequency of the different GW sources today. _f0 is the redshifted Horizon scale at the time of the PT.
        fp = 0.28957
        self.betaH = (8 * np.pi/fp)**(1/3) * max(self.v_wall, self.c_s) / self.RH

        # Scalar field source
        fb_phi_Hz = H_p0_Hz * 0.11 * self.betaH
        A_phi = 0.05 
        n1_phi = 2.4
        n2_phi = -2.4
        a1_phi = 1.2
        h2Omega_phi_peak = Rh2 * A_phi * Kphi**2 / self.betaH**2 # eq. (2.7) of 2403.03723
        h2Omega_phi = self.BPL(f_Hz, fb_phi_Hz, h2Omega_phi_peak, n1_phi, n2_phi, a1_phi)

        # Sound wave source
        A_sw = 0.11
        n1_sw = 3.0
        n2_sw = 1.0
        n3_sw = -3.0
        a1_sw = 2
        a2_sw = 4
        # This shell-thickness estimate assumes detonations or deflagrations.
        # A full hydrodynamic profile would be needed for more general solutions.
        xi_shell = np.abs(self.v_wall - self.c_s) # dimensionless sound shell thickness

        Delta_w = xi_shell / max(self.v_wall, self.c_s) # relative frequency separation between f1 and f2
        f1_sw_Hz = H_p0_Hz * 0.2 / self.RH
        f2_sw_Hz = H_p0_Hz * 0.5 / self.RH / Delta_w

        # Compute the lifetime only from the relevant alpha
        Htau_shell = self.RH / np.sqrt(0.6 * self.kappa_sw * self.alpha_hyd/(1 + self.alpha_hyd) / (4/3))

        Htau_sw = min(1, Htau_shell)
        h2Omega_int_sw = Rh2 * A_sw * Ksw**2 * Htau_sw * self.RH # eq. (2.10) of 2403.03723
        h2Omega_sw_peak_at_f2 = 1 / np.pi * (np.sqrt(2) + 2 * (f2_sw_Hz / f1_sw_Hz) / (1 + f2_sw_Hz**2 / f1_sw_Hz**2)) \
            * h2Omega_int_sw
        h2Omega_sw = self.DBPL(f_Hz, f1_sw_Hz, f2_sw_Hz, h2Omega_sw_peak_at_f2, n1_sw, n2_sw, n3_sw, a1_sw, a2_sw)

        # Turbulence source as in eq. (2.15) of 2403.03723, not the approximation in eq. (2.25)
        if Kturb != 0:
            Omega_s = Kturb
            v_Alfven = np.sqrt(Omega_s / (4/3))
            N = 2  # number of eddy turnovers
            N_Hp_tau_e = N * self.RH / (2 * np.pi * v_Alfven)
            # Duration of the turbulence source in units of Hubble time
            A_turb = 0.085
            a2_turb = 2.15
            f2_turb_Hz = 2.2 * H_p0_Hz / self.RH
            p_aniso = (1 + (f_Hz / f2_turb_Hz)**a2_turb)**(-11/(3 * a2_turb))
            T_turb = np.where(
                f_Hz <= H_p0_Hz / (2 * np.pi) / N_Hp_tau_e,
                np.log(1 + N_Hp_tau_e)**2,
                np.log(1 + H_p0_Hz / (2 * np.pi * f_Hz))**2
            )
            h2Omega_turb = 3 * A_turb * Omega_s**2 * Rh2 * self.RH**3 * (f_Hz / H_p0_Hz)**3 * p_aniso * T_turb
        else:
            h2Omega_turb = f_Hz * 0.0

        # Total GW spectrum
        if source == "all":
            res = h2Omega_phi + h2Omega_sw + h2Omega_turb
        elif source == "Wall collisions":
            res = h2Omega_phi
        elif source == "Sound waves":
            res = h2Omega_sw
        elif source == "Turbulence":
            res = h2Omega_turb
        else:
            print("ERROR: Unknown source")
            exit()
        return res

    def h2Omega_0_sum_old(self, f_Hz : float, source : str ="all") -> float:
        '''
        Alternative SW model as used in sub-gev paper and presented in LISA update paper. The remaining
        sources are described as in 1903.09642.

        Parameters
        ----------
        f_Hz : float
            Frequency in Hz
        source : str
            Source to consider for the GW spectrum. Options are "all",
            "Wall collisions", "Sound waves", "Turbulence".

        Returns
        -------
        float
            GW energy density spectrum value at frequency f_Hz.
        '''

        self.sources = ["Bubble collisions", "Sound waves", "Turbulence"]
        self.h0 = 3.9309363
        self.kappa_sw = self.alphaDS / (0.73 + 0.083 * np.sqrt(self.alphaDS) + self.alphaDS)

        # Kinetic energy fractions. KswDS is used for the computation of the sound wave source lifetime
        _k = self.alpha / (1 + self.alpha)
        Kphi = self.kappa_phi * _k
        Ksw = self.kappa_sw * _k
        Kturb = self.kappa_turb * _k
        KswDS = self.kappa_sw * self.alphaDS / (1 + self.alphaDS) if self.alphaDS is not np.nan else Ksw

        # Peak frequency of the different GW sources today. _f0 is the redshifted Horizon scale at the time of the PT.
        _f0 = 1.65e-5 * self.D**(-1/3) * (self.Treh_SM_GeV / 100)
        _f0 *= (self.g_eff_tot_reh / 100)**(1/2)
        _f0 *= (100 / self.h_eff_tot_reh)**(1/3)
        f0phi_Hz = _f0 * 0.51 / self.RH
        f0sw_Hz = _f0 * 3.4 / self.RH / (self.v_wall - self.c_s)
        f0turb_Hz = _f0 * 3.9 / self.RH / (self.v_wall - self.c_s)

        f0sw_Hz = 8.87619e-6 * self.D**(-1/3) * (8*np.pi)**(1/3.) * self.v_wall / self.RH 
        f0sw_Hz *= (self.Treh_SM_GeV / 100)
        f0sw_Hz *= (self.g_eff_tot_reh/100)**(1/2) * (100/self.h_eff_tot_reh)**(1/3)

        # Amplutude prefactor due to redshift
        Rh2 = (
            c.Omega_gamma_h2
            / self.D**(4.0 / 3.0)
            * (self.h0 / self.h_eff_tot_reh) ** (4.0 / 3.0)
            * self.g_eff_tot_reh
            / self.g0
        )

        # Spectral functions
        def s_phi(x : float) -> float:
            # Based on 1802.05712: Cutting et al., Gravitational waves from vacuum
            # first-order phase transitions: from the envelope to the lattice.
            return 0.024 * x**3 * (1 + 2 * x**2.07)**(-2.18)

        def s_sw(x : float, p : float=3, q : float=4, n : float=2) -> float:
            # Normalization of the (p,q,n) paramterized spectrum
            # Without specifying the normalization, the spectrum is normalized to 1 at x=1
            # In that case: norm = 1/0.687.
            norm = (q/p)**(p/n) * ((p+q)/q)**((p+q)/n)
            norm *= gamma(p/n) * gamma(q/n) / n / gamma((p+q)/n)

            S_tilde = x**p / (q/(p+q) + p / (p+q) * x**n)**((p+q)/n)
            return S_tilde / norm

        def s_turb(x : float) -> float:
            # Based on 0909.0622: Caprini, Durrer and Servant, the stochastic GW
            # background from turbulence and magnetic fields generated by a FOPT.
            turb_correction = 3.9 / (self.v_wall - self.c_s) / self.RH
            # Ratio of f_turb at emission with respect to H at emission
            return 6.8 * x**3 * (1 + x)**(-11/3) / (1 + 8 * np.pi * x * turb_correction)

        # Lifetime of the sound wave source
        Htau_sw = min(1, self.RH / np.sqrt(KswDS / (4/3)))


        # Peak amplitudes
        Omega_phi_peak = Kphi**2 * self.RH**2
        # Keep the normalization used in the current implementation, which
        # differs from the legacy expression by a factor of sqrt(3).
        Omega_sw_peak = Ksw**2 * self.RH * Htau_sw * 0.012 * 3 * np.sqrt(3)
        Omega_turb_peak = Kturb**(3/2) * self.RH * (1 - Htau_sw)

        # Total GW spectrum
        if source == "all":
            res = Rh2 * (Omega_phi_peak * s_phi(f_Hz / f0phi_Hz) 
                        + Omega_sw_peak * s_sw(f_Hz / f0sw_Hz)
                        + Omega_turb_peak * s_turb(f_Hz / f0turb_Hz))
        elif source == "Bubble collisions":
            res = Rh2 * Omega_phi_peak * s_phi(f_Hz / f0phi_Hz)
        elif source == "Sound waves":
            res = Rh2 * Omega_sw_peak * s_sw(f_Hz / f0sw_Hz)
        elif source == "Turbulence":
            res = Rh2 * Omega_turb_peak * s_turb(f_Hz / f0turb_Hz)
        else:
            print("ERROR: Unknown source")
            exit()
        return res

    
    
    """ 
    def h2Omega_0_sum_older(self, f_Hz, source="all"):
        '''
        Calculate the GW energy density spectrum from a FOPT.
        Follows 1903.09642, with adaptation for DS phase transition: dilution D
        due to DS entropy production is included; alpha_DS is the strength of the
        DS PT normalized on the DS radiation bath.
        Assume v_wall = 1 and instantaneous reheating after the PT (at Treh_SM_GeV).
        '''

        # Kinetic energy fractions. KswDS is used for the computation of the sound wave source lifetime
        _k = self.alpha / (1 + self.alpha)
        Kphi = self.kappa_phi * _k
        Ksw = self.kappa_sw * _k
        Kturb = self.kappa_turb * _k
        KswDS = self.kappa_sw * self.alphaDS / (1 + self.alphaDS) if self.alphaDS is not np.nan else Ksw

        # Peak frequency of the different GW sources today. _f0 is the redshifted Horizon scale at the time of the PT.
        _f0 = 1.65e-5 * self.D**(-1/3) * (self.Treh_SM_GeV / 100)
        _f0 *= (self.g_eff_tot_reh / 100)**(1/2)
        _f0 *= (100 / self.h_eff_tot_reh)**(1/3)
        f0phi_Hz = _f0 * 0.51 / self.RH
        f0sw_Hz = _f0 * 3.4 / self.RH / (self.v_wall - self.c_s)
        f0turb_Hz = _f0 * 3.9 / self.RH / (self.v_wall - self.c_s)

        # Amplutude prefactor due to redshift
        Rh2 = (
            Omega_gamma_h2
            / self.D**(4.0 / 3.0)
            * (self.g0 / self.g_eff_tot_reh) ** (4.0 / 3.0)
            * self.h_eff_tot_reh
            / self.h0
        )

        # Spectral functions
        def s_phi(x):
            # Based on 1802.05712: Cutting et al., gravitational waves from
            # vacuum first-order phase transitions: from the envelope to the lattice.
            return 0.024 * x**3 * (1 + 2 * x**2.07)**(-2.18)

        def s_sw(x):
            # Based on 1512.06239: Caprini et al., Science with the space-based
            # interferometer eLISA. II: Gravitational waves from cosmological
            # phase transitions.
            return 0.38 * x**3 * (1 + 0.75 * x**2)**(-3.5)

        def s_turb(x):
            # Based on 0909.0622: Caprini, Durrer and Servant, the stochastic
            # gravitational wave background from turbulence and magnetic fields
            # generated by a first-order phase transition.
            turb_correction = 3.9 / (self.v_wall - self.c_s) / self.RH
            # Ratio of f_turb at emission with respect to H at emission
            return 6.8 * x**3 * (1 + x)**(-11/3) / (1 + 8 * np.pi * x * turb_correction)

        # Lifetime of the sound wave source
        Htau_sw = min(1, self.RH / np.sqrt(KswDS / (4/3)))

        # Peak amplitudes
        Omega_phi_peak = Kphi**2 * self.RH**2
        Omega_sw_peak = Ksw**2 * self.RH * Htau_sw
        Omega_turb_peak = Kturb**(3/2) * self.RH * (1 - Htau_sw)

        # Total GW spectrum
        if source == "all":
            res = Rh2 * (Omega_phi_peak * s_phi(f_Hz / f0phi_Hz) 
                        + Omega_sw_peak * s_sw(f_Hz / f0sw_Hz)
                        + Omega_turb_peak * s_turb(f_Hz / f0turb_Hz))
        elif source == "Bubble collisions":
            res = Rh2 * Omega_phi_peak * s_phi(f_Hz / f0phi_Hz)
        elif source == "Sound waves":
            res = Rh2 * Omega_sw_peak * s_sw(f_Hz / f0sw_Hz)
        elif source == "Turbulence":
            res = Rh2 * Omega_turb_peak * s_turb(f_Hz / f0turb_Hz)
        else:
            print("ERROR: Unknown source")
            exit()
        return res
        
    """
