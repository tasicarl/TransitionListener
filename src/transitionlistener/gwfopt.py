"""
Module to calculate the GW energy spectrum from the phase transition parameters.

What this module computes is described in the documentation under
The gravitational wave spectrum: :doc:`/algorithms/the_spectrum`.

Part of TransitionListener v2
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
from . import config as tlconfig
from scipy.optimize import minimize_scalar

from . import console, print
import rich


class FOPTspectrum():
    """
    Class to compute the gravitational wave spectrum from a first order phase transition.
    """

    SOURCES = {
        "LISA-pipeline24": ("Wall collisions", "Sound waves", "Turbulence"),
        "LISA-update2019": ("Bubble collisions", "Sound waves", "Turbulence"),
        "LewickiVaskonen22": ("Field shells", "Fluid shells"),
    }

    # arXiv:2208.11697 table 1: fits to flat-space simulations of strongly
    # supercooled transitions, Omega_GW = (H/beta)^2 [kappa alpha/(1+alpha)]^2 A S(f/f_p)
    # with the broken power law S of eq. (4.1) and the peak position 2 pi f_p / beta.
    # The "bubbles" columns are scalar-field shells, the "fluid" columns fluid shells.
    LV22_SHAPES_FLAT = {
        "bubbles_envelope": dict(A=0.0378, a=3.08, b=0.98, c=1.91, fp=1.33),
        "bubbles_R2":       dict(A=0.0593, a=1.03, b=1.84, c=1.45, fp=0.64),
        "bubbles_R3":       dict(A=0.0513, a=2.41, b=2.42, c=4.08, fp=0.77),
        "fluid_R2_v1":      dict(A=0.0594, a=1.03, b=1.87, c=1.39, fp=0.57),
        "fluid_R2_vcs":     dict(A=0.0336, a=1.00, b=1.39, c=0.71, fp=0.44),
        "fluid_R3_v1":      dict(A=0.0514, a=2.36, b=2.36, c=3.69, fp=0.66),
        "fluid_R3_vcs":     dict(A=0.0364, a=2.02, b=1.38, c=1.48, fp=0.44),
    }
    # arXiv:2511.15687 (Lewicki and Vaskonen 2025): scalar-field shells simulated with the
    # cosmic expansion during the transition, three models of the shells after collision.
    # Omega_GW = (H/beta)^2 [kappa alpha/(1+alpha)]^2 A(beta) S(f/f_p) with the broken power
    # law S of 2208.11697 eq. (4.1), the causality tail below the horizon frequency of
    # 2511.15687 eq. (21), and
    #   A(beta)              = A [1 + cA (1 - exp(sqrt(H/beta)))]          (eq. (23) there)
    #   2 pi f_p(beta)/a_p H_p = fp beta/H [1 + cf (beta/H)^-1.2]           (eq. (22) there)
    # "dissipative":     the dissipative bulk-flow model (shell energy ~ R^-3 after collision);
    #                    a = b = c = 2 (text below their eq. (21)), A, fp, cA, cf of eqs. (22), (23).
    # "non_dissipative": the bulk-flow model (shell energy ~ R^-2); a = c = 1, b = 2; same
    #                    eqs. (22), (23), which the paper fits to both bulk-flow models jointly.
    # In flat space the two models are the "bubbles_R3" and "bubbles_R2" columns above
    # (2208.11697 sec. 4: T_rr ~ R^-2 is the bulk-flow model; 2511.15687, text below eq. (11)).
    LV22_SHAPES_EXPANDING = {
        "dissipative":     dict(a=2.0, b=2.0, c=2.0, A=0.06, fp=0.7, cA=0.8, cf=1.8),
        "non_dissipative": dict(a=1.0, b=2.0, c=1.0, A=0.06, fp=0.7, cA=0.8, cf=1.8),
    }

    def __init__(self, gwparams_dict: dict, astroparams_dict: dict=
                 {"A": 0.0, "gamma": 0.0}, verbose: bool = False, config=None):
        """Validate inputs, cache parameters and pre-compute spectra grids.

        ``config`` is the model's ``Configuration`` object (``pot.config``): its
        ``gwConf`` selects the source template and the modelling choices of the
        spectrum, its ``percolationConf.f_perc`` the true-vacuum fraction that
        defines the percolation scale. Without it the defaults of
        :class:`transitionlistener.config.Configuration` apply.
        """
        self.gwparams_dict = gwparams_dict
        self.astroparams_dict = astroparams_dict
        # the strength normalised to the dark-sector bath, used by the LISA-update2019
        # template for the sound-wave lifetime when a model supplies it
        self.alphaDS = gwparams_dict.get("alpha_DS", np.nan)
        self._read_modelling_choices(config)
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
            self.h0 = gwparams_dict.get("h0", c.h_eff_today)
            self.betaH = gwparams_dict.get(
                "betaH",
                (8 * np.pi / self.f_perc) ** (1 / 3) / self.RH * max(self.v_wall, self.c_s),
            )  # Default to betaH_RH if not provided
            # betaH_Gamma = H^-1 dlnGamma/dt at T_perc: the speed of the expansion factors when
            # gwConf.expansion_speed = "betaH_Gamma" (checked below); unused otherwise
            self.betaH_Gamma = gwparams_dict.get("betaH_Gamma", np.nan)
        except errors.Timeout:
            raise
        except Exception as err:
            raise errors.GWParamsError(f"Missing necessary GW parameters in gwparams_dict. Error: {err}")
        if not self.use_mean_bubble_separation:
            # beta/H from the slope of the action, (beta/H)_S3, in place of the bubble separation: the
            # separation follows from it by the relation R_* = (8 pi / f_perc)^{1/3} max(v_w, c_s) / beta,
            # so every factor of the spectrum, the expansion factors included, takes this speed
            betaH_S3 = gwparams_dict.get("betaH_S3", np.nan)
            try:
                ok = bool(np.isfinite(betaH_S3) and betaH_S3 > 0)
            except TypeError:
                ok = False
            if not ok:
                raise errors.GWParamsError(
                    "gwConf.use_mean_bubble_separation = False needs a finite, positive 'betaH_S3' in "
                    f"gwparams_dict, got {betaH_S3!r}."
                )
            self.RH = (8 * np.pi / self.f_perc) ** (1 / 3) * max(self.v_wall, self.c_s) / betaH_S3
            self.betaH = float(betaH_S3)
        if self.expansion_suppression and self.expansion_speed == "betaH_Gamma" and self.use_mean_bubble_separation:
            try:
                ok = bool(np.isfinite(self.betaH_Gamma) and self.betaH_Gamma > 0)
            except TypeError:
                ok = False
            if not ok:
                raise errors.GWParamsError(
                    "gwConf.expansion_speed = 'betaH_Gamma' needs a finite, positive 'betaH_Gamma' in "
                    f"gwparams_dict, got {self.betaH_Gamma!r}."
                )

        self.verbose = verbose
        self.sources = list(self.SOURCES[self.source_model])

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

        choices = rich.table.Table(title="GW MODELLING CHOICES", title_justify="left", box=rich.box.ROUNDED)
        choices.add_column("Choice", style="cyan", no_wrap=True)
        choices.add_column("Value", style="orange1")
        for key, value in self.modelling_choices().items():
            choices.add_row(key, str(value))
        console.print("\n", choices)

    def _read_modelling_choices(self, config) -> None:
        """Copy the spectrum's modelling choices from the configuration object.

        The choices are validated against the allowed values in
        :mod:`transitionlistener.config`; a value that is not in the list raises
        :class:`transitionlistener.errors.GWParamsError` rather than falling back
        to a default silently.
        """
        if config is None:
            config = tlconfig.Configuration()
        self.config = config
        gwc = config.gwConf

        def _choice(name, allowed):
            value = str(getattr(gwc, name, getattr(tlconfig.GWConf, name)))
            if value not in allowed:
                raise errors.GWParamsError(
                    f"gwConf.{name} must be one of {allowed}, got {value!r}."
                )
            return value

        self.f_perc = float(config.percolationConf.f_perc)
        if not 0.0 < self.f_perc < 1.0:
            raise errors.GWParamsError(
                f"percolationConf.f_perc must lie strictly between 0 and 1, got {self.f_perc}."
            )
        self.source_model = _choice("source_model", tlconfig.GW_SOURCE_MODELS)
        self.sw_lifetime_factor = _choice("sw_lifetime_factor", tlconfig.GW_SW_LIFETIME_FACTORS)
        self.lv22_shape_field = _choice("lv22_shape_field", tlconfig.GW_LV22_SHAPES_FIELD)
        self.lv22_shape_fluid = _choice("lv22_shape_fluid", tlconfig.GW_LV22_SHAPES_FLUID)
        self.lv22_expanding_shape = _choice("lv22_expanding_shape", tlconfig.GW_LV22_EXPANDING_SHAPES)
        self.causality_tail = _choice("causality_tail", tlconfig.GW_CAUSALITY_TAILS)
        self.kinetic_energy_efficiency = float(
            getattr(gwc, "kinetic_energy_efficiency", tlconfig.GWConf.kinetic_energy_efficiency)
        )
        if not 0.0 < self.kinetic_energy_efficiency <= 1.0:
            raise errors.GWParamsError(
                "gwConf.kinetic_energy_efficiency must lie in (0, 1], "
                f"got {self.kinetic_energy_efficiency}."
            )
        self.expansion_suppression = bool(
            getattr(gwc, "expansion_suppression", tlconfig.GWConf.expansion_suppression)
        )
        self.expansion_speed = _choice("expansion_speed", tlconfig.GW_EXPANSION_SPEEDS)
        self.use_mean_bubble_separation = bool(
            getattr(gwc, "use_mean_bubble_separation", tlconfig.GWConf.use_mean_bubble_separation)
        )

    def modelling_choices(self) -> OrderedDict:
        """The modelling choices this spectrum was built with, for provenance."""
        out = OrderedDict()
        out["f_perc"] = self.f_perc
        out["use_mean_bubble_separation"] = self.use_mean_bubble_separation
        out["source_model"] = self.source_model
        out["expansion_suppression"] = self.expansion_suppression
        if self.expansion_suppression or self.source_model == "LewickiVaskonen22":
            out["causality_tail"] = self.causality_tail
        if self.expansion_suppression:
            out["expansion_speed"] = self.expansion_speed
        if self.source_model == "LewickiVaskonen22":
            if self.expansion_suppression:
                out["lv22_expanding_shape"] = self.lv22_expanding_shape
            else:
                out["lv22_shape_field"] = self.lv22_shape_field
                out["lv22_shape_fluid"] = self.lv22_shape_fluid
        else:
            if self.source_model == "LISA-pipeline24":
                out["kinetic_energy_efficiency"] = self.kinetic_energy_efficiency
            out["sw_lifetime_factor"] = self.sw_lifetime_factor
        return out

    def _sw_lifetime_factor(self, Htau_shell: float) -> float:
        """Upsilon_sw, the effective duration of the sound-wave source in Hubble times.

        ``Htau_shell`` is H tau_sh = R_* H / U_f, the shell lifetime in units of
        the Hubble time. "min" cuts the linear growth of the signal at one
        Hubble time (arXiv:2403.03723); "guo" is the result of integrating the
        source in a radiation-dominated expanding background,
        1 - (1 + 2 H tau_sh)^(-1/2) (arXiv:2007.08537); "guo_capped" first caps
        tau_sh at one Hubble time and then applies Guo's form, as BSMPT does.
        The three agree for H tau_sh << 1 and differ by an order-one factor for
        shells that last about a Hubble time.
        """
        if self.sw_lifetime_factor == "min":
            return min(1.0, Htau_shell)
        if self.sw_lifetime_factor == "guo":
            return 1.0 - 1.0 / np.sqrt(1.0 + 2.0 * Htau_shell)
        return 1.0 - 1.0 / np.sqrt(1.0 + 2.0 * min(1.0, Htau_shell))

    # ---- the cosmic expansion during the transition, arXiv:2511.15687
    def _expansion_betaH(self) -> float:
        """The speed passed to the expansion factors of arXiv:2511.15687.

        ``betaH`` (the speed from the bubble separation) by default, ``betaH_Gamma`` with
        ``gwConf.expansion_speed = "betaH_Gamma"``. Every other factor of the spectrum keeps
        ``betaH``.
        """
        if self.expansion_speed == "betaH_Gamma" and self.use_mean_bubble_separation:
            return self.betaH_Gamma
        return self.betaH

    def _expansion_factors(self, H_p0_Hz: float):
        """Amplitude ratio A(beta)/A(infinity) of eq. (23), peak-frequency ratio
        f_p(beta)/f_p(infinity) of eq. (22) and the horizon frequency of eq. (21).

        The fits were made for an exponential nucleation rate Gamma ~ exp(beta t) and
        are functions of its beta/H_V, with H_V the Hubble rate of the free-energy
        difference of the two phases alone. ``gwConf.expansion_speed`` selects the speed
        passed to them: "betaH_RH", the speed from the bubble separation that every
        other factor of the spectrum uses, or "betaH_Gamma", the growth rate
        H^-1 dlnGamma/dt of this transition at T_perc, with H the Hubble rate of the
        percolation history; it equals beta/H_V up to H(T_perc)/H_V, which is close to
        one in a vacuum-dominated transition. The fits were made for beta/H_V of
        order 4 and above; below 1.52 the amplitude ratio turns negative and is set
        to zero."""
        bH = self._expansion_betaH()
        r_A = max(0.0, 1.0 + 0.8 * (1.0 - np.exp(np.sqrt(1.0 / bH))))
        r_f = 1.0 + 1.8 * bH ** (-1.2)
        return r_A, r_f, self._horizon_frequency(H_p0_Hz, bH)

    @staticmethod
    def _horizon_frequency(H_p0_Hz: float, bH: float) -> float:
        """The horizon frequency today, eq. (21) of arXiv:2511.15687:
        f_H = [1 - 0.6 (beta/H)^-0.67] a_p H_p / (2 pi)."""
        return H_p0_Hz / (2 * np.pi) * max(1e-3, 1.0 - 0.6 * bH ** (-0.67))

    def _causal_tail(self, flat, out, f_Hz, f_H: float):
        """Continue a spectrum as f^3 below the horizon frequency f_H.

        "sharp": f^3 below f_H, the spectrum above (as fitted in
        arXiv:2511.15687). "smooth": the spectrum is multiplied by
        (1 + (f_H/f)^3)^(-(3-a)/3) with a its own logarithmic slope at f_H, which
        is one far above f_H and steepens the slope to 3 far below it without a
        kink. "off": no tail.
        """
        if self.causality_tail == "off":
            return out
        f = np.asarray(f_Hz, dtype=float)
        if self.causality_tail == "sharp":
            edge = flat(np.array([f_H]))[0]
            return np.where(f < f_H, edge * (f / f_H) ** 3, out)
        eps = 1e-3
        pair = flat(np.array([f_H * (1 - eps), f_H * (1 + eps)]))
        with np.errstate(divide="ignore", invalid="ignore"):
            a = float(np.log(pair[1] / pair[0]) / np.log((1 + eps) / (1 - eps)))
        if not np.isfinite(a):
            a = 3.0
        return out * (1.0 + (f_H / f) ** 3) ** (-max(0.0, 3.0 - a) / 3.0)

    @staticmethod
    def lv22_shape_function(x, a: float, b: float, c: float):
        """S(f)/A of eq. (4.1) of arXiv:2208.11697, equal to one at the peak x = f/f_p = 1."""
        return (a + b) ** c / (b * x ** (-a / c) + a * x ** (b / c)) ** c

    def _h2Omega_lv22(self, f_Hz, source: str, Rh2: float, H_p0_Hz: float):
        """The strongly supercooled template of Lewicki and Vaskonen.

        Two broken power laws, one for the scalar-field shells with the energy
        fraction kappa_phi alpha/(1+alpha) and one for the fluid shells with
        kappa_sw alpha/(1+alpha), replace the three sources of the LISA
        templates. In flat space (``expansion_suppression = False``) each is the
        column ``lv22_shape_field`` or ``lv22_shape_fluid`` of table 1 of
        arXiv:2208.11697, continued as f^3 below the horizon frequency according to
        ``causality_tail``. With the cosmic expansion during the transition the
        amplitude, the peak frequency and the tail frequency follow eqs. (23),
        (22) and (21) of arXiv:2511.15687 and the shape is
        ``lv22_expanding_shape`` for both. Both fits are written in terms of
        beta/H, for which the value inferred from the bubble separation R_* H is
        used, except in the two factors of eqs. (22) and (23) that describe the
        expansion during an exponentially growing nucleation rate: these take the
        speed selected by ``gwConf.expansion_speed`` (:meth:`_expansion_betaH`).
        """
        f = np.asarray(f_Hz, dtype=float)
        _k = self.alpha / (1 + self.alpha)
        parts = {"Field shells": (self.kappa_phi * _k, self.lv22_shape_field),
                 "Fluid shells": (self.kappa_sw * _k, self.lv22_shape_fluid)}
        if source != "all" and source not in parts:
            return 0.0 * f
        f_unit = H_p0_Hz / (2 * np.pi)      # a_p H_p / (2 pi) redshifted to today
        bH = self.betaH
        out = 0.0 * f
        for name, (K, shape) in parts.items():
            if source != "all" and source != name:
                continue
            if self.expansion_suppression:
                e = self.LV22_SHAPES_EXPANDING[self.lv22_expanding_shape]
                bX = self._expansion_betaH()
                A = max(0.0, e["A"] * (1 + e["cA"] * (1 - np.exp(np.sqrt(1 / bX)))))  # eq. (23)
                f_p = f_unit * e["fp"] * bH * (1 + e["cf"] * bX ** (-1.2))              # eq. (22)
                a, b, cc = e["a"], e["b"], e["c"]
            else:
                s = self.LV22_SHAPES_FLAT[shape]
                A = s["A"]
                f_p = f_unit * s["fp"] * bH      # table 1 lists 2 pi f_p / beta
                a, b, cc = s["a"], s["b"], s["c"]
            out = out + Rh2 * A * K**2 / bH**2 * self.lv22_shape_function(f / f_p, a, b, cc)
        return out

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
        """The GW spectrum today, h^2 Omega_GW(f), with the template and the
        modelling choices of the configuration (see GWConf).

        Parameters
        ----------
        f_Hz : float
            Frequency in Hz
        source : str
            Source to consider. "all", or one of ``self.sources`` (which
            depend on the template).

        Returns
        -------
        float
            GW energy density spectrum value at frequency ``f_Hz``.
        """
        self.sources = list(self.SOURCES[self.source_model])

        # Amplitude prefactor due to redshift
        Rh2 = c.Omega_gamma_h2 / self.D**(4./3.) * (self.h0 / self.h_eff_tot_reh)**(4./3.) \
            * self.g_eff_tot_reh / self.g0

        # Hubble parameter at the time of the phase transition redshifted to today in Hertz
        H_p0_Hz = 1.65e-5 * self.D**(-1/3) * (self.Treh_SM_GeV / 100)
        H_p0_Hz *= (self.g_eff_tot_reh / 100)**(1/2)
        H_p0_Hz *= (100 / self.h_eff_tot_reh)**(1/3)

        # The scale of the signal: beta/H inferred from the bubble separation R_* H by
        # inverting R_* = (8 pi / f_perc)^{1/3} max(v_w, c_s) / beta, with the true-vacuum
        # fraction f_perc at percolation taken from percolationConf.
        self.betaH = (8 * np.pi / self.f_perc)**(1/3) * max(self.v_wall, self.c_s) / self.RH

        templates = {"LISA-pipeline24": self._h2Omega_lisa_pipeline24,
                     "LISA-update2019": self._h2Omega_lisa_update2019,
                     "LewickiVaskonen22": self._h2Omega_lv22}
        template = templates[self.source_model]
        if not self.expansion_suppression:
            out = template(f_Hz, source, Rh2, H_p0_Hz)
            if self.source_model != "LewickiVaskonen22":
                return out
            # the flat-space fits of 2208.11697 rise as f^a with a = 1 to 3.1 down to f -> 0; causality
            # requires f^3 below the horizon frequency f_H = a_p H_p / 2 pi, eq. (20) of 2511.15687. Its
            # beta-dependent fit, eq. (21), corrects the instantaneous-reheating estimate of a_p H_p for the
            # expansion during the transition, like eqs. (22) and (23), and is applied only with
            # expansion_suppression (gwConf.causality_tail = "off" keeps the fit as published)
            flat = lambda ff: template(ff, source, Rh2, H_p0_Hz)
            return self._causal_tail(flat, out, f_Hz, H_p0_Hz / (2 * np.pi))
        if self.source_model == "LewickiVaskonen22":
            # the fit with expansion replaces the flat-space one; only the tail is added
            r_A, r_f, f_H = self._expansion_factors(H_p0_Hz)
            flat = lambda ff: template(ff, source, Rh2, H_p0_Hz)
            return self._causal_tail(flat, flat(f_Hz), f_Hz, f_H)
        # the LISA templates: the amplitude and the frequencies of the flat-space
        # template are rescaled by the ratios of eqs. (23) and (22) of 2511.15687,
        # and the spectrum is continued as f^3 below the horizon frequency
        r_A, r_f, f_H = self._expansion_factors(H_p0_Hz)
        flat = lambda ff: r_A * template(np.asarray(ff, dtype=float) / r_f, source, Rh2, H_p0_Hz)
        return self._causal_tail(flat, flat(f_Hz), f_Hz, f_H)

    def _h2Omega_lisa_pipeline24(self, f_Hz, source: str, Rh2: float, H_p0_Hz: float):
        """The three-source template of the LISA pipeline paper 2403.03723 in
        flat space: wall collisions (broken power law), sound waves (double
        broken power law with the amplitude and the kinetic-energy factor of the
        Higgsless simulations 2209.04369, and the lifetime factor Upsilon_sw)
        and turbulence."""

        # Kinetic energy fractions.
        # Note: The factor kinetic_energy_efficiency (0.6 by default) accounts for the
        # efficiency in producing kinetic energy in the bulk fluid motion with respect to
        # the single bubble case, as found in Ref. 2209.04369 (Higgsless simulations) by
        # Jinno, Konstandin, Rubira, Stomberg. It is a configuration choice, see GWConf.
        eff = self.kinetic_energy_efficiency
        _k = self.alpha / (1 + self.alpha)
        Kphi = self.kappa_phi * _k
        Ksw = eff * self.kappa_sw * _k
        Kturb = eff * self.kappa_turb * _k

        # Peak frequency of the different GW sources today. _f0 is the redshifted Horizon scale at the time of the PT.
        # The sound-wave and turbulence fits of arXiv:2403.03723 are written in
        # H_* R_* = (8 pi)^{1/3} max(v_w, c_s) H_*/beta, their eq. (2.4). That is a convention
        # rather than a measured separation: arXiv:1910.13125, which they follow, builds it from
        # beta at percolation, calls it an estimate, and notes that a careful definition of R_*
        # still needs work. The simulations the fits come from choose a beta to make themselves
        # dimensionless and report spectra in that same convention, which omits the f_perc^{1/3}.
        # self.RH is a different thing: the separation the solver measures at percolation,
        # n_b(Tperc)^{-1/3}, larger by f_perc^{-1/3} = 1.51 for an exponential rate. Feeding it to
        # a fit whose variable is the convention misplaces the peak and the amplitude, so the
        # convention is reconstructed here from the beta/H the spectrum already uses. The RH output
        # column keeps the measured value, and the collision part is written in beta/H throughout.
        RH_end = (8 * np.pi)**(1/3) * max(self.v_wall, self.c_s) / self.betaH
        self.RH_end = RH_end

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
        f1_sw_Hz = H_p0_Hz * 0.2 / RH_end
        f2_sw_Hz = H_p0_Hz * 0.5 / RH_end / Delta_w

        # Compute the lifetime only from the relevant alpha: H tau_sh = R_* H / U_f with the
        # root-mean-square fluid velocity U_f^2 = (3/4) K_sw.
        Htau_shell = RH_end / np.sqrt(eff * self.kappa_sw * self.alpha_hyd/(1 + self.alpha_hyd) / (4/3))

        Htau_sw = self._sw_lifetime_factor(Htau_shell)
        h2Omega_int_sw = Rh2 * A_sw * Ksw**2 * Htau_sw * RH_end # eq. (2.10) of 2403.03723
        h2Omega_sw_peak_at_f2 = 1 / np.pi * (np.sqrt(2) + 2 * (f2_sw_Hz / f1_sw_Hz) / (1 + f2_sw_Hz**2 / f1_sw_Hz**2)) \
            * h2Omega_int_sw
        h2Omega_sw = self.DBPL(f_Hz, f1_sw_Hz, f2_sw_Hz, h2Omega_sw_peak_at_f2, n1_sw, n2_sw, n3_sw, a1_sw, a2_sw)

        # Turbulence source as in eq. (2.15) of 2403.03723, not the approximation in eq. (2.25)
        if Kturb != 0:
            Omega_s = Kturb
            v_Alfven = np.sqrt(Omega_s / (4/3))
            N = 2  # number of eddy turnovers
            N_Hp_tau_e = N * RH_end / (2 * np.pi * v_Alfven)
            # Duration of the turbulence source in units of Hubble time
            A_turb = 0.085
            a2_turb = 2.15
            f2_turb_Hz = 2.2 * H_p0_Hz / RH_end
            p_aniso = (1 + (f_Hz / f2_turb_Hz)**a2_turb)**(-11/(3 * a2_turb))
            T_turb = np.where(
                f_Hz <= H_p0_Hz / (2 * np.pi) / N_Hp_tau_e,
                np.log(1 + N_Hp_tau_e)**2,
                np.log(1 + H_p0_Hz / (2 * np.pi * f_Hz))**2
            )
            h2Omega_turb = 3 * A_turb * Omega_s**2 * Rh2 * RH_end**3 * (f_Hz / H_p0_Hz)**3 * p_aniso * T_turb
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

    def _h2Omega_lisa_update2019(self, f_Hz, source: str, Rh2: float, H_p0_Hz: float):
        """The template of the sub-GeV analysis 2502.19478 in flat space: sound
        waves as in the LISA update 1910.13125 with the lifetime factor
        Upsilon_sw, bubble collisions from the lattice fit of 1802.05712 and
        turbulence from 0909.0622. The sound-wave efficiency is the fit of
        1004.4187 at the strength ``alpha_DS`` normalised to the dark-sector
        bath when the model supplies it, else at ``alpha``.
        """
        alpha_sw = self.alphaDS if np.isfinite(self.alphaDS) else self.alpha
        kappa_sw = alpha_sw / (0.73 + 0.083 * np.sqrt(alpha_sw) + alpha_sw)
        # These fits are written in the same convention as the pipeline template,
        # R_* H = (8 pi)^{1/3} v_w H/beta (1910.13125, 1903.09642), an estimate built from beta at
        # percolation rather than a measured separation; self.RH, the separation the solver
        # measures at percolation, is larger by f_perc^{-1/3}.
        RH_end = (8 * np.pi)**(1/3) * max(self.v_wall, self.c_s) / self.betaH
        self.RH_end = RH_end

        # Kinetic energy fractions; KswDS sets the lifetime of the sound-wave source
        _k = self.alpha / (1 + self.alpha)
        Kphi = self.kappa_phi * _k
        Ksw = kappa_sw * _k
        Kturb = self.kappa_turb * _k
        KswDS = kappa_sw * alpha_sw / (1 + alpha_sw)

        # Peak frequencies today; _f0 is the redshifted horizon scale at the transition
        _f0 = 1.65e-5 * self.D**(-1/3) * (self.Treh_SM_GeV / 100)
        _f0 *= (self.g_eff_tot_reh / 100)**(1/2)
        _f0 *= (100 / self.h_eff_tot_reh)**(1/3)
        f0phi_Hz = _f0 * 0.51 / RH_end
        f0turb_Hz = _f0 * 3.9 / RH_end / (self.v_wall - self.c_s)
        f0sw_Hz = 8.87619e-6 * self.D**(-1/3) * (8*np.pi)**(1/3.) * self.v_wall / RH_end
        f0sw_Hz *= (self.Treh_SM_GeV / 100)
        f0sw_Hz *= (self.g_eff_tot_reh/100)**(1/2) * (100/self.h_eff_tot_reh)**(1/3)

        # Spectral functions
        def s_phi(x):
            # 1802.05712, Cutting et al.: vacuum transitions from the envelope to the lattice
            return 0.024 * x**3 * (1 + 2 * x**2.07)**(-2.18)

        def s_sw(x, p=3, q=4, n=2):
            # the (p, q, n) parameterisation of 1910.13125, normalised to unit integral
            norm = (q/p)**(p/n) * ((p+q)/q)**((p+q)/n)
            norm *= gamma(p/n) * gamma(q/n) / n / gamma((p+q)/n)
            S_tilde = x**p / (q/(p+q) + p / (p+q) * x**n)**((p+q)/n)
            return S_tilde / norm

        def s_turb(x):
            # 0909.0622, Caprini, Durrer and Servant
            turb_correction = 3.9 / (self.v_wall - self.c_s) / RH_end
            return 6.8 * x**3 * (1 + x)**(-11/3) / (1 + 8 * np.pi * x * turb_correction)

        # Lifetime of the sound-wave source: H tau_sh = R_* H / U_f with U_f^2 = (3/4) K
        Htau_shell = RH_end / np.sqrt(KswDS / (4/3))
        Htau_sw = self._sw_lifetime_factor(Htau_shell)
        Omega_phi_peak = Kphi**2 * RH_end**2
        # 3 K^2 (H tau) (H R / c_s) Omega~ of arXiv:1704.05871, with Omega~ = 0.012 and c_s = 1 / sqrt(3)
        Omega_sw_peak = Ksw**2 * RH_end * Htau_sw * 0.012 * 3 * np.sqrt(3)
        Omega_turb_peak = Kturb**(3/2) * RH_end * (1 - Htau_sw)

        f = np.asarray(f_Hz, dtype=float)
        parts = {"Bubble collisions": Omega_phi_peak * s_phi(f / f0phi_Hz),
                 "Sound waves": Omega_sw_peak * s_sw(f / f0sw_Hz),
                 "Turbulence": Omega_turb_peak * s_turb(f / f0turb_Hz)}
        if source == "all":
            return Rh2 * sum(parts.values())
        if source in parts:
            return Rh2 * parts[source]
        return 0.0 * f

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
