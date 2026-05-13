"""Compute thermodynamic and gravitational-wave observables for transitions.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np
from dataclasses import dataclass
from scipy import interpolate
from transitionlistener.generic_potential import generic_potential
from transitionlistener.phases import PhaseInfo
from transitionlistener import config
from transitionlistener.transitions import TransitionInfo
from . import thermodynamics as td
from . import errors
from transitionlistener.hydrodynamics import Hydrodynamics, calc_kappas
from transitionlistener.bubbledynamics_fixedstep import (
    calcPercAndEvolve,
    calcAlphas,
    calc_betaH_S3,
    calc_betaH_S3_approx,
    calcMeanBubbleSeparation,
    calcMeanBubbleSeparationwExp,
    calcTf,
    HubbleParameter,
    energyDensity,
)

from . import console, print
import rich
from rich.panel import Panel


@dataclass
class TransitionObservableInfo:
    """Collection of transition observable data."""
    v_wall: float = np.nan
    c_s: float = np.nan
    c_s_sym: float = np.nan
    c_s_bro: float = np.nan
    Tperc: float = np.nan
    Tperc_SM_GeV: float = np.nan
    Tcrit_SM_GeV: float = np.nan
    xi_crit: float = np.nan
    Tnuc_SM_GeV: float = np.nan
    Treh: float = np.nan
    Treh_SM_GeV: float = np.nan
    alpha: float = np.nan
    alpha_theta: float = np.nan
    alpha_p: float = np.nan
    alpha_e: float = np.nan
    alpha_hyd: float = np.nan
    alpha_inf: float = np.nan
    alpha_eq: float = np.nan
    RH: float = np.nan
    g_eff_tot_reh: float = np.nan
    h_eff_tot_reh: float = np.nan
    g0: float = np.nan
    h0: float = np.nan
    betaH_S3: float = np.nan
    betaH_RH: float = np.nan
    Tf_SM_GeV: float = np.nan
    kappa_phi: float = np.nan
    kappa_sw: float = np.nan
    kappa_turb: float = np.nan
    D: float = np.nan
    step: float = np.nan
    total_steps: float = np.nan
    WARNING_too_weak_to_compute_perc: bool = False
    WARNING_no_perc_splines: bool = False
    WARNING_betaH_small: bool = False
    WARNING_not_T0_global_min: bool = False


@dataclass
class TransitionContext:
    """Collect shared inputs and outputs for a single transition."""

    tr: TransitionInfo
    pot: generic_potential
    phases: list[PhaseInfo]
    GWconfig: config.GWConf
    PercolationConf: config.PercolationConf
    derived_param_names: list[str]
    verbose: bool
    derived_params: dict
    Tnuc: float
    outdict: dict
    phase_symmetric: PhaseInfo
    phase_broken: PhaseInfo


@dataclass
class PercolationResult:
    """Hold percolation-related outputs and splines when available."""

    Tperc: float
    TSYM: np.ndarray | None
    H: np.ndarray | None
    P: np.ndarray | None
    TBRO: np.ndarray | None
    S: np.ndarray | None
    Pint: interpolate.interp1d | None
    Sint: interpolate.interp1d | None
    Hint: interpolate.interp1d | None
    Pexpint: interpolate.interp1d | None
    soundSpSqint: interpolate.interp1d | None
    scalef_r_int: interpolate.interp1d | None
    TBROint: interpolate.interp1d | None
    successful: bool


class TransitionObservables:
    """Evaluate derived observables for every transition of a potential."""

    def __init__(self, pot: generic_potential, phases: list[PhaseInfo],
                 transitions: list[TransitionInfo], verbose=False):
        """Store inputs and immediately compute all derived quantities."""
        self.pot = pot
        self.phases = phases
        self.GWconfig = pot.config.gwConf
        self.PercolationConf = pot.config.percolationConf
        self.transitions = transitions
        self.verbose = verbose
        self.derived_param_names = [name for name in config.all_observables.keys()]

        if verbose:
            console.print(
                "[green bold]\nStart percolation computation...[/]",
            )

        self.transitionObservables = self.computeTransitionObservables()
        self.check_T0_global_min()

    def check_T0_global_min(self):
        """Check whether the global minimum at T = Tmin is the last minimum.
        If not, raise an error.
        """
        T0 = self.pot.Tmin
        last_vev = self.phases[self.transitions[-1].low_phase].valAt(self.pot.Tmin)

        if not np.allclose(last_vev, self.pot.X0):
            # The global minimum at T = 0 is not the last minimum!

            for i, tr in enumerate(self.transitions):
                if tr is not None and tr.type == 1:
                    self.transitions[i].derived_params["WARNING:not_T0_global_min"] = True
                    self.transitionObservables[i]["WARNING:not_T0_global_min"] = True

            msg = (
                "The global minimum at T = {} is not the last minimum found "
                "by the phase tracing algorithm. This may be unwanted, for "
                "example if the potential is expected to have a unique "
                "global minimum at T = {} or when identifying the last phase "
                "with the electroweak vacuum."
            ).format(T0 * self.pot.conversionFactor, T0 * self.pot.conversionFactor)
            
            if self.GWconfig.check_if_T0_global_min:
                console.print(f"[bold red]ERROR:[/bold red] {msg}")
                raise errors.GlobalMinimumError(msg)
            elif self.verbose:
                console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")
                console.print(
                    "[bold yellow]WARNING:[/bold yellow] "
                    "Continuing, because the check_if_T0_global_min flag is "
                    "set to False."
                )
        else:
            # The global minimum at T = 0 is the last minimum.
            for i, tr in enumerate(self.transitions):
                if tr is not None and tr.type == 1:
                    self.transitions[i].derived_params["WARNING:not_T0_global_min"] = False
                    self.transitionObservables[i]["WARNING:not_T0_global_min"] = False
                    
            if self.verbose:
                msg = (
                    "The global minimum at T_min is the last minimum found "
                    "by the phase tracing algorithm."
                )
                console.print(
                        f"[bold cyan]INFO:[/bold cyan] {msg}"
                    )

    def computeTransitionObservables(self):
        """
        Calculate the GW observables for each transition in the
        transitions_dict. This is done by calling the
        `computeTransitionObservables_one_transition`
        function from the `bubbledynamics` module.

        Raises
        ------
        err
            Raises an error if GW observables cannot be calculated.

        Returns
        -------
        transitionObservables: dict
            A dictionary with the GW observables for each transition.
        """
        self.transitionObservables = {i: {} for i in range(len(self.transitions))}
        self.percolationResults = {i: None for i in range(len(self.transitions))}
        self.outdicts = {i: None for i in range(len(self.transitions))}

        for i, tr in enumerate(self.transitions):
            if tr is not None and tr.type == 1:
                try:
                    derived_params_dict_tr, percol, outdict = self.computeTransitionObservables_one_transition(
                        tr, nAction=self.PercolationConf.n_action
                    )
                    self.transitions[i].derived_params = derived_params_dict_tr
                    self.percolationResults[i] = percol
                    self.outdicts[i] = outdict
                    self.transitionObservables[i] = derived_params_dict_tr
                except Exception as err:
                    if self.verbose:
                        print("Error calculating GW parameters: ", err)
                    raise err
        return self.transitionObservables

    def _initialize_transition_context(self, tr: TransitionInfo) -> TransitionContext:
        """Prepare reusable data structures for a single transition."""
        derived_params = {
            "WARNING:too_weak_to_compute_perc": False,
            "WARNING:no_perc_splines": False,
            "WARNING:betaH_small": False,
        }
        return TransitionContext(
            tr=tr,
            pot=self.pot,
            phases=self.phases,
            GWconfig=self.GWconfig,
            PercolationConf=self.PercolationConf,
            derived_param_names=self.derived_param_names,
            verbose=self.verbose,
            derived_params=derived_params,
            Tnuc=tr.Tnuc,
            outdict=tr.full_tunneling_info,
            phase_symmetric=self.phases[tr.high_phase],
            phase_broken=self.phases[tr.low_phase],
        )

    def _ensure_wall_velocity(self, ctx: TransitionContext, T: float) -> None:
        """Populate v_wall if requested by configuration."""
        if "v_wall" not in ctx.derived_param_names:
            return
        GWconfig = ctx.GWconfig
        derived = ctx.derived_params
        if GWconfig.wall_velocity == "LTE":
            hydr = Hydrodynamics(ctx.pot, ctx.phase_symmetric, ctx.phase_broken, ctx.verbose)
            derived["v_wall"] = hydr.calcWallVelocityLTE(T)
        elif GWconfig.wall_velocity == "c":
            derived["v_wall"] = 1.0
        elif GWconfig.wall_velocity == "WallGo":
            raise NotImplementedError("WallGo is not implemented yet.")
        else:
            vw = 0.0
            try:
                vw = float(GWconfig.wall_velocity)
            except:
                raise ValueError("wall_velocity must be 'LTE' or a float. " +
                                 "Not: ", GWconfig.wall_velocity, "!")
            if vw <= 0.0 or vw > 1.0:
                raise ValueError("Wall velocity must be 0 < vw <= 1.0." +
                                 "Not ", vw, "!")
            else:
                derived["v_wall"] = vw

    def _ensure_sound_speed(self, ctx: TransitionContext, T: float) -> None:
        """Populate sound speed values when requested."""
        if "c_s" not in ctx.derived_param_names:
            return
        derived = ctx.derived_params
        GWconfig = ctx.GWconfig
        if ctx.verbose:
            print("Calculating sound speed...")
        if GWconfig.sound_speed == "compute":
            hydr = Hydrodynamics(ctx.pot, ctx.phase_symmetric, ctx.phase_broken, ctx.verbose)
            c_sb = hydr.calc_cs(T, sym=False)
            c_ss = hydr.calc_cs(T, sym=True)
            derived["c_s"] = c_sb
            derived["c_s_sym"] = c_ss
            derived["c_s_bro"] = c_sb
        else:
            cs = 0.0
            try:
                cs = float(GWconfig.sound_speed)
            except:
                raise NotImplementedError(
                    "The GWconfig.sound_speed option is not implemented yet: "
                    + GWconfig.sound_speed
                )
            if cs <= 1.0 and cs > 0:
                derived["c_s"] = cs
                derived["c_s_sym"] = cs
                derived["c_s_bro"] = cs
            else:
                raise ValueError("Speed of sound must be 0 < cs <= 1." +
                                 " Not ", cs, "!")
            

    def _compute_transition_strength(self, ctx: TransitionContext) -> float:
        """Return alpha_theta at nucleation and handle logging."""
        weak_threshold = ctx.PercolationConf.weak_threshold
        _, alpha_theta_nuc, _, _, _, _ = calcAlphas(
            ctx.Tnuc, ctx.pot, ctx.phase_symmetric, ctx.phase_broken,
            verbose=ctx.verbose
        )
        if ctx.verbose:
            print("alpha at nucleation =", np.squeeze(alpha_theta_nuc))
        return np.squeeze(alpha_theta_nuc)

    def _compute_percolation(
        self,
        ctx: TransitionContext,
        alpha_theta_nuc: float,
        rtol: float,
        nAction: int,
    ) -> PercolationResult:
        """Either reuse nucleation data or perform the percolation computation."""
        pot = ctx.pot
        derived = ctx.derived_params
        verbose = ctx.verbose
        GWconfig = ctx.GWconfig
        weak_threshold = ctx.PercolationConf.weak_threshold

        if alpha_theta_nuc < weak_threshold:
            if verbose:
                print(
                    "alpha at nucleation is too small (alpha =",
                    alpha_theta_nuc,
                    "<",
                    weak_threshold,
                    "), using Tnuc as Treh",
                )
            derived["Treh_SM_GeV"] = ctx.Tnuc * pot.conversionFactor
            derived["WARNING:too_weak_to_compute_perc"] = True
            derived["WARNING:no_perc_splines"] = True
            return PercolationResult(
                Tperc=ctx.Tnuc,
                TSYM=None,
                H=None,
                P=None,
                TBRO=None,
                S=None,
                Pint=None,
                Sint=None,
                Hint=None,
                Pexpint=None,
                soundSpSqint=None,
                scalef_r_int=None,
                TBROint=None,
                successful=False,
            )

        if verbose:
            msg = (
                f"alpha at nucleation is large enough "
                f"(alpha = {alpha_theta_nuc:.3e} > {weak_threshold:.3e}). "
                "Continue with percolation computation."
            )
            print(msg)

        try:
            Tperc, TSYM, H, P, TBRO, S, P_exp, scalef_r, soundSpSq = calcPercAndEvolve(
                ctx.outdict,
                ctx.Tnuc,
                ctx.phase_symmetric,
                ctx.phase_broken,
                pot,
                rtol=rtol,
                nAction=nAction,
                verbose=verbose,
                vw=ctx.derived_params.get("v_wall", 1.0),
            )
        except Exception as err:
            if verbose:
                print("Fatal error in calcPercAndEvolve: ", err)
            raise

        try:
            if verbose:
                print("Calculating percolation splines...")
            finite_mask = np.isfinite(P)
            Pint = interpolate.interp1d(
                TSYM[finite_mask],
                P[finite_mask],
                kind="cubic",
                fill_value=(0, 1),
                bounds_error=False,
            )
            Pexpint = interpolate.interp1d(
                TSYM[finite_mask],
                P_exp[finite_mask],
                kind="cubic",
                fill_value=(0, 1),
                bounds_error=False,
            )
            soundSqSqint = interpolate.interp1d(
                TSYM[finite_mask],
                soundSpSq[finite_mask],
                kind="cubic",
                bounds_error=False,
            )
            scalef_r_int = interpolate.interp1d(
                TSYM[finite_mask],
                scalef_r[finite_mask],
                kind="cubic",
                bounds_error=False,
            )
            Sint = interpolate.interp1d(
                TSYM[np.isfinite(S)],
                S[np.isfinite(S)],
                kind="cubic",
                fill_value=(-np.inf, np.inf),
                bounds_error=False,
            )
            Hint = interpolate.interp1d(TSYM, H, kind="cubic")
            TBROint = interpolate.interp1d(TSYM, TBRO, kind="cubic")
        except Exception as err:
            derived["WARNING:no_perc_splines"] = True
            if verbose:
                print("Error in computing percolation splines: ", err)
                msg = (
                    "WARNING: Cannot calculate the splines. Continue with the "
                    "percolation temperature, but no splines will be available "
                    "for further calculations."
                )
                console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")
            Pint = Sint = Hint = TBROint = Pexpint = soundSqSqint = scalef_r_int = None

        return PercolationResult(
            Tperc=Tperc,
            TSYM=TSYM,
            H=H,
            P=P,
            TBRO=TBRO,
            S=S,
            Pint=Pint,
            Sint=Sint,
            Hint=Hint,
            Pexpint=Pexpint,
            soundSpSqint=soundSqSqint,
            scalef_r_int=scalef_r_int,
            TBROint=TBROint,
            successful=True,
        )

    def _populate_temperature_observables(
        self,
        ctx: TransitionContext,
        percolation: PercolationResult,
    ) -> None:
        """Fill temperature-related entries in the derived parameters."""
        derived = ctx.derived_params
        pot = ctx.pot

        if "Tperc_SM_GeV" in ctx.derived_param_names:
            if ctx.verbose:
                print("Calculating percolation temperature...")
            derived["Tperc"] = percolation.Tperc
            derived["Tperc_SM_GeV"] = percolation.Tperc * pot.conversionFactor

        if "Tcrit_SM_GeV" in ctx.derived_param_names:
            if ctx.verbose:
                print("Calculating critical temperature...")
            derived["Tcrit_SM_GeV"] = ctx.tr.Tcrit * pot.conversionFactor

        if "xi_crit" in ctx.derived_param_names:
            if ctx.verbose:
                print("Calculating critical vev over temperature...")
            T_crit = ctx.tr.Tcrit
            vev_crit = np.sqrt(np.sum(ctx.phase_broken.valAt(T_crit) ** 2))
            derived["xi_crit"] = vev_crit / T_crit

        if "Tnuc_SM_GeV" in ctx.derived_param_names:
            if ctx.verbose:
                print("Calculating nucleation temperature...")
            derived["Tnuc_SM_GeV"] = ctx.tr.Tnuc * pot.conversionFactor

        if "Treh_SM_GeV" in ctx.derived_param_names:
            if ctx.verbose:
                print("Calculating reheating temperature...")
            if derived.get("WARNING:no_perc_splines", False):
                if ctx.verbose:
                    msg = (
                        "WARNING: Set Treh_SM_GeV to Tperc_SM_GeV "
                        "because no percolation splines were found."
                    )
                    console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")
                Treh = percolation.Tperc
            else:
                Treh = percolation.TBROint(percolation.Tperc) if percolation.TBROint else percolation.Tperc
            derived["Treh"] = Treh
            derived["Treh_SM_GeV"] = Treh * pot.conversionFactor

    def _compute_alpha_parameters(
        self,
        ctx: TransitionContext,
        percolation: PercolationResult,
    ) -> None:
        """Compute alpha-related quantities if requested."""
        if "alpha" not in ctx.derived_param_names:
            return
        if ctx.verbose:
            print("Calculating alpha parameters...")
        alpha_p, alpha_theta, alpha_e, alpha_hyd, alpha_inf, alpha_eq = calcAlphas(
            percolation.Tperc, ctx.pot, ctx.phase_symmetric, ctx.phase_broken,
            verbose=ctx.verbose
        )
        derived = ctx.derived_params
        derived["alpha"] = np.squeeze(alpha_theta)
        derived["alpha_theta"] = np.squeeze(alpha_theta)
        derived["alpha_p"] = np.squeeze(alpha_p)
        derived["alpha_e"] = np.squeeze(alpha_e)
        derived["alpha_hyd"] = np.squeeze(alpha_hyd)
        derived["alpha_inf"] = np.squeeze(alpha_inf)
        derived["alpha_eq"] = np.squeeze(alpha_eq)

    def _compute_beta_and_separation(
            self,
            ctx: TransitionContext,
            percolation: PercolationResult,
            tmin: float,
            tmax: float,
    ) -> None:
        """Populate beta/H variants, RH and Tf if configured."""
        derived = ctx.derived_params
        pot = ctx.pot
        verbose = ctx.verbose
        outdict = ctx.outdict
        phase_symmetric = ctx.phase_symmetric
        phase_broken = ctx.phase_broken

        if "betaH_S3_approx" in ctx.derived_param_names:
            if verbose:
                print("Calculating beta/H (approximation through derivative of bounce action)...")
                derived["betaH_S3_approx"] = calc_betaH_S3_approx(
                    percolation.Tperc, outdict, pot, phase_symmetric, phase_broken, tmin, tmax, verbose=verbose
                )

        if "RH" in ctx.derived_param_names:
            if verbose:
                print("Calculating mean bubble separation...")
            if (
                    derived.get("WARNING:no_perc_splines", False)
                    or percolation.Sint is None
                    or percolation.TSYM is None
            ):
                if verbose:
                    msg = (
                        "WARNING: Set RH to Nan "
                        "because no percolation splines were found."
                    )
                    console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")
                derived["RH"] = np.nan
            else:
                Rsep = calcMeanBubbleSeparation(
                    percolation.Tperc,
                    percolation.TSYM[0],
                    percolation.Sint,
                    percolation.Pint,
                    percolation.Hint,
                )
                Rsep_exp = calcMeanBubbleSeparationwExp(
                    percolation.Tperc,
                    percolation.TSYM[0],
                    percolation.Sint,
                    percolation.Pexpint,
                    percolation.Hint,
                    percolation.soundSpSqint,
                    percolation.scalef_r_int,
                )
                RsepHperc = Rsep * percolation.Hint(percolation.Tperc)
                derived["RH"] = RsepHperc

        # This combines the dark-sector contribution with the SM bath and
        # therefore assumes a DS + SM setup.
        if "g_eff_tot_reh" in ctx.derived_param_names:
            if verbose:
                print("Calculating g_eff_tot_reh...")
            bosons = pot.boson_massSq(ctx.phase_broken.valAt(percolation.Tperc), percolation.Tperc)
            fermions = pot.fermion_massSq(ctx.phase_broken.valAt(percolation.Tperc))
            gefftotreh = td.e_geffDS(
                bosons*(~pot.mass_spectrum.is_SM_bosons),
                fermions*(~pot.mass_spectrum.is_SM_fermions),
                percolation.Tperc,
            ) + td.e_geffSM(percolation.Tperc, pot.conversionFactor)
            derived["g_eff_tot_reh"] = gefftotreh

        # This combines the dark-sector contribution with the SM bath and
        # therefore assumes a DS + SM setup.
        if "h_eff_tot_reh" in ctx.derived_param_names:
            if verbose:
                print("Calculating h_eff_tot_reh...")
            bosons = pot.boson_massSq(ctx.phase_broken.valAt(percolation.Tperc), percolation.Tperc)
            fermions = pot.fermion_massSq(ctx.phase_broken.valAt(percolation.Tperc))
            hefftotreh = td.s_geffDS(
                bosons*(~pot.mass_spectrum.is_SM_bosons),
                fermions*(~pot.mass_spectrum.is_SM_fermions),
                percolation.Tperc,
            ) + td.s_geffSM(percolation.Tperc, pot.conversionFactor)
            derived["h_eff_tot_reh"] = hefftotreh

        if "g0" in ctx.derived_param_names:
            derived["g0"] = 2

        if "h0" in ctx.derived_param_names:
            derived["h0"] = 3.91

        if (
                "betaH_S3" in ctx.derived_param_names
                or "betaH_RH" in ctx.derived_param_names
                or "Tf_SM_GeV" in ctx.derived_param_names
        ):
            self._populate_beta_related(
                ctx,
                percolation,
                tmin,
                tmax,
            )

    def _populate_beta_related(
        self,
        ctx: TransitionContext,
        percolation: PercolationResult,
        tmin: float,
        tmax: float,
    ) -> None:
        """Gather betaH variants and Tf once percolation data is available."""
        derived = ctx.derived_params
        verbose = ctx.verbose
        pot = ctx.pot

        if "betaH_S3" in ctx.derived_param_names:
            if verbose:
                print("Calculating betaH_S3 (through the percolation splines)...")
            if derived.get("WARNING:no_perc_splines", False) or percolation.Sint is None:
                if verbose:
                    msg = (
                        "WARNING: Approximate betaH_S3 only by using calc_betaH_S3_approx "
                        "because no percolation splines were found."
                    )
                    console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")

                derived["betaH_S3"] = calc_betaH_S3_approx(
                    percolation.Tperc,
                    ctx.outdict,
                    ctx.pot,
                    ctx.phase_symmetric,
                    ctx.phase_broken,
                    tmin,
                    tmax,
                    verbose=verbose,
                )
            else:
                derived["betaH_S3"] = calc_betaH_S3(
                    percolation.Tperc,
                    percolation.Sint,
                    ctx.outdict,
                    ctx.pot,
                    ctx.phase_symmetric,
                    ctx.phase_broken,
                    verbose,
                )

        if "betaH_RH" in ctx.derived_param_names:
            if derived.get("WARNING:no_perc_splines", False):
                if verbose:
                    msg = (
                        "Set betaH_RH to np.nan "
                        "because no percolation splines were found."
                    )
                    console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")
                derived["betaH_RH"] = np.nan
            else:
                f_perc = ctx.PercolationConf.f_perc
                derived["betaH_RH"] = (
                    (8 * np.pi) ** (1 / 3)
                    / derived["RH"]
                    * max(derived.get("v_wall", 1.0), derived.get("c_s", 1 / np.sqrt(3)))
                    * (1 / f_perc) ** (1 / 3)
                )

        if "Tf_SM_GeV" in ctx.derived_param_names:
            if verbose:
                print("Calculating Tf_SM_GeV...")
            if (
                derived.get("WARNING:no_perc_splines", False)
                or percolation.Pint is None
                or percolation.TSYM is None
            ):
                if verbose:
                    msg = (
                        "Set Tf_SM_GeV to np.nan because no "
                        "percolation splines were found."
                    )
                    console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")
                Tf_val = np.nan
            else:
                try:
                    Tf_val = calcTf(
                        percolation.Tperc,
                        percolation.TSYM[-1],
                        percolation.Pint,
                        pot,
                        verbose=verbose,
                    )
                except errors.PercolationError as err:
                    if verbose:
                        msg = (
                            "Could not compute the temperature at which the "
                            "transition ends (Tf_SM_GeV). Set it to np.nan instead."
                        )
                        console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")
                        console.print("Error message: ", err)
                    Tf_val = np.nan
            derived["Tf_SM_GeV"] = Tf_val * pot.conversionFactor if np.isfinite(Tf_val) else np.nan

    def _compute_kappa_parameters(
        self,
        ctx: TransitionContext,
        percolation: PercolationResult
    ) -> None:
        """Calculate kappa parameters if required.

        Parameters
        ----------
        ctx : TransitionContext
        percolation : PercolationResult

        Returns
        -------
        None. 
        """
        names = ctx.derived_param_names
        if not {"kappa_phi", "kappa_sw", "kappa_turb"} & set(names):
            return
        if ctx.verbose:
            print("Calculating kappa parameters...")
        _, alpha_thetaP, _, alpha_hydP, alpha_infP, alpha_eqP = calcAlphas(
            percolation.Tperc, ctx.pot, ctx.phase_symmetric, ctx.phase_broken,
            verbose=ctx.verbose
        )
        # Compute the initial bubble radius R0. We take it to be
        # the radius of the bubble at the nucleation temperature,
        # which is the radius at which the field is halfway
        # between the start and endpoint of the transition. Note
        # that the endpoint of the transition is not necessarily
        # the broken phase minimum, rather the point from which on
        # the field just rolls down to the broken phase minimum, i.e.
        # the endpoint in the instanton solution.

        # this is the dictionary with the transition
        # parameters at the nucleation temperature
        instanton = ctx.tr.instanton
        R_, phi_ = instanton.profile1D.R, instanton.profile1D.Phi
        phi_middle = (phi_[0] + phi_[-1]) / 2
        idx = np.argmin(np.abs(phi_ - phi_middle))

        # this is the radius at which the field is halfway between the
        # start and endpoint of the transition
        R0 = R_[idx]

        RH = ctx.derived_params.get("RH")
        if RH is None:
            overlap_tmin = max(float(ctx.phase_symmetric.Tmin), float(ctx.phase_broken.Tmin))
            overlap_tmax = min(float(ctx.phase_symmetric.Tmax), float(ctx.phase_broken.Tmax))
            betaH_approx = calcBetaHapprox(
                percolation.Tperc,
                ctx.outdict,
                ctx.pot,
                ctx.phase_symmetric,
                ctx.phase_broken,
                overlap_tmin,
                overlap_tmax,
                verbose=ctx.verbose,
            )
            f_perc = ctx.PercolationConf.f_perc
            RH = (8 * np.pi / f_perc) ** (1 / 3) / betaH_approx * max(
                ctx.derived_params.get("v_wall", 1.0),
                ctx.derived_params.get("c_s", 1 / np.sqrt(3))
            )

        # Bubble size at time of collision (percolation)
        Hstar = HubbleParameter(
            energyDensity(ctx.pot, ctx.phase_symmetric, percolation.Tperc),
            ctx.pot.conversionFactor
        )
        Rstar = RH / Hstar

        kappa_phi, kappa_sw, kappa_turb = calc_kappas(
            alpha_hydP,
            alpha_infP,
            alpha_eqP,
            ctx.derived_params.get("v_wall", 1.0),
            ctx.derived_params.get("c_s", 1 / np.sqrt(3)),
            Rstar,
            R0,
            ctx.GWconfig,
        )
        ctx.derived_params["kappa_phi"] = kappa_phi
        ctx.derived_params["kappa_sw"] = kappa_sw
        ctx.derived_params["kappa_turb"] = kappa_turb

    def _finalize_results(self, ctx: TransitionContext) -> dict:
        """Ensure all requested parameters are present and return dict."""
        derived = ctx.derived_params
        beta_S3 = derived.get("betaH_S3", np.nan)
        beta_RH = derived.get("betaH_RH", np.nan)

        if "WARNING:betaH_small" in ctx.derived_param_names:
            derived["WARNING:betaH_small"] = derived.get("betaH_RH", np.inf) < 10

        if "WARNING:betaH_very_small" in ctx.derived_param_names:
            derived["WARNING:betaH_very_small"] = derived.get("betaH_RH", np.inf) < 3

        if "WARNING:nucleationRate_nonexponential" in ctx.derived_param_names:
            derived["WARNING:nucleationRate_nonexponential"] = (
                np.isfinite(beta_S3) and beta_S3 < 0
            )

        if "WARNING:betaH_nonfinite" in ctx.derived_param_names:
            derived["WARNING:betaH_nonfinite"] = (
                not np.isfinite(beta_S3) or not np.isfinite(beta_RH)
            )

        if "WARNING:betaH_mismatch" in ctx.derived_param_names:
            mismatch = False
            if np.isfinite(beta_S3) and np.isfinite(beta_RH):
                abs_beta_S3 = abs(beta_S3)
                abs_beta_RH = abs(beta_RH)
                if abs_beta_S3 == 0 or abs_beta_RH == 0:
                    mismatch = not np.isclose(abs_beta_S3, abs_beta_RH)
                else:
                    mismatch = max(
                        abs_beta_RH / abs_beta_S3,
                        abs_beta_S3 / abs_beta_RH,
                    ) >= 10
            derived["WARNING:betaH_mismatch"] = mismatch
            
        if "WARNING:not_T0_global_min" in ctx.derived_param_names:
            # This flag is set in the check_T0_global_min function,
            # when checking all transitions after initialization.
            derived["WARNING:not_T0_global_min"] = False

        # Check if there are any names in derived_param_names which are not
        # in the derived_params_dict
        for name in ctx.derived_param_names:
            if name not in derived:
                msg = (
                    f"{name} not found in derived_params_dict. "
                    "It might be that this variable still needs to be implemented. "
                    "It is added to the dictionary with value NaN."
                )
                console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")
                derived[name] = np.nan

        return derived

    def computeTransitionObservables_one_transition(self, tr : TransitionInfo,
                                                    rtol=1e-4, nAction=50):
        """Calculate the GW observables after the phase tracing and the
        computation of the nucleation temperature have been performed.
        This function assumes kinetic equilibrium between the dark sector and
        the SM.

        Parameters
        ----------
        tr : TransitionInfo
            The transition information for which to compute observables.
        rtol : float, optional
            Relative tolerance for the computation. Default is 1e-4.
        nAction : int, optional
            Number of action points to consider. Default is 50.
        Returns
        ----------
        GWparams : tuple (float)
            Dictionary with the GW parameters."""

        ctx = self._initialize_transition_context(tr)
        if ctx.verbose:
            print(
                "\nStart calculating GW observables for transition from phase",
                tr.high_phase,
                "to phase",
                tr.low_phase,
                "at Tnuc =",
                tr.Tnuc * self.pot.conversionFactor,
                "GeV",
            )

        tmin = max(float(ctx.phase_symmetric.Tmin), float(ctx.phase_broken.Tmin))
        tmax = min(float(ctx.phase_symmetric.Tmax), float(ctx.phase_broken.Tmax))
        if tmin >= tmax:
            raise errors.PercolationError(
                "No overlapping temperature interval between traced phases: "
                f"Tsym in [{ctx.phase_symmetric.Tmin:.8g}, {ctx.phase_symmetric.Tmax:.8g}], "
                f"Tbro in [{ctx.phase_broken.Tmin:.8g}, {ctx.phase_broken.Tmax:.8g}]."
            )

        alpha_theta_nuc = self._compute_transition_strength(ctx)
        percolation = self._compute_percolation(ctx, alpha_theta_nuc, rtol, nAction)

        self._ensure_wall_velocity(ctx, percolation.Tperc)
        self._ensure_sound_speed(ctx, percolation.Tperc)

        self._populate_temperature_observables(ctx, percolation)
        self._compute_alpha_parameters(ctx, percolation)
        self._compute_beta_and_separation(ctx, percolation, tmin, tmax)
        self._compute_kappa_parameters(ctx, percolation)

        derived = ctx.derived_params
        if "D" in ctx.derived_param_names:
            derived["D"] = 1
        if "step" in ctx.derived_param_names:
            derived["step"] = ctx.tr.step
        if "total_steps" in ctx.derived_param_names:
            derived["total_steps"] = ctx.tr.total_steps

        return self._finalize_results(ctx), percolation, ctx.outdict

    def __iter__(self):
        """Enable iteration over the derived observables dictionary."""
        return iter(self.transitionObservables)

    def __getitem__(self, key: str):
        """Allow dictionary-style access to an individual transition."""
        return self.transitionObservables[key]

    def __len__(self):
        """Return the number of transitions for which observables were computed."""
        return len(self.transitionObservables)

    def keys(self):
        """Expose the transition keys."""
        return self.transitionObservables.keys()

    def items(self):
        """Return key/value pairs of the transition observables."""
        return self.transitionObservables.items()

    def printDerivedParams(self, derived_param_names):
        """Print a rich-formatted table of the derived parameters per transition."""
        nTransitions = len(self.transitions)
        print("\nComputed thermodynamic quantities for", nTransitions, "transition(s):\n")
        for i, tr in enumerate(self.transitions):
            if tr.type == 1:
                derived_params_dict = self.transitionObservables[i]
                table = rich.table.Table(
                    title=f"Transition {i+1}: First-order phase transition",
                    title_justify="left",
                    box=rich.box.ROUNDED,
                )
                table.add_column("Parameter", style="cyan", no_wrap=True)
                table.add_column("Value", style="orange1")
                for j, name in enumerate(derived_param_names):
                    table.add_row(name, f"{derived_params_dict[name]:.10e}")
                console.print(table)

            if tr.type == 2:
                table = rich.table.Table(
                    title=f"Transition {i+1}: Second-order phase transition",
                    title_justify="left",
                    box=rich.box.ROUNDED,
                )
                table.add_column("Parameter", style="cyan", no_wrap=True)
                table.add_column("Value", style="orange1")
                table.add_row("Start phase", str(tr.high_phase))
                table.add_row("End phase", str(tr.low_phase))
                table.add_row("Tcrit / GeV", str(tr.Tcrit * self.pot.conversionFactor))
                console.print(table)
