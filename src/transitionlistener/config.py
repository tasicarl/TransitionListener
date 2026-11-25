"""This module contains the configurations.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import copy


class TracingConf:
    """Phase-tracing and tunneling computation parameters."""

    internal_scale = 1000      # Reference energy scale (internal units); sets the magnitude of the stable minimum.
    Tmax_factor = 2.5               # Tmax_factor * internal_scale = max tracing temperature.
    diftol = 1                      # Max field-space distance between two phases (internal units) before they merge.
    Z2_cutof_factor = -5            # For Z2-symmetric potentials: reject phases with field < Z2_cutof_factor * diftol.
    tracing_derivative_order = 4    # Finite-difference derivative order: 2 or 4.
    tracing_field_accuracy = 1e-3   # Target field-space accuracy during tracing (1e-3 works well).
    tracing_temp_accuracy = 1e-3    # Target temperature accuracy during tracing.
    gen_mirror_phases = False       # If True, also generate mirror-transformed phases of the potential.
    pair_transition_over_images = False  # If True, compute the bounce on every image of the
                                    # false vacuum and keep the smallest action, at the cost of
                                    # one bounce per image wherever the false vacuum is not at a
                                    # fixed point of the group. Off by default because the phase
                                    # tracing now puts both phases on the same image, which is
                                    # what the pairing was added to repair; see
                                    # `pathDeformation.imagesToPairOver`.
    nucleation_Ttol = 1e-8          # Nucleation-temperature tolerance (internal units).
    do_tachyon_test = True          # If True, verify the potential is stable at T = 0.
    approx_strength_threshold = 1e-4  # Transitions with DV / T^4 below this are treated as second-order.
    do_high_T_phase_check = True    # Hotfix toggle for high-T phase checks (needed by the vev-flip-flop model).

    tracing_args = {
        "dtstart": 1e-4,         # Initial trace step for traceMultiMin (scaled by tHigh - tLow).
        "tjump": 1e-5,           # Branch-jump size when tracing continues after a phase endpoint.
        "dtabsMax": 20.0,        # Max absolute step is largest of abs(dtstart)*dtabsMax ...
        "dtfracMax": 0.25,       # ... and t*dtfracMax in internal units.
        "dtmin": 1e-6,           # Min step (relative to dtstart) before assuming transition ends.
        "deltaX_tol": 1.2,       # deltaX_tol*deltaX_target = max x-error before shrinking the step.
        "minratio": 1e-4,        # Smallest/largest Hessian eigenvalue ratio before branch ends.
        "local_min_args": {      # Forwarded to phases.findApproxLocalMin() from traceMultiMin().
            "n": 100,            # Interpolation samples along x1->x2 for local-min detection.
            "edge": 0.05,        # Fraction trimmed off each end before searching.
        },
    }
    """Arguments forwarded to the phase-tracing routines."""
    tunneling_params = {
        "tunneling_findProfile_params": {
            "phitol": 1e-10,     # Field-space tolerance for the overshoot/undershoot profile.
            "xtol": 1e-10,       # Radial shooting tolerance for the bounce match.
            "rmin": 1e-4,
            "rmax": 1e4,
            "npoints": 500,      # Radial support points for the bounce profile.
        },
        "V_spline_samples": 1000,  # Points used to spline V(x) along the bounce path; None evaluates V directly.
        "tunneling_init_params": {
            "phi_eps": 1e-6,     # Initial offset from the false vacuum.
            "rscale": None,      # Manual radial scale; None = auto-estimate.
        },
        "deformation_deform_params": {
            "startstep": 2e-3,
            "fRatioConv": 2e-2,  # Convergence on normal/tangential force ratio along the path. Auto-tightened to 3e-3 for Ndim>=2 models in generic_potential.__init__.
            "converge_0": 1.5,   # Early-stop threshold for the force-ratio at the first stage.
            "fRatioIncrease": 5.0,
            "maxiter": 500,
            "verbose": False,
        },
    }
    """Arguments forwarded to the bounce-action / tunneling routines."""

    def __init__(self):
        """Give every model its own copy of the mutable default dicts."""
        self.tracing_args = copy.deepcopy(TracingConf.tracing_args)
        self.tunneling_params = copy.deepcopy(TracingConf.tunneling_params)


class PercolationConf:
    """Settings for the percolation solver."""

    # Solver mode and integral backend.
    algorithm_mode = "adaptive_step_size"       # "adaptive_step_size" (default) or "fixed_step_size".
    integral_method = "ode"               # Percolation integral backend: "ode" or "double_integral".
    time_temperature_mode = "sound_speed" # dT/dt relation: "sound_speed" or "bag".
    entropy_scheme_diagnostic = True  # Compare both entropy schemes on the percolation support.
    # Flag the point when the two entropy schemes' 3 c_s^2 differ, in the median over the
    # percolation support, by more than this fraction. That median estimates the relative
    # shift the mean bubble separation takes between the schemes, so the criterion is
    # "flag a point whose R_* is uncertain at about the five per cent level from this
    # modelling choice". It sits a little below five so that the scatter of the estimate does
    # not hide such a point: calibrated over 91 points run in both schemes, the estimate lies
    # between 0.88 and 1.29 times the shift it predicts for eight points in ten. The precise
    # value is not critical and four per cent is the round one; on that calibration sample it
    # flags one point of the ninety-one, whose separation moves by 3.7%. Lower it to see the
    # weaker cases: 3.5% flags six, all of which move by more than 2.9%.
    entropy_scheme_warn_threshold = 0.04  # bubbledynamics.ENTROPY_SCHEME_DEFAULT_THRESHOLD
    entropy_definition = "dof_table"      # Thermodynamics of the expansion history:
                                          # "dof_table" (counted degrees of freedom, default)
                                          # or "eff_potential" (-dV/dT of the potential).

    # Percolation targets.  Iperc is computed as -ln(1 - f_perc) at use sites.
    f_perc = 0.28957                      # True-vacuum fraction at percolation.
    f_start = 1e-3                        # Max initial true-vacuum fraction; below this, widen the T-range.
    f_final = 0.99                        # Min final true-vacuum fraction for a completed percolation.

    # Iteration controls shared by both percolation passes.
    weight = 1 / 1.5                      # Fraction of support points spent between Tnuc and Tperc (fixed_step_size mode).
    # Raised from 10. On the 2HDM edge line every point that failed with an unresolved
    # percolation profile reported reaching this support-update limit with 112-119 of its
    # action evaluations still unspent, so the iteration count, not the action budget, was
    # what stopped the refinement; the measured cost of the extra iterations on that line is
    # about 2% more action solves, because the budget they spend was already there. Note the
    # setting is shared with the fixed_step_size backend, where it caps the step-2 and step-3
    # range-adjustment iterations instead, each of which recomputes the whole profile.
    maxit = 20                            # Max iterations per percolation pass (step 2 and step 3 each).
    rel_increment = 0.10                  # Relative T-range growth when the initial range is too small.
    max_boundary_ratio = 0.45             # Max share of T-range allowed on plateaus where P_true = 0 or 1.

    # Support-bank budget (adaptive step size mode). DZW, "dynamic zoom window", is the
    # internal name of the adaptive solver (the *_dynamiczoomwindow_* functions).
    n_action_min = 15                     # Min support points built before a DZW sweep.
    n_action_increment = 5                # Support points added per DZW refinement iteration.
    n_action_max = 60                     # Max new support points per DZW support update; points with a cached action are free.
    max_action_temperatures = 100         # Max action evaluations of the step-1 scout while it brackets the transition.

    # fixed_step_size support count.
    n_action = 30                         # Fixed support count used by the fixed_step_size path.

    # P(T)-jump refinement and final acceptance.
    large_delta_p_refine_threshold = 0.1  # Neighbour DeltaP above this triggers extra refinement.
    large_delta_p_success_threshold = 0.2 # Neighbour DeltaP above this blocks acceptance.

    # Diagnostics and optional rescue paths.
    jitter_GH4_threshold = 1.0            # log10(Gamma/H^4) jitter tolerance before flagging unresolved action.
    jitter_rescue = True                  # If True, rerun with tunneltight precision on jitter detection.
    n_jitter_save = 20                    # Max jitter diagnostic samples saved to disk.

    # Observable accuracy targets (step 2 and step 3).
    acc_tperc = 1e-2                      # Relative accuracy goal on Tperc.
    acc_tfinal = 1e-2                     # Relative accuracy goal on Tfinal.
    acc_rh = 1e-2                         # Relative accuracy goal on RH.

    # Largest change of log10(Gamma/H^4) allowed between neighbouring support points. The
    # percolation integral interpolates the logarithm of the nucleation rate between the
    # samples, so an interval across which the rate rises by many decades is not resolved:
    # the integral stays at zero through the interval and then jumps, which the controller
    # then mistakes for a sharp transition and tries to refine at the wrong place. Seen on
    # the abelian dark Higgs benchmark (g = 1, v = 0.1 GeV, lambda = 0.03323), where a
    # 4.6 MeV wide gap in the support (35.9 to 40.5 MeV) carried 290 decades of rate and
    # the point was lost with error code 10 although its transition is healthy. Tunable
    # per run as percolation_max_log10_rate_step; zero switches the criterion off.
    max_log10_rate_step = 12.0

    # Strength-based validity flag.
    weak_threshold = 1e-4                 # Below this alpha at Tperc, the percolation result is flagged invalid.

    # beta/H from the action derivative, T d(S3/T)/dT at Tperc, estimated by a local least-squares
    # quadratic through the action samples of the percolation support. Interpolating splines are not
    # used for this: the adaptive support places samples as close as 1e-6 Tperc apart, and their
    # slope turns an irregular 1e-2 error in S3/T into errors of hundreds in beta/H.
    # CAUTION: the defaults below were validated on ONE model class only, the 2HDM BSMPT benchmark
    # (lambda3 line, 5 points against converged references, 4 bounce-precision variants): with 11
    # samples the fit reproduced the converged beta/H within 0.5 %. The adaptive support there put
    # 56-108 samples in total, 6-36 within 0.3 % of Tperc, and the 11 nearest spanned at most 0.95 %
    # of Tperc. Models with a sparser support near Tperc, or with sharper structure in S3/T on the
    # scale of that span, may need different values - re-validate before trusting them there.
    betaH_S3_fit_points = 11              # Support samples nearest Tperc used for the fit.
    betaH_S3_fit_check_points = 7         # Smaller subset refitted as a stability check.
    betaH_S3_fit_rel_tol = 0.03           # WARNING if the two fits differ by more than this (2HDM: median 0.14 %, max 2.0 %).
    betaH_S3_fit_rescue = True            # If True, recompute the fit's actions with tunneltight deformation when the two fits disagree.
    betaH_S3_fit_min_per_side = 2         # Samples required both below and above Tperc; otherwise fall back.
    betaH_S3_fit_max_rel_span = 0.02      # Largest |T/Tperc - 1| allowed among the samples; otherwise fall back.
    betaH_S3_fallback_rel_step = 2e-3     # Fallback: 5 new actions at Tperc (1 + k step), k = -2..2, same fit.


class GWConf:
    """Settings for calculating the gravitational-wave signal."""

    use_mean_bubble_separation = True  # The speed of the spectrum: True, (beta/H)_RH from the mean bubble separation at
                                       # percolation; False, (beta/H)_S3 from the slope of the action, in every factor
                                       # (the separation then follows from it, and expansion_speed is ignored).
    wall_velocity = "LTE"              # "LTE", "c", "WallGo", or a float in (0, 1].
    bw_collisions = "off"              # Bubble-wall collisions: "off", "full" (kappa_phi=1), or "NLO".
    turbulence = "off"                 # Turbulence contribution: "off" or a model key.
    dilution = False                   # If True, apply a dilution factor to the GW signal.
    epsilon_turbulence = 0.1           # Energy fraction redirected into turbulence (0 disables).
    sound_speed = "compute"            # Sound-speed strategy: "compute" (broken-phase cs), "1/3" (bag value), or a float in (0, 1).
    coupled_hydrodynamics = True       # Couple the dark and visible sectors in the hydro solve.
    check_if_T0_global_min = False     # If True, require the T=0 minimum to be global (else just warn).
    # The shipped PTA free-spectrum product the likelihoods are evaluated against. Written
    # out rather than imported from `observability`, which pulls in matplotlib and the
    # optional PTA stack; a test holds this to `observability.DEFAULT_PTA_FREE_SPECTRUM`.
    pta_free_spectrum = "ng15_hd_v2"

    # Modelling choices of the spectrum itself. They are the nuisance parameters
    # of a parameter inference and can be set per run from the input dictionary
    # through the ``gw_<name>`` keys handled in runtime_options.
    source_model = "LISA-pipeline24"   # The template. "LISA-pipeline24": the three sources of the LISA pipeline paper 2403.03723,
                                       # with the sound waves as fitted to the Higgsless simulations of 2209.04369; "LISA-update2019":
                                       # the sound waves of the LISA update 1910.13125 with the collision (1802.05712) and turbulence
                                       # (0909.0622) sources of the sub-GeV analysis 2502.19478; "LewickiVaskonen22": the broken power
                                       # laws of 2208.11697 fitted to strongly supercooled scalar-field and fluid shells, one per shell type.
    kinetic_energy_efficiency = 0.6    # Fluid kinetic energy relative to the single-bubble estimate kappa_sw alpha/(1+alpha), from the
                                       # Higgsless simulations of 2209.04369; values between 0.4 and 0.8 are realistic (LISA-pipeline24 only).
    expansion_suppression = False      # The cosmic expansion during the transition, 2511.15687: the amplitude of eq. (23), the peak
                                       # frequency of eq. (22) and the causal f^3 tail below the horizon frequency of eq. (21). For the
                                       # LISA templates the ratios A(beta)/A(infinity) and f_p(beta)/f_p(infinity) rescale the flat-space
                                       # template; for LewickiVaskonen22 the fit with expansion replaces the flat-space one. False: flat space.
    sw_lifetime_factor = "min"         # Form of the sound-wave lifetime factor Upsilon_sw, always applied: "min" = min(1, H tau_sh)
                                       # (2403.03723), "guo" = 1 - (1 + 2 H tau_sh)^(-1/2) (2007.08537), "guo_capped" = Guo's form with
                                       # tau_sh first capped at one Hubble time (BSMPT).
    lv22_shape_field = "bubbles_R3"    # LewickiVaskonen22: column of table 1 of 2208.11697 for the scalar-field shells (energy kappa_phi).
    lv22_shape_fluid = "fluid_R3_v1"   # LewickiVaskonen22: column for the fluid shells (energy kappa_sw).
    lv22_expanding_shape = "dissipative"  # LewickiVaskonen22 with expansion, the models of 2511.15687: "dissipative" (dissipative bulk
                                       # flow, a = b = c = 2; bubbles_R3 in flat space) or "non_dissipative" (bulk flow, a = c = 1, b = 2;
                                       # bubbles_R2 in flat space).
    causality_tail = "smooth"          # Join of the f^3 tail below the horizon frequency, applied when expansion_suppression is on
                                       # (f_H of 2511.15687 eq. (21)) and to the flat-space LewickiVaskonen22 fits (f_H = a_p H_p/2 pi,
                                       # eq. (20) there): "smooth" (no kink),
                                       # "sharp" (as fitted in 2511.15687) or "off" (the flat-space fits as published in 2208.11697).
    expansion_speed = "betaH_RH"       # The speed in the three expansion factors of 2511.15687 (amplitude, peak frequency, horizon
                                       # frequency) when expansion_suppression is on. "betaH_RH": the speed from the bubble separation,
                                       # as in every other factor of the spectrum. "betaH_Gamma": H^-1 dlnGamma/dt at T_perc, with H the
                                       # Hubble rate of the percolation history, the growth rate of the nucleation rate; it equals the
                                       # beta/H_V of the exponential rate the fits were made for up to H(T_perc)/H_V, which is close to
                                       # one in a vacuum-dominated transition. Needs the observable betaH_Gamma, which only the adaptive
                                       # solver computes (the fixed step size solver refuses this choice). Every other factor keeps
                                       # betaH_RH.


# Allowed values of the GWConf modelling choices, shared by the input-dictionary
# overrides (runtime_options) and the spectrum itself (gwfopt).
GW_SOURCE_MODELS = ("LISA-pipeline24", "LISA-update2019", "LewickiVaskonen22")
GW_SW_LIFETIME_FACTORS = ("min", "guo", "guo_capped")
GW_LV22_SHAPES_FIELD = ("bubbles_envelope", "bubbles_R2", "bubbles_R3")
GW_LV22_SHAPES_FLUID = ("fluid_R2_v1", "fluid_R2_vcs", "fluid_R3_v1", "fluid_R3_vcs")
GW_LV22_EXPANDING_SHAPES = ("dissipative", "non_dissipative")
GW_CAUSALITY_TAILS = ("smooth", "sharp", "off")
GW_EXPANSION_SPEEDS = ("betaH_RH", "betaH_Gamma")
GW_BW_COLLISIONS = ("off", "full", "NLO")


class NucleationConf:
    """Settings for the bubble nucleation rate.

    By default `rate_mode` "S3overT_4D" is the hard-coded 4D rate
    T^4 (S3/2piT)^{3/2} exp(-S3/T), and the validity monitor is inert unless a model supplies the ratios it needs.
    A model with a 3D potential (one that mixes in ``one_field_rate_3d.ThreeDimensionalRate``) switches on the 3D
    effective-theory rate in its own defaults; the values below are those of every other model.
    """

    rate_mode = "S3overT_4D"      # "S3overT_4D" (default, 4D) or "model" (model supplies the rate).
    # Whether the model's own rate expression is used on top of its action.
    # At leading order it is not needed: supplying the action already gives the
    # 3D rate, because the two expressions are the same function. Set this once
    # the fluctuation determinants are in, where the prefactor really differs.
    include_determinants = False
    keep_4d_action = False        # rate_mode "model": also solve the 4D bounce, for comparison (stored as action_4d).
    # Order of the hard -> soft matching relations. "NLO" uses DRalgo's
    # next-to-leading-order thermal masses and g3^2 with the one-loop quartic. "LO" uses
    # the leading-order masses and g3^2, which carry no mubar, so nothing cancels the 4D matching
    # scale: at sub-GeV vevs the scale band on T_perc is 11-15% against 4% at NLO.
    matching_order = "NLO"        # "NLO" or "LO".
    mu4_over_T = 7.0553           # Central 4D matching scale in units of T: 4 pi exp(-gamma_E).
    input_scheme = "CW"           # Conformal U(1) models: "CW" or "zero_quartic" (lambda(mu0) = 0); see the 3D model files.
    mu3_choice = "g3sq"           # Conformal U(1) models: the 3D scale, "g3sq" or "gT" (g(mu0) T).
    # The A0^2 phi^2 coupling h3 in the temporal mass m_D^2 + h3 phi^2 / 2 of the 3D rate (LO potential, temporal
    # determinant, double-counting subtraction): "h3", the NLO-matched coupling; "2g3sq", its leading-order value.
    temporal_coupling = "h3"
    # include_determinants = True with no table attached: the grid to use instead of the one shipped for the model
    # (transitionlistener.eft_rate.nlo_tables), and whether to add the dimension-six shift the grid stores apart from Delta S.
    nlo_grid_file = None
    include_dim6 = False


class Configuration:
    """Container holding one independent configuration object per subsystem."""

    def __init__(self):
        """Start every model with independent configuration objects."""
        self.tracingConf = TracingConf()
        self.gwConf = GWConf()
        self.percolationConf = PercolationConf()
        self.nucleationConf = NucleationConf()


all_observables = {
    "alpha": R"$\alpha$",
    "alpha_theta": R"$\alpha_{\theta}$",
    "alpha_thetabar": R"$\alpha_{\bar\theta}$",
    "alpha_inf": R"$\alpha_\infty$",
    "alpha_eq": R"$\alpha_\mathrm{eq}$",
    "betaH_S3": R"$(\beta /H)_{S_3}$",
    "betaH_RH": R"$(\beta /H)_{RH}$",
    "betaH_Gamma": R"$(\beta /H)_\Gamma$",
    "RH": R"$RH$",
    "Treh_SM_GeV": R"$T^{\mathrm{reh}}_{\mathrm{SM}}$ / $\mathrm{GeV}$",
    "Treh_DS_GeV": R"$T^{\mathrm{reh}}_{\mathrm{DS}}$ / $\mathrm{GeV}$",
    "Tperc_SM_GeV": R"$T^{\mathrm{perc}}_{\mathrm{SM}}$ / $\mathrm{GeV}$",
    "g_eff_tot_reh": R"$g^{\mathrm{reh}}_{\mathrm{tot}}$",
    "h_eff_tot_reh": R"$h^{\mathrm{reh}}_{\mathrm{tot}}$",
    "kappa_phi": R"$\kappa_{\phi}$",
    "kappa_sw": R"$\kappa_{\mathrm{sw}}$",
    "kappa_turb": R"$\kappa_{\mathrm{turb}}$",
    "g0": R"$g_0$",
    "h0": R"$h_0$",
    "v_wall": R"$v_{\mathrm{wall}}$",
    "D": R"$D$",
    "c_s": R"$c_\mathrm{s}$",
    "c_s_sym": R"$c_\mathrm{s}^\mathrm{sym}$",
    "c_s_bro": R"$c_\mathrm{s}^\mathrm{bro}$",
    "step": R"$\mathrm{step}$",
    "total_steps": R"$\mathrm{total\ steps}$",
    "Tnuc_SM_GeV": R"$T^{\mathrm{nuc}}_{\mathrm{SM}}$ / $\mathrm{GeV}$",
    "Tcrit_SM_GeV": R"$T^{\mathrm{crit}}_{\mathrm{SM}}$ / $\mathrm{GeV}$",
    "Tf_SM_GeV": R"$T^{\mathrm{final}}_{\mathrm{SM}}$ / $\mathrm{GeV}$",
    "xi_crit": R"$\xi_{\mathrm{c}}$",
    "WARNING:too_weak_to_compute_perc": R"$\mathrm{WARNING:} \mathrm{too\ weak\ to\ compute\ perc}$",
    "WARNING:no_perc_splines": R"$\mathrm{WARNING:} \mathrm{no\ perc\ splines}$",
    # Whether the accepted percolation profile met acc_tperc, acc_tfinal and acc_rh. The
    # targets are advisory -- the controller accepts either way, because by then refinement
    # has nothing left to try -- so this is the only record of whether they were honoured.
    # Warning true means a target was measured and missed, or a required observable could not
    # be determined at all. Warning false with finite uncertainties means every target that
    # could be measured was met; that includes the common case where `acc_rh` could not be
    # judged, because `DIAG:accuracy_rh_uncertainty` is a difference against a previous
    # estimate of the mean bubble separation or it is inf, and an inf there says the target was
    # never measured rather than missed. Tperc and Tf fall back to their own bracket widths and
    # so are always measurable from a single profile.
    # Warning false with nan uncertainties means no verdict was formed at all: the
    # fixed_step_size backend, which has no such controller, and the adaptive controller's
    # iteration-limit exit.
    "WARNING:accuracy_targets_missed":
        R"$\mathrm{WARNING:}\ \mathrm{accuracy\ targets\ missed}$",
    "DIAG:accuracy_tperc_uncertainty": R"$\mathrm{DIAG:}\ \delta T_\mathrm{perc}/T_\mathrm{perc}$",
    "DIAG:accuracy_tfinal_uncertainty": R"$\mathrm{DIAG:}\ \delta T_\mathrm{f}/T_\mathrm{f}$",
    "DIAG:accuracy_rh_uncertainty": R"$\mathrm{DIAG:}\ \delta R_\mathrm{H}/R_\mathrm{H}$",
    # The wall-velocity branch test assumes the matching residual of `arxiv:2303.10171`
    # crosses zero once below the Jouguet velocity; WallGo (`arxiv:2411.04970`) makes the same
    # assumption and notes it can fail for strong transitions. Where it fails the velocity is a
    # choice among branches, and WallGo's practice is to flag such an outcome rather than
    # resolve it silently.
    "WARNING:v_wall_branch_ambiguous":
        R"$\mathrm{WARNING:}\ v_\mathrm{wall} \mathrm{\ branch\ ambiguous}$",
    "WARNING:v_wall_no_solution":
        R"$\mathrm{WARNING:}\ v_\mathrm{wall} \mathrm{\ has\ no\ solution}$",
    "WARNING:betaH_small": R"$\mathrm{WARNING:} (\beta/H)_{RH} \mathrm{\ too\ small}$",
    "WARNING:betaH_very_small": R"$\mathrm{WARNING:} (\beta/H)_{RH} \mathrm{\ very\ small}$",
    "WARNING:betaH_mismatch": R"$\mathrm{WARNING:} (\beta/H)_{S_3} \mathrm{\ vs\ } (\beta/H)_{RH} \mathrm{\ mismatch}$",
    "WARNING:betaH_nonfinite": R"$\mathrm{WARNING:} (\beta/H)_{S_3} \mathrm{\ or\ } (\beta/H)_{RH} \mathrm{\ nonfinite}$",
    "WARNING:betaH_S3_fit_unstable": R"$\mathrm{WARNING:} (\beta/H)_{S_3} \mathrm{\ fit\ unstable}$",
    "WARNING:action_rescue_attempted": R"$\mathrm{WARNING:} \mathrm{action\ rescue\ attempted}$",
    "WARNING:action_rescue_failed": R"$\mathrm{WARNING:} \mathrm{action\ rescue\ failed}$",
    "DIAG:action_rescue_attempts": R"$\mathrm{DIAG:} \mathrm{actions\ recomputed}$",
    "DIAG:runtime_s": R"$\mathrm{DIAG:} \mathrm{runtime\ [s]}$",
    "WARNING:daisy_outside_validity":
        R"$\mathrm{WARNING:} \mathrm{daisy\ outside\ validity}$",
    "DIAG:daisy_over_radiation": R"$\mathrm{DIAG:} V_\mathrm{daisy}/V_\mathrm{rad}$",
    "WARNING:eft3d_outside_validity": R"$\mathrm{WARNING:} \mathrm{3D\ EFT\ outside\ validity}$",
    "DIAG:eft3d_expansion_3d": R"$\mathrm{DIAG:} g_3^2/(\pi m_3)$",
    "DIAG:eft3d_hard": R"$\mathrm{DIAG:} m_\mathrm{heavy}/(2\pi T)$",
    "DIAG:eft3d_dim6": R"$\mathrm{DIAG:} |\Delta S_6|/S_3$",
    "WARNING:noisy_c_s":
        R"$\mathrm{WARNING:} c_\mathrm{s} \mathrm{\ step\ dependent}$",
    "DIAG:c_s_step_change": R"$\mathrm{DIAG:} \Delta c_\mathrm{s}/c_\mathrm{s}$",
    "WARNING:entropy_scheme_sensitive":
        R"$\mathrm{WARNING:} \mathrm{entropy\ scheme\ matters}$",
    "DIAG:entropy_scheme_cs2_spread": R"$\mathrm{DIAG:} \Delta(3c_s^2)/3c_s^2$",
    "DIAG:entropy_scheme_lna_gap": R"$\mathrm{DIAG:} |\Delta \ln a|$",
    "WARNING:nucleationRate_nonexponential": R"$\mathrm{WARNING:} \mathrm{nucleation\ rate\ nonexponential}$",
    "WARNING:spline_tnuc_unavailable": R"$\mathrm{WARNING:} T_\mathrm{nuc} \mathrm{\ spline\ unavailable}$",
    "WARNING:spline_tnuc_not_reached": R"$\mathrm{WARNING:} T_\mathrm{nuc} \mathrm{\ criterion\ not\ reached}$",
    "WARNING:spline_tnuc_failed": R"$\mathrm{WARNING:} T_\mathrm{nuc} \mathrm{\ spline\ failed}$",
    "WARNING:not_T0_global_min": R"$\mathrm{WARNING:} T=0 \mathrm{\ not\ global\ min}$",
    "WARNING:false_vacuum_not_shrinking": R"$\mathrm{WARNING:} \mathcal{V}_\mathrm{false} \mathrm{\ growing\ at\ } T_\mathrm{perc}$",
}
